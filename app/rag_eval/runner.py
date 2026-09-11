"""
RAG 评估执行模块。

可以把这个文件理解成“评估流程主脚本”，主要负责 4 件事：

1. 把测试知识通过真实导入链路写入 Milvus；
2. 读取题库，逐条走真实查询链路；
3. 统计每一层的召回效果；
4. 输出汇总结果和报告文件。

这个文件不追求过度抽象，优先保证：
- 顺着往下读就能看懂；
- 关键参数和返回值写清楚；
- 每一步为什么做能说清楚。
"""

import json
import time
from pathlib import Path
from statistics import mean
from unittest.mock import patch

from app.infra.vector_store.milvus_gateway import milvus_gateway
from app.process.import_.agent.nodes.node_bge_embedding import node_bge_embedding
from app.process.import_.agent.nodes.node_import_milvus import node_import_milvus
from app.process.import_.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.process.import_.agent.state import create_default_state
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.query.agent.state import create_query_default_state
from app.rag_eval.dataset import (
    ARTIFACTS_DIR,
    TEST_FILE_TITLE,
    TEST_ITEM_NAME,
    build_batch_eval_cases,
    build_import_chunks,
    build_web_search_docs,
    load_batch_eval_cases,
    write_batch_eval_cases,
)
from app.rag_eval.metrics import evaluate_query_state
from app.shared.clients import mongo_history_utils
from app.shared.config.embedding_config import embedding_config
from app.shared.config.milvus_config import milvus_config
from app.shared.config.reranker_config import reranker_config


LAYER_LABELS = {
    "embedding_chunks": "普通检索",
    "hyde_embedding_chunks": "HyDE检索",
    "rrf_chunks": "RRF融合",
    "reranked_docs": "最终重排结果",
}


def insert_env_ready() -> bool:
    """
    判断“测试数据入库”所需环境是否齐全。

    返回值：
    - True：可以执行入库
    - False：缺少必要配置
    """
    return bool(
        milvus_config.milvus_url
        and milvus_config.chunks_collection
        and milvus_config.item_name_collection
        and (embedding_config.bge_m3_path or embedding_config.bge_m3)
    )


def batch_eval_ready() -> bool:
    """
    判断“批量评测”所需环境是否齐全。

    和入库相比，这里额外要求 reranker 模型可用。

    返回值：
    - True：可以执行批量评测
    - False：缺少必要配置
    """
    return bool(
        milvus_config.milvus_url
        and milvus_config.chunks_collection
        and milvus_config.item_name_collection
        and (embedding_config.bge_m3_path or embedding_config.bge_m3)
        and reranker_config.bge_reranker_large
    )


def milvus_ready() -> bool:
    """
    判断 MilvusClient 是否已经初始化成功。

    返回值：
    - True：Milvus 可用
    - False：Milvus 不可用
    """
    return milvus_gateway.client is not None


def close_mongo_client() -> None:
    """
    关闭评测过程中可能创建的 Mongo 连接。
    """
    mongo_tool = getattr(mongo_history_utils, "_history_mongo_tool", None)
    if mongo_tool is not None:
        mongo_tool.client.close()


def _query_chunk_rows_with_retry(
    milvus_client,
    *,
    file_title: str,
    expected_count: int,
    retry_times: int = 5,
) -> list[dict]:
    """
    查询刚写入的 chunk 数据，并做几次短暂重试。

    参数：
    - milvus_client：当前项目的 Milvus 客户端
    - file_title：本次导入文件标题，用于过滤出当前测试数据
    - expected_count：期望查回多少条 chunk
    - retry_times：最多重试次数

    返回值：
    - list[dict]：查询到的 chunk 行数据

    为什么要重试：
    - 向量库刚写完数据时，马上查询有时会出现短暂延迟。
    """
    for _ in range(retry_times):
        chunk_rows = milvus_client.query(
            collection_name=milvus_gateway.chunk_collection_name,
            filter=f"item_name == '{TEST_ITEM_NAME}'",
            output_fields=["chunk_id", "item_name", "title", "part", "file_title", "content"],
        )
        chunk_rows = [row for row in chunk_rows if row.get("file_title") == file_title]
        if len(chunk_rows) >= expected_count:
            chunk_rows.sort(key=lambda row: row["part"])
            return chunk_rows
        time.sleep(0.3)
    return []


def insert_batch_eval_dataset() -> dict:
    """
    写入评测测试数据，并生成题库文件。

    返回值：
    - item_name：本次写入的主体名称
    - item_rows：item_name 集合查询结果
    - chunk_rows：chunk 集合查询结果
    - case_count：本次生成的题库问题数量

    执行步骤：
    1. 检查环境；
    2. 构造导入 state；
    3. 走真实导入节点；
    4. 查询回刚写入的数据；
    5. 根据真实 chunk_id 生成题库；
    6. 返回本次入库结果。
    """
    if not insert_env_ready():
        raise RuntimeError("缺少 Milvus 或 Embedding 配置，无法执行评测数据入库。")

    milvus_client = milvus_gateway.client
    if milvus_client is None:
        raise RuntimeError("MilvusClient 未成功初始化，无法执行评测数据入库。")

    # 1. 准备导入 state。
    # 参数说明：
    # - task_id：本次导入任务标识
    # - file_title：这份测试知识的标题
    # - chunks：待导入的测试知识数据集
    state = create_default_state(
        task_id="rag_eval_batch_insert",
        file_title=TEST_FILE_TITLE,
        chunks=build_import_chunks(),
    )

    # 2. 固定主体识别结果，保证测试数据每次都稳定。
    # 这里不让大模型自由识别，是为了避免测试数据每次导入出来的主体名称不一致。
    # todo:  with patch("app.rag.import_.item_name_service.recognize_item_name_by_chunks - 执行",  返回必须 return_value=TEST_ITEM_NAME):
    with patch("app.rag.import_.item_name_service.recognize_item_name_by_chunks", return_value=TEST_ITEM_NAME):
        state = node_item_name_recognition(state)  # item_name  item_name存储到向量数据库

        # 3. 继续走真实导入链路：向量化 -> 写入 Milvus。
        state = node_bge_embedding(state)
        state = node_import_milvus(state)

    # 4. 查询 item_name 集合，确认主体索引已经写入。
    item_rows = milvus_client.query(
        collection_name=milvus_gateway.item_collection_name,
        filter=f"file_title == '{TEST_FILE_TITLE}'",
        output_fields=["item_name", "file_title"],
    )
    if not item_rows:
        raise RuntimeError("item_name 集合中未查询到评测数据。")

    # 5. 查询 chunk 集合，拿到真实 chunk_id。
    # 后面题库标注和指标统计都基于这些真实 id。
    chunk_rows = _query_chunk_rows_with_retry(
        milvus_client,
        file_title=TEST_FILE_TITLE,
        expected_count=len(build_import_chunks()),
    )
    if len(chunk_rows) != len(build_import_chunks()):
        raise RuntimeError("chunks 集合中的评测数据数量与预期不一致。")

    # 6. 用“真实 chunk_id”生成题库，这样后面的评测才能精确比对命中情况。
    case_list = build_batch_eval_cases(
        gold_chunk_ids=[str(row["chunk_id"]) for row in chunk_rows],
        expected_item_names=[TEST_ITEM_NAME],
    )
    write_batch_eval_cases(case_list)

    return {
        "item_name": state["item_name"],
        "item_rows": item_rows,
        "chunk_rows": chunk_rows,
        "case_count": len(case_list),
    }


def run_query_eval_case(case_data: dict) -> dict:
    """
    执行单条问题评测。

    参数：
    - case_data：单条题库数据，至少包含：
      - case_id
      - question
      - expected_item_names
      - gold_chunk_ids
      - must_hit_chunk_ids

    返回值：
    - dict：这道题在 4 层查询链路上的评测结果

    这一层只关心一件事：
    给定一条题库问题，让真实查询链路完整跑一遍，
    最后把查询结果和标注答案做对比。
    """
    # 1. 构造查询 state。这里直接把题库里的 expected_item_names 放进去，
    # 避免主体识别波动影响“召回评测”本身。
    state = create_query_default_state(
        session_id=f"{case_data['case_id']}_query",
        original_query=case_data["question"],
        rewritten_query=case_data["question"],
        item_names=case_data["expected_item_names"],
        is_stream=False,
        web_search_docs=build_web_search_docs(),
    )

    # 2. 固定 HyDE 输出，避免大模型随机性影响评测稳定性。
    # 这样每次跑同一题，HyDE 这一层的检索输入都保持一致。
    with patch(
        "app.rag.query.hyde_search_service.call_llm_answer",
        return_value="HAK 180 局部烫印通过设置 50mm 起点和 170mm 终点实现区域控制。",
    ):
        # 3. 逐层执行真实查询链路。
        # 每个节点执行完后，结果都会回写到 state 里，供后续评测计算使用。
        state.update(node_search_embedding(state))
        state.update(node_search_embedding_hyde(state))
        state = node_rrf(state)
        state = node_rerank(state)

    # 4. 用 metrics 模块统一计算这条题的评测结果。
    return evaluate_query_state(result_state=state, expected=case_data)


def summarize_eval_results(eval_results: list[dict]) -> dict:
    """
    对多条题目的评测结果做平均汇总。

    参数：
    - eval_results：所有题目的详细评测结果

    返回值：
    - dict：整体平均指标
    """
    if not eval_results:
        return {"case_count": 0, "avg_item_name_hit_rate": 0.0, "layers": {}}

    layer_names = ("embedding_chunks", "hyde_embedding_chunks", "rrf_chunks", "reranked_docs")
    summary = {
        "case_count": len(eval_results),
        "avg_item_name_hit_rate": round(mean(result["item_name_hit_rate"] for result in eval_results), 4),
        "layers": {},
    }

    for layer_name in layer_names:
        summary["layers"][layer_name] = {
            "avg_precision": round(mean(result["layers"][layer_name]["precision"] for result in eval_results), 4),
            "avg_recall": round(mean(result["layers"][layer_name]["recall"] for result in eval_results), 4),
            "avg_must_hit_rate": round(
                mean(result["layers"][layer_name]["must_hit_rate"] for result in eval_results),
                4,
            ),
        }

    return summary


def _to_chinese_case_result(eval_result: dict) -> dict:
    """
    将单条评测结果转成更适合直接写报告的中文结构。
    """
    chinese_layers = {}
    for layer_name, layer_result in eval_result.get("layers", {}).items():
        chinese_layers[LAYER_LABELS.get(layer_name, layer_name)] = {
            "检索结果chunk_id列表": layer_result.get("retrieved_chunk_ids", []),
            "检索结果数量": layer_result.get("retrieved_count", 0),
            "标注相关chunk_id列表": layer_result.get("gold_chunk_ids", []),
            "标注相关数量": layer_result.get("gold_count", 0),
            "命中chunk_id列表": layer_result.get("hit_chunk_ids", []),
            "命中数量": layer_result.get("hit_count", 0),
            "必须命中chunk_id列表": layer_result.get("must_hit_chunk_ids", []),
            "必须命中结果列表": layer_result.get("must_hit_ids", []),
            "必须命中数量": layer_result.get("must_hit_count", 0),
            "精确率": layer_result.get("precision", 0.0),
            "召回率": layer_result.get("recall", 0.0),
            "必命中率": layer_result.get("must_hit_rate", 0.0),
        }

    return {
        "用例ID": eval_result.get("case_id", ""),
        "问题": eval_result.get("question", ""),
        "预期主体列表": eval_result.get("expected_item_names", []),
        "识别主体列表": eval_result.get("predicted_item_names", []),
        "主体命中率": eval_result.get("item_name_hit_rate", 0.0),
        "分层结果": chinese_layers,
    }


def _to_chinese_summary(summary: dict) -> dict:
    """
    将汇总结果转成中文结构，方便直接展示或落报告。
    """
    chinese_layers = {}
    for layer_name, layer_summary in summary.get("layers", {}).items():
        chinese_layers[LAYER_LABELS.get(layer_name, layer_name)] = {
            "平均精确率": layer_summary.get("avg_precision", 0.0),
            "平均召回率": layer_summary.get("avg_recall", 0.0),
            "平均必命中率": layer_summary.get("avg_must_hit_rate", 0.0),
        }

    return {
        "用例总数": summary.get("case_count", 0),
        "平均主体命中率": summary.get("avg_item_name_hit_rate", 0.0),
        "分层汇总": chinese_layers,
    }


def save_batch_eval_report(
    eval_results: list[dict],
    summary: dict,
    report_name: str = "rag_eval_batch_report.json",
) -> Path:
    """
    保存批量评测报告。

    报告里同时放两部分：
    1. 汇总结果；
    2. 每道题的详细结果。

    参数：
    - eval_results：每道题的详细评测结果
    - summary：整体汇总结果
    - report_name：报告文件名，默认 `rag_eval_batch_report.json`

    返回值：
    - Path：报告文件路径
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS_DIR / report_name
    report_path.write_text(
        json.dumps(
            {
                "汇总结果": _to_chinese_summary(summary),
                "详细结果": [_to_chinese_case_result(result) for result in eval_results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report_path


def run_batch_eval(case_list: list[dict] | None = None) -> dict:
    """
    执行整批评测。

    参数：
    - case_list：可选，自定义题库；不传时默认从题库文件读取

    返回值：
    - eval_results：每条题目的详细结果
    - summary：整体平均指标
    - report_path：报告文件路径

    流程按顺序就是：
    1. 检查环境；
    2. 读取题库；
    3. 逐题调用 `run_query_eval_case()`；
    4. 汇总指标；
    5. 保存报告；
    6. 返回结果。
    """
    if not batch_eval_ready():
        raise RuntimeError("缺少批量评测依赖配置，无法执行批量检索评测。")
    if not milvus_ready():
        raise RuntimeError("MilvusClient 未成功初始化，无法执行批量检索评测。")

    real_case_list = case_list if case_list is not None else load_batch_eval_cases()
    if not real_case_list:
        raise RuntimeError("未找到批量评测样本，请先执行评测数据入库。")

    # 逐题跑查询链路，拿到每一道题的分层评测结果。
    eval_results = [run_query_eval_case(case_data) for case_data in real_case_list]
    # 再把所有题的结果做平均汇总。
    summary = summarize_eval_results(eval_results)
    # 最后把结果写到报告文件。
    report_path = save_batch_eval_report(eval_results, summary)
    return {
        "eval_results": eval_results,
        "summary": summary,
        "report_path": report_path,
    }
