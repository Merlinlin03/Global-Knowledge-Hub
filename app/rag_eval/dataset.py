"""
评估样本定义模块。

这个文件只负责维护评估用的数据：
1. 测试知识数据集；
2. 测试问题题库；
3. 题库文件的读写位置。

阅读顺序建议：
1. 先看 `build_import_chunks()`，理解测试知识有哪些；
2. 再看 `build_batch_eval_cases()`，理解评测问题怎么设计；
3. 最后看 `write_batch_eval_cases()` / `load_batch_eval_cases()`。
"""

import json
from pathlib import Path


# 当前演示样本对应的主体名称与文件标题。
TEST_ITEM_NAME = "HAK 180"
TEST_FILE_TITLE = "HAK 180"

# 评估过程中生成的题库文件和报告统一放到包内 artifacts 目录。
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
GENERATED_BATCH_CASES_FILE = ARTIFACTS_DIR / "hak180_eval_cases.json"


def build_import_chunks() -> list[dict]:
    """
    构造测试知识切片。

    返回值：
    - list[dict]：可直接送入项目导入链路的 chunk 列表

    字段说明：
    - title：当前切片标题
    - parent_title：当前切片所属章节
    - part：切片顺序，后面生成题库时会按这个顺序理解知识范围
    - file_title：文件标题，这里和主体名称保持一致，便于回查
    - content：切片正文

    设计原则：
    - 测试问题不必很多，但测试知识数据要稍微丰富一些；
    - 这样评测更像真实知识库，而不是只拿 2~3 条文本做演示。
    """
    return [
        # 参数设置类
        {
            "title": "局部烫印范围设置",
            "parent_title": "局部烫印模式",
            "part": 1,
            "file_title": TEST_FILE_TITLE,
            "content": (
                "HAK 180 在局部烫印模式下，若想只在纸张顶部 50mm 到 170mm 的区域转印烫金膜，"
                "应将起始位置设置为 50mm，结束位置设置为 170mm，然后保存参数。"
            ),
        },
        # 操作步骤类
        {
            "title": "局部烫印操作步骤",
            "parent_title": "局部烫印模式",
            "part": 2,
            "file_title": TEST_FILE_TITLE,
            "content": (
                "进入 HAK 180 操作面板的局部烫印菜单，选择定位设置，依次输入 50mm 起点和 "
                "170mm 终点，确认后执行试印。"
            ),
        },
        # 注意事项类
        {
            "title": "局部烫印注意事项",
            "parent_title": "局部烫印模式",
            "part": 3,
            "file_title": TEST_FILE_TITLE,
            "content": (
                "设置 HAK 180 局部烫印范围前，需要确认纸张原点、膜带张力和压印位置正常，"
                "否则会出现转印区域偏移。"
            ),
        },
        # 参数保存与生效类
        {
            "title": "局部烫印参数保存",
            "parent_title": "局部烫印模式",
            "part": 4,
            "file_title": TEST_FILE_TITLE,
            "content": (
                "HAK 180 在完成局部烫印的起点和终点设置后，需要先保存当前参数，"
                "再进行下一步试印，否则修改后的区间不会生效。"
            ),
        },
        # 常见问题类
        {
            "title": "局部烫印常见偏移原因",
            "parent_title": "局部烫印模式",
            "part": 5,
            "file_title": TEST_FILE_TITLE,
            "content": (
                "HAK 180 局部烫印出现转印区域偏移时，常见原因包括纸张原点错误、"
                "膜带张力异常、压印位置不准以及定位参数未重新保存。"
            ),
        },
        # 结果确认类
        {
            "title": "局部烫印试印确认",
            "parent_title": "局部烫印模式",
            "part": 6,
            "file_title": TEST_FILE_TITLE,
            "content": (
                "HAK 180 完成局部烫印参数设置后，应先执行试印，观察 50mm 到 170mm 的"
                "转印区间是否准确，再进入正式生产。"
            ),
        },
        # 异常处理类
        {
            "title": "局部烫印异常处理",
            "parent_title": "局部烫印模式",
            "part": 7,
            "file_title": TEST_FILE_TITLE,
            "content": (
                "如果 HAK 180 局部烫印试印后发现位置偏差，应重新检查纸张原点、"
                "膜带张力、压印位置，并再次确认起点终点参数是否正确。"
            ),
        },
        # 场景说明类
        {
            "title": "局部烫印模式说明",
            "parent_title": "局部烫印模式",
            "part": 8,
            "file_title": TEST_FILE_TITLE,
            "content": (
                "HAK 180 的局部烫印模式用于只在纸张指定区域进行烫金转印，"
                "常用于局部图案、标题区域或指定装饰区域的加工。"
            ),
        },
    ]


def build_web_search_docs() -> list[dict]:
    """
    构造联网占位结果。

    返回值：
    - list[dict]：模拟联网检索结果的列表

    字段说明：
    - title：网页标题
    - snippet：网页摘要
    - url：网页地址

    为什么只放一条固定数据：
    - 当前项目 rerank 阶段要求同时存在本地候选和联网候选；
    - 评测重点是本地召回链路，所以这里用稳定占位数据避免联网波动干扰结果。
    """
    return [
        {
            "title": "联网占位结果",
            "snippet": "这是一条联网占位结果，用于满足 rerank 入参，不提供关键答案。",
            "url": "https://example.com/hak180-placeholder",
        }
    ]


def build_batch_eval_cases(
    gold_chunk_ids: list[str],
    expected_item_names: list[str] | None = None,
) -> list[dict]:
    """
    生成批量评测题库。

    参数：
    - gold_chunk_ids：本次测试知识真实入库后得到的 chunk_id 列表
      这里必须用“真实入库后的 id”，因为后续指标计算就是拿这些 id 对比命中情况。
    - expected_item_names：预期主体列表；不传时默认使用 `TEST_ITEM_NAME`

    返回值：
    - list[dict]：批量评测题库

    每个问题都带 3 类标注：
    - `expected_item_names`：预期主体
    - `gold_chunk_ids`：相关 chunk
    - `must_hit_chunk_ids`：关键 chunk

    设计选择：
    - 测试知识数据集做大一点；
    - 测试问题保留 2 条代表性问题；
    - 避免靠堆很多相似问题制造“样本很多”的假象。
    """
    expected_names = expected_item_names or [TEST_ITEM_NAME]

    # 按知识主题给 chunk 做分组，后面写题目时更容易看懂。
    #
    # 当前 `gold_chunk_ids` 的顺序，默认和 `build_import_chunks()` 里的 part 一一对应：
    # - [0] -> part 1：局部烫印范围设置
    # - [1] -> part 2：局部烫印操作步骤
    # - [2] -> part 3：局部烫印注意事项
    # - [3] -> part 4：局部烫印参数保存
    # - [4] -> part 5：局部烫印常见偏移原因
    # - [5] -> part 6：局部烫印试印确认
    # - [6] -> part 7：局部烫印异常处理
    # - [7] -> part 8：局部烫印模式说明
    #
    # case 1 关注“参数设置 + 操作流程”，所以取前 4 条：
    # - part 1：范围设置
    # - part 2：操作步骤
    # - part 3：注意事项
    # - part 4：参数保存
    setting_and_step_chunks = gold_chunk_ids[:4]
    # case 2 关注“注意事项 + 异常排查”，所以取 part 3 ~ part 7：
    # - part 3：注意事项
    # - part 4：参数保存
    # - part 5：偏移原因
    # - part 6：试印确认
    # - part 7：异常处理
    caution_and_fix_chunks = gold_chunk_ids[2:7]

    return [
        {
            "case_id": "hak180_local_region_eval_001",
            "question": "HAK 180 在局部烫印时怎么设置 50mm 到 170mm 区域，设置后还要做什么？",
            "expected_item_names": expected_names,
            # `gold_chunk_ids` 表示“这道题所有算相关的标准答案范围”。
            # 只要检索命中了这里面的 chunk，都算命中相关内容。
            # 这一题关心的是设置方法、操作流程、注意点和保存动作，
            # 所以相关范围取 `setting_and_step_chunks`。
            "gold_chunk_ids": setting_and_step_chunks or gold_chunk_ids,
            # `must_hit_chunk_ids` 表示“这道题绝对不能漏掉的关键 chunk”。
            # 这一题最关键的是：
            # - part 1：范围设置
            # - part 2：操作步骤
            # 所以后面评估 `must_hit_rate` 时，会重点看这两条是否被召回。
            #
            # `or gold_chunk_ids[:1]` 是兜底逻辑：
            # 如果测试数据极少，切片后为空，至少保留第一条作为关键 chunk。
            "must_hit_chunk_ids": gold_chunk_ids[:2] or gold_chunk_ids[:1],
            "tags": ["售后", "参数设置", "操作步骤", "局部烫印"],
        },
        {
            "case_id": "hak180_local_region_eval_002",
            "question": "HAK 180 局部烫印前后要注意什么，出现位置偏移一般该检查哪些项目？",
            "expected_item_names": expected_names,
            # 这一题的标准相关范围更偏“注意事项 + 排查处理”。
            # 所以相关答案范围取：
            # - part 3：注意事项
            # - part 4：参数保存
            # - part 5：偏移原因
            # - part 6：试印确认
            # - part 7：异常处理
            "gold_chunk_ids": caution_and_fix_chunks or gold_chunk_ids,
            # 这一题最关键的是“注意什么”和“偏移为什么发生”。
            # 所以把 part 3 ~ part 5 设成必须命中的关键 chunk：
            # - part 3：注意事项
            # - part 4：参数保存
            # - part 5：偏移原因
            #
            # 这些 chunk 会用于计算 `must_hit_rate`：
            # 命中了几条关键 chunk / 一共配置了几条关键 chunk。
            "must_hit_chunk_ids": gold_chunk_ids[2:5] or gold_chunk_ids[:1],
            "tags": ["售后", "注意事项", "异常处理", "局部烫印"],
        },
    ]


def write_batch_eval_cases(case_list: list[dict]) -> None:
    """
    将题库写入磁盘。

    参数：
    - case_list：要保存的题库列表

    这样可以把“准备测试数据”和“执行评测”拆成两步。
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_BATCH_CASES_FILE.write_text(
        json.dumps(case_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_batch_eval_cases() -> list[dict]:
    """
    读取批量评测题库。

    返回值：
    - list[dict]：题库列表；如果文件不存在则返回空列表

    题库文件不存在时，通常说明还没先执行测试数据入库。
    """
    if not GENERATED_BATCH_CASES_FILE.exists():
        return []
    return json.loads(GENERATED_BATCH_CASES_FILE.read_text(encoding="utf-8"))
