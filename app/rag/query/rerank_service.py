import time

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import RERANK_MAX_INPUT_TOKENS, RERANK_SUMMARY_CHAR_RATIO, RERANK_MIN_SUMMARY_CHARS, \
    RERANK_MAX_TOPK, RERANK_MIN_TOPK, RERANK_GAP_ABS, RERANK_GAP_RATIO
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.infra.llm.providers import llm_providers

#   1.获取并且校验参数(state) rewritten_query  rrf_chunks  web_search_docs
#            2.数据格式化处理   rrf_chunks {chunk_id,title,parent_title,part,file_title,content,item_name,score,type,url}
#                             web_search_docs {snippt , title,url}
#                             -> 一种格式 -> 给模型了
#                没有    / chunk_id -> 数据库的有的标识
#                snippet / content -> text : 回答参考内容
#                title  / title   -> title : 标题
#                没有    / score   -> score : web 0  milvus rrf的分 -> reranker打分
#                没有    / type    -> mcp web  数据库 milvus
#                url    /  没用    -> url [图片地址]
#                reranker_list
#            3.组装问题和答案的列表(rewritten_query,reranker_list) -> question_answer_pair_list [[],[],[]]
#                获取问题
#                判断问题token的长度
#                循环reranker_list获取答案 text
#                   判断text的长度
#                     超
#                        模型压缩 -> 调用..
#                   装数据pair
#                返回结果
#            4. reranker模型打分+排序
#                reranker.compute_score([question_answer_pair_list [问题,答案]]) -> [scores]
#                [scores] -> reranker_list { score : x }  -> zip
#                sort排序
#                reranker_list {text 没有压缩} -> 分数 + 排序
#            5. 动态topk截取数据
#                reranker_list  min  max  topk
#                  1 2 3 4 5 6 7 8
#                    断崖值 0.3
@step_log("get_data_and_validates")
def get_data_and_validates(state):
    #1.获取参数
    rewritten_query = state.get("rewritten_query")
    rrf_chunks = state.get("rrf_chunks",[])
    web_search_docs = state.get("web_search_docs",[])
    #2.非空判断
    if not rewritten_query or len(rrf_chunks) == 0 or len(web_search_docs) == 0:
        logger.error(f"rewritten_query,rrf_chunks,web_search_docs可能为空,业务无法继续进行,提前终止!")
        raise ValueError(f"rewritten_query,rrf_chunks,web_search_docs可能为空,业务无法继续进行,提前终止!")
    return rewritten_query,rrf_chunks,web_search_docs

@step_log("deal_rrf_and_web_result")
def deal_rrf_and_web_result(rrf_chunks, web_search_docs):
    # 1. 定义一个列表
    reranker_docs = []
    # 2. 先循环rrf
    for chunk in rrf_chunks:
        reranker_docs.append({
            "chunk_id":chunk.get("chunk_id"),
            "text":chunk.get("content"),
            "title":chunk.get("title"),
            "score":0, # rrf分 -> reranker打的分
            "type":"milvus",
            "url":None
        })
    # 3. 再循环web_search
    for doc in web_search_docs:
        reranker_docs.append({
            "chunk_id": None,
            "text": doc.get("snippet"),
            "title": doc.get("title"),
            "score": 0,  # rrf分 -> reranker打的分
            "type": "web",
            "url": doc.get("url")
        })

    return reranker_docs

@step_log("create_question_answer_list")
def create_question_answer_list(rewritten_query, reranker_docs):
    question_answer_pair_list = []
    # 1. 获取rewritten_query并且计算token数量
    reranker_model =  llm_providers.reranker_model()
    tokenizer =  reranker_model.tokenizer
    # 算的时候,只需要算我这个字符占有token列表,不用考虑前后的特殊标识
    rewritten_query_tokens_list =  tokenizer.encode(rewritten_query,add_special_tokens=False)
    rewritten_query_token_len = len(rewritten_query_tokens_list)
    # 2. 循环reranker_docs获取每个text答案
    for doc in reranker_docs:
        # 3. 答案的长度判读
        answer = doc.get("text") # 答案
        answer_token_len = len(tokenizer.encode(answer,add_special_tokens=False))
        # 4. 超长了调用模型进行压缩
        # reranker固定4个分割符号
        if rewritten_query_token_len + answer_token_len + 4 > RERANK_MAX_INPUT_TOKENS:
            # 调用模型进行压缩
            # limit = 答案的token / 1.3 -> int -> 50 max
            limit = max(
                RERANK_MIN_SUMMARY_CHARS,
                int((RERANK_MAX_INPUT_TOKENS - 4 - rewritten_query_token_len) / RERANK_SUMMARY_CHAR_RATIO))
            # 加载提示词
            rerank_text_refine_str =  load_prompt("rerank_text_refine",question=rewritten_query,answer=answer,limit=limit)
            # 封装message
            messages = [
                HumanMessage(
                    content=rerank_text_refine_str
                )
            ]
            # 封装调用链
            chains = llm_providers.chat() | StrOutputParser()
            # 执行获取结果
            answer = chains.invoke(messages)
        # 5. 答案一定处理过了
        # question_answer_pair_list answer -> 可能被压缩 -> 只用于打分
        question_answer_pair_list.append([rewritten_query,answer])
    # 6. 返回结果
    return question_answer_pair_list


@step_log("use_reranker_deal_score")
def use_reranker_deal_score(question_answer_pair_list, reranker_docs):
    # 1. 调用reranker打分
    reranker_model = llm_providers.reranker_model()
    # normalize=True 归一化 避免负分  0 - 1分之间
    scores_list =  reranker_model.compute_score(question_answer_pair_list,normalize=True)
    # scores_list == question_answer_pair_list == reranker_docs {score}
    # 2. 同步遍历 分 -> reranker_docs
    for score , doc in zip(scores_list,reranker_docs):
        doc["score"] = score
    # 3. 倒序排序
    reranker_docs.sort(key=lambda x : x.get("score",0),reverse=True)

@step_log("dyn_limit_reranker_docs")
def dyn_limit_reranker_docs(reranker_docs):
    """
      动态结果截取! topk个
         RERANK_MAX_TOPK: int = 5  -> 最多10个
         RERANK_MIN_TOPK: int = 2  -> 最少2个
         RERANK_GAP_RATIO: float = 20% -> 断崖百分比  ->  1 - 2 / 1     0.3 0.2 -> (0.3 - 0.2) / 0.3 = 33%
         RERANK_GAP_ABS: float = 0.2   -> 断崖分差值  ->  0.8  ->  0.5  跳过  大 多了 影响准确了  小  少了 召回率
      topk -> ???
    :param reranker_docs:
    :return:
    """
    # 第一个版本! 只考虑 前后指针断崖判断即可
    top_max:int=RERANK_MAX_TOPK
    top_min:int=RERANK_MIN_TOPK
    gap_abs:float=RERANK_GAP_ABS # 0.2
    gap_ratio:float=RERANK_GAP_RATIO # 0.2
    # todo: 累计断崖 第一个(min-1)-> 最新一个判断
    #情况1: max > len -> max = len
    top_max = min(top_max,len(reranker_docs))
    # 情况3: topk可能没有值 场景1: min > max  场景2: 没有断崖
    # 准备topk 要动态截取的数量
    topk:int= top_max
    # 情况2: min > max  正常人 min 小于 max ! max = len
    if top_max > top_min:
        # 循环指针
        # pre_index 从最小值开始
        #       到max的前一个(top_max-2)
        for pre_index in range(top_min-1,top_max-1):
            # 获取pre_index对应前置分数
            pre_score = reranker_docs[pre_index].get("score",0.0)
            # 获取next_index对应后置分数
            next_score = reranker_docs[pre_index+1].get("score", 0.0)
            # 分差
            abs_score = pre_score - next_score
            ratio     = abs_score / pre_score
            # 判断断崖
            if abs_score > gap_abs or ratio > gap_ratio:
                # 出现了断崖!
                # 就可以截取到前置指针的位置
                topk = pre_index + 1
                break
    # 获取top k (动态)
    final_reranker_docs = reranker_docs[:topk]
    # 返回结果
    return final_reranker_docs


@step_log("rerank_documents")
def rerank_documents(state: QueryGraphState) -> QueryGraphState:
    """
    重排序服务：
    1. 合并 RRF 和 Web Search 的文档
    2. 使用 BGE Reranker 模型计算相关性得分
    3. 根据得分动态截断，智能截取 TopK
    4. 回写 reranked_docs
    """
    # 1.获取并且校验参数(state) rewritten_query  rrf_chunks  web_search_docs
    rewritten_query,rrf_chunks,web_search_docs = get_data_and_validates(state)
    # 2. 数据格式化处理
    reranker_docs = deal_rrf_and_web_result(rrf_chunks,web_search_docs)
    # 3. 组装问题和答案的列表(rewritten_query,reranker_list) -> question_answer_pair_list [[],[],[]]
    question_answer_pair_list:list[list[str]] = create_question_answer_list(rewritten_query,reranker_docs)
    # 4. 打分+排序
    logger.info(f"排序和打分之前的数据:{reranker_docs}")
    use_reranker_deal_score(question_answer_pair_list,reranker_docs)
    logger.info(f"排序和打分之后的数据:{reranker_docs}")
    # 5. 动态截取数据
    reranker_docs = dyn_limit_reranker_docs(reranker_docs)
    # 6. 更新state
    state["reranked_docs"] = reranker_docs
    return state