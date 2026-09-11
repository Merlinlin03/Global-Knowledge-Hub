import sys

from app.shared.runtime.logger import node_log
from app.rag.query.rerank_service import rerank_documents
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_rerank")
def node_rerank(state):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    state = rerank_documents(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    mock_rrf_chunks = [
        {"chunk_id": "local_1", "content": "二大爷在家里也算比较亲的! 因为他是爸爸的亲哥哥!有血缘关系!", "title": "算法介绍"},
        {"chunk_id": "local_2", "content": "大舅也算比较轻的!因为他是妈妈的亲哥哥!也有血缘关系!俗话说得好:娘亲舅大!", "title": "模型介绍"},
        {"chunk_id": "local_3", "content": "老舅", "title": "模型介绍"},
        {"chunk_id": "local_4", "content": "大姑", "title": "模型介绍"},
        {"chunk_id": "local_5", "content": "那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打那收到货撒旦撒旦极客飒登记卡萨达卡稍等哈时间跨度哈手机卡等哈手机卡等哈手打", "title": "模型介绍"},
    ]
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "老姨"},
    ]
    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "请问哪种亲戚关系更近？",
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False,
    }
    result = node_rerank(mock_state)
    print(result)