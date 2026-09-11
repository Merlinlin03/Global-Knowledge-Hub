import sys

from app.shared.runtime.logger import node_log
from app.rag.query.web_search_service import search_by_web
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_web_search_mcp")
def node_web_search_mcp(state):
    """
    节点功能：调用外部搜索引擎补充信息
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    web_search_docs = search_by_web(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    return {
        "web_search_docs":web_search_docs
    }

if __name__ == "__main__":
    test_state = {
        "session_id": "xxxx",
        "is_stream": False,
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
    }
    result_state = node_web_search_mcp(test_state)
    print(result_state)