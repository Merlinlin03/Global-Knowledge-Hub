import sys

from app.shared.runtime.logger import node_log
from app.rag.query.hyde_search_service import search_by_hyde
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    hyde_embedding_chunks = search_by_hyde(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {
        "hyde_embedding_chunks":hyde_embedding_chunks
    }

if __name__ == "__main__":
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False,
    }
    result = node_search_embedding_hyde(mock_state)
    print(result)