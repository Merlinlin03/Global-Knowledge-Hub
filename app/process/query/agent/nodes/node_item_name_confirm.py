import json
import sys

from app.shared.runtime.logger import node_log
from app.rag.query.item_name_confirm_service import confirm_item_name
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.infra.persistence.history_repository import history_repository

@node_log("node_item_name_confirm")
def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心业务主体。
    输入：state['original_query']
    输出：更新 state['item_names']
    """
    # 先登记节点开始，前端进度区可以立即感知"主体确认"已启动。
    add_running_task(state["session_id"], "node_item_name_confirm", state["is_stream"])
    # 调用 rag/query service 层
    state = confirm_item_name(state)
    # 识别完成后写入完成列表，方便前端展示当前节点已结束。
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    return state


if __name__ == "__main__":
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "他怎么用？",
        "is_stream": False,
    }
    result_state = node_item_name_confirm(mock_state)
    print(result_state)
