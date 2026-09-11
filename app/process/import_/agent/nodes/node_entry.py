import json

from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.entry_service import resolve_input_file

@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    """
    add_running_task(state["task_id"], "node_entry")
    # 调用业务逻辑
    state = resolve_input_file(state)
    add_done_task(state["task_id"], "node_entry")
    return state



if __name__ == '__main__':
    from app.shared.runtime.logger import logger
    from app.process.import_.agent.state import create_default_state
    # 单元测试：覆盖不支持类型、MD、PDF三种场景
    logger.info("===== 开始node_entry节点单元测试 =====")

    # 测试1: 不支持的TXT文件
    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="联想海豚用户手册.txt"
    )
    result_1 =  node_entry(test_state1)
    print(f"第一次测试结果: \n {json.dumps(result_1, indent=4, ensure_ascii=False)}")
    # 测试2: MD文件
    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="小米用户手册.md"
    )
    result_2 = node_entry(test_state2)
    print(f"第二次测试结果: \n {json.dumps(result_2, indent=4, ensure_ascii=False)}")
    # 测试3: PDF文件
    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="万用表的使用.pdf"
    )
    result_3 = node_entry(test_state3)

    print(f"第三次测试结果: \n {json.dumps(result_3, indent=4, ensure_ascii=False)}")

    logger.info("===== 结束node_entry节点单元测试 =====")