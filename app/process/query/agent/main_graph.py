from langgraph.graph import StateGraph, END

from app.process.query.agent.nodes.node_answer_output import node_answer_output
from app.process.query.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.query.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger

# 1. 创建一个编译对象
query_graph_builder = StateGraph(QueryGraphState)

# 2.添加节点
query_graph_builder.add_node(node_item_name_confirm)
query_graph_builder.add_node(node_search_embedding)
query_graph_builder.add_node(node_search_embedding_hyde)
query_graph_builder.add_node(node_web_search_mcp)
query_graph_builder.add_node(node_rrf)
query_graph_builder.add_node(node_rerank)
query_graph_builder.add_node(node_answer_output)

# 3.添加入口节点+条件边
query_graph_builder.set_entry_point("node_item_name_confirm")

# 路由函数
def after_node_item_name_confirm(state:QueryGraphState):
    """
       判断item_names有 [明确]或者 没有 [可选 | 不确定]
       判断answer 没有 [明确] 或者 有 [可选 | 不确定]
    :param state:
    :return:
    """
    if not state.get("answer"):
        # 没有answer => 有明确item_names 可以继续多路召回
        logger.info(f"本次问题有明确的item_names:{state.get('item_names')},正常进入多路召回~")
        return "node_search_embedding","node_search_embedding_hyde","node_web_search_mcp"
    else:
        logger.info(f"本次问题没有明确的item_names,跳到回答node_answer_output节点~")
        return "node_answer_output"

query_graph_builder.add_conditional_edges("node_item_name_confirm",after_node_item_name_confirm,{
    "node_search_embedding":"node_search_embedding",
    "node_search_embedding_hyde":"node_search_embedding_hyde",
    "node_web_search_mcp":"node_web_search_mcp",
    "node_answer_output":"node_answer_output"
})

# 4.添加静态边
query_graph_builder.add_edge("node_search_embedding","node_rrf")
query_graph_builder.add_edge("node_search_embedding_hyde","node_rrf")
query_graph_builder.add_edge("node_web_search_mcp","node_rrf")
query_graph_builder.add_edge("node_rrf","node_rerank")
query_graph_builder.add_edge("node_rerank","node_answer_output")
query_graph_builder.add_edge("node_answer_output",END)

# 5.编译对象
query_app = query_graph_builder.compile()
