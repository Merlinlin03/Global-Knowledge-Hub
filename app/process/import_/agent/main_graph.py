from dotenv import load_dotenv
from langgraph.constants import START
from langgraph.graph import StateGraph, END

from app.process.import_.agent.state import ImportGraphState
from app.process.import_.agent.nodes.node_entry import node_entry
from app.process.import_.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.process.import_.agent.nodes.node_md_img import node_md_img
from app.process.import_.agent.nodes.node_document_split import node_document_split
from app.process.import_.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.process.import_.agent.nodes.node_bge_embedding import node_bge_embedding
from app.process.import_.agent.nodes.node_import_milvus import node_import_milvus
from app.shared.runtime.logger import logger

# 1. 定义图的构建对象,并且指定全局state
import_graph_builder = StateGraph(ImportGraphState)

# 2. 添加图节点
# import_graph_builder.add_node("node_entry",node_entry)
import_graph_builder.add_node(node_entry)
import_graph_builder.add_node(node_pdf_to_md)
import_graph_builder.add_node(node_md_img)
import_graph_builder.add_node(node_document_split)
import_graph_builder.add_node(node_item_name_recognition)
import_graph_builder.add_node(node_bge_embedding)
import_graph_builder.add_node(node_import_milvus)

# 3. 设置起始节点
# import_graph_builder.add_edge(START,"node_entry")
import_graph_builder.set_entry_point("node_entry")

# 4. 起始节点后的条件边设置
def node_entry_after(state:ImportGraphState) -> str:
    """
      判断类型文件  is_md_read_enabled = True  or  is_pdf_read_enabled = True
    :param state:
    :return: 目标节点名称
    """
    if state.get("is_md_read_enabled",False) :
        # 是md
        # 日志: 核心点
        # 日志: 交代清楚 有理有据有目标
        logger.info(f"传入的文件地址:{state.get('local_file_path')} , 判断传入的文件是md类型,所以跳转到node_md_img")
        return "node_md_img"
    elif state.get("is_pdf_read_enabled",False):
        # 是pdf
        logger.info(f"传入的文件地址:{state.get('local_file_path')} , 判断传入的文件是pdf类型,所以跳转到node_pdf_to_md")
        return "node_pdf_to_md"
    else:
        # 不是能处理的类型
        # debug info warning error
        logger.warning(f"传入的文件地址:{state.get('local_file_path')} , 不支持该文档类型处理! 只能处理 md / pdf格式数据!")
        return END

"""
  添加条件边
     参数1: 起始节点 str 节点名
     参数2: 路由函数 state -> 业务逻辑 -> return "节点名称","节点名称","节点名称"
     参数3: path_map dict 显示的配置路由关系! 供静态打印使用
"""
import_graph_builder.add_conditional_edges("node_entry",node_entry_after,{
    "node_md_img":"node_md_img",
    "node_pdf_to_md":"node_pdf_to_md",
    END:END
})

# 5. 设置静态边
import_graph_builder.add_edge("node_pdf_to_md","node_md_img")
import_graph_builder.add_edge("node_md_img","node_document_split")
import_graph_builder.add_edge("node_document_split","node_item_name_recognition")
import_graph_builder.add_edge("node_item_name_recognition","node_bge_embedding")
import_graph_builder.add_edge("node_bge_embedding","node_import_milvus")
import_graph_builder.add_edge("node_import_milvus",END)

# 6. 编译图对象
import_app = import_graph_builder.compile()

# 执行图对象...

if __name__ == "__main__":
    from app.shared.utils.path_util import PROJECT_ROOT
    import os
    from app.shared.runtime.logger import logger

    # 全流程测试：验证PDF导入→Milvus入库完整链路
    logger.info("===== 开始执行知识图谱导入全流程测试 =====")

    # 1. 构造测试文件路径（复用你项目的doc目录）
    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 2. 构造输出目录（存放MD/图片等中间文件）
    test_output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(test_output_dir, exist_ok=True)  # 不存在则创建

    # 3. 校验测试PDF文件是否存在
    if not os.path.exists(test_pdf_path):
        logger.error(f"全流程测试失败：测试PDF文件不存在，路径：{test_pdf_path}")
        logger.info("请检查文件路径，或手动将测试文件放入项目根目录的doc文件夹中")
    else:
        # 4. 构造测试状态（贴合实际业务入参，开启PDF解析开关）
        test_state = ImportGraphState({
            "task_id": "test_kg_import_workflow_001",  # 测试任务ID
            "local_file_path": test_pdf_path,  # 测试PDF文件路径
            "local_dir": test_output_dir,  # 中间文件输出目录
            "is_pdf_read_enabled": False,  # 开启PDF解析（核心开关）
            "is_md_read_enabled": False  # 关闭MD解析
        })
        try:
            logger.info(f"测试任务启动，PDF文件路径：{test_pdf_path}")
            logger.info(f"中间文件输出目录：{test_output_dir}")
            logger.info("开始执行全流程节点，依次执行：entry→pdf2md→md_img→split→item_name→embedding→milvus")

            # 5. 执行LangGraph全流程（流式执行，打印节点执行进度）
            final_state = None
            for step in import_app.stream(test_state, stream_mode="values"):
                # 打印当前执行完成的节点（流式输出更直观）
                current_node = list(step.keys())[-1] if step else "未知节点"
                logger.info(f"✅ 节点执行完成：{current_node}")
                final_state = step  # 保存最终状态

            # 6. 全流程执行完成，结果预览和核心指标打印
            if final_state:
                logger.info("-" * 80)
                logger.info("===== 全流程测试执行成功，核心结果预览 =====")

                # 提取核心结果指标
                chunks = final_state.get("chunks", [])
                chunk_count = len(chunks)
                md_content = final_state.get("md_content", "")[:150]  # MD内容前150字符
                item_name = final_state.get("item_name", "未识别")  # 主体名称
                has_embedding = all("dense_vector" in c and "sparse_vector" in c for c in chunks) if chunks else False
                has_chunk_id = all("chunk_id" in c for c in chunks) if chunks else False

                # 打印核心指标
                logger.info(f"📄 PDF转MD内容预览（前150字符）：{md_content}...")
                logger.info(f"🏷️  识别的主体名称：{item_name}")
                logger.info(f"📝 文档切分总切片数：{chunk_count}")
                logger.info(f"🔍 所有切片是否完成向量化：{'是' if has_embedding else '否'}")
                logger.info(f"🗄️  所有切片是否完成Milvus入库（含chunk_id）：{'是' if has_chunk_id else '否'}")
                logger.info(f"📂 最终状态包含的核心键：{list(final_state.keys())}")
                logger.info("-" * 80)
        except Exception as e:
            logger.exception(f"===== 全流程测试运行失败 =====")
    logger.info("===== 知识图谱导入全流程测试结束 =====")