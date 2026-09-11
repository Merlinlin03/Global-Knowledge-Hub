import json
from pathlib import Path
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNKS_SPLIT_TOP_NUMBER
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.infra.llm.providers import llm_providers
from app.infra.vector_store.milvus_gateway import milvus_gateway


# 1. 获取并且校验参数(state) -> chunks , file_title
#              1.1 获取参数 file_title chunks md_path
#              1.2 非空校验 file_title -> md_path chunks -> md_path / json -> 读取..
#              1.3 返回结果
@step_log("get_data_and_validates")
def get_data_and_validates(state:ImportGraphState) -> tuple[list[dict[str,Any]],str]:
    """
     获取并且校验
    :param state:
    :return:
    """
    # 1. 获取相关的参数
    md_path = state.get("md_path")
    file_title = state.get("file_title")
    chunks = state.get("chunks")
    # 2. 校验赋值
    if not file_title:
        if md_path and Path(md_path).exists():
            file_title =  Path(md_path).stem
        else:
            file_title = "default_title"
        logger.warning(f"file_title没有值,给与默认值:{file_title}")

    if not chunks:
        if md_path:
            # 获取json chunks的地址
            chunks_json_obj:Path = Path(md_path).with_name(f"{Path(md_path).stem}.json")
            chunks = json.loads(chunks_json_obj.read_text(encoding="utf-8"))
        if not chunks:
            logger.error(f"chunks为空,读取本地备份文件依然为空,业务无法继续进行!")
            raise ValueError(f"chunks为空,读取本地备份文件依然为空,业务无法继续进行!")

    return chunks,file_title

@step_log("recognize_item_name_by_chunks")
def recognize_item_name_by_chunks(chunks, file_title)->str:
    """
    识别item_name
    :param chunks:
    :param file_title:
    :return:
    """
    # 2.1 获取语言模型对象 chat()
    chain_client = llm_providers.chat()
    # 2.2 加载提示词字符串 .prompt ...
    system_prompt_text:str = load_prompt("product_recognition_system")
    user_context:str = ""
    """
    # 声明常量,切块的截取数量 默认: 10
    CHUNKS_SPLIT_TOP_NUMBER = 10
    """
    used_chunks = chunks[:CHUNKS_SPLIT_TOP_NUMBER]
    """
      context = 
         第1 2 3切块: 标题: title,内容: content \n
         chunk \n
         chunk \n
         chunk \n
    """
    for index,  chunk in enumerate(used_chunks,start=1):
        user_context += f"第{index}部分: 标题为:{chunk.get('title')} , 内容为: {chunk.get('content')} \n"
    user_prompt_text: str = load_prompt("item_name_recognition",file_title=file_title,context = user_context)
    # 2.3 封装成Message
    messages = [
        SystemMessage(
            content=system_prompt_text
        ),
        HumanMessage(
            content=user_prompt_text
        )
    ]
    # 2.4 封装langchain调用chains ...
    chains = chain_client | StrOutputParser()
    # 2.5 执行获取item_name
    item_name = chains.invoke(messages)
    # 2.6 非空校验 空 -> file_title兜底
    if not item_name:
        item_name = file_title
        logger.warning(f"模型未识别到item_name使用file_title:{file_title}赋予默认值!")
    # 2.7 返回item_name
    return item_name

@step_log("chunk_update_item_name")
def chunk_update_item_name(chunks, item_name):

    for chunk in chunks:
        chunk['item_name'] = item_name

    logger.info(f"完成chunk的item_name属性更新: {item_name}")

@step_log("prepared_milvus_item_name_collection")
def prepared_milvus_item_name_collection():
    """

    :return:
    """
    # 0.准备工作(客户端 + 集合名称)
    item_name_collection_name = milvus_gateway.item_name_collection_name
    milvus_client = milvus_gateway.milvus_client
    # 1.判断是否有集合
    has_item_name_collection =  milvus_client.has_collection(collection_name=item_name_collection_name)
    # 2.有直接返回
    if has_item_name_collection:
        # 有 True
        logger.info(f"{item_name_collection_name}集合已经存在无需创建!")
        return
    # 3.没有创建集合
    logger.info(f"{item_name_collection_name}集合不存在,开始创建!")
    # 创建: schema field  主键  索引..
    # 3.1 准备schema -> 这里可以添加field列信息
    schema = milvus_client.create_schema(
        auto_id=True, # 主键自增长
        enable_dynamic_field=True, # schema设置列的信息! True插入了没有提前设定好的列,也可以进行存储
    )
    # 列 名称 类型 长度限制..
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    # FLOAT_VECTOR -> 32 没有启动fp16加速 cpu或者gpu没有加速   FLOAT16_VECTOR -> gpu fp16
    # dim = 维度  -> 根据嵌入式模型生成的维度  bge-m3 1024
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 3.2 索引问题
    # 索引就是一种更高效的查询数据 (红黑树 / hash表 / 多图导航... / 倒排索引)
    # 索引不是必须的! 但是很有必要的! 频繁查询的列需要建立索引提高查询效率!
    # dense_vector  sparse_vector
    # 3.3. Prepare index parameters
    index_params = milvus_client.prepare_index_params()

    # 稠密
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW", # 稠密向量: 1. AUTOAINDEX (不明确底层实现原理) 2. FLAT  IVF_FLAT  HNSW
        # FLAT 蛮力 暴力 全盘搜索
        # IVF_FLAT 分类存储,每类有一个中心点,先比较中心点确定类别,在细化搜索 [准确 / 效率 比较中和]
        # HNSW 多层图导航 类似 地图 ... 逐层向下查找 .. (效率 / 准确 比较高 占有空间最大)
        metric_type="COSINE", # 稠密向量相似度可以选: COSINE = IP -> 速度更快一些 归一化   L2
        params = {
            "M": 64, # 每个点最大的链接数量
            "efConstruction": 100  # 考虑链接的范围 在这个范围内确定 M
        }
    )

    # 稀疏
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",  # 倒排索引
        metric_type="IP",  # 稠密向量相似度可以选: IP -> 推荐  BM25 -> 3.0 默认取消了
        params={"inverted_index_algo": "DAAT_MAXSCORE"} # 根据权重值做优化,降低一些低权重数据的排名!!!
    )

    # 创建集合
    milvus_client.create_collection(
        collection_name=item_name_collection_name,
        schema=schema,
        index_params=index_params
    )

@step_log("delete_and_insert_item_name")
def delete_and_insert_item_name(item_name, file_title):
    """
    完成item_name替换工作. ..
    :param item_name:
    :param file_title:
    :return:
    """
    # 1. 根据item_name生成稠密和稀疏向量(嵌入式模型)
    result = llm_providers.generate_embeddings([item_name])
    item_name_dense  = result.get("dense")[0]
    item_name_sparse = result.get("sparse")[0]
    # 2. 删除之前file_title对应就的数据
    milvus_client = milvus_gateway.milvus_client
    # 幂等操作
    milvus_client.delete(
        collection_name=milvus_gateway.item_name_collection_name,
        filter=f"file_title == '{file_title}'"
        #filter=f"file_title == 'hk180烫金机安全手册'"
    )
    # 插入数据
    data = [
        {
            "file_title":file_title,
            "item_name":item_name,
            "dense_vector":item_name_dense,
            "sparse_vector":item_name_sparse
        }
    ]
    milvus_client.insert(
        collection_name=milvus_gateway.item_name_collection_name,
        data = data
    )
    logger.info(f"完成{item_name}的数据更新或者插入!")

@step_log("recognize_and_index_item_name")
def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    """
    主体识别服务：
    1. 基于 chunks 构造上下文
    2. 调用 LLM 识别 item_name
    3. 将 item_name 回填到 state 和 chunks
    4. 同步写入主体名称索引
    """
    # 1. 获取并且校验参数(state) -> chunks , file_title
    chunks , file_title= get_data_and_validates(state)
    # 2. 识别item_name
    item_name = recognize_item_name_by_chunks(chunks,file_title)
    # 3. 给chunks的切块补全item_name属性
    chunk_update_item_name(chunks,item_name)
    # 4. 提前准备对应的集合数据
    prepared_milvus_item_name_collection()
    # 5. 插入item_name数据
    delete_and_insert_item_name(item_name, file_title)
    # 6. 更新state
    state['chunks'] = chunks
    state['item_name'] = item_name
    return state