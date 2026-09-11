import json
from pathlib import Path
from typing import Any

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import EMBEDDING_BATCH_SIZE
from app.shared.runtime.logger import logger, step_log
from app.infra.llm.providers import llm_providers

@step_log("get_data_and_validates")
def get_data_and_validates(state:ImportGraphState) -> tuple[list[dict[str,Any]],str]:
    """
     获取并且校验
    :param state:
    :return:
    """
    # 1. 获取相关的参数
    md_path = state.get("md_path")
    item_name = state.get("item_name")
    chunks = state.get("chunks")
    # 2. 校验赋值
    if not item_name:
        if md_path and Path(md_path).exists():
            item_name =  Path(md_path).stem
        else:
            item_name = "default_item_name"
        logger.warning(f"item_name没有值,给与默认值:{item_name}")

    if not chunks:
        if md_path:
            # 获取json chunks的地址
            chunks_json_obj:Path = Path(md_path).with_name(f"{Path(md_path).stem}.json")
            chunks = json.loads(chunks_json_obj.read_text(encoding="utf-8"))
        if not chunks:
            logger.error(f"chunks为空,读取本地备份文件依然为空,业务无法继续进行!")
            raise ValueError(f"chunks为空,读取本地备份文件依然为空,业务无法继续进行!")

    return chunks,item_name

@step_log("batch_generate_embeddings")
def batch_generate_embeddings(chunks:list[dict[str,Any]], item_name:str, batch_number:int = EMBEDDING_BATCH_SIZE) -> list[dict[str,Any]]:
    """
    批量生成
    :param chunks:
    :param item_name:
    :param batch_number:
    :return:
    """
    logger.info(f"未生成向量之前的示例:{chunks[0]}")
    # 1. 获取chunks长度
    length = len(chunks)
    # 2. 循环遍历
    # length = 20
    # 0 - 19 0 1 2 3 4 ... 19
    # 0 6 12
    for index in range(0,length,batch_number):
        # 获取本次批量chunks
        # [{1},{2},{3}]
        current_chunks = chunks[index:index+batch_number]
        # 进行拼接 item_name + content
        # [item_name+1,+2,+3]
        current_content_list = [ f"主体:{item_name},内容:{chunk.get('content')}" for chunk in current_chunks]
        # 批量生成向量
        # result =  {dense: [[],[],[]] , sparse: [{},{},{}]}
        result = llm_providers.generate_embeddings(current_content_list)
        # 回填 current_chunks
        for i,current_chunk in enumerate(current_chunks):
            current_chunk['dense_vector'] = result.get("dense")[i]
            current_chunk['sparse_vector'] = result.get("sparse")[i]
    logger.info(f"生成向量之后的示例:{chunks[0]}")
    return chunks

@step_log("generate_chunk_embeddings")
def generate_chunk_embeddings(state: ImportGraphState) -> ImportGraphState:
    """
    向量化服务：
    1. 读取 chunks
    2. 生成 dense_vector / sparse_vector
    3. 将向量结果补充回 chunks
    """
    #   1. 获取并校验参数(state) -> chunks / item_name
    chunks,item_name=get_data_and_validates(state)

    # 2. 批量进行向量生成 (chunks , item_name , batch_number:int = 6) -> chunks
    embeddings_content = batch_generate_embeddings(chunks,item_name,EMBEDDING_BATCH_SIZE)
    #3. 更新state embeddings_content
    state['embeddings_content'] = embeddings_content
    return state