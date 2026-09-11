import time

from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger
from app.infra.llm.providers import llm_providers
from app.infra.vector_store.milvus_gateway import milvus_gateway

def get_data_and_validates(state:QueryGraphState)-> tuple[list[str],str]:
    """"""
    # 1. 获取数据
    item_names = state.get("item_names",[])
    rewritten_query = state.get("rewritten_query")
    # 2. 非空校验
    if len(item_names) == 0 or not rewritten_query:
        logger.error(f"关联的主体或者重写的问题为空,业务无法继续,提前终止!")
        raise ValueError(f"关联的主体或者重写的问题为空,业务无法继续,提前终止!")
    # 3. 返回结果
    return item_names,rewritten_query


def search_by_milvus(item_names:list[str], rewritten_query:str):
    # 1. rewritten_query进行向量化
    result =  llm_providers.generate_embeddings([rewritten_query])
    # 2. 组装reqs请求列表(AnnSearchRequest)
    reqs = milvus_gateway.create_requests(
        dense_vector=result['dense'][0],
        sparse_vector=result['sparse'][0],
        expr=f"item_name in  {item_names}",
        limit=5*2
    )
    # 3. 进行混合检索处理
    milvus_result =  milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name,
        reqs=reqs,
        ranker_weights=(0.6,0.4),
        norm_score=True,
        limit=5,
        output_fields=[
            # chunk_id file_title title parent_title part item_name content x x
            #  {} -> lm -> 润色
            "chunk_id",
            "file_title",
            "title",
            "parent_title",
            "part",
            "item_name",
            "content"
        ]
    )
    # milvus_result = [ [id/chunk_id:主键,distance:分,entity:{}] ] -> 外面没有意义! 保证对称性 单列检索
    # 4. 返回结果
    return milvus_result[0] if milvus_result and len(milvus_result) >0 else []


def deal_milvus_list(milvus_list):
    # {id / chunk_id,distance,entityL{}} -> {}
    embedding_chunks = []
    for item in milvus_list:
        entity = item.get("entity",{})
        embedding_chunks.append({
            "chunk_id": entity.get("chunk_id") ,  # item.get("id") or item.get("chunk_id")
            "score": item.get("distance",0.0),
            "title":entity.get("title"),
            "file_title":entity.get("file_title"),
            "parent_title":entity.get("parent_title"),
            "part":entity.get("part"),
            "item_name":entity.get("item_name"),
            "content":entity.get("content"),
            "source": "milvus",  # 直接查询 或者假设性查询  milvus 网络检索 web
            "url":""
        })
    return embedding_chunks

def search_by_embedding(state: QueryGraphState) -> QueryGraphState:
    """
    向量检索服务：
    1. 根据改写后的问题和限定的商品范围
    2. 利用 BGEM3 混合检索（稠密+稀疏）技术
    3. 从 Milvus 向量数据库中召回 Top-K 最相关的知识切片
    4. 回写 embedding_chunks
    """
    # 1. 获取并校验参数(state) -> item_names rewritten_query
    item_names,rewritten_query = get_data_and_validates(state)
    # 2. 进行向量的混合+条件检索(item_names rewritten_query) -> [{id/chunk_id:x,distance:0.9,entity:{输出的field} }]
    milvus_list = search_by_milvus(item_names,rewritten_query)
    # 3. 进行结果的统一格式化处理( [{id/chunk_id:x,distance:0.9,entity:{输出的field} }]) -> [{id:x,输出的field,score:,type:milvus,url:""},{},{}]
    embedding_chunks = deal_milvus_list(milvus_list)
    # 4. 返回结果即可
    return embedding_chunks