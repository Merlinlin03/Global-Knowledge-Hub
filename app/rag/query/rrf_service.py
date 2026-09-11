import time

from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import NODE_RRF_K, NODE_RRF_LIMIT_TOP
from app.shared.runtime.logger import logger, step_log

@step_log("get_data_and_validates")
def get_data_and_validates(state:QueryGraphState):
    #1.获取参数
    embedding_chunks = state.get("embedding_chunks",[])
    hyde_embedding_chunks = state.get("hyde_embedding_chunks",[])
    # 2.非空校验
    if len(embedding_chunks) == 0 or len(hyde_embedding_chunks) == 0:
        logger.error(f"embedding_chunks或者hyde_embedding_chunks数据为空,业务无法继续进行,提前终止!")
        raise ValueError(f"embedding_chunks或者hyde_embedding_chunks数据为空,业务无法继续进行,提前终止!")
    # 3. 返回结果
    return embedding_chunks,hyde_embedding_chunks

@step_log("use_by_rrf")
def use_by_rrf(rrf_list:list,top:int=NODE_RRF_LIMIT_TOP,k:int=NODE_RRF_K):
    """
       进行权重排名计算
    :param rrf_list:
    :param top:
    :param k:
    :return:
    """
    # 1.提前准备两个字典
    score_dict:dict[str,float] = {}  # chunk_id , 0.56
    chunk_dict:dict[str,dict]  = {}  # chunk_id , chunk [去重复]
    # 2. 先循环每一路 rrf_list = [(1.0,[]),(1.0,[])]
    for weight,current_chunks_list in rrf_list:
        for rank,chunk in enumerate(current_chunks_list,start=1):
            # rrf => 排名 -> 循环的顺序就是排名的顺序
            # chunk => {id,title,content,item_name,score,type,url...}
            # 上一次计算的得分  + 权重 * 1 / k + rank
            # todo
            score_dict[chunk.get('chunk_id')] = score_dict.get(chunk.get("chunk_id"),0.0)+weight * (1/(k+rank))
            # chunk_dict[chunk.get('chunk_id')] = chunk # 每次覆盖 保留最后一次数据
            chunk_dict.setdefault(chunk.get('chunk_id'),chunk)  # 每次检查 保留第一次数据 [优雅]
    # 3. 分已经计算完毕了
    # score_dict = {chunk_id:分,chunk_id:分 至多10个 至少5个} -> 没有排名吧
    # chunk_dict = {chunk_id:chunk score = 分  -> 排名 -> topk -> rrf_chunks  }
    chunk_list = []
    for chunk_id,score in score_dict.items():
        # chunk_id == 分
        chunk = chunk_dict.get(chunk_id)
        chunk["score"] = score # milvus打的分替换成rrf排序的分析
        chunk_list.append(chunk)
        # chunk_dict = {chunk_id:chunk -> score}
    #4. 排序处理
    chunk_list.sort(key = lambda x : x.get('score',0),reverse=True)
    #5. 截取最高的topk
    rrf_chunks = chunk_list[:top]
    return rrf_chunks

@step_log("fuse_by_rrf")
def fuse_by_rrf(state: QueryGraphState) -> QueryGraphState:
    """
    RRF 融合服务：
    1. 合并来自不同检索源的文档列表
    2. 应用 RRF 算法消除分数差异
    3. 给出综合排名最高的文档列表（Top 10）
    4. 回写 rrf_chunks
    """
    # 1. 获取并且校验参数 (state )embedding_chunks  hyde_embedding_chunks
    embedding_chunks , hyde_embedding_chunks = get_data_and_validates(state)
    # 2. 封装数据结构 -> list -> [( 权重,list -> 有顺序 -> 向量数据库查询 -> 分高),(权重,list)]
    # 目标方便遍历和获取对应的权重
    rrf_list = [(1.0,embedding_chunks),(1.0,hyde_embedding_chunks)]
    # 3. 使用rrf算法进行数据处理( list) -> rrf_chunks[5]
    rrf_chunks = use_by_rrf(rrf_list)
    # 4. 修改state = rrf_chunks
    state['rrf_chunks']=rrf_chunks
    # 5. 返回state
    return state