from typing import Any

from app.infra.config.providers import infra_config
from app.shared.clients import get_milvus_client, create_hybrid_search_requests, hybrid_search


class MilvusGateway:
    # 提供两个属性 集合名
    @property
    def chunk_collection_name(self):
        return infra_config.milvus_config.chunks_collection
    # 新添加
    @property
    def item_collection_name(self):
        return infra_config.milvus_config.item_name_collection

    # 提供一个获取客户端
    @property
    def item_name_collection_name(self):
        return infra_config.milvus_config.item_name_collection

    @property
    def milvus_client(self):
        return get_milvus_client()

    # 新添加
    @property
    def client(self):
        return get_milvus_client()
        # 新引入

    def create_requests(
            self,
            dense_vector: list[float],
            sparse_vector: dict[int, float],
            expr: str = None,
            limit: int = 5,
    ):
        return create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=expr,
            limit=limit,
        )

    def hybrid_search(
            self,
            *,
            collection_name: str,
            reqs: list[Any],
            ranker_weights: tuple[float, float] = (0.5, 0.5),
            norm_score: bool = False,
            limit: int = 5,
            output_fields: list[str] | None = None,
            search_params: dict | None = None,
    ):
        return hybrid_search(
            client=self.milvus_client,
            collection_name=collection_name,
            reqs=reqs,
            ranker_weights=ranker_weights,
            norm_score=norm_score,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params,
        )

milvus_gateway = MilvusGateway()