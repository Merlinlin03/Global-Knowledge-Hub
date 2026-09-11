from app.shared.config.embedding_config import embedding_config, EmbeddingConfig
from app.shared.config.lm_config import lm_config, LLMConfig
from app.shared.config.bailian_mcp_config import mcp_config, McpConfig
from app.shared.config.milvus_config import milvus_config, MilvusConfig
from app.shared.config.mineru_config import mineru_config, MinerUConfig
from app.shared.config.minio_config import minio_config, MinIOConfig
from app.shared.config.reranker_config import reranker_config, RerankerConfig
from app.shared.config.settings_config import settings, AppSettings

from dataclasses import dataclass , field

# 创建一个类型 赋值对应的config对象! 导入当前一个类型即可获取所有config

@dataclass
class InfraConfig:
    # 属性名 : 类型 object = 默认值
    embedding_config:EmbeddingConfig = field(default_factory=lambda : embedding_config)  # 复制了一个对象
    # lm_config:LLMConfig=field(default=lm_config) = lm_config:LLMConfig=lm_config
    lm_config:LLMConfig=field(default_factory=lambda :lm_config)
    mcp_config:McpConfig=field(default_factory=lambda : mcp_config)
    milvus_config:MilvusConfig=field(default_factory=lambda : milvus_config)
    mineru_config:MinerUConfig=field(default_factory=lambda : mineru_config)
    minio_config:MinIOConfig=field(default_factory=lambda : minio_config)
    reranker_config:RerankerConfig=field(default_factory=lambda : reranker_config)
    settings:AppSettings=field(default_factory=lambda : settings)
    #name:str="哈哈哈"

infra_config = InfraConfig()

# python的安全机制!!
# InfraConfig -> 属性 = 默认值 -> 另外一个模块中的对象 mineru_config
# InfraConfig -> embedding_config -> 对象
#                                            每个模块 不同的类型 指向同一个对象 地址引用
# embedding_config -> embedding_config -> 对象

# lm_config -> 对象 | [] {} ->  field(default_factory=lambda : embedding_config)  # 复制了一个对象
#           -> str bool 数字 () -> 可以直接复制

if __name__ == "__main__":
    print(infra_config.lm_config.api_key)
    print(infra_config.mcp_config.mcp_base_url)