"""
MCP 配置模块，负责读取联网检索相关环境变量。
"""
from dataclasses import dataclass

from app.shared.config.common import env_str


@dataclass
class McpConfig:
    mcp_base_url: str
    api_key: str

# 不使用 @dataclass 的普通类写法（等价效果）
# class McpConfig:
#     # 初始化方法
#     def __init__(self, mcp_base_url: str, api_key: str):
#         self.mcp_base_url = mcp_base_url  # 实例属性赋值
#         self.api_key = api_key           # 实例属性赋值

mcp_config = McpConfig(
    mcp_base_url=env_str("MCP_DASHSCOPE_BASE_URL"),
    api_key=env_str("OPENAI_API_KEY"),
)
