# Global Knowledge Hub

Global Knowledge Hub 是一个面向出海 App 运营场景的 RAG 知识库平台，用于沉淀海外产品运营、用户支持、订阅退款、广告体验、隐私合规、本地化 SOP 和版本发布等业务知识，并为运营助手、反馈分派、广告异常诊断等 Agent 模块提供可检索、可引用的知识依据。

## 项目定位

本项目聚焦“出海业务知识库 RAG 平台”这一模块，不是完整的 AI 出海运营大平台。它解决的问题是：当运营、客服或产品人员提出业务问题时，系统能够从内部知识文档中召回相关依据，结合多路检索、重排和回答约束，输出有边界的知识库答案。

## 核心能力

- 支持 PDF / Markdown 文档导入。
- 使用 MinerU 将 PDF 解析为 Markdown，并保留图片资源。
- 基于 Markdown 标题进行语义切分，并对过长/过短切片做二次处理。
- 使用 LLM 识别文档对应的业务主体，例如退款政策、用户反馈 SOP、广告投诉处理规范、隐私合规 FAQ。
- 使用 BGE-M3 生成 dense / sparse 向量，写入 Milvus。
- 查询时先做业务主体识别和问题改写，再进行普通向量检索、HyDE 检索和联网搜索。
- 使用 RRF 融合多路召回结果，再通过 BGE Reranker 重排。
- 回答生成阶段限制模型只能基于检索内容、历史对话和已确认业务主体作答，降低幻觉。
- 支持 SSE 流式输出、会话历史记录和导入任务状态追踪。
- 内置 RAG 评估模块，可评估主体命中率、precision、recall 和 must-hit 命中率。

## 技术架构

```text
导入链路
PDF / Markdown
-> 文件类型识别
-> PDF 转 Markdown
-> 图片资源处理
-> Markdown 语义切分
-> 业务主体识别
-> BGE-M3 向量化
-> Milvus 入库

查询链路
用户问题
-> 历史会话辅助的业务主体识别
-> 问题改写
-> 普通向量检索 + HyDE 检索 + Web Search
-> RRF 融合
-> BGE Reranker 重排
-> 受约束回答生成
```

## 技术栈

- Python 3.11+
- FastAPI
- LangGraph
- LangChain
- Milvus
- MongoDB
- MinIO
- BGE-M3 Embedding
- BGE Reranker
- OpenAI-compatible LLM API
- MinerU PDF 解析
- DashScope WebSearch MCP

## 目录结构

```text
app/
  api/                 # FastAPI 导入服务和查询服务
  process/             # LangGraph 导入/查询编排
  rag/                 # RAG 导入、检索、重排、回答生成服务
  rag_eval/            # RAG 评估脚本与指标
  resources/
    demo_docs/         # 出海业务演示文档
    html/              # 导入页和查询页
    prompts/           # RAG 链路 Prompt
  shared/              # 配置、客户端、模型与工具函数
docker-compose.yml     # Milvus / MinIO / MongoDB / etcd
pyproject.toml         # Python 依赖声明
uv.lock                # uv 锁文件
```

## 启动方式

### 1. 准备环境变量

复制 `.env.example` 为 `.env`，并填写 LLM、Embedding、Milvus、MongoDB、MinIO、MinerU 和 MCP 配置。

```bash
cp .env.example .env
```

`.env` 包含真实 API Key 和本地服务配置，只保留在本机，不要提交到 GitHub。

### 2. 启动基础设施

该命令会启动 Milvus、MinIO、etcd 和 MongoDB。首次运行需要拉取镜像，如果网络中断，重复执行同一命令即可继续。

```bash
docker compose up -d
```

### 3. 安装依赖

```bash
uv sync
```

### 4. 启动导入服务

```bash
uv run uvicorn app.api.http.import_server:app --host 0.0.0.0 --port 8000
```

如果 `8000` 端口被占用，可以换成 `8012`：

```bash
uv run uvicorn app.api.http.import_server:app --host 0.0.0.0 --port 8012
```

### 5. 启动查询服务

```bash
uv run uvicorn app.api.http.query_server:app --host 0.0.0.0 --port 8001
```

### 6. 打开页面

- 导入页面：http://127.0.0.1:8000/import/html
- 查询页面：http://127.0.0.1:8001/html

如果导入服务使用 `8012` 端口，则导入页面为：http://127.0.0.1:8012/import/html

## 演示方式

先在导入页面上传 `app/resources/demo_docs` 下的 Markdown 文档，等待导入完成后在查询页面提问。

可尝试的问题：

- 印尼市场用户投诉广告过多时，运营侧应该如何处理？
- 用户反馈已订阅但仍看到广告，客服应该先确认哪些信息？
- 海外用户请求退款时，哪些情况需要转人工？
- 隐私合规 FAQ 中，用户要求删除账号数据应该怎么回复？
