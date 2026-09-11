"""
RAG 评估子系统统一入口。

目标：
1. 将评估数据、执行流程、指标统计收敛到一个独立包中。
2. 便于在结构相近的其他 RAG 项目中整体迁移。
3. 对外只暴露一个测试类，降低使用成本。

说明：
优先使用 `RagEvalTester`，不要关心包内的其他实现细节。
"""

from app.rag_eval.tester import RagEvalTester

__all__ = ["RagEvalTester"]
