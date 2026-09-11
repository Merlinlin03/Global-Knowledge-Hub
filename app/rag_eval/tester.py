"""
RAG 评估测试类。

这个文件故意写得很薄。
你只需要记住一个类和两个方法：

- `RagEvalTester.run_insert_test_data()`
- `RagEvalTester.run_eval()`

其他具体流程都放在 `runner.py` 里。

当前设计思路是：
- 测试知识数据尽量多一些；
- 测试问题保持少量但有代表性。
"""


class RagEvalTester:
    """
    RAG 评估统一入口类。

    最简单用法：

    ```python
    from app.rag_eval import RagEvalTester

    tester = RagEvalTester()
    tester.run_insert_test_data()
    tester.run_eval()
    ```
    """

    def run_insert_test_data(self) -> dict:
        """
        插入评测测试数据。

        返回值包含：
        - item_name：写入的主体名称
        - item_rows：item_name 集合查询结果
        - chunk_rows：chunks 集合查询结果
        - case_count：生成的评测用例数量
        """
        from app.rag_eval.runner import close_mongo_client, insert_batch_eval_dataset

        try:
            return insert_batch_eval_dataset()
        finally:
            close_mongo_client()

    def run_eval(self) -> dict:
        """
        运行批量评测。

        返回值包含：
        - eval_results：每条问题的详细评测结果
        - summary：批量汇总结果
        - report_path：评测报告文件路径
        """
        from app.rag_eval.runner import close_mongo_client, run_batch_eval

        try:
            return run_batch_eval()
        finally:
            close_mongo_client()
