import copy
import json
from typing import TypedDict
from app.shared.runtime.logger import logger


class ImportGraphState(TypedDict):

    # 传入的数据 (12)
    task_id: str
    local_file_path: str

    # 文件类型标识
    md_path : str
    is_md_read_enabled : bool

    pdf_path: str
    is_pdf_read_enabled: bool

    file_title : str  #文件名

    # 输出文件地址
    local_dir : str # 存储生成文件地址 文件夹

    md_content: str # md文件内容

    chunks : list # 装切割后的文本块 dict

    item_name: str  # 主体 file_title兜底

    embeddings_content: list # 带有向量的切块


# 准备一个state对象
default_state:ImportGraphState ={
    "task_id": "",
    "local_file_path": "",
    "md_path": "",
    "is_md_read_enabled": False,
    "pdf_path": "",
    "is_pdf_read_enabled": False,
    "file_title": "",
    "local_dir": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": []
}

# 定义一个可以更新对象属性的函数.并且返回更新后的对象
def create_default_state(**kwargs) -> ImportGraphState:
    """
      创建一个对象! 更新指定的属性 形参列表中指定要更新的属性
      **kwargs = local_file_path = value  -> dict {local_file_path : key}
    :return:
    """
    # 赋值对象内容! 进行复制(copy)内容更新!!
    # 深拷贝 : 不仅拷贝第一层属性,也会copy嵌套属性  新的 = copy.deepcopy(对象)
    # 浅拷贝 : 仅拷贝第一层属性,嵌套属性依然共享  copy.copy  |  dict.copy()
    new_state = copy.deepcopy(default_state)
    # default_state 全局唯一对象! 多次更新,值进行共享!
    new_state.update(kwargs)
    return new_state


def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染。
    """
    return copy.deepcopy(default_state)

if __name__ == "__main__":
    state = create_default_state(task_id="007周星驰",local_file_path="xx/xxx/md.md")
    # json json数据转换和备份
    # json.dump  dumps = >  dict -> json   dump dict写到晚不的 .json格式的文件  dumps dict 转成 json字符串
    #  不带s都是文件
    # json.load  loads = >  json -> dict   load 加载外部的json文件 dict  loads 把json字符串转成 dict
    logger.info("本次生成的state:\n{}", json.dumps(state,indent=4,ensure_ascii=False))
