# import模块所有json相关的类型
from pydantic import BaseModel


class UploadResponseSchema(BaseModel):
    code:int = 200
    message:str | None=None
    task_ids:list[str]

class StatusResponseSchema(BaseModel):
    code:int=200  # 业务状态
    task_id:str   # task_id
    status:str    # 当前task_id解析文件的整体状态 正在进行 已经完成或者失败
    done_list:list[str]    # 获取本次task_id对应已经完成的节点
    running_list:list[str] # 获取本次task_id正在进行的节点