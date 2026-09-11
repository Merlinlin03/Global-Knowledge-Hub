from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger , step_log
from pathlib import Path

@step_log("resolve_input_file")
def resolve_input_file(state: ImportGraphState) -> ImportGraphState:
    """
     进行文件类型调用分发任务
    :param state:
    :return:
    """
    # 1. 先获取 local_file_path参数 state
    local_file_path = state.get("local_file_path")
    # 2. local_file_path进行非空校验 -> 空 -> 直接抛出异常 FileNotFound....
    if not local_file_path:
        logger.error(f"local_file_path未空,无法继续业务,提前终止!")
        raise FileNotFoundError(f"local_file_path未空,无法继续业务,提前终止!")
    # 3. 判断是不是md -> md_path is_md_read_enabled is_pdf_read_enabled = False
    # str .endswith 以什么结尾  startswith 以什么开头
    if local_file_path.endswith(".md"):
        state['md_path'] = local_file_path
        state['is_md_read_enabled'] = True
        state['is_pdf_read_enabled'] = False
    # 4. 判断是不是pdf -> is_md_read_enabled = False pdf_path  is_pdf_read_enabled
    elif local_file_path.endswith(".pdf"):
        state['pdf_path'] = local_file_path
        state['is_pdf_read_enabled'] = True
        state['is_md_read_enabled'] = False
    # 扩展文件类型
    # elif.....
    # 5. 都不是做好警告提示 is_md_read_enabled  pdf_path  is_pdf_read_enabled = False 提前结束
    else:
        logger.warning(f"{local_file_path}对应的文件类型无法解析,我们只支持md/pdf格式类型,提前终止,跳转到END节点!")
        state['is_pdf_read_enabled'] = False
        state['is_md_read_enabled'] = False
        return state
    # 6. 获取file_title参数 同步更新state
    # c://xx/xxx/xxx/xxx/xx.md xx.pdf
    # xx.md xx.pdf
    # local_file_path.split("/")[-1]
    # Path  获取当前路径地址  属性 .name -> 文件名带后缀 .stem 文件名没有后缀  .suffix 后缀 .parent 获取上一层文件夹 .parents 获取父文件夹列表
    #       函数  read_text()  read_bytes()   writer_text()  writer_bytes()
    file_title = Path(local_file_path).stem
    # 7. 返回处理后的state
    state['file_title'] = file_title
    return state