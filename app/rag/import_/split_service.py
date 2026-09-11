import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN, CHUNK_MAX_SIZE
from app.shared.runtime.logger import logger, step_log


#   md_content , file_title =  load_markdown_content(state)
#          1. 获取三个参数  md_content , file_title , md_path
#          2. md_content 非空校验 -> 空 -> md_path 校验 读取..
#          3. file_title 非空校验 -> 空 -> md_path 校验 stem  -> default...
#          4. 统一换成符号(数据清洗)  md_content \n\r |  \r  -> \n -> state[md_content] ..
#          5. 返回md_content file_title
@step_log("load_markdown_content")
def load_markdown_content(state:ImportGraphState) -> tuple[str,str]:
    """
    获取参数和校验
    :param state:
    :return:
    """
    md_content = state.get("md_content")
    file_title = state.get("file_title")
    md_path    = state.get("md_path")

    # md_content校验
    if not md_content:
        if md_path and Path(md_path).exists():
            logger.warning(f"md_content内容为空,从备份地址:{md_path}再次读取数据!!")
            md_content = Path(md_path).read_text(encoding="utf-8")
        if not md_content:
            logger.error(f"md_content为空,尝试从md_path读取,依然为空,业务无法继续进行,提前终止!")
            raise ValueError(f"md_content为空,尝试从md_path读取,依然为空,业务无法继续进行,提前终止!")

    if not file_title:
        if md_path and Path(md_path).exists():
            file_title = Path(md_path).stem
        if not file_title:
            file_title = "default"
        state['file_title'] = file_title
        logger.warning(f"file_title为空,启动默认值机制,赋值后:{file_title}")

    # 数据清晰 统一换行符号
    md_content = md_content.replace("\r\n","\n").replace("\r","\n")
    state['md_content'] = md_content
    return md_content,file_title

#  split_document(md_content,file_title) -> list[dict{title,content,file_title}]

@step_log("split_chunks_document")
def split_chunks_document(md_content, file_title) -> list[dict[str,Any]]:
    """
       根据标题完成语义切割!!

       md_content

       adada
       adsadsa
       asdasdsa
       ## 标题1 \n
       文本内容 \n
       文本内容 \n
       ![]()   ...
       ```python \n
       import json \n
       # 这是python代码 \n
       xxxx
       ```
       文本内容
       文本内容
       ## 标题2
       文本内容
       文本内容
       ![]()
       ```python
       import json
       # 这是python代码
       xxxx
       ```
       文本内容
       文本内容



      思路:
         md_content - \n - 逐行 - 切割 -> list [line]
         逐行判断 -> line -> 标题  line -> 代码块 (状态)  line -> 普通行

    :param md_content:
    :param file_title:
    :return:
    """
    # 1. md_content按行切割 \n
    md_content_lines:list[str] = md_content.split("\n")
    # 2. 准备数据(记录当前标题和标题行以及代码块 历史chunks)
    chunks:list[dict[str,Any]] = [] # [{title,content,file_title}
    current_title:str = None
    current_title_lines:list[str] = []
    is_code_block = False
    # 判断是不是标题 -> 正则
    # 前面可以有空格 # - ###### 一个空格 内容 1
    # ^ 开头匹配
    # 空格 =  \s
    # 量词  * 0 - n + 1 - N  {5,10}
    title_reg = re.compile(r"^\s*#{1,6}\s.+")
    # 3. 循环行 -> 判断是不是标题 是不是带块 是不是空行 是不是普通
    for line in md_content_lines:
        # 空行
        line_strip = line.strip()
        if not line_strip:
            # 空行
            logger.warning("处理碰到空行!跳过本次处理!")
            continue
        # 判断是不是代码块
        if line_strip.startswith("```") or line_strip.startswith("~~~"):
            # ```  ~~~ 进入或者退出
            is_code_block = not is_code_block
            current_title_lines.append(line_strip)
            continue
        # 判断是不是有效标题
        if not is_code_block and title_reg.match(line_strip):
            # 是 情况1: 下一个标题开始了 上一次是不是需要结算了!
            #    情况2: 第一次是第一个标题 别结算了
            #  只要最近的有效标题 有标题 有内容...
            if current_title and len(current_title_lines) > 1:
                # 第二个....
                chunks.append(
                    {
                        "content": "\n".join(current_title_lines),
                        "title":current_title,
                        "file_title":file_title
                    }
                )

            if not current_title and len(current_title_lines) > 0:
                current_title_lines.append(line_strip)
            else:
                current_title_lines = [line_strip] # 将标题设置第一行字符串
            # 开启新的赛季了
            current_title = line_strip  # 新的设置为当前处理标签
        else:
            # 不是  在代码块 普通行 or 不在带块 不是 # 开头
            current_title_lines.append(line_strip)

    # 最后一次可能没有被结算
    if current_title and len(current_title_lines) > 1:
        # 第二个....
        chunks.append(
            {
                "content": "\n".join(current_title_lines),
                "title": current_title,
                "file_title": file_title
            }
        )

    # 还一种尴尬场景,也可以能不结算 [整个文档没有标题]
    if len(chunks) == 0 and len(current_title_lines) > 0:
        chunks.append({
            "content":"\n".join(current_title_lines),
            "title":"default",
            "file_title":file_title
        })

    logger.info(f"完成了语义标题切割,切块数量为:{len(chunks)}")

    return chunks

@step_log("_split_long_chunk")
def _split_long_chunk(chunk) -> list[dict[str,Any]]:
    """
      主要目标: 过长的进行短切...
      使用技术: langchain递归切割器..
    :param chunk:
    :return:
    """
    """
       chunk
           title # xxxx
           file_tile hk180烫金机
           content  "\n".json(current_title_lines) -> 1. # xxxx\n2. 内容1 ||| 3.内容2 
       注意: 不能只让第一部分有标题 
          1. 先将content标题移除
          2. 定义标准标题 prefix title\n
    """
    # 1. 清晰原有的content去掉标签前缀..
    content = chunk.get("content")
    title   = chunk.get("title")
    file_title = chunk.get("file_title")

    # + 1 因为去掉 \n符号 = 1  r"\n" = 2
    clear_content = content[len(title) + 1:] # 2. 内容1 ||| 3.内容2
    # 2. 定义公共标准标签前缀
    sub_content_prefix = title + "\n"
    """
    # xxxx
    内容....
    """
    # 3. 定义递归切割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE - len(sub_content_prefix),  # 因为确保标题数量去除了
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";"],
    )

    sub_chunk_list = []

    # 4. 使用递归切割器进行content切割
    for part_index ,  split_text in enumerate(splitter.split_text(clear_content),start=1):
        # 5. 拼接每个子chunk内容
        split_text = split_text.strip()
        content_sub_new = sub_content_prefix + split_text
        # chunk -> 拼接提示词 -> llm  title 核心的信息
        # title 标题信息 以及我们是内部第几个部分  title_part
        sub_chunk_list.append({
            "title":f"{title}_第{part_index}部分",
            "content":content_sub_new,
            "file_title":file_title,
            "parent_title":title,
            "part":part_index
        })
    # 6. 最终返回结果
    logger.info(f"进入标题:{title},完成切割后,切成:{len(sub_chunk_list)}块!")
    return sub_chunk_list

@step_log("_merge_short_chunks_same_parent_title")
def _merge_short_chunks_same_parent_title(refine_list)-> list[dict[str,Any]]:
    """
     同一个标题下,过短(400)进行合并,不超过(1000)
    :param refine_list:
    :return:
    """
    """
      先指向一个基础(pre),作为参照! 
      如果base小于400,尝试将后面的合并入.. 
      合并的前提是: base < 400  同一个parent_title  合并后小于1000 
    """
    merged_chunk_list = []
    # 1.定义一个合并的base chunk 变量
    base_chunk =None # 要合并入的 不动 判断400  [1]
    # 2.循环处理chunk_list挪动要合并的元素
    for next_chunk in  refine_list:
        # 第一次给base_chunk赋值
        if not base_chunk:
            base_chunk = next_chunk
            logger.info(f"短合并第一次进入,设置base_chunk内容!")
            continue
        # 3.合并的逻辑
        # base_chunk -> content < 400 条件1
        is_short_chunk = len(base_chunk.get("content")) < CHUNK_MIN
        # next_chunk -> parent_title == base_chunk -> parent_title  and parent_title 非空
        # 确保不同语义标题一定不能合并!!!
        is_same_parent_title = base_chunk.get("parent_title") and base_chunk.get("parent_title") == next_chunk.get("parent_title")

        if is_short_chunk and is_same_parent_title:
            # 短 + 同一个标题 (可能被合并)
            # base_chunk + next_chunk content + <= 1000
            # 有可能是同一个标题
            # title + \n  base   +   title + \n  next
            # title 切块之前的! 切块之后 title_1  title -> parent_title
            next_content = next_chunk.get("content")[len(next_chunk.get("parent_title")) + 1:]
            # next_content -> len -> 400 - 600 不能合并
            # if 400 600:    400 base -> 300  next -> 500 合并 减少碎片 < 400
            #     # base next -> 加 -> add..
            #     base_chunk = None
            #     continue
            is_not_long = (len(base_chunk.get("content")) + len(next_content)) <= CHUNK_MAX_SIZE
            if is_not_long:
                # 没有超过1000
                base_chunk['content'] = base_chunk.get("content") + "\n" + next_content
                continue
            else:
                merged_chunk_list.append(base_chunk)
                base_chunk = next_chunk  # 切换指向! next作为基础判断
                continue
        else:
            # 不能合并 base 大于400 要不然不是同一个标题
            # base <-- next
            merged_chunk_list.append(base_chunk)
            base_chunk = next_chunk # 切换指向! next作为基础判断
            continue
    # 跳出循环
    if base_chunk:
        merged_chunk_list.append(base_chunk)
    logger.info(f"进行短合并,合并之前:{len(refine_list)},合并之后:{len(merged_chunk_list)}")
    return merged_chunk_list


@step_log("refine_chunks")
def refine_chunks(chunks) -> list[dict[str,Any]]:
    """
    进行精细切割!
      长 -> 600 -> 短切
      短 -> 400 -> 合并 -> 1000
    :param chunks:
    :return:
    """
    # 1. 定义接收最终结果的list
    refine_list = []
    # 2. 进行循环处理查看是否过长
    for chunk in chunks:
        content = chunk.get("content")
        if len(content) > CHUNK_SIZE:
            long_chunk_list = _split_long_chunk(chunk)
            refine_list.extend(long_chunk_list) #[{}.{}]
        else:
            refine_list.append(chunk)
    # 3. 进行短合并处理
    refine_list = _merge_short_chunks_same_parent_title(refine_list)
    # 4. 补全属性..
    for chunk in refine_list:
        if "parent_title" not in chunk:
            chunk['parent_title'] = chunk.get('title',"default_title")
        if "part" not in chunk:
            chunk['part'] = 1
    # 5. 最终返回结果
    logger.info(f"完成chunks的精细处理! 进入切块数量:{len(chunks)},处理后:{len(refine_list)}")
    return refine_list

@step_log("backup_chunks_json")
def backup_chunks_json(refine_chunks_list, md_path:str):
    """
    数据备份
    :param refine_chunks_list:
    :param param:
    :return:
    """
    # 1. 获取目标的地址  文件夹 / 文件名_new.json
    json_path_obj:Path = Path(md_path).with_name(f"{Path(md_path).stem}.json")
    # 2. 目标位置写入字符串
    json_path_obj.write_text(json.dumps(refine_chunks_list,indent=4,ensure_ascii=False),encoding="utf-8")
    logger.info(f"已经将切片数据,备份到{json_path_obj}位置!!")

@step_log("split_document")
def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分服务：
    1. 按标题层级做一级粗切
    2. 对超长文本做二次细切
    3. 构造 chunks 列表
    4. 回写 chunks
    """

    # 1. 获取参数的校验
    md_content, file_title = load_markdown_content(state)
    # 2. 确保语义切割,根据标题切割(只保留关联标题)
    chunks : list[dict[str,Any]] = split_chunks_document(md_content, file_title)
    # 3. 进行精细切割处理
    refine_chunks_list = refine_chunks(chunks)
    # 4. 修改state -> chunks
    state['chunks'] = refine_chunks_list
    # 5. 备份refine_chunks_list  -> [{},{}] -> json字符串 -> 本地磁盘
    backup_chunks_json(refine_chunks_list,state['md_path'])
    return state