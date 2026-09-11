import re
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import SUPPORTED_IMAGE_EXTENSIONS
from app.shared.runtime.logger import logger, step_log
from app.infra.llm.providers import llm_providers
from app.shared.runtime.load_prompt import load_prompt
import base64
from mimetypes import guess_type
from app.shared.utils.rate_limit_utils import apply_api_rate_limit
from app.infra.object_storage.minio_gateway import minio_gateway

# 4. 获取数据并且校验(state -> md_path) -> md_content , images_path_obj , md_path_obj
#           4.1 获取md_path属性 state
#           4.2 非空和文件存性校验
#           4.3 读取md_content内容通过md_path,md_content -> state [必须存,因为可能没有图片,后续不处理]
#           4.4 获取图片文件夹地址
#               md_path_obj -> .parent / 'images' -> images_path_obj
#           4.5 返回 return ...
@step_log("validates_and_get_data")
def validates_and_get_data(state:ImportGraphState) -> tuple[str,Path,Path]:
    """
      获取并且校验
    :param state:
    :return:
    """
    # 1. 获取md_path
    md_path = state.get("md_path")
    # 2. 判断存在性
    if not md_path:
        logger.error(f"核心参数md_path为空,业务无法继续,提前终止!")
        raise ValueError(f"核心参数md_path为空,业务无法继续,提前终止!")
    md_path_obj:Path = Path(md_path)
    if not md_path_obj.exists():
        logger.error(f"md_path地址为:{md_path},但是没有真实的文件,业务无法继续,提前终止!")
        raise FileNotFoundError(f"md_path地址为:{md_path},但是没有真实的文件,业务无法继续,提前终止!")
    # 3. 读取md_content
    md_content = md_path_obj.read_text(encoding="utf-8")
    if not md_content:
        logger.error(f"md_path地址为:{md_path},有真实的文件,但是内容为空!业务无法继续,提前终止!")
        raise ValueError(f"md_path地址为:{md_path},有真实的文件,但是内容为空!业务无法继续,提前终止!")
    state['md_content'] = md_content
    # 4. 获取images文件夹地址
    images_path_obj:Path = md_path_obj.parent / 'images'
    return md_content,images_path_obj,md_path_obj

@step_log("scan_images")
def scan_images(images_path_obj:Path,md_content:str) -> list[tuple[str,str,tuple[str,str]]]:
    """
    进行图片扫描!!
    :param images_path_obj:
    :param md_content:
    :return:
    """
    # 创建容器
    image_context = []
    #6.1 遍历图片文件夹images 获取每一个文件
    for image_obj in images_path_obj.iterdir():
        image_name = image_obj.name
        # 6.2 判断文件是图片 -> 后缀 -> list []
        if image_obj.suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            logger.warning(f"当前文件名:{image_name},不是图片,无需处理,直接跳过本次!")
            continue
        # 6.3 使用图片的名字 name -> md中查找是否引用 [正则 回顾]
        image_reg = re.compile(r'\!\[.*?\]\(.*?'+re.escape(image_name)+r'.*?\)')
        # 找第一个匹配对象
        match = image_reg.search(md_content)
        if not match:
            logger.warning(f"{image_name}图片没有被md内容引用,无需处理,直接跳过本次!")
            continue
        # 6.4 被引用获取引用的信息 start end
        start = match.start()
        end   = match.end()
        # start_1 = match.span()[0]
        # end_1 = match.span()[1]
        # 6.5 上文  [start-100  :start]  下文 [end:end+100]
        # [ )
        pre_context = md_content[max(0,start-100):start]
        post_content = md_content[end:min(end+100,len(md_content))]
        # 6.6 拼接本次数据 (图片名,str(obj),(上文,下文)) -> append ..
        image_context.append(
            (
                image_name,str(image_obj),
                (pre_context,post_content)
            )
        )
    return image_context

@step_log("summarize_images")
def summarize_images(image_content:list[tuple[str,str,tuple[str,str]]], stem:str) ->dict[str,str] :
    """
         7.1 获取模型对象 vision_chat()
         7.2 封装提示词(图片 / 文本)
         7.3 封装一个调用chains StrOutputParser()
         7.4 执行调用链获取返回结果
         7.5 封装一个dict,并且返回即可
    :param image_content:
    :param stem:
    :return:
    """

    image_summaries = {}

    # 7.1 获取模型对象 vision_chat()
    vision_client = llm_providers.vision_chat()

    # 循环处理每一张图片 content => 内容 context => 上下文
    for image_name,image_path_str, image_context  in image_content:

        # 添加访问限制
        apply_api_rate_limit()

        # 7.2 封装提示词(图片 / 文本)
        # 导入文本提示词
        image_text =  load_prompt("image_summary",root_folder = stem , image_content = image_context)
        # 处理图片的base64字符串
        image_path_obj:Path = Path(image_path_str)
        image_base64_str:str = base64.b64encode(image_path_obj.read_bytes()).decode(encoding="utf-8")
        human_message = HumanMessage(
            content = [
                {
                    "type":"text",
                    "text":image_text
                },
                {
                    "type":"image_url",
                    # url : "图片网络地址 | base64字符串"
                    # mimetype ???? 标准文件类型 (多邮件文件类型) text/html  image/png image/jpeg  video/mp4
                    # from mimetypes import guess_type
                    # tuple[str mimetype类型 image/jpeg ,str 压缩格式 gz | None]  =  guess_type(文件名.jpg.gz)
                    "image_url":{"url":f"data:{guess_type(image_name)[0]};base64,{image_base64_str}"}
                }
            ]
        )
        # 7.3 封装一个调用chains StrOutputParser()
        chains = vision_client | StrOutputParser()
        # 7.4 调用chains 链在执行的时候,加入的message列表
        # 视觉模型限制是每分钟3000次调用!!!
        image_summary =  chains.invoke([human_message])
        image_summaries[image_name] = image_summary
        logger.info(f"完成:{image_name}图片意图识别,识别内容:{image_summary}")

    return image_summaries


# 8. 将图片传递到minio,并且md_content内容的替换
#          md_content_new : str =  upload_images_and_replace(md_content,summarize_images:dict,image_context_list,md_path_obj.stem)
#          8.1 获取minio客户端对象
#          8.2 先删除当前文件图片(minio)
#          8.3 循环image_context_list向minio上传文件 -> dict[str:image_name,str:网络地址]4
#          8.4 循环图片和网络地址对应的字典,逐一替换md_content内容..
#          8.5 返回md_content_new

@step_log("upload_images_and_replace")
def upload_images_and_replace(md_content:str, summaries_images:dict[str,str],
                              image_content:list[tuple[str,str,tuple[str,str]]], stem:str) -> str:
    """
    进行文件上传和md_content替换
    :param md_content: 旧md_content ![](./...)
    :param summarize_images: 图片的描述 {图片名.jpg = 描述内容 }
    :param image_content: [(图片名,图片地址,(上,下))]
    :param stem: 文件名 文件夹名...
    :return: 新的md_content内容
    """
    # 8.1 获取minio客户端对象
    minio_client = minio_gateway.minio_client
    # 8.2 先删除当前文件图片(minio)
    """
        bucket_name 
            /upload-images
                /hk180烫金机安全手册
                   /xxx.jpg
                   /xxx.jpg
    """
    # 先查询
    # [Object(object_name) ]
    select_object_list = minio_client.list_objects(
        bucket_name=minio_gateway.bucket_name,
        # prefix查询的时候, 前面必须是不能添加 /   -> minio_img_dir 自带 / 开头
        prefix=minio_gateway.minio_img_dir[1:] + "/" + stem,
        recursive=True # 获取前缀下的文件夹中的所有对象
    )
    # 再删除
    # select_object_list : list[Object object_name ] -> delete_object_list list[DeleteObject object_name ]
    delete_object_list = [ DeleteObject(select_object.object_name)  for select_object in select_object_list ]
    errors = minio_client.remove_objects(minio_gateway.bucket_name,delete_object_list=delete_object_list)
    for error in errors:
        logger.warning(f"文件删除状态:{error}")

    # 8.3 上传文件
    image_urls = {} # key = image_name value = url
    for image_name,image_path_str, _ in image_content:
        # 健壮性 每张网络请求一次! 可能失败! 失败不影响整体
        try:
            # 图片上传
            # fput
            object_name = minio_gateway.minio_img_dir + "/" + stem + "/" + image_name
            minio_client.fput_object(
                bucket_name=minio_gateway.bucket_name,
                object_name=object_name,
                file_path=image_path_str,
                content_type=guess_type(image_name)[0]
            )
            # 图片访问地址 ..
            # http or https :// 端点 : 9000 / 桶名 / 对象名
            image_url = minio_gateway.build_image_url(stem,image_name)
            image_urls[image_name] = image_url
            logger.info(f"{image_name}已经上传到minio服务器,访问地址:{image_url}")
        except Exception as e:
            logger.warning(f"{image_name}图片上传minio失败,跳过本次,继续运行!")
            continue
    # 8.4 进行内容判断 image_urls
    if not image_urls:
        logger.warning(f"图片上传全部失败!直接使用原md_content处理即可!")
        return md_content

    # 8.5 进行md_content内容替换
    for image_name,image_url in image_urls.items():
        # summaries_images (image_name , 总结描述)
        image_summary = summaries_images.get(image_name)
        # md_content -> xxx ![](image_name) -> ![image_summary](image_url)
        # 定义正则 image_name查找
        reg = re.compile(r"\!\[.*?\]\(.*?"+re.escape(image_name)+r".*?\)")
        """
          参数1: 要替换入的内容  字符串  f"![{image_summary}]({image_url})"
                /2 正 sub代表着分组替换  
                lambda _ : f"![{image_summary}]({image_url})"  -> 匿名函数 -> 返回值不会处理,直接当做完整替换
          参数2: 要被替换的文档  字符串  md_content
          返回值: 替换后的文档   md_content 
        """
        md_content = reg.sub(lambda _: f"![{image_summary}]({image_url})",md_content)

    return md_content

@step_log("backup_new_md_content")
def backup_new_md_content(md_content_new:str, md_path_obj:Path) -> str:
    """
      进行md_content_new备份,防止后续确定,再次读取!
      也方便调试查看!!
    :param md_content_new:
    :param md_path_obj:
    :return:
    """
    #  md_path_obj -> 文件名_new.md
    # 获取目标的path
    md_new_path_obj:Path = md_path_obj.with_name(f"{md_path_obj.stem}_new{md_path_obj.suffix}")
    md_new_path_obj.write_text(md_content_new,encoding="utf-8")
    logger.info(f"将md_content_new进行备份,备份地址为:{md_new_path_obj}")
    return str(md_new_path_obj)

@step_log("enrich_markdown_images")
def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    """
    Markdown 图片增强服务：
    1. 扫描 Markdown 中的图片
    2. 调用多模态模型生成图片说明
    3. 上传图片到 MinIO
    4. 替换 Markdown 图片地址并回写 md_content
    """
    # 1. 参数校验和获取
    md_content,images_path_obj,md_path_obj = validates_and_get_data(state)
    # 2. 没有文件,提前终止...
    if (not images_path_obj.exists()) or images_path_obj.is_file() or (len(list(images_path_obj.iterdir()))==0):
        logger.info(f"{md_path_obj}文档对应的images为空或者没有图片!不需要图片识别,提前结束当前节点!")
        return state
    # 3. 获取每张图的信息 图片名 地址 (上,下)
    image_content: list[tuple[str,str,tuple[str,str]]] =  scan_images(images_path_obj, md_content)
    # 4. 进行图片内容识别(视觉模型)
    image_summaries:dict[str,str] = summarize_images(image_content, md_path_obj.stem)
    # 5. 文件上传和md_content内容替换 
    md_content_new: str = upload_images_and_replace(md_content,
                                                    image_summaries, image_content, md_path_obj.stem)
    # 6. 修改state md_content
    state['md_content'] = md_content_new
    # 7. 备份md_content -> 文件名_new.md -> state[md_path]
    md_path_new:str = backup_new_md_content(md_content_new,md_path_obj)
    state['md_path'] = md_path_new
    return state