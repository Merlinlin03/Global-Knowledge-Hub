import shutil
import time

import requests

from app.process.import_.agent.state import ImportGraphState
from pathlib import Path

from app.rag.import_.config import PDF_PARSE_SERVICE_LOCAL_DIR, MINERU_MODEL_VERSION, MINERU_POLL_TIMEOUT_SECONDS, \
    MINERU_POLL_INTERVAL_SECONDS
from app.shared.runtime.logger import logger, PROJECT_ROOT, step_log
from app.infra.config.providers import infra_config


#        1. 参数获取和校验 (state -> str)-> (pdf_path:Path,local_dir:Path)  validate_pdf_paths
#           # 获取参数 并且校验 非空 | 真实存在性校验
#           1.1 state获取 pdf_path 和 local_dir : str
#           1.2 进行pdf_path非空校验
#               空 -> 打印日志error -> 异常 ValueError
#           1.3 进行local_path非空校验
#               空 -> 打印日志warning -> 给与默认值  项目根路径 / output
#           1.4 将pdf_path local_dir 转成Path
#           1.5 pdf_path_obj:Path 判断是否存在
#               不存在,打印日志error -> 异常 FileNotFoundError
#           1.6 local_dir_obj:Path 是不是目录(is_dir())
#               不是(不存在,或者不是目录)
#               打印日志warning -> 创建 local_dri_obj对应的文件夹
#           1.7 返回结果 return pdf_path_obj , local_dir_obj

@step_log("validate_pdf_paths")
def validate_pdf_paths(state:ImportGraphState) -> tuple[Path,Path]:
    """
    进行参数获取和校验! 并且完成文件存在性和文件夹的创建....
    :param state:
    :return:
    """
    # 1.1 state获取 pdf_path 和 local_dir : str
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")
    # 1.2 进行pdf_path非空校验
    #  空 -> 打印日志error -> 异常 ValueError
    if not pdf_path:
        logger.error(f"pdf_path参数为空,业务无法继续进行,提前终止!")
        raise ValueError(f"pdf_path参数为空,业务无法继续进行,提前终止!")
    # 1.3 进行local_path非空校验
    #  空 -> 打印日志warning -> 给与默认值  项目根路径 / output
    if not local_dir:
        logger.warning(f"local_dir未空,我们给与默认值,默认为: 项目根地址/output 文件夹!")
        """
          Path  name stem suffix  read... writer... 
                拼接地址 Path  / Path or "xxx"
                local_dir_obj = PROJECT_ROOT / "output" / "xxx"  -> 推荐
        """
        local_dir:Path = PROJECT_ROOT / PDF_PARSE_SERVICE_LOCAL_DIR
        # 后续节点使用
        state["local_dir"] = str(local_dir)
    # 1.4 将pdf_path local_dir 转成Path
    pdf_path_obj:Path = Path(pdf_path)
    # Path(str | Path) 兼容语法
    local_dir_obj:Path = Path(local_dir)
    # 1.5 pdf_path_obj:Path 判断是否存在
    # 不存在,打印日志error -> 异常 FileNotFoundError
    if not pdf_path_obj.exists():
        logger.error(f"存在pdf_path地址:{str(pdf_path_obj)},但是地址没有对应文件,业务无法继续进行,提前终止!")
        raise FileNotFoundError(f"存在pdf_path地址:{str(pdf_path_obj)},但是地址没有对应文件,业务无法继续进行,提前终止!")
    # 1.6 local_dir_obj:Path 是不是目录(is_dir())
    # 不是(不存在,或者不是目录)
    # 打印日志warning -> 创建 local_dri_obj对应的文件夹
    if not local_dir_obj.is_dir():
        logger.warning(f"存在local_dir地址:{str(local_dir_obj)},但是没有对应的文件夹,我们创建对应的文件夹,业务继续!")
        # parents 如果有多层文件夹也会一定创建  就是你们在学linux命令的时候 mkdir -P x/x/x/x/x
        # exist_ok 如果存在,也不会报错,也不重复创建!
        local_dir_obj.mkdir(parents=True,exist_ok=True)
    # 1.7 返回结果 return pdf_path_obj , local_dir_obj
    return pdf_path_obj,local_dir_obj

@step_log("upload_pdf_and_poll")
def upload_pdf_and_poll(pdf_path_obj:Path) -> str:
     """
     进行minerU的交互的zip的获取
     :param pdf_path_obj:
     :return:
     """
     # 2.1 向minerU服务器发送请求申请上传地址 (batch_id / url)
     header = {
         "Content-Type": "application/json",
         # .env配置文件 -> config / minerU_config / infra / config / providers / miner_u ..
         "Authorization": f"Bearer {infra_config.mineru_config.api_key}"
     }
     data = {
         "files": [
             {"name": f"{pdf_path_obj.name}"}
         ],
         # p... vlm  html
         "model_version": MINERU_MODEL_VERSION
     }

     url = f"{infra_config.mineru_config.base_url}/file-urls/batch"
     """
        url 地址 headers 请求头  params ? 后参数  json 请求体参数  data 请求体 字节数据
     """

     response = requests.post(url=url,headers=header,json=data)
     # 所有的网络请求,必须两步骤判定 1. 状态码 200 http的网络状态  2. 业务状态必须成功
     if response.status_code != 200:
         logger.error(f"向minerU服务器申请上传文件解析,但是http状态码为:{response.status_code},状态错误,无法继续业务!")
         raise RuntimeError(f"向minerU服务器申请上传文件解析,但是http状态码为:{response.status_code},状态错误,无法继续业务!")
     response_dict = response.json()
     if response_dict.get('code',-1) != 0:
         logger.error(f"向minerU服务器申请上传文件解析,网络状态正常,服务业务状态异常,code = {response_dict.get('code',-1)}"
                      f"错误原因:{response_dict.get('msg')},无法继续业务!")
         raise RuntimeError(f"向minerU服务器申请上传文件解析,网络状态正常,服务业务状态异常,code = {response_dict.get('code',-1)}"
                      f"错误原因:{response_dict.get('msg')},无法继续业务!")
     # 网络没问题 / 业务也没问题
     batch_id = response_dict.get('data',{}).get('batch_id')
     file_upload_urls = response_dict.get('data',{}).get('file_urls',[])
     file_upload_url = None
     if len(file_upload_urls) > 0:
         file_upload_url = file_upload_urls[0]

     if not batch_id:
         logger.error(f"申请minerU解析文件,返回的batch_id为空,业务无法继续进行! 业务中断!")
         raise ValueError(f"申请minerU解析文件,返回的batch_id为空,业务无法继续进行! 业务中断!")


     logger.info(f"完成上传文件申请,batch_id:{batch_id},上传文件预签名地址:{file_upload_url}")
     #2.2 向指定的url地址发起网络请求并且上传pdf文件
     # file_upload_url  第一次请求申请地址 -> minerU -> 文件服务器 -> 开辟了一个空间 ->  空间对应的地址 -> 返回
     # 置换 -> 开辟的空间 -> 换成我们本次上传的文件 -> put
     # 预签名地址  第三方文件服务器的地址(想要往服务器上传文件需要认证) ->
     # 方案1: 上传的时候 传入token 认证  方案2: 预先认证(免检)  http://oss?3879732947294729
     # 预先签名地址 -> put(代码) ->  代理(vpn) 添加额外的请求头... -> 电脑的网络(网卡)  ->   文件服务器  很大概率会报错(认为你是免检,但是你中间干了不该干的事)
     # 尽量让请求更加干净 不要携带其他不相关的代理头
     # 预先签名 -> 服务器对你检查 -> 越严格
     with requests.Session() as session:
         # session (1.复用请求请求对象 2. 属性设置了以后,可以不信任当前系统的环境,保证请求的整洁性) 和 requests 都可以发起请求
         session.trust_env = False
         # 请求就按照代码的方式传递参数,不额外添加请求内容
         # 也不一定能传递成功!! 代理太强了!!!
         upload_response =  session.put(url=file_upload_url,data=pdf_path_obj.read_bytes())
         # 判断http的响应状态码 200
         # 判断业务状态码 code == 0  为啥? 因为不是一个接口 就是文件服务器特殊的上传地址 只有网络状态码 没有业务
         if upload_response.status_code != 200:
             logger.error(f"向:{file_upload_url}上传文件,服务器返回的网络状态码为:{upload_response.status_code},业务失败,提前终止!")
             raise RuntimeError(f"向:{file_upload_url}上传文件,服务器返回的网络状态码为:{upload_response.status_code},业务失败,提前终止!")


     #2.3 轮询向minerU获取batch_id解析状态 zip_url
     # 获取minerU解析结果
     # 方案1: 回调 (minerU -> 我们的服务器 fastapi)  申请地址的时候 请求体中 callback = 我们的地址
     # 方案2: 轮询 (我们 -> 3s -> minerU -> batch_id -> 解析结果) [我们]
     # 准备数据格式
     result_url = f"{infra_config.mineru_config.base_url}/extract-results/batch/{batch_id}"
     start_time = time.time()
     # 声明一个循环
     while True:
         # 1. 先判断时间 是否超时 600
         if time.time() - start_time >= MINERU_POLL_TIMEOUT_SECONDS:
             logger.error(f"轮询获取{batch_id}对应的解析结果超时! 耗时为: {time.time() - start_time}")
             raise TimeoutError(f"轮询获取{batch_id}对应的解析结果超时! 耗时为: {time.time() - start_time}")
         # 2. 没有超时向接口发起请求获取解析结果
         try:
            poll_result = requests.get(result_url,headers=header)
         except Exception as e:
            logger.warning(f"申请结果出现网络波动{str(e)},稍后再试!")
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue
         # 3. 网络状态判定
         # 1 2 3 4 5
         if poll_result.status_code != 200:
             # 5 给机会  客户端 -> 服务器
             # 4 不给机会  客户端 -> 一定错误
             if 500 <= poll_result.status_code < 600:
                # 错误是可以给机会! 这次不行了
                logger.warning(f"申请结果出现网络状态错误:{poll_result.status_code},稍后再试,等待服务器修复!")
                time.sleep(MINERU_POLL_INTERVAL_SECONDS)
                continue
             else:
                 logger.error(
                     f"获取:{batch_id}对应的解析结果,服务器访问报错,http的状态码:{poll_result.status_code},错误无法修复!业务失败,提前终止!")
                 raise RuntimeError(
                     f"获取:{batch_id}对应的解析结果,服务器访问报错,http的状态码:{poll_result.status_code},错误无法修复!业务失败,提前终止!")
         # 4. 业务状态判定
         poll_result_dict = poll_result.json()
         if poll_result_dict.get('code',-1) != 0:
             # 业务失败
             # 不给机会
             logger.error(
                 f"获取:{batch_id}对应的解析结果,业务状态报错! 业务状态码:{poll_result_dict.get('code',-1)},错误信息:{poll_result_dict.get('msg')},业务失败,提前终止!")
             raise RuntimeError(
                 f"获取:{batch_id}对应的解析结果,业务状态报错! 业务状态码:{poll_result_dict.get('code',-1)},错误信息:{poll_result_dict.get('msg')},业务失败,提前终止!")
         # 5. 获取解析结果和状态判定
         extract_result_list = poll_result_dict.get('data',{}).get('extract_result',[])
         if len(extract_result_list) == 0:
            # 错误是可以给机会! 这次不行了
            logger.warning(f"解析结果extract_result_list为空,跳过本次!稍后再试")
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue
         extract_result = extract_result_list[0]
         state = extract_result.get('state')

         if state == 'done':
            # 解析完毕
            full_zip_url = extract_result.get('full_zip_url')
            if not full_zip_url:
                # 不给机会
                logger.error(
                    f"获取:{batch_id}对应的解析结果,任务已经完成,但是full_zip_url没有地址!业务失败,提前终止!")
                raise ValueError(
                    f"获取:{batch_id}对应的解析结果,任务已经完成,但是full_zip_url没有地址!业务失败,提前终止!")
            # 2.4 返回zip_url ...
            return full_zip_url
         elif state == 'failed':
            # 解析完毕,失败了
            # 不给机会
            logger.error(
                f"获取:{batch_id}对应的解析结果,任务解析失败!业务失败,提前终止!")
            raise ValueError(
                f"获取:{batch_id}对应的解析结果,任务解析失败!业务失败,提前终止!")
         else:
            # 正在解析中...
            logger.warning("本次解析,没有获得结果,继续下一次!!!")
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue

@step_log("download_and_extract_markdown")
def download_and_extract_markdown(zip_url:str, local_dir_obj:Path, file_name:str) -> Path:
    """
       进行地址下载和解压,以及重命名! 最终返回md_path_obj
    :param zip_url:
    :param local_dir_obj:
    :param file_name:
    :return:
    """
    # 1.下载数据 [重试3次]
    response = requests.get(zip_url,timeout=MINERU_POLL_TIMEOUT_SECONDS)
    # 文件服务器! -> response - status_code
    if response.status_code != 200:
        logger.error(f"向指定地址:{zip_url}下载zip文件报错,状态码为:{response.status_code},业务无法继续进行!!")
        raise RuntimeError(f"向指定地址:{zip_url}下载zip文件报错,状态码为:{response.status_code},业务无法继续进行!!")

    # 准备zip文件对象
    zip_file_obj:Path = local_dir_obj / f"{file_name}.zip"
    """
      response 
          .status_code 
          .json()  -> 服务器返回的json字符串 -> dict
          .text    -> 服务器返回的json字符串 -> str -> json.loads...
          .content -> 服务器返回的字节数据   
    """
    zip_file_obj.write_bytes(response.content)
    # 2.解压数据
    # 创建一个解压后的文件夹  output / 文件名
    zip_extract_dir:Path = local_dir_obj / file_name

    if zip_extract_dir.is_dir():
        # 证明上一次解压过! 清空数据,避免脏数据
        # mysql -> 事务 -> 脏读 不可重复度 虚读幻读
        # shutil-> shell utils
        # shutil.rmtree -> 递归清空 清空文件夹内容和文件夹本身
        shutil.rmtree(zip_extract_dir)
    # 无论是否存在,都需要创建
    zip_extract_dir.mkdir(parents=True,exist_ok=True)
    # 解压
    """
       参数1: 要解压的压缩包
       参数2: 解压到的文件夹
      unpack_archive 解压  使用简单,支持所有格式的压缩包
    """
    shutil.unpack_archive(zip_file_obj,zip_extract_dir)
    # 压缩
    # shutil.make_archive(压缩的文件名 stem ,压缩的格式 zip tar tar.gz , 那个文件夹进行压缩 地址)
    # 3.重命名
    # 找到文件夹的指定类型文件 .md
    # Path
    md_obj_list: list[Path] = list(zip_extract_dir.rglob("*.md")) # 递归搜索
    # for 生成器.. -> list(生成器) -> list -> len...
    if len(md_obj_list) == 0:
        logger.error(f"向指定地址:{zip_url}下载zip文件,解压后发现没有md文件,业务无法继续进行!!")
        raise ValueError(f"向指定地址:{zip_url}下载zip文件,解压后发现没有md文件,业务无法继续进行!!")
    # 重命名
    # 情况1: 就是文件名 -> 直接return ..
    for current_md_obj in md_obj_list:
        if current_md_obj.stem == file_name:
            logger.info(f"向指定地址:{zip_url}下载zip文件,解压后的文件名,等于原文件名{file_name},直接返回!!")
            return current_md_obj
    md_obj_path:Path = None
    # 情况2: full -> 记录
    for current_md_obj in md_obj_list:
        if current_md_obj.stem == 'full':
            md_obj_path = current_md_obj
            break
    # 情况3: xxxx -> 记录
    if not md_obj_path:
        md_obj_path = md_obj_list[0]
    # 重命名
    # rename -> 真的会修改磁盘!!
    logger.info(f"触发了md文件的重命名机制,原名称:{md_obj_path.stem},目标名称:{file_name}")
    # rename 可以完成重命名!  删除旧的 复制新的
    # 参数方式1: 新的文件名 -> 不是完成地址相对地址 -> 创建在执行文件所在的文件夹  node_pdf_to_md   process / import_ / agent / notes
    # 参数方式2: 指定具体的完整地址
    # md_obj_path.rename(f"{file_name}.md")
    # md_obj_path c://xx/x/x/x/full.md    md_obj_path.with_name(f"{file_name}.md") -> 生成一个新的Path c://xx/x/x/x/文件名.md
    md_obj_path = md_obj_path.rename(md_obj_path.with_name(f"{file_name}.md"))
    return md_obj_path

@step_log("parse_pdf_to_markdown")
def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    """
      进行pdf转成md业务!
      最后修改state md_path属性
    """
    # 1.获取并且校验参数 validate_pdf_paths(state) -> tuple(pdf_path_obj: Path, local_dir_obj: Path)
    pdf_path_obj, local_dir_obj = validate_pdf_paths(state)
    # 2.minerU解析pdf文件并返回zip的下载地址
    zip_url:str = upload_pdf_and_poll(pdf_path_obj)
    # 3. 根据zip_url下载并解压和重命名md文件
    md_path_obj:Path =  download_and_extract_markdown(zip_url,local_dir_obj,pdf_path_obj.stem)
    # 4. 更新state md_path
    state['md_path'] = str(md_path_obj)
    return state