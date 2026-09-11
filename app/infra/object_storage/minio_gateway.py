# 引入现有的功能,进行汇总!
from minio import Minio

from app.shared.clients.minio_utils import get_minio_client
from app.infra.config.providers import infra_config

class MinioGateway:

    # 提供获取桶名称的函数
    @property
    def bucket_name(self):
        return infra_config.minio_config.bucket_name

    # 提供获取图片前缀的函数
    @property
    def minio_img_dir(self):
        return infra_config.minio_config.minio_img_dir

    # 提供获取minio_client的函数
    @property
    def minio_client(self):
        return get_minio_client()

    # minio上传文件不会返回访问地址 -> 我们拼接访问地址  http https  端点  桶 对象名
    # 封装一个拼接访问地址的函数
    def build_image_url(self,stem:str,image_name:str):
        """
          object_name = upload-images / 文件名 / 图片名称.png
        :param object_name:
        :return:
        """
        image_url = "https://" if infra_config.minio_config.minio_secure else "http://" + (f"{infra_config.minio_config.endpoint}"
                           f"/{infra_config.minio_config.bucket_name}{infra_config.minio_config.minio_img_dir}/{stem}/{image_name}")
        return image_url

minio_gateway = MinioGateway()