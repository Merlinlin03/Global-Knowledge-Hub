import os

from app.shared.runtime.logger import logger, PROJECT_ROOT



def load_prompt(name: str, **kwargs) -> str:
    """
    加载提示词并渲染变量占位符
    :param name: 提示词文件名（不带.prompt后缀，如image_summary）
    :param **kwargs: 需渲染的变量键值对（如root_folder="测试文件", image_content=("上文内容", "下文内容")）
    :return: 渲染后的最终提示词字符串
    """
    # 1. 拼接提示词路径（你的原有逻辑，完全保留）
    prompt_path = PROJECT_ROOT / 'app'  / 'resources' / 'prompts' / f'{name}.prompt'

    # 2. 校验文件是否存在（可选，避免文件不存在直接报错）
    if not prompt_path.exists():
        raise FileNotFoundError(f"提示词文件不存在：{prompt_path.absolute()}")

    # 3. 读取纯文本提示词（你的原有逻辑）
    raw_prompt = prompt_path.read_text(encoding='utf-8')

    # 4. 核心：如果传了参数，渲染占位符；没传参，直接返回原文本
    if kwargs:
        # f"{question}xx"
        rendered_prompt = raw_prompt.format(**kwargs)
        logger.debug(f"提示词渲染成功，替换变量：{list(kwargs.keys())}")
        return rendered_prompt
    return raw_prompt



if __name__ == '__main__':
    text = load_prompt("rerank_text_refine",question="问题",answer="答案",limit=20)
    print(text)

    from dotenv import load_dotenv
    # override=True 覆盖系统环境变量 OPEN_API_KEY || .env OPEN_API_KEY
    #   1. 建议删除系统环境变量的key!!
    #   2. override=True 项目下 .env 覆盖系统环境变量
    load_dotenv(override=True)
    os.getenv("OPENAI_API_KEY") # -> 系统环境变量