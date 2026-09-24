# ============================================================
# Demo 意义：
#   这是最基础的 LangChain 模型调用示例——创建 ChatOpenAI 模型实例后，
#   直接用一句自然语言调用模型并获取回复，不涉及任何提示模板或链。
#
# 学习目的：
#   1. 掌握如何用 ChatOpenAI 连接大模型（model、api_key、base_url）
#   2. 理解 temperature 参数对输出随机性的影响
#   3. 认识 model.invoke() 统一调用方式，以及返回的 AIMessage 对象
#      （取纯文本用 response.content，而不是直接打印整个对象）
#
# 适用场景：
#   快速验证模型连通性、做一次性问答、跑单个 prompt 的临时脚本，
#   以及任何不需要模板复用、不需要链式组合的简单调用。
# ============================================================
from pyexpat import model

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

model = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    # 温度参数，用于控制模型的输出随机性 0-1之间 0表示更确定的输出，1表示更随机的输出
    temperature=0,
)

response = model.invoke("用一句话解释 LangChain")
print(response.content)