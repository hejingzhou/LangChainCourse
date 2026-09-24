# ============================================================
# Demo 意义：
#   本示例演示如何用 ChatPromptTemplate 构建带角色区分的提示词模板，
#   再用变量填充模板后调用模型，实现"同一模板 + 不同输入"的复用。
#
# 学习目的：
#   1. 理解 system / human 等消息角色的含义与作用
#   2. 掌握 ChatPromptTemplate.from_messages 及 {变量} 占位符用法
#   3. 学会用 prompt.invoke(...) 把变量填入模板、得到消息列表再传给模型
#
# 适用场景：
#   需要区分系统设定与用户输入的结构化提示，例如角色扮演、法考答疑、
#   客服机器人等"固定人设 + 动态问题"的场景。
# ============================================================
from dotenv import load_dotenv
# langchain-openai: ChatOpenAI 提供 LangChain 对 OpenAI 系列聊天模型的封装能力
from langchain_openai import ChatOpenAI
import os
# langchain-core: ChatPromptTemplate 提供构建和管理聊天提示模板的核心能力
from langchain_core.prompts import ChatPromptTemplate, message, prompt

load_dotenv()

# 1. 创建 DeepSeek 模型
model = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    # 温度参数，用于控制模型的输出随机性 0-1之间 0表示更确定的输出，1表示更随机的输出
    temperature=0,
)

# 2. 创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位耐心的法律专家老师。用适合 {level} 的方式回答，不超过 200 字。"),
    ("human", "请解释 {topic}"),
])

# 3. 将变量填入模板，得到消息列表
message = prompt.invoke({
    "level": "法盲",
    "topic": "怎么认定正当防卫，什么是防卫过当？",
})

# 4. 调用模型，得到响应
response = model.invoke(message)

# 5. 打印响应内容
print(response.content)