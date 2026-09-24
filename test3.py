# ============================================================
# Demo 意义：
#   本示例演示 LangChain 的 LCEL 管道语法：用 | 把「提示模板 → 大模型 →
#   输出解析器」串成一条链，一次 invoke 即完成"格式化→生成→转文本"全过程。
#
# 学习目的：
#   1. 理解 LCEL（LangChain Expression Language）链式组合思想
#   2. 认识 StrOutputParser 的作用：把 AIMessage 对象转成纯文本
#   3. 掌握 chain = prompt | model | parser 的写法及其带来的简洁性
#
# 适用场景：
#   需要把多个组件串成固定流程并拿到纯文本结果的场景，是后续构建
#   RAG 检索链、Agent 等复杂工作流的基础写法。
# ============================================================
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate, message, prompt
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    # 温度参数，用于控制模型的输出随机性 0-1之间 0表示更确定的输出，1表示更随机的输出
    temperature=0,
)

# 基于指定模板创建聊天提示模板，{level}和{topic}为动态占位符，用于后续填充具体参数生成完整提示词
prompt = ChatPromptTemplate.from_template("用适合 {level} 的语言解释 {topic}，不超过三句话。")

# 初始化字符串输出解析器，用于将模型返回的AIMessage对象转换为可直接使用的纯文本格式
parser = StrOutputParser()

# 使用LangChain LCEL管道语法（|）组合处理链：按「提示模板格式化→大模型生成→解析器转纯文本」的顺序串联组件
chain = prompt | model | parser

answer = chain.invoke({
    "level": "初学者",
    "topic": "RAG",
})

print(answer)