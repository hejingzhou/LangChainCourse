---
title: LangChain Course
slug: langchain-course-code
date: "2026-09-24"
updated: "2026-09-24"
description: 从模型调用入门到 RAG 检索问答与 Tool Calling/Agent 的 9 个递进课程，附完整可运行代码。
tags: [LangChain, RAG, Agent, LLM, DeepSeek]
---

# LangChain 课程文档

> 本课程通过 9 个递进的 demo（test1.py ~ test9.py），从"调用一次大模型"逐步深入到 RAG 检索问答和 Tool Calling / Agent。
> 每个 demo 都围绕一个核心概念，附完整可运行代码，以及学习目标、学习目的与适用场景。

## 课程总览

| 编号 | 文件 | 主题 | 核心能力 |
| --- | --- | --- | --- |
| 1 | test1.py | 模型调用入门 | 用 ChatOpenAI 连接大模型并获取回复 |
| 2 | test2.py | 提示词模板 | 用 system/human 角色构建可复用模板 |
| 3 | test3.py | LCEL 链式组合 | 用 `\|` 把 prompt → model → parser 串成链 |
| 4 | test4.py | 模板 + 真实数据 | 把业务数据动态填充进模板 |
| 5 | test5.py | 结构化输出 | 用 Pydantic 让模型输出可解析的 JSON |
| 6 | test6.py | 文档加载与向量化 | PDF 切分 + Embedding + 存入向量库 |
| 7 | test7.py | RAG 检索问答 | 检索 → 拼上下文 → 生成答案 |
| 8 | test8.py | Tool Calling（手写） | 理解工具调用三步底层机制 |
| 9 | test9.py | Agent 封装 | 用 create_agent 自动完成工具循环 |

**学习路径**：① 模型调用与提示词基础（1~4）→ ② 结构化输出（5）→ ③ RAG 检索增强生成（6~7）→ ④ 工具调用与 Agent（8~9）

---

## 第 1 课：模型调用入门（test1.py）

### 一句话概括
创建 `ChatOpenAI` 模型实例，用一句自然语言调用模型并拿到回复，是最基础的 LangChain 用法。

### 学习目标
- 掌握用 `ChatOpenAI` 连接大模型的三个关键参数：`model`、`api_key`、`base_url`
- 理解 `temperature` 参数对输出随机性的影响（0 更确定，1 更随机）
- 认识 `model.invoke()` 统一调用方式，以及返回的 `AIMessage` 对象（取纯文本用 `response.content`）

### 学习目的
建立"调用大模型"的最小可运行认知，为后续所有课程打下基础——后面每一个 demo 都复用了这套模型连接方式。

### 适用场景
- 快速验证模型连通性、API key 是否有效
- 一次性问答、跑单个 prompt 的临时脚本
- 不需要模板复用、不需要链式组合的简单调用

### 核心知识点
- `load_dotenv()`：从 `.env` 文件读取环境变量（API key 不硬编码）
- `ChatOpenAI`：LangChain 对 OpenAI 系列聊天模型的封装（DeepSeek 也走 OpenAI 兼容协议）
- `response.content`：`invoke` 返回的是 `AIMessage` 对象，取纯文本要访问 `.content` 属性

### 完整代码
```python
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
```

> ⚠️ 注：原始 test1.py 里多了一行 `from pyexpat import model`（第 16 行），这是一处误加的导入，会导致 `ImportError`。上面已去掉，运行时应删除该行。

---

## 第 2 课：提示词模板（test2.py）

### 一句话概括
用 `ChatPromptTemplate.from_messages` 构建带角色区分的提示词，再用变量填充后调用模型。

### 学习目标
- 理解 `system` / `human` 等消息角色的含义与作用
- 掌握 `from_messages` 以及 `{变量}` 占位符的用法
- 学会用 `prompt.invoke(...)` 把变量填入模板、得到消息列表再传给模型

### 学习目的
实现"同一模板 + 不同输入"的复用——把固定的系统人设和动态的用户问题分离，让提示词可维护、可复用。

### 适用场景
- 需要区分"系统设定"与"用户输入"的结构化提示
- 角色扮演、法考答疑、客服机器人等"固定人设 + 动态问题"场景

### 核心知识点
- `ChatPromptTemplate.from_messages`：用元组 `("角色", "内容")` 构建多角色模板
- `system` 角色：设定模型的人设、规则、约束
- `human` 角色：真正的用户输入
- `{变量}` 占位符：用 `prompt.invoke({"变量": 值})` 填充

### 完整代码
```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. 创建 DeepSeek 模型
model = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
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
```

---

## 第 3 课：LCEL 链式组合（test3.py）

### 一句话概括
用 LCEL 管道语法 `|` 把「提示模板 → 大模型 → 输出解析器」串成一条链，一次 `invoke` 完成全过程。

### 学习目标
- 理解 LCEL（LangChain Expression Language）链式组合思想
- 认识 `StrOutputParser` 的作用：把 `AIMessage` 对象转成纯文本
- 掌握 `chain = prompt | model | parser` 的写法及其带来的简洁性

### 学习目的
掌握 LangChain 最核心的链式组合语法。LCEL 是后续构建 RAG 检索链、Agent 等复杂工作流的基础——所有组件都可以用 `|` 自由拼接。

### 适用场景
- 需要把多个组件串成固定流程并拿到纯文本结果
- 作为后续 RAG 链、Agent 的底层写法基础

### 核心知识点
- LCEL：用 `|` 管道符把 Runnable 组件串联，数据从左向右流动
- `StrOutputParser`：把模型的 `AIMessage` 输出对象解析成纯字符串
- 对比：第 2 课要手动 `prompt.invoke` 再 `model.invoke` 再取 `.content`，这里一条链搞定

### 完整代码
```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

# 基于指定模板创建聊天提示模板，{level}和{topic}为动态占位符
prompt = ChatPromptTemplate.from_template("用适合 {level} 的语言解释 {topic}，不超过三句话。")

# 初始化字符串输出解析器，把模型返回的 AIMessage 对象转换为纯文本
parser = StrOutputParser()

# 使用 LCEL 管道语法（|）组合处理链
chain = prompt | model | parser

answer = chain.invoke({
    "level": "初学者",
    "topic": "RAG",
})

print(answer)
```

---

## 第 4 课：模板 + 真实数据（test4.py）

### 一句话概括
用 `from_template` 构建单模板，并把真实的业务数据（桌面文件名列表）动态填充进模板。

### 学习目标
- 掌握 `from_template` 单模板用法（对比第 2 课的 `from_messages`）
- 学会用真实数据（`os.listdir` 读取文件系统）填充模板
- 理解多变量（`files`、`level`、`topic`）同时填充

### 学习目的
把 LangChain 和真实业务数据结合——模型的价值在于处理真实数据，而不是写死的文本。这一课演示了"动态数据 + 提示模板"的完整闭环。

### 适用场景
- 需要动态注入业务数据的场景（文件列表、数据库查询结果、API 返回值等）
- 把模型能力接入现有系统数据流的过渡练习

### 核心知识点
- `ChatPromptTemplate.from_template`：单个字符串模板，含多个 `{变量}`
- `os.listdir` + `os.path.isfile`：读取目录并过滤出文件
- `prompt.invoke` 一次填充多个变量

### 完整代码
```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

def getDesktopFileNames():
    """获取桌面上的所有文件名"""
    desktop_path = r"C:\Users\Administrator\Desktop"
    # 获取桌面目录下的所有条目
    entries = os.listdir(desktop_path)
    # 过滤出文件（不包括文件夹）
    file_names = [entry for entry in entries if os.path.isfile(os.path.join(desktop_path, entry))]
    return file_names

prompt = ChatPromptTemplate.from_template("以下是桌面文件名列表：{files}，请用适合 {level} 的方式回答 {topic} 。")

model = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

# 1. 先拿到桌面上的文件名列表
file_names = getDesktopFileNames()

# 2. 把 files、level、topic 三个变量一起填进模板
messages = prompt.invoke({
    "files": file_names,
    "level": "小白",
    "topic": "都有哪些文件类型，给我类型列表",
})

# 3. 把提示发给模型，拿到回复
response = model.invoke(messages)

# 4. 打印回复内容
print(response.content)
```

---

## 第 5 课：结构化输出（test5.py）

### 一句话概括
用 `with_structured_output` + Pydantic 让模型输出可解析的结构化 JSON，而不是自由文本。

### 学习目标
- 用 Pydantic `BaseModel` 定义输出数据结构
- 用 `Field(description=...)` 为每个字段添加语义约束
- 掌握 `model.with_structured_output(结构, method=...)` 用法
- 理解嵌套模型（一个模型的字段类型是另一个模型）

### 学习目的
让模型的输出从"给人看的自由文本"变成"给程序用的结构化数据"——直接拿到类型安全的对象，方便后续解析、入库、传给下游系统。

### 适用场景
- 信息抽取、文本分类、情感分析
- 需要把模型结果作为结构化数据返回给 API 或写入数据库
- 内容审核、简历解析、合同要素提取等

### 核心知识点
- Pydantic `BaseModel` + `Field`：定义字段和描述
- `with_structured_output`：包装模型，约束输出结构
- 嵌套模型：字段类型可以是另一个 `BaseModel`
- `method` 参数：`"function_calling"`（通过函数调用约束）、`"json_schema"`、`"json_mode"` 等

### 完整代码
```python
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import os

load_dotenv()

class AnalysisResult(BaseModel):
    key_point: str = Field(description="核心观点")
    confidence: float = Field(description="置信度，0-1")

class ArticleAnalysis(BaseModel):
    category: str = Field(description="文章类别...")
    sentiment: str = Field(description="...")
    summary: str = Field(description="...")
    result: AnalysisResult = Field(description="分析结果")

model = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}},
)

# 将模型包装为结构化输出模型：
# 1. 第一个参数 ArticleAnalysis 指定输出的 Pydantic 数据结构
# 2. method="function_calling" 表示通过函数调用机制约束生成结构化 JSON
structured_model = model.with_structured_output(
    ArticleAnalysis,
    method="function_calling",
)

# 注意：原文件里这里是一大段「埃隆·马斯克传」的测试文章文本，为节省篇幅已省略
result = structured_model.invoke("（此处省略：一大段待分析的文章内容）")

print("类别：", result.category)
print("情绪：", result.sentiment)
print("摘要：", result.summary)
print("分析结果：", result.result)
print("分析结果：", result.result.key_point)
print("置信度：", result.result.confidence)
```

> 注：`result.result.key_point`、`result.result.confidence` 演示了嵌套模型的字段访问方式。

---

## 第 6 课：文档加载与向量化（test6.py）

### 一句话概括
完成 RAG 的"数据准备"环节：加载 PDF → 切分文本 → 生成向量 → 存入本地向量库 Chroma。

### 学习目标
- 用 `PyPDFLoader` 加载 PDF（一页通常变成一个 Document）
- 用 `RecursiveCharacterTextSplitter` 切分文本，理解 `chunk_size` 和 `chunk_overlap`
- 用 `HuggingFaceEmbeddings` 加载本地中文向量模型（`bge-small-zh-v1.5`）
- 理解 `normalize_embeddings=True`（L2 归一化，让点积等于余弦相似度）
- 用 `Chroma.from_documents` 生成向量并持久化

### 学习目的
理解 RAG 的"检索"靠什么工作——本质是把文本转成向量，通过向量相似度找相关内容。这一课是所有 RAG 应用的"数据地基"。

### 适用场景
- 构建知识库、文档问答系统的数据预处理
- 为 PDF、Word、网页等文档建立可检索的向量索引

### 核心知识点
- `PyPDFLoader`：文档加载器
- `RecursiveCharacterTextSplitter`：递归字符切分，`chunk_size` 控制块大小、`chunk_overlap` 控制相邻块重叠
- `HuggingFaceEmbeddings`：本地 embedding 模型，`device="cpu"` 用 CPU 运行
- `Chroma`：本地向量数据库，`persist_directory` 指定持久化目录

### 完整代码
```python
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 读取 PDF：通常一页变成一个 Document
pages = PyPDFLoader("埃隆马斯克传记.pdf").load()

# 2. 把每页切成较小文本块
splitter = RecursiveCharacterTextSplitter(
    # chunk_size：每个文本块的最大字符数
    chunk_size=500,
    # chunk_overlap：相邻文本块重叠的字符数，避免切分边界丢失上下文
    chunk_overlap=80,
)

chunks = splitter.split_documents(pages)

# 3. 本地 Embedding 模型（首次执行会下载模型，需要网络和几分钟）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 4. 生成向量，并存入本地 Chroma 数据库
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

print(f"PDF 页数：{len(pages)}")
print(f"文本块数量：{len(chunks)}")

for singleChunk in chunks:
    print(f"第{chunks.index(singleChunk)+1}个文本块：{singleChunk.page_content}\n")
    print(f"第{chunks.index(singleChunk)+1}个文本块的元数据：{singleChunk.metadata}\n")
```

---

## 第 7 课：RAG 检索问答（test7.py）

### 一句话概括
完整演示 RAG（检索增强生成）：从向量库检索相关内容 → 拼成上下文 → 让模型基于资料生成答案。

### 学习目标
- 用 `Chroma` 重新打开第 6 课创建的向量库（`persist_directory` 传目录而非 sqlite 文件）
- 用 `as_retriever` 把向量库变成检索器，理解其返回 `list[Document]`
- 用 `format_docs` 把检索结果拼成上下文
- 组装 RAG 链：`retriever → format_docs → prompt → model → parser`
- 理解提示词里"只能依据参考资料回答"的约束作用

### 学习目的
理解 RAG 的核心价值——让模型基于"检索到的真实资料"回答，而不是依赖可能过时或幻觉的内部知识。这是企业知识库、文档问答的基石。

### 适用场景
- 文档问答、知识库助手
- 企业知识检索、客服知识库
- 任何需要"模型答案有据可查"的场景

### 核心知识点
- `vector_store.as_retriever(search_kwargs={"k": 4})`：返回相似度最高的 4 条
- RAG 链的组装方式：检索器输出 → 格式化 → 填充提示词 → 生成
- Embedding 模型必须与写入时一致（否则向量空间不匹配）

### 完整代码
```python
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import os

load_dotenv()

# 1. 必须使用和第 6 课完全相同的 Embedding 模型
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 2. 重新打开第 6 课创建的向量库
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
)

# 3. 把向量库变成检索器
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

# 4. 将检索到的资料拼成给模型看的上下文
def format_docs(docs: list[Document]) -> str:
    result = []
    for doc in docs:
        page = doc.metadata.get("page", 0) + 1
        result.append(f"[来源：第 {page} 页]\n{doc.page_content}")
    return "\n\n".join(result)

# 5. 提示词：限制模型只能依据资料回答
prompt = ChatPromptTemplate.from_template("""
你是一个 PDF 文档问答助手。

请严格遵守以下规则：
1. 只能根据"参考资料"回答。
2. 资料没有答案时，必须回答："资料中没有找到答案。"
3. 不要使用自己的外部知识补充。
4. 回答结尾列出参考页码。

参考资料：
{context}

用户问题：
{question}
""")

# 6. DeepSeek 只负责生成最终回答
model = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

# 7. 组装 RAG 链
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": lambda question: question,
    }
    | prompt
    | model
    | StrOutputParser()
)

# 8. 定义要问的问题
question = "SpaceX成立于哪一年？"

# 9. 运行 RAG 链
answer = rag_chain.invoke(question)

print("===== 最终答案 =====")
print(answer)
```

---

## 第 8 课：Tool Calling 手写版（test8.py）

### 一句话概括
用 `@tool` 定义工具 + `bind_tools` 绑定，手写"执行工具 → 回传结果 → 再问模型"三步，看清工具调用的底层机制。

### 学习目标
- 用 `@tool` 装饰器把普通函数变成模型可调用的工具（`docstring` 是给模型看的说明）
- 用 `model.bind_tools([...])` 让模型知道有哪些工具可用
- 理解模型只返回 `tool_calls`（工具名 + 参数），**不会自己执行**
- 手动执行工具、用 `ToolMessage` 把结果回传给模型

### 学习目的
理解 Tool Calling 的三步底层机制，避免"为什么模型只给我 tool_calls 不给结果"的困惑。这是理解 Agent 工作原理的前提。

### 适用场景
- 需要让模型调用外部函数、API、数据库的场景
- 精细控制工具调用流程、需要介入每一步的场景

### 核心知识点
- `@tool`：把函数包装成工具，函数的 `docstring` 会被模型当作工具说明
- `bind_tools`：绑定工具，返回带工具能力的模型
- `response.tool_calls`：模型返回的调用意图（`name` + `args`）
- `ToolMessage`：把工具执行结果回传给模型
- `finish_reason="tool_calls"` 且 `content=""`：说明模型在等代码执行工具

### 完整代码
```python
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import os

load_dotenv()


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""
    return a * b


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。此函数返回的是示例数据，不是真实天气。"""
    weather_data = {
        "北京": "晴，18°C",
        "上海": "小雨，22°C",
        "杭州": "多云，20°C",
    }
    return weather_data.get(city, "暂时没有该城市的天气数据。")


model = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
    # Tool Calling 时关闭思考模式
    extra_body={"thinking": {"type": "disabled"}},
)

model_with_tools = model.bind_tools([
    multiply,
    get_weather,
])

response = model_with_tools.invoke("北京天气如何？顺便计算 23 乘以 7。")

print(response.tool_calls)
print(response)

# ---- 以上只是第一步：模型返回了"要调用哪些工具 + 参数"，不会真正执行 ----

# 1. 工具名 -> 工具对象 的映射
tool_map = {
    "multiply": multiply,
    "get_weather": get_weather,
}

# 2. 逐个执行模型要求的工具，收集真实结果
tool_messages = []
for tc in response.tool_calls:
    func = tool_map[tc["name"]]
    result = func.invoke(tc["args"])   # 真正执行 multiply / get_weather
    tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

# 3. 把工具执行结果回传给模型，让模型生成最终回答
final_response = model_with_tools.invoke(
    [HumanMessage(content="北京天气如何？顺便计算 23 乘以 7。"), response] + tool_messages
)

print("\n===== 最终回答 =====")
print(final_response.content)
```

---

## 第 9 课：Agent 封装（test9.py）

### 一句话概括
用 `create_agent` 一行封装，自动完成「执行工具 → 回传结果 → 再问模型」的循环，直到模型不再调用工具。

### 学习目标
- 掌握 `create_agent(model, tools)` 的现代封装写法
- 理解它与第 8 课手写三步的对应关系
- 知道要传**原始 model**（不是 `bind_tools` 后的），`create_agent` 内部会自动 bind
- 知道结果从 `result["messages"][-1].content` 获取

### 学习目的
掌握生产级 Agent 写法。手写版用于理解机制，封装版用于实际开发——`create_agent` 能自动处理多轮工具调用、结果回传等复杂情况。

### 适用场景
- 日常开发中的工具调用 / Agent 应用
- 需要模型多轮调用工具、链式调用多个工具的场景

### 核心知识点
- `create_agent`：LangChain 1.x 推荐的 Agent 构建方式，底层基于 langgraph
- 自动工具循环：执行工具 → 回传 → 再问模型，直到模型给出最终回答
- `result["messages"]`：完整的对话历史，`[-1]` 是最终回答

### 完整代码
```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import os

load_dotenv()


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""
    return a * b


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。此函数返回的是示例数据，不是真实天气。"""
    weather_data = {
        "北京": "晴，18°C",
        "上海": "小雨，22°C",
        "杭州": "多云，20°C",
    }
    return weather_data.get(city, "暂时没有该城市的天气数据。")


model = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
    # Tool Calling 时关闭思考模式
    extra_body={"thinking": {"type": "disabled"}},
)

# create_agent 自动完成「执行工具 -> 回传结果 -> 再问模型」的循环
# 注意：传原始 model，不是 bind_tools 后的，内部会自动 bind
agent = create_agent(
    model=model,
    tools=[multiply, get_weather],
)

result = agent.invoke(
    {"messages": [("user", "北京天气如何？顺便计算 23 乘以 7。")]}
)

print("===== 最终回答 =====")
print(result["messages"][-1].content)
```

---

## 常见问题与注意事项

1. **API key 管理**：始终通过 `.env` + `load_dotenv()` 读取，不要把 key 硬编码进代码，也不要提交 `.env` 到 Git 仓库（本课程已用 `.gitignore` 排除）。
2. **Embedding 模型一致性**：RAG 写入向量库和读取向量库时，必须使用完全相同的 embedding 模型，否则向量空间不匹配、检索失效。
3. **`persist_directory` 传目录**：Chroma 的 `persist_directory` 传的是目录路径（如 `./chroma_db`），不是 `chroma.sqlite3` 文件本身。
4. **DeepSeek 工具调用需关闭思考模式**：test8/test9 中通过 `extra_body={"thinking": {"type": "disabled"}}` 关闭思考模式，否则工具调用可能异常。
5. **模型只"点菜"不"做菜"**：Tool Calling 中模型只返回 `tool_calls`（意图），真正执行工具、回传结果要靠代码（或 `create_agent` 自动完成）。
6. **test1.py 的误导入**：原始文件中 `from pyexpat import model` 是误加的一行，运行前应删除，否则会报 `ImportError`。
