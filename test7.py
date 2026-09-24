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
    # model_name：使用的 Hugging Face 中文向量模型
    model_name="BAAI/bge-small-zh-v1.5",
    # model_kwargs：传给底层模型加载器的参数；device="cpu" 表示用 CPU 运行
    model_kwargs={"device": "cpu"},
    # encode_kwargs：向量化参数
    #   normalize_embeddings=True 的意义/目的：对每个向量做 L2 归一化，
    #   让所有向量的长度都变成 1。
    #   核心操作：把向量的每个分量都除以该向量自己的模长（L2 范数），
    #   即 v_normalized = v / ||v||。
    #   为什么这么做：归一化后，两个向量的“点积”就等于“余弦相似度”，
    #   检索时更公平、更稳定，不受文本长度影响。
    #   举例：向量 [3, 4] 的模长是 5，归一化后变成 [0.6, 0.8]，长度变为 1。
    encode_kwargs={"normalize_embeddings": True},
)

# 2. 重新打开第 6 课创建的向量库
vector_store = Chroma(
    # persist_directory：向量库的持久化目录，必须和第 6 课写入时的目录一致（./chroma_db）
    # 注意：这里传的是“目录”，不是 chroma.sqlite3 文件本身
    persist_directory="./chroma_db",
    # embedding_function：加载向量库时使用的 Embedding 模型，必须和写入时保持一致
    embedding_function=embedding_model,
)

# 3. 把向量库变成检索器
#    as_retriever 返回一个 VectorStoreRetriever（检索器）对象，
#    它本质上是一个 Runnable：传入问题字符串，返回与之相关的 Document 列表 list[Document]。
retriever = vector_store.as_retriever(
    # search_kwargs：检索参数；k=4 表示返回相似度最高的 4 条结果
    search_kwargs={"k": 4}
)

# 4. 将检索到的资料拼成给模型看的上下文
def format_docs(docs: list[Document]) -> str:
    # docs：检索器返回的 Document 列表，每个元素含 page_content（正文）和 metadata（元数据）
    result = []

    for doc in docs:
        page = doc.metadata.get("page", 0) + 1
        result.append(
            f"[来源：第 {page} 页]\n{doc.page_content}"
        )

    return "\n\n".join(result)

# 5. 提示词：限制模型只能依据资料回答
prompt = ChatPromptTemplate.from_template("""
你是一个 PDF 文档问答助手。

请严格遵守以下规则：
1. 只能根据“参考资料”回答。
2. 资料没有答案时，必须回答：“资料中没有找到答案。”
3. 不要使用自己的外部知识补充。
4. 回答结尾列出参考页码。

参考资料：
{context}

用户问题：
{question}
""")

# 6. DeepSeek 只负责生成最终回答
model = ChatOpenAI(
    # model：使用的模型名称
    model="deepseek-v4-pro",
    # api_key：API 密钥，从环境变量 DEEPSEEK_API_KEY 读取
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    # base_url：API 服务地址（DeepSeek 官方接口）
    base_url="https://api.deepseek.com",
    # temperature：采样温度，0 表示输出最确定、最稳定，适合事实性问答
    temperature=0,
)

# 7. 组装 RAG 链
rag_chain = (
    {
        # context：先用 retriever 检索，再把结果交给 format_docs 拼成上下文
        "context": retriever | format_docs,
        # question：把输入的问题原样透传给模板中的 {question}
        "question": lambda question: question,
    }
    | prompt
    | model
    # StrOutputParser：把模型的输出对象转成纯字符串
    | StrOutputParser()
)

# ============ 主流程：提问 -> 执行 -> 输出答案 ============

# 8. 定义要问的问题
question = "SpaceX成立于哪一年？"

# 9. 运行 RAG 链：内部会先检索、再拼提示词、最后让模型生成答案
answer = rag_chain.invoke(question)

# 10. 输出最终答案
print("===== 最终答案 =====")
print(answer)

# ============ 以下为调试观察（可选，不影响主流程） ============

# 观察 1：单独调用检索器，查看检索到了哪些内容
docs = retriever.invoke(question)
print("\n===== 检索到的内容 =====")
for i, doc in enumerate(docs, start=1):
    page = doc.metadata.get("page", 0) + 1
    print(f"--- 第 {i} 条（来源第 {page} 页）---")
    print(doc.page_content)
    print()

# 观察 2：查看投喂给 DeepSeek 的完整提示词
filled_prompt = prompt.format(context=format_docs(docs), question=question)
print("===== 投喂给 DeepSeek 的完整提示词 =====")
print(filled_prompt)
