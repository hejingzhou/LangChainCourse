from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 读取 PDF：通常一页变成一个 Document
#    参数：PDF 文件的路径（字符串），可以是相对路径或绝对路径
pages = PyPDFLoader("埃隆马斯克传记.pdf").load()

# 2. 把每页切成较小文本块
splitter = RecursiveCharacterTextSplitter(
    # chunk_size：每个文本块的最大字符数（500 表示每块最多约 500 个字符）
    chunk_size=500,
    # chunk_overlap：相邻文本块之间重叠的字符数（80 表示每块首尾重叠 80 个字符，
    #                用于避免在切分边界处丢失上下文信息）
    chunk_overlap=80,
)

chunks = splitter.split_documents(pages)

# 3. 本地 Embedding 模型
# 首次执行会下载模型，需要网络和几分钟时间
embeddings = HuggingFaceEmbeddings(
    # model_name：要使用的 Hugging Face 预训练模型名称
    #             bge-small-zh-v1.5 是中文向量模型，体积小、适合 CPU 本地运行
    model_name="BAAI/bge-small-zh-v1.5",
    # model_kwargs：传给底层模型加载器的参数
    #               device="cpu" 表示在 CPU 上运行（没有 GPU 时使用）
    model_kwargs={"device": "cpu"},
    # encode_kwargs：文本编码（向量化）时的参数
    #               normalize_embeddings=True 表示对生成的向量做 L2 归一化，
    #               归一化后向量长度为 1，便于用余弦相似度检索
    encode_kwargs={"normalize_embeddings": True},
)

# 4. 生成向量，并存入本地 Chroma 数据库
vector_store = Chroma.from_documents(
    # documents：要写入向量库的文档列表（这里是切分后的 chunks）
    documents=chunks,
    # embedding：用于把文本转成向量的 embedding 模型
    embedding=embeddings,
    # persist_directory：向量数据的本地持久化目录，
    #                    之后可通过 Chroma(persist_directory=...) 重新加载
    persist_directory="./chroma_db",
)

print(f"PDF 页数：{len(pages)}")
print(f"文本块数量：{len(chunks)}")

# print("\n第一个文本块：")
# print(chunks[0].page_content)

# print("\n第一个文本块的元数据：")
# print(chunks[0].metadata)

for singleChunk in chunks:
    print(f"第{chunks.index(singleChunk)+1}个文本块：{singleChunk.page_content}\n")
    print(f"第{chunks.index(singleChunk)+1}个文本块的元数据：{singleChunk.metadata}\n")
