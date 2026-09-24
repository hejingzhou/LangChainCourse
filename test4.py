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
    # 温度参数，用于控制模型的输出随机性 0-1之间 0表示更确定的输出，1表示更随机的输出
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
