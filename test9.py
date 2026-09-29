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
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    },
)

# 封装版：create_agent 会自动完成「执行工具 -> 回传结果 -> 再问模型」的循环，
# 直到模型不再要求调用工具为止（test8.py 里手写的三步，这里一行搞定）。
# 注意：这里传的是原始 model，不是 bind_tools 后的模型，
# create_agent 内部会自动 bind 这些 tools。
agent = create_agent(
    model=model,
    tools=[multiply, get_weather],
)

# 只需给一条用户消息，agent 自己管理工具调用、结果回传和多轮循环
result = agent.invoke(
    {"messages": [("user", "北京天气如何？顺便计算 23 乘以 7。")]}
)

# result["messages"] 是完整的对话历史，最后一条是模型的最终回答
print("===== 最终回答 =====")
print(result["messages"][-1].content)
