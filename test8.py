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
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    },
)

model_with_tools = model.bind_tools([
    multiply,
    get_weather,
])

response = model_with_tools.invoke(
    "北京天气如何？顺便计算 23 乘以 7。"
)

print(response.tool_calls)
print(response)

# ---- 以上只是 Tool Calling 的第一步：模型返回了"要调用哪些工具 + 参数" ----
# 模型本身不会执行这些函数，也不会算出结果。执行和计算需要代码来完成。

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

# 3. 把工具执行结果回传给模型，让模型基于结果生成最终回答
final_response = model_with_tools.invoke(
    [HumanMessage(content="北京天气如何？顺便计算 23 乘以 7。"), response] + tool_messages
)

print("\n===== 最终回答 =====")
print(final_response.content)   