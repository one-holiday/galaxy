import os
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse  # 建议使用FastAPI的StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import asyncio  # 用于处理流式响应

# --------------------------
# 1. 加载环境变量和初始化AI客户端
# --------------------------
load_dotenv()  # 加载.env文件（确保路径正确）
load_dotenv(dotenv_path=r"D:\pythonproject\deepseek_deploy\ini.env")  # 手动指定路径（如果需要）

# 验证环境变量
print("Is ARK_API_KEY present:", os.getenv("ARK_API_KEY") is not None)
print("Is ARK_MODEL_ID present:", os.getenv("ARK_MODEL_ID") is not None)

# 初始化OpenAI客户端（连接DeepSeek模型）
client = OpenAI(
    api_key=os.getenv("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# --------------------------
# 2. 定义AI模型调用函数（复用query_deepseek）
# --------------------------
def query_deepseek(prompt, system_prompt="You are a helpful AI assistant", stream=False):
    try:
        response = client.chat.completions.create(
            model=os.getenv("ARK_MODEL_ID"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            stream=stream
        )

        if stream:
            # 流式响应：返回生成器（供FastAPI流式输出）
            def stream_generator():
                full_response = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield content  # 逐段返回
                print("\n流式响应完成")
            return stream_generator()  # 返回生成器
        else:
            # 非流式响应：直接返回结果
            return response.choices[0].message.content
    except Exception as e:
        print(f"Request failed: {str(e)}")
        return None

# --------------------------
# 3. 初始化FastAPI服务
# --------------------------
app = FastAPI()

# 允许跨域，配置CORS，允许前端域名（如 http://localhost:8080）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"], #替换为前端实际地址（必须精确匹配）
    allow_credentials=True,
    allow_methods=["*"], # 允许所有HTTP方法（包括OPTIONS预检请求）
    allow_headers=["*"], # 允许所有请求头
    expose_headers=["*"], # 允许所有响应头，可选：暴露响应头给前端
    max_age=3600 # 预检请求的缓存时间（秒），减少重复预检
)

# 定义前端输入格式
class InputData(BaseModel):
    input: str  # 用户输入的prompt
    stream: bool = False  # 是否启用流式响应（默认关闭）

# --------------------------
# 4. 定义API接口，调用AI模型
# --------------------------
    @app.post("/api/ai")
    async def ai_api(data: InputData):
        if data.stream:
            generator = query_deepseek(data.input, stream=True)
            # 添加SSE必要头部，防止连接被缓存或中断
            return StreamingResponse(
                generator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",  # 禁止缓存
                    "Connection": "keep-alive",  # 保持连接
                    "X-Accel-Buffering": "no"  # 禁止反向代理缓冲
                }
            )
        else:
            result = query_deepseek(data.input, stream=False)
            return {"result": result}
    # --------------------------
    # 启动服务
    # --------------------------
    # 命令行运行：uvicorn app:app --host 0.0.0.0 --port 5000 --reload
    # API地址：http://localhost:5000/api/ai

