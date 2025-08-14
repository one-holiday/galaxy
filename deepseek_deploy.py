import os
from openai import OpenAI
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()
load_dotenv(dotenv_path=r"D:\pythonproject\deepseek_deploy\ini.env")
# 新增：打印环境变量是否获取成功
print("ARK_API_KEY是否存在:", os.getenv("ARK_API_KEY") is not None)
print("ARK_MODEL_ID是否存在:", os.getenv("ARK_MODEL_ID") is not None)


# 初始化客户端
client = OpenAI(
    api_key=os.getenv("ARK_API_KEY"),  # 从环境变量获取API密钥
    base_url="https://ark.cn-beijing.volces.com/api/v3",  # 方舟API基础URL
)

def query_deepseek(prompt, system_prompt="你是有帮助的AI助手", stream=False):
    """
    查询DeepSeek模型的函数

    参数:
    prompt - 用户输入的问题
    system_prompt - 系统角色设定
    stream - 是否使用流式响应
    """
    try:
        # 创建聊天补全
        response = client.chat.completions.create(
            model=os.getenv("ARK_MODEL_ID"),  # 模型Endpoint ID
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # 控制随机性 (0-2)
            max_tokens=1024,  # 最大生成长度
            top_p=0.9,  # 核采样概率
            stream=stream  # 是否流式输出
        )

        # 处理流式响应
        if stream:
            full_response = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            print()  # 换行
            return full_response

        # 处理非流式响应
        return response.choices[0].message.content

    except Exception as e:
        print(f"请求失败: {str(e)}")
        return None


if __name__ == "__main__":
    # 示例用法
    print("----- 标准请求 -----")
    response = query_deepseek("常见的十字花科植物有哪些？")
    print(response)

    print("\n----- 流式请求 -----")
    query_deepseek("用Python实现快速排序算法", stream=True)


