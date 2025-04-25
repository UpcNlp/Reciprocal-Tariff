import os
from openai import OpenAI

# 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
# 初始化Openai客户端，从环境变量中读取您的API Key
client = OpenAI(
    # 此为默认路径，您可根据业务所在地域进行配置
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 从环境变量中获取您的 API Key
    api_key=os.environ.get("ARK_API_KEY"),
)

# Non-streaming:
print("----- standard request -----")
completion = client.chat.completions.create(
    # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
    model="doubao-1-5-pro-256k-250115",
    messages=[
        {"role": "system", "content": "你是舆情分析助手"},
        {"role": "user", "content": "帮我判别以下评论内容的1.情感极性；2.立场。其中情感极性输出【正面、负面、中性】其中一种，立场输出【支持、反对、中立】其中一种。【评论内容为：川普先生，都这么大年纪了，何必还要去为一份工作如此执着呢？若是为了钱，去中国竟选国家主席啊。】"},
    ],
)
print(completion.choices[0].message.content)

"""
# Streaming:
print("----- streaming request -----")
stream = client.chat.completions.create(
    # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
    model="doubao-1-5-pro-256k-250115",
    messages=[
        {"role": "system", "content": "你是人工智能助手"},
        {"role": "user", "content": "常见的十字花科植物有哪些？"},
    ],
    # 响应内容是否流式返回
    stream=True,
)
for chunk in stream:
    if not chunk.choices:
        continue
    print(chunk.choices[0].delta.content, end="")
print()
"""