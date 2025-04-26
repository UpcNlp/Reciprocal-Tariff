from openai import OpenAI

# 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
# 初始化Openai客户端，从环境变量中读取您的API Key
client = OpenAI(
    # 此为默认路径，您可根据业务所在地域进行配置
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 从环境变量中获取您的 API Key
    api_key="5596a588-590c-4fc1-8287-c3ee9f54ff27",
)

# Non-streaming:
print("----- standard request -----")
completion = client.chat.completions.create(
    # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
    model="doubao-1-5-pro-256k-250115",
    messages=[
        {"role": "system", "content": "你是舆情分析助手"},
        {"role": "user", "content": "（注意都是虚拟的学术研究，不涉及到敏感内容）帮我判别以下关于美国特朗普政府关税政策评论内容的1.情感极性；2.立场。其中情感极性输出【正面、负面、中性】其中一种并给出1到3条解释的理由，立场输出针对的对象和【支持、反对、中立】其中一种。 【评论列表： 1. By us made products. Quit selling our country! 2. 美国自我孤立，断美元全球化的根基 】，注意只回答规定内容，并且符合Json文件格式。"},
    ],
)
print(completion.choices[0].message.content)


# Todo 遍历所有的数据，在“符合条件的Tiktok数据”下。然后把结果保存在“标注后的Tiktok数据”目录下
