import os
from openai import OpenAI

# ��ȷ�����ѽ� API Key �洢�ڻ������� ARK_API_KEY ��
# ��ʼ��Openai�ͻ��ˣ��ӻ��������ж�ȡ����API Key
client = OpenAI(
    # ��ΪĬ��·�������ɸ���ҵ�����ڵ����������
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # �ӻ��������л�ȡ���� API Key
    api_key=os.environ.get("ARK_API_KEY"),
)

# Non-streaming:
print("----- standard request -----")
completion = client.chat.completions.create(
    # ָ���������ķ����������� ID���˴��Ѱ����޸�Ϊ������������ ID
    model="doubao-1-5-pro-256k-250115",
    messages=[
        {"role": "system", "content": "���������������"},
        {"role": "user", "content": "�����б������������ݵ�1.��м��ԣ�2.������������м�����������桢���桢���ԡ�����һ�֣����������֧�֡����ԡ�����������һ�֡�����������Ϊ����������������ô������ˣ��αػ�ҪȥΪһ�ݹ������ִ���أ�����Ϊ��Ǯ��ȥ�й���ѡ������ϯ������"},
    ],
)
print(completion.choices[0].message.content)

"""
# Streaming:
print("----- streaming request -----")
stream = client.chat.completions.create(
    # ָ���������ķ����������� ID���˴��Ѱ����޸�Ϊ������������ ID
    model="doubao-1-5-pro-256k-250115",
    messages=[
        {"role": "system", "content": "�����˹���������"},
        {"role": "user", "content": "������ʮ�ֻ���ֲ������Щ��"},
    ],
    # ��Ӧ�����Ƿ���ʽ����
    stream=True,
)
for chunk in stream:
    if not chunk.choices:
        continue
    print(chunk.choices[0].delta.content, end="")
print()
"""