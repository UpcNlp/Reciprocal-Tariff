# project_root/process_data.py

import os
import re
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

# ——— Step 1: 读取数据，按视频拆分 ———
INPUT_FILE = '../Tiktok_评论采集_八爪鱼RPA数据表格_20250422-121433.xlsx'  # 根据实际路径调整
OUTPUT_DIR = '../Data'

def extract_video_id(url: str) -> str:
    """从 TikTok 视频链接中提取数字 ID"""
    m = re.search(r'/video/(\d+)', url)
    return m.group(1) if m else url

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # 标准化列名，方便后续处理
    df = df.rename(columns={
        '作品链接': 'video_url',
        '回复用户': 'user',
        '回复评论': 'comment'
    })
    df['video_id'] = df['video_url'].apply(extract_video_id)
    return df

# ——— Step 2: 获取每个视频的发表时间，打上时间标签（具体到哪一天） ———
def extract_publish_date(video_url: str) -> str:
    """
    抓取 TikTok 视频页面，解析 JSON 数据中的 createTime，
    并返回 ISO 格式日期字符串（YYYY-MM-DD）。
    """
    try:
        resp = requests.get(video_url, headers={
            'User-Agent': 'Mozilla/5.0'
        }, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        script = soup.find('script', id='SIGI_STATE')
        data = json.loads(script.string)
        vid = extract_video_id(video_url)
        ts = data['ItemModule'][vid]['createTime']
        return datetime.fromtimestamp(int(ts)).date().isoformat()
    except Exception:
        return None  # 若抓取失败，可后续手动补充

# ——— Step 3: 对每个视频下的评论按用户分组，并打标签 ———
import regex  # pip install regex

def is_pure_emoji(text: str) -> bool:
    """检查字符串是否仅由 Emoji 组成"""
    # \p{Emoji} 匹配 Emoji， anchors 保证整串都是 Emoji
    return bool(regex.fullmatch(r'(?:\p{Emoji_Presentation}|\p{Emoji}\uFE0F)+', text))

def detect_language(text: str) -> str:
    """
    简单区分中/英/混合/其他：
      - 仅中文 → 'Chinese'
      - 仅英文 → 'English'
      - 中英文混合 → 'Mixed'
      - 其他字符（数字、符号等）→ 'Other'
    """
    has_cn = bool(regex.search(r'\p{IsHan}', text))
    has_en = bool(regex.search(r'[A-Za-z]', text))
    if has_cn and has_en:
        return 'Mixed'
    elif has_cn:
        return 'Chinese'
    elif has_en:
        return 'English'
    else:
        return 'Other'

# ——— 主流程 ———
def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(INPUT_FILE)
    results = {}

    # 按视频 ID 分组
    # vid：分组后的唯一标识（以'video_id'分组，唯一标识即为'video_id'）
    # group：唯一标识下的子DataFrame
    for vid, group in df.groupby('video_id'):
        video_url = group['video_url'].iloc[0]
        pub_date = extract_publish_date(video_url)
        comment_groups = []

        # 按用户分组，同一用户多条评论算一组
        for user, sub in group.groupby('user'):
            comments = sub['comment'].tolist()
            full_text = ' '.join([str(element) for element in comments])
            comment_groups.append({
                'user':        user,
                'comments':    comments,
                'language':    detect_language(full_text),
                'pure_emoji':  is_pure_emoji(full_text)
            })

        results[vid] = {
            'video_id':     vid,
            'video_url':    video_url,
            'publish_date': pub_date,
            'comment_groups': comment_groups
        }

        # 写入单个视频的 JSON
        out_path = os.path.join(OUTPUT_DIR, f'video_{vid}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results[vid], f, ensure_ascii=False, indent=2)
        print(f'✔ 已输出 {out_path}')

if __name__ == '__main__':
    main()
