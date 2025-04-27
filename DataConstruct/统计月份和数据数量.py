from datetime import datetime
import os
import json
import sys


def main():
    # 读取目录“D:\projects\TCSS\RT\Reciprocal-Tariff\符合条件的Tiktok数据\”下面所有的json文件
    input_dir = r"D:\projects\TCSS\RT\Reciprocal-Tariff\符合条件的Tiktok数据"
    month_count = {}
    # 遍历所有的json文件
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                # 读取json文件
                with open(os.path.join(root, file), "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                    # 遍历数据
                    comments_data = data["comment_groups"]
                    timestamp = data["publish_date"]
                    if timestamp:
                        # 将字符串m-d转换为日期格式
                        date = timestamp
                        # date = datetime.fromtimestamp(timestamp).strftime("%M-%d") # 报错
                        # 统计月份和数据数量
                        if date not in month_count.keys():
                            month_count[date] = {
                                "total_video": 1,
                                "English": 0,
                                "Chinese": 0,
                                "Other": 0,
                                "pure_emoji": 0,
                            }
                        else:
                            month_count[date]["total_video"] += 1

                        # 遍历并统计所有的语言和对应的数量
                        for comment in comments_data:
                            # print(comment["pure_emoji"] == True)
                            # print(type(comment["pure_emoji"]))
                            if comment["language"] == "English":
                                month_count[date]["English"] += 1
                            if comment["language"] == "Chinese":
                                month_count[date]["Chinese"] += 1
                            if comment["language"] == "Other":
                                month_count[date]["Other"] += 1
                            if comment["pure_emoji"] == True:
                                month_count[date]["pure_emoji"] += 1
    # 打印mothth_count，按照月份排序
    month_count = dict(sorted(month_count.items(), key=lambda x: datetime.strptime(x[0], "%m-%d")))
    # 打印月份和数据数量
    print("月份和数据数量：")
    for month, count in month_count.items():
        print(f"{month}: {count}")
        '''
        # 将结果保存到文件
        output_file = os.path.join(input_dir, "month_count.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(month_count, f, ensure_ascii=False, indent=4)
        '''
    # print(month_count)

if __name__ == "__main__":
    sys.exit(int(main() or 0))