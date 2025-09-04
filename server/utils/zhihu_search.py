

import requests
import time
from datetime import datetime
import pandas as pd


def get_zhihu_answer_comments(answer_id, limit=20, offset=0):
    """
    get commentes from zhihu answers
    :param answer_id: the targeted answer
    :param limit: maximum number of comments per request
    :param offset: offset (starting from offset.value)
    :return: 评论列表
    """
    url = f"https://www.zhihu.com/api/v4/answers/{answer_id}/root_comments"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": f"https://www.zhihu.com/question/{answer_id}",
    }

    params = {
        "order": "normal",
        "limit": limit,
        "offset": offset,
        "status": "open",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None


def parse_comments(comment_data):
    """
    解析评论数据
    :param comment_data: 原始评论数据
    :return: 解析后的评论列表
    """
    comments = []

    if not comment_data or "data" not in comment_data:
        return comments

    for item in comment_data["data"]:
        print(item)
        try:
            # 获取用户省份信息
            province = item.get("address_text", {})

            comment = {
                "id": item.get("id", ""),
                "content": item.get("content", ""),
                "created_time": datetime.fromtimestamp(item.get("created_time", 0)).strftime('%Y-%m-%d %H:%M:%S'),
                "user_id": item.get("author", {}).get("id", ""),
                "user_name": item.get("author", {}).get("member", {}).get("name", ""),
                "province": province,  # 用户省份
                "like_count": item.get("like_count", 0),  # 点赞数
                "child_comment_count": item.get("child_comment_count", 0),  # 子评论数
            }
            comments.append(comment)
        except Exception as e:
            print(f"解析评论出错: {e}, 原始数据: {item}")
            continue

    return comments


def get_all_comments(answer_id, max_count=100):
    """
    获取所有评论
    :param answer_id: 回答ID
    :param max_count: 最大获取数量
    :return: 所有评论列表
    """
    all_comments = []
    offset = 0
    limit = 20  # 每次请求获取20条

    while True:
        print(f"正在获取第 {offset // limit + 1} 页评论...")
        data = get_zhihu_answer_comments(answer_id, limit, offset)

        if not data or "data" not in data:
            print("没有更多评论或获取失败")
            break

        comments = parse_comments(data)
        all_comments.extend(comments)

        # 检查是否还有更多评论或达到最大数量
        print(data.get("paging", {}))
        print( len(comments))
        if data.get("paging", {}).get("is_end", True) and len(comments) == 0:
        # if data.get("paging", {}).get("is_end", True) or len(comments) == 0:
            print("已到达最后一页")
            break

        if len(all_comments) >= max_count:
            all_comments = all_comments[:max_count]
            print(f"已达到最大获取数量 {max_count}")
            break

        offset += limit
        time.sleep(2)  # 防止请求过于频繁

    return all_comments


def save_to_file(comments, answer_id, filename="zhihu_comments"):
    """
    将评论保存到文件
    :param comments: 评论列表
    :param filename: 文件名
    """
    try:
        o_file = '{}_{}.csv'.format(filename, answer_id)
        df = pd.DataFrame(comments)
        df.to_csv(o_file, index=False, encoding='utf-8')
            # json.dump(comments, f, ensure_ascii=False, indent=2)
        print(f"评论已保存到 {o_file}")
    except Exception as e:
        print(f"保存文件失败: {e}")


class Zhihu:

    def __init__(self):
        # 示例：替换为你想要爬取的知乎回答ID
        answer_ids = ['81964408445','82586149604','82493740255','81348057992','81748398040','81531639383'] #
        province_dict = {}
        for answer_id in answer_ids:
            print(f"开始爬取回答 {answer_id} 的评论...")
            comments = get_all_comments(answer_id, max_count=500)  # 最多获取100条评论


            if comments:
                print(f"共获取到 {len(comments)} 条评论")

                for i, comment in enumerate(comments, 1):  # 打印前3条作为示例
                    if i <= 3:
                        print(f"\n评论示例 {i}:")
                        print(f"用户ID: {comment['user_id']}")
                        print(f"用户名: {comment['user_name']}")
                        print(f"省份: {comment['province']}")
                        print(f"时间: {comment['created_time']}")
                        print(f"内容: {comment['content']}")
                        print(f"点赞数: {comment['like_count']}")

                    if comment['province'] not in province_dict:
                        province_dict[comment['province']] = 0
                    province_dict[comment['province']] = province_dict[comment['province']] + 1
                        # 保存到文件
                save_to_file(comments, answer_id)
            else:
                print("没有获取到任何评论")

        for prvn in province_dict:
            print("{}\t{}".format(prvn, province_dict[prvn]))