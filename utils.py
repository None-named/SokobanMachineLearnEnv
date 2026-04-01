import os
def parse_map_file(file_path):
    """
    读取地图文件并转换为二维列表

    参数:
        file_path (str): 地图文件的路径

    返回:
        list: 二维列表形式的地图数据
    """
    map_data = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 去除行尾的换行符和首尾空白字符
                line = line.strip()

                # 跳过空行
                if not line:
                    continue

                # 将每一行字符串转换为字符列表
                row = list(line)
                map_data.append(row)

        return map_data

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
        return []
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return []

def get_latest_file_concise(folder_path):
    """更简洁的获取最新文件的方式"""
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    if not all_files:
        return None

    # 使用 lambda 函数作为 key 来排序
    latest_filename = max(all_files, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
    return os.path.join(folder_path, latest_filename)