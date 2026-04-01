import os
import json
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

def read_json_variable(file_path, key=None):
    """
    读取 JSON 文件中的变量。

    参数:
        file_path (str): JSON 文件的路径。
        key (str, 可选): 如果 JSON 是字典格式，指定要获取的特定键的值。如果为 None，则返回整个 JSON 对象。

    返回:
        dict, list, or any: 解析后的数据或指定键的值。如果出错则返回 None。
    """
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件 '{file_path}' 不存在。")
        return None

    try:
        # 2. 打开并读取文件 (建议指定 encoding='utf-8')
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 3. 如果指定了 key，则返回对应的值
        if key is not None:
            if isinstance(data, dict) and key in data:
                return data[key]
            else:
                print(f"⚠️ 警告: 键 '{key}' 在文件中未找到，或文件内容不是字典格式。")
                return None

        # 4. 如果没有指定 key，返回整个解析后的对象
        return data

    except json.JSONDecodeError as e:
        print(f"❌ 错误: JSON 格式无效 - {e}")
        return None
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        return None