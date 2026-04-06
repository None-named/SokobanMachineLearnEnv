import os
import json
from pathlib import Path


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


def get_latest_item(directory_path, file_only=False, dir_only=False):
    """
    获取指定目录下最新的文件或文件夹路径。

    :param directory_path: 目标目录路径 (str 或 Path)
    :param file_only: 如果为 True，只查找文件
    :param dir_only: 如果为 True，只查找文件夹
    :return: 最新的 Path 对象，如果目录为空则返回 None
    """
    path = Path(directory_path)

    if not path.exists():
        raise FileNotFoundError(f"目录不存在: {directory_path}")

    # 获取目录下的所有条目
    items = list(path.iterdir())

    # 根据参数进行过滤
    if file_only:
        items = [item for item in items if item.is_file()]
    elif dir_only:
        items = [item for item in items if item.is_dir()]

    if not items:
        return None

    # 使用 os.path.getmtime 获取修改时间并排序
    # key 函数返回每个路径的修改时间戳
    latest_item = max(items, key=lambda x: os.path.getmtime(x))

    return latest_item


def get_latest_file_concise(folder_path, include_dirs=False):
    """获取最新文件/文件夹，include_dirs 控制是否包含文件夹"""
    # 根据 include_dirs 参数决定获取列表的方式
    if include_dirs:
        # 获取所有文件和文件夹
        all_items = os.listdir(folder_path)
        # 过滤出确实存在的项（文件和文件夹）
        all_items = [item for item in all_items if os.path.exists(os.path.join(folder_path, item))]
    else:
        # 只获取文件
        all_items = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    if not all_items:
        return None

    # 使用 lambda 函数作为 key 来排序
    latest_item_name = max(all_items, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
    return os.path.join(folder_path, latest_item_name)


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


def collect_txt_files(root_dir, recursive=False):
    """
    收集指定目录下的所有 .txt 文件。

    参数:
        root_dir (str): 根目录路径
        recursive (bool): 是否递归搜索子目录 (默认: False)

    返回:
        list: 包含所有 .txt 文件绝对路径的列表
    """
    root_path = Path(root_dir)
    txt_files = []

    if recursive:
        # 使用 rglob 进行递归搜索 (包括子目录)
        # rglob('*.txt') 相当于 os.walk 递归查找
        pattern = '*.txt'
        files = root_path.rglob(pattern)
    else:
        # 使用 glob 仅搜索当前目录
        pattern = '*.txt'
        files = root_path.glob(pattern)

    # 过滤并确保只获取文件 (排除同名目录的情况)
    txt_files = [str(file) for file in files if file.is_file()]

    return txt_files


def get_directories(root_dir, recursive=False):
    """
    获取目录下的所有文件夹。

    参数:
        root_dir (str): 目标根目录路径
        recursive (bool): 是否递归获取所有子目录 (默认: False)

    返回:
        list: 包含所有文件夹路径的列表
    """
    root_path = Path(root_dir)
    dirs = []

    if recursive:
        # rglob('*') 会递归匹配所有路径
        # 配合 filter 筛选出目录
        dirs = [str(d) for d in root_path.rglob('*') if d.is_dir()]
    else:
        # iterdir() 仅遍历当前层级
        dirs = [str(d) for d in root_path.iterdir() if d.is_dir()]

    return dirs


def get_subfolder_names(directory_path, recursive=False):
    """
    获取指定目录下所有子文件夹的名称。

    参数:
        directory_path (str 或 Path): 要扫描的目录路径。
        recursive (bool): 是否递归获取所有层级的子文件夹。默认为 False。

    返回:
        list: 子文件夹名称的列表。如果 recursive=False，返回直接子文件夹；
              如果 recursive=True，返回所有层级的子文件夹（使用相对路径）。

    示例:
        >>> get_subfolder_names("./my_folder")
        ['docs', 'images', 'src']

        >>> get_subfolder_names("./my_folder", recursive=True)
        ['docs', 'images', 'src', 'docs/api', 'docs/guides', 'images/icons']
    """
    path = Path(directory_path)
    if not path.is_dir():
        return []

    if not recursive:
        # 仅获取直接子文件夹
        return [item.name for item in path.iterdir() if item.is_dir()]
    else:
        # 递归获取所有层级的子文件夹
        result = []
        for item in path.iterdir():
            if item.is_dir():
                result.append(item.name)
                # 递归获取子文件夹下的内容，并添加相对路径前缀
                sub_folders = get_subfolder_names(item, recursive=True)
                result.extend(f"{item.name}/{sub}" for sub in sub_folders)
        return result


def split_list(lst, chunk_size):
    """
    将列表 lst 按 chunk_size 的长度分割成子列表
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
