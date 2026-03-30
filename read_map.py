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


# 使用示例
if __name__ == "__main__":
    map_list = parse_map_file("map.txt")

    # 打印结果
    print("map = [")
    for row in map_list:
        print(f"    {row},")
    print("]")

    # 或者直接赋值给变量
    map_variable = parse_map_file("map.txt")