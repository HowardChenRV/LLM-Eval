import json
import os

"""
    1. 获取自统计大盘: https://metabase.llm-ai.com/question/57-in-out-token-map
    2. 下载为json文件后格式如下:
    [
    {"In Token":"0","Out Token":"0","Count":"4,296,988"},
    {"In Token":"0","Out Token":"250","Count":"40,040"},
    {"In Token":"0","Out Token":"500","Count":"5,294"},
    {"In Token":"0","Out Token":"750","Count":"1,724"},
    ...
    ]
"""

def calculate_percentages(json_file_path):
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"Error: File '{json_file_path}' does not exist.")
        return

    # 加载 JSON 文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 去掉 Count 中的逗号并转换为整数
    for entry in data:
        entry["In Token"] = int(entry["In Token"].replace(",", ""))
        entry["Out Token"] = int(entry["Out Token"].replace(",", ""))
        entry["Count"] = int(entry["Count"].replace(",", ""))

    # 计算总数
    total_count = sum(entry["Count"] for entry in data)

    # 计算每组的百分比并过滤小于 0.01% 的项
    filtered_data = []
    for entry in data:
        entry["Percentage"] = (entry["Count"] / total_count) * 100
        if entry["Percentage"] >= 0.01:
            filtered_data.append(entry)

    # 按百分比从大到小排序
    sorted_data = sorted(filtered_data, key=lambda x: x["Percentage"], reverse=True)

    # 打印结果
    print("Results (Percentage >= 0.01%):")
    for entry in sorted_data:
        print(f"In Token: {entry['In Token']}-{entry['In Token']+500},\tOut Token: {entry['Out Token']}-{entry['Out Token']+250},\t"
              f"Count: {entry['Count']},\tPercentage: {entry['Percentage']:.4f}%")
        

file_path = "query_result_2024-11-15T19_49_17.430054+08_00.json"
calculate_percentages(file_path)

"""
2024年11月15日统计结果如下:
    Results (Percentage >= 0.01%):
    In Token: 0-500,        Out Token: 0-250,       Count: 3984918, Percentage: 73.5932%
    In Token: 500-1000,     Out Token: 0-250,       Count: 975396,  Percentage: 18.0136%
    In Token: 1000-1500,    Out Token: 0-250,       Count: 145965,  Percentage: 2.6957%
    In Token: 1500-2000,    Out Token: 0-250,       Count: 58177,   Percentage: 1.0744%
    In Token: 0-500,        Out Token: 250-500,     Count: 28814,   Percentage: 0.5321%
    In Token: 2000-2500,    Out Token: 0-250,       Count: 24099,   Percentage: 0.4451%
    In Token: 1000-1500,    Out Token: 500-750,     Count: 20270,   Percentage: 0.3743%
    In Token: 1000-1500,    Out Token: 750-1000,    Count: 17130,   Percentage: 0.3164%
    In Token: 1500-2000,    Out Token: 1000-1250,   Count: 16961,   Percentage: 0.3132%
    In Token: 1500-2000,    Out Token: 1250-1500,   Count: 10319,   Percentage: 0.1906%
    In Token: 2500-3000,    Out Token: 0-250,       Count: 10076,   Percentage: 0.1861%
    In Token: 1000-1500,    Out Token: 250-500,     Count: 9740,    Percentage: 0.1799%
    In Token: 1500-2000,    Out Token: 750-1000,    Count: 9462,    Percentage: 0.1747%
    In Token: 2000-2500,    Out Token: 1500-1750,   Count: 6472,    Percentage: 0.1195%
    In Token: 500-1000,     Out Token: 250-500,     Count: 6443,    Percentage: 0.1190%
    In Token: 3000-3500,    Out Token: 0-250,       Count: 5260,    Percentage: 0.0971%
    In Token: 0-500,        Out Token: 500-750,     Count: 4992,    Percentage: 0.0922%
    In Token: 2000-2500,    Out Token: 1750-2000,   Count: 4686,    Percentage: 0.0865%
    In Token: 1500-2000,    Out Token: 250-500,     Count: 3728,    Percentage: 0.0688%
    In Token: 2000-2500,    Out Token: 1250-1500,   Count: 3470,    Percentage: 0.0641%
    In Token: 3500-4000,    Out Token: 0-250,       Count: 2958,    Percentage: 0.0546%
    In Token: 1500-2000,    Out Token: 500-750,     Count: 2759,    Percentage: 0.0510%
    In Token: 2500-3000,    Out Token: 2000-2250,   Count: 2608,    Percentage: 0.0482%
    In Token: 1500-2000,    Out Token: 1500-1750,   Count: 2365,    Percentage: 0.0437%
    In Token: 2000-2500,    Out Token: 250-500,     Count: 2332,    Percentage: 0.0431%
    In Token: 500-1000,     Out Token: 500-750,     Count: 2201,    Percentage: 0.0406%
    In Token: 1000-1500,    Out Token: 1000-1250,   Count: 2180,    Percentage: 0.0403%
    In Token: 2500-3000,    Out Token: 2250-2500,   Count: 2128,    Percentage: 0.0393%
    In Token: 2000-2500,    Out Token: 1000-1250,   Count: 1861,    Percentage: 0.0344%
    In Token: 0-500,        Out Token: 750-1000,    Count: 1709,    Percentage: 0.0316%
    In Token: 4000-4500,    Out Token: 0-250,       Count: 1599,    Percentage: 0.0295%
    In Token: 2500-3000,    Out Token: 250-500,     Count: 1533,    Percentage: 0.0283%
    In Token: 2500-3000,    Out Token: 1750-2000,   Count: 1496,    Percentage: 0.0276%
    In Token: 500-1000,     Out Token: 750-1000,    Count: 1488,    Percentage: 0.0275%
    In Token: 2000-2500,    Out Token: 2000-2250,   Count: 1439,    Percentage: 0.0266%
    In Token: 3500-4000,    Out Token: 1000-1250,   Count: 1372,    Percentage: 0.0253%
    In Token: 2000-2500,    Out Token: 750-1000,    Count: 1313,    Percentage: 0.0242%
    In Token: 3000-3500,    Out Token: 750-1000,    Count: 1177,    Percentage: 0.0217%
    In Token: 3500-4000,    Out Token: 750-1000,    Count: 1128,    Percentage: 0.0208%
    In Token: 3000-3500,    Out Token: 1000-1250,   Count: 1116,    Percentage: 0.0206%
    In Token: 3000-3500,    Out Token: 2500-2750,   Count: 1062,    Percentage: 0.0196%
    In Token: 3000-3500,    Out Token: 250-500,     Count: 1021,    Percentage: 0.0189%
    In Token: 3000-3500,    Out Token: 2750-3000,   Count: 1010,    Percentage: 0.0187%
    In Token: 2500-3000,    Out Token: 1000-1250,   Count: 903,     Percentage: 0.0167%
    In Token: 2500-3000,    Out Token: 2500-2750,   Count: 881,     Percentage: 0.0163%
    In Token: 4500-5000,    Out Token: 0-250,       Count: 861,     Percentage: 0.0159%
    In Token: 2500-3000,    Out Token: 750-1000,    Count: 779,     Percentage: 0.0144%
    In Token: 2500-3000,    Out Token: 1500-1750,   Count: 765,     Percentage: 0.0141%
    In Token: 5000-5500,    Out Token: 0-250,       Count: 748,     Percentage: 0.0138%
    In Token: 2500-3000,    Out Token: 1250-1500,   Count: 729,     Percentage: 0.0135%
    In Token: 2000-2500,    Out Token: 500-750,     Count: 726,     Percentage: 0.0134%
    In Token: 3000-3500,    Out Token: 2250-2500,   Count: 638,     Percentage: 0.0118%
    In Token: 3500-4000,    Out Token: 250-500,     Count: 603,     Percentage: 0.0111%
    In Token: 3500-4000,    Out Token: 3500-3750,   Count: 549,     Percentage: 0.0101%
"""