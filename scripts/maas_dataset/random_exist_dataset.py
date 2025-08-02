"""如果需要的输入输出范围比较窄，符合的数据比较少，可以用这个脚本扩增下"""

import json
import random

input_path = "maas_kimi_k2_in9345_out315_num50.json"
multiple = 200      # 扩增多少倍，预期数 / 现有数


def expand_date_by_shuffle():
    with open(input_path, "r") as f:
        datas = json.load(f)
        print(len(datas))

    res = []
    for item in datas:
        res.append(item)
        prompt_template = item["prompt"].split(" ")
        for i in range(multiple):
            random.shuffle(prompt_template)
            res.append({
                "prompt": " ".join(prompt_template),
                "prompt_len": item["prompt_len"],
                "output_len": item["output_len"]
            })

    print(len(res))
    file_name = input_path.replace('num50', 'num10050')
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(res, file, ensure_ascii=False, indent=4)
    print(file_name)
    return


if __name__ == '__main__':
    expand_date_by_shuffle()
