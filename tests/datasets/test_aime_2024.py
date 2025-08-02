from llm_eval.datasets.aime_2024 import AIME2024
import json

def save_dataset_as_jsonl(dataset, file_path: str):
    """将 Hugging Face 的 Dataset 对象保存为 jsonl 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")
            
            
def test_load_dataset():
    dataset = AIME2024()
    print(dataset.dataset)
    

    dataset = AIME2024()
    save_dataset_as_jsonl(dataset.dataset, "AIME2024.jsonl")