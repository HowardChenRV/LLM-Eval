#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict, Any

def generate_prompt_body(model: str, system_msg: str, user_msg: str) -> Dict[str, Any]:
    """生成prompt_body字典"""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    }

def generate_dataset(num_samples: int, model: str, system_msg: str, user_msg: str) -> List[Dict[str, Any]]:
    """生成数据集样本"""
    dataset = []
    for i in range(1, num_samples + 1):
        dataset.append({
            "idx": f"request-{i}",
            "prompt_body": generate_prompt_body(model, system_msg, user_msg)
        })
    return dataset

def save_to_jsonl(data: List[Dict[str, Any]], output_path: str) -> None:
    """保存数据到jsonl文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='生成适配CustomBodyDataset的jsonl文件')
    parser.add_argument('output_path', type=str, help='输出文件路径(.jsonl)')
    parser.add_argument('--num_samples', type=int, default=10, help='生成样本数量(默认:10)')
    parser.add_argument('--model', type=str, default="gpt-4o-mini", help='模型名称(默认:gpt-4o-mini)')
    parser.add_argument('--system_msg', type=str, default="You are a helpful assistant.", 
                       help='系统消息(默认:"You are a helpful assistant.")')
    parser.add_argument('--user_msg', type=str, default="What is 2+2?", 
                       help='用户消息(默认:"What is 2+2?")')
    
    args = parser.parse_args()
    
    if not args.output_path.lower().endswith('.jsonl'):
        print("错误: 输出文件必须以.jsonl结尾")
        return
    
    dataset = generate_dataset(args.num_samples, args.model, args.system_msg, args.user_msg)
    save_to_jsonl(dataset, args.output_path)
    print(f"成功生成{args.num_samples}条样本数据到: {args.output_path}")

if __name__ == '__main__':
    main()