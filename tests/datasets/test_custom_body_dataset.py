import pytest
import os
from datasets import Dataset
from llm_eval.datasets.local import CustomBodyDataset

@pytest.fixture
def valid_jsonl_file(tmp_path):
    # 创建有效的JSONL测试文件
    file_path = tmp_path / "test.jsonl"
    content = """{"idx": "request-1", "body": {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}}
{"idx": "request-2", "body": {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}}"""
    file_path.write_text(content)
    return str(file_path)

@pytest.fixture
def invalid_json_file(tmp_path):
    # 创建无效的JSON测试文件
    file_path = tmp_path / "test.json"
    content = """[{"idx": "request-1", "body": {"model": "gpt-4"}}]"""
    file_path.write_text(content)
    return str(file_path)

def test_load_valid_jsonl(valid_jsonl_file):
    # 测试加载有效的JSONL文件
    dataset = CustomBodyDataset.load(valid_jsonl_file)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert dataset[0]["idx"] == "request-1"
    assert dataset[1]["body"]["model"] == "gpt-4"

def test_check_local_dataset_valid(valid_jsonl_file):
    # 测试验证有效的JSONL文件
    filename = CustomBodyDataset.check_local_dataset(valid_jsonl_file)
    assert filename == "test"

def test_check_local_dataset_invalid_extension(invalid_json_file):
    # 测试文件扩展名验证
    with pytest.raises(ValueError, match="not a JSONL file"):
        CustomBodyDataset.check_local_dataset(invalid_json_file)

def test_check_local_dataset_nonexistent():
    # 测试不存在的文件路径
    with pytest.raises(ValueError, match="does not exist"):
        CustomBodyDataset.check_local_dataset("nonexistent.jsonl")

def test_check_local_dataset_directory(tmp_path):
    # 测试传入目录路径
    with pytest.raises(ValueError, match="not a file"):
        CustomBodyDataset.check_local_dataset(str(tmp_path))

def test_req_generator(valid_jsonl_file):
    # 测试req_generator正常情况
    dataset = CustomBodyDataset(path=valid_jsonl_file)
    generator = dataset.req_generator(2)
    
    # 验证生成器返回的内容
    first_req = next(generator)
    assert isinstance(first_req, dict)
    assert first_req["idx"] == "request-1"
    assert first_req["body"]["model"] == "gpt-4"
    
    second_req = next(generator)
    assert second_req["idx"] == "request-2"
    
    # 验证生成器结束
    with pytest.raises(StopIteration):
        next(generator)

def test_req_generator_insufficient_data(valid_jsonl_file):
    # 测试请求数量超过数据集大小的情况
    dataset = CustomBodyDataset(path=valid_jsonl_file)
    with pytest.raises(AssertionError, match="Dataset num not enough"):
        list(dataset.req_generator(3))

def test_req_generator_zero_requests(valid_jsonl_file):
    # 测试请求数量为0的情况
    dataset = CustomBodyDataset(path=valid_jsonl_file)
    generator = dataset.req_generator(0)
    assert len(list(generator)) == 0

def test_req_generator_partial_requests(valid_jsonl_file):
    # 测试部分请求的情况
    dataset = CustomBodyDataset(path=valid_jsonl_file)
    generator = dataset.req_generator(1)
    requests = list(generator)
    assert len(requests) == 1
    assert requests[0]["idx"] == "request-1" 