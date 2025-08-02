from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class TestType(Enum):
    """
    测试类型枚举
    """
    STATIC_INFERENCE_PERFORMANCE = 1  # 推理性能 - 静态
    SERVING_INFERENCE_PERFORMANCE = 2  # 推理性能 - serving
    STATIC_INFERENCE_CORRECTNESS = 3  # 推理正确性 - 静态
    SERVING_INFERENCE_CORRECTNESS = 4  # 推理正确性 - serving
    MODEL_INFERENCE_EVALUATION = 5  # 模型推理评价
    LENOVO_DEMO_PERFORMANCE = 6 # 联想AIPC lenovo-demo 性能测试专用
    LLAMA_BENCH_PERFORMANCE = 7 # 联想AIPC llama-bench 性能测试专用


class DataType(Enum):
    """
    数据类型枚举
    """
    TEST_STATISTIC = 1  # 本次测试结果的统计数据
    TEST_PROCESS = 2  # 本次测试的过程记录数据，需要时存储


class DatasetSource(Enum):
    """
    数据集来源
    """
    OPEN_SOURCE = 0  # 开源
    ORIGINAL = 1  # 原创闭源


class Language(Enum):
    """
    数据集使用的语言
    """
    ENGLISH = 0  # 英文
    CHINESE = 1  # 中文
    MIXED = 2  # 混合


class ModelSource(Enum):
    """
    模型来源
    """
    OPEN_SOURCE = 0  # 开源模型
    API = 1  # 闭源API


@dataclass
class EvaluationResult:
    """
    评测结果数据
    """
    dataset_name: str  # 数据集名称
    dataset_source: DatasetSource  # 数据集来源
    version: str  # 数据集版本
    dataset_category: str  # 子数据集分类，没有子数据集保持和源数据集一致
    score: float  # 测试得分
    language: Optional[Language] = None  # 数据集语言
    prompt_num: Optional[int] = None  # 题目总数
    match_num: Optional[int] = None  # 与baseline的匹配数
    download_url: Optional[str] = None  # 测试结果下载地址
    tester: Optional[str] = None  # 测试人
    hardware: Optional[str] = None  # 芯片
    hardware_num: Optional[int] = None  # 芯片数
    model: Optional[str] = None  # 模型
    quantization_method: Optional[str] = None  # 量化方法
    model_source: Optional[ModelSource] = None  # 模型来源
    framework: Optional[str] = None  # 推理引擎
    framework_version: Optional[str] = None  # 引擎版本
    test_source: Optional[str] = None  # 测试来源
    labels: Optional[Dict[str, Any]] = None  # 自定义标签
