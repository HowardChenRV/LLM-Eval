import os
from typing import Optional
from datasets import Dataset
from llm_eval.utils.registry import register_dataset
from llm_eval.datasets.local import CustomBodyDataset


@register_dataset("yuanbao_ds_r1")
class YuanbaoDSR1(CustomBodyDataset):
    """腾讯元宝R1版本数据集"""
    
    @staticmethod
    def load(path: Optional[str] = None) -> Dataset:
        """加载R1版本数据集
        
        Args:
            path: 可选的覆盖路径，如果为None则使用默认路径
            
        Returns:
            Dataset: 加载后的数据集对象
        """
        default_path = os.path.join(
            os.path.dirname(__file__),
            "tencent_yuanbao_deepseek_r1_20250307_num_651.jsonl"
        )
        return CustomBodyDataset.load(path or default_path)


@register_dataset("yuanbao_ds_v3")
class YuanbaoDSV3(CustomBodyDataset):
    """腾讯元宝V3版本数据集"""
    
    @staticmethod
    def load(path: Optional[str] = None) -> Dataset:
        """加载V3版本数据集
        
        Args:
            path: 可选的覆盖路径，如果为None则使用默认路径
            
        Returns:
            Dataset: 加载后的数据集对象
        """
        default_path = os.path.join(
            os.path.dirname(__file__),
            "tencent_yuanbao_deepseek_v3_20250306_num_649.jsonl"
        )
        return CustomBodyDataset.load(path or default_path)