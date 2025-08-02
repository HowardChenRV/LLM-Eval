from typing import Union, Iterator, Dict

from datasets import load_dataset, Dataset
from llm_eval.utils.registry import register_dataset
from llm_eval.utils.datasets import download_dataset_by_hub
from llm_eval.datasets.local import CustomBodyDataset


@register_dataset("mmbench_cn")
class MMBenchCN(CustomBodyDataset):
    """
    MMBench是OpenCompass研究团队自建的视觉语言模型评测数据集
    Datasets: https://modelscope.cn/datasets/lmms-lab/MMBench
    """

    @staticmethod
    def load(path: str = "") -> Dataset:
        """
        Args:
            path: 可选的覆盖路径，如果为None则使用默认路径

        Returns:
            Dataset: 加载后的数据集对象
        """
        if path:
            dataset = load_dataset("json", data_files=path)
        else:
            dataset = download_dataset_by_hub("shallowdream16/MMBench_CN_test_4636", name="default", split="all")
        return dataset

    def req_generator(self, num_requests: int) -> Iterator[Dict]:
        dataset_num = len(self.dataset)
        assert dataset_num >= num_requests, f"Dataset num not enough, expect={num_requests}, actual={dataset_num}, Please reduce the num_requests."

        count = 0
        for data in self.dataset:
            if count >= num_requests:
                break
            # print(f"Generating request {count + 1}/{num_requests}: {data}")

            # 修改下key，兼容custom格式
            data["body"] = data["prompt_body"]
            data.pop("prompt_body")

            yield dict(data)
            count += 1
