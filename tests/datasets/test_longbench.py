import pytest
from llm_eval.datasets.longbench.longbench import LongbenchDataset


def test_load_longbench():
    dataset = LongbenchDataset()
    print(dataset)