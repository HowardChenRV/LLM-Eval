from llm_eval.datasets.sharegpt import SharegptDataset
from llm_eval.utils.tokenizer import get_tokenizer
from llm_eval.utils.registry import dataset_registry
from llm_eval.utils.datasets import dowanload_dataset_by_url
from datasets import load_dataset
from modelscope.msdatasets import MsDataset



def test_load_dataset():

    dataset_class = dataset_registry.get_class("sharegpt")
    print(dataset_registry.all_classes())
    file_path = "/share/datasets/tmp_share/chenyonghua/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
    dataset = dataset_class(path=file_path)
    print(dataset.dataset)

    num_request = 10
    tokenizer = get_tokenizer("/share/datasets/public_models/Meta-Llama-3-8B-Instruct/")
    for req in dataset.perf_req_generator(num_requests=num_request, tokenizer=tokenizer):
        print(req)

    
def test_download_dataset():
    dataset = SharegptDataset()
    print(dataset.dataset)
