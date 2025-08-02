from llm_eval.datasets.math_500 import Math500
from llm_eval.utils.tokenizer import get_tokenizer


def test_load_dataset():
    dataset = Math500()
    print(dataset.dataset)

    num_request = 200
    tokenizer = get_tokenizer("/Users/howardchen/DeepSeek-R1", trust_remote_code=True)

    for req in dataset.perf_req_generator(num_requests=num_request, tokenizer=tokenizer):
        print(req)