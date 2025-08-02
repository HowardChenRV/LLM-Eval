from llm_eval.datasets.lp_resume.resume_screening import ResumeScreening
from llm_eval.utils.tokenizer import get_tokenizer


def test_load_dataset():
    dataset = ResumeScreening()
    print(dataset.dataset)

    num_request = 10
    tokenizer = get_tokenizer("/Users/howardchen/DeepSeek-R1", trust_remote_code=True)

    for req in dataset.perf_req_generator(num_requests=num_request, tokenizer=tokenizer):
        print(req)