### Quick Execution
Modify configuration as needed
   ```shell
    # Execute by modifying test parameters via command line
    llm-eval lenovo_perf_eval --env production --port 8060 --model {tokenizer directory} --warmup-times 1 --repetition 3 --output-prefix {test report naming}
    # Execute by modifying test parameters via yaml configuration
    llm-eval lenovo_perf_eval --env production --use-yaml --case-path examples/edge_aipc_eval/Qwen2.5-1.5B-Instruct.yaml