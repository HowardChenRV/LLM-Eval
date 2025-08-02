### 快速执行
视具体情况修改配置
   ```shell
    # 通过命令行修改测试参数执行
    llm-eval lenovo_perf_eval --env production --port 8060 --model {tokenizer目录} --warmup-times 1 --repetition 3 --output-prefix {测试报告命名}
    # 通过yaml配置修改测试参数执行
    llm-eval lenovo_perf_eval --env production --use-yaml --case-path examples/edge_aipc_eval/Qwen2.5-1.5B-Instruct.yaml
   ```