# llama-bench-test 工具

## 概述

llama-bench-test 是一个自动化测试工具，用于运行 llama-bench 性能测试并收集结果。它支持多模型测试配置，并将测试结果发送到 Kafka 进行集中统计和分析。

## 功能特点

- 支持从 YAML 配置文件加载测试用例
- 可同时测试多个不同模型
- 实时输出测试日志和结果
- 测试结果自动上报到 Kafka
- 详细的错误处理和日志记录
- 支持 Git 分支和提交信息记录

## 使用说明

### 运行环境
windows AIPC，参考文档 xxx

### 依赖安装

```bash
# 依赖 python 3.10～3.12
pip install -r requirement.txt
```

### 配置文件

配置文件 `llama_bench_case.yaml` 示例：

```yaml
test_case:
  model_list:
    - model_name: Qwen2.5-1.5B-Instruct
      quantization_method: Q4_0
      model_path: \\path\to\model.gguf
    - model_name: Qwen2.5-3B-Instruct
      quantization_method: Q4_0
      model_path: \\path\to\model.gguf
  prompt_tokens: "64,128"  # 输入token数量
  gen_tokens: "32,64"      # 生成token数量

metrics:
  kafka:
    host: 8.140.201.231    # Kafka服务器地址
    port: 9094             # Kafka端口

logging:
  level: INFO              # 日志级别
  file_path: llama_bench_test_log  # 日志目录
  format: "%(asctime)s [%(levelname)s] - %(message)s - %(filename)s:%(lineno)d"

meta_info:
  tester: CI/CD            # 测试者信息
```

### 运行测试

基本用法：
```bash
python auto_run_llama_bench.py --llama-bench-path path/to/llama-bench.exe
```

带Git信息运行：
```bash
python auto_run_llama_bench.py \
  --llama-bench-path path/to/llama-bench.exe \
  --branch main \
  --commit abc123
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --llama-bench-path | llama-bench可执行文件路径 | llama-bench.exe |
| --branch | Git分支名称 | None |
| --commit | Git提交哈希 | None |

## 日志说明

测试日志会保存在 `llama_bench_test_log` 目录下，文件名格式为 `runtime_YYYYMMDD_HHMMSS.log`。

日志包含以下信息：
- 测试开始/结束时间
- 每个模型的测试结果
- 错误信息（如果有）
- 性能数据

## 注意事项

1. 确保模型文件路径可访问
2. Kafka服务器需要提前配置好
3. 测试时间可能较长，建议在性能稳定的环境中运行
4. 日志目录会自动创建，无需手动创建