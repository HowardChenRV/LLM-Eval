FROM quay.io/ascend/vllm-ascend:v0.8.5rc1

WORKDIR /app

COPY . .

# 1. 先强制通过清华源升级pip（关键步骤）
RUN python -m pip install --no-cache-dir --upgrade pip \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn
# 2. 然后配置永久镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装llm-eval核心包及常用可选依赖
RUN pip install -e .

# 设置容器启动命令
CMD ["llm-eval", "serving_perf_eval", "-h"]
