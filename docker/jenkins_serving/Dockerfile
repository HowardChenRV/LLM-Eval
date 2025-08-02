FROM ubuntu:22.04

# 1. 安装 OpenJDK 21
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-21-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. 把 JAVA_HOME 写进环境变量
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# 3. 拷贝项目
WORKDIR /app/LLM-Eval

RUN ln -s /usr/bin/python3 /usr/local/bin/python

COPY . .

# 4. 升级 pip 并设置清华源
RUN python -m pip install --no-cache-dir --upgrade pip \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --trusted-host pypi.tuna.tsinghua.edu.cn && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 5. 安装项目依赖
RUN pip install -e .

CMD ["llm-eval", "serving_perf_eval", "-h"]
