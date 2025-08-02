from qwen_agent.agents import Assistant
import os

# Define LLM
llm_cfg = {
    'model': 'Qwen3-30B-A3B',
    'model_server': 'https://cloud.llm-ai.com/maas/v1',
    'api_key': 'sk-das5tysnwiphukwp',
}

# Define Tools
tools = [
    {
        'mcpServers': {  # 修改此处：用 uv run 替代 uvx
            'time': {
                'command': 'uv',
                'args': [
                    'run',
                    '--index-url', 'https://pypi.tuna.tsinghua.edu.cn/simple',  # 国内源加速
                    'mcp-server-time', '--local-timezone=Asia/Shanghai'
                ]
            },
            'fetch': {
                'command': 'uv',
                'args': [
                    'run',
                    '--index-url', 'https://pypi.tuna.tsinghua.edu.cn/simple',
                    'mcp-server-fetch'
                ]
            }
        }
    },
    'code_interpreter',  # Built-in tools
]

# Define Agent
bot = Assistant(llm=llm_cfg, function_list=tools)

# Streaming generation
messages = [{'role': 'user', 'content': 'https://qwenlm.github.io/blog/ Introduce the latest developments of Qwen'}]
for responses in bot.run(messages=messages):
    pass
print(responses)
