def do_qwen_image(image_path,model_name):
    import requests
    import os
    import base64
    # 读取图片文件并转换为base64
    with open(image_path, 'rb') as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    api_key = "sk-das5uesnxk5skugn"
    url = "https://cloud.llm-ai.com/maas/v1/chat/completions"
    
    # api_key = "sk-5ba31bf7ccc146c5bceb420af9031bb8"
    # url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "请充当万能识别OCR助手，将图片中的所有文字、排版、图片内容信息完整输出，严格使用Markdown格式，100%忠实于图片内容与结构，无任何内容缺失或更改。保留所有文字、标点、空格、标题、分段、列表、引用、字体样式（如加粗、斜体、下划线）、字体大小等。数学公式使用LaTeX语法，代码按Markdown代码块格式输出，表格使用Markdown表格格式并完整呈现所有列与行。严格还原所有缩进、对齐、图标、分割线等布局。如果图片无文字信息，则简洁输出这个图片中的内容、场景、细节等信息。请严格遵循：输出的第一个字符即为图片内容，不得添加任何说明、解释、前缀、后缀内容。绝对禁止省略任何字符、符号或内容，忠实呈现原始内容和结构，不得生成主观信息，禁止生成解释或扩展性描述。不得省略任何内容，不得使用省略号代替内容。"
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "stream": False
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']


# res = do_qwen_image("/Users/howardchen/Dev/QA/LLM-Eval/tests/qwen-vl/20250507-181001.jpeg","qwen2.5-vl-7b-instruct")
# print(res)
res = do_qwen_image("/Users/howardchen/Dev/QA/LLM-Eval/tests/qwen-vl/20250507-181001.jpeg","qwen2.5-vl-32b-instruct")
print(res)
# res = do_qwen_image("/Users/howardchen/Dev/QA/LLM-Eval/tests/qwen-vl/20250507-181001.jpeg","qwen2.5-vl-72b-instruct")
# print(res)