import json
import requests
import re
from llm_eval.datasets.math_500 import Math500

system_prompt = """
你是一个数据治理专家，负责处理题目数据并生成标准化 JSON 输出，只返回 JSON 格式的输出，禁止返回任何解释性文字。请严格按照以下要求执行，并确保>所有公式用 LaTeX 格式标注。以下是你的任务流程：
1.任务背景
    你负责处理题目数据，包括检查题干完整性、提取题目内容，并确保所有公式用 LaTeX 格式标注只返回 JSON 格式的输出，禁止返回任何解释性文字。
2.任务步骤
    步骤 1：题干完整性检查
        检查内容：判断题干内容是否完整。
        错误判定：如果题干中出现以下关键词（如 "如图"、"下图"、"上图"、"图中"、"in the picture"、"inferred from the passage"、"According to the text"、"Chapter 4 of your"）且缺少完整内容，则判定为题干错误。
        返回格式：如果题干不完整，返回空 JSON 对象 {}。
    步骤 2：题干正确性检查
        检查内容：确认题干内容是否正确或有缺失。
        公式处理：如果题干涉及公式，将原内容公式按 LaTeX 源码替换并用 [tex] 标注格式返回。
        阶段判断：根据题干判断阶段（small school、middle school、high school、College）。
        内容格式：过滤 HTML 标签，并将 <br>、<br />、</p > 替换为 \n。
        错误判定：如果题干有问题（如逻辑不清、信息不全），返回空 JSON 对象 {}。
    步骤 3：题型判断
        选择题：qtype 为 option。
        解答题：qtype 为 solution。
        判断题：qtype 为 judge。
        填空题：qtype 为 blank。
    步骤 4：题干和选项分离
        题干内容：提取题干内容，放入 question 字段。
        选项内容：如果是选择题，将选项内容放入 options 字段，格式为 "A: 选项1\nB: 选项2\nC: 选项3\nD: 选项4"。
        填空题处理：如果题型为填空题，题干中的下划线（___ 或 ________）需保留。
   步骤 5：答案和解析处理
        答案提取：根据题干 question 从 answers_content 提取题目答案，放入 answer 字段。
        如果是选择题，answer 字段应为选项标识（如 A、B、C、D）。如果选项缺失，返回空 JSON 对象 {}。
        如果是填空题，answer 字段应为填空内容。
        解析提取：将解析内容放入 analysis 字段。
        公式处理：检查 answer 和 analysis 内容中的公式，按 LaTeX 源码替换并用 [tex] 标注格式。
        内容格式：过滤 HTML 标签，并将 <br>、<br />、</p > 替换为 \n
   步骤 6：难度标注
        难度等级：标注题的难度，字段为 difficulty，范围为 1-5 个等级（1 为最简单，5 为最难）。
3. 输出格式
    返回json字符串，而不是一个包含 Markdown 代码块，包含以下字段：
    id: 题目 ID（字符串）
    subjectName: 学科名称（字符串）
    question: 题目内容（字符串）
answer: 答案内容（字符串）
    gradeGroupName: 阶段（字符串，可选值为 small school、middle school、high school、College）
    analysis: 解析内容（字符串）
    difficulty: 难度等级（整数，范围 1-5）
    qtype: 题目类型（字符串，可选值为 option、solution、judge、blank）
    options: 选择题选项（字符串，格式为 "A: 选项1\nB: 选项2\nC: 选项3\nD: 选项4"，仅当 qtype 为 option 时存在）
    knowledge: 题的知识点（字符串）
4. 注意事项
    返回格式：只返回JSON 格式的输出，禁止返回任何解释性文字。
    公式标注：所有公式用 [tex] 标注，例如 [tex]f(x) = x^2[/tex]。
    填空题下划线：如果题型为填空题，题干中的下划线（___ 或 ________）需保留。
    保留图表格式：如果题干中包含图表（如图片、表格等），保留其原始格式，不做修改。
    内容格式：过滤 HTML 标签，并将 <br>、<br />、</p > 替换为 \n。
    换行符：确保所有换行符使用 \n，而不是 <br>、<br />、<br />\n、<br > 或其他格式。
    选择题答案：如果题型为 option，answer 字段应为选项标识（如 A、B、C、D），而不是选项值或选项加值。"""

# 无问
DEEPSEEK_API_URL = "https://cloud.llm-ai.com/maas/deepseek-v3/nvidia/chat/completions"
API_KEY = "sk-c7ano2vnaszd6hzt"
model = "deepseek-v3"

# deepseek官方
# DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
# API_KEY = "sk-f8cbff9440bb4c25937a498de5d2d9c1"
# model = "deepseek-chat"

# 阿里云百炼
# DEEPSEEK_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
# API_KEY = "sk-cf5186e8e3f84479ac9ddeb3ee544e90"
# model = "deepseek-v3"

dataset = Math500()

user_prompt = json.dumps(dataset.dataset[0])
# user_prompt = "你是谁"
print("user_prompt=",user_prompt)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]
# 调用 Deepseek API
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
data = {
    "model": model,
    "messages": messages,
    "stream": False,
    "top_p": 1,
    # "presence_penalty": 1,
    # "response_format" : {
    # 'type': 'json_object'
    # }
}
response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
# print(response.text)
if response.status_code == 200:
    content = response.json()['choices'][0]['message']['content']
    print("-"*100)
    print(content)
    print("-"*100)
    try:
        json_pattern = r'```json\s*({.*?})\s*```'
        match = re.search(json_pattern, content, re.DOTALL)
        json_content = match.group(1).strip()
        insert_one = json.loads(json_content)
    except:
        insert_one = {}