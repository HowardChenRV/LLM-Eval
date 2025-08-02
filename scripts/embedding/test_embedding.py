from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = '/Users/howardchen/DeepSeek-R1'

# 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 输入 prompt
# prompt =  ", so I need to find the radius of a sphere where the volume and the surface area are numerically equal. Hmm, that means if the volume is, say, 100 cubic units, the surface area is also 100 square units. Even though they have different units, their numerical values are the same. Alright, let's start by recalling the formulas for the volume and surface area of a sphere.\n\nThe volume \\( V \\) of a sphere is given by \\( \\frac{4}{3}\\pi r^3 \\), right? And the surface area \\( A \\) is \\( 4\\pi r^2 \\). The problem says these two are numerically equal, so I can set them equal to each other:\n\n\\( \\frac{4}{3}\\pi r^3 = 4\\pi r^2 \\)\n\nHmm, okay. Let me write that down more clearly:\n\n\\[\n\\frac{4}{3}\\pi r^3 = 4\\pi r^2\n\\]\n\nNow, I need to solve for \\( r \\). Let's see. First, maybe I can simplify this equation by dividing both sides by something common. Both sides have a \\( 4\\pi r^2 \\), I think. Let me check.\n\nIf I divide both sides by \\( 4\\pi r^2 \\), that should be allowed as long as \\( r \\neq 0 \\), which makes sense because a sphere with radius 0 isn't really a sphere. So dividing both sides by \\( 4\\pi r^2 \\):\n\nLeft side: \\( \\frac{4}{3}\\pi r^3 \\div 4\\pi r^2 = \\frac{4}{3} \\div 4 \\times \\pi \\div \\pi \\times r^3 \\div r^2 \\)\n\nSimplifying each part:\n\n\\( \\frac{4}{3} \\div 4 = \\frac{4}{3} \\times \\frac{1}{4} = \\frac{1}{3} \\)\n\n\\( \\pi \\div \\pi = 1 \\)\n\n\\( r^3 \\div r^2 = r^{3-2} = r \\)\n\nSo left side simplifies to \\( \\frac{1}{3} \\times 1 \\times r = \\frac{r}{3} \\)\n\nRight side: \\( 4\\pi r^2 \\div 4\\pi r^2 = 1 \\)\n\nSo now the equation is:\n\n\\( \\frac{r}{3} = 1 \\)\n\nMultiply both sides by 3:\n\n\\( r = 3 \\)\n\nWait, so the radius is 3 units? Let me check that again to make sure I didn't make a mistake.\n\nStarting with the original equation:\n\n\\( \\frac{4}{3}\\pi r^3 = 4\\pi r^2 \\)\n\nDivide both sides by \\( 4\\pi \\):\n\n\\( \\frac{1}{3} r^3 = r^2 \\)\n\nThen multiply both sides by 3:\n\n\\( r^3 = 3r^2 \\)\n\nSubtract \\( 3r^2 \\) from both sides:\n\n\\( r^3 - 3r^2 = 0 \\)\n\nFactor out \\( r^2 \\):\n\n\\( r^2(r - 3) = 0 \\)\n\nSo, the solutions are \\( r^2 = 0 \\) or \\( r - 3 = 0 \\), which gives \\( r = 0 \\) or \\( r = 3 \\). Since radius can't be 0, the only valid solution is \\( r = 3 \\).\n\nOkay, that seems to check out. Let me verify by plugging back into the original formulas.\n\nSurface area: \\( 4\\pi (3)^2 = 4\\pi \\times 9 = 36\\pi \\)\n\nVolume: \\( \\frac{4}{3}\\pi (3)^3 = \\frac{4}{3}\\pi \\times 27 = 36\\pi \\)\n\nWait, both are 36π? So numerically, if we consider the numerical value without units, they are both 36π. But the problem states that the volume and surface area are numerically equal. Hmm, does that mean they want the numerical value ignoring the units? So even though surface area is in square units and volume is in cubic units, their numerical values are the same. So if the radius is 3, then the numerical values are both 36π. But 36π is not equal to 36π? Wait, they are equal. But π is a constant. So does that mean the answer is 3?\n\nWait, but the problem says \"numerically equal\". So if you just consider the numbers, not the units. So if the volume is 36π cubic units and the surface area is 36π square units, then numerically, they are both 36π. But π is approximately 3.1416, so 36π is about 113.097. So both the volume and surface area would be approximately 113.097 in their respective units. So numerically, they are equal. So the answer is 3.\n\nBut wait, the problem says \"the volume and surface area, in cubic units and square units, respectively, are numerically equal\". So they are equal as numbers, disregarding the units. So even though one is in cubic units and the other in square units, their numerical values are the same. So if the radius is 3, then both are 36π, which is the same number. So 36π is equal to 36π, so that works. So radius 3 is correct.\n\nBut let me think again. Suppose the radius was 3, then surface area is 4π*(3)^2 = 36π, and volume is (4/3)π*(3)^3 = 36π. So both are 36π. So numerically, they are equal. So 36π is equal to 36π. So that's correct.\n\nAlternatively, if the problem had said that the volume and surface area are equal in measure, considering units, that would be impossible because cubic units can't equal square units. But since it's just the numerical values, they can be equal. So 36π is numerically equal to 36π. So radius 3 is correct.\n\nTherefore, the answer is 3 units.\n\n**Final Answer**\nThe radius of the sphere is \\boxed{3} units.\n"
prompt =  "\n\nTo find the radius of a sphere whose volume and surface area are numerically equal, we start with the formulas for the volume \\( V \\) and surface area \\( A \\) of a sphere:\n\n\\[\nV = \\frac{4}{3}\\pi r^3\n\\]\n\\[\nA = 4\\pi r^2\n\\]\n\nWe set these equal to each other since their numerical values are the same:\n\n\\[\n\\frac{4}{3}\\pi r^3 = 4\\pi r^2\n\\]\n\nDividing both sides by \\( 4\\pi r^2 \\):\n\n\\[\n\\frac{1}{3}r = 1\n\\]\n\nSolving for \\( r \\):\n\n\\[\nr = 3\n\\]\n\nTo verify, we substitute \\( r = 3 \\) back into the original formulas:\n\n- Surface area: \\( 4\\pi (3)^2 = 36\\pi \\)\n- Volume: \\( \\frac{4}{3}\\pi (3)^3 = 36\\pi \\)\n\nBoth the surface area and volume are numerically equal to \\( 36\\pi \\), confirming the solution is correct.\n\nThus, the radius of the sphere is \\boxed{3} units."


# 分词和生成 token
tokens = tokenizer.tokenize(prompt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# 将 token IDs 转换为模型的输入格式
input_ids = torch.tensor([token_ids])

# 生成 embedding
# with torch.no_grad():
#     outputs = model(input_ids)
#     embeddings = outputs.last_hidden_state

# 输 prompt 分词、token、embedding 向量
print("Prompt分词:", tokens)
print("Token IDs:", token_ids)
# print("Embedding 向量:", embeddings)
print("Token len:", len(token_ids))
