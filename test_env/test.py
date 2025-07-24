import dspy
import os

from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的变量

# 设置语言模型
# DSPy 中的语言模型默认是缓存的。如果重复相同的调用，将获得相同的输出。但可以通过设置 cache=False 来关闭缓存。
lm = dspy.LM('openai/qwen2.5-32b-instruct', api_base=os.getenv('QW_API_BASE'), api_key=os.getenv('QW_API_KEY'),
             cache=False)
dspy.configure(lm=lm)

# Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
qa = dspy.ChainOfThought('question -> answer')

# Run with the default LM configured with `dspy.configure` above.
response = qa(question="陕西省的省会是哪里？")
print(response.answer)
