import os
import dspy
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的变量

# 设置语言模型
lm = dspy.LM('ollama_chat/qwen2.5:32b', api_base=os.getenv("OLLAMA_A800_API_BASE"), api_key='', cache=False)
dspy.configure(lm=lm)


# 定义一个签名，它描述了输入输出的结构。
class QuestionAnswer(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()


# 创建一个模块，使用之前定义的签名。
qa_module = dspy.Predict(QuestionAnswer)

# 提出一个问题并获取答案。
question = "你是谁？"
response = qa_module(question=question)

print(response.answer)
