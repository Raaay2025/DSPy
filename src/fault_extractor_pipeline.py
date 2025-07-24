import os
import json
import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate import Evaluate
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from src.dsp_model import FaultExtractor
from src.bert_utils import bert_similarity_metric
from dspy import Example


def load_data(file_path):
    """
    加载 JSON 数据文件。
    :param file_path: 文件路径
    :return: 加载后的数据列表
    """
    with open(file_path, "r", encoding="utf-8-sig") as f:
        samples = json.load(f)
    print("读取样本数:", len(samples))
    print("示例数据:", samples[:2])
    return samples


def build_dspy_dataset(samples):
    """
    将原始数据转换为 DSPy 的 Example 格式。
    :param samples: 原始数据列表
    :return: DSPy Example 列表
    """
    train_set = []
    for sample in samples:
        example = Example({
            'input': sample['input'],
            **sample.get('extract', {})
        }).with_inputs('input')
        train_set.append(example)
    print("构建数据集完成，示例数据:", train_set[:2])
    return train_set


def split_dataset(dataset, test_size=0.2, random_state=42):
    """
    使用 train_test_split 划分训练集和验证集。
    :param dataset: DSPy Example 列表
    :param test_size: 验证集比例
    :param random_state: 随机种子
    :return: 划分后的 trainset 和 valset
    """
    trainset, valset = train_test_split(dataset, test_size=test_size, random_state=random_state)
    print("训练集样本数:", len(trainset))
    print("验证集样本数:", len(valset))
    print("示例数据:", trainset[:2], valset[:2])
    return trainset, valset


def configure_language_model():
    """
    加载环境变量并配置 Ollama 语言模型。
    :return: 初始化的 FaultExtractor 模型
    """
    load_dotenv()  # 加载 .env 文件中的变量

    lm = dspy.LM(
        'ollama_chat/qwen2.5:32b',
        api_base=os.getenv("OLLAMA_A800_API_BASE"),
        api_key='',
        cache=False
    )
    dspy.configure(lm=lm)
    print("语言模型已配置")

    model = FaultExtractor()
    return model


def train_model(model, train_set):
    """
    使用 BootstrapFewShotWithRandomSearch 优化模型。
    :param model: 待优化的模型
    :param train_set: 训练集
    :return: 优化后的模型
    """
    config = dict(
        max_labeled_demos=4,
        max_bootstrapped_demos=4,
        num_candidate_programs=10,
        num_threads=20,
    )

    teleprompter = BootstrapFewShotWithRandomSearch(
        metric=bert_similarity_metric,
        **config
    )

    print("开始模型优化...")
    optimized_module = teleprompter.compile(model, trainset=train_set)
    print("模型优化完成")
    return optimized_module


def save_model(model, model_path="optimized_fault_extractor.json"):
    """
    保存优化后的模型。
    :param model: 模型对象
    :param model_path: 保存路径
    """
    model.save(model_path)
    print("模型已保存至:", model_path)


def evaluate_model(model, val_set, name="模型"):
    """
    在验证集上评估模型。
    :param model: 模型对象
    :param val_set: 验证集
    :param name: 模型名称（用于输出）
    """
    evaluator = Evaluate(devset=val_set,
                         metric=bert_similarity_metric,
                         num_threads=5,
                         display_progress=True)
    score = evaluator(model)
    print(f"{name}评分: {score}")


def main():
    # 加载数据
    file_path = "../raw_data/故障类.json"
    samples = load_data(file_path)

    # 构建 DSPy 数据集
    dataset = build_dspy_dataset(samples)

    # 划分训练集和验证集
    train_set, val_set = split_dataset(dataset)

    # 配置语言模型、模型初始化
    model = configure_language_model()

    # 模型训练
    optimized_model = train_model(model, train_set)

    # 保存模型
    save_model(optimized_model)

    # 评估优化前后模型
    evaluate_model(model, val_set, "优化前模块")
    evaluate_model(optimized_model, val_set, "优化后模块")


if __name__ == "__main__":
    main()
