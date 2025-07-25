import os
import json
import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from src.dsp_model import *

def read_json_file(file_path):
    """
    读取 JSON 文件，提取 input 和 faulty_call 字段。
    :param file_path: 文件路径
    :return: 包含 input 和 label 的字典列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        try:
            items = json.load(f)
            for item in items:
                input_text = item.get('input', '').strip()
                label = item.get('faulty_call', '-1').strip()
                if input_text and label in ('0', '1'):
                    data.append({
                        'input': input_text,
                        'label': '故障' if label == '1' else '非故障'
                    })
        except json.JSONDecodeError as e:
            print(f"解析错误：{e}")
    return data


def load_data(project_root):
    """
    加载故障类和非故障类数据并合并。
    :param project_root: 项目根目录
    :return: 合并后的数据列表
    """
    fault_file = os.path.join(project_root, "raw_data", "故障类.json")
    non_fault_file = os.path.join(project_root, "raw_data", "非故障类.json")

    print("故障类文件路径:", fault_file)
    print("非故障类文件路径:", non_fault_file)

    fault_data = read_json_file(fault_file)
    non_fault_data = read_json_file(non_fault_file)

    all_data = fault_data + non_fault_data

    print("读取样本数:", len(all_data))
    print("故障样本数:", len(fault_data))
    print("非故障样本数:", len(non_fault_data))
    print("示例数据:", all_data[0])

    return all_data


def split_dataset(data, test_size=0.2, random_state=42):
    """
    使用 train_test_split 划分训练集和验证集。
    :param data: 原始数据
    :param test_size: 验证集比例
    :param random_state: 随机种子
    :return: 划分后的数据集
    """
    X = [item['input'] for item in data]
    y = [item['label'] for item in data]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print("总样本数:", len(data))
    print("训练集样本数:", len(X_train))
    print("验证集样本数:", len(X_val))

    return X_train, X_val, y_train, y_val


def build_dspy_datasets(X_train, y_train, X_val, y_val):
    """
    将数据转换为 DSPy 的 Example 格式。
    :param X_train: 训练集输入
    :param y_train: 训练集标签
    :param X_val: 验证集输入
    :param y_val: 验证集标签
    :return: train_set, val_set
    """
    train_set = [
        dspy.Example(input=text, label=label).with_inputs("input")
        for text, label in zip(X_train, y_train)
    ]

    val_set = [
        dspy.Example(input=text, label=label).with_inputs("input")
        for text, label in zip(X_val, y_val)
    ]

    return train_set, val_set


class ClassificationSignature(dspy.Signature):
    """中文故障分类任务的输入输出定义"""
    input = dspy.InputField(desc="输入文本")
    label = dspy.OutputField(desc="分类标签，“故障”或“非故障”")


def configure_language_model_cot():
    """
    配置 Ollama 语言模型。
    """
    load_dotenv()  # 加载 .env 文件中的变量

    lm = dspy.LM(
        'ollama_chat/qwen2.5:32b',
        api_base=os.getenv("OLLAMA_A800_API_BASE"),
        api_key='',
        cache=False
    )
    dspy.configure(lm=lm)

    model = FaultClassifierChainOfThought()
    return model


def configure_language_model_predict():
    """
    配置 Ollama 语言模型。
    """
    load_dotenv()  # 加载 .env 文件中的变量

    lm = dspy.LM(
        'ollama_chat/qwen2.5:32b',
        api_base=os.getenv("OLLAMA_A800_API_BASE"),
        api_key='',
        cache=False
    )
    dspy.configure(lm=lm)

    model = FaultClassifierPredict()
    return model


def metric(gold, pred, trace=None):
    """
    评估指标：判断预测是否正确。
    """
    return gold.label == pred.label


def train_model_bfswrs(model, train_set):
    """
    使用 BootstrapFewShotWithRandomSearch 优化模型。
    """
    config = dict(
        max_labeled_demos=4,
        max_bootstrapped_demos=4,
        num_candidate_programs=20,
        num_threads=10
    )

    teleprompter = BootstrapFewShotWithRandomSearch(metric=metric, **config)
    optimized_model = teleprompter.compile(model, trainset=train_set)

    return optimized_model


def train_model_v2(model, train_set):
    """
    使用 MIPROv2 优化模型
    :param model: 待优化的模型（FaultClassifier 实例）
    :param train_set: DSPy Example 列表
    :return: 优化后的模型
    """

    # MIPROv2 初始化参数
    # 用于初始化优化器，不会在 compile 阶段动态改变
    init_config = dict(
        metric=metric,  # 评估函数，判断预测是否正确（必填）
        max_bootstrapped_demos=8,  # 每个样本中使用的自举示例数（推荐 6~8）
        max_labeled_demos=8,  # 每个样本中使用的标注示例数（推荐 6~8）
        num_candidates=40,  # 每轮生成的候选 prompt 数量（推荐 30~50）
        num_threads=10,  # 并行线程数（根据 CPU 核心数调整）
        init_temperature=0.7,  # 生成 prompt 的温度（降低温度让 prompt 更稳定）
        seed=42,  # 随机种子（保证可复现）
        verbose=True,  # 显示优化过程（调试时建议开启）
        track_stats=True,  # 是否记录统计信息（用于分析优化过程）
        auto=None,  # 设置为 None 以手动配置参数
        max_errors=10,  # 最大错误容忍数（防止优化过程崩溃）
        log_dir="logs/mipro_runs"  # 日志保存路径（用于记录 prompt 演变）
    )

    # MIPROv2 编译参数
    # 用于控制优化过程中的训练行为
    compile_config = dict(
        num_trials=40,  # 总共优化多少轮（推荐 30~50）
        minibatch=True,  # 使用小批量验证（适用于小样本数据）
        minibatch_size=30,  # 小批量大小（推荐 20~30，防止资源占用过高）
        minibatch_full_eval_steps=10,  # 每隔多少步进行一次全量验证（推荐 5~10）
        program_aware_proposer=True,  # 使用程序感知的 proposer（提升 prompt 质量）
        data_aware_proposer=True,  # 使用数据感知的 proposer（利用数据特征）
        view_data_batch_size=10,  # 查看数据的批量大小（影响 proposer 的上下文）
        tip_aware_proposer=True,  # 使用 TIP 意识 proposer（提升泛化能力）
        fewshot_aware_proposer=True,  # 使用少样本意识 proposer（提升 few-shot 效果）
        requires_permission_to_run=False,  # 不需要权限运行（用于跳过权限检查）
        provide_traceback=None  # 不提供 traceback（用于控制错误输出）
    )

    # 初始化优化器
    teleprompter = MIPROv2(**init_config)

    print("开始模型优化...")
    # 执行优化编译
    optimized_model = teleprompter.compile(
        student=model,
        trainset=train_set,
        **compile_config
    )
    print("模型优化完成")

    return optimized_model


def evaluate_model(model, val_set):
    """
    在验证集上评估模型性能。
    """
    evaluator = Evaluate(devset=val_set, metric=metric, num_threads=4, display_progress=True)
    score = evaluator(model)
    print("验证集准确率:", score)
    return score


def save_model(model, path="optimized_fault_classifier.json"):
    """
    保存优化后的模型。
    """
    model.save(path)
    print("模型已保存至:", path)


def load_model(model_path="optimized_fault_classifier.json"):
    """
    加载模型并恢复状态。
    """
    loaded_model = FaultClassifierChainOfThought()

    with open(model_path, 'r', encoding='utf-8') as f:
        state = json.load(f)

    loaded_model.load_state(state)
    print("模型已成功加载")
    return loaded_model


def predict(model, input_text):
    """
    对输入文本进行分类预测。
    """
    result = model(input=input_text)
    return result.label


def main():
    # 获取项目根目录
    project_root = os.path.dirname(os.getcwd())

    # 读取数据
    all_data = load_data(project_root)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = split_dataset(all_data)

    # 构建 DSPy 数据集
    train_set, val_set = build_dspy_datasets(X_train, y_train, X_val, y_val)

    # 配置语言模型
    model = configure_language_model_cot()

    # 训练优化模型
    optimized_model = train_model_bfswrs(model, train_set)

    # 评估优化模型
    evaluate_model(optimized_model, val_set)

    # 保存模型
    save_model(optimized_model)

    # 加载模型并进行推理测试
    loaded_model = load_model()
    sample_input = "设备无法启动，电源指示灯不亮。"
    prediction = predict(loaded_model, sample_input)
    print(f"输入: {sample_input} -> 预测结果: {prediction}")

    # 模型对比（优化前后）
    print("优化前模块评分：")
    evaluate_model(model, val_set)

    print("优化后模块评分：")
    evaluate_model(loaded_model, val_set)


# 使用ChainOfThought + BootstrapFewShotWithRandomSearch 优化模型
if __name__ == "__main__":
    main()
