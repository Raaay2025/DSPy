import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

# 全局变量，用于缓存 BERT 模型和分词器，避免重复加载
_bert_tokenizer = None
_bert_model = None

# 判断当前是否有 GPU 可用（cuda），否则使用 CPU
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_bert_model():
    """加载 BERT 模型和分词器"""
    global _bert_tokenizer, _bert_model
    if _bert_tokenizer is None or _bert_model is None:
        _bert_tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese/')
        _bert_model = BertModel.from_pretrained('../bert-base-chinese')
        _bert_model.to(_device)  # 将模型移动到 GPU 或 CPU
        _bert_model.eval()  # 设置为评估模式，不进行梯度更新
    return _bert_tokenizer, _bert_model


def get_bert_similarity(text1, text2):
    """
    使用 BERT 模型计算两个文本之间的语义相似度
    :param text1: 文本1
    :param text2: 文本2
    :return: 余弦相似度（0~1）
    """
    if text1 == "无" and text2 == "无":
        return 1.0  # 如果两个字段都为“无”，视为匹配

    tokenizer, model = get_bert_model()

    # 对文本进行编码
    inputs1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True, max_length=128).to(_device)
    inputs2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True, max_length=128).to(_device)

    # 推理阶段不计算梯度
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # 提取 [CLS] 向量作为句子表示
    vec1 = outputs1.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    vec2 = outputs2.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    # 计算余弦相似度
    similarity = cosine_similarity([vec1], [vec2])[0][0]

    # 将余弦相似度从 [-1, 1] 映射到 [0, 1]
    return (similarity + 1) / 2


def bert_similarity_metric(gold, pred, trace=None):
    """
    计算 gold 和 pred 之间的平均语义相似度
    :param gold: 标注样本（Example 对象）
    :param pred: 模型预测（Example 对象）
    :param trace: 可选参数，用于调试或跟踪
    :return: 平均相似度（0~1）
    """
    # 获取所有需要比较的字段名（排除 input 字段）
    fields = [k for k in gold.toDict().keys() if k != 'input']

    total_similarity = 0.0
    valid_fields = 0

    for field in fields:
        gold_val = getattr(gold, field, "无")
        pred_val = getattr(pred, field, "无")

        # 如果 gold_val 不存在，跳过该字段
        if gold_val == "无" and pred_val == "无":
            similarity = 1.0  # 两个字段都为“无”，视为匹配
        elif gold_val == "无" or pred_val == "无":
            similarity = 0.0  # 一个为“无”，一个不为“无”，视为不匹配
        else:
            similarity = get_bert_similarity(gold_val, pred_val)

        total_similarity += similarity
        valid_fields += 1

    # 返回平均相似度
    return total_similarity / valid_fields if valid_fields else 0.0
