from src.fault_classifier_pipeline import *

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.getcwd()))

# 读取数据
all_data = load_data(project_root)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = split_dataset(all_data)

# 构建 DSPy 数据集
train_set, val_set = build_dspy_datasets(X_train, y_train, X_val, y_val)

# 配置语言模型
model = configure_language_model_predict()

# 加载模型并进行推理测试
loaded_model = load_model_predict()
sample_input = "设备无法启动，电源指示灯不亮。"
prediction = predict(loaded_model, sample_input)
print(f"输入: {sample_input} -> 预测结果: {prediction}")

# 模型对比（优化前后）
print("优化前模块评分：")
evaluate_model(model, val_set)

print("优化后模块评分：")
evaluate_model(loaded_model, val_set)
