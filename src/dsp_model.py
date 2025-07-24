import dspy


class ClassificationSignature(dspy.Signature):
    """中文故障分类任务的输入输出定义"""
    input = dspy.InputField(desc="输入文本")
    label = dspy.OutputField(desc="分类标签，故障或非故障")


class FaultClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ClassificationSignature)

    def forward(self, input):
        return self.predict(input=input)


class FaultExtractionSignature(dspy.Signature):
    """从故障类通信文本中提取20个结构化字段"""
    input = dspy.InputField(desc="故障类通信文本")

    # 20 个输出字段
    fault_equipment = dspy.OutputField(desc="故障设备")
    fault_time = dspy.OutputField(desc="故障时间")
    region = dspy.OutputField(desc="调管范围")
    voltage_level = dspy.OutputField(desc="电压等级")
    weather_condition = dspy.OutputField(desc="天气情况")
    fault_reason_and_check_result = dspy.OutputField(desc="故障原因及检查结果")
    fault_recovery_time = dspy.OutputField(desc="故障恢复时间")
    illustrate = dspy.OutputField(desc="处置详情")
    line_name = dspy.OutputField(desc="线路名称")
    power_supply_time = dspy.OutputField(desc="送电时间")
    fault_phase = dspy.OutputField(desc="故障相别")
    protect_info = dspy.OutputField(desc="保护信息")
    plant_station_name = dspy.OutputField(desc="厂站名称")
    bus_name = dspy.OutputField(desc="母线名称")
    bus_type = dspy.OutputField(desc="母线类型")
    handling_status = dspy.OutputField(desc="处理情况")
    detailed_description = dspy.OutputField(desc="详细情况")
    expecteddefect_elimination_time = dspy.OutputField(desc="预计消缺时间")
    protection_action = dspy.OutputField(desc="继电保护动作情况")
    trip_details = dspy.OutputField(desc="故障详情")
    unit_num = dspy.OutputField(desc="设备编号")
    manufacturer = dspy.OutputField(desc="设备厂家")
    production_date = dspy.OutputField(desc="投产年月")


class FaultExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(FaultExtractionSignature)

    def forward(self, input):
        return self.predict(input=input)
