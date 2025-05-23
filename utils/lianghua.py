import torch
import torch.nn as nn
import block
import torch.quantization


# 动态量化（适用于LSTM/Linear层）
def dynamic_quantize_model(model):
    # 对Linear层进行动态量化
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv1d, nn.LayerNorm},  # 指定要量化的模块类型
        dtype=torch.qint8
    )
    return quantized_model

#
# # 静态量化（需要校准数据）
# def static_quantize_model(model, calibration_data):
#     # 设置量化配置
#     model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
#
#     # 模型融合（需要根据模型结构调整）
#     def fuse_modules(model):
#         for name, module in model.named_children():
#             if isinstance(module, CBS_1D):
#                 # 融合Conv+BN+ReLU
#                 torch.quantization.fuse_modules(
#                     module,
#                     ['conv', 'bn', 'act'],
#                     inplace=True
#                 )
#             # 递归处理子模块
#             fuse_modules(module)
#
#     fuse_modules(model)
#
#     # 插入量化/反量化桩
#     model = torch.quantization.add_quant_dequant(model)
#
#     # 准备模型
#     prepared_model = torch.quantization.prepare(model)
#
#     # 校准（需要少量校准数据）
#     with torch.no_grad():
#         for data in calibration_data:
#             _ = prepared_model(data)
#
#     # 转换量化模型
#     quantized_model = torch.quantization.convert(prepared_model)
#     return quantized_model


# 使用示例
device = torch.device('cpu')

# 加载原始模型
model = block.DualBranchTimeSeriesPredictor(
    d_model=256,  # 根据实际配置修改
    n_heads=8,
    d_ff=512,
    num_layers=3,
    seq_len=40,
    pred_len=10
).to(device)
model.load_state_dict(torch.load('../model/DualBranchTimeSeriesPredictor_new_best.pth', map_location=device))
model.eval()

# 方式1：动态量化
quantized_model_dynamic = dynamic_quantize_model(model)

# 方式2：静态量化（需要提供校准数据）
# calibration_loader = ...  # 你的校准数据加载器
# quantized_model_static = static_quantize_model(model, calibration_loader)

# 保存量化模型
torch.save(quantized_model_dynamic, 'quantized_model_dynamic.pth')
# torch.save(quantized_model_static.state_dict(), 'quantized_model_static.pth')

# 推理测试样例
test_input = torch.randn(1, 40, 4)  # 示例输入
# 对比原始模型和量化模型输出
with torch.no_grad():
    original_output = model(test_input)
    quantized_output = quantized_model_dynamic(test_input)

# 计算输出差异
diff = (original_output[0] - quantized_output[0]).abs().mean()
print(f"输出平均差异: {diff.item():.6f}")  # 正常应小于1e-3

print(quantized_model_dynamic.state_dict().keys())