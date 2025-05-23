import matplotlib.pyplot as plt
import block
from preprocessing import create_dataset  # 确保路径正确
from scipy.spatial.transform import Rotation as R
import os
import time
import numpy as np
import torch

np.set_printoptions(suppress=True)
# ==================== 1. 加载模型 ====================
device = 'cpu'
vision = True # 是否使用可视化
sample = slice(1000,1001) # 选择全部使用slice(None)
error_thredholds = [10,40,62.5,100,150,200] # 误差阈值
config = {
    'input_size': 40,
    'output_size': 10,
    'd_model': 256,
    'n_heads': 8,
    'd_ff': 512,
    'num_layers': 3,
    'encoders_save_path': 'model/Encoders_best.pth',
    'class_fc_save_path': 'model/class_fc_best.pth',
    'pre_coords_save_path': 'model/pre_coords_class0.pth'
}

# 初始化模型
encoder = block.Encoders(
    d_model=config['d_model'],
    n_heads=config['n_heads'],
    d_ff=config['d_ff'],
    num_layers=config['num_layers'],
).to(device)
class_fc = block.class_fc(
    seq_len=config['input_size'],
    pred_len=config['output_size'],
    d_model=config['d_model'],
).to(device)
pre_coords = block.pre_coords(
    seq_len=config['input_size'],
    pred_len=config['output_size'],
    d_model=config['d_model'],
).to(device)

# 加载权重
save_path = config['encoders_save_path']
if os.path.exists(save_path):
    encoder.load_state_dict(torch.load(save_path, map_location=device))
    print(f"成功加载预训练权重: {save_path}")
encoder.eval()
save_path = config['class_fc_save_path']
if os.path.exists(save_path):
    class_fc.load_state_dict(torch.load(save_path, map_location=device))
    print(f"成功加载预训练权重: {save_path}")
class_fc.eval()
save_path = config['pre_coords_save_path']
if os.path.exists(save_path):
    pre_coords.load_state_dict(torch.load(save_path, map_location=device))
    print(f"成功加载预训练权重: {save_path}")
pre_coords.eval()

# ==================== 2. 读取数据并进行预测 ====================
# 加载数据
data = np.load("model/time_armor_series_dataset2.npz")
motion_type = data['motion_type']
direction_type = data['direction']
# print(motion_type)
# print(direction_type)
inputs, labels = create_dataset(data, input_size=config['input_size'], output_size=config['output_size'])

# 选取测试数据
test_inputs = inputs[sample]  # 选择最近的 200 个样本进行可视化
test_labels = labels[sample]
test_inputs = torch.tensor(test_inputs, dtype=torch.float32).to(device)
intermediate = encoder(test_inputs)
# print(intermediate.shape)

classes = class_fc(intermediate)

# 增加预热和循环测试逻辑
warmup_cycles = 100  # 预热循环次数
test_cycles = 1000  # 正式测试循环次数

# 预热阶段 (避免首次运行受冷启动影响)
with torch.no_grad():
    for _ in range(warmup_cycles):
        classes_squeezed = classes.squeeze(-1)
        sorted_classes = torch.sort(classes_squeezed, dim=1)[0]
        trimmed = sorted_classes[:, 1:-1]
        average = trimmed.mean(dim=1)

# 正式测试
total_time = 0
with torch.no_grad():
    for _ in range(test_cycles):
        start_time = time.perf_counter()  # 更高精度计时

        classes_squeezed = classes.squeeze(-1)
        sorted_classes = torch.sort(classes_squeezed, dim=1)[0]
        trimmed = sorted_classes[:, 1:-1]
        average = trimmed.mean(dim=1)

        total_time += time.perf_counter() - start_time

# 输出结果
avg_time_ms = (total_time / test_cycles) * 1000
print(f"测试次数: {test_cycles}次")
print(f"总耗时: {total_time * 1000:.3f}ms")
print(f"平均耗时: {avg_time_ms:.3f}ms")
print(f"标准差: ±{np.std([total_time / test_cycles] * test_cycles) * 1000:.3f}ms")