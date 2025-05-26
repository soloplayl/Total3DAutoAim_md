import torch
import numpy as np
import matplotlib.pyplot as plt
import block
from vane_train import create_dataset  # 确保路径正确
from scipy.spatial.transform import Rotation as R
import os
np.set_printoptions(suppress=True)
# ==================== 1. 加载模型 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision = True # 是否使用可视化
sample = slice(0,1000) # 选择全部使用slice(None)
error_thredholds = [10,40,62.5,100,150,200] # 误差阈值
config = {
    'feature_dim': 4,  # 特征维度7是:t, x, y, z, rot(3 dims) 4是t, x, y, z
    'input_size': 20,
    'output_size': 10,
    'offset' : 36,
    'd_model': 256,
    'n_heads': 8,
    'd_ff': 256,
    'num_layers': 3,
    'model_type': 'vane_transformer', # vane_transformer/ DualBranchTimeSeriesPredictor
    'total_transformer_save_path': 'model/vane_model/total_Predictor.pth',
    'vane_transformer_save_path': 'model/vane_model/vane_Predictor.pth',
}


# 初始化模型
if config['model_type'] == 'DualBranchTimeSeriesPredictor':
    model = block.DualBranchTimeSeriesPredictor(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        input_dim=config['feature_dim'],
        seq_len=config['input_size'],
        pred_len=config['output_size'],
    ).to(device)
    save_path = config['total_transformer_save_path']
elif config['model_type'] == 'vane_transformer':
    model = block.TimeSeriesTransformer(
        input_size=config['input_size'],
        output_size=config['output_size'],
        num_layers=config['num_layers'],
        input_dim=config['feature_dim'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff']
    ).to(device)
    save_path = config['vane_transformer_save_path']
else:
    raise ValueError(f"Unsupported model type: {config['model_type']}")

# 加载权重
if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"成功加载预训练权重: {save_path}")
model.eval()

# ==================== 2. 读取数据并进行预测 ====================
# 加载数据
data = np.load("model/datasets/vane_dataset_yunsu.npz")
motion_type = data['motion_type']
direction_type = data['direction']
# print(motion_type)
# print(direction_type)
inputs, labels = create_dataset(data, input_size=config['input_size'], output_size=config['output_size'], offset=config['offset'])
print(inputs.shape)

# 选取测试数据
test_inputs = inputs[sample]  # 选择最近的 200 个样本进行可视化
test_labels = labels[sample]
# print(test_inputs)
# 转换为 PyTorch Tensor
# ==================== 新增：分批预测逻辑 ====================
batch_size = 2048  # 根据显存容量调整此数值

pred_coords, pred_trans, pred_rot, pred_radius, pred_class = [],[],[],[],[]
# 逐批处理数据
if  config['model_type'] == 'DualBranchTimeSeriesPredictor':
    for i in range(0, len(test_inputs), batch_size):
        # 获取当前批次数据
        batch_inputs = test_inputs[i:i + batch_size]

        # 转换为tensor并传输到设备
        batch_tensor = torch.tensor(batch_inputs, dtype=torch.float32).to(device)

        # 预测并收集结果
        with torch.no_grad():
            bpred_coords, bpred_trans, bpred_rot, bpred_radius,bpred_class = model(batch_tensor)
            bpred_coords = bpred_coords.cpu().numpy()
            bpred_trans = bpred_trans.cpu().numpy()
            bpred_rot = bpred_rot.cpu().numpy()
            bpred_radius = bpred_radius.cpu().numpy()
            bpred_class = bpred_class.cpu().numpy()
            pred_coords.append(bpred_coords)
            pred_trans.append(bpred_trans)
            pred_rot.append(bpred_rot)
            pred_radius.append(bpred_radius)
            pred_class.append(bpred_class)
elif config['model_type'] == 'vane_transformer':
    for i in range(0, len(test_inputs), batch_size):
        # 获取当前批次数据
        batch_inputs = test_inputs[i:i + batch_size]

        # 转换为tensor并传输到设备
        batch_tensor = torch.tensor(batch_inputs, dtype=torch.float32).to(device)

        # 预测并收集结果
        with torch.no_grad():
            bpred_coords = model(batch_tensor)
            bpred_coords = bpred_coords.cpu().numpy()
            pred_coords.append(bpred_coords)

# print('b:',np.concatenate(pred_coords, axis=0).shape)

# ==================== 修改后的MSE计算部分 ====================
pred_coords_all = np.concatenate(pred_coords, axis=0)  # [batch_size, output_steps, 3]
real_coords_all = test_labels[..., :3]  # [batch_size, output_steps, 3]
# 计算所有时间步和所有样本的MSE
MSE = np.mean((real_coords_all - pred_coords_all) ** 2)
print('All steps MSE:', MSE)

# 若需单独分析各时间步：
for step in range(pred_coords_all.shape[1]):
    mse_step = np.mean((real_coords_all[:, step, :] - pred_coords_all[:, step, :]) ** 2)
    print(f'Step {step} MSE: {mse_step:.6f}')



pred_step = -1 # 取最后一个预测步（索引 -1）即为预测0.5s=dt*50 dt与你生成数据有关，即你目标检测的速度dt=0.01s
pred_coords = np.concatenate(pred_coords, axis=0)[:, pred_step]

# print(pred_coords)
real = test_labels[:, pred_step,:]  # 真实值的最后一个时间步
# print("real.shape:",real.shape)
real_coords = real[:,:3]




# 初始化累积计数数组
cumulative_counts = np.zeros(len(error_thredholds))

# 统计每个阈值下的累积样本数
for i in range(len(real_coords)):
    error = np.sqrt((real_coords[i,0] - pred_coords[i,0]) ** 2
             +(real_coords[i,1] - pred_coords[i,1]) ** 2
             +(real_coords[i,2] - pred_coords[i,2]) ** 2)*1000
    for j in range(len(error_thredholds)):
        if error < error_thredholds[j]:
            cumulative_counts[j] += 1

interval_counts = np.zeros_like(cumulative_counts)
interval_counts[0] = cumulative_counts[0]
for j in range(1, len(error_thredholds)):
    interval_counts[j] = cumulative_counts[j] - cumulative_counts[j-1]

# 计算百分比
acc = interval_counts / len(real_coords)

# 最后一个区间（>200mm）的百分比
error_gt_200 = (len(real_coords) - cumulative_counts[-1]) / len(real_coords)
print(f"error < 10mm:{acc[0]*100:.2f}%")
print(f"10mm < error < 40mm:{acc[1]*100:.2f}%")
print(f"40mm < error < 62.5mm:{acc[2]*100:.2f}%")
print(f"62.5mm < error < 100mm:{acc[3]*100:.2f}%")
print(f"100mm < error < 150mm:{acc[4]*100:.2f}%")
print(f"150mm < error < 200mm:{acc[5]*100:.2f}%")
print(f"error > 200mm:{error_gt_200*100:.2f}%")

errors = np.sqrt((real_coords[:,0] - pred_coords[:,0]) ** 2
             +(real_coords[:,1] - pred_coords[:,1]) ** 2
             +(real_coords[:,2] - pred_coords[:,2]) ** 2)
print("avg_error:", np.mean(errors)*1000, "mm")

# ==================== 4. Matplotlib 动态绘图 ====================
if vision:
    # 创建图形和坐标轴
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('圆周运动轨迹动态演示')
    ax.grid(True)
    # ax.axis('equal')

    # ax.set_xlim(0.0, 10000.0)  # x 的范围
    # ax.set_ylim(-5000.0, 5000.0)  # y 的范围
    # ax.set_zlim(-5000.0, 5000.0)  # z 的绝对值范围


    # 控制绘制速度
    for i in range(len(real_coords)):
        if i % 10== 0:
            ax.scatter(pred_coords[i,0], pred_coords[i,1], pred_coords[i,2], color='red',marker='x', s=100, zorder=5, label='预测点')
            ax.scatter(real_coords[i,0], real_coords[i,1], real_coords[i,2], color='blue', s=50, zorder=5, label='真实点')
            error = np.sqrt((real_coords[i, 0] - pred_coords[i, 0]) ** 2
                            + (real_coords[i, 1] - pred_coords[i, 1]) ** 2
                            + (real_coords[i, 2] - pred_coords[i, 2]) ** 2)
            ax.text(0.05, 0.95, 0,
                    f'误差: {error*1000:.2f} mm',
                    transform=ax.transAxes, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8))
            plt.pause(0.001)  # 调整这个值控制播放速度，0.001对应约1000帧/秒
    plt.show()
