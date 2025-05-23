import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import block
import os
import time
import loss

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 1. 数据预处理 ====================

class WindowGenerator:
    def __init__(self, input_width, label_width, offsite=40):
        self.input_width = input_width  # 输入窗口大小 (n)
        self.label_width = label_width  # 预测窗口大小 (n)
        self.offsite = offsite  # 预测窗口偏置

    def split_window(self, features):
        inputs = features[:, :self.input_width, :7]  # 只取txyzrot作为输入
        labels = features[:, self.input_width + self.offsite: self.input_width + self.offsite + self.label_width,
                 1:]  # 只取xyz作为label
        return inputs, labels


def create_dataset(data, input_size=10, output_size=10, offset=40):
    """创建时间窗口数据集"""
    generator = WindowGenerator(
        input_width=input_size,
        label_width=output_size,
        offsite=offset
    )

    # 生成差分特征
    t_diff = (data['t'] - data['t'][:, 0][:, np.newaxis])[..., np.newaxis]  # (4000, 200, 1)
    x_diff = (data['x'] - data['x'][:, 0][:, np.newaxis])[..., np.newaxis]  # (4000, 200, 1)
    y_diff = (data['y'] - data['y'][:, 0][:, np.newaxis])[..., np.newaxis]  # (4000, 200, 1)
    z_diff = (data['z'] - data['z'][:, 0][:, np.newaxis])[..., np.newaxis]  # (4000, 200, 1)

    # data['rot'] 已是 (4000, 200, 3)
    rot_data = data['rot']
    """    
    # 如果有车体中心平移数据的情况，此处计算结果为 (4000, 200, 3)
    # trans_data = data['trans'] - np.stack([data['z'][:, 0], data['x'][:, 0], data['y'][:, 0]], axis=-1)[:, np.newaxis,
    #                              :]
    # data['radius'] 原始形状 (4000,)，扩展为 (4000, 1, 1)，再沿时间步复制到 (4000, 200, 1)
    radius_data = data['radius'][:, None, None]  # (4000, 1, 1)
    radius_data = np.tile(radius_data, (1, data['z'].shape[1], 1))  # (4000, 200, 1)
    # features = np.concatenate([t_diff, x_diff, y_diff, z_diff, rot_data, trans_data], axis=-1)
    """

    # 使用 np.concatenate (各数组形状均为 (4000, 200, C)) 按最后一个维度拼接
    features = np.concatenate([t_diff, x_diff, y_diff, z_diff, rot_data], axis=-1)

    num_samples, num_points, _ = features.shape  # 获取处理后的实际特征维度
    window_length = input_size + offset + output_size  # 例如90
    num_windows = num_points - window_length + 1
    # 生成滑动窗口
    sequences = []
    for sample in features:
        for i in range(num_windows):
            sequences.append(sample[i:i + window_length])

    dataset = np.array(sequences)
    inputs, labels = generator.split_window(dataset)
    return inputs, labels


# ==================== 2. 数据集类 ====================
class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


# ==================== 3. 训练流程 ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path, patience=10,
                scheduler=None):
    model.train()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    # train_real_losses = []
    val_losses = []
    total_start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # 单个epoch开始时间
        model.train()
        epoch_train_loss = 0.0
        epoch_train_coord_loss = 0.0
        epoch_train_rot_loss = 0.0
        epoch_train_RotAngle_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # 模型输出4个分支：
            # pred_coords: [batch, output_steps, 3]
            # pred_trans: [batch, output_steps, 3]
            # pred_rot: [batch, output_steps, 3]
            # pred_radius: [batch, output_steps, 1]
            # pred_class: [batch, output_steps, 1]
            pred_coords, pred_trans, pred_rot, pred_radius, pred_class = model(inputs)

            # 从 labels 中提取各部分 ground truth，
            # 假设 labels 的 shape 为 [batch, output_steps, 9]
            # 分别为： [x, y, z, rot(3 dims)] //没用trans
            gt_coords = labels[..., :3]  # 第0~2通道：坐标
            gt_rot = labels[..., 3:6]  # 第3~5通道：旋转向量
            # 计算总损失及各项损失
            total_loss, loss_coord, loss_rot, loss_theta = criterion(
                pred_coords, gt_coords,
                pred_rot, gt_rot,
            )

            total_loss.backward()
            optimizer.step()
            epoch_train_loss += total_loss.item()
            epoch_train_coord_loss += loss_coord.item()
            epoch_train_rot_loss += loss_rot.item()
            epoch_train_RotAngle_loss += loss_theta.item()
        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        avg_train_coord_loss = epoch_train_coord_loss / len(train_loader)
        avg_train_rot_loss = epoch_train_rot_loss / len(train_loader)
        avg_train_RotAngle_loss = epoch_train_RotAngle_loss / len(train_loader)

        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_coord_loss = 0.0
        epoch_val_rot_loss = 0.0
        epoch_val_RotAngle_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                pred_coords, pred_trans, pred_rot, pred_radius, pred_class = model(inputs)

                # 同样从 labels 中提取各分支目标
                gt_coords = labels[..., :3]
                gt_rot = labels[..., 3:6]

                val_loss, _loss_coord, _loss_rot, _loss_theta = criterion(
                    pred_coords, gt_coords,
                    pred_rot, gt_rot
                )
                epoch_val_loss += val_loss.item()
                epoch_val_coord_loss += _loss_coord.item()
                epoch_val_rot_loss += _loss_rot.item()
                epoch_val_RotAngle_loss += _loss_theta.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_coord_loss = epoch_val_coord_loss / len(val_loader)
        avg_val_rot_loss = epoch_val_rot_loss / len(val_loader)
        avg_val_RotAngle_loss = epoch_val_RotAngle_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # ========== 计时和日志 ==========
        epoch_time = time.time() - epoch_start_time  # 当前epoch耗时
        remaining_time = (num_epochs - epoch - 1) * epoch_time  # 剩余预估时间
        # 打印训练和验证损失
        # print(f"Patience {epochs_no_improve}, Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_real_loss:.4f}, Val Loss: {avg_val_real_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Patience [{epochs_no_improve}] | "
              f"Time: {epoch_time:.2f}s | ETA: {remaining_time / 60:.1f}min | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}\n"
              f"Train => Coord: {avg_train_coord_loss:.4f} | "
              f"Rot: {avg_train_rot_loss:.4f} | "
              f"RotAngle: {avg_train_RotAngle_loss:.4f} | "
              f"Total: {avg_train_loss:.4f}\n"
              
              f"Val   => Coord: {avg_val_coord_loss:.4f} |"
              f"Rot: {avg_val_rot_loss:.4f} | "
              f"RotAngle: {avg_val_RotAngle_loss:.4f} | "
              f"Total: {avg_val_loss:.4f}\n"
              )

        # ========== 新增代码 ==========
        # 在每个epoch结束后更新学习率
        if scheduler is not None:
            scheduler.step()

        # =============================
        torch.save(model.state_dict(), 'model/DualBranchTimeSeriesPredictor_last.pth')
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            print("saving best model...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}!")
                break
    # ========== 最终统计 ==========
    total_time = time.time() - total_start_time
    print(f"\nTraining completed in {total_time / 60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Training and Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()

    return model


# ==================== 5. 配置参数 ====================
config = {
    'data_path': 'model/datasets/time_armor_series_Rot.npz',
    'input_size': 10,  # 输入窗口大小 dt=0.015 t = 20*0.015=0.3s
    'output_size': 12,  # 输出窗口大小 pred_len=10 offset=30 dt=0.015 pred_t=30*0.015=0.45s
    'offset': 8,  # 预测窗口偏置
    'batch_size': 2048,
    'test_ratio': 0.2,
    'epochs': 100,
    'lr': 5e-4,
    'model_type': 'DualBranchTimeSeriesPredictor',
    'new_transformer_save_path': 'model/DualBranchTimeSeriesPredictor_infantry.pth',
    'd_model': 256,
    'n_heads': 8,
    'd_ff': 512,
    'num_layers': 3,
    'eta_min' : 1e-4 # 余旋退火参数
}
# --- 冻结配置字典（0表示冻结，1表示可训练）---
freeze_config = {
    "encoders": 1,  # 所有Encoder层
    "class_fc": 0,  # 分类分支
    "pre_coords": 1  # 位置分支
}
# ==================== 6. 执行流程 ====================
if __name__ == "__main__":
    data = np.load(config['data_path'])
    print('data length:', len(data['x']))
    inputs, labels = create_dataset(data, input_size=config['input_size'], output_size=config['output_size'],
                                    offset=config['offset'])
    print("inputs shape:", inputs.shape)
    print("labels shape:", labels.shape)
    # 训练集和测试集划分
    train_size = int((1 - config['test_ratio']) * inputs.shape[0])
    train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
    train_labels, test_labels = labels[:train_size], labels[train_size:]

    train_dataset = TimeSeriesDataset(train_inputs, train_labels)
    test_dataset = TimeSeriesDataset(test_inputs, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # n_bins = 32
    # buffer = 0.05  # 边界扩展比例
    # bins = torch.linspace(-1 - buffer, 1 + buffer, n_bins)

    # 在模型初始化时动态选择
    if config['model_type'] == 'DualBranchTimeSeriesPredictor':
        model = block.DualBranchTimeSeriesPredictor(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            num_layers=config['num_layers'],
            input_dim=7,
            seq_len=config['input_size'],
            pred_len=config['output_size'],
        ).to(device)
    print('-----------start save model-----------')

    save_path = config['new_transformer_save_path']
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"成功加载预训练权重: {save_path}")


    # criterion = loss.TotalLoss(coord_loss_weight=1.0, trans_loss_weight=1.0,
    #                            rot_loss_weight=10000.0, radius_loss_weight=10.0,
    #                            v_loss_weight=100.0,theta_loss_weight=100.0,
    #                            loss_class_weight=1000.0,theta0_loss_weight=100.0).to(device)  # 使用自定义的空间损失函数[1.0,1.0,10000,1.0]

    # criterions_weight
    criterion = loss.TotalLoss(coord_loss_weight=1.0,
                               rot_loss_weight=10000.0,
                               theta_loss_weight=5000.0
                               ).to(device)



    # --- 根据配置冻结参数 ---
    for module_name, flag in freeze_config.items():
        if hasattr(model, module_name):  # 检查模块是否存在
            module = getattr(model, module_name)
            for param in module.parameters():
                param.requires_grad = (flag == 1)
        else:
            print(f"Warning: Module {module_name} not found in model, skipped freezing.")
    # optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    #########################################
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)  # 建议改用AdamW
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-4
    )
    #########################################
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['epochs'],
        save_path=save_path,
        patience=50,  # 连续10次验证损失无改进则早停
    )
