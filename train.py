import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import block
import os
import time
import loss
import pandas as pd
import argparse


# ==================== 1. 数据预处理 ====================

class WindowGenerator:
    def __init__(self, input_width, label_width, input_dim=3, offsite=40):
        self.input_width = input_width  # 输入窗口大小 (n)
        self.label_width = label_width  # 预测窗口大小 (n)
        self.offsite = offsite  # 预测窗口偏置
        self.input_dim = input_dim  # 输入特征维度

    def split_window(self, features):
        inputs = features[:, :self.input_width, :self.input_dim]
        if self.input_dim == 3 or self.input_dim == 6:
            labels = features[:, self.input_width + self.offsite: self.input_width + self.offsite + self.label_width,
                     :3]  # 只取xyz作为label
        else:
            labels = features[:, self.input_width + self.offsite: self.input_width + self.offsite + self.label_width,
                     1:4]  # 只取xyz作为label
        return inputs, labels


def create_dataset(data, input_dim=3, input_size=10, output_size=10, offset=40, type='vane_generate', unit='mm'):
    """创建时间窗口数据集"""
    generator = WindowGenerator(
        input_width=input_size,
        label_width=output_size,
        offsite=offset
    )
    # print("data[...,0]:",data[...,0].shape)
    # 生成差分特征
    print("\033[93m默认:t单位s，xyz单位mm，rot单位rad\033[0m")
    print("\033[93m如果输入是m请修改unit参数为'm'，如果输入是mm请忽略...\033[0m")
    if type == 'vane_generate':
        x_diff = (np.cos(data['theta']) * np.sin(data['phi']))[:, :, np.newaxis]
        y_diff = (np.cos(data['theta']) * np.cos(data['phi']))[:, :, np.newaxis]
        z_diff = (np.sin(data['theta']))[:, :, np.newaxis]
        # 使用 np.concatenate 按最后一个维度拼接
        features = np.concatenate([x_diff * 1000, y_diff * 1000, z_diff * 1000], axis=-1)
    elif type == 'txt':
        if input_dim == 3:
            x_diff = (data[..., 0])[..., np.newaxis]
            y_diff = (data[..., 1])[..., np.newaxis]
            z_diff = (data[..., 2])[..., np.newaxis]
            # 使用 np.concatenate 按最后一个维度拼接
            if unit == 'm':
                features = np.concatenate([x_diff * 1000, y_diff * 1000, z_diff * 1000], axis=-1)
            elif unit == 'mm':
                features = np.concatenate([x_diff, y_diff, z_diff], axis=-1)
            else:
                raise ValueError(f"Unsupported unit: {unit}. Use 'm' or 'mm'.")
        elif input_dim == 4:
            t_diff = (data[..., 0])[..., np.newaxis]
            x_diff = (data[..., 1])[..., np.newaxis]
            y_diff = (data[..., 2])[..., np.newaxis]
            z_diff = (data[..., 3])[..., np.newaxis]
            # 使用 np.concatenate 按最后一个维度拼接
            if unit == 'm':
                features = np.concatenate([t_diff, x_diff * 1000, y_diff * 1000, z_diff * 1000], axis=-1)
            elif unit == 'mm':
                features = np.concatenate([t_diff, x_diff, y_diff, z_diff], axis=-1)
            else:
                raise ValueError(f"Unsupported unit: {unit}. Use 'm' or 'mm'.")
        elif input_dim == 6:
            x_diff = (data[..., 1])[..., np.newaxis] - (data[..., 4])[..., np.newaxis]
            y_diff = (data[..., 2])[..., np.newaxis] - (data[..., 5])[..., np.newaxis]
            z_diff = (data[..., 3])[..., np.newaxis] - (data[..., 6])[..., np.newaxis]
            if unit == 'm':
                features = np.concatenate([x_diff * 1000, y_diff * 1000, z_diff * 1000], axis=-1)
            elif unit == 'mm':
                features = np.concatenate([x_diff, y_diff, z_diff], axis=-1)
            else:
                raise ValueError(f"Unsupported unit: {unit}. Use 'm' or 'mm'.")
        elif input_dim == 7:
            t_diff = (data[..., 0])[..., np.newaxis]
            x_diff = (data[..., 1])[..., np.newaxis]
            y_diff = (data[..., 2])[..., np.newaxis]
            z_diff = (data[..., 3])[..., np.newaxis]
            rot = (data[..., 4:7])[..., np.newaxis]
            if unit == 'm':
                features = np.concatenate([t_diff, x_diff * 1000, y_diff * 1000, z_diff * 1000, rot], axis=-1)
            elif unit == 'mm':
                features = np.concatenate([t_diff, x_diff, y_diff, z_diff, rot], axis=-1)
            else:
                raise ValueError(f"Unsupported unit: {unit}. Use 'm' or 'mm'.")
        else:
            raise ValueError(f"Unsupported input dimension: {input_dim}")
    elif type == 'armor_generate':
        x_diff = (data['x'])[:, :, np.newaxis]
        y_diff = (data['y'])[:, :, np.newaxis]
        z_diff = (data['z'])[:, :, np.newaxis]
        features = np.concatenate([x_diff, y_diff, z_diff], axis=-1)
    else:
        raise ValueError(f"Unsupported data type: {type}")
    print("x_diff:", x_diff.shape)

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
    # 关键修改：对每个窗口减去其输入部分的起始点
    if input_dim == 3 or input_dim == 4 or input_dim==6:
        base_points = inputs[:, 0, :]  # 取每个输入窗口的第一个点 [B, 3]
        base_points = base_points[:, np.newaxis, :]  # 扩展维度 [B, 1, 3]

        # 输入/标签统一减去基准点
        inputs = inputs - base_points
        labels = labels - base_points
    elif input_dim == 7:
        base_points = inputs[:, 0, :4]
        base_points = base_points[:, np.newaxis, :]
        # 输入/标签统一减去基准点
        inputs[..., :4] = inputs[..., :4] - base_points
        output_size[...:4] = labels[..., :4] - base_points
    else:
        raise ValueError(f"Unsupported input dimension: {input_dim}")

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
                scheduler=None, device='cpu'):
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
            if config['model_type'] == 'DualBranchTimeSeriesPredictor':
                pred_coords, pred_trans, pred_rot, pred_radius, pred_class = model(inputs)
            elif config['model_type'] == 'vane_transformer':
                pred_coords = model(inputs)

            # 从 labels 中提取各部分 ground truth，
            # 假设 labels 的 shape 为 [batch, output_steps, 9]
            # 分别为： [x, y, z, rot(3 dims)]

            gt_coords = labels[..., :3]  # 第0~2通道：坐标
            # 计算总损失及各项损失
            total_loss, loss_coord = criterion(
                pred_coords, gt_coords,
            )

            total_loss.backward()
            optimizer.step()
            epoch_train_loss += total_loss.item()
            epoch_train_coord_loss += loss_coord.item()

        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        avg_train_coord_loss = epoch_train_coord_loss / len(train_loader)

        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_coord_loss = 0.0
        epoch_val_rot_loss = 0.0
        epoch_val_RotAngle_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                if config['model_type'] == 'DualBranchTimeSeriesPredictor':
                    pred_coords, pred_trans, pred_rot, pred_radius, pred_class = model(inputs)
                elif config['model_type'] == 'vane_transformer':
                    pred_coords = model(inputs)

                # 同样从 labels 中提取各分支目标
                gt_coords = labels[..., :3]

                val_loss, _loss_coord = criterion(
                    pred_coords, gt_coords,
                )
                epoch_val_loss += val_loss.item()
                epoch_val_coord_loss += _loss_coord.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_coord_loss = epoch_val_coord_loss / len(val_loader)
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
              f"Total: {avg_train_loss:.4f}\n"

              f"Val   => Coord: {avg_val_coord_loss:.4f} |"
              f"Total: {avg_val_loss:.4f}\n"
              )

        # ========== 新增代码 ==========
        # 在每个epoch结束后更新学习率
        if scheduler is not None:
            scheduler.step()

        # =============================
        torch.save(model.state_dict(), 'model/vane_Predictor_last.pth')
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
    'data_path': 'model/datasets/processed_log.txt',
    'feature_dim': 3,  # 特征维度7是:t, x, y, z, rot(3 dims) 4是t, x, y, z 3是x, y, z
    'input_size': 20,  # 输入窗口大小 dt=0.0125 t = 20*0.0125=0.25s
    'output_size': 10,  # 输出窗口大小 pred_len=10 offset=36 dt=0.0125 pred_t=(36+10)*0.0125=0.45s+0.125s=0.575s
    'offset': 36,  # 预测窗口偏置
    'batch_size': 2048,
    'test_ratio': 0.2,
    'epochs': 1000,
    'lr': 1e-3,
    'patience': 30,
    'model_type': 'DualBranchTimeSeriesPredictor',  # vane_transformer/DualBranchTimeSeriesPredictor
    'total_transformer_save_path': 'model/vane_model/total_Predictor.pth',
    'vane_transformer_save_path': 'model/vane_model/vane_Predictor.pth',
    'd_model': 256,
    'n_heads': 8,
    'd_ff': 512,  # 前馈网络维度这个ffn层设为512的的话建议训练所有状态学习，单学习能量机关256即可，如果预测自瞄建议512
    'num_layers': 3,
    'eta_min': 1e-7,  # 余旋退火参数
    'data_mode': 'txt',  # vane/armor/txt 对应vane_generate/armor_generate/自己的txt文件训练集
    'unit': 'mm',  # 数据单位 mm/m
    'vision': False,  # preict显示图像
    'sample': None
}
# --- 冻结配置字典（0表示冻结，1表示可训练）---
freeze_config = {
    "encoders": 1,  # 所有Encoder层
    "class_fc": 0,  # 分类分支
    "pre_coords": 1  # 位置分支
}


def train_init():
    print(config['patience'])
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if config['data_mode'] == 'txt':
        data = pd.read_csv(config['data_path'], sep='\s+', header=None)
        print("原始数据形状:", data.shape[0])
        numpy_data = data.to_numpy()  # 转换为NumPy数组
        # 重塑形状
        data = numpy_data.reshape(1, data.shape[0], data.shape[1])
        print("data.shape:", data.shape)
        inputs, labels = create_dataset(data, input_dim=config['feature_dim'], input_size=config['input_size'],
                                        output_size=config['output_size'],
                                        offset=config['offset'], type='txt', unit=config['unit'])
    elif config['data_mode'] == 'vane':
        data = np.load(config['data_path'])
        print('data length:', len(data['theta']))
        inputs, labels = create_dataset(data, input_dim=config['feature_dim'], input_size=config['input_size'],
                                        output_size=config['output_size'],
                                        offset=config['offset'], type='vane_generate', unit='m')
    elif config['data_mode'] == 'armor':
        data = np.load(config['data_path'])
        print('data length:', len(data['theta']))
        inputs, labels = create_dataset(data, input_dim=config['feature_dim'], input_size=config['input_size'],
                                        output_size=config['output_size'],
                                        offset=config['offset'], type='armor_generate', unit='mm')
    else:
        raise ValueError(f"Unsupported data mode: {config['data_mode']}")

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

    print('-----------start save model-----------')

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"成功加载预训练权重: {save_path}")

    # criterions_weight
    criterion = loss.TotalLoss(coord_loss_weight=1.0,
                               rot_loss_weight=0.0,
                               theta_loss_weight=0.0
                               ).to(device)

    if config['model_type'] == 'DualBranchTimeSeriesPredictor':
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
        eta_min=config['eta_min']
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
        patience=30,  # 连续10次验证损失无改进则早停
        device=device,
    )


# ==================== 6. 执行流程 ====================
if __name__ == "__main__":
    train_init()
