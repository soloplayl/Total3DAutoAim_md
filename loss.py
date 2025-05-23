import torch
import torch.nn as nn
import block
import torch.optim as optim
import torch.nn.functional as F
class TotalLoss(nn.Module):
    """
    定义一个封装了坐标、平移、旋转和半径损失的总体损失模块，
    可以直接使用 criterion = TotalLoss().to(device) 的方式加载到设备上。
    """

    def __init__(self, coord_loss_weight=1.0,
                 rot_loss_weight=10000.0,
                 theta_loss_weight=100.0):
        """
        :param coord_loss_weight: 坐标损失的权重
        :param rot_loss_weight: 旋转损失的权重
        :param radius_loss_weight: 半径损失的权重
        """
        super(TotalLoss, self).__init__()
        self.coord_loss_weight = coord_loss_weight

        self.rot_loss_weight = rot_loss_weight

        self.theta_loss_weight = theta_loss_weight

        # 这里使用均方误差作为示例损失函数
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        # criterion = nn.SmoothL1Loss()  # 平滑L1损失函数收敛有点慢
    def forward(self, pred_coords, gt_coords,
                pred_rot, gt_rot
             ):
        """
        计算总体损失，同时返回各部分的损失值。

        :param pred_coords: 模型预测的坐标，[batch, output_steps, 3]
        :param gt_coords:  ground truth 坐标，[batch, output_steps, 3]
        :param pred_trans: 模型预测的平移向量，[batch, output_steps, 3]
        :param gt_trans: ground truth 平移向量，[batch, output_steps, 3]
        :param pred_rot: 模型预测的旋转向量，[batch, output_steps, 3]
        :param gt_rot: ground truth 旋转向量，[batch, output_steps, 3]
        :param pred_radius: 模型预测的半径，[batch, output_steps, 1]
        :param gt_radius: ground truth 半径，[batch, output_steps, 1]
        :return: 一个 tuple, (总损失, 坐标损失, 平移损失, 旋转损失, 半径损失)
        """
        # 坐标损失：整体位置的均方误差
        loss_coord = self.mse_loss(pred_coords, gt_coords)


        # 如果需要，可以转换为旋转矩阵后计算 geodesic 距离
        loss_rot = self.mse_loss(pred_rot, gt_rot)
        # 计算旋转向量的角度损失
        loss_theta = self.mse_loss(torch.norm(pred_rot, dim=-1, keepdim=True), torch.norm(gt_rot, dim=-1, keepdim=True))


        total_loss = (self.coord_loss_weight * loss_coord +
                      self.rot_loss_weight * loss_rot +
                      self.theta_loss_weight * loss_theta
                      )

        return total_loss, loss_coord, loss_rot, loss_theta

# 使用示例
if __name__ == '__main__':
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 参数设置
    batch_size = 32
    seq_len = 40
    input_features = 4
    output_steps = 10

    # 构造模型并移动到 device
    model = block.DualBranchTimeSeriesPredictor(input_dim=4, seq_len=seq_len, pred_len=output_steps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 构造 dummy 输入并移动到 device
    dummy_input = torch.randn(batch_size, seq_len, input_features).to(device)

    # 假设 ground truth 坐标、平移、旋转向量和半径
    gt_coords = torch.randn(batch_size, output_steps, 3).to(device)
    gt_trans = torch.randn(batch_size, output_steps, 3).to(device)
    gt_rot = torch.randn(batch_size, output_steps, 3).to(device)
    gt_radius = torch.randn(batch_size, output_steps, 1).abs().to(device)

    # 前向传播
    pred_coords, pred_trans, pred_rot, pred_radius = model(dummy_input)

    criterion = TotalLoss(coord_loss_weight=1.0, trans_loss_weight=1.0,
                          rot_loss_weight=1.0, radius_loss_weight=1.0).to(device)

    # 计算损失
    total_loss, loss_coord, loss_trans, loss_rot, loss_radius = criterion(
        pred_coords, gt_coords, pred_trans, gt_trans, pred_rot, gt_rot, pred_radius, gt_radius)

    # 优化步骤
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 打印各部分损失
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Coordinate Loss: {loss_coord.item():.4f}")
    print(f"Translation Loss: {loss_trans.item():.4f}")
    print(f"Rotation Loss: {loss_rot.item():.4f}")
    print(f"Radius Loss: {loss_radius.item():.4f}")