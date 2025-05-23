import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np

class CBS_1D(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, act=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.norm = nn.LayerNorm(out_features)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.linear(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim  # 词向量维度
        self.n_heads = n_heads
        self.d_head = dim // n_heads

        self.key_dim = int(self.dim * 0.5)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * n_heads
        h = dim + nh_kd * 2

        self.wq = Linear(dim, nh_kd, act=False)
        self.wk = Linear(dim, nh_kd, act=False)
        self.wv = Linear(dim, dim, act=False)
        self.pe = Linear(dim, dim, act=False)
        self.out = Linear(dim, dim, act=False)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projection
        q = self.wq(q).view(batch_size, -1, self.n_heads, self.key_dim).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.n_heads, self.key_dim).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.scale)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim) + self.pe(
            v.reshape(batch_size, -1, self.dim))
        return self.out(context)


class FeedForward(nn.Module):
    def __init__(self, dim, d_ff=2048):
        super().__init__()
        self.linear1 = Linear(dim, d_ff)
        self.linear2 = Linear(d_ff, dim, act=False)

    def forward(self, x):
        return self.linear2(self.linear1(x))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)

        self.ffn = FeedForward(d_model, d_ff=d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


def rotation_matrix_from_vector(rot_vec):
    """
    将 rot_vec（形状: [batch, output_steps, 3]）转换为旋转矩阵，
    利用罗德里格斯公式。
    """
    # 计算旋转角度（范数）
    angle = torch.norm(rot_vec, dim=-1, keepdim=True)  # [batch, output_steps, 1]
    # 防止除零，计算归一化旋转轴
    axis = rot_vec / (angle + 1e-6)  # [batch, output_steps, 3]

    # 计算 cos 和 sin（扩展一维方便广播）
    cos = torch.cos(angle).unsqueeze(-1)  # [batch, output_steps, 1, 1]
    sin = torch.sin(angle).unsqueeze(-1)  # [batch, output_steps, 1, 1]

    # 构造外积: axis_outer = axis * axis^T，形状 [batch, output_steps, 3, 3]
    axis_unsq = axis.unsqueeze(-1)  # [batch, output_steps, 3, 1]
    axis_transpose = axis_unsq.transpose(-2, -1)  # [batch, output_steps, 1, 3]
    outer = torch.matmul(axis_unsq, axis_transpose)  # [batch, output_steps, 3, 3]

    # 构造反对称矩阵 K，注意使用归一化后的 axis
    x = axis[..., 0].unsqueeze(-1)  # [batch, output_steps, 1]
    y = axis[..., 1].unsqueeze(-1)
    z = axis[..., 2].unsqueeze(-1)
    zeros = torch.zeros_like(x)
    K = torch.cat([
        torch.cat([zeros, -z, y], dim=-1).unsqueeze(-2),
        torch.cat([z, zeros, -x], dim=-1).unsqueeze(-2),
        torch.cat([-y, x, zeros], dim=-1).unsqueeze(-2)
    ], dim=-2)  # [batch, output_steps, 3, 3]

    # 单位矩阵 I
    I = torch.eye(3, device=rot_vec.device).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

    # 罗德里格斯公式：R = I*cos(angle) + (1 - cos(angle)) * outer + sin(angle) * K
    R = cos * I + (1 - cos) * outer + sin * K  # [batch, output_steps, 3, 3]
    return R


class LinearResBlock(nn.Module):
    def __init__(self, in_features):
        super(LinearResBlock, self).__init__()
        self.fc1_ = Linear(in_features, in_features * 2, act=True)
        self.fc2_ = Linear(in_features * 2, in_features, act=False)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x):
        res = x
        x = self.fc2_(self.fc1_(x))
        return self.act(self.norm(x + res))


class Encoders(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, cross_feature=False):
        super(Encoders, self).__init__()
        self.cross_feature = cross_feature
        self.fc1 = nn.Linear(7, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(num_layers)])

    def forward(self, x):
        # 输入的特征是车坐标系下的 [t, x, y, z, rot]
        if self.cross_feature:
            # 提取每个特征并保持三维结构
            x0 = x[:, :, [0]]  # 形状保持 [1024, 40, 1]
            x1 = x[:, :, [1]]
            x2 = x[:, :, [2]]
            x3 = x[:, :, [3]]

            # 计算交互特征并保持三维结构
            x0x1 = (x[:, :, 0] * x[:, :, 1]).unsqueeze(-1)  # 通过 unsqueeze 恢复第三维
            x0x2 = (x[:, :, 0] * x[:, :, 2]).unsqueeze(-1)
            x0x3 = (x[:, :, 0] * x[:, :, 3]).unsqueeze(-1)

            # 沿第三维度拼接 t,x,y,z,rot,tx,ty,tz
            x = torch.cat([x, x0x1, x0x2, x0x3], dim=-1)

        x = self.fc1(x)  # (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x)
        return x


class trans_fc(nn.Module):
    def __init__(self, seq_len, pred_len, d_model):
        super(trans_fc, self).__init__()
        self.trans_fc = nn.Sequential(
            CBS_1D(seq_len, pred_len, k=3, s=1, p=1),
            LinearResBlock(d_model),
            LinearResBlock(d_model),
            nn.Linear(d_model, 3)
        )

    def forward(self, x):
        return self.trans_fc(x)  # [batch, output_steps, 3]


# class rot_fc(nn.Module):
#     def __init__(self, seq_len, pred_len, d_model):
#         super(rot_fc, self).__init__()
#         self.rot_fc = nn.Sequential(
#             CBS_1D(seq_len, pred_len, k=3, s=1, p=1),
#             nn.Linear(d_model, d_model),
#             nn.SiLU(),
#             nn.Linear(d_model, 3)
#         )
#
#     def forward(self, x):
#         return self.rot_fc(x)  # [batch, output_steps , 3]

class rot_fc(nn.Module):
    def __init__(self, seq_len, pred_len, d_model):
        super(rot_fc, self).__init__()
        self.rot_fc = nn.Sequential(
            CBS_1D(seq_len, pred_len, k=3, s=1, p=1),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3)  # 原始输出3维向量
        )

    def forward(self, x):
        # 获取原始输出向量
        raw_vec = self.rot_fc(x)  # [batch, output_steps, 3]

        # # 计算模长并保持维度
        # angle = torch.norm(raw_vec, dim=-1, keepdim=True)  # [batch, output_steps, 1]
        #
        # # 使用sigmoid将模长限制在0-1范围，然后扩展到0-2π
        # scaled_angle = torch.sigmoid(angle) * 2 * math.pi  # [batch, output_steps, 1]
        #
        # # 计算单位方向向量（防止除以零）
        # direction = raw_vec / (angle + 1e-6)  # [batch, output_steps, 3]
        #
        # # 组合最终的旋转向量：方向 * 缩放后的角度
        # final_rot_vec = direction * scaled_angle

        return raw_vec  # [batch, output_steps, 3]

class radius_fc(nn.Module):
    def __init__(self, seq_len, pred_len, d_model):
        super(radius_fc, self).__init__()
        self.output_steps = pred_len
        self.radius_fc = nn.Sequential(
            CBS_1D(seq_len, 1, k=3, s=1, p=1),  # 调整时间步维度
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),  # 输出1维半径
            nn.ReLU()
        )

    def forward(self, x):
        return self.radius_fc(x).expand(-1, self.output_steps, -1)  # [batch, output_steps, 1]


class class_fc(nn.Module):
    def __init__(self,seq_len, pred_len, d_model):
        super(class_fc, self).__init__()
        self.output_steps = pred_len
        self.class_fc = nn.Sequential(
            CBS_1D(seq_len, 1, k=3, s=1, p=1),  # 调整时间步维度
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),  # 输出1维类别
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.class_fc(x).expand(-1, self.output_steps, -1)  # [batch, output_steps, 1]

class pre_coords(nn.Module):
    def __init__(self, seq_len=40, pred_len=10, d_model=256,):
        """
        input_size: 输入时序长度（特征数可以根据实际情况扩展）
        hidden_size: 隐藏层维度
        output_steps: 预测未来的时间步数
        """
        super(pre_coords, self).__init__()
        self.input_size = seq_len
        self.output_steps = pred_len

        # 平移分支：输出平移向量 T (3维)
        self.trans_fc = trans_fc(seq_len, pred_len, d_model)

        # 旋转分支：输出3D旋转向量 r (3维)
        self.rot_fc = rot_fc(seq_len, pred_len, d_model)

        # 半径分支：输出每个时间步的半径标量
        self.radius_fc = radius_fc(seq_len, pred_len, d_model)

        # 装甲板初始位置分支
        self.init_angle_fc = nn.Sequential(
            CBS_1D(seq_len, 1, k=3, s=1, p=1),
            nn.Linear(d_model, 2),
            # nn.Tanh()  # 输出范围 [-1, 1]，后续映射到 [0, 2π]
        )

    def forward(self, x):
        """
        x: [batch, seq_len, input_features]
        返回：
            pred_coords: [batch, output_steps, 3]，最终预测的装甲板坐标
            trans_out: 预测的平移部分
            rot_vec: 预测的3D旋转向量
        """
        batch_size = x.size(0)
        # residual 平移预测
        trans_out = self.trans_fc(x)  # [batch, output_steps, 3]

        # 旋转预测
        rot_vec = self.rot_fc(x)  # [batch, output_steps , 3]

        # 半径预测 [batch, output_steps]
        radius_pred_expanded = self.radius_fc(x)  # [batch, 1, 1]

        # # 预测初始角度 theta0 平面坐标系
        # theta0 = self.theta_fc(x)        # [batch, 1, 1]
        # theta0 = theta0.squeeze(-1)      # [batch, 1]
        # theta0 = (theta0 + 1) * np.pi    # 映射到 [0, 2π]
        #
        # # 生成初始点 p0 = [cos(theta0), sin(theta0), 0] * radius
        # p0 = torch.stack([
        #     torch.sin(theta0),
        #     torch.zeros_like(theta0),
        #     torch.cos(theta0)
        # ], dim=-1)  # [batch, 3]
        #
        # # 扩展 p0 到每个时间步并乘以半径
        # p0 = (p0.expand(-1, self.output_steps, -1))* radius_pred_expanded   # [batch, output_steps, 3]

        # 球坐标角度预测
        angles = self.init_angle_fc(x)  # [batch, 1, 2]
        theta_phi = angles.squeeze(1)  # [batch, 2]

        # 分解角度
        # theta = (theta_phi[:, 0] + 1) * np.pi  # 方位角[0, 2π]
        # phi = (theta_phi[:, 1] + 1) * np.pi / 2  # 极角[0, π]

        # 扩展到所有时间步
        theta = theta_phi[:, 0].unsqueeze(1).expand(-1, self.output_steps)  # [batch, output_steps]
        phi = theta_phi[:, 1].unsqueeze(1).expand(-1, self.output_steps)  # [batch, output_steps]
        radius = radius_pred_expanded.squeeze(-1)  # [batch, output_steps]

        # 球坐标系转笛卡尔坐标 x正前，y正左，z正上
        x_coord = radius * torch.sin(phi) * torch.cos(theta)
        y_coord = radius * torch.sin(phi) * torch.sin(theta)
        z_coord = radius * torch.cos(phi)

        # 组合初始点坐标
        p0 = torch.stack([x_coord, y_coord, z_coord], dim=-1)  # [batch, output_steps, 3]

        # 将预测的旋转向量转换为旋转矩阵
        R = rotation_matrix_from_vector(rot_vec)  # [batch, output_steps, 3, 3]
        rotated = torch.matmul(R, p0.unsqueeze(-1)).squeeze(-1)  # [batch, output_steps, 3]

        # 最终预测坐标
        pred_coords = trans_out + rotated

        return pred_coords, trans_out, rot_vec, radius_pred_expanded

class DualBranchTimeSeriesPredictor(nn.Module):
    def __init__(self, d_model=256, n_heads=8, d_ff=512, num_layers=3, input_dim=7, seq_len=40, pred_len=10):
        """
        input_size: 输入时序长度（特征数可以根据实际情况扩展）
        hidden_size: 隐藏层维度
        output_steps: 预测未来的时间步数
        """
        super(DualBranchTimeSeriesPredictor, self).__init__()
        self.input_size = seq_len
        self.output_steps = pred_len

        # 叠加多个 Encoder 层
        self.encoders = Encoders(d_model, n_heads, d_ff, num_layers)

        # 位置预测
        self.pre_coords = pre_coords(seq_len, pred_len, d_model)

        # 类别损失
        self.class_fc = class_fc(seq_len, pred_len, d_model)

    def forward(self, x):
        """
        x: [batch, seq_len, input_features]
        返回：
            pred_coords: [batch, output_steps, 3]，最终预测的装甲板坐标
            trans_out: 预测的平移部分
            rot_vec: 预测的3D旋转向量
        """
        x = self.encoders(x)
        # 类别预测
        class_pred_expanded = self.class_fc(x)  # [batch, 1, 1]

        pred_coords, trans_out, rot_vec, radius_pred_expanded =self.pre_coords(x)

        return pred_coords, trans_out, rot_vec, radius_pred_expanded, class_pred_expanded


# 示例用法
if __name__ == '__main__':
    # 假设输入40个时间步，每个时间步4维特征（比如时间和归一化后的xyz）
    batch_size = 1
    seq_len = 40
    input_features = 7
    hidden_size = 128
    output_steps = 10  # 最后预测0.4s-0.5s的数据

    model = DualBranchTimeSeriesPredictor(input_dim=input_features,
                                          seq_len=40,
                                          pred_len=10)

    dummy_input = torch.randn(batch_size, seq_len, input_features)
    pred_coords, trans, rot, radius,class_pred_expanded = model(dummy_input)
    print("Predicted coordinates shape:", pred_coords.shape)  # 应为 [1, 10, 3]
    print('平移向量:', trans.shape)
    print('旋转向量:', rot)
    angle = torch.norm(rot, dim=-1, keepdim=True)
    print('旋转角度:', angle)
    print('旋转半径:', radius.shape)