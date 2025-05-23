import numpy as np

class WindowGenerator:
    def __init__(self, input_width, label_width, offsite=40):
        self.input_width = input_width  # 输入窗口大小 (n)
        self.label_width = label_width  # 预测窗口大小 (n)
        self.offsite = offsite  # 预测窗口偏置

    def split_window(self, features):
        inputs = features[:, :self.input_width, :4]  # 只取txyz作为输入
        labels = features[:, self.input_width + self.offsite: self.input_width + self.offsite + self.label_width,
                 1:]  # 只取xyz作为label
        # print(inputs.shape)
        # print(labels.shape)
        return inputs, labels


def create_dataset(data, input_size=40, output_size=10, offset=40):
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

    # 平移数据，此处计算结果为 (4000, 200, 3)
    trans_data = data['trans'] - np.stack([data['z'][:, 0], data['x'][:, 0], data['y'][:, 0]], axis=-1)[:, np.newaxis,
                                 :]

    # data['radius'] 原始形状 (4000,)，扩展为 (4000, 1, 1)，再沿时间步复制到 (4000, 200, 1)
    radius_data = data['radius'][:, None, None]  # (4000, 1, 1)
    radius_data = np.tile(radius_data, (1, data['z'].shape[1], 1))  # (4000, 200, 1)
    # class
    class_data = data['class'][:, None, None]  # (4000, 1, 1)
    class_data = np.tile(class_data, (1, data['z'].shape[1], 1))  # (4000, 200, 1)
    # omega
    omega_data = data['omega'][:, None, None]  # (4000, 1, 1)
    omega_data = np.tile(omega_data, (1, data['z'].shape[1], 1))  # (4000, 200, 1)
    # 使用 np.concatenate (各数组形状均为 (4000, 200, C)) 按最后一个维度拼接
    features = np.concatenate([t_diff, z_diff, x_diff, y_diff , rot_data, trans_data, radius_data, class_data, omega_data], axis=-1)

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
