import numpy as np
from tqdm import tqdm


def generate_dataset(num_samples, config):
    dataset = {
        't': np.zeros((num_samples, config['n_points'])),
        'theta': np.zeros((num_samples, config['n_points'])),
        'motion_type': np.zeros(num_samples, dtype=int),  #  0表示匀速,1变速
        'direction': np.zeros(num_samples, dtype=int),   #  -1表示正,1表示逆
        'phi': np.zeros((num_samples, config['n_points'])), # 倾斜角（0到π）
    }

    for i in tqdm(range(num_samples), desc="生成进度"):
        # 判断是否为匀角速度样本
        is_uniform = np.random.rand() < config.get('uniform_ratio', 0.0)
        sign = np.random.choice([-1, 1])
        dataset['direction'][i] = sign  # 1表示变速

        # 为每个时间点的傾斜角添加噪声并保持范围
        phi_base = np.random.uniform(0, np.pi)
        phi_noise = config['noise_level'] * np.pi * np.random.randn(config['n_points'])
        dataset['phi'][i] = (phi_base + phi_noise) % np.pi  # 使用模运算保持范围
        if is_uniform:
            # 匀角速度样本参数
            a = 0.0
            omega = sign * np.pi / 3
            dataset['motion_type'][i] = 0  # 0表示匀速
        else:
            # 振荡样本参数：随机正反转
            a = np.random.uniform(
                low=config['bounds'][0, 0],
                high=config['bounds'][0, 1]
            )
            phi = np.random.uniform(
                low=config['bounds'][2, 0],
                high=config['bounds'][2, 1]
            )
            # 随机生成正负角速度（绝对值在配置范围内）
            omega = np.random.uniform(
                low=config['bounds'][1, 0],
                high=config['bounds'][1, 1]
            )
            dataset['motion_type'][i] = 1  # 1表示变速

        # 生成时间序列（公共部分）
        intervals = config['base_interval'] + np.random.uniform(
            -config['max_perturbation'],
            config['max_perturbation'],
            size=config['n_points'] - 1
        )
        t_base = np.random.uniform(0,  3)
        t = np.concatenate((
            [t_base],
            t_base + np.cumsum(intervals)
        ))
        # print(t)
        # 计算角度值
        if is_uniform:
            theta = (omega * t) % (2 * np.pi)

            print("theta[0]:",theta[0])
        else:
            phase = omega * t + phi
            cos_term = -a / omega * np.cos(phase)
            linear_term = (config['linear_coeff'] - a) * t
            theta = sign*(cos_term + linear_term) % (2 * np.pi)

        # 添加噪声
        theta += config['noise_level'] * np.pi * np.random.randn(config['n_points'])

        # 存储数据
        dataset['t'][i] = t
        dataset['theta'][i] = theta


    return dataset

# 配置参数
CONFIG = {
    'bounds': np.array([
        [0.780, 1.045],   # a 的范围
        [1.884, 2.000],   # omega 的绝对值范围（振荡样本）
        [0, 2 * np.pi]    # phi 的范围
    ]),
    'start_time': 0.5,
    'base_interval': 0.0125, # 基础时间间隔（秒）
    'max_perturbation': 0.002, # 最大扰动（秒）
    'samples': 1000,  # 生成样本数量
    'n_points': 500, # 每个样本的时间点数量
    'linear_coeff': 2.090,
    'noise_level': 0.01,
    'uniform_ratio': 0.4  # 匀角速度样本占比（0.5表示50%）
}

# 示例使用
if __name__ == "__main__":

    dataset = generate_dataset(CONFIG['samples'], CONFIG)
    np.savez_compressed('../model/datasets/vane_dataset'+str(CONFIG['samples'])+'.npz', **dataset)

    # 验证样本
    sample_idx = 0
    print(f"\n样本 {sample_idx} 参数:")
    # print(f"a={dataset['params'][sample_idx][0]:.4f}, ω={dataset['params'][sample_idx][1]:.4f}, φ={dataset['params'][sample_idx][2]:.4f}")
    print(f"时间范围: {dataset['t'][sample_idx][0]:.3f}s 到 {dataset['t'][sample_idx][-1]:.3f}s")