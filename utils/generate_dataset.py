import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_dataset(num_samples, config):
    """
    生成时间序列数据集
    Args:
        num_samples: 生成样本数量
        config: 配置字典，包含以下键值对：
            - base_interval: 基础时间间隔
            - max_perturbation: 最大时间扰动
            - n_points: 数据点数量
            - start_time: 起始时间
            - radius_bounds: 旋转半径范围
            - yunsu_bounds: 角速度范围
            - uniform_ratio: 小陀螺概率
            - noise_level: 噪声水平
            - car_center_bounds: 车辆中心范围
            - center_v_bounds: 车辆速度范围
            - center_a_bounds: 车辆加速度范围
    Returns:
        dataset: 生成的数据集，包含以下字段：
            - t: 时间序列
            - x: X坐标序列
            - y: Y坐标序列
            - z: Z坐标序列
            - rot: 旋转向量
            - trans: 平移向量
            - radius: 旋转半径
            - theta0: 装甲板初始角度
            - motion_type: 小陀螺运动类型
            - car_center: 车辆中心坐标
            - car_motion_type: 车辆运动类型
            - direction: 运动方向
            - class: 样本类别
            - omega: 角速度


    """
    dataset = {
        't': np.zeros((num_samples, config['n_points'])),
        'x': np.zeros((num_samples, config['n_points'])),
        'y': np.zeros((num_samples, config['n_points'])),
        'z': np.zeros((num_samples, config['n_points'])),
        'rot': np.zeros((num_samples, config['n_points'],3)),
        'trans': np.zeros((num_samples, config['n_points'],3)),
        'radius': np.zeros(num_samples, dtype=int),
        'theta0': np.zeros(num_samples, dtype=int),
        'motion_type': np.zeros(num_samples, dtype=int),  #  0表示小陀螺匀速,1小陀螺静止
        'car_motion_type':np.zeros(num_samples, dtype=int), # 0表示车匀速，1表示车加速，2表示静止，3匀速上坡，4加速上坡
        'direction': np.zeros(num_samples, dtype=int),   #  -1表示正,1表示逆
        'class': np.zeros(num_samples, dtype=int), #1:匀速小陀螺 0: 其他
        'omega': np.zeros(num_samples, dtype=float),  # 角速度
    }
    for i in tqdm(range(num_samples), desc="生成进度"):
        is_uniform = np.random.rand() < 1  # 小陀螺概率
        is_uniform2 = np.random.rand() < 0 # 车运动概率
        sign = np.random.choice([-1, 1])
        dataset['direction'][i] = sign  # 1表示变速
        if is_uniform:
            radius = np.random.uniform(
                low=config['radius_bounds'][0],
                high=config['radius_bounds'][1]
            )
        else:
            radius = 0
        dataset['radius'][i] = radius
        # 生成车辆中心
        x = np.random.uniform(
            low=config['car_center_bounds'][0, 0],
            high=config['car_center_bounds'][0, 1]
        )
        y = np.random.uniform(
            low=config['car_center_bounds'][1, 0],
            high=config['car_center_bounds'][1, 1]
        )
        z = np.random.uniform(
            low=config['car_center_bounds'][2, 0],
            high=config['car_center_bounds'][2, 1]
        )

        # 生成非均匀时间序列
        intervals = config['base_interval'] + np.random.uniform(
            -config['max_perturbation'],
            config['max_perturbation'],
            size=config['n_points'] - 1
        )
        t = np.concatenate((
            [config['start_time']],
            config['start_time'] + np.cumsum(intervals)
        ))

        car_init_center = np.array([x, y, z], dtype=np.float32)  # 初始化装甲板绕中心旋转点坐标
        car_center = np.zeros((len(t), 3))
        dataset['car_motion_type'][i] = 2
        vx = 0
        vz = 0
        vy = 0
        ax = 0
        az = 0
        ay = 0
        is_uniform3 = 0  # 变速概率
        is_uniform4 = 0  # 上下坡概率
        if is_uniform2:
            is_uniform3 = np.random.rand() < 0  # 变速概率
            is_uniform4 = np.random.rand() < 0.25  # 上下坡概率
            vx = np.random.uniform(
                low=config['center_v_bounds'][0],
                high=config['center_v_bounds'][1]
            )*np.random.choice([-1, 1])
            vy = np.random.uniform(
                low=config['center_v_bounds'][0],
                high=config['center_v_bounds'][1]
            )*np.random.choice([-1, 1])

            dataset['car_motion_type'][i] = 0 # 匀速
            if is_uniform4:
                vz = np.random.uniform(
                low=config['center_v_bounds'][0],
                high=config['center_v_bounds'][1]
            )*np.random.choice([-1, 1])

            if is_uniform3:
                dataset['car_motion_type'][i] = 1  # 加速
                ax = np.random.uniform(
                    low=config['center_a_bounds'][0],
                    high=config['center_a_bounds'][1]
                )*np.random.choice([-1, 1])
                ay = np.random.uniform(
                    low=config['center_a_bounds'][0],
                    high=config['center_a_bounds'][1]
                )*np.random.choice([-1, 1])
                if is_uniform4:
                    dataset['car_motion_type'][i] = 4
                    az = np.random.uniform(
                        low=config['center_a_bounds'][0],
                        high=config['center_a_bounds'][1]
                    )*np.random.choice([-1, 1])
            if dataset['car_motion_type'][i]==0:
                if is_uniform4:
                    dataset['car_motion_type'][i] = 3

            car_center[:, 0] = vx * t + 0.5 * ax * t * t
            car_center[:, 1] = vy * t + 0.5 * ay * t * t
            car_center[:, 2] = vz * t + 0.5 * az * t * t
            car_center += car_init_center
        else:
            car_center += car_init_center
        # else:
        #     print(car_center)


        if is_uniform:
            # 匀角速度样本参数
            omega = np.random.uniform(
                low=config['yunsu_bounds'][0],
                high=config['yunsu_bounds'][1]
            )
            dataset['motion_type'][i] = 0  # 0表示匀速
        else:
            omega = 0
            dataset['motion_type'][i] = 1  # 1表示静止

        dataset['omega'][i] = sign * omega  # 角速度
        # print( dataset['omega'][i])
        # 计算角度序列（支持可变角速度）
        theta = sign * (omega * t) % (2 * np.pi)
        # print(theta)
        # 如需可变角速度，可改为：theta = np.cumsum(omega_sequence * intervals)
        theta += config['noise_level'] * np.pi * np.random.randn(config['n_points'])
        # 生成装甲板位置序列（向量化操作）
        if is_uniform4:
            # 生成随机初始装甲板位置
            theta0 = np.random.uniform(0, 2 * np.pi)
            dataset['theta0'][i] = theta0  # 新增字段保存初始角度
            armor_local = radius * np.array([np.cos(theta0), np.sin(theta0), 0])
            # 生成与速度方向垂直的旋转轴
            v = np.array([vx,vy,vz])
            v_norm = v / np.linalg.norm(v)
            rand_vec = np.array([0,vy,0])
            rand_vec = rand_vec/np.linalg.norm(rand_vec)
            axis = np.cross(v_norm, rand_vec)
            axis /= np.linalg.norm(axis)
            # print(axis)
            rotations = R.from_rotvec(np.outer(theta, axis))
            # print(rotations.as_rotvec().shape)
            armor_pos = rotations.apply(armor_local) + car_center
        else:
            # armor_pos = np.empty((len(t), 3))
            theta0 = np.random.uniform(0, 2 * np.pi)
            dataset['theta0'][i] = theta0  # 新增字段保存初始角度
            armor_local = radius * np.array([np.cos(theta0), np.sin(theta0), 0])
            # armor_pos[:, 0] = radius * np.cos(theta)  # X坐标
            # armor_pos[:, 1] = radius * np.sin(theta)  # Y坐标
            # armor_pos[:, 2] = 0  # Z坐标固定
            axis = np.array([0, 0, -1])  # 绕z轴旋转
            rotations = R.from_rotvec(np.outer(theta, axis))
            armor_pos = rotations.apply(armor_local) + car_center


        # 保存旋转向量和平移向量
        dataset['rot'][i] = rotations.as_rotvec()
        dataset['trans'][i] = car_center
        # 添加xyz轴随机抖动（σ=±4mm高斯分布）
        armor_pos[:, 2] += np.random.normal(
            loc=0.0,
            scale=4,
            size=[config['n_points']]
        )
        armor_pos[:, 0] += np.random.normal(
            loc=0.0,
            scale=4,
            size=[config['n_points']]
        )
        armor_pos[:, 1] += np.random.normal(
            loc=0.0,
            scale=4,
            size=[config['n_points']]
        )
        if dataset['motion_type'][i]==0 and dataset['car_motion_type'][i]!=2:
            # print('dataset[motion_type][i]:',dataset['motion_type'][i])
            # print("dataset['car_motion_type'][i]",dataset['car_motion_type'][i])
            dataset['class'][i]=1
        else:
            dataset['class'][i]=0

        # 存储数据
        dataset['t'][i] = t
        dataset['x'][i] = armor_pos[:, 0]
        dataset['y'][i] = armor_pos[:, 1]
        dataset['z'][i] = armor_pos[:, 2]

    return dataset

# 参数配置
CONFIG = {
    'base_interval': 0.015,  # 基础时间间隔
    'max_perturbation': 0.002,  # 最大时间扰动
    'n_points': 200,  # 数据点数量 base_interval*n_points=2s center_v_bounds*2s=6000mm
    'start_time': 0,  # 起始时间
    'radius_bounds': np.array([50, 400]),  # 旋转半径
    'yunsu_bounds': np.array([0, 3 * np.pi]),  # 角速度（rad/s）
    'uniform_ratio': 0.5,
    'noise_level': 0.005,
    'car_center_bounds': np.array([
        [2000.0, 8000.0],  # x 的范围
        [-3000.0, 3000.0],  # y 的范围
        [0.0, 8000.0]  # z 的范围
    ]),
    'center_v_bounds': np.array([1000, 2000]),
    'center_a_bounds': np.array([0, 500]),
}

if __name__ == "__main__":
    dataset = generate_dataset(2000, CONFIG)
    np.savez_compressed('../model/datasets/time_armor_series_Rot.npz', **dataset)

    # 验证样本
    sample_idx = 0
    print(f"\n样本 {sample_idx} 参数:")
    # print(f"a={dataset['params'][sample_idx][0]:.4f}, ω={dataset['params'][sample_idx][1]:.4f}, φ={dataset['params'][sample_idx][2]:.4f}")
    print(f"时间范围: {dataset['t'][sample_idx][0]:.3f}s 到 {dataset['t'][sample_idx][-1]:.3f}s")
