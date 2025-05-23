import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体（解决中文显示问题）
rcParams['font.sans-serif'] = ['SimHei']  # 微软雅黑也可以尝试 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
dataset = np.load('model/time_armor_series_dataset.npz')
# 获取样本总数（假设所有字段的样本数一致）
num_samples = dataset['x'].shape[0]  # 假设x的形状为(num_samples, time_steps)

# 随机选择一个样本索引
sample_idx = np.random.randint(0, num_samples)  # 均匀分布随机采样

x = dataset['x'][sample_idx]
y = dataset['y'][sample_idx]
z = dataset['z'][sample_idx]
print(z[0])
print(x[0])
print(y[0])
print('====================')
print(z[1])
print(x[1])
print(y[1])
print('====================')
print(z[2])
print(x[2])
print(y[2])
motion_types = ['小陀螺匀速','小陀螺静止']
car_motion_types = ['车匀速','车加速','车静止','匀速上坡','加速上坡']

car_motion_type = dataset['car_motion_type'][sample_idx]

motion_type = dataset['motion_type'][sample_idx]
for i in range(2):
    if motion_type==i:
        motion_type=motion_types[i]
        break
for i in range(5):
    if car_motion_type==i:
        car_motion_type=car_motion_types[i]
        break

direction_type = dataset['direction'][sample_idx]
radius = dataset['radius'][sample_idx]
# 创建画布

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Z')
ax.set_ylabel('X')
ax.set_zlabel('Y')
ax.set_title('圆周运动轨迹动态演示')
ax.grid(True)
ax.axis('equal')

ax.set_xlim(0.0, 10000.0)# z 的范围
ax.set_ylim(-5000.0, 5000.0) # x 的范围
ax.set_zlim(-5000.0, 5000.0)# y 的绝对值范围

# 控制绘制速度
n_points = x.shape[0]
for frame in range(n_points):
    if frame%5==0:
        ax.scatter(z[frame], x[frame], y[frame], color='red', s=3, zorder=5)
        ax.text(0.05, 0.95, 0,
                f'idx:{sample_idx},时间: {dataset["t"][sample_idx, frame]:.2f}s,运动方式:{motion_type},运动方向:{direction_type},旋转半径:{radius},车:{car_motion_type}',
                transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
        plt.pause(0.001)  # 调整这个值控制播放速度，0.001对应约1000帧/秒
plt.plot(z,x,y)
plt.show()
