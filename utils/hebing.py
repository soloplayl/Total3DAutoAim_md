import numpy as np

# 加载两个数据集
dataset = np.load('../model/datasets/time_armor_series_yunsu.npz')
dataset1 = np.load('../model/datasets/time_armor_series_stats.npz')
dataset2 = np.load('../model/datasets/time_armor_series_Rot.npz')
# 创建合并后的字典
merged_data = {}

# 确保两个数据集包含相同的键
# assert set(dataset.files) == set(dataset1.files), "数据集结构不一致"
print("dataset:",len(dataset['class']),"\n",
      "dataset1:",len(dataset1['class']),"\n",
      "dataset2:",len(dataset2['class']))
# 按样本维度合并每个字段
for key in dataset.files:
    merged_data[key] = np.concatenate([dataset[key], dataset1[key], dataset2[key]], axis=0)

# 保存合并后的数据集
np.savez_compressed('../model/datasets/time_armor_series_RotOrTrans_20000.npz', **merged_data)

print(f"合并完成，总样本数: {len(merged_data['class'])}")