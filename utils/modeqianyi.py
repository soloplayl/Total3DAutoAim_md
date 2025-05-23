import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import block
import os
import time
import loss

mode = 'pre_coords'  # 'all', 'encoders', 'class', 'pre_coords'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = '../model/DualBranchTimeSeriesPredictor_infantry_10_12_best_RotOrTrans.pth'
new_save_path = '../model/infantry10_12_8/pre_coords_10_12_8_class0.pth'
# model = block.DualBranchTimeSeriesPredictor(
#     d_model=256, n_heads=8, d_ff=512, num_layers=3, input_dim=4, seq_len=10, pred_len=12
# ).to(device)
# model = block.Encoders(
#     d_model=256, n_heads=8, d_ff=512, num_layers=3
# ).to(device)
# model = block.class_fc(
#      seq_len=10, pred_len=12, d_model=256
# ).to(device)
model = block.pre_coords(
     seq_len=10, pred_len=12,d_model=256
).to(device)
if os.path.exists(save_path):
    pretrained_dict = torch.load(save_path, map_location=device)
    model_dict = model.state_dict()
    renamed_pretrained_dict = {}

    print(pretrained_dict.keys())
    print(model_dict.keys())
    if mode =='all':
        for k, v in pretrained_dict.items():
            new_k = k
            # 修改 encoder 层前缀 后面替代前面
            if new_k.startswith('fc1.'):
                new_k = new_k.replace('fc1.', 'encoders.fc1.')
            elif new_k.startswith('encoders.'):
                new_k = new_k.replace('encoders.', 'encoders.layers.')
            # 修改 trans_fc 层前缀
            elif new_k.startswith('trans_fc.'):
                new_k = new_k.replace('trans_fc.', 'pre_coords.trans_fc.trans_fc.')
            # 修改 rot_fc 层前缀
            elif new_k.startswith('rot_fc.'):
                new_k = new_k.replace('rot_fc.', 'pre_coords.rot_fc.rot_fc.')
            # 修改 radius_fc 层前缀
            elif new_k.startswith('radius_fc.'):
                new_k = new_k.replace('radius_fc.', 'pre_coords.radius_fc.radius_fc.')
            # 修改 class_fc 层前缀
            elif new_k.startswith('class_fc.'):
                new_k = new_k.replace('class_fc.', 'class_fc.class_fc.')

            if new_k == 'p0':
                new_k = 'pre_coords.p0'
            if new_k in model_dict:
                renamed_pretrained_dict[new_k] = v
    elif mode == 'encoders':
        for k, v in pretrained_dict.items():
            new_k = k
            if new_k.startswith('encoders.'):
                new_k = new_k.replace('encoders.', '')
            if new_k in model_dict:
                renamed_pretrained_dict[new_k] = v

    elif mode == 'class':
        for k, v in pretrained_dict.items():
            new_k = k
            if new_k.startswith('class_fc.'):
                print(1)
                new_k = new_k.replace('class_fc.class_fc.', 'class_fc.')
            if new_k in model_dict:
                renamed_pretrained_dict[new_k] = v

    elif mode == 'pre_coords':
        for k, v in pretrained_dict.items():
            new_k = k
            if new_k.startswith('pre_coords.'):
                new_k = new_k.replace('pre_coords.', '')
            if new_k in model_dict:
                renamed_pretrained_dict[new_k] = v
    model_dict.update(renamed_pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    print(f"成功加载以下匹配层：")
    for k in renamed_pretrained_dict.keys():
        print(f" - {k}")
    print('待加载模型参数总个数:',len(model_dict))
    print(f"共加载 {len(renamed_pretrained_dict)}/{len(pretrained_dict)} 个参数")

    torch.save(model.state_dict(), new_save_path)
    print(f"模型权重已保存到 {new_save_path}")