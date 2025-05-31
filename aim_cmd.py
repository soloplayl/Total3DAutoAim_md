import argparse
import re
import train as vt
import predicted as pt
import export as et
import utils.generate_dataset as gdarmor
import utils.generate_dataset_vane as gdvane
import numpy as np
class ConfigParser:
    def __init__(self, default_config):
        self.default_config = default_config
        self.parser = argparse.ArgumentParser(
            description='config parser',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False
        )
        self.parser.add_argument('params', nargs='*', help='parameters (e.g., lr=0.1 batch_size=32)')
        self.parser.add_argument('-h', '--help', action='store_true', help='show this help message and exit')

    def parse_args(self):
        args, unknown = self.parser.parse_known_args()

        if args.help:
            self.print_help()
            exit(0)

        # 解析YOLO风格参数
        config_updates = {}
        for param in args.params:
            match = re.match(r'([a-zA-Z_]\w*)=([\w.+-]+)', param)
            if match:
                key, value = match.groups()
                # 尝试转换数值类型
                try:
                    if '.' in value:
                        config_updates[key] = float(value)
                    else:
                        config_updates[key] = int(value)
                except ValueError:
                    config_updates[key] = value
            else:
                print(f"\033[91m警告: 忽略无效参数格式 '{param}' (应使用 key=value 格式)\033[0m")

        # 合并默认配置和更新值
        config = {**self.default_config, **config_updates}

        # 确保feature_dim是整数类型
        if 'feature_dim' in config:
            config['feature_dim'] = int(config['feature_dim'])

        return config

    def print_help(self):
        print("\n\033[1;36m参数解析器\033[0m")  # 青色加粗标题
        print("\033[93m用法: python aim_cmd.py [key1=value1] [key2=value2] ...\033[0m")  # 黄色说明
        print("\n\033[1;35m示例:\033[0m")  # 紫色加粗示例标题
        print("  python aim_cmd.py feature_dim=7 lr=0.001 batch_size=4096")
        print("  python aim_cmd.py model_type=vane_transformer unit=m d_model=512 epochs=5000")
        print("\n\033[1;34m支持的配置参数:\033[0m")  # 蓝色加粗参数标题
        for key, value in self.default_config.items():
            print(f"\033[93m{key}\033[0m: \033[92m{value}\033[0m (\033[94m{type(value).__name__}\033[0m)")
        print("\n\033[1;33m使用 -h 或 --help 显示此帮助信息\033[0m")  # 黄色提示

        # print("\n使用 -h 或 --help 显示此帮助信息")


# 默认配置
default_config = {
    'mode': 'train',  # train/predict/export/vane_generate/armor_generate 采用训练模式/预测模式/导出模型/能量机关数据生成/自瞄装甲板数据生成
    'data_path': 'model/datasets/processed_log.txt', # 数据集路径
    'feature_dim': 3,  # 特征维度7是:t, x, y, z, rot(3 dims) 4是t, x, y, z 3是x, y, z
    'input_size': 20,  # 输入窗口大小 dt=0.0125 t = 20*0.0125=0.25s
    'output_size': 10,  # 输出窗口大小 pred_len=10 offset=36 dt=0.0125 pred_t=(36+10)*0.0125=0.45s+0.125s=0.575s
    'offset': 36,  # 预测窗口偏置
    'batch_size': 1024, # 批量大小
    'test_ratio': 0.2, # 测试集比例
    'epochs': 1000, # 训练轮数
    'lr': 1e-3, # 学习率
    'patience': 30,  # 提前停止的耐心值
    'model_type': 'DualBranchTimeSeriesPredictor',  # vane_transformer/DualBranchTimeSeriesPredictor
    'total_transformer_save_path': 'model/vane_model/total_Predictor.pth', # total_transformer模型保存路径
    'vane_transformer_save_path': 'model/vane_model/vane_Predictor.pth', # vane_transformer模型保存路径
    'd_model': 256, # 模型维度 d_model=256
    'n_heads': 8, # 注意力头数
    'd_ff': 512,  # 前馈网络维度这个ffn层设为512的的话建议训练所有状态学习，单学习能量机关256即可，如果预测自瞄建议512
    'num_layers': 3, # 编码器层数
    'eta_min': 1e-7,  # 余旋退火参数
    'data_mode': 'txt',  # vane/armor/txt 对应vane_generate/armor_generate/自己的txt文件训练集
    'unit': 'mm',  # 数据单位 mm/m
    'vision': False,  # mode=predict的参数用于显示图像
    'sample': None # 采样数，None表示全部数据mode=predict的参数,在生成数据集里面表示生成样本数
}

# --- 冻结配置字典（0表示冻结，1表示可训练）---
default_freeze_config = {
    "encoders": 1,  # 所有Encoder层
    "class_fc": 0,  # 分类分支
    "pre_coords": 1  # 位置分支
}

if __name__ == "__main__":
    import torch

    print("===== PyTorch环境检测 =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available!")

    print("\n===== 运行方式检测 =====")
    try:
        # 用于检测是否在命令行中运行
        import sys

        if 'ipykernel' in sys.modules:
            print("运行环境: Jupyter Notebook")
        elif 'google.colab' in sys.modules:
            print("运行环境: Google Colab")
        else:
            print("运行环境: 命令行终端")
    except:
        print("运行环境检测失败")
    # 创建解析器
    parser = ConfigParser(default_config)

    # 解析参数并获取配置
    config = parser.parse_args()


    # 打印最终配置 - 添加颜色显示
    print("\n\033[1;36m最终使用的配置:\033[0m")  # 青色加粗标题

    # 计算最长键名用于对齐
    max_key_length = max(len(key) for key in config.keys())

    for key, value in config.items():
        # 如果值是字符串，添加引号
        value_str = f"'{value}'" if isinstance(value, str) else str(value)

        # 为不同类型的值分配不同颜色
        color_code = '\033[95m'  # 亮紫色 - 字符串
        if isinstance(value, int):
            color_code = '\033[96m'  # 青色 - 整数
        elif isinstance(value, float):
            color_code = '\033[92m'  # 绿色 - 浮点数

        # 格式化输出，键名黄色显示
        print(f"  \033[93m{key.ljust(max_key_length)}\033[0m : {color_code}{value_str}\033[0m (\033[94m{type(value).__name__}\033[0m)")

    # 这里可以添加训练代码
    if config['mode'] == 'train':
        # 更新vane_train模块的配置
        vt.config = config
        vt.freeze_config = default_freeze_config
        vt.train_init()
    if config['mode'] == 'predict':
        pt.predict_init(config)
    if config['mode'] == 'export':
        et.config = config
        et.export_init()
    if config['mode'] == 'vane_generate':
        dataset = gdvane.generate_dataset(config['sample'], gdvane.CONFIG)
        np.savez_compressed('./model/datasets/vane_dataset'+str(config['sample'])+'.npz', **dataset)
    if config['mode'] == 'armor_generate':
        dataset = gdarmor.generate_dataset(config['sample'], gdarmor.CONFIG)
        np.savez_compressed('./model/datasets/armor_dataset'+str(config['sample'])+'.npz', **dataset)