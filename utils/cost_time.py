import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import block
# 设置中文字体（无需额外安装）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体，Windows 系统自带
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
def measure_inference_time(model, input_shape, device='cuda', num_warmup=10, num_repeats=100, batch_size=1):
    """
    测量模型的推理时间
    Args:
        model: 已加载的模型
        input_shape: 输入张量的形状 (seq_len, input_dim)
        device: 'cuda' 或 'cpu'
        num_warmup: 预热次数（避免首次运行的初始化开销）
        num_repeats: 正式测量的重复次数
        batch_size: 批处理大小
    Returns:
        avg_time: 平均推理时间（毫秒）
        std_time: 时间标准差（毫秒）
    """
    # 确保模型在评估模式
    model.eval().to(device)

    # 生成随机输入数据
    seq_len, input_dim = input_shape
    dummy_input = torch.randn(batch_size, seq_len, input_dim).to(device)

    # 预热（不计算时间）
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # 测量时间（GPU需要同步）
    if device == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        timings = []
        with torch.no_grad():
            for _ in range(num_repeats):
                start_event.record()
                _ = model(dummy_input)
                end_event.record()
                torch.cuda.synchronize()  # 等待CUDA操作完成
                timings.append(start_event.elapsed_time(end_event))  # 毫秒
    else:
        timings = []
        with torch.no_grad():
            for _ in range(num_repeats):
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                timings.append((end_time - start_time) * 1000)  # 毫秒

    # 计算统计量
    avg_time = np.mean(timings)
    std_time = np.std(timings)
    return avg_time, std_time


def plot_inference_time(batch_sizes, avg_times, std_times, title="cpu_i710代(不含AVX512)推理模型时间"):
    """
    绘制推理时间对比图
    """
    plt.figure(figsize=(10, 6))
    plt.errorbar(batch_sizes, avg_times, yerr=std_times, fmt='-o', capsize=5)
    plt.xlabel("Batch Size")
    plt.ylabel("推理时间 (毫秒)")
    plt.title(title)
    plt.grid(True)
    plt.savefig("cpu_i710_inference_time.png")
    plt.close()


if __name__ == "__main__":
    # 示例：测量不同批处理大小的时间
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    input_shape = (40, 10)  # 输入形状 (seq_len, input_dim)
    print(torch.__config__.show())
    # 加载你的模型
    model = block.DualBranchTimeSeriesPredictor(seq_len=40, pred_len=10).to(device)
    model.load_state_dict(torch.load("../model/DualBranchTimeSeriesPredictor_new_best.pth", map_location=device))
    # model = torch.load("model/quantized_model_dynamic.pth",
    #                    map_location=device,
    #                    weights_only=False)  # 关闭安全模式

    # 测量不同批处理大小的时间
    batch_sizes = [1, 2, 4, 8, 16]
    avg_times = []
    std_times = []

    for bs in batch_sizes:
        avg, std = measure_inference_time(model, input_shape, device=device, batch_size=bs)
        avg_times.append(avg)
        std_times.append(std)
        print(f"Batch Size: {bs}, 平均时间: {avg:.2f} ± {std:.2f} ms")

    # 绘制结果
    plot_inference_time(batch_sizes, avg_times, std_times)