import torch
import torch.nn as nn
import block
import onnxruntime as ort
import numpy as np
config = {
    'data_path': 'model/datasets/tvec_data.npz',
    'input_size': 10,  # 输入窗口大小 dt=0.015 t = 20*0.015=0.3s
    'output_size': 12,  # 输出窗口大小 pred_len=10 offset=30 dt=0.015 pred_t=30*0.015=0.45s
    'offset': 8,  # 预测窗口偏置
    'batch_size': 5120,
    'test_ratio': 0.2,
    'epochs': 10000,
    'lr': 1e-5,
    'model_type': 'DualBranchTimeSeriesPredictor',
    'new_transformer_save_path': 'model/DualBranchTimeSeriesPredictor_infantry_10_12_best_real_new.pth',
    'd_model': 256,
    'n_heads': 8,
    'd_ff': 512,
    'num_layers': 3
}


# 1. 准备模型 ----------------------------------------------
def prepare_model():
    # 初始化模型
    model = block.DualBranchTimeSeriesPredictor(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        input_dim=4,
        seq_len=config['input_size'],
        pred_len=config['output_size'],
    )
    # 加载预训练权重
    checkpoint = torch.load(
        config['new_transformer_save_path'],
        map_location='cpu'
    )
    model.load_state_dict(checkpoint)
    model.eval()  # 切换为推理模式

    # 打印模型结构验证
    # print("模型结构:")
    # print(model)

    # 验证PyTorch模型输出形状
    dummy_input = torch.randn(1, 10, 4)
    with torch.no_grad():
        output = model(dummy_input)

    # 保存PyTorch输出供后续验证
    pt_output = output[0].numpy()

    print("\nPyTorch输出形状验证:", output[0].shape)
    assert output[0].shape == (1, 12, 3), "PyTorch模型输出形状不符合要求"

    return model


# 2. 导出ONNX模型 ------------------------------------------
def export_onnx(model):
    # 创建虚拟输入
    dummy_input = torch.randn(1, 10, 4)

    # 包装模型处理多输出问题
    class WrappedModel(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.model = original_model

        def forward(self, x):
            return self.model(x)[0]  # 只返回第一个输出

    wrapped_model = WrappedModel(model)
    wrapped_model.eval()

    # 导出参数
    onnx_path = "pred_model.onnx"
    input_names = ["input"]
    output_names = ["output"]

    # 执行导出
    torch.onnx.export(
        wrapped_model,  # 使用包装后的模型
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=14,
        dynamic_axes={
            # 如需动态维度可在此配置
            # "input": {0: "batch_size", 1: "sequence"},
            # "output": {0: "batch_size", 1: "sequence"}
        },
        verbose=True
    )

    print(f"\nONNX模型已保存至: {onnx_path}")
    return onnx_path


# 3. 验证ONNX模型 ------------------------------------------
def validate_onnx(onnx_path):
    # 创建推理会话
    ort_session = ort.InferenceSession(onnx_path)

    # 使用与PyTorch相同的输入数据
    input_data = np.random.randn(1, 10, 4).astype(np.float32)

    # 执行推理
    outputs = ort_session.run(
        output_names=None,
        input_feed={"input": input_data}
    )

    # 形状验证
    print("\nONNX输出形状验证:", outputs[0].shape)
    assert outputs[0].shape == (1, 12, 3), "ONNX模型输出形状不符合要求"

    # 数值验证（使用相同输入）
    dummy_input = torch.tensor(input_data)
    with torch.no_grad():
        pt_output_new = model(dummy_input)[0].numpy()

    onnx_output = outputs[0]

    # 计算差异
    diff = np.abs(pt_output_new - onnx_output).max()
    print(f"最大数值差异: {diff:.6f}")
    assert diff < 1e-4, "数值差异超过阈值"

    print("验证成功！输出形状和数值均符合预期")


if __name__ == "__main__":
    # 完整流程
    model_pth = prepare_model()
    # onnx_path = export_onnx(model_pth)
    # validate_onnx(onnx_path)
# 4. 转换为IR模型 ------------------------------------------
    import openvino as ov
    from pathlib import Path
    import numpy as np

    # 确保输出目录存在
    output_dir = Path("ir_pred_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用新版API转换模型
    ov_model = ov.convert_model("pred_model.onnx")

    # 序列化为IR格式
    ov.save_model(ov_model, output_dir / "pred_model.xml",compress_to_fp16=True)

    print(f"IR模型已保存至: {output_dir.resolve()}")

# 5. 验证IR模型 ------------------------------------------
    import numpy as np
    from openvino.runtime import Core

    # 初始化OpenVINO核心
    ie = Core()

    # 读取IR模型
    model = ie.read_model(model='ir_pred_model/pred_model.xml')
    compiled_model = ie.compile_model(model=model, device_name='CPU')

    # 获取输入输出信息
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    print("11111:",output_layer)
    # 生成测试数据
    input_data = np.random.randn(1, 10, 4).astype(np.float32)

    # 执行推理
    result = compiled_model([input_data])[output_layer]

    # 验证输出形状和数值
    print("OpenVINO输出形状:", result[0].shape)
    assert result[0].shape == (12, 3), "输出形状错误"


    dummy_input = torch.tensor(input_data)
    with torch.no_grad():
        pt_output = model_pth(dummy_input)
    # 对比原始PyTorch输出
    diff = np.abs(pt_output[0] - result[0]).max()
    print(f"与PyTorch的最大差异: {diff:.6f}")
    assert diff < 1e-1, "数值差异过大"