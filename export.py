import torch
import torch.nn as nn
import block
import onnxruntime as ort
import numpy as np
import aim_cmd

config = aim_cmd.default_config

# 1. 准备模型 ----------------------------------------------
def prepare_model():
    # 初始化模型
    model = block.DualBranchTimeSeriesPredictor(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        input_dim=config['feature_dim'],
        seq_len=config['input_size'],
        pred_len=config['output_size'],
    )
    # 加载预训练权重
    checkpoint = torch.load(
        config['total_transformer_save_path'],
        map_location='cpu'
    )
    model.load_state_dict(checkpoint)
    model.eval()  # 切换为推理模式

    # 打印模型结构验证
    # print("模型结构:")
    # print(model)

    # 验证PyTorch模型输出形状
    dummy_input = torch.randn(1, config['input_size'], config['feature_dim'])
    with torch.no_grad():
        output = model(dummy_input)

    # 保存PyTorch输出供后续验证
    pt_output = output[0].numpy()

    print("\nPyTorch输出形状验证:", output[0].shape)
    assert output[0].shape == (1, config['output_size'],3), "PyTorch模型输出形状不符合要求"

    return model


# 2. 导出ONNX模型 ------------------------------------------
def export_onnx(model):
    # 创建虚拟输入
    dummy_input = torch.randn(1, config['input_size'], config['feature_dim'])

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
def validate_onnx(onnx_path,model):
    # 创建推理会话
    ort_session = ort.InferenceSession(onnx_path)

    # 使用与PyTorch相同的输入数据
    input_data = np.random.randn(1, config['input_size'], config['feature_dim']).astype(np.float32)

    # 执行推理
    outputs = ort_session.run(
        output_names=None,
        input_feed={"input": input_data}
    )

    # 形状验证
    print("\nONNX输出形状验证:", outputs[0].shape)
    assert outputs[0].shape == (1, config['output_size'],3), "ONNX模型输出形状不符合要求"

    # 数值验证（使用相同输入）
    dummy_input = torch.tensor(input_data)
    with torch.no_grad():
        pt_output_new = model(dummy_input)[0].numpy()

    onnx_output = outputs[0]

    # 计算差异
    diff = np.abs(pt_output_new - onnx_output).max()
    print(f"最大数值差异: {diff:.6f}")
    assert diff < 1e-3, "数值差异超过阈值"

    print("验证成功！输出形状和数值均符合预期")

def export_init():
    model_pth = prepare_model()
    onnx_path = export_onnx(model_pth)
    validate_onnx(onnx_path, model_pth)
    # 4. 转换为IR模型 ------------------------------------------
    import openvino as ov
    from pathlib import Path
    import numpy as np

    # 确保输出目录存在
    output_dir = Path("ir_pred_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用新版API转换模型
    ov_model = ov.convert_model("pred_model.onnx")
    print("onnx模型")
    # 序列化为IR格式
    ov.save_model(ov_model, output_dir / "pred_model.xml", compress_to_fp16=False)

    print(f"IR模型已保存至: {output_dir.resolve()}")

    # 5. 验证IR模型 ------------------------------------------
    import numpy as np
    from openvino.runtime import Core

    # 初始化OpenVINO核心
    ie = Core()

    # 读取IR模型
    ir_model = ie.read_model(model='ir_pred_model/pred_model.xml')
    compiled_model = ie.compile_model(model=ir_model, device_name='CPU')

    # 获取输入输出信息
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    print("11111:", output_layer)
    # 生成测试数据
    input_data = np.random.randn(1, config['input_size'], config['feature_dim']).astype(np.float32)

    # 执行推理
    result = compiled_model([input_data])[output_layer]

    # 验证输出形状和数值
    print("OpenVINO输出形状:", result[0].shape)
    assert result[0].shape == (config['output_size'], 3), "输出形状错误"

    dummy_input = torch.tensor(input_data)
    with torch.no_grad():
        pt_output = model_pth(dummy_input)
    # 对比原始PyTorch输出
    diff = np.abs(pt_output[0] - result[0]).max()
    print(f"与PyTorch的最大差异: {diff:.6f}")
    assert diff < 1e-1, "数值差异过大"
if __name__ == "__main__":
    export_init()