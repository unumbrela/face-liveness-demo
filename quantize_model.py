import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# --- 配置区域 ---
# 1. 指定原始的、未量化的ONNX模型路径
model_fp32 = './model.onnx'

# 2. 指定量化后模型的输出路径
model_quant = './model.quant.onnx'

# --- 量化代码 ---
def quantize_onnx_model():
    print(f"开始量化模型: {model_fp32}")
    
    # 执行动态量化
    # quantize_dynamic 会将权重转换为INT8，并在运行时动态量化激活值
    # 这是一种无需校准数据集的便捷方法
    quantize_dynamic(
        model_input=model_fp32,      # 输入模型路径
        model_output=model_quant,    # 输出模型路径
        weight_type=QuantType.QInt8  # 将权重转换为INT8
    )

    print(f"模型量化完成！已保存到: {model_quant}")
    print("现在请检查原始模型和量化后模型的大小。")

if __name__ == '__main__':
    quantize_onnx_model()