import torch
from models.PIMNet import PIMNet # Assuming PIMNet is in the models directory
from dataload.data_util import Config, Data
# --- 配置区域 ---
# 1. 初始化你的模型结构
#    这里的参数需要和你训练时使用的模型结构参数一致
model_pth = "pths/polyp/epoch_30.pth"
test_paths = "/mnt/d/BaiduDownload/data/UltraEdit"
cfg = Config(datapath='/mnt/d/BaiduDownload/data/UltraEdit', # Make sure this path is correct
                savepath='./pths/', mode='train',
                batch=8, lr=1e-4, momen=0.9, decay=5e-4, epoch=30, lr_decay_gamma=0.1)

model = PIMNet(cfg)

checkpoint = torch.load(model_pth, weights_only=False,map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

# 3. 设置为评估模式
model.eval()

# 4. 创建一个符合模型输入的虚拟输入 (dummy input)
#    你需要知道你的模型输入尺寸，例如 (batch_size, channels, height, width)
#    假设输入是 3x224x224 的图像
dummy_input = torch.randn(1, 3, 224, 224) 

# 5. 定义输出的ONNX文件名
onnx_output_path = "PIMNet.onnx"

# --- 执行导出 ---
print("开始导出到 ONNX...")
torch.onnx.export(model,               # 你实例化的模型
                  dummy_input,         # 虚拟输入
                  onnx_output_path,    # 输出文件名
                  export_params=True,
                  opset_version=12,    # 一个常用的版本
                  do_constant_folding=True,
                  input_names = ['input'],   # 输入名
                  output_names = ['output'], # 输出名
                  dynamic_axes={'input' : {0 : 'batch_size'}, # 动态轴，允许不同的批次大小
                                'output' : {0 : 'batch_size'}})
print(f"模型已成功导出到 {onnx_output_path}")