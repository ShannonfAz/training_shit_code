"""
export_weights.py
将训练好的 GRU 模型权重导出为 C 头文件。
运行后生成 gru_weights.h，供嵌入式推理使用。
"""

import torch
import numpy as np
from bydrun64 import GRUFixedLenClassifier

# 加载模型
model = GRUFixedLenClassifier(input_dim=5, hidden_dim=16, num_layers=1, num_classes=6, dropout=0.0)
model.load_state_dict(torch.load('road_shape_gru_64.pth', map_location='cpu'))
model.eval()

gru = model.gru

# 提取并转置权重，以匹配 C 代码中列优先的乘法
W_ih = gru.weight_ih_l0.detach().numpy()   # (48, 5)
W_hh = gru.weight_hh_l0.detach().numpy()   # (48, 16)
b_ih = gru.bias_ih_l0.detach().numpy()     # (48,)
b_hh = gru.bias_hh_l0.detach().numpy()     # (48,)

fc_w = model.fc.weight.detach().numpy()    # (6, 16)
fc_b = model.fc.bias.detach().numpy()      # (6,)

ln_w = model.ln.weight.detach().numpy()    # (16,)
ln_b = model.ln.bias.detach().numpy()      # (16,)

# 将所有权重展平为 C 数组
def to_c_array(arr, name):
    flat = arr.flatten()
    return f"const float {name}[{len(flat)}] = {{\n    " + ", ".join(f"{v:.8f}f" for v in flat) + "\n};"

with open("gru_weights.h", "w") as f:
    f.write("// GRU 模型权重（自动生成，不要手动修改）\n")
    f.write("#pragma once\n\n")
    f.write(to_c_array(W_ih, "W_ih"))       # 48x5 = 240
    f.write("\n")
    f.write(to_c_array(W_hh, "W_hh"))       # 48x16 = 768
    f.write("\n")
    f.write(to_c_array(b_ih, "b_ih"))       # 48
    f.write("\n")
    f.write(to_c_array(b_hh, "b_hh"))       # 48
    f.write("\n")
    f.write(to_c_array(fc_w, "fc_weight"))  # 6x16 = 96
    f.write("\n")
    f.write(to_c_array(fc_b, "fc_bias"))    # 6
    f.write("\n")
    f.write(to_c_array(ln_w, "ln_gamma"))   # 16
    f.write("\n")
    f.write(to_c_array(ln_b, "ln_beta"))    # 16

print("✅ gru_weights.h 生成成功")