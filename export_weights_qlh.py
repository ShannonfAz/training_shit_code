"""
export_weights.py
将训练好的 GRU 模型权重（input_dim=3, hidden_dim=8）导出为 C 头文件。
运行后生成 gru_weights.h，供嵌入式推理使用。
"""

import torch
import numpy as np
from bydrun64_qlh import GRUFixedLenClassifier

# 加载新模型（参数必须与训练完全一致）
model = GRUFixedLenClassifier(
    input_dim=3, hidden_dim=8, num_layers=1, num_classes=6, dropout=0.0
)
model.load_state_dict(torch.load('road_shape_gru_64.pth', map_location='cpu'))
model.eval()

gru = model.gru

# 提取权重（形状已自动适配新维度）
W_ih = gru.weight_ih_l0.detach().numpy()   # (3*8, 3) = (24, 3)
W_hh = gru.weight_hh_l0.detach().numpy()   # (3*8, 8) = (24, 8)
b_ih = gru.bias_ih_l0.detach().numpy()     # (24,)
b_hh = gru.bias_hh_l0.detach().numpy()     # (24,)

fc_w = model.fc.weight.detach().numpy()    # (6, 8)
fc_b = model.fc.bias.detach().numpy()      # (6,)

ln_w = model.ln.weight.detach().numpy()    # (8,)
ln_b = model.ln.bias.detach().numpy()      # (8,)

# 将所有权重展平为 C 数组
def to_c_array(arr, name):
    flat = arr.flatten()
    return f"const float {name}[{len(flat)}] = {{\n    " + ", ".join(f"{v:.8f}f" for v in flat) + "\n};"

with open("gru_weights.h", "w") as f:
    f.write("// GRU 模型权重（input_dim=3, hidden_dim=8）\n")
    f.write("// 自动生成，不要手动修改\n")
    f.write("#pragma once\n\n")
    f.write(to_c_array(W_ih, "W_ih"))       # 24x3 = 72
    f.write("\n")
    f.write(to_c_array(W_hh, "W_hh"))       # 24x8 = 192
    f.write("\n")
    f.write(to_c_array(b_ih, "b_ih"))       # 24
    f.write("\n")
    f.write(to_c_array(b_hh, "b_hh"))       # 24
    f.write("\n")
    f.write(to_c_array(fc_w, "fc_weight"))  # 6x8 = 48
    f.write("\n")
    f.write(to_c_array(fc_b, "fc_bias"))    # 6
    f.write("\n")
    f.write(to_c_array(ln_w, "ln_gamma"))   # 8
    f.write("\n")
    f.write(to_c_array(ln_b, "ln_beta"))    # 8

print("✅ gru_weights.h 生成成功（总参数量约", 72+192+24+24+48+6+8+8, "个 float）")