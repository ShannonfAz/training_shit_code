"""
export_lstm_weights.py
将 LSTM 冠军模型（input_dim=4, hidden_dim=8）的权重 + 特征统计量导出为 C 头文件。
运行后生成 lstm_weights.h，供 lstmrun.cpp 使用。
用法: ~/pythonvenv/bin/python export_lstm_weights.py  （在 lstm/ 目录下运行）
"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # lstm/

# 加载 LSTM 模型（参数必须与 train_lstm.py 完全一致）
from train_lstm import LSTMFixedLenClassifier

model = LSTMFixedLenClassifier(input_dim=4, hidden_dim=8, num_layers=1, num_classes=6, dropout=0.0)
model.load_state_dict(torch.load('road_shape_lstm.pth', map_location='cpu'))
model.eval()

lstm = model.lstm
W_ih = lstm.weight_ih_l0.detach().numpy()   # (4*8, 4) = (32, 4)，门顺序 [i, f, g, o]
W_hh = lstm.weight_hh_l0.detach().numpy()   # (4*8, 8) = (32, 8)
b_ih = lstm.bias_ih_l0.detach().numpy()     # (32,)
b_hh = lstm.bias_hh_l0.detach().numpy()     # (32,)
fc_w = model.fc.weight.detach().numpy()     # (6, 8)
fc_b = model.fc.bias.detach().numpy()       # (6,)
ln_w = model.ln.weight.detach().numpy()     # (8,)
ln_b = model.ln.bias.detach().numpy()       # (8,)

# 特征统计量（4 维）
stats = np.load('feat_stats_lstm.npz')
feat_mean = stats['mean']   # (4,)
feat_std = stats['std']     # (4,)

def to_c_array(arr, name):
    flat = arr.flatten()
    return f"const float {name}[{len(flat)}] = {{\n    " + ", ".join(f"{v:.8f}f" for v in flat) + "\n};"

with open("cpp/lstm_weights.h", "w") as f:
    f.write("// LSTM 模型权重（input_dim=4, hidden_dim=8, 门顺序 [i,f,g,o]）+ 特征统计量\n")
    f.write("// 由 export_lstm_weights.py 自动生成，不要手动修改\n")
    f.write("#pragma once\n\n")
    f.write(to_c_array(W_ih, "W_ih"))
    f.write("\n")
    f.write(to_c_array(W_hh, "W_hh"))
    f.write("\n")
    f.write(to_c_array(b_ih, "b_ih"))
    f.write("\n")
    f.write(to_c_array(b_hh, "b_hh"))
    f.write("\n")
    f.write(to_c_array(fc_w, "fc_weight"))
    f.write("\n")
    f.write(to_c_array(fc_b, "fc_bias"))
    f.write("\n")
    f.write(to_c_array(ln_w, "ln_gamma"))
    f.write("\n")
    f.write(to_c_array(ln_b, "ln_beta"))
    f.write("\n")
    f.write(to_c_array(feat_mean, "FEAT_MEAN"))
    f.write("\n")
    f.write(to_c_array(feat_std, "FEAT_STD"))

n = W_ih.size + W_hh.size + b_ih.size + b_hh.size + fc_w.size + fc_b.size + ln_w.size + ln_b.size
print(f"✅ cpp/lstm_weights.h 生成成功（模型参数 {n} 个 float + 特征统计量 8 个 float）")
