import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from torch.nn.utils.rnn import pack_padded_sequence

# ========== 参数 ==========
input_dim = 2
hidden_dim = 64
num_layers = 2
num_classes = 8
dropout = 0.1
seq_len = 100          # 固定序列长度（根据你数据设定）
batch_size = 1         # TFLM 推荐 batch=1

# ========== 原始模型（仅用于加载权重）==========
class OriginalGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, packed_input):
        _, hn = self.gru(packed_input)
        return self.fc(hn[-1])

# ========== 导出用静态模型（无打包，直接处理填充序列）==========
class ExportGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.gru(x)          # out: (batch, seq_len, hidden_dim)
        out = out[:, -1, :]           # 强行取最后一个时间步
        return self.fc(out)

# ---------- 1. 加载原始权重 ----------
device = 'cpu'
original_model = OriginalGRU(input_dim, hidden_dim, num_layers, num_classes, dropout)
state_dict = torch.load('road_shape_gru.pth', map_location=device)
original_model.load_state_dict(state_dict)
original_model.eval()

# ---------- 2. 复制权重到导出模型 ----------
export_model = ExportGRU(input_dim, hidden_dim, num_layers, num_classes, dropout)
export_model.load_state_dict(original_model.state_dict())
export_model.eval()

# ---------- 3. 导出 ONNX（固定输入形状）----------
dummy_input = torch.randn(batch_size, seq_len, input_dim)
onnx_path = "model.onnx"
torch.onnx.export(
    export_model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    opset_version=11,          # 建议 ≥11
    do_constant_folding=True,
    export_params=True,
    dynamic_axes=None          # 完全固定 shape
)
print(f"✅ ONNX 模型已导出至 {onnx_path}")

# ---------- 4. ONNX → TensorFlow SavedModel ----------
# 需要 pip install onnx onnx-tf
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load(onnx_path)
tf_rep = prepare(onnx_model)          # 生成 TensorFlow 计算图
tf_saved_model_path = "model_tf"
tf_rep.export_graph(tf_saved_model_path)
print(f"✅ TensorFlow SavedModel 已保存至 {tf_saved_model_path}")

# ---------- 5. 转换为 TFLite（仅内置算子，推荐整数量化）----------
converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_path)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# 量化配置（可选，但 TFLM 上强烈推荐）
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# 如果有代表性数据可放开下方注释，以改善量化精度
# def representative_dataset():
#     for _ in range(100):
#         data = np.random.randn(1, seq_len, input_dim).astype(np.float32)
#         yield [data]
# converter.representative_dataset = representative_dataset

tflite_model = converter.convert()
tflite_path = "model.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"✅ TFLite 模型已保存至 {tflite_path}")

# ---------- 6. 验证 TFLite 推理（与 PyTorch 导出模型对比）----------
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test_input = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]['index'])

with torch.no_grad():
    pt_output = export_model(torch.from_numpy(test_input)).numpy()

print("TFLite 输出:", tflite_output)
print("PyTorch 输出:", pt_output)
print("最大绝对误差:", np.max(np.abs(tflite_output - pt_output)))