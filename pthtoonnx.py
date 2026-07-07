import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import onnx
import onnxruntime

# -------------------- 1. 导出专用模型（与之前相同） --------------------
class GRUPointClassifierExport(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, num_classes=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, padded_input, lengths):
        # padded_input: (batch, max_len, 2)  已按长度降序排列的填充序列
        # lengths: (batch,) 真实长度，CPU int64
        lengths_cpu = lengths.cpu()
        packed = pack_padded_sequence(padded_input, lengths_cpu,
                                      batch_first=True, enforce_sorted=True)
        packed_output, hn = self.gru(packed)
        last_hidden = hn[-1]          # (batch, hidden_dim)
        out = self.fc(last_hidden)    # (batch, num_classes)
        return out


# -------------------- 2. 加载训练权重 --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 8   # 与训练时一致

model_export = GRUPointClassifierExport(input_dim=2, hidden_dim=64,
                                        num_layers=2, num_classes=num_classes,
                                        dropout=0.1)
state_dict = torch.load('road_shape_gru.pth', map_location='cpu')
model_export.load_state_dict(state_dict)
model_export.eval()

# 导出时建议直接使用 CPU，避免 CUDA 带来的额外复杂性
model_export.cpu()


# -------------------- 3. 准备示例输入并导出 ONNX（修正点） --------------------
batch_size = 2
max_len = 5
padded_input = torch.randn(batch_size, max_len, 2)      # CPU 张量
lengths = torch.tensor([5, 3], dtype=torch.int64)       # 必须降序，CPU

onnx_path = "road_shape_gru.onnx"
torch.onnx.export(
    model_export,
    (padded_input, lengths),
    onnx_path,
    input_names=["padded_input", "lengths"],
    output_names=["logits"],
    dynamic_axes={
        "padded_input": {0: "batch", 1: "seq_len"},
        "lengths": {0: "batch"},
        "logits": {0: "batch"}
    },
    opset_version=11,               # 11 完全支持 PackedSequence
    dynamo=False,                   # ★ 关键：禁用 dynamo，使用传统 TorchScript 导出
    do_constant_folding=True,
)

print(f"ONNX model saved to {onnx_path}")

# 验证 ONNX 模型结构
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")


# -------------------- 4. 推理验证 --------------------
# 准备测试数据（3条序列，长度 4,6,2 -> 降序为 6,4,2）
seq1 = np.random.randn(4, 2).astype(np.float32)
seq2 = np.random.randn(6, 2).astype(np.float32)
seq3 = np.random.randn(2, 2).astype(np.float32)
sequences = [seq2, seq1, seq3]               # 已降序
lengths_val = np.array([6, 4, 2], dtype=np.int64)
max_len_val = 6
padded_val = np.zeros((3, max_len_val, 2), dtype=np.float32)
for i, seq in enumerate(sequences):
    padded_val[i, :len(seq)] = seq

# PyTorch 推理
with torch.no_grad():
    pt_out = model_export(torch.from_numpy(padded_val),
                          torch.from_numpy(lengths_val)).numpy()

# ONNX Runtime 推理
ort_session = onnxruntime.InferenceSession(onnx_path)
ort_out = ort_session.run(None, {"padded_input": padded_val,
                                 "lengths": lengths_val})[0]

diff = np.abs(pt_out - ort_out).max()
print(f"Max difference between PyTorch and ONNX Runtime: {diff:.6f}")
assert diff < 1e-4, "Mismatch detected!"
print("Verification passed!")