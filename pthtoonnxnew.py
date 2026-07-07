"""
export_onnx.py
将训练好的 PyTorch GRU 模型导出为 ONNX 格式。

使用前请确保：
    pip install torch onnx
"""

import torch
import numpy as np

# ==================== 1. 模型定义（必须与训练时完全一致） ====================
class GRUFixedLenClassifier(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, num_layers=1, num_classes=6, dropout=0.0):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.ln = torch.nn.LayerNorm(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, 64, 5)
        out, _ = self.gru(x)          # (batch, 64, hidden_dim)
        out = self.dropout(out)       # 推理时若 dropout=0 则无影响
        feat = out.mean(dim=1)        # 平均池化
        feat = self.ln(feat)
        logits = self.fc(feat)
        return logits


# ==================== 2. 加载训练好的权重 ====================
model = GRUFixedLenClassifier(input_dim=5, hidden_dim=16, num_layers=1, num_classes=6, dropout=0.0)
model.load_state_dict(torch.load('road_shape_gru_64.pth', map_location='cpu'))
model.eval()

# ==================== 3. 创建虚拟输入 ====================
# 输入形状必须与模型期望一致: (batch_size, seq_len, features) = (1, 64, 5)
dummy_input = torch.randn(1, 64, 5)

# ==================== 4. 导出 ONNX ====================
onnx_path = "road_shape_gru.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch'},   # 允许变 batch 大小
        'output': {0: 'batch'}
    },
    opset_version=11,           # opset 11 对循环网络支持较好
    verbose=False
)

print(f"ONNX 模型已保存至: {onnx_path}")

# ==================== 5. 验证（可选） ====================
import onnx
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX 模型验证通过！")