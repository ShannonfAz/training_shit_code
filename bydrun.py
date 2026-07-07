import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence

# ---------- 模型定义（必须与训练时完全相同）----------
class GRUPointClassifier(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=512, num_layers=2, num_classes=8, dropout=0.1):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, packed_input):
        _, hn = self.gru(packed_input)
        last_hidden = hn[-1]          # (batch, hidden_dim)
        return self.fc(last_hidden)

# ---------- 加载模型 ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUPointClassifier(num_classes=8)    # 根据实际类别数修改
model.load_state_dict(torch.load('road_shape_gru.pth', map_location=device))
model.to(device)
model.eval()

# ---------- 标签名称映射（示例，请与训练时一致）----------
idx_to_label = {
    0: "zhixian",
    1: "shizi",
    2: "daodazuohuandao",
    3: "jinruzuohuandao",
    4: "likaizuohuandao",
    5: "daodayouhuandao",
    6: "jinruyouhuandao",
    7: "likaiyouhuandao"
}

# ---------- 推理函数（返回所有类别概率）----------
def predict_all_probs(points, img_width=320.0, img_height=240.0):
    """
    points: np.array 或 list，形状 (L, 2)，如 [[114, 51], [41, 91], ...]
    返回：概率列表（长度为类别数）和预测的类别索引
    """
    # 转换为张量并归一化
    if not isinstance(points, torch.Tensor):
        pts = torch.tensor(points, dtype=torch.float32)
    else:
        pts = points.float()
    pts[:, 0] /= img_width
    pts[:, 1] /= img_height
    pts = pts.unsqueeze(0)   # (1, L, 2)

    # 构造 PackedSequence
    lengths = torch.tensor([pts.size(1)], dtype=torch.long)
    packed_input = pack_padded_sequence(pts, lengths, batch_first=True, enforce_sorted=True)
    packed_input = packed_input.to(device)

    # 前向传播
    with torch.no_grad():
        logits = model(packed_input)           # (1, num_classes)
        probs = torch.softmax(logits, dim=1)   # 概率分布
        probs = probs.squeeze(0).cpu().numpy() # 转为 numpy 数组
        pred_idx = int(np.argmax(probs))

    return probs, pred_idx

# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 模拟一组新点集（请替换为实际数据）
    new_points = np.array([
        [
            126,
            238
        ],
        [
            125,
            150
        ],
        [
            77,
            161
        ],
        [
            19,
            62
        ],
        [
            37,
            53
        ],
        [
            56,
            87
        ],
        [
            91,
            87
        ],
        [
            120,
            62
        ],
        [
            124,
            22
        ],
        [
            155,
            8
        ],
        [
            205,
            28
        ],
        [
            205,
            237
        ]
    ])

    probs, pred_id = predict_all_probs(new_points)

    print("各类别概率：")
    for idx, prob in enumerate(probs):
        class_name = idx_to_label.get(idx, f"class_{idx}")
        print(f"  {class_name:12s}: {prob:.4f}")

    print(f"\n预测类别: {idx_to_label[pred_id]} (索引 {pred_id})")