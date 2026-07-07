import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

# ---------- 新的可直接脚本化的模型 ----------
class ScriptableGRUModel(nn.Module):
    """
    该模型接收已归一化的填充序列和长度，内部完成打包、GRU 和分类。
    """
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, num_classes=8, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, padded_input, lengths):
        # padded_input: (batch, max_len, 2), lengths: (batch,)
        packed = pack_padded_sequence(padded_input, lengths, batch_first=True, enforce_sorted=True)
        _, hn = self.gru(packed)
        last_hidden = hn[-1]                     # 取最后一层的隐藏状态
        out = self.fc(last_hidden)
        return out

# ---------- 加载原始权重并迁移到新模型 ----------
# 原始训练时用的类（与训练代码一致，仅用于加载权重）
class OriginalGRU(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, num_classes=8, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, packed_input):
        _, hn = self.gru(packed_input)
        return self.fc(hn[-1])

# 1. 用原始类加载权重
device = 'cpu'
original_model = OriginalGRU(num_classes=8)          # 请确认 num_classes 与实际一致
state_dict = torch.load('road_shape_gru.pth', map_location=device)
original_model.load_state_dict(state_dict)
original_model.eval()

# 2. 创建可脚本化模型并复制权重
scriptable_model = ScriptableGRUModel(num_classes=8)
scriptable_model.load_state_dict(original_model.state_dict())   # 层名完全相同，可直接复制
scriptable_model.eval()

# 3. 使用 torch.jit.script 导出（没有包装调用，不会出现类型推断错误）
scripted_model = torch.jit.script(scriptable_model)
scripted_model.save('road_shape_gru.pt')

print("✅ 成功导出为 road_shape_gru.pt")