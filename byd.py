import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

# -------------------- 1. 数据集定义 --------------------
class RoadShapeDataset(Dataset):
    """
    道路形状点集数据集。
    每条数据: 点序列 (seq_len, 2), 标签 (int)
    """
    def __init__(self, point_sequences, labels):
        self.sequences = point_sequences
        self.labels = labels
        self.img_width = 320.0
        self.img_height = 240.0

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.as_tensor(self.sequences[idx], dtype=torch.float32)
        seq[:, 0] /= self.img_width
        seq[:, 1] /= self.img_height
        label = torch.as_tensor(self.labels[idx], dtype=torch.long)
        return seq, label

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in sequences])
    lengths, perm_idx = lengths.sort(0, descending=True)
    sequences = [sequences[i] for i in perm_idx]
    labels = torch.stack([labels[i] for i in perm_idx])
    padded_seq = pad_sequence(sequences, batch_first=True)
    packed_seq = pack_padded_sequence(padded_seq, lengths, batch_first=True, enforce_sorted=True)
    return packed_seq, labels

# -------------------- 2. GRU 分类模型 --------------------
class GRUPointClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, num_classes=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, packed_input):
        packed_output, hn = self.gru(packed_input)
        last_hidden = hn[-1]
        out = self.fc(last_hidden)
        return out

# -------------------- 3. 全量训练 --------------------
def train_full(model, data_loader, epochs=30, lr=1e-3, device='cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for packed_input, labels in data_loader:
            packed_input = packed_input.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(packed_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)

        scheduler.step()
        avg_loss = total_loss / len(data_loader.dataset)
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_loss:.4f}")

    return model

# -------------------- 4. 主流程 --------------------
def main():
    # 加载全部数据（不划分）
    with open("byd_2026_5_22.json", "r") as f:
        raw_data = json.load(f)

    label_map = {
        "zhixian": 0, "shizi": 1, "daodazuohuandao": 2, "jinruzuohuandao": 3,
        "likaizuohuandao": 4, "daodayouhuandao": 5, "jinruyouhuandao": 6, "likaiyouhuandao": 7
    }
    sequences = [np.array(d["point"], dtype=np.float32) for d in raw_data]
    labels = [label_map[d["label"]] for d in raw_data]

    # 全量数据集
    full_dataset = RoadShapeDataset(sequences, labels)
    full_loader = DataLoader(full_dataset, batch_size=286, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUPointClassifier(input_dim=2, hidden_dim=512, num_layers=2, num_classes=8, dropout=0.1)

    print("全量训练开始...")
    trained_model = train_full(model, full_loader, epochs=100, lr=1e-3, device=device)

    # 保存模型
    torch.save(trained_model.state_dict(), "road_shape_gru.pth")
    print("全量训练完成，模型已保存为 road_shape_gru.pth")

if __name__ == "__main__":
    main()