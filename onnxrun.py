import numpy as np
import onnxruntime as ort
import json

# -------------------- 配置 --------------------
ONNX_MODEL_PATH = "road_shape_gru.onnx"
IMG_WIDTH = 320.0
IMG_HEIGHT = 240.0

# 标签映射（与训练时一致）
LABEL_MAP = {
    0: "zhixian",
    1: "shizi",
    2: "daodazuohuandao",
    3: "jinruzuohuandao",
    4: "likaizuohuandao",
    5: "daodayouhuandao",
    6: "jinruyouhuandao",
    7: "likaiyouhuandao"
}
NUM_CLASSES = len(LABEL_MAP)

# -------------------- 推理引擎 --------------------
class RoadShapeInference:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)

    def preprocess(self, sequences):
        """
        预处理多个变长序列，返回模型所需输入。
        参数:
            sequences: list of np.ndarray, 每个形状 (L_i, 2)，原始坐标（未归一化）
        返回:
            padded_input: np.ndarray, 形状 (batch, max_len, 2)
            lengths: np.ndarray, 形状 (batch,), int64
            sort_idx: np.ndarray, 用于还原顺序的索引（按长度降序排列）
        """
        # 1. 归一化
        norm_seqs = []
        for seq in sequences:
            seq = np.array(seq, dtype=np.float32).copy()
            seq[:, 0] /= IMG_WIDTH
            seq[:, 1] /= IMG_HEIGHT
            norm_seqs.append(seq)

        # 2. 按长度降序排列
        lengths = np.array([s.shape[0] for s in norm_seqs])
        sort_idx = np.argsort(-lengths)          # 降序索引
        lengths = lengths[sort_idx]
        sorted_seqs = [norm_seqs[i] for i in sort_idx]

        # 3. 填充到相同长度
        max_len = lengths[0]
        batch = len(sorted_seqs)
        padded_input = np.zeros((batch, max_len, 2), dtype=np.float32)
        for i, seq in enumerate(sorted_seqs):
            padded_input[i, :seq.shape[0], :] = seq

        return padded_input, lengths.astype(np.int64), sort_idx

    def predict(self, sequences):
        """
        对多个序列进行类别预测。
        参数:
            sequences: 同 preprocess
        返回:
            pred_classes: list of int, 每个序列的预测类别
            probs: np.ndarray, 形状 (batch, num_classes)，预测概率
        """
        # 预处理
        padded_input, lengths, sort_idx = self.preprocess(sequences)

        # ONNX 推理
        ort_inputs = {
            "padded_input": padded_input,
            "lengths": lengths
        }
        logits = self.session.run(["logits"], ort_inputs)[0]  # (batch, num_classes)

        # Softmax 得到概率
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        # 还原到原始输入顺序
        inv_sort_idx = np.argsort(sort_idx)
        probs = probs[inv_sort_idx]
        pred_classes = np.argmax(probs, axis=1).tolist()

        return pred_classes, probs

    def predict_single(self, sequence):
        """单条序列预测，返回类别和概率"""
        preds, probs = self.predict([sequence])
        return preds[0], probs[0]

# -------------------- 示例：从 JSON 文件读取并推理 --------------------
def load_sequences_from_json(json_path):
    """加载 JSON 文件中的点序列。格式示例：[{"point": [[x1,y1],...], "label": "zhixian"}, ...]"""
    with open(json_path, "r") as f:
        data = json.load(f)
    sequences = [np.array(d["point"], dtype=np.float32) for d in data]
    labels = [d.get("label", None) for d in data]   # 如果没有标签则为 None
    return sequences, labels

if __name__ == "__main__":
    # 初始化推理器
    inferencer = RoadShapeInference(ONNX_MODEL_PATH)

    # 方式1：直接使用几个手工构建的序列测试
    # # 模拟一个直道（水平线）
    # xs = np.linspace(50, 200, 10)
    # ys = np.full_like(xs, 100) + np.random.randn(10) * 2
    # seq_straight = np.stack([xs, ys], axis=1)
    #
    # # 模拟一个弯道（圆弧）
    # t = np.linspace(0, np.pi/2, 12)
    # seq_curve = np.stack([80 + 60*np.cos(t), 120 + 60*np.sin(t)], axis=1)
    #
    # # 单条预测
    # cls_id, prob = inferencer.predict_single(seq_straight)
    # print(f"直道序列预测类别: {LABEL_MAP[cls_id]}, 概率: {prob}")
    #
    # cls_id, prob = inferencer.predict_single(seq_curve)
    # print(f"弯道序列预测类别: {LABEL_MAP[cls_id]}, 概率: {prob}")
    #
    # # 批量预测
    # batch = [seq_straight, seq_curve]
    # pred_classes, probs = inferencer.predict(batch)
    # for i, (cls_id, prob) in enumerate(zip(pred_classes, probs)):
    #     print(f"批次序列{i}: 预测类别 {LABEL_MAP[cls_id]}, 概率 {prob}")

    # 方式2：从 JSON 文件加载真实数据推理（如果有）
    json_file = "byd_2026_5_22.json"   # 替换为实际路径
    sequences, true_labels = load_sequences_from_json(json_file)
    pred_classes, probs = inferencer.predict(sequences)
    for i, cls_id in enumerate(pred_classes):
        print(f"样本{i}: 预测 {LABEL_MAP[cls_id]}, 真实 {true_labels[i] if true_labels[i] else 'N/A'}")