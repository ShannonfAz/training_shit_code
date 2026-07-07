"""
build_tflite_direct.py
使用 TensorFlow 基础算子重建 GRU 模型（固定 batch=1，仅用安全算子），
加载 PyTorch 权重并导出为 TFLM 兼容的 TFLite 文件。

要求：
    pip install tensorflow torch numpy
    Python 3.11 或 3.13 均可，TensorFlow 版本建议 2.12+
"""

import numpy as np
import tensorflow as tf
import torch

# ========== 根据你的项目修改导入 ==========
from bydrun64 import GRUFixedLenClassifier  # 你的训练模型定义文件

# ==================== 1. 从 PyTorch 加载训练好的权重 ====================
pytorch_model = GRUFixedLenClassifier(
    input_dim=5, hidden_dim=16, num_layers=1, num_classes=6, dropout=0.0
)
pytorch_model.load_state_dict(torch.load('road_shape_gru_64.pth', map_location='cpu'))
pytorch_model.eval()

# 提取各层权重（PyTorch -> TensorFlow 格式）
gru = pytorch_model.gru
weight_ih = gru.weight_ih_l0.detach().numpy().T   # (input, 3*hidden) = (5, 48)
weight_hh = gru.weight_hh_l0.detach().numpy().T   # (hidden, 3*hidden) = (16, 48)
bias_ih   = gru.bias_ih_l0.detach().numpy()       # (48,)
bias_hh   = gru.bias_hh_l0.detach().numpy()       # (48,)

fc_weight = pytorch_model.fc.weight.detach().numpy().T  # (hidden, num_classes) = (16, 6)
fc_bias   = pytorch_model.fc.bias.detach().numpy()      # (6,)

ln_gamma = pytorch_model.ln.weight.detach().numpy()     # (16,)
ln_beta  = pytorch_model.ln.bias.detach().numpy()       # (16,)


# ==================== 2. 用安全算子搭建 TensorFlow 模型 ====================
class TfGRUFromScratch(tf.keras.Model):
    def __init__(self, hidden_dim=16, num_classes=6):
        super().__init__()
        self.hidden_dim = hidden_dim

    def call(self, x):
        # x: (1, 64, 5)  固定 batch=1，避免动态形状
        batch_size = 1   # 固定为 1，嵌入式单样本推理

        # 初始化隐藏状态，静态形状，不产生动态 FILL 节点
        h = tf.zeros((batch_size, self.hidden_dim), dtype=tf.float32)

        outputs = []
        for t in range(64):
            xt = x[:, t, :]                      # (1, 5)
            gate_x = tf.matmul(xt, self.W_ih) + self.b_ih  # (1, 48)
            gate_h = tf.matmul(h, self.W_hh) + self.b_hh    # (1, 48)

            # 拆分三个门
            r = tf.sigmoid(gate_x[:, :self.hidden_dim] + gate_h[:, :self.hidden_dim])
            z = tf.sigmoid(gate_x[:, self.hidden_dim:2*self.hidden_dim] + gate_h[:, self.hidden_dim:2*self.hidden_dim])
            n = tf.tanh(gate_x[:, 2*self.hidden_dim:] + r * gate_h[:, 2*self.hidden_dim:])

            # 安全实现 (1 - z) -> 1 + (-z)，不产生 SUB 算子
            one = tf.constant(1.0, dtype=tf.float32)
            neg_z = tf.multiply(z, tf.constant(-1.0, dtype=tf.float32))
            one_minus_z = tf.add(one, neg_z)
            h = tf.add(tf.multiply(one_minus_z, n), tf.multiply(z, h))
            outputs.append(h)

        # 平均池化
        all_h = tf.stack(outputs, axis=1)          # (1, 64, hidden)
        feat = tf.reduce_mean(all_h, axis=1)       # (1, hidden)

        # 安全 LayerNorm（只用 ADD, MUL, MEAN, RSQRT）
        mean = tf.reduce_mean(feat, axis=-1, keepdims=True)
        neg_mean = tf.multiply(mean, tf.constant(-1.0, dtype=tf.float32))
        feat_centered = tf.add(feat, neg_mean)

        feat_sq = tf.multiply(feat_centered, feat_centered)   # 平方
        variance = tf.reduce_mean(feat_sq, axis=-1, keepdims=True)
        inv_std = tf.math.rsqrt(tf.add(variance, tf.constant(1e-5, dtype=tf.float32)))

        feat_norm = tf.multiply(feat_centered, inv_std)
        feat = tf.add(tf.multiply(self.ln_gamma, feat_norm), self.ln_beta)

        # 全连接层
        logits = tf.matmul(feat, self.fc_weight) + self.fc_bias
        return logits


# 实例化模型
tf_model = TfGRUFromScratch(hidden_dim=16, num_classes=6)

# 手动赋值权重（将 PyTorch 权重复制到 TF 变量）
tf_model.W_ih = tf.Variable(weight_ih, dtype=tf.float32, trainable=False)
tf_model.W_hh = tf.Variable(weight_hh, dtype=tf.float32, trainable=False)
tf_model.b_ih = tf.Variable(bias_ih, dtype=tf.float32, trainable=False)
tf_model.b_hh = tf.Variable(bias_hh, dtype=tf.float32, trainable=False)
tf_model.fc_weight = tf.Variable(fc_weight, dtype=tf.float32, trainable=False)
tf_model.fc_bias = tf.Variable(fc_bias, dtype=tf.float32, trainable=False)
tf_model.ln_gamma = tf.Variable(ln_gamma, dtype=tf.float32, trainable=False)
tf_model.ln_beta = tf.Variable(ln_beta, dtype=tf.float32, trainable=False)

# 运行一次以创建所有变量
dummy_input = tf.random.normal((1, 64, 5))
_ = tf_model(dummy_input)


# ==================== 3. 验证输出一致性（PyTorch vs TensorFlow）====================
test_input = np.random.randn(1, 64, 5).astype(np.float32)
tf_out = tf_model(test_input).numpy()

torch_input = torch.from_numpy(test_input)
with torch.no_grad():
    torch_out = pytorch_model(torch_input).numpy()

diff = np.abs(tf_out - torch_out).max()
print(f"PyTorch 与 TensorFlow 输出的最大差异: {diff:.6f}（应小于 1e-5）")

# ==================== 4. 转换为 TFLite ====================
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

# 强制使用内置算子，避免引入实验性或自定义算子
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 固定输入形状，彻底消除动态 FILL
converter.input_spec = [tf.TensorSpec(shape=(1, 64, 5), dtype=tf.float32)]

tflite_model = converter.convert()

with open('road_shape_gru.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite 模型已保存为 road_shape_gru.tflite")

# 打印模型中的算子列表（供核对）
print("\n模型使用的算子列表：")
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
ops = interpreter._get_ops_details()
unique_ops = set(op['op_name'] for op in ops)
for op in sorted(unique_ops):
    print(f"  {op}")