"""
convert_onnx_to_tflite.py
利用 onnx2tf 将 ONNX 模型转为 TensorFlow SavedModel，再生成 TFLite。

依赖：
    pip install onnx onnx2tf tensorflow
"""

import os
import onnx
import tensorflow as tf
import onnx2tf

# ==================== 1. 加载 ONNX 模型 ====================
onnx_path = "road_shape_gru.onnx"
onnx_model = onnx.load(onnx_path)
print("ONNX 模型加载成功。")

# ==================== 2. ONNX → TensorFlow SavedModel ====================
tf_model_dir = "tf_model_from_onnx"

# onnx2tf 的 convert 函数，直接生成 SavedModel 或 TFLite
onnx2tf.convert(
    input_onnx_file_path=onnx_path,
    output_folder_path=tf_model_dir,
    output_signaturedefs=True,   # 生成 signature
    copy_onnx_input_output_names_to_tflite=True,
    non_verbose=True,
)
print(f"TensorFlow SavedModel 已保存至: {tf_model_dir}")

# ==================== 3. SavedModel → TFLite ====================
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
# 可选：开启优化
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = "road_shape_gru.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"TFLite 模型已保存至: {tflite_path}")

# ==================== 4. 验证 TFLite 模型 ====================
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("输入详情:", input_details)
print("输出详情:", output_details)

# 简单推理测试
import numpy as np
test_input = np.random.randn(1, 64, 5).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print("推理输出 shape:", output.shape)
print("TFLite 模型验证成功！")