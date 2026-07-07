import numpy as np

data = np.load('feat_stats.npz')
mean = data['mean']  # shape (5,)
std  = data['std']   # shape (5,)

print("// 请将以下内容复制到 grurun.cpp 中替换 FEAT_MEAN 和 FEAT_STD")
print("static const float FEAT_MEAN[5] = {", ", ".join(f"{v:.8f}f" for v in mean), "};")
print("static const float FEAT_STD[5]  = {", ", ".join(f"{v:.8f}f" for v in std),  "};")