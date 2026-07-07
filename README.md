训练用的垃圾代码，全程vibe coding，本人不会写py，慎用。

数据集：byd_2026_5_26.json

训练：byd64_qlh.py

推理（py）：bydrun64_qlh.py

导出权重：

export_weights_qlh.py

feat_stats.py

导出权重后你会看到一个更新了的gru_weight.h，你需要把它丢进克隆下来的隔壁仓库的project/model/ 里面，然后去project/code/grurun.cpp里把第11，12行换成feat_stats.py生成的东西

然后就可以开跑了
