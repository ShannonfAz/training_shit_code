#ifndef _LSTMRUN_HPP_
#define _LSTMRUN_HPP_

#include <vector>
#include <cmath>

// LSTM 推理类（input_dim=4, hidden_dim=8），权重来自 lstm_weights.h
class LstmClassifier {
public:
    static constexpr int kSeqLen = 64;
    static constexpr int kInputDim = 4;    // dx, dy, ddir_x, ddir_y
    static constexpr int kHiddenDim = 8;
    static constexpr int kNumClasses = 6;

    const char* labels[6] = {
        "zhixian", "shizi", "daodazuohuandao",
        "jinruzuohuandao", "daodayouhuandao", "jinruyouhuandao"
    };

    LstmClassifier();
    ~LstmClassifier();

    bool init();

    // 输入 64 个点（x,y），返回 6 类概率
    std::vector<float> run(const std::vector<std::pair<float, float>>& points);

private:
    static inline float sigmoid(float x);
    static inline float tanh(float x);

    // 预处理：64 个点 -> (64, 4) 特征（Z-score 归一化）
    void preprocess(const std::vector<std::pair<float, float>>& points, float feat[][kInputDim]);
};

#endif
