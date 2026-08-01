#include "lstmrun.hpp"
#include "lstm_weights.h"   // 由 export_lstm_weights.py 生成（input_dim=4, hidden_dim=8）
#include <cmath>

// ======================== 激活函数 ========================
inline float LstmClassifier::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
inline float LstmClassifier::tanh(float x) {
    return std::tanh(x);
}

LstmClassifier::LstmClassifier()  = default;
LstmClassifier::~LstmClassifier() = default;

bool LstmClassifier::init() {
    return true;
}

// ======================== 推理主函数 ========================
std::vector<float> LstmClassifier::run(const std::vector<std::pair<float, float>>& points) {
    if (points.size() != kSeqLen) return {};

    float features[kSeqLen][kInputDim];
    preprocess(points, features);

    // ---------- 1. LSTM 前向传播 ----------
    float h[kHiddenDim] = {0};
    float c[kHiddenDim] = {0};
    float all_h[kSeqLen][kHiddenDim];

    constexpr int kGateDim = 4 * kHiddenDim;   // 4 门: [i, f, g, o]

    for (int t = 0; t < kSeqLen; ++t) {
        float gate_x[kGateDim];
        float gate_h[kGateDim];

        // W_ih * x_t + b_ih
        for (int i = 0; i < kGateDim; ++i) {
            float sum = b_ih[i];
            for (int j = 0; j < kInputDim; ++j) {
                sum += features[t][j] * W_ih[i * kInputDim + j];
            }
            gate_x[i] = sum;
        }

        // W_hh * h + b_hh
        for (int i = 0; i < kGateDim; ++i) {
            float sum = b_hh[i];
            for (int j = 0; j < kHiddenDim; ++j) {
                sum += h[j] * W_hh[i * kHiddenDim + j];
            }
            gate_h[i] = sum;
        }

        // 门控运算（PyTorch 门顺序: i, f, g, o）
        for (int i = 0; i < kHiddenDim; ++i) {
            float ig = sigmoid(gate_x[i] + gate_h[i]);
            float fg = sigmoid(gate_x[i + kHiddenDim] + gate_h[i + kHiddenDim]);
            float gg = tanh(gate_x[i + 2 * kHiddenDim] + gate_h[i + 2 * kHiddenDim]);
            float og = sigmoid(gate_x[i + 3 * kHiddenDim] + gate_h[i + 3 * kHiddenDim]);
            c[i] = fg * c[i] + ig * gg;
            h[i] = og * tanh(c[i]);
        }

        // 保存当前时间步输出（用于平均池化）
        for (int i = 0; i < kHiddenDim; ++i) {
            all_h[t][i] = h[i];
        }
    }

    // ---------- 2. 平均池化 ----------
    float feat[kHiddenDim] = {0};
    for (int i = 0; i < kHiddenDim; ++i) {
        for (int t = 0; t < kSeqLen; ++t) {
            feat[i] += all_h[t][i];
        }
        feat[i] /= kSeqLen;
    }

    // ---------- 3. LayerNorm ----------
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < kHiddenDim; ++i) mean += feat[i];
    mean /= kHiddenDim;
    for (int i = 0; i < kHiddenDim; ++i) {
        float diff = feat[i] - mean;
        var += diff * diff;
    }
    var = var / kHiddenDim + 1e-5f;
    float inv_std = 1.0f / std::sqrt(var);
    for (int i = 0; i < kHiddenDim; ++i) {
        feat[i] = (feat[i] - mean) * inv_std;
        feat[i] = feat[i] * ln_gamma[i] + ln_beta[i];
    }

    // ---------- 4. 全连接层 ----------
    float logits[kNumClasses] = {0};
    for (int i = 0; i < kNumClasses; ++i) {
        for (int j = 0; j < kHiddenDim; ++j) {
            logits[i] += feat[j] * fc_weight[i * kHiddenDim + j];
        }
        logits[i] += fc_bias[i];
    }

    // ---------- 5. Softmax ----------
    float max_logit = logits[0];
    for (int i = 1; i < kNumClasses; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < kNumClasses; ++i) {
        logits[i] = std::exp(logits[i] - max_logit);
        sum += logits[i];
    }
    std::vector<float> probs(kNumClasses);
    for (int i = 0; i < kNumClasses; ++i) {
        probs[i] = logits[i] / sum;
    }
    return probs;
}

// ======================== 预处理 ========================
void LstmClassifier::preprocess(const std::vector<std::pair<float, float>>& points,
                                float feat[][kInputDim]) {
    // 1. 位移 (dx, dy)
    float diff[kSeqLen][2];
    for (int i = 0; i < kSeqLen - 1; ++i) {
        diff[i][0] = points[i+1].first - points[i].first;
        diff[i][1] = points[i+1].second - points[i].second;
    }
    diff[kSeqLen-1][0] = diff[kSeqLen-2][0];
    diff[kSeqLen-1][1] = diff[kSeqLen-2][1];

    // 2. 步长和方向向量
    float dir_vec[kSeqLen][2];
    for (int i = 0; i < kSeqLen; ++i) {
        float len = std::sqrt(diff[i][0]*diff[i][0] + diff[i][1]*diff[i][1]) + 1e-8f;
        dir_vec[i][0] = diff[i][0] / len;
        dir_vec[i][1] = diff[i][1] / len;
    }

    // 3. 方向变化 (ddir_x, ddir_y)
    float ddir[kSeqLen][2];
    for (int i = 0; i < kSeqLen - 1; ++i) {
        ddir[i][0] = dir_vec[i+1][0] - dir_vec[i][0];
        ddir[i][1] = dir_vec[i+1][1] - dir_vec[i][1];
    }
    ddir[kSeqLen-1][0] = ddir[kSeqLen-2][0];
    ddir[kSeqLen-1][1] = ddir[kSeqLen-2][1];

    // 4. Z-score 归一化并填入特征（4 维：dx, dy, ddir_x, ddir_y）
    for (int i = 0; i < kSeqLen; ++i) {
        feat[i][0] = (diff[i][0] - FEAT_MEAN[0]) / FEAT_STD[0];
        feat[i][1] = (diff[i][1] - FEAT_MEAN[1]) / FEAT_STD[1];
        feat[i][2] = (ddir[i][0] - FEAT_MEAN[2]) / FEAT_STD[2];
        feat[i][3] = (ddir[i][1] - FEAT_MEAN[3]) / FEAT_STD[3];
    }
}
