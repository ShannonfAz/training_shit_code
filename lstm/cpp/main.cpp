// LSTM C++ 推理验证 + 计时
// 用法: 在 build/ 目录执行 ./lstm_infer （默认读取 ../2026_8_1_v1.json）
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include "lstmrun.hpp"

// ---------- 极简 JSON 解析（仅适用本数据集格式） ----------
struct Sample {
    std::string label;
    std::vector<std::pair<float, float>> points;   // 64 个 (x, y)
};

static std::vector<Sample> parse_dataset(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string& s = ss.str();

    std::vector<Sample> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t lab = s.find("\"label\": \"", i);
        if (lab == std::string::npos) break;
        lab += 10;
        size_t lab_end = s.find('"', lab);
        Sample smp;
        smp.label = s.substr(lab, lab_end - lab);

        size_t pt = s.find("\"point\": [", lab_end);
        if (pt == std::string::npos) break;
        pt += 10;   // 已指向第一个内层 '['
        size_t p = pt;
        while (true) {
            while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) p++;
            if (p >= s.size() || s[p] == ']') break;   // 点数组结束
            if (s[p] != '[') { std::cerr << "格式异常 pos=" << p << std::endl; return out; }
            size_t comma = s.find(',', p);
            size_t rb = s.find(']', comma);
            smp.points.emplace_back(
                std::stof(s.substr(p + 1, comma - p - 1)),
                std::stof(s.substr(comma + 1, rb - comma - 1)));
            p = rb + 1;
            while (p < s.size() && (s[p] == ',' || s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) p++;
        }
        pt = p;   // 指向外层 ']' 之后
        out.push_back(std::move(smp));
        i = pt;
    }
    return out;
}

int main(int argc, char* argv[]) {
    const std::string json_path = (argc > 1) ? argv[1] : "../2026_8_1_v1.json";
    auto data = parse_dataset(json_path);
    if (data.empty() || data[0].points.size() != 64) {
        std::cerr << "数据解析失败或格式不符" << std::endl;
        return 1;
    }
    std::cout << "加载 " << data.size() << " 个样本" << std::endl;

    LstmClassifier clf;
    clf.init();

    // ---------- 全量推理 + 统计 ----------
    int correct = 0;
    float worst_margin = 1.0f;
    int worst_idx = -1;
    std::vector<int> per_label_total(6, 0), per_label_correct(6, 0);
    std::vector<std::string> errors;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < data.size(); ++i) {
        std::vector<float> probs = clf.run(data[i].points);
        int pred = (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());
        std::vector<float> sorted = probs;
        std::sort(sorted.rbegin(), sorted.rend());
        float margin = sorted[0] - sorted[1];

        int true_id = -1;
        for (int k = 0; k < 6; ++k) {
            if (clf.labels[k] == data[i].label) { true_id = k; break; }
        }
        per_label_total[true_id]++;
        if (pred == true_id) {
            correct++;
            per_label_correct[true_id]++;
        } else {
            errors.push_back("JSON#" + std::to_string(i) + ": " + data[i].label +
                             " -> " + clf.labels[pred]);
        }
        if (margin < worst_margin) {
            worst_margin = margin;
            worst_idx = (int)i;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
                    / (double)data.size();

    std::cout << "\n========== 推理统计 ==========" << std::endl;
    std::cout << "总样本数: " << data.size() << "，正确: " << correct
              << "，错误: " << data.size() - correct
              << "，准确率: " << (double)correct / data.size() << std::endl;
    for (int k = 0; k < 6; ++k) {
        std::cout << "  " << clf.labels[k] << ": " << per_label_correct[k] << "/"
                  << per_label_total[k] << std::endl;
    }
    for (const auto& e : errors) std::cout << "  错误: " << e << std::endl;
    if (worst_idx >= 0) {
        std::cout << "最弱样本: JSON#" << worst_idx << " (" << data[worst_idx].label
                  << ")，margin " << worst_margin << std::endl;
    }
    std::cout << "平均推理耗时: " << avg_ns << " ns/样本 ("
              << avg_ns / 1000.0 << " µs)" << std::endl;

    // ---------- 纯前向基准（单样本重复 N 次，与 Python bench 对齐） ----------
    const int N = 100000;
    auto probe = data[0].points;
    for (int i = 0; i < 100; ++i) clf.run(probe);   // 预热
    auto b0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) clf.run(probe);
    auto b1 = std::chrono::high_resolution_clock::now();
    double bench_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(b1 - b0).count()
                      / (double)N;
    std::cout << "纯前向基准(" << N << "次): " << bench_ns << " ns/样本 ("
              << bench_ns / 1000.0 << " µs)" << std::endl;

    return 0;
}
