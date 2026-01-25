#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

class Uint8NeuralNetwork {
private:
    // 网络结构参数
    static const int INPUT_DIM = 48;   // 16 pixels × 3 channels
    static const int HIDDEN_DIM = 16;  // 隐藏层神经元数
    static const int OUTPUT_DIM = 3;   // 分类数
    
    // 量化权重 (uint8)
    std::vector<uint8_t> fc1_weight;  // [16, 48]
    std::vector<uint8_t> fc1_bias;    // [16]
    std::vector<uint8_t> fc2_weight;  // [3, 16]
    std::vector<uint8_t> fc2_bias;    // [3]
    
    // 量化参数 (用于反量化，如果需要精确计算)
    float fc1_w_min, fc1_w_max;
    float fc1_b_min, fc1_b_max;
    float fc2_w_min, fc2_w_max;
    float fc2_b_min, fc2_b_max;
    
    // ReLU激活函数 (直接截断负值)
    inline uint8_t relu(int32_t x) {
        if (x < 0) return 0;
        if (x > 255) return 255;
        return static_cast<uint8_t>(x);
    }
    
public:
    Uint8NeuralNetwork() {
        fc1_weight.resize(HIDDEN_DIM * INPUT_DIM);
        fc1_bias.resize(HIDDEN_DIM);
        fc2_weight.resize(OUTPUT_DIM * HIDDEN_DIM);
        fc2_bias.resize(OUTPUT_DIM);
    }
    
    // 加载权重文件
    bool load_weights(const char* filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "无法打开权重文件: " << filename << std::endl;
            return false;
        }
        
        // 读取量化参数
        file.read(reinterpret_cast<char*>(&fc1_w_min), sizeof(float));
        file.read(reinterpret_cast<char*>(&fc1_w_max), sizeof(float));
        file.read(reinterpret_cast<char*>(&fc1_b_min), sizeof(float));
        file.read(reinterpret_cast<char*>(&fc1_b_max), sizeof(float));
        file.read(reinterpret_cast<char*>(&fc2_w_min), sizeof(float));
        file.read(reinterpret_cast<char*>(&fc2_w_max), sizeof(float));
        file.read(reinterpret_cast<char*>(&fc2_b_min), sizeof(float));
        file.read(reinterpret_cast<char*>(&fc2_b_max), sizeof(float));
        
        // 读取权重尺寸 (验证用)
        uint32_t h1, w1, h2, w2;
        file.read(reinterpret_cast<char*>(&h1), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&w1), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&h2), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&w2), sizeof(uint32_t));
        
        if (h1 != HIDDEN_DIM || w1 != INPUT_DIM || h2 != OUTPUT_DIM || w2 != HIDDEN_DIM) {
            std::cerr << "权重尺寸不匹配!" << std::endl;
            return false;
        }
        
        // 读取权重数据
        file.read(reinterpret_cast<char*>(fc1_weight.data()), HIDDEN_DIM * INPUT_DIM);
        file.read(reinterpret_cast<char*>(fc1_bias.data()), HIDDEN_DIM);
        file.read(reinterpret_cast<char*>(fc2_weight.data()), OUTPUT_DIM * HIDDEN_DIM);
        file.read(reinterpret_cast<char*>(fc2_bias.data()), OUTPUT_DIM);
        
        file.close();
        
        std::cout << "权重加载成功!" << std::endl;
        std::cout << "FC1: [" << HIDDEN_DIM << ", " << INPUT_DIM << "]" << std::endl;
        std::cout << "FC2: [" << OUTPUT_DIM << ", " << HIDDEN_DIM << "]" << std::endl;
        
        return true;
    }
    
    // 矩阵乘法: output = input × weight^T + bias
    void matmul_add_bias(
        const std::vector<uint8_t>& input,
        const std::vector<uint8_t>& weight,
        const std::vector<uint8_t>& bias,
        std::vector<int32_t>& output,
        int in_dim,
        int out_dim
    ) {
        for (int i = 0; i < out_dim; i++) {
            int32_t sum = 0;
            
            // 点积
            for (int j = 0; j < in_dim; j++) {
                sum += static_cast<int32_t>(input[j]) * 
                       static_cast<int32_t>(weight[i * in_dim + j]);
            }
            
            // 加bias (缩放到同一量级)
            sum = sum / 256 + static_cast<int32_t>(bias[i]);
            
            output[i] = sum;
        }
    }
    
    // 推理
    int inference(const std::vector<uint8_t>& pixel_data) {
        if (pixel_data.size() != INPUT_DIM) {
            std::cerr << "输入尺寸错误! 需要 " << INPUT_DIM << " 个uint8值" << std::endl;
            return -1;
        }
        
        // Layer 1: FC + ReLU
        std::vector<int32_t> hidden_raw(HIDDEN_DIM);
        matmul_add_bias(pixel_data, fc1_weight, fc1_bias, hidden_raw, INPUT_DIM, HIDDEN_DIM);
        
        // 应用ReLU并量化到uint8
        std::vector<uint8_t> hidden(HIDDEN_DIM);
        for (int i = 0; i < HIDDEN_DIM; i++) {
            hidden[i] = relu(hidden_raw[i]);
        }
        
        // Layer 2: FC
        std::vector<int32_t> output(OUTPUT_DIM);
        matmul_add_bias(hidden, fc2_weight, fc2_bias, output, HIDDEN_DIM, OUTPUT_DIM);
        
        // Argmax
        int max_idx = 0;
        int32_t max_val = output[0];
        for (int i = 1; i < OUTPUT_DIM; i++) {
            if (output[i] > max_val) {
                max_val = output[i];
                max_idx = i;
            }
        }
        
        return max_idx;
    }
    
    // 批量推理并统计准确率
    void evaluate(const char* test_file) {
        std::ifstream file(test_file);
        if (!file.is_open()) {
            std::cerr << "无法打开测试文件: " << test_file << std::endl;
            return;
        }
        
        int total = 0;
        int correct = 0;
        std::string line;
        
        while (std::getline(file, line)) {
            // 解析一行数据
            std::vector<uint8_t> pixels;
            int label;
            
            size_t pos = 0;
            for (int i = 0; i < INPUT_DIM; i++) {
                size_t next = line.find(',', pos);
                int val = std::stoi(line.substr(pos, next - pos));
                pixels.push_back(static_cast<uint8_t>(val));
                pos = next + 1;
            }
            label = std::stoi(line.substr(pos));
            
            // 推理
            int predicted = inference(pixels);
            
            total++;
            if (predicted == label) {
                correct++;
            }
        }
        
        file.close();
        
        float accuracy = 100.0f * correct / total;
        std::cout << "\n测试结果:" << std::endl;
        std::cout << "总样本数: " << total << std::endl;
        std::cout << "正确数: " << correct << std::endl;
        std::cout << "准确率: " << accuracy << "%" << std::endl;
    }
};


int main(int argc, char* argv[]) {
    std::cout << "=== Uint8 量化神经网络推理引擎 ===" << std::endl;
    
    Uint8NeuralNetwork nn;
    
    // 加载权重
    if (!nn.load_weights("weights_uint8.bin")) {
        return 1;
    }
    
    // 测试单个样本
    std::cout << "\n--- 单样本测试 ---" << std::endl;
    std::vector<uint8_t> test_pixel = {
        128, 64, 32, 255, 128, 0, 64, 128, 255, 32, 64, 128,
        255, 0, 128, 128, 255, 64, 32, 128, 255, 64, 32, 128,
        100, 150, 200, 50, 100, 150, 200, 50, 100, 150, 200, 50,
        75, 125, 175, 225, 75, 125, 175, 225, 75, 125, 175, 225
    };
    
    int result = nn.inference(test_pixel);
    std::cout << "预测类别: " << result << std::endl;
    
    // 评估整个测试集
    std::cout << "\n--- 测试集评估 ---" << std::endl;
    nn.evaluate("train_data.txt");  // 用训练数据评估（实际应该用单独的测试集）
    
    return 0;
}
