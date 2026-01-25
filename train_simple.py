import numpy as np
import struct

# ============================================
# 1. 生成示例数据
# ============================================
def generate_sample_data(filename, num_samples=1000):
    """生成示例训练数据"""
    print(f"生成 {num_samples} 个训练样本...")
    
    with open(filename, 'w') as f:
        for i in range(num_samples):
            # 随机生成16个像素，每个像素3个通道
            pixels = np.random.randint(0, 256, size=48, dtype=np.uint8)
            
            # 根据像素特征生成标签
            avg_brightness = pixels.mean()
            if avg_brightness < 85:
                label = 0
            elif avg_brightness < 170:
                label = 1
            else:
                label = 2
            
            # 写入文件
            line = ','.join(map(str, pixels)) + ',' + str(label) + '\n'
            f.write(line)
    
    print(f"数据已保存到 {filename}")


# ============================================
# 2. 简单的神经网络训练（numpy实现）
# ============================================
class SimpleNN:
    def __init__(self, input_dim=48, hidden_dim=16, output_dim=3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 随机初始化权重
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.01
        self.b2 = np.zeros(output_dim)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, X):
        # 归一化输入
        X = X / 255.0
        
        # Layer 1
        self.z1 = np.dot(X, self.W1.T) + self.b1
        self.a1 = self.relu(self.z1)
        
        # 模拟量化
        self.a1 = np.clip(np.round(self.a1 * 255), 0, 255) / 255.0
        
        # Layer 2
        self.z2 = np.dot(self.a1, self.W2.T) + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        X = X / 255.0
        
        # 反向传播
        dz2 = self.a2.copy()
        dz2[range(m), y] -= 1
        dz2 /= m
        
        dW2 = np.dot(dz2.T, self.a1)
        db2 = np.sum(dz2, axis=0)
        
        da1 = np.dot(dz2, self.W2)
        dz1 = da1 * (self.z1 > 0)
        
        dW1 = np.dot(dz1.T, X)
        db1 = np.sum(dz1, axis=0)
        
        # 更新权重
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X, y, epochs=100, batch_size=32):
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # 随机打乱
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            total_loss = 0
            correct = 0
            
            # Mini-batch训练
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                # Forward
                probs = self.forward(batch_X)
                
                # 计算损失
                batch_loss = -np.log(probs[range(len(batch_y)), batch_y] + 1e-8).mean()
                total_loss += batch_loss
                
                # 计算准确率
                predictions = np.argmax(probs, axis=1)
                correct += np.sum(predictions == batch_y)
                
                # Backward
                self.backward(batch_X, batch_y)
            
            if (epoch + 1) % 20 == 0:
                accuracy = 100 * correct / n_samples
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Acc: {accuracy:.2f}%')


# ============================================
# 3. 量化和导出权重
# ============================================
def quantize_weights(weight, bits=8):
    """将权重量化到uint8"""
    max_val = 2**bits - 1
    w = weight
    
    # 找到权重范围
    w_min = w.min()
    w_max = w.max()
    
    # 量化到[0, 255]
    if w_max - w_min < 1e-8:  # 避免除0
        w_quantized = np.zeros_like(w, dtype=np.uint8)
    else:
        w_quantized = np.round((w - w_min) / (w_max - w_min) * max_val)
        w_quantized = np.clip(w_quantized, 0, max_val).astype(np.uint8)
    
    return w_quantized, w_min, w_max


def export_weights_to_cpp(model, output_file):
    """导出权重为C++可读的二进制格式"""
    
    # 量化每个参数
    fc1_w_q, fc1_w_min, fc1_w_max = quantize_weights(model.W1)
    fc1_b_q, fc1_b_min, fc1_b_max = quantize_weights(model.b1)
    fc2_w_q, fc2_w_min, fc2_w_max = quantize_weights(model.W2)
    fc2_b_q, fc2_b_min, fc2_b_max = quantize_weights(model.b2)
    
    # 保存为二进制文件
    with open(output_file, 'wb') as f:
        # 保存量化参数
        f.write(struct.pack('f', fc1_w_min))
        f.write(struct.pack('f', fc1_w_max))
        f.write(struct.pack('f', fc1_b_min))
        f.write(struct.pack('f', fc1_b_max))
        f.write(struct.pack('f', fc2_w_min))
        f.write(struct.pack('f', fc2_w_max))
        f.write(struct.pack('f', fc2_b_min))
        f.write(struct.pack('f', fc2_b_max))
        
        # 保存权重尺寸
        f.write(struct.pack('I', fc1_w_q.shape[0]))  # 16
        f.write(struct.pack('I', fc1_w_q.shape[1]))  # 48
        f.write(struct.pack('I', fc2_w_q.shape[0]))  # 3
        f.write(struct.pack('I', fc2_w_q.shape[1]))  # 16
        
        # 保存量化后的权重 (uint8)
        f.write(fc1_w_q.tobytes())
        f.write(fc1_b_q.tobytes())
        f.write(fc2_w_q.tobytes())
        f.write(fc2_b_q.tobytes())
    
    print(f"\n权重已导出到 {output_file}")
    print(f"Layer1 weight: {fc1_w_q.shape}, range: [{fc1_w_min:.4f}, {fc1_w_max:.4f}]")
    print(f"Layer1 bias: {fc1_b_q.shape}, range: [{fc1_b_min:.4f}, {fc1_b_max:.4f}]")
    print(f"Layer2 weight: {fc2_w_q.shape}, range: [{fc2_w_min:.4f}, {fc2_w_max:.4f}]")
    print(f"Layer2 bias: {fc2_b_q.shape}, range: [{fc2_b_min:.4f}, {fc2_b_max:.4f}]")
    
    # 保存为文本格式
    with open(output_file + '.txt', 'w') as f:
        f.write("# FC1 Weight (16x48)\n")
        for row in fc1_w_q:
            f.write(','.join(map(str, row)) + '\n')
        f.write("\n# FC1 Bias (16)\n")
        f.write(','.join(map(str, fc1_b_q)) + '\n')
        f.write("\n# FC2 Weight (3x16)\n")
        for row in fc2_w_q:
            f.write(','.join(map(str, row)) + '\n')
        f.write("\n# FC2 Bias (3)\n")
        f.write(','.join(map(str, fc2_b_q)) + '\n')


# ============================================
# 主程序
# ============================================
if __name__ == '__main__':
    print("=" * 60)
    print("Uint8 量化神经网络训练和导出")
    print("=" * 60)
    
    # 1. 生成数据
    print("\n步骤1: 生成训练数据")
    print("-" * 60)
    generate_sample_data('train_data.txt', num_samples=1000)
    
    # 2. 加载数据
    print("\n步骤2: 加载数据")
    print("-" * 60)
    data = []
    labels = []
    with open('train_data.txt', 'r') as f:
        for line in f:
            values = list(map(int, line.strip().split(',')))
            data.append(values[:-1])
            labels.append(values[-1])
    
    X = np.array(data, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"加载了 {len(X)} 个样本")
    print(f"输入维度: {X.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 3. 训练模型
    print("\n步骤3: 训练模型")
    print("-" * 60)
    model = SimpleNN(input_dim=48, hidden_dim=16, output_dim=3)
    model.train(X, y, epochs=100, batch_size=32)
    
    # 4. 导出权重
    print("\n步骤4: 导出量化权重")
    print("-" * 60)
    export_weights_to_cpp(model, 'weights_uint8.bin')
    
    # 5. 最终测试
    print("\n步骤5: 测试准确率")
    print("-" * 60)
    probs = model.forward(X)
    predictions = np.argmax(probs, axis=1)
    accuracy = 100 * np.sum(predictions == y) / len(y)
    print(f"训练集准确率: {accuracy:.2f}%")
    
    print("\n" + "=" * 60)
    print("完成! 现在可以编译和运行C++推理引擎:")
    print("  make")
    print("  ./inference_uint8")
    print("=" * 60)
