import numpy as np
import struct
import matplotlib.pyplot as plt
import argparse

class ConfigurableNN:
    """可配置深度和宽度的神经网络"""
    def __init__(self, layer_sizes):
        """
        参数:
            layer_sizes: list, 每层的神经元数
            例如: [48, 32, 16, 3] 表示:
                  输入48 -> 隐藏层32 -> 隐藏层16 -> 输出3
                  这是3层网络（2个隐藏层 + 1个输出层）
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        print(f"创建神经网络:")
        print(f"  - 网络深度: {self.num_layers}层")
        print(f"  - 网络结构: {' -> '.join(map(str, layer_sizes))}")
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers):
            # He初始化
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
            print(f"  - Layer {i+1}: [{layer_sizes[i]}, {layer_sizes[i+1]}]")
        
        # 训练历史
        self.history = {
            'loss': [],
            'accuracy': []
        }
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, X, quantize=True):
        """前向传播"""
        # 归一化输入
        activations = [X / 255.0]
        
        for i in range(self.num_layers):
            z = np.dot(activations[-1], self.weights[i].T) + self.biases[i]
            
            if i < self.num_layers - 1:  # 隐藏层用ReLU
                a = self.relu(z)
                if quantize:
                    # 模拟uint8量化
                    a = np.clip(np.round(a * 255), 0, 255) / 255.0
            else:  # 输出层用Softmax
                a = self.softmax(z)
            
            activations.append(a)
        
        return activations
    
    def backward(self, X, y, learning_rate=0.01):
        """反向传播"""
        m = X.shape[0]
        
        # 前向传播
        activations = self.forward(X, quantize=False)
        
        # 计算输出层梯度
        delta = activations[-1].copy()
        delta[range(m), y] -= 1
        delta /= m
        
        # 保存中间激活值的线性组合（用于计算导数）
        z_values = []
        for i in range(self.num_layers):
            z = np.dot(activations[i], self.weights[i].T) + self.biases[i]
            z_values.append(z)
        
        # 反向传播
        for i in range(self.num_layers - 1, -1, -1):
            # 计算梯度
            dW = np.dot(delta.T, activations[i])
            db = np.sum(delta, axis=0)
            
            # 更新权重
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
            # 传播到前一层
            if i > 0:
                delta = np.dot(delta, self.weights[i]) * self.relu_derivative(z_values[i-1])
    
    def train(self, X, y, epochs=100, batch_size=32, learning_rate=0.01, verbose=True):
        """训练模型"""
        n_samples = X.shape[0]
        
        print(f"\n开始训练:")
        print(f"  - 训练样本: {n_samples}")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Learning rate: {learning_rate}")
        print("-" * 60)
        
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
                activations = self.forward(batch_X, quantize=False)
                probs = activations[-1]
                
                # 计算损失
                batch_loss = -np.log(probs[range(len(batch_y)), batch_y] + 1e-8).mean()
                total_loss += batch_loss
                
                # 计算准确率
                predictions = np.argmax(probs, axis=1)
                correct += np.sum(predictions == batch_y)
                
                # Backward
                self.backward(batch_X, batch_y, learning_rate)
            
            # 记录历史
            avg_loss = total_loss / (n_samples // batch_size)
            accuracy = 100 * correct / n_samples
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(accuracy)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1:3d}/{epochs}] Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%')
        
        print("-" * 60)
        print(f"训练完成! 最终准确率: {accuracy:.2f}%")
    
    def predict(self, X):
        """预测"""
        activations = self.forward(X, quantize=True)
        return np.argmax(activations[-1], axis=1)
    
    def plot_training_history(self, save_path='training_history.png'):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['loss']) + 1)
        
        # Loss曲线
        ax1.plot(epochs, self.history['loss'], 'b-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Accuracy曲线
        ax2.plot(epochs, self.history['accuracy'], 'g-', linewidth=2, label='Training Accuracy')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training Accuracy over Epochs', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n训练曲线已保存到: {save_path}")
        plt.close()


def generate_sample_data(filename, num_samples=1000, num_pixels=16):
    """生成示例训练数据"""
    print(f"生成训练数据:")
    print(f"  - 样本数: {num_samples}")
    print(f"  - 像素数: {num_pixels}")
    print(f"  - 输入维度: {num_pixels * 3} (RGB)")
    
    with open(filename, 'w') as f:
        for i in range(num_samples):
            # 随机生成像素
            pixels = np.random.randint(0, 256, size=num_pixels*3, dtype=np.uint8)
            
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
    
    print(f"数据已保存到: {filename}")


def load_data(filename):
    """加载数据"""
    data = []
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            values = list(map(int, line.strip().split(',')))
            data.append(values[:-1])
            labels.append(values[-1])
    
    X = np.array(data, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    print(f"\n加载数据:")
    print(f"  - 样本数: {len(X)}")
    print(f"  - 输入维度: {X.shape[1]}")
    print(f"  - 标签分布: {np.bincount(y)}")
    
    return X, y


def quantize_weights(weight, bits=8):
    """量化权重"""
    max_val = 2**bits - 1
    w = weight
    w_min = w.min()
    w_max = w.max()
    
    if w_max - w_min < 1e-8:
        w_quantized = np.zeros_like(w, dtype=np.uint8)
    else:
        w_quantized = np.round((w - w_min) / (w_max - w_min) * max_val)
        w_quantized = np.clip(w_quantized, 0, max_val).astype(np.uint8)
    
    return w_quantized, w_min, w_max


def export_weights_to_cpp(model, output_file):
    """导出权重为C++格式"""
    print(f"\n导出权重:")
    
    with open(output_file, 'wb') as f:
        # 保存网络结构
        f.write(struct.pack('I', model.num_layers))
        for size in model.layer_sizes:
            f.write(struct.pack('I', size))
        
        # 量化并保存每层的权重和偏置
        for i in range(model.num_layers):
            w_q, w_min, w_max = quantize_weights(model.weights[i])
            b_q, b_min, b_max = quantize_weights(model.biases[i])
            
            # 保存量化参数
            f.write(struct.pack('f', w_min))
            f.write(struct.pack('f', w_max))
            f.write(struct.pack('f', b_min))
            f.write(struct.pack('f', b_max))
            
            # 保存权重
            f.write(w_q.tobytes())
            f.write(b_q.tobytes())
            
            print(f"  Layer {i+1}: W{w_q.shape}, range=[{w_min:.4f}, {w_max:.4f}]")
    
    print(f"\n权重已保存到: {output_file}")
    
    # 同时保存文本格式
    with open(output_file + '.txt', 'w') as f:
        f.write(f"# Network Structure: {' -> '.join(map(str, model.layer_sizes))}\n")
        f.write(f"# Num Layers: {model.num_layers}\n\n")
        
        for i in range(model.num_layers):
            w_q, _, _ = quantize_weights(model.weights[i])
            b_q, _, _ = quantize_weights(model.biases[i])
            
            f.write(f"# Layer {i+1} Weight ({w_q.shape[0]}x{w_q.shape[1]})\n")
            for row in w_q:
                f.write(','.join(map(str, row)) + '\n')
            
            f.write(f"\n# Layer {i+1} Bias ({len(b_q)})\n")
            f.write(','.join(map(str, b_q)) + '\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可配置的Uint8量化神经网络训练')
    
    # 网络结构参数
    parser.add_argument('--num_pixels', type=int, default=16, 
                        help='输入像素数 (默认: 16)')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[16],
                        help='隐藏层神经元数 (默认: [16]). 例如: --hidden_layers 32 16 表示两个隐藏层')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='分类数 (默认: 3)')
    
    # 训练参数
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='生成的训练样本数 (默认: 1000)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (默认: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小 (默认: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='学习率 (默认: 0.01)')
    
    # 输出参数
    parser.add_argument('--data_file', type=str, default='train_data.txt',
                        help='训练数据文件 (默认: train_data.txt)')
    parser.add_argument('--weight_file', type=str, default='weights_uint8.bin',
                        help='权重输出文件 (默认: weights_uint8.bin)')
    parser.add_argument('--plot_file', type=str, default='training_history.png',
                        help='训练曲线图文件 (默认: training_history.png)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("可配置Uint8量化神经网络训练")
    print("=" * 60)
    
    # 构建网络结构
    input_dim = args.num_pixels * 3  # RGB
    layer_sizes = [input_dim] + args.hidden_layers + [args.num_classes]
    
    print(f"\n配置:")
    print(f"  输入像素数: {args.num_pixels}")
    print(f"  输入维度: {input_dim}")
    print(f"  隐藏层: {args.hidden_layers}")
    print(f"  输出类别: {args.num_classes}")
    print(f"  网络结构: {' -> '.join(map(str, layer_sizes))}")
    print(f"  网络深度: {len(layer_sizes) - 1}层")
    
    # 1. 生成数据
    print("\n" + "=" * 60)
    print("步骤1: 生成训练数据")
    print("=" * 60)
    generate_sample_data(args.data_file, args.num_samples, args.num_pixels)
    
    # 2. 加载数据
    print("\n" + "=" * 60)
    print("步骤2: 加载数据")
    print("=" * 60)
    X, y = load_data(args.data_file)
    
    # 3. 创建并训练模型
    print("\n" + "=" * 60)
    print("步骤3: 创建并训练模型")
    print("=" * 60)
    model = ConfigurableNN(layer_sizes)
    model.train(X, y, epochs=args.epochs, batch_size=args.batch_size, 
                learning_rate=args.learning_rate, verbose=True)
    
    # 4. 绘制训练曲线
    print("\n" + "=" * 60)
    print("步骤4: 绘制训练曲线")
    print("=" * 60)
    model.plot_training_history(args.plot_file)
    
    # 5. 导出权重
    print("\n" + "=" * 60)
    print("步骤5: 导出权重")
    print("=" * 60)
    export_weights_to_cpp(model, args.weight_file)
    
    # 6. 最终测试
    print("\n" + "=" * 60)
    print("步骤6: 最终测试")
    print("=" * 60)
    predictions = model.predict(X)
    accuracy = 100 * np.sum(predictions == y) / len(y)
    print(f"训练集准确率: {accuracy:.2f}%")
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  ✓ {args.data_file}")
    print(f"  ✓ {args.weight_file}")
    print(f"  ✓ {args.weight_file}.txt")
    print(f"  ✓ {args.plot_file}")
    print("\n下一步: 使用C++推理引擎进行推理")
