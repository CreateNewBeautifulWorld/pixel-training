import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import itertools

class NeuralNetworkExplorer:
    """网络架构探索器 - FP16训练"""
    def __init__(self, layer_sizes):
        """
        参数:
            layer_sizes: list, 每层神经元数
            例如: [12, 8, 4, 1] 表示 12输入 -> 8 -> 4 -> 1输出
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        # 使用FP16初始化权重
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers):
            # He初始化
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i]).astype(np.float16) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1], dtype=np.float16)
            self.weights.append(w)
            self.biases.append(b)
        
        # 训练历史
        self.history = {
            'loss': [],
            'val_loss': []
        }
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(np.float16)
    
    def forward(self, X):
        """前向传播 - 回归任务"""
        # 归一化输入到[0,1]
        activations = [X.astype(np.float16) / 255.0]
        
        for i in range(self.num_layers):
            z = np.dot(activations[-1], self.weights[i].T) + self.biases[i]
            
            if i < self.num_layers - 1:  # 隐藏层用ReLU
                a = self.relu(z)
            else:  # 输出层不用激活（线性回归）
                a = z
            
            activations.append(a)
        
        return activations
    
    def mse_loss(self, predictions, targets):
        """均方误差损失"""
        return np.mean((predictions - targets) ** 2)
    
    def backward(self, X, y, learning_rate=0.01):
        """反向传播 - MSE损失"""
        m = X.shape[0]
        
        # 前向传播
        activations = self.forward(X)
        predictions = activations[-1].flatten()
        
        # 计算输出层梯度 (MSE导数)
        delta = (predictions - y).reshape(-1, 1) / m
        
        # 保存中间激活值
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
            self.weights[i] = (self.weights[i] - learning_rate * dW).astype(np.float16)
            self.biases[i] = (self.biases[i] - learning_rate * db).astype(np.float16)
            
            # 传播到前一层
            if i > 0:
                delta = np.dot(delta, self.weights[i]) * self.relu_derivative(z_values[i-1])
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, learning_rate=0.01, verbose=False):
        """训练模型"""
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # 随机打乱
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch训练
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                self.backward(batch_X, batch_y, learning_rate)
            
            # 计算训练和验证损失
            train_pred = self.forward(X_train)[-1].flatten()
            train_loss = self.mse_loss(train_pred, y_train)
            
            val_pred = self.forward(X_val)[-1].flatten()
            val_loss = self.mse_loss(val_pred, y_val)
            
            self.history['loss'].append(float(train_loss))
            self.history['val_loss'].append(float(val_loss))
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'  Epoch [{epoch+1:3d}/{epochs}] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        return float(val_loss)  # 返回最终验证损失
    
    def predict(self, X):
        """预测"""
        activations = self.forward(X)
        return activations[-1].flatten()


def generate_regression_data(num_samples=1000, num_pixels=16, noise_level=0.1):
    """
    生成回归数据
    输出 = 像素亮度的某种函数 + 噪声
    """
    X = np.random.randint(0, 256, size=(num_samples, num_pixels * 3), dtype=np.uint8)
    
    # 目标函数：归一化后的平均亮度 + 一些非线性
    X_normalized = X.astype(np.float32) / 255.0
    y = np.mean(X_normalized, axis=1)  # 平均亮度
    y = y ** 1.5  # 非线性变换
    y += np.random.randn(num_samples) * noise_level  # 添加噪声
    y = np.clip(y, 0, 1)  # 限制在[0,1]
    
    return X, y.astype(np.float32)


def grid_search_architecture(
    num_pixels_range=[1, 2, 4, 8, 16, 32, 64],
    hidden_layers_range=[0, 1, 2, 3, 4, 5],
    hidden_neurons_range=[4, 8, 16, 32],
    num_samples=1000,
    epochs=50,
    output_dir='exploration_results'
):
    """
    网格搜索最优架构
    
    参数:
        num_pixels_range: 测试的像素数列表
        hidden_layers_range: 测试的隐藏层数列表
        hidden_neurons_range: 测试的隐藏层神经元数列表
        num_samples: 每个配置的训练样本数
        epochs: 每个配置的训练轮数
        output_dir: 结果输出目录
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    total_configs = len(num_pixels_range) * len(hidden_layers_range) * len(hidden_neurons_range)
    config_idx = 0
    
    print("=" * 80)
    print("网络架构探索 - Grid Search")
    print("=" * 80)
    print(f"总配置数: {total_configs}")
    print(f"像素数范围: {num_pixels_range}")
    print(f"隐藏层数范围: {hidden_layers_range}")
    print(f"神经元数范围: {hidden_neurons_range}")
    print("=" * 80)
    
    for num_pixels in num_pixels_range:
        # 生成数据
        print(f"\n生成数据: {num_pixels} 像素...")
        X, y = generate_regression_data(num_samples=num_samples, num_pixels=num_pixels)
        
        # 划分训练/验证集
        split = int(0.8 * num_samples)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        input_dim = num_pixels * 3
        
        for num_hidden_layers in hidden_layers_range:
            for hidden_neurons in hidden_neurons_range:
                config_idx += 1
                
                # 构建网络结构
                if num_hidden_layers == 0:
                    # 直接从输入到输出
                    layer_sizes = [input_dim, 1]
                else:
                    # 有隐藏层
                    layer_sizes = [input_dim] + [hidden_neurons] * num_hidden_layers + [1]
                
                # 计算参数量
                total_params = sum(
                    layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]
                    for i in range(len(layer_sizes) - 1)
                )
                
                print(f"\n[{config_idx}/{total_configs}] 测试配置:")
                print(f"  像素数: {num_pixels}")
                print(f"  网络结构: {' -> '.join(map(str, layer_sizes))}")
                print(f"  隐藏层数: {num_hidden_layers}")
                print(f"  每层神经元: {hidden_neurons if num_hidden_layers > 0 else 'N/A'}")
                print(f"  参数量: {total_params}")
                
                # 训练模型
                model = NeuralNetworkExplorer(layer_sizes)
                final_val_loss = model.train(
                    X_train, y_train, X_val, y_val,
                    epochs=epochs, batch_size=32, learning_rate=0.01,
                    verbose=False
                )
                
                final_train_loss = model.history['loss'][-1]
                
                print(f"  最终 Train Loss: {final_train_loss:.6f}")
                print(f"  最终 Val Loss: {final_val_loss:.6f}")
                
                # 记录结果
                result = {
                    'config_id': config_idx,
                    'num_pixels': num_pixels,
                    'input_dim': input_dim,
                    'num_hidden_layers': num_hidden_layers,
                    'hidden_neurons': hidden_neurons,
                    'layer_sizes': layer_sizes,
                    'total_params': total_params,
                    'final_train_loss': final_train_loss,
                    'final_val_loss': final_val_loss,
                    'train_history': model.history['loss'],
                    'val_history': model.history['val_loss']
                }
                results.append(result)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f'exploration_results_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\n结果已保存到: {results_file}")
    
    return results, results_file


def analyze_results(results, output_dir='exploration_results'):
    """分析探索结果"""
    
    print("\n" + "=" * 80)
    print("结果分析")
    print("=" * 80)
    
    # 按验证损失排序
    sorted_results = sorted(results, key=lambda x: x['final_val_loss'])
    
    # Top 10配置
    print("\nTop 10 最佳配置 (按验证损失):")
    print("-" * 80)
    for i, result in enumerate(sorted_results[:10], 1):
        print(f"\n排名 #{i}:")
        print(f"  网络结构: {' -> '.join(map(str, result['layer_sizes']))}")
        print(f"  像素数: {result['num_pixels']}")
        print(f"  隐藏层数: {result['num_hidden_layers']}")
        print(f"  神经元数: {result['hidden_neurons']}")
        print(f"  参数量: {result['total_params']}")
        print(f"  验证损失: {result['final_val_loss']:.6f}")
    
    # 可视化
    print("\n生成可视化图表...")
    
    # 1. 参数量 vs 验证损失
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    params = [r['total_params'] for r in results]
    val_losses = [r['final_val_loss'] for r in results]
    
    axes[0, 0].scatter(params, val_losses, alpha=0.6)
    axes[0, 0].set_xlabel('参数量')
    axes[0, 0].set_ylabel('验证损失')
    axes[0, 0].set_title('参数量 vs 验证损失')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 隐藏层数 vs 验证损失
    hidden_layers = [r['num_hidden_layers'] for r in results]
    axes[0, 1].scatter(hidden_layers, val_losses, alpha=0.6)
    axes[0, 1].set_xlabel('隐藏层数')
    axes[0, 1].set_ylabel('验证损失')
    axes[0, 1].set_title('隐藏层数 vs 验证损失')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 输入维度 vs 验证损失
    input_dims = [r['input_dim'] for r in results]
    axes[1, 0].scatter(input_dims, val_losses, alpha=0.6)
    axes[1, 0].set_xlabel('输入维度')
    axes[1, 0].set_ylabel('验证损失')
    axes[1, 0].set_title('输入维度 vs 验证损失')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Top 5训练曲线
    for i, result in enumerate(sorted_results[:5]):
        label = f"{' -> '.join(map(str, result['layer_sizes']))}"
        axes[1, 1].plot(result['val_history'], label=label, linewidth=2)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('验证损失')
    axes[1, 1].set_title('Top 5 配置训练曲线')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'exploration_analysis.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"可视化图表已保存: {plot_file}")
    plt.close()
    
    # 生成对比表
    print("\n生成详细对比表...")
    comparison_file = os.path.join(output_dir, 'comparison_table.txt')
    with open(comparison_file, 'w') as f:
        f.write("网络架构探索 - 对比表\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'排名':<6} {'网络结构':<30} {'像素':<6} {'层数':<6} {'神经元':<8} {'参数量':<10} {'验证损失':<12}\n")
        f.write("-" * 120 + "\n")
        
        for i, result in enumerate(sorted_results, 1):
            structure = ' -> '.join(map(str, result['layer_sizes']))
            f.write(f"{i:<6} {structure:<30} {result['num_pixels']:<6} "
                   f"{result['num_hidden_layers']:<6} {result['hidden_neurons']:<8} "
                   f"{result['total_params']:<10} {result['final_val_loss']:<12.6f}\n")
    
    print(f"对比表已保存: {comparison_file}")
    
    # 返回最佳配置
    best_config = sorted_results[0]
    print("\n" + "=" * 80)
    print("🏆 最佳配置:")
    print("=" * 80)
    print(f"  网络结构: {' -> '.join(map(str, best_config['layer_sizes']))}")
    print(f"  像素数: {best_config['num_pixels']}")
    print(f"  隐藏层数: {best_config['num_hidden_layers']}")
    print(f"  神经元数: {best_config['hidden_neurons']}")
    print(f"  参数量: {best_config['total_params']}")
    print(f"  验证损失: {best_config['final_val_loss']:.6f}")
    print("=" * 80)
    
    return best_config


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='神经网络架构探索')
    
    # 探索范围
    parser.add_argument('--pixels', type=int, nargs='+', default=[1, 2, 4, 8, 16],
                        help='测试的像素数列表 (默认: 1 2 4 8 16)')
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='测试的隐藏层数列表 (默认: 0 1 2 3)')
    parser.add_argument('--neurons', type=int, nargs='+', default=[4, 8, 16],
                        help='测试的神经元数列表 (默认: 4 8 16)')
    
    # 训练参数
    parser.add_argument('--samples', type=int, default=1000,
                        help='每个配置的训练样本数 (默认: 1000)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='每个配置的训练轮数 (默认: 50)')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='exploration_results',
                        help='结果输出目录 (默认: exploration_results)')
    
    args = parser.parse_args()
    
    # 运行探索
    results, results_file = grid_search_architecture(
        num_pixels_range=args.pixels,
        hidden_layers_range=args.layers,
        hidden_neurons_range=args.neurons,
        num_samples=args.samples,
        epochs=args.epochs,
        output_dir=args.output_dir
    )
    
    # 分析结果
    best_config = analyze_results(results, output_dir=args.output_dir)
    
    print("\n\n探索完成! 🎉")
    print(f"结果文件: {results_file}")
    print(f"可视化图: {args.output_dir}/exploration_analysis.png")
    print(f"对比表: {args.output_dir}/comparison_table.txt")
