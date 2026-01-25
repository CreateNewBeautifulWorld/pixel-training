# 🔍 神经网络架构探索指南

## 📋 两阶段方法论

### 阶段1️⃣: 探索 (FP16)
**目标**: 找到最优网络结构  
**方法**: Grid Search / Random Search  
**精度**: FP16（快速，精度够用）  
**输出**: 回归任务（单个float输出）  

### 阶段2️⃣: 收敛 (Uint8)
**目标**: 量化部署到硬件  
**方法**: 用阶段1找到的最优结构  
**精度**: Uint8量化  
**输出**: 适合硬件的格式  

---

## 🚀 快速开始

### 最小探索（快速测试）
```bash
# 测试3种像素数 × 3种层数 × 2种神经元 = 18个配置
python explore_architecture.py \
    --pixels 4 8 16 \
    --layers 0 1 2 \
    --neurons 8 16 \
    --epochs 50
```
**预计时间**: 5-10分钟  
**适合**: 快速验证

---

### 中等探索（推荐）
```bash
# 测试5种像素数 × 4种层数 × 3种神经元 = 60个配置
python explore_architecture.py \
    --pixels 1 2 4 8 16 \
    --layers 0 1 2 3 \
    --neurons 4 8 16 \
    --epochs 50
```
**预计时间**: 30-60分钟  
**适合**: 一般项目

---

### 完整探索（详尽）
```bash
# 测试7种像素数 × 6种层数 × 4种神经元 = 168个配置
python explore_architecture.py \
    --pixels 1 2 4 8 16 32 64 \
    --layers 0 1 2 3 4 5 \
    --neurons 4 8 16 32 \
    --epochs 100 \
    --samples 2000
```
**预计时间**: 3-5小时  
**适合**: 重要项目，需要找到全局最优

---

## 📊 参数说明

### 探索范围参数

**--pixels**  
测试的像素数列表
```bash
--pixels 1 2 4 8 16 32 64

# 说明：
# 1像素  = 3输入 (RGB)      → 极小网络
# 4像素  = 12输入            → 小网络
# 16像素 = 48输入            → 中等网络
# 64像素 = 192输入           → 大网络
```

**--layers**  
测试的隐藏层数列表
```bash
--layers 0 1 2 3 4 5

# 说明：
# 0层：直接 input → output  → 线性模型
# 1层：input → hidden → output
# 2层：input → h1 → h2 → output
# 3+层：深度网络
```

**--neurons**  
测试的每层神经元数列表
```bash
--neurons 4 8 16 32

# 说明：
# 4：  窄网络，最省资源
# 8：  小网络
# 16： 中等网络
# 32： 宽网络
```

### 训练参数

**--samples**  
每个配置的训练样本数
```bash
--samples 1000  # 默认，快速
--samples 2000  # 更准确
--samples 5000  # 最准确
```

**--epochs**  
每个配置的训练轮数
```bash
--epochs 50   # 快速探索
--epochs 100  # 标准
--epochs 200  # 充分训练
```

**--output_dir**  
结果输出目录
```bash
--output_dir exploration_results  # 默认
--output_dir my_exploration       # 自定义
```

---

## 📁 输出文件

运行后会生成：

```
exploration_results/
├── exploration_results_YYYYMMDD_HHMMSS.json  ← 详细结果（JSON）
├── exploration_analysis.png                   ← 可视化分析
└── comparison_table.txt                       ← 对比表格
```

### 1. JSON结果文件
包含每个配置的详细信息：
```json
{
  "config_id": 1,
  "num_pixels": 4,
  "input_dim": 12,
  "num_hidden_layers": 1,
  "hidden_neurons": 8,
  "layer_sizes": [12, 8, 1],
  "total_params": 105,
  "final_train_loss": 0.0123,
  "final_val_loss": 0.0156,
  "train_history": [...],
  "val_history": [...]
}
```

### 2. 可视化分析图
包含4个子图：
- **左上**: 参数量 vs 验证损失
- **右上**: 隐藏层数 vs 验证损失
- **左下**: 输入维度 vs 验证损失
- **右下**: Top 5配置训练曲线

### 3. 对比表格
排序后的所有配置对比：
```
排名   网络结构              像素  层数  神经元  参数量    验证损失
1      12 -> 8 -> 1         4     1     8       105      0.001234
2      24 -> 16 -> 1        8     1     16      417      0.001456
...
```

---

## 📊 实际示例

### 示例1: 快速探索
```bash
python explore_architecture.py \
    --pixels 4 8 \
    --layers 0 1 \
    --neurons 8 \
    --epochs 30
```

**配置数**: 2 × 2 × 1 = 4  
**时间**: ~2分钟  

**可能结果**:
```
Top 3配置:
1. 12 -> 8 -> 1    (Val Loss: 0.0012)
2. 24 -> 8 -> 1    (Val Loss: 0.0015)
3. 12 -> 1         (Val Loss: 0.0023)
```

---

### 示例2: 标准探索
```bash
python explore_architecture.py \
    --pixels 2 4 8 16 \
    --layers 0 1 2 \
    --neurons 8 16 \
    --epochs 50
```

**配置数**: 4 × 3 × 2 = 24  
**时间**: ~15分钟  

**可能发现**:
- 0层（线性）通常不够好
- 1层+16神经元 vs 2层+8神经元差不多
- 输入太小（2像素）限制性能

---

### 示例3: 深入探索
```bash
python explore_architecture.py \
    --pixels 1 2 4 8 16 32 \
    --layers 0 1 2 3 \
    --neurons 4 8 16 32 \
    --epochs 100
```

**配置数**: 6 × 4 × 4 = 96  
**时间**: ~2小时  

**可能发现**:
- Sweet spot: 8像素 + 2层 + 16神经元
- 超过3层收益递减
- 32神经元参数量翻倍但提升不大

---

## 🎯 如何解读结果

### 查看Top配置
```bash
# 运行完成后，查看对比表
cat exploration_results/comparison_table.txt | head -20
```

**关注点**:
1. **验证损失**: 越小越好
2. **参数量**: 在满足精度下越小越好
3. **过拟合**: train_loss << val_loss 说明过拟合

---

### 可视化分析

**图1: 参数量 vs 损失**
- 寻找"拐点"：损失不再明显下降的参数量
- 最小参数量达到可接受损失

**图2: 层数 vs 损失**
- 看哪个层数最优
- 通常1-3层足够

**图3: 输入维度 vs 损失**
- 输入太小会限制性能
- 输入太大浪费资源

**图4: 训练曲线**
- Top 5配置的收敛情况
- 平滑下降 = 好
- 震荡 = 可能需要调参

---

## 💡 优化建议

### 如果所有配置损失都很高

**原因**: 任务太难或数据质量差  

**解决**:
1. 增加训练样本: `--samples 5000`
2. 增加训练轮数: `--epochs 200`
3. 检查数据生成函数

---

### 如果大网络和小网络损失接近

**原因**: 任务简单，小网络就够用  

**建议**: 选择最小的满足要求的网络

---

### 如果深层网络没有提升

**原因**: 任务不需要深层特征  

**建议**: 用浅层网络（1-2层）

---

## 🔄 探索后的下一步

### 1. 记录最佳配置
```bash
# 查看最佳配置
cat exploration_results/comparison_table.txt | head -5

# 例如发现最优: 48 -> 16 -> 1
```

### 2. 用最佳配置训练Uint8版本
```bash
# 使用之前的可配置脚本
python train_configurable.py \
    --num_pixels 16 \
    --hidden_layers 16 \
    --num_classes 1 \     # 注意：改为回归
    --epochs 200
```

### 3. 量化部署
```bash
# 编译C++推理
make

# 部署到硬件
```

---

## 📈 预期结果模式

### 典型发现

**输入规模**:
- 1-2像素: 信息不足，损失高
- 4-8像素: 性价比高
- 16-32像素: 性能好，资源多
- 64像素: 边际收益递减

**隐藏层数**:
- 0层: 只能拟合线性关系
- 1层: 可以拟合大部分简单非线性
- 2-3层: 复杂任务
- 4+层: 通常过拟合

**神经元数**:
- 4-8: 简单任务
- 16: 大部分任务的sweet spot
- 32+: 除非数据量大，否则过拟合

---

## ⚡ 加速技巧

### 并行探索（手动）
```bash
# 终端1
python explore_architecture.py --pixels 1 2 4 --layers 0 1 \
    --output_dir exp1

# 终端2
python explore_architecture.py --pixels 8 16 32 --layers 2 3 \
    --output_dir exp2

# 然后合并结果
```

### 粗搜索 + 细搜索
```bash
# 第1步：粗搜索
python explore_architecture.py \
    --pixels 1 4 16 64 \
    --layers 0 1 2 \
    --neurons 8 32 \
    --epochs 30

# 查看结果，发现16像素+1层+8神经元最好

# 第2步：细搜索
python explore_architecture.py \
    --pixels 12 16 20 \
    --layers 1 \
    --neurons 6 8 10 12 \
    --epochs 100
```

---

## 🎓 完整工作流程

```bash
# Step 1: 快速探索（找大致范围）
python explore_architecture.py \
    --pixels 4 8 16 \
    --layers 0 1 2 \
    --neurons 8 16 \
    --epochs 30

# Step 2: 查看结果
cat exploration_results/comparison_table.txt | head -10

# Step 3: 细化探索（假设发现8像素最好）
python explore_architecture.py \
    --pixels 6 8 10 \
    --layers 1 2 \
    --neurons 12 16 20 \
    --epochs 100

# Step 4: 选定最优配置，用Uint8训练
# 假设最优是: 24 -> 16 -> 1
python train_configurable.py \
    --num_pixels 8 \
    --hidden_layers 16 \
    --epochs 200

# Step 5: 部署
make
./inference_uint8
```

---

## 📝 注意事项

1. **FP16 vs Uint8**: 探索用FP16，部署用Uint8
2. **回归 vs 分类**: 探索是回归（输出float），部署可能是分类
3. **数据一致性**: 确保探索和部署用相同的数据分布
4. **过拟合**: 关注train loss vs val loss差距
5. **随机性**: 多次运行取平均更可靠

---

## 🔍 调试技巧

### 如果程序崩溃
```bash
# 减少配置数
python explore_architecture.py \
    --pixels 4 \
    --layers 1 \
    --neurons 8 \
    --epochs 10
```

### 如果内存不足
```bash
# 减少样本数
python explore_architecture.py --samples 500
```

### 如果想看详细训练过程
```bash
# 修改代码，将verbose=True
# 或减少配置数以便观察
```

---

## 🎉 总结

**探索脚本做什么**:
✅ 自动测试多种网络配置  
✅ 使用FP16快速训练  
✅ 记录所有结果  
✅ 可视化对比  
✅ 找出最优配置  

**你需要做什么**:
1. 运行探索脚本
2. 查看结果和图表
3. 选择最优配置
4. 用选定配置训练Uint8版本
5. 部署到硬件

**预期收益**:
- 🎯 找到最优网络结构
- ⚡ 避免盲目试错
- 📊 数据驱动的决策
- 💰 节省开发时间

开始你的探索之旅吧！🚀
