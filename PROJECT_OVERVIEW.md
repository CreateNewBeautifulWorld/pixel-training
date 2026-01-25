# 🎯 Uint8量化神经网络 - 完整工具集

## 📦 项目包含3个版本

### 版本1️⃣: 简单版 (入门学习)
**文件**: `train_simple.py`

```bash
python train_simple.py
```

- ✅ 最简单，快速上手
- ✅ 固定结构：48→16→3
- ✅ 5分钟完成训练
- 🎯 适合：初学者、快速验证

---

### 版本2️⃣: 可配置版 (实战部署)
**文件**: `train_configurable.py`

```bash
python train_configurable.py \
    --num_pixels 16 \
    --hidden_layers 32 16 \
    --num_classes 3 \
    --epochs 200
```

- ✅ 完全自定义深度/宽度
- ✅ 训练过程可视化
- ✅ 命令行参数控制
- 🎯 适合：实际项目、需要调优

---

### 版本3️⃣: 架构探索版 (最优结构) ⭐ 新增
**文件**: `explore_architecture.py`

```bash
python explore_architecture.py \
    --pixels 1 2 4 8 16 32 64 \
    --layers 0 1 2 3 4 5 \
    --neurons 4 8 16 32 \
    --epochs 100
```

- ✅ 自动测试多种配置
- ✅ FP16快速训练
- ✅ 找到最优网络结构
- ✅ 可视化对比所有配置
- 🎯 适合：不知道用什么结构、需要找全局最优

---

## 🔄 推荐工作流程

### 场景A: 我知道要什么结构
```bash
# 直接用可配置版训练
python train_configurable.py \
    --num_pixels 16 \
    --hidden_layers 16 \
    --epochs 200
```

---

### 场景B: 我不知道用什么结构 ⭐
```bash
# Step 1: 探索找最优
python explore_architecture.py \
    --pixels 4 8 16 \
    --layers 0 1 2 \
    --neurons 8 16 \
    --epochs 50

# Step 2: 查看结果
cat exploration_results/comparison_table.txt

# Step 3: 用最优配置训练
# 假设最优是: 24 -> 16 -> 1
python train_configurable.py \
    --num_pixels 8 \
    --hidden_layers 16 \
    --epochs 200
```

---

### 场景C: 我只是想快速了解
```bash
# 用简单版
python train_simple.py
```

---

## 📊 功能对比表

| 功能 | 简单版 | 可配置版 | 探索版 |
|------|--------|---------|--------|
| **难度** | ⭐ | ⭐⭐ | ⭐⭐⭐ |
| **灵活性** | 固定 | 可配置 | 自动探索 |
| **训练曲线** | ❌ | ✅ | ✅ |
| **多配置对比** | ❌ | ❌ | ✅ |
| **找最优结构** | ❌ | 手动 | 自动 |
| **适合场景** | 学习 | 实战 | 研究 |
| **运行时间** | 1分钟 | 5分钟 | 30分钟-数小时 |

---

## 📁 完整文件列表

```
项目根目录/
│
├─ 训练脚本（3个版本）
│  ├─ train_simple.py              ← 简单版
│  ├─ train_configurable.py        ← 可配置版
│  └─ explore_architecture.py      ← 探索版 ⭐新增
│
├─ 推理和部署
│  ├─ inference_uint8.cpp          ← C++推理引擎
│  ├─ uint8_nn_rtl.v               ← Verilog RTL设计
│  └─ Makefile                     ← 编译脚本
│
├─ 文档（按版本）
│  ├─ QUICKSTART.md                ← 快速入门（简单版）
│  ├─ README.md                    ← 项目总览
│  ├─ CONFIG_GUIDE.md              ← 配置指南（可配置版）
│  ├─ COMPARISON.md                ← 配置对比（可配置版）
│  ├─ NEW_FEATURES.md              ← 新功能说明
│  ├─ EXPLORATION_GUIDE.md         ← 探索指南 ⭐新增
│  └─ WORKFLOW_SUMMARY.md          ← 工作流程总结 ⭐新增
│
├─ 示例结果
│  ├─ training_history.png         ← 标准配置训练曲线
│  ├─ training_shallow.png         ← 浅层配置训练曲线
│  ├─ training_deep.png            ← 深层配置训练曲线
│  ├─ exploration_analysis.png     ← 探索结果可视化 ⭐新增
│  └─ comparison_table.txt         ← 配置对比表 ⭐新增
│
└─ 其他
   ├─ 使用说明.txt                 ← 中文说明
   ├─ 下载说明.md                  ← 下载指引
   └─ 项目结构.md                  ← 架构说明
```

---

## 🎯 按需求选择版本

### 我想...

**快速了解项目**
→ 用简单版：`python train_simple.py`

**训练自己的网络结构**
→ 用可配置版：`python train_configurable.py --hidden_layers 32 16`

**不知道用什么结构，想找最优**
→ 用探索版：`python explore_architecture.py`

**看训练过程曲线**
→ 用可配置版或探索版（都生成PNG图）

**对比多种配置**
→ 用探索版（自动对比）

**部署到硬件**
→ 先用探索版找最优 → 再用可配置版训练 → 最后C++推理

---

## 💡 典型使用场景

### 场景1: 学生作业
```bash
# 用简单版，5分钟搞定
python train_simple.py
make
./inference_uint8

# 交作业 ✅
```

---

### 场景2: 公司项目（已知结构）
```bash
# 用可配置版
python train_configurable.py \
    --num_pixels 16 \
    --hidden_layers 32 16 \
    --epochs 300

# 查看训练曲线
open training_history.png

# 部署
make && ./inference_uint8
```

---

### 场景3: 研究项目（需要最优）
```bash
# 第1天：粗探索
python explore_architecture.py \
    --pixels 4 8 16 32 \
    --layers 0 1 2 3 \
    --neurons 8 16 32 \
    --epochs 50

# 第2天：查看结果，细探索
cat exploration_results/comparison_table.txt
# 发现16像素+2层最好

python explore_architecture.py \
    --pixels 12 16 20 \
    --layers 1 2 3 \
    --neurons 12 16 20 \
    --epochs 100

# 第3天：用最优配置训练
python train_configurable.py \
    --num_pixels 16 \
    --hidden_layers 16 16 \
    --epochs 500

# 发论文 ✅
```

---

## 🚀 快速开始指南

### 第1次使用（10分钟）

```bash
# 1. 简单版快速体验
python train_simple.py

# 2. 编译运行
make
./inference_uint8

# 3. 看效果
# 显示准确率100%，成功！
```

---

### 第2次使用（30分钟）

```bash
# 1. 可配置版尝试不同配置
python train_configurable.py --hidden_layers 32 16

# 2. 查看训练曲线
open training_history.png

# 3. 调整参数再试
python train_configurable.py --hidden_layers 64 32
```

---

### 第3次使用（2小时）

```bash
# 1. 探索版找最优
python explore_architecture.py \
    --pixels 4 8 16 \
    --layers 1 2 3 \
    --neurons 8 16 \
    --epochs 100

# 2. 分析结果
cat exploration_results/comparison_table.txt
open exploration_results/exploration_analysis.png

# 3. 用最优配置部署
python train_configurable.py [最优配置]
make
./inference_uint8
```

---

## 📚 文档阅读顺序

### 初学者路径
1. QUICKSTART.md - 5分钟快速入门
2. README.md - 理解项目
3. 运行简单版 - 动手实践

### 进阶用户路径
1. CONFIG_GUIDE.md - 学习配置
2. COMPARISON.md - 对比分析
3. 运行可配置版 - 尝试调优

### 研究者路径
1. EXPLORATION_GUIDE.md - 探索方法
2. WORKFLOW_SUMMARY.md - 完整流程
3. 运行探索版 - 找全局最优

---

## ⚙️ 系统要求

### 必需
- Python 3.6+
- numpy
- matplotlib (用于可视化)

### 可选
- g++ (用于C++推理)
- Verilog仿真器 (用于RTL)

### 安装
```bash
pip install numpy matplotlib
```

---

## 🎉 新增功能总结

### ⭐ 探索版本 (v3.0)

**新增内容**：
1. `explore_architecture.py` - 架构探索脚本
2. `EXPLORATION_GUIDE.md` - 详细使用指南
3. `WORKFLOW_SUMMARY.md` - 工作流程总结
4. 示例探索结果和可视化图

**核心功能**：
- 自动测试多种配置（像素数×层数×神经元数）
- FP16快速训练
- 回归任务（输出单个float）
- 可视化对比所有配置
- 自动找出最优网络结构

**适用场景**：
- 不确定用什么网络结构
- 需要找到全局最优配置
- 研究不同结构的性能
- 资源受限需要最优性价比

---

## 📊 版本演进

```
v1.0 (简单版)
├─ 固定结构
├─ 快速入门
└─ 学习原理

v2.0 (可配置版)
├─ v1.0所有功能
├─ 自定义结构
├─ 训练可视化
└─ 命令行配置

v3.0 (探索版) ← 当前
├─ v2.0所有功能
├─ 自动探索
├─ 多配置对比
├─ 最优结构推荐
└─ FP16训练
```

---

## 💬 常见问题

**Q: 三个版本可以混用吗？**  
A: 可以！它们输出格式兼容，C++推理引擎通用。

**Q: 探索版本很慢怎么办？**  
A: 减少配置数，或者先粗搜索再细搜索。

**Q: 我应该用哪个版本？**  
A: 
- 学习 → 简单版
- 已知结构 → 可配置版  
- 找最优 → 探索版

**Q: 探索版的FP16和部署的Uint8不同怎么办？**  
A: 探索只是找结构，找到后用Uint8重新训练该结构。

**Q: 能直接跳过探索版吗？**  
A: 可以，但可能错过更优配置。探索版帮你科学地找最优解。

---

## 🎯 总结

你现在拥有**完整的工具链**：

1. **简单版** - 入门学习 ✅
2. **可配置版** - 灵活训练 ✅
3. **探索版** - 找最优结构 ✅
4. **C++推理** - 高效部署 ✅
5. **RTL设计** - 硬件加速 ✅

**完整工作流**：
```
探索 → 找最优结构
  ↓
配置训练 → 用最优结构训练
  ↓
量化 → Uint8
  ↓
部署 → C++ / RTL
```

开始你的神经网络之旅吧！🚀
