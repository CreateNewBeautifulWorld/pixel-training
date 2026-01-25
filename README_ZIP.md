# 📦 Uint8量化神经网络 - 完整工具包

**文件**: `uint8_neural_network_complete.zip`  
**大小**: 369 KB  
**文件数**: 25个  
**版本**: v3.0 - 架构探索版

---

## 🎯 这个包是做什么的？

这是一个**完整的神经网络工具集**，从探索最优结构到硬件部署的全流程工具：

1. **探索阶段** - 用FP16自动找最优网络结构
2. **训练阶段** - 用找到的最优结构训练模型
3. **部署阶段** - 量化到Uint8并部署到C++/硬件

---

## 📁 包含文件清单

### 🐍 Python脚本（3个）

1. **explore_architecture.py** ⭐ 新增
   - 架构探索工具（FP16）
   - 自动测试多种配置
   - 找到最优网络结构
   ```bash
   python explore_architecture.py \
       --pixels 1 2 4 8 16 32 64 \
       --layers 0 1 2 3 4 5 \
       --neurons 1 2 4 8 16 32 \
       --epochs 100
   ```

2. **train_configurable.py**
   - 可配置训练脚本
   - 支持自定义深度/宽度
   - 生成训练曲线图
   ```bash
   python train_configurable.py \
       --num_pixels 16 \
       --hidden_layers 32 16 \
       --epochs 200
   ```

3. **train_simple.py**
   - 简单固定版本
   - 快速入门学习
   ```bash
   python train_simple.py
   ```

---

### ⚙️ C++/硬件部署（3个）

1. **inference_uint8.cpp** - C++推理引擎
2. **uint8_nn_rtl.v** - Verilog RTL设计
3. **Makefile** - 编译脚本

```bash
make
./inference_uint8
```

---

### 📚 文档（10个）

**入门文档**:
1. **PROJECT_OVERVIEW.md** ⭐ 先看这个！
   - 项目总览
   - 三个版本对比
   - 快速开始指南

2. **QUICKSTART.md**
   - 5分钟快速入门
   - 简单版使用说明

**探索版文档** ⭐ 核心：
3. **EXPLORATION_GUIDE.md**
   - 架构探索详细指南
   - 参数说明
   - 使用示例

4. **WORKFLOW_SUMMARY.md**
   - 完整工作流程
   - 从探索到部署
   - 实际案例分析

**可配置版文档**:
5. **CONFIG_GUIDE.md**
   - 可配置版使用指南
   - 配置示例
   - 优化建议

6. **COMPARISON.md**
   - 不同配置对比分析
   - 性能测试结果

7. **NEW_FEATURES.md**
   - 新功能说明
   - 版本演进

**其他文档**:
8. **README.md** - 项目说明
9. **使用说明.txt** - 中文简明说明
10. **下载说明.md** - 下载和安装指引
11. **项目结构.md** - 文件结构说明

---

### 📊 示例结果（6个）

**训练曲线图**:
1. **training_history.png** - 标准配置（48→16→3）
2. **training_shallow.png** - 浅层配置（12→8→3）
3. **training_deep.png** - 深层配置（48→32→16→3）

**探索结果** ⭐:
4. **exploration_analysis.png** - 12个配置可视化对比
5. **comparison_table.txt** - 配置排名表

**数据文件**:
6. **train_data.txt** - 示例训练数据（1000样本）
7. **weights_uint8.bin** - 预训练权重（二进制）
8. **weights_uint8.bin.txt** - 预训练权重（文本）

---

## 🚀 3步快速开始

### 第1步：解压
```bash
unzip uint8_neural_network_complete.zip
cd uint8_neural_network_complete/
```

### 第2步：安装依赖
```bash
pip install numpy matplotlib
```

### 第3步：选择你的路径

**路径A - 快速学习**（5分钟）
```bash
python train_simple.py
make
./inference_uint8
```

**路径B - 探索最优结构**（30分钟-数小时）⭐
```bash
# 快速探索
python explore_architecture.py \
    --pixels 4 8 16 \
    --layers 0 1 2 \
    --neurons 8 16 \
    --epochs 50

# 查看结果
cat exploration_results/comparison_table.txt
open exploration_results/exploration_analysis.png
```

**路径C - 直接训练部署**（15分钟）
```bash
python train_configurable.py \
    --num_pixels 16 \
    --hidden_layers 16 \
    --epochs 200

make
./inference_uint8
```

---

## 📖 推荐阅读顺序

### 首次使用
1. 📄 **PROJECT_OVERVIEW.md** - 了解整体
2. 📘 **QUICKSTART.md** - 快速上手
3. 💻 运行 `train_simple.py` - 动手实践

### 使用探索功能
1. 📗 **EXPLORATION_GUIDE.md** - 学习探索方法
2. 📊 **WORKFLOW_SUMMARY.md** - 理解完整流程
3. 🔍 运行 `explore_architecture.py` - 找最优结构

### 深入使用
1. 📕 **CONFIG_GUIDE.md** - 配置详解
2. 📈 **COMPARISON.md** - 性能分析
3. ⚙️ 部署到硬件

---

## 🎯 核心特性

### ⭐ 架构探索（新增）
- ✅ 自动测试多种配置（像素数×层数×神经元数）
- ✅ FP16快速训练
- ✅ 回归输出（单float）
- ✅ 可视化对比所有配置
- ✅ 自动找出最优结构

### 📐 灵活配置
- ✅ 可配置深度（0-5层）
- ✅ 可配置宽度（1-32神经元）
- ✅ 可配置输入（1-64像素）
- ✅ 训练过程可视化

### 🔧 完整部署
- ✅ Uint8量化
- ✅ C++推理引擎
- ✅ Verilog RTL设计
- ✅ 硬件友好

---

## 💡 典型应用场景

### 场景1：研究项目
```
探索 → 找最优 → 训练 → 发论文
```
使用：`explore_architecture.py` + `train_configurable.py`

### 场景2：工业部署
```
探索 → 找最优 → 训练 → 量化 → C++/RTL部署
```
使用：完整工具链

### 场景3：学习神经网络
```
简单版 → 理解原理 → 可配置版 → 探索版
```
使用：从简单到复杂，逐步深入

---

## 🔍 探索功能详解

### 你的需求配置
```bash
python explore_architecture.py \
    --pixels 1 2 4 8 16 32 64 \     # 输入：1-64像素
    --layers 0 1 2 3 4 5 \          # 深度：0-5层
    --neurons 1 2 4 8 16 32 \       # 宽度：1-32神经元
    --epochs 100 \                  # 训练轮数
    --samples 2000                  # 样本数
```

**配置数**: 7 × 6 × 6 = **252个配置**  
**时间**: 6-10小时  
**输出**: JSON结果 + 可视化图 + 排名表

### 输出示例
```
Top 3配置：
1. 24->8->8->1    (8像素, 2层, 8神经元)  Loss: 0.019  参数: 281
2. 24->16->16->1  (8像素, 2层, 16神经元) Loss: 0.026  参数: 689
3. 12->8->1       (4像素, 1层, 8神经元)  Loss: 0.027  参数: 113
```

---

## 🛠️ 系统要求

### 必需
- Python 3.6+
- numpy
- matplotlib

### 可选
- g++ (C++推理)
- Verilog仿真器 (RTL验证)

### 安装
```bash
pip install numpy matplotlib
```

---

## 📊 性能指标

| 工具 | 时间 | 难度 | 功能 |
|-----|------|------|------|
| train_simple.py | 1分钟 | ⭐ | 固定训练 |
| train_configurable.py | 5分钟 | ⭐⭐ | 灵活训练 |
| explore_architecture.py | 30分钟-数小时 | ⭐⭐⭐ | 自动探索 |

---

## 🎓 学习路径

```
第1天：train_simple.py
      ↓ 理解基础

第2天：train_configurable.py
      ↓ 学习配置

第3天：explore_architecture.py
      ↓ 自动探索

第4天：部署到硬件
      ↓ 完整项目
```

---

## 💬 常见问题

**Q: 从哪个文件开始？**  
A: 先看 `PROJECT_OVERVIEW.md`，然后选择合适的工具。

**Q: 探索需要多久？**  
A: 快速探索30分钟，完整探索6-10小时。建议分阶段。

**Q: FP16和Uint8有什么区别？**  
A: FP16用于快速探索，Uint8用于硬件部署。探索找结构，部署用Uint8。

**Q: 如何选择最优配置？**  
A: 看 `comparison_table.txt`，选验证损失最低且参数量合适的。

**Q: 能否跳过探索直接部署？**  
A: 可以，但可能不是最优。探索帮你科学地找最优结构。

---

## 📞 技术支持

- 📖 详细文档在各个 .md 文件中
- 💡 示例结果在 .png 和 .txt 文件中
- 🔍 代码中有详细注释

---

## 🎉 总结

这个工具包提供：

✅ **3种训练方式**（简单/可配置/探索）  
✅ **完整文档**（10个详细文档）  
✅ **示例结果**（训练曲线+探索对比）  
✅ **部署工具**（C++ + RTL）  
✅ **从探索到部署的完整流程**  

**立即开始你的神经网络项目吧！** 🚀

---

**版本**: v3.0  
**更新**: 2025-01-25  
**许可**: MIT License
