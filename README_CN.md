# K-ISA 项目

## 项目概述

K-ISA 是一个高性能向量计算库，专注于实现高效的数据分析和机器学习算法。该项目提供了一系列优化的向量操作和算法实现，尤其适用于金融数据分析和简化的LLM（大型语言模型）计算。

## 主要功能

- **向量操作基础库**：提供高效的向量运算支持
- **订单簿事件分析**：使用经验动态建模(EDM)分析订单簿事件的泊松到达率
- **简化的LLM计算**：实现包括矩阵-向量乘法、注意力机制、FFT处理等在内的简化LLM前向传播计算
- **经验动态建模(EDM)**：用于时间序列预测和非线性系统分析

## 安装指南

### 前提条件

- C编译器 (gcc/clang)
- 数学库支持 (libm)
- 对于ARM架构，支持NEON指令集

### 编译安装

```bash
git clone https://github.com/yourusername/kisa.git
cd kisa
mkdir build && cd build
cmake ..
make
make install
```

## 使用示例

### 1. 订单簿事件分析

```bash
# 基本使用方式
./orderbook_edm_analysis

# 使用自定义参数
./orderbook_edm_analysis --events 2000 --window 200 --dimension 4 --threads 8
```

该示例会分析订单簿事件的到达率，计算自相关性，执行经验动态建模预测，并输出详细的分析结果。

### 2. LLM计算示例

```bash
./llm_calc
```

这个示例展示了简化的LLM前向传播计算，包括注意力机制、位置编码、层归一化等关键组件。

## 项目结构

```
kisa/
├── include/             # 头文件
│   └── kisa.h           # 主要API定义
├── src/                 # 源代码
│   ├── core/            # 核心功能实现
│   ├── apps/            # 应用程序
│   │   ├── orderbook_edm_analysis.c  # 订单簿分析应用
│   │   └── llm_calc.c   # LLM计算应用
├── examples/            # 示例代码
├── tests/               # 测试文件
└── docs/                # 文档

```

## 订单簿EDM分析详解

`orderbook_edm_analysis.c` 程序模拟订单簿事件流，并使用EDM分析不同类型事件的到达率和模式：

1. 新订单事件 (New)
2. 修改订单事件 (Modify)
3. 删除订单事件 (Delete)
4. 交易执行事件 (Execute)

该程序使用EDM来预测未来事件率并分析事件之间的动态关系。它能够处理实际数据或生成模拟数据，提供详细的相关性分析和非线性特性分析。

## LLM计算详解

`llm_calc.c` 实现了简化的LLM前向传播计算，包括：

1. 矩阵-向量乘法（模拟线性层）
2. 向量激活函数（使用ReLU）
3. 多头注意力机制
4. 使用FFT进行高级序列处理（频域卷积和特征提取）
5. 层归一化
6. 位置编码
7. 经验动态建模(EDM)

## 性能优化

K-ISA针对不同架构提供了优化：

- 对于ARM架构，使用NEON SIMD指令集
- 对于通用架构，提供标准C实现
- 支持多线程并行计算

## 贡献指南

我们欢迎任何形式的贡献，包括但不限于：

- 报告问题
- 提交功能请求
- 提交代码改进
- 完善文档

请通过Pull Request或Issue与我们交流。

## 许可证

本项目采用 [MIT 许可证](LICENSE) 进行授权。 