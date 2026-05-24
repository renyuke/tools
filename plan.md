# 时间序列预测学习项目计划

## Context

用户正在学习时间序列预测，需要：用过去两年（24个月）的每月工作量数据来预测接下来两个月的工作量。目前项目目录为空，没有数据，需要从零搭建一个完整的学习项目。

## 方案概述

在项目目录下创建 Python virtualenv，使用 Jupyter Notebook 的形式提供完整的交互式学习体验，包含：模拟数据生成 → 多种预测方法 → 误差验证 → 可视化。

## 技术选型

- **Python 版本**: 3.11（`C:\Python\Python311\python.exe`）
- **环境**: 项目目录内创建 `.venv` 虚拟环境
- **核心依赖**: pandas, numpy, matplotlib, statsmodels, scikit-learn, pmdarima, jupyter
- **交付形式**: Jupyter Notebook（.ipynb），含 markdown 讲解和代码

## 计划步骤

### Step 1: 环境搭建
- 确认 Python 版本（3.11 or 3.13）
- 创建 `.venv` 虚拟环境
- 安装依赖包
- 创建 `.gitignore`

### Step 2: 创建模拟数据
- 生成 24 个月（2024-06 ~ 2026-05）的每月工作量数据
- 包含三种成分：**趋势**（缓慢增长）+ **季节**（每年周期波动）+ **噪声**（随机扰动）
- 数据保存为 CSV 文件，方便查看

### Step 3: 实现预测方法

在 Notebook 中依次实现以下方法，每个方法都有 markdown 解释：

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| 1. **Naive 季节法** | 用去年同期的值作为预测 | 基线对比 |
| 2. **移动平均 (SMA)** | 用过去 N 个月的均值 | 平滑随机波动 |
| 3. **指数平滑 (Holt-Winters)** | 考虑趋势+季节性的加权平滑 | 有趋势和季节的数据 |
| 4. **ARIMA/SARIMA** | 差分自回归移动平均模型 | 经典统计方法 |
| 5. **线性回归** | 用月份序号+季节虚拟变量做特征 | 简单 ML 方法 |

### Step 4: 误差验证

使用 **最后 2 个月作为验证集**（模拟"预测未来"的场景），计算三种误差：

- **MAE** (Mean Absolute Error) — 平均绝对误差
- **RMSE** (Root Mean Square Error) — 均方根误差（对大误差更敏感）
- **MAPE** (Mean Absolute Percentage Error) — 平均绝对百分比误差（直观）

可视化对比：实际值 vs 预测值折线图

### Step 5: 用最优方法预测未来 2 个月

- 基于验证集误差选择最优模型
- 用全部 24 个月数据重新训练
- 预测 2026-06、2026-07 两个月的值

## 关键文件

| 文件 | 说明 |
|------|------|
| `data/workload.csv` | 模拟的每月工作量数据 |
| `notebooks/time_series_forecast.ipynb` | 主 Notebook，包含所有内容 |
| `.venv/` | Python 虚拟环境 |
| `.gitignore` | Git 忽略规则 |

## 验证方式

1. 运行 Notebook 所有 cell，确认无报错
2. 检查可视化图表是否正确展示数据趋势和预测结果
3. 验证误差指标计算正确
4. 确认 future_predictions 输出为 2026-06 和 2026-07 两个值

## 用户已确认

- **Python**: 3.11
- **交付形式**: Jupyter Notebook
- **数据**: 模拟工作量数据（单位：小时/月），含趋势+季节+噪声
