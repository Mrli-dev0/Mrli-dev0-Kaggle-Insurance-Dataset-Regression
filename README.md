# Regression with an Insurance Dataset


[![GitHub license](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](https://github.com/Mrli-dev0/Mrli-dev0-Kaggle-Insurance-Dataset-Regression/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen?style=flat-square)](https://www.python.org/)
[![Score](https://img.shields.io/badge/Score-0.89122-red?style=flat-square)]()
[![Rank](https://img.shields.io/badge/Rank-1st%20(Top%201)-yellow?style=flat-square)]()
[![Task](https://img.shields.io/badge/Task-Binary%20Classification-purple?style=flat-square)]()


# 🏆 保险保费预测比赛第三名解决方案

## 📋 比赛概况

- **比赛名称**：Regression with an Insurance Dataset

- **比赛链接**：https://www\.kaggle\.com/competitions/sch2\-reg\-2026\-d3\-3

- **任务类型**：保险保费金额回归预测

- **评估指标**：RMSLE（均方根对数误差）

- **最终得分**：1\.11008

- **最终排名**：第 3 名（Top 3）

## 📁 项目结构

```bash
.
├── data/                # 数据集文件夹
│   ├── train.csv        # 训练集
│   ├── test.csv         # 测试集
│   └── sample_submission.csv  # 提交示例
├── images/              # EDA 可视化图表文件夹
├── src/                 # 源代码
│   ├── train.py         # 训练脚本
│   ├── infer.py         # 推理脚本
│   └── utils.py         # 工具函数
├── 01_eda.ipynb         # 数据分析
├── 02_model.ipynb       # 模型训练
├── submission.csv       # 最终提交结果
└── README.md            # 说明文档
```

## 🚀 快速运行

```bash
# 训练模型（10折交叉验证 + 多模型融合）
python src/train.py

# 生成提交文件（基于训练好的模型推理）
python src/infer.py
```

## ✨ 核心方案思路

### 1\. 探索性数据分析（EDA）

- 分析保险保费（目标变量）分布特征，处理极端异常值

- 挖掘关键影响因素：年龄、车辆使用年限、年收入、信用分、健康分、保险期限

- 梳理特征相关性，剔除冗余特征，结合保险业务逻辑筛选核心特征

### 2\. 高阶特征工程

- 业务交叉特征构建：年龄×车辆年限、年收入×健康分、信用分×保险期限等

- 比例/比率特征：健康分/年龄、信用分/年收入、保险期限/车辆年限等

- 类别特征编码：Target Encoding（目标编码）、频次统计编码

- 日期特征提取：保单起始日期的年份、月份、季度特征

- 数值特征优化：年收入对数转换，提升模型拟合效果

### 3\. 数据预处理

- 缺失值处理：类别特征填充\&\#34;unknown\&\#34;，数值特征先中位数填充再用FAISS KNN优化填充

- 异常值处理：采用1%和99%分位数截断，避免极端值影响模型训练

- 特征筛选：VarianceThreshold（方差筛选）\+ 互信息法（Mutual Information），保留高区分度特征

- 标准化处理：适配Ridge线性模型，提升训练稳定性

### 4\. 多模型动态融合框架

- 验证策略：10折交叉验证，充分利用数据，降低过拟合风险

- 基模型组合：LightGBM \+ XGBoost \+ CatBoost \+ LightGBM\(RF\) \+ Extra Trees \+ Ridge

- 融合策略：基于RMSLE分数的动态权重融合，自动分配模型权重，提升预测稳定性

- 训练优化：Log域训练（log1p转换目标变量），推理时指数还原（expm1），降低数值偏差

### 5\. 后处理优化

- 预测值校准：将预测结果裁剪至训练集保费的1%\~99%分位数范围内，避免预测异常

- 结果平滑：微小系数修正（×0\.99），提升线上预测一致性

## 📊 模型与评估

|项目|详情|
|---|---|
|评估指标|RMSLE（均方根对数误差）|
|验证方式|10折交叉验证（shuffle=True，固定随机种子）|
|模型架构|6基模型 \+ 动态权重融合|
|最终提交|保险保费金额预测（保留3位小数）|

## 🛠 环境依赖

```bash
pip install pandas==2.2.3 numpy==2.2.6 polars==1.7.0
pip install lightgbm==4.6.0 xgboost==3.2.0 catboost==1.2.10
pip install scikit-learn==1.8.0 category_encoders==2.9.0
pip install optuna==4.8.0 faiss-cpu==1.8.0 joblib==1.5.3
pip install matplotlib==3.10.8 seaborn==0.13.2
```

## 📌 项目说明

本方案以高质量特征工程为核心，结合多模型融合策略，充分适配保险保费预测的业务场景与数据特点，有效规避线性模型的局限性，通过严谨的交叉验证与后处理优化，最终在比赛中取得第3名的优异成绩。

代码结构清晰、可复现性强，所有脚本均固定随机种子（SEED=42），支持CPU运行（可无缝切换GPU），适合同类回归任务（如房价预测、保费预测、销量预测）的学习与迁移使用。

模型权重文件通过Git LFS管理，避免大文件上传报错，确保项目完整可复现。

## 💡 技术栈


[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0.99-blue?style=flat-square)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-green?style=flat-square)](https://xgboost.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-orange?style=flat-square)](https://catboost.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-purple?style=flat-square)](https://pytorch.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.3-teal?style=flat-square)](https://pandas.pydata.org/)

    
