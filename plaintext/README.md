# Kaggle房价预测机器学习课程设计
竞赛地址：House Prices: Advanced Regression Techniques
## 项目说明
仅使用传统机器学习（无深度学习），模型包含Lasso、随机森林、XGBoost加权融合，任务为房屋售价回归预测，评价指标RMSLE。
## 目录结构
1. data：数据集、字段说明文档
2. code：完整训练代码
3. result：Kaggle提交预测文件
## 运行依赖
pip install pandas numpy scikit-learn xgboost
## 实验结果
Lasso 5折RMSLE：0.132
随机森林 5折RMSLE：0.125
XGBoost 5折RMSLE：0.118
融合模型 5折RMSLE：0.112
线上Kaggle得分：0.115
