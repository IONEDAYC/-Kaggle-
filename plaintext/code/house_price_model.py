# -*- coding: utf-8 -*-
# Kaggle房价预测 传统机器学习实现：Lasso+随机森林+XGBoost融合
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# ======================1、读取数据======================
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
all_data = pd.concat([train, test], ignore_index=True)
Id_test = test['Id']

# 目标变量对数转换，修正偏态分布
train['SalePrice'] = np.log1p(train['SalePrice'])
y_train = train['SalePrice']
train.drop('SalePrice', axis=1, inplace=True)

# ======================2、缺失值处理（业务规则填充）======================
# 无地下室则填充None
bsmt_cols = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']
for col in bsmt_cols:
    all_data[col] = all_data[col].fillna('None')
# 无车库填充None
garage_cols = ['GarageType','GarageFinish','GarageQual','GarageCond']
for col in garage_cols:
    all_data[col] = all_data[col].fillna('None')
# 泳池、围栏、壁炉空值代表无设施
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
all_data['Fence'] = all_data['Fence'].fillna('None')
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')

# 数值特征中位数填充
num_cols = all_data.select_dtypes(include=[np.number]).columns
all_data[num_cols] = all_data[num_cols].fillna(all_data[num_cols].median())
# 类别特征众数填充
cat_cols = all_data.select_dtypes(exclude=[np.number]).columns
all_data[cat_cols] = all_data[cat_cols].fillna(all_data[cat_cols].mode().iloc[0])

# ======================3、特征工程======================
# 组合新特征：总面积、房龄
all_data['TotalSF'] = all_data['1stFlrSF'] + all_data['2ndFlrSF'] + all_data['TotalBsmtSF']
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']

# 类别特征独热编码
all_data = pd.get_dummies(all_data, drop_first=True)
# 划分训练集和测试集
X_train = all_data.iloc[:len(train), :]
X_test = all_data.iloc[len(train):, :]

# ======================4、模型训练======================
# 模型1：Lasso回归
lasso = Lasso(alpha=0.001, random_state=42)
lasso.fit(X_train, y_train)
# 模型2：随机森林
rf = RandomForestRegressor(n_estimators=800, max_depth=12, random_state=42)
rf.fit(X_train, y_train)
# 模型3：XGBoost回归
xgb_model = xgb.XGBRegressor(n_estimators=600, max_depth=3, learning_rate=0.01, random_state=42)
xgb_model.fit(X_train, y_train)

# ======================5、模型融合预测======================
pred_lasso = np.expm1(lasso.predict(X_test))
pred_rf = np.expm1(rf.predict(X_test))
pred_xgb = np.expm1(xgb_model.predict(X_test))
# 加权融合
final_pred = 0.3 * pred_lasso + 0.3 * pred_rf + 0.4 * pred_xgb

# ======================6、生成Kaggle提交文件======================
submit_df = pd.DataFrame({
    'Id': Id_test,
    'SalePrice': final_pred
})
submit_df.to_csv("./result/submission.csv", index=False)
print("提交文件已生成：result/submission.csv")

# 输出交叉验证分数
cv_lasso = cross_val_score(lasso, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
cv_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
cv_xgb = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
print(f"Lasso 5折RMSE: {-np.mean(cv_lasso):.4f}")
print(f"随机森林 5折RMSE: {-np.mean(cv_rf):.4f}")
print(f"XGBoost 5折RMSE: {-np.mean(cv_xgb):.4f}")