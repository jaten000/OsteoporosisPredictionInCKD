# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:04:35 2024

@author: jatenhsu
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 匯入資料
file_path = 'C:/...../Data.csv'
data = pd.read_csv(file_path)

# 排除 Osteogenesis Imperfecta並計算所有其他特徵之間的相關性矩陣, drop 'Osteogenesis Imperfecta' due to All = 0
features = data.drop(['Osteogenesis Imperfecta'], axis=1)
correlation_matrix = features.corr()

# 調整圖表大小，增大字體
plt.figure(figsize=(28, 28))  # 增大圖表的大小
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, annot_kws={"size": 10})  # 設定字體大小為 8
plt.title('Feature Correlation Matrix', fontsize=16)
plt.xticks(rotation=90, ha='right', fontsize=12)  # 調整 X 軸標籤字體大小及旋轉角度
plt.yticks(fontsize=12)  # 調整 Y 軸標籤字體大小
plt.tight_layout()

# 儲存圖片
output_path = 'C:/...../feature_correlation_matrix_adjusted.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

# Feature selection with correlation 和繪製 correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 匯入資料
file_path = 'C:/...../Data.csv'
data = pd.read_csv(file_path)

# 設定 outcome 和 features
outcome = 'Osteoporosis'

# 排除 PERSON_ID 和 Osteogenesis Imperfecta(已實驗得知關連性很低)
features = data.drop(['PERSON_ID', 'Osteogenesis Imperfecta'], axis=1)

# 計算每個特徵與 outcome 的相關性
correlation_matrix = features.corrwith(data[outcome]).to_frame(name='correlation')

# 排除 Osteoporosis 本身在 y 軸上顯示
correlation_matrix = correlation_matrix.drop(index=[outcome])

import pandas as pd

# 排序相關性，正相關大的在上方，負相關大的在下方
correlation_matrix_sorted = correlation_matrix.sort_values(by='correlation', ascending=False)

# 繪製排序後的相關矩陣圖
plt.figure(figsize=(8, 8))
sns.heatmap(correlation_matrix_sorted, annot=True, cmap='coolwarm', cbar=True)
plt.title('Correlation with Osteoporosis (Sorted by Correlation)')
plt.tight_layout()

# 儲存圖片
output_path = 'C:/...../correlation_matrix_sorted.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()


# 使用 HistGradientBoostingClassifier 的特徵重要性進行特徵選擇
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# 匯入資料
file_path = 'C:/...../Data.csv'
data = pd.read_csv(file_path)

# 設定 outcome 和 features
outcome = 'Osteoporosis'
X = data.drop(['PERSON_ID', outcome], axis=1)
y = data[outcome]

# 確保所有特徵都是數值型資料，若有類別型特徵，需進行編碼
X = X.select_dtypes(include=[np.number])

# 建立支援 NaN 的模型
model = HistGradientBoostingClassifier()

# 訓練模型
model.fit(X, y)

# 使用 permutation_importance 計算特徵重要性
result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)

# 獲取特徵重要性分數
feature_importances = result.importances_mean
features_list = X.columns

# 將特徵和重要性結合成 DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': features_list,
    'Importance': feature_importances
})

# 按重要性排序，選出前 10 個特徵
top_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

print('Selected Features based on importance:')
print(top_features)

# 繪製特徵重要性圖表
plt.figure(figsize=(10, 8))
plt.barh(top_features['Feature'], top_features['Importance'])
plt.gca().invert_yaxis()
plt.xlabel('Feature Importance')
plt.title('Top 10 important features')
plt.tight_layout()

# 儲存圖片
output_path = 'C:/...../feature_importance.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()


# Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# 使用 RandomForestClassifier 進行 RFE 選出最好的 10 個 features
model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X, y)

# 印出選出的 features
selected_rfe_features = features_list[rfe.support_]
print('Selected RFE Features:', selected_rfe_features)


# 使用 RFECV 並繪製特徵數量與交叉驗證分數圖
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# 設定 outcome 和 features
outcome = 'Osteoporosis'
X = data.drop(['PERSON_ID', outcome], axis=1)
y = data[outcome]

# 建立支援 NaN 的模型
model = RandomForestClassifier()

# 使用交叉驗證進行 Recursive Feature Elimination with Cross Validation (RFECV)
cv = StratifiedKFold(n_splits=5)
rfecv = RFECV(estimator=model, step=1, cv=cv, scoring='accuracy')
rfecv.fit(X, y)

# 印出最佳的 features 和最適合的 feature 數量
optimal_features = X.columns[rfecv.support_]
print('Optimal number of features:', rfecv.n_features_)
print('Best features:', optimal_features)

# 使用 `cv_results_` 提取交叉驗證分數並繪製圖表
plt.figure(figsize=(10, 8))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validation Score (Accuracy)')
plt.title('Number of Features vs. Cross-Validation Scores')
plt.tight_layout()

# 儲存圖片
output_path_rfecv = 'C:/...../rfecv_plot.png'
plt.savefig(output_path_rfecv, dpi=300, bbox_inches='tight')
plt.show()


