# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 18:28:24 2024

@author: jatenhsu
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 匯入資料
file_path = "C:/...../Data.csv"
data = pd.read_csv(file_path)

# 2. 只保留 outcome 和指定的 features
selected_features = ["Osteoporosis", "Cre", "Age", "Weight", "Height", "Albumin", "iPTH", "Glucose", "HGB"]
data = data[selected_features]

# 3. 分割資料成訓練集和測試集，確保 "Osteoporosis" 的比例相似
train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    stratify=data["Osteoporosis"],
    random_state=42
)

# 4. 檢查並匹配缺失值比例
missing_features = ["Weight", "Height", "Albumin", "iPTH", "Glucose", "HGB"]

print("Feature - Train Missing Ratio, Test Missing Ratio")
for feature in missing_features:
    train_missing_ratio = train_data[feature].isna().mean()
    test_missing_ratio = test_data[feature].isna().mean()
    print(f"{feature} - {train_missing_ratio:.2f}, {test_missing_ratio:.2f}")

# 5. 檢查資料集
print(train_data.head())
print(test_data.head())

# 6. 將 train_data 和 test_data 存成 CSV 檔案
train_file_path = "C:/...../train_data.csv"
test_file_path = "C:/...../test_data.csv"

# 儲存訓練集和測試集
train_data.to_csv(train_file_path, index=False)
test_data.to_csv(test_file_path, index=False)

print(f"Train data saved to {train_file_path}")
print(f"Test data saved to {test_file_path}")
