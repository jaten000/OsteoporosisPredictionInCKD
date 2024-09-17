# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:01:23 2024

@author: jatenhsu
"""
# 獨立分開訓練集和測試集，先用訓練集
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 匯入資料
train_file_path = "C:/...../train_data.csv"
data = pd.read_csv(train_file_path)

# 2. 分割資料，確保 "Osteoporosis" 比例接近
train_data, val_data = train_test_split(data, test_size=0.2, stratify=data["Osteoporosis"], random_state=42)

# 3. 多重插補缺失值
missing_features = ["Weight", "Height", "Albumin", "iPTH", "Glucose", "HGB"]
imputer = IterativeImputer()
train_data_imputed = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
val_data_imputed = pd.DataFrame(imputer.transform(val_data), columns=val_data.columns)

# 4. 標準化連續型特徵
continuous_features = ["Cre", "Age", "Weight", "Height", "Albumin", "iPTH", "Glucose", "HGB"]
scaler = StandardScaler()
train_data_imputed[continuous_features] = scaler.fit_transform(train_data_imputed[continuous_features])
val_data_imputed[continuous_features] = scaler.transform(val_data_imputed[continuous_features])

# 選擇要使用的 features
selected_features = ["Cre", "Age", "Weight", "Height", "Albumin", "iPTH", "Glucose", "HGB"]
# 如果需要另一組 features，比如這一組，則可以替換使用這一行：


# 定義 X_train, y_train, X_val, y_val，只使用選擇的特徵
X_train = train_data_imputed[selected_features]
y_train = train_data_imputed["Osteoporosis"]
X_val = val_data_imputed[selected_features]
y_val = val_data_imputed["Osteoporosis"]


# 5. 定義模型
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input

def create_ann_model(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))  # 使用 Input 層來定義輸入維度
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 二元分類，輸出為機率
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 包含原有模型和新增的ANN模型
models = {
    'Logistic Regression': LogisticRegression(),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(silent=True),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'ANN': create_ann_model(len(continuous_features))  # 新增ANN模型
}

# 6. 訓練模型並進行5-fold交叉驗證
results = {}
for model_name, model in models.items():
    if model_name == 'ANN':
        # 將ANN模型的資料轉為TensorFlow格式
        ann_train_data = train_data_imputed[selected_features].values
        ann_val_data = val_data_imputed[selected_features].values
        
        # 使用一部分驗證集資料來進行早期停止（避免過擬合）
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        
        # 訓練ANN模型
        model.fit(ann_train_data, train_data_imputed["Osteoporosis"].values, 
                  validation_data=(ann_val_data, val_data_imputed["Osteoporosis"].values), 
                  epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)
        
        # 預測並計算驗證集上的機率和分類結果
        val_probs = model.predict(ann_val_data).ravel()
        val_preds = (val_probs >= 0.5).astype(int)
    else:
        # 其他模型的訓練和預測
        model.fit(train_data_imputed[selected_features], train_data_imputed["Osteoporosis"])
        val_preds = model.predict(val_data_imputed[selected_features])
        val_probs = model.predict_proba(val_data_imputed[selected_features])[:, 1]

    # 使用5-fold交叉驗證
    if model_name != 'ANN':  # ANN 不支持 cross_val_score，略過這部分
        cv_scores = cross_val_score(model, val_data_imputed[selected_features], val_data_imputed["Osteoporosis"], cv=5)
        print(f'{model_name} - 5-Fold CV Scores: {cv_scores}, Mean CV Score: {cv_scores.mean():.2f}')
    
    # 計算評估指標
    accuracy = accuracy_score(val_data_imputed["Osteoporosis"], val_preds)
    precision = precision_score(val_data_imputed["Osteoporosis"], val_preds)
    recall = recall_score(val_data_imputed["Osteoporosis"], val_preds)
    f1 = f1_score(val_data_imputed["Osteoporosis"], val_preds)
    auc = roc_auc_score(val_data_imputed["Osteoporosis"], val_probs)

    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc
    }

    print(f"{model_name}: Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, AUC: {auc:.2f}")

# 7. 匯入 test_data，補缺失值與標準化
test_file_path = "C:/...../test_data.csv"
test_data = pd.read_csv(test_file_path)

test_data_imputed = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)
test_data_imputed[continuous_features] = scaler.transform(test_data_imputed[continuous_features])

# 選擇要使用的 features
selected_features = ["Cre", "Age", "Weight", "Height", "Albumin", "iPTH", "Glucose", "HGB"]
# 如果需要另一組 features，比如這一組，則可以替換使用這一行：

# 8. 應用模型於測試集
for model_name, model in models.items():
    if model_name == 'ANN':
        # 將ANN模型的資料轉為TensorFlow格式
        ann_test_data = test_data_imputed[selected_features].values
        
        # 預測機率和分類結果
        test_probs = model.predict(ann_test_data).ravel()
        test_preds = (test_probs >= 0.5).astype(int)
    else:
        test_preds = model.predict(test_data_imputed[selected_features])
        test_probs = model.predict_proba(test_data_imputed[selected_features])[:, 1]


# 8. 計算ANN model SHAP 值並繪製 summary plot

import shap
import matplotlib.pyplot as plt

# 隨機選擇部分數據進行解釋，以加快計算速度
background = ann_train_data[np.random.choice(ann_train_data.shape[0], 100, replace=False)]

# 使用 KernelExplainer 來計算 SHAP 值
explainer = shap.KernelExplainer(models['ANN'].predict, background)

# 計算 SHAP 值
shap_values = explainer.shap_values(ann_train_data)

# 移除多餘的維度，將 (4232, 7, 1) 轉換為 (4232, 7)
shap_values = np.squeeze(shap_values)

# 確認 SHAP 值的形狀是否與資料相符
print(f"shap_values shape: {shap_values.shape}")
print(f"ann_train_data shape: {ann_train_data.shape}")

# 繪製 SHAP summary plot
plt.figure(figsize=(10, 10))
shap.summary_plot(shap_values, features=ann_train_data, feature_names=continuous_features, show=False)

# 設置 X 軸範圍為 -1.0 到 1.0，確保左右對稱
plt.xlim([-1.0, 1.0])

plt.savefig('C:/...../ANN_SHAP_Summary.tiff', dpi=300)
plt.show()
