# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:01:23 2024

@author: jatenhsu
"""
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
import tensorflow as tf  # 導入 TensorFlow
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import GridSearchCV

class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn=None, **params):
        self.build_fn = build_fn or create_ann_model  # 如果沒有指定，使用默認模型函數
        self.params = params
        self.model = None

    def set_params(self, **params):
        self.params.update(params)
        return self

    def get_params(self, deep=True):
        return self.params

    def fit(self, X, y, epochs=100, batch_size=32):
        model_params = {key: self.params[key] for key in self.params if key not in ['epochs', 'batch_size']}
        self.model = self.build_fn(**model_params)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, 
               validation_data=(X_val, y_val),  # 加入驗證數據
               callbacks=[early_stopping])
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")

    def predict_proba(self, X):
        return self.model.predict(X)

# 定義ANN模型
def create_ann_model(input_dim, dropout_rate=0.5, optimizer='adam'):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 包裝模型
model = KerasClassifierWrapper(build_fn=create_ann_model, input_dim=X_train.shape[1])

# 定義超參數範圍
param_grid = {
    'batch_size': [32, 64, 128, 256],  # 減少批次大小的選項
    'epochs': [50, 100, 200, 300],
    'dropout_rate': [0.05, 0.1, 0.2, 0.3],  # 減少 dropout 的選項
    'optimizer': ['adam', 'rmsprop']  # 優化器
}

# 使用 GridSearchCV 進行網格搜索
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)

# 訓練模型，進行超參數調整
grid_result = grid_search.fit(X_train, y_train)

# 輸出最佳超參數和結果
print(f"Best score: {grid_result.best_score_}")
print(f"Best parameters: {grid_result.best_params_}")


# 包含其它模型和ANN模型
models = {
    'Logistic Regression': LogisticRegression(),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(silent=True),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'ANN': create_ann_model(X_train.shape[1])  # 使用ANN模型
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

# 9. 繪製 ROC 曲線 (Validation 和 Test)
# 定義每個模型的顏色
model_colors = {
    "Logistic Regression": "blue",
    "XGBoost": "orange",
    "LightGBM": "green",
    "CatBoost": "red",
    "SVM": "purple",
    "Decision Tree": "pink",
    "Random Forest": "cyan",
    "KNN": "brown",
    "ANN": "magenta"  
}


for dataset_name, data_imputed in zip(['Validation', 'Test'], [val_data_imputed, test_data_imputed]):
    plt.figure(figsize=(8, 8))

    # 創建一個列表來存儲每個模型的名稱和AUC值
    auc_list = []

    for model_name, model in models.items():
        if model_name == 'ANN':
            ann_data_imputed = data_imputed[selected_features].values
            probs = model.predict(ann_data_imputed).ravel()  # ANN 特別處理
        else:
            probs = model.predict_proba(data_imputed[selected_features])[:, 1]

        fpr, tpr, _ = roc_curve(data_imputed["Osteoporosis"], probs)
        auc = roc_auc_score(data_imputed["Osteoporosis"], probs)

        # 保存模型名稱和AUC值到列表
        auc_list.append((model_name, auc, fpr, tpr))

    # 根據AUC值從高到低進行排序
    auc_list.sort(key=lambda x: x[1], reverse=True)

    # 按照排序後的AUC值繪製ROC曲線，並分配固定顏色
    for model_name, auc, fpr, tpr in auc_list:
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})', color=model_colors[model_name])

    # 繪製No Skill的基線
    plt.plot([0, 1], [0, 1], linestyle='--', label="No Skill", color='yellowgreen')

    # 標籤和圖例
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for Different Models ({dataset_name} Set)")

    # 調整圖例順序為根據AUC排序的結果
    sorted_labels = [f'{model_name} (AUC = {auc:.2f})' for model_name, auc, _, _ in auc_list]
    sorted_labels.append("No Skill")  # 添加 No Skill
    plt.legend(sorted_labels, loc='best')

    # 儲存和顯示圖片
    plt.axis('square')
    plt.savefig(f"C:/...../ROC_Curve_Comparison_{dataset_name}.tiff", dpi=300)
    plt.show()

