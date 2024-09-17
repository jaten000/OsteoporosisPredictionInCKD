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
train_file_path = "C:/Users/jatenhsu/Desktop/ckdosteoporosis/train_data.csv"
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
test_file_path = "C:/Users/jatenhsu/Desktop/ckdosteoporosis/test_data.csv"
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
    plt.figure(figsize=(12, 12))

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
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', color=model_colors[model_name])  # 顯示三位小數

    # 繪製No Skill的基線
    plt.plot([0, 1], [0, 1], linestyle='--', label="No Skill", color='yellowgreen')

    # 標籤和圖例
    plt.xlabel("False Positive Rate", fontsize=18)  # 調整x軸字體大小
    plt.ylabel("True Positive Rate", fontsize=18)  # 調整y軸字體大小
    plt.title(f"ROC Curve for Different Models ({dataset_name} Set)", fontsize=20)  # 調整標題字體大小
    
    # 調整X軸和Y軸的刻度字體大小
    plt.tick_params(axis='both', which='major', labelsize=16)

    # 調整圖例順序為根據AUC排序的結果
    sorted_labels = [f'{model_name} (AUC = {auc:.3f})' for model_name, auc, _, _ in auc_list]  # 顯示三位小數
    sorted_labels.append("No Skill")  # 添加 No Skill
    plt.legend(sorted_labels, loc='best', fontsize=16)  # 增加字體大小

    # 儲存和顯示圖片
    plt.axis('square')
    plt.savefig(f"C:/Users/jatenhsu/Desktop/ckdosteoporosis/ROC_Curve_Comparison_{dataset_name}.tiff", dpi=300)
    plt.show()



# 12. 計算測試集和驗證集的各項評估指標並匯出成CSV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(y_true, y_pred, y_probs, model_name, dataset_type, group_type):
    accuracy = accuracy_score(y_true, y_pred)
    
    # 使用 zero_division=0 來避免警告，這樣如果沒有正樣本，Precision 和 Recall 將被設為 0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 檢查 y_true 中是否包含多於一個類別，以防止 ROC AUC 計算錯誤
    if len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, y_probs)
    else:
        auc = None  # 當只有一個類別時，AUC 無法定義
    
    return {
        'Model': model_name,
        'Dataset': dataset_type,
        'Group': group_type,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc
    }

metrics_results = []

# 分別計算 Validation 和 Test 的指標
for dataset_name, data_imputed in zip(['Validation', 'Test'], [val_data_imputed, test_data_imputed]):
    for model_name, model in models.items():
        if model_name == 'ANN':
            ann_data_imputed = data_imputed[selected_features].values
            preds = (model.predict(ann_data_imputed).ravel() >= 0.5).astype(int)  # ANN 特別處理
            probs = model.predict(ann_data_imputed).ravel()
        else:
            preds = model.predict(data_imputed[selected_features])
            probs = model.predict_proba(data_imputed[selected_features])[:, 1]

        # 全部樣本的計算
        metrics_results.append(calculate_metrics(data_imputed["Osteoporosis"], preds, probs, model_name, dataset_name, "All"))


# 13. 95%信賴區間
import numpy as np
from sklearn.utils import resample

# 定義一個函數來計算 95% 信賴區間
def calculate_95ci(metric_values):
    lower_bound = np.percentile(metric_values, 2.5)
    upper_bound = np.percentile(metric_values, 97.5)
    return lower_bound, upper_bound

# 修改 calculate_metrics 函數以支持 bootstrap 來計算95%信賴區間
def calculate_metrics_with_ci(y_true, y_pred, y_probs, model_name, dataset_type, group_type, n_bootstrap=1000):
    accuracy_scores, precision_scores, recall_scores, f1_scores, auc_scores = [], [], [], [], []

    # 進行 bootstrap resampling
    for _ in range(n_bootstrap):
        indices = resample(np.arange(len(y_true)), replace=True)  # 隨機抽取有放回的樣本
        y_true_resampled = y_true[indices]
        y_pred_resampled = y_pred[indices]
        y_probs_resampled = y_probs[indices]

        # 計算每次重抽樣的評估指標
        accuracy_scores.append(accuracy_score(y_true_resampled, y_pred_resampled))
        precision_scores.append(precision_score(y_true_resampled, y_pred_resampled, zero_division=0))
        recall_scores.append(recall_score(y_true_resampled, y_pred_resampled, zero_division=0))
        f1_scores.append(f1_score(y_true_resampled, y_pred_resampled, zero_division=0))
        
        if len(set(y_true_resampled)) > 1:  # 防止 y_true 只有一個類別時 ROC AUC 計算錯誤
            auc_scores.append(roc_auc_score(y_true_resampled, y_probs_resampled))
        else:
            auc_scores.append(np.nan)  # 當只有一個類別時，設置 AUC 為 NaN

    # 檢查 auc_scores 是否有有效值
    auc_scores_clean = [score for score in auc_scores if not np.isnan(score)]
    
    if len(auc_scores_clean) > 0:  # 確保有有效的 AUC 值
        auc_mean = np.nanmean(auc_scores_clean)  # 忽略 NaN 的情況計算平均值
        auc_ci = calculate_95ci(auc_scores_clean)  # 計算信賴區間
        auc_result = f'{auc_mean:.3f} ({auc_ci[0]:.3f} - {auc_ci[1]:.3f})'
    else:
        auc_result = 'N/A'  # 當無法計算 AUC 時，返回 'N/A'

    # 計算每個評估指標的平均值和95%CI
    accuracy_mean = np.mean(accuracy_scores)
    accuracy_ci = calculate_95ci(accuracy_scores)

    precision_mean = np.mean(precision_scores)
    precision_ci = calculate_95ci(precision_scores)

    recall_mean = np.mean(recall_scores)
    recall_ci = calculate_95ci(recall_scores)

    f1_mean = np.mean(f1_scores)
    f1_ci = calculate_95ci(f1_scores)

    return {
        'Model': model_name,
        'Dataset': dataset_type,
        'Group': group_type,
        'Accuracy': f'{accuracy_mean:.3f} ({accuracy_ci[0]:.3f} - {accuracy_ci[1]:.3f})',
        'Precision': f'{precision_mean:.3f} ({precision_ci[0]:.3f} - {precision_ci[1]:.3f})',
        'Recall': f'{recall_mean:.3f} ({recall_ci[0]:.3f} - {recall_ci[1]:.3f})',
        'F1 Score': f'{f1_mean:.3f} ({f1_ci[0]:.3f} - {f1_ci[1]:.3f})',
        'AUC': auc_result  # 返回計算的 AUC 結果或 'N/A'
    }


# 計算 Validation 和 Test 的指標並添加95% CI
metrics_results = []

for dataset_name, data_imputed in zip(['Validation', 'Test'], [val_data_imputed, test_data_imputed]):
    for model_name, model in models.items():
        if model_name == 'ANN':
            ann_data_imputed = data_imputed[selected_features].values
            preds = (model.predict(ann_data_imputed).ravel() >= 0.5).astype(int)  # ANN 特別處理
            probs = model.predict(ann_data_imputed).ravel()
        else:
            preds = model.predict(data_imputed[selected_features])
            probs = model.predict_proba(data_imputed[selected_features])[:, 1]

        # 全部樣本的計算
        metrics_results.append(calculate_metrics_with_ci(data_imputed["Osteoporosis"].values, preds, probs, model_name, dataset_name, "All"))

# 將結果保存為 DataFrame 並匯出為 CSV
metrics_df = pd.DataFrame(metrics_results)
metrics_df.to_csv("C:/Users/jatenhsu/Desktop/ckdosteoporosis/classification_metrics_validation_test_grouped_with_ci.csv", index=False)

print(f"Results with 95% CI saved to C:/Users/jatenhsu/Desktop/ckdosteoporosis/classification_metrics_validation_test_grouped_with_ci.csv")


# Calibration Curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# 定義模型顏色
calibration_colors = {
    "ANN": "magenta",
    "CatBoost": "red",
    "LightGBM": "green",
    "XGBoost": "orange",
    "Random Forest": "cyan"
}

# 計算和繪製 Calibration Curve
plt.figure(figsize=(12, 10))

for model_name, model in models.items():
    if model_name in ["ANN", "CatBoost", "LightGBM", "XGBoost", "Random Forest"]:
        if model_name == 'ANN':
            probs = test_probs  # 已經預測的機率
        else:
            probs = model.predict_proba(test_data_imputed[selected_features])[:, 1]  # 其他模型的預測機率

        # 使用 calibration_curve 函數
        fraction_of_positives, mean_predicted_value = calibration_curve(test_data_imputed["Osteoporosis"], probs, n_bins=10)

        # 繪製 Calibration 曲線
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label=f'{model_name}', color=calibration_colors[model_name])

# 完美校準的基準線
plt.plot([0, 1], [0, 1], linestyle='--', label="Perfectly calibrated", color="black")

# 標籤
plt.xlabel("Predicted probability", fontsize=16)  # X軸字體大小
plt.ylabel("True probability", fontsize=16)  # Y軸字體大小
plt.title("Calibration Curve for Selected Models", fontsize=18)  # 標題字體大小
plt.legend(loc='best', fontsize=16)  # 圖例字體大小
plt.tick_params(axis='both', which='major', labelsize=16)  # 調整 X 和 Y 軸刻度的字體大小

# 去除網格
plt.grid(False)

# 保存圖片為TIFF格式，300dpi
plt.savefig("C:/Users/jatenhsu/Desktop/ckdosteoporosis/Calibration_Curve.tiff", dpi=300, format='tiff')

# 顯示圖片
plt.show()


# Decision Curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc
from sklearn.calibration import calibration_curve

# Function to plot decision curves
def plot_decision_curve(results, test_data_imputed, selected_features, models, file_path):
    plt.figure(figsize=(14, 12))

    thresholds = np.linspace(0.01, 0.99, 99)
    net_benefit_all = [0.68 - threshold for threshold in thresholds]  # ALL 的曲線
    net_benefit_none = np.zeros_like(thresholds)  # NONE 的曲線

    # 繪製 ALL 和 NONE 的曲線
    plt.plot(thresholds, net_benefit_all, 'k--', label='ALL', color='gray', linestyle='--', linewidth=2)
    plt.plot(thresholds, net_benefit_none, 'k-', label='NONE', color='black', linewidth=2)

    # 依據模型的結果繪製決策曲線
    model_colors = {
        'ANN': 'magenta',
        'CatBoost': 'red',
        'LightGBM': 'green',
        'XGBoost': 'orange',
        'Random Forest': 'cyan'
    }

    for model_name in ['ANN', 'CatBoost', 'LightGBM', 'XGBoost', 'Random Forest']:
        model = models[model_name]

        # 針對 ANN 模型處理資料格式
        if model_name == 'ANN':
            test_data_input = test_data_imputed[selected_features].values
            probs = model.predict(test_data_input).ravel()
        else:
            probs = model.predict_proba(test_data_imputed[selected_features])[:, 1]

        net_benefits = []
        for threshold in thresholds:
            tp = np.sum((probs >= threshold) & (test_data_imputed['Osteoporosis'] == 1))
            fp = np.sum((probs >= threshold) & (test_data_imputed['Osteoporosis'] == 0))
            fn = np.sum((probs < threshold) & (test_data_imputed['Osteoporosis'] == 1))
            tn = np.sum((probs < threshold) & (test_data_imputed['Osteoporosis'] == 0))

            # 計算決策收益 (Net Benefit)
            benefit = (tp / len(probs)) - (fp / len(probs)) * (threshold / (1 - threshold))
            net_benefits.append(benefit)

        plt.plot(thresholds, net_benefits, label=model_name, color=model_colors[model_name], linewidth=2)

    # 圖片設置
    plt.title('Decision Curve Analysis', fontsize=20)
    plt.xlabel('Threshold Probability', fontsize=18)
    plt.ylabel('Net Benefit', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper right', fontsize=18)
    plt.grid(False)

    # 設定Y軸最小值為0.0，移除負值部分
    plt.ylim(bottom=-0.05)

    # 保存圖片到指定路徑
    plt.savefig(file_path, format='tiff', dpi=300)
    plt.close()

# 文件存儲路徑
output_file_path = "C:/Users/jatenhsu/Desktop/ckdosteoporosis/decision_curve.tiff"

# 調用函數，繪製決策曲線
plot_decision_curve(results, test_data_imputed, selected_features, models, output_file_path)

# 顯示圖片
plt.show()
