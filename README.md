# OsteoporosisPredictionInCKD
Machine learning models to predict osteoporosis in patients with chronic kidney disease stage 3–5 and end-stage kidney disease
1. Download "Data.csv"
2. Run "Feature selection.py" to select features.
3. In our study, 8 features (age, creatinine, height, weight, albumin, glucose, intact parathyroid hormone, hemoglobin) for ANN model achieved the highest predictive performance
4. Run "8feature_stratified_split.py", then you will get "train_data.csv" and "test_test.csv)
5. Run "Main_code.py", and you will get the predictive performance metrics (AUC, precision, recall, accuracy, and F1 score) of the nine ML models (logistic regression, XGBoost, LightGBM, CatBoost, SVM, decision tree, random forest, k-nearest neighbors, and artificial neural network); the ROC curve comparison for different models; the Confusion matrices and predictive probabilities histograms for different models; Calibration curve (A) and decision curve analysis (B) for five selected models in osteoporosis prediction.
6. Run "SHAP.py", and you will get the SHAP summary plot for ANN model
7. Run "GridSearchCV_ANN_Hyperparameter.py to tune the hyperparameter of ANN model.
