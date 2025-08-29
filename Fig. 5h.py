import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
file_path = ""
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls, sheet_name='Sheet1')

# Rename the first column
df.rename(columns={'Unnamed: 0': 'Group'}, inplace=True)

# Features and labels
X = df[['Behav.', 'Bio', 'Neural']]
y = df['Group'].map({'sham': 0, 'TES': 1})  # sham=0, TES=1

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SVM model
svm_model = SVC(kernel='linear', random_state=42)

# 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_cv_pred = cross_val_predict(svm_model, X_scaled, y, cv=cv)

# Confusion matrix (raw counts)
cm_cv = confusion_matrix(y, y_cv_pred)

# Convert to row-wise percentages (recall-based view)
cm_cv_percent = (cm_cv.T / cm_cv.sum(axis=1)).T * 100

# Display as percentages
disp_cv = ConfusionMatrixDisplay(confusion_matrix=cm_cv_percent, display_labels=['Sham', 'TES'])
disp_cv.plot(cmap='Blues', values_format='.1f')
plt.title("Confusion Matrix (%) - Row-wise (Recall per Class)")
plt.show()
