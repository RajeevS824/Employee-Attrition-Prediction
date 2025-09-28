# ==========================================================
# Importing Necessary Libraries
# ==========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib   # For saving and loading trained models

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ==========================================================
# Load Dataset
# ==========================================================
df = pd.read_excel("Employee-Attrition.xlsx")

print("Initial Shape:", df.shape)
print(df.info())
print("Columns:", df.columns.tolist())


# ==========================================================
# Data Cleaning
# ==========================================================
# Dropping columns that are not useful for prediction
drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
df = df.drop(columns=drop_cols)
print("After Dropping Useless Columns:", df.shape)


# ==========================================================
# Exploratory Data Analysis (EDA)
# ==========================================================
df_attr = df.copy()  # keep a copy for EDA

# 1. Attrition distribution
plt.figure(figsize=(5,4))
sns.countplot(x='Attrition', data=df_attr, hue='Attrition', palette="Set2", legend=False)
plt.title("Overall Attrition Distribution")
plt.show()

# 2. Monthly income vs Attrition
plt.figure(figsize=(7,4))
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df_attr, hue='Attrition', palette="Set3", legend=False)
plt.title("Monthly Income vs Attrition")
plt.show()

# 3. Age distribution by Attrition
plt.figure(figsize=(7,4))
sns.kdeplot(df_attr[df_attr['Attrition']=='Yes']['Age'], label="Left", fill=True, color="red")
sns.kdeplot(df_attr[df_attr['Attrition']=='No']['Age'], label="Stayed", fill=True, color="green")
plt.title("Age Distribution by Attrition")
plt.legend()
plt.show()


# ==========================================================
# Define Target & Selected Features
# ==========================================================
target = 'Attrition'
features = [
    'OverTime', 'MaritalStatus', 'DistanceFromHome',
    'JobRole', 'Department', 'TotalWorkingYears',
    'JobLevel', 'YearsInCurrentRole', 'MonthlyIncome',
    'Age', 'YearsWithCurrManager', 'StockOptionLevel',
    'YearsAtCompany', 'JobInvolvement'
]


# ==========================================================
# Barplots for Features vs Target
# ==========================================================
for col in features:
    plt.figure(figsize=(6, 4))
    sns.barplot(x=df[col], y=df[target], estimator=np.mean, errorbar=None)
    plt.title(f"{col} vs Attrition (Selected Feature)")
    plt.ylabel("Attrition Rate")
    plt.xlabel(col)
    plt.xticks(rotation=45)
    plt.show()

# Barplots for other columns
other_cols = [col for col in df.columns if col not in features + [target]]
for col in other_cols:
    plt.figure(figsize=(6, 4))
    sns.barplot(x=df[col], y=df[target], estimator=np.mean, errorbar=None)
    plt.title(f"{col} vs Attrition (Other Column)")
    plt.ylabel("Attrition Rate")
    plt.xlabel(col)
    plt.xticks(rotation=45)
    plt.show()


# ==========================================================
# Encode Target Column
# ==========================================================
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
print("Target Distribution:\n", df['Attrition'].value_counts())


# ==========================================================
# Identify Categorical & Numeric Columns
# ==========================================================
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("Categorical Columns:", cat_cols)
print("Numeric Columns:", num_cols)


# ==========================================================
# Label Encoding for Categorical Columns
# ==========================================================
encoder = LabelEncoder()
for col in ['BusinessTravel', 'Department', 'EducationField',
            'Gender', 'JobRole', 'MaritalStatus', 'OverTime']:
    df[col] = encoder.fit_transform(df[col])


# ==========================================================
# Correlation Analysis
# ==========================================================
corr_matrix = df.corr()
plt.figure(figsize=(20, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

print("Correlation with Attrition (%):\n", corr_matrix['Attrition'].sort_values(ascending=False) * 100)


# ==========================================================
# Select Features & Target for Modeling
# ==========================================================
X = df[features]
y = df[target]


# ==========================================================
# Outlier Visualization
# ==========================================================
plt.figure(figsize=(20, 6))
sns.boxplot(data=X)
plt.title("Before Scaling")
plt.show()


# ==========================================================
# Feature Scaling
# ==========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(12, 6))
sns.boxplot(data=X_scaled)
plt.title("After Scaling")
plt.show()


# ==========================================================
# Handle Class Imbalance (SMOTE)
# ==========================================================
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

print("Before SMOTE:\n", y.value_counts())
print("After SMOTE:\n", y_res.value_counts())


# ==========================================================
# Train-Test Split
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


# ==========================================================
# Random Forest Model
# ==========================================================
rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Save and reload model (joblib)
joblib.dump(rf_model, "attrition_model.pkl")
rf_model = joblib.load("attrition_model.pkl")

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:,1]

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf) * 100)
print("Random Forest AUROC:", roc_auc_score(y_test, y_prob_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Random Forest (Attrition)")
plt.show()


# ==========================================================
# Logistic Regression Model
# ==========================================================
log_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:,1]

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log) * 100)
print("Logistic Regression AUROC:", roc_auc_score(y_test, y_prob_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))


# ==========================================================
# Decision Tree Model
# ==========================================================
dt_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
y_prob_dt = dt_model.predict_proba(X_test)[:,1]

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt) * 100)
print("Decision Tree AUROC:", roc_auc_score(y_test, y_prob_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))


# ==========================================================
# XGBoost Model
# ==========================================================
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:,1]

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb) * 100)
print("XGBoost AUROC:", roc_auc_score(y_test, y_prob_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))


# ==========================================================
# Naive Bayes Model
# ==========================================================
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
y_prob_nb = nb_model.predict_proba(X_test)[:,1]

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb) * 100)
print("Naive Bayes AUROC:", roc_auc_score(y_test, y_prob_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))


# ==========================================================
# Support Vector Machine (SVM) Model
# ==========================================================
svm_model = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)[:,1]

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm) * 100)
print("SVM AUROC:", roc_auc_score(y_test, y_prob_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))


# ==========================================================
# AdaBoost Model
# ==========================================================
ada_model = AdaBoostClassifier(n_estimators=200, random_state=42)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)
y_prob_ada = ada_model.predict_proba(X_test)[:,1]

print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada) * 100)
print("AdaBoost AUROC:", roc_auc_score(y_test, y_prob_ada))
print("Classification Report:\n", classification_report(y_test, y_pred_ada))
