# 👨‍💼 Employee Attrition & Performance Dashboard
    https://employee-attrition-prediction-ecmujpvixzodrf2nfxrh9v.streamlit.app/
📌 **Project Overview**
Employee turnover is one of the biggest challenges in HR management, leading to increased costs, reduced productivity, and disruption in teams. This project builds an end-to-end machine learning solution with interactive dashboards to help:

* **HR Teams** → identify at-risk employees and design retention strategies.
* **Management** → optimize costs by reducing recruitment & training needs.
* **Researchers** → study attrition trends, performance drivers, and employee engagement.

The project integrates **Python (EDA & ML), Scikit-learn (models), and Streamlit (dashboard)** to deliver actionable insights into attrition risk and performance prediction.

---

## 🛠️ What I Did in This Project

### 1. Data Preparation (Python + Pandas)

* Loaded & cleaned HR Analytics Employee Attrition dataset.
* Dropped irrelevant columns (`EmployeeCount`, `Over18`, `StandardHours`).
* Encoded categorical variables (LabelEncoder).
* Applied feature scaling (`StandardScaler` ).
* Balanced imbalanced classes with **SMOTE**.

### 2. Exploratory Data Analysis (EDA)

* Visualized **attrition distribution**, income levels, age trends.
* Correlation heatmap of key factors (e.g., overtime, job satisfaction).
* Bar plots & box plots: attrition vs income, department, promotions.

### 3. Machine Learning Models (Scikit-learn )

Implemented and compared multiple models:

* Logistic Regression
* Decision Tree
* Random Forest
* Naive Bayes
* Support Vector Machine (SVM)
* AdaBoost

**Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Classification Report.

### 4. Streamlit Dashboard

* **Page 1 – Home:** Project objectives and dataset preview.
* **Page 2 – Attrition Prediction:**

  * Predict whether an employee is likely to leave.
  * Inputs: Overtime, Job Level, Income, Tenure, etc.
  * Business insights + HR recommendations.
* **Page 3 – Performance Prediction:**

  * Predict employee performance rating.
  * Inputs: Years with Manager, Salary Hike %, Job Involvement, etc.
  * Automatic suggestions for low, average, and high performers.

---

## 🎯 Motive of the Project

* Reduce employee attrition by identifying risk early.
* Improve workforce planning with performance insights.
* Support HR teams with data-driven decision making.

---

## 🌍 Real-Life Use Cases

* **HR:** Proactively retain employees at high risk of leaving.
* **Management:** Align salaries, promotions, and workloads with data-driven insights.
* **Organizations:** Reduce recruitment and training costs by improving retention.
* **Researchers:** Study the drivers of employee satisfaction, engagement, and turnover.

---

## ✅ Conclusion

This project demonstrates how **EDA + Machine Learning + Streamlit** can work together to transform raw HR data into actionable insights. By integrating predictive models and interactive dashboards, organizations can improve employee retention, optimize costs, and foster a more engaged workforce.


