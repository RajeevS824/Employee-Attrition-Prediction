# 📌 Employee Attrition Analysis & Prediction

**HR Analytics | Machine Learning | Streamlit Dashboard**
 
This project focuses on analyzing employee attrition, identifying key factors influencing turnover, and building machine learning models to predict both **employee attrition** and **performance ratings**. An interactive **Streamlit application** is included for real-time prediction and insights.

---

    https://employee-attrition-prediction-ecmujpvixzodrf2nfxrh9v.streamlit.app/
    
## 📘 **Project Title**

### **Employee Attrition Analysis and Prediction**

---

## 🎯 **Skills Takeaway**

* Data Preprocessing & Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Machine Learning (Classification Models)
* Model Evaluation
* Streamlit App Development
* HR Analytics & Business Insights

---

## 🏢 **Domain**

### **HR Analytics – Predicting & Preventing Employee Attrition**

---

# 🚨 Problem Statement

Employee turnover significantly impacts organizations through recruitment costs, productivity loss, and workflow disruption.
This project aims to:

✔ Understand patterns and key reasons behind employee attrition
✔ Identify at-risk employees proactively
✔ Build predictive ML models to support HR decision-making
✔ Provide dashboards & insights to help implement retention strategies

---

# 📌 **Business / Real-Life Use Cases**

### **1. Employee Retention**

Detect employees likely to leave & take preventive actions (bonuses, role changes, training).

### **2. Cost Optimization**

Reduce hiring, onboarding, and training costs caused by high attrition.

### **3. Workforce Planning**

Identify departments or roles with high turnover & improve HR policies.

### **4. Performance Management**

Predict performance ratings to plan promotions or training programs.

### **5. HR Strategy Development**

Use analytics-based insights instead of assumptions.

---

# 🧠 **Approach (Step-by-Step)**

### **1. Data Collection & Preprocessing**

* Load dataset (Excel)
* Remove unnecessary constant columns
* Handle missing values & outliers
* Encode categorical variables using Label Encoding
* Normalize numerical features with StandardScaler
* Balance target using SMOTE

### **2. Exploratory Data Analysis (EDA)**

* Attrition distribution
* Salary vs attrition
* Overtime impact
* Department-wise attrition
* Age/Experience vs job satisfaction

### **3. Feature Engineering**

* Tenure buckets
* Work-life balance indicators
* Promotion gap features
* Manager relationship indicators

### **4. Model Building**

Models used:

* Logistic Regression
* Decision Tree
* Random Forest Classifier ✔ (best performance)

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* AUC-ROC
* Confusion Matrix

### **5. Dashboard & Visualization**

Built using **Streamlit**:

* Real-time Attrition Prediction
* Performance Rating Prediction
* HR Recommendations
* Interactive forms
* Automated insights

---

# 📊 **Results**

### ✔ **Predictive Accuracy**

* Achieved high accuracy (>85% in balanced conditions).
* Random Forest outperformed other models.

### ✔ **Key Attrition Drivers**

* Overtime
* Low Job Satisfaction
* Low Salary
* Long commute distance
* Poor work-life balance
* Years since last promotion

### ✔ **At-Risk Employee Identification**

* Model provides probability of attrition
* HR can prioritize retention efforts

### ✔ **Visual Dashboards**

* Streamlit app gives real-time predictions
* Easy interface for HR teams

---

# 📈 **Evaluation Metrics**

| Metric               | Description                                              |
| -------------------- | -------------------------------------------------------- |
| **Accuracy**         | Correct predictions / total inputs                       |
| **Precision**        | Correctly predicted attrition out of predicted attrition |
| **Recall**           | Correctly identified actual attrition cases              |
| **F1 Score**         | Harmonic mean of precision & recall                      |
| **AUC-ROC**          | Binary classification performance                        |
| **Confusion Matrix** | TP, TN, FP, FN breakdown                                 |

---

# 💼 **Business & Technical Impact**

### ✔ Reduced Attrition (10–20% possible)

Better retention strategies based on predictions.

### ✔ Significant Cost Savings

Lower hiring and training costs.

### ✔ Data-Driven HR Strategy

Insights help avoid bias-driven decisions.

### ✔ Workforce Stability

Predictive insights improve workforce planning.

---

# 🚀 **Future Enhancements**

* Add more ML models (XGBoost, CatBoost, ANN)
* Deploy Streamlit app on cloud (AWS / GCP / Azure)
* Add Employee Promotion Prediction Model
* Add Explainable AI (SHAP) for feature importance
* Integrate HR chatbot for automated insights
* Build a full HRMS dashboard with login authentication

---

# 🏗 **System Architecture**

```
           +-----------------------------+
           |        Raw Dataset          |
           +--------------+--------------+
                          |
                          v
              Data Preprocessing Engine
        (Cleaning, Encoding, Scaling, SMOTE)
                          |
                          v
               Machine Learning Models
       (Attrition Model + Performance Model)
                          |
                          v
                Streamlit Web Interface
         - User Inputs
         - Predictions (Attrition/Performance)
         - HR Recommendations
                          |
                          v
                  Final Output to HR
```

---

# 🔧 **Features**

### **1. Attrition Prediction Module**

* Predict leave/stay
* Probability score
* HR Recommendations

### **2. Performance Prediction Module**

* Predict rating (1–4)
* Suggest training & growth actions

### **3. Data Preview**

* View dataset inside the app

### **4. Intelligent HR Insights**

* Automated guidance based on model predictions

---

# 📁 **Project Structure**

```
Employee-Attrition-Prediction/
│
├── streamlit_app.py           # Main Streamlit application

├── data/                      # Employee-Attrition.xlsx
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
 
```

---

# ▶️ **How to Run the Project**

### **1. Clone the Repository**

```bash
git clone https://github.com/RajeevS824/Employee-Attrition-Prediction
cd Employee-Attrition-Prediction
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run the Streamlit App**

```bash
streamlit run streamlit_app.py
```

### **4. Upload Dataset**

Ensure `Employee-Attrition.xlsx` is in the project folder.

---

# 🛠 **Tech Stack**

### **Programming & Libraries**

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Matplotlib

### **Model**

* Random Forest Classifier
* Logistic Regression (optional)

### **Frontend / Dashboard**

* **Streamlit**

### **Tools**

* Jupyter Notebook
* GitHub
* Excel

---

# 🗂 **Dataset**

Dataset contains 35+ employee features including:

* Age
* Department
* Salary
* Job Role
* Job Satisfaction
* Overtime
* Tenure
* Work-Life Balance
* Promotion History
* Manager Interaction
* Performance Rating

(Full description included in README)

---

# 🎯 **Example Predictions**

### ✔ Attrition Prediction

Predict if employee will stay or leave.

### ✔ Performance Rating Prediction

Predict rating from 1–4.

### ✔ Promotion Likelihood (Future)

Predict time to next promotion.

---


