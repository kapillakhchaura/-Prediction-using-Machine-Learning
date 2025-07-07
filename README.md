Heart Disease Prediction using Machine Learning

Internship Task 2 – CODTECH IT SOLUTIONS  
Intern: Kapil Lakhchaura  
Duration: May 30, 2025 – July 30, 2025  


Project Objective

The objective of this project is to build a Machine Learning model to predict whether a patient has heart disease or not, based on clinical features. This is a binary classification task using the processed Cleveland Heart Disease dataset from the UCI Machine Learning Repository.



Dataset Details

Name: Processed Cleveland Heart Disease Dataset  
Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)  
Instances: 303  
Used Attributes: 14 medical features (out of 76 total)  
Target Variable: `num` → Converted to `0` (No Disease), `1` (Disease Present)



Features Used

| Feature      | Description |
|--------------|-------------|
| age          | Age in years |
| sex          | Sex (1 = male, 0 = female) |
| cp           | Chest pain type (1–4) |
| trestbps     | Resting blood pressure |
| chol         | Serum cholesterol |
| fbs          | Fasting blood sugar > 120 mg/dl |
| restecg      | Resting electrocardiographic results |
| thalach      | Max heart rate achieved |
| exang        | Exercise induced angina |
| oldpeak      | ST depression induced by exercise |
| slope        | Slope of the ST segment |
| ca           | Number of major vessels (0–3) |
| thal         | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect) |
| target       | 0 = No Disease, 1 = Disease |



Tools & Technologies Used

Python  
Pandas, NumPy  
Scikit-learn  
PyCharm (IDE)



Steps Performed

1. Loaded the dataset using `pandas`
2. Replaced missing values (`?`) and dropped incomplete rows
3. Converted all data to numeric types
4. Converted target column to binary (0 or 1)
5. Split data into training and testing sets (80:20)
6. Trained a Logistic Regression model
7. Evaluated using accuracy score and classification report



Model Results

text
Accuracy Score: 88.33%
Classification Report:

              precision    recall  f1-score   support
           0       0.91       0.89       0.90        36
           1       0.84       0.88       0.86        24

    accuracy                           0.88        60
   macro avg       0.88       0.88       0.88        60
weighted avg       0.88       0.88       0.88        60
