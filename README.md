# Machine_Learning_Projects



### **Project Title: Car Price Prediction Using Machine Learning in Python**

The Car Price Prediction project focuses on building a machine learning model to estimate the selling price of used cars based on various key attributes. With the growing online used-car market, predicting accurate car prices helps sellers set fair prices and buyers make informed decisions. This project demonstrates the complete machine learning pipeline—from data preprocessing and feature engineering to model training, evaluation, and optimization.

#### 🔹 **Objective:**
To develop a regression model that predicts car prices using features such as brand, model, manufacturing year, fuel type, transmission, engine size, kilometers driven, and ownership history.

#### 🔹 **Tools and Libraries Used:**
- **Python** – Programming language for data manipulation and modeling  
- **pandas**, **NumPy** – Data preprocessing, manipulation, and analysis  
- **matplotlib**, **seaborn** – Data visualization and exploratory data analysis (EDA)  
- **scikit-learn** – Machine learning models, preprocessing, and evaluation  
- **LabelEncoder**, **OneHotEncoder** – Encoding categorical variables  
- **Linear Regression**, **Ridge**, **Lasso**, **Random Forest Regressor** – Supervised learning models for regression  

#### 🔹 **Project Workflow:**

1. **Data Collection & Loading:**  
   The dataset was collected from online car listing platforms (e.g., Kaggle, OLX, CarDekho) and loaded using pandas.

2. **Data Cleaning & Preprocessing:**  
   - Removed irrelevant columns and duplicate records  
   - Handled missing values using imputation strategies  
   - Converted categorical variables (e.g., fuel type, transmission) using Label Encoding and One-Hot Encoding  
   - Derived new features like car age from the manufacturing year

3. **Exploratory Data Analysis (EDA):**  
   - Visualized price trends across car brands and models  
   - Analyzed correlations between numerical features and car price  
   - Detected and handled outliers in mileage and price data

4. **Model Building:**  
   - Trained multiple regression models, including **Linear Regression**, **Ridge Regression**, **Lasso**, and **Random Forest Regressor**  
   - Evaluated models using metrics like **R² Score**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**  
   - **Random Forest** yielded the best performance with balanced bias-variance trade-off
