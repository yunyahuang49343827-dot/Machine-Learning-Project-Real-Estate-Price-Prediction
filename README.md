# 🏡 Real Estate Price Prediction with Machine Learning  

## 📌 Project Overview  
This project was developed for **BISM7217 – Machine Learning for Analysis (UQ, 2025)**.  
The aim is to predict whether a property in the Australian real estate market will be sold at a **higher** or **lower** price than its listed price.  

The task was a **binary classification problem**, with a dataset of approximately 7,000 property transactions.  

---

## 📊 Dataset  
- **Source:** Australian property sales (Feb 2022 – Feb 2023)  
- **Target variable:** `price_outcome`  
  - *Higher*: sold price > listed price  
  - *Lower*: sold price < listed price  
- **Features included:**  
  - Property details (beds, baths, size, suburb info)  
  - Listing details (listed price, days on market, description text, listed date)  

---

## 🔧 Methods  

### 1. Data Exploration & Preprocessing  
- Removed invalid cases (`price_outcome = Equal`, unrealistic `days_on_market`).  
- Filled missing values using median imputation.  
- Visualised distributions to check label balance and outliers.  

### 2. Text Preparation  
- Cleaned property descriptions using **NLTK** (tokenization, stopword removal, lemmatization).  
- Attempted sentiment features, but found limited predictive value.  

### 3. Feature Engineering  
- **Price Ratio**: listed price ÷ suburb median price  
- **Hot Suburb Flag**: binary indicator for popular suburbs  
- **Listing Month**: extracted from listed date  
- **Average Room Size**: property size ÷ number of bedrooms  

### 4. Model Building  
- Implemented **Decision Tree, Random Forest, SVM, Naïve Bayes** classifiers.  
- **Random Forest** performed best due to robustness with mixed feature types.  

### 5. Evaluation & Tuning  
- Metrics: precision, recall, F1-score, confusion matrix, ROC-AUC.  
- Random Forest baseline achieved **AUC = 0.902**, with strong recall for “Higher” outcomes.  
- Hyperparameter tuning with GridSearchCV reduced overfitting but lowered AUC to 0.82.  
- Key predictors: **Price Ratio** and **Listing Month**; sentiment features had minimal effect.  

---

## ✅ Results & Insights  
- **Random Forest** was the most effective model overall.  
- **Pricing strategy:** Properties priced close to the suburb median and listed between **May–September** were more likely to sell above the listed price.  
- Text sentiment from descriptions had limited predictive value.  
- **Recommendation:** Agencies should align listing prices with market medians and optimize listing timing.  

---

## 💻 Technologies Used  
- **Python**: pandas, numpy, scikit-learn, seaborn, matplotlib  
- **NLTK**: text preprocessing (tokenization, stopword removal, lemmatization)  
- **Jupyter Notebook**: analysis and reporting  
