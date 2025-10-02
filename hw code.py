# 📦 Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 📥 Download NLTK resources (only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 📄 Load the dataset
file_path = r"C:\Users\user\Desktop\Machine learning\HW\Assignment_Data.xlsx"
df = pd.read_excel(file_path)

# -----------------------------------
# ✅ Numerical Data Exploration & Preparation
# -----------------------------------

# Remove 'Equal' from target variable
df = df[df['price_outcome'].isin(['Higher', 'Lower'])].reset_index(drop=True)

# Fill missing values
df['number_of_baths'] = df['number_of_baths'].fillna(df['number_of_baths'].median())
df['number_of_parks'] = df['number_of_parks'].fillna(0)  # or use median
df['property_size'] = df['property_size'].fillna(df['property_size'].median())
df['suburb_median_price'] = df['suburb_median_price'].fillna(df['suburb_median_price'].median())

# Convert and split date into year and month
df['listed_date'] = pd.to_datetime(df['listed_date'])
df['listed_year'] = df['listed_date'].dt.year
df['listed_month'] = df['listed_date'].dt.month

# Feature: Average room size
df['avg_room_size'] = df['property_size'] / df['number_of_beds'].replace(0, 1)

# 🔢 Descriptive statistics
print("\n🔍 Numeric summary:\n", df.describe())

# 📊 Visualizations

# Heatmap of missing values
plt.figure(figsize=(8, 4))
sns.heatmap(df.isnull(), cbar=False, cmap='YlOrBr')
plt.title("Missing Value Heatmap")
plt.show()

# Label distribution
df['price_outcome'].value_counts().plot(kind='bar', color='skyblue', title="Price Outcome Distribution")
plt.xlabel("Price Outcome")
plt.ylabel("Count")
plt.show()

# Histogram of property size
plt.figure(figsize=(6, 4))
sns.histplot(df['property_size'], bins=20, kde=True)
plt.title("Distribution of Property Size")
plt.xlabel("Property Size (sqm)")
plt.ylabel("Frequency")
plt.show()

# Boxplot of listed price by outcome
plt.figure(figsize=(8, 4))
sns.boxplot(x='price_outcome', y='listed_price', data=df)
plt.title("Listed Price by Price Outcome")
plt.xlabel("Price Outcome")
plt.ylabel("Listed Price")
plt.show()

# -----------------------------------
# ✅ Text Preparation (Clean listing_description)
# -----------------------------------

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation & special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Clean text column
df['description_cleaned'] = df['listing_description'].fillna('').apply(clean_text)

# Show cleaned result samples
print(df[['listing_description', 'description_cleaned']].head(2))

# -----------------------------------
# ✅ Save the cleaned dataset to Excel
# -----------------------------------
output_path = r"C:\Users\user\Desktop\Machine learning\HW\Assignment_Data_Cleaned.xlsx"
df.to_excel(output_path, index=False)
print(f"✅ Cleaned dataset saved to:\n{output_path}")
