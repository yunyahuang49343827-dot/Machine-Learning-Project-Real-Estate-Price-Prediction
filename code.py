#Use this cell to import all the required libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, cross_val_score

# Load dataset
df = pd.read_excel(r"C:\7217 ML-ASS\Assignment_Data.xlsx")

# Check basic data information
print("Basic Information：")
print(df.info())

# Distribution of target variable
print("\nDistribution of target variable 'price_outcome':")
print(df['price_outcome'].value_counts())

# Remove 'Equal'
df = df[df['price_outcome'] != 'Equal']
print("\nNumber of records after removing 'Equal':", df.shape[0])

# Check missing values in numerical columns
num_cols = [ 'listed_price', 'days_on_market', 'number_of_beds', 
    'number_of_baths', 'number_of_parks', 'property_size',
    'suburb_days_on_market', 'suburb_median_price']
print("\nMissing value summary for numerical columns:")
print(df[num_cols].isna().sum())

# Fill missing property_size
if df['property_size'].isna().sum() > 0:
    median_size = df['property_size'].median()
    df['property_size'] = df['property_size'].fillna(median_size)
    print(f"\nFilled missing values in property_size with median value {median_size}.")

# Remove invalid days_on_market
num_negative = df[df['days_on_market'] < 0].shape[0]
print(f"Number of rows where days_on_market < 0: {num_negative}")
df = df[df['days_on_market'] >= 0]
print(f"Number of rows after removing negative days_on_market: {df.shape[0]}")

max_days = df['days_on_market'].max()
print(f"Maximum value of days_on_market: {max_days}")

df = df[df['days_on_market'] <= 500]
print(f"Number of rows after removing days_on_market > 500: {df.shape[0]}")

# Plot target distribution
sns.countplot(x=df['price_outcome'])
plt.title('Price Outcome Distribution')
plt.show()

# Plots for listed_price and days_on_market
x_limits_main = {'listed_price': (0, 4_000_000), 'days_on_market': (0, 300)}

for col in ['listed_price', 'days_on_market']:
    plt.figure(figsize=(6, 4))
    lower, upper = x_limits_main[col]
    filtered_data = df[df[col] <= upper][col]

    sns.histplot(filtered_data, kde=True)
    plt.title(f'{col} Distribution')
    plt.xlim(lower, upper)
    plt.ylabel('Number of Properties')
    ax = plt.gca()

    if col == 'listed_price':
        ax.xaxis.set_major_locator(mtick.MultipleLocator(1_000_000))
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x/1e6)}M'))
    else:
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    plt.show()

# Multi subplot: beds, baths, parks, property_size
cols = ['number_of_beds', 'number_of_baths', 'number_of_parks', 'property_size']
x_limits = {'number_of_beds': (0, 6), 'number_of_baths': (0, 6), 'number_of_parks': (0, 6), 'property_size': (0, 1250)}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, col in zip(axes.flatten(), cols):
    lower, upper = x_limits[col]
    filtered_data = df[df[col] <= upper][col]

    if col in ['number_of_beds', 'number_of_baths', 'number_of_parks']:
        sns.histplot(filtered_data, kde=False, binwidth=1, ax=ax)
    else:
        sns.histplot(filtered_data, kde=True, ax=ax)

    ax.set_title(f'{col} Distribution')
    ax.set_xlim(lower, upper)
    ax.set_ylabel('Number of Properties')

    if col == 'property_size':
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    else:
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x)}'))

plt.tight_layout()
plt.show()

# Category plots
category_cols = ['property_classification', 'property_sub_classification']
colors = {'Higher': '#FFA500', 'Lower': '#B0B0B0'}

for col in category_cols:
    prop_df = (df.groupby([col, 'price_outcome']).size()
                 .groupby(level=0).apply(lambda x: x / x.sum())
                 .unstack())
    prop_df.index = prop_df.index.get_level_values(0)

    prop_df[['Higher', 'Lower']].plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f'{col} vs Price Outcome')
    plt.ylabel('Proportion')

    if col == 'property_sub_classification':
        plt.xticks(rotation=90)
    else:
        plt.xticks(rotation=0)

    plt.show()

# Text cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df['cleaned_description'] = df['listing_description'].apply(clean_text)
df['description_word_count'] = df['listing_description'].apply(lambda x: len(str(x).split()))
df['description_char_count'] = df['listing_description'].apply(lambda x: len(str(x)))
df['description_avg_word_length'] = df['listing_description'].apply(
    lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
)

# ==== STEP 1: TF-IDF (1-2gram) + SVD 50維 ====
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=5)
X_tfidf = tfidf.fit_transform(df['cleaned_description'].fillna(""))

svd = TruncatedSVD(n_components=50, random_state=42)
tfidf_svd = svd.fit_transform(X_tfidf)

tfidf_cols = [f"tfidf_svd_{i+1}" for i in range(tfidf_svd.shape[1])]
tfidf_df = pd.DataFrame(tfidf_svd, columns=tfidf_cols, index=df.index)

df = pd.concat([df, tfidf_df], axis=1)
print("✅ Added TF-IDF SVD features:", tfidf_df.shape)

# y
df_filtered['price_outcome_binary'] = df_filtered['price_outcome'].apply(lambda x: 1 if x == 'Higher' else 0)

# 丟掉不該進模型的欄位（加入 cleaned_description）
drop_cols = [
    'price_outcome', 'price_outcome_binary',
    'property_address', 'listing_description', 'listed_date',
    'cleaned_description'  # << 新增：不用再把文字本體編碼進模型
]

X = df_filtered.drop(columns=[c for c in drop_cols if c in df_filtered.columns]).copy()
y = df_filtered['price_outcome_binary']

# 仍然會自動包含剛剛新增的 tfidf_svd_1..50（它們是數值欄）
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

print("✅ X columns sample:", X.columns[:10])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # << 加入 stratify
)


# Sentiment
sia = SentimentIntensityAnalyzer()
tqdm.pandas()
df['listing_description'] = df['listing_description'].fillna('')
df['description_sentiment'] = df['listing_description'].progress_apply(lambda x: sia.polarity_scores(str(x))['compound'])
print(df[['listing_description', 'description_sentiment']].iloc[1:50])



# Sentiment plots
plt.figure(figsize=(8, 4))
sns.histplot(df['description_sentiment'], bins=50, kde=True, color='#4C72B0')
plt.title('Description Sentiment Score Distribution')
plt.xlabel('Sentiment Score (compound)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(
    x="price_outcome",
    y="description_sentiment",
    data=df
)
plt.title('Description Sentiment by Price Outcome')
plt.xlabel('Price Outcome')
plt.ylabel('Sentiment Score')
plt.show()

# Suburb median imputation
df['suburb_median_price_filled'] = df.groupby('property_classification')['suburb_median_price'].transform(
    lambda x: x.fillna(x.median())
)
print("✅ Completed median fill per property_classification for suburb_median_price")

# Filter unreasonable values
high_median_cap = df['suburb_median_price_filled'].quantile(0.99)
df_clean = df[(df['listed_price'] > 100) & (df['suburb_median_price_filled'] <= high_median_cap)]
print(f"✅ Cleaned dataset: {df_clean.shape[0]} / {df.shape[0]} rows retained")

# Feature engineering
df_clean['price_ratio'] = df_clean['listed_price'] / df_clean['suburb_median_price_filled']
print("✅ Added feature: price_ratio")

# Outlier removal
lower_bound = df_clean['price_ratio'].quantile(0.01)
upper_bound = df_clean['price_ratio'].quantile(0.99)
original_count = df_clean.shape[0]
df_filtered = df_clean[(df_clean['price_ratio'] >= lower_bound) & (df_clean['price_ratio'] <= upper_bound)]
new_count = df_filtered.shape[0]
print(f"✅ Removed price_ratio outliers → {new_count} / {original_count}")

# Hot suburb flag
suburb_days_mean = df_filtered['suburb_days_on_market'].mean()
df_filtered['hot_suburb_flag'] = df_filtered['suburb_days_on_market'].apply(lambda x: 1 if pd.notnull(x) and x < suburb_days_mean else 0)

# Listing month
df_filtered['listing_month'] = pd.to_datetime(df_filtered['listed_date']).dt.month

# Encode and split
df_filtered['price_outcome_binary'] = df_filtered['price_outcome'].apply(lambda x: 1 if x == 'Higher' else 0)
X = df_filtered.drop(columns=['price_outcome', 'price_outcome_binary', 'property_address', 'listing_description', 'listed_date'])
y = df_filtered['price_outcome_binary']
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

print("✅ X dtypes after cleanup:")
print(X.dtypes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest baseline
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("=== Random Forest Report ===")
print(classification_report(y_test, rf_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

rf_probs = rf.predict_proba(X_test)[:, 1]
print(f"Random Forest ROC-AUC: {roc_auc_score(y_test, rf_probs):.3f}")
RocCurveDisplay.from_estimator(rf, X_test, y_test, name='Random Forest')
plt.show()

# Tuned Random Forest
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [5, 10],
    'min_samples_split': [5, 10],
    'max_features': ['sqrt'],
    'class_weight': ['balanced']
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("✅ Best Parameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_
best_rf_preds = best_rf.predict(X_test)

print("\n=== Tuned Random Forest Report ===")
print(classification_report(y_test, best_rf_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, best_rf_preds))

cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='f1', n_jobs=-1)
print(f"Cross-validated F1 scores: {cv_scores}")
print(f"Mean F1 score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

best_rf_probs = best_rf.predict_proba(X_test)[:, 1]
print(f"Tuned Random Forest ROC-AUC: {roc_auc_score(y_test, best_rf_probs):.3f}")
RocCurveDisplay.from_estimator(best_rf, X_test, y_test, name='Tuned Random Forest')
plt.show()
