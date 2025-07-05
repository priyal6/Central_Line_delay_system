import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import lightgbm as lgb
import requests
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings('ignore')
import sys
import os


if os.name == 'nt':  # Windows
    os.system('chcp 65001 > nul')  # Set console to UTF-8
    sys.stdout.reconfigure(encoding='utf-8')

# === Load and Clean Data ===
print("Loading and cleaning data...")
try:
    df = pd.read_csv('data/central_train_log.csv', parse_dates=['timestamp', 'predictedArrival'])
    print(f"Original dataset shape: {df.shape}")
    
    # Check for missing values
    print(f"Missing values in predictedArrival: {df['predictedArrival'].isna().sum()}")
    df.dropna(subset=['predictedArrival'], inplace=True)
    print(f"Shape after dropping NaN: {df.shape}")
    
    # Ensure timestamps are timezone-aware
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['predictedArrival'] = pd.to_datetime(df['predictedArrival'], utc=True)
    
    print("Data loaded successfully!")
    
except FileNotFoundError:
    print("Error: Could not find 'data/central_train_log.csv'")
    print("Please ensure the file exists in the correct path.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

#Feature Engineering
print("\nEngineering features...")

# Calculate delay in seconds
df['delay_sec'] = (df['predictedArrival'] - df['timestamp']).dt.total_seconds()

# Binary classification: delay > 2 minutes (120 seconds)
df['delay_bin'] = (df['delay_sec'] > 120).astype(int)

# Time-based features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_peak_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

# Additional features for better prediction
df['month'] = df['timestamp'].dt.month
df['is_rush_morning'] = df['hour'].isin([7, 8, 9]).astype(int)
df['is_rush_evening'] = df['hour'].isin([17, 18, 19]).astype(int)

# Station encoding
df['station'] = df['station'].astype(str)

# Display basic statistics
print(f"Delay distribution:")
print(f"  No delay (≤2 min): {(df['delay_bin'] == 0).sum()} ({(df['delay_bin'] == 0).mean():.1%})")
print(f"  Delayed (>2 min): {(df['delay_bin'] == 1).sum()} ({(df['delay_bin'] == 1).mean():.1%})")

print(f"\nDelay statistics:")
print(f"  Mean delay: {df['delay_sec'].mean():.1f} seconds")
print(f"  Median delay: {df['delay_sec'].median():.1f} seconds")
print(f"  Max delay: {df['delay_sec'].max():.1f} seconds")

# === Prepare Features ===
# Basic features
features = ['hour', 'day_of_week', 'is_weekend', 'is_peak_hour', 'month', 
           'is_rush_morning', 'is_rush_evening']

# Add station encoding if there are multiple stations
if df['station'].nunique() > 1:
    print(f"\nFound {df['station'].nunique()} unique stations")
    
    top_stations = df['station'].value_counts().head(10).index
    for station in top_stations:
        df[f'station_{station}'] = (df['station'] == station).astype(int)
        features.append(f'station_{station}')

X = df[features]
y = df['delay_bin']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Features used: {features}")

# === Train/Test Split ===
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# === Train LightGBM Model ===
print("\nTraining LightGBM model...")


clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=1
)
clf.fit(X_train, y_train)

#EVALULATE MODEL
print("\nEvaluating model...")
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

#VISUALIZE 
plt.figure(figsize=(15, 10))

# 1. Feature Importance
plt.subplot(2, 3, 1)
importance = clf.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(importance)
plt.barh(range(len(sorted_idx)), importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.title("Feature Importance")
plt.xlabel("Importance")

# 2. Confusion Matrix
plt.subplot(2, 3, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")

# 3. Delay Distribution by Hour
plt.subplot(2, 3, 3)
hourly_delay = df.groupby('hour')['delay_bin'].mean()
plt.bar(hourly_delay.index, hourly_delay.values)
plt.title("Delay Rate by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Delay Rate")

# 4. Delay Distribution by Day of Week
plt.subplot(2, 3, 4)
dow_delay = df.groupby('day_of_week')['delay_bin'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
plt.bar(range(7), dow_delay.values)
plt.xticks(range(7), days)
plt.title("Delay Rate by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Delay Rate")

# 5. Delay Distribution
plt.subplot(2, 3, 5)
plt.hist(df['delay_sec'], bins=50, alpha=0.7, edgecolor='black')
plt.axvline(x=120, color='red', linestyle='--', label='2 min threshold')
plt.title("Delay Distribution")
plt.xlabel("Delay (seconds)")
plt.ylabel("Frequency")
plt.legend()

# 6. ROC Curve
from sklearn.metrics import roc_curve
plt.subplot(2, 3, 6)
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.tight_layout()
plt.show()

# === Additional Analysis ===
print("\nAdditional Analysis:")

# Peak hours analysis
peak_delays = df[df['is_peak_hour'] == 1]['delay_bin'].mean()
non_peak_delays = df[df['is_peak_hour'] == 0]['delay_bin'].mean()
print(f"Peak hours delay rate: {peak_delays:.1%}")
print(f"Non-peak hours delay rate: {non_peak_delays:.1%}")

# Weekend vs weekday analysis
weekend_delays = df[df['is_weekend'] == 1]['delay_bin'].mean()
weekday_delays = df[df['is_weekend'] == 0]['delay_bin'].mean()
print(f"Weekend delay rate: {weekend_delays:.1%}")
print(f"Weekday delay rate: {weekday_delays:.1%}")

# Model insights
print(f"\nModel Performance Summary:")
print(f"- The model achieved {roc_auc_score(y_test, y_proba):.1%} ROC-AUC score")
print(f"- Most important features: {feature_names[np.argsort(importance)[-3:][::-1]].tolist()}")

# === Prediction Function ===
def predict_delay(hour, day_of_week, is_weekend=None, station=None):
    """
    Predict delay probability for given conditions
    """
    if is_weekend is None:
        is_weekend = 1 if day_of_week >= 5 else 0
    
    
    feature_dict = {
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'is_peak_hour': 1 if hour in [7, 8, 9, 17, 18, 19] else 0,
        'month': datetime.now().month,
        'is_rush_morning': 1 if hour in [7, 8, 9] else 0,
        'is_rush_evening': 1 if hour in [17, 18, 19] else 0,
    }
    
    
    for feature in features:
        if feature.startswith('station_'):
            feature_dict[feature] = 0
    
   
    input_df = pd.DataFrame([feature_dict])
    
   
    prob = clf.predict_proba(input_df)[0][1]
    return prob

# Example predictions
print(f"\nExample Predictions:")
print(f"Monday 8 AM: {predict_delay(8, 0):.1%} delay probability")
print(f"Friday 6 PM: {predict_delay(18, 4):.1%} delay probability")
print(f"Saturday 2 PM: {predict_delay(14, 5):.1%} delay probability")

print("\nModel training complete!")
