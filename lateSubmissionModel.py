"""
Late Submission Detector - Fixed Version
Name - ADITYA BHARDWAJ
Section - D2
Roll No - 07
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load the fixed dataset
print("Loading dataset...")
df = pd.read_csv("DataSet.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S')

# Remove duplicate submissions (keep first submission per student)
df = df.sort_values('Timestamp').drop_duplicates(
    subset=['Student Full Name (As per University Records)'], 
    keep='first'
)
print(f"After removing duplicates: {df.shape}")

# Define deadline
deadline = pd.Timestamp("2026-01-18 18:00:00")

# Create target variable (0 = on time, 1 = late)
df['late'] = (df['Timestamp'] > deadline).astype(int)

# Feature Engineering
df['hour'] = df['Timestamp'].dt.hour
df['minute'] = df['Timestamp'].dt.minute
df['day'] = df['Timestamp'].dt.day
df['weekday'] = df['Timestamp'].dt.weekday  # 0=Monday, 6=Sunday
df['is_weekend'] = (df['weekday'] >= 5).astype(int)

# Time category
def get_time_category(hour):
    if 6 <= hour < 12:
        return 0  # Morning
    elif 12 <= hour < 17:
        return 1  # Afternoon
    elif 17 <= hour < 21:
        return 2  # Evening
    elif 21 <= hour < 24:
        return 3  # Night
    else:
        return 4  # Late night/Early morning

df['time_category'] = df['hour'].apply(get_time_category)

# Hours after deadline (0 for on-time submissions)
df['hours_after_deadline'] = np.maximum(
    (df['Timestamp'] - deadline).dt.total_seconds() / 3600, 0
)

# Encode Section (D1=0, D2=1)
df['section_encoded'] = df['B.Tech Section'].map({'D1': 0, 'D2': 1})

# Select features
feature_columns = ['hour', 'weekday', 'is_weekend', 'time_category', 
                   'section_encoded', 'hours_after_deadline']

X = df[feature_columns]
y = df['late']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
print(f"Late submissions in training: {y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f}%)")
print(f"Late submissions in testing: {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.1f}%)")

# Train Random Forest model
print("\n" + "="*50)
print("Training Random Forest Classifier...")
print("="*50)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n" + "="*50)
print("TEST SET RESULTS")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['On Time', 'Late']))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

# Feature Importance
print("\n" + "="*50)
print("FEATURE IMPORTANCE")
print("="*50)
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# Analyze submissions by time
print("\n" + "="*50)
print("SUBMISSION PATTERN ANALYSIS")
print("="*50)

# Count submissions by hour
submission_counts = df.groupby(df['Timestamp'].dt.hour).size()
print("\nSubmissions by hour of day:")
print(submission_counts.sort_index())

# Late vs On-time by section
section_stats = df.groupby('B.Tech Section')['late'].agg(['count', 'sum', 'mean'])
section_stats.columns = ['Total', 'Late', 'Late Percentage']
print("\nSection-wise statistics:")
print(section_stats)

# Late vs On-time by time category
time_stats = df.groupby('time_category')['late'].agg(['count', 'sum', 'mean'])
time_category_names = {0: 'Morning', 1: 'Afternoon', 2: 'Evening', 3: 'Night', 4: 'Late Night'}
time_stats.index = time_stats.index.map(time_category_names)
time_stats.columns = ['Total', 'Late', 'Late Percentage']
print("\nTime category-wise statistics:")
print(time_stats)

# Prediction function
def predict_submission(timestamp_str, section):
    """
    Predict if a submission will be late
    """
    timestamp = pd.to_datetime(timestamp_str)
    deadline = pd.Timestamp("2026-01-18 18:00:00")
    
    features = {
        'hour': timestamp.hour,
        'weekday': timestamp.weekday(),
        'is_weekend': int(timestamp.weekday() >= 5),
        'time_category': get_time_category(timestamp.hour),
        'section_encoded': 0 if section == 'D1' else 1,
        'hours_after_deadline': max((timestamp - deadline).total_seconds() / 3600, 0)
    }
    
    feature_df = pd.DataFrame([features])[feature_columns]
    feature_scaled = scaler.transform(feature_df)
    
    prediction = model.predict(feature_scaled)[0]
    probability = model.predict_proba(feature_scaled)[0][1]
    
    return prediction, probability

# Test predictions
print("\n" + "="*50)
print("SAMPLE PREDICTIONS")
print("="*50)

test_cases = [
    ("18/01/2026 14:00:00", "D2"),  # Afternoon before deadline
    ("18/01/2026 19:00:00", "D2"),  # Evening after deadline
    ("19/01/2026 10:00:00", "D1"),  # Next day morning
]

for time_str, section in test_cases:
    pred, prob = predict_submission(time_str, section)
    status = "LATE" if pred == 1 else "ON TIME"
    print(f"Submission at {time_str} ({section}): {status} (confidence: {max(prob,1-prob):.1%})")