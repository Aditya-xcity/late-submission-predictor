# 📊 Student Submission Behavior Analyzer (Late Submission Predictor)

## 🚀 Overview

This project uses Machine Learning to analyze and predict whether a student will submit an assignment **on time or late** based on submission patterns.

It is built using real-world student submission data and demonstrates a complete ML pipeline — from data preprocessing to model evaluation.

---

## 🎯 Objective

To predict:

> Will a student submit an assignment late or on time?

---

## 🗂️ Dataset Description

The dataset contains:

* Timestamp of submission
* Student name
* Section (D1 / D2)
* Roll numbers
* Assignment submission links

---

## ⚙️ Features Used

After preprocessing, the following features are extracted:

* `hour` → Submission hour
* `weekday` → Day of the week
* `is_weekend` → Weekend indicator
* `time_category` → Morning / Afternoon / Evening / Night
* `section_encoded` → Section converted to numeric

---

## 🧠 Model Used

* **Random Forest Classifier**

Why Random Forest?

* Handles non-linear patterns well
* Works with small datasets
* Provides feature importance

---

## 🧪 ML Pipeline

1. Load dataset
2. Clean data (remove duplicates)
3. Convert timestamps
4. Feature engineering
5. Encode categorical data
6. Create target variable (`late`)
7. Train-test split
8. Model training
9. Evaluation

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Cross-validation

---

## 🔍 Key Insights

* Most submissions occur during **evening and night**
* Majority of students submit **late**
* Submission timing strongly influences lateness
* Section-wise behavior differences observed

---

## ⚠️ Challenges & Learnings

* **Class imbalance** (more late submissions than on-time)
* **Data leakage avoided** by removing post-deadline features
* Small dataset limits generalization

---

## 📈 Sample Output

* Predicts submission status (Late / On Time)
* Provides confidence score
* Generates statistical analysis of submission patterns

---

## 🛠️ Tech Stack

* Python
* Pandas
* Scikit-learn
* NumPy

---

## 📁 Project Structure

```
├── lateSubmissionModel.py
├── dataset.csv
├── README.md
```

---

## 💡 Future Improvements

* Add more data for better accuracy
* Deploy as a web app
* Add visualization dashboard
* Track individual student behavior over time

---

## 🏁 Conclusion

This project demonstrates how basic machine learning can be applied to real-world student data to extract insights and build predictive models.

---

## 👨‍💻 Author

**Aditya Bhardwaj**
B.Tech CSE

---
