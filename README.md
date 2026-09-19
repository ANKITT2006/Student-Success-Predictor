# 🎓 Student Success Predictor (AI/ML Project)

## 📌 Overview

The **Student Success Predictor** is a machine learning system designed to analyze student academic and behavioral indicators, benchmark multiple ML models, and predict academic outcomes (Pass / Fail) with confidence probabilities and personalized actionable recommendations.

---

## 🚀 Key Features

* 🧠 **Multi-Model Benchmarking**: Trains and evaluates **5 machine learning algorithms**:
  - Logistic Regression
  - Random Forest Classifier
  - Decision Tree Classifier
  - Support Vector Machine (SVC with Probability Calibration)
  - Gradient Boosting Classifier
* 📊 **Comprehensive Metrics**: Evaluates models on **Accuracy, Precision, Recall, F1-Score**, and generates **Confusion Matrices**.
* ⚡ **Intelligent Auto-Selection**: Automatically picks the best-performing model based on F1-Score while allowing users to switch models dynamically.
* 🔮 **What-If Improvement Simulator**: Simulates how incremental improvements in attendance (+10%), study hours (+1-2h), or assignment marks directly raise success probability.
* 📈 **Auto-Generated Visualizations**:
  - `model_comparison.png`: Side-by-side grouped bar chart comparing Accuracy and F1-score across all 5 models.
  - `feature_importance.png`: Feature impact rankings from Random Forest.
  - `confusion_matrix.png`: Heatmap breakdown of classification performance.
* 📁 **Batch CSV Prediction**: Generates sample CSV files and processes bulk student records to export enriched prediction files (`predictions_output.csv`).
* 💡 **Personalized AI Action Plans**: Context-aware risk classification (Low, Moderate, Elevated, Critical) and targeted advice tailored to student strengths and weaknesses.

---

## 🛠️ Tech Stack

* **Programming Language:** Python 3
* **Libraries:**
  * NumPy
  * Pandas
  * Scikit-Learn
  * Matplotlib
  * Colorama

---

## 💻 How to Run

### 1. Interactive CLI (Recommended)
Launch the interactive menu to run single-student predictions, what-if simulations, model switching, and chart generation:
```bash
python student_success_predictor.py
```

### 2. Automated Benchmark & Verification
Run the benchmark directly without interactive prompts:
```bash
python student_success_predictor.py --benchmark
```

### 3. Generate Visual Charts Only
```bash
python student_success_predictor.py --visualize
```

### 4. Full Verification Suite
```bash
python student_success_predictor.py --test
```

---

## 📈 Model Performance Benchmark

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **Random Forest** | 98.67% | 100.00% | 98.51% | 99.25% |
| **Decision Tree** | 96.00% | 98.48% | 97.01% | 97.74% |
| **Support Vector Machine** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **Gradient Boosting** | 98.67% | 100.00% | 98.51% | 99.25% |

---

## 👤 Author

ANKIT NAG

* GitHub: [ANKIT04042006](https://github.com/ANKIT04042006)
* LinkedIn: [Ankit Nag](https://www.linkedin.com/in/ankit-nag-6b580637a)
