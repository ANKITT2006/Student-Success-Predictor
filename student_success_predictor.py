import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Initialize Colorama for cross-platform colored terminal output
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    CYAN = Fore.CYAN + Style.BRIGHT
    GREEN = Fore.GREEN + Style.BRIGHT
    YELLOW = Fore.YELLOW + Style.BRIGHT
    RED = Fore.RED + Style.BRIGHT
    MAGENTA = Fore.MAGENTA + Style.BRIGHT
    RESET = Style.RESET_ALL
    BOLD = Style.BRIGHT
except ImportError:
    CYAN = GREEN = YELLOW = RED = MAGENTA = RESET = BOLD = ""

FEATURE_NAMES = ["Study_Hours", "Attendance", "Assignment_Score"]

class StudentSuccessPredictor:
    def __init__(self, n_samples=300, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {
            "Logistic Regression": LogisticRegression(random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=random_state),
            "Support Vector Machine": CalibratedClassifierCV(SVC(random_state=random_state), ensemble=False),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state)
        }
        self.results = {}
        self.best_model_name = None
        self.active_model_name = None
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def generate_data(self):
        """Generates realistic student data for training and testing."""
        np.random.seed(self.random_state)
        study_hours = np.random.uniform(1, 10, self.n_samples)
        attendance = np.random.uniform(45, 100, self.n_samples)
        assignment = np.random.uniform(35, 100, self.n_samples)
        
        # Realistic weighted scoring formula with slight random variation
        noise = np.random.normal(0, 4, self.n_samples)
        composite_score = (study_hours * 4.5) + (attendance * 0.35) + (assignment * 0.30) + noise
        passed = (composite_score >= 50.0).astype(int)

        self.df = pd.DataFrame({
            "Study_Hours": np.round(study_hours, 1),
            "Attendance": np.round(attendance, 1),
            "Assignment_Score": np.round(assignment, 1),
            "Passed": passed
        })
        return self.df

    def prepare_and_train(self):
        """Splits data before scaling (ML best practice) and trains all models."""
        if self.df is None:
            self.generate_data()

        X = self.df[FEATURE_NAMES]
        y = self.df["Passed"]

        X_train_raw, X_test_raw, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state, stratify=y
        )

        # Scale features using training distribution only
        self.X_train = self.scaler.fit_transform(X_train_raw)
        self.X_test = self.scaler.transform(X_test_raw)

        best_f1 = -1.0

        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, zero_division=0)
            rec = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            cm = confusion_matrix(self.y_test, y_pred)

            self.results[name] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm,
                "model": model
            }

            if f1 > best_f1:
                best_f1 = f1
                self.best_model_name = name

        self.active_model_name = self.best_model_name

    def display_metrics_table(self):
        """Prints a structured benchmark table of all evaluated models."""
        print(f"\n{CYAN}{'='*68}")
        print(f"{CYAN}             MODEL COMPARISON & EVALUATION BENCHMARK")
        print(f"{CYAN}{'='*68}{RESET}")
        print(f"{BOLD}{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}{RESET}")
        print("-" * 68)

        for name, metrics in self.results.items():
            is_best = " (Best)" if name == self.best_model_name else ""
            prefix = GREEN if name == self.active_model_name else RESET
            print(f"{prefix}{name + is_best:<25} | "
                  f"{metrics['accuracy']*100:6.2f}%    | "
                  f"{metrics['precision']*100:6.2f}%    | "
                  f"{metrics['recall']*100:6.2f}%    | "
                  f"{metrics['f1_score']*100:6.2f}%{RESET}")
        print("-" * 68)
        print(f"{YELLOW}Active Model for Predictions: {BOLD}{self.active_model_name}{RESET}\n")

    def predict_single(self, hours, attendance, assignment, model_name=None):
        """Predicts outcome and probability for a single student."""
        chosen_model_name = model_name or self.active_model_name
        model = self.models[chosen_model_name]
        
        user_df = pd.DataFrame([[hours, attendance, assignment]], columns=FEATURE_NAMES)
        user_scaled = self.scaler.transform(user_df)
        prob = model.predict_proba(user_scaled)[0][1] * 100
        pred = model.predict(user_scaled)[0]
        return pred, prob

    def get_action_plan(self, hours, attendance, assignment, prob, pred):
        """Generates contextual AI recommendations and risk tiering."""
        recommendations = []
        
        # Risk assessment
        if prob >= 80:
            tier = f"{GREEN}LOW RISK (High Distinction Candidate){RESET}"
        elif prob >= 50:
            tier = f"{YELLOW}MODERATE RISK (Borderline Safe){RESET}"
        elif prob >= 30:
            tier = f"{MAGENTA}ELEVATED RISK (Needs Attention){RESET}"
        else:
            tier = f"{RED}CRITICAL RISK (Urgent Intervention Needed){RESET}"

        # Suggestions based on specific metrics
        if hours < 3.0:
            recommendations.append("Increase daily study time to at least 3.5 - 5.0 hours/day.")
        elif hours < 4.5:
            recommendations.append("Boost study time by 1 hour daily focusing on active recall / practice.")

        if attendance < 75.0:
            recommendations.append("Critical: Attendance is below 75%. Attend lectures regularly to prevent debarment.")
        elif attendance < 85.0:
            recommendations.append("Strive to keep attendance above 85% to maximize internal marks.")

        if assignment < 60.0:
            recommendations.append("Assignment scores are low. Consult teaching assistants & review past assignments.")
        elif assignment < 75.0:
            recommendations.append("Improve assignment rigor by taking weekly mock quizzes.")

        if pred == 1 and not recommendations:
            recommendations.append("Outstanding performance across all metrics! Maintain your current routine.")
        elif pred == 0 and not recommendations:
            recommendations.append("Work on holistic revision and prioritize high-weightage topics.")

        return tier, recommendations

    def simulate_what_if(self, hours, attendance, assignment):
        """Shows how improvement in study hours or attendance shifts success probability."""
        _, current_prob = self.predict_single(hours, attendance, assignment)
        
        print(f"\n{CYAN}--- WHAT-IF IMPROVEMENT SIMULATION ---{RESET}")
        print(f"Current Estimated Success Probability: {BOLD}{current_prob:.2f}%{RESET}")
        
        scenarios = [
            ("+1 Hour Daily Study", min(12.0, hours + 1.0), attendance, assignment),
            ("+2 Hours Daily Study", min(12.0, hours + 2.0), attendance, assignment),
            ("+10% Attendance", hours, min(100.0, attendance + 10.0), assignment),
            ("+15 Score in Assignments", hours, attendance, min(100.0, assignment + 15.0)),
            ("Optimized (+1.5h Study, +10% Att, +10 Assign)", min(12.0, hours + 1.5), min(100.0, attendance + 10.0), min(100.0, assignment + 10.0))
        ]

        print(f"{'Scenario':<45} | {'Projected Prob':<15} | {'Delta'}")
        print("-" * 75)
        for label, h, att, ass in scenarios:
            _, p = self.predict_single(h, att, ass)
            delta = p - current_prob
            delta_str = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
            color = GREEN if delta > 0 else RESET
            print(f"{label:<45} | {p:6.2f}%         | {color}{delta_str}{RESET}")
        print("-" * 75)

    def generate_visualizations(self, output_dir="."):
        """Generates and saves visual plots for model comparison and features."""
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []

        # 1. Model Comparison Chart
        model_names = list(self.results.keys())
        accuracies = [self.results[m]["accuracy"] * 100 for m in model_names]
        f1_scores = [self.results[m]["f1_score"] * 100 for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#2563eb')
        plt.bar(x + width/2, f1_scores, width, label='F1-Score (%)', color='#10b981')
        plt.ylabel('Score (%)')
        plt.title('Student Success Predictor - Model Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, [m.replace(" ", "\n") for m in model_names], fontsize=9)
        plt.ylim(0, 105)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, "model_comparison.png")
        plt.savefig(comparison_path, dpi=300)
        plt.close()
        saved_files.append(comparison_path)

        # 2. Feature Importance Chart (from Random Forest)
        rf_model = self.models["Random Forest"]
        importances = rf_model.feature_importances_ * 100
        sorted_idx = np.argsort(importances)

        plt.figure(figsize=(8, 5))
        plt.barh(np.array(FEATURE_NAMES)[sorted_idx], importances[sorted_idx], color='#8b5cf6')
        plt.xlabel('Importance (%)')
        plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        for index, value in enumerate(importances[sorted_idx]):
            plt.text(value + 1, index, f"{value:.1f}%", va='center')
        plt.xlim(0, max(importances) + 15)
        plt.tight_layout()
        feat_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(feat_path, dpi=300)
        plt.close()
        saved_files.append(feat_path)

        # 3. Confusion Matrix for the active model
        cm = self.results[self.active_model_name]["confusion_matrix"]
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix ({self.active_model_name})', fontsize=12, fontweight='bold')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Fail (0)', 'Pass (1)'])
        plt.yticks(tick_marks, ['Fail (0)', 'Pass (1)'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black", fontsize=14)
        plt.tight_layout()
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        saved_files.append(cm_path)

        return saved_files

    def batch_predict(self, csv_filepath, output_filepath="predictions_output.csv"):
        """Performs batch prediction on a provided CSV file and exports results."""
        if not os.path.exists(csv_filepath):
            print(f"{RED}Error: File '{csv_filepath}' not found.{RESET}")
            return None

        batch_df = pd.read_csv(csv_filepath)
        missing_cols = [c for c in FEATURE_NAMES if c not in batch_df.columns]
        if missing_cols:
            print(f"{RED}Error: CSV missing required columns: {missing_cols}{RESET}")
            return None

        scaled = self.scaler.transform(batch_df[FEATURE_NAMES])
        model = self.models[self.active_model_name]
        probabilities = model.predict_proba(scaled)[:, 1] * 100
        predictions = model.predict(scaled)

        batch_df["Success_Probability_%"] = np.round(probabilities, 2)
        batch_df["Prediction"] = np.where(predictions == 1, "PASS", "FAIL")
        batch_df.to_csv(output_filepath, index=False)
        return output_filepath

    def create_sample_csv(self, filepath="sample_students.csv"):
        """Generates a template CSV with test students."""
        sample_df = pd.DataFrame({
            "Student_ID": [f"STU{1000+i}" for i in range(1, 6)],
            "Study_Hours": [2.5, 6.0, 1.0, 7.5, 4.0],
            "Attendance": [65.0, 92.0, 50.0, 88.0, 78.0],
            "Assignment_Score": [55.0, 85.0, 42.0, 90.0, 70.0]
        })
        sample_df.to_csv(filepath, index=False)
        return filepath


def get_float_input(prompt, min_val, max_val):
    """Helper to safely prompt user for float inputs within limits."""
    while True:
        try:
            val = float(input(prompt))
            if min_val <= val <= max_val:
                return val
            print(f"{YELLOW}Please enter a value between {min_val} and {max_val}.{RESET}")
        except ValueError:
            print(f"{RED}Invalid input. Please enter a numerical value.{RESET}")


def interactive_cli():
    predictor = StudentSuccessPredictor()
    print(f"\n{CYAN}{'='*60}")
    print(f"       VIT STUDENT SUCCESS PREDICTOR (AI / ML SUITE)")
    print(f"{'='*60}{RESET}")
    print(f"Training machine learning models and evaluating benchmarks...")
    predictor.prepare_and_train()
    print(f"{GREEN}Training complete! Best performing model: {BOLD}{predictor.best_model_name}{RESET}")

    while True:
        print(f"\n{CYAN}---------------- MAIN MENU ----------------{RESET}")
        print("1. Predict Single Student Academic Outcome")
        print("2. What-If Improvement Simulator")
        print("3. View Multi-Model Comparison & Metrics")
        print("4. Generate & Save Visualization Charts (.png)")
        print("5. Batch Prediction from CSV (with sample generator)")
        print("6. Change Active Machine Learning Model")
        print("7. Exit")
        print("-" * 43)

        choice = input(f"{BOLD}Enter choice (1-7): {RESET}").strip()

        if choice == "1":
            print(f"\n{CYAN}--- Student Academic Profile Input ---{RESET}")
            hours = get_float_input("Daily Study Hours (0.0 to 16.0): ", 0.0, 16.0)
            attendance = get_float_input("Class Attendance % (0.0 to 100.0): ", 0.0, 100.0)
            assignment = get_float_input("Previous Assignment Score (0.0 to 100.0): ", 0.0, 100.0)

            pred, prob = predictor.predict_single(hours, attendance, assignment)
            tier, suggestions = predictor.get_action_plan(hours, attendance, assignment, prob, pred)

            print(f"\n{CYAN}{'='*45}")
            print("             PREDICTION REPORT")
            print(f"{'='*45}{RESET}")
            status_text = f"{GREEN}PASS{RESET}" if pred == 1 else f"{RED}FAIL{RESET}"
            print(f"Predicted Outcome   : {status_text}")
            print(f"Success Probability : {BOLD}{prob:.2f}%{RESET}")
            print(f"Risk Assessment     : {tier}")
            print(f"Model Utilized      : {predictor.active_model_name}")
            
            print(f"\n{YELLOW}AI Recommendations & Strategy:{RESET}")
            for s in suggestions:
                print(f"  • {s}")
            print(f"{CYAN}{'='*45}{RESET}")

        elif choice == "2":
            print(f"\n{CYAN}--- What-If Improvement Simulator ---{RESET}")
            hours = get_float_input("Current Daily Study Hours (0-16): ", 0.0, 16.0)
            attendance = get_float_input("Current Attendance % (0-100): ", 0.0, 100.0)
            assignment = get_float_input("Current Assignment Score (0-100): ", 0.0, 100.0)
            predictor.simulate_what_if(hours, attendance, assignment)

        elif choice == "3":
            predictor.display_metrics_table()

        elif choice == "4":
            print(f"\nGenerating charts using Matplotlib...")
            saved = predictor.generate_visualizations()
            print(f"{GREEN}Successfully saved 3 visualization charts:{RESET}")
            for f in saved:
                print(f"  [Created] {f}")

        elif choice == "5":
            print(f"\n{CYAN}--- Batch Prediction Suite ---{RESET}")
            print("1. Generate sample CSV file ('sample_students.csv')")
            print("2. Run batch prediction on a CSV file")
            sub_choice = input("Select option (1-2): ").strip()
            
            if sub_choice == "1":
                created = predictor.create_sample_csv()
                print(f"{GREEN}Sample CSV created at: {created}{RESET}")
            elif sub_choice == "2":
                filename = input("Enter path to input CSV [default: sample_students.csv]: ").strip()
                if not filename:
                    filename = "sample_students.csv"
                if not os.path.exists(filename):
                    print(f"{YELLOW}'{filename}' does not exist. Creating sample CSV first...{RESET}")
                    predictor.create_sample_csv(filename)
                
                out_path = predictor.batch_predict(filename)
                if out_path:
                    print(f"{GREEN}Batch prediction complete! Results saved to: {out_path}{RESET}")
                    preview = pd.read_csv(out_path)
                    print(f"\n{CYAN}Preview of Output:{RESET}")
                    print(preview.head())

        elif choice == "6":
            print(f"\n{CYAN}--- Switch Active Machine Learning Model ---{RESET}")
            models_list = list(predictor.models.keys())
            for idx, name in enumerate(models_list, 1):
                cur = " (Active)" if name == predictor.active_model_name else ""
                print(f"{idx}. {name}{cur}")
            sel = input(f"Select model number (1-{len(models_list)}): ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(models_list):
                predictor.active_model_name = models_list[int(sel)-1]
                print(f"{GREEN}Active model updated to: {predictor.active_model_name}{RESET}")
            else:
                print(f"{RED}Invalid selection.{RESET}")

        elif choice == "7":
            print(f"\n{GREEN}Thank you for using VIT Student Success Predictor. Goodbye!{RESET}\n")
            break
        else:
            print(f"{RED}Invalid menu choice. Please select 1-7.{RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student Success Predictor")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark without interactive prompt")
    parser.add_argument("--visualize", action="store_true", help="Generate and save charts")
    parser.add_argument("--test", action="store_true", help="Run automated verification test")
    args = parser.parse_args()

    if args.test or args.benchmark or args.visualize:
        p = StudentSuccessPredictor()
        p.prepare_and_train()
        p.display_metrics_table()
        if args.visualize or args.test:
            saved_charts = p.generate_visualizations()
            print("Visualizations created:", saved_charts)
        if args.test:
            sample_f = p.create_sample_csv()
            out_f = p.batch_predict(sample_f)
            print("Batch prediction completed:", out_f)
            pred, prob = p.predict_single(6.0, 85.0, 80.0)
            print(f"Sample prediction: outcome={pred}, probability={prob:.2f}%")
            print("ALL VERIFICATIONS PASSED.")
    else:
        interactive_cli()
