#============================================
# Name

# Dange Aydenfu      id:0545/16
#  Hubeyb Awol       id:1067/16
#============================================
"""
Practical Engineering Case Study: Pump Condition Classification (Multiclass)
Algorithms:
1. Support Vector Machines (SVM)
2. Naïve Bayes (Gaussian Naïve Bayes)
3. Decision Tree
4. Random Forest
5. AdaBoost
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# ============================================================
# PART 1. CREATE DATASET (30 RECORDS - 5 FEATURES, 3 CLASSES)
# Class Mapping: 0 = Normal, 1 = Warning, 2 = Fault
# ============================================================

data = [

    # ID, Temperature_C, Vibration_mm_s, Speed_RPM, Operating_Hours, Failure_Risk

    [1,  46, 1.1, 1490,  850, 0],
    [2,  49, 1.4, 1500, 1100, 0],
    [3,  54, 1.7, 1480, 1450, 0],
    [4,  57, 1.9, 1510, 1750, 0],
    [5,  59, 2.1, 1470, 2050, 0],
    [6,  52, 1.3, 1520, 1050, 0],
    [7,  61, 2.3, 1460, 2250, 0],
    [8,  56, 1.6, 1500, 1550, 0],
    [9,  63, 2.5, 1450, 2450, 0],
    [10, 51, 1.2, 1515,  950, 0],
    [11, 65, 2.7, 1440, 2650, 0],
    [12, 58, 2.0, 1490, 1850, 0],
    [13, 62, 2.4, 1470, 2150, 0],
    [14, 55, 1.8, 1505, 1350, 0],
    [15, 60, 2.2, 1480, 1950, 0],

    [16, 79, 5.5, 1380, 4900, 1],
    [17, 86, 6.2, 1350, 5800, 1],
    [18, 89, 6.8, 1330, 6500, 1],
    [19, 75, 4.8, 1400, 4600, 1],
    [20, 93, 7.5, 1300, 7200, 1],
    [21, 83, 5.9, 1370, 5500, 1],
    [22, 97, 8.1, 1270, 7900, 1],
    [23, 74, 4.6, 1410, 4300, 1],
    [24, 88, 6.7, 1320, 6800, 1],
    [25, 85, 6.4, 1340, 6200, 1],
    [26, 100, 8.5, 1250, 8400, 1],
    [27, 77, 5.2, 1390, 4800, 1],
    [28, 91, 7.2, 1290, 7400, 1],
    [29, 82, 5.8, 1360, 5700, 1],
    [30, 95, 7.8, 1280, 7700, 1]

]

columns = [
    "Machine_ID", "Temperature_C", "Vibration_mm_s",
    "Speed_RPM", "Pressure_bar", "Operating_Hours", "Pump_Condition"
]
df = pd.DataFrame(data, columns=columns)

print("\nDATASET: 30 PUMP RECORDS")
print(df.to_string(index=False))
print("\nClass distribution:")
print(df["Pump_Condition"].value_counts().rename(index={0: "Normal", 1: "Warning", 2: "Fault"}))

# Separate predictors (X) and target (y).
feature_cols = ["Temperature_C", "Vibration_mm_s", "Speed_RPM", "Pressure_bar", "Operating_Hours"]
X = df[feature_cols]
y = df["Pump_Condition"]

# Initial data visualization.
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="Temperature_C", y="Vibration_mm_s",
    hue="Pump_Condition", palette="viridis", s=100
)
plt.title("Machine temperature vs. vibration (Normal=0, Warning=1, Fault=2)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Vibration (mm/s)")
plt.legend(title="Pump Condition")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df[feature_cols + ["Pump_Condition"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix of engineering variables")
plt.tight_layout()
plt.show()

# ============================================================
# PART 2. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")

# ============================================================
# PART 3. DEFINE FIVE INDEPENDENT CLASSIFIERS
# ============================================================

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])

decision_tree_model = DecisionTreeClassifier(
    max_depth=4, random_state=42
)

random_forest_model = RandomForestClassifier(
    n_estimators=100, max_depth=4, random_state=42
)

adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

models = {
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model
}

# ============================================================
# PART 4. TRAIN, PREDICT, EVALUATE, AND VISUALIZE EACH MODEL
# ============================================================

results = []
predictions = {}
class_names_list = ["Normal", "Warning", "Fault"]

for model_name, model in models.items():
    print("\n" + "=" * 65)
    print(f"{model_name.upper()}")
    print("=" * 65)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1, 2],
        target_names=class_names_list,
        zero_division=0
    ))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=class_names_list,
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.set_title(f"{model_name}: Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ============================================================
# PART 5. COMPARE THE FIVE ALGORITHMS
# ============================================================

results_df = pd.DataFrame(results).set_index("Algorithm")
print("\nMODEL COMPARISON")
print(results_df.round(3))

results_df.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.05))
plt.title("Comparison of five supervised learning algorithms")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ============================================================
# PART 6. INTERPRET TREE-BASED FEATURE IMPORTANCE
# ============================================================

tree_models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

for name, model in tree_models.items():
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    print(f"\n{name} feature importance:")
    print(importance.round(3))

    plt.figure(figsize=(7, 4))
    plt.barh(importance.index, importance.values)
    plt.title(f"{name}: Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Engineering feature")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(16, 8))
plot_tree(
    decision_tree_model,
    feature_names=feature_cols,
    class_names=class_names_list,
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree learned from the training records")
plt.tight_layout()
plt.show()

# ============================================================
# PART 7. TEST A NEW ENGINEERING OBSERVATION
# ============================================================

new_machine = pd.DataFrame([{
    "Temperature_C": 75,
    "Vibration_mm_s": 4.0,
    "Speed_RPM": 1410,
    "Pressure_bar": 14.8,
    "Operating_Hours": 3400
}])

print("\nNEW MACHINE OBSERVATION")
print(new_machine)

for name, model in models.items():
    predicted_class = int(model.predict(new_machine)[0])
    label = class_names_list[predicted_class]
    print(f"{name:15s} prediction: {label}")

print("\nEND OF PRACTICAL CASE STUDY")