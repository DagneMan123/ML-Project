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
    # ID, Temp_C, Vib_mm_s, Speed_RPM, Pressure_bar, Op_Hours, Pump_Condition
    [1,  48, 1.2, 1480, 10.5,  900, 0],
    [2,  51, 1.5, 1490, 11.0, 1200, 0],
    [3,  55, 1.8, 1475, 10.8, 1500, 0],
    [4,  58, 2.0, 1500, 11.2, 1800, 0],
    [5,  60, 2.2, 1460, 11.5, 2100, 0],
    [6,  53, 1.4, 1510, 10.7, 1100, 0],
    [7,  62, 2.4, 1450, 12.0, 2300, 0],
    [8,  57, 1.7, 1495, 11.1, 1600, 0],
    [9,  64, 2.6, 1440, 12.3, 2500, 0],
    [10, 50, 1.3, 1505, 10.6, 1000, 0],
    [11, 72, 3.8, 1420, 14.5, 3200, 1],
    [12, 74, 4.1, 1410, 15.0, 3500, 1],
    [13, 70, 3.5, 1430, 14.0, 3000, 1],
    [14, 76, 4.4, 1400, 15.5, 3800, 1],
    [15, 73, 3.9, 1415, 14.8, 3300, 1],
    [16, 75, 4.2, 1405, 15.2, 3600, 1],
    [17, 71, 3.6, 1425, 14.2, 3100, 1],
    [18, 77, 4.6, 1395, 15.8, 4000, 1],
    [19, 72, 3.7, 1418, 14.6, 3250, 1],
    [20, 76, 4.3, 1398, 15.4, 3900, 1],
    [21, 85, 6.1, 1360, 19.5, 5700, 2],
    [22, 99, 8.4, 1260, 24.0, 8100, 2],
    [23, 82, 5.8, 1370, 18.8, 5200, 2],
    [24, 90, 7.0, 1310, 21.5, 7000, 2],
    [25, 87, 6.7, 1330, 20.2, 6400, 2],
    [26, 102, 8.8, 1240, 25.5, 8600, 2],
    [27, 80, 5.5, 1380, 18.0, 5000, 2],
    [28, 93, 7.5, 1280, 22.0, 7600, 2],
    [29, 84, 6.0, 1350, 19.0, 5900, 2],
    [30, 97, 8.1, 1270, 23.5, 7900, 2],
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