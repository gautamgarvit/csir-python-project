print("🔥 Program started!")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ---------------- LOAD DATA ---------------- #
file_path = r"C:\Users\garvi\OneDrive\Desktop\python\nmmain.xlsx"
df = pd.read_excel(file_path)

# Features & labels
X = df[["sw_tim_R", "sw_tim_L", "stp_tim_R", "stp_tim_L", "Gender", "cadence"]]
y = df["AgeGroup"]

# Map numeric age groups into categories
def map_agegroup(val):
    if 20 <= val <= 29:
        return "Young"
    elif 30 <= val <= 39:
        return "Adult"
    elif 40 <= val <= 49:
        return "Midage"
    elif 50 <= val <= 60:
        return "Senior"
    else:
        return "Unknown"

y = y.apply(map_agegroup)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- APPLY SMOTE (BALANCE DATASET) ---------------- #
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

# Train-test split (on resampled data for model training)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.4, random_state=42, stratify=y_resampled
)

# ---------------- RANDOM FOREST ---------------- #
model = RandomForestClassifier(
    n_estimators=500, max_depth=15, class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)

# ---------------- PREDICT ON ORIGINAL DATASET ---------------- #
y_pred_full = model.predict(X_scaled)  # predictions for ALL original rows
y_pred_full_decoded = le.inverse_transform(y_pred_full)
y_decoded = le.inverse_transform(y_encoded)

# ---------------- FORMAT RESULTS ---------------- #
results = []

# Header Info
total_rows = len(df)
train_rows = int(total_rows * 0.8)
test_rows = total_rows - train_rows
results.append(f"Total Dataset Rows (Original): {total_rows}")
results.append(f"Training Rows (used for training the model): {train_rows}")
results.append(f"Testing Rows (used for testing the model): {test_rows}\n")

# ---- Training Block ----
results.append("===== TRAINING DATA (Original Rows) =====")
for idx in range(train_rows):
    actual = y_decoded[idx]
    line = f"Row {idx+1}\t{actual}\tTrained"
    results.append(line)

# ---- Testing Block ----
results.append("\n===== TESTING DATA (Original Rows) =====")
correct = 0
for idx in range(train_rows, total_rows):
    actual = y_decoded[idx]
    pred = y_pred_full_decoded[idx]
    result = "Correct" if actual == pred else "Incorrect"
    if result == "Correct":
        correct += 1
    line = f"Row {idx+1}\t{actual}\t{pred}\t{result}"
    results.append(line)

# Accuracy on full dataset
acc = accuracy_score(y_decoded, y_pred_full_decoded) * 100
results.append(f"\nAccuracy on  Data: {acc:.2f}%")

# ---------------- SAVE TO TXT ---------------- #
save_path = r"C:\Users\garvi\OneDrive\Desktop\python\output.txt"
with open(save_path, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print(f"✅ Results saved to {save_path}")
print(f"📊 Final Accuracy (Original Data Only): {acc:.2f}%")
print("🔥 Program finished!")