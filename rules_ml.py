import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# === Step 1: Read and Parse Logs ===
log_file = "system_logs.txt"
log_entries = []

with open(log_file, "r") as file:
    for line in file:
        match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) (.+)", line.strip())
        if match:
            timestamp, level, message = match.groups()
            log_entries.append([timestamp, level, message])

# Convert to DataFrame
df = pd.DataFrame(log_entries, columns=["timestamp", "level", "message"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

# === Step 2: Rule-based Anomaly Detection (Error Spikes) ===
error_counts = Counter(df[df["level"] == "ERROR"]["timestamp"].dt.floor("30s"))
threshold = 3  # Customize threshold for spikes

print("\n🚨 Rule-based Anomaly Detection:")
for time, count in error_counts.items():
    if count > threshold:
        print(f"⚠️ Spike detected! {count} ERROR logs in 30 seconds at {time}")

# === Step 3: Feature Engineering for AI Model ===
level_mapping = {"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
df["level_score"] = df["level"].map(level_mapping)
df["message_length"] = df["message"].apply(len)

# Features for ML model
X = df[["level_score", "message_length"]]

# Train/Test split
X_train, X_test, df_train, df_test = train_test_split(X, df, test_size=0.3, random_state=42)

# === Step 4: AI-based Anomaly Detection (Isolation Forest) ===
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)

df_test["anomaly"] = model.predict(X_test)
df_test["is_anomaly"] = df_test["anomaly"].apply(lambda x: "❌ Anomaly" if x == -1 else "✅ Normal")

anomalies = df_test[df_test["is_anomaly"] == "❌ Anomaly"]

print("\n🤖 AI-based Anomaly Detection (Isolation Forest):")
print(f"Total anomalies detected: {anomalies.shape[0]}")
print(anomalies[["timestamp", "level", "message", "is_anomaly"]])

# === Step 5: Show Final Logs with Flags ===
print("\n📊 Full Log DataFrame (sample with anomaly labels where available):")
df_final = pd.concat([df_train, df_test])  # merge back train + test with anomaly labels
print(df_final.head(20))
