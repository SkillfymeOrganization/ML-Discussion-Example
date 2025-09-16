import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# Read log file
log_file_path = "system_logs.txt"  # Update with your file path if needed
with open(log_file_path, "r") as file:
    logs = file.readlines()

# Preporcessing - Parse logs into a structured DataFrame
data = []
for log in logs:
    parts = log.strip().split(" ", 3)  # Ensure the message part is captured fully
    if len(parts) < 4:
        continue  # Skip malformed lines
    timestamp = parts[0] + " " + parts[1]
    level = parts[2]
    message = parts[3]
    data.append([timestamp, level, message])

df = pd.DataFrame(data, columns=["timestamp", "level", "message"])

# Convert timestamp to datetime format for sorting
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Assign numeric scores to log levels
level_mapping = {"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
df["level_score"] = df["level"].map(level_mapping)

# Add message length as a new feature
df["message_length"] = df["message"].apply(len)
print(df.columns)
print(df.head)
# Split into train and test sets
X = df[["level_score", "message_length"]]
X_train, X_test, df_train, df_test = train_test_split(
    X, df, test_size=0.3, random_state=42
)

# AI Model for Anomaly Detection (Isolation Forest)
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)

# Predict anomalies on test set
df_test["anomaly"] = model.predict(X_test)

# Mark anomalies in a readable format
df_test["is_anomaly"] = df_test["anomaly"].apply(lambda x: "❌ Anomaly" if x == -1 else "✅ Normal")

# Print only detected anomalies in test set
anomalies = df_test[df_test["is_anomaly"] == "❌ Anomaly"]
# print("\n🔍 **Detected Anomalies in Test Set:**\n", anomalies)

# # AI Model for Anomaly Detection (Isolation Forest)
# model = IsolationForest(contamination=0.1, random_state=42)  # Lower contamination for better accuracy
# df["anomaly"] = model.fit_predict(df[["level_score", "message_length"]])

# # Mark anomalies in a readable format
# df["is_anomaly"] = df["anomaly"].apply(lambda x: "❌ Anomaly" if x == -1 else "✅ Normal")

# # Print only detected anomalies
# anomalies = df[df["is_anomaly"] == "❌ Anomaly"]
print("\n🔍 **Detected Anomalies:**\n", df_test.shape, anomalies.shape)


