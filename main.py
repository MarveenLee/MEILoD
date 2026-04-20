import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("GAN.csv")

y_raw = df["Activity"]

emg_cols = df.columns[:8]
imu_cols = df.columns[8:-1]

df = df.dropna()

df[emg_cols] = df[emg_cols].clip(-5, 5)
df[imu_cols] = df[imu_cols].clip(-20, 20)

scaler = StandardScaler()
df[emg_cols.tolist() + imu_cols.tolist()] = scaler.fit_transform(
    df[emg_cols.tolist() + imu_cols.tolist()]
)

window_size = 100
stride = 50

def extract_features(window):
    feats = []
    feats.append(window.mean().values)
    feats.append(window.std().values)
    feats.append((window[emg_cols]**2).mean().values)
    return np.concatenate(feats)

X, y = [], []

for i in range(0, len(df) - window_size, stride):
    window = df.iloc[i:i+window_size]
    feat = extract_features(window[emg_cols.tolist() + imu_cols.tolist()])
    X.append(feat)
    y.append(window["Activity"].mode()[0])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

plt.figure(figsize=(10,4))
plt.plot(df[emg_cols[0]].values[:1000], label="EMG")
plt.plot(df[imu_cols[0]].values[:1000], label="IMU")
plt.legend()
plt.title("EMG vs IMU")
plt.show()