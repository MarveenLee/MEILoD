"""
MEILoD多模态数据处理Pipeline
适配：EMG + IMU → 活动分类（walking/jogging/stair_ascent/stair_descent）
输出：准确率 + 混淆矩阵 + 特征可视化
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 数据加载 ====================
def load_data(file_path):
    """加载MEILoD数据，自动识别列名"""
    df = pd.read_csv(file_path)
    
    # 提取EMG列（前8列）
    emg_cols = [col for col in df.columns if 'Rectus' in col or 'Vastus' in col or 'Semitendinosus' in col]
    emg_cols = emg_cols[:8]  # 取前8个EMG通道
    
    # 提取IMU列（加速度+陀螺仪，6个传感器 × 6轴 = 36列）
    acc_cols = [col for col in df.columns if 'ACC' in col][:18]  # 前18个ACC列（6传感器×3轴）
    gyro_cols = [col for col in df.columns if 'GYRO' in col][:18]  # 前18个GYRO列
    imu_cols = acc_cols + gyro_cols
    
    # 提取标签列
    label_col = 'Activity'
    
    print(f"✓ 加载完成: {len(df)} 行")
    print(f"  - EMG通道: {len(emg_cols)}")
    print(f"  - IMU通道: {len(imu_cols)}")
    print(f"  - 标签分布:\n{df[label_col].value_counts()}")
    
    return df, emg_cols, imu_cols, label_col

# ==================== 2. 数据清洗与预处理 ====================
def preprocess_data(df, emg_cols, imu_cols):
    """数据清洗：去噪、滤波、归一化"""
    df_clean = df.copy()
    
    # 2.1 去除异常值（3σ原则）
    for col in emg_cols + imu_cols:
        mean, std = df_clean[col].mean(), df_clean[col].std()
        df_clean[col] = df_clean[col].clip(mean - 3*std, mean + 3*std)
    
    # 2.2 EMG信号整流（取绝对值）
    for col in emg_cols:
        df_clean[col] = np.abs(df_clean[col])
    
    # 2.3 低通滤波（EMG：10Hz，IMU：5Hz）
    fs = 200  # 假设采样率200Hz（实际需查数据集文档）
    for col in emg_cols:
        b, a = signal.butter(4, 10/(fs/2), btype='low')
        df_clean[col] = signal.filtfilt(b, a, df_clean[col])
    
    for col in imu_cols:
        b, a = signal.butter(4, 5/(fs/2), btype='low')
        df_clean[col] = signal.filtfilt(b, a, df_clean[col])
    
    # 2.4 归一化（Z-score）
    scaler = StandardScaler()
    df_clean[emg_cols + imu_cols] = scaler.fit_transform(df_clean[emg_cols + imu_cols])
    
    print("✓ 数据预处理完成：去噪 → 整流 → 滤波 → 归一化")
    return df_clean

# ==================== 3. 滑动窗口特征提取 ====================
def extract_window_features(df, emg_cols, imu_cols, window_size=200, step=100):
    """
    滑动窗口提取时域/频域特征
    window_size: 窗口大小（1秒 @200Hz）
    step: 步长（0.5秒）
    """
    features = []
    labels = []
    
    total_windows = (len(df) - window_size) // step
    print(f"提取特征中... 预计 {total_windows} 个窗口")
    
    for start in tqdm(range(0, len(df) - window_size, step)):
        end = start + window_size
        window = df.iloc[start:end]
        
        feat_vec = []
        
        # 3.1 对每个EMG通道提取特征
        for col in emg_cols:
            data = window[col].values
            feat_vec.extend([
                np.mean(data),          # 均值
                np.std(data),           # 标准差
                np.max(data),           # 最大值
                np.min(data),           # 最小值
                np.ptp(data),           # 峰峰值
                np.sqrt(np.mean(data**2)),  # RMS
                np.sum(data**2),        # 能量
                kurtosis(data),         # 峰度
                skew(data)              # 偏度
            ])
        
        # 3.2 对每个IMU通道提取特征
        for col in imu_cols[:6]:  # 取前6个IMU通道避免维度爆炸
            data = window[col].values
            feat_vec.extend([
                np.mean(data),
                np.std(data),
                np.max(data),
                np.min(data),
                np.sqrt(np.mean(data**2))
            ])
        
        # 3.3 频域特征（FFT主频能量）
        for col in emg_cols[:2] + imu_cols[:2]:  # 采样部分通道
            data = window[col].values
            fft_vals = np.abs(np.fft.fft(data))[:len(data)//2]
            if len(fft_vals) > 0:
                feat_vec.append(np.sum(fft_vals**2))  # 频域能量
            else:
                feat_vec.append(0)
        
        features.append(feat_vec)
        labels.append(window['Activity'].mode()[0])  # 窗口内多数标签
    
    print(f"✓ 特征提取完成: {len(features)} 个样本, {len(feat_vec)} 个特征")
    return np.array(features), np.array(labels)

# ==================== 4. 模型训练与评估 ====================
def train_and_evaluate(X, y):
    """训练随机森林分类器并输出准确率"""
    
    # 4.1 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 将整数标签转为字符串（修复TypeError）
    target_names = [str(label) for label in le.classes_]
    # 可选：给标签起有意义的名字
    label_names = {
        '0': 'Walking',
        '1': 'Jogging', 
        '2': 'Stair_Ascent',
        '3': 'Stair_Descent'
    }
    target_names_str = [label_names.get(name, name) for name in target_names]
    
    print(f"标签映射: {dict(zip(target_names_str, range(len(le.classes_))))}")
    
    # 4.2 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 4.3 训练模型
    print("\n训练RandomForest中...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # 4.4 预测与评估
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"🎯 模型准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*50}\n")
    
    # 详细分类报告（使用字符串标签名）
    print("分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names_str))
    
    # 交叉验证
    cv_scores = cross_val_score(clf, X, y_encoded, cv=5)
    print(f"5折交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names_str, yticklabels=target_names_str)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    
    # 特征重要性（可选）
    feature_importance = clf.feature_importances_
    print(f"\n特征重要性范围: [{feature_importance.min():.4f}, {feature_importance.max():.4f}]")
    
    return clf, accuracy

# ==================== 5. 可视化分析 ====================
def visualize_data(df, emg_cols, imu_cols):
    """数据可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # 5.1 EMG原始信号
    ax = axes[0, 0]
    for col in emg_cols[:4]:
        ax.plot(df[col][:1000], label=col.split('_')[0])
    ax.set_title('EMG Signals (first 1000 samples)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude (normalized)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 5.2 IMU加速度
    ax = axes[0, 1]
    acc_cols = [c for c in imu_cols if 'ACC' in c][:3]
    for col in acc_cols:
        ax.plot(df[col][:1000], label=col)
    ax.set_title('IMU Acceleration Signals')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Acceleration (G)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 5.3 标签分布
    ax = axes[1, 0]
    label_counts = df['Activity'].value_counts()
    ax.bar(label_counts.index, label_counts.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_title('Activity Label Distribution')
    ax.set_xlabel('Activity (0=Walk,1=Jog,2=StairUp,3=StairDown)')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    
    # 5.4 EMG功率谱
    ax = axes[1, 1]
    for col in emg_cols[:2]:
        f, Pxx = signal.periodogram(df[col].values, fs=200)
        ax.semilogy(f, Pxx, label=col.split('_')[0])
    ax.set_title('EMG Power Spectral Density')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power/Frequency (dB/Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_visualization.png', dpi=150)
    plt.show()
    print("✓ 可视化图表已保存: data_visualization.png")

# ==================== 主流程 ====================
def main():
    print("="*60)
    print("MEILoD多模态数据处理Pipeline")
    print("EMG + IMU → 活动分类（外骨骼应用场景）")
    print("="*60)
    
    # Step 1: 加载数据
    # file_path = input("请输入MEILoD CSV文件路径: ").strip()
    file_path = "GAN.csv"
    df, emg_cols, imu_cols, label_col = load_data(file_path)
    
    # Step 2: 数据清洗
    df_clean = preprocess_data(df, emg_cols, imu_cols)
    
    # Step 3: 可视化
    visualize_data(df_clean, emg_cols, imu_cols)
    
    # Step 4: 特征提取
    X, y = extract_window_features(df_clean, emg_cols, imu_cols, 
                                   window_size=200, step=100)
    
    # Step 5: 模型训练
    model, accuracy = train_and_evaluate(X, y)
    
    print("\n" + "="*60)
    print(f"✅ Pipeline完成！最终准确率: {accuracy:.2%}")
    print("="*60)
    print("\n输出文件:")
    print("  - confusion_matrix.png (混淆矩阵)")
    print("  - data_visualization.png (数据可视化)")

if __name__ == "__main__":
    main()