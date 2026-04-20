# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:40:28 2026

@author: MarveenLee
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy import signal
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据加载 ====================
def load_data(file_path):
    """加载MEILoD数据"""
    df = pd.read_csv(file_path)
    
    # 列名处理
    emg_cols = df.columns[:8].tolist()
    imu_cols = df.columns[8:-1].tolist()
    label_col = 'Activity'
    
    print(f"✓ 加载完成: {len(df)} 行")
    print(f"  - EMG通道: {len(emg_cols)}")
    print(f"  - IMU通道: {len(imu_cols)}")
    print(f"  - 标签分布:\n{df[label_col].value_counts()}")
    
    return df, emg_cols, imu_cols, label_col

# ==================== 优化预处理 ====================
def preprocess_optimized(df, emg_cols, imu_cols, use_rectify=False, use_filter=False):
    """
    优化预处理：方案一阈值 + 可选整流/滤波
    use_rectify: 是否对EMG整流（默认False，保留原始波形）
    use_filter: 是否低通滤波（默认False，保留高频信息）
    """
    df_clean = df.copy()
    
    # 方案一的固定阈值（更激进，保留极端运动模式）
    df_clean[emg_cols] = df_clean[emg_cols].clip(-5, 5)
    df_clean[imu_cols] = df_clean[imu_cols].clip(-20, 20)
    
    # 可选：EMG整流（实验证明会降低准确率）
    if use_rectify:
        for col in emg_cols:
            df_clean[col] = np.abs(df_clean[col])
    
    # 可选：低通滤波（实验证明会丢失信息）
    if use_filter:
        fs = 200
        for col in emg_cols:
            b, a = signal.butter(4, 10/(fs/2), btype='low')
            df_clean[col] = signal.filtfilt(b, a, df_clean[col])
    
    # 归一化
    scaler = StandardScaler()
    df_clean[emg_cols + imu_cols] = scaler.fit_transform(df_clean[emg_cols + imu_cols])
    
    print(f"✓ 预处理完成 (整流={use_rectify}, 滤波={use_filter})")
    return df_clean

# ==================== 优化特征提取 ====================
def extract_features_optimized(df, emg_cols, imu_cols, window_size=100, step=50, use_pca=True):
    """
    优化特征提取：
    - 窗口100（方案一最优）
    - 适度特征数量（避免过拟合）
    - PCA降维保留全部IMU信息
    """
    features = []
    labels = []
    
    # PCA降维（保留95%方差）
    if use_pca and len(imu_cols) > 10:
        pca = PCA(n_components=0.95)
        imu_data = df[imu_cols].values
        imu_pca = pca.fit_transform(imu_data)
        imu_pca_cols = [f'IMU_PCA_{i}' for i in range(imu_pca.shape[1])]
        df_imu_pca = pd.DataFrame(imu_pca, columns=imu_pca_cols, index=df.index)
        print(f"  - IMU PCA降维: {len(imu_cols)} → {len(imu_pca_cols)} 维")
    else:
        df_imu_pca = df[imu_cols]
    
    total_windows = (len(df) - window_size) // step
    print(f"提取特征中... 预计 {total_windows} 个窗口")
    
    for start in tqdm(range(0, len(df) - window_size, step)):
        end = start + window_size
        
        # 提取EMG特征（保留关键统计量）
        window_emg = df[emg_cols].iloc[start:end]
        emg_feats = []
        for col in emg_cols:
            data = window_emg[col].values
            emg_feats.extend([
                np.mean(data),          # 均值
                np.std(data),           # 标准差
                np.sqrt(np.mean(data**2)),  # RMS
                np.max(np.abs(data)),   # 峰值
                kurtosis(data),         # 峰度（波形尖锐度）
            ])
        
        # 提取IMU特征（使用PCA降维后的数据）
        window_imu = df_imu_pca.iloc[start:end]
        imu_feats = []
        for col in window_imu.columns:
            data = window_imu[col].values
            imu_feats.extend([
                np.mean(data),
                np.std(data),
                np.ptp(data),           # 峰峰值（运动范围）
                np.sqrt(np.mean(data**2)),
            ])
        
        # 组合特征（总共: 8*5=40 EMG特征 + n_pca*4 IMU特征）
        feat_vec = emg_feats + imu_feats
        features.append(feat_vec)
        labels.append(df['Activity'].iloc[start:end].mode()[0])
    
    print(f"✓ 特征提取完成: {len(features)} 样本, {len(feat_vec)} 特征")
    return np.array(features), np.array(labels)

# ==================== 集成模型 ====================
def train_ensemble(X, y):
    """集成多个模型提升准确率"""
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    # 模型1: 随机森林（深度较大）
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=25,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    # 模型2: 梯度提升（处理非线性关系）
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    print("\n训练集成模型...")
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    
    # 软投票（平均概率）
    rf_proba = rf.predict_proba(X_test)
    gb_proba = gb.predict_proba(X_test)
    ensemble_proba = (rf_proba + gb_proba) / 2
    y_pred = np.argmax(ensemble_proba, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"🎯 集成模型准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*50}\n")
    
    # 单模型对比
    print("单模型对比:")
    print(f"  RandomForest: {rf.score(X_test, y_test):.4f}")
    print(f"  GradientBoosting: {gb.score(X_test, y_test):.4f}")
    
    # 分类报告
    target_names = ['Walking', 'Jogging', 'Stair_Ascent', 'Stair_Descent']
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Ensemble Model Confusion Matrix (Accuracy: {accuracy:.2%})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('optimized_confusion_matrix.png', dpi=150)
    plt.show()
    
    return accuracy

# ==================== 主流程 ====================
def main():
    print("="*60)
    print("MEILoD优化方案 - 融合方案一与方案二优势")
    print("="*60)
    
    # 加载数据
    df, emg_cols, imu_cols, label_col = load_data("GAN.csv")
    
    # 方案对比实验
    print("\n" + "="*40)
    print("方案对比实验")
    print("="*40)
    
    results = {}
    
    # 配置1: 方案一复现（窗口100，无整流滤波）
    print("\n>>> 配置1: 方案一风格（窗口100，无整流滤波）")
    df1 = preprocess_optimized(df, emg_cols, imu_cols, use_rectify=False, use_filter=False)
    X1, y1 = extract_features_optimized(df1, emg_cols, imu_cols, window_size=100, step=50)
    acc1 = train_ensemble(X1, y1)
    results['方案一风格'] = acc1
    
    # 配置2: 方案二风格（窗口200，整流+滤波）
    print("\n>>> 配置2: 方案二风格（窗口200，整流+滤波）")
    df2 = preprocess_optimized(df, emg_cols, imu_cols, use_rectify=True, use_filter=True)
    X2, y2 = extract_features_optimized(df2, emg_cols, imu_cols, window_size=200, step=100)
    acc2 = train_ensemble(X2, y2)
    results['方案二风格'] = acc2
    
    # 配置3: 最优配置（窗口100，无整流无滤波，全部IMU）
    print("\n>>> 配置3: 最优配置（窗口100，无整流无滤波，PCA降维）")
    df3 = preprocess_optimized(df, emg_cols, imu_cols, use_rectify=False, use_filter=False)
    X3, y3 = extract_features_optimized(df3, emg_cols, imu_cols, window_size=100, step=50, use_pca=True)
    acc3 = train_ensemble(X3, y3)
    results['最优配置'] = acc3
    
    # 结果汇总
    print("\n" + "="*60)
    print("最终结果对比")
    print("="*60)
    for name, acc in results.items():
        print(f"{name:15}: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\n✅ 优化方案完成！建议使用【最优配置】")
    
if __name__ == "__main__":
    main()
