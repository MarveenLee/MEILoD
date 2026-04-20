https://data.mendeley.com/datasets/ydz48cby4t/1

MEILoD – Multimodal EMG–IMU Locomotion Dataset
Published: 13 March 2026
|
Version 1
|
DOI:
10.17632/ydz48cby4t.1
Contributors:
Mohamed Khaled Farouk
,
Ahmed Abdellatif Hamed IBRAHIM
,
Mohamed Fawzy El-Khatib
,
Mohammed I. Awad
,
Mostafa R. A. Atia
Description
MEILoD (Multimodal EMG–IMU Locomotion Dataset) is a synchronized wearable sensor dataset designed for multimodal Human Activity Recognition (HAR) and lower-limb gait analysis. The dataset contains surface electromyography (EMG) and 6-axis inertial measurement unit (IMU) signals collected using the DELSYS Trigno™ Avanti wireless system from nine healthy male participants (age 12–39) performing four locomotor activities: walking, jogging, stair ascent, and stair descent. EMG signals were recorded using eight sensors placed bilaterally on the Rectus Femoris, Vastus Lateralis, Vastus Medialis, and Semitendinosus muscles.

MEILoD was collected as part of a thesis project “Deep Learning Approaches for Human Gait Assessment Using Wearable Sensors” to support research in wearable HAR, EMG–IMU fusion, gait modeling, and assistive exoskeleton development. The dataset is released in raw format to enable reproducible preprocessing, benchmarking, and methodological comparison across deep learning architectures.

The dataset is provided in two versions: MEILoD v1.0 – Raw Dataset, containing nine participant-level CSV files and a merged file including all subjects; and MEILoD v1.1 – GAN-Augmented Dataset, which includes participant-level and merged augmented files. The augmentation was performed using a time-series Generative Adversarial Network (GAN) to mitigate class imbalance, increasing the dataset volume by approximately 12%. Augmentation was applied to minority classes only, reducing walking-class dominance and improving class balance without artificially inflating classification accuracy.

Download All 395 MB

Files

rar
MEILoD v1.0 – Raw Dataset.rar
195 MB

rar
MEILoD v1.1 – GAN-Augmented.rar
200 MB

png
RAW Vs GAN.png
104 KB
Categories
Human Activity in Medical Context, Activity Recognition, Electromyography, Wearable Sensor, Convolutional Neural Network, Deep Learning, Neural Network, Long Short-Term Memory Network, Generative Adversarial Network, Data Augmentation






pip install numpy pandas matplotlib scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

