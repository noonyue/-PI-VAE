"""
生成全部模型的L1药物分类混淆矩阵大图
包含：PLS-DA, SVM, RandomForest, CNN, LSTM, Transformer (使用Raw特征)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 添加父目录以导入pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pi_vae_pipeline import load_data, preprocess_spectra

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9

# ========== 定义模型类 ==========

class PLSDAWrapper:
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_numeric = np.array([np.where(self.classes_ == c)[0][0] for c in y])
        n_comp = min(self.n_components, max(1, len(self.classes_) - 1))
        self.model = PLSRegression(n_components=n_comp)
        self.model.fit(X, y_numeric)
        return self

    def predict(self, X):
        y_cont = self.model.predict(X).ravel()
        y_idx = np.clip(np.round(y_cont).astype(int), 0, len(self.classes_) - 1)
        return self.classes_[y_idx]

class CNN1D(nn.Module):
    def __init__(self, input_len, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            out = self.features(dummy)
            flat = out.view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)

class LSTM1D(nn.Module):
    def __init__(self, input_len, n_classes):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h)

class TransformerClassifier(nn.Module):
    def __init__(self, input_len, n_classes):
        super().__init__()
        self.embed = nn.Linear(1, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embed(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

def train_deep_model(model, X_train, y_train, X_test, y_test, epochs=50, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # 预测
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_test_tensor)
        y_pred = outputs.argmax(dim=1).cpu().numpy()

    return y_pred

# ========== 加载数据 ==========
print("Loading data...")
X_uv_raw, X_nir_raw, y_drug, y_manufacturer = load_data('Sampedata0.xlsx')

X_uv = preprocess_spectra(X_uv_raw)
X_nir = preprocess_spectra(X_nir_raw)
X_raw = np.hstack([X_uv, X_nir])

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_drug, test_size=0.2, random_state=42, stratify=y_drug
)

# 将标签编码为数值
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ========== 训练并获取混淆矩阵 ==========
models_results = {}

# 1. PLS-DA
print("Training PLS-DA...")
plsda = PLSDAWrapper(n_components=5)
plsda.fit(X_train, y_train)
y_pred_plsda = plsda.predict(X_test)
models_results['PLS-DA'] = confusion_matrix(y_test, y_pred_plsda)

# 2. SVM
print("Training SVM...")
svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
models_results['SVM'] = confusion_matrix(y_test, y_pred_svm)

# 3. RandomForest
print("Training RandomForest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
models_results['RandomForest'] = confusion_matrix(y_test, y_pred_rf)

# 4. CNN
print("Training CNN...")
cnn = CNN1D(input_len=X_raw.shape[1], n_classes=len(np.unique(y_drug)))
y_pred_cnn_encoded = train_deep_model(cnn, X_train, y_train_encoded, X_test, y_test_encoded, epochs=50)
y_pred_cnn = le.inverse_transform(y_pred_cnn_encoded)
models_results['CNN'] = confusion_matrix(y_test, y_pred_cnn)

# 5. LSTM
print("Training LSTM...")
lstm = LSTM1D(input_len=X_raw.shape[1], n_classes=len(np.unique(y_drug)))
y_pred_lstm_encoded = train_deep_model(lstm, X_train, y_train_encoded, X_test, y_test_encoded, epochs=50)
y_pred_lstm = le.inverse_transform(y_pred_lstm_encoded)
models_results['LSTM'] = confusion_matrix(y_test, y_pred_lstm)

# 6. Transformer
print("Training Transformer...")
transformer = TransformerClassifier(input_len=X_raw.shape[1], n_classes=len(np.unique(y_drug)))
y_pred_transformer_encoded = train_deep_model(transformer, X_train, y_train_encoded, X_test, y_test_encoded, epochs=50)
y_pred_transformer = le.inverse_transform(y_pred_transformer_encoded)
models_results['Transformer'] = confusion_matrix(y_test, y_pred_transformer)

# ========== 绘制大图 ==========
print("Generating confusion matrix figure...")

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

drug_labels = [f'药物{i}' for i in range(9)]

for idx, (model_name, cm) in enumerate(models_results.items()):
    row = idx // 3
    col = idx % 3
    ax = fig.add_subplot(gs[row, col])

    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 绘制热图
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=drug_labels, yticklabels=drug_labels,
                cbar_kws={'label': '归一化计数'}, ax=ax, linewidths=0.5)

    # 计算准确率
    accuracy = np.trace(cm) / np.sum(cm)

    # 设置标题
    panel_label = chr(97 + idx)  # a, b, c, ...
    ax.set_title(f'({panel_label}) {model_name} (准确率: {accuracy:.2%})',
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('预测标签', fontsize=10)
    ax.set_ylabel('真实标签', fontsize=10)

# 添加总标题
fig.suptitle('L1药物分类 - 全部模型混淆矩阵对比',
             fontsize=16, fontweight='bold', y=0.98)

# 保存图片
plt.savefig('figures/all_models_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("All models confusion matrix saved: figures/all_models_confusion_matrix.png")
plt.close()
