import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt
from scipy.stats import ttest_rel
from itertools import combinations
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample
from sklearn.base import clone

files = sorted(glob.glob('data/**/*.csv', recursive=True))

def label_of(path):
    return os.path.basename(path).replace('.csv', '').split('_')[1]

def preprocess(x, fs=1000):
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=0)
    b, a = butter(4, [20/(fs/2), 450/(fs/2)], btype='band')
    return filtfilt(b, a, x, axis=0)

def make_windows(x, win=200, hop=100):
    n = (len(x) - win) // hop + 1
    return np.stack([x[i*hop : i*hop+win] for i in range(n)])

def td_features(w, th=0.01):
    rms = np.sqrt(np.mean(w**2, axis=0))
    mav = np.mean(np.abs(w), axis=0)
    d = np.diff(w, axis=0)
    wl = np.sum(np.abs(d), axis=0)
    zc = np.sum(((w[:-1]*w[1:]) < 0) & (np.abs(d) >= th), axis=0)
    ssc = np.sum((d[:-1]*d[1:]) < 0, axis=0)
    return np.concatenate([rms, mav, wl, zc, ssc])

X_all, y_all = [], []
for p in files:
    lbl = label_of(p)
    raw_x = np.loadtxt(p, delimiter=',', skiprows=1)[:, :5]
    xf = preprocess(raw_x)
    
    for i in range(30):
        start = i * 5000 + 1500
        seg = xf[start:start+2000]
        if len(seg) == 2000:
            feats = [td_features(w) for w in make_windows(seg)]
            X_all.append(np.mean(feats, axis=0))
            y_all.append(lbl)

X_all = np.array(X_all)
y_all = np.array(y_all)

X_sub, y_sub = resample(X_all, y_all, n_samples=1216, stratify=y_all, random_state=42)

Xtr, Xte, ytr, yte = train_test_split(
    X_sub, y_sub, test_size=247, stratify=y_sub, random_state=42
)

print(f"학습셋 배열: {Xtr.shape}")
print(f"테스트셋 배열: {Xte.shape}\n")

sc = StandardScaler()
Xtr_s = sc.fit_transform(Xtr)
Xte_s = sc.transform(Xte)

le = LabelEncoder()
ytr_e = le.fit_transform(ytr)
yte_e = le.transform(yte)

models = {
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovr'),
    'RF' : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGB': XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, 
                         subsample=0.9, colsample_bytree=0.8, random_state=42)
}

print("--- 1. Hold-out 평가 및 계산 비용 ---")
rows = []
for k, m in models.items():
    # 학습 시간 측정
    t0 = time.time()
    m.fit(Xtr_s, ytr_e)
    train_t = time.time() - t0
    
    t0 = time.time()
    for _ in range(100): 
        m.predict(Xte_s[:1])
    infer_ms = (time.time() - t0) * 10
    
    print(f'{k}: 학습 {train_t:.3f}s, 추론 {infer_ms:.3f} ms/sample')
    
    p = m.predict(Xte_s)
    rows.append([k,
        accuracy_score(yte_e, p),
        precision_score(yte_e, p, average='macro'),
        recall_score(yte_e, p, average='macro'),
        f1_score(yte_e, p, average='macro')
    ])

print("\n[성능 비교표]")
df = pd.DataFrame(rows, columns=['model', 'acc', 'precision', 'recall', 'f1'])
print(df.round(4))

print("\n--- 2. 혼동 행렬 시각화 ---")
cm = confusion_matrix(yte_e, models['RF'].predict(Xte_s))
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(8), le.classes_, rotation=45)
ax.set_yticks(range(8), le.classes_)
for i in range(8):
    for j in range(8):
        if cm[i,j]: 
            ax.text(j, i, cm[i,j], ha='center', va='center', fontsize=8)
plt.colorbar(im)
plt.tight_layout()
plt.savefig('confusion8.png', dpi=120)
print("'confusion8.png' 이미지 저장 완료")

print("\n--- 3. 5-Fold 교차검증 결과 ---")
skf = StratifiedKFold(5, shuffle=True, random_state=42)
fold_acc = {k: [] for k in models}

le_cv = LabelEncoder()
le_cv.fit(y_sub)

for tr_i, va_i in skf.split(X_sub, y_sub):
    Xa, Xb = X_sub[tr_i], X_sub[va_i]
    ya, yb = y_sub[tr_i], y_sub[va_i]
    
    sc_cv = StandardScaler()
    Xa = sc_cv.fit_transform(Xa)
    Xb = sc_cv.transform(Xb)
    
    ya = le_cv.transform(ya)
    yb = le_cv.transform(yb)
    
    for k, proto in models.items():
        m = clone(proto)
        m.fit(Xa, ya)
        fold_acc[k].append(m.score(Xb, yb))

for k, v in fold_acc.items():
    print(f'{k}: {np.mean(v)*100:.2f} ± {np.std(v)*100:.2f} %')

print("\n--- 4. 통계적 유의성 검정 ---")
for a, b in combinations(models.keys(), 2):
    t, p = ttest_rel(fold_acc[a], fold_acc[b])
    if not np.isfinite(p): 
        print(f'{a} vs {b}: 검정 불가')
        continue
    verdict = '유의함' if p < 0.05 else '유의하지 않음'
    print(f'{a} vs {b}: p = {p:.4f} ({verdict})')