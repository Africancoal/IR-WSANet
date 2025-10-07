from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------- 指标 & 数据 -----------
metrics  = ['Precision', 'Recall', 'mAP$_{0.5}$',
            'mAP$_{0.5:0.95}$', 'F1', 'GFLOPs', 'FPS']
datasets = ['SIDD‑City', 'SIDD‑Mountain', 'HIT‑UAV']
models   = ['YOLO‑v10', 'IR‑WSANet']

data = {
    'SIDD‑City': {
        'YOLO‑v10'  : [99.5, 93.5, 94.7, 75.2, 96.6, 24.8, 41.9],
        'IR‑WSANet': [99.8, 94.4, 97.2, 76.4, 97.6, 27.9, 42.8]},
    'SIDD‑Mountain': {
        'YOLO‑v10'  : [86.9, 73.2, 80.6, 48.3, 79.8, 24.8, 41.9],
        'IR‑WSANet': [88.1, 74.8, 82.6, 49.2, 80.4, 27.9, 42.8]},
    'HIT‑UAV': {
        'YOLO‑v10'  : [88.7, 70.1, 72.7, 48.3, 78.3, 24.8, 41.9],
        'IR‑WSANet': [89.4, 77.8, 82.8, 54.8, 83.2, 27.9, 42.8]},
}

# ----------- 画图 -----------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(metrics))

total_width = 0.8
bar_w = total_width / (len(datasets) * len(models))

colors  = ['#4C72B0', '#55A868', '#C44E52']     # 数据集颜色
hatches = {'YOLO‑v10': '//', 'IR‑WSANet': ''}   # 模型区分样式

for d_idx, ds in enumerate(datasets):
    for m_idx, model in enumerate(models):
        offset = (-total_width/2) + (d_idx*len(models)+m_idx+0.5)*bar_w
        ax.bar([xi + offset for xi in x],
               data[ds][model],
               width=bar_w,
               color=colors[d_idx],
               edgecolor='black',
               hatch=hatches[model])

# ----------- 合并图例（右上角）-----------
model_handles = [
    Patch(facecolor='white', edgecolor='black', hatch='//', label='YOLO‑v10'),
    Patch(facecolor='white', edgecolor='black', hatch='',   label='IR‑WSANet')
]
dataset_handles = [
    Patch(facecolor=colors[i], edgecolor='black', label=datasets[i])
    for i in range(len(datasets))
]

# 合并图例列表并放在右上角
combined_handles = model_handles + dataset_handles
ax.legend(handles=combined_handles, loc='upper right',


fontsize=10, title_fontsize=11,
          frameon=True, ncol=1)

# ----------- 美化 -----------
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylabel('Score / Value', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.6)
fig.tight_layout()

# ----------- 导出 -----------
plt.savefig('wsanet_vs_yolov10_datasets_metrics_final.png', dpi=2500, bbox_inches='tight')
plt.show()
