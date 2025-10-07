import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------- 设置字体 & 风格 ----------------
mpl.rcParams['font.family'] = 'Times New Roman'  # 设置字体
mpl.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# ---------------- 数据 ----------------
data = {
    'Model': ['SSD', 'Faster R-CNN', 'YOLO-v7', 'YOLO-v8',
              'YOLO-v10', 'RT-DETR', 'IR-WSANet'],
    'mAP0.5_0.95': [74.1, 72.3, 74.2, 75.7, 75.2, 74.5, 76.4],
    'Recall': [82.2, 80.4, 92.9, 93.4, 93.5, 94.1, 94.4],
    'F1': [88.9, 87.3, 96.0, 96.9, 96.6, 96.7, 97.6],
    'Precision': [96.7, 95.5, 99.4, 99.8, 99.5, 99.4, 99.8]
}
df = pd.DataFrame(data)

# ---------------- 配色 & 样式 ----------------
colors = {
    'mAP0.5_0.95': '#1f77b4',  # 蓝
    'Recall': '#2ca02c',  # 绿
    'F1': '#ff7f0e',  # 橙
    'Precision': '#d62728'  # 红
}
linestyles = {
    'mAP0.5_0.95': '-',
    'Recall': '--',
    'F1': '-.',
    'Precision': ':'
}

# ---------------- 绘图 ----------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=2500)

ir_idx = df[df['Model'] == 'IR-WSANet'].index[0]

for metric in ['mAP0.5_0.95', 'Recall', 'F1', 'Precision']:
    ax.plot(df['Model'], df[metric],
            label=metric,
            color=colors[metric],
            linestyle=linestyles[metric],
            marker='o',
            linewidth=2.2,
            markersize=6)

    # 标注 IR-WSANet 点
    y_val = df.loc[ir_idx, metric]
    ax.annotate(f'{y_val}', (ir_idx, y_val), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=9)

# ---------------- 细节优化 ----------------
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Performance on SIDD-City Dataset', fontsize=14, fontweight='bold')
ax.set_xticklabels(df['Model'], rotation=45, fontsize=11)
ax.set_yticks(range(70, 101, 5))
ax.set_ylim([70, 101])
ax.tick_params(axis='y', labelsize=11)
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()

# ---------------- 保存 ----------------
save_path = 'accuracy_metrics_highres.png'
fig.savefig(save_path, dpi=2500)
emf_save_path = 'accuracy_metrics_SCI.emf'
fig.savefig(emf_save_path, format='emf', bbox_inches='tight')
print(f'Figure saved to: {save_path}')
plt.show()
