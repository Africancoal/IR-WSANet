import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------- 设置字体 & 风格 ----------------
mpl.rcParams['font.family'] = 'Times New Roman'  # 设置字体
mpl.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# ---------------- 数据 ----------------
data = {
    'Model': ['SSD', 'Faster R-CNN', 'YOLO-v7', 'YOLO-v8',
              'YOLO-v10', 'PARE-YOLO', 'IR-WSANet'],
    'mAP0.5_0.95': [43.5, 41.5, 45.8, 49.2, 48.3, 52.7, 54.8],
    'Recall': [68.3, 65.8, 69.7, 72.7, 70.1, 78.0, 77.8],
    'F1': [74.2, 71.4, 75.3, 77.8, 78.3, 81.1, 83.2],
    'Precision': [81.4, 78.1, 81.8, 83.6, 88.7, 89.2, 89.4]
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
ax.set_title('Performance on HIT-UAV Dataset', fontsize=14, fontweight='bold')
ax.set_xticklabels(df['Model'], rotation=45, fontsize=11)
ax.set_yticks(range(40, 96, 5))
ax.set_ylim([40, 96])
ax.tick_params(axis='y', labelsize=11)
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()

# ---------------- 保存 ----------------
save_path = 'accuracy_metrics_highres.png'
fig.savefig(save_path, dpi=2500)
print(f'Figure saved to: {save_path}')
plt.show()
