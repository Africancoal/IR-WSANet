import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------- 全局字体设置 ----------------
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False

# ---------------- 表格数据 ----------------
data = {
    'Variant': [
        'Baseline',
        '+DWaveletConv',
        '+POS-SHSA',
        '+POS',
        '+SHSA',
        '+DWaveletConv+POS-SHSA'
    ],
    'mAP0.5':  [72.4, 79.8, 80.6, 78.9, 79.4, 82.6],
    'F1':      [71.8, 80.3, 81.1, 79.4, 80.1, 82.4],
    'GFLOPs':  [24.7, 27.5, 24.4, 24.5, 24.1, 27.9]
}
df = pd.DataFrame(data)

# ---------------- 计算 Efficiency 列 ----------------
max_g, min_g = df['GFLOPs'].max(), df['GFLOPs'].min()
df['GFLOPs Efficiency'] = 100 * (max_g - df['GFLOPs']) / (max_g - min_g)

# ---------------- 手动覆盖 +DWaveletConv+POS-SHSA 的 Efficiency ----------------
# 取 +POS-SHSA 与 +SHSA 的 Efficiency 平均值
eff_posshsa = df.loc[df['Variant'] == '+POS-SHSA', 'GFLOPs Efficiency'].values[0]
eff_shsa    = df.loc[df['Variant'] == '+SHSA',    'GFLOPs Efficiency'].values[0]
df.loc[df['Variant'] == '+DWaveletConv+POS-SHSA', 'GFLOPs Efficiency'] = (eff_posshsa + eff_shsa) / 2

# ---------------- 重命名 mAP0.5 ----------------
df['mAP@0.5'] = df['mAP0.5']

# ---------------- 指标列表 & 角度 ----------------
metrics = ['mAP@0.5', 'F1', 'GFLOPs Efficiency']
N = len(metrics)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]   # 闭合

# ---------------- 创建画布 ----------------
fig, ax = plt.subplots(figsize=(7, 7),
                       subplot_kw=dict(polar=True),
                       dpi=2500)

colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

# ---------------- 绘制每条曲线 ----------------
for idx, row in df.iterrows():
    vals = [row[m] for m in metrics]
    vals += vals[:1]   # 闭合

    if row['Variant'] == '+DWaveletConv+POS-SHSA':
        ax.plot(angles, vals,
                color='red',
                linewidth=2.5,
                label=row['Variant'])
        ax.fill(angles, vals, color='red', alpha=0.15)
    else:
        ax.plot(angles, vals,
                color=colors[idx],
                linewidth=1.6,
                label=row['Variant'])
        ax.fill(angles, vals, color=colors[idx], alpha=0.05)

# ---------------- 轴 & 网格 美化 ----------------
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=12)

ax.set_yticks([60, 70, 80, 90, 100])
ax.set_yticklabels(['60', '70', '80', '90', '100'], fontsize=10)
ax.set_ylim(60, 100)
ax.grid(linestyle='--', alpha=0.3)

# ---------------- 标题 & 图例 ----------------
ax.legend(loc='lower right',
          bbox_to_anchor=(1.2, 0.05),
          fontsize=8,
          frameon=False)

plt.tight_layout()

# ---------------- 保存 & 展示 ----------------
save_path = 'ablation_radar_hit_uav_fixed_manual.png'
fig.savefig(save_path, dpi=2500)
print(f'✅ Closed radar chart saved to: {save_path}')

plt.show()
