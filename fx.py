import numpy as np
import matplotlib.pyplot as plt
import os

# 定义Leaky ReLU函数
def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

# 定义Leaky ReLU的导函数
def leaky_relu_derivative(x, alpha=0.1):
    return np.where(x > 0, 1, alpha)

# 生成x值范围（-10到10之间，步长0.1）
x = np.arange(-10, 10, 0.1)

# 计算对应的函数值
alpha = 0.1  # 泄漏系数
y_leaky_relu = leaky_relu(x, alpha)
y_derivative = leaky_relu_derivative(x, alpha)

# 创建图形和坐标轴
plt.figure(figsize=(10, 6), dpi=120)  # 设置更高的DPI以获得更好的分辨率

# 在同一坐标系中绘制两条曲线
plt.plot(x, y_leaky_relu, 'b-', linewidth=2, label='Leaky ReLU (α=0.1)')
plt.plot(x, y_derivative, 'r-', linewidth=2, label='Derivative Leaky ReLU')

# 添加图例和标题
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel('x', fontsize=12)

# 添加关键参考线
plt.axhline(y=0, color='b', linestyle='--', alpha=0.3)  # 零点
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)  # 导函数的渐近线
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)  # 添加垂直参考线

# 标记导函数的不同区域
plt.annotate('$LeakyReLU\'(x) = α$', xy=(-5, 0.1), xytext=(-8, 0.5),
             arrowprops=dict(arrowstyle='->', color='black'), color='red', fontsize=10)
plt.annotate('$LeakyReLU\'(x) = 1$', xy=(5, 0.9), xytext=(2, 0.7),
             arrowprops=dict(arrowstyle='->', color='black'), color='red', fontsize=10)

# 标记函数的不连续点
plt.annotate('MAX (x=0)', xy=(0, 0), xytext=(0.5, -0.5),
             arrowprops=dict(arrowstyle='->', color='gray'), color='gray', fontsize=10)

# 设置坐标轴范围
plt.xlim(-10, 10)
plt.ylim(-1.5, 10.5)  # 值域为[-∞,∞)，这里设置为-1.5到10.5

# 添加双Y轴标签
plt.ylabel('Leaky ReLU(x) (blue)', color='b', fontsize=12)
plt.tick_params(axis='y', labelcolor='b')

# 添加第二个Y轴用于导函数
ax2 = plt.gca().twinx()
ax2.set_ylabel('Leaky ReLU\'(x) (red)', color='r', fontsize=12)
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylim(-0.2, 1.2)  # 导函数的Y轴范围

# 添加网格和调整布局
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# 保存图像到本地
output_dir = os.path.join(os.path.expanduser('~'), 'Downloads')  # 保存到用户下载目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, 'leaky_relu_and_derivative.png')

# 以高分辨率保存图像
dpi = 300  # 设置高DPI值
plt.savefig(output_path, dpi=dpi, bbox_inches='tight')

# 计算图像分辨率（像素）
width_inches, height_inches = plt.gcf().get_size_inches()
width_pixels = int(width_inches * dpi)
height_pixels = int(height_inches * dpi)

# 显示保存信息
print(f"图像已保存至: {output_path}")
print(f"图像分辨率: {width_pixels} × {height_pixels} 像素 (DPI={dpi})")
print(f"文件大小: {os.path.getsize(output_path)/1024:.2f} KB")

# 显示图像
plt.show()