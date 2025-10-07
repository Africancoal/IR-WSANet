from ultralytics import YOLOv10
import torch
import os
import shutil
from pathlib import Path

try:
    from ultralytics.nn.modules.SHSA import SHSA  # 导入 SHSA
except ImportError:
    raise ImportError("无法导入 SHSA 模块，请检查 ultralytics/nn/modules/SHSA.py 是否定义了 SHSA 类")
# 检查 GPU 可用性
if torch.cuda.is_available():
    device = '0'  # 使用第 0 个 GPU
else:
    raise Exception("CUDA is not available")

# 定义路径
model_path = "./weights/last.pt"  # 模型权重路径
source_path = "C:/data/SIDD-voc/city/JPEGImages"  # 输入图片目录
save_name = "analysis"  # 保存结果的子目录名称
no_detection_dir = "no_detections"  # 未检测到目标的子文件夹名称

# 加载模型
model = YOLOv10(model_path)

# 执行目标检测
results = model(source=source_path,
                name=save_name,
                conf=0.45,  # 置信度阈值
                save=True,  # 保存检测结果
                device=device  # 指定设备
                )

# 获取保存路径
save_dir = Path(f"runs/detect/{save_name}")
no_detection_path = save_dir / no_detection_dir  # 未检测到目标的子文件夹完整路径

# 创建未检测到目标的子文件夹
os.makedirs(no_detection_path, exist_ok=True)

# 处理检测结果
for result in results:
    # 获取当前图像的文件名和保存路径
    img_path = Path(result.path)  # 原始图像路径
    saved_img_path = save_dir / img_path.name  # 保存的检测结果图像路径

    # 检查是否有检测到目标（边界框数量）
    if result.boxes is None or len(result.boxes) == 0:
        # 如果没有检测到目标，移动到 no_detections 子文件夹
        if saved_img_path.exists():
            shutil.move(str(saved_img_path), str(no_detection_path / img_path.name))
            print(f"未检测到目标，移动 {img_path.name} 到 {no_detection_path}")
    else:
        # 如果检测到目标，保留在 analysis 目录下
        print(f"检测到目标，保留 {img_path.name} 在 {save_dir}")

print(f"检测完成，结果保存在 runs/detect/{save_name}/")
print(f"未检测到目标的图像保存在 runs/detect/{save_name}/{no_detection_dir}/")