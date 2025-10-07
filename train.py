import torch
from ultralytics import YOLOv10
import os
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.YOLOv10DetectionModel'])
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
# 加载模型
model = YOLOv10('/hy-tmp/yolosaha/yolov10s.pt')
model = YOLOv10("/hy-tmp/yolo/ultralytics/cfg/models/v10/yolov10s.yaml")

# 开始训练
model.train(
    data='/hy-tmp/yolosaha/data.yaml',
    epochs=200,
    imgsz=512,
    batch=4,
    task='detect',
    amp = False,  # 保持混合精度开启
    optimizer = 'Adam',  # 改用原生Adam
    workers = 0,  # 减少数据加载进程
    device = 0  # 确保只使用单卡
)