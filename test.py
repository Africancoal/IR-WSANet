import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import yaml_load, LOGGER

yaml_path = "C:/Users/Hasee/PycharmProjects/pythonProject/yolowW/ultralytics/cfg/models/v10/yolov10s-w.yaml"  # Replace with your YAML path
model = DetectionModel(cfg=yaml_path, ch=3, nc=1, verbose=True)
print(model)