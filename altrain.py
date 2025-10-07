import torch
import os
import sys
import yaml
import numpy as np  # 新增 numpy 导入
from datetime import datetime
from ultralytics import YOLOv10

# 设置环境变量（优化显存分配）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# 定义模型路径配置
MODEL_CONFIGS = {
    "yolov10_SHSA_CF": {
        "cfg": "/hy-tmp/yolo-main/ultralytics/cfg/models/v10/yolov10sH.yaml",
        "weights": "/hy-tmp/yolo-main/yolov10s.pt"
    }
}
CODE_DIR = "/hy-tmp/yolo-main/"

def numpy_to_python(obj):
    """递归转换 NumPy 标量为 Python 原生类型"""
    if isinstance(obj, np.generic):
        return obj.item()  # 转换标量
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(v) for v in obj]
    else:
        return obj

def run_experiment(model_type):
    # 动态添加模型代码路径到sys.path
    if CODE_DIR not in sys.path:
        sys.path.insert(0, CODE_DIR)

    try:
        cfg = MODEL_CONFIGS[model_type]
        # 检查文件是否存在
        if not os.path.exists(cfg["cfg"]):
            raise FileNotFoundError(f"Config file missing: {cfg['cfg']}")
        if not os.path.exists(cfg["weights"]):
            raise FileNotFoundError(f"Weights file missing: {cfg['weights']}")

        # 带时间戳的实验名称
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{model_type}_exp_{timestamp}"

        # 初始化模型
        model = YOLOv10(cfg["cfg"])  # 先加载配置文件
        model.load(cfg["weights"])  # 再加载预训练权重

        # 训练配置
        results = model.train(
            data="/hy-tmp/yoloshsa/data1.yaml",
            epochs=200,
            imgsz=640,
            batch=16,
            task="detect",
            amp=True,
            optimizer="Adam",
            #lr0=0.0001,              # 基础学习率 0.0001
            #weight_decay=0.0001,     # 权重衰减 0.0001
            #warmup_epochs=3,         # 预热阶段（需根据数据集大小调整）
            #momentum=0.9,            # 动量 0.9
            #patience=0,             # 耐心值 30
            workers=0,
            device=0,
            name=exp_name,
            project="/hy-tmp/ablation_experiments",
            seed=42,
            deterministic=True
        )

        # 保存完整结果（关键修改点）
        result_path = f"/hy-tmp/ablation_experiments/{exp_name}_full_results.yaml"
        with open(result_path, "w") as f:
            # 构建结果字典
            result_data = {
                "model_type": model_type,
                "config": cfg,
                "metrics": {
                    "map50": results.box.map50,
                    "map": results.box.map,
                    "precision": results.box.mp,
                    "recall": results.box.mr,
                    "f1": 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-6),
                    "maps": list(results.box.maps),
                },
                "speed": results.speed,
                "nparams": sum(p.numel() for p in model.model.parameters()),
                "gflops": model.info()[3] if isinstance(model.info(), tuple) else "N/A",
                "nc": model.model.model[-1].nc,
                "names": model.model.names,
            }
            # 执行递归类型转换
            cleaned_data = numpy_to_python(result_data)
            yaml.dump(cleaned_data, f, default_flow_style=False)  # 禁用流式风格

        print(f"✅ Experiment {exp_name} results saved to {result_path}")

    finally:
        # 恢复sys.path
        if CODE_DIR in sys.path:
            sys.path.remove(CODE_DIR)

if __name__ == "__main__":
    for model_type in MODEL_CONFIGS.keys():
        print(f"\n🚀 Starting experiment for {model_type}")
        run_experiment(model_type)