from ultralytics import YOLO

# 加载 .pt 模型（可以替换为自己的路径）
model = YOLO('garbage_classification.pt')

# 导出为 ONNX 格式
model.export(format='onnx', opset=12, dynamic=True)

print("Export Successful!!!")
