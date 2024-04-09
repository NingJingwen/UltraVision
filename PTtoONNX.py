#更多功能参数介绍：https://docs.ultralytics.com/modes/export/#key-features-of-export-mode
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('runs/detect/mask_train_v1/weights/best.pt')  # load a custom trained model
    # Export the model
    model.export(format='onnx')