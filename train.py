
from ultralytics import YOLO
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if __name__ == '__main__':
    # # 训练数据集：
    model = YOLO('yolov8n.pt')  # 如果要训练如pose，该对应的权重即可
    results = model.train(data='data.yaml', epochs=100)

    #预测结果
    # model = YOLO('yolov8n-pose.pt') #常用模型yolov8n-seg.pt、yolov8n.pt、yolov8n-pose.pt
    # model.predict("ultralytics/assets", save=True) #测试图片文件夹，并且设置保存True

#如果中断后，可以改为以下代码：
# model = YOLO('last.pt')  # last.pt文件的路径
# results = model.train(resume=True)