from ultralytics import YOLO
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import logging
from flask import Flask,request, jsonify
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy as np
import cv2
import ast
import json
import base64
from PIL import Image
import io

logging.basicConfig(level=logging.DEBUG)  # DEBUG及以上的日志信息都会显示

app = Flask(__name__)
api = Api(app)

@app.route('/')
def index():
    return 'Hi Dear, please input your specific path'

class Images(Resource):

    def get(self):
        data=pd.read_csv('record.csv') #read csv
        data=data.to_dict()
        return {'data': data}, 200

    def post(self):

        # input check, only support json
        if request.content_type != "application/json":
            return jsonify({'error':'Invalid Content-Type'}), 400

        # converted into ndarray accepted by yolo model
        parser = reqparse.RequestParser()  # 初始化
        parser.add_argument('img_base64', required=True)  # only support json input
        args = parser.parse_args()  # 将参数解析为字典
        img_byte = base64.b64decode(args['img_base64']) # base64 decode,converted into byte
        img_np_arr = np.fromstring(img_byte, np.uint8)
        frame = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)  # converted into ndarray


        # inference
        model = YOLO('mask.pt')
        results=model.predict(frame, save=True,classes=[0,1,2])
        im_array = results[0].plot() # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1]) # RGB PIL image

        # parsed into base64
        image_data = io.BytesIO() # 创建一个BytesIO对象，用于临时存储图像数据
        im.save(image_data,format="JPEG") # 将图像保存到BytesIO对象中，格式为JPEG
        image_data_bytes = image_data.getvalue() # 将BytesIO对象的内容转换为字节串
        output = base64.b64encode(image_data_bytes).decode() # 将图像数据编码为Base64字符串

        return {'data': output},200  # 返回200 OK数据


if __name__ == '__main__':
    api.add_resource(Images, '/images')
    app.run(host='0.0.0.0', port='5000')
    # # 训练数据集：
    # model = YOLO('yolov8n.pt')  # 如果要训练如pose，该对应的yaml和权重即可
    # results = model.train(data='data.yaml', epochs=100)

    #预测结果
    # model = YOLO('mask.pt') #常用模型yolov8n-seg.pt、yolov8n.pt、yolov8n-pose.pt
    # model.predict("test01.png", save=True,classes=[0,1,2]) #测试图片文件夹，并且设置保存True

    #如果中断后，可以改为以下代码：
    # model = YOLO('last.pt')  # last.pt文件的路径
    # results = model.train(resume=True)