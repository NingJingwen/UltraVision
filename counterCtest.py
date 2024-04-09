# 项目说明：该项目是基于yolov8代码由微智启软件工作室改进，支持鼠标绘制区域计数

# 操作指南
# 1、鼠标中键点击窗口，就可以绘制区域点了
# 2、当绘制好区域后，按空格键就会把坐标转成Polygon对象
# 3、如果要修改模型或者检测其他资源，修改run方法里面的参数即可，75行附近
# 备注：由于可能有多个区域，所以采用随机颜色，如果想采用固定颜色，可以把new_regions对象里面region_color写上固定的RGB颜色如(30, 255, 215)，
# 再把random_color置随机颜色的两行代码删除即可

import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import random
from shapely.geometry import Polygon
from shapely.geometry.point import Point
import os
from PIL import ImageFont, ImageDraw, Image

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
from copy import deepcopy

# 用于存储跟踪历史记录
track_history = defaultdict(list)
# 变量初始化为None，表示当前没有选中的区域
current_region = None  # 当前区域
current_point = None  # 当前鼠标点
detected_count= 0 #统计检测到的总数
up=[] #上行数量
down=[] #下行数量
total_passing=[] #走过区域的总数
total_region=0 #现在在区域的总数
# 两个区域的计数值，颜色，拖拽状态
all_regions = []
new_regions = [
    {
        'polygon': [],
        'counts': 0,
        'dragging': False,
        'region_color': None,
        'text_color': (0, 0, 0),
    }
]
# 它接受五个参数：事件类型、鼠标坐标x、鼠标坐标y、标志位和参数。
def mouse_callback(event, x, y, flags, param):
    global current_region
    global current_point
    # 鼠标左键点击
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in all_regions:
            # 判断鼠标的xy坐标是否在循环遍历的区域
            if region['polygon'].contains(Point((x, y))):
                current_region = region
                current_region['dragging'] = True
                current_region['offset_x'] = x
                current_region['offset_y'] = y
    # 当鼠标移动时执行。如果当前选中的区域不为空且处于拖拽状态，则计算鼠标移动的距离，并根据偏移量更新区域的多边形形状。同时更新偏移量。
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region['dragging']:
            dx = x - current_region['offset_x']
            dy = y - current_region['offset_y']
            current_region['polygon'] = Polygon([
                (p[0] + dx, p[1] + dy) for p in current_region['polygon'].exterior.coords])
            current_region['offset_x'] = x
            current_region['offset_y'] = y

    # 当鼠标左键抬起时执行。如果当前选中的区域不为空且处于拖拽状态，则将拖拽状态设置为False
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region['dragging']:
            current_region['dragging'] = False
    # 鼠标中键按下,记录所有点的坐标，加入新的坐标数组
    elif event == cv2.EVENT_MBUTTONDOWN:
        current_point = (x, y)
        new_regions[0]['polygon'].append((x, y))

def run(
        weights='yolov8n.pt',
        source="",
        device='',
        view_img=False,  # 是否显示图像
        save_img=False,  # 是否保存图像
        exist_ok=False,  # 是否存在
        classes=None,
        line_thickness=2,
        track_thickness=2,
        region_thickness=2,
):
    # 初始化帧计数器为0
    vid_frame_count = 0
    global current_point  # 使用全局变量
    global detected_count
    global up
    global down
    global total_passing
    global total_region
    # 如果检测的资源是摄像头，那么是字符串格式，需要转为数字格式
    if source.isdigit():
        source=int(source)
    # 检查视频源路径是否存在，如果不存在则抛出FileNotFoundError异常
    elif not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # 设置YOLO模型，并根据设备类型选择使用CPU或GPU
    model = YOLO(f'{weights}')
    model.to('cuda') if device == '0' else model.to('cpu')

    # 提取模型中的类别名称
    names = model.model.names
    print(f'type of names {type(names)}')

    # 打开视频文件并获取帧宽度、帧高度、帧率和编码格式
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*'mp4v')

    # 设置输出目录，并在其中创建一个新视频文件
    save_dir = increment_path(Path('output') / 'exp', exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(source,int):
        source=str(source)
    video_writer = cv2.VideoWriter(str(save_dir / f'{Path(source).stem}.mp4'), fourcc, fps, (frame_width, frame_height))
    font = ImageFont.truetype("Alibaba-PuHuiTi-Bold.ttf", size=20, encoding="unic")
    # 遍历视频的每一帧，进行目标检测和跟踪
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1
        scale = 0.3  # 视频缩放比例
        if frame_width>1920 and frame_height>1080:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        # 显示、绘制当前点坐标到窗口上
        if current_point:
            cv2.circle(frame, current_point, 5, (0, 255, 0), 2)
        for i in range(1, len(new_regions[0]['polygon'])):
            cv2.line(frame, new_regions[0]['polygon'][i - 1], new_regions[0]['polygon'][i], (0, 255, 0), 2)

        # Extract the results
        results = model.track(frame, persist=True, classes=classes)
        # 如果检测结果中包含目标框，则提取目标框的位置、ID和类别
        if results[0].boxes.id is not None:
            # 获取所有的目标框的位置信息xyxy，并存储在boxes变量中，其中cpu（）表示将数据从GPU转移到CPU，使用CPU运行时候的选择项
            boxes = results[0].boxes.xyxy.cpu()
            # 获取所有目标框的id信息，并存储到track_ids变量中，其中tolist()表示将数据转为列表类型
            track_ids = results[0].boxes.id.int().cpu().tolist()
            # 获取所有目标框类别序号，并存储到clss变量中
            clss = results[0].boxes.cls.cpu().tolist()
            detected_count=len(clss) # 该帧的类别总数
            # 创建一个注释器对象，在图像上增加线条和文字，用于在图像上绘制目标框和跟踪线
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))
            # 同时遍历每个目标框、跟踪ID和类别，绘制目标框、跟踪线和区域计数。使用zip,就不需要写3个for循环了，简洁代码。
            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, str(names[cls])+str(track_id), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center
                # print(f'type of bbox_center {type(bbox_center)}')
                for i, region in enumerate(all_regions):
                    if region['polygon'].contains(Point((bbox_center[0],bbox_center[1]))):
                        # print(f"type of region[polygon] {type(region['polygon'])}")
                        region['counts']+=1
                        # 判断之前是否有记录过track_id,没有则记录
                        if track_id not in total_passing:
                            total_passing.append(track_id)
                            # 如果是在第一个区域，那么把追踪的ID加入上行的数组
                            if i == 0:
                                if track_id not in up:
                                    up.append(track_id)
                            # 如果是第二个区域，那么加入下行的数组
                            elif i == 1:
                                if track_id not in down:
                                    down.append(track_id)

                track = track_history[track_id]  # 跟踪线
                track.append((float(bbox_center[0]), float(bbox_center[1])))  # 追踪物体的xy坐标
                # 如果跟踪线长度大于30，删除第一个点
                if len(track) > 30:
                    track.pop(0)
                # 将跟踪线中的点坐标连接起来
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                # 在视频帧上绘制跟踪线
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)
                # {
                #     'name': 'YOLOv8 Rectangle Region',
                #     'polygon': Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),  # Polygon points
                #     'counts': 0,  *
                #     'dragging': False,
                #     'region_color': (37, 255, 225),  # BGR Value
                #     'text_color': (0, 0, 0),  # Region Text Color
                # }
                # 检查目标框是否在指定的区域内，如果是，则增加该区域的计数


        # 绘制所有区域的边界框和文本标签
        for region in all_regions:
            total_region += region['counts']
            region_label = str(region['counts'])  # 区域个数
            region_color = region['region_color']  # 区域边界线颜色
            region_text_color = region['text_color']  # 区域文本颜色
            polygon_coords = np.array(region['polygon'].exterior.coords, dtype=np.int32)  # 将多边形的外边界坐标转换为一个NumPy数组
            centroid_x, centroid_y = int(region['polygon'].centroid.x), int(
                region['polygon'].centroid.y)  # 计算多边形的中心点的坐标，方便把计数的位置放到中间
            text_size, _ = cv2.getTextSize(region_label,
                                           cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=0.7,
                                           thickness=line_thickness)
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            # 绘制区域内计数的背景
            cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5),
                          region_color, -1)
            # 计数的数字
            cv2.putText(frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color,
                        line_thickness)
            # frame：表示要绘制多边形的图像。
            # [polygon_coords]：表示多边形的顶点坐标列表。每个顶点坐标是一个二元组 (x, y)。
            # isClosed=True：表示多边形是否闭合。如果为True，则多边形的第一个顶点和最后一个顶点相同，形成一个封闭的形状。如果为False，则多边形不闭合。
            # color=region_color：表示多边形的颜色。可以是一个整数（表示灰度值）或一个三元组（表示BGR颜色值）。
            # thickness=region_thickness：表示多边形的线条粗细。可以是正数（表示线条宽度）或负数（表示填充多边形内部）。
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)

        # 如果需要显示图像，则显示带有目标框和跟踪线的图像
        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow('YOLOv8 Counter')
                cv2.setMouseCallback('YOLOv8 Counter', mouse_callback)

            # 由于CV2对中文显示不友好，需要将 OpenCV 图像转换为 PIL 图像
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # 在 PIL 图像上绘制文字
            draw = ImageDraw.Draw(frame_pil)
            txt = f"检测到物体的总数是：{detected_count}\n现在总共有{total_region}个在划线区域内\n上行的数量是：{len(up)}\n下行的数量是：{len(down)}\n总共有{len(total_passing)}个经过区域"
            #（20,10）是文字开头的初始坐标
            draw.text((20, 10), txt, font=font, fill=(255, 0, 0))
            # 将 PIL 图像转换回 OpenCV 图像
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            # 遍历绘制区域的线条
            for i in range(1, len(new_regions[0]['polygon'])):
                cv2.line(frame, new_regions[0]['polygon'][i - 1], new_regions[0]['polygon'][i], (0, 255, 0), 2)

            cv2.imshow('YOLOv8 Counter', frame)
        # 如果需要保存图像，则将当前帧写入输出视频文件
        if save_img:
            video_writer.write(frame)
        # 重置每个区域的计数
        detected_count=0
        total_region=0
        # up = 0  # 上行数量
        # down = 0  # 下行数量
        # total_passing = []
        for region in all_regions:  # Reinitialize count for each region
            region['counts'] = 0
        # 如果按下'q'键，则退出循环
        key = cv2.waitKey(1)
        # 英文状态'q'键退出，或者右上角按关闭按钮，退出循环
        if key == ord('q') or cv2.getWindowProperty('YOLOv8 Counter wzq', cv2.WND_PROP_VISIBLE) < 1:
            break
        # 当按下空格键时，如果存在未连接的点，把它们连起来
        elif key == ord(' '):
            # 如果坐标大于等于3个，可以把它们封装成Polygon对象
            if len(new_regions[0]['polygon']) >= 3:

                new_regions[0]['polygon'] = Polygon(new_regions[0]['polygon'])
                # 置随机颜色
                random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                new_regions[0]['region_color'] = random_color
                # 把坐标加入all_regions
                # all_regions.append(new_regions[0])
                all_regions.append(deepcopy(new_regions[0]))
                # 清空new_regions里面的坐标
                new_regions[0]['polygon'] = []
                # 清空当前坐标点
                current_point = None
            else:
                print("坐标个数必须大于3")

        elif key == ord('c'):
            new_regions[0]['polygon'] = []
            # 清空当前坐标点
            current_point = None

    # 释放帧计数器、视频写入器和视频捕获器，并关闭所有OpenCV窗口
    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='initial weights path')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--source', default='./data/inference_test/car.mp4', type=str, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--view-img', action='store_true', default=True, help='show results')
    parser.add_argument('--save-img', action='store_true', default=True, help='save results')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--line-thickness', type=int, default=2, help='bounding box thickness')
    parser.add_argument('--track-thickness', type=int, default=2, help='Tracking line thickness')
    parser.add_argument('--region-thickness', type=int, default=4, help='Region thickness')
    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
