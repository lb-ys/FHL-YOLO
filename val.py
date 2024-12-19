import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/amov/paper/训练结果/yolov8s/yolov8s/weights/best.pt')
    model.val(data=r'/home/amov/code/FHL-YOLO/ultralytics/ultralytics/cfg/datasets/my-data.yaml',
              split='test',
              imgsz=640,
              batch=1,
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
              project='runs/val',
              name='exp',
              )