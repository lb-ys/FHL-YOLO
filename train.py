import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/amov/code/FHL-YOLO/ultralytics/ultralytics/cfg/models/v8/FHL-YOLO.yaml')

    # model.load('yolov8n.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度

    model.train(data=r'/home/amov/code/FHL-YOLO/ultralytics/ultralytics/cfg/datasets/my-data.yaml',
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache=False,
                imgsz=640,
                epochs=10,
                single_cls=False,  # 是否是单类别检测
                batch=1,
                close_mosaic=10,
                workers=0,
                device=' ',
                optimizer='SGD', # using SGD
                # resume='runs/train/exp21/weights/last.pt', # 如过想续训就设置last.pt的地址
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='runs',
                name='exp',
                )