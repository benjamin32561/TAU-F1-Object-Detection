import torch
import numpy as np

CLASS_CONF_THRESHOLD = 0.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class YOLOv5():
    def __init__(self,model_path:str):
        self.image_dimensions = (640,640)
        self.model = None
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(DEVICE)
        except Exception as e:
            self.get_logger().error(e)
            exit()
        self.model.conf = CLASS_CONF_THRESHOLD
    
    def DetectObbjects(self,img:np.ndarray):
        results = self.model([img])
        results_df = results.pandas().xyxy[0]
        return results_df