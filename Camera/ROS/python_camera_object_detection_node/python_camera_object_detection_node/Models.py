from abc import abstractmethod
import torch
import numpy as np

CLASS_CONF_THRESHOLD = 0.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BaseModel():
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.image_dimensions = (640,640)
        self.model_path = model_path
        self.model = None

    @abstractmethod
    def DetectObjects(self,img:np.ndarray,show:bool=False):
        pass
    
    @abstractmethod
    def ModelResultsToMsgStr(self,res):
        pass

class YOLOv5(BaseModel):
    def __init__(self, model_path:str=''):
        super().__init__(model_path)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)  # local model
        self.model = self.model.to(DEVICE)
        self.model.conf = CLASS_CONF_THRESHOLD
    
    def DetectObjects(self,img:np.ndarray,show:bool=False):
        results = self.model([img])
        if show:
            results.show()
        results_df = results.pandas().xyxy[0]
        return results_df

    def ModelResultsToMsgStr(self,res):
        pass