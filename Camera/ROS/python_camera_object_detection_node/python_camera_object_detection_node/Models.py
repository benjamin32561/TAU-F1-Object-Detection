import numpy as np

CLASS_CONF_THRESHOLD = 0.5

class YOLOv5():
    def __init__(self,model_path:str):
        self.model = None
    
    def DetectObbjects(self,img:np.ndarray):
        return ''