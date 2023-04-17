#tvt split
DATASET_SPLIT_RATIO = {"train":0.8,"val":0.2}

#source path consts
SRC_PATH = "/content/dataset/"
IMAGES_SUB_FOLDER = "img"
LABELS_SUB_FOLDER = "ann"

#source json consts
OBJECTS = "objects"
CLASS_TITLE = "classTitle"
POINTS = "points"
EXT = "exterior"
SIZE = "size"
WIDTH = "width"
HEIGHT = "height"
ID = "id"

IMG_EXTEN = ".jpg"

MOVE_IMAGES = True

BOUNDING_BOX_AREA_THRESH = 0.0009 #0.000025
MIN_BBX_WIDTH = 100
MIN_BBX_HEIGHT = 100

NOISE_CLASS = ['unknwon_cone']