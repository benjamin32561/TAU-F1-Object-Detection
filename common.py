import json
import os
import shutil
from glob import glob
import random
import copy

#constants
#tvt split
PHASES = {"train":0.75,"val":0.15,"test":0.1}

#source path consts
SRC_PATH = "../dataset/"
IMAGE_SUB_FOLDER = "images"
LABELS_SUB_FOLDER = "labels"

#source json consts
OBJECTS = "objects"
CLASS_TITLE = "classTitle"
POINTS = "points"
EXT = "exterior"
SIZE = "size"
WIDTH = "width"
HEIGHT = "height"

IMG_EXTEN = ".jpg"

#functions
def CreateDirectories(dirs:list):
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def GetDataFromJson(file_path:str):
    f = open(file_path,'r')
    json_data = json.load(f)
    f.close()
    return json_data

def GetSubFolders(path:str):
    #getting sub folders in some path
    folders = glob(path+"*/", recursive = True)
    return folders, len(folders)

def GetFilesInDir(dir_path:str):
    file_list = os.listdir(dir_path)
    random.shuffle(file_list)
    return file_list, len(file_list)

def GetCurrentPhase(nof:int,i:int,phases:list):
    per_sum = 0
    for phase in phases.keys():
        per_sum+=phases[phase]
        if i <= per_sum*nof:
            return phase

def CopyImage(src_folder:str,dst_folder:str,original_filename:str):
    #save image in new location with correct format
    f = os.path.join(src_folder, original_filename)
    short_filename = original_filename[:-4]
    new_f = os.path.join(dst_folder, short_filename+IMG_EXTEN)
    shutil.copyfile(f, new_f)
    return short_filename

def GetImgWHFromJson(json_data:dict):
    return json_data[SIZE][HEIGHT], json_data[SIZE][WIDTH]

def WriteJsonFiles(path_and_data:dict):
    #write json data to files
    for path in path_and_data.keys():
        with open(path,'w+') as json_file:
            json.dump(path_and_data[path],json_file,indent=4)