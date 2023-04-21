import json
import os
import shutil
import random
import copy
from glob import glob
from .consts import DATASET_SPLIT_RATIO, IMG_EXTEN, SIZE, WIDTH, HEIGHT, MOVE_IMAGES, \
                    POINTS, EXT

"""
desription: creates all directories that the list contains
input:
	dirs - list od directories to create
output:
"""
def CreateDirectories(dirs:list):
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

"""
desription: return data from json file
input:
	file_path - json file path relative to run file
output:
	dict - the json data from the json file
"""
def GetDataFromJson(file_path:str):
    f = open(file_path,'r')
    json_data = json.load(f)
    f.close()
    return json_data

"""
desription: gets all sub-folders in path
input:
	path - the location from which to extract the sub-folders
output:
	list - list of all sub-folders
	int - amount of sub-folders
"""
def GetSubFolders(path:str):
    #getting sub folders in some path
    folders = glob(path+"*/", recursive = True)
    return folders, len(folders)

"""
desription: gets all files from directory and shuffling it
input:
	dir_path - directory path to get file list from
output:
	list - list of all files (shuffled)
	int - amount of files
"""
def GetFilesInDir(dir_path:str, shuffle=True):
    file_list = os.listdir(dir_path)
    if shuffle:
        random.shuffle(file_list)
    return file_list, len(file_list)

"""
desription: returns current phase of split
input:
	nof - number of files 
	i - index of current file
output:
	string - current phase
"""
def GetCurrentPhase(nof:int,i:int):
    per_sum = 0
    for phase in DATASET_SPLIT_RATIO.keys():
        per_sum+=DATASET_SPLIT_RATIO[phase]
        if i <= per_sum*nof:
            return phase

"""
desription: cpies image from source location to new
			location and setting image extention to IMG_EXTEN
input:
	src_folder - source folder of image
	dst_folder - destination folder for image
	original_filename - original filename (with original extention)
output:
	string - the file name without the extention
"""
def CopyImage(src_folder:str,dst_folder:str,original_filename:str):
    #save image in new location with correct format
    f = os.path.join(src_folder, original_filename)
    short_filename = original_filename[:-4]
    new_f = os.path.join(dst_folder, short_filename+IMG_EXTEN)
    if MOVE_IMAGES:
        shutil.move(f, new_f)
    else:
        shutil.copyfile(f, new_f)
    return short_filename

"""
desription: gets image height and width
input:
	json_data - json data of original dataset
output:
	int - height of image
	int - width of image
"""
def GetImgHWFromJson(json_data:dict):
    return json_data[SIZE][HEIGHT], json_data[SIZE][WIDTH]

"""
desription: saves json data to disk
input:
	path_and_data - key is the full json filepath
					and value is the json data of the file
output:
"""
def WriteJsonFiles(path_and_data:dict):
    #write json data to files
    for key,val in path_and_data.items():
        with open(key,'w+') as json_file:
            json.dump(val,json_file,indent=4)

"""
desription: returns width and height of given bounding box, bounding box variable format is same as original dataset
input:
	bbx - the bounding box data
output:
    - float, width
    - float, height
"""
def GetBbxWH(bbx):
    start_end = bbx[POINTS][EXT]
    x1,y1 = start_end[0][0],start_end[0][1]
    x2,y2 = start_end[1][0],start_end[1][1]
    width, height = x2-x1, y2-y1
    return width, height

"""
desription: return area of given bounding box, bounding box variable format is same as original dataset
input:
	bbx - the bounding box data
output: float, the area of the givan bounding box
"""
def GetBbxArea(bbx):
    start_end = bbx[POINTS][EXT]
    x1,y1 = start_end[0][0],start_end[0][1]
    x2,y2 = start_end[1][0],start_end[1][1]
    delta_x, delta_y = x2-x1, y2-y1
    return delta_x*delta_y

