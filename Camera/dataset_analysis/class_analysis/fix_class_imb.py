import sys
import os
import seaborn as sns
sys.path.append(os.path.abspath('../../'))
import matplotlib.pyplot as plt
import common.functions as cf
from loguru import logger
from common.consts import IMAGES_SUB_FOLDER, LABELS_SUB_FOLDER, \
                        SRC_PATH, OBJECTS, CLASS_TITLE

def main():
    #getting sub folder data
    folders, n_folders = cf.GetSubFolders(SRC_PATH)
    n_folder = 1

    class_type_cnt = {}
    for src_sub_path in folders: #iterating through sub folders
        n_folder+=1

        #image and lable sorce folders
        src_images = src_sub_path+IMAGES_SUB_FOLDER
        src_labels = src_sub_path+LABELS_SUB_FOLDER

        #loading image list and shuffeling it for random split
        files, nof = cf.GetFilesInDir(src_images)

        ten_per = nof//10
        for img_idx, original_filename in enumerate(files): #iterating through images in sub folder

            #load image data json file
            json_file_path = os.path.join(src_labels, original_filename+".json")
            img_json_data = cf.GetDataFromJson(json_file_path)

            for bbx in img_json_data[OBJECTS]:
                if bbx[CLASS_TITLE] not in class_type_cnt.keys():
                    class_type_cnt[bbx[CLASS_TITLE]] = 1
                else:
                    class_type_cnt[bbx[CLASS_TITLE]]+=1
    print("Class Distrebution before clean: ")

    print(class_type_cnt)
    sns.displot(y=class_type_cnt.values(), x=class_type_cnt.keys())

    
    print("Class Distrebution after clean: ")
    

if __name__ == '__main__':
    main()