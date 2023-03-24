import sys
import os
import numpy as np
import seaborn as sns
sys.path.append(os.path.abspath('../../'))
import matplotlib.pyplot as plt
import common.functions as cf
from loguru import logger
from common.consts import IMAGES_SUB_FOLDER, LABELS_SUB_FOLDER, \
                        SRC_PATH, OBJECTS, CLASS_TITLE, ID
from common.functions import GetBbxArea, GetImgHWFromJson

def main():
    #getting sub folder data
    folders, _ = cf.GetSubFolders(SRC_PATH)
    n_folder = 1

    class_type_cnt = {}
    bbx_data = []
    for src_sub_path in folders: #iterating through sub folders
        n_folder+=1

        #image and lable sorce folders
        src_images = src_sub_path+IMAGES_SUB_FOLDER
        src_labels = src_sub_path+LABELS_SUB_FOLDER

        #loading image list and shuffeling it for random split
        files, _ = cf.GetFilesInDir(src_images)

        for original_filename in files: #iterating through images in sub folder

            #load image data json file
            json_file_path = os.path.join(src_labels, original_filename+".json")
            img_json_data = cf.GetDataFromJson(json_file_path)

            for bbx in img_json_data[OBJECTS]:
                if bbx[CLASS_TITLE] not in class_type_cnt.keys():
                    class_type_cnt[bbx[CLASS_TITLE]] = 1
                else:
                    class_type_cnt[bbx[CLASS_TITLE]]+=1
                
                current_bbx_data = {}
                current_bbx_data["fie_path"] = json_file_path
                bbx_area = GetBbxArea(bbx)
                w,h = GetImgHWFromJson(img_json_data)
                current_bbx_data["rel_area"] = bbx_area/(w*h)
                current_bbx_data[ID] = bbx[ID]
                bbx_data.append(current_bbx_data)

    print("Class Distrebution before clean: ")

    print(class_type_cnt)# create a barplot using Seaborn
    sns.set(style="whitegrid")
    ax = sns.barplot(x=list(class_type_cnt.keys()), y=list(class_type_cnt.values()))

    # set the labels and title
    ax.set(xlabel='Class Names', ylabel='Number of Objects', title='Class Object Count')

    # display the graph
    plt.show()
    min_cass_am = class_type_cnt.keys()[np.argmin(class_type_cnt.values())]
    print(min_cass_am)

    
    print("Class Distrebution after clean: ")
    

if __name__ == '__main__':
    main()