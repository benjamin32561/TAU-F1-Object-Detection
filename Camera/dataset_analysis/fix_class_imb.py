import sys
import os
import numpy as np
import seaborn as sns
sys.path.append(os.path.abspath('../'))
import matplotlib.pyplot as plt
import common.functions as cf
from loguru import logger
import argparse
from common.consts import IMAGES_SUB_FOLDER, LABELS_SUB_FOLDER, \
                        SRC_PATH, OBJECTS, CLASS_TITLE, ID
from common.functions import GetBbxArea, GetImgHWFromJson, WriteJsonFiles

def main():
    parser = argparse.ArgumentParser(description='My script')

    # add a --save_at argument to the parser
    parser.add_argument('--save_at', type=str, default='/content/', help='folder path to save graphs at')

    # parse the command-line arguments
    args = parser.parse_args()

    #getting sub folder data
    folders, _ = cf.GetSubFolders(SRC_PATH)
    n_folder = 1

    class_type_cnt = {}
    bbx_data = {}
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
                    bbx_data[bbx[CLASS_TITLE]] = []
                else:
                    class_type_cnt[bbx[CLASS_TITLE]]+=1
                
                current_bbx_data = {}
                current_bbx_data["file_path"] = json_file_path
                bbx_area = GetBbxArea(bbx)
                current_bbx_data["rel_area"] = bbx_area
                current_bbx_data[ID] = bbx[ID]
                bbx_data[bbx[CLASS_TITLE]].append(current_bbx_data)

    print("Class Distrebution before clean: ")

    print(class_type_cnt)# create a barplot using Seaborn
    sns.set(style="whitegrid")
    ax = sns.barplot(x=list(class_type_cnt.keys()), y=list(class_type_cnt.values()))
    # set the labels and title
    ax.set(xlabel='Class Names', ylabel='Number of Objects', title='Class Object Count')
    # display the graph
    plt.savefig(os.path.join(args.save_at, "Class Dist Before Imb Fix.png"))

    min_class = list(class_type_cnt.keys())[np.argmin(list(class_type_cnt.values()))]
    min_class_am = class_type_cnt[min_class]
    print(min_class_am)
    final = {}
    for key in class_type_cnt.keys():
        final[key] = min_class_am

    #fixing class imb by deleting n smallest objects from each class
    del bbx_data[min_class] #removing min class
    file_path_id = {}
    for key in bbx_data.keys(): #sorting each array and saving bbx id to remove by file_path
        #sorting
        bbx_data_to_rem = sorted(bbx_data[key], key=lambda x: x["rel_area"])[:-min_class_am]
        for bbx in bbx_data_to_rem: #saving bbx to del by json file path
            file_path = bbx["file_path"]
            if file_path not in file_path_id.keys():
                file_path_id[file_path] = []
            file_path_id[file_path].append(bbx[ID])
    for file_path in file_path_id.keys(): #deleting bbx from files
        img_json_data = cf.GetDataFromJson(file_path)
        old_img_bbx = img_json_data[OBJECTS]
        bbx_id_to_rem = file_path_id[file_path]
        img_json_data[OBJECTS] = [d for d in old_img_bbx if d[ID] not in bbx_id_to_rem]
        WriteJsonFiles({file_path:img_json_data})
    
    print("\nClass Distrebution after clean: ")

    print(final)
    # create a barplot using Seaborn
    plt.clf()
    sns.set(style="whitegrid")
    ax = sns.barplot(x=list(final.keys()), y=list(final.values()))
    # set the labels and title
    ax.set(xlabel='Class Names', ylabel='Number of Objects', title='Class Object Count')
    # display the graph
    plt.savefig(os.path.join(args.save_at, "Class Dist After Imb Fix.png"))

if __name__ == '__main__':
    main()