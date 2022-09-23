from consts import *

def main():
    #creating dataset folders
    for phase in DATASET_SPLIT_RATIO.keys():
        os.makedirs(DST_PATH+phase+'/'+IMAGE_SUB_FOLDER, exist_ok=True)
        os.makedirs(DST_PATH+phase+'/'+LABELS_SUB_FOLDER, exist_ok=True)

    #getting sub folder data
    folders, n_folders = GetSubFolders(SRC_PATH)
    n_folder = 1

    for src_sub_path in folders: #iterating through sub folders
        logger.info("{0}/{1} folders".format(n_folder,n_folders))
        n_folder+=1

        #image and lable sorce folders
        src_images = src_sub_path+IMAGE_SUB_FOLDER
        src_labels = src_sub_path+LABELS_SUB_FOLDER

        #loading image list and shuffeling it for random split
        files, nof = GetFilesInDir(src_images)

        ten_per = nof//10
        for img_idx, original_filename in enumerate(files): #iterating through images in sub folder
            #print progress
            if img_idx%ten_per==0:
                logger.info("{0}/{1}".format(img_idx,nof))

            #get current phase in split
            phase = GetCurrentPhase(nof, img_idx)
            dst_images = os.path.join(DST_PATH+phase,IMAGE_SUB_FOLDER)
            dst_labels = os.path.join(DST_PATH+phase,LABELS_SUB_FOLDER)

            #save image in new location with correct format
            filename = CopyImage(src_images, dst_images, original_filename)

            #load image data json file
            json_file_path = os.path.join(src_labels, original_filename+".json")
            img_json_data = GetDataFromJson(json_file_path)

            #extract image data from json
            img_h, img_w = GetImgHWFromJson(img_json_data)

            #extracting bounding box data from json file
            #creating a list of lines to write to the text file
            all_lines = []
            for bbx in img_json_data[OBJECTS]:
                line = CreateBoundingBoxLineByYOLOFormat(bbx, img_h, img_w)
                all_lines.append(line)

            #save labels to text file
            text_f = os.path.join(dst_labels, filename+".txt")
            with open(text_f,"w+") as text_file:
                text_file.write('\n'.join(all_lines))

if __name__ == '__main__':
    main()