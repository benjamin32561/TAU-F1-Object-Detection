import os

ROOTDIR ='C:\\Users\\ben32\\Desktop\\TAU-F1-Object-Detection'
FILENAME = "amz_01376"

def main():
    for subdir, dirs, files in os.walk(ROOTDIR):
        for file in files:
            if FILENAME in file:
                print(subdir)
                print(file)
                break

if __name__ == '__main__':
    main()