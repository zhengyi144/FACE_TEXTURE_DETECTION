import os
import numpy as np
import dlib
import cv2

def saveFaceImage(filePath,savePath):
    fileList=os.listdir(filePath)
    for fileName in fileList:
        imagePath=os.path.join(filePath,fileName)
        image=cv2.imread(imagePath)
        rect=detectFace(image)
        if rect is not None:
            faceImage=image[rect[0][1]:rect[1][1],rect[0][0]:rect[1][0],:]
            saveImagePath=os.path.join(savePath,fileName)
            cv2.imwrite(saveImagePath,faceImage)


def detectFace(image,num=1):
    detector=dlib.get_frontal_face_detector()
    rects=detector(image,num)
    if len(rects)==0:
        return 
    rect=[(rects[0].left(), rects[0].top()), (rects[0].right(), rects[0].bottom())]
    return rect

def main():
    filePath=r'\\ai\zhyi\dataset\CASIA_DB_IMAGES\train'
    savePath=r'\\ai\zhyi\dataset\CASIA_DB_IMAGES\train_faces'
    saveFaceImage(filePath,savePath)


if __name__ == "__main__":
    main()
