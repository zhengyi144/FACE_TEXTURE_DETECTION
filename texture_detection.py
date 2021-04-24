import cv2
import os
import numpy as np
from utils import *
import matplotlib.pyplot as plt

def convertColor(image,hsv=True,yuv=True,merge=True):
    image_hsv=np.zeros_like(image)
    image_yuv=np.zeros_like(image)

    if hsv:
        image_hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    if yuv:
        image_yuv=cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
    if merge:
        images=[]
        for i in range(3):
            images.append(image_hsv[:,:,i])
            images.append(image_yuv[:,:,i])
        return images
    return image_hsv,image_yuv

def saveHistDataset(datasetPath,outPath,name='RA_val_256_hist'):
    print(os.getcwd())
    txtFilePath=os.path.join(datasetPath,'cbnData/val.txt')
    savePath=os.path.join(outPath,'%s_dataset.txt'%name)
    print(savePath)
    f=open(savePath,'w+')
    fread=open(txtFilePath)
    while True:
        line=fread.readline()
        if not line:
            break
        line=line.split('\t')
        imagePath=datasetPath+line[0]
        label=line[-1].strip('\n') 
        src=cv2.imread(imagePath)
        if src is None:
            continue
        #resize
        image=cv2.resize(src,(64,64))
        hsv_yuv=convertColor(image)
        hists=[]
        #print(np.shape(hsv_yuv[0]))
        for i in range(len(hsv_yuv)):
            lbp=circularLBPOptimization(hsv_yuv[i],1,8)
            hist=calcHist(lbp)
            hists.append(hist)
        hists=np.reshape(hists,[6*256])
        line=map(str,hists)
        line=" ".join(line)
        f.write(str(line)+','+label+'\n')   
    f.close()
    fread.close()

def calCASIAHistDataset(datasetPath,outPath,name='train_256_hist'):
    fileList=os.listdir(datasetPath)
    savePath=os.path.join(outPath,'%s_dataset.txt'%name)
    print(savePath)
    f=open(savePath,'a+')
    for fileName in fileList:
        imagePath=os.path.join(datasetPath,fileName)
        label=fileName.split('.')[0].split('_')[-1]   #1 real,0 print attack,2 video attack
        src=cv2.imread(imagePath)
        if src is None or src.shape[0]<64 or src.shape[1]<64:
            continue
        #resize
        image=cv2.resize(src,(64,64))
        hsv_yuv=convertColor(image)
        hists=[]
        #print(np.shape(hsv_yuv[0]))
        for i in range(len(hsv_yuv)):
            #计算各个通道的LBP
            lbp=circularLBPOptimization(hsv_yuv[i],1,8)
            hist=calcHist(lbp)
            hists.append(hist)
        hists=np.reshape(hists,[6*256])
        line=map(str,hists)
        line=" ".join(line)
        f.write(str(line)+','+label+'\n')   
    f.close()  



def main():
    datasetPath='./dataset/CASIA_DB_IMAGES/test_faces'
    outPath='./dataset/CASIA_DB_IMAGES'
    calCASIAHistDataset(datasetPath,outPath,name='test_256_hist')

if __name__=="__main__":
    main()


