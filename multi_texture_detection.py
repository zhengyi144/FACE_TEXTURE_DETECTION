import cv2
import os
import numpy as np
from utils import *
import threading
from texture_detection import convertColor

class TaskThread(threading.Thread):
    def __init__(self,threadID,fileList,begin,end,writer):
        threading.Thread.__init__(self)
        self.threadID=threadID
        self.fileList=fileList
        self.begin=begin
        self.end=end
        self.writer=writer
    
    def run(self):
        print("start thread:"+str(self.threadID))
        for i in range(self.begin,self.end):
            imagePath=self.fileList[i]
            line=imageProcess(imagePath)
            global threadLock
            threadLock.acquire()
            saveFeatures(self.writer,line)
            threadLock.release()

def imageProcess(imagePath):
    src=cv2.imread(imagePath['imagePath'])
    image=cv2.resize(src,(64,64))
    hsv_yuv=convertColor(image)
    hists=[]
    #print(np.shape(hsv_yuv[0]))
    for i in range(len(hsv_yuv)):
        #计算各个通道的LBP
        lbp=uniformLBP(hsv_yuv[i],1,8)
        hist=calcHist(lbp,num=59)
        hists.append(hist)
    hists=np.reshape(hists,[6*59])
    line=map(str,hists)
    line=" ".join(line)
    line=str(line)+','+imagePath['label']+'\n'
    return line

def saveFeatures(f,line):
    f.write(line)

def calCASIAHistDataset():
    datasetPath='./dataset/CASIA_DB_IMAGES/test_faces'
    fileList=os.listdir(datasetPath)
    imagePathList=[]
    for fileName in fileList:
        imagePath=os.path.join(datasetPath,fileName)
        label=fileName.split('.')[0].split('_')[-1] 
        imagePathList.append({"imagePath":imagePath,"label":label})
    return imagePathList

def readReplayFiles():
    datasetPath="./dataset/REPLAY_ATTACK/anti_spoofing"
    txtFilePath=os.path.join(datasetPath,'cbnData/val.txt')
    fread=open(txtFilePath)
    imagePathList=[]
    while True:
        line=fread.readline()
        if not line:
            break
        line=line.split('\t')
        imagePath=datasetPath+line[0]
        label=line[-1].strip('\n')
        imagePathList.append({"imagePath":imagePath,"label":label})
    return imagePathList


if __name__=="__main__":
    imagePathList=readReplayFiles() #calCASIAHistDataset()
    outPath=  './dataset/REPLAY_ATTACK/anti_spoofing/cbnData'   #'./dataset/CASIA_DB_IMAGES'
    savePath=os.path.join(outPath,'RA_train_59.txt')
    f=open(savePath,'a+')
    threadLock=threading.Lock()
    num=len(imagePathList)
    N=10 if num/10000>10 else 5
    print(N)
    threadList=[]
    fregLen=int((num+N-1)/N)
    print(fregLen)
    for i in range(N):
        begin=i*fregLen
        end=(i+1)*fregLen if (i+1)*fregLen<num else num
        myThread=TaskThread(i,imagePathList,begin,end,f)
        myThread.start()
        threadList.append(myThread)

    for myThread in threadList:
        myThread.join()

    print("writer finish!")
    f.close()
    