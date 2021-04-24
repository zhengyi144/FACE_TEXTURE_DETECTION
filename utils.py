import numpy as np
import math

'''
  https://blog.csdn.net/quincuntial/article/details/50541815
'''
def originLBP(src):
    dst = np.zeros(src.shape,dtype=src.dtype)
    for i in range(1,src.shape[0]-1):
        for j in range(1,src.shape[1]-1):
            center = src[i][j]
            code = 0  
            code |= (src[i-1][j-1] >= center) << 7  
            code |= (src[i-1][j  ] >= center) << 6  
            code |= (src[i-1][j+1] >= center) << 5  
            code |= (src[i  ][j+1] >= center) << 4  
            code |= (src[i+1][j+1] >= center) << 3 
            code |= (src[i+1][j  ] >= center) << 2  
            code |= (src[i+1][j-1] >= center) << 1  
            code |= (src[i  ][j-1] >= center) << 0  

            dst[i-1][j-1]= code  
    return dst

def rotateInvariant(src,radius,n_points):
    height,width=src.shape
    dst=np.zeros((height,width),dtype=src.dtype)
    for r in range(height):
        for c in range(width):
            currentVal=src[r,c]
            minVal=currentVal
            for k in range(n_points):
                temp=(currentVal>>(n_points-k))|(currentVal<<k)
                if(temp<minVal):
                    minVal=temp
            dst[r,c]=minVal
    return dst

def circularLBP(src,radius,n_points):
    height=src.shape[0]
    width=src.shape[1]
    dst=src.copy()
    src.astype(dtype=np.float32)
    dst.astype(dtype=np.float32)
    neighbors=np.zeros((1,n_points),dtype=np.int32)
    lbp_value=np.zeros((1,n_points),dtype=np.uint8)
    for x in range(radius,width-radius-1):
        for y in range(radius,height-radius-1):
            lbp=0
            for n in range(n_points):
                theta=float(2*np.pi*n)/n_points
                x_n=x+radius*np.cos(theta)
                y_n=y-radius*np.sin(theta)
                
                x1=int(math.floor(x_n))
                y1=int(math.floor(y_n))
                
                x2=int(math.ceil(x_n))
                y2=int(math.ceil(y_n))
                
                tx = np.abs(x_n - x1)
                ty = np.abs(y_n - y1)
                
                w1 = (1 - tx) * (1 - ty)
                w2 = tx * (1 - ty)
                w3 = (1 - tx) * ty
                w4 = tx * ty

                neighbor = src[y1, x1] * w1 + src[y2, x1] * w2 + src[y1, x2] * w3 + src[y2, x2] * w4

                #neighbor=src[y1,x1]*(x2-x_n)*(y2-y_n)+src[y2,x1]*(x_n-x1)*(y2-y_n)+src[y1,x2]*(x2-x_n)*(y_n-y1)+src[y2,x2]*(x_n-x1)*(y_n-y1)
                neighbors[0,n]=neighbor
                
            center=src[y,x]
            for n in range(n_points):
                if neighbors[0,n]>center:
                    lbp_value[0,n]=1
                else:
                    lbp_value[0,n]=0 
                lbp+=lbp_value[0,n]*2**n
            
            dst[y,x]=int(lbp/(2**n_points-1)*255)       
    return dst
            

def circularLBPOptimization(src,radius,n_points,_rotateInvariant=False):
    height,width=src.shape
    dst=np.zeros((height,width),dtype=np.int32)

    for k in range(n_points):
        theta=float(2*np.pi*k)/n_points
        rx=radius*np.cos(theta)
        ry=-radius*np.sin(theta)

        x1=int(math.floor(rx))
        y1=int(math.floor(ry))
        
        x2=int(math.ceil(rx))
        y2=int(math.ceil(ry))
        
        tx = np.abs(rx - x1)
        ty = np.abs(ry - y1)
        #print(tx,ty)
    
        w1 = (1 - tx) * (1 - ty)
        w2 = tx * (1 - ty)
        w3 = (1 - tx) * ty
        w4 = tx * ty

        for i in range(radius,height-radius):
            for j in range(radius,width-radius):
                center=src[i,j]
                neighbor=src[i+y1,j+x1]*w1+src[i+y2,j+x1]*w2+src[i+y1,j+x2]*w3+src[i+y2,j+x2]*w4
                #print(neighbor)
                dst[i-radius,j-radius]|=(neighbor>center)<<(n_points-k-1)
    
    if _rotateInvariant:
        dst=rotateInvariant(dst,radius,n_points)
    return dst

def uniformLBP(src,radius,n_points,_rotateInvariant=False):
    height,width=src.shape
    dst=circularLBPOptimization(src,radius,n_points)
    valueMap=uniformMapping()
    for i in range(radius,height-radius):
        for j in range(radius,width-radius):
            dst[i-radius,j-radius]=valueMap[dst[i-radius,j-radius]]
    return dst

def ColALBP(image,lbp_r=1,co_r=2,normalize=False):
    '''
    Co-occurrence of Adjacent Local Binary Patterns algorithm 
    Input image with shape (height, width, channels)
    Input lbp_r is radius for adjacent local binary patterns
    Input co_r is radius for co-occurence of the patterns
    Output features with length 1024 * number of channels
    '''
    h, w, c = image.shape
    # normalize face
    image = (image - np.mean(image, axis=(0,1))) / (np.std(image, axis=(0,1)) + 1e-8)
    # albp and co-occurrence per channel in image
    histogram = np.empty(0, dtype=np.int)
    for i in range(image.shape[2]):
        C = image[lbp_r:h-lbp_r, lbp_r:w-lbp_r, i].astype(float)
        X = np.zeros((4, h-2*lbp_r, w-2*lbp_r))
        # adjacent local binary patterns
        X[0, :, :] = image[lbp_r  :h-lbp_r  , lbp_r+lbp_r:w-lbp_r+lbp_r, i] - C
        X[1, :, :] = image[lbp_r-lbp_r:h-lbp_r-lbp_r, lbp_r  :w-lbp_r  , i] - C
        X[2, :, :] = image[lbp_r  :h-lbp_r  , lbp_r-lbp_r:w-lbp_r-lbp_r, i] - C
        X[3, :, :] = image[lbp_r+lbp_r:h-lbp_r+lbp_r, lbp_r  :w-lbp_r  , i] - C
        X = (X>0).reshape(4, -1)#0,1
        # co-occurrence of the patterns
        A = np.dot(np.array([1, 2, 4, 8]), X) #（0-15）
        A = A.reshape(h-2*lbp_r, w-2*lbp_r) + 1
        hh, ww = A.shape
        
        D  = (A[co_r  :hh-co_r  , co_r  :ww-co_r  ] - 1) * 16 - 1   
        Y1 =  A[co_r  :hh-co_r,   co_r+co_r:ww-co_r+co_r] + D
        Y2 =  A[co_r-co_r:hh-co_r-co_r, co_r+co_r:ww-co_r+co_r] + D
        Y3 =  A[co_r-co_r:hh-co_r-co_r, co_r  :ww-co_r  ] + D
        Y4 =  A[co_r-co_r:hh-co_r-co_r, co_r-co_r:ww-co_r-co_r] + D
        
        #histogram 256
        Y1 = np.bincount(Y1.ravel(), minlength=256)
        Y2 = np.bincount(Y2.ravel(), minlength=256)
        Y3 = np.bincount(Y3.ravel(), minlength=256)
        Y4 = np.bincount(Y4.ravel(), minlength=256)
        pattern = np.concatenate((Y1, Y2, Y3, Y4))
        histogram = np.concatenate((histogram, pattern))
    # normalize the histogram and return it
    if normalize:
        features = (histogram - np.mean(histogram)) / np.std(histogram)
    else:
        features=histogram
    return features

def getHopCount(value):
    temp=np.zeros(8,dtype=np.int)
    n=7
    count=0
    while value:
        temp[n]=value&1
        value>>=1
        n-=1
    for i in range(8):
        if temp[i] !=temp[(i+1)%8]:
            count+=1
    return count

def uniformMapping():
    temp=1
    valueMap=np.zeros(256,np.int)
    for i in range(256):
        if getHopCount(i)<3:
            valueMap[i]=temp
            temp+=1
    return valueMap

def calcHist(src,num=256):
    hist=np.bincount(src.ravel(), minlength=num)
    return hist

