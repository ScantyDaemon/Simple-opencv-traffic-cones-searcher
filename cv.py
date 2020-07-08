import cv2
import numpy as np
from matplotlib import pyplot as plt
def empty(a):
    pass

cap = cv2.VideoCapture('cones2.mp4')


path = 'cones2.jpg'

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

img = cv2.imread(path)
imgContour = img.copy()
def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area>500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if objCor ==3: objectType ="Tri"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio >0.98 and aspRatio <1.03: objectType= "Square"
                else:objectType="Rectangle"
            elif objCor>4: objectType= "Circles"
            else:objectType="None"
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imgContour,objectType,
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                        (0,0,0),2)
 






#cv2.namedWindow("TrackBars")
#cv2.resizeWindow("TrackBars",640,240)
#cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
#cv2.createTrackbar("Hue Max","TrackBars",19,179,empty)
#cv2.createTrackbar("Sat Min","TrackBars",110,255,empty)
#cv2.createTrackbar("Sat Max","TrackBars",240,255,empty)
#cv2.createTrackbar("Val Min","TrackBars",153,255,empty)
#cv2.createTrackbar("Val Max","TrackBars",255,255,empty)
 
#
plt.rcParams['figure.figsize'] = 10, 10


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_HSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
#lower = np.array([0,68,5])
#upper = np.array([179,255,255])
#mask = cv2.inRange(imgHSV,lower,upper)
#imgResult = cv2.bitwise_and(img,img,mask=mask)
#img_HSV = imgResult
plt.imshow(img_HSV)
plt.show()
img_thresh_low = cv2.inRange(img_HSV, np.array([0, 68, 5]), np.array([15, 255, 255])) #всё что входит в "левый красный"
img_thresh_high = cv2.inRange(img_HSV, np.array([159, 135, 135]), np.array([255, 255, 255])) #всё что входит в "правый красный"

img_thresh = cv2.bitwise_or(img_thresh_low, img_thresh_high)
plt.imshow(img_thresh)
plt.show()
f, axarr = plt.subplots(nrows=1, ncols=2)
axarr[0].imshow(img_thresh_low)
axarr[1].imshow(img_thresh_high)

kernel = np.ones((5, 5))
img_thresh_opened = cv2.morphologyEx(img_thresh_low, cv2.MORPH_OPEN, kernel)
img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 5)
imgCanny = cv2.Canny(img_thresh_blurred, 80, 160)

#imgContour = imgCanny.copy()



# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

getContours(imgCanny)
cv2.imshow('result',imgContour)







#imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
#h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
#s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
#s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
#v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
#v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
#print(h_min,h_max,s_min,s_max,v_min,v_max)
            #4 12 125 255 94 255

 
 
    # cv2.imshow("Original",img)
    # cv2.imshow("HSV",imgHSV)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Result", imgResult)
 
#while True:
#img = imgResult

#    cv2.waitKey(1)


##############


 
#imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)

#imgCanny = cv2.Canny(imgBlur,50,50)
#getContours(imgCanny)
 
imgBlank = np.zeros_like(img)
#imgStack = stackImager(0.8,([imgclear],[imgContour]))





    # calculate optical flow
    #p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    #good_new = p1[st==1]
    #good_old = p0[st==1]
    # draw the tracks
    #for i,(new,old) in enumerate(zip(good_new, good_old)):
    #    a,b = new.ravel()
    #    c,d = old.ravel()
    #    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #    frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    #getContours(frame_gray)
    #img = cv2.add(frame,mask)
    #cv2.imshow('frame',frame)
    # Now update the previous frame and previous points
    #old_gray = frame_gray.copy()
    #p0 = good_new.reshape(-1,1,2)







cv2.waitKey(0) 
#####################