import cv2
import numpy as np
from matplotlib import pyplot as plt
def empty(a):
    pass




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

#img = cv2.imread(path)
#imgContour = img.copy()
#def getContours(img):
#    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#    for cnt in contours:
#        area = cv2.contourArea(cnt)
#        print(area)
#        if area>500:
#            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
#            peri = cv2.arcLength(cnt,True)
#            #print(peri)
#            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
#            print(len(approx))
#            objCor = len(approx)
#            x, y, w, h = cv2.boundingRect(approx)
#            if objCor ==3: objectType ="Tri"
#            elif objCor == 4:
#                aspRatio = w/float(h)
#                if aspRatio >0.98 and aspRatio <1.03: objectType= "Square"
#                else:objectType="Rectangle"
#            elif objCor>4: objectType= "Circles"
#            else:objectType="None"
#            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
#            global mask 
#            mask = cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            #cv2.putText(imgContour,objectType,(x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,0),2)
 






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
plt.rcParams.update({'figure.max_open_warning': 5})



#imgContour = imgCanny.copy()

#0 24 43 255 0 255
#cv2.namedWindow("TrackBars")
#cv2.resizeWindow("TrackBars",640,240)
#cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
#cv2.createTrackbar("Hue Max","TrackBars",24,179,empty)
#cv2.createTrackbar("Sat Min","TrackBars",43,255,empty)
#cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
#cv2.createTrackbar("Val Min","TrackBars",0,255,empty)
#cv2.createTrackbar("Val Max","TrackBars",255,255,empty)


cap = cv2.VideoCapture('cones2.mp4')
ret, old_frame = cap.read()
mask = np.zeros_like(old_frame)
imgContour =np.zeros_like(old_frame)
while(cap.isOpened()):
    ret, framer = cap.read()
    try:
        frame = framer.copy()
        anotherframe = framer.copy()
    except:
        break
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_HSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    #lower = np.array([0,68,5])
    #upper = np.array([179,255,255])
    #mask = cv2.inRange(imgHSV,lower,upper)
    #imgResult = cv2.bitwise_and(img,img,mask=mask)
    #img_HSV = imgResult
    #
    ##plt.imshow(img_HSV)
    ##plt.show()
    #

 

    #h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    #h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    #s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    #s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    #v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    #v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    #print(h_min,h_max,s_min,s_max,v_min,v_max)
    #lower = np.array([h_min,s_min,v_min])
    #upper = np.array([h_max,s_max,v_max])
    #mask = cv2.inRange(imgHSV,lower,upper)
    #imgResult = cv2.bitwise_and(img,img,mask=mask)


    lower = np.array([0,73,99])
    upper = np.array([46,255,255])
    img_thresh_low = cv2.inRange(img_HSV, np.array([0,73,99]), np.array([46,255,255])) #всё что входит в "левый красный"
    img_thresh_high = cv2.inRange(img_HSV, np.array([0, 75, 137]), np.array([184, 255, 255])) #всё что входит в "правый красный"

    img_thresh = cv2.bitwise_or(img_thresh_low, img_thresh_high)
    #
    ##plt.imshow(img_thresh)
    ##plt.show()
    #
    #f, axarr = plt.subplots(nrows=1, ncols=2)

    #axarr[0].imshow(img_thresh_low)
    #axarr[1].imshow(img_thresh_high)
    
#0 10 43 255 71 255


    


    kernel = np.ones((5, 5))
    img_thresh_opened = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 5)
    img_edges = cv2.Canny(img_thresh_blurred, 80, 160)
    contours, _= cv2.findContours(np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros_like(img_edges)
    cv2.drawContours(img_contours, contours, -1, (255,255,255), 2)
    approx_contours = []
    for c in contours:
        approx = cv2.approxPolyDP(c, 10, closed = True)
        approx_contours.append(approx)
    img_approx_contours = np.zeros_like(img_edges)
    cv2.drawContours(img_approx_contours, approx_contours, -1, (255,255,255), 1)
    all_convex_hulls = []
    for ac in approx_contours:
        all_convex_hulls.append(cv2.convexHull(ac))
    img_all_convex_hulls = np.zeros_like(img_edges)
    cv2.drawContours(img_all_convex_hulls, all_convex_hulls, -1, (255,255,255), 2)
    convex_hulls_3to10 = []
    for ch in all_convex_hulls:
        if 3 <= len(ch) <= 10:
            convex_hulls_3to10.append(cv2.convexHull(ch))

    img_convex_hulls_3to10 = np.zeros_like(img_edges)
    cv2.drawContours(img_convex_hulls_3to10, convex_hulls_3to10, -1, (255,255,255), 2)
    approx_contours = []
    contours, _= cv2.findContours(np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        approx = cv2.approxPolyDP(c, 10, closed = True)
        approx_contours.append(approx)
    img_all_convex_hulls = np.zeros_like(img_edges)
    cv2.drawContours(img_all_convex_hulls, all_convex_hulls, -1, (255,255,255), 2)
    convex_hulls_3to10 = []
    for ch in all_convex_hulls:
        if 3 <= len(ch) <= 10:
            convex_hulls_3to10.append(cv2.convexHull(ch))

    img_convex_hulls_3to10 = np.zeros_like(img_edges)
    cv2.drawContours(img_convex_hulls_3to10, convex_hulls_3to10, -1, (255,255,255), 2)
    def convex_hull_pointing_up(ch):
        #Определяет, направлен ли контур наверх.
        #Если да, то это конус
        
        # точки контура выше центра и ниже 
        points_above_center, points_below_center = [], []
    
        x, y, w, h = cv2.boundingRect(ch) # координаты левого верхнего угла описывающего прямоугольника, ширина и высота
        aspect_ratio = w / h # отношение ширины прямоугольника к высоте

        # если прямоугольник узкий, продолжаем определение. Если нет, то контур не подходит
        if aspect_ratio < 0.8:
            # каждую точку контура классифицируем как лежащую выше или ниже центра
            vertical_center = y + h / 2

            for point in ch:
                if point[0][1] < vertical_center: # если координата y точки выше центра, то добавляем эту точку в список точек выше центра
                    points_above_center.append(point)
                elif point[0][1] >= vertical_center:
                    points_below_center.append(point)

            # определяем координаты x крайних точек, лежащих ниже центра
            left_x = points_below_center[0][0][0]
            right_x = points_below_center[0][0][0]
            for point in points_below_center:
                if point[0][0] < left_x:
                    left_x = point[0][0]
                if point[0][0] > right_x:
                    right_x = point[0][0]

            # проверяем, лежат ли верхние точки контура вне "основания". Если да, то контур не подходит
            for point in points_above_center:
                if (point[0][0] < left_x) or (point[0][0] > right_x):
                    return False
        else:
            return False
        
        return True

    # определяем, является ли контур конусом. Если да, то сохраняем и строим для него описывающий прямоугольник
    cones = []
    bounding_rects = []
    for ch in convex_hulls_3to10:
        if convex_hull_pointing_up(ch):
            cones.append(ch)
            rect = cv2.boundingRect(ch)
            bounding_rects.append(rect)

    img_cones = np.zeros_like(img_edges)
    cv2.drawContours(img_cones, cones, -1, (255,255,255), 2)
    for rect in bounding_rects:
        cv2.rectangle(anotherframe, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 3)
    #cv2.imshow('anotherframe',anotherframe)


    #masker = cv2.inRange(img_edges,lower,upper)
    imgConesOnly = cv2.bitwise_and(framer,framer,mask=img_edges)
    imgCanny = cv2.Canny(img_convex_hulls_3to10, 80, 160)
    #imgCanny = cv2.Canny(masker, 80, 160)


    contours,hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area>100:
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x, y, w, h = cv2.boundingRect(approx)
            mask = cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
    
    shot = cv2.add(anotherframe,mask)
    cv2.imshow('frame',shot)
    print(plt.get_fignums())
    imgContour =np.zeros_like(old_frame)
    mask=np.zeros_like(old_frame)

    #cv2.imshow('test',imgConesOnly)
    key = cv2.waitKey(1)
    if key == ord('p'):
        cv2.waitKey(-1)
    #plt.cla()
    #cv2.imshow('frame',imgCanny)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#while(1):
    #ret,frame = cap.read()
    #framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #imger = cv2.add(frame,masking)
    #cv2.imshow('frame',imger)


    #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #img = cv.add(frame,mask)
    #img = cv2.add(frame)
    #ret, old_frame = cap.read()


#getContours(imgCanny)
#cv2.imshow('result',imgContour)






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
 
#imgBlank = np.zeros_like(img)
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