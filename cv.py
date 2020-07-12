import cv2
import numpy as np
from matplotlib import pyplot as plt
def empty(a):
    pass

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


plt.rcParams['figure.figsize'] = 10, 10
plt.rcParams.update({'figure.max_open_warning': 5})
#0 24 43 255 0 255
#cv2.namedWindow("TrackBars")
#cv2.resizeWindow("TrackBars",640,640)
#cv2.createTrackbar("Hue Min","TrackBars",0,255,empty)
#cv2.createTrackbar("Hue Max","TrackBars",46,255,empty)
#cv2.createTrackbar("Sat Min","TrackBars",127,255,empty)
#cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
#cv2.createTrackbar("Val Min","TrackBars",58,255,empty)
#cv2.createTrackbar("Val Max","TrackBars",255,255,empty)
#cv2.createTrackbar("Hue Min1","TrackBars",110,255,empty)
#cv2.createTrackbar("Hue Max1","TrackBars",224,255,empty)
#cv2.createTrackbar("Sat Min1","TrackBars",64,255,empty)
#cv2.createTrackbar("Sat Max1","TrackBars",220,255,empty)
#cv2.createTrackbar("Val Min1","TrackBars",137,255,empty)
#cv2.createTrackbar("Val Max1","TrackBars",255,255,empty)

lower = np.array([0,127,58])
upper = np.array([46,255,255])
lower1 = np.array([110,64,137])
upper1 = np.array([224,220,255])
cap = cv2.VideoCapture('coner.mp4')
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
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_HSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_thresh_low = cv2.inRange(img_HSV, lower, upper) #всё что входит в "левый красный"
    img_thresh_high = cv2.inRange(img_HSV, lower1, upper1) #всё что входит в "правый красный"
    img_thresh = cv2.bitwise_or(img_thresh_low, img_thresh_high)
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
    imgConesOnly = cv2.bitwise_and(framer,framer,mask=img_edges)
    imgCanny = cv2.Canny(img_convex_hulls_3to10, 80, 160)
    contours,hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area>100:
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x, y, w, h = cv2.boundingRect(approx)
            mask = cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
    shot = cv2.add(anotherframe,mask)
    cv2.imshow('Result',shot)
    print(plt.get_fignums())
    imgContour =np.zeros_like(old_frame)
    mask=np.zeros_like(old_frame)
    #cv2.imshow('test1',img_thresh)
    #cv2.imshow('test2',img_convex_hulls_3to10)
    key = cv2.waitKey(1)
    if key == ord('p'):
        cv2.waitKey(-1)

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(0) 
