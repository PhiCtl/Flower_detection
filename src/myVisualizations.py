import cv2
import numpy as np

def nothing(x):
    pass

def create_color_trackbar(test_img): # TODO HSV
    test = cv2.resize(test_img, (int(test_img.shape[1]/5), int(test_img.shape[0]/5)))
    img = np.zeros(test.shape, np.uint8)
    cv2.namedWindow('Color_tuning')

    # create trackbars for color change
    cv2.createTrackbar('R','Color_tuning',0,255,nothing)
    cv2.createTrackbar('G','Color_tuning',0,255,nothing)
    cv2.createTrackbar('B','Color_tuning',0,255,nothing)

# create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image',0,1,nothing)

    while(1):
        img_ = np.hstack((img, test))
        cv2.imshow('Color tuning',img_)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R','Color_tuning')
        g = cv2.getTrackbarPos('G','Color_tuning')
        b = cv2.getTrackbarPos('B','Color_tuning')
        s = cv2.getTrackbarPos(switch,'Color_tuning')

        if s == 0:
            img[:] = 0
        else:
            img[:] = [b,g,r]

    cv2.destroyAllWindows()

#TODO bounding box track bar and mask centroid track bar from camera infos