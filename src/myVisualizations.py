import cv2
import numpy as np
from flower_segmentation import*

def nothing(x):
    pass

def create_hsv_trackbar(test_img, scale = 1): # TODO HSV
    new_height = int(test_img.shape[0] * scale)
    new_width = int(test_img.shape[1] * scale)
    img = cv2.resize(test_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = img.copy()

    cv2.namedWindow('Mask_tuning')

    # create trackbars for color change
    cv2.createTrackbar('H','Mask_tuning',0,180,nothing)
    cv2.createTrackbar('SL','Mask_tuning',0,255,nothing)
    cv2.createTrackbar('SH', 'Mask_tuning', 0, 255, nothing)
    cv2.createTrackbar('VL','Mask_tuning',0,255,nothing)
    cv2.createTrackbar('VH', 'Mask_tuning', 0, 255, nothing)
    cv2.createTrackbar('d','Mask_tuning',0,50,nothing)

# create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'Mask_tuning',0,1,nothing)

    while(1):

        cv2.imshow('Mask_tuning',mask)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        h = cv2.getTrackbarPos('H','Mask_tuning')
        sl = cv2.getTrackbarPos('SL','Mask_tuning')
        sh = cv2.getTrackbarPos('SH','Mask_tuning')
        vl = cv2.getTrackbarPos('VL','Mask_tuning')
        vh = cv2.getTrackbarPos('VH', 'Mask_tuning')
        sw = cv2.getTrackbarPos(switch,'Mask_tuning')
        d = cv2.getTrackbarPos('s','Mask_tuning')

        if sw == 0:
            mask[:] = 0
        else:
            low = np.array([h - d, sl, vl], dtype=np.float32)
            upp = np.array([h + d, sh, vh], dtype=np.float32)
            mask = cv2.inRange(img, low, upp)
            #mask = cv2.erode(mask, kernel, iterations=it)

    cv2.destroyAllWindows()

def create_canny_slidebar(img, scale = 1):

    def kmeans(img, nb=3):
        output = img.copy()
        clust = output.reshape((-1, 3))
        clust = np.float32(clust)  # should be flattened and of type float32
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # max iter and accuracy epsilon
        K = nb
        ret, label, center = cv2.kmeans(clust, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        clust = res.reshape((output.shape))

        return clust

    def get_contour(img, thresh=100):
        contours, _ = cv2.findContours(img,)

    new_height = int(img.shape[0] * scale)
    new_width = int(img.shape[1] * scale)
    output = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    edge = np.zeros(output.shape, np.uint8)
    else_img = np.zeros_like(output)

    cv2.namedWindow('Canny threshold')

    # create trackbars for color change
    cv2.createTrackbar('L', 'Canny threshold', 0, 255, nothing)
    cv2.createTrackbar('H', 'Canny threshold', 0, 255, nothing)
    cv2.createTrackbar('k', 'Canny threshold', 2,30, nothing)
    cv2.createTrackbar('t', 'Canny threshold', 2,10, nothing)

    # create switch for ON/OFF functionality
    blur = '0 : OFF \n1 : ON'
    cv2.createTrackbar(blur, 'Canny threshold', 0, 1, nothing)

    while (1):
        img_ = np.vstack((edge, else_img))
        cv2.imshow('Canny threshold', img_)

        wk = cv2.waitKey(1) & 0xFF
        if wk == 27:
            break

        # get current positions of four trackbars
        l = cv2.getTrackbarPos('L', 'Canny threshold')
        h = cv2.getTrackbarPos('H', 'Canny threshold')
        b = cv2.getTrackbarPos(blur, 'Canny threshold')
        t = cv2.getTrackbarPos('t', 'Canny threshold')
        k = cv2.getTrackbarPos('k', 'Canny threshold')

        if b == 1 :
            else_img = cv2.blur(output, (k, k))
            edge = cv2.Canny(else_img, l, h)
        else:
            else_img = kmeans(output, nb=t)
            edge = cv2.Canny(else_img, l, h)




    cv2.destroyAllWindows()


#TODO bounding box track bar and mask centroid track bar from camera infos
if __name__ == "__main__":
    img = cv2.imread('../data_supp/strawb_test.jpg')
    #create_color_trackbar(img[c:d,a:b])
    #create_canny_slidebar(img[c:d,a:b])
    #create_hsv_trackbar(img, scale=0.5)
    detector = Object_Detection()
    detector.set_picture(img)
    detector.get_mask(sens=30, it=1, type='flower', scale=0.5, plot=True)