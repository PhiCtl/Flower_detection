from myVisualizations import*
import os

def get_mask(img, sens=10, it=2, scale=1, plot=False):  # OK

    """Finds objects in frame
    args: sensitivity of color range
          number of iterations to erode the mask
          scaling of picture
          plot resulting mask"""

    if plot: print("Getting mask...")
    # Smoothen
    output = img.copy()
    output = cv2.medianBlur(output, 5)

    # Resize
    # new_height = int(output.shape[0]*scale)
    # new_width  = int(output.shape[1]*scale)
    # output = cv2.resize(output, (new_width, new_height),interpolation = cv2.INTER_AREA)

    # Define HSV orange color range for thresholding TUNED :)
    low_orange = np.array([16.8 / 2 - sens, 0.5 * 255, 0.5 * 255], dtype=np.float32)
    upp_orange = np.array([16.8 / 2 + sens, 255, 255], dtype=np.float32)

    # Opening to get rid of the small background artifacts -> #TODO : tune size of opening element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # From bgr to hsv colorspace
    hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)

    # Threshold and erode
    mask = cv2.inRange(hsv, low_orange, upp_orange)
    mask = cv2.erode(mask, kernel, iterations=it)


    if plot:  # TODO correct segfault in plot mode
        # Bitwise and mask and original picture
        res = cv2.bitwise_and(output, output, mask=mask)
        cv2.imshow('result', res)
        cv2.imshow('mask HSV', mask)
        cv2.imshow('img', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Closing windows...")
    if plot: print("Done")

def run():

    img = cv2.imread("../data_train/20210413_162558.jpg")
    print(os.getcwd())
    create_color_trackbar(img)


if __name__ == "__main__":
    run()