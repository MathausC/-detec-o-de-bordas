import sys
import cv2 as cv
import numpy as np

def main(argv):

    scale = 1
    delta = 0
    ddepth = cv.CV_16S
    
    
    if len(argv) < 1:
        return -1

    src = cv.imread(argv[0], cv.IMREAD_COLOR)

    if src is None:
        print ('Error opening image: ' + argv[0])
        return -1

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    
    
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    
    
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    
    
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)    
    
    cv.imshow('Sobel', grad)
    cv.waitKey(0)

    verticalKernel = np.matrix([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    horizontalKernel = verticalKernel.transpose()

    grad2_x = cv.filter2D(gray, ddepth, horizontalKernel)
    grad2_y = cv.filter2D(gray, ddepth, verticalKernel)

    abs_grad2_x = cv.convertScaleAbs(grad2_x)
    abs_grad2_y = cv.convertScaleAbs(grad2_y)

    grad2 = cv.addWeighted(abs_grad2_x, 0.5, abs_grad2_y, 0.5, 0)

    cv.imshow('Prewitt', grad2)
    cv.waitKey(0)
    
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])