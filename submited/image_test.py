import numpy as np
import cv2
import glob

test_images_names="test_images/*.jpg"
'''
definition of a tool to check the HSV filters
'''

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(gray, sobel_kernel=3, thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    abs_sobel = np.sqrt(np.add(np.multiply(abs_sobelx, abs_sobelx), np.multiply(abs_sobely, abs_sobely)))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return mag_binary

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    abs_sobel = np.arctan2(abs_sobely, abs_sobelx)
    # scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    dir_binary = np.zeros_like(abs_sobel)
    dir_binary[(abs_sobel >= thresh[0]) & (abs_sobel <= thresh[1])] = 1
    return dir_binary

# Create a window
cv2.namedWindow('image')
cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
images = glob.glob(test_images_names)
nbr_images = len(images)

def nothing(x):
    pass

# Create trackbars
cv2.createTrackbar('Test_images', 'Trackbars', 0, nbr_images-1,nothing)
cv2.createTrackbar('HSV ok', 'Trackbars', 0, 1,nothing)
cv2.createTrackbar('H-', 'Trackbars', 0, 180,nothing)
cv2.createTrackbar('H+', 'Trackbars', 0, 180,nothing)
cv2.createTrackbar('S-', 'Trackbars', 0, 255,nothing)
cv2.createTrackbar('S+', 'Trackbars', 0, 255,nothing)
cv2.createTrackbar('V-', 'Trackbars', 0, 255,nothing)
cv2.createTrackbar('V+', 'Trackbars', 0, 255,nothing)
cv2.createTrackbar('Thres ok', 'Trackbars', 0, 1,nothing)
cv2.createTrackbar('kernel (2n+3)', 'Trackbars', 0, 2,nothing)
cv2.createTrackbar('Sx-', 'Trackbars', 0, 255,nothing)
cv2.createTrackbar('Sx+', 'Trackbars', 0, 255,nothing)
cv2.createTrackbar('Sy-', 'Trackbars', 0, 255,nothing)
cv2.createTrackbar('Sy+', 'Trackbars', 0, 255,nothing)
cv2.createTrackbar('Sobel-', 'Trackbars', 0, 255,nothing)
cv2.createTrackbar('Sobel+', 'Trackbars', 0, 255,nothing)
cv2.createTrackbar('dir-', 'Trackbars', 0, 90,nothing)
cv2.createTrackbar('dir+', 'Trackbars', 0, 90,nothing)
# initialise trackbar position
cv2.setTrackbarPos('Test_images', 'Trackbars', 0)
cv2.setTrackbarPos('HSV ok', 'Trackbars', 0)
cv2.setTrackbarPos('H-', 'Trackbars', 0)
cv2.setTrackbarPos('H+', 'Trackbars', 180)
cv2.setTrackbarPos('S-', 'Trackbars', 0)
cv2.setTrackbarPos('S+', 'Trackbars', 255)
cv2.setTrackbarPos('V-', 'Trackbars', 0)
cv2.setTrackbarPos('V+', 'Trackbars',  255)
cv2.setTrackbarPos('Thres ok', 'Trackbars', 0)
cv2.setTrackbarPos('kernel (2n+3)', 'Trackbars', 1)
cv2.setTrackbarPos('Sx-', 'Trackbars', 0)
cv2.setTrackbarPos('Sx+', 'Trackbars', 255)
cv2.setTrackbarPos('Sy-', 'Trackbars', 0)
cv2.setTrackbarPos('Sy+', 'Trackbars', 255)
cv2.setTrackbarPos('Sobel-', 'Trackbars', 0)
cv2.setTrackbarPos('Sobel+', 'Trackbars', 255)
cv2.setTrackbarPos('dir-', 'Trackbars', 10)
cv2.setTrackbarPos('dir+', 'Trackbars', 90)

while 1:
    # Get slider values
    hsv=cv2.getTrackbarPos('HSV ok', 'Trackbars')
    h = cv2.getTrackbarPos('H-', 'Trackbars'), cv2.getTrackbarPos('H+', 'Trackbars')
    s = cv2.getTrackbarPos('S-', 'Trackbars'), cv2.getTrackbarPos('S+', 'Trackbars')
    v = cv2.getTrackbarPos('V-', 'Trackbars'), cv2.getTrackbarPos('V+', 'Trackbars')
    # Thresholds
    thres = cv2.getTrackbarPos('Thres ok', 'Trackbars')
    ksize = 1+2*cv2.getTrackbarPos('kernel (2n+3)', 'Trackbars')
    Sx= cv2.getTrackbarPos('Sx-', 'Trackbars'),cv2.getTrackbarPos('Sx+', 'Trackbars')
    Sy= cv2.getTrackbarPos('Sy-', 'Trackbars'),cv2.getTrackbarPos('Sy+', 'Trackbars')
    Sobel = cv2.getTrackbarPos('Sobel-', 'Trackbars'),cv2.getTrackbarPos('Sobel+', 'Trackbars')
    dir= cv2.getTrackbarPos('dir-', 'Trackbars'),cv2.getTrackbarPos('dir+', 'Trackbars')
    dir=np.dot(dir,2/np.pi)
    lower = np.array([h[0], s[0], v[0]], np.uint8)
    upper = np.array([h[1], s[1], v[1]], np.uint8)

    # Combine original image and mask
    img_nbr = cv2.getTrackbarPos('Test_images', 'Trackbars')
    img = cv2.cvtColor(cv2.imread(images[img_nbr]), cv2.COLOR_BGR2HSV)

    # Calculate mask using thresholds
    mask = cv2.inRange(img, lower, upper)

    # HSV filter application
    if hsv==1 :
        new = cv2.bitwise_and(img,img,mask = mask)
        new = cv2.cvtColor(new, cv2.COLOR_HSV2BGR)
    else:
        new = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # thresholding filter application
    gray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    if thres==1:
        gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=Sx)
        grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=Sy)
        mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=Sobel)
        dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=dir)
        # combine tresholding functions
        combined = np.zeros_like(gradx)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) | (dir_binary == 1))] = 255
        new=combined
    cv2.imshow('image', new)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break
cv2.destroyAllWindows()