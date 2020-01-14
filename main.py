import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import glob, os
import numexpr as ne

boundingBoxSelected = 0
cornersCaptured = 0
refPt = []
n_colors = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, .1)
flags = cv2.KMEANS_RANDOM_CENTERS
cap = cv2.VideoCapture('colors.mp4')
cv2.namedWindow('Calibration', cv2.WINDOW_FULLSCREEN)

def grabScreenBoundingBoxCoordinate(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, frame, boundingBoxSelected, cornersCaptured
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        cornersCaptured=cornersCaptured+1
    if cornersCaptured == 2:
        boundingBoxSelected = 1
cv2.setMouseCallback("Calibration", grabScreenBoundingBoxCoordinate)
firstRound = 1
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        if not boundingBoxSelected:
            cv2.imshow('Calibration', frame)
            cv2.polylines(frame, np.int32([refPt]), 1, (0, 0, 255))
        else:
            if firstRound:
                cv2.setMouseCallback("Calibration", lambda *args : None)
                #cv2.destroyWindow("Calibration")
                minX, maxX, minY, maxY = min([refPt[0][0], refPt[1][0]]), max([refPt[0][0], refPt[1][0]]), min([refPt[0][1], refPt[1][1]]), max([refPt[0][1], refPt[1][1]])
                firstRound = 0
            frameCrop = frame[minY:maxY, minX:maxX]
            frameCrop = cv2.pyrDown(frameCrop)
            #frameCrop = cv2.cvtColor(frameCrop, cv2.COLOR_BGR2GRAY)
            #ret, thresh = cv2.threshold(frameCrop, 127, 255, 0)
            #cv2.imshow('thresh', thresh)
            #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(frameCrop, contours, -1, (120, 255, 155), 3)
            cv2.imshow('Calibration', frameCrop)
            pixels = np.float32(frameCrop.reshape(-1, 3))
            _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 5, flags)
            print(palette)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
# index = 0
# dom_patch = np.zeros(shape=(picCount, n_colors, 3), dtype=np.uint8)
# for file in glob.glob("*.JPG"):
#     img = io.imread(file)
#     pixels = np.float32(img.reshape(-1, 3))
#     #bincount_numexpr_app(img)
#     _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
#     _, counts = np.unique(labels, return_counts=True)
#     indices = np.argsort(counts)[::-1]
#     freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
#     rows = np.int_(img.shape[0]*freqs)
# #    dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
#     for i in range(n_colors):
#         dom_patch[index, i, :] += np.uint8(palette[indices[i]])
#     index = index+1
#
# fig, ax1 = plt.subplots(1, 1, figsize=(12,6))
# ax1.imshow(dom_patch)
# ax1.set_title('Dominant colors')
# ax1.axis('off')
# fig.savefig('colorImage.png')

