import cv2 as cv
import sys

if len(sys.argv) < 2:
    sys.exit(1)
s, t = 5, 40
img = cv.imread(sys.argv[1])
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# blurred = cv.GaussianBlur(gray, (s, s), 0)
# cv.imwrite("blurred.png", blurred)
edges = cv.Canny(gray, t, t-1)
img_e = img.copy()
img_e[edges == 0] = 255
img_e[edges == 255] = 0
cv.imwrite("img_e.png", img_e)
cv.imwrite("edges.png", edges)
