import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('./eegs.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh= cv2.threshold(gray, 149, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
area = [cv2.contourArea(j) for j in contours]
area_sort = np.argsort(area)[::-1]
count = 0
for i in area_sort[1:]:
        if area[i] > 1000:
            cnt = contours[i]
            cv2.drawContours(img, [cnt], 0, (20, 100, 200), 3)
            count += 1
print(count)
plt.imshow(img)