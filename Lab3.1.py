import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('D:/CPV/coin2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (11,11), 0)
canny = cv2.Canny(blur, 30, 150, 3)
dilated = cv2.dilate(canny, (1,1), iterations = 2)
(contour, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, contour, -1, (0,255,0), 2)
plt.imshow(rgb)
plt.show()

print('Number of coins are: ', len(contour))

size = []
for i in range(len(contour)):
    size.append(cv2.contourArea(contour[i]))
size.sort()
count_group = []
max_error = 300
root = size[0]
group = 1
count = 1
for i in range(1 , len(size) + 1):
    if i == len(size):
        count_group.append(count)
        break
    if size[i] - root > max_error:
        count_group.append(count)
        count = 1
        group = group + 1
        root = size[i]
    else:
        count = count + 1

print("The type of coins is:  " + str(group))

for i in range(group):
    print("the type of coins of group " + str(i) + " is " + str(count_group[i]))

img = cv2.imread('D:/CPV/eggs.png')
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
plt.imshow(img)

