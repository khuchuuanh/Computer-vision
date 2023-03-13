from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
def alignment(img1_color, img2_color, value1 = 5000, value2 = 0.9):
    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape
    
    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(value1)
    
    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)
    
    # Match features between the two images.
    # We create a Brute Force matcher with 
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    
    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)


    matches = list(matches)
    # Sort matches on the basis of their Hamming distance.
    matches.sort(key = lambda x: x.distance)
    
    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*value2)]
    no_of_matches = len(matches)
    
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    
    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt
    
    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    
    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color,
                        homography, (width, height))
    return transformed_img


temp = cv2.imread("temp.jpg")
img = cv2.imread("cmt1.jpg")

img_aft = alignment(img, temp)

plt.axis("off")
plt.figure(figsize = (10, 10))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img_aft, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))


temp = cv2.imread("temp.jpg")
img = cv2.imread("cmt2.jpeg")

img_aft = alignment(img, temp)

plt.axis("off")
plt.figure(figsize = (10, 10))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img_aft, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))


temp = cv2.imread("temp.jpg")
img = cv2.imread("cmt3.jpg")

img_aft = alignment(img, temp)

plt.axis("off")
plt.figure(figsize = (10, 10))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img_aft, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))

temp = cv2.imread("temp.jpg")
img = cv2.imread("cmt4.jpg")

img_aft = alignment(img, temp)

plt.axis("off")
plt.figure(figsize = (10, 10))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img_aft, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))

temp = cv2.imread("temp.jpg")
img = cv2.imread("cmt5.jpg")

img_aft = alignment(img, temp)

plt.axis("off")
plt.figure(figsize = (10, 10))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img_aft, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))

plt.show()