import cv2
import numpy as np
import matplotlib.pyplot as plt

I_1 = cv2.imread('photo_1.jpg', 0)
I_2 = cv2.imread('photo_2.jpg', 0)

sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.04,edgeThreshold=10,sigma=1.6)
kp1,des1 = sift.detectAndCompute(I_1,None)
kp2,des2 = sift.detectAndCompute(I_2,None)

matcher = cv2.BFMatcher()
matches = matcher.knnMatch(des1,des2,k=2)

r = .3

good_matches = []
for m,n in matches:
    # Compute the ratio between best match m, and second best match n here
    if m.distance < n.distance * r:      
      good_matches.append(m)

img = cv2.drawMatches(I_1,kp1,I_2,kp2,good_matches,None,flags=2) 
plt.imshow(img)
plt.show()
