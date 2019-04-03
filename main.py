import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
from sklearn import neighbors
from sklearn import metrics
import imutils


def preprocess(img, y1, y2, x1, x2):
    x = img[y1:y2, x1: x2]
    x = imutils.resize(x, width=50, height=50)
    kernel = np.ones((2, 2), np.uint8)
    # x = cv2.dilate(x, kernel, iterations=10)
    return x


img = cv2.imread('handwritten.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# thresholding
ret, img = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)

# dilate the image
kernel = np.ones((5, 5), np.uint8)
img = cv2.dilate(img, kernel, iterations=10)


eight = preprocess(img,120, 340, 220, 440 )
plt.imshow( eight)
plt.show()

plus = preprocess(img, 120, 340, 450, 670)
plt.imshow(plus)
plt.show()

three = preprocess(img, 120, 340, 600, 820)
plt.imshow(three)
plt.show()


# img1
# # make it binary
# one = (one < 100).astype(int)
plus = (plus < 100).astype(int)
three = (three < 100).astype(int)
eight = (eight < 100).astype(int)

print("loading classifier")
knnPickel = pickle.load(open("./knn.classifier", "rb"))
knn = knnPickel["knn"]
# print(knn.predict(one.reshape(1,-1)), knn.predict(plus.reshape(1,-1)), knn.predict(two.reshape(1,-1)), )
print(knn.predict(eight.reshape(1,-1)), knn.predict(plus.reshape(1,-1)), knn.predict(three.reshape(1,-1)))

# print("Accuracy:",metrics.accuracy_score(testLabels, testRes))