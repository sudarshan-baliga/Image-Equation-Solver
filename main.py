import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
import imutils


def preprocessRegion(img, y1, y2, x1, x2):
    x = img[y1:y2, x1: x2]
    x = imutils.resize(x, width=50, height=50)
    return x

def slidingWindow(img):
    width, height = img.shape
    count = 3
    regions = []
    x1, x2 = 30, 250
    while(count):
        regions.append(preprocessRegion(img ,0,220,x1, x2))
        x1 += 200
        x2 += 200
        count -= 1
    return regions


img = cv2.imread('cropped.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# thresholding
ret, img = cv2.threshold(gray,80,255,cv2.THRESH_BINARY_INV)

# dilate the image
kernel = np.ones((5, 5), np.uint8)
img = cv2.dilate(img, kernel, iterations=10)

print("loading classifier")
knnPickel = pickle.load(open("./knn.classifier", "rb"))
knn = knnPickel["knn"]


regions = slidingWindow(img)
for reg in regions:
    plt.imshow(reg)
    plt.show()
    reg = (reg < 100).astype(int)
    print(reg.shape)
    print(knn.predict(reg.reshape(1,-1)))

    # cv2.imshow("eight", region)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
# plus = preprocessRegion(img, 120, 340, 450, 670)


# three = preprocessRegion(img, 120, 340, 600, 820)
# plt.imshow(three)
# plt.show()


# img1
# # make it binary
# one = (one < 100).astype(int)
# plus = (plus < 100).astype(int)
# three = (three < 100).astype(int)
# eight = (eight < 100).astype(int)

# print("loading classifier")
# knnPickel = pickle.load(open("./knn.classifier", "rb"))
# knn = knnPickel["knn"]
# print(knn.predict(one.reshape(1,-1)), knn.predict(plus.reshape(1,-1)), knn.predict(two.reshape(1,-1)), )
# print(knn.predict(eight.reshape(1,-1)), knn.predict(plus.reshape(1,-1)), knn.predict(three.reshape(1,-1)))

# print("Accuracy:",metrics.accuracy_score(testLabels, testRes))