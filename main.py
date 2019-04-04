import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
import imutils
from converter import convert


def preprocessRegion(img, y1, y2, x1, x2):
    x = img[y1:y2, x1: x2]
    x = imutils.resize(x, width=50, height=50)
    return x

def slidingWindow(img):
    height, width = img.shape
    img2 = img.copy()
    regions = []
    x1, x2 = 30, 250
    while(x2 <= width):
        img2 = img.copy()
        regions.append(preprocessRegion(img ,0,220,x1, x2))
        cv2.rectangle(img2, (x1, 0), (x2, 220), (255,0,0), 2)
        cv2.imshow("sliding window", img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        x1 += 200
        x2 += 200
    return regions


img = cv2.imread('handwritten1.png')

plt.imshow(img)
plt.title("original image")
plt.show()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# thresholding
ret, img = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)
plt.imshow(img, cmap = 'gray')
plt.title("segmented image")
plt.show()

# dilate the image
kernel = np.ones((5, 5), np.uint8)
img = cv2.dilate(img, kernel, iterations=8)
plt.imshow(img, cmap = 'gray')
plt.title("dilated image")
plt.show()

print("loading classifier")
knnPickel = pickle.load(open("./knn2.classifier", "rb"))
knn = knnPickel["knn"]

# get the regions with text
regions = slidingWindow(img)
result = []
for reg in regions:
    reg = (reg < 50).astype(int)
    sym = knn.predict(reg.reshape(1,-1))
    plt.imshow(reg, cmap='gray')
    plt.title("region with symbol " + sym[0])
    plt.show()
    result.append(sym[0])
for char in result:
    print(char, end=" ")
print("=", convert(result))