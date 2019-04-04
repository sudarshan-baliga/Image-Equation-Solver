import cv2
import numpy as np
from postfix import B

def convert(string):
    #print("From source : ",string)
    obj = B()
    return obj.callToPostfix(string)