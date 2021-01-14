import cv2


path = r'D:\111.png'

img = cv2.imread(path)
img = cv2.resize(img, (500,300))
cv2.imwrite(r'D:\222.jpg', img, )
