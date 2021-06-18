import cv2
import numpy as np
from PIL import Image

harcasPath = "haarcascade_frontalface_default.xml"
mask_path = "mask.png"

faceCascade = cv2.CascadeClassifier(harcasPath)

mask = Image.open(mask_path)


def thug_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 2.1)
    background = Image.fromarray(image)

    for(x, y, w, h) in faces:
        resized_mask = mask.resize((w, h), Image.ANTIALIAS)

        offset = (x, y)
        background.paste(resized_mask, offset, mask=resized_mask)

    return np.asarray(background)


cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if ret:

        img = thug_mask(frame)
        cv2.imshow("Face", img)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()
