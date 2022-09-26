import time
import numpy as np
import cv2
from PIL import Image
from pytesseract import pytesseract

ref_point = []
crop = False


def show_text(inp_text):
    img = np.ones(shape=(512, 512, 3), dtype=np.int16)
    cv2.putText(img=img, text=inp_text, org=(150, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3,
                color=(0, 255, 0), thickness=3)
    cv2.imshow("TEXT", img)



###########################################

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()

    cv2.imshow('frame', frame)

    if cv2.waitKey(1):
        pass
    cv2.imwrite('sih.jpg', frame)
    time.sleep(1)


    path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.tesseract_cmd = path_to_tesseract
    img = Image.open('sih.jpg')
    img = img.crop((100, 100, 200, 500))
    img.show()
    text = pytesseract.image_to_string(img)
    print(text)

    ##text ka function nahi chla abhi mrse
    #print(text)

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()