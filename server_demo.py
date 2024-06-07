import requests
import numpy as np
import cv2

url = 'http://127.0.0.1:8000/predict'  # URL of the server endpoint

# Open the image file in binary mode
with open('camera_38.png', 'rb') as image_file:
    files = {'file': image_file}
    response = requests.post(url, files=files)


print(response.status_code)

byte_string = response.content
np_img = np.frombuffer(byte_string, np.uint8)
img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

img = img * 20 # to make objects visible

cv2.imwrite('mask_image.png', img)
