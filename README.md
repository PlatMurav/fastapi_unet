# Segmentation Model with TensorFlow and TFRecords
## Overview
This repository contains code for starting the FastAPI server and uses previously created model to do segmentation task.
<br>The model architecture used in this repository is a Convolutional Neural Network (CNN) based on U-Net and designed for semantic segmentation tasks. It takes input images and outputs segmentation masks with class labels for each pixel.
<br>The whole training process can be seen in other repositories (such as **tfds_Unet**).

### Files Description
- `main.py`: contains the FastAPI server creation and post method.
- `model.py`: contains the Segmentator class, responsible for the model and datasets creation and images pre/post-processing.
- `camera_38.png`: an image used to test the model and server.
- `server_demo.py`: is just a test file used to connect to our server and upload the image and get response.
- `model_t800.keras`: our model that was previously created and trained.
- `mask_image.png`: the result of our post method.


## Requirements
To be albe to use the model you should have:
* trained model
* an image containing one or more objects


# Main.py
*The server code is contained in main.py*
### Intro
At first, we need:
1. Create an instance of **Segmentator class** (*so we could pre/post-process images and then handle them to the model*)
2. Create an instance of **FastAPI class**. 
```python
from model import Segmentator

app = FastAPI()
segmentator = Segmentator()
```
### Initializing the model
At frist we should check if the model exists and raise an error if it does not.
```python
filename = 'model_t800.keras'
if not os.path.exists(filename):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
else:
    model = tf.keras.models.load_model('model_t800.keras')
```
Since we have our model in the same folder it will be downloaded and put in the variable "model".
<br>**Ensure** your TensorFlow model is saved and place it in the appropriate path.

### Starting the server
The code below runs the FastAPI application using Uvicorn when the script is executed directly. The app will be available at http://127.0.0.1:8000.

```python
# Running the server
uvicorn.run(app, host='127.0.0.1', port=8000)
```

## @app.Post
### Reading the file
We take an image file uploaded via an HTTP request, read its contents, and convert it into a format suitable for further processing with OpenCV
```python
# The await file.read() reads the file contents (asynchronously) into a bytes object called "contents"
contents = await file.read()

# np.frombuffer creates a 1-dimensional NumPy array from the buffer (bytes object) contents
np_img = np.frombuffer(contents, np.uint8)

# Decode the image using OpenCV 
image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)  
```
**Notes**:
* "file" is an instance of UploadFile provided by FastAPI when a file is uploaded via a POST request.
* "np_img" contains the raw byte data of the image.
* The result, image, is a NumPy array representing the decoded image. The shape of this array will typically be (height, width, 3) for a color image.

### Pre/post-processing and convertation

```python
# Preprocess the image to the required format for the model.
image = segmentator.preprocess_image(img)

# Performs prediction and returns the mask of an image
prd_mask = model.predict(image)

# Postprocess the model output to a human-readable format
final_mask = segmentator.postprocess_image(prd_mask)

# Encodes the numpy array (or previously tensor) to a PNG image
ret, buffer = cv2.imencode('.png', final_mask.numpy())

# Convert the buffer (the image) to a byte string so it could be sent back
byte_string = buffer.tobytes()
```
**Notes**:
*  Postprocess method deletes batch-dimension and uses argmax() funtion.
*  Preprocess method includes resizing and normalizing the image.

### Response
We return the encoded image as an HTTP response. 
```python
# Sets the content of the HTTP response to byte_string
return Response(content=byte_string, media_type="image/png")
```
**Notes**:
* *Response* is a class from FastAPI that allows you to create a custom HTTP response.
* byte_string is the image data that has been encoded as a byte string using OpenCV.
* a_type="image/jpeg" sets the Content-Type header of the HTTP response to "image/png".

# Server_demo.py

### Url
At first we define the server endpoint
```python
url = 'http://127.0.0.1:8000/predict'
```
### Request
1. Open the image file in binary mode
2. Create a dictionary to hold the file for the POST request
3. Send the POST request with the image file
```python
with open('camera_38.png', 'rb') as image_file:
    files = {'file': image_file}
    response = requests.post(url, files=files)
```
### Processing response
1. We get the content of the response (which is expected to be a byte string).
```python
byte_string = response.content
```
2. **Convert the byte string** to a NumPy array of type uint8 (unsigned 8-bit integer).
```python
np_img = np.frombuffer(byte_string, np.uint8)
```
3. **Decode the NumPy array into an image** using OpenCV, and convert it to grayscale.
```python
img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
```
### Scaling and Saving
The pixel values of the image are multiplied by 20. This enhances the visibility of the objects in the image, as it increases the intensity of the pixel values.
<br>The processed image is saved to a file named 'mask_image.png'
```python
img = img * 20

cv2.imwrite('mask_image.png', img)
```
