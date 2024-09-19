from PIL import Image
import cv2
import numpy as np 
import requests

# loading the image
# grabbing from URL and using Python's PIL Image and OpenCV
image_url = 'https://a57.foxnews.com/media.foxbusiness.com/BrightCove/854081161001/201805/2879/931/524/854081161001_5782482890001_5782477388001-vs.jpg'
response = requests.get(image_url, stream=True)
image = Image.open(response.raw)
image = image.resize((450,250))

# converting to Numpy array - helps with further processing
# becuase Numpy is easier for math operations
image_np = np.array(image)

# grayscale aids with reducing memory, noise, computation 
image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

# # Show the grayscale image
# cv2.imshow("Grayscale Image", image_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# gaussian blur - smooths image by averaging pixel values
# reduces noise, enhances features, makes it easier to do more analysis
image_blur = cv2.GaussianBlur(image_gray, (7,7), 0)
# cv2.imshow("Gaussian Blur", image_blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# dilation expands bright regions in image -- enhance object boundary + remove noise
image_dil = cv2.dilate(image_blur, np.ones((3,3)))
# cv2.imshow("dilation", image_dil)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# morphological closing
# fills gaps, smoothes edges, reduces noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
closing = cv2.morphologyEx(image_dil, cv2.MORPH_CLOSE, kernel)

car_cascade_src = 'cars.xml'
car_cascade = cv2.CascadeClassifier(car_cascade_src)
cars = car_cascade.detectMultiScale(closing, 1.1, 1)

for (x, y, w, h) in cars:
    cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Convert the annotated image to PIL Image format and display it
annotated_image = Image.fromarray(image_np)
annotated_image.show()

# Close the window when a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()