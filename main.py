import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import time

# loading the stored model from file
model = load_model(r'/home/user/Downloads/Fire-64x64-color-v7-soft.h5')

cap = cv2.VideoCapture(r'/home/user/Downloads/Fire Blaze.avi')
time.sleep(2)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

while cap.isOpened:
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame, 0)

        IMG_SIZE = 64
        # IMG_SIZE = 224

        # for i in range(2500):
        #    cap.read()

        while (1):

            rval, image = cap.read()
            if rval == True:
                orig = image.copy()

                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                image = image.astype("float") / 255.0
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)

                tic = time.time()
                fire_prob = model.predict(image)[0][0] * 100
                toc = time.time()
                print("Time taken = ", toc - tic)
                print("FPS: ", 1 / np.float64(toc - tic))
                print("Fire Probability: ", fire_prob)
                print("Predictions: ", model.predict(image))
                print(image.shape)

                label = "Fire Probability: " + str(fire_prob)
                cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                out.write(frame)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) == ord('q'):
                break

cap.release()
out.release()
cv2.destroyAllWindows()
