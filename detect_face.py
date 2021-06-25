import dlib
import os
import cv2
from imutils import face_utils
import imutils
import numpy as np

detector = dlib.get_frontal_face_detector()

pred_loc = os.path.join('shape_predictor_68_face_landmarks.dat')
# image_loc = os.path.join(base_dir, 'apps/faceapi/pd_distance_api/image.JPEG')
cap = cv2.VideoCapture(0)
predictor = dlib.shape_predictor(pred_loc)
# image = cv2.imread(image_loc)  # VideoCapture(0)
while True:
    ret, image = cap.read()
    image_orig = imutils.resize(image, width=200)
    image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(image, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
        bottom, top = np.max(shape[:, 1]), np.min(shape[:, 1])
        left, right = np.max(shape[:, 0]), np.min(shape[:, 0])

        image_orig = cv2.rectangle(image_orig, (left, top), (right, bottom), (255, 0, 0), 0)

    # show the frame
    image_orig = imutils.resize(image_orig, width=640)
    cv2.imshow("Frame", image_orig)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
