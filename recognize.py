import face_recognition
import cv2
import numpy as np
import os

orig_dir = 'images/known/'
dataset = dict()
for image in os.listdir(orig_dir):
    name = image.split('.')[0]
    array_image = cv2.imread(os.path.join(orig_dir, image))
    image_encoding = face_recognition.face_encodings(array_image)[0]
    dataset[name] = {'image': array_image, 'encoding': image_encoding}

known_face_encodings = [dataset[obj]['encoding'] for obj in dataset.keys()]
known_face_names = [obj for obj in dataset.keys()]

cap = cv2.VideoCapture(0)
out = cv2.VideoWriter(os.path.join(), -1, 20.0, (640, 480))

while True:
    ret, image = cap.read()
    try:
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.face_distance(known_face_encodings, face_encoding)
            name = ""

            if matches.any():
                min_distance = np.min(matches)
                print(matches)
                if min_distance < 0.50:
                    name = known_face_names[np.argmin(matches)]
                image = cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, name, (left, bottom), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            cv2.imshow('test', image)
            out.write(image)
    except Exception as e:
        print(e)
        cv2.imshow('test', image)
        out.write(image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
