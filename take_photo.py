import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0
orig_dir = 'images/known/'
name = input('Enter new name: ')
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "{}{}.png".format(orig_dir, name)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        exit()
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
