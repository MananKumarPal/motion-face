import cv2
import time

video = cv2.VideoCapture(1)

a = 0

while True:
    a = a+1
    check, frame = video.read()
    print(check)
    print(frame)

    Manan = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # time.sleep(2)
    cv2.imshow("Capturing", Manan)
    # cv2.imshow("Capturing",frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

print(a)
video.release()
cv2.destroyAllWindows()
