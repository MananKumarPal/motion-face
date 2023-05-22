import cv2
import time
import pandas
from datetime import datetime

first_frame = None
# to prevent index out of bound error at the beginning
status_list = [None, None]
times = []
df = pandas.DataFrame(columns=["Start", "End"])
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(1)  # 0 is the default camera (webcam) of my computer

time.sleep(2)
while True:
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    check, frame = video.read()
    status = 0
    Manan = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Manan = cv2.GaussianBlur(Manan, (21, 21), 0)  # for better accuracy
    faces = face_cascade.detectMultiScale(Manan,
                                      scaleFactor=1.3,
                                      minNeighbors=5)
    if first_frame is None:
        first_frame = Manan
        continue  # skip the rest of the lines in this iteration

    delta_frame = cv2.absdiff(first_frame, Manan)
    thres_frame = cv2.threshold(delta_frame, 80, 255, cv2.THRESH_BINARY)[1]
    thres_frame = cv2.dilate(thres_frame, None, iterations=2)

    # find all contours in the thres_frame
    (cnts, _) = cv2.findContours(thres_frame.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter the contours that have more than 1000 pixels
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue

        status = 1  # first object found entering the frame
        # draw a rectangle around the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 3)  # green rectangle

    status_list.append(status)

    # we only need to retain the last two items in the list
    status_list = status_list[-2:]
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    cv2.imshow("Manan Frame", Manan)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thres_frame)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        if status == 1:
            # treat quitting as the last exit time
            times.append(datetime.now())
        break
    # end of loop

print(status_list)
print(times)

for i in range(0, len(times), 2):
    df = df.append({"Start": times[i], "End": times[i+1]}, ignore_index=True)

df.to_csv("Times.csv")  # output the csv file for plotting

video.release()
cv2.destroyAllWindows()
