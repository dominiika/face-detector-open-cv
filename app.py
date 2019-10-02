import cv2
import pandas
from datetime import datetime

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
video = cv2.VideoCapture(0)
visible_face_list = [0]
times = []
df = pandas.DataFrame(columns=["Appears", "Disappears"])

while True:
    check, frame = video.read()
    visible_face = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)

    for (x, y, w, h) in faces:
        visible_face = 1
        color = (255, 0, 150)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        font = cv2.FONT_HERSHEY_SIMPLEX
        time = str(datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        color = (255, 255, 255)
        stroke = 1
        cv2.putText(frame, time, (x - 10, y - 8), font, 1, color, stroke, cv2.LINE_AA)

    cv2.imshow("Capturing faces", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        if visible_face == 1:
            times.append('Camera turned off')
        break

    visible_face_list.append(visible_face)

    if visible_face_list[-1] == 1 and visible_face_list[-2] == 0:
        times.append(datetime.now())
        image_name = "images/" + datetime.now().strftime("%d-%m-%Y_%H:%M:%S")+'.jpg'

        cv2.imwrite(image_name, frame)
    elif visible_face_list[-1] == 0 and visible_face_list[-2] == 1:
        times.append(datetime.now())

for i in range(0, len(times), 2):
    df = df.append({"Appears": times[i], "Disappears": times[i+1]}, ignore_index=True)

df_name = "Times_" + datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + ".csv"
df.to_csv(df_name)

video.release()
cv2.destroyAllWindows()
