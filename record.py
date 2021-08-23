#!/usr/bin/python

import cv2

capture = cv2.VideoCapture(0)

x = input()
# video recorder
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(x, fourcc, 20.0, (640, 480))


# record video
while (capture.isOpened()):
    ret, frame = capture.read()
    video_writer.write(frame)
    cv2.imshow('Video Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
video_writer.release()
cv2.destroyAllWindows()