# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import imutils
import time
import cv2
import pickle

from cv.cvutils import *
from structures.boxtracker import BoxTracker


class FRVideoFaces():

    def __init__(self):
        self.detection_method = "cnn"
        print("[INFO] loading recognizer...")
        self.recognizer = pickle.loads(open("data/output/recognizer.pickle", "rb").read())
        print("[INFO] loading labels...")
        self.le = pickle.loads(open("data/output/le.pickle", "rb").read())

    def frvideofaces(self, source=None, use_ct=0.1):
        # if a video path was not supplied, grab a reference to the webcam
        if source is None:
            print("[INFO] starting video stream...")
            vs = VideoStream(src=0).start()
            time.sleep(2.0)
        # otherwise, grab a reference to the video file
        else:
            print("[INFO] opening video file...")
            vs = cv2.VideoCapture(source)

        # instantiate our box tracker
        bt = BoxTracker(maxDisappeared=2, maxDistance=40)

        # start over frames from the video file stream
        while True:
            # grab the frame from the threaded video stream
            frame = vs.read()

            if source is None:
                frame = frame
            else:
                frame = frame[1]

            # if we are viewing a video and we did not grab a frame then we
            # have reached the end of the video
            if frame is None:
                break

            # convert the input frame from BGR to RGB then resize it to have
            # a width of 750px (to speedup processing)
            frame = imutils.resize(frame, width=750)

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input frame, then compute
            # the facial embeddings for each face
            boxes = face_recognition.face_locations(frame,
                model=self.detection_method)
            encodings = face_recognition.face_encodings(frame, boxes)

            # initialize the current status along with our list of bounding
            # box rectangles returned by either [1] our object detector or
            # (2) the correlation trackers
            rects = []

            # loop over the detections
            for i in range(len(boxes)):
                # trying to keep the correspondence of boxes, encodings and rects
                rects.append(mcoord_to_coord(boxes[i]))

            # use the box tracker to associate the (1) old object
            # boxes with (2) the newly computer object boxes
            objects = bt.update(rects)

            # loop over the tracked objects
            for (objectID, box) in objects.items():
                (startX, startY, endX, endY) = box.astype("int")

                tbox = npcoord_to_coord(box)
                if tbox in rects:
                    idx = rects.index(tbox)
                    enc = encodings[idx]

                    preds = self.recognizer.predict_proba(enc.reshape(1, -1))[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = self.le.classes_[j]
                    color = (0, 0, 255)
                    if proba > use_ct:
                        color = (0, 255, 0)
                    text = "{} {:.2f}".format(name, proba)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)
                    cv2.putText(frame, text, (startX, endY + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # show the output frame
            cv2.imshow(source, frame)
            key = cv2.waitKey(1) & 0xFF

            # if the 'q' key was pressed, break from the loop
            if key == ord("q"):
                break

        # if we are using a webcam, release the pointer
        if source is None:
            vs.stop()
        # otherwise, release the file pointer
        else:
            vs.release()

        # do a bit of cleanup
        cv2.destroyAllWindows()


if __name__ == "__main__":
    frv = FRVideoFaces()
    frv.frvideofaces("data/videos/lunch_scene.mp4", use_ct=0.65)
