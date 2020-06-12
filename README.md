# face-recognition
Recognition of faces from video.

## introduction
This program is implemented from [Face recognition with OpenCV, Python, and deep learning](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/) and the classification part of [OpenCV Face Recognition](https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/).  The classification section of the algorithm was changed because classification in the first linked site uses a one-by-one distance comparison.  This would cause the speed of matching to vary linearly with the number of candidate matches.

## requirements
* Python 2.7
* [imutils 0.5.3](https://pypi.org/project/imutils/)
* [face_recognition 1.3.0](https://pypi.org/project/face-recognition/)
* [dlib 19.19.0](https://pypi.org/project/dlib/)
* OpenCV 3.4.0
