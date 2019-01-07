# Facial-Detection in Python

Similar concept with Depth-Sensor project just without the additional information
of depth to look for faces within a distance.

This is small-scale program to count the amount of user looking at the camera to
determine if they are 'reading'. Of course, there are different and better method
of determining with better sophisticated program and high-end hardware.

## Prerequisite
1. [Python 2.7.15](https://www.python.org/downloads/release/python-2715/)
2. External Library
    1. [OpenCV](https://pypi.org/project/opencv-python/)
    2. [Dlib](https://pypi.org/project/dlib/) - required CMake downloaded and installed
    3. [imutils](https://pypi.org/project/imutils/)
3. [Pre-trained facial landmark detector](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

This program is based on the tutorial, created by Adrian Rosebrock, linked below:

https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/

https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/

In the tutorial, Rosebrock teaches how to find the landmark of a face and counting
blinking of the eye. However, what we required is their method of detecting a frontal
face. As such, there will be a few snippet of code required to run this program
which will simplify things.

Firstly, have to import packages into the script like the tutorial:

    #import necessary packages
    from imutils.video import VideoStream
    from imutils import face_utils
    import numpy as np
    import argparse
    import imutils
    import time
    import dlib
    import cv2

If you do not have those libraries: mainly **imutils**, **dlib** and **cv2** - use
"pip install" command in command prompt to download them. Click on the link in the
prerequisite to be directed to the site for more information.

    pip install opencv-python
    pip install imutils
    pip install dlib

If "pip install dlib" fails, you might not have download and install [CMake](https://cmake.org/download/)

If "pip" command is not found, setup your environment path by going to
"System Properties" -> "Environment Variables..." -> "System variables" ->
"Path" -> "New" and add the path (default path: C:\Python27\Scripts).

To allow the logging of timestamp of the face, add these import into your script:

    # packages for logging and flushing to cmd screen on runtime
    import sys # flushing to cmd screen
    import datetime # logging the time of face detection and last seen

    # the path of the log file
    logfilePath = open("logtime.txt", "a+") # will automatically create or append to existing

The 'sys' import will allow command prompt the flush out information while running
the script. Remember to set your text file path to log timestamp!

Next, we are going to allow argument to be passed to the call of the script:

    # allowing arguments to be passed when executing script
    ap = argparse.ArgumentParser()
    ap.add_argument("-p","--shape-predictor",required=True,help="path to facial landmark predictor")
    args = vars(ap.parse_args())

This will allow the pre-trained facial landmark (download in the prerequisite) to
be loaded and used in the script. The landmark are not required but it serves as
a second verification of the front of the face instead of the side profile.

To be able to detect the face and predict the landmark, initialise the function
and assign the call to a variable:

    # dlib calls to detect frontal face and eyes (optional - 2nd verification)
    detect_face = dlib.get_frontal_face_detector()
    predict_eyes = dlib.shape_predictor(args["shape_predictor"]) # the file is passed when calling the script: python frontal_face_detection.py --shape-predictor shape_predictor_68_face_landmarks.dat

Start the video streaming and assign to a variable to retrieve frame information:

    # starting the video stream thread
    vs = VideoStream(src=0).start()
    time.sleep(2.0) # time delay to let devices warm up

The sleep is called to let the device buff time to launch.

Before we get into the loop of processing the video, create 4 empty list variables:

    # variables to keep track of faces and index to be removed from list
    faces_onscreen = [] # store all the faces meeting the condition appearing in that current loop of frame
    faces_verifying = [] # store all the first encounter of faces to be verify
    faces_logging = [] # store all the faces to be logged with timestamp
    removeIndex = [] # store the indexes to be remove in a list

Last step before looping the video is to grab the indexes of the facial
landmark of the left and right eye:

    # grabbing the indexes of the facial landmarks for the left and right eyes
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

Again, landmark is not required as it is for second verification.

Now to process the video. To start, add the following code:

    # looping over each frames from the video stream
    while True:
        # grab the current frame from video to be process
        frame = vs.read()
        frame = imutils.resize(frame, width=450) # resizing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale for lower processing demand

This is allow the script to read each frame to resize and convert to grayscale.

As soon as you have the image/frame, you can now detect the faces in it.

    # detect faces in the grayscale frame
    rects = detect_face(gray, 0)
    for x in rects: # loop through all detected face
       # if 2nd verification is not required, uncomment the line below and comment the 2nd verification process
       ###faces_onscreen.append((x.left(), x.top(), x.right(), x.bottom())) # add to the list

       # optional 2nd verification - finding the eyes within the face
       shape = predict_eyes(gray, x)
       shape = face_utils.shape_to_np(shape)
       leftEye = shape[lStart:lEnd]
       rightEye = shape[rStart:rEnd]
       # if there aren't two eyes (not facing forward) - do not consider to log
       # otherwise, consider to log
       if len(leftEye) > 0 and len(rightEye) > 0:
           faces_onscreen.append((x.left(), x.top(), x.right(), x.bottom())) # add to the list


In this process, we are detecting the faces in the image and find if it has more
than 2 eyes within the region of interest. Of course, excluding the eye it would
be a simpler code.

By appending to "faces_onscreen", you're 'confirming' it's a frontal face. Of course,
it is not 100% frontal as a slight deviation could be counted. The extra verification
from the eyes, might or might not improve the accuracy as well. But in this case,
it should suffice.

Next few steps is to verify and start logging the face if verified. Firstly,
adding the face to be verify:

    # add new faces to verifying list and updating movement made by verifying and verified faces
    for (x, y, w, h) in faces_onscreen:
        faceExist = False
        # skipping verified faces
        index = 0
        for (x2, y2, w2, h2, st2, ls2) in faces_logging:
            # if two faces overlapped -> consider as same face
            if not (w < x2) and not (x > w2) and not (h < y2) and not (y > h2):
                faceExist = True
                faces_logging[index] = (x, y, w, h, st2, time.time()) # reassign face new values -> to deal with movement
                break
            # if similiar coordinate -> same face
            if abs(x - x2) < 75 and abs(y - y2) < 75 and abs(w - w2) < 75 and abs(h - h2) < 75:
                faceExist = True
                faces_logging[index] = (x, y, w, h, st2, time.time()) # reassign face new values -> to deal with movement
                break
            index += 1
        # skipping verifying faces
        index = 0
        for (x2, y2, w2, h2, st2, ls2) in faces_verifying:
            if not (w < x2) and not (x > w2) and not (h < y2) and not (y > h2):
                faceExist = True
                faces_verifying[index] = (x, y, w, h, st2, time.time())  # reassign face new values -> to deal with movement
                break
            if abs(x - x2) < 75 and abs(y - y2) < 75 and abs(w - w2) < 75 and abs(h - h2) < 75:
                faceExist = True
                faces_verifying[index] = (x, y, w, h, st2, time.time())  # reassign face new values -> to deal with movement
                break
            index += 1
        # adding new faces to verifying list
        if not faceExist:
            faces_verifying.append((x, y, w, h, time.time(), time.time()))

While it is adding face to be verify, it will also adjust verifying face
and verified face to compensate with the movement.

Of course, if a person is just passing by and is not REALLY looking at the screen,
it will also detect it. However, we don't want to record that so we have a time
recorded for time delay by processing as followed:

    # remove verifying faces if not on screen
    index = 0
    for (x, y, w, h, st, ls) in faces_verifying:
        faceExist = False
        for (x2, y2, w2, h2) in faces_onscreen:
            if not (w < x2) and not (x > w2) and not (h < y2) and not (y > h2):
                faceExist = True
                break
            if abs(x - x2) < 75 and abs(y - y2) < 75 and abs(w - w2) < 75 and abs(h - h2) < 75:
                faceExist = True
                break
        # add indexes of face to be removed
        if not faceExist:
            if time.time() - st > 2:  # started more than 2 seconds ago (time can be change to your requirement)
                if time.time() - ls > 0.25:  # but last since was more than 0.25 second ago (time can be change to your requirement)
                    removeIndex.append(index)
        index += 1


    # remove face from list
    removeIndex.reverse() # allow removing from 'back to front' to prevent moving index causing incorrect removal
    for x in removeIndex:
        faces_verifying.pop(x)
    # clear the removing list
    del removeIndex[:]

Okay, so in the process we are checking is if the verifying face is still exist or
not - by using the same method to determine if it is the same face. If it doesn't
exist anymore (can be a range of reasons: walking pass, looking away, etc), it
will not register the face. However, there will be a condition to remove it which
is: if it is started 2 seconds ago and last seen is more than 0.25 seconds ago.
This is to give buffer time for user to look back to register. To remove the face,
the index of the face will be added into a list to be looped. It is deleting
from descending order as ascending order will not delete it properly - example
deleting 1,3 and 5 ascendingly will delete 1, 4, and 7, as the indexing will move
down, causing an error in deletion.

Similarly, verified face will also be checked if it has started more than 2 second
ago and adding into a new list while removing from verifying face list.

    # add verified faces after appearing for more than 2 seconds (time can be change to your requirement)
    index = 0
    for (x, y, w, h, st, ls) in faces_verifying:
       if time.time() - st > 2: # started more than 2 seconds ago (time can be change to your requirement)
           faces_logging.append((x, y, w, h, st, ls))
           removeIndex.append(index) # remove from verifying list
       index += 1
    # remove face from list
    removeIndex.reverse()
    for x in removeIndex:
       faces_verifying.pop(x)
    # clear the removing list
    del removeIndex[:]

After the face is verified, you will begin to log the time when it disappear. Again,
it will find the face in the faces in the frame to check if it still exist, if it
doesn't and - this is important, fulfil the condition of not been last seen 2 seconds
ago, it will be removed. While removing, it will log the time it started and the
duration of the interaction.

    # removing verified face if not on screen
    index = 0
    for (x, y, w, h, st, ls) in faces_logging:
        faceExist = False
        for (x2, y2, w2, h2) in faces_onscreen:
            if not (w < x2) and not (x > w2) and not (h < y2) and not (y > h2):
                faceExist = True
                break
            if abs(x - x2) < 75 and abs(y - y2) < 75 and abs(w - w2) < 75 and abs(h - h2) < 75:
                faceExist = True
                break

        # remove face
        if not faceExist:
            if time.time() - ls > 2: # if last seen is more than 2seconds ago (time can be change to your requirement)
                removeIndex.append(index)
        index += 1


    # remove verified face and log to text file
    removeIndex.reverse()
    for x in removeIndex:
        sys.stdout.write("Person " + str(x) + " looked at screen for " + "{0:.2f}".format((time.time() - faces_logging[x][4] - 1.5)) + " second, started at " + str( datetime.datetime.fromtimestamp(faces_logging[x][4]).strftime('%Y-%m-%d %H:%M:%S')) + "\n")
        sys.stdout.flush()

        logfilePath.write(str(datetime.datetime.fromtimestamp(faces_logging[x][4]).strftime('%Y-%m-%d %H:%M:%S')) + " -> " + "{0:.2f}".format((time.time() - faces_logging[x][4] - 1.5)) + " seconds\n")

        faces_logging.pop(x)
    del removeIndex[:]

And all the logging is completed. The following code will just display the camera
and faces being verified for ease of knowing what is being verified. "del removeIndex[:]"
is there to empty out the list from current cycle - it can be placed at the top
or bottom of the loop.

    # draw bounding box around the face that are being log
    for (x, y, w, h, st, ls) in faces_logging:
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)


    # show the output frame ------------------------
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

And to quit the process, the following code will handle all the processes prior to quitting.

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # do a bit of cleanup
    logfilePath.close()
    cv2.destroyAllWindows()
    vs.stop()

A keypress is used to allow the user to quit the program easily. Everything will
be proceed before exiting the program to prevent errors.

Some useful resources while doing this project:

https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning

https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20170830

https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/

https://stackoverflow.com/questions/41912372/dlib-installation-on-windows-10

https://docs.opencv.org/3.4/d7/d8b/tutorial_py_face_detection.html

https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
