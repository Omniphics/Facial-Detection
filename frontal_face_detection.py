#import necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# packages for logging and flushing to cmd screen on runtime
import sys # flushing to cmd screen
import datetime # logging the time of face detection and last seen

# the path of the log file
logfilePath = open("logtime.txt", "a+") # will automatically create or append to existing

# variables to keep track of faces and index to be removed from list
faces_onscreen = [] # store all the faces meeting the condition appearing in that current loop of frame
faces_verifying = [] # store all the first encounter of faces to be verify
faces_logging = [] # store all the faces to be logged with timestamp
removeIndex = [] # store the indexes to be remove in a list

# allowing arguments to be passed when executing script
ap = argparse.ArgumentParser()
ap.add_argument("-p","--shape-predictor",required=True,help="path to facial landmark predictor")
args = vars(ap.parse_args())

# dlib calls to detect frontal face and eyes (optional - 2nd verification)
detect_face = dlib.get_frontal_face_detector()
predict_eyes = dlib.shape_predictor(args["shape_predictor"]) # the file is passed when calling the script: python frontal_face_detection.py --shape-predictor shape_predictor_68_face_landmarks.dat

# grabbing the indexes of the facial landmarks for the left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# starting the video stream thread
vs = VideoStream(src=0).start()
time.sleep(2.0) # time delay to let devices warm up

# looping over each frames from the video stream
while True:
    # grab the current frame from video to be process
    frame = vs.read()
    frame = imutils.resize(frame, width=450) # resizing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale for lower processing demand


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

    del faces_onscreen[:] # reset list for next cycle


    # draw bounding box around the face that are being log
    for (x, y, w, h, st, ls) in faces_logging:
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)


    # show the output frame ------------------------
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
logfilePath.close()
cv2.destroyAllWindows()
vs.stop()
