import cv2
import cvzone
import mediapipe
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640,360,[20,50],invert=True)

# Left Eye IDs + Right Eye IDs
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243,  # Left Eye
          263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398] # Right Eye
ratioList = []

blinkCounter = 0
counter = 0 # a helper variable to avoid multiple counts for a single blink
color = (255,0,255)
while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    success, img = cap.read()
    img,faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img,face[id],5,color,cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]

        leftLeft = face[130]
        leftRight = face[243]
        verLength, _ = detector.findDistance(leftUp,leftDown) #vertical length
        horLength, _ = detector.findDistance(leftLeft,leftRight)#horizontal length

        cv2.line(img,leftUp,leftDown,(0,200,0),3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

        ratio = int((verLength/horLength)*100)
        ratioList.append(ratio)
        if len(ratioList) > 3:   #to make the blink detection more stable and less noisy
            ratioList.pop(0)
        ratioAverage = sum(ratioList)/len(ratioList)

        if ratioAverage < 30 and counter ==0:
            blinkCounter += 1
            color = (0,200,0)
            counter = 1
        if counter !=0:
            counter +=1
            if counter > 10:
                counter =0
                color = (255,0,255)

        cvzone.putTextRect(img,f'Blink Count:{blinkCounter}',(100,100), colorR= color)

        imgPlot = plotY.update(ratioAverage, color)
        #cv2.imshow("ImagePlot", imgPlot)

        img = cv2.resize(img,(640,360))
        imgStack = cvzone.stackImages([img,imgPlot],1,2)

    else:
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, img], 1, 2)

    #img = cv2.resize(img,(640,360))
    cv2.imshow("Image", imgStack)
    key = cv2.waitKey(25)
    
    # Close frame when close button is clicked or 'q' is pressed
    if key == ord('q') or cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()