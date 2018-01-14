import cv2
import numpy as np
import math

def main():
    vc = cv2.VideoCapture(1)
    retval, frame = vc.read()
    while retval:
        cv2.imshow("Cryo", frame)

        # Grab next frame from webcam
        retval, frame = vc.read()

        if cv2.waitKey(1) == 27:
            break

def det_blobs(dev=1):
    # detect blobs in the frame fr
    # return the number of blobs
    # blurred = cv.GaussianBlur(fr, (3, 3), 0)

    # Setup BlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 2000
    params.maxArea = 4000000
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.3
    
    # Filter by Convexity
    params.filterByConvexity = False
    #params.minConvexity = 0.87
    
    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.6

    # Distance Between Blobs
    # params.minDistBetweenBlobs = 20
    
    # Create a detector with the parameters
    # Create a detector with the parameters
    opencv_version = (cv2.__version__).split('.')
    if int(opencv_version[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)

    camera = cv2.VideoCapture(dev)   # 0 on Dragon and 1 on laptop

    while camera.isOpened():
        retval, im = camera.read() # 720 x 1280 camera rgb sensor
        im = im[:, 200:1042, :]
        overlay = im.copy()

        h,w,ch = im.shape
        # print('im size: ({}, {})'.format(h,w))

        blurred = cv2.GaussianBlur(im, (7, 7), 0)
        keypoints = detector.detect(blurred)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # for k in keypoints:
        #     cv2.circle(overlay, (int(k.pt[0]), int(k.pt[1])), int(k.size/2), (0, 0, 255), -1)
        #     cv2.line(overlay, (int(k.pt[0])-20, int(k.pt[1])), (int(k.pt[0])+20, int(k.pt[1])), (0,0,0), 3)
        #     cv2.line(overlay, (int(k.pt[0]), int(k.pt[1])-20), (int(k.pt[0]), int(k.pt[1])+20), (0,0,0), 3)

        opacity = 0.5
        cv2.addWeighted(overlay, opacity, im, 1 - opacity, 0, im)

	    # Uncomment to resize to fit output window if needed
	    #im = cv2.resize(im, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        cv2.imshow("Output", im)

        k = cv2.waitKey(20) & 0xff
        if k == 27:
        	break

    camera.release()
    cv2.destroyAllWindows()

def colorDistance(kCol, uCol):
    # compute sum of square of differences between each channel 
    d = (kCol[0] - uCol[0])**2 + (kCol[1] - uCol[1])**2 + \
        (kCol[2] - uCol[2])**2
        
    # square root of sum is the Euclidean distance between the
    # two colors
    return math.sqrt(d)

def det_apples_oranges(dev=1):
    # detect blobs in the frame fr
    # return the number of blobs
    # blurred = cv.GaussianBlur(fr, (3, 3), 0)
    opencv_version = (cv2.__version__).split('.')

    camera = cv2.VideoCapture(dev)   # 0 on Dragon and 1 on laptop

    while camera.isOpened():
        retval, im = camera.read() # 720 x 1280 camera rgb sensor
        im = im[0:635, 200:1042, :]
        overlay = im.copy() # for display of what is apple and orange

        h,w,ch = im.shape
        # print('im size: ({}, {})'.format(h,w))

        blurred = cv2.GaussianBlur(im, (7, 7), 0)
        
        blurHSV = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        t = 145
        (t, binary) = cv2.threshold(blurHSV[:, :, 1], t, 255, cv2.THRESH_BINARY)

        if int(opencv_version[0]) < 3:
            (contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elif:
            (_, contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        avg_len = np.mean([len(c) for c in contours])
        print('averange length of contours {}'.format(avg_len))
        filtered_contours = [c for c in contours if len(c) >= 100]
        print('{} good contours.'.format(len(filtered_contours)))

        apple = (45, 89, 153)
        orange = (31, 81, 164)

        appleCount = 0
        orangeCount = 0

        for i, c in enumerate(filtered_contours):
            print('Contour {}'.format(i))
    
            # find centroid of shape
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            # find average color for 9 pixel kernel around centroid
            b = im[cy - 4 : cy + 5, cx - 4 : cx + 5, 0]
            g = im[cy - 4 : cy + 5, cx - 4 : cx + 5, 1]
            r = im[cy - 4 : cy + 5, cx - 4 : cx + 5, 2]

            bAvg = np.mean(b)
            gAvg = np.mean(g)
            rAvg = np.mean(r)
    
            # find distances to known reference colors
            dist = []
            dist.append(colorDistance(apple, (bAvg, gAvg, rAvg)))
            dist.append(colorDistance(orange, (bAvg, gAvg, rAvg)))
    
            (x, y, w, h) = cv2.boundingRect(c)
    
            # which one is closest?
            minDist = min(dist)
            # if it was yellow, count the shape
            if dist[0] == minDist:
                # apple green rectangle.
                appleCount += 1
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)
            elif dist[1] == minDist:
                # orange - yellow
                orangeCount += 1
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 255), 3)

        cv2.imshow("Output", overlay)

        k = cv2.waitKey(20) & 0xff
        if k == 27:
        	break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # main()
    # det_blobs()
    det_apples_oranges()
