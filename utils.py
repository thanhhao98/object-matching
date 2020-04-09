import cv2
import numpy as np
def featureMatching(dataImg1,dataImg2,minMatchCount=10,flannIndexKdtree=1):
    img1, kp1, des1 = dataImg1
    img2, kp2, des2 = dataImg2
    index_params = dict(algorithm = flannIndexKdtree, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(des1,des2,k=2)
    except:
        return None, None, None
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>minMatchCount:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w,d = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        try:
            dst = cv2.perspectiveTransform(pts,M)
        except Exception as e:
            dst = None
    else:
        dst = None
    return dst, good, matches

def templateMatching(template,img, method):
    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    return top_left