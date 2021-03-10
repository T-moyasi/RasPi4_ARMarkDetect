import cv2
import numpy as np
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
cap = cv2.VideoCapture(0)
ColorCyan = (255, 255, 0)
ColorRed = (0,0,255)
ColorBlack = (0,0,0)
cameraMatrix=np.array(
        [[516.47230949,   0.        , 271.67603794],
       [  0.        , 515.13458388, 254.19658506],
       [  0.        ,   0.        ,   1.        ]])
distCoeffs = np.array( [-0.05112526, -0.14365194,  0.00367056, -0.01412392,  0.11600701])

def detectMarks(frame):
    #最小辺を指定したくなったとき用
    MIN_EDGE = 25
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # オープニング・クロージングによるノイズ除去
    element8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
    oc = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, element8)
    oc = cv2.morphologyEx(oc, cv2.MORPH_CLOSE, element8)

    # マーカ検出
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary) 
    print("{} detect.".format(len(corners)))
    if len(corners) > 0:
        # 検出したマーカに枠を書く
        #aruco.drawDetectedMarkers(frame, corners, ids, ColorRed)
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.27, cameraMatrix, distCoeffs)
        #z-axisのネガポジ判定
        PorN = 0
        for i in range(len(corners)):
            if(rvec[i][0][2]<0):
                PorN = PorN-1
            if(rvec[i][0][2]>0):
                PorN = PorN+1
        for i in range(len(corners)):
            if(PorN > 0):
                #print("posi nega : Positive")
                if(rvec[i][0][2]<0):
                    rvec[i][0][2]=-rvec[i][0][2]
            if(PorN < 0):
                #print("posi nega : Negative")
                if(rvec[i][0][2]>0):
                    rvec[i][0][2]=-rvec[i][0][2] 
            #print(rvec[i][0][2])
        for i in range(len(corners)):
            #aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec[i][0], tvec[i][0], 1)
            if ids[i] == 0:
                euler1 = rodrigues2euler(rvec[i][0],tvec[i][0])
                print(ids[i],corners[i][0][0],euler1)
                point1 = tuple(corners[i][0][0].astype(np.int))
            if ids[i] == 1:
                euler2 = rodrigues2euler(rvec[i][0],tvec[i][0])
                print(ids[i],corners[i][0][0],euler2)
                point2 = tuple(corners[i][0][1].astype(np.int))
            if ids[i] == 2:
                euler3 = rodrigues2euler(rvec[i][0],tvec[i][0])
                print(ids[i],corners[i][0][0],euler3)
                point3 = tuple(corners[i][0][3].astype(np.int))
            if ids[i] == 3:
                euler4 = rodrigues2euler(rvec[i][0],tvec[i][0])
                print(ids[i],corners[i][0][0],euler4)
                point4 = tuple(corners[i][0][2].astype(np.int))
            if ids[i] == 4:
                euler5 = rodrigues2euler(rvec[i][0],tvec[i][0])
                print(ids[i],corners[i][0][0],euler5)
                point5 = tuple(corners[i][0][2].astype(np.int))
        for i ,corner in enumerate(corners):
            points = corner[0].astype(np.double)
            point = [0,0]
            for j in range(4):
                point +=points[j]
            point /= 4
            #print(point)
        if len(corners) == 5:
            x=mean(point1[0],point2[0],point3[0],point4[0],point5[0])
            y=mean(point1[1],point2[1],point3[1],point4[1],point5[1])
            point6 = tuple((x,y))
            return point1,point2,point3,point4,point5,point6
        else:
            return None
    else:
        return None

#Line Segment
def LSeg(point1,point2):
    return ((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)**0.5

#Heron's formula
def heron(point1,point2,point3):
    L1 = LSeg(point1,point2)
    L2 = LSeg(point2,point3)
    L3 = LSeg(point3,point1)
    s = (L1+L2+L3)/2
    return (s*(s-L1)*(s-L2)*(s-L3))**0.5

#Calculate the area of the rectangle from 4 points
def calArea(point1,point2,point3,point4):
    area = heron(point1,point2,point3)+heron(point2,point3,point4)
    return area

def rodrigues2euler(rvec,tvec):
    rvec_matrix = cv2.Rodrigues(rvec)
    rvec_matrix = rvec_matrix[0]
    t_tvec =tvec[np.newaxis,:].T
    com_matrix = np.hstack((rvec_matrix,t_tvec))
    euler2 = cv2.decomposeProjectionMatrix(com_matrix)[6]
    euler = np.array([euler2[0][0]+180,euler2[1][0]+180,euler2[2][0]+180])
    #print(euler)
    return euler

def mean(*args):
    sum = 0.0
    for i in args:
        sum += i
    return sum/len(args)

def main():
    print(cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320))
    print(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240))
    baseArea = 0.0
    while True:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        points = detectMarks(frame)

        if points is not None:
            #cv2.rectangle(frame, tuple(points[0]), tuple(points[3]), ColorBlack,10)
            cv2.line(frame,tuple(points[0]),tuple(points[1]),ColorBlack,thickness=4, lineType=cv2.LINE_AA)
            cv2.line(frame,tuple(points[0]),tuple(points[2]),ColorBlack,thickness=4, lineType=cv2.LINE_AA)
            cv2.line(frame,tuple(points[1]),tuple(points[3]),ColorBlack,thickness=4, lineType=cv2.LINE_AA)
            cv2.line(frame,tuple(points[2]),tuple(points[3]),ColorBlack,thickness=4, lineType=cv2.LINE_AA)
            area=calArea(points[0],points[1],points[2],points[3])
            print("x,y:",points[5])
            if(baseArea == 0.0):
                baseArea=area
                print(baseArea)
            
            if((baseArea-area)>100):
                print("z:minus")
            elif((baseArea-area)<-100):
                print("z:plus")
            else:
                print("z:zero")
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()               
          
if __name__ == '__main__' :
    main()