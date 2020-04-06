import math
import numpy as np
rbox1=[0,0,5,20]
rbox2=[1,2,20,1]
p11=[0,0]
p12=[1,0]
p21=[0,1]
p22=[1,1]
weighIOU, weightDIOU, weightAIOU = 0.8, 0.1, 0.1
distance_of_central = ((rbox1[0] - rbox2[0]) ** 2 + (rbox1[1] - rbox2[1]) ** 2) ** 0.5
diagonal_lenth = distance_of_central + (((rbox1[2]) ** 2 + (rbox2[0]) ** 2) ** 0.5) * 0.5 + (
            ((rbox2[2]) ** 2 + (rbox2[0]) ** 2) ** 0.5) * 0.5

# IOU = area_inter / (area1 + area2 - area_inter)
IOU = 0.55
DIOU = 1 - distance_of_central ** 2 / diagonal_lenth ** 2
AIOU = 1 - (4 / (math.pi ** 2)) * (math.atan(rbox1[2] / rbox1[3]) - math.atan(rbox2[2] / rbox2[3])) ** 2
# print(IOU,DIOU,AIOU)

# print(IOU * weighIOU + DIOU * weightDIOU + AIOU * weightAIOU)

def a(rbox1,rbox2):
    if rbox1[2] >= rbox1[3]:
       long1 = rbox1[2]
    else :
       long1 = rbox1[3]
    if rbox2[2] >= rbox2[3]:
       long2 = rbox1[2]
    else :
       long2 = rbox1[3]
    # print(long1,long2)
    # return print(1 - (((long1-long2)**2)**0.5+0 + distance_of_central) / 0.5)
a(rbox1,rbox2)

def b():
    # 两点间的距离
    d1= ((p11[0]-p21[0])**2+(p11[1]-p21[1])**2)**0.5
    d2= ((p11[0]-p22[0])**2+(p11[1]-p22[1])**2)**0.5
    d3= ((p12[0]-p21[0])**2+(p12[1]-p21[1])**2)**0.5
    d4= ((p12[0]-p22[0])**2+(p12[1]-p22[1])**2)**0.5

    l1= ((p11[0]-p12[0])**2+(p11[1]-p12[1])**2)**0.5
    l2= ((p21[0]-p22[0])**2+(p21[1]-p22[1])**2)**0.5

    d_translation=(d1+d2+d3+d4)/4-(l1+l2)/2 # ? 2 ro 4 ?

    d_closestpoint=2


    theta=1
    d_angle=min(l1,l2)*math.sin(theta)
    # print(d1,d2,d3,d4,l1,l2)
    # print(math.sin(theta),d_angle)

    d_st12=d_closestpoint+0.25*d_angle+d_translation
    d_st21 = d_closestpoint + 0.25 * d_angle + d_translation

    d_srtaightLin=min(d_st12,d_st21)
    print(d_st12,d_srtaightLin)
b()

# 求两条线的角度
def slope(x1, y1, x2, y2):  # Line slope given two points:
    return (y2 - y1) / ((x2 - x1)+0.000001)

def angle(s1, s2):
    return math.degrees(math.atan((s2 - s1) / (1 + (s2 * s1))))

lineA = ((0,0), (0,1))
lineB = ((0,0), (1,0))

slope1 = slope(lineA[0][0], lineA[0][1], lineA[1][0], lineA[1][1])
slope2 = slope(lineB[0][0], lineB[0][1], lineB[1][0], lineB[1][1])

ang = angle(slope1, slope2)
theta=ang*2*math.pi/360

# 求点到直线的距离,从P3垂直到P1和P2之间绘制的线的距离
p1=np.array([0,0])
p2=np.array([0,10])
p3=np.array([5,7])
d=abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))
print(d,"3333")
