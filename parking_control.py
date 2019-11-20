# This is the main code to run the pi car.

import time

import numpy as np

import driver

DEBUG = False

OFFSET = 371

LOT_LENGTH = 52
LOT_WIDTH = 26


def park(X, Y, angle=0):
    """ Parks the car to the lot.

    Args:
        X: bias in x-axis
        Y: bias in y-axis
        angle: bias of angle

    Returns:
        None
    """
    R = 29.5  # 需要调参，与servo参数相匹配

    k = 0.1
    k1 = 2
    # Y=Y-2;

    gama = (Y-R*np.sin(angle))/(X-R-R*np.cos(angle))
    X1 = ((X-R-R*np.cos(angle))**2)/(R**2)

    n = 0

    while ((np.abs(k1-k)/np.abs(k)) > 0.0001)and n < 30:
        k = k1
        FK = k**4+(8-X1*(gama**2+1))*k**2+(16-(4*gama**2+4)*X1)
        F1K = 4*k**3+2*(8-X1*(gama**2+1))*k
        k1 = k-(FK/F1K)
        n = n+1

    k = k1
    l = k*R
    tana = (k+2*gama)/(gama*k-2)
    a = np.arctan(tana)
    print("a=", a)
    print("l=", l)
    print("angle=", angle)

    # a=np.pi*0
    # l=10

    if n >= 30:
        a = 0.995807703453048
        l = 6.708203932504121
        print("[warning] ----------------no answer------------------------")
    else:
        print("----------------- solution found --------------------")

    dist1f = (a+angle)*R
    dist2f = l
    dist3f = a*R

    dist1 = int(178*np.abs(dist1f) + 63)
    dist2 = int(178*np.abs(dist2f) + 63)
    dist3 = int(178*np.abs(dist3f) + 63)

    sign1 = np.abs(dist1f)/dist1f
    sign2 = np.abs(dist2f)/dist2f
    sign3 = np.abs(dist3f)/dist3f

    kt = 0.35  # 需要调参，将dist转化为时间参数,与motor参数相匹配

    time1 = kt*dist1f+1
    time2 = np.abs(kt*dist2f)+1
    time3 = kt*dist3f+1

    time_gap = 0.7

    print("distant1=", dist1)
    print("distant2=", dist2)
    print("distant3=", dist3)

    print("distant1f=", dist1f)
    print("distant2f=", dist2f)
    print("distant3f=", dist3f)

    print("inX=", X)
    print("inY=", Y)
    print("calX=", R-2*R*np.cos(a)+R*np.cos(angle)+l*np.sin(a))
    print("calY=", 2*R*np.sin(a)+R*np.sin(angle)+l*np.cos(a))

    motorspeed = -0.05

    d = driver.driver()

    d.setStatus(motor=0, servo=-1,  dist=0, mode="distance")
    time.sleep(1)
    d.setStatus(motor=motorspeed*sign1, servo=-1,  dist=dist1, mode="distance")
    time.sleep(time1)

    d.setStatus(motor=0, servo=0,  dist=0, mode="distance")
    time.sleep(time_gap)
    d.setStatus(motor=motorspeed*sign2, servo=0,  dist=dist2, mode="distance")
    time.sleep(time2)

    d.setStatus(motor=0, servo=1,  dist=0, mode="distance")
    time.sleep(time_gap)
    d.setStatus(motor=motorspeed*sign3, servo=1,  dist=dist3, mode="distance")
    time.sleep(time3)

    d.setStatus(motor=0.0, servo=0.0, dist=0x00, mode="stop")
    # time.sleep(time_gap);
    d.close()
    del d
    print("==========piCar Client Fin==========")


def reverse(X, Y, angle=0):
    """ Reverse operation of the park()

    Args:
        X: bias in x-axis
        Y: bias in Y-axis
        angle: bias in angle

    Returns:
        None
    """
    R = 29.5  # 需要调参，与servo参数相匹配

    k = 0.1
    k1 = 2

    gama = (Y-R*np.sin(angle))/(X-R-R*np.cos(angle))
    X1 = ((X-R-R*np.cos(angle))**2)/(R**2)

    n = 0

    while ((np.abs(k1 - k)/np.abs(k)) > 0.0001) and n < 30:
        k = k1
        FK = k**4 + (8 - X1*(gama**2 + 1))*k**2 + (16 - (4*gama**2 + 4)*X1)
        F1K = 4*k**3 + 2*(8 - X1*(gama**2 + 1))*k
        k1 = k - (FK/F1K)
        n = n + 1

    k = k1
    l = k*R
    tana = (k+2*gama)/(gama*k - 2)
    a = np.arctan(tana)
    print("a=", a)
    print("l=", l)
    print("angle=", angle)

    if n >= 30:
        a = 0.995807703453048
        l = 6.708203932504121
        print("[warning] ----------------no answer------------------------")
    else:
        print("----------------- solution found --------------------")

    dist1f = (a+angle)*R
    dist2f = l
    dist3f = a*R

    dist1 = int(178*np.abs(dist1f) + 63)
    dist2 = int(178*np.abs(dist2f) + 63)
    dist3 = int(178*np.abs(dist3f) + 63)

    sign1 = np.abs(dist1f)/dist1f
    sign2 = np.abs(dist2f)/dist2f
    sign3 = np.abs(dist3f)/dist3f

    kt = 0.35  # 需要调参，将dist转化为时间参数,与motor参数相匹配

    time1 = kt*dist1f + 1
    time2 = np.abs(kt*dist2f) + 1
    time3 = kt*dist3f + 1

    time_gap = 0.7

    print("distant1=", dist1)
    print("distant2=", dist2)
    print("distant3=", dist3)

    print("distant1f=", dist1f)
    print("distant2f=", dist2f)
    print("distant3f=", dist3f)

    print("inX=", X)
    print("inY=", Y)
    print("calX=", R - 2*R*np.cos(a)+R*np.cos(angle) + l*np.sin(a))
    print("calY=", 2*R*np.sin(a) + R*np.sin(angle) + l*np.cos(a))

    motorspeed = 0.05

    d = driver.driver()

    d.setStatus(motor=0, servo=1, dist=0, mode="distance")
    time.sleep(1)
    d.setStatus(motor=motorspeed*sign1, servo=1, dist=dist1, mode="distance")
    time.sleep(time1)

    d.setStatus(motor=0, servo=0,  dist=0, mode="distance")
    time.sleep(time_gap)
    d.setStatus(motor=motorspeed*sign2, servo=0, dist=dist2, mode="distance")
    time.sleep(time2)

    d.setStatus(motor=0, servo=-1,  dist=0, mode="distance")
    time.sleep(time_gap)
    d.setStatus(motor=motorspeed*sign3, servo=-1, dist=dist3, mode="distance")
    time.sleep(time3)

    d.setStatus(motor=0.0, servo=0.0, dist=0x00, mode="stop")
    time.sleep(time_gap)
    d.close()
    del d
    print("==========piCar Client Fin==========")
