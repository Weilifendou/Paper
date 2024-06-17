from MVGigE import *


def OpenCam():
    r, hCam = MVOpenCamByIndex(0)  # 根据相机的索引返回相机句柄

    if hCam == 0:
        if r == MVST_ACCESS_DENIED:
            print('无法打开相机，可能正被别的软件控制!')
        else:
            print('无法打开相机!')
        return 0
    return hCam


def StartGrab(hCam):
    r, img = MVGetImgBuf(hCam)

    if img is None:
        print('error: ', r)
        MVCloseCam(hCam)
        return 0

    if MVStartGrabWindow(hCam) != MVST_SUCCESS:
        print("MVStartGrabWindow error")
        MVCloseCam(hCam)
        return 0

    return 1, img


def RealTime():
    hCam = OpenCam()
    if hCam == 0:
        quit()

    r, img = MVGetImgBuf(hCam)

    if img is None:
        print('error: ', r)
        MVCloseCam(hCam)
        return 0

    if MVStartGrabWindow(hCam) != MVST_SUCCESS:
        print("MVStartGrabWindow error")
        MVCloseCam(hCam)
        return 0
    while cv2.getWindowProperty('SeekBar', cv2.WND_PROP_VISIBLE) >= 1:
        res, id = MVGetSampleGrabBuf(hCam, img, 50)
        if res == MVST_SUCCESS:
            ProcessImage(img)
        if cv2.waitKey(1) == 27: break
    MVStopGrab(hCam)  # 停止采集
    MVCloseCam(hCam)