from MVGigE import *
import time
import numpy as np
import cv2
import msvcrt

imgShow=np.zeros((1,1,1), np.uint8)

def callback(info, value):
	global cnt
	global imgShow

	image,id = MVInfo2Img(info)

	if imgShow.shape[0] == 1:
		imgShow = image.copy()
	else:
		imgShow = image

	return 0

def testCB():
	global imgShow
	r, hCam = MVOpenCamByIndex(0)  # 根据相机的索引返回相机句柄
	
	if(hCam == 0):
		if(r == MVST_ACCESS_DENIED):
			print('无法打开相机，可能正被别的软件控制!')
			return
		else:
			print('无法打开相机!')
			return
	
	cb = MVStreamCB(callback)

	if MVStartGrab(hCam, cb, hCam) != MVST_SUCCESS:
		print("StartGrab error")
		MVCloseCam(hCam)
		return

	cv2.namedWindow("image")
	while(True):
		if imgShow.shape[0] != 1:
			cv2.imshow('image', imgShow)
		key = cv2.waitKey(10)
		if key != -1:
			break

	MVStopGrab(hCam)  # 停止采集
	MVCloseCam(hCam)
	return

if __name__ == '__main__':
	testCB()