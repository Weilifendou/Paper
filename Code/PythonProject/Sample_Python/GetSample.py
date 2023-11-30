from MVGigE import *
import time
import numpy as np
import cv2
import datetime
import argparse


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

	return 1,img
	
def CapBin():
	print('press Esc to quit')
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

	while True:
		res, id = MVGetSampleGrabBuf(hCam, img, 10)
		if res == MVST_SUCCESS:
			raw = cv2.resize(img, (1200, 800))
			raw = cv2.flip(raw, -1) #参数为0上下翻转，1为左右翻转，-1为上下左右均翻转
			cv2.imshow('img',raw)
			if cv2.waitKey(1) == 27:
				break
		else:
			print(res)

	MVStopGrab(hCam)  # 停止采集
	MVCloseCam(hCam)

if __name__ == '__main__':
	CapBin()