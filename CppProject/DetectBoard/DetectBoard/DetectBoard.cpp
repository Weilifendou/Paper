#include "DetectBoard.h"

DetectBoard::DetectBoard(QWidget *parent)
    : QWidget(parent)
	, shotNumber(0)
	, saveImageFlag(false)
	, HCflag(true)
	, m_hProperty(nullptr)
	, displayScale(4)
{
    ui.setupUi(this);
	init();
}


void DetectBoard::init()
{
	bool MVInitLibState = false;
	if (MVInitLib() == MVST_SUCCESS) MVInitLibState = true;

	if (MVInitLibState)
	{
		int linkCameraNumber = 0;
		MVGetNumOfCameras(&linkCameraNumber);								//获取连接到计算机上的相机的数量
		if (linkCameraNumber)
		{
			bool MVInitLibLink = false;
			if (MVOpenCamByIndex(0, &hCamera) == MVST_SUCCESS) MVInitLibLink = true;
			if (MVInitLibLink) CameraInitAfter();
		}
	}
}

void DetectBoard::CameraInitAfter()
{
	int hCameraW, hCameraH;
	MVGetWidth(hCamera, &hCameraW);
	MVGetHeight(hCamera, &hCameraH);
	MV_PixelFormatEnums mPixelFormat;
	MVGetPixelFormat(hCamera, &mPixelFormat);
	mImage.CreateByPixelFormat(hCameraW, hCameraH, mPixelFormat);

	MVSetTriggerMode(hCamera, TriggerMode_On);
	MVSetTriggerSource(hCamera, TriggerSource_Software);

	if (m_hProperty == nullptr)
	{
		MVCamInfo CamInfo;
		MVGetCameraInfo(0, &CamInfo);
		MVCamProptySheetInit(&m_hProperty, hCamera, 0, LPCTSTR("Propty"));
	}
}

int __stdcall StreamCB(MV_IMAGE_INFO *pInfo, ULONG_PTR nUserVal)
{
	DetectBoard *pDlg = (DetectBoard *)nUserVal;
	return (pDlg->OnStreamCB(pInfo));
}

int DetectBoard::OnStreamCB(MV_IMAGE_INFO *pInfo)
{
	MVInfo2Image(hCamera, pInfo, &mImage);
	drawImageForWindow();
	return 0;
}

void DetectBoard::drawImageForWindow()
{
	wImage = mImage.GetWidth() / displayScale;
	hImage = mImage.GetHeight() / displayScale;
	m_ShowImage = qt_pixmapFromWinHBITMAP(mImage.GetHBitmap()).scaled(wImage, hImage, Qt::KeepAspectRatio).toImage();
	const QPixmap temp = QPixmap::fromImage(m_ShowImage);
	ui.label->setPixmap(temp);
}
void DetectBoard::on_lxcj_clicked()
{
	MVSetTriggerMode(hCamera, TriggerMode_Off);
	MVSetTriggerSource(hCamera, TriggerSource_Software);

	MVStartGrab(hCamera, StreamCB, (ULONG_PTR)this);
}
