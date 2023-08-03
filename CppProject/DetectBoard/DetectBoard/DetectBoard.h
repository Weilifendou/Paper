#pragma once

#include <QtWidgets/QWidget>
#include "ui_DetectBoard.h"


#include "MVImage.h"
#include "MVGigE.h"
#include "MVCamProptySheet.h"
#include <qtwinextras/qwinfunctions.h>
Q_GUI_EXPORT QPixmap qt_pixmapFromWinHBITMAP(HBITMAP bitmap, int hbitmapFormat = 0);

#ifdef _DEBUG
#define		CODEC(str)			QString::fromLocal8Bit(str)
#define		WARNINGS			CODEC("错误")
#define		INFORMATION			CODEC("提示")
#define		SAVEPATH			CODEC("选择保存位置")
#define		IMSG(str)			QMessageBox::information(0,INFORMATION,CODEC(str),QMessageBox::Ok);
#define		WMSG(str)			QMessageBox::warning(0,WARNINGS,CODEC(str),QMessageBox::Ok);
#else
#pragma comment(lib, "Shell32.lib")
#pragma execution_character_set("utf-8")
#define		WARNINGS			("错误")
#define		INFORMATION			("提示")
#define		SAVEPATH			("选择保存位置")
#define		IMSG(str)			QMessageBox::information(this,INFORMATION,(str),QMessageBox::Ok);
#define		WMSG(str)			QMessageBox::warning(this,WARNINGS,(str),QMessageBox::Ok);
#endif

class DetectBoard : public QWidget
{
    Q_OBJECT
public:
    DetectBoard(QWidget *parent = Q_NULLPTR);
	enum BUTTONPRESSTYPE { CONTINUITYSHOT = 0, SINGLESHOT, OUTSHOT, STOPSHOT, CAMERAERR };
	int OnStreamCB(MV_IMAGE_INFO *);


public slots:
	void on_lxcj_clicked();


protected:
	void init();
	void CameraInitAfter();
	void drawImageForWindow();

private:
    Ui::DetectBoardClass ui;

	DWORD	m_nIdxCam;
	HANDLE hCamera, m_hProperty;
	MVImage mImage, pDstImage;

	unsigned int shotNumber;
	short hImage, wImage;

	QImage tImage, m_ShowImage;
	bool saveImageFlag;
	QString saveImagePath;
	unsigned short displayScale;

	BUTTONPRESSTYPE currentModel;

	//计数互斥
	bool HCflag;
};
