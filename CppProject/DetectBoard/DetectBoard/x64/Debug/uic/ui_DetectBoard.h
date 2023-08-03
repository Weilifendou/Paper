/********************************************************************************
** Form generated from reading UI file 'DetectBoard.ui'
**
** Created by: Qt User Interface Compiler version 5.12.6
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DETECTBOARD_H
#define UI_DETECTBOARD_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DetectBoardClass
{
public:
    QPushButton *lxcj;
    QLabel *label;

    void setupUi(QWidget *DetectBoardClass)
    {
        if (DetectBoardClass->objectName().isEmpty())
            DetectBoardClass->setObjectName(QString::fromUtf8("DetectBoardClass"));
        DetectBoardClass->resize(981, 787);
        lxcj = new QPushButton(DetectBoardClass);
        lxcj->setObjectName(QString::fromUtf8("lxcj"));
        lxcj->setGeometry(QRect(70, 30, 75, 23));
        label = new QLabel(DetectBoardClass);
        label->setObjectName(QString::fromUtf8("label"));
        label->setGeometry(QRect(40, 100, 881, 631));

        retranslateUi(DetectBoardClass);

        QMetaObject::connectSlotsByName(DetectBoardClass);
    } // setupUi

    void retranslateUi(QWidget *DetectBoardClass)
    {
        DetectBoardClass->setWindowTitle(QApplication::translate("DetectBoardClass", "DetectBoard", nullptr));
        lxcj->setText(QApplication::translate("DetectBoardClass", "\350\277\236\347\273\255\351\207\207\351\233\206", nullptr));
        label->setText(QApplication::translate("DetectBoardClass", "TextLabel", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DetectBoardClass: public Ui_DetectBoardClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DETECTBOARD_H
