/********************************************************************************
** Form generated from reading UI file 'Samples_Qt_TriggerCount.ui'
**
** Created by: Qt User Interface Compiler version 5.12.6
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SAMPLES_QT_TRIGGERCOUNT_H
#define UI_SAMPLES_QT_TRIGGERCOUNT_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <mvscrollarea.h>

QT_BEGIN_NAMESPACE

class Ui_Samples_Qt_TriggerCountClass
{
public:
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QPushButton *lxcj;
    QPushButton *rcf;
    QPushButton *wcf;
    QPushButton *tz;
    QSpacerItem *horizontalSpacer_3;
    QLabel *label_2;
    QComboBox *displayScale;
    QSpacerItem *horizontalSpacer_5;
    QPushButton *Set;
    QSpacerItem *horizontalSpacer_4;
    QPushButton *saveImage;
    QSpacerItem *horizontalSpacer_2;
    QLabel *label;
    QLabel *pssl;
    QSpacerItem *horizontalSpacer;
    MVScrollArea *scrollArea;
    QWidget *scrollAreaWidgetContents;
    QGridLayout *gridLayout;
    QLabel *showImage;

    void setupUi(QWidget *Samples_Qt_TriggerCountClass)
    {
        if (Samples_Qt_TriggerCountClass->objectName().isEmpty())
            Samples_Qt_TriggerCountClass->setObjectName(QString::fromUtf8("Samples_Qt_TriggerCountClass"));
        Samples_Qt_TriggerCountClass->resize(888, 760);
        Samples_Qt_TriggerCountClass->setStyleSheet(QString::fromUtf8("font: 75 10pt \"\345\276\256\350\275\257\351\233\205\351\273\221\";"));
        verticalLayout = new QVBoxLayout(Samples_Qt_TriggerCountClass);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(2, 6, 2, 2);
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        lxcj = new QPushButton(Samples_Qt_TriggerCountClass);
        lxcj->setObjectName(QString::fromUtf8("lxcj"));

        horizontalLayout->addWidget(lxcj);

        rcf = new QPushButton(Samples_Qt_TriggerCountClass);
        rcf->setObjectName(QString::fromUtf8("rcf"));

        horizontalLayout->addWidget(rcf);

        wcf = new QPushButton(Samples_Qt_TriggerCountClass);
        wcf->setObjectName(QString::fromUtf8("wcf"));

        horizontalLayout->addWidget(wcf);

        tz = new QPushButton(Samples_Qt_TriggerCountClass);
        tz->setObjectName(QString::fromUtf8("tz"));
        tz->setEnabled(false);

        horizontalLayout->addWidget(tz);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_3);

        label_2 = new QLabel(Samples_Qt_TriggerCountClass);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout->addWidget(label_2);

        displayScale = new QComboBox(Samples_Qt_TriggerCountClass);
        displayScale->addItem(QString());
        displayScale->addItem(QString());
        displayScale->addItem(QString());
        displayScale->addItem(QString());
        displayScale->setObjectName(QString::fromUtf8("displayScale"));

        horizontalLayout->addWidget(displayScale);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_5);

        Set = new QPushButton(Samples_Qt_TriggerCountClass);
        Set->setObjectName(QString::fromUtf8("Set"));

        horizontalLayout->addWidget(Set);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_4);

        saveImage = new QPushButton(Samples_Qt_TriggerCountClass);
        saveImage->setObjectName(QString::fromUtf8("saveImage"));
        saveImage->setEnabled(true);

        horizontalLayout->addWidget(saveImage);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);

        label = new QLabel(Samples_Qt_TriggerCountClass);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout->addWidget(label);

        pssl = new QLabel(Samples_Qt_TriggerCountClass);
        pssl->setObjectName(QString::fromUtf8("pssl"));
        pssl->setAlignment(Qt::AlignCenter);

        horizontalLayout->addWidget(pssl);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);


        verticalLayout->addLayout(horizontalLayout);

        scrollArea = new MVScrollArea(Samples_Qt_TriggerCountClass);
        scrollArea->setObjectName(QString::fromUtf8("scrollArea"));
        scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        scrollArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QString::fromUtf8("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 882, 715));
        gridLayout = new QGridLayout(scrollAreaWidgetContents);
        gridLayout->setSpacing(0);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setContentsMargins(-1, 0, 0, 0);
        showImage = new QLabel(scrollAreaWidgetContents);
        showImage->setObjectName(QString::fromUtf8("showImage"));
        showImage->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(showImage, 0, 0, 1, 1);

        scrollArea->setWidget(scrollAreaWidgetContents);

        verticalLayout->addWidget(scrollArea);


        retranslateUi(Samples_Qt_TriggerCountClass);

        displayScale->setCurrentIndex(1);


        QMetaObject::connectSlotsByName(Samples_Qt_TriggerCountClass);
    } // setupUi

    void retranslateUi(QWidget *Samples_Qt_TriggerCountClass)
    {
        Samples_Qt_TriggerCountClass->setWindowTitle(QApplication::translate("Samples_Qt_TriggerCountClass", "Samples_Qt_TriggerCount", nullptr));
        lxcj->setText(QApplication::translate("Samples_Qt_TriggerCountClass", "\350\277\236\347\273\255\351\207\207\351\233\206", nullptr));
        rcf->setText(QApplication::translate("Samples_Qt_TriggerCountClass", "\350\275\257\350\247\246\345\217\221", nullptr));
        wcf->setText(QApplication::translate("Samples_Qt_TriggerCountClass", "\345\244\226\350\247\246\345\217\221", nullptr));
        tz->setText(QApplication::translate("Samples_Qt_TriggerCountClass", "\345\201\234\346\255\242", nullptr));
        label_2->setText(QApplication::translate("Samples_Qt_TriggerCountClass", "\346\230\276\347\244\272\346\257\224\344\276\213:", nullptr));
        displayScale->setItemText(0, QApplication::translate("Samples_Qt_TriggerCountClass", "20%", nullptr));
        displayScale->setItemText(1, QApplication::translate("Samples_Qt_TriggerCountClass", "25%", nullptr));
        displayScale->setItemText(2, QApplication::translate("Samples_Qt_TriggerCountClass", "50%", nullptr));
        displayScale->setItemText(3, QApplication::translate("Samples_Qt_TriggerCountClass", "100%", nullptr));

        Set->setText(QApplication::translate("Samples_Qt_TriggerCountClass", "\345\217\202\346\225\260\350\256\276\347\275\256", nullptr));
        saveImage->setText(QApplication::translate("Samples_Qt_TriggerCountClass", "\344\277\235\345\255\230\345\233\276\347\211\207", nullptr));
        label->setText(QApplication::translate("Samples_Qt_TriggerCountClass", "\346\213\215\346\221\204\346\225\260\351\207\217:", nullptr));
        pssl->setText(QApplication::translate("Samples_Qt_TriggerCountClass", "0", nullptr));
        showImage->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class Samples_Qt_TriggerCountClass: public Ui_Samples_Qt_TriggerCountClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SAMPLES_QT_TRIGGERCOUNT_H
