#include "DetectBoard.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    DetectBoard w;
    w.show();
    return a.exec();
}
