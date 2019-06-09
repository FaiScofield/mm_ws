#include <time.h>
#include <iostream>
#include <vector>
//#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <boost/filesystem.hpp>

#include <QMainWindow>
#include <QLabel>
#include <QApplication>

using namespace cv;
using namespace std;

const int imgRows = 64;
const int imgCols = 870;
const bool saveDepthMap = false;
const string outputFolder = "/home/vance/output/kitti_04_dm/";

bool fixPixelThisRow(Mat& src, int row, int col) {
    Mat thisRow;
    src.row(row).copyTo(thisRow);

    vector<bool> rowFlag(imgCols, false);
    int s = 0, e = imgCols - 1;
    for (int c = 0; c < imgCols; ++c) {
        if (thisRow.data[c] != 0) {
            rowFlag[c] = true;
        }
    }
    for (int i = 0; i < col; ++i) {
        if (rowFlag[i])
            s = i;
    }
    for (int i = imgCols - 1; i > col; --i) {
        if (rowFlag[i])
            e = i;
    }
    if (s == 0 || e == imgCols - 1) {
//        printf("skip this (%d, %d) for start = %d, end = %d\n", row, col, s, e);
        return false;
    }

    int m = e - s;
    uchar delta = thisRow.data[e] - thisRow.data[s];
    for (int i = s + 1; i < e; ++i) {
        thisRow.data[i] = thisRow.data[s] + static_cast<uchar>((i-s)*delta/m);
    }
    thisRow.copyTo(src.row(row));

    return true;
}

bool fixPixelThisCol(Mat& src, int row, int col) {
    Mat thisCol;
    src.col(col).copyTo(thisCol);

    vector<bool> colFlag(imgRows, false);
    int s = 0, e = imgRows - 1;
    for (int r = 0; r < imgRows; ++r) {
        if (thisCol.data[r] != 0) {
            colFlag[r] = true;
        }
    }

    for (int i = 0; i < row; ++i) {
        if (colFlag[i])
            s = i;
    }
    for (int i = imgRows - 1; i > row; --i) {
        if (colFlag[i])
            e = i;
    }
    if (s == 0 || e == imgRows - 1) {
//        printf("skip this (%d, %d) for start = %d, end = %d\n", row, col, s, e);
        return false;
    }

    int m = e - s;
    uchar delta = thisCol.data[e] - thisCol.data[s];
    for (int i = s + 1; i < e; ++i) {
        thisCol.data[i] = thisCol.data[s] + static_cast<uchar>((i-s)*delta/m);
    }
    thisCol.copyTo(src.col(col));

    return true;
}

// 将一个深度图可视化
void ShowMatWithPng(const Mat& src,  QLabel* lImage)
{
    // 找最大最小值
    Mat abs;
    double minVal, maxVal;
    minMaxIdx(src, &minVal, &maxVal);
    convertScaleAbs(src, abs, 255 / maxVal);

    // 归一化到[0,1]内并赋予颜色
    QImage qimage(imgCols, imgRows, QImage::Format_RGB16);
    for (int row=0; row<imgRows; ++row) {
      for (int col=0; col<imgCols; ++col) {
        uchar v = src.at<uchar>(row, col);
        if ((v < minVal) || (v > maxVal)) {
          qimage.setPixel(col, row, QColor::fromHsv(0, 0, 0).rgb() ); // H, S, V
        } else {
          // cut off value, so it is within [minVal...maxVal]
          double d = 1.0 - (v - minVal)/(maxVal - minVal); // scale to [0...1]
          qimage.setPixel(col, row, QColor::fromHsv(359.0 - d*d*d*359.0, 255.0, d*255.0).rgb() ); // H, S, V
        }
      }
    }

    lImage->setPixmap(QPixmap::fromImage(qimage));
    lImage->show();
}

// 自定义均值滤波
void interpolation(Mat& image, int row, int col, int cellSize)
{
    if (image.ptr(row)[col] != 0)
        return;

    size_t validNeighbourNum = 0;
    uchar value = 0;
    int s, e;
    if (cellSize % 2 == 0)
        s = -cellSize/2;
    else
        s = -(cellSize-1)/2;
    e = cellSize + s - 1;

    for (int i = s; i <= e; ++i) {
        for (int j = s; j <= e; ++j) {
            if (row + i >= 0 && row + i < image.rows &&
                col + j >= 0 && col + j < image.cols &&
                image.ptr(row+i)[col+j] != 0 ) {
                value += image.ptr(row+i)[col+j];
                validNeighbourNum++;
            } else
                continue;
        }
    }

    if (validNeighbourNum != 0)
        image.ptr(row)[col] = static_cast<uchar>(value/validNeighbourNum);
}

// 线性插值
void linearInterpolation(const Mat& src, Mat& output) {
    src.copyTo(output);
    for (int r = 8; r < imgRows; r++) {
        for (int c = 0; c < imgCols; c++) {
            if (src.at<uchar>(r,c) == 0) {
                bool fix = fixPixelThisCol(output, r, c);
                if (!fix)
                    fix = fixPixelThisRow(output, r, c);
                if (!fix)
                    interpolation(output, r, c, 5);
            }
        }
    }
}



int main(int argc, char **argv)
{
    QApplication a(argc, argv);

    if (argc < 2) {
        printf("Usage: %s <depth image folder>\n", argv[0]);
        return -1;
    }

    // read folder images
    string folder = argv[1];
    boost::filesystem::path path(folder);
    if (!boost::filesystem::exists(path)) {
        cerr << argv[1] << " Folder dons't exist!\n";
        return -1;
    }

    vector<string> files;
    boost::filesystem::directory_iterator end_iter;
    for (boost::filesystem::directory_iterator iter(path);
         iter != end_iter; ++iter) {
        if (boost::filesystem::is_regular_file(iter->status()))
            files.push_back(iter->path().string());

        if (boost::filesystem::is_directory(iter->status()))
            continue;
    }

    if (files.empty())
        printf("Not pictures in the folder!\n");
    else
        printf("Read %ld pictures in the folder.\n", files.size());
    sort(files.begin(), files.end());

    // depth image smooth
    Mat i_depth(imgRows, imgCols, CV_32FC1);
    Mat mat_16u(imgRows, imgCols, CV_16UC1);
    Mat i_pixFilter(imgRows, imgCols, CV_16UC1);

//    QLabel *originalImage, *fixedImage, *linearInterp;
//    originalImage = new QLabel();
//    originalImage->setBackgroundRole(QPalette::Base);
//    originalImage->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
//    originalImage->setScaledContents(true);
//    originalImage->setText(QString("originalImage"));
//    fixedImage = new QLabel();
//    fixedImage->setBackgroundRole(QPalette::Base);
//    fixedImage->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
//    fixedImage->setScaledContents(true);
//    fixedImage->setText("fixedImage");
//    linearInterp = new QLabel();
//    linearInterp->setBackgroundRole(QPalette::Base);
//    linearInterp->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
//    linearInterp->setScaledContents(true);
//    linearInterp->setText("linearInterp");
    for (size_t i = 0; i < files.size(); ++i) {
        cout << "read file from: " << files[i] << endl;

        i_depth = imread(files[i], -1);

        Mat depthAbs, depthColor;
        double minVal, maxVal;
        minMaxIdx(i_depth, &minVal, &maxVal);
        convertScaleAbs(i_depth, depthAbs, 255 / maxVal);
        applyColorMap(depthAbs, depthColor, cv::COLORMAP_HSV);    // HSV
        imshow("Origal Depth Color", depthAbs);
        cvMoveWindow("Origal Depth Color", 50, 0);
//        ShowMatWithPng(depthAbs, originalImage);

        // 自定义均值滤波
        Mat intABS, intColor;
        Mat intMat = i_depth.clone();
        for (int i = 8; i < intMat.rows; ++i) {
            for (int j = 0; j < intMat.cols; ++j) {
                if (intMat.ptr(i)[j] == 0)
                    interpolation(intMat, i, j, 4);
            }
        }
        minMaxIdx(intMat, &minVal, &maxVal);
        convertScaleAbs(intMat, intABS, 255 / maxVal);
        applyColorMap(intABS, intColor, cv::COLORMAP_HSV);
        imshow("Interpolation Depth Color", intABS);
        cvMoveWindow("Interpolation Depth Color", 50, 200);
//        ShowMatWithPng(intABS, fixedImage);

        // 线性插值
        Mat linear, linearColor;
        linearInterpolation(depthAbs, linear);
        applyColorMap(linear, linearColor, cv::COLORMAP_HSV);    //
        imshow("linear interpolation", linear);
        cvMoveWindow("linear interpolation", 50, 400);
//        ShowMatWithPng(linear, linearInterp);

        // save image
        if (saveDepthMap) {
            size_t s = files[i].find_last_of('/');
            size_t e = files[i].find_last_of('.');
            string outputfile = outputFolder + "interp_depth/" + files[i].substr(s+1, e-s) + ".png";
            cv::imwrite(outputfile, intMat);
            outputfile = outputFolder + "interp_depth_abs/" + files[i].substr(s+1, e-s) + ".png";
            cv::imwrite(outputfile, intABS);
            outputfile = outputFolder + "interp_depth_abs_color/" + files[i].substr(s+1, e-s) + ".png";
            cv::imwrite(outputfile, intColor);
        }
        waitKey(100);
    }

    // 关闭窗口，设备
    destroyAllWindows();

//    delete[] depthData;
//    delete[] averagedDepthData;
//    delete[] pixelFilterData;

    std::system("pause");
    return 0;
}
