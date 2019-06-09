
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

void fillHole(const Mat srcBw, Mat &dstBw)
{
    Size m_Size = srcBw.size();
    Mat Temp = Mat::zeros(m_Size.height+2, m_Size.width+2, srcBw.type()); // 延展图像
    srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));

    cv::floodFill(Temp, Point(0, 0), Scalar(255));

    Mat cutImg;// 裁剪延展的图像
    Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);

    dstBw = srcBw | (~cutImg);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image with holes>\n", argv[0]);
        return -1;
    }

    Mat img = cv::imread(argv[1], -1);

    Mat gray;
    img.convertTo(gray, CV_8UC1);
//    cv::cvtColor(img, gray, CV_RGB2GRAY);

//    Mat bw;
//    cv::threshold(img, bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    Mat bwFill;
    fillHole(gray, bwFill);

    imshow("before", gray);
    imshow("after", bwFill);

    waitKey();

    return 0;
}
