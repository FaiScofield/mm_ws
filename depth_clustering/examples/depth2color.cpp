#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;

int main(int argc, char *argv[])
{
    Mat image = imread("/home/vance/dataset/mrt/scenario1/scan00021.png", CV_8U);
    if (!image.data)
        fprintf(stderr, "error in reading image.\n");

    imshow("original image", image);

    Mat imageShow;
    if (image.type() == CV_32F) {
        printf("convert CV_32F to CV_8UC3\n");
        image.convertTo(imageShow, CV_8UC3);
    }
    if (image.type() == CV_8U) {
        printf("convert gray image to color\n");
        cvtColor(image, imageShow, COLOR_GRAY2RGB);
    }

    for (int r = 20; r < 40; ++r)
        for (int c = 20; c < 40; ++c)
            imageShow.at<Vec3b>(r, c) = Vec3b(0, 0, 255);

    imshow("color image", imageShow);
    waitKey(0);

    return 0;
}

