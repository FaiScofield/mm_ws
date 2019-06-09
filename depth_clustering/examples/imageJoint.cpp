#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <vector>

using boost::ends_with;
using namespace std;
namespace fs = boost::filesystem;

enum MODE {
    SAME = 0,
    CONTINUE = 1,
    RANDOM = 2
};

bool saveJointedImage = true;
string outputFolder = "/home/vance/output/joint_images/";
MODE mode = SAME;


vector<string> readFolderFiles(string folder) {
    vector<string> fileNames;

    fs::path path(folder);
    if (!fs::exists(path))
        return fileNames;

    fs::directory_iterator end_iter;
    for (fs::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (fs::is_directory(iter->status()))
            continue;
        if (ends_with(iter->path().string(), ".txt") ||
            ends_with(iter->path().string(), ".cfg"))
            continue;

        if (fs::is_regular_file(iter->status()))
            fileNames.push_back(iter->path().string());
    }

    sort(fileNames.begin(), fileNames.end());

    return fileNames;
}

void imageJoint(vector<string> files, MODE mode=SAME) {
    printf("Dealing with all images...\n");

    cv::Mat tmp = cv::imread(files[0], -1);
    int type = tmp.type();
    printf("Image type: %d, rows = %d, cols = %d\n", type, tmp.rows, tmp.cols);
    cv::Mat jointedImage(5*tmp.rows, tmp.cols, type);
    printf("Set jointedImage type: %d, rows = %d, cols = %d\n", jointedImage.type(),
           jointedImage.rows, jointedImage.cols);

    vector<cv::Mat> images;
    switch (mode) {
        case SAME: {
            printf("joint type = SAME\n");
            for (int i = 0; i < files.size(); ++i) {
                cv::Mat currentImage = cv::imread(files[i], -1);
                for (int k = 1; k < 6; ++k) {
                     currentImage.rowRange(0, tmp.rows).copyTo(jointedImage.rowRange((k-1)*tmp.rows, k*tmp.rows));
                }
                cv::imshow("Jointed Image", jointedImage);
                cv::waitKey(200);
                images.push_back(jointedImage.clone());
            }
            break;
        }
        case CONTINUE: {
            printf("joint type = CONTINUE\n");
            int numToSave = files.size() - files.size() % 5;
            for (int i = 0; i < numToSave; i += 5) {
                for (int k = 1; k < 6; ++k) {
                    cv::Mat currentImage = cv::imread(files[i+k], -1);
                    currentImage.rowRange(0, tmp.rows).copyTo(jointedImage.rowRange((k-1)*tmp.rows, k*tmp.rows));
                }
                cv::imshow("Jointed Image", jointedImage);
                cv::waitKey(200);
                images.push_back(jointedImage.clone());
            }
            break;
        }
        case RANDOM: {
            printf("joint type = RANDOM\n");
            int numToSave = files.size() / 5;
            vector<bool> visited(files.size(), false);
            for (int i = 0; i < numToSave; i++) {
                int k = 1;
                while (k < 6) {
                    int index = rand() % files.size();
                    if (visited[index])
                        continue;
                    visited[index] = true;
                    cv::Mat currentImage = cv::imread(files[index], -1);
                    currentImage.rowRange(0, tmp.rows).copyTo(jointedImage.rowRange((k-1)*tmp.rows, k*tmp.rows));
                    k++;
                }
                cv::imshow("Jointed Image", jointedImage);
                cv::waitKey(200);
                images.push_back(jointedImage.clone());
            }
            break;
        }
    }
    printf("Generate %ld joint images.\n", images.size());


    if (saveJointedImage) {
        printf("Writing joint images to %s\n", outputFolder.c_str());
        for (size_t i = 0; i < images.size(); ++i) {
            char id[1024];
            sprintf(id, "%6d", i);
            string name = outputFolder + id + ".png";
            cv::imwrite(name, images[i]);
        }
        printf("Done.\n");
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <depth image folder> [mode(same=0/continue=1)]\n", argv[0]);
        return -1;
    } else if (atoi(argv[2]) < 0 || atoi(argv[2]) > 2) {
        fprintf(stderr, "Error input for mode type, for image-joint with:\n"
                        "  same 5 pictures, use 0;\n"
                        "  continue 5 picturesm use 1.\n"
                        "  random 5 pictures, use 2.\n");
        return -1;
    } else {
        mode = MODE(atoi(argv[2]));
    }

    vector<string> fileNames = readFolderFiles(string(argv[1]));
    if (fileNames.empty()) {
        printf("Not depth image in the folder!\n");
        return  -1;
    }
    else
        printf("Read %ld files in the folder.\n", fileNames.size());

    imageJoint(fileNames, mode);


    return 0;
}
