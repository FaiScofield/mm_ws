#include "clusterers/image_based_clusterer.h"
#include "communication/abstract_client.h"
#include "ground_removal/depth_ground_remover.h"
#include "projections/cloud_projection.h"
#include "projections/spherical_projection.h"
#include "utils/folder_reader.h"
#include "utils/velodyne_utils.h"
#include "image_labelers/diff_helpers/diff_factory.h"


#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <stdlib.h>
#include <exception>

#include <boost/thread/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace cv;
using boost::ends_with;
namespace fs = boost::filesystem;
namespace dc = depth_clustering;

string outputFolder = "/home/vance/output/segment/";  // 请以"/"结尾
bool savePictures = true;
pcl::visualization::CloudViewer g_viewer("Cluster Viewer");

class Segment2DM {
private:
    unique_ptr<dc::ProjectionParams> _projParams = nullptr;

    pcl::PointCloud<pcl::PointXYZI>::Ptr _fullCloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr _segCloud;

    string _origDataFolder, _segDataFolder;
    vector<string> _origFiles, _segFiles;
    string _currentFileName;

    Mat _depthImage;    // CV_32F
    Mat _depthImageAbs;
    Mat _depthImageColor;

    Mat _interpDepthImage;          // CV_8U
    Mat _interpDepthImageAbs;
    Mat _interpDepthImageColor;     // CV_8UC3
    Mat _clustersProjectionImage;   // CV_8UC3

public:
    Segment2DM(string folder1, string folder2);
    vector<string> readFolderFiles(const string& folder);
    void cloudFromTxtFile(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const string& fileIn);
    void cloudFromBinFile(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const string& fileIn);
    void interpolation(Mat& image, int row, int col, int cellSize);
};

Segment2DM::Segment2DM(string folder1, string folder2): _origDataFolder(folder1), _segDataFolder(folder2) {
    _projParams = dc::ProjectionParams::HDL_64_EQUAL();

    if (savePictures) {
        system(("mkdir -p " + outputFolder).c_str());
        system(("cd " + outputFolder + "; mkdir interp_depth_abs").c_str());
        system(("cd " + outputFolder + "; mkdir interp_depth_color").c_str());
        system(("cd " + outputFolder + "; mkdir cluster_to_depth").c_str());
    }

    _origFiles = readFolderFiles(_origDataFolder);
    _segFiles = readFolderFiles(_segDataFolder);
    if (_origFiles.size() < 1 || _segFiles.size() < 1) {
        fprintf(stderr, "No file in the path!\n");
        return;
    }
    if (_origFiles.size() != _segFiles.size()) {
        fprintf(stderr, "Wrong size of two data folders!\n");
        return;
    }

    // 逐个处理
    for (size_t i = 0; i < _origFiles.size(); ++i) {
        printf("Loading cloud from %s\n", _origFiles[i].c_str());

        // 加载点云
        _fullCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
        _segCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
        cloudFromTxtFile(_fullCloud, _origFiles[i]);
        pcl::io::loadPCDFile(_segFiles[i], *_segCloud);
        if (_fullCloud->empty() || _segCloud->empty())
            continue;
        printf("cloud size of %s is %ld\n", _origFiles[i].c_str(), _fullCloud->size());
        printf("cloud size of %s is %ld\n", _segFiles[i].c_str(), _segCloud->size());
        g_viewer.showCloud(_fullCloud);

        // 点云投影
        _depthImage = Mat(64, 870, CV_32FC1);
        for (const auto& p : _fullCloud->points) {
            float dist_to_sensor = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
            if (dist_to_sensor < 0.01f) continue;
            auto angle_rows = dc::Radians::FromRadians(asin(p.z / dist_to_sensor));
            auto angle_cols = dc::Radians::FromRadians(atan2(p.y, p.x));
            size_t bin_rows = _projParams->RowFromAngle(angle_rows);
            size_t bin_cols = _projParams->ColFromAngle(angle_cols);

            _depthImage.at<float>(bin_rows, bin_cols) = dist_to_sensor;
        }
        double minVal, maxVal;
        minMaxIdx(_depthImage, &minVal, &maxVal);
        convertScaleAbs(_depthImage, _depthImageAbs, 255 / maxVal);
        applyColorMap(_depthImageAbs, _depthImageColor, COLORMAP_HSV);    // HSV
        imshow("Original Depth Image Abs", _depthImageAbs);
        imshow("Original Depth Image Color", _depthImageColor);

        // 深度图填补空隙
        _interpDepthImage = _depthImageAbs.clone();
        for (int i = 0; i < _interpDepthImage.rows; ++i) {
            for (int j = 0; j < _interpDepthImage.cols; ++j) {
                if (_interpDepthImage.ptr(i)[j] == 0) {
                    interpolation(_interpDepthImage, i, j, 3);
                }
            }
        }
        applyColorMap(_interpDepthImage, _interpDepthImageColor, COLORMAP_RAINBOW);
        imshow("Interpolation Depth Image Abs", _interpDepthImage);
        imshow("Interpolation Depth Image Abs Color", _interpDepthImageColor);

        // segment投影到深度图
        cvtColor(_interpDepthImage, _clustersProjectionImage, COLOR_GRAY2RGB);
        for (auto& p : _segCloud->points) {
            float dist_to_sensor = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
            if (dist_to_sensor < 0.01f) continue;
            auto angle_rows = dc::Radians::FromRadians(asin(p.z / dist_to_sensor));
            auto angle_cols = dc::Radians::FromRadians(atan2(p.y, p.x));
            size_t bin_rows = _projParams->RowFromAngle(angle_rows);
            size_t bin_cols = _projParams->ColFromAngle(angle_cols);

            // 聚类上红色
            _clustersProjectionImage.at<Vec3b>(bin_rows, bin_cols) = Vec3b(0, 0, 255);
        }
        imshow("segment to DM", _clustersProjectionImage);

        // 保存深度图
        string outputfile;
        if (savePictures) {
            size_t s = _origFiles[i].find_last_of('/');
            size_t e = _origFiles[i].find_last_of('.');
            _currentFileName = _origFiles[i].substr(s+1, e-s-1);

            outputfile = outputFolder + "interp_depth_abs/" + _currentFileName + ".png";
            imwrite(outputfile, _interpDepthImage);
            outputfile = outputFolder + "interp_depth_color/" + _currentFileName + ".png";
            imwrite(outputfile, _interpDepthImageColor);
            outputfile = outputFolder + "cluster_to_depth/" + _currentFileName + ".png";
            imwrite(outputfile, _clustersProjectionImage);
        }

        waitKey(0);
    }
    destroyAllWindows();
}


void Segment2DM::cloudFromBinFile(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const string& fileIn)
{
    int32_t num = 1000000;
    float *data = (float*)malloc(num*sizeof(float));
    float *px = data+0;
    float *py = data+1;
    float *pz = data+2;
    float *pr = data+3;

    // load point cloud
    FILE *stream;
    stream = fopen(fileIn.c_str(),"rb");
    num = fread(data, sizeof(float), num, stream)/4;
    for (int32_t i=0; i<num; i++) {
        pcl::PointXYZI point;
        point.x = *px;
        point.y = *py;
        point.z = *pz;
        point.intensity = *pr;
        cloud->points.push_back(point);
        px += 4;
        py += 4;
        pz += 4;
        pr += 4;
    }
    fclose(stream);
}


void Segment2DM::cloudFromTxtFile(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const string& fileIn)
{
    locale::global(std::locale("en_US.UTF-8"));
    ifstream ifs(fileIn.c_str());
    for (std::string line; std::getline(ifs, line, '\n');) {
        vector<string> coords_str;
        boost::split(coords_str, line, boost::is_any_of(" "));
        if (coords_str.size() != 4) {
            fprintf(stderr, "ERROR: format of line is wrong.\n");
            continue;
        }
        pcl::PointXYZI point;
        point.x = std::stof(coords_str[0]);
        point.y = std::stof(coords_str[1]);
        point.z = std::stof(coords_str[2]);
        point.intensity = std::stof(coords_str[3]);
        cloud->push_back(point);
    }
    ifs.close();
}

vector<string> Segment2DM::readFolderFiles(const string& folder) {
    vector<string> res;

    fs::path path(folder);
    if (!fs::exists(path))
        return res;

    fs::directory_iterator end_iter;
    for (fs::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (fs::is_directory(iter->status()))
            continue;
        if (/*ends_with(iter->path().string(), ".txt") ||*/
            ends_with(iter->path().string(), ".cfg"))
            continue;

        if (fs::is_regular_file(iter->status()))
            res.push_back(iter->path().string());
    }

    if (res.empty())
        printf("Not laser data in the folder!\n");
    else
        printf("Read %ld files in the folder.\n", _origFiles.size());

    sort(res.begin(), res.end());

    return res;
}


void Segment2DM::interpolation(Mat& image, int row, int col, int cellSize)
{
    if (row < 0 || row >= image.rows || col < 0 || col >= image.cols)
        return;
    if (image.ptr(row)[col] != 0)
        return;

    size_t validNeighbourNum = 0;
    ushort value = 0;
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
                image.ptr(row+i)[col+j] > 0 ) {
                value += image.ptr(row+i)[col+j];
                validNeighbourNum++;
            } else
                continue;
        }
    }

    if (validNeighbourNum != 0)
        image.ptr(row)[col] = static_cast<uchar>(static_cast<float>(value)/validNeighbourNum);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("Usage: %s <lidar data folder> <segment lidar data folder>\n", argv[0]);
        return -1;
    }

    Segment2DM pc2dm(argv[1], argv[2]);

    return 0;
}
