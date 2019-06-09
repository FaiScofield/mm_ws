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

string outputFolder = "/home/vance/output/no_ground/";  // 请以"/"结尾
bool savePictures = true;

dc::Cloud::Ptr CloudFromFile(const string &file_name,
        const dc::ProjectionParams &proj_params) {
    dc::Cloud::Ptr cloud = nullptr;
    if (ends_with(file_name, ".pcd")) {
        pcl::PointCloud<pcl::PointXYZL> pcl_cloud;
        pcl::io::loadPCDFile(file_name, pcl_cloud);
        cloud = dc::Cloud::FromPcl<pcl::PointXYZL>(pcl_cloud);
        cloud->InitProjection(proj_params);
    } else if (ends_with(file_name, ".png") || ends_with(file_name, ".exr")) {
        cloud = dc::Cloud::FromImage(dc::MatFromDepthPng(file_name), proj_params);
    } else if (ends_with(file_name, ".txt")) {
        cloud = dc::ReadKittiCloudTxt(file_name);
        cloud->InitProjection(proj_params);
    } else if (ends_with(file_name, ".bin")) {
        cloud = dc::ReadKittiCloud(file_name);
        cloud->InitProjection(proj_params);
    }
    return cloud;
}


class Pointcloud2DM {
private:
    unique_ptr<dc::ProjectionParams> _proj_params = nullptr;

    unique_ptr<
        dc::ImageBasedClusterer<dc::LinearImageLabeler<>> > _clusterer = nullptr;
    unique_ptr<dc::DepthGroundRemover> _ground_rem = nullptr;

    dc::Cloud::Ptr _cloud;
    std::unordered_map<uint16_t, dc::Cloud> _clusters;

    string _data_folder;
    vector<string> _file_names;
    string _current_file_name;
    vector<string> _current_object_labels;

    Mat _depth_image;  // CV_32F
    Mat _depth_image_abs;
    Mat _depth_image_color;

    Mat _interp_depth_image;        // CV_8U
    Mat _interp_depth_image_abs;
    Mat _interp_depth_image_color;  // CV_8UC3

    Mat _clusters_projection_image; // CV_8UC3

    Mat _angle_image, _angle_image_color;
    Mat _smoothed_image;
    Mat _no_ground_image, _no_ground_image_abs;;
    Mat _depth_image_repaired;

    // 参数
    int _min_cluster_size;
    int _max_cluster_size;
    int _smooth_window_size;
    dc::Radians _angle_tollerance;
    dc::Radians _ground_remove_angle;
    dc::DiffFactory::DiffType _diff_type;


//    pcl::PointCloud<pcl::PointXYZI>::Ptr _cloud_intensity_512;
    pcl::PointCloud<pcl::PointXYZI>::Ptr _cloud_intensity;
    pcl::PointCloud<pcl::PointXYZ>::Ptr _original_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr _no_ground_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud_out;

public:
    Pointcloud2DM(string folder);

    void readFolderFiles();

};

Pointcloud2DM::Pointcloud2DM(string folder): _data_folder(folder) {
//    _cloud.reset(new dc::Cloud);
    _cloud_out.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    _original_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    _no_ground_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    _cloud_intensity.reset(new pcl::PointCloud<pcl::PointXYZI>());
//    _cloud_intensity_512.reset(new pcl::PointCloud<pcl::PointXYZI>());

    _proj_params = dc::ProjectionParams::HDL_64_EQUAL();
//    _diff_type = dc::DiffFactory::DiffType::LINE_DIST_PRECOMPUTED;
    _diff_type = dc::DiffFactory::DiffType::ANGLES_PRECOMPUTED;
//    _diff_type = dc::DiffFactory::DiffType::ANGLES;
//    _diff_type = dc::DiffFactory::DiffType::SIMPLE;
    _angle_tollerance = dc::Radians::FromDegrees(10);
    _ground_remove_angle = dc::Radians::FromDegrees(10);
    _min_cluster_size = 30;
    _max_cluster_size = 5000;
    _smooth_window_size = 5;

    _clusterer.reset(new dc::ImageBasedClusterer<dc::LinearImageLabeler<>>(
                         _angle_tollerance, _min_cluster_size, _max_cluster_size));
    _clusterer->SetDiffType(_diff_type);
    _ground_rem.reset(new dc::DepthGroundRemover(*_proj_params,
                         _ground_remove_angle, _smooth_window_size));
    _ground_rem->AddClient(_clusterer.get());
//    _clusterer->AddClient(nullptr);

    if (savePictures) {
        system(("mkdir -p " + outputFolder).c_str());
        system(("cd " + outputFolder + "; mkdir original_depth").c_str());
        system(("cd " + outputFolder + "; mkdir original_depth_abs").c_str());
        system(("cd " + outputFolder + "; mkdir original_depth_color").c_str());
//        system(("cd " + outputFolder + "; mkdir interp_depth").c_str());
//        system(("cd " + outputFolder + "; mkdir interp_depth_abs").c_str());
//        system(("cd " + outputFolder + "; mkdir interp_depth_color").c_str());
//        system(("cd " + outputFolder + "; mkdir cluster_to_depth").c_str());
//        system(("cd " + outputFolder + "; mkdir label_image").c_str());
//        system(("cd " + outputFolder + "; mkdir angle_image").c_str());
//        system(("cd " + outputFolder + "; mkdir smoothed_image").c_str());
        system(("cd " + outputFolder + "; mkdir no_ground_image").c_str());
        system(("cd " + outputFolder + "; mkdir no_ground_image_abs").c_str());
    }


    readFolderFiles();
    if (_file_names.size() < 1) {
        fprintf(stderr, "No file in the path!\n");
        return;
    }
    for (size_t i = 0; i < _file_names.size(); ++i) {
        printf("Loading cloud from %s\n", _file_names[i].c_str());

        _cloud.reset(new dc::Cloud);
        _cloud = CloudFromFile(_file_names[i], *_proj_params);
        if (_cloud->points().empty())
            continue;
        _depth_image = _cloud->projection_ptr()->depth_image().clone();

        double minVal, maxVal;
        minMaxIdx(_depth_image, &minVal, &maxVal);
        convertScaleAbs(_depth_image, _depth_image_abs, 255 / maxVal);
        applyColorMap(_depth_image_abs, _depth_image_color, COLORMAP_HSV);    // HSV
        imshow("Origal Depth Image", _depth_image);
        imshow("Origal Depth Image Abs", _depth_image_abs);
        imshow("Origal Depth Image Abs Color", _depth_image_color);

        _depth_image.convertTo(_no_ground_image, CV_8U);
        imshow("_no_ground_image", _no_ground_image);

        // 保存深度图
        string outputfile;
        if (savePictures) {
            size_t s = _file_names[i].find_last_of('/');
            size_t e = _file_names[i].find_last_of('.');
            _current_file_name = _file_names[i].substr(s+1, e-s-1);

            outputfile = outputFolder + "original_depth/" + _current_file_name + ".png";
            imwrite(outputfile, _depth_image);
            outputfile = outputFolder + "original_depth_abs/" + _current_file_name + ".png";
            imwrite(outputfile, _depth_image_abs);
            outputfile = outputFolder + "original_depth_color/" + _current_file_name + ".png";
            imwrite(outputfile, _depth_image_color);

            outputfile = outputFolder + "no_ground_image/" + _current_file_name + ".png";
            imwrite(outputfile, _no_ground_image);
            outputfile = outputFolder + "no_ground_image_abs/" + _current_file_name + ".png";
            imwrite(outputfile, _no_ground_image);
        }

        waitKey(10);
    }
    destroyAllWindows();
}



void Pointcloud2DM::readFolderFiles() {
    _file_names.clear();

    fs::path path(_data_folder);
    if (!fs::exists(path))
        return;

    fs::directory_iterator end_iter;
    for (fs::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (fs::is_directory(iter->status()))
            continue;
        if (/*ends_with(iter->path().string(), ".txt") ||*/
            ends_with(iter->path().string(), ".cfg"))
            continue;

        if (fs::is_regular_file(iter->status()))
            _file_names.push_back(iter->path().string());
    }

    if (_file_names.empty())
        printf("Not laser data in the folder!\n");
    else
        printf("Read %ld files in the folder.\n", _file_names.size());

    sort(_file_names.begin(), _file_names.end());
}


int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Usage: %s <lidar data folder>\n", argv[0]);
        return -1;
    }

    Pointcloud2DM pc2dm(argv[1]);

    return 0;
}
