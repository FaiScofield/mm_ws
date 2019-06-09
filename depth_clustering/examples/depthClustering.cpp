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

string outputFolder = "/home/vance/output/kitti05/";  // 请以"/"结尾
bool savePictures = true;
bool savePointcloud = false;

pcl::visualization::CloudViewer g_viewer("Cluster Viewer");

//int g_v1(0), g_v2(1);
//boost::shared_ptr<pcl::visualization::PCLVisualizer> g_visualizer(new pcl::visualization::PCLVisualizer("3D Viewer"));


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

    Mat _label_image;
    Mat _clusters_projection_image; // CV_8UC3

    Mat _angle_image, _angle_image_color;
    Mat _smoothed_image;
    Mat _no_ground_image;
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
    void interpolation(Mat&, size_t, size_t, size_t);
    void DMSmooth();
    void getClusters();
    void labelProjection();

    bool readCloudFromBinaryFile(const string&, pcl::PointCloud<pcl::PointXYZI>::Ptr);
    void depthImage2Cloud(const Mat&, pcl::PointCloud<pcl::PointXYZI>&);

};

Pointcloud2DM::Pointcloud2DM(string folder): _data_folder(folder) {
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

    if (savePictures) {
//        system(("mkdir " + outputFolder).c_str());
        system(("mkdir -p " + outputFolder + "original_depth").c_str());
        system(("cd " + outputFolder + "; mkdir original_depth_abs").c_str());
        system(("cd " + outputFolder + "; mkdir original_depth_color").c_str());
        system(("cd " + outputFolder + "; mkdir interp_depth").c_str());
        system(("cd " + outputFolder + "; mkdir interp_depth_abs").c_str());
        system(("cd " + outputFolder + "; mkdir interp_depth_color").c_str());
        system(("cd " + outputFolder + "; mkdir cluster_to_depth").c_str());
        system(("cd " + outputFolder + "; mkdir label_image").c_str());
//        system(("cd " + outputFolder + "; mkdir angle_image").c_str());
//        system(("cd " + outputFolder + "; mkdir smoothed_image").c_str());
        system(("cd " + outputFolder + "; mkdir no_ground_image").c_str());
    }
    if (savePointcloud) {
        system(("cd " + outputFolder + "; mkdir pcd_original").c_str());
        system(("cd " + outputFolder + "; mkdir pcd_clusters").c_str());
//        system(("cd " + outputFolder + "; mkdir pcd_512").c_str());
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
        switch (_depth_image.type()) {  // _interp_depth_image
        case cv::DataType<float>::type: {
            // we have received a depth image
            fprintf(stderr, "[INFO] received depth.\n");
            auto diff_helper_ptr =
                dc::DiffFactory::Build(_diff_type, &_depth_image, _proj_params.get());
            _angle_image_color = diff_helper_ptr->Visualize();
            break;
        }
        case cv::DataType<uint16_t>::type: {
            fprintf(stderr, "[INFO] received labels.\n");
            _angle_image_color = dc::AbstractImageLabeler::LabelsToColor(_depth_image);
            break;
        }
        default: {
          fprintf(stderr, "[ERROR] unknown type Mat received.\n");
          return;
        }
        }
        if (_diff_type == dc::DiffFactory::DiffType::LINE_DIST_PRECOMPUTED ||
            _diff_type == dc::DiffFactory::DiffType::ANGLES_PRECOMPUTED) {
            imshow("Angle Image Color", _angle_image_color);
        }

        _depth_image.convertTo(_interp_depth_image, CV_8U);
        DMSmooth();

        _ground_rem->OnNewObjectReceived(*_cloud, 0);

        // 显示各种图片
        _angle_image = _ground_rem->_angle_image;
        _smoothed_image = _ground_rem->_smoothed_image;
        _no_ground_image = _ground_rem->_no_ground_image;
        imshow("_angle_image", _angle_image);
//        imshow("_smoothed_image", _smoothed_image);
//        imshow("_no_ground_image", _no_ground_image);

        // 加载原始点云至pcl格式
        _original_cloud->points.clear();
        _cloud_intensity->points.clear();
        if (ends_with(_file_names[i], ".pcd")) {
            pcl::io::loadPCDFile(_file_names[i], *_original_cloud);
        } else if (ends_with(_file_names[i], ".bin")) {
            readCloudFromBinaryFile(_file_names[i], _cloud_intensity);
            pcl::copyPointCloud(*_cloud_intensity, *_original_cloud);
        }

        getClusters();

        labelProjection();

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

            outputfile = outputFolder + "interp_depth/" + _current_file_name + ".png";
            imwrite(outputfile, _interp_depth_image);
            outputfile = outputFolder + "interp_depth_abs/" + _current_file_name + ".png";
            imwrite(outputfile, _interp_depth_image_abs);
            outputfile = outputFolder + "interp_depth_color/" + _current_file_name + ".png";
            imwrite(outputfile, _interp_depth_image_color);

            outputfile = outputFolder + "angle_image/" + _current_file_name + ".png";
            imwrite(outputfile, _angle_image);
//            outputfile = outputFolder + "smoothed_image/" + _current_file_name + ".png";
//            imwrite(outputfile, _smoothed_image);
//            outputfile = outputFolder + "no_ground_image/" + _current_file_name + ".png";
//            imwrite(outputfile, _no_ground_image);

            outputfile = outputFolder + "cluster_to_depth/" + _current_file_name + ".png";
            imwrite(outputfile, _clusters_projection_image);
        }
        // 保存聚类点云
        if (savePointcloud) {
            if (_cloud_out->points.size() == 0)
                fprintf(stderr, "Error in saving pcd file.\n");

            _cloud_out->height = 1;
            _cloud_out->width = _cloud_out->points.size();
            outputfile = outputFolder + "pcd_clusters/" + _current_file_name + ".pcd";
            pcl::io::savePCDFileASCII(outputfile, *_cloud_out);

//            depthImage2Cloud(_depth_image, *_cloud_intensity);
//            outputfile = outputFolder + "pcd_512/" + _current_file_name + ".pcd";
//            pcl::io::savePCDFileASCII(outputfile, *_cloud_intensity);
        }

        //    string s1 = "original cloud " + _current_file_name;
        //    string s2 = "cluster cloud " + _current_file_name;
        //    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud(_original_cloud, 255, 255, 255);
        //    g_visualizer->addPointCloud<pcl::PointXYZ>(_original_cloud, cloud, s1, g_v1);

        //    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(_cloud_out);
        //    g_visualizer->addPointCloud<pcl::PointXYZRGB>(_cloud_out, rgb, s2, g_v2);
        //    g_visualizer->spinOnce();

        waitKey(10);
    }
    destroyAllWindows();
}


bool Pointcloud2DM::readCloudFromBinaryFile(const string& fileName, pcl::PointCloud<pcl::PointXYZI>::Ptr currentCloud) {
    int32_t num = 1000000;
    float *data = (float*)malloc(num*sizeof(float));
    float *px = data+0;
    float *py = data+1;
    float *pz = data+2;
    float *pr = data+3;

    FILE *stream;
    stream = fopen(fileName.c_str(), "rb");
    num = fread(data, sizeof(float), num, stream)/4;
    for (int32_t i=0; i<num; i++) {
        pcl::PointXYZI point;
        point.x = *px;
        point.y = *py;
        point.z = *pz;
        point.intensity = *pr;
        currentCloud->points.push_back(point);
        px += 4;
        py += 4;
        pz += 4;
        pr += 4;
    }
    fclose(stream);

    if (currentCloud->size())
      return true;
    else
      return false;
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

void Pointcloud2DM::interpolation(Mat& image, size_t row, size_t col, size_t cellSize)
{
//    if (image.ptr(row)[col] == 0)
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
            if (static_cast<int>(row) + i >= 0 && static_cast<int>(row) + i < image.rows &&
                static_cast<int>(col) + j >= 0 && static_cast<int>(col) + j < image.cols &&
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

void Pointcloud2DM::DMSmooth() {
    double minVal, maxVal;
    minMaxIdx(_interp_depth_image, &minVal, &maxVal);
    convertScaleAbs(_interp_depth_image, _depth_image_abs, 255 / maxVal);
    applyColorMap(_depth_image_abs, _depth_image_color, COLORMAP_HSV);    // HSV
    imshow("Origal Depth Image", _depth_image);
    imshow("Origal Depth Image Abs", _depth_image_abs);
    imshow("Origal Depth Image Abs Color", _depth_image_color);

    size_t zeroNum = 0;
    for (int i = 0; i < _interp_depth_image.rows; ++i) {
        for (int j = 0; j < _interp_depth_image.cols; ++j) {
            if (_interp_depth_image.ptr(i)[j] == 0) {
                interpolation(_interp_depth_image, i, j, 3);
                zeroNum++;
            }
        }
    }
    minMaxIdx(_interp_depth_image, &minVal, &maxVal);
    convertScaleAbs(_interp_depth_image, _interp_depth_image_abs, 255 / maxVal);
    applyColorMap(_interp_depth_image_abs, _interp_depth_image_color, COLORMAP_RAINBOW);
    imshow("Interpolation Depth Image", _interp_depth_image);
    imshow("Interpolation Depth Image Abs", _interp_depth_image_abs);
    imshow("Interpolation Depth Image Abs Color", _interp_depth_image_color);

    cout << "[INFO] Tatal zero depth: " << zeroNum << endl;
}

void Pointcloud2DM::getClusters() {
    if (!_cloud->projection_ptr()) {
      fprintf(stderr, "[ERROR] projection not initialized in cloud.\n");
      fprintf(stderr, "[ERROR] cannot label this cloud.\n");
      return;
    }

    _clusters = _clusterer->_clusters;
    printf("[INFO] Clusters number befor filter: %ld\n", _clusters.size());

    // filter out unfitting clusters
    std::vector<uint16_t> labels_to_erase;
    for (const auto& kv : _clusters) {
        const auto& cluster = kv.second;
        Eigen::Vector3f center = Eigen::Vector3f::Zero();
        Eigen::Vector3f extent = Eigen::Vector3f::Zero();
        Eigen::Vector3f max_point(std::numeric_limits<float>::lowest(),
                                  std::numeric_limits<float>::lowest(),
                                  std::numeric_limits<float>::lowest());
        Eigen::Vector3f min_point(std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::max());
        for (const auto& point : cluster.points()) {
            center = center + point.AsEigenVector();
            min_point << std::min(min_point.x(), point.x()),
              std::min(min_point.y(), point.y()),
              std::min(min_point.z(), point.z());
            max_point << std::max(max_point.x(), point.x()),
              std::max(max_point.y(), point.y()),
              std::max(max_point.z(), point.z());
        }
        center /= cluster.size();
        if (min_point.x() < max_point.x()) {
            extent = max_point - min_point;
        }

        // 去掉中心太远的，30m
        if (center.x() * center.x() + center.y() * center.y() > 900.0f) {
            labels_to_erase.push_back(kv.first);
            continue;
        }
        // 尽量去掉离马路两边太远的，一般是花丛什么的，8m
        if (center.x() > 30.0f || center.y() > 6.0f || center.z() > 1.5f) {
            labels_to_erase.push_back(kv.first);
            continue;
        }
        // 去掉某一维尺度太小的，0.2m
        if (extent.x() < 0.2f || extent.y() < 0.2f || extent.z() < 0.3f) {
            labels_to_erase.push_back(kv.first);
            continue;
        }
        // 去掉某一维尺度太大的，5m
        if (extent.x() > 5.0f || extent.y() > 5.0f || extent.z() > 5.0f) {
            labels_to_erase.push_back(kv.first);
            continue;
        }
    }
    for (auto label : labels_to_erase) {
        _clusters.erase(label);
    }
    printf("[INFO] Clusters number after filter: %ld\n", _clusters.size());
}


void Pointcloud2DM::labelProjection() {
    if (_clusters.size() < 1) {
        fprintf(stderr, "[ERROR] No clusters this frame!\n");
        return;
    }
    if (_interp_depth_image_abs.data == nullptr)
        return;

    // 投影到深度图里并给颜色
    cvtColor(_interp_depth_image_abs, _clusters_projection_image, COLOR_GRAY2RGB);
    for (auto& c : _clusters) {
        auto& points = c.second.points();
        for (size_t i = 0; i < points.size(); ++i) {
            const auto& p = points[i];
            float dist_to_sensor = p.DistToSensor2D();
            if (dist_to_sensor < 0.01f) {
              continue;
            }
            auto angle_rows = dc::Radians::FromRadians(asin(p.z() / dist_to_sensor));
            auto angle_cols = dc::Radians::FromRadians(atan2(p.y(), p.x()));
//            size_t bin_rows = p.ring();
            size_t bin_rows = _proj_params->RowFromAngle(angle_rows);
            size_t bin_cols = _proj_params->ColFromAngle(angle_cols);

            // 聚类上红色
            _clusters_projection_image.at<Vec3b>(bin_rows, bin_cols) = Vec3b(0, 0, 255);
        }
    }
    imshow("DM Projection", _clusters_projection_image);

    // 可视化点云
    _cloud_out->points.clear();
    pcl::PointXYZRGB thisPoint;
    for (auto& c : _clusters) {
        auto& points = c.second.points();

        uint8_t r, g, b;
        r = uint8_t(random() % 255);
        g = uint8_t(random() % 255);
        b = uint8_t(random() % 255);
        uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
        for (size_t i = 0; i < points.size(); ++i) {
            dc::RichPoint p = points[i];
            thisPoint.x = p.x();
            thisPoint.y = p.y();
            thisPoint.z = p.z();
            thisPoint.rgb = *reinterpret_cast<float*>(&rgb);
            _cloud_out->points.push_back(thisPoint);
        }
    }
    g_viewer.showCloud(_cloud_out);
}

void Pointcloud2DM::depthImage2Cloud(const Mat& image, pcl::PointCloud<pcl::PointXYZI>& cloud) {
    cloud.clear();
    for (int r = 0; r < image.rows; ++r) {
        for (int c = 0; c < image.cols; ++c) {
            float depth = image.at<float>(r, c);
            dc::Radians angle_z = _proj_params->AngleFromRow(r);
            dc::Radians angle_xy = _proj_params->AngleFromCol(c);
            pcl::PointXYZI p;
            p.x = depth * cosf(angle_z.val()) * cosf(angle_xy.val());
            p.y = depth * cosf(angle_z.val()) * sinf(angle_xy.val());
            p.z = depth * sinf(angle_z.val());
            p.intensity = 0.8;
            cloud.push_back(p);
        }
    }
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
