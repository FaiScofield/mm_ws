#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
namespace fs = boost::filesystem;
namespace ba = boost::algorithm;

void readFolder(const string& folder, vector<string>& files)
{
    files.clear();

    fs::path path(folder);
    if (!fs::exists(path))
        return;

    fs::directory_iterator end_iter;
    for (fs::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (fs::is_directory(iter->status()))
            continue;
        if (ba::ends_with(iter->path().string(), ".cfg"))
            continue;

        if (fs::is_regular_file(iter->status()))
            files.push_back(iter->path().string());
    }

    if (files.empty()) {
        ROS_ERROR("Not laser data in the folder!");
        return;
    }
    else
        ROS_INFO("Read %d files in the folder.", files.size());

    sort(files.begin(), files.end());
}

void cloudFromBinFile(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const string& fileIn)
{
    ROS_INFO("Reading file %s", fileIn.c_str());
    /*
//    fstream file(fileIn.c_str(), std::ios::in | std::ios::binary);
//    if (file.good()) {
//        file.seekg(0, std::ios::beg);
//        for (int i = 0; file.good() && !file.eof(); ++i) {
//            pcl::PointXYZI point;
//            file.read(reinterpret_cast<char*>(&point.x), sizeof(float));
//            file.read(reinterpret_cast<char*>(&point.y), sizeof(float));
//            file.read(reinterpret_cast<char*>(&point.z), sizeof(float));
//            file.read(reinterpret_cast<char*>(&point.intensity), sizeof(float));
//            cloud->points.push_back(point);
//        }
//        file.close();
//    }
*/
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

void cloudFromTxtFile(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const string& fileIn) {
    std::locale::global(std::locale("en_US.UTF-8"));
    fprintf(stderr, "Reading cloud from %s.\n", fileIn.c_str());

    std::ifstream file(fileIn.c_str());
    for (std::string line; std::getline(file, line, '\n');) {
        // here we parse the line
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
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "readDataFromBinFiles");
    ros::NodeHandle nh;

    ros::Publisher veloPub = nh.advertise<sensor_msgs::PointCloud2>("/kitti/velo/pointcloud", 1);

    string folder;
    nh.param<string>("folder_of_bin_files", folder, string("/home/vance/dataset/mm/persion/data"));
    ROS_INFO("Reading data from: %s", folder.c_str());

    vector<string> files;
    if (folder.size() > 0)
        readFolder(folder, files);
    else {
        ROS_ERROR("Please set the folder of .bin/.txt files for laser data.");
        return -1;
    }
    ROS_INFO("Publishing topic...");

    bool done = false;
    size_t i = 0, num = files.size();
    sensor_msgs::PointCloud2 msg;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;

    ros::Rate loop_rate(10);
//    while (ros::ok() && !done) {
    for (; i < num; ++i) {
        cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
        if (boost::ends_with(files[i], ".bin"))
            cloudFromBinFile(cloud, files[i]);
        if (boost::ends_with(files[i], ".txt"))
            cloudFromTxtFile(cloud, files[i]);
        ROS_INFO("Read %ld points.", cloud->points.size());

        pcl::toROSMsg(*cloud, msg);
        msg.header.seq = static_cast<uint>(i);
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = "/velodyne";

        veloPub.publish(msg);

        if (i == num - 1) {
            ROS_INFO("Process done.");
            done = true;
        }

//        ros::spinOnce();

//        loop_rate.sleep();
    }



    return 0;
}
