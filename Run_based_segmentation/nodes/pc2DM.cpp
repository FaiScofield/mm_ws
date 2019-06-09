#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <velodyne_pointcloud/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>

#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

namespace pointcloud_to_dense_map {

struct PointXYZIRL
{
    PCL_ADD_POINT4D;                    // quad-word XYZ
    float    intensity;                 ///< laser intensity reading
    uint16_t ring;                      ///< laser ring number
    uint16_t label;                     ///< point label
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // ensure proper alignment
} EIGEN_ALIGN16;

}; // namespace

#define SLRPointXYZIRL pointcloud_to_dense_map::PointXYZIRL
#define VPoint velodyne_pointcloud::PointXYZIR
// Register custom point struct according to PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(pointcloud_to_dense_map::PointXYZIRL,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (uint16_t, ring, ring)
                                  (uint16_t, label, label))

#define dist(a,b) sqrt(((a).x-(b).x)*((a).x-(b).x)+((a).y-(b).y)*((a).y-(b).y))


class Pointcloud2DM {
private:
    ros::NodeHandle node_handle_;
    ros::Subscriber points_node_sub_;
    ros::Publisher dense_map_pub_;

    // parameters
    std::string input_topic_;
    std::string output_topic_;
    int sensor_model_, sensor_resolution_;

    // data
    cv::Mat dm_;
    cv::Mat dm_show_;
    cv_bridge::CvImage cb_;

public:
    Pointcloud2DM();
    // Call back funtion.
    void laser_callback_(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg);

};

Pointcloud2DM::Pointcloud2DM(): node_handle_("~") {
    ROS_INFO("Inititalizing Point Cloud to Dense Map...");
    node_handle_.param<std::string>("input_topic", input_topic_, "/kitti/velo/pointcloud");
    std::cout << input_topic_ << std::endl;
    ROS_INFO("Input Topic: %s", input_topic_.c_str());
    node_handle_.param<std::string>("output_topic", output_topic_, "/dense_map");
    ROS_INFO("Output Topic: %s", output_topic_.c_str());
    node_handle_.param("sensor_model", sensor_model_, 64);
    ROS_INFO("Sensor Model: %d", sensor_model_);
    node_handle_.param("sensor_resolution", sensor_resolution_, 1800);
    ROS_INFO("Sensor Resolution: %d", sensor_resolution_);

    points_node_sub_ = node_handle_.subscribe(input_topic_, 2, &Pointcloud2DM::laser_callback_, this);
    dense_map_pub_ = node_handle_.advertise<sensor_msgs::Image>(output_topic_, 10);

    dm_ = cv::Mat(sensor_model_, sensor_resolution_, CV_32FC1);
    dm_show_ = cv::Mat(sensor_model_, sensor_resolution_, CV_16UC1);

    cb_.encoding = "mono8";
    cb_.header.stamp = ros::Time::now();
    cb_.header.frame_id = "camera";
}


void Pointcloud2DM::laser_callback_(const sensor_msgs::PointCloud2ConstPtr& in_cloud_msg) {
    ROS_INFO("Recived a meaasge.");

    // Msg to pointcloud
    pcl::PointCloud<VPoint>::Ptr laserCloudIn(new pcl::PointCloud<VPoint>());
//    laserCloudIn.reset;
    pcl::fromROSMsg(*in_cloud_msg, *laserCloudIn);

    // projection
    float max = 0.0, min = 50000.0;
    for (auto& p : laserCloudIn->points) {
        float range = sqrt(p.x * p.x + p.y * p.y/* + p.z * p.z*/);
        int row = p.ring;/*asin(p.z / range) * 180 / M_PI*/
        if (row < 0 || row >= sensor_model_)
            continue;
        int col = -round((atan2(p.x, p.y) * 180 / M_PI -90.0)/0.2) + sensor_resolution_/2;

        if (col >= sensor_resolution_)
            col -= sensor_resolution_;
        if (col < 0 || col >= sensor_resolution_)
            continue;

        dm_.at<float>(row, col) = range;

        if (range > max) max = range;
        if (range < min) min = range;
    }

    ROS_INFO("Projection done. Max range: %f, min range: %f", max, min);

    for (int i = 0; i < sensor_model_; ++i) {
        for (int j = 0; j < sensor_resolution_; ++j) {
            dm_show_.at<uchar>(i, j) = dm_.at<float>(i,j)*255/(max-min);
//            ROS_INFO("Range: %f, gray: %d", dm_.at<float>(i, j), dm_show_.at<uchar>(i, j));
        }
    }


    double minv, maxv;
    cv::minMaxLoc(dm_, &minv, &maxv);

    cv::Mat show;
    dm_.convertTo(show, CV_16U, (2<<16)/(maxv-minv), -minv*(2<<16)/(maxv-minv));
    cv::imshow("a", show);
    cv::waitKey(20);
    cb_.image = show;
    sensor_msgs::Image::Ptr dm_messgae = cb_.toImageMsg();
    dense_map_pub_.publish(dm_messgae);

//    Mat dm_abs, dm_color;
//    double minVal, maxVal;
//    minMaxIdx(dm_, &minVal, &maxVal);
//    ROS_INFO("Max range: %f, min range: %f", maxVal, minVal);
//    convertScaleAbs(dm_, dm_abs, 255 / maxVal);
//    applyColorMap(dm_abs, dm_color, COLORMAP_HSV);    // HSV
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "pc2DM");
//  ros::NodeHandle nh;

    Pointcloud2DM pc2dm;

    ros::spin();

    return 0;
}
