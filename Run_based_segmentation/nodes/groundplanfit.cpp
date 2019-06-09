/*
    @file groundplanfit.cpp
    @brief ROS Node for ground plane fitting

    This is a ROS node to perform ground plan fitting.
    Implementation accoriding to <Fast Segmentation of 3D Point Clouds: A Paradigm>

    In this case, it's assumed that the x,y axis points at sea-level,
    and z-axis points up. The sort of height is based on the Z-axis value.

    @author Vincent Cheung(VincentCheungm)
    @bug Sometimes the plane is not fit.
*/

#include <iostream>
#include <fstream>

// For disable PCL complile lib, to use PointXYZIR
#define PCL_NO_PRECOMPILE

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/io/io.h>

#include <velodyne_pointcloud/point_types.h>

//#include <dynamic_reconfigure/server.h>
//#include <points_preprocessor_usi/paramsConfig.h>

// using eigen lib
#include <Eigen/Dense>

//Customed Point Struct for holding clustered points
namespace scan_line_run
{
  /** Euclidean Velodyne coordinate, including intensity and ring number, and label. */
  struct PointXYZIRL
  {
    PCL_ADD_POINT4D;                    // quad-word XYZ
    float    intensity;                 ///< laser intensity reading
    uint16_t ring;                      ///< laser ring number
    uint16_t label;                     ///< point label
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // ensure proper alignment
  } EIGEN_ALIGN16;

}; // namespace scan_line_run

#define SLRPointXYZIRL scan_line_run::PointXYZIRL
//#define VPoint velodyne_pointcloud::PointXYZIR
#define VPoint pcl::PointXYZI
#define RUN pcl::PointCloud<SLRPointXYZIRL>
// Register custom point struct according to PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(scan_line_run::PointXYZIRL,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (uint16_t, ring, ring)
                                  (uint16_t, label, label))


using Eigen::MatrixXf;
using Eigen::JacobiSVD;
using Eigen::VectorXf;

const float ang_res_y = 26.8f/63.0f;

/*
    @brief Compare function to sort points. Here use z axis.
    @return z-axis accent
*/
bool point_cmp(VPoint a, VPoint b){
    return a.z < b.z;
}

/*
    @brief Ground Plane fitting ROS Node.
    @param Velodyne Pointcloud topic.
    @param Sensor Model.
    @param Sensor height for filtering error mirror points.
    @param Num of segment, iteration, LPR
    @param Threshold of seeds distance, and ground plane distance

    @subscirbe:/velodyne_points
    @publish:/points_no_ground, /points_ground
*/
class GroundPlaneFit{
public:
    GroundPlaneFit();
    void saveNoGroundPointcloud();
//    void updateParams(points_preprocessor_usi::paramsConfig &config, uint32_t level);

private:
    pcl::PointCloud<SLRPointXYZIRL>::Ptr _pcGroundSeeds;
    pcl::PointCloud<SLRPointXYZIRL>::Ptr _pcGround;
    pcl::PointCloud<SLRPointXYZIRL>::Ptr _pcNoGround;
    pcl::PointCloud<SLRPointXYZIRL>::Ptr _pcAllLabel;

    ros::NodeHandle _nodeHandle;
    ros::Subscriber _subPointsNode;
    ros::Publisher _pubGroundPoints;
    ros::Publisher _pubGroundlessPoints;
    ros::Publisher _pubAllPoints;

    std::string _pointTopic;

    int _sensorModel;
    double _sensorHeight;
    int _maxIteration;
    int _numLPR;
    double _thresholdSeeds;
    double _thresholDistance;

    bool _flagSaveGroundlessPoints;
    int _frameID;
    std::string _outputDirectory;


    void velodyneCallback(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg);
    void estimatePlane(void);
    void extractInitialSeeds(const pcl::PointCloud<SLRPointXYZIRL>& p_sorted);

    // Model parameter for ground plane fitting
    // The ground plane model is: ax+by+cz+d=0
    // Here normal:=[a,b,c], d=d
    // th_dist_d_ = threshold_dist - d
    float _d;
    MatrixXf _normal;
    float _threshold_d;
};

/*
    @brief Constructor of GPF Node.
    @return void
*/
GroundPlaneFit::GroundPlaneFit():_nodeHandle("~"), _frameID(0) {
    _pcGroundSeeds.reset(new pcl::PointCloud<SLRPointXYZIRL>());
    _pcGround.reset(new pcl::PointCloud<SLRPointXYZIRL>());
    _pcNoGround.reset(new pcl::PointCloud<SLRPointXYZIRL>());
    _pcAllLabel.reset(new pcl::PointCloud<SLRPointXYZIRL>());

    // Init ROS related
    ROS_INFO("[Groundplanfit] [Groundplanfit] Inititalizing Ground Plane Fitter...");
//    _nodeHandle.param<std::string>("point_topic", _pointTopic, "/velodyne_points");
    _nodeHandle.param<std::string>("point_topic", _pointTopic, std::string("/kitti/velo/pointcloud"));
    ROS_INFO("[Groundplanfit] Input Point Cloud: %s", _pointTopic.c_str());

    _nodeHandle.param("sensor_model", _sensorModel, 64);
    ROS_INFO("[Groundplanfit] Sensor Model: %d", _sensorModel);

    _nodeHandle.param("sensor_height", _sensorHeight, 2.2);
    ROS_INFO("[Groundplanfit] Sensor Height: %f", _sensorHeight);

    _nodeHandle.param("max_iter", _maxIteration, 5);
    ROS_INFO("[Groundplanfit] Num of Iteration: %d", _maxIteration);

    _nodeHandle.param("num_lpr", _numLPR, 200);
    ROS_INFO("[Groundplanfit] Num of LPR: %d", _numLPR);

    _nodeHandle.param("th_seeds", _thresholdSeeds, 0.5);    // 1.2
    ROS_INFO("[Groundplanfit] Seeds Threshold: %f", _thresholdSeeds);

    _nodeHandle.param("th_dist", _thresholDistance, 0.3);
    ROS_INFO("[Groundplanfit] Distance Threshold: %f", _thresholDistance);

    _nodeHandle.param("save_no_ground_cloud", _flagSaveGroundlessPoints, false);
    ROS_INFO("[Groundplanfit] Save no_ground_pointcloud: %d", _flagSaveGroundlessPoints);

    _nodeHandle.param<std::string>("output_directory", _outputDirectory, std::string("/home/vance/output"));
    if (_flagSaveGroundlessPoints) {
        ROS_INFO("[Groundplanfit] Output Director for no_ground_pointcloud: %s", _outputDirectory.c_str());
        system(("mkdir -p " + _outputDirectory + "/pcd").c_str());
        system(("mkdir -p " + _outputDirectory + "/bin").c_str());
    }

    // Listen to velodyne topic
    _subPointsNode = _nodeHandle.subscribe(_pointTopic, 2, &GroundPlaneFit::velodyneCallback, this);

    // Publish Init
    std::string groundlessTopic, groundTopic;
    _nodeHandle.param<std::string>("no_ground_point_topic", groundlessTopic, "/points_no_ground");
    ROS_INFO("[Groundplanfit] No Ground Output Point Cloud: %s", groundlessTopic.c_str());

    _nodeHandle.param<std::string>("ground_point_topic", groundTopic, "/points_ground");
    ROS_INFO("[Groundplanfit] Only Ground Output Point Cloud: %s", groundTopic.c_str());

    _pubGroundlessPoints = _nodeHandle.advertise<sensor_msgs::PointCloud2>(groundlessTopic, 2);
    _pubGroundPoints = _nodeHandle.advertise<sensor_msgs::PointCloud2>(groundTopic, 2);
    _pubAllPoints =  _nodeHandle.advertise<sensor_msgs::PointCloud2>("/all_points", 2);
}

/*
    @brief The function to estimate plane model. The
    model parameter `normal_` and `d_`, and `th_dist_d_`
    is set here.
    The main step is performed SVD(UAV) on covariance matrix.
    Taking the sigular vector in U matrix according to the smallest
    sigular value in A, as the `normal_`. `d_` is then calculated
    according to mean ground points.

    @param g_ground_pc:global ground pointcloud ptr.

*/
void GroundPlaneFit::estimatePlane(void){
    // Create covarian matrix in single pass.
    // TODO: compare the efficiency.
    Eigen::Matrix3f cov;
    Eigen::Vector4f pc_mean;
    pcl::computeMeanAndCovarianceMatrix(*_pcGround, cov, pc_mean);
    // Singular Value Decomposition: SVD
    JacobiSVD<MatrixXf> svd(cov, Eigen::DecompositionOptions::ComputeFullU);
    // use the least singular vector as normal
    _normal = (svd.matrixU().col(2));
    // mean ground seeds value
    Eigen::Vector3f seeds_mean = pc_mean.head<3>();

    // according to normal.T*[x,y,z] = -d
    _d = -(_normal.transpose()*seeds_mean)(0,0);
    // set distance threhold to `th_dist - d`
    _threshold_d = _thresholDistance - _d;

    // return the equation parameters
}


/*
    @brief Extract initial seeds of the given pointcloud sorted segment.
    This function filter ground seeds points accoring to heigt.
    This function will set the `g_ground_pc` to `g_seed_pc`.
    @param p_sorted: sorted pointcloud

    @param ::num_lpr_: num of LPR points
    @param ::th_seeds_: threshold distance of seeds
    @param ::

*/
void GroundPlaneFit::extractInitialSeeds(const pcl::PointCloud<SLRPointXYZIRL>& p_sorted){
    // LPR is the mean of low point representative
    double sum = 0;
    int cnt = 0;

    // Calculate the mean height value.
    for (size_t i=0; i<p_sorted.points.size() && cnt<_numLPR; i++){
        sum += p_sorted.points[i].z;
        cnt++;
    }
    double lpr_height = (cnt != 0 ? sum/cnt : 0);// in case divide by 0

    _pcGroundSeeds->clear();

    // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
    for (size_t i=0; i<p_sorted.points.size(); i++){
        if (p_sorted.points[i].z < lpr_height + _thresholdSeeds){
            _pcGroundSeeds->points.push_back(p_sorted.points[i]);
        }
    }
    // return seeds points
}

/*
    @brief Velodyne pointcloud callback function. The main GPF pipeline is here.
    PointCloud SensorMsg -> Pointcloud -> z-value sorted Pointcloud
    ->error points removal -> extract ground seeds -> ground plane fit mainloop
*/
void GroundPlaneFit::velodyneCallback(const sensor_msgs::PointCloud2ConstPtr& in_cloud_msg){
    _pcGroundSeeds.reset(new pcl::PointCloud<SLRPointXYZIRL>());
    _pcGround.reset(new pcl::PointCloud<SLRPointXYZIRL>());
    _pcNoGround.reset(new pcl::PointCloud<SLRPointXYZIRL>());
    _pcAllLabel.reset(new pcl::PointCloud<SLRPointXYZIRL>());

    // 1.Msg to pointcloud
    sensor_msgs::PointCloud2 tmpCloud = *in_cloud_msg;
    tmpCloud.fields[3].name = "intensity";
    pcl::PointCloud<pcl::PointXYZI> laserCloudIn;
    pcl::fromROSMsg(tmpCloud, laserCloudIn);

    // 2.Sort on Z-axis value.
    sort(laserCloudIn.points.begin(), laserCloudIn.end(), point_cmp);
    size_t size = laserCloudIn.size();

    // 3.Error point removal
    // As there are some error mirror reflection under the ground,
    // here regardless point under 2*sensor_height
    // Sort point according to height, here uses z-axis in default
    pcl::PointCloud<VPoint>::iterator it = laserCloudIn.points.begin();
    for(int i=0; i<laserCloudIn.points.size(); i++){
        if (laserCloudIn.points[i].z < -2*_sensorHeight){
            it++;
        } else {
            break;
        }
    }
    laserCloudIn.points.erase(laserCloudIn.points.begin(), it);
//    ROS_INFO("[Groundplanfit] Erase %ld error points for too low z value\n",
//             size - laserCloudIn.size());

    // For mark ground points and hold all points
    SLRPointXYZIRL point;
    for(size_t i = 0; i<laserCloudIn.points.size(); i++){
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;
        point.intensity = laserCloudIn.points[i].intensity;
        point.label = 0u;// 0 means uncluster

        float range = sqrt(point.x * point.x + point.y * point.y);
        point.ring = (atan2(point.z, range) * 180 / M_PI + 24.8)  / ang_res_y;

        if (point.ring < 0 || point.ring >= _sensorModel)
            continue;

        _pcAllLabel->points.push_back(point);
    }
//    ROS_INFO("[Groundplanfit] skip %ld points for wrong ring value",
//             laserCloudIn.size() - _pcAllLabel->size());


    // 4. Extract init ground seeds.
    extractInitialSeeds(*_pcAllLabel);   // set g_seed_pc
    _pcGround = _pcGroundSeeds;

    // 5. Ground plane fitter mainloop
    for(int i=0; i<_maxIteration; i++){
        estimatePlane();

        _pcGround->clear();
        _pcNoGround->clear();

        //pointcloud to matrix
        MatrixXf points(_pcAllLabel->points.size(), 3);
        int j = 0;
        for(auto& p : _pcAllLabel->points){
            points.row(j++) << p.x, p.y, p.z;
        }
        // ground plane model
        VectorXf result = points * _normal;

        // threshold filter
        for(int r=0; r<result.rows(); r++){
            if(result[r] < _threshold_d){
                _pcAllLabel->points[r].label = 1u; // means ground
                _pcGround->points.push_back(_pcAllLabel->points[r]);
            } else {
                _pcAllLabel->points[r].label = 0u;// means not ground and non clusterred
                _pcNoGround->points.push_back(_pcAllLabel->points[r]);
            }
        }
//        ROS_INFO("[Groundplanfit] iteration %d get %ld ground points",
//                 i, _pcGround->size());

        // 至少迭代4次，最多_maxIteration次
        if (i > 2) {
//            ROS_INFO("[Groundplanfit] iteration %d get %ld ground points",
//                     i, _pcGround->size());
            float ratio = static_cast<float>(_pcGround->size()) / _pcAllLabel->size();
            if (ratio > 0.65f)
                break;
        }
    }


    // publish ground points
    sensor_msgs::PointCloud2 ground_msg;
    pcl::toROSMsg(*_pcGround, ground_msg);
    ground_msg.header.stamp = in_cloud_msg->header.stamp;
    ground_msg.header.frame_id = in_cloud_msg->header.frame_id;
    _pubGroundPoints.publish(ground_msg);
    // publish not ground points
    sensor_msgs::PointCloud2 groundless_msg;
    pcl::toROSMsg(*_pcNoGround, groundless_msg);
    groundless_msg.header.stamp = in_cloud_msg->header.stamp;
    groundless_msg.header.frame_id = in_cloud_msg->header.frame_id;
    _pubGroundlessPoints.publish(groundless_msg);
    // publish all points
    sensor_msgs::PointCloud2 all_points_msg;
    pcl::toROSMsg(*_pcAllLabel, all_points_msg);
    all_points_msg.header.stamp = in_cloud_msg->header.stamp;
    all_points_msg.header.frame_id = in_cloud_msg->header.frame_id;
    _pubAllPoints.publish(all_points_msg);


    // save point cloud data
    if (_flagSaveGroundlessPoints)
        saveNoGroundPointcloud();

    _pcAllLabel->clear();
}

// 保存去地面后的点云
void GroundPlaneFit::saveNoGroundPointcloud() {
    size_t num = _pcNoGround->size();
    char filename1[128], filename2[128];
    sprintf(filename1, "%06d.pcd", _frameID);
    sprintf(filename2, "%06d.bin", _frameID);
    std::string filename_pcd, filename_bin;
    filename_pcd = _outputDirectory + "/pcd/" + filename1;
    filename_bin = _outputDirectory + "/bin/" + filename2;

    // save pcd file.
    ROS_INFO("[Groundplanfit] Saving no_ground_pointcloud to %s", filename_pcd.c_str());
    _pcNoGround->height = 1;
    _pcNoGround->width = static_cast<uint32_t>(num);
    pcl::io::savePCDFileASCII(filename_pcd, *_pcNoGround);

/*
    std::ofstream ofs(filename_pcd);
    ofs << "VERSION 0.7\n"
        << "FIELDS x y z intensity ring label\n"
        << "SIZE 4 4 4 4 2 2\n"
        << "TYPE F F F F U U\n"
        << "COUNT 1 1 1 1 1 1\n"
        << "WIDTH " << g_not_ground_pc->size() << std::endl
        << "HEIGHT 1\n"
        << "VIEWPOINT 0 0 0 1 0 0 0\n"
        << "POINTS " << g_not_ground_pc->size() << std::endl
        << "DATA ascii\n";
    for (int i = 0; i < g_all_pc->size(); ++i) {
        if (g_all_pc->points[i].label != 0u)
            continue;
        ofs << std::fixed
            << g_all_pc->points[i].x << " "
            << g_all_pc->points[i].y << " "
            << g_all_pc->points[i].z << " "
            << g_all_pc->points[i].intensity << " "
            << g_all_pc->points[i].ring << " "
            << g_all_pc->points[i].label << std::endl;
    }
    ofs.close();
*/

    // save bin file
    ROS_INFO("[Groundplanfit] Saving no_ground_pointcloud to %s", filename_bin.c_str());
    float* data = (float*)malloc(5*num*sizeof(float));
    for (size_t i = 0; i < num; ++i) {
        data[5*i+0] = _pcNoGround->points[i].x;
        data[5*i+1] = _pcNoGround->points[i].y;
        data[5*i+2] = _pcNoGround->points[i].z;
        data[5*i+3] = _pcNoGround->points[i].intensity;
        data[5*i+4] = _pcNoGround->points[i].label;
    }
    FILE* fb;
    fb = fopen(filename_bin.c_str(), "wb");
    fwrite(data, sizeof(float), 5*num, fb);
    fclose(fb);

    _frameID++;
}

//void GroundPlaneFit::updateParams(points_preprocessor_usi::paramsConfig &config,
//                                  uint32_t level) {
//    _sensorHeight = config.sensor_height;
//    _numSegmentation = config.num_seg;
//    _numIteration = config.num_iter;
//    _numLPR = config.num_lpr;
//    _thresholdSeeds = config.th_seeds;
//    _thresholDistance = config.th_dist;

//    ROS_INFO("[Groundplanfit] Parameters updated!");
//}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "GroundPlaneFit");

//    dynamic_reconfigure::Server<points_preprocessor_usi::paramsConfig> server;
//    dynamic_reconfigure::Server<points_preprocessor_usi::paramsConfig>::CallbackType f;

    GroundPlaneFit node;

//    f = boost::bind(&GroundPlaneFit::updateParams, boost::ref(node), _1, _2);
//    server.setCallback(f);

    ros::spin();

    return 0;

}
