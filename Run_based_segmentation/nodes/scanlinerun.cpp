/*
    @file scanlinerun.cpp
    @brief ROS Node for scan line run

    This is a ROS node to perform scan line run clustring.
    Implementation accoriding to <Fast Segmentation of 3D Point Clouds: A Paradigm>

    In this case, it's assumed that the x,y axis points at sea-level,
    and z-axis points up. The sort of height is based on the Z-axis value.

    @author Vincent Cheung(VincentCheungm)
    @bug .
*/

#include <iostream>
#include <forward_list>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <velodyne_pointcloud/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>

#include <opencv2/core/core.hpp>

//#include <dynamic_reconfigure/server.h>
//#include <points_preprocessor_usi/paramsConfig.h>

//struct ScanLinePoint;
using namespace std;


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

// @Vance: 两点属于same run只算水平距离合理性？？？
//#define dist(a,b) sqrt(((a).x-(b).x)*((a).x-(b).x)+((a).y-(b).y)*((a).y-(b).y))

double dist(const SLRPointXYZIRL& a, const SLRPointXYZIRL& b) {
    return sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z));
}

//struct ScanLinePoint {
//    float ring;
//    float row;

//    ScanLinePoint(float a, float b): ring(a), row(b) {}

//    bool operator < (const struct ScanLinePoint& slp) const {
//        return this->ring < slp.ring ? true : (this->row < slp.row);
//    }
//};



/*
    @brief Scan Line Run ROS Node.
    @param Velodyne Pointcloud Non Ground topic.
    @param Sensor Model.
    @param Threshold between points belong to the same run
    @param Threshold between runs

    @subscirbe:/all_points
    @publish:/slr
*/
class ScanLineRun{
public:
    ScanLineRun();
//    void updateParams(points_preprocessor_usi::paramsConfig &config, uint32_t level);

private:
    ros::NodeHandle _nodeHandle;
    ros::Subscriber _subPointsNode;
    ros::Publisher _pubClusterPoints;
    ros::Publisher _pubRingPoints;

    std::string _pointTopic;

    int _sensorModel;       // also means number of sensor scan line.
    double _thresholdRun;   // thresold of distance of points belong to the same run.
    double _thresholdMerge; // threshold of distance of runs to be merged.

    // For organization of points.
    std::vector<std::vector<SLRPointXYZIRL> > _laserFrame;
    std::vector<SLRPointXYZIRL> _laserRow;

    std::vector<std::forward_list<SLRPointXYZIRL*> > _runs; // For holding all runs.
//    std::vector<std::set<ScanLinePoint>> _allRunsIndex;

    uint16_t _maxLabel;// max run labels, for disinguish different runs.
    std::vector<std::vector<size_t> > _nogroundIndex;// non ground point index.

    // Call back funtion.
    void velodyneCallback(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg);
    // For finding runs on a scanline.
    void findRuns(size_t scanline);
    // For update points cluster label after merge action.
    void updateLabels(size_t scanline);
    // For merge `current` run to `target` run.
    void mergeRuns(uint16_t cur_label, uint16_t target_label);

    /// @deprecated methods for smart index
    // Smart idx according to paper, but not useful in my case.
    int smartIndex(int local_idx, int n_i, int n_j, bool inverse);

    // Dummy object to occupy idx 0.
    std::forward_list<SLRPointXYZIRL*> _dummy;

    cv::Mat _laserFrameLabel;
    std::vector<pcl::PointCloud<SLRPointXYZIRL>> _allClusters;
};

/*
    @brief Constructor of SLR Node.
    @return void
*/
ScanLineRun::ScanLineRun(): _nodeHandle("~") {
    // Init ROS related
    ROS_INFO("[Scanlinerun  ] Inititalizing Scan Line Run Cluster...");
//    _nodeHandle.param<std::string>("point_topic", _pointTopic, "/all_points");
    _nodeHandle.param<std::string>("point_topic", _pointTopic, "/points_no_ground");
    ROS_INFO("[Scanlinerun  ] point_topic: %s", _pointTopic.c_str());

    _nodeHandle.param("sensor_model", _sensorModel, 64);
    ROS_INFO("[Scanlinerun  ] Sensor Model: %d", _sensorModel);

    // Init Ptrs with vectors
    for (int i=0; i<_sensorModel; i++) {
        std::vector<size_t> dummy_vec;
        _nogroundIndex.push_back(dummy_vec);
    }

    // Init LiDAR frames with vectors and points
    SLRPointXYZIRL p_dummy;
    p_dummy.intensity = -1;// Means unoccupy by any points
//    laser_row_ = std::vector<SLRPointXYZIRL>(2251, p_dummy);
    _laserRow = std::vector<SLRPointXYZIRL>(1800, p_dummy);
    _laserFrame = std::vector<std::vector<SLRPointXYZIRL> >(_sensorModel, _laserRow);
    _laserFrameLabel = cv::Mat::zeros(_sensorModel, 1800, CV_8UC1);

    // Init runs, idx 0 for interest point, and idx 1 for ground points
    _maxLabel = 1;
    _runs.push_back(_dummy);
    _runs.push_back(_dummy);
//    _allRunsIndex.resize(20000);

    _nodeHandle.param("th_run", _thresholdRun, 0.15);
    ROS_INFO("[Scanlinerun  ] Point-to-Run Threshold: %f", _thresholdRun);

    _nodeHandle.param("th_merge", _thresholdMerge, 0.5);
    ROS_INFO("[Scanlinerun  ] RUN-to-RUN Distance Threshold: %f", _thresholdMerge);

    // Subscriber to velodyne topic
    _subPointsNode = _nodeHandle.subscribe(_pointTopic, 2, &ScanLineRun::velodyneCallback, this);

    // Publisher Init
    std::string cluster_topic;
    _nodeHandle.param<std::string>("cluster", cluster_topic, "/slr");
    ROS_INFO("[Scanlinerun  ] Cluster Output Point Cloud: %s", cluster_topic.c_str());
    _pubClusterPoints = _nodeHandle.advertise<sensor_msgs::PointCloud2 >(cluster_topic, 10);
}


//void ScanLineRun::updateParams(points_preprocessor_usi::paramsConfig &config,
//                               uint32_t level) {
//    _thresholdRun = config.th_run;
//    _thresholdMerge = config.th_merge;

//    printf("[Scanlinerun  ] Parameters updated!\n");
//}


/*
    @brief Read points from the given scan_line.
    The distance of two continuous points will be labelled to the same run.
    Clusterred points(`Runs`) stored in `runs_[cluster_id]`.

    @param scan_line: The scan line to find runs.
    @return void
*/
void ScanLineRun::findRuns(size_t scanLine) {
    // If there is no non-ground points of current scanline, skip.
    size_t point_size = _nogroundIndex[scanLine].size();
    if (point_size <= 0)
        return;

    size_t non_g_pt_idx = _nogroundIndex[scanLine][0]; // The first non ground point
    size_t non_g_pt_idx_l = _nogroundIndex[scanLine][point_size - 1]; // The last non ground point

    /* Iterate all non-ground points, and compute and compare the distance
    of each two continous points. At least two non-ground points are needed.
    */
    for (size_t i_idx = 0; i_idx < point_size - 1; i_idx++) {
        size_t i = _nogroundIndex[scanLine][i_idx];
        size_t i1 = _nogroundIndex[scanLine][i_idx+1];

        if (i_idx == 0) {
            // The first point, make a new run.
            SLRPointXYZIRL& p_0 = _laserFrame[scanLine][i];
            _maxLabel += 1;
            _runs.push_back(_dummy);
            _laserFrame[scanLine][i].label = _maxLabel;
            _runs[p_0.label].insert_after(_runs[p_0.label].cbefore_begin(), &_laserFrame[scanLine][i]);

//            _allRunsIndex[p_0.label].insert(ScanLinePoint(scanLine,i));

//            printf("p_0.label == %d\n", p_0.label);

//            _laserFrameLabel.at<uchar>(scanLine, i) = 1;
            if (p_0.label == 0)
                ROS_ERROR("p_0.label == 0");
        }

        // Compare with the next point
        SLRPointXYZIRL& p_i = _laserFrame[scanLine][i];
        SLRPointXYZIRL& p_i1 = _laserFrame[scanLine][i1];

        // If next point is ground point, skip.
        if (p_i1.label == 1u) {
            // Add to ground run `runs_[1]` (ground)
            _runs[p_i1.label].insert_after(_runs[p_i1.label].cbefore_begin(), &_laserFrame[scanLine][i1]);
            ROS_ERROR("ground point in noground points!");
            continue;
        }

        /* If cur point is not ground and next point is within threshold,
         * then make it the same run. Else, to make a new run.
         */
        if (p_i.label != 1u && dist(p_i, p_i1) < _thresholdRun) {
            p_i1.label = p_i.label;
        } else {
            _maxLabel += 1;
            p_i1.label = _maxLabel;
            _runs.push_back(_dummy);
        }

        // Insert the index.
        _runs[p_i1.label].insert_after(_runs[p_i1.label].cbefore_begin(), &_laserFrame[scanLine][i1]);
//        _allRunsIndex[p_i1.label].insert(ScanLinePoint(scanLine,i1));

        if (p_i1.label == 0)
            ROS_ERROR("p_i1.label == 0");
    }

    // Compare the last point and the first point, for laser scans is a ring.
    if (point_size > 1) {
        SLRPointXYZIRL &p_0 = _laserFrame[scanLine][non_g_pt_idx];
        SLRPointXYZIRL &p_l = _laserFrame[scanLine][non_g_pt_idx_l];

        // Skip, if one of the start point or the last point is ground point.
        if (p_0.label == 1u || p_l.label == 1u) {
            return ;
        } else if (dist(p_0, p_l) < _thresholdRun) {
            if (p_0.label == 0) {
                ROS_ERROR("Ring Merge to 0 label");
            }
            /// If next point is within threshold, then merge it into the same run.
            mergeRuns(p_l.label, p_0.label);
        }
    } else if (point_size == 1) {
            // The only point, make a new run.
            SLRPointXYZIRL& p_0 = _laserFrame[scanLine][non_g_pt_idx];
            _maxLabel += 1;
            _runs.push_back(_dummy);
            _laserFrame[scanLine][non_g_pt_idx].label = _maxLabel;
            _runs[p_0.label].insert_after(_runs[p_0.label].cbefore_begin(), &_laserFrame[scanLine][non_g_pt_idx]);
//            _allRunsIndex[p_0.label].insert(ScanLinePoint(scanLine,non_g_pt_idx));
    }

}


/*
    @brief Update label between points and their smart `neighbour` point
    above `scan_line`.

    @param scan_line: The current scan line number.
*/
void ScanLineRun::updateLabels(size_t scanLine) {
    // Iterate each point of this scan line to update the labels.
    size_t point_size_j_idx = _nogroundIndex[scanLine].size();
    // Current scan line is emtpy, do nothing.
    if (point_size_j_idx == 0)
        return;

    // Iterate each point of this scan line to update the labels.
    for (size_t j_idx = 0; j_idx < point_size_j_idx; j_idx++) {
        size_t j = _nogroundIndex[scanLine][j_idx];


        SLRPointXYZIRL& p_j = _laserFrame[scanLine][j];

        // Runs above from scan line 0 to scan_line
        for (int l = scanLine - 1; l >= 0; l--) {
            if (_nogroundIndex[l].size() == 0)
                continue;

            // Smart index for the near enough point, after re-organized these points.
            size_t nn_idx = j;

            // 跳过地面点和未标记点
            if (_laserFrame[l][nn_idx].label == -1u || _laserFrame[l][nn_idx].label == 1u) {
                continue;
            }

            // Nearest neighbour point
            SLRPointXYZIRL& p_nn = _laserFrame[l][nn_idx];
            // Skip, if these two points already belong to the same run.
            if (p_j.label == p_nn.label) {
                continue;
            }
            double dist_min = dist(p_j, p_nn);

            /* Otherwise,
            If the distance of the `nearest point` is within `th_merge_`,
            then merge to the smaller run.
            */
            if (dist_min < _thresholdMerge) {
                uint16_t  cur_label = 0, target_label = 0;

                if (p_j.label == 0 || p_nn.label == 0) {
                    ROS_ERROR("p_j.label:%u, p_nn.label:%u", p_j.label, p_nn.label);
                }

                // Merge to a smaller label cluster
                if (p_j.label > p_nn.label) {
                    cur_label = p_j.label;
                    target_label = p_nn.label;
                } else {
                    cur_label = p_nn.label;
                    target_label = p_j.label;
                }

                // Merge these two runs.
                mergeRuns(cur_label, target_label);
            }
        }
    }

}

/*
    @brief Merge current run to the target run.

    @param cur_label: The run label of current run.
    @param target_label: The run label of target run.
*/
void ScanLineRun::mergeRuns(uint16_t cur_label, uint16_t target_label) {
    if (cur_label == 0 || target_label == 0) {
        ROS_ERROR("Error merging runs cur_label:%u target_label:%u", cur_label, target_label);
    }
    // First, modify the label of current run.
    for (auto& p : _runs[cur_label]) {
        p->label = target_label;
    }
    // Then, insert points of current run into target run.
    _runs[target_label].insert_after(_runs[target_label].cbefore_begin(), _runs[cur_label].begin(),_runs[cur_label].end() );
    _runs[cur_label].clear();

//    for (auto& p : _allRunsIndex[cur_label]) {
//        _allRunsIndex[target_label].insert(p);
//    }
//    _allRunsIndex[cur_label].clear();
}

/*
    @brief Smart index for nearest neighbour on scanline `i` and scanline `j`.

    @param local_idx: The local index of point on current scanline.
    @param n_i: The number of points on scanline `i`.
    @param n_j: The number of points on scanline `j`.
    @param inverse: If true, means `local_idx` is on the outsider ring `j`.
    Otherwise, it's on the insider ring `i`.

    @return The smart index.
*/
/*
//[[deprecated("Not useful in my case.")]]
int ScanLineRun::smartIndex(int local_idx, int n_i, int n_j, bool inverse=false) {
    if (inverse==false) {
        // In case of zero-divide.
        if (n_i == 0 ) return 0;
        float rate = (n_j*1.0f)/n_i;
        int idx = floor(rate*local_idx);

        // In case of overflow
        if (idx>n_j) {
            idx = n_j>1?n_j-1:0;
        }
        return idx;
    }else{
        // In case of zero-divide.
        if (n_j == 0 ) return 0;
        float rate = (n_i*1.0f)/n_j;
        int idx = ceil(rate*local_idx);

        // In case of overflow
        if (idx>n_i) {
            idx = n_i>1?n_i-1:0;
        }
        return idx;
    }

}
*/





/*
    @brief Velodyne pointcloud callback function, which subscribe `/all_points`
    and publish cluster points `slr`.
*/
void ScanLineRun::velodyneCallback(const sensor_msgs::PointCloud2ConstPtr& in_cloud_msg) {
    // Msg to pointcloud
    pcl::PointCloud<SLRPointXYZIRL> laserCloudIn;
    pcl::fromROSMsg(*in_cloud_msg, laserCloudIn);

    /// Clear and init.
    // Clear runs in the previous scan.
    _maxLabel = 1;
    if (!_runs.empty()) {
        _runs.clear();
        _runs.push_back(_dummy);// dummy for index `0`
        _runs.push_back(_dummy);// for ground points

//        _allRunsIndex.clear();
    }

    // Init laser frame.
    SLRPointXYZIRL p_dummy;
    p_dummy.intensity = -1;

    _laserFrame = std::vector<std::vector<SLRPointXYZIRL> >(_sensorModel, _laserRow);
    for (int i=0; i<_sensorModel; i++) {
        _nogroundIndex[i].clear();
    }

    // Organize Pointcloud in scanline
    double range = 0;
    int row = 0;
    int nNoGround = 0;
    for (auto& point : laserCloudIn.points) {
        if (point.ring < _sensorModel && point.ring >= 0) {
            // Compute and angle.
            // @Note: In this case, `x` points right and `y` points forward.
//            range = sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
            range = sqrt(point.x*point.x + point.y*point.y);
            if (point.x >= 0) { // 从1/4处算
//                row = int(563 - asin(point.y/range)/0.00279111);
                row = int(450 - asin(point.y/range)/0.003491);
            } else if (point.x < 0 && point.y <= 0) { // 从3/4处算
//                row = int(1688 + asin(point.y/range)/0.00279111);
                row = int(1350 + asin(point.y/range)/0.003491);
            } else {
//                row = int(1688 + asin(point.y/range)/0.00279111);
                row = int(1350 + asin(point.y/range)/0.003491);
            }

            if (row >= 1800 || row < 0) {  // 2250
                ROS_ERROR("[Scanlinerun  ] Row: %d is out of index.", row);
                continue;
            } else {
                _laserFrame[point.ring][row] = point;
            }

            // 1u为地面点,这里非地面点保存到nogroundIndex
            if (point.label != 1u) {
                _nogroundIndex[point.ring].push_back(static_cast<size_t>(row));
                nNoGround++;
            } else {
                _runs[1].insert_after(_runs[1].cbefore_begin(), &point);
//                _allRunsIndex[1].insert(ScanLinePoint(point.ring, row));
            }
        }
    }
    printf("[scanlinrun] Get %d no ground points.\n", nNoGround);

    // Main processing
    for (size_t i=0; i<static_cast<size_t>(_sensorModel); i++) {
        sort(_nogroundIndex[i].begin(), _nogroundIndex[i].end());

        // get runs on current scan line i
        findRuns(i);
        updateLabels(i);
    }

    // Extract Clusters
    // re-organize scan-line points into cluster point cloud
    pcl::PointCloud<SLRPointXYZIRL>::Ptr laserCloud(new pcl::PointCloud<SLRPointXYZIRL>());
    pcl::PointCloud<SLRPointXYZIRL>::Ptr clusters(new pcl::PointCloud<SLRPointXYZIRL>());

    int cnt = 0, numClusters = 0;

    // Re-organize pointcloud clusters for PCD saving or publish
    for (size_t i=2; i<_runs.size(); i++) {
        if (!_runs[i].empty()) {
            cnt++;

            int ccnt = 0;
            // adding run current for publishing
            for (auto& p : _runs[i]) {
                // Reorder the label id
                ccnt++;
                p->label = cnt;
                laserCloud->points.push_back(*p);
                // clusters->points.push_back(*p);
            }
            // clusters->clear();
        }
    }

//    for (size_t i=2; i<_allRunsIndex.size(); i++) {
//        if (!_allRunsIndex[i].empty()) {
//            numClusters++;

//            for (auto& p : _allRunsIndex[i]) {
//                SLRPointXYZIRL point = _laserFrame[p.ring][p.row];
//                point.label = numClusters;
//                clusters->points.push_back(point);
//            }
//        }
//    }
//    printf("[Scanlinerun  ] Total cluster: %d\n", numClusters);

    // Publish Cluster Points
    if (laserCloud->points.size() > 0) {
        sensor_msgs::PointCloud2 cluster_msg;
        pcl::toROSMsg(*laserCloud, cluster_msg);
//        pcl::toROSMsg(*clusters, cluster_msg);
        cluster_msg.header.frame_id = "/velodyne";
        _pubClusterPoints.publish(cluster_msg);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ScanLineRun");


//    dynamic_reconfigure::Server<points_preprocessor_usi::paramsConfig> server;
//    dynamic_reconfigure::Server<points_preprocessor_usi::paramsConfig>::CallbackType f;

    ScanLineRun node;

//    f = boost::bind(&ScanLineRun::updateParams, boost::ref(node), _1, _2);
//    server.setCallback(f);


    ros::spin();

    return 0;

}
