#include <ros/ros.h>

#include <dynamic_reconfigure/server.h>
#include <points_preprocessor_usi/paramsConfig.h>

bool callback(points_preprocessor_usi::paramsConfig  &config, uint32_t level)
{
    ROS_INFO("Reconfigure Request: %f %d %d %d %f %f %f %f",
             config.sensor_height,
             config.num_seg,
             config.num_iter,
             config.num_lpr,
             config.th_seeds,
             config.th_dist,
             config.th_run,
             config.th_merge);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "dynamic_reconfigure_node");
    ros::NodeHandle nh;

    dynamic_reconfigure::Server<points_preprocessor_usi::paramsConfig> server;
    dynamic_reconfigure::Server<points_preprocessor_usi::paramsConfig>::CallbackType f;

    f = boost::bind(&callback, _1, _2);

    server.setCallback(f);

    ROS_INFO("Spinning node");

    ros::spin();


    return 0;
}
