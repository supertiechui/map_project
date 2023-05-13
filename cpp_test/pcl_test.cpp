//
// Created by heht on 23-4-3.
//
#include <iostream>
#include <filesystem> // requires gcc version >= 8

#include <pcl/io/pcd_io.h>    //pcd流头文件
#include <pcl/point_types.h>  //点类型即现成的pointT
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/gicp.h>
#include <queue>

#include "Eigen/Core"

#define pi 3.1415926
namespace fs = std::filesystem;
using std::ios;
using std::cout;
using std::cerr;
using std::endl;
typedef pcl::PointXYZI PointType;

std::vector<double> splitPoseLine(std::string _str_line, char _delimiter) {
    std::vector<double> parsed;
    std::stringstream ss(_str_line);
    std::string temp;
    while (getline(ss, temp, _delimiter)) {
        parsed.push_back(std::stod(temp)); // convert string to "double"
    }
    return parsed;
}

void get_scan_pose(const std::string sequence_scan_dir_, std::vector<std::string> &sequence_scan_names_, std::vector<std::string> &sequence_scan_paths_,
                   const std::string sequence_pose_path_, std::vector<Eigen::Matrix4d> &sequence_scan_poses_, std::vector<Eigen::Matrix4d> &sequence_scan_inverse_poses_)
{
    // sequence bin files


    for(auto& _entry : fs::directory_iterator(sequence_scan_dir_)) {
        sequence_scan_names_.emplace_back(_entry.path().filename());
        sequence_scan_paths_.emplace_back(_entry.path());
    }
    std::sort(sequence_scan_names_.begin(), sequence_scan_names_.end());
    std::sort(sequence_scan_paths_.begin(), sequence_scan_paths_.end());

    int num_total_scans_of_sequence_ = sequence_scan_paths_.size();
    cout<<"\033[1;32m Total : " << num_total_scans_of_sequence_ << " scans in the directory.\033[0m"<<endl;

    // sequence pose file
    std::ifstream pose_file_handle (sequence_pose_path_);
    int num_poses {0};
    std::string strOneLine;
    while (getline(pose_file_handle, strOneLine))
    {
        // str to vec
        std::vector<double> ith_pose_vec = splitPoseLine(strOneLine, ' ');
        if(ith_pose_vec.size() == 12) {
            ith_pose_vec.emplace_back(double(0.0));
            ith_pose_vec.emplace_back(double(0.0));
            ith_pose_vec.emplace_back(double(0.0));
            ith_pose_vec.emplace_back(double(1.0));
        }

        // vec to eig
        Eigen::Matrix4d ith_pose = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(ith_pose_vec.data(), 4, 4);
        Eigen::Matrix4d ith_pose_inverse = ith_pose.inverse();

        // save (move)
//        cout << "Pose of scan: " << sequence_scan_names_.at(num_poses) << endl;
//        cout << ith_pose << endl;
        sequence_scan_poses_.emplace_back(ith_pose);
        sequence_scan_inverse_poses_.emplace_back(ith_pose_inverse);

        num_poses++;
    }
    // check the number of scans and the number of poses are equivalent
    assert(sequence_scan_paths_.size() == sequence_scan_poses_.size());
}

int gicp(const pcl::PointCloud<PointType>::Ptr src_cloud,
         const pcl::PointCloud<PointType>::Ptr tgt_cloud,
         pcl::PointCloud<PointType>::Ptr transformed_source)
{
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    int iterations = 50;
    icp.setMaximumIterations(iterations);
    icp.setInputSource(src_cloud);
    icp.setInputTarget(tgt_cloud);
//    icp.setRANSACOutlierRejectionThreshold(0.0);
    icp.align(*transformed_source);
    icp.setMaximumIterations(1);

    if (icp.hasConverged())
    {
        std::cout << "\nICP has converged, score is " << icp.getFitnessScore() << std::endl;
        std::cout << "\nICP transformation " << iterations << " : cloud_icp -> cloud_in" << std::endl;
        Eigen::Matrix4d transformation_matrix = icp.getFinalTransformation().cast<double>();
        cout<<transformation_matrix<<endl;
    }
    else
    {
        PCL_ERROR("\nICP has not converged.\n");
        return (-1);
    }

    // Visualization
    pcl::visualization::PCLVisualizer viewer("ICP demo");
    // Create two vertically separated viewports
    int v1(0);
    int v2(1);
    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

    // The color we will be using
    float bckgr_gray_level = 0.0;  // Black
    float txt_gray_lvl = 1.0 - bckgr_gray_level;

    // Original point cloud is white
    pcl::visualization::PointCloudColorHandlerCustom<PointType> cloud_in_color_h(src_cloud, (int)255 * txt_gray_lvl, (int)255 * txt_gray_lvl,
                                                                              (int)255 * txt_gray_lvl);
    viewer.addPointCloud(src_cloud, cloud_in_color_h, "cloud_in_v1", v1);
    viewer.addPointCloud(src_cloud, cloud_in_color_h, "cloud_in_v2", v2);

    // Transformed point cloud is green
    pcl::visualization::PointCloudColorHandlerCustom<PointType> cloud_tr_color_h(tgt_cloud, 20, 180, 20);
    viewer.addPointCloud(tgt_cloud, cloud_tr_color_h, "cloud_tr_v1", v1);

    // ICP aligned point cloud is red
    pcl::visualization::PointCloudColorHandlerCustom<PointType> cloud_icp_color_h(transformed_source, 180, 20, 20);
    viewer.addPointCloud(transformed_source, cloud_icp_color_h, "cloud_icp_v2", v2);

    // Adding text descriptions in each viewport
    viewer.addText("White: Original point cloud\nGreen: Matrix transformed point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_1", v1);
    viewer.addText("White: Original point cloud\nRed: ICP aligned point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_2", v2);

    std::stringstream ss;
    ss << iterations;
    std::string iterations_cnt = "ICP iterations = " + ss.str();
    viewer.addText(iterations_cnt, 10, 60, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "iterations_cnt", v2);

    // Set background color
    viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
    viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v2);

    // Set camera position and orientation
    viewer.setCameraPosition(0, 0, 150, 0, 1, 0, 0);
//    viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
    viewer.setSize(1280, 800);
    //    viewer.setSize(1280, 1024);  // Visualiser window size


    // Display the visualiser
    while (!viewer.wasStopped())
    {

    }

    return 1;
}

int main (int argc, char** argv)
{
//    std::string scan_dir_ = "/home/heht/kitti/new/01";
    std::string scan_dir_ = "/home/heht/kitti/sequences/01/velodyne";
    std::vector<std::string> scan_names_;
    std::vector<std::string> scan_paths_;
    std::string pose_path_ = "/home/heht/kitti/sequences/01/poses.txt";
    std::vector<Eigen::Matrix4d> poses_;
    std::vector<Eigen::Matrix4d> inverse_poses; // used for global to local
    std::vector<double> kVecExtrinsicLiDARtoPoseBase =  {-1.857e-03, -9.999e-01, -8.039e-03, -4.784e-03,
                                                         -6.481e-03,  8.0518e-03, -9.999e-01, -7.337e-02,
                                                         9.999e-01, -1.805e-03, -6.496e-03, -3.339e-01,
                                                         0.0,        0.0,        0.0,        1.0};
    Eigen::Matrix4d kSE3MatExtrinsicLiDARtoPoseBase = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(kVecExtrinsicLiDARtoPoseBase.data(), 4, 4);;
    // Base is where of the pose writtened (e.g., for KITTI, poses is usually in camera)
    get_scan_pose(scan_dir_, scan_names_, scan_paths_,
                  pose_path_, poses_,inverse_poses);




    pcl::PointCloud<PointType>::Ptr cloud1(new pcl::PointCloud<PointType>); // 创建点云（指针）
    pcl::PointCloud<PointType>::Ptr cloud2(new pcl::PointCloud<PointType>); // 创建点云（指针）
    pcl::PointCloud<PointType>::Ptr cloud3(new pcl::PointCloud<PointType>); // 创建点云（指针）

//    pcl::io::loadPCDFile<PointType>(scan_paths_[435], *cloud1);
//    pcl::io::loadPCDFile<PointType>(scan_paths_[438], *cloud2);
//    gicp(cloud2, cloud1, cloud3);
    pcl::visualization::CloudViewer viewer("vviie ");
    viewer.showCloud(cloud3);
    while(!viewer.wasStopped())
    {

    }
//    return (0);
}
