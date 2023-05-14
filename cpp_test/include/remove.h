//
// Created by heht on 23-4-20.
//

#ifndef CPP_TEST_REMOVE_H
#define CPP_TEST_REMOVE_H

#include "utility.h"
#include <ikd_Tree.h>
using PointVector = KD_TREE<PointType>::PointVector;
template class KD_TREE<PointType>;
KD_TREE<PointType>::Ptr kdtree_ptr;
KD_TREE<PointType> ikd_Tree;

class Removerter {
private:
    std::string pointcloud_topic;
    // Patchwork++ initialization


    // removert params
    float kVFOV = 50;
    float kHFOV = 360;
    std::pair<float, float> kFOV = {50, 360};

    // sequence info
//    nh.param<std::vector<double>>("removert/ExtrinsicLiDARtoPoseBase", kVecExtrinsicLiDARtoPoseBase, std::vector<double>());
//    kSE3MatExtrinsicLiDARtoPoseBase = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(kVecExtrinsicLiDARtoPoseBase.data(), 4, 4);
//    kSE3MatExtrinsicPoseBasetoLiDAR = kSE3MatExtrinsicLiDARtoPoseBase.inverse();

    std::vector<double> kVecExtrinsicLiDARtoPoseBase =  {-1.857e-03, -9.999e-01, -8.039e-03, -4.784e-03,
                                                        -6.481e-03,  8.0518e-03, -9.999e-01, -7.337e-02,
                                                        9.999e-01, -1.805e-03, -6.496e-03, -3.339e-01,
                                                        0.0,        0.0,        0.0,        1.0};
    Eigen::Matrix4d kSE3MatExtrinsicLiDARtoPoseBase = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(kVecExtrinsicLiDARtoPoseBase.data(), 4, 4);;
    // Base is where of the pose writtened (e.g., for KITTI, poses is usually in camera)
    // if the pose file is obtained via lidar odometry itself, then kMatExtrinsicLiDARtoBase is eye(4)
    Eigen::Matrix4d kSE3MatExtrinsicPoseBasetoLiDAR = kSE3MatExtrinsicLiDARtoPoseBase.inverse();

    // sequence bin files
    bool isScanFileKITTIFormat_ = false;

    std::string sequence_scan_dir_;
    std::vector<std::string> sequence_scan_names_;
    std::vector<std::string> sequence_scan_paths_;
    int num_total_scans_of_sequence_;
    float kDownsampleVoxelSize = 0.05;

    // sequence pose file
    std::string sequence_pose_path_;
    std::vector<Eigen::Matrix4d> sequence_scan_poses_;
    std::vector<Eigen::Matrix4d> sequence_scan_inverse_poses_; // used for global to local

    // target region to removerting
    int start_idx_ = 1;
    int end_idx_ = 991;
    int initilize_idx = 20;

    bool use_keyframe_gap_ = true;
    bool use_keyframe_meter_ = false;
    int keyframe_gap_ = 1;
    float keyframe_gap_meter_ = 2;

    //
    std::vector<float> remove_resolution_list_ = {2.5, 2.0, 1.5};
    std::vector<float> revert_resolution_list_ = {1.0, 0.9, 0.8, 0.7};

    //
    int kNumOmpCores = 16;

    //
    pcl::PointCloud<PointType>::Ptr single_scan;
    pcl::PointCloud<PointType>::Ptr projected_scan;

    float rimg_color_min_ = 0;
    float rimg_color_max_ = 20;
    std::pair<float, float> kRangeColorAxis = {rimg_color_min_, rimg_color_max_}; // meter
    std::pair<float, float> kRangeColorAxisForDiff = {0.0, 0.5}; // meter


    //
    bool kFlagSaveMapPointcloud = true;
    bool kFlagSaveCleanScans = true;
    std::string save_pcd_directory_;

    const float kFlagNoPOINT = 10000.0; // no point constant, 10000 has no meaning, but must be larger than the maximum scan range (e.g., 200 meters)
    const float kValidDiffUpperBound = 200.0; // must smaller than kFlagNoPOINT

    // Static sensitivity
    int kNumKnnPointsToCompare = 2;// static sensitivity (increase this value, less static structure will be removed at the scan-side removal stage)
    float kScanKnnAndMapKnnAvgDiffThreshold = 0.1; // static sensitivity (decrease this value, less static structure will be removed at the scan-side removal stage)

    std::vector<std::string> sequence_valid_scan_names_;
    std::vector<std::string> sequence_valid_scan_paths_;
    std::vector<pcl::PointCloud<PointType>::Ptr> scans_;
    std::vector<pcl::PointCloud<PointType>::Ptr> scans_unground;
    std::vector<pcl::PointCloud<PointType>::Ptr> scans_ground;
    std::vector<pcl::PointCloud<PointType>::Ptr> scans_static_;
    std::vector<pcl::PointCloud<PointType>::Ptr> scans_dynamic_;

    std::string scan_static_save_dir_ = "/home/heht/PCD_/Scans/static";
    std::string scan_dynamic_save_dir_ = "/home/heht/PCD_/Scans/dynamic";
    std::string map_static_save_dir_ = "/home/heht/PCD_/Map";
    std::string map_dynamic_save_dir_ = "/home/heht/PCD_/Map";
    std::string scan_range_save_dir;
    std::string map_range_save_dir;
    std::string diff_range_save_dir;

    std::vector<Eigen::Matrix4d> scan_poses_;
    std::vector<Eigen::Matrix4d> scan_inverse_poses_;

    pcl::PointCloud<PointType>::Ptr map_scan;
    pcl::KdTreeFLANN<PointType>::Ptr kdtree_map_scan;



    pcl::KdTreeFLANN<PointType>::Ptr kdtree_map_global_curr_;
    pcl::KdTreeFLANN<PointType>::Ptr kdtree_scan_global_curr_;


    pcl::PointCloud<PointType>::Ptr map_global_orig_;


    pcl::PointCloud<PointType>::Ptr map_global_curr_; // the M_i. i.e., removert is: M1 -> S1 + D1, D1 -> M2 , M2 -> S2 + D2 ... repeat ...
    pcl::PointCloud<PointType>::Ptr map_local_curr_;

    pcl::PointCloud<PointType>::Ptr map_subset_global_curr_;

    pcl::PointCloud<PointType>::Ptr map_global_curr_static_; // the S_i
    pcl::PointCloud<PointType>::Ptr map_global_curr_dynamic_;  // the D_i

    pcl::PointCloud<PointType>::Ptr map_global_accumulated_static_; // TODO, the S_i after reverted
    pcl::PointCloud<PointType>::Ptr map_global_accumulated_dynamic_;  // TODO, the D_i after reverted

    std::vector<pcl::PointCloud<PointType>::Ptr> static_map_global_history_; // TODO
    std::vector<pcl::PointCloud<PointType>::Ptr> dynamic_map_global_history_; // TODO

    float curr_res_alpha_; // just for tracking current status

    const int base_node_idx_ = 0;

    unsigned long kPauseTimeForClearStaticScanVisualization = 1000; // microsec

    // NOT recommend to use for under 5 million points map input (becausing not-using is just faster)
    const bool kUseSubsetMapCloud = false;
    const float kBallSize = 80.0; // meter

public:
    Removerter();
    ~Removerter();
    void cleanIntensity(pcl::PointCloud<PointType>::Ptr& points);
    void scanMap(
            const std::vector<pcl::PointCloud<PointType>::Ptr>& _scans,
            const std::vector<Eigen::Matrix4d>& _scans_poses,
            pcl::PointCloud<PointType>::Ptr& _ptcloud_to_save );
    float globalPointDistance(PointType p, int idx);
    float kdtreeDistance_global(PointType p, int idx);
    void scanMap( int idx, pcl::PointCloud<PointType>::Ptr& _ptcloud_to_save);
    void findDynamicPointsOfScanByKnn ( int _scan_idx );
    void slideWindows();
    void initizlize();
    void CloudSubtraction(pcl::PointCloud<PointType>::Ptr& cloud0, pcl::PointCloud<PointType>::Ptr& cloud1, pcl::PointCloud<PointType>::Ptr& cloud2);
    void removeGround(std::string bin_path, pcl::PointCloud<PointType>::Ptr& cloud_unground, pcl::PointCloud<PointType>::Ptr& cloud_ground);
    void allocateMemory();
    void get_scan_pose();
    void parseValidScanInfo();
    void readValidScans();

    void mergeScansWithinGlobalCoord(
            const std::vector<pcl::PointCloud<PointType>::Ptr>& _scans,
            const std::vector<Eigen::Matrix4d>& _scans_poses,
            pcl::PointCloud<PointType>::Ptr& _ptcloud_to_save );
    void octreeDownsampling(const pcl::PointCloud<PointType>::Ptr& _src, pcl::PointCloud<PointType>::Ptr& _to_save);
    void voxelDownsampling(const pcl::PointCloud<PointType>::Ptr& _src, pcl::PointCloud<PointType>::Ptr& _to_save);
    void makeGlobalMap();
    void makeInitiGlobalMap();


    void run(void);

    void removeOnce(float _res);
    void revertOnce(float _res);

    void saveCurrentStaticMapHistory(void); // the 0th element is a noisy (original input) (actually not static) map.
    void saveCurrentDynamicMapHistory(void);

    void takeGlobalMapSubsetWithinBall( int _center_scan_idx );
    void transformGlobalMapSubsetToLocal(int _base_scan_idx);

    void transformGlobalMapToLocal(int _base_scan_idx);
    void transformGlobalMapToLocal(int _base_scan_idx, pcl::PointCloud<PointType>::Ptr& _map_local);
    void transformGlobalMapToLocal(const pcl::PointCloud<PointType>::Ptr& _map_global, int _base_scan_idx, pcl::PointCloud<PointType>::Ptr& _map_local);

    cv::Mat scan2RangeImg(const pcl::PointCloud<PointType>::Ptr& _scan,
                          const std::pair<float, float> _fov, /* e.g., [vfov = 50 (upper 25, lower 25), hfov = 360] */
                          const std::pair<int, int> _rimg_size);
    std::pair<cv::Mat, cv::Mat> map2RangeImg(const pcl::PointCloud<PointType>::Ptr& _scan,
                                             const std::pair<float, float> _fov, /* e.g., [vfov = 50 (upper 25, lower 25), hfov = 360] */
                                             const std::pair<int, int> _rimg_size);

    std::vector<int> calcDescrepancyAndParseDynamicPointIdx (const cv::Mat& _scan_rimg, const cv::Mat& _diff_rimg, const cv::Mat& _map_rimg_ptidx);
    std::vector<int> calcDescrepancyAndParseDynamicPointIdxForEachScan( std::pair<int, int> _rimg_shape );

    std::vector<int> getStaticIdxFromDynamicIdx(const std::vector<int>& _dynamic_point_indexes, int _num_all_points);
    std::vector<int> getGlobalMapStaticIdxFromDynamicIdx(const std::vector<int>& _dynamic_point_indexes);

    void parsePointcloudSubsetUsingPtIdx( const pcl::PointCloud<PointType>::Ptr& _ptcloud_orig,
                                          std::vector<int>& _point_indexes, pcl::PointCloud<PointType>::Ptr& _ptcloud_to_save );
    void parseMapPointcloudSubsetUsingPtIdx( std::vector<int>& _point_indexes, pcl::PointCloud<PointType>::Ptr& _ptcloud_to_save );
    void parseStaticMapPointcloudUsingPtIdx( std::vector<int>& _point_indexes );
    void parseDynamicMapPointcloudUsingPtIdx( std::vector<int>& _point_indexes );

    void saveCurrentStaticAndDynamicPointCloudGlobal( void );
    void saveCurrentStaticAndDynamicPointCloudLocal( int _base_pose_idx  = 0);

    // void local2global(const pcl::PointCloud<PointType>::Ptr& _ptcloud_global, pcl::PointCloud<PointType>::Ptr& _ptcloud_local_to_save );
    pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr& _scan_local, int _scan_idx);
    pcl::PointCloud<PointType>::Ptr global2local(const pcl::PointCloud<PointType>::Ptr& _scan_global, int _scan_idx);

    // scan-side removal
    std::pair<pcl::PointCloud<PointType>::Ptr, pcl::PointCloud<PointType>::Ptr> removeDynamicPointsOfScanByKnn ( int _scan_idx );
    void removeDynamicPointsAndSaveStaticScanForEachScan( void );

    void scansideRemovalForEachScan(void);
    void saveCleanedScans(void);
    void saveMapPointcloudByMergingCleanedScans(void);
    void saveInitilizeByMergingCleanedScans(void);
    void scansideRemovalForEachScanAndSaveThem(void);

    void saveStaticScan( int _scan_idx, const pcl::PointCloud<PointType>::Ptr& _ptcloud );
    void saveDynamicScan( int _scan_idx, const pcl::PointCloud<PointType>::Ptr& _ptcloud );

};


#endif //CPP_TEST_REMOVE_H
