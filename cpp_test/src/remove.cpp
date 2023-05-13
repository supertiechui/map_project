//
// Created by heht on 23-4-20.
//

#include "../include/remove.h"


inline float rad2deg(float radians)
{
    return radians * 180.0 / M_PI;
}

inline float deg2rad(float degrees)
{
    return degrees * M_PI / 180.0;
}


void fsmkdir(std::string _path)
{
    if (!fs::is_directory(_path) || !fs::exists(_path))
        fs::create_directories(_path); // create src folder
} //fsmkdir

Removerter::Removerter()
{

    // voxelgrid generates warnings frequently, so verbose off + ps. recommend to use octree (see makeGlobalMap)
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    // pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

    sequence_scan_dir_ = "/home/heht/kitti/new/01";
    if(isScanFileKITTIFormat_)
        sequence_scan_dir_ = "/home/heht/kitti/sequences/01/velodyne";
    sequence_pose_path_ = "/home/heht/kitti/sequences/01/poses.txt";
    save_pcd_directory_ = "/home/heht/PCD_/";

    allocateMemory();

} // ctor


void Removerter::allocateMemory()
{
    map_scan.reset(new pcl::PointCloud<PointType>());
    kdtree_map_scan.reset(new pcl::KdTreeFLANN<PointType>());

    map_global_orig_.reset(new pcl::PointCloud<PointType>());

    map_global_curr_.reset(new pcl::PointCloud<PointType>());
    map_local_curr_.reset(new pcl::PointCloud<PointType>());

    map_global_curr_static_.reset(new pcl::PointCloud<PointType>());
    map_global_curr_dynamic_.reset(new pcl::PointCloud<PointType>());

    map_subset_global_curr_.reset(new pcl::PointCloud<PointType>());

    kdtree_map_global_curr_.reset(new pcl::KdTreeFLANN<PointType>());
    kdtree_scan_global_curr_.reset(new pcl::KdTreeFLANN<PointType>());

    //
    map_global_orig_->clear();
    map_global_curr_->clear();
    map_local_curr_->clear();
    map_global_curr_static_->clear();
    map_global_curr_dynamic_->clear();
    map_subset_global_curr_->clear();

} // allocateMemory


Removerter::~Removerter(){}

void Removerter::cleanIntensity(pcl::PointCloud<PointType>::Ptr& points)
{
    for(int i = 0; i < points->size(); i++)
    {
        points->at(i).intensity = 100;
    }
}
float Removerter::globalPointDistance(PointType pi, int idx) {
    auto pose = scan_inverse_poses_.at(idx);
    Eigen::Vector4d p1(pi.x, pi.y, pi.z, 1.0);
    Eigen::Vector4d p2 = kSE3MatExtrinsicPoseBasetoLiDAR * pose * p1;
    if(p2[0] < 0)
        return 0.2;
    return sqrt(p2[0]*p2[0] + p2[1]*p2[1] + p2[2]*p2[2]);
//    return sqrt((p.x-p2.x)*(p.x-p2.x) + (p.y-p2.y)*(p.y-p2.y) + (p.z-p2.z)*(p.z-p2.z));
}
float Removerter::kdtreeDistance_global(PointType p, int idx)
{
    float distance = globalPointDistance(p, idx);
    if(distance < 30){
        return 0.2;
    }else if(distance < 50){
        return 0.3;
    }else if(distance < 70){
        return 0.8;
    }else{
        return 1.0;
    }
}
//实现：cloud2 = cloud0 - cloud1
//param[in]  pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud1://减数点云
//param[in] pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud0：//被减数点云
//param[out] pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud2：//差点云
void Removerter::CloudSubtraction(pcl::PointCloud<PointType>::Ptr& cloud0, pcl::PointCloud<PointType>::Ptr& cloud1, pcl::PointCloud<PointType>::Ptr& cloud2) {
    float resolution = 0.10f; //八叉树分辨率，根据点云需要自行调整
    pcl::octree::OctreePointCloudChangeDetector<PointType> octree(resolution);

    //添加cloud1到八叉树中
    octree.setInputCloud(cloud1->makeShared());
    octree.addPointsFromInputCloud();

    octree.switchBuffers();

    //添加cloud0到八叉树中
    octree.setInputCloud(cloud0->makeShared());
    octree.addPointsFromInputCloud();

    std::vector<int>newPointIdxVector;   //存储新加入点索引的向量
    octree.getPointIndicesFromNewVoxels(newPointIdxVector);

    cloud2->width = newPointIdxVector.size();
    cloud2->height = 1;
    cloud2->is_dense = false;
    cloud2->points.resize(cloud2->width * cloud2->height);

    for (size_t i = 0; i < newPointIdxVector.size(); i++) {
        cloud2->points[i].x = cloud0->points[newPointIdxVector[i]].x;
        cloud2->points[i].y = cloud0->points[newPointIdxVector[i]].y;
        cloud2->points[i].z = cloud0->points[newPointIdxVector[i]].z;
        cloud2->points[i].intensity = cloud0->points[newPointIdxVector[i]].intensity;
    }
}
void Removerter::voxelDownsampling(const pcl::PointCloud<PointType>::Ptr& _src, pcl::PointCloud<PointType>::Ptr& _to_save){
    // pcdown
    pcl::VoxelGrid<PointType> downsize_filter;
    downsize_filter.setLeafSize(kDownsampleVoxelSize, kDownsampleVoxelSize, kDownsampleVoxelSize);
    downsize_filter.setInputCloud(_src);
    downsize_filter.filter(*_to_save);
}
void Removerter::removeGround(std::string bin_path, pcl::PointCloud<PointType>::Ptr& cloud_unground, pcl::PointCloud<PointType>::Ptr& cloud_ground){
//    std::string bin_path = sequence_valid_scan_paths_[idx];
    Eigen::MatrixXf cloud;
    if( isScanFileKITTIFormat_ ){
        FILE *file = fopen(bin_path.c_str(), "rb");
        if (!file) {
            std::cerr << "error: failed to load " << bin_path << std::endl;
            return;
        }

        std::vector<float> buffer(1000000);
        size_t num_points = fread(reinterpret_cast<char *>(buffer.data()), sizeof(float), buffer.size(), file) / 4;

        cloud.resize(num_points, 4);
        for (int i=0; i<num_points; i++)
        {
            cloud.row(i) << buffer[i*4], buffer[i*4+1], buffer[i*4+2], buffer[i*4+3];
        }
    }else{
        pcl::PointCloud<PointType>::Ptr points (new pcl::PointCloud<PointType>);
        pcl::io::loadPCDFile<PointType> (bin_path, *points);
        cloud.resize(points->points.size(), 4);
        for(int i = 0; i < points->points.size(); i++){
            cloud.row(i) << points->points[i].x, points->points[i].y, points->points[i].z, points->points[i].intensity;
        }
    }

    patchwork::Params patchwork_parameters;
    patchwork_parameters.verbose = false;

    patchwork::PatchWorkpp Patchworkpp(patchwork_parameters);
    // Estimate Ground

    Patchworkpp.estimateGround(cloud);

    // Get Ground and Nonground
    Eigen::MatrixX4f ground     = Patchworkpp.getGround();
    Eigen::MatrixX4f nonground  = Patchworkpp.getNonground();
    double time_taken = Patchworkpp.getTimeTaken();
    for(int i = 0; i < nonground.rows(); ++i) {
        PointType point;
        point.x = nonground.row(i)[0];
        point.y = nonground.row(i)[1];
        point.z = nonground.row(i)[2];
        point.intensity = nonground.row(i)[3];
        cloud_unground->push_back(point);
    }
    for(int i = 0; i < ground.rows(); ++i) {
        PointType point;
        point.x = ground.row(i)[0];
        point.y = ground.row(i)[1];
        point.z = ground.row(i)[2];
        point.intensity = ground.row(i)[3];
        cloud_ground->push_back(point);
    }
//        cout<<"the NO."<<i<<" is "<<ground.row(i)[1]<<endl;
    // Get centers and normals for patches
    Eigen::MatrixX4f centers    = Patchworkpp.getCenters();
    Eigen::MatrixX4f normals    = Patchworkpp.getNormals();
//    cout << "Origianl Points  #: " << cloud.rows() << endl;
//    cout << "Ground Points    #: " << ground.rows() << endl;
//    cout << "Nonground Points #: " << nonground.rows() << endl;
//    cout << "Time Taken : "<< time_taken / 1000000 << "(sec)" << endl;
}
void Removerter::initizlize() {
    // map-side removals
    for(float _rm_res: remove_resolution_list_) {
        removeOnce( _rm_res );
    }

    // if you want to every iteration's map data, place below two lines to inside of the above for loop
    saveCurrentStaticAndDynamicPointCloudGlobal(); // if you want to save within the global points uncomment this line

    // TODO
    // map-side reverts
    // if you want to remove as much as possible, you can use omit this steps
    for(float _rv_res: revert_resolution_list_) {
        revertOnce( _rv_res );
    }

    // scan-side removals
    scansideRemovalForEachScanAndSaveThem();
}
void Removerter::findDynamicPointsOfScanByKnn(int _scan_idx) {
    clock_t beg = clock();
    pcl::PointCloud<PointType>::Ptr scan_orig = scans_.at(_scan_idx);
    auto scan_pose = scan_poses_.at(_scan_idx);

    // curr scan (in global coord)
    pcl::PointCloud<PointType>::Ptr scan_orig_global = local2global(scan_orig, _scan_idx);
//    cleanIntensity(scan_orig_global);
    int num_points_of_a_scan = scan_orig_global->points.size();

    pcl::PointCloud<PointType>::Ptr scan_static_global (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr scan_dynamic_global (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr noisePoint (new pcl::PointCloud<PointType>);
    for (std::size_t pt_idx = 0; pt_idx < num_points_of_a_scan; pt_idx++)
    {
        std::vector<int> topk_indexes_scan;
        std::vector<float> topk_L2dists_scan;
        if(kdtree_map_scan->radiusSearch(scan_orig_global->points[pt_idx], 0.2, topk_indexes_scan, topk_L2dists_scan)){
            scan_static_global->push_back(scan_orig_global->points[pt_idx]);
        }else{
            scan_dynamic_global->push_back(scan_orig_global->points[pt_idx]);
        }
    }

//    voxelDownsampling(noisePoint, scan_static_global);
    clock_t end = clock();
    cout<<"time taken "<<(end - beg)/double(1000000)<<endl;
//    CloudSubtraction(noisePoint, scan_orig_global, scan_dynamic_global);
//    std::string file_name4 = save_pcd_directory_ + "scan_map_static_ex.pcd";
//    pcl::io::savePCDFileBinary(file_name4, *scan_dynamic_global);

    std::string file_name5 = save_pcd_directory_ + "scan_map_static.pcd";
    pcl::io::savePCDFileBinary(file_name5, *scan_static_global);

    std::string file_name6 = save_pcd_directory_ + "scan_map_dynamic.pcd";
    pcl::io::savePCDFileBinary(file_name6, *scan_dynamic_global);

    cout<<"\033[1;32m The scan " << num_points_of_a_scan << "\033[0m"<<endl;
    cout<<"\033[1;32m The number of scan_map is " << map_scan->size() << "\033[0m"<<endl;
    cout<<"\033[1;32m -- The number of static points in a scan: " << scan_static_global->points.size() << "\033[0m"<<endl;
    cout<<"\033[1;32m -- The number of dynamic points in a scan: " << scan_dynamic_global->points.size() << "\033[0m"<<endl;

    usleep( kPauseTimeForClearStaticScanVisualization );
}
void Removerter::slideWindows() {
    map_global_curr_static_->clear();
    map_global_curr_->clear();
    for(int scan_idx = initilize_idx; scan_idx < scans_.size()-5; scan_idx++)
    {
        clock_t beg_idx = clock();
        map_scan->clear();
        scanMap(scan_idx, map_scan);
        if(map_scan->empty())
        {
            cout<<"Map has no points!!!"<<endl;
            return;
        }
        kdtree_map_scan->makeShared();
        kdtree_map_scan->setInputCloud(map_scan);
//        cleanIntensity(map_scan);
        std::string file_name1 = save_pcd_directory_ + "scan_map_"+std::to_string(scan_idx)+ ".pcd";
        pcl::io::savePCDFileBinary(file_name1, *map_scan);

        pcl::PointCloud<PointType>::Ptr scan_orig = scans_.at(scan_idx);
        auto scan_pose = scan_poses_.at(scan_idx);

        // curr scan (in global coord)
        pcl::PointCloud<PointType>::Ptr scan_orig_global = local2global(scan_orig, scan_idx);
        int num_points_of_a_scan = scan_orig_global->points.size();

        pcl::PointCloud<PointType>::Ptr scan_static_global (new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr scan_dynamic_global (new pcl::PointCloud<PointType>);
//        pcl::PointCloud<PointType>::Ptr static_noisePoint (new pcl::PointCloud<PointType>);
        for (std::size_t pt_idx = 0; pt_idx < num_points_of_a_scan; pt_idx++)
        {
            float kd_distance = kdtreeDistance_global(scan_orig_global->points[pt_idx], scan_idx);
            std::vector<int> topk_indexes_scan;
            std::vector<float> topk_L2dists_scan;
            if( kdtree_map_scan->radiusSearch(scan_orig_global->points[pt_idx], 0.2, topk_indexes_scan, topk_L2dists_scan) ){
                scan_static_global->push_back(scan_orig_global->points[pt_idx]);
            }else{
                scan_dynamic_global->push_back(scan_orig_global->points[pt_idx]);
            }
        }
        pcl::PointCloud<PointType>::Ptr scan_static_local = global2local(scan_static_global, scan_idx);
        pcl::PointCloud<PointType>::Ptr scan_dynamic_local = global2local(scan_dynamic_global, scan_idx);
        scans_static_.emplace_back(scan_static_local);
        scans_dynamic_.emplace_back(scan_dynamic_local);
//        CloudSubtraction(static_noisePoint, scan_orig_global, scan_dynamic_global);
//        voxelDownsampling(static_noisePoint, scan_static_global);
//        *map_global_curr_static_ += *scan_static_global;
//        *map_global_curr_static_ += *scan_ground_global;
        clock_t end_idx = clock();

        cout<<"\033[1;33m The scan "<< scan_idx<<" time is " <<(end_idx - beg_idx)/double(1000000) <<"\033[0m"<<endl;
        cout<<"\033[1;32m The number of scan is " << num_points_of_a_scan<< "\033[0m"<<endl;
        cout<<"\033[1;32m -- The number of static points in a scan: " << scan_static_global->points.size() << "\033[0m"<<endl;
        cout<<"\033[1;32m -- The number of dynamic points in a scan: " << scan_dynamic_global->points.size() << "\033[0m"<<endl;

//        usleep( kPauseTimeForClearStaticScanVisualization );
    }
//    clock_t beg_global = clock();
//    octreeDownsampling(map_global_curr_static_, map_global_curr_);
//    clock_t end_global = clock();
//    cout<<"\033[1;31m The global octree downsampling time is " <<(end_global - beg_global)/double(1000000) <<"\033[0m"<<endl;
//    std::string file_name1 = save_pcd_directory_ + "global_static_map.pcd";
//    pcl::io::savePCDFileBinary(file_name1, *map_global_curr_);
}
void Removerter::get_scan_pose()
{
    // sequence bin files


    for(auto& _entry : fs::directory_iterator(sequence_scan_dir_)) {
        sequence_scan_names_.emplace_back(_entry.path().filename());
        sequence_scan_paths_.emplace_back(_entry.path());
    }
    std::sort(sequence_scan_names_.begin(), sequence_scan_names_.end());
    std::sort(sequence_scan_paths_.begin(), sequence_scan_paths_.end());

    cout<<"\033[1;32m Total : " << sequence_scan_paths_.size() << " scans in the directory.\033[0m"<<endl;

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
//    cout<<sequence_scan_paths_.size()<<" "<<sequence_scan_poses_.size()<<endl;
    // check the number of scans and the number of poses are equivalent
    assert(sequence_scan_paths_.size() == sequence_scan_poses_.size());
}

void Removerter::parseValidScanInfo( void )
{
    int num_valid_parsed {0};
    float movement_counter {0.0};
//    std::cout<<start_idx_<<" bbb scan path is "<<end_idx_<<std::endl;
    for(int curr_idx=0; curr_idx < int(sequence_scan_paths_.size()); curr_idx++)
    {

        // check the scan idx within the target idx range
        if(curr_idx > end_idx_ || curr_idx < start_idx_) {
            curr_idx++;
            continue;
        }
        // check enough movement occured (e.g., parse every 2m)
        if(use_keyframe_gap_) {
//            ROS_INFO("key gap is  %d", keyframe_gap_);
            if( remainder(num_valid_parsed, keyframe_gap_) != 0 ) {
                num_valid_parsed++;
                continue;
            }
        }
        if(use_keyframe_meter_) {
            if( 0 /*TODO*/ ) {
                // TODO using movement_counter
            }
        }
//        std::cout<<start_idx_<<" aaa scan pose is "<<sequence_scan_poses_.size()<<std::endl;
        // save the info (reading scan bin is in makeGlobalMap)
        sequence_valid_scan_paths_.emplace_back(sequence_scan_paths_.at(curr_idx));
        sequence_valid_scan_names_.emplace_back(sequence_scan_names_.at(curr_idx));
//        std::cout<<" ccc scan path is "<<sequence_scan_names_[0]<<std::endl;
        scan_poses_.emplace_back(sequence_scan_poses_.at(curr_idx)); // used for local2global
        scan_inverse_poses_.emplace_back(sequence_scan_inverse_poses_.at(curr_idx)); // used for global2local

        //
        num_valid_parsed++;
    }

    if(use_keyframe_gap_) {
        cout<<"\033[1;32m Total " << sequence_valid_scan_paths_.size()
                                            << " nodes are used from the index range [" << start_idx_ << ", " << end_idx_ << "]"
                                            << " (every " << keyframe_gap_ << " frames parsed)\033[0m"<<endl;
    }
} // parseValidScanInfo


void Removerter::readValidScans( void )
// for target range of scan idx
{
    const int cout_interval {10};
    int cout_counter {0};
    std::cout<<"read scans :"<<sequence_valid_scan_paths_.size()<<std::endl;
    for(auto& _scan_path : sequence_valid_scan_paths_)
    {
        // read bin files and save

        pcl::PointCloud<PointType>::Ptr points (new pcl::PointCloud<PointType>); // pcl::PointCloud Ptr is a shared ptr so this points will be automatically destroyed after this function block (because no others ref it).
        pcl::PointCloud<PointType>::Ptr unground (new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr ground (new pcl::PointCloud<PointType>);
        if( isScanFileKITTIFormat_ ) {
            readBin(_scan_path, points); // For KITTI (.bin)
//            removeGround(_scan_path, unground, ground);
        } else {
            pcl::io::loadPCDFile<PointType> (_scan_path, *points); // saved from SC-LIO-SAM's pcd binary (.pcd)
//            removeGround(_scan_path, unground, ground);
        }

        pcl::PointCloud<PointType>::Ptr downsampled_points (new pcl::PointCloud<PointType>);
        voxelDownsampling(points, downsampled_points);

        // save downsampled pointcloud
        scans_.emplace_back(downsampled_points);
//        scans_unground.emplace_back(unground);
//        scans_ground.emplace_back(ground);

        // cout for debug
        cout_counter++;
        if (remainder(cout_counter, cout_interval) == 0) {
            cout << _scan_path << endl;
            cout << "Read a pointcloud with " << points->points.size() << " points." << endl;
            cout << "downsample the pointcloud: " << downsampled_points->points.size() << " points." << endl;
            cout << " ... (display every " << cout_interval << " readings) ..." << endl;
        }
    }
    cout << endl;
} // readValidScans


std::pair<cv::Mat, cv::Mat> Removerter::map2RangeImg(const pcl::PointCloud<PointType>::Ptr& _scan,
                                                     const std::pair<float, float> _fov, /* e.g., [vfov = 50 (upper 25, lower 25), hfov = 360] */
                                                     const std::pair<int, int> _rimg_size)
{
    const float kVFOV = _fov.first;
    const float kHFOV = _fov.second;

    const int kNumRimgRow = _rimg_size.first;
    const int kNumRimgCol = _rimg_size.second;

    // @ range image initizliation
    cv::Mat rimg = cv::Mat(kNumRimgRow, kNumRimgCol, CV_32FC1, cv::Scalar::all(kFlagNoPOINT)); // float matrix, save range value
    cv::Mat rimg_ptidx = cv::Mat(kNumRimgRow, kNumRimgCol, CV_32SC1, cv::Scalar::all(0)); // int matrix, save point (of global map) index

    // @ points to range img
    int num_points = _scan->points.size();
#pragma omp parallel for num_threads(kNumOmpCores)
    for (int pt_idx = 0; pt_idx < num_points; ++pt_idx)
    {
        PointType this_point = _scan->points[pt_idx];
        SphericalPoint sph_point = cart2sph(this_point);

        // @ note about vfov: e.g., (+ V_FOV/2) to adjust [-15, 15] to [0, 30]
        // @ min and max is just for the easier (naive) boundary checks.
        int lower_bound_row_idx {0};
        int lower_bound_col_idx {0};
        int upper_bound_row_idx {kNumRimgRow - 1};
        int upper_bound_col_idx {kNumRimgCol - 1};
        int pixel_idx_row = int(std::min(std::max(std::round(kNumRimgRow * (1 - (rad2deg(sph_point.el) + (kVFOV/float(2.0))) / (kVFOV - float(0.0)))), float(lower_bound_row_idx)), float(upper_bound_row_idx)));
        int pixel_idx_col = int(std::min(std::max(std::round(kNumRimgCol * ((rad2deg(sph_point.az) + (kHFOV/float(2.0))) / (kHFOV - float(0.0)))), float(lower_bound_col_idx)), float(upper_bound_col_idx)));

        float curr_range = sph_point.r;

        // @ Theoretically, this if-block would have race condition (i.e., this is a critical section),
        // @ But, the resulting range image is acceptable (watching via Rviz),
        // @      so I just naively applied omp pragma for this whole for-block (2020.10.28)
        // @ Reason: because this for loop is splited by the omp, points in a single splited for range do not race among them,
        // @         also, a point A and B lied in different for-segments do not tend to correspond to the same pixel,
        // #               so we can assume practically there are few race conditions.
        // @ P.S. some explicit mutexing directive makes the code even slower ref: https://stackoverflow.com/questions/2396430/how-to-use-lock-in-openmp
        if ( curr_range < rimg.at<float>(pixel_idx_row, pixel_idx_col) ) {
            rimg.at<float>(pixel_idx_row, pixel_idx_col) = curr_range;
            rimg_ptidx.at<int>(pixel_idx_row, pixel_idx_col) = pt_idx;
        }
    }

    return std::pair<cv::Mat, cv::Mat>(rimg, rimg_ptidx);
} // map2RangeImg


cv::Mat Removerter::scan2RangeImg(const pcl::PointCloud<PointType>::Ptr& _scan,
                                  const std::pair<float, float> _fov, /* e.g., [vfov = 50 (upper 25, lower 25), hfov = 360] */
                                  const std::pair<int, int> _rimg_size)
{
    const float kVFOV = _fov.first;
    const float kHFOV = _fov.second;

    const int kNumRimgRow = _rimg_size.first;
    const int kNumRimgCol = _rimg_size.second;
    // cout << "rimg size is: [" << _rimg_size.first << ", " << _rimg_size.second << "]." << endl;

    // @ range image initizliation
    cv::Mat rimg = cv::Mat(kNumRimgRow, kNumRimgCol, CV_32FC1, cv::Scalar::all(kFlagNoPOINT)); // float matrix

    // @ points to range img
    int num_points = _scan->points.size();
#pragma omp parallel for num_threads(kNumOmpCores)
    for (int pt_idx = 0; pt_idx < num_points; ++pt_idx)
    {
        PointType this_point = _scan->points[pt_idx];
        SphericalPoint sph_point = cart2sph(this_point);

        // @ note about vfov: e.g., (+ V_FOV/2) to adjust [-15, 15] to [0, 30]
        // @ min and max is just for the easier (naive) boundary checks.
        int lower_bound_row_idx {0};
        int lower_bound_col_idx {0};
        int upper_bound_row_idx {kNumRimgRow - 1};
        int upper_bound_col_idx {kNumRimgCol - 1};
        int pixel_idx_row = int(std::min(std::max(std::round(kNumRimgRow * (1 - (rad2deg(sph_point.el) + (kVFOV/float(2.0))) / (kVFOV - float(0.0)))), float(lower_bound_row_idx)), float(upper_bound_row_idx)));
        int pixel_idx_col = int(std::min(std::max(std::round(kNumRimgCol * ((rad2deg(sph_point.az) + (kHFOV/float(2.0))) / (kHFOV - float(0.0)))), float(lower_bound_col_idx)), float(upper_bound_col_idx)));

        float curr_range = sph_point.r;

        // @ Theoretically, this if-block would have race condition (i.e., this is a critical section),
        // @ But, the resulting range image is acceptable (watching via Rviz),
        // @      so I just naively applied omp pragma for this whole for-block (2020.10.28)
        // @ Reason: because this for loop is splited by the omp, points in a single splited for range do not race among them,
        // @         also, a point A and B lied in different for-segments do not tend to correspond to the same pixel,
        // #               so we can assume practically there are few race conditions.
        // @ P.S. some explicit mutexing directive makes the code even slower ref: https://stackoverflow.com/questions/2396430/how-to-use-lock-in-openmp
        if ( curr_range < rimg.at<float>(pixel_idx_row, pixel_idx_col) ) {
            rimg.at<float>(pixel_idx_row, pixel_idx_col) = curr_range;
        }
    }

    return rimg;
} // scan2RangeImg


void Removerter::mergeScansWithinGlobalCoord(
        const std::vector<pcl::PointCloud<PointType>::Ptr>& _scans,
        const std::vector<Eigen::Matrix4d>& _scans_poses,
        pcl::PointCloud<PointType>::Ptr& _ptcloud_to_save )
{
    // NOTE: _scans must be in local coord
    for(std::size_t scan_idx = 0 ; scan_idx < _scans.size(); scan_idx++)
    {
        auto ii_scan = _scans.at(scan_idx); // pcl::PointCloud<PointType>::Ptr
        auto ii_pose = _scans_poses.at(scan_idx); // Eigen::Matrix4d
        // local to global (local2global)
        pcl::PointCloud<PointType>::Ptr scan_global_coord(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*ii_scan, *scan_global_coord, kSE3MatExtrinsicLiDARtoPoseBase);
        pcl::transformPointCloud(*scan_global_coord, *scan_global_coord, ii_pose);

        // merge the scan into the global map
        *_ptcloud_to_save += *scan_global_coord;
    }
} // mergeScansWithinGlobalCoord


void Removerter::octreeDownsampling(const pcl::PointCloud<PointType>::Ptr& _src, pcl::PointCloud<PointType>::Ptr& _to_save)
{
    pcl::octree::OctreePointCloudVoxelCentroid<PointType> octree( kDownsampleVoxelSize );
    octree.setInputCloud(_src);
    octree.defineBoundingBox();
    octree.addPointsFromInputCloud();
    pcl::octree::OctreePointCloudVoxelCentroid<PointType>::AlignedPointTVector centroids;
    octree.getVoxelCentroids(centroids);

    // init current map with the downsampled full cloud
    _to_save->points.assign(centroids.begin(), centroids.end());
    _to_save->width = 1;
    _to_save->height = _to_save->points.size(); // make sure again the format of the downsampled point cloud
//    cout<<"\033[1;32m Downsampled pointcloud have: " << _to_save->points.size() << " points.\033[0m"<<endl;
    cout << endl;
} // octreeDownsampling
void Removerter::scanMap(const std::vector<pcl::PointCloud<PointType>::Ptr> &_scans,
                         const std::vector<Eigen::Matrix4d> &_scans_poses,
                         pcl::PointCloud<PointType>::Ptr &_ptcloud_to_save)
{
    // NOTE: _scans must be in local coord

    for(std::size_t scan_idx = _scans.size()-1 ; scan_idx >= _scans.size()-10; scan_idx--)
    {
//        if(scan_idx > 33 && scan_idx < 37)
//            continue;
        auto ii_scan = _scans.at(scan_idx); // pcl::PointCloud<PointType>::Ptr
        auto ii_pose = _scans_poses.at(scan_idx); // Eigen::Matrix4d
        // local to global (local2global)
        pcl::PointCloud<PointType>::Ptr scan_global_coord(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*ii_scan, *scan_global_coord, kSE3MatExtrinsicLiDARtoPoseBase);
        pcl::transformPointCloud(*scan_global_coord, *scan_global_coord, ii_pose);

        // merge the scan into the global map
        *_ptcloud_to_save += *scan_global_coord;
    }

} // mergeScansWithinGlobalCoord
void Removerter::scanMap(int idx, pcl::PointCloud<PointType>::Ptr& _ptcloud_to_save) {
    if(idx < 5 || idx > scans_.size()-5){
        cout<<"Error idx for submap!"<<endl;
        return;
    }
    cout<<"The number of static scan is "<<scans_static_.size()<<endl;
    for(std::size_t scan_idx = scans_static_.size() - 1 ; scan_idx >= scans_static_.size() - 5; scan_idx--)
    {
        auto ii_scan = scans_static_.at(scan_idx);
        auto ii_pose = scan_poses_.at(scan_idx); // Eigen::Matrix4d
        // local to global (local2global)
        pcl::PointCloud<PointType>::Ptr scan_global_coord(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*ii_scan, *scan_global_coord, kSE3MatExtrinsicLiDARtoPoseBase);
        pcl::transformPointCloud(*scan_global_coord, *scan_global_coord, ii_pose);
        // merge the scan into the global map
        *_ptcloud_to_save += *scan_global_coord;
    }
    for(std::size_t scan_idx = idx + 2 ; scan_idx < idx + 5; scan_idx++)
    {
        auto ii_scan = scans_.at(scan_idx); // pcl::PointCloud<PointType>::Ptr
        auto ii_pose = scan_poses_.at(scan_idx); // Eigen::Matrix4d
        // local to global (local2global)
        pcl::PointCloud<PointType>::Ptr scan_global_coord(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*ii_scan, *scan_global_coord, kSE3MatExtrinsicLiDARtoPoseBase);
        pcl::transformPointCloud(*scan_global_coord, *scan_global_coord, ii_pose);
        // merge the scan into the global map
        *_ptcloud_to_save += *scan_global_coord;
    }
}

void Removerter::makeGlobalMap( void )
{
    // transform local to global and merging the scans
    map_global_orig_->clear();
    map_global_curr_->clear();

    mergeScansWithinGlobalCoord(scans_, scan_poses_, map_global_orig_);
    cout<<"\033[1;32m Map pointcloud (having redundant points) have: " << map_global_orig_->points.size() << " points.\033[0m"<<endl;
    cout<<"\033[1;32m Downsampling leaf size is " << kDownsampleVoxelSize << " m.\033[0m"<<endl;

    // remove repeated (redundant) points
    // - using OctreePointCloudVoxelCentroid for downsampling
    // - For a large-size point cloud should use OctreePointCloudVoxelCentroid rather VoxelGrid
    octreeDownsampling(map_global_orig_, map_global_curr_);

    // save the original cloud
    if( kFlagSaveMapPointcloud ) {
        // in global coord
        std::string static_global_file_name = save_pcd_directory_ + "OriginalNoisyMapGlobal.pcd";
        pcl::io::savePCDFileBinary(static_global_file_name, *map_global_curr_);
        cout<<"\033[1;32m The original pointcloud is saved (global coord): " << static_global_file_name << "\033[0m"<<endl;

        // in local coord (i.e., base_node_idx == 0 means a start idx is the identity pose)
        int base_node_idx = base_node_idx_;
        pcl::PointCloud<PointType>::Ptr map_local_curr (new pcl::PointCloud<PointType>);
        transformGlobalMapToLocal(map_global_curr_, base_node_idx, map_local_curr);
        std::string static_local_file_name = save_pcd_directory_ + "OriginalNoisyMapLocal.pcd";
        pcl::io::savePCDFileBinary(static_local_file_name, *map_local_curr);
        cout<<"\033[1;32m The original pointcloud is saved (local coord): " << static_local_file_name << "\033[0m"<<endl;
    }
    // make tree (for fast ball search for the projection to make a map range image later)
    // if(kUseSubsetMapCloud) // NOT recommend to use for under 5 million points map input
    //     kdtree_map_global_curr_->setInputCloud(map_global_curr_);

    // save current map into history
    // TODO
    // if(save_history_on_memory_)
    //     saveCurrentStaticMapHistory();

} // makeGlobalMap


void Removerter::transformGlobalMapSubsetToLocal(int _base_scan_idx)
{
    Eigen::Matrix4d base_pose_inverse = scan_inverse_poses_.at(_base_scan_idx);

    // global to local (global2local)
    map_local_curr_->clear();
    pcl::transformPointCloud(*map_subset_global_curr_, *map_local_curr_, base_pose_inverse);
    pcl::transformPointCloud(*map_local_curr_, *map_local_curr_, kSE3MatExtrinsicPoseBasetoLiDAR);

} // transformGlobalMapSubsetToLocal


void Removerter::transformGlobalMapToLocal(int _base_scan_idx)
{
    Eigen::Matrix4d base_pose_inverse = scan_inverse_poses_.at(_base_scan_idx);

    // global to local (global2local)
    map_local_curr_->clear();
    pcl::transformPointCloud(*map_global_curr_, *map_local_curr_, base_pose_inverse);
    pcl::transformPointCloud(*map_local_curr_, *map_local_curr_, kSE3MatExtrinsicPoseBasetoLiDAR);

} // transformGlobalMapToLocal


void Removerter::transformGlobalMapToLocal(int _base_scan_idx, pcl::PointCloud<PointType>::Ptr& _map_local)
{
    Eigen::Matrix4d base_pose_inverse = scan_inverse_poses_.at(_base_scan_idx);

    // global to local (global2local)
    _map_local->clear();
    pcl::transformPointCloud(*map_global_curr_, *_map_local, base_pose_inverse);
    pcl::transformPointCloud(*_map_local, *_map_local, kSE3MatExtrinsicPoseBasetoLiDAR);

} // transformGlobalMapToLocal


void Removerter::transformGlobalMapToLocal(
        const pcl::PointCloud<PointType>::Ptr& _map_global,
        int _base_scan_idx, pcl::PointCloud<PointType>::Ptr& _map_local)
{
    Eigen::Matrix4d base_pose_inverse = scan_inverse_poses_.at(_base_scan_idx);

    // global to local (global2local)
    _map_local->clear();
    pcl::transformPointCloud(*_map_global, *_map_local, base_pose_inverse);
    pcl::transformPointCloud(*_map_local, *_map_local, kSE3MatExtrinsicPoseBasetoLiDAR);

} // transformGlobalMapToLocal


void Removerter::parseMapPointcloudSubsetUsingPtIdx( std::vector<int>& _point_indexes, pcl::PointCloud<PointType>::Ptr& _ptcloud_to_save )
{
    // extractor
    pcl::ExtractIndices<PointType> extractor;
    boost::shared_ptr<std::vector<int>> index_ptr = boost::make_shared<std::vector<int>>(_point_indexes);
    extractor.setInputCloud(map_global_curr_);
    extractor.setIndices(index_ptr);
    extractor.setNegative(false); // If set to true, you can extract point clouds outside the specified index

    // parse
    _ptcloud_to_save->clear();
    extractor.filter(*_ptcloud_to_save);
} // parseMapPointcloudSubsetUsingPtIdx


void Removerter::parseStaticMapPointcloudUsingPtIdx( std::vector<int>& _point_indexes )
{
    // extractor
    pcl::ExtractIndices<PointType> extractor;
    boost::shared_ptr<std::vector<int>> index_ptr = boost::make_shared<std::vector<int>>(_point_indexes);
    extractor.setInputCloud(map_global_curr_);
    extractor.setIndices(index_ptr);
    extractor.setNegative(false); // If set to true, you can extract point clouds outside the specified index

    // parse
    map_global_curr_static_->clear();
    extractor.filter(*map_global_curr_static_);
} // parseStaticMapPointcloudUsingPtIdx


void Removerter::parseDynamicMapPointcloudUsingPtIdx( std::vector<int>& _point_indexes )
{
    // extractor
    pcl::ExtractIndices<PointType> extractor;
    boost::shared_ptr<std::vector<int>> index_ptr = boost::make_shared<std::vector<int>>(_point_indexes);
    extractor.setInputCloud(map_global_curr_);
    extractor.setIndices(index_ptr);
    extractor.setNegative(false); // If set to true, you can extract point clouds outside the specified index

    // parse
    map_global_curr_dynamic_->clear();
    extractor.filter(*map_global_curr_dynamic_);
} // parseDynamicMapPointcloudUsingPtIdx


void Removerter::saveCurrentStaticAndDynamicPointCloudGlobal( void )
{
    if( ! kFlagSaveMapPointcloud )
        return;

    std::string curr_res_alpha_str = std::to_string(curr_res_alpha_);

    // dynamic
    std::string dyna_file_name = map_dynamic_save_dir_ + "/DynamicMapInitizlizeGlobalResX" + curr_res_alpha_str + ".pcd";
    pcl::io::savePCDFileBinary(dyna_file_name, *map_global_curr_dynamic_);
    cout<<"\033[1;32m -- a pointcloud is saved: " << dyna_file_name << "\033[0m"<<endl;

    // static
    std::string static_file_name = map_static_save_dir_ + "/StaticMapInitizlizeGlobalResX" + curr_res_alpha_str + ".pcd";
    pcl::io::savePCDFileBinary(static_file_name, *map_global_curr_static_);
    cout<<"\033[1;32m -- a pointcloud is saved: " << static_file_name << "\033[0m"<<endl;
} // saveCurrentStaticAndDynamicPointCloudGlobal


void Removerter::saveCurrentStaticAndDynamicPointCloudLocal( int _base_node_idx )
{
    if( ! kFlagSaveMapPointcloud )
        return;

    std::string curr_res_alpha_str = std::to_string(curr_res_alpha_);

    // dynamic
    pcl::PointCloud<PointType>::Ptr map_local_curr_dynamic (new pcl::PointCloud<PointType>);
    transformGlobalMapToLocal(map_global_curr_dynamic_, _base_node_idx, map_local_curr_dynamic);
    std::string dyna_file_name = map_dynamic_save_dir_ + "/DynamicMapMapsideLocalResX" + curr_res_alpha_str + ".pcd";
    pcl::io::savePCDFileBinary(dyna_file_name, *map_local_curr_dynamic);
    cout<<"\033[1;32m -- a pointcloud is saved: " << dyna_file_name << "\033[0m"<<endl;

    // static
    pcl::PointCloud<PointType>::Ptr map_local_curr_static (new pcl::PointCloud<PointType>);
    transformGlobalMapToLocal(map_global_curr_static_, _base_node_idx, map_local_curr_static);
    std::string static_file_name = map_static_save_dir_ + "/StaticMapMapsideLocalResX" + curr_res_alpha_str + ".pcd";
    pcl::io::savePCDFileBinary(static_file_name, *map_local_curr_static);
    cout<<"\033[1;32m -- a pointcloud is saved: " << static_file_name << "\033[0m"<<endl;

} // saveCurrentStaticAndDynamicPointCloudLocal


std::vector<int> Removerter::calcDescrepancyAndParseDynamicPointIdx
        (const cv::Mat& _scan_rimg, const cv::Mat& _diff_rimg, const cv::Mat& _map_rimg_ptidx)
{
    int num_dyna_points {0}; // TODO: tracking the number of dynamic-assigned points and decide when to stop removing (currently just fixed iteration e.g., [2.5, 2.0, 1.5])

    std::vector<int> dynamic_point_indexes;
    for (int row_idx = 0; row_idx < _diff_rimg.rows; row_idx++) {
        for (int col_idx = 0; col_idx < _diff_rimg.cols; col_idx++) {
            float this_diff = _diff_rimg.at<float>(row_idx, col_idx);
            float this_range = _scan_rimg.at<float>(row_idx, col_idx);

            float adaptive_coeff = 0.05; // meter, // i.e., if 4m apart point, it should be 0.4m be diff (nearer) wrt the query
            float adaptive_dynamic_descrepancy_threshold = adaptive_coeff * this_range; // adaptive descrepancy threshold
            // float adaptive_dynamic_descrepancy_threshold = 0.1;

            if( this_diff < kValidDiffUpperBound // exclude no-point pixels either on scan img or map img (100 is roughly 100 meter)
                && this_diff > adaptive_dynamic_descrepancy_threshold /* dynamic */)
            {  // dynamic
                int this_point_idx_in_global_map = _map_rimg_ptidx.at<int>(row_idx, col_idx);
                dynamic_point_indexes.emplace_back(this_point_idx_in_global_map);

                // num_dyna_points++; // TODO
            }
        }
    }

    return dynamic_point_indexes;
} // calcDescrepancyAndParseDynamicPointIdx


void Removerter::takeGlobalMapSubsetWithinBall( int _center_scan_idx )
{
    Eigen::Matrix4d center_pose_se3 = scan_poses_.at(_center_scan_idx);
    PointType center_pose;
    center_pose.x = float(center_pose_se3(0, 3));
    center_pose.y = float(center_pose_se3(1, 3));
    center_pose.z = float(center_pose_se3(2, 3));

    std::vector<int> subset_indexes;
    std::vector<float> pointSearchSqDisGlobalMap;
    kdtree_map_global_curr_->radiusSearch(center_pose, kBallSize, subset_indexes, pointSearchSqDisGlobalMap, 0);
    parseMapPointcloudSubsetUsingPtIdx(subset_indexes, map_subset_global_curr_);
} // takeMapSubsetWithinBall


std::vector<int> Removerter::calcDescrepancyAndParseDynamicPointIdxForEachScan( std::pair<int, int> _rimg_shape )
{
    std::vector<int> dynamic_point_indexes;
    // dynamic_point_indexes.reserve(100000);
    for(std::size_t idx_scan=0; idx_scan < initilize_idx; ++idx_scan) {
        // curr scan
        pcl::PointCloud<PointType>::Ptr _scan = scans_.at(idx_scan);

        // scan's pointcloud to range img
        cv::Mat scan_rimg = scan2RangeImg(_scan, kFOV, _rimg_shape); // openMP inside

        // map's pointcloud to range img
        if( kUseSubsetMapCloud ) {
            takeGlobalMapSubsetWithinBall(idx_scan);
            transformGlobalMapSubsetToLocal(idx_scan); // the most time comsuming part 1
        } else {
            // if the input map size (of a batch) is short, just using this line is more fast.
            // - e.g., 100-1000m or ~5 million points are ok, empirically more than 10Hz
            transformGlobalMapToLocal(idx_scan);
        }
        auto [map_rimg, map_rimg_ptidx] = map2RangeImg(map_local_curr_, kFOV, _rimg_shape); // the most time comsuming part 2 -> so openMP applied inside

        // diff range img
        const int kNumRimgRow = _rimg_shape.first;
        const int kNumRimgCol = _rimg_shape.second;
        cv::Mat diff_rimg = cv::Mat(kNumRimgRow, kNumRimgCol, CV_32FC1, cv::Scalar::all(0.0)); // float matrix, save range value
        cv::absdiff(scan_rimg, map_rimg, diff_rimg);

        // parse dynamic points' indexes: rule: If a pixel value of diff_rimg is larger, scan is the further - means that pixel of submap is likely dynamic.
        std::vector<int> this_scan_dynamic_point_indexes = calcDescrepancyAndParseDynamicPointIdx(scan_rimg, diff_rimg, map_rimg_ptidx);
        dynamic_point_indexes.insert(dynamic_point_indexes.end(), this_scan_dynamic_point_indexes.begin(), this_scan_dynamic_point_indexes.end());
        // save scan_rimg
//        if(curr_res_alpha_ == 2.5)
//        {
//            std::string curr_res_alpha_str = std::to_string(curr_res_alpha_);
//            std::string file_name_orig = sequence_valid_scan_names_.at(idx_scan).substr(0,6) + "X";
//            std::string scan_range_name = scan_range_save_dir + "/" +file_name_orig + curr_res_alpha_str.substr(0,3) + ".jpg";
//            cv::Mat dst;
//            dst = convertColorMappedImg(scan_rimg, kRangeColorAxis);
//            cv::imwrite(scan_range_name, dst);
//
//            std::string map_range_name = map_range_save_dir + "/" +file_name_orig + curr_res_alpha_str.substr(0,3) + ".jpg";
//            cv::Mat dst1;
//            dst1 = convertColorMappedImg(map_rimg, kRangeColorAxis);
//            cv::imwrite(map_range_name, dst1);
//
//            std::string diff_range_name = diff_range_save_dir + "/" +file_name_orig + curr_res_alpha_str.substr(0,3) + ".jpg";
//            cv::Mat dst2;
//            dst2 = convertColorMappedImg(diff_rimg, kRangeColorAxis);
//            cv::imwrite(diff_range_name, dst2);
//        }

        // visualization


        std::pair<float, float> kRangeColorAxisForPtIdx {0.0, float(map_global_curr_->points.size())};

    } // for_each scan Done

    // remove repeated indexes
    std::set<int> dynamic_point_indexes_set (dynamic_point_indexes.begin(), dynamic_point_indexes.end());
    std::vector<int> dynamic_point_indexes_unique (dynamic_point_indexes_set.begin(), dynamic_point_indexes_set.end());

    return dynamic_point_indexes_unique;
} // calcDescrepancyForEachScan


std::vector<int> Removerter::getStaticIdxFromDynamicIdx(const std::vector<int>& _dynamic_point_indexes, int _num_all_points)
{
    std::vector<int> pt_idx_all = linspace<int>(0, _num_all_points, _num_all_points);

    std::set<int> pt_idx_all_set (pt_idx_all.begin(), pt_idx_all.end());
    for(auto& _dyna_pt_idx: _dynamic_point_indexes) {
        pt_idx_all_set.erase(_dyna_pt_idx);
    }

    std::vector<int> static_point_indexes (pt_idx_all_set.begin(), pt_idx_all_set.end());
    return static_point_indexes;
} // getStaticIdxFromDynamicIdx


std::vector<int> Removerter::getGlobalMapStaticIdxFromDynamicIdx(const std::vector<int>& _dynamic_point_indexes)
{
    int num_all_points = map_global_curr_->points.size();
    return getStaticIdxFromDynamicIdx(_dynamic_point_indexes, num_all_points);
} // getGlobalMapStaticIdxFromDynamicIdx



void Removerter::saveCurrentStaticMapHistory(void)
{
    // deep copy
    pcl::PointCloud<PointType>::Ptr map_global_curr_static (new pcl::PointCloud<PointType>);
    *map_global_curr_static = *map_global_curr_;

    // save
    static_map_global_history_.emplace_back(map_global_curr_static);
} // saveCurrentStaticMapHistory


void Removerter::removeOnce( float _res_alpha )
{
    // filter spec (i.e., a shape of the range image)
    curr_res_alpha_ = _res_alpha;

    std::pair<int, int> rimg_shape = resetRimgSize(kFOV, _res_alpha);
    float deg_per_pixel = 1.0 / _res_alpha;
    cout<<"\033[1;32m Removing starts with resolution: x" << _res_alpha << " (" << deg_per_pixel << " deg/pixel)\033[0m"<<endl;
    cout<<"\033[1;32m -- The range image size is: [" << rimg_shape.first << ", " << rimg_shape.second << "].\033[0m"<<endl;
    cout<<"\033[1;32m -- The number of map points: " << map_global_curr_->points.size() << "\033[0m"<<endl;
    cout<<"\033[1;32m -- ... starts cleaning ... " << "\033[0m"<<endl;

    // map-side removal: remove and get dynamic (will be removed) points' index set
    std::vector<int> dynamic_point_indexes = calcDescrepancyAndParseDynamicPointIdxForEachScan( rimg_shape );
    cout<<"\033[1;32m -- The number of dynamic points: " << dynamic_point_indexes.size() << "\033[0m"<<endl;
    parseDynamicMapPointcloudUsingPtIdx(dynamic_point_indexes);

    // static_point_indexes == complemently indexing dynamic_point_indexes
    std::vector<int> static_point_indexes = getGlobalMapStaticIdxFromDynamicIdx(dynamic_point_indexes);
    cout<<"\033[1;32m -- The number of static points: " << static_point_indexes.size() << "\033[0m"<<endl;
    parseStaticMapPointcloudUsingPtIdx(static_point_indexes);

    // Update the current map and reset the tree
    map_global_curr_->clear();
    *map_global_curr_ = *map_global_curr_static_;

    // if(kUseSubsetMapCloud) // NOT recommend to use for under 5 million points map input
    //     kdtree_map_global_curr_->setInputCloud(map_global_curr_);

} // removeOnce


void Removerter::revertOnce( float _res_alpha )
{
    std::pair<int, int> rimg_shape = resetRimgSize(kFOV, _res_alpha);
    float deg_per_pixel = 1.0 / _res_alpha;
    cout<<"\033[1;32m Reverting starts with resolution: x" << _res_alpha << " (" << deg_per_pixel << " deg/pixel)\033[0m"<<endl;
    cout<<"\033[1;32m -- The range image size is: [" << rimg_shape.first << ", " << rimg_shape.second << "].\033[0m"<<endl;
    cout<<"\033[1;32m -- ... TODO ... \033[0m"<<endl;

    // TODO

} // revertOnce


void Removerter::parsePointcloudSubsetUsingPtIdx( const pcl::PointCloud<PointType>::Ptr& _ptcloud_orig,
                                                  std::vector<int>& _point_indexes, pcl::PointCloud<PointType>::Ptr& _ptcloud_to_save )
{
    // extractor
    pcl::ExtractIndices<PointType> extractor;
    boost::shared_ptr<std::vector<int>> index_ptr = boost::make_shared<std::vector<int>>(_point_indexes);
    extractor.setInputCloud(_ptcloud_orig);
    extractor.setIndices(index_ptr);
    extractor.setNegative(false); // If set to true, you can extract point clouds outside the specified index

    // parse
    _ptcloud_to_save->clear();
    extractor.filter(*_ptcloud_to_save);
} // parsePointcloudSubsetUsingPtIdx


pcl::PointCloud<PointType>::Ptr Removerter::local2global(const pcl::PointCloud<PointType>::Ptr& _scan_local, int _scan_idx)
{
    Eigen::Matrix4d scan_pose = scan_poses_.at(_scan_idx);

    pcl::PointCloud<PointType>::Ptr scan_global(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*_scan_local, *scan_global, kSE3MatExtrinsicLiDARtoPoseBase);
    pcl::transformPointCloud(*scan_global, *scan_global, scan_pose);

    return scan_global;
}

pcl::PointCloud<PointType>::Ptr Removerter::global2local(const pcl::PointCloud<PointType>::Ptr& _scan_global, int _scan_idx)
{
    Eigen::Matrix4d base_pose_inverse = scan_inverse_poses_.at(_scan_idx);

    pcl::PointCloud<PointType>::Ptr scan_local(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*_scan_global, *scan_local, base_pose_inverse);
    pcl::transformPointCloud(*scan_local, *scan_local, kSE3MatExtrinsicPoseBasetoLiDAR);

    return scan_local;
}

std::pair<pcl::PointCloud<PointType>::Ptr, pcl::PointCloud<PointType>::Ptr>
Removerter::removeDynamicPointsOfScanByKnn ( int _scan_idx )
{
    // curr scan (in local coord)
    pcl::PointCloud<PointType>::Ptr scan_orig = scans_.at(_scan_idx);
    auto scan_pose = scan_poses_.at(_scan_idx);

    // curr scan (in global coord)
    pcl::PointCloud<PointType>::Ptr scan_orig_global = local2global(scan_orig, _scan_idx);
    kdtree_scan_global_curr_->setInputCloud(scan_orig_global);
    int num_points_of_a_scan = scan_orig_global->points.size();

    //
    pcl::PointCloud<PointType>::Ptr scan_static_global (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr scan_dynamic_global (new pcl::PointCloud<PointType>);
    for (std::size_t pt_idx = 0; pt_idx < num_points_of_a_scan; pt_idx++)
    {
        std::vector<int> topk_indexes_scan;
        std::vector<float> topk_L2dists_scan;
        kdtree_scan_global_curr_->nearestKSearch(scan_orig_global->points[pt_idx], kNumKnnPointsToCompare, topk_indexes_scan, topk_L2dists_scan);
        float sum_topknn_dists_in_scan = accumulate( topk_L2dists_scan.begin(), topk_L2dists_scan.end(), 0.0);
        float avg_topknn_dists_in_scan = sum_topknn_dists_in_scan / float(kNumKnnPointsToCompare);

        std::vector<int> topk_indexes_map;
        std::vector<float> topk_L2dists_map;
        kdtree_map_global_curr_->nearestKSearch(scan_orig_global->points[pt_idx], kNumKnnPointsToCompare, topk_indexes_map, topk_L2dists_map);
        float sum_topknn_dists_in_map = accumulate( topk_L2dists_map.begin(), topk_L2dists_map.end(), 0.0);
        float avg_topknn_dists_in_map = sum_topknn_dists_in_map / float(kNumKnnPointsToCompare);

        //
        if ( std::abs(avg_topknn_dists_in_scan - avg_topknn_dists_in_map) < kScanKnnAndMapKnnAvgDiffThreshold) {
            scan_static_global->push_back(scan_orig_global->points[pt_idx]);
        } else {
            scan_dynamic_global->push_back(scan_orig_global->points[pt_idx]);
        }
    }

    // again global2local because later in the merging global map function, which requires scans within each local coord.
    pcl::PointCloud<PointType>::Ptr scan_static_local = global2local(scan_static_global, _scan_idx);
    pcl::PointCloud<PointType>::Ptr scan_dynamic_local = global2local(scan_dynamic_global, _scan_idx);

    cout<<"\033[1;32m The scan " << sequence_valid_scan_paths_.at(_scan_idx) << "\033[0m"<<endl;
    cout<<"\033[1;32m -- The number of static points in a scan: " << scan_static_local->points.size() << "\033[0m"<<endl;
    cout<<"\033[1;32m -- The number of dynamic points in a scan: " << num_points_of_a_scan - scan_static_local->points.size() << "\033[0m"<<endl;

    usleep( kPauseTimeForClearStaticScanVisualization );

    return std::pair<pcl::PointCloud<PointType>::Ptr, pcl::PointCloud<PointType>::Ptr> (scan_static_local, scan_dynamic_local);

} // removeDynamicPointsOfScanByKnn


void Removerter::saveStaticScan( int _scan_idx, const pcl::PointCloud<PointType>::Ptr& _ptcloud )
{

    std::string file_name_orig = sequence_valid_scan_names_.at(_scan_idx);
    std::string file_name = scan_static_save_dir_ + "/" + file_name_orig.substr(0,6) + ".pcd";
    cout<<"\033[1;32m Scan " << _scan_idx << "'s static points is saved (" << file_name << ")\033[0m"<<endl;
    pcl::PointCloud<PointType>::Ptr scan_global_coord(new pcl::PointCloud<PointType>());
    scan_global_coord = local2global(_ptcloud ,_scan_idx);
    pcl::io::savePCDFileBinary(file_name, *scan_global_coord);
} // saveStaticScan


void Removerter::saveDynamicScan( int _scan_idx, const pcl::PointCloud<PointType>::Ptr& _ptcloud )
{
    std::string file_name_orig = sequence_valid_scan_names_.at(_scan_idx);
    std::string file_name = scan_dynamic_save_dir_ + "/" + file_name_orig.substr(0,6) + ".pcd";
    cout<<"\033[1;32m Scan " << _scan_idx << "'s dynamic points is saved (" << file_name << ")\033[0m"<<endl;
    pcl::PointCloud<PointType>::Ptr scan_global_coord(new pcl::PointCloud<PointType>());
    scan_global_coord = local2global(_ptcloud ,_scan_idx);
    pcl::io::savePCDFileBinary(file_name, *scan_global_coord);
} // saveDynamicScan


void Removerter::saveCleanedScans(void)
{
    if( ! kFlagSaveCleanScans )
        return;

    for(std::size_t idx_scan=0; idx_scan < scans_static_.size(); idx_scan++) {
        saveStaticScan(idx_scan, scans_static_.at(idx_scan));
        saveDynamicScan(idx_scan, scans_dynamic_.at(idx_scan));
    }
} // saveCleanedScans

void Removerter::saveInitilizeByMergingCleanedScans() {
    // static map
    {
        pcl::PointCloud<PointType>::Ptr map_global_static_scans_merged_to_verify_full (new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr map_global_static_scans_merged_to_verify (new pcl::PointCloud<PointType>);
        mergeScansWithinGlobalCoord(scans_static_, scan_poses_, map_global_static_scans_merged_to_verify_full);
        octreeDownsampling(map_global_static_scans_merged_to_verify_full, map_global_static_scans_merged_to_verify);

        // global
        std::string local_file_name = map_static_save_dir_ + "/StaticInitizlizeMapGlobal.pcd";
        pcl::io::savePCDFileBinary(local_file_name, *map_global_static_scans_merged_to_verify);
        cout<<"\033[1;32m  [For verification] A static pointcloud (cleaned scans merged) is saved (global coord): " << local_file_name << "\033[0m"<<endl;
    }

    // dynamic map
    {
        pcl::PointCloud<PointType>::Ptr map_global_dynamic_scans_merged_to_verify_full (new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr map_global_dynamic_scans_merged_to_verify (new pcl::PointCloud<PointType>);
        mergeScansWithinGlobalCoord(scans_dynamic_, scan_poses_, map_global_dynamic_scans_merged_to_verify_full);
        octreeDownsampling(map_global_dynamic_scans_merged_to_verify_full, map_global_dynamic_scans_merged_to_verify);

        // global
        std::string local_file_name = map_dynamic_save_dir_ + "/DynamicInitizlizeMapGlobal.pcd";
        pcl::io::savePCDFileBinary(local_file_name, *map_global_dynamic_scans_merged_to_verify);
        cout<<"\033[1;32m  [For verification] A dynamic pointcloud (cleaned scans merged) is saved (global coord): " << local_file_name << "\033[0m"<<endl;
    }
}
void Removerter::saveMapPointcloudByMergingCleanedScans(void)
{
    // merge for verification
    if( ! kFlagSaveMapPointcloud )
        return;

    // static map
    {
        pcl::PointCloud<PointType>::Ptr map_global_static_scans_merged_to_verify_full (new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr map_global_static_scans_merged_to_verify (new pcl::PointCloud<PointType>);
        mergeScansWithinGlobalCoord(scans_static_, scan_poses_, map_global_static_scans_merged_to_verify_full);
        octreeDownsampling(map_global_static_scans_merged_to_verify_full, map_global_static_scans_merged_to_verify);

        // global
        std::string local_file_name = map_static_save_dir_ + "/StaticMapGlobal.pcd";
        pcl::io::savePCDFileBinary(local_file_name, *map_global_static_scans_merged_to_verify);
        cout<<"\033[1;32m  [For verification] A static pointcloud (cleaned scans merged) is saved (global coord): " << local_file_name << "\033[0m"<<endl;

//        // local
//        pcl::PointCloud<PointType>::Ptr map_local_static_scans_merged_to_verify (new pcl::PointCloud<PointType>);
//        int base_node_idx = base_node_idx_;
//        transformGlobalMapToLocal(map_global_static_scans_merged_to_verify, base_node_idx, map_local_static_scans_merged_to_verify);
//        std::string global_file_name = map_static_save_dir_ + "/StaticMapInitizlizeMapLocal.pcd";
//        pcl::io::savePCDFileBinary(global_file_name, *map_local_static_scans_merged_to_verify);
//        cout<<"\033[1;32m  [For verification] A static pointcloud (cleaned scans merged) is saved (local coord): " << global_file_name << "\033[0m"<<endl;
    }

    // dynamic map
    {
        pcl::PointCloud<PointType>::Ptr map_global_dynamic_scans_merged_to_verify_full (new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr map_global_dynamic_scans_merged_to_verify (new pcl::PointCloud<PointType>);
        mergeScansWithinGlobalCoord(scans_dynamic_, scan_poses_, map_global_dynamic_scans_merged_to_verify_full);
        octreeDownsampling(map_global_dynamic_scans_merged_to_verify_full, map_global_dynamic_scans_merged_to_verify);

        // global
        std::string local_file_name = map_dynamic_save_dir_ + "/DynamicMapGlobal.pcd";
        pcl::io::savePCDFileBinary(local_file_name, *map_global_dynamic_scans_merged_to_verify);
        cout<<"\033[1;32m  [For verification] A dynamic pointcloud (cleaned scans merged) is saved (global coord): " << local_file_name << "\033[0m"<<endl;

//        // local
//        pcl::PointCloud<PointType>::Ptr map_local_dynamic_scans_merged_to_verify (new pcl::PointCloud<PointType>);
//        int base_node_idx = base_node_idx_;
//        transformGlobalMapToLocal(map_global_dynamic_scans_merged_to_verify, base_node_idx, map_local_dynamic_scans_merged_to_verify);
//        std::string global_file_name = map_dynamic_save_dir_ + "/DynamicMapInitizlizeMapLocal.pcd";
//        pcl::io::savePCDFileBinary(global_file_name, *map_local_dynamic_scans_merged_to_verify);
//        cout<<"\033[1;32m  [For verification] A dynamic pointcloud (cleaned scans merged) is saved (local coord): " << global_file_name << "\033[0m"<<endl;
    }
} // saveMapPointcloudByMergingCleanedScans


void Removerter::scansideRemovalForEachScan( void )
{
    // for fast scan-side neighbor search
    kdtree_map_global_curr_->setInputCloud(map_global_curr_);

    // for each scan
    for(std::size_t idx_scan=0; idx_scan < initilize_idx; idx_scan++) {
        auto [this_scan_static, this_scan_dynamic] = removeDynamicPointsOfScanByKnn(idx_scan);
        scans_static_.emplace_back(this_scan_static);
        scans_dynamic_.emplace_back(this_scan_dynamic);
    }
} // scansideRemovalForEachScan


void Removerter::scansideRemovalForEachScanAndSaveThem( void )
{
    scansideRemovalForEachScan();
//    saveCleanedScans();
//    saveInitilizeByMergingCleanedScans();
} // scansideRemovalForEachScanAndSaveThem


void Removerter::run( void )
{
    // load scan and poses
    get_scan_pose();
    parseValidScanInfo();
    readValidScans();

    makeGlobalMap();
    initizlize();
    slideWindows();
//
    saveCleanedScans();
    saveMapPointcloudByMergingCleanedScans();
//    int idx = initilize_idx + 5;
//    pcl::PointCloud<PointType>::Ptr scan_global_coord(new pcl::PointCloud<PointType>());
//    scan_global_coord = local2global(scans_[idx] ,idx);
//    std::string file_name2 = save_pcd_directory_ + "scan.pcd";
//    pcl::io::savePCDFileBinary(file_name2, *scan_global_coord);
//
//    map_scan->clear();
//
//    scanMap(scans_, scan_poses_, map_scan);
////    cleanIntensity(map_scan);
//    std::string file_name1 = save_pcd_directory_ + "scan_map.pcd";
//    pcl::io::savePCDFileBinary(file_name1, *map_scan);
//    kdtree_map_scan->setInputCloud(map_scan);
//
//    findDynamicPointsOfScanByKnn(idx);
//    std::string file_name3 = save_pcd_directory_ + "scan_map_knn.pcd";
//    pcl::io::savePCDFileBinary(file_name3, *map_scan);


    // construct initial map using the scans and the corresponding poses
//    makeGlobalMap();
//
//    // map-side removals
//    for(float _rm_res: remove_resolution_list_) {
//        removeOnce( _rm_res );
//    }
//
//    // if you want to every iteration's map data, place below two lines to inside of the above for loop
//    saveCurrentStaticAndDynamicPointCloudGlobal(); // if you want to save within the global points uncomment this line
//    saveCurrentStaticAndDynamicPointCloudLocal(base_node_idx_); // w.r.t specific node's coord. 0 means w.r.t the start node, as an Identity.
//
//    // TODO
//    // map-side reverts
//    // if you want to remove as much as possible, you can use omit this steps
//    for(float _rv_res: revert_resolution_list_) {
//        revertOnce( _rv_res );
//    }
//
//    // scan-side removals
//    scansideRemovalForEachScanAndSaveThem();

//    remove ground test
//    pcl::PointCloud<PointType>::Ptr ground (new pcl::PointCloud<PointType>);
//    pcl::PointCloud<PointType>::Ptr unground (new pcl::PointCloud<PointType>);
//    removeGround(sequence_valid_scan_paths_[30], unground, ground);
//    std::string file_name1 = save_pcd_directory_ + "scan_ground.pcd";
//    pcl::io::savePCDFileBinary(file_name1, *ground);
//    std::string file_name2 = save_pcd_directory_ + "scan_unground.pcd";
//    pcl::io::savePCDFileBinary(file_name2, *unground);

//    point distance test
//    pcl::PointCloud<PointType>::Ptr point = scans_[35];
////    pcl::PointCloud<PointType>::Ptr point = local2global(scans_[35], 35);
//    pcl::PointCloud<PointType>::Ptr newpoint(new pcl::PointCloud<PointType>);
//    int num_of_scan = point->points.size();
//    int num_100 = 0;
//    int num_100_200 = 0;
//    int num_200_300 = 0;
//    int num_300 = 0;
//    for (int i = 0; i < num_of_scan; ++i) {
//        float distance = pointDistance(point->points[i]);
////        float distance = globalPointDistance(point->points[i], 35);
////        if(distance >= 60)
////            newpoint->points.push_back(point->points[i]);
//
//        if(distance < 20){
//            num_100++;
//        }else if(distance < 50){
//            num_100_200++;
//        }else if(distance < 80){
//            num_200_300++;
//        }else{
//            num_300++;
//        }
//    }
////    std::string file_name2 = save_pcd_directory_ + "scan_xyz.pcd";
////    pcl::io::savePCDFileBinary(file_name2, *newpoint);
//    cout<<"distance < 20 is "<<num_100<<endl;
//    cout<<"distance < 50 is "<<num_100_200<<endl;
//    cout<<"distance < 80 is "<<num_200_300<<endl;
//    cout<<"distance > 80 is "<<num_300<<endl;
}