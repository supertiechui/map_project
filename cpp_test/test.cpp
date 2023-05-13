#include <iostream>
#include <fstream>
#include <string>
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Jacobi"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "pcl/point_cloud.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp>
#include <unordered_map>
#include <set>
#include <queue>
#include <map>

using namespace std;
using namespace nvinfer1;

class Logger : public ILogger {
public:
    void set_verbosity(bool verbose) { _verbose = verbose; }
    void log(Severity severity, const char* msg) noexcept {
        if (_verbose) {
            switch (severity) {
                case Severity::kINTERNAL_ERROR:
                    std::cerr << "INTERNAL_ERROR: ";
                    break;
                case Severity::kERROR:
                    std::cerr << "ERROR: ";
                    break;
                case Severity::kWARNING:
                    std::cerr << "WARNING: ";
                    break;
                case Severity::kINFO:
                    std::cerr << "INFO: ";
                    break;
                default:
                    std::cerr << "UNKNOWN: ";
                    break;
            }
            std::cout << msg << std::endl;
        }
    }

private:
    bool _verbose = false;
};
template<typename T>
cv::Mat convertColorMappedImg (const cv::Mat &_src, std::pair<T, T> _caxis)
{
    T min_color_val = _caxis.first;
    T max_color_val = _caxis.second;

    cv::Mat image_dst;
    image_dst = 255 * (_src - min_color_val) / (max_color_val - min_color_val);
    image_dst.convertTo(image_dst, CV_8UC1);

    cv::applyColorMap(image_dst, image_dst, cv::COLORMAP_JET);

    return image_dst;
}
void cvTest()
{
    string path1 = "/home/heht/PCD/range/scan/000452X2.5.jpg";
    cv::Mat dst;
    std::pair<float, float> kRangeColorAxis = std::pair<float, float> {0, 20}; // meter
    //以灰度读取原图
    cv::Mat img = cv::imread(path1, 0);
    dst = convertColorMappedImg(img,kRangeColorAxis);
//    cv::applyColorMap(img, img,6);//i代表不同的色带：[0,21]
    cv::imshow("img",dst);
    cv::waitKey();
}
void cpp_test()
{
    string bin_path = "/home/heht/kitti/sequences/01/velodyne/000435.bin";

    FILE *file = fopen(bin_path.c_str(), "rb");
    if (!file) {
        std::cerr << "error: failed to load " << bin_path << std::endl;
        return;
    }

    std::vector<float> buffer(1000000);
    int num_points = fread(reinterpret_cast<char *>(buffer.data()), sizeof(float), buffer.size(), file) / 4;

//    int j=0;
//    for (auto &p: ground) {
//        eigen_ground.row(j++) << p.x, p.y, p.z;
//    }
//    Eigen::MatrixX3f centered = eigen_ground.rowwise() - eigen_ground.colwise().mean();
//    Eigen::MatrixX3f cov = (centered.adjoint() * centered) / double(eigen_ground.rows() - 1);
//    cloud.resize(num_points, 4);
    Eigen::MatrixX4f cloud(num_points, 4);
    for (int i=0; i<num_points; i++)
    {
        cloud.row(i) << buffer[i*4], buffer[i*4+1], buffer[i*4+2], buffer[i*4+3];
    }
    Eigen::MatrixX3f test = cloud.block(0,0,cloud.rows(),3).rowwise()-cloud.block(0,0,cloud.rows(),3).colwise().mean();
//    cout<<cloud<<endl;
    cout<<cloud.block(0,0,cloud.rows(),3)<< endl;
}

class MyLinkedList {
public:
    struct ListNode
    {
        int val;
        ListNode* next;
        ListNode(int val_) : val(val_), next(nullptr){}
    };

    MyLinkedList() {
        Head = new ListNode(0);
        size_ = 0;
    }

    int get(int index) {
        ListNode* cur = Head;
        for(int i = 0; i < index; i++)
            cur = cur->next;
        return cur->next->val;
    }

    void addAtHead(int val) {
        ListNode* l1 = new ListNode(val);
        ListNode* l2 = Head->next;
        Head->next = l1;
        l1->next = l2;
        size_++;
        printLinkList();
    }

    void addAtTail(int val) {
        ListNode* cur = Head;
        for(int i = 0; i < size_; i++)
            cur = cur->next;
        ListNode* newNode = new ListNode(val);
        newNode->next = cur->next;
        cur->next = newNode;

        size_++;
        printLinkList();
    }

    void addAtIndex(int index, int val) {
        if(index < 0 || index > size_)
        {
            cout<<"error!"<<endl;
            return;
        }
        ListNode* cur = Head;
        for(int i = 0; i < index; i++)
            cur = cur->next;
        ListNode* newNode = new ListNode(val);
        newNode->next = cur->next;
        cur->next = newNode;

        size_++;
        printLinkList();
    }

    void deleteAtIndex(int index) {
        if(index < 0 || index > size_)
        {
            cout<<"error!"<<endl;
            return;
        }
        ListNode* cur = Head;
        for(int i = 0; i < index; i++)
            cur = cur->next;
        ListNode* del = cur->next;
        cur->next = cur->next->next;
        delete del;
        del = nullptr;
        size_--;
    }

    void printLinkList()
    {
        if (size_ == 0)
        {
            return;
        }
        ListNode* pCur = Head->next;
        while (pCur != nullptr)
        {
            cout << pCur->val << " ";
            pCur = pCur->next;
        }
        cout << endl;
    }
private:
    ListNode* Head;
    int size_;


};

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList* obj = new MyLinkedList();
 * int param_1 = obj->get(index);
 * obj->addAtHead(val);
 * obj->addAtTail(val);
 * obj->addAtIndex(index,val);
 * obj->deleteAtIndex(index);
 */
int main()
{
    cpp_test();
//    cvTest();
//    Logger gLogger;
//    // 定义IBuilder和iNetworkDefinition
//    IBuilder* builder = createInferBuilder(gLogger);
//    IBuilderConfig* config = builder->createBuilderConfig();
//    INetworkDefinition* network = builder->createNetworkV2(1);
//
//// 创建ONNX的解析器
//    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);
//
//// 获取模型
//    const char* onnx_filename = "/home/heht/catkin_net/src/rangenet_lib/darknet53/model.onnx";
//    cout<<onnx_filename<<endl;
//    cout<<"find ? "<<parser->parseFromFile(onnx_filename, static_cast<int>(ILogger::Severity::kWARNING));

    return 0;
}

