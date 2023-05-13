cmake_minimum_required(VERSION 3.1)
project(cpp_test)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_VERSION VERSION_LESS "3.15")
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ListPrepend.cmake")
    list_prepend(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
else()
    list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
endif()

# cuda10.0
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt 7.0.0.11
set(TENSORRT_ROOT /home/heht/TensorRT/TensorRT-8.2.2.1)
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)
find_package(TensorRT PATHS cmake)
if(TENSORRT_FOUND)
    message("TensorRT SUCCESS!\n")
endif()

include(FetchContent)
find_package(Eigen3 REQUIRED QUIET)
find_package(OpenCV REQUIRED)
find_package(PCL 1.10 REQUIRED)
#find_package(Open3D HINTS REQUIRED)
#list(APPEND Open3D_LIBRARIES dl)
#message(STATUS "Found Open3D ${Open3D_VERSION}")
# include directories

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/patchworkpp/include
        ${EIGEN3_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        # ${GTSAM_INCLUDE_DIR}
)
link_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/patchworkpp/include
        ${PCL_LIBRARY_DIRS}
        ${OpenCV_LIBRARY_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
#        ${Open3D_LIBRARY_DIRS}
        # ${GTSAM_LIBRARY_DIRS}
)
add_subdirectory(patchworkpp)
set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${Open3D_LIBRARIES}
        PATCHWORK::patchworkpp
        )

add_library(myslam SHARED src/remove.cpp src/utility.cpp)
target_link_libraries(myslam  ${THIRD_PARTY_LIBS})
# Range Image Projection
add_executable(my_removert src/main.cpp)
target_link_libraries(my_removert myslam)

add_executable(cpp_test test.cpp )
target_link_libraries(cpp_test ${OpenCV_LIBS} ${PCL_LIBRARY_DIRS} ${TENSORRT_LIBRARIES})
#
#add_executable(pcll pcl_test.cpp )
#target_link_libraries(pcll ${OpenCV_LIBS} ${PCL_LIBRARIES})
