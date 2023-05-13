#include "../include/utility.h"

bool cout_debug = false;


void readBin(std::string _bin_path, pcl::PointCloud<PointType>::Ptr _pcd_ptr)
{
 	std::fstream input(_bin_path.c_str(), ios::in | ios::binary);
	if(!input.good()){
		cerr << "Could not read file: " << _bin_path << endl;
		exit(EXIT_FAILURE);
	}
	input.seekg(0, ios::beg);
  
	for (int ii=0; input.good() && !input.eof(); ii++) {
		PointType point;

		input.read((char *) &point.x, sizeof(float));
		input.read((char *) &point.y, sizeof(float));
		input.read((char *) &point.z, sizeof(float));
		input.read((char *) &point.intensity, sizeof(float));

		_pcd_ptr->push_back(point);
	}
	input.close();
}

std::vector<double> splitPoseLine(std::string _str_line, char _delimiter) {
    std::vector<double> parsed;
    std::stringstream ss(_str_line);
    std::string temp;
    while (getline(ss, temp, _delimiter)) {
        parsed.push_back(std::stod(temp)); // convert string to "double"
    }
    return parsed;
}

SphericalPoint cart2sph(const PointType & _cp)
{ // _cp means cartesian point

    if(cout_debug){
        cout << "Cartesian Point [x, y, z]: [" << _cp.x << ", " << _cp.y << ", " << _cp.z << endl;
    }

    SphericalPoint sph_point {
         std::atan2(_cp.y, _cp.x), 
         std::atan2(_cp.z, std::sqrt(_cp.x*_cp.x + _cp.y*_cp.y)),
         std::sqrt(_cp.x*_cp.x + _cp.y*_cp.y + _cp.z*_cp.z)
    };    
    return sph_point;
}


std::pair<int, int> resetRimgSize(const std::pair<float, float> _fov, const float _resize_ratio)
{
    // default is 1 deg x 1 deg 
    float alpha_vfov = _resize_ratio;    
    float alpha_hfov = _resize_ratio;    

    float V_FOV = _fov.first;
    float H_FOV = _fov.second;

    int NUM_RANGE_IMG_ROW = std::round(V_FOV*alpha_vfov);
    int NUM_RANGE_IMG_COL = std::round(H_FOV*alpha_hfov);

    std::pair<int, int> rimg {NUM_RANGE_IMG_ROW, NUM_RANGE_IMG_COL};
    return rimg;
}


std::set<int> convertIntVecToSet(const std::vector<int> & v) 
{ 
    std::set<int> s; 
    for (int x : v) { 
        s.insert(x); 
    } 
    return s; 
}

float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

