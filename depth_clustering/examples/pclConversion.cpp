#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
//#include <pcl/io/
#include <pcl/visualization/cloud_viewer.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

using namespace cv;
using namespace std;
using boost::ends_with;
namespace bp = boost::program_options;
namespace bf = boost::filesystem;

enum FORMAT {
    PCD = 0, BIN, TXT, PNG
};


void readCloudTxt(const string& file, pcl::PointCloud<pcl::PointXYZI>& cloud) {
    cloud.clear();
    locale::global(std::locale("en_US.UTF-8"));
    ifstream ifs(file.c_str());
    for (std::string line; std::getline(ifs, line, '\n');) {
        vector<string> coords_str;
        boost::split(coords_str, line, boost::is_any_of(" "));
        if (coords_str.size() != 4) {
            fprintf(stderr, "ERROR: format of line is wrong.\n");
            continue;
        }
        pcl::PointXYZI point;
        point.x = std::stof(coords_str[0]);
        point.y = std::stof(coords_str[1]);
        point.z = std::stof(coords_str[2]);
        point.intensity = std::stof(coords_str[3]);
        cloud.push_back(point);
    }
    ifs.close();
}

void readCloudBinary(const string& file, pcl::PointCloud<pcl::PointXYZI>& cloud) {
    cloud.clear();
    int32_t num = 1000000;
    float *data = (float*)malloc(num*sizeof(float));
    float *px = data+0;
    float *py = data+1;
    float *pz = data+2;
    float *pr = data+3;

    // load point cloud
    FILE *stream;
    stream = fopen(file.c_str(), "rb");
    num = fread(data, sizeof(float), num, stream)/4;
    for (int32_t i = 0; i < num; i++) {
        pcl::PointXYZI point;
        point.x = *px;
        point.y = *py;
        point.z = *pz;
        point.intensity = *pr;
        cloud.push_back(point);
        px += 4;
        py += 4;
        pz += 4;
        pr += 4;
    }
    fclose(stream);
}

void loadCloudFromFile(const string& file, pcl::PointCloud<pcl::PointXYZI>& cloud) {
    if (ends_with(file, ".pcd")) {
        pcl::io::loadPCDFile(file, cloud);
    } else if (ends_with(file, ".png") || ends_with(file, ".exr")) {
//        Mat image = imread(file, -1);
//        readCloudImage(image, cloud);
    } else if (ends_with(file, ".txt")) {
        readCloudTxt(file, cloud);
    } else if (ends_with(file, ".bin")) {
        readCloudBinary(file, cloud);
    }
}

vector<string> readFolderFiles(const string& folder) {
    vector<string> files;

    bf::path folderPath(folder);
    if (!bf::exists(folderPath))
        return files;

    bf::directory_iterator end_iter;
    for (bf::directory_iterator iter(folderPath); iter != end_iter; ++iter) {
        if (bf::is_directory(iter->status()))
            continue;

        if (bf::is_regular_file(iter->status()))
            files.push_back(iter->path().string());
    }

    return files;
}

bool checkFormat(const string& sf, FORMAT& of) {
    if (sf == "pcd")
        of = PCD;
    else if (sf == "bin")
        of = BIN;
    else if (sf == "txt")
        of = TXT;
    else if (sf == "png")
        of = PNG;
    else
        return false;

    return true;
}

void saveCloudWithFormat(const string& outFile, const FORMAT format,
                         const pcl::PointCloud<pcl::PointXYZI>& cloud) {
    switch (format) {
    case PCD: {
        pcl::io::savePCDFileASCII(outFile, cloud);
        break;
    }
    case BIN: {
        float* data = (float*)malloc(4*cloud.size()*sizeof(float));
        for (size_t i = 0; i < cloud.size(); ++i) {
            data[4*i+0] = cloud.points[i].x;
            data[4*i+1] = cloud.points[i].y;
            data[4*i+2] = cloud.points[i].z;
            data[4*i+3] = cloud.points[i].intensity;
        }
        FILE* fb;
        fb = fopen(outFile.c_str(), "wb");
        fwrite(data, sizeof(float), 4*cloud.size(), fb);
        fclose(fb);
        break;
    }
    case TXT: {
        ofstream ofs(outFile.c_str(), ios_base::out);
        for (size_t i = 0; i < cloud.size(); ++i) {
            ofs << cloud[i].x << " " << cloud[i].y << " "
                << cloud[i].z << " " << cloud[i].intensity << "\n";
        }
        ofs.close();
        break;
    }
    case PNG: {
        break;
    }
    default:
        break;
    }
}

int main(int argc, char **argv)
{
    cout << "Welcome to use point cloud conversion tool.\n * use -h to find usage." << endl;

    //解析参数
    string inFolder, outFolder;
    string outFormat;
    try {
        bp::options_description desc{"Options"};  //选项描述器
        desc.add_options()
          ("help,h", "Show Help")
          ("input-folder,i", bp::value<string>(), "Input data folder (with file format: .pcd/.bin/.txt/.png)")
          ("output-folder,o", bp::value<string>(), "Output data folder")
          ("output-format,f", bp::value<string>(), "Output format (pcd/bin/txt/png)");

        bp::variables_map vm; //选项存储器
        bp::store(parse_command_line(argc, argv, desc), vm);
        notify(vm);
        if (vm.count("help")) {
            cout << desc << endl;
            return 0;
        }
        if (vm.count("input-folder")  || vm.count("i"))
            inFolder = vm["input-folder"].as<string>();
        if (vm.count("output-folder") || vm.count("o"))
            outFolder = vm["output-folder"].as<string>();
        if (vm.count("output-format") || vm.count("f"))
            outFormat = vm["output-format"].as<string>();
    } catch (const bp::error& ex) {
        cerr << ex.what() << endl;
    }
    cout << endl << "Parameters show below:" << endl;
    cout << "input folder: " << inFolder << endl;
    cout << "output folder: " << outFolder << endl;
    cout << "output format: " << outFormat << endl;
    if (!bf::exists(inFolder)) {
        cerr << "[Error] input folder doesn't exist!" << endl;
        return -1;
    }
    FORMAT format;
    if (!checkFormat(outFormat, format)) {
        cerr << "[Error] Wrong output format!" << endl;
        return -1;
    }
    cout << "Parser sucessed." << endl;

    //读入数据
    vector<string> inFiles = readFolderFiles(inFolder);
    if (inFiles.empty()) {
        cerr << "[Error] Not laser data in the folder!" << endl;
        return -1;
    } else {
        cout << "Read " << inFiles.size() << " files in the folder." << endl;
    }
    sort(inFiles.begin(), inFiles.end());

    //格式转换并输出
    cout << "Turning data to new format..." << endl;
    system(("mkdir -p " + outFolder).c_str());
    if (!ends_with(outFolder, "/"))
        outFolder += "/";
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    for (const auto& f : inFiles) {
        cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
        loadCloudFromFile(f, *cloud);

        size_t s = f.find_last_of('/');
        size_t e = f.find_last_of('.');
        string fileName = outFolder + f.substr(s+1, e-s) + outFormat;
        saveCloudWithFormat(fileName, format, *cloud);
    }

    return 0;
}
