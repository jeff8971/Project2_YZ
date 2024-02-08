#include <iostream>
#include <vector>
#include "csv_util.h"
// 假设上述代码已经包含

int main() {
    // 写入CSV文件
    char filename[] = "test.csv";
    char image_filename[] = "image1.png";
    std::vector<float> image_data = {0.1, 0.2, 0.3};
    append_image_data_csv(filename, image_filename, image_data, true); // true to reset file
    
    // 读取CSV文件
    std::vector<char *> filenames;
    std::vector<std::vector<float>> data;
    read_image_data_csv(filename, filenames, data, true); // true to echo file content
    
    // 清理分配的内存
    for(auto &fname : filenames) {
        delete[] fname;
    }
    
    return 0;
}
