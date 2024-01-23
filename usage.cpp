#include <iostream>

void usage(){
    std::cout<< "Usage: "<< std::endl;
    std::cout<< "\t gpu_status "<< std::endl;
    std::cout<< "\t\t prints gpu information in human-readable format."<< std::endl;
    std::cout<< "\t gpu_status -h,--help"<< std::endl;
    std::cout<< "\t\tThis help message."<< std::endl;
    std::cout<< "\t gpu_status --output=sql"<< std::endl;
    std::cout<< "\t\t formating suitable for uploading into sql"<< std::endl;
}
