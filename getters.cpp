#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <ctime>
#include <vector>
#include <cuda_runtime_api.h>
#include <nvml.h>
#include <unistd.h>

std::string getHostName(){
    const size_t bufferSize = 1024; // Define the buffer size as 1 kB
    std::vector<char> buffer(bufferSize);  

    // Call gethostname and check for errors
    if (gethostname(buffer.data(), bufferSize) != 0) {
	std::cerr << "Error getting hostname" << std::endl;
	exit(1);
    }
    std::string hostname(buffer.data());
    return hostname;
}

std::string getProcessName(unsigned int pid) {
    std::stringstream ss;
    ss << "/proc/" << pid << "/comm";
    std::ifstream commFile(ss.str().c_str());
    std::string processName;
    if (commFile.good()) {
	std::getline(commFile, processName);
    } else {
	processName = "Unknown";
    }
    return processName;
}

std::string getCommandName(unsigned int pid) {
    std::stringstream ss;
    ss << "/proc/" << pid << "/cmdline";
    std::ifstream commFile(ss.str().c_str());
    std::string commandName;
    if (commFile.good()) {
	std::getline(commFile, commandName);
    } else {
	commandName = "Unknown";
    }
    return commandName;
}
