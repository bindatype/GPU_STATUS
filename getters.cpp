#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <ctime>
#include <vector>
#include <cuda_runtime_api.h>
#include <nvml.h>

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
