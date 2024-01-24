#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <ctime>
#include <vector>
#include <cuda_runtime_api.h>
#include <nvml.h>
#include "printHumanReadableResults.h"
#include "printSQLOutput.h"
#include "usage.h"

bool outputSQL = false;
std::ofstream sqlFile;


int main(int argc, char**argv) {
// 
    for (int i = 1; i < argc; ++i) {
	if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
	    usage();
	    exit(0);
	}
	if (strcmp(argv[i], "--output=sql") == 0) {
	    outputSQL = true;
	}
    }

    if (outputSQL) {
	printSQLOutput();
    }else{
	printHumanReadableResults();
    }

    return 0;
}

