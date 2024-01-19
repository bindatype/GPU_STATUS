#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <ctime>
#include <vector>
#include <cuda_runtime_api.h>
#include <nvml.h>
#include "printHumanReadableResults.h"

bool outputSQL = false;
std::ofstream sqlFile;


int main(int argc, char**argv) {

	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "--output=sql") == 0) {
			outputSQL = true;
			sqlFile.open("output.sql");
		}
	}

	if (outputSQL) {

	}else{
		printHumanReadableResults();
	}

	return 0;
}
