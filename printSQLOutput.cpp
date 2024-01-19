#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <ctime>
#include <vector>
#include <cuda_runtime_api.h>
#include <nvml.h>
#include "getters.h"


void printSQLOutput(){
	nvmlReturn_t nvmlResult;

	// Initialize NVML -- do not call  nvmlSystemGet'ters before initializing!
	nvmlResult = nvmlInit();
	if (nvmlResult != NVML_SUCCESS) {
		std::cerr << "Failed to initialize NVML: " << nvmlErrorString(nvmlResult) << std::endl;
	}
	cudaError_t cudaResult;

	int deviceCount;
	std::time_t unixTime = std::time(0);

	// Get cuda and nvidia driber versions
	char nvidiaDriverVersion[30];
	int cudaDriverVersion=0;
	nvmlSystemGetDriverVersion(nvidiaDriverVersion,sizeof(nvidiaDriverVersion));
	nvmlSystemGetCudaDriverVersion(&cudaDriverVersion);

	// Get number of CUDA devices
	cudaResult = cudaGetDeviceCount(&deviceCount); 
	if (cudaResult != cudaSuccess) {
		std::cerr << "Failed to get CUDA device count: " << cudaGetErrorString(cudaResult) << std::endl;
		nvmlShutdown();
	}


	nvmlDevice_t device;
	nvmlUtilization_t utilization;
	nvmlMemory_t memory;
	unsigned int temperature, power;
	char name[64];

	// Iterate over devices
	for (int i = 0; i < deviceCount; ++i) {
		std::cout << "INSERT IGNORE INTO TableX (timestamp,nvidia_driver,cuda_driver";
		std::cout << ",device_id,device_name,serial_number,num_cores,compute_mode";
		std::cout << ",persist_mode,power_state,power,energy,mem_used,mem_total";
		std::cout << ",temperature,gpu_util,gpu_mem_util,curr_ecc_mode,pend_ecc_mode";
		std::cout << ",ecc_mem_errs,l1_errs,l2_errs,reg_file_errs";
		std::cout << ",pid,commandname,usedGpuMemory";
		std::cout << ") VALUES (";
		std::cout << "\""<< unixTime<<"\"";
		std::cout << ",\""<< nvidiaDriverVersion <<"\"";
		std::cout << ",\""<< cudaDriverVersion <<"\"";
		//		std::cout << ");";

		// Get handle to the NVML device
		nvmlResult = nvmlDeviceGetHandleByIndex(i, &device);
		if (nvmlResult != NVML_SUCCESS) {
			std::cerr << "Failed to get handle for device " << i << ": " << nvmlErrorString(nvmlResult) << std::endl;
			continue;
		}

		// Note to future self - see https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries
		char serialnumber[NVML_DEVICE_SERIAL_BUFFER_SIZE];
		unsigned int numCores=0;
		unsigned long long energy = 0;
		nvmlEccErrorCounts_t eccCounts;
		nvmlEnableState_t  currentECCMode, pendingECCMode;
		nvmlMemoryErrorType_t errorType;
		nvmlEccCounterType_t counterType;
		nvmlEnableState_t mode;
		nvmlPstates_t pState;
		nvmlComputeMode_t computemode;

		// Get device properties
		nvmlResult = nvmlDeviceGetName(device, name, sizeof(name));
		nvmlResult = nvmlDeviceGetSerial(device, serialnumber,NVML_DEVICE_SERIAL_BUFFER_SIZE);
		nvmlResult = nvmlDeviceGetUtilizationRates(device, &utilization);
		nvmlResult = nvmlDeviceGetPersistenceMode (device, &mode);
		nvmlResult = nvmlDeviceGetMemoryInfo(device, &memory);
		nvmlResult = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
		nvmlResult = nvmlDeviceGetPowerUsage(device, &power);
		nvmlResult = nvmlDeviceGetNumGpuCores(device, &numCores );
		nvmlResult = nvmlDeviceGetPowerState(device,&pState);
		nvmlResult = nvmlDeviceGetEccMode(device,&currentECCMode,&pendingECCMode);
		nvmlResult = nvmlDeviceGetDetailedEccErrors(device, NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_VOLATILE_ECC, &eccCounts);
		nvmlResult = nvmlDeviceGetComputeMode(device,&computemode);
		nvmlResult = nvmlDeviceGetTotalEnergyConsumption (device, &energy );

		std::cout << ",\""<< i <<"\"";
		std::cout << ",\""<< name <<"\"";
		std::cout << ",\""<< serialnumber <<"\"";
		std::cout << ",\""<< numCores <<"\"";
		switch (computemode) {
			case NVML_COMPUTEMODE_DEFAULT :
				std::cout << ",\"default\"" ;
				break;
			case NVML_COMPUTEMODE_EXCLUSIVE_THREAD:
				std::cout << ",\"exclusive_thread\"" ;
				break;
			case NVML_COMPUTEMODE_PROHIBITED:
				std::cout << ",\"prohibited\"" ;
				break;
			case NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
				std::cout << ",\"exclusive_process\"" ;
				break;
			default:
				std::cout << ",\"unknown\"" ;

		}
		if ( NVML_FEATURE_ENABLED == mode ){
			std::cout << ",\"on\"" ;
		} else {
			std::cout << ",\"off\"" ;
		}
		if ( NVML_PSTATE_0 == pState ){
			std::cout << ",\"max\"" ;
		} else {
			std::cout << ",\"suboptimal\"" ;
		}
		std::cout << ",\""<< power/1000. <<"\"";
		std::cout << ",\""<< energy/3.6e+9 <<"\"";
		std::cout << ",\""<< memory.used / 1048576 <<"\"";
		std::cout << ",\""<< memory.total / 1048576 <<"\"";
		std::cout << ",\""<< temperature <<"\"";
		std::cout << ",\""<< utilization.gpu <<"\"";
		std::cout << ",\""<< utilization.memory <<"\"";
		if ( NVML_FEATURE_ENABLED == currentECCMode){
			std::cout << ",\"on\"" ;
		} else {
			std::cout << ",\"off\"" ;
		}
		if ( NVML_FEATURE_ENABLED == pendingECCMode ){
			std::cout << ",\"on\"" ;
		} else {
			std::cout << ",\"off\"" ;
		}
		std::cout << ",\""<< eccCounts.deviceMemory <<"\"";
		std::cout << ",\""<< eccCounts.l1Cache <<"\"";
		std::cout << ",\""<< eccCounts.l2Cache <<"\"";
		std::cout << ",\""<< eccCounts.registerFile <<"\"";


		// Get process information
		unsigned int processCount = 0;
		nvmlResult = nvmlDeviceGetComputeRunningProcesses(device, &processCount, NULL);
		std::vector<nvmlProcessInfo_t> processes(processCount);
		nvmlResult = nvmlDeviceGetComputeRunningProcesses(device, &processCount, &processes[0]);
		if (nvmlResult == NVML_SUCCESS && processCount > 0) {
			for (unsigned int j = 0; j < processCount; ++j) {
				std::string processName = getProcessName(processes[j].pid);
				std::string commandName = getCommandName(processes[j].pid);
				std::cout << ",\""<< processes[j].pid <<"\"";
				std::cout << ",\""<< commandName <<"\"";
				std::cout << ",\""<< processes[j].usedGpuMemory / 1048576 <<"\"";
			}
		}
		std::cout << ");" << std::endl;
	}
	nvmlShutdown();
}
