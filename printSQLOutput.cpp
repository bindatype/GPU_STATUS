#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <ctime>
#include <vector>
#include <cuda_runtime_api.h>
#include <nvml.h>
#include <unistd.h>
#include "getters.h"

void printSQLOutput() {
    nvmlReturn_t nvmlResult;

    // Initialize NVML -- do not call nvmlSystemGet'ters before initializing!
    nvmlResult = nvmlInit();
    if (nvmlResult != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(nvmlResult) << std::endl;
        return;
    }
    cudaError_t cudaResult;

    int deviceCount;
    std::time_t unixTime = std::time(0);

    // Get cuda and nvidia driver versions
    char nvidiaDriverVersion[30];
    int cudaDriverVersion = 0;
    nvmlSystemGetDriverVersion(nvidiaDriverVersion, sizeof(nvidiaDriverVersion));
    nvmlSystemGetCudaDriverVersion(&cudaDriverVersion);

    // Get number of CUDA devices
    cudaResult = cudaGetDeviceCount(&deviceCount);
    if (cudaResult != cudaSuccess) {
        std::cerr << "Failed to get CUDA device count: " << cudaGetErrorString(cudaResult) << std::endl;
        nvmlShutdown();
        return;
    }

    nvmlDevice_t device;
    nvmlUtilization_t utilization;
    nvmlMemory_t memory;
    unsigned int temperature, power, enforcedPowerLimit;
    char name[64];

    // Iterate over devices
    for (int i = 0; i < deviceCount; ++i) {
        std::cout << "INSERT IGNORE INTO gpu_monitoring (timestamp,hostname,nvidia_driver,cuda_driver";
        std::cout << ",device_id,device_name,serial_number,num_cores,compute_mode";
        std::cout << ",persist_mode,power_state,power,enforced_power_limit,energy,mem_used,mem_total";
        std::cout << ",temperature,gpu_util,gpu_mem_util,curr_ecc_mode,pend_ecc_mode";
        std::cout << ",ecc_mem_errs,l1_errs,l2_errs,reg_file_errs";
        std::cout << ",pid_1,commandname_1,usedGpuMemory_1";
        std::cout << ",pid_2,commandname_2,usedGpuMemory_2";
        std::cout << ") VALUES (";
        std::cout << "\"" << unixTime << "\"";
        std::cout << ",\"" << getHostName() << "\"";
        std::cout << ",\"" << nvidiaDriverVersion << "\"";
        std::cout << ",\"" << cudaDriverVersion << "\"";

        // Get handle to the NVML device
        nvmlResult = nvmlDeviceGetHandleByIndex(i, &device);
        if (nvmlResult != NVML_SUCCESS) {
            std::cerr << "Failed to get handle for device " << i << ": " << nvmlErrorString(nvmlResult) << std::endl;
            continue;
        }

        // Note to future self - see https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries
        char serialnumber[NVML_DEVICE_SERIAL_BUFFER_SIZE];
        unsigned int numCores = 0;
        unsigned long long energy = 0;
        nvmlEccErrorCounts_t eccCounts;
        nvmlEnableState_t currentECCMode, pendingECCMode;
        nvmlMemoryErrorType_t errorType;
        nvmlEccCounterType_t counterType;
        nvmlEnableState_t mode;
        nvmlPstates_t pState;
        nvmlComputeMode_t computemode;

        // Get device properties
        nvmlResult = nvmlDeviceGetName(device, name, sizeof(name));
        nvmlResult = nvmlDeviceGetSerial(device, serialnumber, NVML_DEVICE_SERIAL_BUFFER_SIZE);
        nvmlResult = nvmlDeviceGetUtilizationRates(device, &utilization);
        nvmlResult = nvmlDeviceGetPersistenceMode(device, &mode);
        nvmlResult = nvmlDeviceGetMemoryInfo(device, &memory);
        nvmlResult = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
        nvmlResult = nvmlDeviceGetPowerUsage(device, &power);
        nvmlResult = nvmlDeviceGetNumGpuCores(device, &numCores);
        nvmlResult = nvmlDeviceGetPowerState(device, &pState);
        nvmlResult = nvmlDeviceGetEccMode(device, &currentECCMode, &pendingECCMode);
        nvmlResult = nvmlDeviceGetDetailedEccErrors(device, NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_VOLATILE_ECC, &eccCounts);
        nvmlResult = nvmlDeviceGetComputeMode(device, &computemode);
        nvmlResult = nvmlDeviceGetTotalEnergyConsumption(device, &energy);

        // Get enforced power limit
        nvmlResult = nvmlDeviceGetEnforcedPowerLimit(device, &enforcedPowerLimit);
        if (nvmlResult != NVML_SUCCESS) {
            std::cerr << "Failed to get enforced power limit for device " << i << ": " << nvmlErrorString(nvmlResult) << std::endl;
            continue;
        }

        std::cout << ",\"" << i << "\"";
        std::cout << ",\"" << name << "\"";
        std::cout << ",\"" << serialnumber << "\"";
        std::cout << ",\"" << numCores << "\"";
        switch (computemode) {
            case NVML_COMPUTEMODE_DEFAULT:
                std::cout << ",\"default\"";
                break;
            case NVML_COMPUTEMODE_EXCLUSIVE_THREAD:
                std::cout << ",\"exclusive_thread\"";
                break;
            case NVML_COMPUTEMODE_PROHIBITED:
                std::cout << ",\"prohibited\"";
                break;
            case NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
                std::cout << ",\"exclusive_process\"";
                break;
            default:
                std::cout << ",\"unknown\"";
        }
        if (NVML_FEATURE_ENABLED == mode) {
            std::cout << ",\"on\"";
        } else {
            std::cout << ",\"off\"";
        }
        if (NVML_PSTATE_0 == pState) {
            std::cout << ",\"max\"";
        } else {
            std::cout << ",\"suboptimal\"";
        }
        std::cout << ",\"" << power / 1000. << "\"";
        std::cout << ",\"" << enforcedPowerLimit / 1000. << "\"";
        std::cout << ",\"" << energy / 3.6e+9 << "\"";
        std::cout << ",\"" << memory.used / 1048576 << "\"";
        std::cout << ",\"" << memory.total / 1048576 << "\"";
        std::cout << ",\"" << temperature << "\"";
        std::cout << ",\"" << utilization.gpu << "\"";
        std::cout << ",\"" << utilization.memory << "\"";
        if (NVML_FEATURE_ENABLED == currentECCMode) {
            std::cout << ",\"on\"";
        } else {
            std::cout << ",\"off\"";
        }
        if (NVML_FEATURE_ENABLED == pendingECCMode) {
            std::cout << ",\"on\"";
        } else {
            std::cout << ",\"off\"";
        }
        std::cout << ",\"" << eccCounts.deviceMemory << "\"";
        std::cout << ",\"" << eccCounts.l1Cache << "\"";
        std::cout << ",\"" << eccCounts.l2Cache << "\"";
        std::cout << ",\"" << eccCounts.registerFile << "\"";

        // Get process information
        std::string processName;
        std::string commandName;
        unsigned int processCount = 0;
        nvmlResult = nvmlDeviceGetComputeRunningProcesses(device, &processCount, NULL);
        std::vector<nvmlProcessInfo_t> processes(processCount);
        nvmlResult = nvmlDeviceGetComputeRunningProcesses(device, &processCount, &processes[0]);

        // Kludge to get last 6 (process info) fields to format properly.
        // Handles at most, 2 processes per gpu. That seems to be rare for us and 3 processes seems never to happen. 

        if (nvmlResult == NVML_SUCCESS && processCount > 0) {
            switch (processCount) {
                case 1: {
                    processName = getProcessName(processes[0].pid);
                    commandName = getCommandName(processes[0].pid);
                    std::cout << ",\"" << processes[0].pid << "\"";
                    std::cout << ",\"" << commandName << "\"";
                    std::cout << ",\"" << processes[0].usedGpuMemory / 1048576 << "\"";
                    std::cout << ",NULL";
                    std::cout << ",NULL";
                    std::cout << ",NULL";
                    break;
                }
                case 2: {
                    processName = getProcessName(processes[0].pid);
                    commandName = getCommandName(processes[0].pid);
                    std::cout << ",\"" << processes[0].pid << "\"";
                    std::cout << ",\"" << commandName << "\"";
                    std::cout << ",\"" << processes[0].usedGpuMemory / 1048576 << "\"";
                    processName = getProcessName(processes[1].pid);
                    commandName = getCommandName(processes[1].pid);
                    std::cout << ",\"" << processes[1].pid << "\"";
                    std::cout << ",\"" << commandName << "\"";
                    std::cout << ",\"" << processes[1].usedGpuMemory / 1048576 << "\"";
                    break;
                }
                default:
                    std::cout << ",NULL";
                    std::cout << ",NULL";
                    std::cout << ",NULL";
                    std::cout << ",NULL";
                    std::cout << ",NULL";
                    std::cout << ",NULL";
            }
        } else {
            std::cout << ",NULL";
            std::cout << ",NULL";
            std::cout << ",NULL";
            std::cout << ",NULL";
            std::cout << ",NULL";
            std::cout << ",NULL";
        }
        std::cout << ");" << std::endl;
    }
    nvmlShutdown();
}