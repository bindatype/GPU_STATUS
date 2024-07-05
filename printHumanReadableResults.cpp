#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <ctime>
#include <vector>
#include <cuda_runtime_api.h>
#include <nvml.h>
#include "getters.h"

void printHumanReadableResults() {
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
    std::cout << "Timestamp: " << unixTime << std::endl;
    std::cout << "NVIDIA Driver: " << nvidiaDriverVersion << std::endl;
    std::cout << "Cuda Driver: " << cudaDriverVersion << std::endl;

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

        std::cout << "\033[1;4mDevice " << i << "\033[0m - " << name << " Serial Number: " << serialnumber << std::endl;
        std::cout << "  Number GPU Cores: " << numCores << std::endl;
        switch (computemode) {
            case NVML_COMPUTEMODE_DEFAULT:
                std::cout << "  Compute Mode: Default compute mode -- multiple contexts per device." << std::endl;
                break;
            case NVML_COMPUTEMODE_EXCLUSIVE_THREAD:
                std::cout << "  Compute Mode: only one context per device, usable from one thread at a time." << std::endl;
                break;
            case NVML_COMPUTEMODE_PROHIBITED:
                std::cout << "  Compute Mode: no contexts per device." << std::endl;
                break;
            case NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
                std::cout << "  Compute Mode: only one context per device, usable from multiple threads at a time." << std::endl;
                break;
            default:
                std::cout << "  Compute Mode: Unable to determine compute mode." << std::endl;
        }
        if (NVML_FEATURE_ENABLED == mode) {
            std::cout << "  Persistence Mode: \033[1;32mON\033[0m" << std::endl;
        } else {
            std::cout << "  Persistence Mode: \033[1;31mOFF\033[0m" << std::endl;
        }
        if (NVML_PSTATE_0 == pState) {
            std::cout << "  Power State: \033[1;32mMax Performance\033[0m" << std::endl;
        } else {
            std::cout << "  Power State: \033[1;31mSuboptimal Performance\033[0m" << std::endl;
        }
        if (power <= 50000) {
            std::cout << "  Power: \033[1;36m" << power / 1000. << "\033[0m W" << std::endl;
        } else if (power > 50000 && power <= 70000) {
            std::cout << "  Power: \033[1;33m" << power / 1000. << "\033[0m W" << std::endl;
        } else {
            std::cout << "  Power: \033[1;31m" << power / 1000. << "\033[0m W" << std::endl;
        }
        std::cout << "  Enforced Power Limit: " << enforcedPowerLimit / 1000. << " W" << std::endl;
        std::cout << "  Total Energy Consumed: " << energy / 3.6e+9 << " kWh since last driver-reload" << std::endl;
        std::cout << "  Memory Usage: " << memory.used / 1048576 << " MB of " << memory.total / 1048576 << " MB" << std::endl;
        if (temperature <= 40) {
            std::cout << "  Temperature: \033[1;36m" << temperature << "\033[0m\u00b0 C" << std::endl;
        } else if (temperature > 40 && temperature <= 70) {
            std::cout << "  Temperature: \033[1;33m" << temperature << "\033[0m\u00b0 C" << std::endl;
        } else {
            std::cout << "  Temperature: \033[1;31m" << temperature << "\033[0m\u00b0 C" << std::endl;
        }
        std::cout << "  GPU Utilization: " << utilization.gpu << "%" << std::endl;
        std::cout << "  Memory Utilization: " << utilization.memory << "%" << std::endl;
        if (NVML_FEATURE_ENABLED == currentECCMode) {
            std::cout << "  Current ECC Mode: \033[1;32mON\033[0m " << std::endl;
        } else {
            std::cout << "  Current ECC Mode: \033[1;31mOFF\033[0m " << std::endl;
        }
        if (NVML_FEATURE_ENABLED == pendingECCMode) {
            std::cout << "  Pending ECC Mode: \033[1;32mON\033[0m " << std::endl;
        } else {
            std::cout << "  Pending ECC Mode: \033[1;31mOFF\033[0m " << std::endl;
        }
        std::cout << "  ECC Mem Errors: " << eccCounts.deviceMemory << std::endl;
        std::cout << "  L1 Cache Errors: " << eccCounts.l1Cache << std::endl;
        std::cout << "  L2 Cache Errors: " << eccCounts.l2Cache << std::endl;
        std::cout << "  Register File Errors: " << eccCounts.registerFile << std::endl;

        // Get process information
        unsigned int processCount = 0;
        nvmlResult = nvmlDeviceGetComputeRunningProcesses(device, &processCount, NULL);
        std::vector<nvmlProcessInfo_t> processes(processCount);
        nvmlResult = nvmlDeviceGetComputeRunningProcesses(device, &processCount, &processes[0]);
        if (nvmlResult == NVML_SUCCESS && processCount > 0) {
            for (unsigned int j = 0; j < processCount; ++j) {
                std::string processName = getProcessName(processes[j].pid);
                std::string commandName = getCommandName(processes[j].pid);
                std::cout << "  Process ID: " << processes[j].pid << " (" << commandName << ") using " << processes[j].usedGpuMemory / 1048576 << " MB" << std::endl;
            }
        }
    }
    nvmlShutdown();
}