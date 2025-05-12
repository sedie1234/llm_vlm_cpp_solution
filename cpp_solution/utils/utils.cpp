#include <chrono>
#include <iostream>
#include <fstream>
#include <string>


#include "utils.h"

void MyTimer::start() {
    begin = std::chrono::high_resolution_clock::now();
}

void MyTimer::stop() {
    end = std::chrono::high_resolution_clock::now();
}

double MyTimer::elapsed_us() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

size_t getPeakRSS() {
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.rfind("VmHWM:", 0) == 0) { // Peak resident set size (kB)
            size_t peak_kb;
            sscanf(line.c_str(), "VmHWM: %zu kB", &peak_kb);
            return peak_kb; // kB 단위
        }
    }
    return 0;
}

