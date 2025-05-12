#pragma once

#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

class MyTimer {
public:
    MyTimer(){
        
    }
    void start();
    void stop();
    double elapsed_us() const;

private:
    std::chrono::high_resolution_clock::time_point begin, end;
};

size_t getPeakRSS();

template<typename T>
double computeAverage(const std::vector<T>& vec) {
    if (vec.empty()) return 0.0;
    T sum = std::accumulate(vec.begin(), vec.end(), T{0});
    return static_cast<double>(sum) / vec.size();
}