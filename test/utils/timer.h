#pragma once
#include<chrono>

class Timer
{
    typedef std::chrono::time_point<std::chrono::system_clock> tp;
public:
    Timer() = default;

    static uint64_t Now()
    {
        auto now = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
        return static_cast<uint64_t>(now.time_since_epoch().count());
    }

    static uint64_t Duration(uint64_t start, uint64_t end)
    {
        return (end - start);
    }
};