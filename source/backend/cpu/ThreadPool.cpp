//
//  ThreadPool.cpp
//  MNN
//
//  Created by MNN on 2019/06/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_USE_THREAD_POOL
#include "backend/cpu/ThreadPool.hpp"
#include <string.h>
#include <MNN/MNNDefine.h>
#ifdef DEBUG_TIMES
#include <chrono>
#endif
#ifdef __ANDROID__
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#endif
//#define MNN_THREAD_LOCK_CPU
//#define MNN_USE_DYNAMIC_WORK_INDEX

#ifndef MNN_THREAD_POOL_MAX_TASKS
#define MNN_THREAD_POOL_MAX_TASKS 2
#endif
namespace MNN {
ThreadPool* ThreadPool::gInstance = nullptr;
static std::mutex gInitMutex;

#ifdef DEBUG_TIMES
extern int g_cur_step;
extern std::vector<std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds>> g_steps;
extern std::vector<std::tuple<int, std::chrono::nanoseconds, std::chrono::nanoseconds>> g_times;
#endif

int ThreadPool::init(int number) {
    if (1 >= number) {
        return 1;
    }
    std::lock_guard<std::mutex> _l(gInitMutex);
    if (nullptr != gInstance) {
        if (gInstance->number() < number) {
            return gInstance->number();
        }
    }
    if (nullptr == gInstance) {
        gInstance = new ThreadPool(number);
    }
    return number;
}
void ThreadPool::destroy() {
    std::lock_guard<std::mutex> _l(gInitMutex);
    if (nullptr != gInstance) {
        delete gInstance;
        gInstance = nullptr;
    }
}
#ifdef MNN_THREAD_LOCK_CPU
static int getNumberOfCPU() {
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp) {
        return 1;
    }
    int number = 0;
    char buffer[1024];
    while (!feof(fp)) {
        char* str = fgets(buffer, 1024, fp);
        if (!str) {
            break;
        }
        if (memcmp(buffer, "processor", 9) == 0) {
            number++;
        }
    }
    fclose(fp);
    if (number < 1) {
        number = 1;
    }
    return number;
}

static int getCPUMaxFreqKHz(int cpuID) {
    char path[256];
    sprintf(path, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuID);
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state", cpuID);
        fp = fopen(path, "rb");
        if (!fp) {
            sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuID);
            fp = fopen(path, "rb");
            if (!fp) {
                return -1;
            }
            int maxfrequency = -1;
            fscanf(fp, "%d", &maxfrequency);
            fclose(fp);
            return maxfrequency;
        }
    }
    int maxfrequency = 0;
    while (!feof(fp)) {
        int frequency = 0;
        int history   = fscanf(fp, "%d %*d", &frequency);
        if (history != 1) {
            break;
        }
        if (frequency > maxfrequency) {
            maxfrequency = frequency;
        }
    }
    fclose(fp);
    return maxfrequency;
}

static std::vector<int> sortCPUIDByMaxFrequency(int maxNumbers) {
    const int cpuNumbers = getNumberOfCPU();
    if (cpuNumbers == 0) {
        return {};
    }
    std::vector<int> cpuIDs;
    std::vector<std::pair<int, int>> cpusFrequency;
    cpusFrequency.resize(cpuNumbers);
    for (int i = 0; i < cpuNumbers; ++i) {
        int frequency           = getCPUMaxFreqKHz(i);
        cpusFrequency[i].first  = frequency;
        cpusFrequency[i].second = i;
    }
    maxNumbers = std::min(maxNumbers, cpuNumbers);
    std::sort(cpusFrequency.rbegin(), cpusFrequency.rend());
    cpuIDs.resize(maxNumbers);
    for (int i = 0; i < maxNumbers; ++i) {
        cpuIDs[i] = cpusFrequency[i].second;
    }
    // FUNC_PRINT(cpusFrequency[0].first);
    return cpuIDs;
}

static int setSchedAffinity(const std::vector<int>& cpuIDs) {
#define __NCPUBITS (8 * sizeof(unsigned long))
    typedef struct {
        unsigned long __bits[CPU_SETSIZE / __NCPUBITS];
    } cpu_set_t;

    // set affinity for thread

    pid_t pid = gettid();
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int i = 1; i < (int)cpuIDs.size(); i++) {
        CPU_SET(cpuIDs[i], &mask);
    }

    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
    if (syscallret) {
        MNN_PRINT("syscall error %d\n", syscallret);
        return -1;
    }

    return 0;
}

#endif // arch
ThreadPool::ThreadPool(int numberThread) {
    mNumberThread = numberThread;
    mActiveCount.store(0, std::memory_order_relaxed);
#ifdef MNN_USE_DYNAMIC_WORK_INDEX
    mTaskAvailable.reset(new std::atomic_bool[MNN_THREAD_POOL_MAX_TASKS]);
#else
    mTaskAvailable.resize(MNN_THREAD_POOL_MAX_TASKS);
#endif
    mTasks.resize(MNN_THREAD_POOL_MAX_TASKS);
#ifdef DEBUG_TIMES
    g_times.resize(numberThread);
#endif
    for (int t = 0; t < mTasks.size(); ++t) {
#ifdef MNN_USE_DYNAMIC_WORK_INDEX
        mTaskAvailable[t].store(true, std::memory_order_relaxed);
#else
        mTaskAvailable[t] = true;
#endif
        mTasks[t].second.reserve(mNumberThread);
        for (int i = 0; i < mNumberThread; ++i) {
            mTasks[t].second.emplace_back(new std::atomic_bool{false});
        }
    }
#ifdef MNN_THREAD_LOCK_CPU
    std::vector<int> sortedCPUIDs = sortCPUIDByMaxFrequency(numberThread);
#endif
    for (int i = 1; i < mNumberThread; ++i) {
        int threadIndex = i;
#ifdef MNN_THREAD_LOCK_CPU
        mWorkers.emplace_back([this, sortedCPUIDs, threadIndex]() {
#else
        mWorkers.emplace_back([this, threadIndex]() {
#endif
#ifdef MNN_THREAD_LOCK_CPU
            int res = setSchedAffinity(sortedCPUIDs);
#endif
#ifdef DEBUG_TIMES
            auto &record = g_times[threadIndex];
#endif
            while (mActiveCount.load(std::memory_order_relaxed) >= 0) {
#ifdef DEBUG_TIMES
                auto t1 = std::chrono::high_resolution_clock::now();
#endif
                while (mActiveCount.load(std::memory_order_relaxed) > 0) {
                    for (int i = 0; i < MNN_THREAD_POOL_MAX_TASKS; ++i) {
                        if (mTasks[i].second[threadIndex]->load(std::memory_order_relaxed)) {
                            std::atomic_thread_fence(std::memory_order_acquire);
#ifdef DEBUG_TIMES
                            auto t2 = std::chrono::high_resolution_clock::now();
#endif
                            mTasks[i].first.first(threadIndex);
                            mTasks[i].second[threadIndex]->store(false, std::memory_order_release);
#ifdef DEBUG_TIMES
                            std::get<1>(record) += std::chrono::high_resolution_clock::now() - t2;
                            std::get<0>(record)++;
#endif
                        }
                    }
                    std::this_thread::yield();
                }
#ifdef DEBUG_TIMES
                std::get<2>(record) = std::chrono::high_resolution_clock::now() - t1;
#endif
                std::unique_lock<std::mutex> _l(mQueueMutex);
                mCondition.wait(_l, [this] { return mActiveCount.load(std::memory_order_acquire) > 0; });
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    mActiveCount.store(-1, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> _l(mQueueMutex);
      mCondition.notify_all();
    }
    for (auto& worker : mWorkers) {
        worker.join();
    }
    for (auto& task : mTasks) {
        for (auto c : task.second) {
            delete c;
        }
    }
}

int ThreadPool::acquireWorkIndex() {
    if (nullptr == gInstance) {
        return INVALID_WORK_INDEX;
    }
#ifdef MNN_USE_DYNAMIC_WORK_INDEX
    return DYNAMIC_WORK_INDEX;
#else
    std::lock_guard<std::mutex> _l(gInstance->mQueueMutex);
    for (int i = 0; i < MNN_THREAD_POOL_MAX_TASKS; ++i) {
        if (gInstance->mTaskAvailable[i]) {
            gInstance->mTaskAvailable[i] = false;
            return i;
        }
    }
    return INVALID_WORK_INDEX;
#endif
}
void ThreadPool::releaseWorkIndex(int index) {
    if (nullptr == gInstance) {
        return;
    }
#ifdef MNN_USE_DYNAMIC_WORK_INDEX
    return;
#else
    if (index < 0 || index >= MNN_THREAD_POOL_MAX_TASKS) {
        return;
    }
    std::lock_guard<std::mutex> _l(gInstance->mQueueMutex);
    gInstance->mTaskAvailable[index] = true;
#endif
}

int ThreadPool::active(int acquiredWorkIndex) {
    if (nullptr == gInstance) {
        return acquiredWorkIndex;
    }
    int workIndex       = acquiredWorkIndex;
#ifdef MNN_USE_DYNAMIC_WORK_INDEX
    if (gInstance->mActiveCount.load(std::memory_order_relaxed) < MNN_THREAD_POOL_MAX_TASKS) {
        std::atomic_thread_fence(std::memory_order_acquire);
        for (int i = 0; i < MNN_THREAD_POOL_MAX_TASKS; ++i) {
            auto& task = gInstance->mTaskAvailable[i];
            if (task.load(std::memory_order_relaxed) && task.exchange(false, std::memory_order_relaxed)) {
                workIndex = i;
                break;
            }
        }
    }
    if (workIndex == acquiredWorkIndex) {
        return INVALID_WORK_INDEX;
    }
#endif
    gInstance->mActiveCount.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> _l(gInstance->mQueueMutex);
    gInstance->mCondition.notify_all();
    return workIndex;
}
int ThreadPool::deactive(int workIndexInUse) {
    if (nullptr == gInstance) {
        return workIndexInUse;
    }
    int newWorkIndex = workIndexInUse;
#ifdef MNN_USE_DYNAMIC_WORK_INDEX
    if (workIndexInUse < 0) {
        return newWorkIndex;
    }
    newWorkIndex = DYNAMIC_WORK_INDEX;
    gInstance->mTaskAvailable[workIndexInUse].store(true, std::memory_order_relaxed);
#endif
    gInstance->mActiveCount.fetch_sub(1, std::memory_order_relaxed);
    return newWorkIndex;
}

void ThreadPool::enqueue(TASK&& task, int index) {
    if (1 >= task.second || 0 > index) {
        for (int i = 0; i < task.second; ++i) {
            task.first(i);
        }
        return;
    }
    MNN_ASSERT(nullptr != gInstance);
    gInstance->enqueueInternal(std::move(task), index);
}

void ThreadPool::enqueueInternal(TASK&& task, int index) {
#ifdef DEBUG_TIMES
    auto t0 = std::chrono::high_resolution_clock::now();
#endif
    /**
    if (mActiveCount.load(std::memory_order_relaxed) == 0) {
        for (int i = 0; i < task.second; ++i) {
            task.first(i);
        }
#ifdef DEBUG_TIMES
        g_steps[g_cur_step].first += std::chrono::high_resolution_clock::now() - t0;
#endif
        return;
    }
    // */
    int workSize = task.second;
    if (workSize > mNumberThread) {
        mTasks[index].first = std::make_pair(
            [workSize, &task, this](int tId) {
                for (int v = tId; v < workSize; v += mNumberThread) {
                    task.first(v);
                }
            },
            mNumberThread);
        workSize = mNumberThread;
    } else {
        mTasks[index].first = std::move(task);
    }
    {
        for (int i = 1; i < workSize; ++i) {
            mTasks[index].second[i]->store(true, std::memory_order_relaxed);
        }
    }
    std::atomic_thread_fence(std::memory_order_acq_rel);
    mTasks[index].first.first(0);
#ifdef DEBUG_TIMES
    std::get<1>(g_times[0]) += std::chrono::high_resolution_clock::now() - t0;
    std::get<0>(g_times[0])++;
#endif
    bool complete = true;
    do {
        std::this_thread::yield();
        complete = true;
        for (int i = 1; i < workSize; ++i) {
            if (mTasks[index].second[i]->load(std::memory_order_relaxed)) {
                complete = false;
                break;
            }
        }
        // FUNC_PRINT(notComplete);
    } while (!complete);
#ifdef DEBUG_TIMES
    std::get<2>(g_times[0]) += std::chrono::high_resolution_clock::now() - t0;
    g_steps[g_cur_step].first += std::chrono::high_resolution_clock::now() - t0;
#endif
    std::atomic_thread_fence(std::memory_order_acquire);
}
} // namespace MNN
#endif
