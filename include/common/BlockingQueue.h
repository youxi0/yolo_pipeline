#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>

//模块:线程安全阻塞队列
//作用:不同线程之间传递FrameData、推理结果、待发送数据
//特点:队列满时生产者阻塞,队列空时消费者阻塞,stop后所有阻塞线程都会退出

template <typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t maxSize = 8)
        : maxSize_(maxSize) {}

    bool push(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        notFull_.wait(lock, [&]() { return stopped_ || queue_.size() < maxSize_; });

        if (stopped_) {
            return false;
        }

        queue_.push_back(item);
        notEmpty_.notify_one();
        return true;
    }

    bool push(T&& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        notFull_.wait(lock, [&]() { return stopped_ || queue_.size() < maxSize_; });

        if (stopped_) {
            return false;
        }

        queue_.push_back(std::move(item));
        notEmpty_.notify_one();
        return true;
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        notEmpty_.wait(lock, [&]() { return stopped_ || !queue_.empty(); });

        if (queue_.empty()) {
            return false;
        }

        item = std::move(queue_.front());
        queue_.pop_front();
        notFull_.notify_one();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        notEmpty_.notify_all();
        notFull_.notify_all();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear();
        notFull_.notify_all();
    }

private:
    size_t maxSize_ = 8;
    bool stopped_ = false;
    std::deque<T> queue_;
    std::mutex mutex_;
    std::condition_variable notEmpty_;
    std::condition_variable notFull_;
};
