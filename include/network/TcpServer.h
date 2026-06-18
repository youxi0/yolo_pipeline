#pragma once

#include "network/PacketProtocol.h"

#include <atomic>
#include <mutex>
#include <string>

namespace blade::net {

//模块:TcpServer负责监听端口、等待Qt客户端连接、收发自定义协议Packet
//说明:这里按一个服务端对应一个Qt客户端设计,适合工业检测界面显示

class TcpServer {
public:
    explicit TcpServer(uint16_t port);
    ~TcpServer();

    bool startListen();
    bool waitForClient();
    bool sendPacket(const Packet& packet);
    bool receivePacket(Packet& packet);
    bool isClientConnected() const;
    void closeClient();
    void stop();

private:
    bool writeExact(const uint8_t* data, size_t size);
    bool readExact(uint8_t* data, size_t size);

private:
    uint16_t port_ = 0;
    int listenFd_ = -1;
    int clientFd_ = -1;
    std::atomic<bool> TCP_running_{false};
    mutable std::mutex clientMutex_;
    std::mutex sendMutex_;
};

} // namespace blade::net
