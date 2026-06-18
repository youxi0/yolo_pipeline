#include "network/TcpServer.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace blade::net {

TcpServer::TcpServer(uint16_t port)
    : port_(port) {}

TcpServer::~TcpServer() {
    stop();
}

bool TcpServer::startListen() {
    listenFd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd_ < 0) {
        std::cerr << "socket failed: " << std::strerror(errno) << std::endl;
        return false;
    }

    // 服务器程序经常需要快速重启。如果不设置 SO_REUSEADDR，当服务器主动关闭连接或异常退出后，
    // 操作系统可能将端口保留在 TIME_WAIT 状态（通常持续几十秒），导致再次启动时绑定失败，
    // 报错 “Address already in use”。启用该选项后，内核允许新的套接字重用该端口，从而服务器可以立即重启。
    int opt = 1;
    ::setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);

    if (::bind(listenFd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "bind failed: " << std::strerror(errno) << std::endl;
        stop();
        return false;
    }

    if (::listen(listenFd_, 1) < 0) {
        std::cerr << "listen failed: " << std::strerror(errno) << std::endl;
        stop();
        return false;
    }

    TCP_running_ = true;
    
    std::cout << "TCP server listening on port " << port_ << std::endl;
    return true;
}

bool TcpServer::waitForClient() {
    if (listenFd_ < 0) {
        return false;
    }

    sockaddr_in clientAddr{};
    socklen_t len = sizeof(clientAddr);
    int fd = ::accept(listenFd_, reinterpret_cast<sockaddr*>(&clientAddr), &len);

    if (fd < 0) {
        if (TCP_running_) {
            std::cerr << "accept failed: " << std::strerror(errno) << std::endl;
        }
        return false;
    }

    char ip[64] = {0};
    ::inet_ntop(AF_INET, &clientAddr.sin_addr, ip, sizeof(ip));

    {
        std::lock_guard<std::mutex> lock(clientMutex_);
        if (clientFd_ >= 0) {
            ::close(clientFd_);
        }
        clientFd_ = fd;
    }

    std::cout << "Qt client connected: " << ip << ":" << ntohs(clientAddr.sin_port) << std::endl;
    return true;
}

bool TcpServer::sendPacket(const Packet& packet) {
    std::lock_guard<std::mutex> sendLock(sendMutex_);

    if (!isClientConnected()) {
        return false;
    }

    auto headerBytes = encodeHeader(packet.header);

    if (!writeExact(headerBytes.data(), headerBytes.size())) {
        closeClient();
        return false;
    }

    if (!packet.payload.empty() && !writeExact(packet.payload.data(), packet.payload.size())) {
        closeClient();
        return false;
    }

    return true;
}

bool TcpServer::receivePacket(Packet& packet) {
    if (!isClientConnected()) {
        return false;
    }

    uint8_t headerBytes[kPacketHeaderSize];
    if (!readExact(headerBytes, kPacketHeaderSize)) {
        closeClient();
        return false;
    }

    if (!decodeHeader(headerBytes, packet.header)) {
        std::cerr << "Invalid packet header." << std::endl;
        closeClient();
        return false;
    }

    packet.payload.resize(packet.header.payloadSize);

    if (packet.header.payloadSize > 0) {
        if (!readExact(packet.payload.data(), packet.payload.size())) {
            closeClient();
            return false;
        }

        uint32_t got = crc32(packet.payload.data(), packet.payload.size());
        if (got != packet.header.crc32) {
            std::cerr << "CRC mismatch. packet=" << packet.header.crc32 << " got=" << got << std::endl;
            closeClient();
            return false;
        }
    }

    return true;
}

bool TcpServer::isClientConnected() const {
    std::lock_guard<std::mutex> lock(clientMutex_);
    return clientFd_ >= 0;
}

void TcpServer::closeClient() {
    std::lock_guard<std::mutex> lock(clientMutex_);
    if (clientFd_ >= 0) {
        ::shutdown(clientFd_, SHUT_RDWR);
        ::close(clientFd_);
        clientFd_ = -1;
        std::cout << "Qt client disconnected." << std::endl;
    }
}

void TcpServer::stop() {
    TCP_running_ = false;
    closeClient();

    if (listenFd_ >= 0) {
        ::shutdown(listenFd_, SHUT_RDWR);
        ::close(listenFd_);
        listenFd_ = -1;
    }
}

bool TcpServer::writeExact(const uint8_t* data, size_t size) {
    size_t sent = 0;

    while (sent < size) {
        int fd = -1;
        {
            std::lock_guard<std::mutex> lock(clientMutex_);
            fd = clientFd_;
        }

        if (fd < 0) {
            return false;
        }

        ssize_t n = ::send(fd, data + sent, size - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            std::cerr << "send failed: " << std::strerror(errno) << std::endl;
            return false;
        }

        if (n == 0) {
            return false;
        }

        sent += static_cast<size_t>(n);
    }

    return true;
}

bool TcpServer::readExact(uint8_t* data, size_t size) {
    size_t received = 0;

    while (received < size) {
        int fd = -1;
        {
            std::lock_guard<std::mutex> lock(clientMutex_);
            fd = clientFd_;
        }

        if (fd < 0) {
            return false;
        }

        ssize_t n = ::recv(fd, data + received, size - received, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            std::cerr << "recv failed: " << std::strerror(errno) << std::endl;
            return false;
        }

        if (n == 0) {
            return false;
        }

        received += static_cast<size_t>(n);
    }

    return true;
}

} // namespace blade::net
