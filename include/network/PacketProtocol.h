#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace blade::net {

//模块:应用层协议包类型
//IMAGE_JPEG:发送可视化后的JPG图像
//META_JSON:发送当前帧检测框、类别、置信度等文本信息
//HEARTBEAT_PING/PONG:心跳包,用于判断客户端是否断开
//COMMAND:Qt客户端发控制命令,例如start/stop/quit
//ERROR_TEXT:发送错误信息

enum class PacketType : uint16_t {
    IMAGE_JPEG = 1,
    META_JSON = 2,
    HEARTBEAT_PING = 3, //心跳请求包,用于检测连接是否还活着
    HEARTBEAT_PONG = 4, //心跳响应包,收到PING后回复PONG
    COMMAND = 5,
    ERROR_TEXT = 6
};

constexpr uint32_t kPacketMagic = 0x424C4144; // ASCII: BLAD
constexpr uint16_t kPacketVersion = 1;        //协议版本号,用于客户端和服务端兼容检查
constexpr size_t kPacketHeaderSize = 36;

// #pragma pack(push, n)：将当前的对齐设置压入内部栈中，然后设置新对齐为 n字节。
// #pragma pack(pop)：从栈顶弹出上一次保存的设置，恢复旧的对齐方式。
// 保证PacketHeader在内存中的大小严格等于36字节
#pragma pack(push, 1)
struct PacketHeader {
    uint32_t magic = kPacketMagic;
    uint16_t version = kPacketVersion;
    uint16_t type = 0;                  //数据包类型,对应PacketType
    uint32_t flags = 0;                 //预留标志位,后续可用于压缩、加密、关键帧等扩展
    uint64_t sequence = 0;              //包序号,用于调试丢包、乱序、统计帧率
    uint64_t timestampMs = 0;
    uint32_t payloadSize = 0;           //payload数据长度,接收端根据它继续读取指定字节数
    uint32_t crc32 = 0;                 //校验码
};
#pragma pack(pop)

struct Packet {
    PacketHeader header;
    std::vector<uint8_t> payload;       //协议数据体,存放真正要传输的数据
};

uint64_t nowMs();
//作用:根据payload数据计算校验码
uint32_t crc32(const uint8_t* data, size_t size);

//作用:把PacketHeader结构体转换成36字节的二进制数组
std::vector<uint8_t> encodeHeader(const PacketHeader& header);

//作用:把接收到的36字节二进制数据解析成PacketHeader结构体
bool decodeHeader(const uint8_t* data, PacketHeader& header);

//作用:根据包类型、包序号和payload自动生成Packet
Packet makePacket(PacketType type, uint64_t sequence, const std::vector<uint8_t>& payload);

//作用:把std::string文本转换成字节数组后再调用makePacket
Packet makeTextPacket(PacketType type, uint64_t sequence, const std::string& text);

//作用:把PacketType转换成可读字符串
std::string packetTypeName(PacketType type);

} // namespace blade::net
