#include "network/PacketProtocol.h"

#include <chrono>
#include <cstring>

namespace blade::net {

namespace {

void writeU16(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>(v & 0xFF));
}

void writeU32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>(v & 0xFF));
}

void writeU64(std::vector<uint8_t>& out, uint64_t v) {
    for (int i = 7; i >= 0; --i) {
        out.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFF));
    }
}

uint16_t readU16(const uint8_t* data) {
    return static_cast<uint16_t>((static_cast<uint16_t>(data[0]) << 8) | data[1]);
}

uint32_t readU32(const uint8_t* data) {
    return (static_cast<uint32_t>(data[0]) << 24) |
           (static_cast<uint32_t>(data[1]) << 16) |
           (static_cast<uint32_t>(data[2]) << 8) |
           static_cast<uint32_t>(data[3]);
}

uint64_t readU64(const uint8_t* data) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v = (v << 8) | static_cast<uint64_t>(data[i]);
    }
    return v;
}

} // namespace

uint64_t nowMs() {
    using namespace std::chrono;
    auto now = steady_clock::now();
    return static_cast<uint64_t>(duration_cast<milliseconds>(now.time_since_epoch()).count());
}

uint32_t crc32(const uint8_t* data, size_t size) {
    uint32_t crc = 0xFFFFFFFFu;

    for (size_t i = 0; i < size; ++i) {
        crc ^= static_cast<uint32_t>(data[i]);
        for (int bit = 0; bit < 8; ++bit) {
            uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }

    return ~crc;
}

std::vector<uint8_t> encodeHeader(const PacketHeader& header) {
    std::vector<uint8_t> out;
    out.reserve(kPacketHeaderSize);

    writeU32(out, header.magic);
    writeU16(out, header.version);
    writeU16(out, header.type);
    writeU32(out, header.flags);
    writeU64(out, header.sequence);
    writeU64(out, header.timestampMs);
    writeU32(out, header.payloadSize);
    writeU32(out, header.crc32);

    return out;
}

bool decodeHeader(const uint8_t* data, PacketHeader& header) {
    header.magic = readU32(data + 0);
    header.version = readU16(data + 4);
    header.type = readU16(data + 6);
    header.flags = readU32(data + 8);
    header.sequence = readU64(data + 12);
    header.timestampMs = readU64(data + 20);
    header.payloadSize = readU32(data + 28);
    header.crc32 = readU32(data + 32);

    return header.magic == kPacketMagic && header.version == kPacketVersion;
}

Packet makePacket(PacketType type, uint64_t sequence, const std::vector<uint8_t>& payload) {
    Packet packet;
    packet.header.magic = kPacketMagic;
    packet.header.version = kPacketVersion;
    packet.header.type = static_cast<uint16_t>(type);
    packet.header.flags = 0;
    packet.header.sequence = sequence;
    packet.header.timestampMs = nowMs();
    packet.header.payloadSize = static_cast<uint32_t>(payload.size());
    packet.header.crc32 = payload.empty() ? 0 : crc32(payload.data(), payload.size());
    packet.payload = payload;
    return packet;
}

Packet makeTextPacket(PacketType type, uint64_t sequence, const std::string& text) {
    std::vector<uint8_t> payload(text.begin(), text.end());
    return makePacket(type, sequence, payload);
}

std::string packetTypeName(PacketType type) {
    switch (type) {
    case PacketType::IMAGE_JPEG:
        return "IMAGE_JPEG";
    case PacketType::META_JSON:
        return "META_JSON";
    case PacketType::HEARTBEAT_PING:
        return "HEARTBEAT_PING";
    case PacketType::HEARTBEAT_PONG:
        return "HEARTBEAT_PONG";
    case PacketType::COMMAND:
        return "COMMAND";
    case PacketType::ERROR_TEXT:
        return "ERROR_TEXT";
    default:
        return "UNKNOWN";
    }
}

} // namespace blade::net
