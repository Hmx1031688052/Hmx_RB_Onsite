import socket
import fcntl
import struct


def get_ip_address(ifname):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return socket.inet_ntoa(
            fcntl.ioctl(
                s.fileno(),
                0x8915,  # SIOCGIFADDR
                struct.pack('256s', bytes(ifname[:15], "utf-8")))[20:24])
    except Exception as e:
        pass
    finally:
        s.close()


if __name__ == "__main__":
    ip = get_ip_address('eno2')  # '192.168.0.110'
    print(ip)
