import serial
import time

# /dev/ttyTHS1 is the J41 Header UART
# Baud rate is set to 9600 (common standard)
serial_port = serial.Serial(
    port="/dev/ttyTHS1",
    baudrate=9600,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)

try:
    print("Serial port opened. Sending data...")
    while True:
        message = "A 50\r\nB 180"
        serial_port.write(message.encode())
        print(f"Sent: {message.strip()}")
        time.sleep(1)

except KeyboardInterrupt:
    print("\nExiting Program")

finally:
    serial_port.close()
