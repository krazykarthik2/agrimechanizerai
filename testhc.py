import serial
import time


# HC-05 default baud rate is usually 9600
# Pins: TX (8), RX (10) -> /dev/ttyTHS1
port = serial.Serial("/dev/ttyTHS1", baudrate=9600, timeout=1)

print("--- Bluetooth Test Started ---")
print("Make sure your phone is connected to the HC-05 app.")

try:
    while True:
        # 1. Send data to the phone
        message = "Jetson says Hello!\r\n"
        port.write(message.encode())
        print("Sent to Phone: Jetson says Hello!")

        # 2. Check if phone sent anything back
        if port.in_waiting > 0:
            incoming = port.readline().decode('utf-8').strip()
            print(f"Received from Phone: {incoming}")

        time.sleep(2)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    port.close()
