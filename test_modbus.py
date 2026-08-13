from pymodbus.client import ModbusTcpClient
import time 
client = ModbusTcpClient('10.17.0.41', port=502)
coil=[1280, 1281, 1282, 1283, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303]
client.connect()
client.write_coil(address=12, value=True)
time.sleep(10)  
client.close()