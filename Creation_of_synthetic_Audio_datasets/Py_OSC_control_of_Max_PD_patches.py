import argparse
from pythonosc import udp_client
import random

# Parse command line arguments
parser = argparse.ArgumentParser(description='Send OSC messages to a Max MSP 8 patch')
parser.add_argument('--host', type=str, default='localhost',
                    help='the IP address or hostname of the OSC server (default: localhost)')
parser.add_argument('--port', type=int, default=8000,
                    help='the port number of the OSC server (default: 8000)')
args = parser.parse_args()

# Create OSC client
client = udp_client.SimpleUDPClient('127.0.0.1', args.port)

print(f'Started UDP client with Host: 127.0.0.1, and port: {args.port}')

# Send some OSC messages
client.send_message('avgRate', 90)
client.send_message('minRadius', 50)
client.send_message('maxRadius', 70)

print('Finished creating synthetic dataset')