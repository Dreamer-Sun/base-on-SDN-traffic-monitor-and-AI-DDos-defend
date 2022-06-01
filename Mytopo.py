from mininet.net import Mininet
from mininet.cli import CLI
from mininet.topo import Topo
from mininet.node import OVSSwitch, OVSController, Controller, RemoteController


net = Mininet()
c0 = net.addController('control', controller=RemoteController, ip='127.0.0.1', port=6653)

h0 = net.addHost('h0')
h1 = net.addHost('h1')
h2 = net.addHost('h2')
s0 = net.addSwitch('s0')
net.addLink(h0, s0)
net.addLink(h1, s0)
net.addLink(h2, s0)
h0.setIP('192.168.1.1', 24)
h1.setIP('192.168.1.2', 24)
h2.setIP('192.168.1.3', 24)
net.start()
CLI(net)
net.stop()

