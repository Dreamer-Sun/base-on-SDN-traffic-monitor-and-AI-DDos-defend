import json

from ryu.base import app_manager
from ryu.app import simple_switch_13
from operator import attrgetter
from ryu.ofproto import ofproto_v1_3
from ryu.controller.handler import set_ev_cls
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER,DEAD_DISPATCHER 
from ryu.lib import hub
from ryu.app.wsgi import ControllerBase
from ryu.app.wsgi import Response
from ryu.app.wsgi import route
from ryu.app.wsgi import WSGIApplication

monitor_instance_name = 'monitor_api_app'
urlQuery = '/monitor/mactable/getinfo'

class Monitor(simple_switch_13.SimpleSwitch13):   #继承simple_switch_13的功能
	OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]	#set openflow version 1.3
	_CONTEXTS = {'wsgi': WSGIApplication}

	def __init__(self, *args, **kwargs):	  #初始化函数
		super(Monitor, self).__init__(*args, **kwargs)
		self.datapaths = {}   #初始化成员变量，用来存储数据
		self.monitor_thread = hub.spawn(self._monitor)   #用协程方法执行_monitor方法，这样其他方法可以被其他协程执行。 hub.spawn()创建协程
		self.monitor_info = {}
		wsgi = kwargs['wsgi']
		wsgi.register(Monitor_Controller,
						{monitor_instance_name: self})

	@set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
	#通过ryu.controller.handler.set_ev_cls装饰器（decorator）进行注册，在运行时，ryu控制器就能知道Monitor这个模块的函数_state_change_handler监听了一个事件
	def _state_change_handler(self, event):  #交换机状态发生变化后，让控制器数据于交换机一致
		datapath = event.datapath
		if event.state == MAIN_DISPATCHER:   # 在MAIN_DISPATCHER状态下，交换机处于上线状态
			if datapath.id not in self.datapaths:
				self.logger.debug('register datapath: %016x', datapath.id)    #十六进制输出
				self.datapaths[datapath.id] = datapath
			elif event.state == DEAD_DISPATCHER:
				if datapath.id in self.datapaths:
					self.logger.debug('unregister datapath %016x', datapath.id)
					del self.datapaths[datapath.id]

	#对交换机发送请求,获取终端信息
	def _monitor(self):
		while True:			#对已注册交换机发出统计信息获取请求每2秒无限地重复一次
			for dp in self.datapaths.values():
				self._request_stats(dp)
				self.monitor_info.setdefault(dp.id, [])
			hub.sleep(1)

	def _request_stats(self, datapath):
		self.logger.debug('send stats request: %016x', datapath.id)
		#定义openflow解析器
		ofproto = datapath.ofproto
		ofp_parser = datapath.ofproto_parser

		#发送流量状态请求，获取流量状态数据
		request = ofp_parser.OFPFlowStatsRequest(datapath)
		datapath.send_msg(request)
		#
		request = ofp_parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
		datapath.send_msg(request)

	@set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
	def _port_stats_reply_handler(self, event):
		body = event.msg.body   #消息体
		self.logger.info('datapath         port      '
                         'rx-pkts  rx-bytes rx-error '  
                         'tx-pkts  tx-bytes tx-error ')    # rx-pkts:receive packets tx-pks:transmit packets
		self.logger.info('---------------- -------- '
						'-------- -------- -------- '
						'-------- -------- -------- ')
		
		tmp_key = ('port', 'rx-pkts', 'rx-bytes', 'rx-error', 'tx-pkts', 'tx-bytes', 'tx-error')
		tmp_dict = dict.fromkeys(tmp_key)
		tmp_list = []
		for stat in sorted(body,key=attrgetter('port_no')):     #attrgetter：属性获取工具
			self.logger.info('%016x %8x %8d %8d %8d %8d %8d %8d',
							event.msg.datapath.id, stat.port_no,
							stat.rx_packets, stat.rx_bytes, stat.rx_errors,
							stat.tx_packets, stat.tx_bytes, stat.tx_errors)
			tmp_dict['port'] = stat.port_no
			tmp_dict['rx-pkts'] = stat.rx_packets
			tmp_dict['rx-bytes'] = stat.rx_bytes
			tmp_dict['rx-error'] = stat.rx_errors
			tmp_dict['tx-pkts'] = stat.tx_packets
			tmp_dict['tx-bytes'] = stat.tx_bytes
			tmp_dict['tx-error'] = stat.tx_errors
			tmp_list.append(tmp_dict.copy())
			self.monitor_info[event.msg.datapath.id] = tmp_list
		print(self.monitor_info)
	@set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
	def _flow_stats_reply_handler(self, event):
		body = event.msg.body
		self.logger.info('datapath         '
						'in-port  eth-dst           '
						'out-port packets  bytes')
		self.logger.info('---------------- '
						'-------- ----------------- '
						'-------- -------- --------')
		for stat in sorted([flow for flow in body if flow.priority==1]
							,key=lambda flow:(flow.match['in_port'],flow.match['eth_dst'])):
			self.logger.info('%016x %8x %17s %8x %8d %8d',
							event.msg.datapath.id,stat.match['in_port'],
							stat.match['eth_dst'],stat.instructions[0].actions[0].port,
 							stat.packet_count,stat.byte_count)

class Monitor_Controller(ControllerBase):

	def __init__(self, req, link, data, **config):
		super(Monitor_Controller, self).__init__(req, link, data, **config)
		self.monitor_app = data[monitor_instance_name]

	@route('monitor', urlQuery, methods=['GET'])
	def get_monitor_info(self, req, **kwargs):
		monitor_name = self.monitor_app

		if not monitor_name.monitor_info:
			return Response(status=404)
		body = json.dumps(monitor_name.monitor_info)
		return Response(content_type='application/json', body=body)
		
