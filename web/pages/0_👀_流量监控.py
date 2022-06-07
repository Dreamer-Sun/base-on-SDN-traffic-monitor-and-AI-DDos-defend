import streamlit as st
import time
import datetime
import requests
import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development


def main():
	st.set_page_config(page_title = 'monitor_web', page_icon=":mag:", layout="wide")
	st.title('流量监控&智能DDos防御 :sunglasses:')
	# creating a single-element container.
	placeholder = st.empty()
	lasttime = ""
	while True:
		# 获取数据
		url ='http://0.0.0.0:8080/monitor/mactable/getinfo'
		response = requests.get(url)
		print(response.json())
		data = response.json()
		# 重新整理数据整理为：
		# [1, "23165156156", 1, 14, 1076, 0, 14, 5575, 0]
		# [
		# 	"id": "213453650875718",
		# 	"port": 1,
		# 	"rx-pkts": 14,
		# 	"rx-bytes": 1076,
		# 	"rx-error": 0,
		# 	"tx-pkts": 14,
		# 	"tx-bytes": 5575,
		# 	"tx-error": 0
		# ]
		# datalist = [["id", "port", "rx-pkts", 	"rx-bytes", "rx-error", 	  "tx-pkts", "tx-bytes", "tx-error"]]
		datalist = [["路由器", "端口", "接收数据包数", "接收数据量", "接收错误数据包数", "发送数据包数", "发送数据量", "发送错误数据包数"]]
		for p in data:
			for q in data[p]:
				print("q", q)
				temp = [p, q["port"], q["rx-pkts"], q["rx-bytes"], q["rx-error"], q["tx-pkts"], q["tx-bytes"], q["tx-error"]]
				datalist.append(temp)
		df = pd.DataFrame(datalist[1:],columns=datalist[0])
		print("df:", df)

		# 绘制实时变化表格
		# table.dataframe(df, 2000, 2000)

		with placeholder.container():
			st.subheader('上一次更新时间:' + lasttime)
			st.subheader('当前更新时间:' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
			lasttime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
			st.dataframe(df)
			time.sleep(1)




if __name__ == '__main__':
	main()