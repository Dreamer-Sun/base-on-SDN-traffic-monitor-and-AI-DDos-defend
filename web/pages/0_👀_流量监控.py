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

	# 先发一次请求 获取路由等基本信息 设置选项值
	df = GetDataFrame()
	option1 = st.selectbox('请选择路由器编号', pd.unique(df["路由器"]))
	
	print("op", option1)
	if option1:
		option2 = st.selectbox('请选择路由器端口', pd.unique(df[df["路由器"]==option1]["端口"]))
	print(option1, option2)
	lastdata = df[(df['路由器'] == option1) & (df['端口'] == option2)].iloc[0]
	# print("lastdata", lastdata)
	# print("lastdatas", lastdata["接收数据包数(个)"].item())
	# creating a single-element container.
	lasttime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
	lastoption1 = option1
	lastoption2 = option2
	placeholder = st.empty()
	# linechart1 = [[lasttime, lasttime, lasttime, lasttime], ["接收数据包数(个)", "接收错误数据包数(个)", "发送数据包数(个)", "发送错误数据包数(个)"], [lastdata["接收数据包数(个)"].item(), lastdata["接收错误数据包数(个)"].item(), lastdata["发送数据包数(个)"].item(), lastdata["发送错误数据包数(个)"].item()]]
	# linechart2 = [[lasttime, lasttime], ["接收数据量(byte)", "发送数据量(byte)"], [lastdata["接收数据量(byte)"].item(), lastdata["发送数据量(byte)"].item()]]
	linechart1 = [[lasttime, "接收数据包数(个)", lastdata["接收数据包数(个)"].item()], [lasttime, "接收错误数据包数(个)", lastdata["接收错误数据包数(个)"].item()], [lasttime, "发送数据包数(个)", lastdata["发送数据包数(个)"].item()], [lasttime, "发送错误数据包数(个)", lastdata["发送错误数据包数(个)"].item()]]
	linechart2 = [[lasttime, "接收数据量(byte)", lastdata["接收数据量(byte)"].item()], [lasttime, "发送数据量(byte)", lastdata["发送数据量(byte)"].item()]]
	linechartcolumns = ["时间", "类型", "数据"]
	while True:
		df = GetDataFrame()
		with placeholder.container():
			st.caption('上一次更新时间:' + lasttime)
			st.caption('当前更新时间:' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))	
			lasttime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
			
			# 监控端口
			st.markdown("### 单独监控")
			st.caption("路由器ID: " + str(option1) )
			st.caption("路由器端口: " + str(option2) )
			
			if option1 and option2:
				# create six columns
				kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
				nowdata = df[(df['路由器'] == option1) & (df['端口'] == option2)].iloc[0]
				# fill in those three columns with respective metrics or KPIs
				# print("nowdata", nowdata, "11", nowdata["接收数据包数(个)"][0])
				kpi1.metric(
					label="接收数据包个数",
					value=nowdata["接收数据包数(个)"].item(),
					delta=nowdata["接收数据包数(个)"].item() - lastdata["接收数据包数(个)"].item(),
				)

				kpi2.metric(
					label="接收数据量(byte)",
					value=nowdata["接收数据量(byte)"].item(),
					delta=nowdata["接收数据量(byte)"].item() - lastdata["接收数据量(byte)"].item(),
				)

				kpi3.metric(
					label="接收错误数据包数(个)",
					value=nowdata["接收错误数据包数(个)"].item(),
					delta=nowdata["接收错误数据包数(个)"].item() - lastdata["接收错误数据包数(个)"].item(),
				)

				kpi4.metric(
					label="发送数据包数(个)",
					value=nowdata["发送数据包数(个)"].item(),
					delta=nowdata["发送数据包数(个)"].item() - lastdata["发送数据包数(个)"].item(),
				)

				kpi5.metric(
					label="发送数据量(byte)",
					value=nowdata["发送数据量(byte)"].item(),
					delta=nowdata["发送数据量(byte)"].item() - lastdata["发送数据量(byte)"].item(),
				)

				kpi6.metric(
					label="发送错误数据包数(个)",
					value=nowdata["发送错误数据包数(个)"].item(),
					delta=nowdata["发送错误数据包数(个)"].item() - lastdata["发送错误数据包数(个)"].item(),
				)
				# 绘制监控端口流量折线图
				# 如果切换端口了清空折线图数据
				if option1 != lastoption1 or option2 != lastoption2:
					linechart1 = [[lasttime, "接收数据包数(个)",nowdata["接收数据包数(个)"].item()], [lasttime, "接收错误数据包数(个)", nowdata["接收错误数据包数(个)"].item()], [lasttime, "发送数据包数(个)", nowdata["发送数据包数(个)"].item()], [lasttime, "发送错误数据包数(个)", nowdata["发送错误数据包数(个)"].item()]]
					linechart2 = [[lasttime, "接收数据量(byte)", nowdata["接收数据量(byte)"].item()], [lasttime, "发送数据量(byte)", nowdata["发送数据量(byte)"].item()]]
				st.markdown('### 监控流量图')
				lich1, lich2 = st.columns(2)
				if nowdata["接收数据包数(个)"].item() != lastdata["接收数据包数(个)"].item():
					linechart1.append([lasttime, "接收数据包数(个)", nowdata["接收数据包数(个)"].item()])

				if nowdata["接收错误数据包数(个)"].item() != lastdata["接收错误数据包数(个)"].item():
					linechart1.append([lasttime, "接收错误数据包数(个)",nowdata["接收错误数据包数(个)"].item()])

				if nowdata["发送数据包数(个)"].item() != lastdata["发送数据包数(个)"].item():
					linechart1.append([lasttime, "发送数据包数(个)", nowdata["发送数据包数(个)"].item()])

				if nowdata["发送错误数据包数(个)"].item() != lastdata["发送错误数据包数(个)"].item():
					linechart1.append([lasttime, "发送错误数据包数(个)", nowdata["发送错误数据包数(个)"].item()])

				if nowdata["接收数据量(byte)"].item() != lastdata["接收数据量(byte)"].item():
					linechart2.append([lasttime, "接收数据量(byte)", nowdata["接收数据量(byte)"].item()])

				if nowdata["发送数据量(byte)"].item() != lastdata["发送数据量(byte)"].item():
					linechart2.append([lasttime, "发送数据量(byte)", nowdata["发送数据量(byte)"].item()])	
				print(linechart1)
				lf1 = pd.DataFrame(linechart1, columns=linechartcolumns)
				lf2 = pd.DataFrame(linechart2, columns=linechartcolumns)

				with lich1:
					st.markdown("## 数据包")
					fig = px.line(
						data_frame=lf1, y="数据", x="时间", color="类型"
					)
					st.write(fig)
				
				with lich2:
					st.markdown("## 数据量")
					fig = px.line(
						data_frame=lf2, y="数据", x="时间", color="类型"
					)
					st.write(fig)

				lastdata = nowdata
			# 绘制流量监控表格
			st.markdown("### 流量监控")
			st.dataframe(df)
			time.sleep(1)

def GetDataFrame():
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
		datalist = [["路由器", "端口", "接收数据包数(个)", "接收数据量(byte)", "接收错误数据包数(个)", "发送数据包数(个)", "发送数据量(byte)", "发送错误数据包数(个)"]]
		for p in data:
			for q in data[p]:
				print("q", q)
				temp = [p, q["port"], q["rx-pkts"], q["rx-bytes"], q["rx-error"], q["tx-pkts"], q["tx-bytes"], q["tx-error"]]
				datalist.append(temp)
		df = pd.DataFrame(datalist[1:],columns=datalist[0])
		print("df:", df)
		return df


if __name__ == '__main__':
	main()