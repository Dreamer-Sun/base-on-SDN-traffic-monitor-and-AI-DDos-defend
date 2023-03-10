import streamlit as st
import time
import datetime
import requests
import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # ð data web app development


def main():
	st.set_page_config(page_title = 'monitor_web', page_icon=":mag:", layout="wide")
	st.title('æµéçæ§&æºè½DDosé²å¾¡ :sunglasses:')

	# ååä¸æ¬¡è¯·æ± è·åè·¯ç±ç­åºæ¬ä¿¡æ¯ è®¾ç½®éé¡¹å¼
	df = GetDataFrame()
	option1 = st.selectbox('è¯·éæ©è·¯ç±å¨ç¼å·', pd.unique(df["è·¯ç±å¨"]))
	
	print("op", option1)
	if option1:
		option2 = st.selectbox('è¯·éæ©è·¯ç±å¨ç«¯å£', pd.unique(df[df["è·¯ç±å¨"]==option1]["ç«¯å£"]))
	print(option1, option2)
	lastdata = df[(df['è·¯ç±å¨'] == option1) & (df['ç«¯å£'] == option2)].iloc[0]
	# print("lastdata", lastdata)
	# print("lastdatas", lastdata["æ¥æ¶æ°æ®åæ°(ä¸ª)"].item())
	# creating a single-element container.
	lasttime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
	lastoption1 = option1
	lastoption2 = option2
	placeholder = st.empty()
	# linechart1 = [[lasttime, lasttime, lasttime, lasttime], ["æ¥æ¶æ°æ®åæ°(ä¸ª)", "æ¥æ¶éè¯¯æ°æ®åæ°(ä¸ª)", "åéæ°æ®åæ°(ä¸ª)", "åééè¯¯æ°æ®åæ°(ä¸ª)"], [lastdata["æ¥æ¶æ°æ®åæ°(ä¸ª)"].item(), lastdata["æ¥æ¶éè¯¯æ°æ®åæ°(ä¸ª)"].item(), lastdata["åéæ°æ®åæ°(ä¸ª)"].item(), lastdata["åééè¯¯æ°æ®åæ°(ä¸ª)"].item()]]
	# linechart2 = [[lasttime, lasttime], ["æ¥æ¶æ°æ®é(byte)", "åéæ°æ®é(byte)"], [lastdata["æ¥æ¶æ°æ®é(byte)"].item(), lastdata["åéæ°æ®é(byte)"].item()]]
	linechart1 = [[lasttime, "æ¥æ¶æ°æ®åæ°(ä¸ª)", lastdata["æ¥æ¶æ°æ®åæ°(ä¸ª)"].item()], [lasttime, "æ¥æ¶éè¯¯æ°æ®åæ°(ä¸ª)", lastdata["æ¥æ¶éè¯¯æ°æ®åæ°(ä¸ª)"].item()], [lasttime, "åéæ°æ®åæ°(ä¸ª)", lastdata["åéæ°æ®åæ°(ä¸ª)"].item()], [lasttime, "åééè¯¯æ°æ®åæ°(ä¸ª)", lastdata["åééè¯¯æ°æ®åæ°(ä¸ª)"].item()]]
	linechart2 = [[lasttime, "æ¥æ¶æ°æ®é(byte)", lastdata["æ¥æ¶æ°æ®é(byte)"].item()], [lasttime, "åéæ°æ®é(byte)", lastdata["åéæ°æ®é(byte)"].item()]]
	linechartcolumns = ["æ¶é´", "ç±»å", "æ°æ®"]
	while True:
		df = GetDataFrame()
		with placeholder.container():
			st.caption('ä¸ä¸æ¬¡æ´æ°æ¶é´:' + lasttime)
			st.caption('å½åæ´æ°æ¶é´:' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))	
			lasttime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
			
			# çæ§ç«¯å£
			st.markdown("### åç¬çæ§")
			st.caption("è·¯ç±å¨ID: " + str(option1) )
			st.caption("è·¯ç±å¨ç«¯å£: " + str(option2) )
			
			if option1 and option2:
				# create six columns
				kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
				nowdata = df[(df['è·¯ç±å¨'] == option1) & (df['ç«¯å£'] == option2)].iloc[0]
				# fill in those three columns with respective metrics or KPIs
				# print("nowdata", nowdata, "11", nowdata["æ¥æ¶æ°æ®åæ°(ä¸ª)"][0])
				kpi1.metric(
					label="æ¥æ¶æ°æ®åä¸ªæ°",
					value=nowdata["æ¥æ¶æ°æ®åæ°(ä¸ª)"].item(),
					delta=nowdata["æ¥æ¶æ°æ®åæ°(ä¸ª)"].item() - lastdata["æ¥æ¶æ°æ®åæ°(ä¸ª)"].item(),
				)

				kpi2.metric(
					label="æ¥æ¶æ°æ®é(byte)",
					value=nowdata["æ¥æ¶æ°æ®é(byte)"].item(),
					delta=nowdata["æ¥æ¶æ°æ®é(byte)"].item() - lastdata["æ¥æ¶æ°æ®é(byte)"].item(),
				)

				kpi3.metric(
					label="æ¥æ¶éè¯¯æ°æ®åæ°(ä¸ª)",
					value=nowdata["æ¥æ¶éè¯¯æ°æ®åæ°(ä¸ª)"].item(),
					delta=nowdata["æ¥æ¶éè¯¯æ°æ®åæ°(ä¸ª)"].item() - lastdata["æ¥æ¶éè¯¯æ°æ®åæ°(ä¸ª)"].item(),
				)

				kpi4.metric(
					label="åéæ°æ®åæ°(ä¸ª)",
					value=nowdata["åéæ°æ®åæ°(ä¸ª)"].item(),
					delta=nowdata["åéæ°æ®åæ°(ä¸ª)"].item() - lastdata["åéæ°æ®åæ°(ä¸ª)"].item(),
				)

				kpi5.metric(
					label="åéæ°æ®é(byte)",
					value=nowdata["åéæ°æ®é(byte)"].item(),
					delta=nowdata["åéæ°æ®é(byte)"].item() - lastdata["åéæ°æ®é(byte)"].item(),
				)

				kpi6.metric(
					label="åééè¯¯æ°æ®åæ°(ä¸ª)",
					value=nowdata["åééè¯¯æ°æ®åæ°(ä¸ª)"].item(),
					delta=nowdata["åééè¯¯æ°æ®åæ°(ä¸ª)"].item() - lastdata["åééè¯¯æ°æ®åæ°(ä¸ª)"].item(),
				)
				# ç»å¶çæ§ç«¯å£æµéæçº¿å¾
				# å¦æåæ¢ç«¯å£äºæ¸ç©ºæçº¿å¾æ°æ®
				if option1 != lastoption1 or option2 != lastoption2:
					linechart1 = [[lasttime, "æ¥æ¶æ°æ®åæ°(ä¸ª)",nowdata["æ¥æ¶æ°æ®åæ°(ä¸ª)"].item()], [lasttime, "æ¥æ¶éè¯¯æ°æ®åæ°(ä¸ª)", nowdata["æ¥æ¶éè¯¯æ°æ®åæ°(ä¸ª)"].item()], [lasttime, "åéæ°æ®åæ°(ä¸ª)", nowdata["åéæ°æ®åæ°(ä¸ª)"].item()], [lasttime, "åééè¯¯æ°æ®åæ°(ä¸ª)", nowdata["åééè¯¯æ°æ®åæ°(ä¸ª)"].item()]]
					linechart2 = [[lasttime, "æ¥æ¶æ°æ®é(byte)", nowdata["æ¥æ¶æ°æ®é(byte)"].item()], [lasttime, "åéæ°æ®é(byte)", nowdata["åéæ°æ®é(byte)"].item()]]
				st.markdown('### çæ§æµéå¾')
				lich1, lich2 = st.columns(2)
				if nowdata["æ¥æ¶æ°æ®åæ°(ä¸ª)"].item() != lastdata["æ¥æ¶æ°æ®åæ°(ä¸ª)"].item():
					linechart1.append([lasttime, "æ¥æ¶æ°æ®åæ°(ä¸ª)", nowdata["æ¥æ¶æ°æ®åæ°(ä¸ª)"].item()])

				if nowdata["æ¥æ¶éè¯¯æ°æ®åæ°(ä¸ª)"].item() != lastdata["æ¥æ¶éè¯¯æ°æ®åæ°(ä¸ª)"].item():
					linechart1.append([lasttime, "æ¥æ¶éè¯¯æ°æ®åæ°(ä¸ª)",nowdata["æ¥æ¶éè¯¯æ°æ®åæ°(ä¸ª)"].item()])

				if nowdata["åéæ°æ®åæ°(ä¸ª)"].item() != lastdata["åéæ°æ®åæ°(ä¸ª)"].item():
					linechart1.append([lasttime, "åéæ°æ®åæ°(ä¸ª)", nowdata["åéæ°æ®åæ°(ä¸ª)"].item()])

				if nowdata["åééè¯¯æ°æ®åæ°(ä¸ª)"].item() != lastdata["åééè¯¯æ°æ®åæ°(ä¸ª)"].item():
					linechart1.append([lasttime, "åééè¯¯æ°æ®åæ°(ä¸ª)", nowdata["åééè¯¯æ°æ®åæ°(ä¸ª)"].item()])

				if nowdata["æ¥æ¶æ°æ®é(byte)"].item() != lastdata["æ¥æ¶æ°æ®é(byte)"].item():
					linechart2.append([lasttime, "æ¥æ¶æ°æ®é(byte)", nowdata["æ¥æ¶æ°æ®é(byte)"].item()])

				if nowdata["åéæ°æ®é(byte)"].item() != lastdata["åéæ°æ®é(byte)"].item():
					linechart2.append([lasttime, "åéæ°æ®é(byte)", nowdata["åéæ°æ®é(byte)"].item()])	
				print(linechart1)
				lf1 = pd.DataFrame(linechart1, columns=linechartcolumns)
				lf2 = pd.DataFrame(linechart2, columns=linechartcolumns)

				with lich1:
					st.markdown("## æ°æ®å")
					fig = px.line(
						data_frame=lf1, y="æ°æ®", x="æ¶é´", color="ç±»å"
					)
					st.write(fig)
				
				with lich2:
					st.markdown("## æ°æ®é")
					fig = px.line(
						data_frame=lf2, y="æ°æ®", x="æ¶é´", color="ç±»å"
					)
					st.write(fig)

				lastdata = nowdata
			# ç»å¶æµéçæ§è¡¨æ ¼
			st.markdown("### æµéçæ§")
			st.dataframe(df)
			time.sleep(1)

def GetDataFrame():
	# è·åæ°æ®
		url ='http://0.0.0.0:8080/monitor/mactable/getinfo'
		response = requests.get(url)
		print(response.json())
		data = response.json()
		# éæ°æ´çæ°æ®æ´çä¸ºï¼
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
		datalist = [["è·¯ç±å¨", "ç«¯å£", "æ¥æ¶æ°æ®åæ°(ä¸ª)", "æ¥æ¶æ°æ®é(byte)", "æ¥æ¶éè¯¯æ°æ®åæ°(ä¸ª)", "åéæ°æ®åæ°(ä¸ª)", "åéæ°æ®é(byte)", "åééè¯¯æ°æ®åæ°(ä¸ª)"]]
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