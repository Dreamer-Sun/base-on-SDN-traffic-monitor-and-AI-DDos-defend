import streamlit as st
import time
import datetime
from streamlit_echarts import st_echarts, JsCode, st_pyecharts
import requests
import json
from pyecharts.charts import Bar
from pyecharts import options as opts

def main():
	st.set_page_config(page_title = 'monitor_web', layout='wide', page_icon=":mag:")
	
	#st.title('流量监控&智能DDos防御 :sunglasses:')

	if 'first_visit' not in st.session_state:
		st.session_state.first_visit = True
	else:
		st.session_state.first_visit=False
	#初始化
	if st.session_state.first_visit:
		st.session_state.date_time=datetime.datetime.now() + datetime.timedelta(hours=-9) # Streamlit Cloud的时区是UTC，加8小时即北京时间
		st.balloons()	
	option1 = {
		"legend": {},
		"tooltip": {
			"trigger": 'axis',
			"showContent": "false"
		},
		"dataset": {
		"source": GetTotalFlow(1)
        },
		"xAxis": {"type": 'category'},
		"yAxis": {"gridIndex": 0},
		"grid": {"top": '15%'},
		"series": [
			{"type": 'bar', "smooth": "true", "seriesLayoutBy": 'row', "emphasis": {"focus": 'series'}},
			{"type": 'bar', "smooth": "true", "seriesLayoutBy": 'row', "emphasis": {"focus": 'series'}},
	
		],
			"tooltip": {
				"show": "true",
					},
			"label": {
				"show":"true"
				},
	};
	option2 = {
        "legend": {},
        "tooltip": {
            "trigger": 'axis',
            "showContent": "false"
        },
        "dataset": {
            "source": GetTotalFlow(2)
        },
        "xAxis": {"type": 'category'},
        "yAxis": {"gridIndex": 0},
        "grid": {"top": '15%'},
        "series": [
            {"type": 'line', "smooth": "true", "seriesLayoutBy": 'row', "emphasis": {"focus": 'series'}},
            {"type": 'line', "smooth": "true", "seriesLayoutBy": 'row', "emphasis": {"focus": 'series'}},
        ],
            "tooltip": {
                    "show": "true",
                },
            "label": {
                "show":"true"
    	},
    };
	


	with st.container():
		col1,col2=st.columns(2)
		col1.title('流量监控&智能DDos防御 :sunglasses:')
		col2.metric(label='NowDate', value=str(st.session_state.date_time.date()))
		st.caption('该项目实现网络流量监控、端口流量控制、节点差错排查等基础功能')
		st.caption('在此基础上，通过人工智能算法，实现DDos动态监测')
		st.caption('通过对DDos智能识别，实现动态流量过滤')
		st.caption('')
		st.subheader('一周内流量统计：')
		st1 = st_echarts(options=option1)
		st2 = st_echarts(options=option2)
		
		
		


def GetTotalFlow(index):
	url = "http://0.0.0.0:8080/monitor/getTotalData"
	data = requests.get(url)
	data_text = json.loads(data.text)
	if index == 1:
		tmp = []
		tmp.append(data_text[0])
		tmp.append(data_text[1])
		tmp.append(data_text[2])
		return tmp
	else:
		tmp = []
		tmp.append(data_text[0])
		tmp.append(data_text[3])
		tmp.append(data_text[4])
		return tmp
			
		

if __name__ == '__main__':
	main()