import streamlit as st
import time
import datetime

def main():
	st.set_page_config(page_title = 'monitor_web', layout='wide', page_icon=":mag:")
	
	#st.title('流量监控&智能DDos防御 :sunglasses:')

	if 'first_visit' not in st.session_state:
		st.session_state.first_visit = True
	else:
		st.session_state.first_visit=False
	#初始化
	if st.session_state.first_visit:
		st.session_state.date_time=datetime.datetime.now() + datetime.timedelta(hours=15) # Streamlit Cloud的时区是UTC，加8小时即北京时间
		st.balloons()
	with st.container():
		col1,col2=st.columns(2)
		col1.title('流量监控&智能DDos防御 :sunglasses:')
		col2.metric(label='NowDate', value=str(st.session_state.date_time.date()))
		st.caption('该项目实现网络流量监控、端口流量控制、节点差错排查等基础功能')
		st.caption('在此基础上，通过人工智能算法，实现DDos动态监测')
		st.caption('通过对DDos智能识别，实现动态流量过滤')
	

	

			
		

if __name__ == '__main__':
	main()