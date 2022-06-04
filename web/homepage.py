import streamlit as st
import time
import datetime

def main():
	st.set_page_config(page_title = 'monitor_web', page_icon=":mag:", layout="wide")
	st.title('流量监控&智能DDos防御 :sunglasses:')
	if 'first_visit' not in st.session_state:
		st.session_state.first_visit = True
	else:
		st.session_state.first_visit=False
	#初始化
	if st.session_state.first_visit:
		st.session_state.date_time=datetime.datetime.now() + datetime.timedelta(hours=8) # Streamlit Cloud的时区是UTC，加8小时即北京时间
		st.balloons()
		



if __name__ == '__main__':
	main()