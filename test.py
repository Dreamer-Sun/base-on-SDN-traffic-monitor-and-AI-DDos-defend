import openpyxl
import requests
import json

params = {}
url = "http://0.0.0.0:8080/monitor/getTotalData"
respond = requests.get(url)
print(json.loads(respond.text))