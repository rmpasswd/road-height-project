from bs4 import BeautifulSoup as bsp
import requests


page = requests.get("http://www.ffwc.gov.bd/ffwc_charts/rainfall.php")
# print(page.status_code)
bsob  = bsp(page.content, 'lxml')


rainfall = bsob.select('table')[0].findAll('tr')[9].findAll('td')[4].getText()

if rainfall == "NP":
	rainfall=0