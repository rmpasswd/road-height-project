from bs4 import BeautifulSoup as bsp
from pprint import pprint
import requests

print("ladsjfk;ad")

page = requests.get("http://www.ffwc.gov.bd/ffwc_charts/waterlevel.php")
print(page.status_code)
bsob  = bsp(page.content, 'lxml')

firstTable = bsob.select('table')
rows = firstTable[0].findAll('tr')
bahadurabad = rows[8].findAll('td')[4]
# print(bahadurabad.getText())

date1 = bsob.select('table thead tr')[1].select('td')[3].getText()
date2 = bsob.select('table thead tr')[1].select('td')[4].getText()
print(date1,date2)

wtrlvl1 = bsob.select('table')[0].findAll('tr')[8].findAll('td')[3].getText()
wtrlvl2 = bsob.select('table')[0].findAll('tr')[8].findAll('td')[4].getText()

print(wtrlvl1, wtrlvl2);print("______________________________________________________")
