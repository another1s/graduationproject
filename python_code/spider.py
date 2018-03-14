import urllib.request, urllib.parse, urllib.error
from urllib.request import urlopen
import json
import http
import difflib
from bs4 import BeautifulSoup
# importing the requests library
import requests
import  sys

share_name=list()
share_name=["RBS","SHG","BSMX","SUPV","CIB","SAN","BBD","IBN"]
# api-endpoint

# this function is used for get the eps_value of the stock within four recent quarters
def get_eps(share_name):
    URL1="https://api.iextrading.com/1.0/stock/"+share_name+"/earnings/"
 #   print(URL1)
    data = requests.get(url=URL1)
    print(data.json())
    stock_data=data.json()
    # data is the list that contains the eps values and date
    data=list()
    eps_values=list()
    quarters_=list()
    data.append(share_name)

    for elements in stock_data['earnings']:
        eps_values.append(elements['actualEPS'])
        quarters_.append(elements['EPSReportDate'])
    data.append(eps_values)
    data.append(quarters_)
    return data

def PB_ratio(share_name,date):
    URL="https://api.iextrading.com/1.0/stock/"+share_name+"/stats"
 #   print(URL)
    data=requests.get(url=URL)
  #  print(data.json())
    stock_data=data.json()
    print(stock_data)
    Price=stock_data['priceToBook']
    return Price
def gross_margin(share_name):
    URL = "https://api.iextrading.com/1.0/stock/" + share_name + "/stats"
#    print(URL)
    data = requests.get(url=URL)
    #  print(data.json())
    stock_data = data.json()
    print(stock_data)
    profit = stock_data['profitMargin']
    return profit

for elements in share_name:
    result=get_eps(elements)
    pb=PB_ratio(elements,"2017-05-02")
    profit=gross_margin(elements)
    print(result)
    print(pb)
    print(profit)
#print("Hello")
#pb=PB_ratio("aapl","2017-05-02")
#print(pb)



