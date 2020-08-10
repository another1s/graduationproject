import csv
import numpy as np
import os
import copy



def average(num, lst):
    sum = float(0)
    average_number = 0
    for data in lst:
        if data is not 'None':
           # print(data)
            sum = sum + float(data)
            average_number = float(sum / num)
        else:
            break
    return average_number

def reconstruct(pblist, pb,filename):   #pblist 存储的为每一个公司的所有pb值，y是用来按年份存储各个季度数据
    season1 = {'March':[],'June':[],'September':[],'December':[]}
    season2 = {'March':[],'June':[],'September':[],'December':[]}
    season3 = {'March':[],'June':[],'September':[],'December':[]}
    season4 = {'March':[],'June':[],'September':[],'December':[]}
    season5 = {'March':[],'June':[],'September':[],'December':[]}
    season6 = {'March':[],'June':[],'September':[],'December':[]}

    for lst in pblist:
        date = lst[0]
        if filename == "PLUG_price_to_book_value_data.csv":
            year = date[5:10]
        else:
            year = date[0:4]

        #if filename =='SOHVY_price_to_book_value_data.csv':
       #     print("hellp")
      #  print(type(year))
        #print(lst,year)
        while year == '2018':
            if '-03-' in lst[0]:
                season1['March'].append(lst[1])
            elif '-06-' in lst[0]:
                season1['June'].append(lst[1])
            elif '-09-' in lst[0]:
                season1['September'].append(lst[1])
            elif '-12-' in lst[0]:
                season1['December'].append(lst[1])
            break

        while year == '2017':
            if '-03-' in lst[0]:
                season2['March'].append(lst[1])
            elif '-06-' in lst[0]:
                season2['June'].append(lst[1])
            elif '-09-' in lst[0]:
                season2['September'].append(lst[1])
            elif '-12-' in lst[0]:
                season2['December'].append(lst[1])
            break

        while year == '2016':
            if '-03-' in lst[0]:
                season3['March'].append(lst[1])
            elif '-06-' in lst[0]:
                season3['June'].append(lst[1])
            elif '-09-' in lst[0]:
                season3['September'].append(lst[1])
            elif '-12-' in lst[0]:
                season3['December'].append(lst[1])
            break

        while year == '2015':
            if '-03-' in lst[0]:
                season4['March'].append(lst[1])
            elif '-06-' in lst[0]:
                season4['June'].append(lst[1])
            elif '-09-' in lst[0]:
                season4['September'].append(lst[1])
            elif '-12-' in lst[0]:
                season4['December'].append(lst[1])
            break

        while year == '2014':
            if '-03-' in lst[0]:
                season5['March'].append(lst[1])
            elif '-06-' in lst[0]:
                season5['June'].append(lst[1])
            elif '-09-' in lst[0]:
                season5['September'].append(lst[1])
            elif '-12-' in lst[0]:
                season5['December'].append(lst[1])
            break

        while year == '2013':
            if '-03-' in lst[0]:
                season6['March'].append(lst[1])
            elif '-06-' in lst[0]:
                season6['June'].append(lst[1])
            elif '-09-' in lst[0]:
                season6['September'].append(lst[1])
            elif '-12-' in lst[0]:
                season6['December'].append(lst[1])
            break
  #  print(season1)
    pb.every_year['2018'] = season1
    pb.every_year['2017'] = season2
    pb.every_year['2016'] = season3
    pb.every_year['2015'] = season4
    pb.every_year['2014'] = season5
    pb.every_year['2013'] = season6
    return pb.every_year
class share_data:
    every_year = dict()
    season = {'March':[],'June':[],'September':[],'December':[]}
    year=dict()
    month=dict()
    s_value=list()
    def initial(self):
        pb.season = {'March': [], 'June': [], 'September': [], 'December': []}
        for years in  ['2018','2017','2016','2015','2014','2013']:
            pb.every_year[years]=pb.season
        return pb

    def data(self, address, filename):  # 此函数用来处理pb值，并返回一个list含有五年的市净率（季度数据）
        with open(address+'\\'+filename, 'r') as pbv:
            data = csv.reader(pbv)
            rows = [row for row in data]
        return rows

pb = share_data()

path = "E:\onedrive\graduation\database\pricetobook"
files = os.listdir(path)
for filename in files:
  #  print(filename)
    Pb_value=pb.data(path,filename)

    pb.initial()
    yearly_pb = reconstruct(Pb_value,pb,filename)    # 得到以年数分布四个关键月份pb数据
 #   if filename == "PLUG_price_to_book_value_data.csv" or "SOHVY_price_to_book_value_data.csv":
#        print(yearly_pb)
    for value in yearly_pb:                 # 此应该取得年份
  #      print(value)
        year_value = yearly_pb[value]      # 该年四个月的数据
   #     print(year_value)
        for months in year_value:
            months_value = year_value[months]
        #    print(type(months_value),months_value)
            if (len(months_value) == 0) :
                break
            else:
                months_value = average(len(months_value), months_value)
            year_value[months] = months_value
    #print(yearly_pb)
