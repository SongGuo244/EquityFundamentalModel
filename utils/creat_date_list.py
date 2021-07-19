from datetime import datetime
import pandas as pd

def creat_date_list_monthend(begin_date,end_date):
    return [datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=begin_date, end=end_date,freq='M'))]

if __name__=='__main__':
    begin_date='2013-04-30'
    end_date='2021-07-07'
    date_list=creat_date_list_monthend(begin_date,end_date)
    print()