import pandas as pd
import psycopg2
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta


class DataLoader:
    def __init__(self):
        with open('/home/guosong/PycharmProjects/EquityFundamentalModel/config/db.json', 'r') as load_f:
            self.conn_dic = json.load(load_f)

        pass

    def load_list_ticker_pit(self, period_date):
        try:
            sql = 'select ticker, listdate, delistdate ' \
                  'from datayes.equ ' \
                  'where listdate<{!r} '.format(period_date)

            with psycopg2.connect(database=self.conn_dic['database'], user=self.conn_dic['user'], password=self.conn_dic['password'],
                                  host=self.conn_dic['host'],port=self.conn_dic['port']) as conn:
                df = pd.read_sql(sql, conn)
                df.drop(index=df.loc[df['listdate'] == 'NaN'].index,inplace=True)
                df['delistdate'].replace('NaN',datetime.today().date().strftime('%Y-%m-%d'),inplace=True)
                df=df.loc[(df['listdate']<period_date)&(df['delistdate']>=period_date)]

            return df['ticker'].tolist()
        except:
            raise Exception('load list ticker pit err')

    def load_ticker_name(self,ticker_list):
        try:
            sql = 'select ticker, secshortname ' \
                  'from datayes.equ ' \
                  'where ticker in ({}) '.format(str(ticker_list)[1:-1])

            with psycopg2.connect(database=self.conn_dic['database'], user=self.conn_dic['user'], password=self.conn_dic['password'],
                                  host=self.conn_dic['host'],port=self.conn_dic['port']) as conn:
                df = pd.read_sql(sql, conn)

            return df
        except:
            raise Exception('load ticker name err')

    def load_ticker_industry_pit(self,period_date, ticker_list):
        try:
            sql = 'SELECT ticker, secshortname, intodate, outdate, industryname1, industryname2, industryname3 ' \
                  'FROM datayes.equindustry ' \
                  'where industryversioncd=\'010303\' ' \
                  'and ticker in ({}) '.format(str(ticker_list)[1:-1])

            with psycopg2.connect(database=self.conn_dic['database'], user=self.conn_dic['user'], password=self.conn_dic['password'],
                                  host=self.conn_dic['host'],port=self.conn_dic['port']) as conn:
                df = pd.read_sql(sql, conn)

            df=df.drop(index=df.loc[df['industryname1'].isin(['银行','金融服务','非银金融','房地产'])].index,axis=0)
            df['intodate']=df['intodate'].map(lambda x: datetime.strptime(x,'%Y-%m-%d').date())
            df['outdate']=df['outdate'].replace('NaN',str(datetime.today().date()))
            df['outdate'] = df['outdate'].map(lambda x: datetime.strptime(x, '%Y-%m-%d').date())

            period_date_dt=datetime.strptime(period_date,'%Y-%m-%d').date()
            df=df.loc[(df['intodate']<period_date_dt)&(period_date_dt<=df['outdate'])]

            df=df.loc[df['ticker'].map(lambda x:x[:2] in ['00','30','60'])]

            return df
        except:
            raise Exception('load data err')

    def load_balance_sheet_data_pit(self, ticker_list, period_date):
        try:
            sql = 'SELECT * FROM datayes.fdmtbsindu ' \
                  'where mergedflag=\'1\' ' \
                  'and ticker in ({}) ' \
                  'and publishdate<={!r} ' \
                  'and enddate>=\'2010-03-31\' '.format(str(ticker_list)[1:-1],period_date)

            with psycopg2.connect(database=self.conn_dic['database'], user=self.conn_dic['user'], password=self.conn_dic['password'],
                                  host=self.conn_dic['host'],port=self.conn_dic['port']) as conn:
                df = pd.read_sql(sql, conn)

            return df
        except:
            raise Exception('load balance sheet data err')

    def load_income_statement_data_pit_ttm(self, ticker_list, period_date):
        try:
            sql = 'SELECT * FROM datayes.fdmtisinduttmpit ' \
                  'where ticker in ({}) ' \
                  'and publishdate<={!r} ' \
                  'and enddate>=\'2010-03-31\' '.format(str(ticker_list)[1:-1],period_date)

            with psycopg2.connect(database=self.conn_dic['database'], user=self.conn_dic['user'], password=self.conn_dic['password'],
                                  host=self.conn_dic['host'],port=self.conn_dic['port']) as conn:
                df = pd.read_sql(sql, conn)

            return df
        except:
            raise Exception('load income statement data err')

    def load_cash_flow_data_pit_ttm(self, ticker_list, period_date):
        try:
            sql = 'SELECT * FROM datayes.fdmtcfinduttmpit ' \
                  'where ticker in ({}) ' \
                  'and publishdate<={!r} ' \
                  'and enddate>=\'2010-03-31\' '.format(str(ticker_list)[1:-1],period_date)

            with psycopg2.connect(database=self.conn_dic['database'], user=self.conn_dic['user'], password=self.conn_dic['password'],
                                  host=self.conn_dic['host'],port=self.conn_dic['port']) as conn:
                df = pd.read_sql(sql, conn)

            return df
        except:
            raise Exception('load income statement data err')

    def load_financial_derivative_data_pit(self, ticker_list, period_date):
        try:
            sql = 'SELECT ticker, publishdate, enddate, ' \
                  'tfixedassets, intcl, intdebt, ndebt, ntanassets, workcapital, nworkcapital, ' \
                  'ic, da FROM datayes.fdmtderpit ' \
                  'where ticker in ({}) ' \
                  'and publishdate<={!r} ' \
                  'and enddate>=\'2010-03-31\' '.format(str(ticker_list)[1:-1],period_date)

            with psycopg2.connect(database=self.conn_dic['database'], user=self.conn_dic['user'], password=self.conn_dic['password'],
                                  host=self.conn_dic['host'],port=self.conn_dic['port']) as conn:
                df = pd.read_sql(sql, conn)

            return df
        except:
            raise Exception('load financial derivative data err')

    def load_st_ticker(self, ticker_list, sample_date, period_date, nyears=3):
        sample_date_dt=datetime.strptime(sample_date,'%Y-%m-%d')
        date_end=sample_date_dt+relativedelta(years=nyears)
        date_end=date_end.strftime('%Y-%m-%d')
        if date_end>period_date:
            date_end=period_date

        try:
            sql ='select distinct ticker ' \
                 'from datayes.secst ' \
                 'where ticker in ({}) ' \
                 'and tradedate between {!r} and {!r} '.format(str(ticker_list)[1:-1], sample_date, date_end)

            with psycopg2.connect(database=self.conn_dic['database'], user=self.conn_dic['user'], password=self.conn_dic['password'],
                                  host=self.conn_dic['host'],port=self.conn_dic['port']) as conn:
                df = pd.read_sql(sql, conn)

            return df['ticker'].tolist()
        except:
            raise Exception('load ST tickers err')

    def load_ticker_name_pit(self, ticker_list, period_date):
        # 注意：已退市股票似乎没有PIT名称
        try:
            sql ='select ticker, value as tickername, begindate, enddate ' \
                 'from datayes.secchghistory ' \
                 'where ticker in ({}) ' \
                 'and changtype=\'简称变更\' ' \
                 'and assetclass=\'E\' '.format(str(ticker_list)[1:-1])

            with psycopg2.connect(database=self.conn_dic['database'], user=self.conn_dic['user'], password=self.conn_dic['password'],
                                  host=self.conn_dic['host'],port=self.conn_dic['port']) as conn:
                df = pd.read_sql(sql, conn)

            df['enddate']=df['enddate'].replace('NaN',datetime.today().date().strftime('%Y-%m-%d'))
            df=df.loc[(df['begindate']<=period_date)&(period_date<=df['enddate'])]

            return df
        except:
            raise Exception('load ticker name pit err')

    def load_fund_holdings(self, ticker_list, sample_date, period_date, nmonth=6):
        sample_date_dt = datetime.strptime(sample_date, '%Y-%m-%d')
        date_end = sample_date_dt + relativedelta(months=nmonth)
        date_end = date_end.strftime('%Y-%m-%d')
        if date_end > period_date:
            date_end = period_date

        try:
            df_fund_rate = pd.read_csv('/home/guosong/PycharmProjects/EquityFundamentalModel/data/basic_data/fund_rate.csv')
            df_fund_rate=df_fund_rate.drop('period', axis=1).drop_duplicates(keep='first')
            df_fund_rate=df_fund_rate.loc[df_fund_rate['overallrating'].astype(int)==5] # fund rate is 5
            fund_id_set=set(df_fund_rate['secid'].tolist())

            sql = 'select distinct a.holdingticker ' \
                  'from datayes.fundholdings a ' \
                  'join datayes.fund b ' \
                  'on a.secid=b.secid ' \
                  'where b.category in (\'E\',\'H\') ' \
                  'and b.indexfund=\'NaN\' ' \
                  'and b.etflof=\'NaN\' ' \
                  'and b.isqdii=0 ' \
                  'and b.isfof=0 ' \
                  'and a.holdingsectype=\'E\' ' \
                  'and a.holdingticker in ({}) ' \
                  'and a.publishdate between {!r} and {!r} ' \
                  'and a.secid in ({}) '.format(str(ticker_list)[1:-1], sample_date, date_end, str(fund_id_set)[1:-1])

            with psycopg2.connect(database=self.conn_dic['database'], user=self.conn_dic['user'],
                                  password=self.conn_dic['password'],
                                  host=self.conn_dic['host'], port=self.conn_dic['port']) as conn:
                df = pd.read_sql(sql, conn)

            return set(df['holdingticker'].tolist())

        except:
            raise Exception('load fund holdings err')



if __name__=='__main__':
    ticker='600519'
    # period_date=datetime.today().date().strftime('%Y-%m-%d')
    period_date='2019-02-28'
    ticker_list=DataLoader().load_list_ticker_pit(period_date)
    # df=DataLoader().load_st_ticker_pit(ticker_list,period_date)
    # df_industry=DataLoader().load_ticker_industry_pit(period_date,ticker_list)
    # df_st=DataLoader().load_st_ticker(ticker_list,'2018-01-01','2019-06-01')
    df=DataLoader().load_fund_holdings(ticker_list,'2013-04-30','2021-06-30')
    # df=DataLoader().load_ticker_name_pit(ticker_list,period_date)
    print()
    # ticker_list=df_industry['ticker'].tolist()
    # df=DataLoader().load_balance_sheet_data_pit([ticker],period_date)
    # df_is=DataLoader().load_income_statement_data_pit_ttm(ticker_list,period_date)
    # df_cf=DataLoader().load_cash_flow_data_pit_ttm(ticker_list,period_date)
    # df_der=DataLoader().load_financial_derivative_data_pit(ticker_list,period_date)
    # res=join_list_to_str(ticker_list)
    # print(res)
    # df=DataLoader().load_ticker_name(['600519','600002'])
    print()