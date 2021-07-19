import pandas as pd
import numpy as np
from datalayer.DataLoader import DataLoader
from utils.creat_date_list import creat_date_list_monthend


def cal_feature_coverage(date_list):
    df_statistics=pd.DataFrame()
    for date in date_list:
        print(date)
        df_feature=pd.read_csv('/home/guosong/PycharmProjects/EquityFundamentalModel/data/preprocessed_feature/preprocessed_feature_{}.csv'.format(date))
        ticker_list=DataLoader().load_list_ticker_pit(date)
        df_industry=DataLoader().load_ticker_industry_pit(date,ticker_list)
        df_statistics[date]=pd.Series({'all_ticker_num':df_industry.shape[0],'featured_ticker_num':df_feature.shape[0]})

    df_statistics=df_statistics.T
    df_statistics.to_excel('/home/guosong/PycharmProjects/EquityFundamentalModel/data/analysis/feature_coverage.xlsx',index=True)
    return

def analysis_label(period_date):
    df_feature = pd.read_csv('/home/guosong/PycharmProjects/EquityFundamentalModel/data/sample_label/sample_label_{}.csv'.format(period_date))
    print()

def combine_results(date_list):
    df_all_results=pd.DataFrame()
    for date in date_list:
        df=pd.read_excel('/home/guosong/PycharmProjects/EquityFundamentalModel/data/results/financial_score_{}.xlsx'.format(date))
        df['predict_date']=date
        df=df[['predict_date','ticker','ticker_score']]
        df_all_results=pd.concat([df_all_results,df],axis=0)

    df_all_results.to_csv('/home/guosong/PycharmProjects/EquityFundamentalModel/data/results/all_financial_score.csv',index=False)




if __name__=='__main__':
    date_begin = '2016-01-31'
    date_end = '2021-06-30'
    date_list = creat_date_list_monthend(date_begin, date_end)
    # cal_feature_coverage(date_list)
    # analysis_label('2020-12-31')
    combine_results(date_list)