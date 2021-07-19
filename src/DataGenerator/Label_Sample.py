from datalayer.DataLoader import DataLoader
from utils.creat_date_list import creat_date_list_monthend
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


def labelsample(date_list):
    for period_date in date_list:
        try:
            print(period_date)
            # split the sample data to training set and validation set
            sample_date_begin = '2013-04-30'
            period_date_dt = datetime.strptime(period_date, '%Y-%m-%d').date()
            sample_date_end = datetime(period_date_dt.year, period_date_dt.month, 1).date()
            sample_date_end = (sample_date_end - relativedelta(days=1)).strftime('%Y-%m-%d')  # notice here!!!
            sample_date_list = creat_date_list_monthend(sample_date_begin, sample_date_end)

            # prepare sample data
            df_sample_data = pd.DataFrame()
            for date in sample_date_list:
                # print('prepare sample data {!r}'.format(date))
                df_sample = pd.read_csv('/home/guosong/PycharmProjects/EquityFundamentalModel/data/preprocessed_feature/preprocessed_feature_{}.csv'.format(date))
                df_sample['ticker'] = df_sample['ticker'].map(lambda x: str(x).zfill(6))
                df_sample.set_index('ticker', inplace=True)
                df_sample['label'] = None
                label1 = DataLoader().load_st_ticker(df_sample.index.tolist(), date, period_date)
                label0 = DataLoader().load_fund_holdings(df_sample.index.tolist(), date, period_date)
                df_sample.loc[df_sample.index.isin(label0), 'label'] = 0
                df_sample.loc[df_sample.index.isin(label1), 'label'] = 1
                df_sample['sample_date'] = date
                df_sample_data = pd.concat([df_sample_data, df_sample], axis=0)

            df_sample_data.to_csv('/home/guosong/PycharmProjects/EquityFundamentalModel/data/model_sample/model_sample_{}.csv'.format(period_date),index=True)
        except:
            print('err on {}'.format(period_date))
            continue
    return



if __name__=='__main__':
    # begin_date = '2016-01-31'
    # end_date = '2018-12-31'
    # date_list = creat_date_list_monthend(begin_date, end_date)
    labelsample(['2020-03-31'])
