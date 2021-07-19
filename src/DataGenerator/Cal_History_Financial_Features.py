from src.DataGenerator.Cal_Financial_Feature import CalFinancialFeature
from datalayer.DataLoader import DataLoader
from utils.creat_date_list import creat_date_list_monthend
import pandas as pd


def cal_history_financial_features(date_list):
    for period_date in date_list:
        try:
            df_all_financial_features=CalFinancialFeature(period_date).cal_all_tickers_feature()
            df_all_financial_features.to_csv('/home/guosong/PycharmProjects/EquityFundamentalModel/data/financial_feature/features_{}.csv'.format(period_date),index=True)

        except:
            continue
    pass

def cross_section_data_preprocess(date_list):
    qualitative_feature=['企业现金流肖像+++','企业现金流肖像++-','企业现金流肖像+--','企业现金流肖像+-+',
                         '企业现金流肖像-++','企业现金流肖像-+-','企业现金流肖像--+','企业现金流肖像---',
                         '存货周转率下降&毛利率上升']
    min_ind_sample=10
    for period_date in date_list:
        print(period_date)
        df_features=pd.read_csv('/home/guosong/PycharmProjects/EquityFundamentalModel/data/financial_feature/features_{}.csv'.format(period_date))
        df_features['ticker']=df_features['ticker'].map(lambda x:str(x).zfill(6))
        df_features.set_index('ticker',inplace=True)
        df_features.drop('enddate',axis=1,inplace=True)

        df_features_quan=df_features.drop(qualitative_feature,axis=1) # original quantitative financial features
        df_features_processed=df_features_quan.copy() # preprocessed quantitative financial features

        # industry neutralization
        ticker_list=df_features.index.tolist()
        df_industry=DataLoader().load_ticker_industry_pit(period_date,ticker_list).drop(['secshortname','intodate','outdate'],axis=1).set_index('ticker')

        ind_dic={1:'industryname1',2:'industryname2',3:'industryname3'}
        all_ind=df_industry['industryname3'].drop_duplicates(keep='first').tolist()

        for ind in all_ind:
            ind_class=3
            df_industry_tmp=df_industry.loc[df_industry[ind_dic[ind_class]]==ind]
            ind_ticker_list=df_industry_tmp.index.tolist()

            if len(ind_ticker_list)<min_ind_sample:
                ind_class-=1
                ind_name=df_industry_tmp[ind_dic[ind_class]].iloc[0]
                df_industry_tmp = df_industry.loc[df_industry[ind_dic[ind_class]] == ind_name]
                if df_industry_tmp.shape[0]<min_ind_sample:
                    ind_class -= 1
                    ind_name = df_industry_tmp[ind_dic[ind_class]].iloc[0]
                    df_industry_tmp = df_industry.loc[df_industry[ind_dic[ind_class]] == ind_name]

            df_features_processed.loc[df_features_processed.index.isin(ind_ticker_list)]-=df_features_quan.loc[df_features_quan.index.isin(df_industry_tmp.index)].median()

        df_features_processed=df_features_processed.fillna(0)

        # use MAD to remove outlier
        made=1.4826*((df_features_processed-df_features_processed.median()).abs()).median()
        made.loc[made==0]=df_features_processed[made.loc[made==0].index].std()

        df_features_max=df_features_processed.median()+3*made
        df_features_min=df_features_processed.median()-3*made
        df_features_processed = df_features_processed.clip(df_features_min,df_features_max, axis=1)

        # standerization the data
        df_features_processed=(df_features_processed-df_features_processed.median())/df_features_processed.std()

        df_features_processed[qualitative_feature]=df_features[qualitative_feature]
        df_features_processed.to_csv('/home/guosong/PycharmProjects/EquityFundamentalModel/data/preprocessed_feature/preprocessed_feature_{}.csv'.format(period_date),index=True)

    return 

if __name__=='__main__':
    begin_date = '2013-04-30'
    end_date = '2021-06-30'
    date_list = creat_date_list_monthend(begin_date, end_date)
    # date_list=['2021-06-30']
    # cal_history_financial_features(date_list)
    cross_section_data_preprocess(date_list)