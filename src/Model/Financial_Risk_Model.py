from datalayer.DataLoader import DataLoader
import pandas as pd
import numpy as np
import itertools
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from utils.creat_date_list import creat_date_list_monthend
from utils.reliefF import reliefF
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (balanced_accuracy_score, precision_recall_fscore_support)
from utils.Tri_training import TriTraining


class FinancialRiskModel:
    def __init__(self, period_date):
        self.period_date=period_date
        self.qualitative_feature = ['企业现金流肖像+++', '企业现金流肖像++-', '企业现金流肖像+--', '企业现金流肖像+-+',
                                    '企业现金流肖像-++', '企业现金流肖像-+-', '企业现金流肖像--+', '企业现金流肖像---',
                                    '存货周转率下降&毛利率上升']

        # split the sample data to training set and validation set
        sample_date_begin='2013-04-30'
        period_date_dt=datetime.strptime(self.period_date,'%Y-%m-%d').date()
        sample_date_end = datetime(period_date_dt.year,period_date_dt.month, 1).date()
        sample_date_end=(sample_date_end - relativedelta(days=1)).strftime('%Y-%m-%d') # notice here!!!
        self.sample_date_list=creat_date_list_monthend(sample_date_begin, sample_date_end)

        # avoid overfitting (pargo)
        n_validation=int(len(self.sample_date_list)/4)
        n_purging=6 # 6 months' interval
        self.train_set_datelist = self.sample_date_list[:(len(self.sample_date_list) - n_validation)]
        self.validation_set_datelist=self.sample_date_list[-(n_validation-n_purging):]

        # prepare sample data
        # self.df_sample_data=pd.DataFrame()
        # for date in self.sample_date_list:
        #     print('prepare sample data {!r}'.format(date))
        #     df_sample = pd.read_csv('/home/guosong/PycharmProjects/EquityFundamentalModel/data/preprocessed_feature/preprocessed_feature_{}.csv'.format(date))
        #     df_sample['ticker'] = df_sample['ticker'].map(lambda x: str(x).zfill(6))
        #     df_sample.set_index('ticker', inplace=True)
        #     df_sample['label']=None
        #     label1=DataLoader().load_st_ticker(df_sample.index.tolist(),date,self.period_date)
        #     label0=DataLoader().load_fund_holdings(df_sample.index.tolist(),date,self.period_date)
        #     df_sample.loc[df_sample.index.isin(label0),'label']=0
        #     df_sample.loc[df_sample.index.isin(label1),'label']=1
        #     df_sample['sample_date']=date
        #     self.df_sample_data=pd.concat([self.df_sample_data,df_sample],axis=0)

        self.df_sample_data=pd.read_csv('/home/guosong/PycharmProjects/EquityFundamentalModel/data/model_sample/model_sample_{}.csv'.format(self.period_date),index_col=0)
        self.df_sample_data.index = self.df_sample_data.index.map(lambda x: str(x).zfill(6))

        self.all_features=[] # all financial features ordered by Relief score, after decorrelation
        self.parameter=[1e-3,0.5,100,10,20,0.5]
        self.model=None # trained model (semi-supervised and class-imbalanced)

        self.imp_median = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()

        pass


    def imbalanced_relief(self,df_sample,k_relief=10):
        divid_scale=5
        df_sample_min=df_sample.loc[df_sample['label']==1].drop('label',axis=1)
        df_sample_mar=df_sample.loc[df_sample['label']==0].drop('label',axis=1)
        nsample_mar=df_sample_mar.shape[0]
        nsample_min=df_sample_min.shape[0]
        nloops=int(nsample_mar/nsample_min)
        feature_score=np.zeros(df_sample_min.shape[1])
        for loop in range(nloops):
            df_resampled_min=resample(df_sample_min,replace=True,n_samples=int(nsample_min/divid_scale),random_state=None)
            df_resampled_mar=resample(df_sample_mar,replace=True,n_samples=df_resampled_min.shape[0],random_state=None)
            X=pd.concat([df_resampled_mar,df_resampled_min]).values
            Y=np.append(np.zeros(df_resampled_mar.shape[0]),np.ones(df_resampled_min.shape[0]))
            feature_score+=reliefF(X,Y,k=k_relief)

            pass

        feature_score/=nloops
        df_result=pd.Series(feature_score,index=df_sample_min.columns).sort_values(ascending=False)

        return df_result

    def feature_selection(self):
        df_training_data=self.df_sample_data.loc[self.df_sample_data['sample_date'].isin(self.train_set_datelist)].copy()
        df_training_data_preprocessed=df_training_data.drop(self.qualitative_feature+['label','sample_date'],axis=1).copy()
        df_training_data_preprocessed=df_training_data_preprocessed.fillna(df_training_data_preprocessed.median())
        df_training_data_preprocessed=(df_training_data_preprocessed-df_training_data_preprocessed.median())/df_training_data_preprocessed.std()
        df_training_data_preprocessed=(df_training_data_preprocessed-df_training_data_preprocessed.min())/(df_training_data_preprocessed.max()-df_training_data_preprocessed.min())
        df_training_data_preprocessed[self.qualitative_feature+['label']]=df_training_data[self.qualitative_feature+['label']]

        # df_feature_selection_res=self.imbalanced_relief(df_training_data_preprocessed.loc[~df_training_data_preprocessed['label'].isna()])
        # df_feature_selection_res.to_csv('/home/guosong/PycharmProjects/EquityFundamentalModel/data/feature_selection/feature_selection_result_{}.csv'.format(self.period_date),index=True)

        self.all_features = self.feature_decorrelation(df_training_data_preprocessed.drop('label',axis=1))

        return

    def feature_decorrelation(self,df_sample):
        corr_threshold=0.8
        df_feature_selection_res=pd.read_csv('/home/guosong/PycharmProjects/EquityFundamentalModel/data/feature_selection/feature_selection_result_{}.csv'.format(self.period_date),index_col=0)
        feature_list=df_feature_selection_res.index.tolist()
        df_corr=df_sample.corr()
        feature_kept=[]
        feature_dropped=[]

        for feature in feature_list:
            if feature in feature_dropped:
                continue
            fea_corr=df_corr[feature]
            feature_kept+=[feature]
            feature_dropped+=fea_corr.loc[(fea_corr<1)&(fea_corr.abs()>=corr_threshold)].index.tolist()

        return feature_kept

    @staticmethod
    def easyensemble(para):
        lr=LogisticRegression(penalty='l2',tol=para[0],C=para[1],solver='lbfgs',max_iter=para[2])
        ada=AdaBoostClassifier(lr,n_estimators=para[3],algorithm='SAMME.R')
        eec=EasyEnsembleClassifier(n_estimators=para[4],base_estimator=ada,replacement=True)

        return eec

    @staticmethod
    def grid_search_parameters():
        lr_tol=[1e-3]
        lr_c=[0.5]
        lr_mi=[100]
        ada_n=[10]
        eec_n=[20]
        s_pct=[0.5]

        grid_para=[]
        iter=itertools.product(lr_tol,lr_c,lr_mi,ada_n,eec_n,s_pct)
        for i in iter:
            grid_para.append(list(i))

        return grid_para

    def parameter_tuning(self,grid_para):
        df_training_data = self.df_sample_data.loc[self.df_sample_data['sample_date'].isin(self.train_set_datelist)].copy()
        df_validation_data = self.df_sample_data.loc[self.df_sample_data['sample_date'].isin(self.validation_set_datelist)].copy()
        df_training_data_preprocessed = df_training_data.drop(self.qualitative_feature + ['label','sample_date'], axis=1).copy()
        df_validation_data_preprocessed = df_validation_data.drop(self.qualitative_feature + ['label','sample_date'], axis=1).copy()

        # data preprocessing
        imp_median=SimpleImputer(strategy='median')
        scaler=StandardScaler()

        df_training_data_preprocessed=pd.DataFrame(imp_median.fit_transform(df_training_data_preprocessed),
                                                   index=df_training_data_preprocessed.index,
                                                   columns=df_training_data_preprocessed.columns)

        df_training_data_preprocessed = pd.DataFrame(scaler.fit_transform(df_training_data_preprocessed),
                                                     index=df_training_data_preprocessed.index,
                                                     columns=df_training_data_preprocessed.columns)

        df_validation_data_preprocessed = pd.DataFrame(imp_median.transform(df_validation_data_preprocessed),
                                                     index=df_validation_data_preprocessed.index,
                                                     columns=df_validation_data_preprocessed.columns)

        df_validation_data_preprocessed = pd.DataFrame(scaler.transform(df_validation_data_preprocessed),
                                                       index=df_validation_data_preprocessed.index,
                                                       columns=df_validation_data_preprocessed.columns)

        df_training_data_preprocessed[self.qualitative_feature + ['label']]=df_training_data[self.qualitative_feature + ['label']]
        df_validation_data_preprocessed[self.qualitative_feature + ['label']]=df_validation_data[self.qualitative_feature + ['label']]

        df_grid_search_result=pd.DataFrame()
        ba=0
        selected_para=[]
        for i, para in enumerate(grid_para):
            selected_features=self.all_features[:int(len(self.all_features)*para[5])]

            X_train=df_training_data_preprocessed.loc[~df_training_data_preprocessed['label'].isna(),selected_features].values
            Y_train=df_training_data_preprocessed.loc[~df_training_data_preprocessed['label'].isna(),'label'].values

            X_validation = df_validation_data_preprocessed.loc[~df_validation_data_preprocessed['label'].isna(), selected_features].values
            Y_validation = df_validation_data_preprocessed.loc[~df_validation_data_preprocessed['label'].isna(), 'label'].values

            eec=FinancialRiskModel.easyensemble(para)
            eec.fit(X_train,Y_train)

            Y_pred=eec.predict(X_validation)

            precision,recall,f_score,support=precision_recall_fscore_support(Y_validation,Y_pred)
            ballanced_accuracy=balanced_accuracy_score(Y_validation,Y_pred)
            print()

            if ba<ballanced_accuracy:
                ba=ballanced_accuracy
                selected_para=para

            # res={'lr_tol':para[0],'lr_c':para[1],'lr_mi':para[2],'ada_n':para[3],'eec_n':para[4],'s_pct':para[5],
            #      'precision0':precision[0],'percision1':precision[1],'recall0':recall[0],'recall1':recall[1],
            #      'fscore0':f_score[0],'fscore1':f_score[1],'ballanced_accuracy':ballanced_accuracy}
            # print(res)
            # df_grid_search_result[i]=pd.Series(res)

        # df_grid_search_result.T.to_csv('/home/guosong/PycharmProjects/EquityFundamentalModel/data/parameter_tuning/grid_search_result_new.csv',index=False)
        self.parameter=selected_para

        return

    def model_training(self):
        df_training_data = self.df_sample_data.loc[self.df_sample_data['sample_date'].isin(self.sample_date_list)].copy() # notice here
        df_training_data_preprocessed = df_training_data.drop(self.qualitative_feature + ['label','sample_date'], axis=1).copy()

        # data preprocessing
        df_training_data_preprocessed = pd.DataFrame(self.imp_median.fit_transform(df_training_data_preprocessed),
                                                     index=df_training_data_preprocessed.index,
                                                     columns=df_training_data_preprocessed.columns)

        df_training_data_preprocessed = pd.DataFrame(self.scaler.fit_transform(df_training_data_preprocessed),
                                                     index=df_training_data_preprocessed.index,
                                                     columns=df_training_data_preprocessed.columns)

        df_training_data_preprocessed[self.qualitative_feature + ['label']] = df_training_data[self.qualitative_feature + ['label']]

        selected_features = self.all_features[:int(len(self.all_features) * self.parameter[5])]
        X_train = df_training_data_preprocessed.loc[~df_training_data_preprocessed['label'].isna(), selected_features].values
        Y_train = df_training_data_preprocessed.loc[~df_training_data_preprocessed['label'].isna(), 'label'].values
        X_train_unlabeled = df_training_data_preprocessed.loc[df_training_data_preprocessed['label'].isna(), selected_features].values

        eec = FinancialRiskModel.easyensemble(self.parameter)
        tt=TriTraining(eec)
        tt.fit(X_train,Y_train,X_train_unlabeled)

        self.model=tt

        return

    def model_prediction(self,df_sample):
        df_sample_preprocessed = df_sample.drop(self.qualitative_feature, axis=1).copy()
        df_ticker_name=DataLoader().load_ticker_name(df_sample.index.tolist()).set_index('ticker')
        df_industry=DataLoader().load_ticker_industry_pit(self.period_date,df_sample.index.tolist())
        df_industry=df_industry[['ticker','industryname1','industryname2','industryname3']].set_index('ticker')
        df_ticker_name_pit=DataLoader().load_ticker_name_pit(df_sample.index.tolist(),self.period_date).set_index('ticker')
        df_ticker_name_pit=df_ticker_name_pit['tickername']
        pit_st_list=df_ticker_name_pit.loc[df_ticker_name_pit.map(lambda x: ('ST' in x) | ('退' in x))].index.tolist()


        # data preprocessing
        df_sample_preprocessed = pd.DataFrame(self.imp_median.transform(df_sample_preprocessed),
                                                     index=df_sample_preprocessed.index,
                                                     columns=df_sample_preprocessed.columns)

        df_sample_preprocessed = pd.DataFrame(self.scaler.transform(df_sample_preprocessed),
                                                     index=df_sample_preprocessed.index,
                                                     columns=df_sample_preprocessed.columns)

        df_sample_preprocessed[self.qualitative_feature] = df_sample[self.qualitative_feature]

        selected_features = self.all_features[:int(len(self.all_features) * self.parameter[5])]
        X = df_sample_preprocessed[selected_features].values
        Y_pred=self.model.predict(X)
        Y_pred_proba=self.model.predict_proba(X)

        pred_result=pd.DataFrame(index=df_sample.index)
        pred_result['ticker_name']=df_ticker_name['secshortname']
        pred_result['ticker_score'] = Y_pred_proba[:, 0]
        pred_result['pred']=Y_pred
        pred_result[['industryname1', 'industryname2', 'industryname3']] = df_industry[
            ['industryname1', 'industryname2', 'industryname3']]

        pred_result.loc[pred_result.index.isin(pit_st_list),'pred']=1
        score0=pred_result.loc[pred_result['pred']==0,'ticker_score']
        score1=pred_result.loc[pred_result['pred']==1,'ticker_score']
        score0/=score0.std()
        score1/=score1.std()
        score0=(score0-score0.min())*70/(score0.max()-score0.min())+30
        score1=(score1-score1.min())*20/(score1.max()-score1.min())+9.9
        pred_result.loc[pred_result['pred'] == 0, 'ticker_score']=score0
        pred_result.loc[pred_result['pred'] == 1, 'ticker_score']=score1

        pred_result.sort_values('ticker_score',ascending=False,inplace=True)

        return pred_result

    def run(self):
        self.feature_selection()
        self.model_training()
        df_sample = pd.read_csv('/home/guosong/PycharmProjects/EquityFundamentalModel/data/preprocessed_feature/preprocessed_feature_{}.csv'.format(self.period_date))
        df_sample['ticker'] = df_sample['ticker'].map(lambda x: str(x).zfill(6))
        df_sample.set_index('ticker', inplace=True)

        pred_result=self.model_prediction(df_sample)

        pred_result.to_excel('/home/guosong/PycharmProjects/EquityFundamentalModel/data/results/financial_score_{}.xlsx'.format(self.period_date),index=True)

        pass



if __name__=='__main__':
    period_date='2021-06-30'
    FRM=FinancialRiskModel(period_date)
    # FRM.feature_selection()
    # FRM.feature_decorrelation()
    # grid_para=FRM.grid_search_parameters()
    # FRM.parameter_tuning(grid_para)
    FRM.run()
    # print()
    # date_begin='2016-01-31'
    # date_end='2019-01-31'
    # date_list=creat_date_list_monthend(date_begin, date_end)
    # for period_date in date_list:
    #     print(period_date)
    #     FRM = FinancialRiskModel(period_date)
    #     FRM.run()