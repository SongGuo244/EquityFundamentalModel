import pandas as pd
import numpy as np
import math
from datalayer.DataLoader import DataLoader
from datetime import datetime


class CalTools:
    @staticmethod
    def cal_gmean(sdata, nyears=5):
        try:
            ns=nyears*4
            if len(sdata)<12:
                return np.nan
            cdata=sdata.iloc[-ns:].copy()+1

            if cdata.min()<0: # negetive value in series, return min
                mean=cdata.min()-1
            else:
                mean = cdata.cumprod().iloc[-1]
                mean = math.pow(mean,1.0/len(cdata))-1

            return mean

        except:
            return np.nan

    @staticmethod
    def cal_amean(sdata, nyears=5):
        try:
            ns = nyears * 4
            cdata = sdata.iloc[-ns:].copy()

            return cdata.mean()

        except:
            return np.nan

    @staticmethod
    def cal_compgrowth(sdata, nyears=5):
        try:
            ns = nyears * 4
            if len(sdata) < 12:
                return np.nan

            cdata = sdata.iloc[-ns:].copy()

            cdata = cdata/cdata.shift(periods=4)+1
            cdata = cdata.iloc[4:]

            if cdata.min() < 0:  # negetive value in series, return min
                mean = cdata.min() - 1
            else:
                mean = cdata.cumprod().iloc[-1]
                mean = math.pow(mean, 1.0 / len(cdata)) - 1

            return mean

        except:
            return np.nan

    @staticmethod
    def cal_stability(sdata,nyears=5):
        try:
            ns = nyears * 4
            if len(sdata) < 12:
                return np.nan
            cdata = sdata.iloc[-ns:].copy()
            sta=cdata.mean()/cdata.std()

            return sta
        except:
            return np.nan

    @staticmethod
    def cal_inv_std(sdata, nyears=5):
        try:
            ns = nyears * 4
            if len(sdata) < 12:
                return np.nan
            cdata = sdata.iloc[-ns:].copy()
            sta = 1.0 / cdata.std()

            return sta
        except:
            return np.nan


def avesplit(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


class CalFinancialFeature:
    def __init__(self, period_date):
        self.period_date = period_date
        ticker_list=DataLoader().load_list_ticker_pit(self.period_date)
        self.df_ticker_industry=DataLoader().load_ticker_industry_pit(self.period_date,ticker_list)
        self.all_ticker_list=self.df_ticker_industry['ticker'].tolist()
        self.all_ticker_list.sort()

        pass

    def cal_all_tickers_feature(self):
        df_all_financial_features=pd.DataFrame()
        ticker_iter=avesplit(self.all_ticker_list,300) # batch load the data

        for ticker_list in ticker_iter:
            df_all_bs_data = DataLoader().load_balance_sheet_data_pit(ticker_list, self.period_date)
            df_all_is_ttm_data = DataLoader().load_income_statement_data_pit_ttm(ticker_list, self.period_date)
            df_all_cf_ttm_data = DataLoader().load_cash_flow_data_pit_ttm(ticker_list, self.period_date)
            df_all_financial_der_pit=DataLoader().load_financial_derivative_data_pit(ticker_list,self.period_date)

            df_all_bs_data = df_all_bs_data.drop(['secid', 'partyid', 'exchangecd', 'actpubtime', 'enddaterep', 'mergedflag',
                                                  'accoutingstandards', 'currencycd', 'updatetime'], axis=1)
            df_all_is_ttm_data = df_all_is_ttm_data.drop(['secid', 'secshortname', 'partyid', 'exchangecd', 'isnew', 'iscalc'], axis=1)
            df_all_cf_ttm_data = df_all_cf_ttm_data.drop(['secid', 'secshortname', 'partyid', 'exchangecd', 'isnew', 'iscalc'], axis=1)

            print(self.period_date,ticker_list)
            # cal financial features for single ticker
            for ticker in ticker_list:
                try:
                    df_bs = df_all_bs_data.loc[df_all_bs_data['ticker'] == ticker].copy()
                    df_is = df_all_is_ttm_data.loc[df_all_is_ttm_data['ticker'] == ticker].copy()
                    df_cf = df_all_cf_ttm_data.loc[df_all_cf_ttm_data['ticker'] == ticker].copy()
                    df_fd = df_all_financial_der_pit.loc[df_all_financial_der_pit['ticker'] == ticker].copy()

                    df_financial_data = pd.merge(df_bs, df_is, how='inner', on=['ticker', 'enddate', 'publishdate'])
                    df_financial_data = pd.merge(df_financial_data, df_cf, how='inner',
                                                 on=['ticker', 'enddate', 'publishdate'])
                    df_financial_data = pd.merge(df_financial_data, df_fd, how='inner',
                                                 on=['ticker', 'enddate', 'publishdate'])

                    df_financial_data['enddate'] = df_financial_data['enddate'].map(
                        lambda x: datetime.strptime(x, '%Y-%m-%d').date())
                    df_financial_data['publishdate'] = df_financial_data['publishdate'].map(
                        lambda x: datetime.strptime(x, '%Y-%m-%d').date())

                    df_financial_data = df_financial_data.sort_values(['enddate', 'publishdate'], ascending=True)
                    df_financial_data = df_financial_data.drop_duplicates('enddate', keep='last')
                    df_financial_data = df_financial_data.set_index('enddate')

                    if df_financial_data.shape[0] < 12:
                        print('financial data for {!r} is not enough'.format(ticker))
                        continue

                    df_financial_data.loc[df_financial_data['reporttype'] != 'A', 'da'] = np.nan
                    df_financial_data['da'] = df_financial_data['da'].fillna(method='ffill')

                    df_financial_data.fillna(0, inplace=True)

                    df_financial_features=self.cal_single_ticker_feature(ticker,df_financial_data)

                    df_all_financial_features=pd.concat([df_all_financial_features,df_financial_features],axis=0)

                except:
                    print('calculate feature data for {!r} err'.format(ticker))
                    continue

        return df_all_financial_features


    def cal_single_ticker_feature(self,ticker, df_financial_data):
        # prepare financial data for features calculation

        # EBIAT = 净利润 + 净利息费用 *（1 - 税率）
        df_financial_data['EBIAT']=df_financial_data['nincome']+df_financial_data['finanexp']*0.75
        # EBIT = 利润总额 + 净利息费用
        df_financial_data['EBIT']=df_financial_data['tprofit']+df_financial_data['finanexp']
        # 毛利
        df_financial_data['毛利']=df_financial_data['revenue']-df_financial_data['cogs']

        df_financial_data['应收款项']=df_financial_data['notesreceiv']+df_financial_data['ar']
        df_financial_data['其他应收款项']=df_financial_data['othreceiv']+df_financial_data['divreceiv']+df_financial_data['intreceiv']\
                                    +df_financial_data['reinsurreserreceiv']+df_financial_data['reinsurreceiv']+df_financial_data['premiumreceiv']


        # financial features calculation
        df_financial_features=pd.DataFrame(index=df_financial_data.index)
        df_financial_features['ticker']=ticker
        # ROA=净利润*2/(期初总资产+期末总资产)
        df_financial_features['ROA']=df_financial_data['nincome']*4/(df_financial_data['tassets']
                                                                   +df_financial_data['tassets'].shift(1)
                                                                   +df_financial_data['tassets'].shift(2)
                                                                   +df_financial_data['tassets'].shift(3))
        df_financial_features['ROA'].loc[df_financial_features['ROA'].isna()] = df_financial_data['nincome'] / df_financial_data['tassets']
        df_financial_features['geo_mean(ROA)']=CalTools.cal_gmean(df_financial_features['ROA'])
        df_financial_features['ave/std(ROA)']=CalTools.cal_stability(df_financial_features['ROA'])
        # ROE(平均) = 归属于母公司的净利润 * 2 /（期末归属于母公司的所有者权益 + 期初归属于母公司的所有者权益）
        df_financial_features['ROE'] = df_financial_data['nincomeattrp'] * 4 / (df_financial_data['tequityattrp']
                                                                           + df_financial_data['tequityattrp'].shift(1)
                                                                           + df_financial_data['tequityattrp'].shift(2)
                                                                           + df_financial_data['tequityattrp'].shift(3))
        df_financial_features['ROE'].loc[df_financial_features['ROE'].isna()] = df_financial_data['nincomeattrp'] / \
                                                                                df_financial_data['tequityattrp']
        df_financial_features['geo_mean(ROE)'] = CalTools.cal_gmean(df_financial_features['ROE'])
        df_financial_features['ave/std(ROE)'] = CalTools.cal_stability(df_financial_features['ROE'])
        # ROIC = EBIAT * 2 / (期初投入资本 + 期末投入资本)
        df_financial_features['ROIC']=df_financial_data['EBIAT']* 4 / (df_financial_data['ic']
                                                                       + df_financial_data['ic'].shift(1)
                                                                       + df_financial_data['ic'].shift(2)
                                                                       + df_financial_data['ic'].shift(3))
        df_financial_features['ROIC'].loc[df_financial_features['ROIC'].isna()] = df_financial_data['EBIAT'] / \
                                                                                df_financial_data['ic']
        df_financial_features['geo_mean(ROIC)'] = CalTools.cal_gmean(df_financial_features['ROIC'])
        df_financial_features['ave/std(ROIC)'] = CalTools.cal_stability(df_financial_features['ROIC'])
        # 销售毛利率=（营业收入-营业成本）/营业收入
        df_financial_features['毛利率']=(df_financial_data['revenue']-df_financial_data['cogs'])/df_financial_data['revenue']
        df_financial_features['geo_mean(毛利率)'] = CalTools.cal_gmean(df_financial_features['毛利率'])
        df_financial_features['ave/std(毛利率)'] = CalTools.cal_stability(df_financial_features['毛利率'])
        # 营业利润率
        df_financial_features['营业利润率'] = df_financial_data['operateprofit'] / df_financial_data['revenue']
        df_financial_features['geo_mean(营业利润率)'] = CalTools.cal_gmean(df_financial_features['营业利润率'])
        df_financial_features['ave/std(营业利润率)'] = CalTools.cal_stability(df_financial_features['营业利润率'])
        # 销售净利率 = 净利润 / 营业收入
        df_financial_features['净利率']=df_financial_data['nincome'] / df_financial_data['revenue']
        df_financial_features['geo_mean(净利率)'] = CalTools.cal_gmean(df_financial_features['净利率'])
        df_financial_features['ave/std(净利率)'] = CalTools.cal_stability(df_financial_features['净利率'])
        # 息税前利润率
        df_financial_features['息税前利润率']=df_financial_data['EBIT']/df_financial_data['revenue']
        df_financial_features['geo_mean(息税前利润率)'] = CalTools.cal_gmean(df_financial_features['息税前利润率'])
        df_financial_features['ave/std(息税前利润率)'] = CalTools.cal_stability(df_financial_features['息税前利润率'])
        # 净资产现金回收率=经营现金流入净额*2/(期初净资产+期末净资产)
        df_financial_features['净资产现金回收率'] = df_financial_data['ncfoperatea'] * 4 / (df_financial_data['tshequity']
                                                                          + df_financial_data['tshequity'].shift(1)
                                                                          + df_financial_data['tshequity'].shift(2)
                                                                          + df_financial_data['tshequity'].shift(3))
        df_financial_features['净资产现金回收率'].loc[df_financial_features['净资产现金回收率'].isna()] = df_financial_data['ncfoperatea'] / \
                                                                                  df_financial_data['tshequity']
        df_financial_features['geo_mean(净资产现金回收率)'] = CalTools.cal_gmean(df_financial_features['净资产现金回收率'])
        df_financial_features['ave/std(净资产现金回收率)'] = CalTools.cal_stability(df_financial_features['净资产现金回收率'])
        # 总资产现金回收率=经营现金流入净额*2/(期初总资产+期末总资产)
        df_financial_features['总资产现金回收率'] = df_financial_data['ncfoperatea'] * 4 / (df_financial_data['tassets']
                                                                                    + df_financial_data['tassets'].shift(1)
                                                                                    + df_financial_data['tassets'].shift(2)
                                                                                    + df_financial_data['tassets'].shift(3))
        df_financial_features['总资产现金回收率'].loc[df_financial_features['总资产现金回收率'].isna()] = df_financial_data['ncfoperatea'] / \
                                                                                          df_financial_data['tassets']
        df_financial_features['geo_mean(总资产现金回收率)'] = CalTools.cal_gmean(df_financial_features['总资产现金回收率'])
        df_financial_features['ave/std(总资产现金回收率)'] = CalTools.cal_stability(df_financial_features['总资产现金回收率'])

        df_financial_features['YOY(营业收入)']=df_financial_data['revenue'].pct_change(periods=4)
        df_financial_features['Comp_Growth(营业收入)']=CalTools.cal_compgrowth(df_financial_data['revenue'])

        df_financial_features['YOY(总资产)'] = df_financial_data['tassets'].pct_change(periods=4)
        df_financial_features['Comp_Growth(总资产)'] = CalTools.cal_compgrowth(df_financial_data['tassets'])

        df_financial_features['YOY(净资产)'] = df_financial_data['tshequity'].pct_change(periods=4)
        df_financial_features['Comp_Growth(净资产)'] = CalTools.cal_compgrowth(df_financial_data['tshequity'])

        df_financial_features['YOY(毛利)'] = df_financial_data['毛利'].pct_change(periods=4)
        df_financial_features['Comp_Growth(毛利)'] = CalTools.cal_compgrowth(df_financial_data['毛利'])

        df_financial_features['YOY(营业利润)'] = df_financial_data['operateprofit'].pct_change(periods=4)
        df_financial_features['Comp_Growth(营业利润)'] = CalTools.cal_compgrowth(df_financial_data['operateprofit'])

        df_financial_features['YOY(净利润)'] = df_financial_data['nincome'].pct_change(periods=4)
        df_financial_features['Comp_Growth(净利润)'] = CalTools.cal_compgrowth(df_financial_data['nincome'])

        df_financial_features['YOY(ROA)'] = df_financial_features['ROA'].pct_change(periods=4)
        df_financial_features['Comp_Growth(ROA)'] = CalTools.cal_compgrowth(df_financial_features['ROA'])

        df_financial_features['YOY(ROE)'] = df_financial_features['ROE'].pct_change(periods=4)
        df_financial_features['Comp_Growth(ROE)'] = CalTools.cal_compgrowth(df_financial_features['ROE'])

        df_financial_features['YOY(ROIC)'] = df_financial_features['ROIC'].pct_change(periods=4)
        df_financial_features['Comp_Growth(ROIC)'] = CalTools.cal_compgrowth(df_financial_features['ROIC'])

        df_financial_features['YOY(毛利率)'] = df_financial_features['毛利率'].pct_change(periods=4)
        df_financial_features['Comp_Growth(毛利率)'] = CalTools.cal_compgrowth(df_financial_features['毛利率'])

        df_financial_features['YOY(营业利润率)'] = df_financial_features['营业利润率'].pct_change(periods=4)
        df_financial_features['Comp_Growth(营业利润率)'] = CalTools.cal_compgrowth(df_financial_features['营业利润率'])

        df_financial_features['YOY(净利率)'] = df_financial_features['净利率'].pct_change(periods=4)
        df_financial_features['Comp_Growth(净利率)'] = CalTools.cal_compgrowth(df_financial_features['净利率'])

        df_financial_features['YOY(息税前利润率)'] = df_financial_features['息税前利润率'].pct_change(periods=4)
        df_financial_features['Comp_Growth(息税前利润率)'] = CalTools.cal_compgrowth(df_financial_features['息税前利润率'])

        df_financial_features['YOY(净资产现金回收率)'] = df_financial_features['净资产现金回收率'].pct_change(periods=4)
        df_financial_features['Comp_Growth(净资产现金回收率)'] = CalTools.cal_compgrowth(df_financial_features['净资产现金回收率'])

        # 存货周转率 = 营业成本 * 2 / (期初存货 + 期末存货)
        df_financial_features['存货周转率'] = df_financial_data['cogs'] * 4 / (df_financial_data['inventories']
                                                                          + df_financial_data['inventories'].shift(1)
                                                                          + df_financial_data['inventories'].shift(2)
                                                                          + df_financial_data['inventories'].shift(3))
        df_financial_features['存货周转率'].loc[df_financial_features['存货周转率'].isna()] = df_financial_data['cogs'] / \
                                                                                          df_financial_data['inventories']

        # 应收账款周转率 = 营业收入 * 2 / (期初应收账款 + 期末应收账款)
        df_financial_features['应收账款周转率']= df_financial_data['revenue'] * 4 / (df_financial_data['应收款项']
                                                                          + df_financial_data['应收款项'].shift(1)
                                                                          + df_financial_data['应收款项'].shift(2)
                                                                          + df_financial_data['应收款项'].shift(3))
        df_financial_features['应收账款周转率'].loc[df_financial_features['应收账款周转率'].isna()] = df_financial_data['revenue'] / \
                                                                                          df_financial_data['应收款项']

        # 总资产周转率 = 营业收入 * 2 / (期初总资产 + 期末总资产)
        df_financial_features['总资产周转率'] = df_financial_data['revenue'] * 4 / (df_financial_data['tassets']
                                                                               + df_financial_data['tassets'].shift(1)
                                                                               + df_financial_data['tassets'].shift(2)
                                                                               + df_financial_data['tassets'].shift(3))
        df_financial_features['总资产周转率'].loc[df_financial_features['总资产周转率'].isna()] = df_financial_data['revenue'] / \
                                                                                        df_financial_data['tassets']

        # 固定资产合计周转率 = 营业收入 * 2 / (期初固定资产合计 + 期末固定资产合计)
        df_financial_features['固定资产合计周转率'] = df_financial_data['revenue'] * 4 / (df_financial_data['tfixedassets']
                                                                              + df_financial_data['tfixedassets'].shift(1)
                                                                              + df_financial_data['tfixedassets'].shift(2)
                                                                              + df_financial_data['tfixedassets'].shift(3))
        df_financial_features['固定资产合计周转率'].loc[df_financial_features['固定资产合计周转率'].isna()] = df_financial_data['revenue'] / \
                                                                                      df_financial_data['tfixedassets']

        # 流动资产周转率 = 营业收入 * 2 / (期初流动资产 + 期末流动资产)
        df_financial_features['流动资产周转率'] = df_financial_data['revenue'] * 4 / (df_financial_data['tca']
                                                                               + df_financial_data['tca'].shift(1)
                                                                               + df_financial_data['tca'].shift(2)
                                                                               + df_financial_data['tca'].shift(3))
        df_financial_features['流动资产周转率'].loc[df_financial_features['流动资产周转率'].isna()] = df_financial_data['revenue'] / \
                                                                                            df_financial_data['tca']

        df_financial_features['现金及现金等价物/带息流动负债']=df_financial_data['nceendbal']/df_financial_data['intcl']
        # 流动比率 = 流动资产 / 流动负债
        df_financial_features['流动比率'] = df_financial_data['tca'] / df_financial_data['tcl']
        # 速动比率 =（流动资产 - 存货） / 流动负债
        df_financial_features['速动比率'] = (df_financial_data['tca']-df_financial_data['inventories']) / df_financial_data['tcl']
        # 资产负债率 = 负债总计 / 资产总计
        df_financial_features['资产负债率']=df_financial_data['tliab']/df_financial_data['tassets']
        # 产权比率=总负债 / (所有者权益 - 无形资产)
        df_financial_features['产权比率']=df_financial_data['tliab']/(df_financial_data['tshequity']-df_financial_data['intanassets'])
        # 债务保障率=经营活动现金流量净额 / 负债合计
        df_financial_features['债务保障率']=df_financial_data['ncfoperatea']/df_financial_data['tliab']

        df_financial_features['三费费用/营业收入']=(df_financial_data['sellexp']+df_financial_data['adminexp']+df_financial_data['finanexp'])/df_financial_data['revenue']
        df_financial_features['Mean(三费费用/营业收入)']=CalTools.cal_amean(df_financial_features['三费费用/营业收入'])
        df_financial_features['Inv_Std(三费费用/营业收入)']=CalTools.cal_inv_std(df_financial_features['三费费用/营业收入'])
        df_financial_features['diff(流动比率,速动比率)']=df_financial_features['流动比率']-df_financial_features['速动比率']

        # 单位销售现金净流入=经营活动产生的净现金流/营业收入
        df_financial_features['单位销售现金净流入']=df_financial_data['ncfoperatea']/df_financial_data['revenue']

        df_financial_features['应收账款/营业收入']=df_financial_data['应收款项']/df_financial_data['revenue']
        df_financial_features['Mean(应收账款/营业收入)']=CalTools.cal_amean(df_financial_features['应收账款/营业收入'])

        df_financial_features['其他应收款/总资产'] = df_financial_data['其他应收款项'] / df_financial_data['tassets']
        df_financial_features['Mean(其他应收款/总资产)'] = CalTools.cal_amean(df_financial_features['其他应收款/总资产'])

        df_financial_features['长期待摊费用/总资产']=df_financial_data['ltamorexp'] / df_financial_data['tassets']
        df_financial_features['abs(资产减值损失)/营业收入']=df_financial_data['assetsimpairloss'].abs()/df_financial_data['revenue']
        df_financial_features['营业外收入/营业收入']=df_financial_data['noperateincome']/df_financial_data['revenue']
        df_financial_features['三费费用/毛利']=(df_financial_data['sellexp']+df_financial_data['adminexp']+df_financial_data['finanexp'])/df_financial_data['毛利']
        df_financial_features['Geo_Mean(三费费用/毛利)']=CalTools.cal_gmean(df_financial_features['三费费用/毛利'])

        # 销售商品、提供劳务收到的现金/营业收入
        df_financial_features['销现/营业收入']=df_financial_data['cfrsalegs']/df_financial_data['revenue']
        df_financial_features['Geo_Mean(销现/营业收入)']=CalTools.cal_gmean(df_financial_features['销现/营业收入'])
        df_financial_features['Ave/Std(销现/营业收入)']=CalTools.cal_stability(df_financial_features['销现/营业收入'])

        # 折旧与摊销/经营活动现金流量净额
        df_financial_features['折旧与摊销/经营现金净额']=df_financial_data['da']/df_financial_data['ncfoperatea']
        df_financial_features['Geo_Mean(折旧与摊销/经营现金净额)']=CalTools.cal_gmean(df_financial_features['折旧与摊销/经营现金净额'])
        df_financial_features['Ave/Std(折旧与摊销/经营现金净额)']=CalTools.cal_stability(df_financial_features['折旧与摊销/经营现金净额'])

        # 经营活动现金流量净额 / 净利润
        df_financial_features['经营现金净额/净利润']=df_financial_data['ncfoperatea']/df_financial_data['nincome'].abs()
        df_financial_features['Geo_Mean(经营现金净额/净利润)']=CalTools.cal_gmean(df_financial_features['经营现金净额/净利润'])
        df_financial_features['Ave/Std(经营现金净额/净利润)']=CalTools.cal_stability(df_financial_features['经营现金净额/净利润'])

        df_financial_features['营业成本率']=df_financial_data['cogs']/df_financial_data['revenue']
        df_financial_features['销售费用率']=df_financial_data['sellexp']/df_financial_data['revenue']

        df_financial_features['应收款项/总资产']=df_financial_data['应收款项']/df_financial_data['tassets']
        df_financial_features['Mean(应收款项/总资产)']=CalTools.cal_amean(df_financial_features['应收款项/总资产'])

        df_financial_features['固定资产合计占比']=df_financial_data['tfixedassets']/df_financial_data['tassets']
        df_financial_features['Ave/Std(在建工程/固定资产)']=CalTools.cal_stability((df_financial_data['cip']+df_financial_data['constmaterials'])/
                                                                           (df_financial_data['fixedassets']+df_financial_data['fixedassetsdisp']))

        df_financial_features['无形资产/总资产']=df_financial_data['intanassets']/df_financial_data['tassets']
        df_financial_features['商誉/总资产']=df_financial_data['goodwill']/df_financial_data['tassets']
        df_financial_features['diff(商誉/总资产)']=df_financial_features['商誉/总资产'].diff(periods=4)
        df_financial_features['有息负债/总资产'] = df_financial_data['intdebt'] / df_financial_data['tassets']

        df_financial_features['递延所得税负债/总负债']=df_financial_data['defertaxliab']/df_financial_data['tliab']
        df_financial_features['递延所得税资产/总资产']=df_financial_data['defertaxassets']/df_financial_data['tassets']

        df_financial_features['其他流动资产/流动资产']=df_financial_data['othca']/df_financial_data['tca']
        df_financial_features['其他非流动资产/非流动资产']=df_financial_data['othnca']/df_financial_data['tnca']

        df_financial_features['应收款项/货币资金']=df_financial_data['应收款项']/df_financial_data['cashcequiv']
        df_financial_features['diff(YOY(应收款项),YOY(营业收入))']=df_financial_data['应收款项'].pct_change(periods=4)-df_financial_data['revenue'].pct_change(periods=4)

        df_financial_features['预付款项/营业收入']=df_financial_data['prepayment']/df_financial_data['revenue']
        df_financial_features['Mean(预付款项/营业收入)']=CalTools.cal_amean(df_financial_features['预付款项/营业收入'])
        df_financial_features['Inv_Std(预付款项/营业收入)']=CalTools.cal_stability(df_financial_features['预付款项/营业收入'])

        df_financial_features['diff(预付款项/总资产)']=(df_financial_data['prepayment']/df_financial_data['tassets']).diff(periods=4)

        df_financial_features['其他应付款/总负债']=df_financial_data['othpayable']/df_financial_data['tliab']
        df_financial_features['diff(YOY(存货),YOY(营业成本))']=df_financial_data['inventories'].pct_change(periods=4)\
                                                         -df_financial_data['cogs'].pct_change(periods=4)
        df_financial_features['diff(YOY(长期待摊费用),YOY(营业收入))'] = df_financial_data['ltamorexp'].pct_change(periods=4) \
                                                               - df_financial_data['revenue'].pct_change(periods=4)

        df_financial_features['diff(YOY(管理费用),YOY(营业收入))'] = df_financial_data['adminexp'].pct_change(periods=4) \
                                                               - df_financial_data['revenue'].pct_change(periods=4)

        df_financial_features['Inv_Std(毛利率)']=CalTools.cal_inv_std(df_financial_features['毛利率'])
        df_financial_features['diff((销售费用+管理费用)/营业收入)']=((df_financial_data['sellexp']+df_financial_data['adminexp'])/df_financial_data['revenue']).diff(periods=4)
        df_financial_features['diff(abs(资产减值损失)/营业收入)']=df_financial_features['abs(资产减值损失)/营业收入'].diff(periods=4)
        df_financial_features['(现金/总资产)*(有息负债/总负债)']=(df_financial_data['nceendbal']/df_financial_data['tassets'])*(df_financial_data['intdebt']/df_financial_data['tliab'])
        df_financial_features['diff(应收账款周转率)']=df_financial_features['应收账款周转率'].diff(periods=4)
        df_financial_features['diff(存货周转率)']=df_financial_features['存货周转率'].diff(periods=4)
        df_financial_features['diff(在建工程/总资产)']=((df_financial_data['cip']+df_financial_data['constmaterials'])/df_financial_data['tassets']).diff(periods=4)
        df_financial_features['Inv_Std(应付职工薪酬/总负债)']=CalTools.cal_inv_std(df_financial_data['payrollpayable']/df_financial_data['tliab'])
        df_financial_features['diff(递延所得税资产/总资产)']=df_financial_features['递延所得税资产/总资产'].diff(periods=4)
        df_financial_features['diff(Comp_growth(存货),Comp_growth(营业收入))']=CalTools.cal_compgrowth(df_financial_data['inventories'])-df_financial_features['Comp_Growth(营业收入)']
        df_financial_features['Inv_Std(固定资产合计占比)']=CalTools.cal_inv_std(df_financial_features['固定资产合计占比'])
        df_financial_features['Mean(营业外收入/营业收入)']=CalTools.cal_amean(df_financial_features['营业外收入/营业收入'])
        df_financial_features['diff(营业外收入/营业收入)']=df_financial_features['营业外收入/营业收入'].diff(periods=4)

        df_financial_features['企业现金流肖像+++']=0
        df_financial_features.loc[(df_financial_data['ncfoperatea']>0)&(df_financial_data['ncffrinvesta'] > 0)&(
                df_financial_data['ncffrfinana'] > 0),'企业现金流肖像+++']=1
        df_financial_features['企业现金流肖像++-']=0
        df_financial_features.loc[(df_financial_data['ncfoperatea'] > 0) & (df_financial_data['ncffrinvesta'] > 0) & (
                    df_financial_data['ncffrfinana'] < 0), '企业现金流肖像++-'] = 1
        df_financial_features['企业现金流肖像+--']=0
        df_financial_features.loc[(df_financial_data['ncfoperatea'] > 0) & (df_financial_data['ncffrinvesta'] < 0) & (
                    df_financial_data['ncffrfinana'] < 0), '企业现金流肖像+--'] = 1
        df_financial_features['企业现金流肖像+-+']=0
        df_financial_features.loc[(df_financial_data['ncfoperatea'] > 0) & (df_financial_data['ncffrinvesta'] < 0) & (
                    df_financial_data['ncffrfinana'] > 0), '企业现金流肖像+-+'] = 1
        df_financial_features['企业现金流肖像-++']=0
        df_financial_features.loc[(df_financial_data['ncfoperatea'] < 0) & (df_financial_data['ncffrinvesta'] > 0) & (
                    df_financial_data['ncffrfinana'] > 0), '企业现金流肖像-++'] = 1
        df_financial_features['企业现金流肖像-+-']=0
        df_financial_features.loc[(df_financial_data['ncfoperatea'] < 0) & (df_financial_data['ncffrinvesta'] > 0) & (
                    df_financial_data['ncffrfinana'] < 0), '企业现金流肖像-+-'] = 1
        df_financial_features['企业现金流肖像--+']=0
        df_financial_features.loc[(df_financial_data['ncfoperatea'] < 0) & (df_financial_data['ncffrinvesta'] < 0) & (
                    df_financial_data['ncffrfinana'] > 0), '企业现金流肖像--+'] = 1
        df_financial_features['企业现金流肖像---']=0
        df_financial_features.loc[(df_financial_data['ncfoperatea'] < 0) & (df_financial_data['ncffrinvesta'] < 0) & (
                    df_financial_data['ncffrfinana'] < 0), '企业现金流肖像---'] = 1

        df_financial_features['存货周转率下降&毛利率上升']=0
        df_financial_features.loc[(df_financial_features['存货周转率'].diff(periods=4)<0)&(df_financial_features['毛利率'].diff(periods=4)>0),'存货周转率下降&毛利率上升']=1

        df_financial_features.replace([np.inf,-np.inf],np.nan,inplace=True)

        return df_financial_features.iloc[-1:]


if __name__=='__main__':
    # period_date=datetime.today().date().strftime('%Y-%m-%d')
    period_date='2013-04-30'
    ticker='000408'
    CFF=CalFinancialFeature(period_date)
    CFF.cal_all_tickers_feature()
    # CFF.cal_single_ticker_feature(ticker)
    print()
