import os
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from pathlib import Path

class Simulator:
    def __init__(self):
        self.sample_submission = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'sample_submission.csv'))
        self.max_count = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'max_count.csv'))
        self.stock = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'stock.csv'))
        order = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'order.csv'), index_col=0)   
        order.index = pd.to_datetime(order.index)
        self.order = order
    
    # 원본 DF 복사 후 time column을 index로 바꾸고 해당 DF 반환
    def subprocess(self, df):
        out = df.copy()
        column = 'time'
        
        out.index = pd.to_datetime(out[column])
        out = out.drop([column], axis=1)
        out.index.name = column
        return out
    
    
    
    def get_state(self, data):       #-- 숫자가 있는 공정에 대해서 해당 숫자만 추출
        if 'CHECK' in data:
            return int(data[-1])     #-- 예시. CHECK_1 이면 1만 return
        elif 'CHANGE' in data:
            return int(data[-1])
        else:
            return np.nan            #-- PROCESS와 STOP event에 대해서는 nan
    
    #--- 원형 공정
    def cal_schedule_part_1(self, df):
        columns = ['PRT_1', 'PRT_2', 'PRT_3', 'PRT_4']
        df_set = df[columns]
        df_out = df_set * 0
        
        p = 0.985
        dt = pd.Timedelta(days=23)    #-- 투입 23일 후 생산 완료
        end_time = df_out.index[-1]

        for time in df_out.index:
            out_time = time + dt
            if end_time < out_time:  #-- 위 가정이 맞다면, 투입 23일 후가 end_time보다 크면 의미 없으니 break.
                break
            else:            
                for column in columns:
                    set_num = df_set.loc[time, column]   #-- 특정 날짜(시간)에 해당 PRT 투입량을 set_num으로 지정
                    if set_num > 0:
                        out_num = np.sum(np.random.choice(2, set_num, p=[1-p, p]))     #-- 양품률 98.5% 반영    
                        df_out.loc[out_time, column] = out_num   #-- 23일 후 df_out 데이터프레임에 out_num 입력
 
        df_out['MOL_1'] = 0.0
        df_out['MOL_2'] = 0.0
        df_out['MOL_3'] = 0.0
        df_out['MOL_4'] = 0.0
        df_out['BLK_1'] = 0.0
        df_out['BLK_2'] = 0.0
        df_out['BLK_3'] = 0.0
        df_out['BLK_4'] = 0.0
        return df_out    
    
    def cal_schedule_part_2(self, df, line='A'):
        if line == 'A':
            columns = ['Event_A', 'MOL_A']
        elif line == 'B':
            columns = ['Event_B', 'MOL_B']
        else:
            columns = ['Event_A', 'MOL_A']
            
        schedule = df[columns].copy()
        
        schedule['state'] = 0        #-- state column 생성
        schedule['state'] = schedule[columns[0]].apply(lambda x: self.get_state(x))  
        #-- state column에 get_state 값 넣기
        schedule['state'] = schedule['state'].fillna(method='ffill')
        schedule['state'] = schedule['state'].fillna(0)
        
        schedule_process = schedule.loc[schedule[columns[0]]=='PROCESS']   #-- PROCESS EVENT만 추출
        df_out = schedule.drop(schedule.columns, axis=1)      #-- EVENT와 MOL column 삭제
        df_out['PRT_1'] = 0.0 
        df_out['PRT_2'] = 0.0
        df_out['PRT_3'] = 0.0
        df_out['PRT_4'] = 0.0
        df_out['MOL_1'] = 0.0
        df_out['MOL_2'] = 0.0
        df_out['MOL_3'] = 0.0
        df_out['MOL_4'] = 0.0

        p = 0.975     #-- 성형 공정의 양품률
        times = schedule_process.index               #-- PROCESS 공정이 이뤄지는 시간(날짜)만 선택
        for i, time in enumerate(times):             #-- PROCESS 공정 상태에만 대하여
            value = schedule.loc[time, columns[1]]   #-- 해당 시점 MOL_X 투입 개수
            state = int(schedule.loc[time, 'state']) #-- 해당 시점의 상태
            df_out.loc[time, 'PRT_'+str(state)] = -value
            if i+48 < len(times):                    #-- 최소 공정 시간 48시간 조건
                out_time = times[i+48]
                df_out.loc[out_time, 'MOL_'+str(state)] = value*p #-- 48시간 후 MOL_X의 개수

        df_out['BLK_1'] = 0.0
        df_out['BLK_2'] = 0.0
        df_out['BLK_3'] = 0.0
        df_out['BLK_4'] = 0.0
        return df_out

    # 이걸 보고 있으니 sample_submission과 같은 index (time)을 가진 보유현황(재고) df를 하나 더 만드는 것 같다.
    def cal_stock(self, df, df_order):
        df_stock = df * 0

        blk2mol = {}
        blk2mol['BLK_1'] = 'MOL_1'
        blk2mol['BLK_2'] = 'MOL_2'
        blk2mol['BLK_3'] = 'MOL_3'
        blk2mol['BLK_4'] = 'MOL_4'

        cut = {}            #-- MOL_X 당 생성되는 BLK_X 수
        cut['BLK_1'] = 506
        cut['BLK_2'] = 506
        cut['BLK_3'] = 400
        cut['BLK_4'] = 400

        p = {}
        p['BLK_1'] = 0.851
        p['BLK_2'] = 0.901
        blk_diffs = []
        for i, time in enumerate(df.index):
            month = time.month
            if month == 4:
                p['BLK_3'] = 0.710
                p['BLK_4'] = 0.700        
            elif month == 5:
                p['BLK_3'] = 0.742
                p['BLK_4'] = 0.732
            elif month == 6:
                p['BLK_3'] = 0.759
                p['BLK_4'] = 0.749
            else:
                p['BLK_3'] = 0.0
                p['BLK_4'] = 0.0
                
            if i == 0:
                df_stock.iloc[i] = df.iloc[i]    
            else:
                df_stock.iloc[i] = df_stock.iloc[i-1] + df.iloc[i]
                for column in df_order.columns:           #-- order.csv의 각 column에 대하여
                    val = df_order.loc[time, column]      #-- 해당 블럭의 수요 (BLK_X)
                    if val > 0:
                        mol_col = blk2mol[column]               #-- 그 블럭에 해당하는 몰드 선택
                        mol_num = df_stock.loc[time, mol_col]   #-- 그 몰드가 몇 개 있는지
                        df_stock.loc[time, mol_col] = 0         #-- 왜 선택 날짜의 몰드 값을 0으로 만드냐...
                        
                        blk_gen = int(mol_num*p[column]*cut[column])     #-- 몰드*몰드당 블럭수*양품률
                        blk_stock = df_stock.loc[time, column] + blk_gen #-- 선택 날짜의 블럭 보유량에 생산량 더하기
                        blk_diff = blk_stock - val    #-- 부족/초과분 = 보유량 - 주문량
                        
                        df_stock.loc[time, column] = blk_diff  #-- 해당 날짜에 부족/초과분
                        blk_diffs.append(blk_diff)             #-- 부족/초과분 점수 계산을 위해서 list에 추가
        return df_stock, blk_diffs    

    def subprocess(self, df):  #-- 원본 DF 복사 후 time column을 index로 바꾸고 해당 DF 반환
        out = df.copy()
        column = 'time'

        out.index = pd.to_datetime(out[column])
        out = out.drop([column], axis=1)
        out.index.name = column
        return out
    
    def add_stock(self, df, df_stock):   
        #-- sample_submission과 같은 DF 생성후 첫 행(시작날짜, 2020-04-01 00:00)에 보유량(stock.csv 값) 저장
        df_out = df.copy()
        for column in df_out.columns:
            df_out.iloc[0][column] = df_out.iloc[0][column] + df_stock.iloc[0][column]
        return df_out

    def order_rescale(self, df, df_order):    
        #-- 일자별로 작성된 order.csv 파일을 24시간 기준으로 변경 (오후 6시에 납품이 이뤄지도록)
        df_rescale = df.drop(df.columns, axis=1)
        dt = pd.Timedelta(hours=18)
        for column in ['BLK_1', 'BLK_2', 'BLK_3', 'BLK_4']:
            for time in df_order.index:
                df_rescale.loc[time+dt, column] = df_order.loc[time, column]
        df_rescale = df_rescale.fillna(0)
        return df_rescale


    def cal_stock(self, df, df_order):
        df_stock = df * 0

        blk2mol = {}
        blk2mol['BLK_1'] = 'MOL_1'
        blk2mol['BLK_2'] = 'MOL_2'
        blk2mol['BLK_3'] = 'MOL_3'
        blk2mol['BLK_4'] = 'MOL_4'

        cut = {} # MOL_X 당 생성되는 BLK_X 수
        cut['BLK_1'] = 506
        cut['BLK_2'] = 506
        cut['BLK_3'] = 400
        cut['BLK_4'] = 400

        p = {} # 자르기 공정의 양품률
        p['BLK_1'] = 0.851
        p['BLK_2'] = 0.901
        blk_diffs = []
        for i, time in enumerate(df.index):
            month = time.month # BLK_3, BLK_4의 월별 자르기 양품률 반영
            if month == 4:
                p['BLK_3'] = 0.710
                p['BLK_4'] = 0.700
            elif month == 5:
                p['BLK_3'] = 0.742
                p['BLK_4'] = 0.732
            elif month == 6:
                p['BLK_3'] = 0.759
                p['BLK_4'] = 0.749
            else:
                p['BLK_3'] = 0.0
                p['BLK_4'] = 0.0

            if i == 0:
                df_stock.iloc[i] = df.iloc[i] # 4월 1일자 재고량은 그대로 두고
            else:
                df_stock.iloc[i] = df_stock.iloc[i-1] + df.iloc[i] 
                # schedule_1, 2를 통해 나온 df_out의 값들인 PRT_1~4, MOL_1~4의 시간에 따른 변화량(생산량)을 반영
                for column in df_order.columns: # order.csv의 각 column에 대하여 
                    val = df_order.loc[time, column] # 해당 블럭의 수요 (BLK_X)
                    if val > 0: # 수요가 있으면
                        mol_col = blk2mol[column] # 그 블럭에 해당하는 몰드 선택
                        mol_num = df_stock.loc[time, mol_col] # 그 몰드가 몇 개 있는지
                        df_stock.loc[time, mol_col] = 0 # 왜 선택 날짜의 몰드 값을 0으로 만드냐...
                        
                        blk_gen = int(mol_num*p[column]*cut[column]) # 몰드*몰드당 블럭수*양품률
                        blk_stock = df_stock.loc[time, column] + blk_gen # 선택 날짜의 블럭 보유량에 생산량 더하기
                        blk_diff = blk_stock - val # 부족/초과분 = 보유량 - 주문량
                        
                        df_stock.loc[time, column] = blk_diff # 해당 날짜에 부족/초과분 
                        blk_diffs.append(blk_diff) # 부족/초과분 점수 계산을 위해서 list에 추가
        return df_stock, blk_diffs
    
    
    # 부족분과 초과분을 함께 score에 넣어줌
    def cal_score(self, blk_diffs):
        # Block Order Difference
        blk_diff_m = 0
        blk_diff_p = 0
        for item in blk_diffs:
            if item < 0:
                blk_diff_m = blk_diff_m + abs(item)
            if item > 0:
                blk_diff_p = blk_diff_p + abs(item)
        score = blk_diff_m + blk_diff_p
        return score



    def get_score(self, df):
        df = self.subprocess(df) 
        out_1 = self.cal_schedule_part_1(df)
        out_2 = self.cal_schedule_part_2(df, line='A')
        out_3 = self.cal_schedule_part_2(df, line='B')
        out = out_1 + out_2 + out_3
        out = self.add_stock(out, self.stock)
        order = self.order_rescale(out, self.order)                    
        out, blk_diffs = self.cal_stock(out, order)                    
        score = self.cal_score(blk_diffs) 
        return score, out
