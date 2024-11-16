# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 10:53:37 2024

@author: Diego
"""

import os
import numpy as np
import pandas as pd

from PCSignalGenerator import SignalGenerator

class PCBacktest(SignalGenerator):
    
    def __init__(self) -> None: 
        
        super().__init__()
        
        self.backtest_path = os.path.join(self.data_path, "RawBacktestRtn")
        if os.path.exists(self.backtest_path) == False: os.makedirs(self.backtest_path)
        
    def _prep_tsy(self) -> pd.DataFrame: 
        
        df_tsy = (self.get_tsy_fut().drop(
            columns = ["PX_LAST", "PX_diff", "PX_pct", "CTD_DUR", "FUT_CNVX"]).
            assign(
                date     = lambda x: pd.to_datetime(x.date).dt.date,
                security = lambda x: x.security.str.split(" ").str[0]))
        
        return df_tsy
        
    def get_kalman_rtn(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.backtest_path, "KalmanZScore.parquet")
        try:
            
            if verbose == True: print("Trying to find Kalman data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, getting results")
        
            df_kalman = (SignalGenerator().get_yld_kalman_zscore().dropna().drop(
                columns = [
                    "z_score", "roll_std", "roll_mean", "lag_resid", "resid",
                    "smooth", "lag_smooth"]).
                rename(columns = {
                    "strat_name": "param",
                    "lag_zscore": "signal"}).
                assign(
                    strat_group = "KalmanZScore",
                    strat_name  = lambda x: x.variable + "_KalmanZScore_" + x.param,
                    date        = lambda x: pd.to_datetime(x.date).dt.date))
            
            df_out = (self._prep_tsy().merge(
                right = df_kalman, how = "inner", on = ["date"]).
                assign(signal_bps = lambda x: -1 * np.sign(x.signal) * x.PX_bps))
            
            print("Saving data")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
    def get_pc_zscore_rtn(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.backtest_path, "PCSpreadZscore.parquet")
        try:
            
            if verbose == True: print("Trying to find ZScore data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, generating it")
            df_zscore = (self.get_pc_spread_zscore().dropna().drop(
                columns = ["z_score", "roll_std", "roll_mean", "spread"]).
                rename(columns = {
                    "strat_name": "param",
                    "lag_zscore": "signal"}).
                assign(
                    strat_group = "PCSpreadZscore",
                    strat_name  = lambda x: x.variable + "_PCSpreadZScore_" + x.param,
                    date        = lambda x: pd.to_datetime(x.date).dt.date))
            
            df_out = (self._prep_tsy().merge(
                right = df_zscore, how = "inner", on = ["date"]).
                assign(signal_bps = lambda x: np.where(
                    x.variable == "PC1", 
                    -1 * np.sign(x.signal) * x.PX_bps,
                    np.sign(x.signal) * x.PX_bps)))
            
            if verbose == True: print("Saving data")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out

    def get_pc_ewmac_rtn(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.backtest_path, "PCSpreadEWMAC.parquet")
        try:
            
            if verbose == True: print("Trying to find EWMAC data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, generating it")
            
            df_ewmac = (self.get_pc_spread_ewmac().drop(
                columns = [
                    "spread", "long_mean", "short_mean", "signal", 
                    "lag_value"]).
                dropna().
                rename(columns = {
                    "strat_name": "param",
                    "lag_signal": "signal"}).
                assign(
                    strat_group = "PCSpreadEWMAC",
                    strat_name  = lambda x: x.variable + "_PCSpreadEWMAC_" + x.param,
                    date        = lambda x: pd.to_datetime(x.date).dt.date))

            df_out = (self._prep_tsy().merge(
                right = df_ewmac, how = "inner", on = ["date"]).
                assign(signal_bps = lambda x: np.sign(x.signal) * x.PX_bps))
            
            if verbose == True: print("Saving data")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def get_pc_ewma_rtn(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.backtest_path, "PCSpreadEWMA.parquet")
        try:
            
            if verbose == True: print("Trying to find EWMA data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, generating it")
            
            df_ewma = (self.get_pc_spread_ewma().drop(
                columns = ["spread", "spread_mean", "signal"]).
                dropna().
                rename(columns = {
                    "strat_name": "param",
                    "lag_signal": "signal"}).
                assign(
                    strat_group = "PCSpreadEWMAC",
                    strat_name  = lambda x: x.variable + "_PCSpreadEWMAC_" + x.param,
                    date        = lambda x: pd.to_datetime(x.date).dt.date))

            df_out = (self._prep_tsy().merge(
                right = df_ewma, how = "inner", on = ["date"]).
                assign(signal_bps = lambda x: np.sign(x.signal) * x.PX_bps))
            
            if verbose == True: print("Saving data")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def get_pc_spread_signal_rtn(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.backtest_path, "PCSpread.parquet")
        try:
            
            if verbose == True: print("Trying to find data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, generating it")
            
            df_out = (self.get_pc_spread_signal().drop(
                columns = ["spread"]).
                rename(columns = {"lag_value": "signal"}).
                assign(date = lambda x: pd.to_datetime(x.date).dt.date).
                merge(right = self._prep_tsy(), how = "inner", on = ["date"]).
                assign(signal_bps = lambda x: np.where(
                    x.variable == "PC2", 
                    np.sign(x.signal) * x.PX_bps,
                    -1 * np.sign(x.signal) * x.PX_bps)))
            
            if verbose == True: print("Saving data")
            df_out.to_parquet(path = file_path, engine = "pyarrow")

def main():
    
    pc_backtest = PCBacktest()

    df = pc_backtest.get_pc_spread_signal_rtn(verbose = True)
    df = pc_backtest.get_pc_ewma_rtn(verbose = True)
    df = pc_backtest.get_pc_ewmac_rtn(verbose = True)
    df = pc_backtest.get_pc_zscore_rtn(verbose = True)
    df = pc_backtest.get_kalman_rtn(verbose = True)
    
if __name__ == "__main__": main()