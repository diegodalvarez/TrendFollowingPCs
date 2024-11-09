# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:56:18 2024

@author: Diego
"""

import os
import numpy as np
import pandas as pd

from pykalman import KalmanFilter
from PrincipalComponentsAnalysis import PrincipalComponentAnalysis

class SignalGenerator(PrincipalComponentAnalysis):
    
    def __init__(self) -> None:
        
        super().__init__()
        self.signal_path = os.path.join(self.data_path, "Signals")
        if os.path.exists(self.signal_path) == False: os.makedirs(self.signal_path)
        
    def pre_process_data(self):
        
        futures_path = os.path.join(self.pca_path, "FittedFuturesPCs.parquet")
        df_fut = (pd.read_parquet(
            path = futures_path, engine = "pyarrow").
            reset_index().
            melt(id_vars = "date").
            assign(variable = lambda x: "fut " + x.variable))
        
        treasury_path = os.path.join(self.pca_path, "FittedTreasuryPCs.parquet")
        df_tsy = (pd.read_parquet(
            path = treasury_path, engine = "pyarrow").
            rename(columns = {"variable": "var_name"}).
            melt(id_vars = ["date", "var_name"]).
            assign(variable = lambda x: x.var_name + " " + x.variable).
            drop(columns = ["var_name"]))
        
        spread_path = os.path.join(self.pca_path, "SpreadPCs.parquet")
        df_spread = (pd.read_parquet(
            path = spread_path, engine = "pyarrow").
            reset_index().
            melt(id_vars = ["date", "variable"]).
            assign(variable = lambda x: "spread_" + x.group_var + " " + x.variable).
            drop(columns = ["group_var"]))
        
        df_out = (pd.concat([
            df_fut,
            df_tsy,
            df_spread]))
        
        return df_out
    
    def _ewmac(self, df: pd.DataFrame, short_window: int, long_window: int, d: int = 10) -> pd.DataFrame:
        
        df_out = (df.sort_values(
            "date").
            assign(
                short_window = short_window,
                long_window  = long_window,
                strat_name   = str(short_window) + "x" + str(long_window),
                short_mean   = lambda x: x.value.ewm(span = short_window, adjust = False).mean(),
                long_mean    = lambda x: x.value.ewm(span = long_window, adjust = False).mean(),
                signal       = lambda x: (x.short_mean - x.long_mean) / x.value,
                lag_signal   = lambda x: x.signal.shift(),
                decile       = lambda x: pd.qcut(x = x.signal, q = d, labels = ["D{}".format(i + 1) for i in range(d)]),
                lag_decile   = lambda x: x.decile.shift()).
            drop(columns = [
                "short_mean", "long_mean", "decile", "short_window", 
                "long_window", "signal"]).
            dropna())
        
        return df_out
    
    def _ewmac_signal(self, df: pd.DataFrame, ewmac_signals: list) -> pd.DataFrame:
        
        df_out = (pd.concat([
            self._ewmac(
                df           = df, 
                short_window = ewmac_signal["short_window"], 
                long_window  = ewmac_signal["long_window"])
            for ewmac_signal in ewmac_signals]))
        
        return df_out
    
    def ewmac_signal(self) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "EWMACSignals.parquet")
        try:
            
            df_ewmac = pd.read_parquet(path = file_path, engine = "pyarrow")
            
        except: 
        
            ewmac_signals = ([
                {"short_window": 2 ** (i + 2), "long_window": 2 ** (i + 3)} 
                for i in range(1,5)])
            
            df_input = self.pre_process_data()
            df_ewmac   = (df_input.groupby(
                "variable").
                apply(self._ewmac_signal, ewmac_signals).
                reset_index(drop = True).
                assign(signal_type = "EMACTrend"))
            
            df_ewmac.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_ewmac
    
    def _ewma_signal(self, df: pd.DataFrame, ewma_windows: list, d: int = 10) -> pd.DataFrame: 
        
        df_out = (pd.concat([df.sort_values("date").assign(
            strat_name  = ewma_window,
            signal      = lambda x: x.value.ewm(span = ewma_window, adjust = False).mean(),
            decile      = lambda x: pd.qcut(x = x.signal, q = d, labels = ["D{}".format(i + 1) for i in range(d)]),
            lag_decile  = lambda x: x.decile.shift(),
            lag_signal  = lambda x: x.signal.shift(),
            signal_type = "EMATrend").
            dropna().
            drop(columns = ["decile", "signal"])
            for ewma_window in ewma_windows]))
        
        return df_out
    
    def ewma_signal(self) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "EWMASignals.parquet")
        try:
            
            df_ewma = pd.read_parquet(path = file_path, engine = "pyarrow")
            
        except:
        
            ewma_windows = [2 ** (i + 2) for i in range(2,6)]
            df_ewma = (self.pre_process_data().groupby(
                "variable").
                apply(self._ewma_signal, ewma_windows).
                reset_index(drop = True))
            
            df_ewma.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_ewma
    
    def _get_resid_zscore(self, df: pd.DataFrame, lookback: int, d: int) -> pd.DataFrame: 
        
        df_out = (df.assign(
            smooth_mean   = lambda x: x.lag_smooth.ewm(span = lookback, adjust = False).mean(),
            smooth_std    = lambda x: x.lag_smooth.ewm(span = lookback, adjust = False).std(),
            smooth_zscore = lambda x: ((x.lag_smooth - x.smooth_mean) / x.smooth_std).shift(),
            resid_mean    = lambda x: x.resid.ewm(span = lookback, adjust = False).mean(),
            resid_std     = lambda x: x.resid.ewm(span = lookback, adjust = False).std(),
            resid_zscore  = lambda x: ((x.resid - x.resid_mean) / x.resid_std).shift(),
            smooth_decile = lambda x: pd.qcut(
                x = x.smooth_zscore, q = d, labels = ["D{}".format(i + 1) for i in range(d)]).shift(),
            resid_decile  = lambda x: pd.qcut(
                x = x.resid_zscore, q = d, labels = ["D{}".format(i + 1) for i in range(d)]).shift()).
            dropna().
            drop(columns = [
                "smooth_mean", "smooth_std", "resid_mean", "resid_std",
                "lag_smooth", "resid"]).
            rename(columns = {"variable": "tmp_var"}).
            melt(id_vars = ["date", "tmp_var"]).
            assign(
                tmp1     = lambda x: x.variable.str.split("_").str[0],
                tmp2     = lambda x: x.variable.str.split("_").str[1]).
            drop(columns = ["variable"]).
            pivot(index = ["date", "tmp_var", "tmp1"], columns = "tmp2", values = "value").
            reset_index().
            assign(
                strat_name  = lookback,
                signal_type = lambda x: x.tmp1 + " Kalman").
            drop(columns = ["tmp1"]).
            rename(columns = {
                "tmp_var": "variable",
                "decile" : "lag_decile",
                "zscore" : "lag_signal"}))
        
        return df_out
    
    def _kalman_filter(self, df: pd.DataFrame, lookbacks: list, d: int) -> pd.DataFrame:
        
        print("Working at {}".format(df.name))
        
        df_tmp = df.sort_values("date").dropna()
        kalman_filter = KalmanFilter(
            transition_matrices      = [1],
            observation_matrices     = [1],
            initial_state_mean       = 0,
            initial_state_covariance = 1,
            observation_covariance   = 1,
            transition_covariance    = 0.01)
        
        state_means, state_covariances = kalman_filter.filter(df.value)
        df_kalman = (df.assign(
            smooth     = state_means,
            lag_smooth = lambda x: x.smooth.shift(),
            resid      = lambda x: x.value - x.lag_smooth).
            drop(columns = ["value", "smooth"]).
            dropna())
        
        df_out = (pd.concat([
            self._get_resid_zscore(df_kalman, lookback, d)
            for lookback in lookbacks]))
        
        return df_out
    
    def kalman_signal(
            self, 
            lookbacks: list = [5, 10, 20], 
            d        : int = 10) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "KalmanSignals.parquet")
        try:
            
            df_kalman = pd.read_parquet(path = file_path, engine = "pyarrow")
            
        except:
                
            df_kalman = (self.pre_process_data().groupby(
                "variable").
                apply(self._kalman_filter, lookbacks, d).
                reset_index(drop = True))
        
            df_kalman.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_kalman
    
def main():
        
    _ = SignalGenerator().kalman_signal()
    _ = SignalGenerator().ewmac_signal()
    _ = SignalGenerator().ewma_signal()
    
if __name__ == "__main__": main()