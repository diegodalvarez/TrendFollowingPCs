# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:20:07 2024

@author: Diego
"""

import os
import numpy as np
import pandas as pd

from pykalman import KalmanFilter
from sklearn.decomposition import PCA
from PCTrendDataPrep import TreasuryDataCollect


class SignalGenerator(TreasuryDataCollect):
    
    def __init__(self) -> None: 
        
        super().__init__()
        self.signal_path = os.path.join(self.data_path, "Signals")
        if os.path.exists(self.signal_path) == False: os.makedirs(self.signal_path)
        
    def _get_tsy_pcs(self, n_comps: int = 1) -> pd.DataFrame:
        
        df_wider = (self.get_tsy_rate().drop(
            columns = ["value"]).
            pivot(index = "date", columns = "variable", values = "val_diff"))
        
        df_out = pd.DataFrame(
            data    = PCA(n_components = n_comps).fit_transform(df_wider),
            columns = ["PC{}".format(i + 1) for i in range(n_comps)],
            index   = df_wider.index)
        
        return df_out
    
    def _get_fut_pcs(self, n_comps: int = 1) -> pd.DataFrame: 
        
        df_wider = (self.get_tsy_fut()[
            ["date", "security", "PX_bps"]].
            pivot(index = "date", columns = "security", values = "PX_bps").
            fillna(0))
        
        df_out = pd.DataFrame(
            data    = PCA(n_components = n_comps).fit_transform(df_wider),
            columns = ["PC{}".format(i + 1) for i in range(n_comps)],
            index   = df_wider.index)
        
        return df_out
    
    def _get_pc_spread(self, n_comps: int = 3) -> pd.DataFrame: 
        
        df_yld_diff = (self._get_tsy_pcs(n_comps).reset_index().melt(
            id_vars = "date").
            rename(columns = {"value": "yld_diff"}))
        
        df_fut = (self._get_fut_pcs(n_comps).reset_index().melt(
            id_vars = "date").
            rename(columns = {"value": "fut"}))
        
        df_out = (df_yld_diff.merge(
            right = df_fut, how = "inner", on = ["date", "variable"]).
            assign(spread = lambda x: x.fut - x.yld_diff).
            drop(columns = ["fut", "yld_diff"]))
        
        return df_out
    
    def _lag_spread(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        return(df.sort_values(
            "date").
            assign(lag_value = lambda x: x.spread.shift()).
            dropna())
    
    def get_pc_spread_signal(self, n_comps: int = 3, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "PCSpreadSignal.parquet")
        try:
            
            if verbose == True: print("Trying to find PC Spread Signal")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
        
            if verbose == True: print("Couldn't find data, generating it")
            df_out = (self._get_pc_spread(
                n_comps = n_comps).
                groupby("variable").
                apply(self._lag_spread).
                reset_index(drop = True))
            
            if verbose == True: print("Saving data")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_ewma(self, df: pd.DataFrame, window: int) -> pd.DataFrame: 
        
        return(df.sort_values(
            "date").
            assign(
                strat_name  = str(window),
                spread_mean = lambda x: x.spread.ewm(span = window, adjust = False).mean(),
                signal      = lambda x: x.spread - x.spread_mean,
                lag_signal  = lambda x: x.signal.shift()))
    
    def _get_pc_spread_ewma(self, df: pd.DataFrame, windows: list) -> pd.DataFrame: 
        
        return(pd.concat([self._get_ewma(df, window) for window in windows]))
    
    def get_pc_spread_ewma(
            self, 
            n_comps   : int = 3,
            multiplier: int = 2,
            start     : int = 2, 
            end       : int = 7,
            verbose   : bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "PCSpreadEWMA.parquet")
        try:
            
            if verbose == True: print("Trying to find PC Spread EWMA")
            df_ewma = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
        
            if verbose == True: print("Couldn't find data generating it")
            windows = [multiplier ** (i + 1) for i in range(start, end)]
            df_pcs  = self._get_pc_spread(n_comps)
            
            df_ewma = (df_pcs.groupby(
                "variable").
                apply(self._get_pc_spread_ewma, windows).
                reset_index(drop = True))
            
            if verbose == True: print("Saving data")
            df_ewma.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_ewma
    
    def _get_ewmac(self, df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
        
        return(df.sort_values(
            "date").
            assign(
                strat_name = str(short_window) + "x" + str(long_window),
                short_mean = lambda x: x.spread.ewm(span = short_window, adjust = False).mean(),
                long_mean  = lambda x: x.spread.ewm(span = long_window, adjust = False).mean(),
                signal     = lambda x: (x.short_mean - x.long_mean) / x.spread,
                lag_signal = lambda x: x.signal.shift()))
    
    def _get_pc_spread_ewmac(self, df: pd.DataFrame, windows: list) -> pd.DataFrame: 
        
        return(pd.concat([self._get_ewmac(
            df           = df, 
            short_window = window["short_window"], 
            long_window  = window["long_window"])
            for window in windows]))
    
    def get_pc_spread_ewmac(
            self,
            n_comps   : int = 3,
            multiplier: int = 2,
            start     : int = 2,
            end       : int = 7,
            verbose   : bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "PCSpreadEWMAC.parquet")
        try:
            
            if verbose == True: print("Trying to find PC Spread EWMAC")
            df_ewmac = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, collecting it")
            
            windows = [{
                "short_window": multiplier ** (i + 1),
                "long_window" : multiplier ** (i + 2)}
                for i in range(start, end)]
            
            df_pcs   = self.get_pc_spread_signal(n_comps)
            df_ewmac = (df_pcs.groupby(
                "variable").
                apply(self._get_pc_spread_ewmac, windows).
                reset_index(drop = True))
            
            if verbose == True: print("Saving data")
            df_ewmac.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_ewmac
    
    def _get_zscore(self, df: pd.DataFrame, window: int) -> pd.DataFrame: 
        
        df_out = (df.sort_values(
            "date").
            assign(
                strat_name = str(window),
                roll_mean  = lambda x: x.spread.ewm(span = window, adjust = False).mean(),
                roll_std   = lambda x: x.spread.ewm(span = window, adjust = False).std(),
                z_score    = lambda x: (x.spread - x.roll_mean) / x.roll_std,
                lag_zscore = lambda x: x.z_score.shift()))
        
        return df_out
    
    def _get_pc_spread_zscore(self, df: pd.DataFrame, windows: list) -> pd.DataFrame: 
        
        return(pd.concat([self._get_zscore(df, window) for window in windows]))
    
    def get_pc_spread_zscore(
            self,
            n_comps   : int = 3,
            multiplier: int = 2,
            start     : int = 1,
            end       : int = 7,
            verbose   : bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.signal_path, "PCSpreadZScore.parquet")
        try:
            
            if verbose == True: print("Trying to find z-score data")
            df_zscore = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, generating it")
            
            windows = [multiplier ** (i + 1) for i in range(start, end)]
            df_pcs  = self._get_pc_spread(n_comps)
            
            df_zscore = (df_pcs.groupby(
                "variable").
                apply(self._get_pc_spread_zscore, windows).
                reset_index(drop = True))
            
            if verbose == True: print("Saving data")
            df_zscore.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_zscore
    
    def _get_yld_pc(self, n_comps: int) -> pd.DataFrame: 
        
        df_wider = (self.get_tsy_rate().drop(
            columns = ["val_diff"]).
            pivot(index = "date", columns = "variable", values = "value").
            fillna(0))
        
        df_out = (pd.DataFrame(
            data    = PCA(n_components = n_comps).fit_transform(df_wider),
            columns = ["PC{}".format(i + 1) for i in range(n_comps)],
            index   = df_wider.index).
            reset_index().
            melt(id_vars = "date"))
        
        return df_out
    
    def _get_kalman(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_tmp        = df.sort_values("date").dropna()
        kalman_filter = KalmanFilter(
            transition_matrices      = [1],
            observation_matrices     = [1],
            initial_state_mean       = 0,
            initial_state_covariance = 1,
            observation_covariance   = 1,
            transition_covariance    = 0.01)
        
        state_means, state_covariances = kalman_filter.filter(df_tmp.value)
        df_out = (df_tmp.assign(
            smooth     = state_means,
            lag_smooth = lambda x: x.smooth.shift(),
            resid      = lambda x: x.lag_smooth - x.value,
            lag_resid  = lambda x: x.resid.shift()).
            drop(columns = ["value"]))
        
        return df_out
    
    def _get_resid_zscore(self, df: pd.DataFrame, window: int) -> pd.DataFrame: 
        
        df_out = (df.sort_values(
            "date").
            assign(
                strat_name = str(window),
                roll_mean  = lambda x: x.resid.ewm(span = window, adjust = False).mean(),
                roll_std   = lambda x: x.resid.ewm(span = window, adjust = False).std(),
                z_score    = lambda x: (x.resid - x.roll_mean) / x.roll_std,
                lag_zscore = lambda x: x.z_score.shift()))
        
        return df_out
        
    def _get_yld_kalman_zscore(self, df: pd.DataFrame, windows: list) -> pd.DataFrame: 
        
        return(pd.concat([self._get_resid_zscore(df, window) for window in windows]))
    
    def get_yld_kalman_zscore(
            self,
            n_comps   : int = 3,
            multiplier: int = 2,
            start     : int = 1,
            end       : int = 7,
            verbose   : bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "YldPCKalmanZScore.parquet")
        try:
            
            if verbose == True: print("Trying to find Yield Kalman Z-Score data")
            df_zscore = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, generating it")
            
            windows   = [multiplier ** (i + 1) for i in range(start, end)]
            df_yld_pc = self._get_yld_pc(n_comps)
            df_kalman = (df_yld_pc.groupby(
                "variable").
                apply(self._get_kalman).
                reset_index(drop = True).
                dropna())
            
            windows   = [multiplier ** (i + 1) for i in range(start, end)]
            df_zscore = (df_kalman.groupby(
                "variable").
                apply(self._get_yld_kalman_zscore, windows).
                reset_index(drop = True))
            
            if verbose == True: print("Saving data")
            df_zscore.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_zscore
            
def main() -> None: 
    
    SignalGenerator().get_yld_kalman_zscore()
    SignalGenerator().get_pc_spread_zscore()
    SignalGenerator().get_pc_spread_ewmac(verbose = True)
    SignalGenerator().get_pc_spread_ewma(verbose = True)
    SignalGenerator().get_pc_spread_signal(verbose = True)        
    
if __name__ == "__main__": main()

