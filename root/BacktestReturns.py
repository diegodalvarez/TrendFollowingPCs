# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 09:53:16 2024

@author: Diego
"""

import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from TreasuryDataCollect import TreasuryDataCollect

class Backtest(TreasuryDataCollect):
    
    def __init__(self) -> None: 
        
        super().__init__()
        self.rtn_path    = os.path.join(self.data_path, "SignalRtn")
        self.signal_path = os.path.join(self.data_path, "Signals")
        
        if os.path.exists(self.rtn_path) == False: os.makedirs(self.rtn_path)
        
    def prep_imply_data(self) -> pd.DataFrame:
        
        ewmac_path = os.path.join(self.signal_path, "EWMACSignals.parquet")
        df_ewmac   = (pd.read_parquet(
            path = ewmac_path, engine = "pyarrow").
            drop(columns = ["value", "lag_signal"]))
        
        
        ewma_path = os.path.join(self.signal_path, "EWMASignals.parquet")
        df_ewma   = (pd.read_parquet(
            path = ewma_path, engine = "pyarrow").
            drop(columns = ["value", "lag_signal"]))

        
        kalman_path = os.path.join(self.signal_path, "KalmanSignals.parquet")
        df_kalman   = (pd.read_parquet(
            path = kalman_path, engine = "pyarrow").
            drop(columns = ["lag_signal"]))
        
        df_out = pd.concat([df_ewmac, df_ewma, df_kalman])
        return df_out
    
    def prep_imply_fut_rtn(self) -> pd.DataFrame: 
        
        df_out = (self.get_tsy_fut()[
            ["date", "security", "PX_bps"]].
            assign(security = lambda x: x.security.str.split(" ").str[0]))
        
        return df_out
    
    def prep_signal(self) -> pd.DataFrame: 
        
        ewmac_path = os.path.join(self.signal_path, "EWMACSignals.parquet")
        df_ewmac   = (pd.read_parquet(
            path = ewmac_path, engine = "pyarrow").
            drop(columns = ["lag_decile", "value"]))

        ewma_path = os.path.join(self.signal_path, "EWMASignals.parquet")
        df_ewma   = (pd.read_parquet(
            path = ewma_path, engine = "pyarrow").
            drop(columns = ["lag_decile", "value"]))
        
        kalman_path = os.path.join(self.signal_path, "KalmanSignals.parquet")
        df_kalman   = (pd.read_parquet(
            path = kalman_path, engine = "pyarrow").
            drop(columns = ["lag_decile"]))
        
        df_out = pd.concat([df_ewmac, df_ewma, df_kalman])
        return df_out
    
    def imply_signal(self, outer_decile_avg: int) -> pd.DataFrame: 
        
        df_combined = (self.prep_imply_data().merge(
            right = self.prep_imply_fut_rtn(), how = "inner", on = ["date"]).
            drop(columns = ["date"]).
            groupby(["variable", "strat_name", "lag_decile", "signal_type", "security"]).
            agg(["mean", "std"])
            ["PX_bps"].
            rename(columns = {
                "mean": "mean_rtn",
                "std" : "std_rtn"}).
            assign(sharpe = lambda x: x.mean_rtn / x.std_rtn * np.sqrt(252)).
            reset_index().
            drop(columns = ["mean_rtn", "std_rtn"]))
        
        df_decile_key = (df_combined[
            ["lag_decile"]].
            groupby("lag_decile").
            head(1).
            assign(decile_num = lambda x: x.lag_decile.astype(str).str.replace("D", "").astype(int)).
            sort_values("decile_num"))
        
        keep_deciles = ([i + 1 for i in range(outer_decile_avg)] + 
                        [df_decile_key.decile_num.max() - i for i in range(outer_decile_avg)])
        
        decile_group = (["lower_decile" for i in range(outer_decile_avg)] +
                        ["upper_decile" for i in range(outer_decile_avg)])
        
        df_decile_group = (df_decile_key.query(
            "decile_num == @keep_deciles").
            sort_values("decile_num").
            assign(decile_group = decile_group).
            drop(columns = ["decile_num"]))
        
        df_out = (df_combined.merge(
            right = df_decile_group, how = "inner", on = ["lag_decile"]).
            drop(columns = ["lag_decile"]).
            groupby(["variable", "strat_name", "signal_type", "security", "decile_group"]).
            agg("mean").
            reset_index().
            pivot(
                index = ["variable", "strat_name", "signal_type", "security"],
                columns = "decile_group",
                values = "sharpe").
            reset_index())
        
        return df_out
    
    def implied_signal_rtn(self, outer_decile_avg: int = 2, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.rtn_path, "ImpliedSignalReturns.parquet")
        try:
            
            if verbose == True: print("Trying to find data")
            df_signal = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data generating implied returns")
    
            df_combined = (self.get_tsy_fut()[
                ["date", "security", "PX_bps"]].
                assign(security = lambda x: x.security.str.split(" ").str[0]).
                merge(right = self.imply_signal(outer_decile_avg), how = "inner", on = ["security"]))
            
            df_signal = (self.prep_signal().merge(
                right = df_combined, 
                how   = "inner", 
                on    = ["date", "variable", "strat_name", "signal_type"]).
                assign(
                    strat_name = lambda x: x.strat_name.astype(str),
                    signal_rtn = lambda x: np.where(
                        x.lag_signal < 0, 
                        np.sign(x.lower_decile) * x.PX_bps,
                        np.sign(x.upper_decile) * x.PX_bps)))
            
            if verbose == True: print("Saving Data\n")
            df_signal.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_signal
    
    
df = Backtest().implied_signal_rtn(verbose = True)