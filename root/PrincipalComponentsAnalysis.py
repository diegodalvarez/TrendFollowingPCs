# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:36:52 2024

@author: Diego
"""

import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from TreasuryDataCollect import TreasuryDataCollect

class PrincipalComponentAnalysis(TreasuryDataCollect):
    
    def __init__(self) -> None: 
        
        super().__init__()
        self.pca_path = os.path.join(self.data_path, "PCAData")
        
        if os.path.exists(self.pca_path) == False: os.makedirs(self.pca_path)
        
    def _prep_tsy_data(self) -> pd.DataFrame: 
        
        df_out = (self.get_tsy_rate().rename(
            columns = {
                "value"   : "yld", 
                "val_diff": "yld_diff"}).
            pivot(
                index   = "date", 
                columns = "variable", 
                values  = ["yld", "yld_diff"]).
            reset_index().
            melt(id_vars = "date").
            rename(columns = {None: "var"}).
            pivot(
                index   = ["date", "var"], 
                columns = "variable", 
                values  = "value").
            reset_index().
            rename(columns = {"var": "variable"}))
            
        return df_out
        
    def _get_tsy_pca(self, df: pd.DataFrame, n_components: int) -> pd.DataFrame: 
        
        df_tmp = (df.set_index(
            "date").
            drop(columns = ["variable"]))
        
        df_out = (pd.DataFrame(
            data    = PCA(n_components = n_components).fit_transform(df_tmp),
            columns = ["PC{}".format(i + 1) for i in range(n_components)]).
            assign(date = df_tmp.index))
        
        return df_out
        
    def get_tsy_pca(self, n_components: int = 3, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.pca_path, "FittedTreasuryPCs.parquet")
        try: 
            
            if verbose == True: print("Trying to find Treasury PCA Data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
        
            if verbose == True: print("Couldn't Find Data Collecting it")
            df_out = (self._prep_tsy_data().groupby(
                "variable").
                apply(self._get_tsy_pca, n_components).
                reset_index().
                drop(columns = ["level_1"]))
        
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
    def _get_tsy_pca_loadings(self, df: pd.DataFrame, n_components: int) -> pd.DataFrame:
        
        df_tmp = df.set_index("date").drop(columns = ["variable"])

        df_out = (pd.DataFrame(
            data =    PCA(n_components = n_components).fit(df_tmp).components_,
            index =   ["PC{}".format(i + 1) for i in range(n_components)],
            columns = df_tmp.columns.to_list()))
        
        return df_out
    
    def get_tsy_pca_loadings(self, n_components: int = 3, verbose: int = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.pca_path, "TreasuryPCLoadings.parquet")
        try:
            
            if verbose == True: print("Looking for Treasury PC Loadings")
            df_tmp = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
        
            if verbose == True: print("Generating PC Loadings")
            df_tmp = (self._prep_tsy_data().groupby(
                "variable").
                apply(self._get_tsy_pca_loadings, n_components).
                reset_index().
                rename(columns = {"level_1": "PC"}))
        
            if verbose == True: print("Saving PC Loadings data\n")
            df_tmp.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_tmp
    
    def get_fut_pca(self, n_components: int = 3, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.pca_path, "FittedFuturesPCs.parquet")
        try: 
            
            if verbose == True: print("Trying to find Treasury Futures PCA Data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't Find Data Collecting it")
            df_tmp = (self.get_tsy_fut().assign(
                security = lambda x: x.security.str.split(" ").str[0])
                [["security", "date", "PX_bps"]].
                pivot(index = "date", columns = "security", values = "PX_bps").
                fillna(0))
            
            df_out = (pd.DataFrame(
                data = PCA(n_components = 3).fit_transform(df_tmp),
                columns = ["PC{}".format(i + 1) for i in range(n_components)]).
                assign(date = df_tmp.index).
                set_index("date"))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
    def get_fut_pca_loadings(self, n_components: int = 3, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.pca_path, "FuturesPCLoadings.parquet")
        try:
            
            if verbose == True: print("Trying to find Futures PC Loadings")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
        
            if verbose == True: print("Couldn't find data getting PC Loadings")
            df_tmp = (self.get_tsy_fut().assign(
                security = lambda x: x.security.str.split(" ").str[0])
                [["security", "date", "PX_bps"]].
                pivot(index = "date", columns = "security", values = "PX_bps").
                fillna(0))
            
            df_out = (pd.DataFrame(
                data    = PCA(n_components = n_components).fit(df_tmp).components_,
                index   = ["PC{}".format(i + 1) for i in range(n_components)],
                columns = df_tmp.columns.to_list()))
            
            if verbose == True: print("Saving data")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
    def _prep_yld_pcs(self) -> pd.DataFrame: 
        
        tsy_path = os.path.join(self.pca_path, "FittedTreasuryPCs.parquet")
        
        df_yld_change = (pd.read_parquet(
            path = tsy_path, engine = "pyarrow").
            query("variable == 'yld_diff'").
            drop(columns = ["variable"]).
            set_index("date").
            cumsum().
            reset_index().
            melt(id_vars = "date").
            rename(columns = {"value": "yld_val"}))
        
        
        df_yld = (pd.read_parquet(
            path = tsy_path, engine = "pyarrow").
            query("variable == 'yld'").
            drop(columns = ["variable"]).
            melt(id_vars = "date").
            rename(columns = {"value": "yld_val"}).
            assign(yld_val = lambda x: np.where(x.variable == "PC1", -1 * x.yld_val, x.yld_val)))
        
        df_out = (pd.concat([
            df_yld_change.assign(group_var = "yld_chng"),
            df_yld.assign(group_var = "yld")]))
        
        return df_out
    
    def get_spread_signals(self, verbose: bool = False) -> pd.DataFrame:
        
        spread_path = os.path.join(self.pca_path, "SpreadPCs.parquet")
        try:
            
            if verbose == True: print("Trying to find Spread PCA Data")
            df_combined = pd.read_parquet(path = spread_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
        
            if verbose == True: print("Couldn't Find Data Collecting it")
            fut_path = os.path.join(self.pca_path, "FittedFuturesPCs.parquet")
            
            df_yield = self._prep_yld_pcs()
            df_fut   = (pd.read_parquet(
                path = fut_path, engine = "pyarrow").
                cumsum().
                reset_index().
                melt(id_vars = "date").
                rename(columns = {"value": "fut_val"}))
    
            df_combined = (df_yield.merge(
                right = df_fut, how = "inner", on = ["date", "variable"]).
                assign(spread = lambda x: x.fut_val - x.yld_val).
                drop(columns = ["yld_val", "fut_val"]).
                pivot(index = ["date", "variable"], columns = "group_var", values = "spread").
                reset_index().
                set_index("date"))
            
            if verbose == True: print("Saving data\n")
            df_combined.to_parquet(path = spread_path, engine = "pyarrow")
        
        return df_combined
    
def main():
    
    princ_comp = PrincipalComponentAnalysis()
    
    princ_comp.get_fut_pca(verbose = True)
    princ_comp.get_tsy_pca_loadings(verbose = True)
    
    princ_comp.get_tsy_pca(verbose = True)
    princ_comp.get_fut_pca_loadings(verbose = True)
    
    princ_comp.get_spread_signals(verbose = True)
    
#if __name__ == "__main__": main()