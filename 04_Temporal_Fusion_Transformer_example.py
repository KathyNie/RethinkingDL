# Import dependencies
import os
import sys
import random
import xarray as xr
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
import optuna
from scipy.stats import pearsonr


def bias_metric(y_true,y_pred):
    bias = np.mean(y_pred-y_true)
    return bias
    
def rmse_metric(y_true,y_pred):
    rmse = np.sqrt(np.mean((y_true-y_pred)**2))
    return rmse
    
def corr_metric(y_true,y_pred):
    corr = pearsonr(y_true, y_pred)[0]
    return corr
    
def kge_coefficient(y_true, y_pred):
    """
    Kling-Gupta Efficiency (KGE) Loss Function
    """
    # Calculate mean of observations
    obs_mean = np.mean(y_true)
    
    # Calculate mean of predictions
    pred_mean = np.mean(y_pred)
    
    # Calculate standard deviation of observations
    obs_std = np.std(y_true)
    
    # Calculate standard deviation of predictions
    pred_std = np.std(y_pred)
    
    # Calculate correlation coefficient
    correlation = np.mean((y_true - obs_mean) * (y_pred - pred_mean)) / (obs_std * pred_std)
    
    # Calculate KGE
    kge = 1 - np.sqrt((correlation - 1)**2 + (pred_std / obs_std - 1)**2 + (pred_mean / obs_mean - 1)**2)
    
    return kge

def nse_coefficient(y_true, y_pred):
    """
    NSE Loss Function
    """
    # Calculate mean of observations
    obs_mean = np.mean(y_true)
    
    # Calculate standard deviation of predictions
    sum_of_square = np.sum((y_pred - y_true)**2)/np.sum((y_true - obs_mean)**2)

    nse = 1 - sum_of_square
    
    return nse

def combined_metrics_for_evaluation(y_true,y_pred):
    bias = bias_metric(y_true,y_pred)
    rmse = rmse_metric(y_true,y_pred)
    corr = corr_metric(y_true,y_pred)
    kge  = kge_coefficient(y_true, y_pred)
    nse  = nse_coefficient(y_true, y_pred)
    return bias,rmse,corr,kge,nse


if __name__=="__main__":

    # Specify the input sequence length
    seq_len = int(sys.argv[1])

    inputpath='/path_to_data_1/WMO_basin_avg_monthly'
    inputpath2='/path_to_data_2/WMO_basin_avg_daily'

    # time-varying input features
    OLprep = xr.open_dataset(os.path.join(inputpath,'OL_10km_PREP_WMO_basin_avg_2003_2020_monthly.nc'))
    OLprep = OLprep.transpose('index','ncl10','ncl11').stack(month=('ncl10','ncl11')).reset_index('month')

    OLtemp = xr.open_dataset(os.path.join(inputpath,'OL_10km_TEMP_WMO_basin_avg_2003_2020_monthly.nc'))
    OLtemp = OLtemp.transpose('index','ncl0','ncl1').stack(month=('ncl0','ncl1')).reset_index('month')

    OLlai = xr.open_dataset(os.path.join(inputpath,'OL_10km_LAI_WMO_basin_avg_2003_2020_monthly.nc'))
    OLlai = OLlai.transpose('index','ncl0','ncl1').stack(month=('ncl0','ncl1')).reset_index('month')

    OLssmc = xr.open_dataset(os.path.join(inputpath,'OL_10km_SSMC_WMO_basin_avg_2003_2020_monthly.nc'))
    OLssmc = OLssmc.transpose('index','ncl0','ncl1').stack(month=('ncl0','ncl1')).reset_index('month')
    
    OLtws = xr.open_dataset(os.path.join(inputpath,'OL_10km_TWS_WMO_basin_avg_2003_2020_monthly.nc'))
    OLtws = OLtws.transpose('index','ncl0','ncl1').stack(month=('ncl0','ncl1')).reset_index('month')

    DAlai = xr.open_dataset(os.path.join(inputpath,'DA_10km_LAI_WMO_basin_avg_2003_2020_monthly.nc'))
    DAlai = DAlai.transpose('index','ncl0','ncl1').stack(month=('ncl0','ncl1')).reset_index('month')

    DAssmc = xr.open_dataset(os.path.join(inputpath,'DA_10km_SSMC_WMO_basin_avg_2003_2020_monthly.nc'))
    DAssmc = DAssmc.transpose('index','ncl0','ncl1').stack(month=('ncl0','ncl1')).reset_index('month')
    
    DAtws = xr.open_dataset(os.path.join(inputpath,'DA_10km_TWS_WMO_basin_avg_2003_2020_monthly.nc'))
    DAtws = DAtws.transpose('index','ncl0','ncl1').stack(month=('ncl0','ncl1')).reset_index('month')

    # static input features
    LISinputELEV=xr.open_dataset(os.path.join(inputpath2,'Global_10km_ELEVATION_WMO_basin_avg.nc'))['ELEVATION'].values
    LISinputSLOPE=xr.open_dataset(os.path.join(inputpath2,'Global_10km_SLOPE_WMO_basin_avg.nc'))['SLOPE'].values
    LISinputBasinArea=xr.open_dataset(os.path.join(inputpath2,'Global_10km_BasinArea_WMO_basin_avg.nc'))['LANDMASK'].values
    LISinputCLAYfrac=xr.open_dataset(os.path.join(inputpath2,'Global_10km_Clay_fraction_WMO_basin_avg.nc'))['Band1'].values
    LISinputSILTfrac=xr.open_dataset(os.path.join(inputpath2,'Global_10km_Silt_fraction_WMO_basin_avg.nc'))['Band1'].values
    LISinputSANDfrac=xr.open_dataset(os.path.join(inputpath2,'Global_10km_Sand_fraction_WMO_basin_avg.nc'))['Band1'].values
    LISinputFORESTfrac=xr.open_dataset(os.path.join(inputpath2,'Global_10km_Forest_fraction_WMO_basin_avg.nc'))['LANDCOVER'].values
    LISinputCROPLANDfrac=xr.open_dataset(os.path.join(inputpath2,'Global_10km_Cropland_fraction_WMO_basin_avg.nc'))['LANDCOVER'].values
    
    #static seasonal features
    MeanAnnualPrep=np.mean(OLprep['PREPmo'].values,axis=1)
    MeanAnnualTemp=np.mean(OLtemp['TEMPmo'].values,axis=1)
    MeanAnnualMaxLAI=np.mean(np.max(OLlai['LAImo'].values.reshape(515,-1,12),axis=2),axis=1)

    # Mean Seasonality
    MeanSeaPrep=np.mean(OLprep['PREPmo'].values.reshape(515,-1,12),axis=1)
    MeanSeaTemp=np.mean(OLtemp['TEMPmo'].values.reshape(515,-1,12),axis=1)
    MeanSeaOLLAI=np.mean(OLlai['LAImo'].values.reshape(515,-1,12),axis=1)
    MeanSeaDALAI=np.mean(DAlai['LAImo'].values.reshape(515,-1,12),axis=1)

    LIS_static_input_vars_1D_pre = [MeanAnnualPrep, MeanAnnualTemp, MeanAnnualMaxLAI, LISinputELEV, LISinputSLOPE, LISinputBasinArea, LISinputCLAYfrac, LISinputSILTfrac, LISinputSANDfrac, LISinputFORESTfrac, LISinputCROPLANDfrac]
    LIS_static_input_vars_1D = np.concatenate(LIS_static_input_vars_1D_pre).reshape(len(LIS_static_input_vars_1D_pre),515).transpose()
    LIS_static_input_vars_SeaD = np.stack((MeanSeaPrep, MeanSeaTemp, MeanSeaOLLAI, MeanSeaDALAI),axis = 2)

    dates = pd.date_range(start='2003-01',end='2020-12',freq='MS')
    months = pd.date_range(start='2000-01', periods=12, freq='M').strftime('%b')
    basin_ids = [f'basin_{i:03}' for i in range(1, 516)]
    dynamic_vars_name = ["Precipitation","Temperature","OL_LAI","OL_SSMC","OL_TWS","DA_LAI","DA_SSMC","DA_TWS"]
    dynamic_vars = [OLprep, OLtemp, OLlai, OLssmc, OLtws, DAlai, DAssmc, DAtws]

    def prepare_dataframe(da,da_new_name):
        coords_to_drop = [coord for coord in da.coords if coord not in ["index","month"]]
        da = da.drop_vars(coords_to_drop)
        da = da.rename({"index":"basin","month":"time"})
        da = da.assign_coords(basin=basin_ids,time=dates)
        df = da.to_dataframe().reset_index()
        df = df.rename(columns={df.columns[-1]: da_new_name})
        return df
    

    dynamic_dataframes = []
    for var,varname in zip(dynamic_vars,dynamic_vars_name):
        df = prepare_dataframe(var,varname)
        dynamic_dataframes.append(df)
    
    df_dynamic = dynamic_dataframes[0]
    for df in dynamic_dataframes[1:]:
        df_dynamic = pd.merge(df_dynamic, df, on=["basin", "time"])

    static_data_1D = pd.DataFrame(
        data = LIS_static_input_vars_1D,
        columns = ['annual_avg_precipitation','annual_avg_temperature','annual_avg_lai','elevation','slope','basin_area','clay_frac','silt_frac','sand_frac','forest_frac','cropland_frac'],
        index = basin_ids
    ).reset_index().rename(columns={"index":"basin"})

    LIS_static_input_vars_SeaD_reshaped=LIS_static_input_vars_SeaD.reshape(515*12,4)
    index = pd.MultiIndex.from_product([basin_ids, months], names=["basin", "month"])
    static_sea_df = pd.DataFrame(LIS_static_input_vars_SeaD_reshaped, index=index, columns=['precipitation_climatology','temperature_climatology','OL_lai_climatology','DA_lai_climatology'])

    df_semifinal=pd.merge(df_dynamic,static_data_1D,on="basin")
    df_semifinal['month'] = df_semifinal['time'].dt.strftime('%b')
    df_final=pd.merge(df_semifinal, static_sea_df, on=['basin', 'month'])
    df_final["time_idx"]=df_final['time'].dt.year*12 + df_final['time'].dt.month
    df_final["time_idx"] -= df_final["time_idx"].min()

    def set_seed(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    set_seed()

    # Set up parameters and train|val cutoff
    max_prediction_length = 1
    max_encoder_length = seq_len
    training_cutoff = 10*12
    val_cutoff = 13*12

    # Prepare training set
    training = TimeSeriesDataSet(
        df_final[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="OL_TWS",
        group_ids=["basin"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["basin"],
        static_reals=['annual_avg_precipitation','annual_avg_temperature','annual_avg_lai','elevation','slope','basin_area','clay_frac','silt_frac','sand_frac','forest_frac','cropland_frac'],
        time_varying_known_categoricals=["month"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "Precipitation",
            "Temperature",
            "OL_LAI",
            "OL_SSMC",
        ],
        target_normalizer=GroupNormalizer(
            groups=["basin"], transformation="softplus"
        ),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Prepare val and test set
    df_val = df_final[(df_final['time_idx'] >= training_cutoff-max_encoder_length) & (df_final['time_idx'] < val_cutoff)]
    df_test = df_final[(df_final['time_idx'] > val_cutoff)]

    validation = TimeSeriesDataSet.from_dataset(training, df_val, predict=False, stop_randomization=True)

    # create dataloaders for model
    batch_size = 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    # Prepare a full set for predictions
    fulldataset = TimeSeriesDataSet(
        df_final[lambda x: x.time_idx <= 215],
        time_idx="time_idx",
        target="OL_TWS",
        group_ids=["basin"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["basin"],
        static_reals=['annual_avg_precipitation','annual_avg_temperature','annual_avg_lai','elevation','slope','basin_area','clay_frac','silt_frac','sand_frac','forest_frac','cropland_frac'],
        time_varying_known_categoricals=["month"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "Precipitation",
            "Temperature",
            "OL_LAI",
            "OL_SSMC",
        ],
        target_normalizer=GroupNormalizer(
            groups=["basin"], transformation="softplus"
        ),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Get the best set of parameters from Optuna study
    best_learning_rate = 0.00161997,
    best_hidden_size = 80,
    best_attention_head_size = 5,
    best_dropout = 0.19424359,
    best_hidden_continuous_size = 24

    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("/path_to_models/TFT/OL_lightning_logs")  # logging results to a tensorboard

    class LossTracker(pl.Callback):
        def __init__(self):
            self.train_losses = []
            self.val_losses = []

        def on_validation_epoch_end(self, trainer, pl_module):
            # Store the latest validation loss at the end of each epoch
            val_loss = trainer.callback_metrics.get("val_loss")
            self.val_losses.append(val_loss)

        def on_train_epoch_end(self, trainer, pl_module):
            # Store the latest training loss at the end of each epoch
            train_loss = trainer.callback_metrics.get("train_loss")
            self.train_losses.append(train_loss)
        
    loss_tracker = LossTracker()

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[loss_tracker, lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.00161997,
        hidden_size=80,
        attention_head_size=5,
        dropout=0.19424359,
        hidden_continuous_size=24,
        loss=QuantileLoss(),
        log_interval=1, 
        optimizer="Ranger",
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # fit network
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


    # Load the best model from the saved file
    OL_TFT_best_model_path = trainer.checkpoint_callback.best_model_path
    OL_best_tft = TemporalFusionTransformer.load_from_checkpoint(OL_TFT_best_model_path)

    # Perform predictions
    OL_TFT_prediction_time_series = []

    for i in range(515):
        prediction=OL_best_tft.predict(fulldataset.filter(lambda x: (x.basin == basin_ids[i])))
        # Move prediction tensor to CPU (if it's on GPU), then append
        prediction = prediction.cpu().numpy()
        OL_TFT_prediction_time_series.append(prediction)
    OL_TFT_prediction_time_series=np.array(OL_TFT_prediction_time_series)

    # Save the predictions for future analyses
    np.save(f'/path_to_output/OL_TFT_prediction_time_series_seq_len_{seq_len}.npy',OL_TFT_prediction_time_series)
    


