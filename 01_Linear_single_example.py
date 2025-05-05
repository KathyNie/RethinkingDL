# Import dependencies
import os
import sys
import xarray as xr
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

def single_basin_data_prep(df,basin_ind,data_type,lag_num):
    basin_data = df[df["basin"]==basin_ids[basin_ind]]
    # Add lagged features
    for lag in range(1,lag_num):
        for col in ["Precipitation","Temperature","OL_LAI","OL_SSMC","DA_LAI","DA_SSMC"]:
            basin_data[f"{col}_lag{lag}"] = basin_data[col].shift(lag)
    # Add seasonal dummy variables(note that this function automatically removed dummy for one month to avoid colinearity)
    basin_data = pd.get_dummies(basin_data, columns=["month"], prefix="Month", drop_first=True)
    # Prepare targets
    for step in range(0, 1):
        basin_data[f"OL_TWS_t+{step}"] = basin_data["OL_TWS"].shift(-step)
        basin_data[f"DA_TWS_t+{step}"] = basin_data["DA_TWS"].shift(-step)
    basin_data = basin_data.dropna()
    if data_type == "OL":
        feature_columns =[col for col in basin_data.columns if col.startswith("Precipitation")] + \
                [col for col in basin_data.columns if col.startswith("Temperature")] + \
                [col for col in basin_data.columns if col.startswith("OL_LAI")] + \
                [col for col in basin_data.columns if col.startswith("OL_SSMC")] + \
                [col for col in basin_data.columns if col.startswith("Month")] + \
                [col for col in basin_data.columns if col.startswith("time_idx")]
        target_columns = [col for col in basin_data.columns if col.startswith("OL_TWS_t+")]
        
    elif data_type == "DA":
        feature_columns =[col for col in basin_data.columns if col.startswith("Precipitation")] + \
                [col for col in basin_data.columns if col.startswith("Temperature")] + \
                [col for col in basin_data.columns if col.startswith("DA_LAI")] + \
                [col for col in basin_data.columns if col.startswith("DA_SSMC")] + \
                [col for col in basin_data.columns if col.startswith("Month")] + \
                [col for col in basin_data.columns if col.startswith("time_idx")]
        target_columns = [col for col in basin_data.columns if col.startswith("DA_TWS_t+")]
     
    X = basin_data[feature_columns]
    Y = basin_data[target_columns]

    time=basin_data["time"]
    train_col_index = basin_data['time'].astype(str).tolist().index('2015-12-01')
    X_train = X.iloc[:train_col_index+1]
    Y_train = Y.iloc[:train_col_index+1]
    
    return X,Y,X_train,Y_train,train_col_index,time

def single_basin_quantile_regress(df,basin_ind,data_type,pred_steps,lag_num):
    X,Y,X_train,Y_train,train_col_index,time = single_basin_data_prep(df,basin_ind,data_type,lag_num)
    quantiles = [0.5, 0.025, 0.975]
    models = {step: {} for step in range(pred_steps)}
    coefficients = {step: {} for step in range(pred_steps)}
    for step in range(pred_steps):
        for q in quantiles:
            # Fit quantile regressor for each step and quantile
            model = QuantileRegressor(quantile=q, solver='highs',alpha=0)
            model.fit(X_train, Y_train.iloc[:, step])
            models[step][q] = model
            # Extract the coefficients
            coeffs = model.coef_
            coefficients[step][q] = coeffs
    #predictions
    predictions_original_scale = {step:{q: scalers_linearGlob[basin_ids[i]][f"{data_type}_TWS"].inverse_transform(model.predict(X).reshape(-1,1)).flatten() for q, model in models[step].items()} for step in range(pred_steps)}
    Y_original_scale = pd.DataFrame(scalers_linearGlob[basin_ids[i]][f"{data_type}_TWS"].inverse_transform(Y),columns=Y.columns, index=Y.index)   
    return models,coefficients, predictions_original_scale,time,Y_original_scale,train_col_index

if __name__=="__main__":

    # specify input sequence length
    lag_num = int(sys.argv[1])

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

    def standardize_per_basin_train_only(df, group_col, value_cols,train_length):
  
        df_standardized = df.copy()
        scalers = {}

        for basin in df[group_col].unique():
            scalers[basin] = {}

            # Get data for the current basin
            basin_data = df[df[group_col] == basin]

            # Split data into training and full sets
            training_data = basin_data.iloc[:train_length]

            # Apply StandardScaler to each feature separately
            for col in value_cols:
                scaler = StandardScaler()
                scaler.fit(training_data[[col]])

                # Store the scaler for later use
                scalers[basin][col] = scaler

                # Update the DataFrame with scaled values
                scaled_values = scaler.transform(basin_data[[col]])
                df_standardized.loc[df[group_col] == basin, col] = scaled_values

        return df_standardized, scalers

    # Define the time-varying features and target to be standardized
    time_varying_cols = ["Precipitation","Temperature","OL_LAI","OL_SSMC","OL_TWS","DA_LAI","DA_SSMC","DA_TWS"]
    static_cols = ['annual_avg_precipitation','annual_avg_temperature','annual_avg_lai','elevation','slope','basin_area','clay_frac','silt_frac','sand_frac','forest_frac','cropland_frac']
    # Apply standardization per basin

    # Apply standardization per basin using the training set for linear model (0:144)->(2003-2015)
    df_final_standardized_linearGlob, scalers_linearGlob = standardize_per_basin_train_only(
        df_final, group_col='basin', value_cols=time_varying_cols, train_length=144
    )

    # Use OL dataset as example here, users can switch to DA dataset if needed.
    # Train the Linear_single models and perform predictions.
    OL_group_model = []
    OL_group_coefficients = []
    OL_group_predictions = []
    OL_group_Y = []
    for i in range(515):
        models,coefficients, predictions,time,Y, train_col_index = single_basin_quantile_regress(
            df_final_standardized_linearGlob,i,"OL",1,lag_num)
        OL_group_model.append(models)
        OL_group_coefficients.append(coefficients)
        OL_group_predictions.append(predictions)
        OL_group_Y.append(Y)

    #Save the prediction results for analyses
    OL_group_predictions = np.array(OL_group_predictions)
    np.save(f'/path_to_output/np_save/OL_linear_prediction_time_series_seq_len_{lag_num}.npy',OL_group_predictions)


    

