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
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset
import time
import math
from datetime import datetime
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


class EnhancedLSTM(nn.Module):
    def __init__(self, hidden_size=256, output_size=1, dropout=0.1, num_variables = 3, num_static=5, n_forecast=1, init_method="xavier"):
        super(EnhancedLSTM, self).__init__()
        
        #LSTM layer
        self.lstm = nn.LSTM(num_variables, hidden_size, num_layers=1, batch_first=True)
        
        #dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer to combine LSTM output and static inputs
        self.fc_static = nn.Linear(hidden_size+num_static, hidden_size)

        #Final output layer to forecast multiple time steps
        self.fc_out = nn.Linear(hidden_size, output_size*n_forecast)

        # Initialization method
        self.init_method = init_method
        
        #initalize weights
        self.init_weights()

        # Store the forecast horizon for reshaping the output
        self.n_forecast = n_forecast
        self.output_size = output_size
        
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                if self.init_method == "xavier":
                    nn.init.xavier_uniform_(param.data)
                elif self.init_method == "kaiming":
                    nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
                elif self.init_method == "orthogonal":
                    nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
        
    def forward(self,x_time, x_static):
        
        """
        x_time: Tensor of shape (batch_size, seq_length, num_variables)
        x_static: Tensor of shape (batch_size, num_static)
        """

        # LSTM forward pass
        lstm_out, _ = self.lstm(x_time)
        
        #Apply dropout to LSTM outputs
        lstm_out = self.dropout(lstm_out[:,-1,:])  # (batch_size, hidden_size)

        # Concatenate static inputs with LSTM output
        combined_input = torch.cat((lstm_out, x_static), dim=1)

        # Pass through the first fully connected layer
        combined_output = torch.relu(self.fc_static(combined_input))

        # Generate forecasts through the final output layer
        forecast = self.fc_out(combined_output)  # (batch_size, output_size * n_forecast)
        
        # Reshape output to (batch_size, n_forecast, output_size) for easy interpretation
        forecast = forecast.view(x_time.size(0), self.n_forecast, self.output_size)
        
        return forecast

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        """
        Initialize with a list of quantiles (e.g., [0.1, 0.5, 0.9]).
        """
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, y_pred, y_true):
        """
        Compute the quantile loss.
        Args:
            y_pred: Predicted values (batch_size, n_forecast, num_quantiles)
            y_true: Ground truth values (batch_size, n_forecast, 1)
        Returns:
            Quantile loss averaged over the batch.
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = y_true - y_pred[:, :, i:i+1]
            losses.append(torch.max(q * errors, (q - 1) * errors).mean())
        return torch.stack(losses).mean()

def create_sequences(data, time_varying_cols, static_cols, target_col, seq_length=12, forecast_length=1):
    X_time, X_static, y = [], [], []

    for basin in data['basin'].unique():
        basin_data = data[data['basin'] == basin].sort_values('time')
        
        for i in range(len(basin_data) - seq_length - forecast_length + 1):
            # Time-varying inputs
            time_seq = basin_data.iloc[i:i + seq_length][time_varying_cols].values
            
            # Static inputs (same values throughout the sequence)
            static_seq = basin_data.iloc[i][static_cols].values  # Only take static from the first time step
            
            # Target values (forecast period)
            target_seq = basin_data.iloc[i + seq_length:i + seq_length + forecast_length][target_col].values
            
            X_time.append(time_seq)
            X_static.append(static_seq)
            y.append(target_seq)

    return np.array(X_time), np.array(X_static), np.array(y)

def lstm_prediction_for_basin(basin_id, time_varying_cols,scaler_var, lstm_model_load_best):
    basin_data = df_final_standardized_LSTM[df_final_standardized_LSTM['basin']==basin_ids[basin_id]]
    static_cols = ['annual_avg_precipitation', 'annual_avg_temperature', 
               'annual_avg_lai', 'elevation', 'slope','basin_area','clay_frac','silt_frac','sand_frac','forest_frac','cropland_frac']
    # Extract the static input (same for the whole time series)
    x_static = basin_data[static_cols].iloc[0].to_numpy(dtype=np.float32)  # (num_static,)
    x_static = torch.tensor(x_static).unsqueeze(0)  # Shape: (1, num_static)

    # Prepare predictions for the full time series using sliding windows
    predictions = []

    # Loop through the time series using sliding windows
    for i in range(len(basin_data) - sequence_win):
    # Extract a sequence of time-varying data
        x_time = basin_data[time_varying_cols].iloc[i:i + sequence_win].to_numpy(dtype=np.float32)
        x_time = torch.tensor(x_time).unsqueeze(0)  # Shape: (1, seq_length, num_variables)

        # Make a prediction using the trained model
        with torch.no_grad():
            forecast = lstm_model_load_best(x_time, x_static)  # Shape: (1, n_forecast, output_size)

        # Convert prediction to numpy and store
        predictions.append(forecast.numpy())  # Shape: (n_forecast, output_size)

    # Flatten the predictions into a single array
    predictions = np.concatenate(predictions, axis=0)  # Shape: (total_series_steps,total_forecast_steps,number of quantiles)

    # Inverse transform the predictions to the original scale
    tws_scaler = scalers_LSTM[basin_ids[basin_id]][scaler_var]  # Get the TWS scaler for the specified basin
    predictions_original = tws_scaler.inverse_transform(predictions.reshape(-1,1)).reshape(predictions.shape)  # Shape: (total_series_steps, total_forecast_steps, number_of_quantiles)
    return predictions_original

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":

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

    # Define the time-varying features and target to be standardized
    time_varying_cols = ["Precipitation","Temperature","OL_LAI","OL_SSMC","OL_TWS","DA_LAI","DA_SSMC","DA_TWS"]
    static_cols = ['annual_avg_precipitation','annual_avg_temperature','annual_avg_lai','elevation','slope','basin_area','clay_frac','silt_frac','sand_frac','forest_frac','cropland_frac']

    # Apply standardization per basin using the training set for LSTM (0:132)->(2003-2013)
    df_final_standardized_LSTM, scalers_LSTM = standardize_per_basin_train_only(
        df_final, group_col='basin', value_cols=time_varying_cols, train_length=120
    )

    # Seperate train|val, select time-varying and static variables
    train_df = df_final_standardized_LSTM[df_final_standardized_LSTM['time'].dt.year < 2013]
    start_date=pd.Timestamp("2013-01-01") - pd.DateOffset(months=seq_len)
    val_df = df_final_standardized_LSTM[(df_final_standardized_LSTM['time'] >= start_date) &(df_final_standardized_LSTM['time'] < pd.Timestamp("2016-01-01") )]
    for col in static_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        val_df[col] = pd.to_numeric(val_df[col], errors='coerce')
    OL_time_varying_cols = ['Precipitation','Temperature','OL_LAI','OL_SSMC']
    DA_time_varying_cols = ['Precipitation','Temperature','DA_LAI','DA_SSMC']
    static_cols = ['annual_avg_precipitation', 'annual_avg_temperature', 'annual_avg_lai', 'elevation', 'slope','basin_area','clay_frac','silt_frac','sand_frac','forest_frac','cropland_frac']
    OL_target_col = 'OL_TWS'
    DA_target_col = 'DA_TWS'

    # Shared settings
    sequence_win = seq_len
    pred_step_size = 1

    # Use OL dataset as example here, users can switch to DA dataset if needed.
    # Create sequences for train and validation
    OL_X_time_train, X_static_train, OL_y_train = create_sequences(train_df, OL_time_varying_cols, static_cols, OL_target_col,sequence_win,pred_step_size)
    OL_X_time_val, X_static_val, OL_y_val = create_sequences(val_df, OL_time_varying_cols, static_cols, OL_target_col,sequence_win,pred_step_size)

 
    # Make sure the static data part is float type.
    X_static_train = np.array(X_static_train, dtype=np.float32)
    X_static_val = np.array(X_static_val, dtype=np.float32)

    # Convert to PyTorch tensors
    batch_size=128

    X_static_train = torch.tensor(X_static_train, dtype=torch.float32)
    X_static_val = torch.tensor(X_static_val, dtype=torch.float32)

    OL_X_time_train = torch.tensor(OL_X_time_train, dtype=torch.float32)
    OL_y_train = torch.tensor(OL_y_train, dtype=torch.float32).unsqueeze(-1)
    OL_X_time_val = torch.tensor(OL_X_time_val, dtype=torch.float32)
    OL_y_val = torch.tensor(OL_y_val, dtype=torch.float32).unsqueeze(-1)

    # Create DataLoaders
    OL_train_dataset = TensorDataset(OL_X_time_train, X_static_train, OL_y_train)
    OL_val_dataset = TensorDataset(OL_X_time_val, X_static_val, OL_y_val)

    OL_train_loader = DataLoader(OL_train_dataset, batch_size=batch_size, shuffle=True)
    OL_val_loader = DataLoader(OL_val_dataset, batch_size=batch_size*10)

    def set_seed(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    set_seed()

    # Training parameters
    quantiles=[0.5, 0.025, 0.975]
    num_epochs = 100
    patience = 10
    min_delta = 0.0001
    best_val_loss = np.inf
    counter = 0

    #Best set of hyperparameters from optuna study:
    hidden_size = 512
    l_rate = 0.001595
    dropout = 0.2
    init_method = 'xavier'

    # Training data
    train_loader = OL_train_loader
    val_loader = OL_val_loader

    save_path = f'/path_to_models/best_lstm_model_for_OL_seq_len_{seq_len}.pth'

    # Initialize the EnhancedLSTM model

    lstm_model = EnhancedLSTM(hidden_size=hidden_size, output_size=len(quantiles), dropout=dropout, 
                     num_variables=4, num_static=11, n_forecast=1, init_method=init_method).to(device)
    # Define the loss function and optimizer
    criterion = QuantileLoss(quantiles=quantiles)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=l_rate)

    # Define lists to store losses for plotting later
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        lstm_model.train()  # Set the model to training mode
        train_loss = 0.0

        for X_time_batch, X_static_batch, y_batch in train_loader:

            X_time_batch = X_time_batch.to(device)
            X_static_batch = X_static_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()  # Reset gradients

            # Forward pass
            y_pred = lstm_model(X_time_batch, X_static_batch)

            # Compute loss
            loss = criterion(y_pred, y_batch)
            train_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Validation phase
        lstm_model.eval()  # Set the model to evaluation mode
        val_loss = 0.0

        with torch.no_grad():  # Disable gradient calculation
            for X_time_batch, X_static_batch, y_batch in val_loader:

                X_time_batch = X_time_batch.to(device)
                X_static_batch = X_static_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Forward pass
                y_pred = lstm_model(X_time_batch, X_static_batch)

                # Compute validation loss
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        # Average the losses over the number of batches
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Log losses for later plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print epoch progress
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check if the validation loss improved
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss  # Update the best loss
            counter = 0  # Reset early stopping counter

            # Save the best model
            print(f"Validation loss improved to {val_loss:.4f}. Saving the best model...")
            torch.save(lstm_model.state_dict(), save_path)

        else:
            counter += 1
            print(f"No improvement for {counter} epochs.")

            # Early stopping check
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # Load the models and perform the predictions
    quantiles = [0.5, 0.025, 0.975]
    hidden_size = 512
    l_rate = 0.001595
    dropout = 0.2
    init_method = 'xavier'
    OL_lstm_model_load_best = EnhancedLSTM(hidden_size=hidden_size, output_size=len(quantiles), dropout=dropout, 
                     num_variables=4, num_static=11, n_forecast=1, init_method=init_method)

    # Load the best model's weights
    OL_lstm_model_load_best.load_state_dict(torch.load(f'/path_to_models/best_lstm_model_for_OL_seq_len_{seq_len}.pth'))

    # Save the predictions to a file for future analyses
    OL_lstm_model_load_best.eval()

    OL_LSTM_prediction_time_series = []

    for i in range(515):
        prediction=lstm_prediction_for_basin(i,OL_time_varying_cols,"OL_TWS",OL_lstm_model_load_best)
        OL_LSTM_prediction_time_series.append(prediction)
    OL_LSTM_prediction_time_series=np.array(OL_LSTM_prediction_time_series)

    np.save(f'/path_to_output/OL_LSTM_prediction_time_series_seq_len_{seq_len}.npy',OL_LSTM_prediction_time_series)
