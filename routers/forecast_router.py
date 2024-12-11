import base64
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Literal
from datetime import timedelta
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger
import io
import numpy as np


# Create APIRouter instance
router = APIRouter()

# Logger configuration
logger.add("forecasting.log", rotation="1 MB", retention="10 days", level="DEBUG")

# Constants
CATEGORY_OPTIONS = ['actual', 'forecast', 'rolling forecast', 'prediction']
FREQUENCY_OPTIONS = {'daily': 'D', 'weekly': 'W', 'monthly': 'M'}
TEMP_FILE_PATH = "temp_upload.csv"  # Path to temporarily save uploaded file

# Step 1: Upload and Analyze File
@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save file temporarily
        with open(TEMP_FILE_PATH, "wb") as buffer:
            buffer.write(file.file.read())
        logger.info("File uploaded and saved successfully.")
        
        # Load dataset
        df = pd.read_csv(TEMP_FILE_PATH)
        
        # Analyze columns
        column_info = analyze_dataset(df)
        return JSONResponse(content=column_info)
    except Exception as e:
        logger.error(f"Error uploading or analyzing file: {str(e)}")
        raise HTTPException(status_code=400, detail="Error processing uploaded file")

# Utility function to analyze dataset columns
def analyze_dataset(df: pd.DataFrame):
    timeline_keywords = ['date', 'year', 'month', 'day', 'time']
    exclusion_keywords = ['code', 'id', 'number', 'num', 'sr.no', 'no', 'patient', 'record', 'reference', 'sequence', 'entry', 'count', 'status', 'flag', 'name', 'label', 'type', 'category', 'index']
    
    time_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in timeline_keywords)]
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    continuous_columns = [col for col in numerical_columns if (not any(exclude in col.lower() for exclude in exclusion_keywords)) and df[col].nunique() > 1]

    all_columns = set(df.columns)
    time_set = set(time_columns)
    continuous_set = set(continuous_columns)
    categorical_columns = list(all_columns - time_set - continuous_set)

    column_info = {
        'Timeline_dimensions': time_columns,
        'Key_figures': continuous_columns,
        'Categorical_columns': categorical_columns
    }
    return column_info

# BaseModel for forecast request validation
class ForecastRequest(BaseModel):
    date_column: str
    target_column: str
    filter_column: Optional[str] = None
    filter_value: Optional[str] = None
    frequency: Literal['daily', 'weekly', 'monthly']
    category: Literal['actual', 'forecast', 'rolling forecast', 'prediction']
    train_upto: Optional[str] = None  # Specify up to which date to use for training
    predict_for: Optional[int] = 30   # Duration to predict (default 30 periods)
    user_role: Literal['public', 'private']


# Utility Functions for Forecasting and Rolling Forecast
def preprocess_and_filter(df, date_column, target_column, frequency, filter_column=None, filter_value=None):
    if filter_column and filter_value:
        df = df[df[filter_column] == filter_value]

    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column, target_column])

    if df[date_column].duplicated().any():
        df = df.groupby(date_column)[target_column].mean().reset_index()

    df = df.set_index(date_column)
    df = df.asfreq(FREQUENCY_OPTIONS[frequency])

    return df

def create_lag_features(df, target_column, num_lags):
    """
    Create lag features for the target column.
    """
    for lag in range(1, num_lags + 1):
        df[f"lag_{lag}"] = df[target_column].shift(lag)
    return df.dropna()

def create_features(df, target_column):
    """
    Create enhanced features including lag, rolling, and time-based features.
    """
    # Lag features
    for lag in range(1, 11):  # Experiment with up to 10 lags
        df[f"lag_{lag}"] = df[target_column].shift(lag)
    
    # Rolling statistics
    df['rolling_mean_3'] = df[target_column].rolling(window=3).mean()
    df['rolling_std_3'] = df[target_column].rolling(window=3).std()
    df['rolling_mean_7'] = df[target_column].rolling(window=7).mean()
    df['rolling_std_7'] = df[target_column].rolling(window=7).std()

    # Time-based features
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df['is_weekend'] = df.index.weekday.isin([5, 6]).astype(int)

    return df.dropna()

def perform_random_forest_with_evaluation(train_df, test_df, target_column, forecast_length):
    """
    Use Random Forest for time series forecasting with evaluation against actual data.

    Args:
        train_df (pd.DataFrame): Training dataset.
        test_df (pd.DataFrame): Testing dataset.
        target_column (str): Target column name.
        prediction_length (int): Number of steps to predict.
    
    Returns:
        pd.DataFrame: Forecasted DataFrame.
        dict: Evaluation metrics (MAE, MSE, R², Accuracy).
    """
    # Create features
    train_data = create_features(train_df.copy(), target_column)
    test_data = create_features(test_df.copy(), target_column)

    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    # Calculate the mean of y_test (for accuracy calculation)
    y_mean = np.mean(y_test)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Perform Randomized Search for Hyperparameter Tuning
    rf = RandomForestRegressor(random_state=42)
    param_distributions = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']  
    }

    ts_split = TimeSeriesSplit(n_splits=5)
    rf_search = RandomizedSearchCV(rf, param_distributions, n_iter=50, cv=ts_split, scoring='neg_mean_absolute_error', random_state=42)
    rf_search.fit(X_train, y_train)

    # Best model
    rf_model = rf_search.best_estimator_
    logger.info(f"Best Parameters: {rf_search.best_params_}")

    # Generate predictions
    predictions = rf_model.predict(X_test)

    # Evaluate predictions
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Calculate accuracy (1 - MAE / y_mean)
    accuracy = 1 - (mae / y_mean) if y_mean != 0 else None  # Avoid division by zero

    # Prepare forecast DataFrame
    forecast_df1 = pd.DataFrame({target_column: predictions}, index=test_data.index)

    # Return forecast and evaluation metrics
    evaluation_metrics = {'MAE': mae, 'MSE': mse, 'R²': r2, 'Accuracy': accuracy}
    return forecast_df1, evaluation_metrics


def plot_random_forest_forecast(train_df, forecast_df1, target_column):
    """
    Plot the Random Forest forecast along with the training data.

    Args:
        train_df (pd.DataFrame): Training dataset.
        forecast_df (pd.DataFrame): Forecasted data.
        target_column (str): The target column name.
        
    Returns:
        go.Figure: The Plotly figure object.
    """
    fig = go.Figure()

    # Add actual data to plot
    fig.add_trace(go.Scatter(
        x=train_df.index, 
        y=train_df[target_column], 
        mode='lines', 
        name='Actual Data', 
        line=dict(color='blue')
    ))

    # Add forecasted data to plot
    fig.add_trace(go.Scatter(
        x=forecast_df1.index, 
        y=forecast_df1[target_column], 
        mode='lines', 
        name='Random Forest Prediction', 
        line=dict(color='green', dash='dash')
    ))

    # Customize layout
    fig.update_layout(
        title="Random Forest Forecast with Historical Data",
        xaxis_title="Date",
        yaxis_title=target_column,
        template="plotly_dark",
        hovermode="x"
    )

    return fig

def perform_forecasting_arima(train_df, target_column, forecast_length, frequency='D'):
    model = ARIMA(train_df[target_column], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_length)
    last_date = train_df.index[-1]
    forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_length, freq=FREQUENCY_OPTIONS[frequency])
    forecast_df = pd.DataFrame({target_column: forecast}, index=forecast_index)
    return forecast_df

# Plot functions for each category
def plot_actual(train_df, target_column):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df.index, y=train_df[target_column], mode='lines', name='Actual Data', line=dict(color='blue')))
    fig.update_layout(title="Historical Data (Actual)", xaxis_title="Date", yaxis_title=target_column, template="plotly_dark", hovermode="x")
    return fig

def plot_forecast(train_df, forecast_df, target_column):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df.index, y=train_df[target_column], mode='lines', name='Actual Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[target_column], mode='lines', name='Forecast', line=dict(color='orange')))
    fig.update_layout(title="Forecast with Historical Data", xaxis_title="Date", yaxis_title=target_column, template="plotly_dark", hovermode="x")
    return fig

def plot_rolling_forecast(df, date_column, target_column, window_size, forecast_length, frequency):
    rolling_forecast_df = df.copy()
    for _ in range(forecast_length):
        rolling_forecast = perform_forecasting_arima(rolling_forecast_df[-window_size:], target_column, forecast_length=1, frequency=frequency)
        rolling_forecast_df = pd.concat([rolling_forecast_df, rolling_forecast])
        rolling_forecast_df = rolling_forecast_df.shift(-1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[target_column], mode='lines', name='Actual Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=rolling_forecast_df.index[-forecast_length:], y=rolling_forecast_df[target_column][-forecast_length:], mode='lines', name='Rolling Forecast', line=dict(color='orange')))
    fig.update_layout(title="Rolling Forecast with Drop/Add Approach", xaxis_title="Date", yaxis_title=target_column, template="plotly_dark", hovermode="x")
    return fig

# Access Control for Forecasting
def check_access(user_role, category):
    if user_role == 'private':
        logger.info(f"Private access for {category}. Only the creator can modify and view.")
        return True
    elif user_role == 'public':
        logger.info(f"Public access for {category}. Data can be shared with others.")
        return True
    else:
        logger.error("Invalid user role.")
        return False


# Step 2: Forecasting with User-Selected Columns
# Update the 'prediction' category in the `/forecast` endpoint

@router.post("/forecast")
async def forecast(request: ForecastRequest):
    # Load dataset from saved file
    try:
        df = pd.read_csv(TEMP_FILE_PATH)
        logger.info("Loaded temporary file for forecasting.")
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Temporary file not found. Please upload the file again.")

    # Preprocess data
    df = preprocess_and_filter(df, request.date_column, request.target_column, request.frequency, request.filter_column, request.filter_value)

    # Access control check
    if not check_access(request.user_role, request.category):
        raise HTTPException(status_code=403, detail="Access denied due to insufficient permissions.")

    # Generate plot based on selected category
    if request.category == 'actual':
        fig = plot_actual(df, request.target_column)
    elif request.category == 'forecast':
        forecast_df = perform_forecasting_arima(df, request.target_column, forecast_length=request.forecast_length, frequency=request.frequency)
        fig = plot_forecast(df, forecast_df, request.target_column)
    elif request.category == 'rolling forecast':
        fig = plot_rolling_forecast(df, request.date_column, request.target_column, window_size=len(df) - request.forecast_length, forecast_length=request.forecast_length, frequency=request.frequency)
    start_date = df.index.min()
    end_date = df.index.max()

    # If category is 'prediction', display available timeline
    if request.category == 'prediction':
        if not request.train_upto or not request.predict_for:
            return JSONResponse(content={
                "message": "Please specify 'train_upto' and 'predict_for' parameters.",
                "available_timeline": {"start_date": str(start_date), "end_date": str(end_date)}
            })

        # Convert `train_upto` to datetime and validate
        train_upto = pd.to_datetime(request.train_upto, errors='coerce')
        if train_upto is None or train_upto < start_date or train_upto > end_date:
            raise HTTPException(status_code=400, detail=f"'train_upto' must be between {start_date} and {end_date}.")

        # Calculate train and test splits
        train_df = df[df.index <= train_upto]
        prediction_start_date = train_upto + timedelta(days=1)
        prediction_end_date = prediction_start_date + timedelta(days=request.predict_for - 1)

        if prediction_end_date > end_date:
            raise HTTPException(
                status_code=400,
                detail=f"Prediction range exceeds available data. Prediction end date should be <= {end_date}."
            )

        # Use Random Forest to predict
        forecast_df, evaluation_metrics = perform_random_forest_with_evaluation(
            train_df, 
            df[df.index >= prediction_start_date].iloc[:request.predict_for], 
            request.target_column, 
            request.predict_for
        )
        logger.info(f"Evaluation Metrics: {evaluation_metrics}")

        fig = plot_random_forest_forecast(train_df, forecast_df, request.target_column)

    else:
        raise HTTPException(status_code=400, detail="Invalid category selected.")

    # Save plot to a buffer and return
    img_bytes = io.BytesIO()
    fig.write_image(img_bytes, format='png')
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/png")
    
