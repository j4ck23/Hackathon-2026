from pyexpat import features

from flask import Flask, jsonify, render_template, request
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn.metrics
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import *
from sklearn.linear_model import LinearRegression
app = Flask(__name__)

# This route "/" is the default page, this will run when the browerser runs the URL
@app.route("/")
def index():
    return render_template("index.html")

# This route "/dropdown" is called by a script in the index.html to get the options for the dropdown menu
@app.route("/dropdown")
def dropdown():
    df = pd.read_csv("combined_data.csv") # Read the CSV file
    crops = df["Crop"].unique().tolist() # Get the unique values from the "Crop" column and convert to a list 

    options = [{"value": c, "label": c} for c in crops]# Create a list of dictionaries with "value" and "label" keys for each crop

    return jsonify(options)

# This is called by a acript in the index.html
@app.route("/runModelGlobal")
def runModelGlobal():
    def pre_process(X, y, look_back):
        X_out, y_out = [], []

        for i in range(look_back, len(X)):
            X_out.append(X[i - look_back:i, :])
            y_out.append(y[i, 0])

        return np.array(X_out), np.array(y_out)
    
    # Read the seed from the query string e.g. /run?seed=42, default to 42 if not provided
    seed = int(request.args.get("seed", 42))
    split_year = int(request.args.get("split_year", 2010))
    features_string = request.args.get("features", "")
    features = [f for f in features_string.split(",") if f]
    crop = request.args.get("status") # Read the status (crop) from the query string 

    # # dataset path
    dataset_path = "combined_data.csv"
    dataframe = pd.read_csv(dataset_path)

    df = dataframe.copy()
    df = df[df["Crop"] == crop] # Filter the DataFrame to include only rows where the "Crop" column matches the selected crop
    df["Area"] = df["Area"].astype("category")
    df["Crop"] = df["Crop"].astype("category")

    df["average_rain_fall_mm_per_year"] = pd.to_numeric(df["average_rain_fall_mm_per_year"], errors="coerce")
    cols = [
    "Area",
    "Year",
    "Crop",
    "average_rain_fall_mm_per_year",
    "avg_temp",
    "pesticide_amount",
    "Crop_Yield",
    ]
    df = df.dropna(subset=cols)
    df = df.sort_values('Year')

    train_df = df[df['Year'] < split_year]
    test_df = df[df['Year'] >= split_year]

    X_train = train_df[features]
    y_train = train_df["Crop_Yield"]

    X_test = test_df[features]
    y_test = test_df["Crop_Yield"]

    #Create, fit and run out model
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=seed, enable_categorical=True)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # results /  actuals
    r2 = sklearn.metrics.r2_score(y_test, preds)
    mse = sklearn.metrics.mean_squared_error(y_test, preds)


    #--------------------------------------------------------------------------------------------------LSTM model----------------------------------------------------------------------------

    df1 = df.copy()
    df1 = df1.sort_values('Year')

    categorical_cols = df1.select_dtypes(include=["object", "category"]).columns.tolist() # Get the list of categorical columns in the DataFrame
    df1 = pd.get_dummies(df1, columns=categorical_cols) # Convert categorical columns to dummy variables (one-hot encoding)

    train_df_lstm = df1[df1['Year'] < split_year] # Filter the DataFrame to include only rows where the "Year" column is less than the split_year - repeated from early to not mess with the results of the XGBoost model
    test_df_lstm  = df1[df1['Year'] >= split_year]

    train_df_lstm, test_df_lstm = train_df_lstm.align(test_df_lstm, join='left', axis=1, fill_value=0) # Align the train and test DataFrames to have the same columns, filling missing values with 0
    features_lstm = [col for col in train_df_lstm.columns if col not in ["Crop_Yield",]] # Get the list of feature columns for the LSTM model, excluding "Crop_Yield" 
    X_train_raw = train_df_lstm[features_lstm].values # Extract the feature values for the training set as a NumPy array
    X_test_raw  = test_df_lstm[features_lstm].values

    y_train_raw = train_df_lstm["Crop_Yield"].values.reshape(-1, 1) # Extract the target variable values for the training set as a NumPy array and reshape to be a 2D array with one column
    y_test_raw  = test_df_lstm["Crop_Yield"].values.reshape(-1, 1)

    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train_raw) # Scale the feature values for the training set to be between 0 and 1 using MinMaxScaler
    X_test_scaled  = feature_scaler.transform(X_test_raw)

    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_raw)
    y_test_scaled  = target_scaler.transform(y_test_raw)

    look_back = max(1, min(5, len(train_df_lstm) // 3)) # Set the look_back parameter to be between 1 and 5, or less if the training set is very small

    X_combined = np.vstack((X_train_scaled[-look_back:], X_test_scaled))
    y_combined = np.vstack((y_train_scaled[-look_back:], y_test_scaled))

    X_lstm_train, y_lstm_train = pre_process(X_train_scaled, y_train_scaled, look_back)
    X_lstm_test, y_lstm_test = pre_process(X_combined, y_combined, look_back)

    model_LSTM = Sequential()
    model_LSTM.add(LSTM(16, input_shape=(X_lstm_train.shape[1], X_lstm_train.shape[2])))
    model_LSTM.add(Dense(8))
    model_LSTM.add(Dense(1))
    model_LSTM.compile(loss='mean_squared_error', optimizer='adam')
    model_LSTM.fit(X_lstm_train, y_lstm_train, epochs=100, batch_size=50, verbose=2)

    pred = model_LSTM.predict(X_lstm_test)
    pred_transform = target_scaler.inverse_transform(pred)
    te = target_scaler.inverse_transform(y_lstm_test.reshape(-1,1))
    LSTM_rmse = np.sqrt(sklearn.metrics.mean_squared_error(te, pred_transform))
    LSTM_r2 = sklearn.metrics.r2_score(te, pred_transform)

#------------------------------------------Linear Regression model----------------------------------------------------------------------------
    x_train_lin = X_train.apply(pd.to_numeric, errors='coerce')
    y_train_lin = y_train.apply(pd.to_numeric, errors='coerce')
    x_test_lin = X_test.apply(pd.to_numeric, errors='coerce')
    y_test_lin = y_test.apply(pd.to_numeric, errors='coerce')
    x_train_lin.fillna(0, inplace=True)
    y_train_lin.fillna(0, inplace=True)
    x_test_lin.fillna(0, inplace=True)
    y_test_lin.fillna(0, inplace=True)
    
    lin_model = LinearRegression()
    lin_model.fit(x_train_lin, y_train_lin)
    lin_preds = lin_model.predict(x_test_lin)

    lin_r2 = sklearn.metrics.r2_score(y_test_lin, lin_preds)
    lin_mse = sklearn.metrics.mean_squared_error(y_test_lin, lin_preds)
#---------------------------------------------------------Resulsts---------------------------------------------------------------------------------------------------------------------------------------
    results = [
        {"index": i, 
         "actual": round(float(y_test.iloc[i]), 2), 
         "predicted": round(float(preds[i]), 2)
        }
        for i in range(min(50, len(y_test)))
    ]

    results_LSTM = [
        {"index": i, 
         "actual": round(float(te[i][0]), 2), 
         "predicted": round(float(pred_transform[i][0]), 2)
        }
        for i in range(min(50, len(te)))
    ]

    results_linear = [
        {"index": i, 
         "actual": round(float(y_test_lin.iloc[i]), 2), 
         "predicted": round(float(lin_preds[i]), 2)
        }
        for i in range(min(50, len(y_test_lin)))
    ]

    # metrics
    metrics = {
        "r2": round(r2, 4),
        "mse": round(mse, 4),
        "LSTM_rmse": round(LSTM_rmse, 4),
        "LSTM_r2": round(LSTM_r2, 4),
        "lin_r2": round(lin_r2, 4),
        "lin_mse": round(lin_mse, 4)
    }

    #feature importance results
    importances = [
        {"feature": col, "importance": round(float(v), 4)}
        for col, v in zip(X_train.columns, model.feature_importances_)
    ]
    #return these results to script in index.html
    return jsonify({"results": results, "importances": importances, "metrics": metrics, "results_LSTM": results_LSTM, "results_linear": results_linear})


# Start the server if this file is run directly (python app.py)
if __name__ == "__main__":
    # host="0.0.0.0":
    # listen on all network interfaces without it Flask only listens inside the container 
    # and your browser can't reach it.
    app.run(host="0.0.0.0", port=5000, debug=True)
