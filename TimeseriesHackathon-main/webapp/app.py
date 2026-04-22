from flask import Flask, jsonify, render_template, request
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn.metrics
app = Flask(__name__)

# This route "/" is the default page, this will run when the browerser runs the URL
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dropdown")
def dropdown():
    df = pd.read_csv("combined_data.csv") # Read the CSV file
    crops = df["Crop"].unique().tolist() # Get the unique values from the "Crop" column and convert to a list 

    options = [{"value": c, "label": c} for c in crops]# Create a list of dictionaries with "value" and "label" keys for each crop

    return jsonify(options)

# This is called by a acript in the index.html
@app.route("/runModelGlobal")
def runModelGlobal():
    # Read the seed from the query string e.g. /run?seed=42, default to 42 if not provided
    seed = int(request.args.get("seed", 42))
    split_year = int(request.args.get("split_year", 2010))
    features_string = request.args.get("features", "")
    features = [f for f in features_string.split(",") if f]
    crop = (request.args.get("status")) # Read the status (crop) from the query string 

    # # dataset path
    dataset_path = "combined_data.csv"
    dataframe = pd.read_csv(dataset_path)

    df = dataframe.copy()
    df = df[df["Crop"] == crop]
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

    results = [
        {"index": i, 
         "actual": round(float(y_test.iloc[i]), 2), 
         "predicted": round(float(preds[i]), 2)
        }
        for i in range(min(50, len(y_test)))
    ]

    # metrics
    metrics = {
        "r2": round(r2, 4),
        "mse": round(mse, 4)
    }

    #feature importance results
    importances = [
        {"feature": col, "importance": round(float(v), 4)}
        for col, v in zip(X_train.columns, model.feature_importances_)
    ]
    #return these results to script in index.html
    return jsonify({"results": results, "importances": importances, "metrics": metrics})


# Start the server if this file is run directly (python app.py)
if __name__ == "__main__":
    # host="0.0.0.0":
    # listen on all network interfaces without it Flask only listens inside the container 
    # and your browser can't reach it.
    app.run(host="0.0.0.0", port=5000, debug=True)
