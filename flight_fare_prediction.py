import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from scipy.stats import randint, uniform

warnings.filterwarnings("ignore")

# ── File paths  (update if your files are elsewhere) ────────
TRAIN_PATH = r"C:\Users\amank\Downloads\Data_Train.xlsx"
TEST_PATH  = r"C:\Users\amank\Downloads\Test_set.xlsx"
RF_PATH    = r"C:\Users\amank\Downloads\rf_model.pkl"
XGB_PATH   = r"C:\Users\amank\Downloads\xgb_model.pkl"
META_PATH  = r"C:\Users\amank\Downloads\meta.pkl"
ENC_PATH   = r"C:\Users\amank\Downloads\encoders.pkl"
STACK_PATH = r"C:\Users\amank\Downloads\stack_model.pkl"


# ============================================================
#  SECTION 1 — FEATURE ENGINEERING
# ============================================================

def parse_duration(d):
    d = str(d)
    hours   = int(d.split("h")[0].strip()) if "h" in d else 0
    minutes = int(d.split("h")[-1].replace("m","").strip()) if "m" in d else 0
    return hours * 60 + minutes

def time_slot(h):
    if h < 6:   return 0   # Early morning
    elif h < 12: return 1   # Morning
    elif h < 18: return 2   # Afternoon
    else:        return 3   # Evening/Night

def get_season(m):
    if m in [11, 12, 1, 2]: return 0   # Winter
    elif m in [3, 4, 5, 6]: return 1   # Summer
    else:                    return 2   # Monsoon

TIER_MAP = {
    "Jet Airways": 3, "Vistara": 3, "Air India": 3,
    "Jet Airways Business": 3,
    "Multiple carriers": 2, "IndiGo": 2, "GoAir": 2,
    "SpiceJet": 1, "Air Asia": 1, "Trujet": 1
}

def engineer_features(df):
    df = df.copy()
    df.dropna(inplace=True)

    # Date
    date_col = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    df["Journey_Day"]        = date_col.dt.day
    df["Journey_Month"]      = date_col.dt.month
    df["Journey_Weekday"]    = date_col.dt.weekday
    df["Is_Weekend"]         = (date_col.dt.weekday >= 5).astype(int)
    df["Is_Month_Start"]     = (date_col.dt.day <= 5).astype(int)
    df["Is_Month_End"]       = (date_col.dt.day >= 25).astype(int)
    df["Season"]             = date_col.dt.month.apply(get_season)
    df["Is_Peak_Month"]      = date_col.dt.month.isin([5, 6, 10, 12]).astype(int)
    df["Quarter"]            = date_col.dt.quarter
    df.drop("Date_of_Journey", axis=1, inplace=True)

    # Departure time
    dep = pd.to_datetime(df["Dep_Time"])
    df["Dep_Hour"]            = dep.dt.hour
    df["Dep_Minute"]          = dep.dt.minute
    df["Dep_Time_Slot"]       = df["Dep_Hour"].apply(time_slot)
    df["Is_Early_Morning_Dep"]= (df["Dep_Hour"] < 6).astype(int)
    df["Is_Late_Night_Dep"]   = (df["Dep_Hour"] >= 21).astype(int)
    df.drop("Dep_Time", axis=1, inplace=True)

    # Arrival time
    arr = pd.to_datetime(df["Arrival_Time"])
    df["Arrival_Hour"]        = arr.dt.hour
    df["Arrival_Minute"]      = arr.dt.minute
    df["Arr_Time_Slot"]       = df["Arrival_Hour"].apply(time_slot)
    df.drop("Arrival_Time", axis=1, inplace=True)

    # Duration
    df["Duration_Mins"]       = df["Duration"].apply(parse_duration)
    df["Duration_Hours"]      = df["Duration_Mins"] / 60.0
    df["Duration_Sq"]         = df["Duration_Mins"] ** 2          # captures non-linearity
    df["Is_Short_Flight"]     = (df["Duration_Mins"] < 120).astype(int)
    df["Is_Long_Flight"]      = (df["Duration_Mins"] > 300).astype(int)
    df.drop("Duration", axis=1, inplace=True)

    # Stops
    stop_map = {"non-stop":0,"1 stop":1,"2 stops":2,"3 stops":3,"4 stops":4}
    df["Total_Stops"]         = df["Total_Stops"].map(stop_map)
    df["Is_Non_Stop"]         = (df["Total_Stops"] == 0).astype(int)
    df["Is_Multi_Stop"]       = (df["Total_Stops"] >= 2).astype(int)

    # Airline tier
    df["Airline_Tier"]        = df["Airline"].map(TIER_MAP).fillna(2).astype(int)

    df.drop(["Route","Additional_Info"], axis=1, errors="ignore", inplace=True)

    # Interaction features
    df["Stops_x_Duration"]    = df["Total_Stops"] * df["Duration_Mins"]
    df["Tier_x_Stops"]        = df["Airline_Tier"] * df["Total_Stops"]
    df["Tier_x_Duration"]     = df["Airline_Tier"] * df["Duration_Mins"]
    df["Peak_x_Stops"]        = df["Is_Peak_Month"] * df["Total_Stops"]
    df["Weekend_x_Tier"]      = df["Is_Weekend"] * df["Airline_Tier"]
    df["DepSlot_x_Stops"]     = df["Dep_Time_Slot"] * df["Total_Stops"]
    df["Season_x_Tier"]       = df["Season"] * df["Airline_Tier"]
    df["Month_x_Stops"]       = df["Journey_Month"] * df["Total_Stops"]
    df["Tier_x_NonStop"]      = df["Airline_Tier"] * df["Is_Non_Stop"]
    df["Duration_x_NonStop"]  = df["Duration_Mins"] * df["Is_Non_Stop"]

    return df


# ============================================================
#  SECTION 2 — OUTLIER REMOVAL
# ============================================================

def remove_outliers(X, y, factor=2.5):
    """
    Remove rows where Price is beyond factor * IQR from Q1/Q3.
    Keeps the model focused on the common price range.
    """
    Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
    IQR    = Q3 - Q1
    mask   = (y >= Q1 - factor * IQR) & (y <= Q3 + factor * IQR)
    removed = (~mask).sum()
    print(f"[INFO] Outlier removal: {removed} rows removed "
          f"(price < ₹{Q1 - factor*IQR:.0f} or > ₹{Q3 + factor*IQR:.0f})")
    return X[mask], y[mask]


# ============================================================
#  SECTION 3 — TRAIN
# ============================================================

def train():
    print("\n" + "="*62)
    print("  STEP 1 — TRAINING (High-Accuracy Edition)")
    print("="*62)

    df       = pd.read_excel(TRAIN_PATH)
    df_clean = engineer_features(df)
    print(f"[INFO] Data after feature engineering: {df_clean.shape}")
    print(f"[INFO] Features ({df_clean.shape[1]-1}): {[c for c in df_clean.columns if c != 'Price']}\n")

    # Label encode
    encoders = {}
    for col in ["Airline","Source","Destination"]:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        encoders[col] = le
    joblib.dump(encoders, ENC_PATH)

    X = df_clean.drop("Price", axis=1)
    y = df_clean["Price"]

    # Remove outliers
    X, y = remove_outliers(X, y, factor=2.5)

    # Log-transform target (reduces right skew → better learning)
    y_log = np.log1p(y)

    X_train, X_val, y_train_log, y_val_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    y_val_orig = np.expm1(y_val_log)   # original scale for metrics

    print(f"[INFO] Train: {X_train.shape}  |  Val: {X_val.shape}")
    print(f"[INFO] Price range (after outlier removal): "
          f"₹{y.min():.0f} – ₹{y.max():.0f}\n")

    # ── Random Forest with tuning ────────────────────────────
    print("[INFO] Tuning Random Forest (RandomizedSearchCV) ...")
    rf_params = {
        "n_estimators"    : randint(400, 800),
        "max_depth"       : randint(12, 25),
        "min_samples_split": randint(2, 6),
        "min_samples_leaf": randint(1, 3),
        "max_features"    : ["sqrt", "log2", 0.6, 0.8],
    }
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(
        rf_base, rf_params, n_iter=20, cv=3,
        scoring="neg_mean_absolute_error",
        random_state=42, n_jobs=-1, verbose=0
    )
    rf_search.fit(X_train, y_train_log)
    rf = rf_search.best_estimator_
    print(f"   Best RF params : {rf_search.best_params_}")

    rf_preds_log  = rf.predict(X_val)
    rf_preds_orig = np.expm1(rf_preds_log)

    # ── XGBoost with tuning ──────────────────────────────────
    print("\n[INFO] Tuning XGBoost (RandomizedSearchCV) ...")
    xgb_params = {
        "n_estimators"    : randint(500, 1000),
        "learning_rate"   : uniform(0.01, 0.05),
        "max_depth"       : randint(5, 9),
        "subsample"       : uniform(0.7, 0.25),
        "colsample_bytree": uniform(0.6, 0.35),
        "min_child_weight": randint(1, 6),
        "gamma"           : uniform(0, 0.3),
        "reg_alpha"       : uniform(0, 0.5),
        "reg_lambda"      : uniform(1, 2),
    }
    xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_params, n_iter=20, cv=3,
        scoring="neg_mean_absolute_error",
        random_state=42, n_jobs=-1, verbose=0
    )
    xgb_search.fit(X_train, y_train_log)
    xgb_model = xgb_search.best_estimator_
    print(f"   Best XGB params: {xgb_search.best_params_}")

    xgb_preds_log  = xgb_model.predict(X_val)
    xgb_preds_orig = np.expm1(xgb_preds_log)

    # ── Stacking meta-model ──────────────────────────────────
    print("\n[INFO] Training stacking meta-model (Ridge) ...")
    stack_X_val = np.column_stack([rf_preds_log, xgb_preds_log])
    # Generate OOF (out-of-fold) predictions for meta-model training
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_oof  = np.zeros(len(X_train))
    xgb_oof = np.zeros(len(X_train))
    for tr_idx, oof_idx in kf.split(X_train):
        # RF OOF
        rf_fold = RandomForestRegressor(**rf_search.best_params_, random_state=42, n_jobs=-1)
        rf_fold.fit(X_train.iloc[tr_idx], y_train_log.iloc[tr_idx])
        rf_oof[oof_idx] = rf_fold.predict(X_train.iloc[oof_idx])
        # XGB OOF
        xgb_fold = xgb.XGBRegressor(**xgb_search.best_params_, random_state=42, n_jobs=-1, verbosity=0)
        xgb_fold.fit(X_train.iloc[tr_idx], y_train_log.iloc[tr_idx])
        xgb_oof[oof_idx] = xgb_fold.predict(X_train.iloc[oof_idx])

    stack_X_train = np.column_stack([rf_oof, xgb_oof])
    stack_model   = Ridge(alpha=1.0)
    stack_model.fit(stack_X_train, y_train_log)
    joblib.dump(stack_model, STACK_PATH)

    # Stack predictions
    stack_preds_log  = stack_model.predict(stack_X_val)
    stack_preds_orig = np.expm1(stack_preds_log)

    # ── Find optimal ensemble weights (minimize MAE) ─────────
    best_mae, best_w = 999999, (0.4, 0.4, 0.2)
    for w1 in np.arange(0.2, 0.7, 0.05):
        for w2 in np.arange(0.2, 0.7, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0.05 or w3 > 0.5:
                continue
            blend = w1*rf_preds_orig + w2*xgb_preds_orig + w3*stack_preds_orig
            mae   = mean_absolute_error(y_val_orig, blend)
            if mae < best_mae:
                best_mae = mae
                best_w   = (round(w1,2), round(w2,2), round(w3,2))
    w_rf, w_xgb, w_stack = best_w
    print(f"\n[INFO] Optimal ensemble weights → RF:{w_rf}  XGB:{w_xgb}  Stack:{w_stack}")

    ens_preds = w_rf*rf_preds_orig + w_xgb*xgb_preds_orig + w_stack*stack_preds_orig

    # ── Print metrics ─────────────────────────────────────────
    def get_metrics(actual, predicted):
        mae  = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2   = r2_score(actual, predicted)
        return mae, rmse, r2

    metrics = {}
    results = [
        ("Random Forest", rf_preds_orig),
        ("XGBoost",       xgb_preds_orig),
        ("Stack (Ridge)", stack_preds_orig),
        ("Ensemble",      ens_preds),
    ]
    print("\n" + "="*64)
    print(f"  {'Model':<22} {'MAE (₹)':>10}  {'RMSE (₹)':>10}  {'R²':>8}")
    print(f"  {'-'*60}")
    for name, preds in results:
        mae, rmse, r2 = get_metrics(y_val_orig, preds)
        metrics[name] = {"mae": mae, "rmse": rmse, "r2": r2}
        print(f"  {name:<22} {mae:>10.0f}  {rmse:>10.0f}  {r2:>8.4f}")
    print("="*64)

    # Save everything
    joblib.dump(rf,        RF_PATH)
    joblib.dump(xgb_model, XGB_PATH)
    joblib.dump({
        "metrics"     : metrics,
        "feature_cols": list(X.columns),
        "weights"     : best_w,
    }, META_PATH)
    print(f"\n[INFO] All models saved.")

    # ── Plots ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Flight Fare Prediction — Training Results", fontsize=14, fontweight="bold")

    # Actual vs Predicted
    axes[0].scatter(y_val_orig, ens_preds, alpha=0.3, s=12, color="#378ADD")
    lims = [y_val_orig.min(), y_val_orig.max()]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    axes[0].set_title("Actual vs Predicted (Ensemble)")
    axes[0].set_xlabel("Actual Price (INR)")
    axes[0].set_ylabel("Predicted Price (INR)")
    axes[0].legend()

    # Feature Importance top 15
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(15).sort_values()
    bar_colors = ["#378ADD" if v > feat_imp.median() else "#B5D4F4" for v in feat_imp]
    axes[1].barh(feat_imp.index, feat_imp.values, color=bar_colors)
    axes[1].set_title("Top 15 Feature Importances (RF)")
    axes[1].set_xlabel("Importance Score")

    # MAE & RMSE comparison
    model_names = ["Random Forest", "XGBoost", "Stack (Ridge)", "Ensemble"]
    mae_vals    = [metrics[m]["mae"]  for m in model_names]
    rmse_vals   = [metrics[m]["rmse"] for m in model_names]
    x = np.arange(len(model_names))
    w = 0.35
    axes[2].bar(x - w/2, mae_vals,  w, label="MAE",  color="#378ADD")
    axes[2].bar(x + w/2, rmse_vals, w, label="RMSE", color="#D85A30")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(model_names, fontsize=8, rotation=10)
    axes[2].set_title("MAE vs RMSE by Model")
    axes[2].set_ylabel("Error (INR)")
    axes[2].legend()
    for i, (m, r) in enumerate(zip(mae_vals, rmse_vals)):
        axes[2].text(i-w/2, m+20, f"₹{m:.0f}", ha="center", fontsize=7)
        axes[2].text(i+w/2, r+20, f"₹{r:.0f}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(r"C:\Users\amank\Downloads\training_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[INFO] Plot saved → training_results.png")


# ============================================================
#  SECTION 4 — BULK TEST SET PREDICTION
# ============================================================

def predict_test_set():
    print("\n" + "="*62)
    print("  STEP 2 — BULK TEST SET PREDICTION")
    print("="*62)

    rf          = joblib.load(RF_PATH)
    xgb_model   = joblib.load(XGB_PATH)
    stack_model = joblib.load(STACK_PATH)
    encoders    = joblib.load(ENC_PATH)
    meta        = joblib.load(META_PATH)
    feat_cols   = meta["feature_cols"]
    w_rf, w_xgb, w_stack = meta["weights"]
    val_metrics = meta["metrics"]

    df_test  = pd.read_excel(TEST_PATH)
    df_clean = engineer_features(df_test)

    for col in ["Airline","Source","Destination"]:
        df_clean[col] = encoders[col].transform(df_clean[col].astype(str))

    X_test        = df_clean[feat_cols]
    rf_log        = rf.predict(X_test)
    xgb_log       = xgb_model.predict(X_test)
    stack_log     = stack_model.predict(np.column_stack([rf_log, xgb_log]))

    rf_preds      = np.expm1(rf_log)
    xgb_preds     = np.expm1(xgb_log)
    stack_preds   = np.expm1(stack_log)
    ens_preds     = w_rf*rf_preds + w_xgb*xgb_preds + w_stack*stack_preds

    out = df_test[["Airline","Source","Destination"]].copy().reset_index(drop=True)
    out["RF_Price"]       = rf_preds.round(0).astype(int)
    out["XGB_Price"]      = xgb_preds.round(0).astype(int)
    out["Stack_Price"]    = stack_preds.round(0).astype(int)
    out["Ensemble_Price"] = ens_preds.round(0).astype(int)
    out.to_csv(r"C:\Users\amank\Downloads\predictions.csv", index=False)

    print(f"[INFO] Saved → predictions.csv  ({len(out)} rows)")
    print(out.head(8).to_string(index=False))

    print("\n  [Validation Metrics Reference]")
    print(f"  {'Model':<22} {'MAE (₹)':>10}  {'RMSE (₹)':>10}  {'R²':>8}")
    print(f"  {'-'*54}")
    for name in ["Random Forest","XGBoost","Stack (Ridge)","Ensemble"]:
        m = val_metrics[name]
        print(f"  {name:<22} {m['mae']:>10.0f}  {m['rmse']:>10.0f}  {m['r2']:>8.4f}")


# ============================================================
#  SECTION 5 — PREDICT A SINGLE FLIGHT WITH MAE/RMSE
# ============================================================

def predict_fare(airline, source, destination,
                 journey_date, dep_time, arrival_time, total_stops):
    """
    Predict fare for any single flight.
    Shows RF, XGB, Stack and Ensemble predictions
    each with their MAE and RMSE from validation.

    Parameters
    ----------
    airline       : str   e.g. "IndiGo"
    source        : str   e.g. "Delhi"
    destination   : str   e.g. "Cochin"
    journey_date  : str   "DD/MM/YYYY"
    dep_time      : str   "HH:MM"  24-hour
    arrival_time  : str   "HH:MM"  24-hour
    total_stops   : int   0=non-stop | 1 | 2 | 3
    """
    rf          = joblib.load(RF_PATH)
    xgb_model   = joblib.load(XGB_PATH)
    stack_model = joblib.load(STACK_PATH)
    encoders    = joblib.load(ENC_PATH)
    meta        = joblib.load(META_PATH)
    feat_cols   = meta["feature_cols"]
    w_rf, w_xgb, w_stack = meta["weights"]
    val_m       = meta["metrics"]

    # Validate
    for col, val in [("Airline",airline),("Source",source),("Destination",destination)]:
        known = list(encoders[col].classes_)
        if val not in known:
            print(f"\n[ERROR] '{val}' is not a valid {col}.")
            print(f"  Valid: {known}\n")
            return None

    # Parse inputs
    date_obj = pd.to_datetime(journey_date, format="%d/%m/%Y")
    month    = date_obj.month
    dh, dm   = map(int, dep_time.split(":"))
    ah, am   = map(int, arrival_time.split(":"))
    dur      = (ah * 60 + am) - (dh * 60 + dm)
    if dur < 0: dur += 24 * 60
    tier     = TIER_MAP.get(airline, 2)
    dep_slot = time_slot(dh)
    is_peak  = int(month in [5, 6, 10, 12])
    is_wknd  = int(date_obj.weekday() >= 5)

    row = pd.DataFrame([{
        "Airline"            : encoders["Airline"].transform([airline])[0],
        "Source"             : encoders["Source"].transform([source])[0],
        "Destination"        : encoders["Destination"].transform([destination])[0],
        "Total_Stops"        : total_stops,
        "Journey_Day"        : date_obj.day,
        "Journey_Month"      : month,
        "Journey_Weekday"    : date_obj.weekday(),
        "Is_Weekend"         : is_wknd,
        "Is_Month_Start"     : int(date_obj.day <= 5),
        "Is_Month_End"       : int(date_obj.day >= 25),
        "Season"             : get_season(month),
        "Is_Peak_Month"      : is_peak,
        "Quarter"            : (month - 1) // 3 + 1,
        "Dep_Hour"           : dh,
        "Dep_Minute"         : dm,
        "Dep_Time_Slot"      : dep_slot,
        "Is_Early_Morning_Dep": int(dh < 6),
        "Is_Late_Night_Dep"  : int(dh >= 21),
        "Arrival_Hour"       : ah,
        "Arrival_Minute"     : am,
        "Arr_Time_Slot"      : time_slot(ah),
        "Duration_Mins"      : dur,
        "Duration_Hours"     : dur / 60.0,
        "Duration_Sq"        : dur ** 2,
        "Is_Short_Flight"    : int(dur < 120),
        "Is_Long_Flight"     : int(dur > 300),
        "Is_Non_Stop"        : int(total_stops == 0),
        "Is_Multi_Stop"      : int(total_stops >= 2),
        "Airline_Tier"       : tier,
        "Stops_x_Duration"   : total_stops * dur,
        "Tier_x_Stops"       : tier * total_stops,
        "Tier_x_Duration"    : tier * dur,
        "Peak_x_Stops"       : is_peak * total_stops,
        "Weekend_x_Tier"     : is_wknd * tier,
        "DepSlot_x_Stops"    : dep_slot * total_stops,
        "Season_x_Tier"      : get_season(month) * tier,
        "Month_x_Stops"      : month * total_stops,
        "Tier_x_NonStop"     : tier * int(total_stops == 0),
        "Duration_x_NonStop" : dur * int(total_stops == 0),
    }])

    row = row[feat_cols]

    # Predict in log space → expm1 back
    rf_log    = rf.predict(row)[0]
    xgb_log   = xgb_model.predict(row)[0]
    stack_log = stack_model.predict([[rf_log, xgb_log]])[0]

    rf_pred    = np.expm1(rf_log)
    xgb_pred   = np.expm1(xgb_log)
    stack_pred = np.expm1(stack_log)
    ens_pred   = w_rf*rf_pred + w_xgb*xgb_pred + w_stack*stack_pred

    ens_mae  = val_m["Ensemble"]["mae"]
    ens_rmse = val_m["Ensemble"]["rmse"]
    low      = round(ens_pred - ens_mae)
    high     = round(ens_pred + ens_mae)

    flight_note = ""
    if dur < 120:  flight_note = "  [Short flight]"
    elif dur > 300: flight_note = "  [Long flight]"
    if is_peak:    flight_note += "  [Peak season]"
    if is_wknd:    flight_note += "  [Weekend]"

    print("\n" + "="*66)
    print(f"  ✈  {airline}  |  {source}  →  {destination}")
    print(f"     Date  : {journey_date}    {dep_time} → {arrival_time}")
    print(f"     Stops : {total_stops}     Duration: {dur} mins{flight_note}")
    print("="*66)
    print(f"  {'Model':<22} {'Predicted (₹)':>14}  {'MAE (₹)':>9}  {'RMSE (₹)':>10}")
    print(f"  {'-'*60}")
    for name, pred in [("Random Forest", rf_pred), ("XGBoost", xgb_pred),
                        ("Stack (Ridge)", stack_pred), ("Ensemble ★", ens_pred)]:
        m = val_m.get(name.replace(" ★",""), val_m["Ensemble"])
        print(f"  {name:<22} {round(pred):>14,}  {m['mae']:>9,.0f}  {m['rmse']:>10,.0f}")
    print("="*66)
    print(f"  Likely fare range  : ₹ {low:,}  –  ₹ {high:,}  (±MAE)")
    print("="*66)

    return {
        "Random Forest": {"price": round(rf_pred),    "MAE": round(val_m["Random Forest"]["mae"]),  "RMSE": round(val_m["Random Forest"]["rmse"])},
        "XGBoost"      : {"price": round(xgb_pred),   "MAE": round(val_m["XGBoost"]["mae"]),        "RMSE": round(val_m["XGBoost"]["rmse"])},
        "Stack"        : {"price": round(stack_pred),  "MAE": round(val_m["Stack (Ridge)"]["mae"]),  "RMSE": round(val_m["Stack (Ridge)"]["rmse"])},
        "Ensemble"     : {"price": round(ens_pred),    "MAE": round(ens_mae),                        "RMSE": round(ens_rmse)},
        "Range"        : (low, high),
    }


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":

    # STEP 1 — Train (auto-skips if models exist)
    if os.path.exists(RF_PATH) and os.path.exists(XGB_PATH):
        print("[INFO] Saved models found — skipping training.")
        print("       Delete rf_model.pkl & xgb_model.pkl to retrain.\n")
    else:
        train()

    # STEP 2 — Bulk test set
    predict_test_set()

    # STEP 3 — Individual predictions
    print("\n\n" + "="*66)
    print("  STEP 3 — INDIVIDUAL FLIGHT PREDICTIONS")
    print("="*66)

    predict_fare(
        airline      = "IndiGo",
        source       = "Delhi",
        destination  = "Cochin",
        journey_date = "15/05/2026",
        dep_time     = "06:00",
        arrival_time = "09:25",
        total_stops  = 0
    )

    predict_fare(
        airline      = "Air India",
        source       = "Mumbai",
        destination  = "Kolkata",
        journey_date = "20/06/2026",
        dep_time     = "08:30",
        arrival_time = "13:45",
        total_stops  = 1
    )

    predict_fare(
        airline      = "Vistara",
        source       = "Banglore",
        destination  = "Delhi",
        journey_date = "01/12/2026",
        dep_time     = "05:30",
        arrival_time = "09:00",
        total_stops  = 0
    )

    # ── YOUR FLIGHT HERE ─────────────────────────────────────
    # predict_fare(
    #     airline      = "SpiceJet",
    #     source       = "Chennai",
    #     destination  = "Delhi",
    #     journey_date = "10/07/2026",
    #     dep_time     = "07:00",
    #     arrival_time = "10:30",
    #     total_stops  = 1
    # )

    print("""
  VALID INPUT OPTIONS
  ─────────────────────────────────────────────────────────
  Airlines     : IndiGo, Air India, Jet Airways, SpiceJet,
                 Multiple carriers, GoAir, Vistara, Air Asia,
                 Jet Airways Business, Trujet
  Sources      : Banglore, Kolkata, Delhi, Chennai, Mumbai
  Destinations : Cochin, Banglore, New Delhi, Hyderabad,
                 Kolkata, Delhi
  total_stops  : 0 (non-stop), 1, 2, 3
  journey_date : "DD/MM/YYYY"
  dep/arr time : "HH:MM"  (24-hour format)
  ─────────────────────────────────────────────────────────
    """)
