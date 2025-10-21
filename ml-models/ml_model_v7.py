"""Flask API wrapper for ticket prediction model.

This file dynamically loads `ml_model_v6.2.py` (note the filename contains a dot)
and exposes endpoints to run predictions.

Endpoints
- GET /health -> simple health check
- POST /predict -> JSON body options:
    - {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}
    - {"days": N}  -> predict next N days after the last date in the dataset

Notes
- This will execute and load the model and preprocessing defined in
  `ml_model_v6.2.py` at startup. That script trains an XGBoost model on import,
  so startup may take some time.

Requirements (install in your environment):
    pip install flask pandas numpy scikit-learn xgboost

Run (development):
    python ml_model_v7.py

"""

import os
import traceback
import importlib.util
from datetime import datetime, timedelta
from flask import Flask, request, jsonify

app = Flask(__name__)


def load_model_module():
    """Dynamically load ml_model_v6.2.py as a module and return it.

    We use a dynamic loader because the filename contains a dot which prevents
    normal import syntax.
    """
    base_dir = os.path.dirname(__file__)
    module_path = os.path.join(base_dir, 'ml_model_v6.2.py')
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Model file not found at {module_path}")

    spec = importlib.util.spec_from_file_location('ml_model_v6_2', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load model module at startup (this will run the training present in that file)
try:
    model_mod = load_model_module()
except Exception as e:
    # Keep the traceback for logs; the endpoints will report failure until the
    # module is successfully loaded.
    model_mod = None
    load_error = traceback.format_exc()
    print("Failed to load model module:")
    print(load_error)


@app.route('/', methods=['GET'])
def index():
    return jsonify({'status': 'ok', 'message': 'Ticket prediction API'})


@app.route('/health', methods=['GET'])
def health():
    if model_mod is None:
        return jsonify({'status': 'error', 'message': 'model not loaded', 'error': load_error}), 500
    return jsonify({'status': 'healthy'})


def df_to_json_list(df):
    """Convert prediction DataFrame to list of dicts with ISO date strings."""
    res = []
    for _, row in df.iterrows():
        date_val = row.get('date') or row.get('start_date')
        # Handle pandas Timestamp
        try:
            date_str = pd_to_iso(date_val)
        except Exception:
            date_str = str(date_val)
        res.append({'date': date_str, 'predicted_ticket_volume': float(row['predicted_ticket_volume'])})
    return res


def pd_to_iso(val):
    # Accept pandas Timestamp, numpy datetime64 or datetime
    try:
        # Import here to avoid forcing pandas unless needed
        import pandas as pd
        if isinstance(val, pd.Timestamp):
            return val.strftime('%Y-%m-%d')
    except Exception:
        pass
    if hasattr(val, 'strftime'):
        return val.strftime('%Y-%m-%d')
    return str(val)


@app.route('/predict', methods=['POST'])
def predict():
    global model_mod
    if model_mod is None:
        return jsonify({'status': 'error', 'message': 'model not loaded', 'error': load_error}), 500

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'status': 'error', 'message': 'invalid or missing JSON body'}), 400

    try:
        # Option 1: explicit date range
        if 'start_date' in data and 'end_date' in data:
            start = datetime.fromisoformat(data['start_date']).date()
            end = datetime.fromisoformat(data['end_date']).date()
            if start > end:
                return jsonify({'status': 'error', 'message': 'start_date must be <= end_date'}), 400

            df_pred = model_mod.predict_range(
                model_mod.bst,
                model_mod.scaler,
                model_mod.daily_df,
                pd_to_datetime(start),
                pd_to_datetime(end),
                model_mod.holiday_flags,
                model_mod.major_event_flags,
            )
            # format results
            out = []
            for _, r in df_pred.iterrows():
                out.append({'date': r['date'].strftime('%Y-%m-%d'), 'predicted_ticket_volume': float(r['predicted_ticket_volume'])})
            return jsonify({'status': 'ok', 'predictions': out})

        # Option 2: days window from last date
        if 'days' in data:
            days = int(data['days'])
            if days <= 0 or days > 365:
                return jsonify({'status': 'error', 'message': 'days must be between 1 and 365'}), 400
            last_date = model_mod.daily_df['start_date'].max()
            start = last_date + timedelta(days=1)
            end = last_date + timedelta(days=days)
            df_pred = model_mod.predict_range(
                model_mod.bst,
                model_mod.scaler,
                model_mod.daily_df,
                start,
                end,
                model_mod.holiday_flags,
                model_mod.major_event_flags,
            )
            out = []
            for _, r in df_pred.iterrows():
                out.append({'date': r['date'].strftime('%Y-%m-%d'), 'predicted_ticket_volume': float(r['predicted_ticket_volume'])})
            return jsonify({'status': 'ok', 'predictions': out})

        return jsonify({'status': 'error', 'message': 'request must include start_date+end_date or days'}), 400

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({'status': 'error', 'message': str(e), 'traceback': tb}), 500


def pd_to_datetime(d):
    # helper: accept date or datetime or string and return pd.Timestamp
    import pandas as pd
    if isinstance(d, pd.Timestamp):
        return d
    try:
        return pd.to_datetime(d)
    except Exception:
        return pd.to_datetime(str(d))


if __name__ == '__main__':
    # Development server
    app.run(host='0.0.0.0', port=5000, debug=True)
