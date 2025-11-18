import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import sqlite3
from datetime import datetime, timedelta

class KiranaPipeline:
    def __init__(self):
        # Load the 3 trained models
        # Note: These paths assume models are in the same folder
        try:
            self.model_amount = joblib.load('Models/model_daily_amount.pkl')
            self.model_denom = joblib.load('Models/denomination_split_model.pkl')
            self.model_spikes = joblib.load('Models/model_spikes.pkl')
        except FileNotFoundError:
            # This error will be caught by Streamlit
            raise FileNotFoundError("Model files not found. Make sure .pkl files are in the root directory.")

        self.denoms = [2000, 500, 200, 100, 50, 20, 10, 5, 2, 1]
        self.coins = [1, 2, 5, 10] 

    def get_yesterday_cash_from_db(self):
        """Fetches the total change given yesterday from the DB"""
        try:
            conn = sqlite3.connect('kirana_demo.db')
            cursor = conn.cursor()
            
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            
            query = f"""
            SELECT SUM(change_given) 
            FROM transactions 
            WHERE payment_method = 'cash' 
              AND date(timestamp) = '{yesterday}'
            """
            
            cursor.execute(query)
            result = cursor.fetchone()[0]
            conn.close()
            
            return result if result else 0.0
        except Exception as e:
            print(f"DB Error: {e}. Defaulting to 0. Have you run setup_db.py?")
            return 0.0

    def get_date_features(self, date_str):
        dt = pd.to_datetime(date_str)
        return {
            'day_of_week': dt.dayofweek,
            'month': dt.month,
            'day_of_month': dt.day,
            'is_weekend': 1 if dt.dayofweek >= 5 else 0
        }

    def predict_daily_cash(self, date_features, prev_day_cash):
        """Predicts Total Change Needed for the day."""
        input_data = pd.DataFrame([{
            'day_of_week': date_features['day_of_week'],
            'month': date_features['month'],
            'day_of_month': date_features['day_of_month'],
            'is_weekend': date_features['is_weekend'],
            'lag1': prev_day_cash,
            'lag2': prev_day_cash,
            'lag3': prev_day_cash,
            'lag7': prev_day_cash,
            'roll3': prev_day_cash,
            'roll7': prev_day_cash,
            'roll3_std': 0,
            'roll7_std': 0,
            'roll14': prev_day_cash
        }])
        
        pred = self.model_amount.predict(input_data)[0]
        return max(0, round(float(pred), 2))

    def predict_denominations(self, total_change_needed, date_features):
        """Predicts note inventory by simulating a typical day's transaction mix."""
        scenarios = [
            {'total': 12, 'tender': 20},   
            {'total': 45, 'tender': 100},  
            {'total': 88, 'tender': 100},  
            {'total': 110, 'tender': 200}, 
            {'total': 420, 'tender': 500}, 
            {'total': 1800, 'tender': 2000}
        ]
        
        inputs = []
        for s in scenarios:
            inputs.append({
                'total_amount': s['total'],
                'tendered_amount': s['tender'],
                'day_of_week': date_features['day_of_week'],
                'is_weekend': date_features['is_weekend'],
                'month': date_features['month']
            })
        
        batch_input = pd.DataFrame(inputs)
        batch_preds = self.model_denom.predict(batch_input)
        batch_preds = np.maximum(0, np.round(batch_preds))
        
        basket_note_counts = batch_preds.sum(axis=0)
        basket_total_value = sum([count * val for count, val in zip(basket_note_counts, self.denoms)])
        
        scaling_factor = total_change_needed / basket_total_value if basket_total_value > 0 else 0
            
        final_inventory = {}
        for i, denom in enumerate(self.denoms):
            count = np.ceil(basket_note_counts[i] * scaling_factor).astype(int)
            final_inventory[denom] = count
            
        return final_inventory

    def predict_spike_hours(self, date_features, prev_day_count=50):
        """Predicts hourly traffic."""
        hourly_preds = {}
        hours = list(range(8, 22)) # 8 AM to 9 PM
        
        input_rows = [{'hour': h, 'dayofweek': date_features['day_of_week'], 'lag_24': prev_day_count / 14} for h in hours]
        input_data = pd.DataFrame(input_rows)
        preds = self.model_spikes.predict(input_data)
        
        for i, h in enumerate(hours):
            hourly_preds[h] = max(0, int(preds[i]))
            
        spike_hour = max(hourly_preds, key=hourly_preds.get)
        return spike_hour

    def run_prediction(self, date_str, yesterday_total_cash):
        """Runs the full pipeline and returns a result dictionary."""
        feats = self.get_date_features(date_str)
        
        total_amt = self.predict_daily_cash(feats, yesterday_total_cash)
        denom_split = self.predict_denominations(total_amt, feats)
        spike_h = self.predict_spike_hours(feats)
        
        inventory_val = sum([k * v for k, v in denom_split.items()])
        
        return {
            "Date": date_str,
            "Total_Change": total_amt,
            "Spike_Hour": f"{spike_h}:00 - {spike_h+1}:00",
            "Inventory": denom_split,
            "Inventory_Value": round(inventory_val, 2),
            "Safety_Buffer": round(inventory_val - total_amt, 2)
        }