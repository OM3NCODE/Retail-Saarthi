import sqlite3
import numpy as np
from datetime import datetime, timedelta

def setup_database():
    # Connect to SQLite (It acts just like Postgres for this demo)
    conn = sqlite3.connect('kirana_demo.db')
    cursor = conn.cursor()

    # 1. Create Table (PostgreSQL Compatible Schema)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        tx_id TEXT PRIMARY KEY,
        timestamp DATETIME,
        total_amount REAL,
        payment_method TEXT,
        tendered_amount REAL,
        change_given REAL
    );
    """)

    # 2. Generate 30 Days of Dummy Data
    print("Generating dummy data for the past 30 days...")
    data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    current_date = start_date
    tx_counter = 0
    
    while current_date <= end_date:
        # 20-50 transactions per day
        for _ in range(np.random.randint(20, 50)):
            tx_time = current_date.replace(hour=np.random.randint(9, 21), minute=np.random.randint(0, 60))
            
            # Logic: 60% Cash, 40% UPI
            method = 'cash' if np.random.random() > 0.4 else 'upi'
            total = np.round(np.random.uniform(50, 500), 2)
            
            if method == 'cash':
                # Simulate tendering a higher note (e.g. Total 140, pay 200)
                tendered = np.ceil(total / 50) * 50 
                change = max(0, tendered - total)
            else:
                tendered = total
                change = 0
                
            data.append((f"tx_{tx_counter}", tx_time.strftime("%Y-%m-%d %H:%M:%S"), total, method, tendered, change))
            tx_counter += 1
            
        current_date += timedelta(days=1)

    # 3. Insert Data
    cursor.executemany("INSERT OR REPLACE INTO transactions VALUES (?, ?, ?, ?, ?, ?)", data)
    conn.commit()
    
    # 4. Verify
    print(f"Success! Database 'kirana_demo.db' created with {len(data)} transactions.")
    conn.close()

if __name__ == "__main__":
    setup_database()