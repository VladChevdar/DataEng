# Vlad Chevdar | DataEng S25 - Data Transformation Lab Assignment
import pandas as pd

# 2. Filter
trip_df = pd.read_csv(
    'bc_trip259172515_230215.csv',
    usecols=['EVENT_NO_TRIP', 
             'OPD_DATE', 
             'VEHICLE_ID', 
             'METERS', 
             'ACT_TIME', 
             'GPS_LONGITUDE', 
             'GPS_LATITUDE'
            ]
)

print(f"Number of records: {len(trip_df)}")

# 3. Decode
trip_df['TIMESTAMP'] = (
    pd.to_datetime(trip_df['OPD_DATE'].str[:9], format='%d%b%Y') +
    pd.to_timedelta(trip_df['ACT_TIME'], unit='s')
)

trip_df = trip_df.drop(columns=['OPD_DATE', 'ACT_TIME'])

# 3. Decode (cont)

# 4. Enhance
trip_df['dMETERS'] = trip_df['METERS'].diff()
trip_df['dTIMESTAMP'] = trip_df['TIMESTAMP'].diff().dt.total_seconds()

trip_df["SPEED"] = trip_df["dMETERS"] / trip_df["dTIMESTAMP"]
trip_df["SPEED"] = trip_df["SPEED"].where(trip_df["dTIMESTAMP"] > 0, 0)

trip_df = trip_df.drop(columns=['dMETERS', 'dTIMESTAMP'])

min_speed = trip_df['SPEED'].min()
max_speed = trip_df['SPEED'].max()
avg_speed = trip_df['SPEED'].mean()

print(f"Minimum speed: {min_speed} meters/second")
print(f"Maximum speed: {max_speed} meters/second")
print(f"Average speed: {avg_speed} meters/second")
