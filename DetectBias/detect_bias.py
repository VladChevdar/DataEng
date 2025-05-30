# Vlad Chevdar | DataEng S25 - Detecting Bias Lab Assignment
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from scipy.stats import binomtest, ttest_ind
import io

# Transform the Data
with open("trimet_stopevents_2022-12-07.html", "r") as f:
    html_content = f.read()

tables = pd.read_html(io.StringIO(html_content))
valid_tables = [
    t for t in tables if set(['vehicle_number', 'arrive_time', 'location_id', 'ons', 'offs', 'trip_number']).issubset(t.columns)
]
combined_df = pd.concat(valid_tables, ignore_index=True)
service_date = datetime.strptime("2022-12-07", "%Y-%m-%d")

stops_df = pd.DataFrame({
    'trip_id': combined_df['trip_number'],
    'vehicle_number': pd.to_numeric(combined_df['vehicle_number'], errors='coerce'),
    'location_id': pd.to_numeric(combined_df['location_id'], errors='coerce'),
    'ons': pd.to_numeric(combined_df['ons'], errors='coerce'),
    'offs': pd.to_numeric(combined_df['offs'], errors='coerce'),
    'tstamp': combined_df['arrive_time'].apply(lambda x: service_date + timedelta(seconds=int(x)) if pd.notna(x) and str(x).isdigit() else None)
}).dropna()

stops_df.to_csv("trimet_stops.csv", index=False)

# Validation
print("How many vehicles?", stops_df['vehicle_number'].nunique())
print("How many stop locations?", stops_df['location_id'].nunique())
print("Min timestamp:", stops_df['tstamp'].min())
print("Max timestamp:", stops_df['tstamp'].max())

boarding_events = stops_df[stops_df['ons'] >= 1]
print("Stop events with boarding:", len(boarding_events))
print("Boarding percentage: {:.2f}%".format(len(boarding_events) / len(stops_df) * 100))

loc_df = stops_df[stops_df['location_id'] == 6913]
print("\nLocation 6913 - stops:", len(loc_df))
print("Unique buses:", loc_df['vehicle_number'].nunique())
print("Boarding %: {:.2f}%".format((loc_df['ons'] >= 1).sum() / len(loc_df) * 100 if len(loc_df) > 0 else 0))

veh_df = stops_df[stops_df['vehicle_number'] == 4062]
print("\nVehicle 4062 - stops:", len(veh_df))
print("Total boarded:", veh_df['ons'].sum())
print("Total deboarded:", veh_df['offs'].sum())
print("Boarding %: {:.2f}%".format((veh_df['ons'] >= 1).sum() / len(veh_df) * 100 if len(veh_df) > 0 else 0))

# Vehicles with biased boarding data (“ons”)
total_stops = len(stops_df)
total_boarding = (stops_df['ons'] >= 1).sum()
overall_boarding_rate = total_boarding / total_stops
biased_ons = []

for vehicle_id, group in stops_df.groupby('vehicle_number'):
    n = len(group)
    k = (group['ons'] >= 1).sum()
    if n == 0:
        continue
    p_value = binomtest(k, n, overall_boarding_rate).pvalue
    if p_value < 0.05:
        biased_ons.append({'vehicle_number': vehicle_id, 'p_value': p_value})

if biased_ons:
    df_biased_ons = pd.DataFrame(biased_ons).sort_values("p_value")
    print("\nVehicles with biased boarding (p < 0.05):")
    print(df_biased_ons[['vehicle_number', 'p_value']])
else:
    print("No biased boarding vehicles found.")

# Vehicles with biased GPS data
breadcrumb_df = pd.read_csv("trimet_relpos_2022-12-07.csv")
breadcrumb_df.columns = breadcrumb_df.columns.str.strip().str.lower()

all_relpos = breadcrumb_df['relpos'].dropna().values
biased_gps = []

for vehicle_id, group in breadcrumb_df.groupby('vehicle_number'):
    vehicle_relpos = group['relpos'].dropna().values
    if len(vehicle_relpos) < 2:
        continue
    _, p_value = ttest_ind(vehicle_relpos, all_relpos, equal_var=False)
    if p_value < 0.005:
        biased_gps.append({'vehicle_number': vehicle_id, 'p_value': p_value})

if biased_gps:
    df_biased_gps = pd.DataFrame(biased_gps).sort_values("p_value")
    print("\nVehicles with biased GPS (p < 0.005):")
    print(df_biased_gps.to_string(index=False))
    df_biased_gps.to_csv("biased_gps_vehicles.csv", index=False)
else:
    print("No GPS bias found.")
