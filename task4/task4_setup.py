import os
from pathlib import Path
import json
 
import pandas as pd
from sklearn.model_selection import train_test_split
 
 
TRIPS_PATH = str(Path(os.environ.get("VISTA_DATA_DIR", Path(__file__).resolve().parents[1] / "data")) / "trips_vista_2024_2025.csv.gz")
RANDOM_STATE = 90049
TEST_SIZE = 0.25
LONG_DISTANCE_KM = 30
SMALL_TRAIN_HOUSEHOLDS = 30
LARGE_TRAIN_HOUSEHOLDS = 300
 
 
def load_and_clean_trips(path=TRIPS_PATH):
   
    df = pd.read_csv(path)
    df['cumdist_num'] = pd.to_numeric(df['cumdist'], errors='coerce')
    df = df.dropna(subset=['cumdist_num', 'travtime'])
    df = df[(df['cumdist_num'] > 0) & (df['travtime'] > 0)]
    df = df[df['homeregion_ASGS'] == 'Greater Melbourne']
    return df
 
 
def _household_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    
    unique_hh = df['hhid'].unique()
    train_hh, test_hh = train_test_split(unique_hh, test_size=test_size, random_state=random_state)
    return df[df['hhid'].isin(train_hh)], df[df['hhid'].isin(test_hh)]
 
 
def _household_subset(df, n_households, random_state=RANDOM_STATE):
   
    unique_hh = df['hhid'].unique()
    chosen_hh, _ = train_test_split(unique_hh, train_size=n_households, random_state=random_state)
    return df[df['hhid'].isin(chosen_hh)]
 
 
def build_datasets():
    
    df = load_and_clean_trips()
 
    normal_range = df[df['cumdist_num'] <= LONG_DISTANCE_KM]
    long_distance = df[df['cumdist_num'] > LONG_DISTANCE_KM]
 
    train_pool, dataset3 = _household_split(normal_range)
    dataset1 = _household_subset(train_pool, SMALL_TRAIN_HOUSEHOLDS)
    dataset2 = _household_subset(train_pool, LARGE_TRAIN_HOUSEHOLDS)
    dataset4 = long_distance
 
    return dataset1, dataset2, dataset3, dataset4
 
 
def run_setup():
    dataset1, dataset2, dataset3, dataset4 = build_datasets()
 
    datasets = {'Dataset1': dataset1, 'Dataset2': dataset2, 'Dataset3': dataset3, 'Dataset4': dataset4}
    summary = {}
    for name, data in datasets.items():
        summary[name] = {
            'n_trips': int(len(data)),
            'n_households': int(data['hhid'].nunique()),
            'feature_center_cumdist_km': f"{data['cumdist_num'].mean():.10f}",
            'feature_center_starthour': f"{data['starthour'].mean():.10f}",
            'mean_travtime_min': f"{data['travtime'].mean():.10f}",
        }
 
    with open('task4_setup_datasets_check.json', 'w') as f:
        json.dump(summary, f, indent=2)
 
    return {
        'dataset1': dataset1,
        'dataset2': dataset2,
        'dataset3': dataset3,
        'dataset4': dataset4,
    }
 







































