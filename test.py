from deep_snow.application import predict_sd
from datetime import datetime, timedelta

def generate_dates(end_date_str, start_date_str):
    target_date = datetime.strptime(end_date_str, "%Y%m%d")
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    date_list = []

    while target_date >= start_date:
        date_list.append(target_date.strftime("%Y%m%d"))
        target_date -= timedelta(days=12)

    return date_list

def most_recent_occurrence(date_str: str, mmdd: str) -> str:
    ref_date = datetime.strptime(date_str, "%Y%m%d")
    target_date = datetime(ref_date.year, int(mmdd[:2]), int(mmdd[2:]))
    
    if target_date >= ref_date:
        target_date = target_date.replace(year=ref_date.year - 1)
    
    return target_date.strftime("%Y%m%d")

# set up arguments -71.01 47.03 -69.51 48.53
aoi = {'minlon':-73.349190, 'minlat':45.532808, 'maxlon':-73.296318, 'maxlat':45.560458} # San Juans, CO
begin_date = '20251110'
end_date = '20260227'
snow_off_day = '1109'  # mmdd format
model_path = 'weights/quinn_ResDepth_v10_256epochs' # Current latest model

# Generate date list
date_list = generate_dates(end_date, begin_date)

# Predict snow depth for each date
for target_date in date_list:
    snowoff_date = most_recent_occurrence(target_date, snow_off_day)
    out_dir = f'run/{target_date}/'
    print(f"Predicting snow depth for {target_date} (snowoff: {snowoff_date})...")
    ds = predict_sd(aoi=aoi, target_date=target_date, snowoff_date=snowoff_date, model_path=model_path, out_dir=out_dir)
    print(f"Prediction completed for {target_date}")
