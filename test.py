from deep_snow.application import download_data, apply_model_ensemble, apply_model
from datetime import datetime, timedelta
import time

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

# Set up ensemble model paths (from GitHub workflows)
model_1_path = 'weights/ResDepth_lr0.000457131171011064_weightdecay0.00010523970398286011_epochs62_mintestloss0.00091'
model_2_path = 'weights/ResDepth_lr0.00025036613931876504_weightdecay0.00020109428801183744_epochs63_mintestloss0.00091'
model_3_path = 'weights/ResDepth_lr0.0001572907262097884_weightdecay0.00013101368652881237_epochs98_mintestloss0.00090'
model_4_path = 'weights/ResDepth_lr0.00011563677025564128_weightdecay0.0003567649258551211_epochs63_mintestloss0.00092'
model_5_path = 'weights/ResDepth_lr0.00026633575524604445_weightdecay8.770493085204089e-05_epochs85_mintestloss0.00090'
model_paths_list = [model_1_path, model_2_path, model_3_path, model_4_path, model_5_path]

# Set up arguments
aoi = {'minlon':-73.349190, 'minlat':45.532808, 'maxlon':-73.296318, 'maxlat':45.560458} # San Juans, CO
begin_date = '20251110'
end_date = '20260227'
snow_off_day = '1109'  # mmdd format
cloud_cover = 25
max_retries = 100
retry_delay = 5  # seconds

# Generate date list
date_list = generate_dates(end_date, begin_date)

# Predict snow depth for each date
for target_date in date_list:
    snowoff_date = most_recent_occurrence(target_date, snow_off_day)
    out_dir = f'run/{target_date}/'
    print(f"Predicting snow depth for {target_date} (snowoff: {snowoff_date})...")
    
    buffer_period = 50
    for attempt in range(max_retries):
        try:
            crs = download_data(aoi=aoi, target_date=target_date, buffer_period=buffer_period, snowoff_date=snowoff_date, out_dir=out_dir, cloud_cover=cloud_cover)
            
            # Run Quinn's ResDepth v11 with specific input channels
            quinn_model_path = 'weights/quinn_ResDepth_v11_254epochs'
            if __name__ == "__main__" and __file__:
                quinn_input_channels = ['snowon_vv','delta_cr','green','swir2','ndsi','ndwi','snodas_sd','elevation','latitude','longitude']
                quinn_ds = apply_model(out_dir=out_dir, out_name=f'{target_date}_quinn', crs=crs, write_tif=True, model_path=quinn_model_path, delete_inputs=False, out_crs='utm', gpu=False, input_channels=quinn_input_channels)
            
            # Run ensemble model
            ds = apply_model_ensemble(out_dir=out_dir, out_name=f'{target_date}_deep-snow', crs=crs, write_tif=True, model_paths_list=model_paths_list, delete_inputs=False, out_crs='utm', gpu=False)
            print(f"Prediction completed for {target_date}")
            break  # Exit the loop if successful
        except ValueError as e:
            if str(e) == "Can't load empty sequence":
                buffer_period += 2
                print(f"ValueError encountered: {e}. Increasing buffer_period to {buffer_period} and retrying...")
            elif str(e) == "'list' object cannot be interpreted as an integer":
                buffer_period += 2
                print(f"TypeError encountered: {e}. Increasing buffer_period to {buffer_period} and retrying...")
            else:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise
