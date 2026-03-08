from deep_snow.application import predict_sd

# set up arguments -71.01 47.03 -69.51 48.53
aoi = {'minlon':-73.349190, 'minlat':45.532808, 'maxlon':-73.296318, 'maxlat':45.560458} # San Juans, CO
target_date = '20260218'
snowoff_date = '20251009'
model_path = 'weights/quinn_ResDepth_v10_256epochs' # Current latest model
out_dir = 'run/'

# predict snow depth
ds = predict_sd(aoi=aoi, target_date=target_date, snowoff_date=snowoff_date, model_path=model_path, out_dir=out_dir)
