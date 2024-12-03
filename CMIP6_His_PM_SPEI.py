## Code for calculating daily SPEI data using the Penman-Monteith formula:

import pandas as pd   
import numpy as np
from standard_precip import spi 
from datetime import datetime, timedelta
import tifffile as tf
import os 
from standard_precip.base_sp import BaseStandardIndex
from standard_precip.utils import best_fit_distribution
from joblib import Parallel, delayed

def create_date_array(start_year, end_year):
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    date_array = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.year % 4 == 0 and current_date.month == 2 and current_date.day == 29:
            current_date += timedelta(days=1)
            continue
        date_array.append(current_date)
        current_date += timedelta(days=1)
    return np.array(date_array)

def process_grid(i, j, pet_row, pr_row, time_D, start_year, end_year, scale, reference):
    if np.isnan(reference[i, j]):
        return np.full_like(time_D, np.nan)
    else:
        # 从 pet 和 pr 中提取对应的行数据
        pet_temp1 = pet_row[:]
        pr_temp1 = pr_row[:]
        WB_temp = pr_temp1 - pet_temp1
        WB = pd.DataFrame({ 
            'date': time_D,
            'Water_Balance': WB_temp
        })
        Water_Balance = 'Water_Balance'
        if isinstance(Water_Balance, str):
           Water_Balance = [Water_Balance]
        WB1,WB_sum = BaseStandardIndex.rolling_window_sum(WB, Water_Balance,scale)  
        dist = ['gev', 'nor', 'glo'] 
        WB_sum_str = ','.join(WB_sum)
        WB_data = WB.loc[:,WB_sum_str]
        valid_indices = ~np.isnan(WB_data)
        data_cleaned = WB_data[valid_indices]
        bins = np.linspace(np.min(data_cleaned), np.max(data_cleaned), 25)
        all_fit = best_fit_distribution(data_cleaned, dist_list= dist, fit_type= "mle", bins= 25)
        best_fit=all_fit[0][0]
        spi_daily = spi.SPI()  
        spei_daily = spi_daily.calculate(WB, 'date', 'Water_Balance', startyr=start_year, endyr=end_year, freq="D", scale=scale, 
                                fit_type="mle", dist_type= best_fit)
        spei_name = WB_sum_str + '_calculated_index'
        spei_temp = spei_daily.loc[:,spei_name]
        return spei_temp.values.reshape(-1)

start_year = 1950
end_year = 2014 
scale = 180

reference = tf.imread('/climca/people/xliu/reference/2.5度/reference_no_antarctica.tif')
reference = reference.astype(float)
reference[62:, :] = np.nan

Model_name = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-ESM-1-1-LR', 'CMCC-ESM2', 'EC-Earth3', 
              'EC-Earth3-CC', 'EC-Earth3-Veg', 'EC-Earth3-Veg-LR', 'INM-CM4-8',  'INM-CM5-0', 
              'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM-1-2-HAM','MPI-ESM1-2-HR', 'MPI-ESM1-2-LR',
              'MRI-ESM2-0']

# model = 'CMCC-ESM2'

main_path = "/climca/people/xliu/Daily_SPEI_CMIP6/Data/CMIP6/NC_Download/Historical_data/"

for model in Model_name:

    # Load in the PET data 
    pet_path = main_path + model + '/preproc/download_only/PET_PM/' 
    pet_file = [f for f in os.listdir(pet_path) if f.endswith('.npy')][0]
    pet = np.load(os.path.join(pet_path, pet_file))

    # Load in the pr data 
    pr_path = main_path + model + '/preproc/download_only/pr/' 
    pr_file = [f for f in os.listdir(pr_path) if f.endswith('.npy')][0]
    pr = np.load(os.path.join(pr_path, pr_file))

    time_D = create_date_array(start_year, end_year)

    lat_size, long_size, time_size = pet.shape  

    # Parallel computation with 16 cores
    results = Parallel(n_jobs=48)(
        delayed(process_grid)(
            i, j, pet[i, j, :], pr[i, j, :], time_D, start_year, end_year, scale, reference
        ) for i in range(lat_size) for j in range(long_size)
    )

    # # 检查并行计算结果是否正确
    # for result in results:
    #     if result is None:
    #         print("Warning: Some grid points returned None.")
    #         break

    # Reshape results to match the original spei array
    spei = np.array(results).reshape(lat_size, long_size, -1)
    # spei = np.array(results).reshape(21, 21, -1)
    # spei1= spei[:,:,7]
    # reference11= reference[40:61,50:71]
    out_path = "/climca/people/xliu/Daily_SPEI_CMIP6/Data/SPEI/CMIP6/PM/"
    output_path = out_path + str(scale) + 'days/SPEI_' + str(scale) + 'days_' + model + '_' + str(start_year) + '_' + str(end_year) + '.npy'
    np.save(output_path, spei)  

    print(f"{model} is completed!")
