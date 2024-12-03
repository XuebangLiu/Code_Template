# Code for calculating the annual mean from daily PET, comparing the differences in PET calculated using different methods and data sources, and plotting time and spatial figures.

import numpy as np
import os
import proplot as plot
from scipy.ndimage import zoom
import tifffile as tf

def process_annual_mean(pet_array):
    """
    对输入的三维数组进行处理,每隔365层进行加和,然后再取平均。

    参数：
    pet_ERA5_array: numpy数组,包含三维数据，形状为 (180, 360, 365*year)。

    返回值：
    mean_array: numpy数组,形状为 (180, 360, 365*year),表示每隔365层加和后再取平均的结果。
    """
    # 按照层进行分组，每组包含365层
    num_layers = pet_array.shape[2]
    grouped_layers = [pet_array[:, :, i:i+365] for i in range(0, num_layers, 365)]

    # 对每组进行求和，得到新的34层数组
    summed_layers = [np.sum(group, axis=2) for group in grouped_layers]

    # 对新的多年多层数组进行平均操作
    mean_array = np.mean(summed_layers, axis=0)

    return mean_array

def process_annual(pet_array):
    """
    对输入的三维数组进行处理,每隔365层进行加和,然后再取平均。

    参数：
    pet_ERA5_array: numpy数组,包含三维数据，形状为 (180, 360, 365*year)。

    返回值：
    mean_array: numpy数组,形状为 (180, 360, 365*year),表示每隔365层加和后再取平均的结果。
    """
    # 按照层进行分组，每组包含365层
    num_layers = pet_array.shape[2]
    grouped_layers = [pet_array[:, :, i:i+365] for i in range(0, num_layers, 365)]

    # 对每组进行求和，得到新的34层数组
    summed_layers = [np.sum(group, axis=2) for group in grouped_layers]

    return summed_layers

def global_land_weighted_mean(temp):
    """
    计算全球陆地区域的面积加权平均气温
    
    参数:
    - temp: 2D numpy 数组，形状为 (纬度, 经度),表示气温数据,海洋区域为NaN
    - lat: 1D numpy 数组，表示纬度值（度数）
    - lon: 1D numpy 数组，表示经度值（度数）
    
    返回:
    - 全球陆地区域面积加权平均气温 (float)
    """
    lat = plot.arange(88.75, -88.75, -2.5)
     
    # 将纬度从度数转换为弧度
    lat_rad = np.deg2rad(lat)
    
    # 计算纬度权重
    weights = np.cos(lat_rad)
    
    # 对每个纬度的气温值应用权重，并忽略NaN值
    weighted_temp = temp * weights[:, np.newaxis]
    
    # 使用np.nansum来忽略NaN值进行加权和求和
    total_weighted_temp = np.nansum(weighted_temp)
    total_weights = np.nansum(weights[:, np.newaxis] * (~np.isnan(temp)))

    # 计算全球陆地区域加权平均气温
    global_mean_temp = total_weighted_temp / total_weights
    
    return global_mean_temp

def calculate_yearly_global_mean(tem):
    """
    计算每年的全球面积加权平均气温
    
    参数:
    - tem: 3D numpy 数组，形状为 (年数, 纬度, 经度)，表示多个年份的气温数据
    - lat: 1D numpy 数组，表示纬度值（度数）
    - lon: 1D numpy 数组，表示经度值（度数）
    
    返回:
    - 一维 numpy 数组，包含每年的全球面积加权平均气温
    """
    if isinstance(tem, list):
        num_years = len(tem)  # 获取年份数
        yearly_means = np.zeros(num_years)  # 初始化存储每年加权平均气温的数组

        for i in range(num_years):
            yearly_means[i] = global_land_weighted_mean(tem[i])
    else: 
        num_years = tem.shape[0]  # 获取年份数
        yearly_means = np.zeros(num_years)  # 初始化存储每年加权平均气温的数组

        for i in range(num_years):
            yearly_means[i] = global_land_weighted_mean(tem[i,:,:])
 
    return yearly_means

def calculate_position(ERA5_trend, trend_M):
    # 初始化dis_trend的shape，最后一个维度是与CMIP6模型数量一致
    dis_trend = np.full_like(ERA5_trend, np.nan)
    
    trend_M = np.transpose(trend_M, (1, 2, 0))

    # 获取数据的维度信息
    num_lat, num_lon, num_models = trend_M.shape

    for i in range(num_lat):
        for j in range(num_lon):
            # 获取对应位置的CMIP6值
            cmip6_values = trend_M[i, j, :]

            # 如果该位置的ERA5趋势值为空，或CMIP6有NaN值，跳过
            if np.isnan(reference[i, j]) or np.isnan(cmip6_values).any():
                continue

            # 对CMIP6值进行排序
            sorted_cmip6_values = np.sort(cmip6_values)

            # 获取ERA5的趋势值
            era5_value = ERA5_trend[i, j]

            # 计算ERA5值在排序后的CMIP6数组中的百分比位置
            position = np.searchsorted(sorted_cmip6_values, era5_value, side='right')

            # 计算位置
            dis_trend[i, j] = position

    return dis_trend

reference = tf.imread('/climca/people/xliu/reference/2.5度/reference_no_antarctica.tif')
reference = reference.astype(float)
reference[62:, :] = np.nan

# 加载ERA5 
ERA5_path = '/climca/people/xliu/Daily_SPEI_CMIP6/Data/ERA5/nc_download/'
ERA5_PM = np.load(ERA5_path + '/PET_PM/ERA5_PET_PM_1950-2014.npy')
ERA5_TH = np.load(ERA5_path + '/PET_TH/ERA5_PET_TH_1950-2014.npy')
ERA5_PM_Year_mean = process_annual_mean(ERA5_PM)
ERA5_TH_Year_mean = process_annual_mean(ERA5_TH)
ERA5_PM_Year = process_annual(ERA5_PM)
ERA5_TH_Year = process_annual(ERA5_TH)

# 加载CMIP6
Model_name = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-ESM-1-1-LR', 'CMCC-ESM2', 'EC-Earth3', 
              'EC-Earth3-CC', 'EC-Earth3-Veg', 'EC-Earth3-Veg-LR', 'INM-CM4-8',  'INM-CM5-0', 
              'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM-1-2-HAM','MPI-ESM1-2-HR', 'MPI-ESM1-2-LR',
              'MRI-ESM2-0']

CMIP6_path = '/climca/people/xliu/Daily_SPEI_CMIP6/Data/CMIP6/NC_Download/Historical_data/'
CMIP6_merge_path = '/climca/people/xliu/Daily_SPEI_CMIP6/Scripts/PET_Compare/Year_mean/'

# # 读取计算PM_PET
CMIP6_PM = np.load('/climca/people/xliu/Daily_SPEI_CMIP6/Scripts/PET_Compare/Year_mean/PET/CMIP6_Allmodels_historical_r1i1p1f1_PET_PM_1950-2014.npy') 
CMIP6_PM_Year_mean = np.load('/climca/people/xliu/Daily_SPEI_CMIP6/Scripts/PET_Compare/Year_mean/PET/Year_mean_CMIP6_Allmodels_historical_r1i1p1f1_PET_PM_1950-2014.npy')
CMIP6_PM_Year = np.load('/climca/people/xliu/Daily_SPEI_CMIP6/Scripts/PET_Compare/Year_mean/PET/Year_CMIP6_Allmodels_historical_r1i1p1f1_PET_PM_1950-2014.npy')

# # 读取计算TH_PET
CMIP6_TH = np.load('/climca/people/xliu/Daily_SPEI_CMIP6/Scripts/PET_Compare/Year_mean/PET/CMIP6_Allmodels_historical_r1i1p1f1_PET_TH_1950-2014.npy')
CMIP6_TH_Year_mean =  np.load('/climca/people/xliu/Daily_SPEI_CMIP6/Scripts/PET_Compare/Year_mean/PET/Year_mean_CMIP6_Allmodels_historical_r1i1p1f1_PET_TH_1950-2014.npy')
CMIP6_TH_Year = np.load('/climca/people/xliu/Daily_SPEI_CMIP6/Scripts/PET_Compare/Year_mean/PET/Year_CMIP6_Allmodels_historical_r1i1p1f1_PET_TH_1950-2014.npy')

lon = plot.arange(-178.75, 178.75, 2.5)
lat = plot.arange(88.75, -88.754, -2.5)

# 取多年的平均值
MM_CMIP6_PM_Year_mean = np.mean(CMIP6_PM_Year_mean, axis=0)
MM_CMIP6_TH_Year_mean = np.mean(CMIP6_TH_Year_mean, axis=0)
# 取每年的平均值
MM_CMIP6_PM_Year = np.mean(CMIP6_PM_Year, axis=0)
MM_CMIP6_TH_Year = np.mean(CMIP6_TH_Year, axis=0)


#  ------------------# PET 每年的变化 折线图-------------------------
x = np.linspace(1950, 2014, 65)
# CMIP6每个模型的值
y1 = np.full((65, 16), np.nan)
y2 = np.full((65, 16), np.nan)
for i in range(16):
    y1[:,i] = calculate_yearly_global_mean(process_annual(CMIP6_PM[i,:,:,:]))
    y2[:,i] = calculate_yearly_global_mean(process_annual(CMIP6_TH[i,:,:,:]))

# ERA5与CMIP6_Ensemble值
y = np.full((65, 4), np.nan)
y[:,0]= calculate_yearly_global_mean(process_annual(ERA5_PM))
y[:,1]= calculate_yearly_global_mean(MM_CMIP6_PM_Year)
y[:,2]= calculate_yearly_global_mean(process_annual(ERA5_TH))
y[:,3]= calculate_yearly_global_mean(MM_CMIP6_TH_Year)

# 绘制图表
gs = plot.GridSpec(nrows=1, ncols=1)
fig = plot.figure(figwidth='10cm', refaspect=(1, 1), span=False)
ax = fig.subplot(gs[0])

for spine in ax.spines.values():
    spine.set_linewidth(0.5)

# 多模型
ax.plot(x,y1,color='red orange', alpha=0.1, lw=1) # , legend_kw={ 'labels': 'CMIP6_models'}
ax.plot(x,y1[:,0],color='red orange', alpha=0.1, lw=1, legend_kw={ 'labels': 'CMIP6_models_PM'}) # , legend_kw={ 'labels': 'CMIP6_models'}
ax.plot(x,y2,color='mid blue', alpha=0.1, lw=1) # , legend_kw={ 'labels': 'CMIP6_models'} 
ax.plot(x,y2[:,0],color='mid blue', alpha=0.1, lw=1, legend_kw={ 'labels': 'CMIP6_models_Har'})

# ERA5与CMIP6_Ensemble
labels = ['ERA5_PM', 'Ensemble_CMIP6_PM', 'ERA5_Har', 'Ensemble_CMIP6_Har']
ax.plot(x, y[:, 0], lw=1, color='deep red', legend_kw={'labels': labels[0]})
ax.plot(x, y[:, 1], lw=1, color='red orange', legend_kw={'labels': labels[1]})
ax.plot(x, y[:, 2], lw=1, color='ocean', legend_kw={'labels': labels[2]})
ax.plot(x, y[:, 3], lw=1, color='mid blue', legend_kw={'labels': labels[3]}) 

# for i in range(0,2):
#     ax.plot(x,y[:, i], lw=1.5, cycle='Reds', cycle_kw={'N': 2, 'left': 0.5},  legend_kw={ 'labels': labels[i]})
# for i in range(2,4):
#     ax.plot(x,y[:, i], lw=1.5, cycle='Blues', cycle_kw={'N': 2, 'left': 0.5},  legend_kw={ 'labels': labels[i]})
# cycle = plot.Cycle('seaborn')
# ax.plot(x,y,cycle,
#         legend_kw={ 'labels': ['PET_ERA5_PM', 'PET_ERA5_Har', 'Ensemble_CMIP6_PM', 'Ensemble_CMIP6_Har']})
ax.format(abc=False, xlabel='', ylabel='PET (mm)', 
          xlabel_kw={'fontsize': 10}, ylabel_kw={'fontsize': 10}, 
          ticklabelsize=10)

ax.legend(prop = {'size':8}, ncol = 2, loc='ll',frame= False)
# ax.legend(prop = {'size':10}, ncol = 2, bbox_to_anchor=(0.36,0.10),frame= False)

save_path = "/climca/people/xliu/Daily_SPEI_CMIP6/Scripts/PET_Compare/Year_mean/PET/"
plot.pyplot.savefig(os.path.join(save_path, "Temporal_pet_compare_PM_Har_ERA5_CMIP6_Ensemble_1950_2014.jpg"), dpi=500)


# -----------------PET不同方法年际平均值,ERA5,CMIP6_ensemble-------------------------
# 绘制实测和年际均值PET 
plot.rc.reso = 'lo'#海岸线可以不同分辨率 'hi' 'med' 'lo' 'x-hi' 'xx-hi'
proj = plot.Proj('robin')
array = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,0,0]
]
fig, axs = plot.subplots(array, figwidth=10, proj=proj)
axs.format(
    abc=True,abcloc='l',abcsize=8, abcstyle='a)', gridlabelsize=18,
    labels=False, lonlines=120, latlines=30,
    coast=True,gridminor=False,coastlinewidth=1,
    suptitle='',suptitlesize=14)  # Annual PET for comparative data
cmap=plot.Colormap('RdBu', reverse=True)
cmap1=plot.Colormap('Spectral', reverse=True) #gamma提高颜色深度 , gamma=0.3 NegPos
cmap2=plot.Colormap('NegPos', reverse=True) #gamma提高颜色深度 , gamma=0.3 NegPos
colorbar_kw1={'label': 'PET (mm)','length': 0.7, 'width': 0.09, 'labelsize': 8, 'ticklabelsize': 7, 'ticks': np.arange(100,2501,800)}
colorbar_kw2={'label': 'PET (mm)','length': 0.7, 'width': 0.09, 'labelsize': 8, 'ticklabelsize': 7, 'ticks': np.arange(-30,30.5,15)}
colorbar_kw3={'label': 'Distribution','length': 0.7, 'width': 0.09, 'labelsize': 8, 'ticklabelsize': 7, 'ticks': np.arange(1, 15, 2)}

axs[0].pcolor(lon, lat, ERA5_PM_Year_mean, cmap=cmap1, colorbar='b', colorbar_kw=colorbar_kw1, 
              extend='both',levels = plot.arange(100, 2500))
axs[1].pcolor(lon, lat, MM_CMIP6_PM_Year_mean, cmap=cmap1, colorbar='b', colorbar_kw=colorbar_kw1, 
              extend='both',levels = plot.arange(100, 2500))
axs[2].pcolor(lon, lat, (ERA5_PM_Year_mean - MM_CMIP6_PM_Year_mean)/ERA5_PM_Year_mean*100, cmap=cmap, colorbar='b', 
              colorbar_kw=colorbar_kw2, extend='both',levels = plot.arange(-30, 30))
text1 = global_land_weighted_mean((ERA5_PM_Year_mean - MM_CMIP6_PM_Year_mean)/ERA5_PM_Year_mean*100) 
axs[2].text(-25, -58, str(round(text1, 2))+'%', fontsize=10, color='black',transform='map')
axs[3].pcolor(lon, lat, calculate_position(ERA5_PM_Year_mean, CMIP6_PM_Year_mean), cmap=cmap, colorbar='b', 
              colorbar_kw=colorbar_kw3, extend='both',levels = plot.arange(1, 15, 1))

axs[4].pcolor(lon, lat, ERA5_TH_Year_mean, cmap=cmap1, colorbar='b', colorbar_kw=colorbar_kw1, 
              extend='both',levels = plot.arange(100, 2500))
axs[5].pcolor(lon, lat, MM_CMIP6_TH_Year_mean, cmap=cmap1, colorbar='b', colorbar_kw=colorbar_kw1, 
              extend='both',levels = plot.arange(100, 2500))
axs[6].pcolor(lon, lat, (ERA5_TH_Year_mean - MM_CMIP6_TH_Year_mean)/ERA5_TH_Year_mean*100, cmap=cmap, colorbar='b', 
              colorbar_kw=colorbar_kw2, extend='both',levels = plot.arange(-30, 30))
text2 = global_land_weighted_mean((ERA5_TH_Year_mean - MM_CMIP6_TH_Year_mean)/ERA5_TH_Year_mean*100)
axs[6].text(-15, -58, str(round(text2, 2))+'%', fontsize=10, color='black',transform='map')
axs[7].pcolor(lon, lat, calculate_position(ERA5_TH_Year_mean, CMIP6_TH_Year_mean), cmap=cmap, colorbar='b', 
              colorbar_kw=colorbar_kw3, extend='both',levels = plot.arange(1, 15, 1))

axs[8].pcolor(lon, lat, (ERA5_PM_Year_mean - ERA5_TH_Year_mean)/ERA5_PM_Year_mean*100, cmap=cmap, colorbar='b', 
              colorbar_kw=colorbar_kw2, extend='both',levels = plot.arange(-30, 30))
text3 = global_land_weighted_mean((ERA5_PM_Year_mean - ERA5_TH_Year_mean)/ERA5_PM_Year_mean*100) 
axs[8].text(-15, -58, str(round(text3, 2))+'%', fontsize=10, color='black',transform='map')
axs[9].pcolor(lon, lat, (MM_CMIP6_PM_Year_mean - MM_CMIP6_TH_Year_mean)/MM_CMIP6_PM_Year_mean*100, cmap=cmap, colorbar='b', 
              colorbar_kw=colorbar_kw2, extend='both',levels = plot.arange(-30, 30))
text4 = global_land_weighted_mean((MM_CMIP6_PM_Year_mean - MM_CMIP6_TH_Year_mean)/MM_CMIP6_PM_Year_mean*100) 
axs[9].text(-20, -58, str(round(text4, 2))+'%', fontsize=10, color='black',transform='map')

titlename = ['ERA5_PM', 'Ensemble_CMIP6_PM', 'Difference (a-b)', 'Difference (a to b)', 
             'ERA5_Har', 'Ensemble_CMIP6_Har', 'Difference (e-f)', 'Difference (e to f)', 
             'Difference (a-e)', 'Difference (b-f)']

for ax, titlena in zip(axs, titlename):
    ax.format(title=titlena)

# fig.colorbar(m, loc='b', label='PET (mm)',
#              labelsize=10,ticklabelsize=9, extendsize='1.7em') 

save_path = "/climca/people/xliu/Daily_SPEI_CMIP6/Scripts/PET_Compare/Year_mean/PET/"
plot.pyplot.savefig(os.path.join(save_path, "Spatial_dis_pet_compare_PM_Har_ERA5_CMIP6_Ensemble_1950_2014.jpg"), dpi=500)


# -----------------PET不同方法年际平均值,ERA5,CMIP6_ensemble-------------------------
# 绘制实测和年际均值PET 
plot.rc.reso = 'lo'#海岸线可以不同分辨率 'hi' 'med' 'lo' 'x-hi' 'xx-hi'
proj = plot.Proj('robin')
array = [
    [1,2,3],
    [4,5,6],
    [7,8,0]
]
fig, axs = plot.subplots(array, figwidth=7, proj=proj)
axs.format(
    abc=True,abcloc='l',abcsize=10, abcstyle='a)', gridlabelsize=18,
    labels=False, lonlines=120, latlines=30,
    coast=True,gridminor=False,coastlinewidth=1,
    suptitle='',suptitlesize=14)  # Annual PET for comparative data
cmap1=plot.Colormap('Spectral', reverse=True) #gamma提高颜色深度 , gamma=0.3 NegPos
cmap2=plot.Colormap('NegPos', reverse=True) #gamma提高颜色深度 , gamma=0.3 NegPos
colorbar_kw1={'label': 'PET (mm)','length': 0.7, 'width': 0.09, 'labelsize': 8, 'ticklabelsize': 7, 'ticks': np.arange(100,2501,800)}
colorbar_kw2={'label': 'PET (mm)','length': 0.7, 'width': 0.09, 'labelsize': 8, 'ticklabelsize': 7, 'ticks': np.arange(-500,501,250)}

axs[0].pcolor(lon, lat, ERA5_PM_Year, cmap=cmap1, colorbar='b', colorbar_kw=colorbar_kw1, extend='both',levels = plot.arange(100, 2500))
axs[1].pcolor(lon, lat, MM_CMIP6_PM_Year_mean, cmap=cmap1, colorbar='b', colorbar_kw=colorbar_kw1, extend='both',levels = plot.arange(100, 2500))
axs[2].pcolor(lon, lat, ERA5_PM_Year - MM_CMIP6_PM_Year_mean, cmap=cmap2, colorbar='b', colorbar_kw=colorbar_kw2, extend='both',levels = plot.arange(-500, 500))
text1 = global_land_weighted_mean(ERA5_PM_Year - MM_CMIP6_PM_Year_mean) 
axs[2].text(-25, -58, str(round(text1, 2))+' mm', fontsize=10, color='black',transform='map')
axs[3].pcolor(lon, lat, ERA5_TH_Year, cmap=cmap1, colorbar='b', colorbar_kw=colorbar_kw1, extend='both',levels = plot.arange(100, 2500))
axs[4].pcolor(lon, lat, MM_CMIP6_TH_Year_mean, cmap=cmap1, colorbar='b', colorbar_kw=colorbar_kw1, extend='both',levels = plot.arange(100, 2500))
axs[5].pcolor(lon, lat, ERA5_TH_Year - MM_CMIP6_TH_Year_mean, cmap=cmap2, colorbar='b', colorbar_kw=colorbar_kw2, extend='both',levels = plot.arange(-500, 500))
text2 = global_land_weighted_mean(ERA5_TH_Year - MM_CMIP6_TH_Year_mean) 
axs[5].text(-15, -58, str(round(text2, 2))+' mm', fontsize=10, color='black',transform='map')
axs[6].pcolor(lon, lat, ERA5_PM_Year - ERA5_TH_Year, cmap=cmap2, colorbar='b', colorbar_kw=colorbar_kw2, extend='both',levels = plot.arange(-500, 500))
text3 = global_land_weighted_mean(ERA5_PM_Year - ERA5_TH_Year) 
axs[6].text(-15, -58, str(round(text3, 2))+' mm', fontsize=10, color='black',transform='map')
axs[7].pcolor(lon, lat, MM_CMIP6_PM_Year_mean - MM_CMIP6_TH_Year_mean, cmap=cmap2, colorbar='b', colorbar_kw=colorbar_kw2, extend='both',levels = plot.arange(-500, 500))
text4 = global_land_weighted_mean(MM_CMIP6_PM_Year_mean - MM_CMIP6_TH_Year_mean) 
axs[7].text(-20, -58, str(round(text4, 2))+' mm', fontsize=10, color='black',transform='map')

titlename = ['ERA5_PM', 'Ensemble_CMIP6_PM', 'Difference (a-b)', 'ERA5_Har', 'Ensemble_CMIP6_Har', 
             'Difference (d-e)', 'Difference (a-d)', 'Difference (b-e)', 'Yearly variations']

for ax, titlena in zip(axs, titlename):
    ax.format(title=titlena)

# fig.colorbar(m, loc='b', label='PET (mm)',
#              labelsize=10,ticklabelsize=9, extendsize='1.7em') 

save_path = "/climca/people/xliu/Daily_SPEI_CMIP6/Scripts/PET_Compare/Year_mean/"
plot.pyplot.savefig(os.path.join(save_path, "Spatial_pet_compare_PM_Har_ERA5_CMIP6_Ensemble_1950_2014.jpg"), dpi=500)


# -----------------PET不同方法年际平均值偏差率,ERA5,CMIP6_ensemble-------------------------
# 绘制实测和年际均值PET 
plot.rc.reso = 'lo'#海岸线可以不同分辨率 'hi' 'med' 'lo' 'x-hi' 'xx-hi'
proj = plot.Proj('robin')
array = [
    [1,2,3],
    [4,5,6],
    [7,8,0]
]
fig, axs = plot.subplots(array, figwidth=7, proj=proj)
axs.format(
    abc=True,abcloc='l',abcsize=10, abcstyle='a)', gridlabelsize=18,
    labels=False, lonlines=120, latlines=30,
    coast=True,gridminor=False,coastlinewidth=1,
    suptitle='',suptitlesize=14)  # Annual PET for comparative data
cmap1=plot.Colormap('Spectral', reverse=True) #gamma提高颜色深度 , gamma=0.3 NegPos
cmap2=plot.Colormap('NegPos', reverse=True) #gamma提高颜色深度 , gamma=0.3 NegPos
colorbar_kw1={'label': 'PET (mm)','length': 0.7, 'width': 0.09, 'labelsize': 8, 'ticklabelsize': 7, 'ticks': np.arange(100,2501,800)}
colorbar_kw2={'label': 'PET (mm)','length': 0.7, 'width': 0.09, 'labelsize': 8, 'ticklabelsize': 7, 'ticks': np.arange(-30,30.5,15)}

m= axs[0].pcolor(lon, lat, ERA5_PM_Year, cmap=cmap1, colorbar='b', colorbar_kw=colorbar_kw1, extend='both',levels = plot.arange(100, 2500))
axs[1].pcolor(lon, lat, MM_CMIP6_PM_Year_mean, cmap=cmap1, colorbar='b', colorbar_kw=colorbar_kw1, extend='both',levels = plot.arange(100, 2500))
axs[2].pcolor(lon, lat, (ERA5_PM_Year - MM_CMIP6_PM_Year_mean)/ERA5_PM_Year*100, cmap=cmap2, colorbar='b', colorbar_kw=colorbar_kw2, extend='both',levels = plot.arange(-30, 30))
text1 = global_land_weighted_mean((ERA5_PM_Year - MM_CMIP6_PM_Year_mean)/ERA5_PM_Year*100) 
axs[2].text(-5, -58, str(round(text1, 2))+'%', fontsize=10, color='black',transform='map')
axs[3].pcolor(lon, lat, ERA5_TH_Year, cmap=cmap1, colorbar='b', colorbar_kw=colorbar_kw1, extend='both',levels = plot.arange(100, 2500))
axs[4].pcolor(lon, lat, MM_CMIP6_TH_Year_mean, cmap=cmap1, colorbar='b', colorbar_kw=colorbar_kw1, extend='both',levels = plot.arange(100, 2500))
axs[5].pcolor(lon, lat, (ERA5_TH_Year - MM_CMIP6_TH_Year_mean)/ERA5_TH_Year*100, cmap=cmap2, colorbar='b', colorbar_kw=colorbar_kw2, extend='both',levels = plot.arange(-30, 30))
text2 = global_land_weighted_mean((ERA5_TH_Year - MM_CMIP6_TH_Year_mean)/ERA5_TH_Year*100) 
axs[5].text(-5, -58, str(round(text2, 2))+'%', fontsize=10, color='black',transform='map')
axs[6].pcolor(lon, lat, (ERA5_PM_Year - ERA5_TH_Year)/ERA5_PM_Year*100, cmap=cmap2, colorbar='b', colorbar_kw=colorbar_kw2, extend='both',levels = plot.arange(-30, 30))
text3 = global_land_weighted_mean((ERA5_PM_Year - ERA5_TH_Year)/ERA5_PM_Year*100) 
axs[6].text(-5, -58, str(round(text3, 2))+'%', fontsize=10, color='black',transform='map')
axs[7].pcolor(lon, lat, (MM_CMIP6_PM_Year_mean - MM_CMIP6_TH_Year_mean)/MM_CMIP6_PM_Year_mean*100, cmap=cmap2, colorbar='b', colorbar_kw=colorbar_kw2, extend='both',levels = plot.arange(-30, 30))
text4 = global_land_weighted_mean((MM_CMIP6_PM_Year_mean - MM_CMIP6_TH_Year_mean)/MM_CMIP6_PM_Year_mean*100) 
axs[7].text(-5, -58, str(round(text4, 2))+'%', fontsize=10, color='black',transform='map')

titlename = ['ERA5_PM', 'Ensemble_CMIP6_PM', 'Difference (a-b)', 'ERA5_Har', 'Ensemble_CMIP6_Har', 'Difference (d-e)', 'Difference (a-d)', 'Difference (b-e)']

for ax, titlena in zip(axs, titlename):
    ax.format(title=titlena)

save_path = "/climca/people/xliu/Daily_SPEI_CMIP6/Scripts/PET_Compare/Year_mean/"
plot.pyplot.savefig(os.path.join(save_path, "Spatial_pet_compare_percent_PM_Har_ERA5_CMIP6_Ensemble_1950_2014.jpg"), dpi=500)

# -----------------PET不同方法年际平均值,PM方法，CMIP6多个模型-------------------------
# 绘制实测和年际均值PET 
plot.rc.reso = 'lo'#海岸线可以不同分辨率 'hi' 'med' 'lo' 'x-hi' 'xx-hi'
proj = plot.Proj('robin')
array = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]
]
fig, axs = plot.subplots(array, figwidth=7, proj=proj)
axs.format(
    abc=True,abcloc='l',abcsize=10, abcstyle='a)', gridlabelsize=18,
    labels=False, lonlines=120, latlines=30,
    coast=True,gridminor=False,coastlinewidth=1,
    suptitle='',suptitlesize=14)  # Annual PET for comparative data
cmap1=plot.Colormap('Spectral', reverse=True) #gamma提高颜色深度 , gamma=0.3 NegPos
# colorbar_kw1={'label': 'PET (mm)','length': 0.7, 'width': 0.15, 'ticks': np.arange(500,2501,500)}

for i in range(16):
    m=axs[i].pcolor(lon, lat, CMIP6_PM_Year_mean[i,:,:], cmap=cmap1, extend='both',levels = plot.arange(100, 2500))

titlename = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-ESM-1-1-LR', 'CMCC-ESM2', 'EC-Earth3', 
              'EC-Earth3-CC', 'EC-Earth3-Veg', 'EC-Earth3-Veg-LR', 'INM-CM4-8',  'INM-CM5-0', 
              'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM-1-2-HAM','MPI-ESM1-2-HR', 'MPI-ESM1-2-LR',
              'MRI-ESM2-0']

for ax, titlena in zip(axs, titlename):
    ax.format(title=titlena)

fig.colorbar(m, loc='b', label='PET (mm)',
             labelsize=10,ticklabelsize=10, extendsize='1.7em', length=0.80, width=0.12, ticks=np.arange(100,2501,400))

save_path = "/climca/people/xliu/Daily_SPEI_CMIP6/Scripts/PET_Compare/Year_mean/"
plot.pyplot.savefig(os.path.join(save_path, "Spatial_pet_compare_PM_CMIP6_models_1950_2014.jpg"), dpi=500)

# -----------------PET不同方法年际平均值,TH方法，CMIP6多个模型-------------------------
# 绘制实测和年际均值PET 
plot.rc.reso = 'lo'#海岸线可以不同分辨率 'hi' 'med' 'lo' 'x-hi' 'xx-hi'
proj = plot.Proj('robin')
array = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]
]
fig, axs = plot.subplots(array, figwidth=7, proj=proj)
axs.format(
    abc=True,abcloc='l',abcsize=10, abcstyle='a)', gridlabelsize=18,
    labels=False, lonlines=120, latlines=30,
    coast=True,gridminor=False,coastlinewidth=1,
    suptitle='',suptitlesize=14)  # Annual PET for comparative data
cmap1=plot.Colormap('Spectral', reverse=True) #gamma提高颜色深度 , gamma=0.3 NegPos
# colorbar_kw1={'label': 'PET (mm)','length': 0.7, 'width': 0.15, 'ticks': np.arange(500,2501,500)}

for i in range(16):
    m=axs[i].pcolor(lon, lat, CMIP6_TH_Year_mean[i,:,:], cmap=cmap1, extend='both',levels = plot.arange(100, 2500))

titlename = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-ESM-1-1-LR', 'CMCC-ESM2', 'EC-Earth3', 
              'EC-Earth3-CC', 'EC-Earth3-Veg', 'EC-Earth3-Veg-LR', 'INM-CM4-8',  'INM-CM5-0', 
              'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM-1-2-HAM','MPI-ESM1-2-HR', 'MPI-ESM1-2-LR',
              'MRI-ESM2-0']

for ax, titlena in zip(axs, titlename):
    ax.format(title=titlena)

fig.colorbar(m, loc='b', label='PET (mm)',
             labelsize=10,ticklabelsize=10, extendsize='1.7em', length=0.80, width=0.12, ticks=np.arange(100,2501,400))

save_path = "/climca/people/xliu/Daily_SPEI_CMIP6/Scripts/PET_Compare/Year_mean/"
plot.pyplot.savefig(os.path.join(save_path, "Spatial_pet_compare_TH_CMIP6_models_1950_2014.jpg"), dpi=500)
