## Code for extracting drought events based on SPEI data, extracting temporal characteristics (such as drought start time, end time, and duration), 
## conducting significance analysis, and preparing results for visualization.

import numpy as np
import proplot as plot
import pymannkendall as mk
import tifffile as tf
from scipy.ndimage import zoom
import os

def near_resampling(data):

    lon_vector = plot.arange(-179, 179, 4)
    lat_vector = plot.arange(89, -89, -4)

    resampled_data = zoom(data, (0.625, 0.625), order=1)

    # import matplotlib.pyplot as plt
    # plt.imshow(resampled_data, cmap='jet')

    # 提取重采样后不是 NaN 值的栅格的行列号
    valid_indices = np.where(~np.isnan(resampled_data))
    valid_rows, valid_cols = valid_indices

    # 根据行列号读取经度和纬度值
    valid_lon = lon_vector[valid_cols]
    valid_lat = lat_vector[valid_rows]

    return valid_lat, valid_lon

def significant_mk(array1, significance_level=0.05):

    significant_grids = np.full((72, 144), np.nan)  # 用于存储通过显著性检验的栅格
    slope = np.full((72, 144), np.nan) 

    reference = tf.imread('/climca/people/xliu/reference/2.5度/reference_no_antarctica.tif')
    reference = reference.astype(float)
    reference[62:, :] = np.nan

    for i in range(array1.shape[0]):
        for j in range(array1.shape[1]):


            if np.isnan(reference[i, j]):
               significant_grids[i, j] = np.nan
               continue

            if np.isnan(array1[i, j, :]).any():
               significant_grids[i, j] = np.nan
               continue

            mk_result = mk.original_test(array1[i, j, :], alpha=significance_level)
            # mk_result[7] is slope value, mk_result[2] is p value
            slope[i, j] = mk_result[7]
            if mk_result[2] < significance_level:
                significant_grids[i, j] = mk_result[7]

    return slope, significant_grids

def mark_sig_areas(tem, threshold=0.8):
    """
    根据给定阈值，判断三维数组中每个位置的值是否至少有一定比例非 NaN,
    如果是,则返回一个标记为1的二维数组,否则为 NaN。

    参数:
    tem (numpy.ndarray): 三维数组，形状为 (72, 144, 16)
    threshold (float): 判断非 NaN 的比例阈值，默认为 0.8

    返回:
    numpy.ndarray: 二维数组，形状为 (72, 144),符合条件的标记为1,否则为 NaN
    """
    # 计算每个位置非 NaN 的数量
    non_nan_count = np.sum(~np.isnan(tem), axis=2)

    # 计算满足条件的最小非 NaN 数量
    required_non_nan_count = threshold * tem.shape[2]

    # 创建布尔掩码，判断是否满足阈值条件
    mask = non_nan_count >= required_non_nan_count

    # 创建 (72, 144) 大小的数组，符合条件的标记为 1，否则为 NaN
    tem = np.where(mask, 1, np.nan)

    return tem

def check_sign_ratio(tem):

    # 获取三维数组的尺寸
    height, width, depth = tem.shape
    
    # 初始化一个二维数组，存储结果，使用整数类型
    result = np.zeros((height, width), dtype=int)*np.nan
    
    for i in range(height):
        for j in range(width):
            
            # 取出三维数组的第三个维度上 (depth) 所有的元素
            if np.isnan(reference[i, j]):
               result[i, j] = np.nan
               continue
           
            elements = tem[i, j, :]
            # 计算正数和负数的比例
            positive_ratio = np.sum(elements > 0) / depth
            negative_ratio = np.sum(elements < 0) / depth
            
            # 判断是否相同符号的比例超过80%
            if positive_ratio > 0.8 or negative_ratio > 0.8:
                result[i, j] = 1  # 相同符号的比例超过80%，标记为1
            else:
                result[i, j] = np.nan  # 否则标记为0
                
    return result

def calculate_diff_and_signal(tem1, tem2):
    """
    计算两个二维数组的差值，并生成一个标志数组，判断对应元素是否不同时为正或不同时为负。

    参数:
    tem1 (numpy.ndarray): 第一个二维数组，形状为 (72, 144)
    tem2 (numpy.ndarray): 第二个二维数组，形状为 (72, 144)

    返回:
    tuple: 返回两个数组：
           - diff1: tem1 和 tem2 的差值数组，形状为 (72, 144)
           - diff_sig: 标志数组，形状为 (72, 144)，如果对应元素不同时为正或不同时为负则为 1,否则为 0
    """
    # 计算差值数组
    diff1 = tem1 - tem2

    # 生成标志数组 diff_sig
    # 判断条件: 如果两个数组对应元素不同时为正或不同时为负，则赋值为 1，否则为 0
    diff_sig = np.where((tem1 * tem2) <= 0, 1, np.nan)

    return diff1, diff_sig

def extract_first_layer(temp):
    """
    从给定的 temp 列表中提取每个子项的第一层数据。

    参数:
    temp (list): 一个大小为 64 的列表，每个子项为 (72, 144, 100) 的三维数组

    返回:
    np.ndarray: 大小为 (72, 144, 64) 的新数组，每个位置包含每个子项的第一层数据
    """
    # 初始化一个大小为 (72, 144, 64) 的数组
    new_array = np.full((72, 144, 64), np.nan)
    
    # 遍历 temp 中的每个子项，取出每个子项的第一层数据（第 0 维度）
    for i in range(64):
        new_array[:, :, i] = temp[i][:, :, 0]

    return new_array

def extract_last_valid_values(temp):
    """
    从给定的 temp 列表中提取每个位置的所有层中的最后一个有效数字。

    参数:
    temp (list): 一个大小为 64 的列表，每个子项为 (72, 144, 100) 的三维数组，可能包含 NaN 值

    返回:
    np.ndarray: 大小为 (72, 144, 64) 的新数组，每个位置包含每个年份的最后一个有效数字
    """
    # 初始化一个大小为 (72, 144, 64) 的数组，默认值为 nan
    new_array = np.full((72, 144, 64), np.nan)

    # 遍历 temp 中的每个子项，找出每个位置的最后一个有效数字
    for i in range(64):
        for x in range(72):
            for y in range(144):
                # 提取 temp[i] 中 (x, y) 位置的所有层
                layers = temp[i][x, y, :]
                
                # 找到最后一个有效的非 NaN 数字
                valid_values = layers[~np.isnan(layers)]
                if valid_values.size > 0:
                    # 取最后一个有效数字
                    new_array[x, y, i] = valid_values[-1]

    return new_array

def sum_valid_values(temp):
    """
    计算 temp 列表中每个子项的所有位置的有效数字的和，忽略 NaN。

    参数:
    temp (list): 一个大小为 64 的列表，每个子项为 (72, 144, 100) 的三维数组，可能包含 NaN 值

    返回:
    np.ndarray: 大小为 (72, 144, 64) 的新数组，每个位置包含每个年份所有层的有效值之和
    """
    # 初始化一个大小为 (72, 144, 64) 的数组，用于存储每个位置的有效值之和
    sum_array = np.zeros((72, 144, 64))

    # 遍历 temp 中的每个子项，计算每个位置的有效数字之和
    for i in range(64):
        for x in range(72):
            for y in range(144):
                # 提取 temp[i] 中 (x, y) 位置的所有层
                layers = temp[i][x, y, :]
                
                # 计算非 NaN 数字的和
                sum_array[x, y, i] = np.nansum(layers)
    
    # 检查 sum_array 中所有值为 0 的位置，并将其设置为 NaN
    sum_array[np.all(sum_array == 0, axis=2)] = np.nan
    
    return sum_array

def back_year_length(temp,temp1):
    """
    处理给定的三维数组 temp,将其转换为包含 64 个年份的数据结构，
    每个年份的数据大小为 (72, 144, 100)，用 nan 填充不足的部分。

    参数:
    temp (numpy.ndarray): 形状为 (72, 144, 6500) 的三维数组

    返回:
    list: 包含 64 个子项的列表，每个子项为 (72, 144, 100) 的数据结构
    """
    # temp = ERA5_PM_DS 
    # temp1 = ERA5_PM_DL
    # temp2 = ERA5_PM_DE
    # 定义年份和天数参数
    start_year = 1951
    end_year = 2014
    days_per_year = 365
    total_days = (end_year - start_year + 1) * days_per_year  # 23360天

    # 初始化tem3为包含64个子项的列表，每个子项大小为(72, 144, 100)
    tem3 = [np.full((72, 144, 100), np.nan) for _ in range(64)]
    tem4 = [np.full((72, 144, 100), np.nan) for _ in range(64)]

    # i = 30 
    # j = 80
    # 对每个栅格位置进行处理
    for i in range(72):
        for j in range(144):
            # 取出 temp[i, j, :] 中非 NaN 的值
            valid_values = temp[i, j, :][~np.isnan(temp[i, j, :])]
            valid_values1 = temp1[i, j, :][~np.isnan(temp1[i, j, :])]

            # 将有效值按年份还原，并存储在tem3中
            year_data = [[] for _ in range(64)]  # 初始化64年的数据存储
            year_data1 = [[] for _ in range(64)]  # 初始化64年的数据存储
            for mm, value in enumerate(valid_values):
                if 0 <= value < total_days:
                    year_index = int(value) // days_per_year  # 计算对应的年份
                    day_of_year = int(value) % days_per_year + 1  # 计算该年的第几天
                    year_data[year_index].append(day_of_year)
                    year_data1[year_index].append(valid_values1[mm])
                       
            # 对每个年份的数据进行填充，使每个栅格位置长度固定为100
            for year_index in range(64):
                # 如果有效天数不足100，使用nan填充
                tem3[year_index][i, j, :len(year_data[year_index])] = year_data[year_index][:100]
                tem4[year_index][i, j, :len(year_data1[year_index])] = year_data1[year_index][:100]
                # No additional need to manually fill with np.nan, np.full already does this.
    
    tem33 = extract_first_layer(tem3)
    tem44 = sum_valid_values(tem4)
                
    return tem33,tem44

def back_year_timing(temp):
    """
    处理给定的三维数组 temp,将其转换为包含 64 个年份的数据结构，
    每个年份的数据大小为 (72, 144, 100)，用 nan 填充不足的部分。

    参数:
    temp (numpy.ndarray): 形状为 (72, 144, 6500) 的三维数组

    返回:
    list: 包含 64 个子项的列表，每个子项为 (72, 144, 100) 的数据结构
    """
    # temp = ERA5_PM_DS 
    # 定义年份和天数参数
    start_year = 1951
    end_year = 2014
    days_per_year = 365
    total_days = (end_year - start_year + 1) * days_per_year  # 23360天

    # 初始化tem3为包含64个子项的列表，每个子项大小为(72, 144, 100)
    tem3 = [np.full((72, 144, 100), np.nan) for _ in range(64)]

    # 对每个栅格位置进行处理
    for i in range(72):
        for j in range(144):
            
            # 取出 temp[i, j, :] 中非 NaN 的值
            valid_values = temp[i, j, :][~np.isnan(temp[i, j, :])]

            # 将有效值按年份还原，并存储在tem3中
            year_data = [[] for _ in range(64)]  # 初始化64年的数据存储
            for mm, value in enumerate(valid_values):
                if 0 <= value < total_days:
                    year_index = int(value) // days_per_year  # 计算对应的年份
                    day_of_year = int(value) % days_per_year + 1  # 计算该年的第几天
                    year_data[year_index].append(day_of_year)
                       
            # 对每个年份的数据进行填充，使每个栅格位置长度固定为100
            for year_index in range(64):
                # 如果有效天数不足100，使用nan填充
                tem3[year_index][i, j, :len(year_data[year_index])] = year_data[year_index][:100]
                # No additional need to manually fill with np.nan, np.full already does this.
    
    tem33 = extract_first_layer(tem3)  
       
    return tem33

def calculate_position(ERA5_trend, trend_M):
    # 初始化dis_trend的shape，最后一个维度是与CMIP6模型数量一致
    dis_trend = np.full_like(ERA5_trend, np.nan)

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

scales = 180
Data_path = '/climca/people/xliu/Daily_SPEI_CMIP6/Scripts/SPEI_Compare/Dry_indentidy/'
Out_path = '//climca/people/xliu/Daily_SPEI_CMIP6/Scripts/SPEI_Compare/Daily_Characteristics/'

reference = tf.imread('/climca/people/xliu/reference/2.5度/reference_no_antarctica.tif')
reference = reference.astype(float)
reference[62:, :] = np.nan


# 加载ERA5_PM 干旱特征 
ERA5_PM_DS = np.load(Data_path +  str(scales) + 'days/' + str(scales) + 'days_ERA5_PM_Dry_Start_1950-2014.npy')
ERA5_PM_DE = np.load(Data_path +  str(scales) + 'days/' + str(scales) + 'days_ERA5_PM_Dry_End_1950-2014.npy')
ERA5_PM_DL = np.load(Data_path +  str(scales) + 'days/' + str(scales) + 'days_ERA5_PM_Dry_Length_1950-2014.npy')
E_PM_DS, E_PM_DL = back_year_length(ERA5_PM_DS,ERA5_PM_DL)
E_PM_DE = back_year_timing(ERA5_PM_DS)

# 加载ERA5_Har 干旱特征 
ERA5_Har_DS = np.load(Data_path +  str(scales) + 'days/' + str(scales) + 'days_ERA5_Har_Dry_Start_1950-2014.npy')
ERA5_Har_DE = np.load(Data_path +  str(scales) + 'days/' + str(scales) + 'days_ERA5_Har_Dry_End_1950-2014.npy')
ERA5_Har_DL = np.load(Data_path +  str(scales) + 'days/' + str(scales) + 'days_ERA5_Har_Dry_Length_1950-2014.npy')
E_Har_DS, E_Har_DL = back_year_length(ERA5_Har_DS,ERA5_Har_DL)
E_Har_DE = back_year_timing(ERA5_Har_DS)

# 加载CMIP6_PM 干旱特征 
CMIP6_PM_DS = np.load(Data_path +  str(scales) + 'days/' + str(scales) + 'days_CMIP6_PM_Dry_Start_1950-2014.npy')
CMIP6_PM_DE = np.load(Data_path +  str(scales) + 'days/' + str(scales) + 'days_CMIP6_PM_Dry_End_1950-2014.npy')
CMIP6_PM_DL = np.load(Data_path +  str(scales) + 'days/' + str(scales) + 'days_CMIP6_PM_Dry_Length_1950-2014.npy')
C_PM_DS = []
C_PM_DL = []
C_PM_DE = []
for i in range(16):
    temm1, temm2  = back_year_length(CMIP6_PM_DS[i,:,:,:],CMIP6_PM_DL[i,:,:,:])
    temm3  = back_year_timing(CMIP6_PM_DE[i,:,:,:])
    C_PM_DS.append(temm1)
    C_PM_DL.append(temm2)
    C_PM_DE.append(temm3)
    print(str(i+1))

# 加载CMIP6_Har 干旱特征 
CMIP6_Har_DS = np.load(Data_path +  str(scales) + 'days/' + str(scales) + 'days_CMIP6_Har_Dry_Start_1950-2014.npy')
CMIP6_Har_DE = np.load(Data_path +  str(scales) + 'days/' + str(scales) + 'days_CMIP6_Har_Dry_End_1950-2014.npy')
CMIP6_Har_DL = np.load(Data_path +  str(scales) + 'days/' + str(scales) + 'days_CMIP6_Har_Dry_Length_1950-2014.npy')
C_Har_DS = []
C_Har_DL = []
C_Har_DE = []
for i in range(16):
    temm1, temm2  = back_year_length(CMIP6_Har_DS[i,:,:,:],CMIP6_Har_DL[i,:,:,:])
    temm3  = back_year_timing(CMIP6_Har_DE[i,:,:,:])
    C_Har_DS.append(temm1)
    C_Har_DL.append(temm2)
    C_Har_DE.append(temm3)
    print(str(i+1))


# 绘制图表 
lon = plot.arange(-178.75, 178.75, 2.5)
lat = plot.arange(88.75, -88.754, -2.5)

#  ------------------# PM CMIP6 models DL 变化趋势-------------------------
trend_PM_M = np.full((72, 144, 16), np.nan)  # models 
trend_sig_PM_M = np.full((72, 144, 16), np.nan)
for i in range(16):
    trend_PM_M[:, :, i], trend_sig_PM_M[:, :, i]  = significant_mk(C_PM_DL[i]) 
    print(str(i+1))

plot.rc.reso = 'lo'#海岸线可以不同分辨率 'hi' 'med' 'lo' 'x-hi' 'xx-hi'
proj = plot.Proj('robin')
fig, axs = plot.subplots(ncols=4, nrows=4, figwidth=10, proj=proj)
axs.format(
    abc=True,abcloc='l',abcsize=8, abcstyle='a)', gridlabelsize=18,
    labels=False, lonlines=120, latlines=30,
    coast=True,gridminor=False,coastlinewidth=1,
    suptitle='',suptitlesize=14)  # Annual PET for comparative data
cmap=plot.Colormap('RdBu', reverse=True) #gamma提高颜色深度 , 'DryWet' gamma=0.3 NegPos , reverse=True

for i in range(16): 
    m=axs[i].pcolor(lon, lat, trend_PM_M[:, :, i], cmap=cmap, extend='both',levels = plot.arange(-0.5, 0.5, 0.1))
    lat1, lon1 = near_resampling(trend_sig_PM_M[:, :, i])
    axs[i].scatter(lon1, lat1, s=0.1, color='black')

# for i in range(16): 
#     m=axs[i].contourf(lon, lat, trend_EE[:, :, i], cmap=cmap, extend='both',levels = plot.arange(-2, 2, 0.5))

titlename =  ['ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-ESM-1-1-LR', 'CMCC-ESM2', 'EC-Earth3', 
            'EC-Earth3-CC', 'EC-Earth3-Veg', 'EC-Earth3-Veg-LR', 'INM-CM4-8',  'INM-CM5-0', 
            'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM-1-2-HAM','MPI-ESM1-2-HR', 'MPI-ESM1-2-LR',
            'MRI-ESM2-0']

for ax, titlena in zip(axs, titlename):
    ax.format(title=titlena)

fig.colorbar(m, loc='b', label='Drought length (days/year)',
             labelsize=10,ticklabelsize=9, extendsize='1.7em', length=0.60, width=0.15) 
# 
save_path = Out_path +  str(scales) + 'days/' + str(scales) + 'days_Dry_Length_Trend_CMIP6_PM_1950-2014.jpg'
plot.pyplot.savefig(save_path, dpi=600)

#  ------------------# Har CMIP6 models DL 变化趋势-------------------------
trend_Har_M = np.full((72, 144, 16), np.nan)  # models 
trend_sig_Har_M = np.full((72, 144, 16), np.nan)
for i in range(16):
    trend_Har_M[:, :, i], trend_sig_Har_M[:, :, i]  = significant_mk(C_Har_DL[i]) 
    print(str(i+1))

plot.rc.reso = 'lo'#海岸线可以不同分辨率 'hi' 'med' 'lo' 'x-hi' 'xx-hi'
proj = plot.Proj('robin')
fig, axs = plot.subplots(ncols=4, nrows=4, figwidth=10, proj=proj)
axs.format(
    abc=True,abcloc='l',abcsize=8, abcstyle='a)', gridlabelsize=18,
    labels=False, lonlines=120, latlines=30,
    coast=True,gridminor=False,coastlinewidth=1,
    suptitle='',suptitlesize=14)  # Annual PET for comparative data
cmap=plot.Colormap('RdBu', reverse=True) #gamma提高颜色深度 , 'DryWet' gamma=0.3 NegPos , reverse=True

for i in range(16): 
    m=axs[i].pcolor(lon, lat, trend_Har_M[:, :, i], cmap=cmap, extend='both',levels = plot.arange(-0.5, 0.5, 0.1))
    lat1, lon1 = near_resampling(trend_sig_Har_M[:, :, i])
    axs[i].scatter(lon1, lat1, s=0.1, color='black')

# for i in range(16): 
#     m=axs[i].contourf(lon, lat, trend_EE[:, :, i], cmap=cmap, extend='both',levels = plot.arange(-2, 2, 0.5))

titlename =  ['ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-ESM-1-1-LR', 'CMCC-ESM2', 'EC-Earth3', 
            'EC-Earth3-CC', 'EC-Earth3-Veg', 'EC-Earth3-Veg-LR', 'INM-CM4-8',  'INM-CM5-0', 
            'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM-1-2-HAM','MPI-ESM1-2-HR', 'MPI-ESM1-2-LR',
            'MRI-ESM2-0']

for ax, titlena in zip(axs, titlename):
    ax.format(title=titlena)

fig.colorbar(m, loc='b', label='Drought length (days/year)',
             labelsize=10,ticklabelsize=9, extendsize='1.7em', length=0.60, width=0.15) 
# 
save_path = Out_path +  str(scales) + 'days/' + str(scales) + 'days_Dry_Length_Trend_CMIP6_Har_1950-2014.jpg'
plot.pyplot.savefig(save_path, dpi=600)

#  ------------------# ERA5 与 Ensemble 变化趋势 与差异-------------------------
trend_EE = np.full((72, 144, 4), np.nan)  # ERA5 and Ensemble 
trend_sig_EE = np.full((72, 144, 4), np.nan)
trend_EE[:, :, 0], trend_sig_EE[:, :, 0] = significant_mk(E_PM_DL) 
trend_EE[:, :, 1] = np.median(trend_PM_M, axis=2) 
trend_sig_EE[:, :, 1] = check_sign_ratio(trend_PM_M)
trend_EE[:, :, 2], trend_sig_EE[:, :, 2] = significant_mk(E_Har_DL) 
trend_EE[:, :, 3] = np.median(trend_Har_M, axis=2) 
trend_sig_EE[:, :, 3] = check_sign_ratio(trend_Har_M)

tt1 = trend_EE[:, :, 0]
tt2 = trend_EE[:, :, 1]

diff_EE = np.full((72, 144, 4), np.nan)  # ERA5 and Ensemble 
diff_sig_EE = np.full((72, 144, 4), np.nan)
diff_EE[:, :, 0], diff_sig_EE[:, :, 0] = calculate_diff_and_signal(trend_EE[:, :, 0], trend_EE[:, :, 1]) 
diff_EE[:, :, 1], diff_sig_EE[:, :, 1] = calculate_diff_and_signal(trend_EE[:, :, 2], trend_EE[:, :, 3]) 
diff_EE[:, :, 2], diff_sig_EE[:, :, 2] = calculate_diff_and_signal(trend_EE[:, :, 0], trend_EE[:, :, 2]) 
diff_EE[:, :, 3], diff_sig_EE[:, :, 3] = calculate_diff_and_signal(trend_EE[:, :, 1], trend_EE[:, :, 3]) 

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
cmap=plot.Colormap('RdBu', reverse=True) #gamma提高颜色深度 , 'DryWet' gamma=0.3 NegPos , reverse=True

# cmap1=plot.Colormap('Spectral', reverse=True) #gamma提高颜色深度 , gamma=0.3 NegPos
# cmap2=plot.Colormap('NegPos', reverse=True) #gamma提高颜色深度 , gamma=0.3 NegPos
colorbar_kw1={'label': 'Drought Length (days/year)','length': 0.7, 'width': 0.09, 'labelsize': 8, 'ticklabelsize': 7}
colorbar_kw2={'label': 'Drought Length (days/year)','length': 0.7, 'width': 0.09, 'labelsize': 8, 'ticklabelsize': 7}
colorbar_kw3={'label': 'Position','length': 0.7, 'width': 0.09, 'labelsize': 8, 'ticklabelsize': 7, 'ticks': np.arange(1, 15, 2)}

for i in range(12): 
    if i==0 or i==1 or i==4 or i==5: 
        if i>1:
            j=i-2
        else:
            j=i
        axs[i].pcolor(lon, lat, trend_EE[:, :, j], cmap=cmap, colorbar='b', colorbar_kw=colorbar_kw1, 
                      extend='both',levels = plot.arange(-1, 1, 0.2))
        lat1, lon1 = near_resampling(trend_sig_EE[:, :, j])
        axs[i].scatter(lon1, lat1, s=0.1, color='black')
    elif i==2 or i==6 or i==8 or i==9: 
        if i==2:
            j=0
        elif i==6:
            j=i-5
        elif i==8 or i==9:
            j=i-6
        axs[i].pcolor(lon, lat, diff_EE[:, :, j], cmap=cmap, colorbar='b', colorbar_kw=colorbar_kw1, 
                      extend='both',levels = plot.arange(-0.5, 0.5, 0.1))
        lat1, lon1 = near_resampling(diff_sig_EE[:, :, j])
        axs[i].scatter(lon1, lat1, s=0.1, color='black')
    else:
        if i==3:
            axs[i].pcolor(lon, lat, calculate_position(trend_EE[:,:,0], trend_PM_M), cmap=cmap, 
                          colorbar='b', colorbar_kw=colorbar_kw3, extend='both',levels = plot.arange(1, 15, 1))
        if i==7:
            axs[i].pcolor(lon, lat, calculate_position(trend_EE[:,:,2], trend_Har_M), cmap=cmap, 
                          colorbar='b', colorbar_kw=colorbar_kw3, extend='both',levels = plot.arange(1, 15, 1))           

# for i in range(4): 
#     m=axs[i].contourf(lon, lat, trend_EE[:, :, i], cmap=cmap, extend='both',levels = plot.arange(-3, 3, 1))

titlename = ['ERA5_PM', 'Ensemble_CMIP6_PM', 'Difference (a-b)', 'Distribution (a to b)', 'ERA5_Har', 'Ensemble_CMIP6_Har', 
             'Difference (d-e)','Distribution (a to b)', 'Difference (a-d)', 'Difference (b-e)']

for ax, titlena in zip(axs, titlename):
    ax.format(title=titlena)

save_path = Out_path +  str(scales) + 'days/' + str(scales) + 'days_Dry_Length_Trend_CMIP6_ERA5_1950-2014.jpg'
plot.pyplot.savefig(save_path, dpi=600)
