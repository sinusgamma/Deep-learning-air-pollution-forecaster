#%%
import numpy as np
import pandas as pd


#%%
# read source weather file
df = pd.read_excel(r"data\hourly_lorinc_2011-2018.xls", skiprows=[0,1,2,3,4,5])
df.head()

#%%
# read source pm10 file
df_10 = pd.read_excel(r"data\daily_pm10.xlsx")
df_10.head()

#%%
#retain important data
df = df[[   'dayth',
            'Local time in Budapest / Lorinc (airport)',
            'T',
            'P',
            'U',
            'DD',
            'Ff',
            'RRR']]
df.head()

#%%
# rename header
df = df.rename(index=str, columns={'Local time in Budapest / Lorinc (airport)' : 'datetime',
                                'T' : 'temp',
                                'P' : 'pres',
                                'U' : 'rhu',
                                'DD': 'w_dir',
                                'Ff' : 'w_speed',
                                'RRR': 'prec'})
df.head()

#%%
# to datetime
df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M')
df.head()

#%%
# millimetre of mercury to hpa
df['pres'] = df['pres']*1.3332239
df.head()

#%%
# convert wind strings to numbers
# split the strings
new = df['w_dir'].str.split("the ", n = 1, expand = True)
df['w_dir'] = new[1]

#%%
# examine unique wind strings
df['w_dir'].unique()

#%%
# convert wind-strings to numbers
# None won't matter because the corresponding wind speed is zero
wind_dict = {   'north-west':315, 
                'west-southwest':247.5, 
                'south-southeast':157.5, 
                'south-west':225,
                'south-southwest':202.5, 
                'south':180, 
                'west':270, 
                'west-northwest':292.5,
                'north-northwest':337.5, 
                'east':90, 
                'east-southeast':112.5, 
                'south-east':135,
                'north-east':45, 
                'east-northeast':67.5, 
                None:0, 
                'north-northeast':22.5, 
                'north':360
}
df = df.replace({'w_dir':wind_dict})
df.head()


#%%
#convert wind to u and w components (meridional and zonal components)
# rad = 4.0*atan(1.0)/180
# u = -spd*sin(rad*dir) 
# v = -spd*cos(rad*dir)
df['u'] = -df['w_speed']*np.sin((4.0*np.arctan(1.0)/180)*df['w_dir'])
df['v'] = -df['w_speed']*np.cos((4.0*np.arctan(1.0)/180)*df['w_dir'])
df.head()

#%%
# set datetime to index to enable later resample
df = df.set_index('datetime')
df.head()

#%%
# fill missing precipitation data with 0
df['prec'] = pd.to_numeric(df['prec'], errors='coerce').fillna(0)

#%%
#resample for daily data
# for prec: Because the are 6h, 12h, 24h measurement periods in our
# dataset we keep the largest value from this for the day. This isn't
# exact the exact precipitation during the day, but the period with
# the highest amount. Later with better data it can be replaced.
df_d = pd.DataFrame()
df_d['temp_avr'] = df['temp'].resample('d').mean()
df_d['temp_max'] = df['temp'].resample('d').max()
df_d['temp_min'] = df['temp'].resample('d').min()
df_d['pres'] = df['pres'].resample('d').mean()
df_d['u'] = df['u'].resample('d').mean()
df_d['v'] = df['v'].resample('d').mean()
df_d['prec'] = df['prec'].resample('d').max()
df_d['datetime'] = df_d.index
df_d

#%%
# convert station names to simpler strings
df_10 = df_10.rename(columns={  'Dátum' : 'datetime',
                                'Budapest Budatétény':'Budateteny',
                                'Budapest Csepel':'Csepel',
                                'Budapest Erzsébet tér':'Erzsebet',
                                'Budapest Gergely utca':'Gergely',
                                'Budapest Gilice tér':'Gilice',
                                'Budapest Honvéd':'Honved',
                                'Budapest Káposztásmegyer':'Kaposztas',
                                'Budapest Korakás park':'Korakas',
                                'Budapest Kosztolányi D. tér':'Kosztolanyi',
                                'Budapest Pesthidegkút':'Pesthidegkut',
                                'Budapest Széna tér':'Szena',
                                'Budapest Teleki tér':'Teleki'})
df_10.head()

#%%
# convert to datetime object
df_10['datetime'] = pd.to_datetime(df_10['datetime'], format='%Y.%m.%d')
df_10

#%%
# convert missing data to np.nan
df_10 = df_10.replace('Nincs adat', np.nan)
# convert concentration to number
df_10 = df_10.replace({' ug/m3':''}, regex = True)
df_10.iloc[:,1:] = df_10.iloc[:,1:].apply(pd.to_numeric, errors='coerce')
df_10.head()

#%%
# linear interpolate missing values
df_10.iloc[:,1:] = df_10.iloc[:,1:].interpolate()
df_10.head()

#%%
# merge databases inner join
df_result = pd.merge(df_d, df_10, on='datetime')
# check
df_result.isnull().sum()

#%%
# check
df_result.info()

#%%
# save to csv
df_result.to_csv('data\daily_clean_full.csv', index=False)

#%%
df_full = pd.read_csv('data\daily_clean_full.csv')
df_full

#%%
# sample and show one year period
from matplotlib import pyplot as plt
values = df_full[:365].values
values

#%%
# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]
i = 1
# plot each column
plt.figure()
for group in groups:
	plt.subplot(len(groups), 1, i)
	plt.plot(values[:, group])
	plt.title(df_full.columns[group], y=0.5, loc='right')
	i += 1
plt.show()