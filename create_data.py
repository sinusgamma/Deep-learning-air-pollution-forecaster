#%%
import numpy as np
import pandas as pd


#%%
# read source weather file
df = pd.read_excel(r"E:\MachineLearning\Udacity\PytorchChallenge\air_pollution_project\hourly_lorinc_2011-2018.xls", skiprows=[0,1,2,3,4,5])
df.head()

#%%
# read source pm10 file
df_10 = pd.read_excel(r"E:\MachineLearning\Udacity\PytorchChallenge\air_pollution_project\daily_pm10.xlsx")
df_10

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
# to hpa
df['pres'] = df['pres']*1.3332239
df.head()

#%%
# wind string to numbers
new = df['w_dir'].str.split("the ", n = 1, expand = True)
df['w_dir'] = new[1]

#%%
# examine unique wind strings
df['w_dir'].unique()

#%%
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
#convert wind to u and w components
# rad = 4.0*atan(1.0)/180
# u = -spd*sin(rad*dir) 
# v = -spd*cos(rad*dir)
df['u'] = -df['w_speed']*np.sin((4.0*np.arctan(1.0)/180)*df['w_dir'])
df['v'] = -df['w_speed']*np.cos((4.0*np.arctan(1.0)/180)*df['w_dir'])
df

#%%
# set datetime to index to enable later resample
df = df.set_index('datetime')
df.head()

#%%
# will missing precipitation data with 0
df['prec'] = pd.to_numeric(df['prec'], errors='coerce').fillna(0)

#%%
#resample for daily data
df_d = pd.DataFrame()
df_d['temp_avr'] = df['temp'].resample('d').mean()
df_d['temp_max'] = df['temp'].resample('d').max()
df_d['temp_min'] = df['temp'].resample('d').min()
df_d['pres'] = df['pres'].resample('d').mean()
df_d['u'] = df['u'].resample('d').mean()
df_d['v'] = df['v'].resample('d').mean()
df_d['prec'] = df['prec'].resample('d').max()
df_d['datetime'] = df_d.index

#%%
# set datetime to index
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
df_10['datetime'] = pd.to_datetime(df_10['datetime'])
df_10.head()

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
# merge databases
df_result = pd.merge(df_d, df_10, on='datetime')
df_result.isnull().sum()

#%%
# save to csv
df_result.to_csv('daily_clean_full.csv', index=False)

#%%
df_full = pd.read_csv('daily_clean_full.csv')
df_full