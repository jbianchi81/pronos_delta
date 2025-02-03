import json
import datetime
import requests

import pandas as pd
import numpy as np
from datetime import timedelta,datetime
import datetime
import dateutil
import pytz

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

from dateutil import parser

from pytz import timezone
localtz = timezone('America/Argentina/Buenos_Aires')

def createEmptyDataFrame():
    df = pd.DataFrame({"fecha":pd.Series(dtype='datetime64[ns]'),"h_obs":pd.Series(dtype='float')})
    df.set_index(df['fecha'], inplace=True)
    df.index.tz_localize("UTC")
    del df["fecha"]
    return df

def getObs(series_id,timestart,timeend):
    response = requests.get(
        apiLoginParams["url"] + '/obs/puntual/observaciones',
        params={
            'series_id': series_id,
            'timestart': timestart,
            'timeend': timeend
        },
        headers={'Authorization': 'Bearer ' + apiLoginParams["token"]},
    )
    json_response = response.json()
    if(not len(json_response)):
        print("No obs data found for series_id: %i" % series_id)
        return createEmptyDataFrame()
    df_obs = pd.DataFrame.from_dict(json_response)
    df_obs = df_obs.rename(columns={'timestart':'fecha','valor':'h_obs'})
    df_obs = df_obs[['fecha','h_obs']]
    df_obs['fecha'] = pd.to_datetime(df_obs['fecha']).dt.round('min')            # Fecha a formato fecha -- CAMBIADO PARA QUE CORRA EN PYTHON 3.5
    df_obs['h_obs'] = df_obs['h_obs'].astype(float)
    df_obs['fecha'] = pd.to_datetime(df_obs['fecha']).dt.round('min')            # Fecha a formato fecha
    df_obs.set_index(df_obs['fecha'], inplace=True)
    del df_obs['fecha']
    return df_obs


with open("config.json") as f:
    config = json.load(f)
apiLoginParams = config["api"]


f_inicio = datetime.datetime(2010,1,1)
f_fin = datetime.datetime(2025,2,1)
serie_obs = 33

df_Obs = getObs(serie_obs,f_inicio,f_fin)
print(df_Obs)

fig = plt.figure(figsize=(17, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(df_Obs.index, df_Obs['h_obs'])
plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
plt.tick_params(axis='both', labelsize=18)
plt.xlabel('Fecha', size=18)
plt.ylabel('Nivel [m]', size=18)
plt.legend(prop={'size':20},loc=0)
plt.tight_layout()
plt.show()