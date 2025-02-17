import json
import datetime
import requests

import pandas as pd
import numpy as np
from datetime import timedelta,datetime

import dateutil
import pytz

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

from dateutil import parser

from pytz import timezone
localtz = timezone('America/Argentina/Buenos_Aires')

"""# Credenciales de la API"""

with open("config.json") as f:
    config = json.load(f)
apiLoginParams = config["api"]


def getLastProno(cal_id=288,filter={}):
    response = requests.get(
        apiLoginParams["url"] + '/sim/calibrados/' + str(cal_id) + '/corridas/last',
        params = {
            **filter,
            "includeProno": True
        },
        headers = {'Authorization': 'Bearer ' + apiLoginParams["token"]},
    )
    json_response = response.json()
    if not len(json_response['series']):
        raise Exception("No series found in retrieved forecast run with filter: %s" % str(filter))
    df_sim = pd.DataFrame.from_dict(json_response['series'][0]['pronosticos'],orient='columns')
    df_sim = df_sim.rename(columns={"timestart":'fecha',"timeend":'fecha2',"valor":'h_sim',"qualifier":'main'})
    df_sim = df_sim[['fecha','h_sim']]
    df_sim['fecha'] = pd.to_datetime(df_sim['fecha'])
    df_sim['h_sim'] = df_sim['h_sim'].astype(float)
    df_sim = df_sim.sort_values(by='fecha')
    df_sim.set_index(df_sim['fecha'], inplace=True)
    # df_sim.index = df_sim.index.tz_convert(None)
    del df_sim['fecha']
    return df_sim, parser.parse(json_response["forecast_date"])

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

def createEmptyDataFrame():
    df = pd.DataFrame({"fecha":pd.Series(dtype='datetime64[ns]'),"h_obs":pd.Series(dtype='float')})
    df.set_index(df['fecha'], inplace=True)
    df.index.tz_localize("UTC")
    del df["fecha"]
    return df

def eliminaSaltos(df,umbral):
    df['dif_F'] = df['h_obs'].diff(periods=1).abs()
    df['dif_B'] = df['h_obs'].diff(periods=-1).abs()
    df = df.fillna(0)
    df = df[(df['dif_F']<umbral) & (df['dif_B']<umbral)]
    return df

def prono2serie(df,main_colname="h_sim",members={'e_pred_01':'p01','e_pred_99':'p99'},series_id=3398):
    df_simulado = df.copy().reset_index()
    # df_simulado['fecha'] = df_simulado['fecha'].dt.tz_localize("America/Argentina/Buenos_Aires") # timezone.localize(df_simulado['fecha'])
    column_mapper = {
        'fecha': 'timestart'
    }
    column_mapper[main_colname] = 'valor'
    df_para_upsert = df_simulado[['fecha',main_colname]].rename(axis=1, mapper=column_mapper,inplace = False).copy()
    # print(df_para_upsert)
    df_para_upsert['qualifier'] = 'main'
    for member in members:
        column_mapper = { 'fecha': 'timestart'}
        column_mapper[member] = "valor"
        df_para_upsert = pd.concat([df_para_upsert, df_simulado[['fecha',member]].rename(axis=1, mapper=column_mapper)], ignore_index=True)
        df_para_upsert['qualifier'] = df_para_upsert['qualifier'].fillna(members[member])
    df_para_upsert['timeend'] = df_para_upsert['timestart']  # .map(lambda a : a.isoformat())
    return {
                'series_table': 'series',
                'series_id': series_id,
                'pronosticos': json.loads(df_para_upsert.to_json(orient='records',date_format='iso'))
           }


def prono2json(series,forecast_date=datetime.now()):
    return {
        'forecast_date': forecast_date.isoformat(), #.tz_localize("America/Argentina/Buenos_Aires").isoformat(), # df_Sim['fecha_emision'].max().isoformat(),
        'series': series
    }

def outputjson(data,outputfile):
    output = open(outputfile,'w')
    output.write(json.dumps(data))
    output.close()

def uploadProno(data,cal_id,responseOutputFile):
    response = requests.post(
        apiLoginParams["url"] + '/sim/calibrados/' + str(cal_id) + '/corridas',
        data=json.dumps(data),
        headers={'Authorization': 'Bearer ' + apiLoginParams["token"], 'Content-type': 'application/json'},
    )
    print("prono upload, response code: " + str(response.status_code))
    print("prono upload, reason: " + response.reason)
    if(response.status_code == 200):
        if(responseOutputFile):
            outresponse = open(responseOutputFile,"w")
            outresponse.write(json.dumps(response.json()))
            outresponse.close()

def uploadPronoSeries(series,cal_id=440,forecast_date=datetime.now(),outputfile=None,responseOutputFile=None):
    data = prono2json(series,forecast_date)
    if(outputfile):
        outputjson(data,outputfile)
        uploadProno(data,cal_id,responseOutputFile)

def outputcsv(df,outputfile):
    csvoutput = open(outputfile,"w")
    csvoutput.write(df.to_csv())
    csvoutput.close()




