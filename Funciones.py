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

from pytz import timezone
localtz = timezone('America/Argentina/Buenos_Aires')

"""# Credenciales de la API"""

with open("Configuracion/config.json") as f:
    config = json.load(f)
apiLoginParams = config["api"]

def readSerie(series_id,timestart=None,timeend=None,tipo="puntual",use_proxy=False):
    params = {}
    if timestart is not None and timeend is not None:
        params = {
            "timestart": timestart if isinstance(timestart,str) else timestart.isoformat(),
            "timeend": timeend if isinstance(timestart,str) else timeend.isoformat()
        }
    response = requests.get("%s/obs/%s/series/%i" % (config["api"]["url"], tipo, series_id),
        params = params,
        headers = {'Authorization': 'Bearer ' + config["api"]["token"]},
        proxies = config["proxy_dict"] if use_proxy else None
    )
    if response.status_code != 200:
        raise Exception("request failed: %s" % response.text)
    json_response = response.json()
    return json_response

def observacionesListToDataFrame(data: list):
    if len(data) == 0:
        raise Exception("empty list")
    data = pd.DataFrame.from_dict(data)
    data.index = data["timestart"].apply(tryParseAndLocalizeDate)
    data.sort_index(inplace=True)
    return data[["valor",]]

def tryParseAndLocalizeDate(date_string,timezone='America/Argentina/Buenos_Aires'):
    date = dateutil.parser.isoparse(date_string) if isinstance(date_string,str) else date_string
    if date.tzinfo is None or date.tzinfo.utcoffset(date) is None:
        try:
            date = pytz.timezone(timezone).localize(date)
        except pytz.exceptions.NonExistentTimeError:
            print("NonexistentTimeError: %s" % str(date))
            return None
    else:
        date = date.astimezone(pytz.timezone(timezone))
    return date

def Consulta_id_corridas(id0_modelo):
    ## Carga Simulados
    response = requests.get(
        'https://alerta.ina.gob.ar/a6/sim/calibrados/'+str(id0_modelo)+'/corridas',
        params={'qualifier':'main','includeProno':False},
        headers={'Authorization': 'Bearer ' + config["api"]["token"]},)
    json_response = response.json()
    return json_response

def cargasim(id_modelo,corrida_id,estacion_id): ## Consulta los pronosticos
    ## Carga Simulados
    response = requests.get(
        'https://alerta.ina.gob.ar/a6/sim/calibrados/'+str(id_modelo)+'/corridas/'+str(corrida_id)+'?estacion_id='+str(estacion_id)+'&includeProno=true',
        headers = {'Authorization': 'Bearer ' + config["api"]["token"]})    
        #params={'qualifier':'main','estacion_id':str(estacion_id),'includeProno':True},

    json_response = response.json()
    
    try:
        df_sim = pd.DataFrame.from_dict(json_response['series'][0]['pronosticos'],orient='columns')
        df_sim = df_sim.rename(columns={'timestart':'fecha','valor':'h_sim'})
        df_sim = df_sim[['fecha','h_sim']]
        df_sim['fecha'] = pd.to_datetime(df_sim['fecha'])
        df_sim['h_sim'] = df_sim['h_sim'].astype(float)

        df_sim = df_sim.sort_values(by='fecha')
        df_sim.set_index(df_sim['fecha'], inplace=True)
        df_sim.index = df_sim.index.tz_convert("America/Argentina/Buenos_Aires")
        #df_sim['fecha'] = df_sim.index
        df_sim['cor_id'] = corrida_id
        df_sim = df_sim.reset_index(drop=True)
        return df_sim
    except:
        print('Son datos para:',corrida_id)
        df_sim = pd.DataFrame()
        return df_sim

def plotFinal(df_obs,df_sim,nameout='productos/plot_final.png',titulo='puerto',ydisplay=1,xytext=(-300,-200),ylim=(-1,2.5),markersize=None,text_xoffset=(-8,-8),obs_label='Nivel Observado',extraObs=None,extraObsLabel='Nivel Observado 2', fecha_emision = None, bandaDeError=('e_pred_01','e_pred_99'),obsLine=False,nombre_estacion="Estación",niveles_alerta={}, cero=0):
    # if nombre_estacion != 'NuevaPalmira':
    df_sim.index = df_sim.index.tz_convert(tz="America/Argentina/Buenos_Aires")
    if not isinstance(df_obs,type(None)):
        df_obs.index = df_obs.index.tz_convert(tz="America/Argentina/Buenos_Aires")
        # print(df_obs.index)
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(1, 1, 1)
    # ax.title('Previsión de niveles a corto plazo en el puerto')
    ax.plot(df_sim.index, df_sim['Y_predic'], '-',color='b',label='Nivel Pronosticado (*)',linewidth=3)
    if not isinstance(df_obs, type(None)):
        ax.plot(df_obs.index, df_obs['h_obs'],'o',color='k',label=obs_label,linewidth=3)
        if obsLine:
            ax.plot(df_obs.index, df_obs['h_obs'],'-',color='k',linewidth=1,markersize=markersize)
    if not isinstance(extraObs,type(None)):
        if nombre_estacion != 'NuevaPalmira': extraObs.index = extraObs.index.tz_convert("America/Argentina/Buenos_Aires")
        ax.plot(extraObs.index, extraObs['h_obs'],'o',color='grey',label=extraObsLabel,linewidth=3,alpha=0.5)
        ax.plot(extraObs.index, extraObs['h_obs'],'-',color='grey',linewidth=1,alpha=0.5)
    ax.plot(df_sim.index, df_sim[bandaDeError[0]],'-',color='k',linewidth=0.5,alpha=0.75,label='_nolegend_')
    ax.plot(df_sim.index, df_sim[bandaDeError[1]],'-',color='k',linewidth=0.5,alpha=0.75,label='_nolegend_')
    ax.fill_between(df_sim.index,df_sim[bandaDeError[0]], df_sim[bandaDeError[1]],alpha=0.1,label='Banda de error')
    # Lineas: 1 , 1.5 y 2 mts
    xmin=df_sim.index.min()
    xmax=df_sim.index.max()
    # Niveles alerta
    if niveles_alerta.get("aguas_bajas"):
        plt.hlines(niveles_alerta["aguas_bajas"], xmin, xmax, colors='y', linestyles='-.', label='Aguas Bajas',linewidth=1.5)
    if niveles_alerta.get("alerta"):
        plt.hlines(niveles_alerta["alerta"], xmin, xmax, colors='y', linestyles='-.', label='Alerta',linewidth=1.5)
    if niveles_alerta.get("evacuacion"):
        plt.hlines(niveles_alerta["evacuacion"], xmin, xmax, colors='r', linestyles='-.', label='Evacuación',linewidth=1.5)
    # fecha emision
    if fecha_emision:
        if fecha_emision.tzinfo is not None and fecha_emision.tzinfo.utcoffset(fecha_emision) is not None:
            ahora = fecha_emision
        else:
            ahora = localtz.localize(fecha_emision)
    elif not isinstance(df_obs, type(None)):
        ahora = df_obs.index.max()
    else: 
        ahora = localtz.localize(datetime.now())
    plt.axvline(x=ahora,color="black", linestyle="--",linewidth=2)#,label='Fecha de emisión')
    bbox = dict(boxstyle="round", fc="0.7")
    arrowprops = dict(
        arrowstyle="->",
        connectionstyle="angle,angleA=0,angleB=90,rad=10")
    offset = 10
    #xycoords='figure pixels',
    xdisplay = ahora + timedelta(days=1.0)
    ax.annotate('Pronóstico a 4 días',
        xy=(xdisplay, ydisplay), xytext=(text_xoffset[0]*offset, -offset), textcoords='offset points',
        bbox=bbox, fontsize=18)#arrowprops=arrowprops
    xdisplay = ahora - timedelta(days=2)
    ax.annotate('Días pasados',
        xy=(xdisplay, ydisplay), xytext=(text_xoffset[1]*offset, -offset), textcoords='offset points',
        bbox=bbox, fontsize=18)
    ax.annotate('Fecha de emisión',
        xy=(ahora, -0.35),fontsize=15, xytext=(ahora+timedelta(days=0.3), -0.30), arrowprops=dict(facecolor='black',shrink=0.05))
    # ax.annotate('(*) Esta previsión surge de aplicar el Modelo Matemático del Delta del PHC-SLH-INA, forzado por el caudal pronosticado del río Paraná \n de acuerdo al SIyAH-INA y por el nivel del Río de la Plata en el arco San Fernando - Nueva Palmira pronosticado por el SHN-SMN. \n Es una herramienta preliminar de pronóstico para utilizar en la emergencia hídrica, que se irá ajustando en el tiempo para \n generar información más confiable.',
    #            xy=(xdisplay, 0), xytext=xytext, textcoords='offset points', fontsize=11)
    fig.subplots_adjust(bottom=0.2,right=0.8)
    plt.figtext(0,0,'          (*) Esta previsión surge de aplicar el Modelo Matemático del Delta del Programa de Hidráulica Computacional (PHC) de la Subgerencia \n          del Laboratorio de Hidráulica (SLH) del Instituto Nacional del Agua (INA), forzado por el caudal pronosticado del río Paraná de acuerdo \n          al Sistema de Información y Alerta Hidrológico (SIyAH-INA) y por el nivel del Río de la Plata en el arco San Fernando - Nueva Palmira \n          pronosticado por el Servicio de Hidrografía Naval (SHN) y el Servicio Meteorológico Nacional (SMN). Es una herramienta preliminar \n          de pronóstico para utilizar en la emergencia hídrica, que se irá ajustando en el tiempo para generar información más confiable. \n \n',fontsize=12,ha="left")
    if cero is not None:
        plt.figtext(0,0,'          (**) El cero de la escala de ' + nombre_estacion + ' corresponde a ' + str(cero+0.53) +' mMOP / '+ str(cero) +' mIGN \n',fontsize=12,ha="left")
    if ylim:
        ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlim(xmin,xmax)
    ax.tick_params(labeltop=False, labelright=True)
    plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
    plt.tick_params(axis='both', labelsize=16)
    plt.xlabel('Fecha', size=16)
    plt.ylabel('Nivel [m] Referido al cero local (**)', size=20)
    plt.legend(prop={'size':18},ncol=1 )#loc=2,
    plt.title('Previsión de niveles a corto plazo en ' + nombre_estacion,fontsize=20)
    #### TABLA
    h_resumne = [0,6,12,18]
    df_prono = df_sim[df_sim.index > ahora ].copy()
    df_prono['Hora'] = df_prono.index.hour
    df_prono['Dia'] = df_prono.index.day
    df_prono = df_prono[df_prono['Hora'].isin(h_resumne)].copy()
    #print(df_prono)
    df_prono['Y_predic'] = df_prono['Y_predic'].round(2)
    df_prono['Hora'] = df_prono['Hora'].astype(str)
    df_prono['Hora'] = df_prono['Hora'].replace('0', '00')
    df_prono['Hora'] = df_prono['Hora'].replace('6', '06')
    df_prono['Dia'] = df_prono['Dia'].astype(str)
    df_prono['Fechap'] = df_prono['Dia']+' '+df_prono['Hora']+'hrs'
    df_prono = df_prono[['Fechap','Y_predic',]]
    #print(df_prono)
    cell_text = []
    for row in range(len(df_prono)):
        cell_text.append(df_prono.iloc[row])
        print(cell_text)
    columns = ('Fecha','Nivel',)
    table = plt.table(cellText=cell_text,
                      colLabels=columns,
                      bbox = (1.08, 0, 0.2, 0.5))
    table.set_fontsize(12)
    #table.scale(2.5, 2.5)  # may help
    date_form = DateFormatter("%H hrs \n %d-%b",tz=df_sim.index.tz)
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_minor_locator(mdates.HourLocator((3,9,15,21,)))
    ## FRANJAS VERTICALES
    start_0hrs = df_sim.index.min().date()
    end_0hrs = (df_sim.index.max() + timedelta(hours=12)).date()
    list0hrs = pd.date_range(start_0hrs,end_0hrs)
    i = 1
    while i < len(list0hrs):
        ax.axvspan(list0hrs[i-1] + timedelta(hours=3), list0hrs[i] + timedelta(hours=3), alpha=0.1, color='grey')
        i=i+2
    plt.savefig(nameout, format='png')
    plt.close()
    if nombre_estacion != 'NuevaPalmira':
      df_sim.index = df_sim.index.tz_convert(tz="UTC")
      if not isinstance(df_obs,type(None)):
          df_obs.index = df_obs.index.tz_convert(tz="UTC")
      if not isinstance(extraObs,type(None)):
          extraObs.index = extraObs.index.tz_convert("UTC")

def outputcsv(df,outputfile):
    csvoutput = open(outputfile,"w")
    csvoutput.write(df.to_csv())
    csvoutput.close()

def prono2serie(df,main_colname="h_sim",members={'e_pred_01':'p01','e_pred_99':'p99'},series_id=3398):
    df_simulado = df.copy().reset_index()
    # df_simulado['fecha'] = df_simulado['fecha'].dt.tz_localize("America/Argentina/Buenos_Aires") # timezone.localize(df_simulado['fecha'])
    column_mapper = {
        'fecha': 'timestart'
    }
    column_mapper[main_colname] = 'valor'
    df_para_upsert = df_simulado[['fecha',main_colname]].rename(axis=1, mapper=column_mapper,inplace = False)
    # print(df_para_upsert)
    df_para_upsert['qualifier'] = 'main'
    for member in members:
        column_mapper = { 'fecha': 'timestart'}
        column_mapper[member] = "valor"
        df_para_upsert = pd.concat([df_para_upsert, df_simulado[['fecha',member]].rename(axis=1, mapper=column_mapper)], ignore_index=True)
        df_para_upsert['qualifier'].fillna(value=members[member],inplace=True)
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
