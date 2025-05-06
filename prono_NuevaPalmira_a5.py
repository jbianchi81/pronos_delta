import json
import datetime
import requests
from a5client import Crud

import pandas as pd
# import numpy as np
from datetime import timedelta,datetime

import dateutil
import pytz
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from pytz import timezone
localtz = timezone('America/Argentina/Buenos_Aires')


''' import sys, getopt
    #import seaborn as sns
    from dateutil import parser
    import logging
    import psycopg2
'''

"""# Credenciales de la API"""

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir,"config.json")
with open(config_path) as f:
    config = json.load(f)
apiLoginParams = config["api"]

client = Crud(apiLoginParams["url"], apiLoginParams["token"])

def readSerie(series_id,timestart=None,timeend=None,tipo="puntual",use_proxy=False):
    params = {}
    if timestart is not None and timeend is not None:
        params = {
            "timestart": timestart if isinstance(timestart,str) else timestart.isoformat(),
            "timeend": timeend if isinstance(timestart,str) else timeend.isoformat()
        }
    response = requests.get("%s/obs/%s/series/%i" % (apiLoginParams["url"], tipo, series_id),
        params = params,
        headers = {'Authorization': 'Bearer ' + apiLoginParams["token"]},
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

def Consulta_id_corridas(id0):
    ## Carga Simulados
    response = requests.get(
        '%s/sim/calibrados/%i/corridas' % (apiLoginParams["url"], id0),
        params={'qualifier':'main','includeProno':False},
        headers={'Authorization': 'Bearer %s' % apiLoginParams["token"]},)
    json_response = response.json()
    return json_response

def cargasim(id_modelo,series_id, qualifier, forecast_timestart : datetime = datetime.now() - timedelta(days=7)): ## Consulta los pronosticos
    ## Carga Simulados
    json_response = client.readSeriePronoConcat(id_modelo, series_id, qualifier = qualifier, forecast_timestart = forecast_timestart)
    
    df_sim = pd.DataFrame.from_dict(json_response['pronosticos'],orient='columns')
    df_sim = df_sim.rename(columns={'timestart':'fecha','valor':'h_sim'})
    df_sim = df_sim[['fecha','h_sim']]
    df_sim['fecha'] = pd.to_datetime(df_sim['fecha'])
    df_sim['h_sim'] = df_sim['h_sim'].astype(float)

    df_sim = df_sim.sort_values(by='fecha')
    df_sim = df_sim.set_index(df_sim['fecha'])
    df_sim.index = df_sim.index.tz_convert("America/Argentina/Buenos_Aires")
    df_sim = df_sim.drop('fecha',axis=1)
    #df_sim['fecha'] = df_sim.index
    # df_sim = df_sim.reset_index(drop=True)
    
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
    ax.annotate('Pronóstico a 3 días',
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
    print(df_prono)
    df_prono['Y_predic'] = df_prono['Y_predic'].round(2)
    df_prono['Hora'] = df_prono['Hora'].astype(str)
    df_prono['Hora'] = df_prono['Hora'].replace('0', '00')
    df_prono['Hora'] = df_prono['Hora'].replace('6', '06')
    df_prono['Dia'] = df_prono['Dia'].astype(str)
    df_prono['Fechap'] = df_prono['Dia']+' '+df_prono['Hora']+'hrs'
    df_prono = df_prono[['Fechap','Y_predic',]]
    print(df_prono)
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

def corrigeNuevaPalmira(plots=False,upload=True,output_csv=True):
    ## Consulta id de las corridas
    id_modelo = 707 # 308 
    estacion_id = 1843
    series_id_sim = 6071
    qualifier = "median"

    ahora = datetime.now(pytz.timezone('America/Argentina/Buenos_Aires')).replace(hour=0, minute=0, second=0, microsecond=0)
    fecha_pronos_calib = ahora - timedelta(days=7) # Corrige con los ultimos x días

    ##################### Conulta Simulado #####################

    # 1 - Consulta ID Corridas
    json_res = Consulta_id_corridas(id_modelo)
    print('Cantidad de corridas: ',len(json_res))
    
    lst_corridas = []
    lst_pronoday = []
    for corridas in range(len(json_res)):
        lst_corridas.append(json_res[corridas]['id'])
        lst_pronoday.append(json_res[corridas]['forecast_date'])
    df_id = pd.DataFrame(lst_pronoday, index =lst_corridas,columns=['forecast_date',])
    df_id['forecast_date'] = pd.to_datetime(df_id['forecast_date'])
    df_id = df_id[df_id['forecast_date']>fecha_pronos_calib]            # Filtra las corridas viejas

    # 2 - Consulta Corridas
    df_base_unico = cargasim(id_modelo, series_id_sim, qualifier)

    # for index, row in df_id.iterrows():                         # Carga las series simuladas usnado los ID de las corridas
    #     df_corr_i = cargasim(id_modelo,index,estacion_id)
    #     if df_corr_i.empty: continue
    #     df_corr_i.set_index(df_corr_i['fecha'], inplace=True)
    #     # df_corr_i.index = df_corr_i.index.tz_convert(None) - timedelta(hours=3)
    #     # df_corr_i['fecha'] = df_corr_i.index
    #     # df_corr_i.reset_index(drop=True)
    #     #df_base = pd.concat([df_base, df_corr_i], ignore_index=True)

    #     df_base_unico.update(df_corr_i)                         # Actualiza los pronos 
    #     df_base_unico = pd.concat([df_base_unico, df_corr_i[~df_corr_i.index.isin(df_base_unico.index)]])   # guarda los nuevos

    ##################### Conulta Observado #####################
    f_inicio = df_base_unico.index.min()
    f_fin = df_base_unico.index.max()

    #Est_Uruguay = {3280:'NuevaPalmira',80:'Colon'}

    df_obs_npal = readSerie(3280,f_inicio,f_fin)
    df_obs_npal = observacionesListToDataFrame(df_obs_npal["observaciones"])
    df_obs_npal["series_id"] = 3280
    df_obs_npal = df_obs_npal.rename(columns={'valor':'h_obs'})

    '''
    df_obs_col = readSerie(80,f_inicio,f_fin)
    df_obs_col = observacionesListToDataFrame(df_obs_col["observaciones"])
    df_obs_col["series_id"] = 80

    df_obs = pd.concat([df_obs_npal,df_obs_col])
    df_Obs = pd.pivot_table(df_obs, values='h_obs', index=['fecha'], columns=['series_id'])

    print("Resample horario")
    df_Obs_H = df_Obs.resample('H').mean()

    # Nombres a los calumnas
    df_Obs_H = df_Obs_H.rename(columns=Est_Uruguay)

    # tz-aware to tz-naive
    df_Obs_H.index = pd.to_datetime(df_Obs_H.index).tz_convert('-03:00').tz_localize(None)
    '''    
    ### PRIMERA CORRECCION
    df_base_unico.index = df_base_unico.index + timedelta(hours=2)

    ## Union 
    # #Crea DF base:
    # indexUnico = pd.date_range(start=df_Sim0['fecha'].min(), end=df_Sim0['fecha'].max(), freq='H')	    #Fechas desde f_inicio a f_fin con un paso de 5 minutos
    # df_base = pd.DataFrame(index = indexUnico)								#Crea el Df con indexUnico
    # df_base.index.rename('fecha', inplace=True)							    #Cambia nombre incide por Fecha
    # df_base.index = df_base.index.round("H")

    # Une obs y sim
    df_base_unico = df_base_unico.join(df_obs_npal, how = 'left')
    df_base_unico = df_base_unico.dropna().copy()    
    # print(df_base_unico.head())

    # df_base['CUru_dly'] = df_base['Colon'].shift(24*4)
    # df_base['CUru_dly'] = df_base['CUru_dly'].interpolate(limit_direction="forward")

    # df_base['h_met_Mavg'] = df_base['A01'].rolling(24, min_periods=1).mean()

    # del df_base['Colon']
    # df_base = df_base.dropna()

    print('Cantidad de datos de entrenamiento:',len(df_base_unico))
    #print(df_base.tail(5))
    ###########################

    ## Modelo
    print(df_base_unico.head())

    train = df_base_unico[:].copy()
    var_obj = 'h_obs'
    covariav = ['h_sim',]#['altura_astro','A01','h_met_Mavg','CUru_dly']
    lr = linear_model.LinearRegression()
    X_train = train[covariav]
    Y_train = train[var_obj]
    lr.fit(X_train,Y_train)

    # Create the test features dataset (X_test) which will be used to make the predictions.
    X_test = train[covariav].values 
    # The labels of the model
    Y_test = train[var_obj].values
    Y_predictions = lr.predict(X_test)
    train['Y_predictions'] = Y_predictions

    # The coefficients
    print('Coefficients B0: \n', lr.intercept_)
    print('Coefficients: \n', lr.coef_)

    # The mean squared error
    mse = mean_squared_error(Y_test, Y_predictions)
    print('Mean squared error: %.5f' % mse)
    # The coefficient of determination: 1 is perfect prediction
    coefDet = r2_score(Y_test, Y_predictions)
    print('r2_score: %.5f' % coefDet)
    train['Error_pred'] =  train['Y_predictions']  - train[var_obj]

    quant_Err = train['Error_pred'].quantile([.01,.05,.95,.99])

    #############   Pronóstico  #############
    
    # Cosulta ultimo prono sin corregir
    index = df_id.index.max()
    fecha_emision = df_id.loc[index,'forecast_date']


    df_last_prono = cargasim(id_modelo, series_id_sim, qualifier)
    
    # pd.DataFrame(columns=['h_sim','cor_id'])
    # df_id_last = df_id[-4:]

    # for index, row in df_id_last.iterrows():                         # Carga las series simuladas usnado los ID de las corridas
    #     df_corr_i = cargasim(id_modelo,index,estacion_id)
    #     if df_corr_i.empty: continue
    #     df_corr_i.set_index(df_corr_i['fecha'], inplace=True)
    #     # df_corr_i.index = df_corr_i.index.tz_convert(None) - timedelta(hours=3)
    #     # df_corr_i['fecha'] = df_corr_i.index
    #     # df_corr_i.reset_index(drop=True)
    #     #df_base = pd.concat([df_base, df_corr_i], ignore_index=True)

    #     df_last_prono.update(df_corr_i)                         # Actualiza los pronos 
    #     df_last_prono = pd.concat([df_last_prono, df_corr_i[~df_corr_i.index.isin(df_last_prono.index)]])   # guarda los nuevos
    
    #print(df_last_prono.cor_id.unique())
    df_last_prono.index = df_last_prono.index + timedelta(hours=2)
    df_last_prono = df_last_prono[['h_sim',]]

    # print(df_obs_npal.head())
    # print(df_last_prono.head())

    covariav = ['h_sim',]
    prediccion = lr.predict(df_last_prono[covariav].values)
    df_last_prono['Y_predic'] = prediccion
    
    # fig = plt.figure(figsize=(15, 8))
    # ax = fig.add_subplot(1, 1, 1)
    
    # ax.plot(df_base_unico.index, df_base_unico['h_obs'],'.',label='Nueva Palmira')    
    # # ax.plot(df_obs_col.index, df_obs_col['valor'],'-',label='Colon')
    # ax.plot(df_base_unico.index, df_base_unico['h_sim'],label='h_sim')
    # ax.plot(df_last_prono.index, df_last_prono['h_sim'],label='Last Prono')
    # ax.plot(df_last_prono.index, df_last_prono['Y_predic'],label='Last Prono Corregido')

    # plt.grid(True,axis='y', which='both', color='0.75', linestyle='-.',linewidth=0.3)
    # plt.tick_params(axis='y', labelsize=14)
    # plt.tick_params(axis='x', labelsize=14,rotation=20)
    # plt.xlabel('Mes', size=18)
    # plt.ylabel('Nivel [m]', size=18)# 'Caudal [m'+r'$^3$'+'/s]'
    # plt.legend(prop={'size':16},loc=0,ncol=1)
    # plt.show()
    # plt.close()
        
    df_last_prono['e_pred_01'] = df_last_prono['Y_predic'] + quant_Err[0.01]
    df_last_prono['e_pred_99'] = df_last_prono['Y_predic'] + quant_Err[0.99]
    
    df_Obs = df_obs_npal[['h_obs',]]    # df_Obs = df_Obs.rename(columns={'NuevaPalmira':'h_obs'})
    df_Sim = df_last_prono[['Y_predic','e_pred_01','e_pred_99']].copy()
    df_Sim.index = df_Sim.index.to_pydatetime()
    df_Sim['fecha'] = df_Sim.index

    # PLOT FINAL
    plotFinal(df_Obs,df_Sim,ydisplay=3.4,text_xoffset=(0.5,0.5),ylim=(-0.5,3.5),nameout='productos/Prono_NPalmira.png',nombre_estacion="NuevaPalmira",cero=None,fecha_emision=fecha_emision)

    if output_csv:
        outputcsv(df_Sim,"productos/prono_NuevaPalmira.csv")

    series = [
        prono2serie(df_Sim,main_colname="Y_predic",members={'e_pred_01':'p01','e_pred_99':'p99'},series_id=26203)
    ]

    ## UPLOAD PRONOSTICO
    if upload:
        uploadPronoSeries(series,cal_id=433,forecast_date=fecha_emision,outputfile="productos/prono_NuevaPalmira.json",responseOutputFile="productos/pronoresponse.json")
    else:
        data = prono2json(series,forecast_date=fecha_emision)
        outputjson(data,"productos/prono_NuevaPalmira.json")

if __name__ == "__main__":
    corrigeNuevaPalmira(plots=False,upload=True,output_csv=True)




