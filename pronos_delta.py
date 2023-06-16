import sys, getopt
import requests, psycopg2
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta,datetime
import matplotlib.pyplot as plt
#import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import json
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from dateutil import parser
from pytz import timezone
localtz = timezone('America/Argentina/Buenos_Aires')

"""# Credenciales de la API"""

with open("config.json") as f:
    config = json.load(f)

apiLoginParams = config["api"]

"""### FUNCIONES ###"""

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
        df_para_upsert = df_para_upsert.append(df_simulado[['fecha',member]].rename(axis=1, mapper=column_mapper), ignore_index=True)
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


def getLastProno(cal_id=288,filter={}):
    response = requests.get(
        apiLoginParams["url"] + '/sim/calibrados/' + str(cal_id) + '/corridas/last',
        params = filter,
        headers = {'Authorization': 'Bearer ' + apiLoginParams["token"]},
    )
    json_response = response.json()
    df_sim = pd.DataFrame.from_dict(json_response['series'][0]['pronosticos'],orient='columns')
    df_sim = df_sim.rename(columns={0:'fecha',1:'fecha2',2:'h_sim',3:'main'})
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

def plotObsVsSim(df):
    fig = plt.figure(figsize=(17, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df.index, df['h_obs'], 'b-',label='Observado')
    ax.plot(df.index, df['h_sim'], 'r-',label='Simulado')
    plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
    plt.tick_params(axis='both', labelsize=18)
    plt.xlabel('Fecha', size=18)
    plt.ylabel('Nivel [m]', size=18)
    plt.legend(prop={'size':20},loc=0)
    plt.tight_layout()
    plt.show()

def plotObsVsSimVsPred(df,train):
        fig = plt.figure(figsize=(17, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(df.index, df['h_sim'], 'b-',label='Simulado')
        ax.plot(df.index, df['h_obs'], 'r-',label='Observado')
        ax.plot(train.index, train['Y_predictions'], 'k-',label='Ajuste RL')
        plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
        plt.tick_params(axis='both', labelsize=18)
        plt.xlabel('Fecha', size=18)
        plt.ylabel('Nivel [m]', size=18)
        plt.legend(prop={'size':20},loc=0)
        plt.tight_layout()
        plt.show()

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
    plt.legend(prop={'size':18},loc=2,ncol=1 )
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


def corrigeZarate(plots=False, upload=True, output_csv=False):
    """# Carga los datos"""

    ## Carga Simulados en Zarate
    df_Zarate_sim, fecha_emision = getLastProno(288,{'estacion_id': '5907', 'var_id': 2})
    print(fecha_emision)
    ## Carga Observados

    f_inicio = df_Zarate_sim.index.min()
    f_fin = df_Zarate_sim.index.max()
    Estaciones = {5907:'Zarate',}
    # Estaciones a consultar
    idDest = 5907
    idSerieOrigen = 29437
    df_Zarate_Obs = getObs(idSerieOrigen,f_inicio,f_fin)

    # Elimina saltos
    df_Zarate_Obs = eliminaSaltos(df_Zarate_Obs,0.25)
    # sns.histplot(data=df_Zarate_Obs, x="dif_F")
    # plt.show()
    # df_Zarate_Obs.set_index(df_Zarate_Obs['fecha'].dt.tz_convert(None), inplace=True)
    # del df_Zarate_Obs['fecha']

    ###### Correccion 1 el simulado en zarate adelanta de 1 hora 
    df_Zarate_sim.index = df_Zarate_sim.index + timedelta(hours=1) # - timedelta(hours=2)

    ## Union
    df_Zarate = df_Zarate_sim.join(df_Zarate_Obs, how = 'outer')
    df_Zarate['h_sim'] = df_Zarate['h_sim'].interpolate(method='linear',limit=4)
    #df_Zarate['h_sim_Mavg'] = df_Zarate['h_sim'].rolling(4, min_periods=1).mean()

    ##Plot
    if plots:
        plotObsVsSim(df_Zarate)

    """# Modelo RL"""

    train0 = df_Zarate.copy()
    train0['Error0'] = train0['h_sim'] - train0['h_obs']
    train0['dif_F1'] = train0['h_sim'].diff(periods=1).abs()
    train0['dif_F2'] = train0['h_sim'].diff(periods=2).abs()
    train0['dif_F3'] = train0['h_sim'].diff(periods=3).abs()

    train0['dif_B1'] = train0['h_sim'].diff(periods=-1).abs()
    train0['dif_B2'] = train0['h_sim'].diff(periods=-2).abs()
    train0['dif_B3'] = train0['h_sim'].diff(periods=-3).abs()

    train0['dif_F'] = train0['dif_F1'].rolling(3, min_periods=1).mean()
    train0['dif_B'] = train0['dif_B1'].rolling(3, min_periods=1).mean()

    ## Modelo
    train = train0[:].copy()
    train = train.dropna()

    var_obj = 'h_obs'
    #var_obj = 'Error0'
    covariav = ['h_sim','dif_F1','dif_B1','dif_F2','dif_B2','dif_F3','dif_B3']
    covariav = ['h_sim','dif_F1','dif_B1','dif_F2','dif_B2']
    covariav = ['h_sim','dif_F','dif_B']
    covariav = ['h_sim',]

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
    #train['Y_predictions'] = train['h_sim'] - train['Y_predictions'] 

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
    quant_Err = train['Error_pred'].quantile([.001,.05,.95,.999])

    ## Plot
    if plots:
        plotObsVsSimVsPred(df_Zarate,train)

    """# Predicción  en Zarate

    Predice sobre todos los datos de Zarate.  Se vuelve a calcular porque al entrenar el modelo hay datos que se eliminan por no tener un observado para comparar. Al predecir si se tiene estos datos en cuenta y se completa la serie.
    """
    df_Zarate_sim2 = df_Zarate_sim.copy()
    X_input = df_Zarate_sim2[['h_sim',]].values

    # Prediccion
    df_Zarate_sim2['Y_predic'] = lr.predict(X_input)

    horas_plot = 24*10
    df_Zarate_sim2 = df_Zarate_sim2[-horas_plot:]
    # df_simulado['e_pred_05'] = df_simulado['Y_predic'] + quant_Err[0.05]
    # df_simulado['e_pred_95'] = df_simulado['Y_predic'] + quant_Err[0.95]
    df_Zarate_sim2['e_pred_01'] = df_Zarate_sim2['Y_predic'] + quant_Err[0.001]
    df_Zarate_sim2['e_pred_99'] = df_Zarate_sim2['Y_predic'] + quant_Err[0.999]

    # SERIES 2 upload #

    series = [
        prono2serie(df_Zarate_sim2,main_colname="Y_predic",members={'e_pred_01':'p01','e_pred_99':'p99'},series_id=29534)
    ]

    # PLOT FINAL
    plotFinal(df_Zarate_Obs,df_Zarate_sim2,'productos/Prono_Zarate.png',ydisplay=-0.75,text_xoffset=(-2,-8),xytext=(-320,-200),ylim=(-1,2.5),nombre_estacion="Zárate",niveles_alerta={"aguas_bajas":0.3},cero=0.42,fecha_emision=fecha_emision) # ,"alerta":2,"evacuacion":2.2

    ## UPLOAD PRONOSTICO
    if upload:
        uploadPronoSeries(series,cal_id=440,forecast_date=fecha_emision,outputfile="productos/prono_zarate.json",responseOutputFile="productos/pronoresponse.json")
    else:
        data = prono2json(series,forecast_date=fecha_emision)
        outputjson(data,"productos/prono_zarate.json")
    
    if output_csv:
        outputcsv(df_Zarate_sim2,"productos/Prono_Zarate.csv")


def corrigeAtucha(plots=False, upload=True, output_csv=False):
    """# Carga los datos"""

    ## Carga Simulados en Atucha
    df_Atucha_sim, fecha_emision = getLastProno(288,{'estacion_id': '151', 'var_id': 2})
    print(fecha_emision)
    ## Carga Observados

    f_inicio = df_Atucha_sim.index.min()
    f_fin = df_Atucha_sim.index.max()
    Estaciones = {151:'Atucha',}
    # Estaciones a consultar
    idDest = 151
    idSerieOrigen = 151 
    df_Atucha_Obs = getObs(idSerieOrigen,f_inicio,f_fin)

    # Elimina saltos
    df_Atucha_Obs = eliminaSaltos(df_Atucha_Obs,0.25)
    # sns.histplot(data=df_Atucha_Obs, x="dif_F")
    # plt.show()
    # df_Atucha_Obs.set_index(df_Atucha_Obs['fecha'].dt.tz_convert(None), inplace=True)
    # del df_Atucha_Obs['fecha']

    ###### Correccion 1 el simulado en Atucha adelanta 1 hora
    df_Atucha_sim.index = df_Atucha_sim.index + timedelta(hours=1) # - timedelta(hours=2)

    ## Union
    df_Atucha = df_Atucha_sim.join(df_Atucha_Obs, how = 'outer')
    df_Atucha['h_sim'] = df_Atucha['h_sim'].interpolate(method='linear',limit=4)
    #df_Atucha['h_sim_Mavg'] = df_Atucha['h_sim'].rolling(4, min_periods=1).mean()

    ##Plot
    if plots:
        plotObsVsSim(df_Atucha)

    """# Modelo RL"""

    train0 = df_Atucha.copy()
    train0['Error0'] = train0['h_sim'] - train0['h_obs']
    train0['dif_F1'] = train0['h_sim'].diff(periods=1).abs()
    train0['dif_F2'] = train0['h_sim'].diff(periods=2).abs()
    train0['dif_F3'] = train0['h_sim'].diff(periods=3).abs()

    train0['dif_B1'] = train0['h_sim'].diff(periods=-1).abs()
    train0['dif_B2'] = train0['h_sim'].diff(periods=-2).abs()
    train0['dif_B3'] = train0['h_sim'].diff(periods=-3).abs()

    train0['dif_F'] = train0['dif_F1'].rolling(3, min_periods=1).mean()
    train0['dif_B'] = train0['dif_B1'].rolling(3, min_periods=1).mean()

    ## Modelo
    train = train0[:].copy()
    train = train.dropna()

    var_obj = 'h_obs'
    #var_obj = 'Error0'
    covariav = ['h_sim','dif_F1','dif_B1','dif_F2','dif_B2','dif_F3','dif_B3']
    covariav = ['h_sim','dif_F1','dif_B1','dif_F2','dif_B2']
    covariav = ['h_sim','dif_F','dif_B']
    covariav = ['h_sim',]

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
    #train['Y_predictions'] = train['h_sim'] - train['Y_predictions'] 

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
    quant_Err = train['Error_pred'].quantile([.001,.05,.95,.999])

    ## Plot
    if plots:
        plotObsVsSimVsPred(df_Atucha,train)

    """# Predicción  en Atucha

    Predice sobre todos los datos de Atucha.  Se vuelve a calcular porque al entrenar el modelo hay datos que se eliminan por no tener un observado para comparar. Al predecir si se tiene estos datos en cuenta y se completa la serie.
    """
    df_Atucha_sim2 = df_Atucha_sim.copy()
    X_input = df_Atucha_sim2[['h_sim',]].values

    # Prediccion
    df_Atucha_sim2['Y_predic'] = lr.predict(X_input)

    horas_plot = 24*10
    df_Atucha_sim2 = df_Atucha_sim2[-horas_plot:]
    # df_simulado['e_pred_05'] = df_simulado['Y_predic'] + quant_Err[0.05]
    # df_simulado['e_pred_95'] = df_simulado['Y_predic'] + quant_Err[0.95]
    df_Atucha_sim2['e_pred_01'] = df_Atucha_sim2['Y_predic'] + quant_Err[0.001]
    df_Atucha_sim2['e_pred_99'] = df_Atucha_sim2['Y_predic'] + quant_Err[0.999]

    # SERIES 2 upload #

    series = [
        prono2serie(df_Atucha_sim2,main_colname="Y_predic",members={'e_pred_01':'p01','e_pred_99':'p99'},series_id=3403)
    ]

    # PLOT FINAL
    plotFinal(df_Atucha_Obs,df_Atucha_sim2,'productos_res/Prono_Atucha.png',ydisplay=0,xytext=(-420,-120),ylim=(-0.5,3.5),nombre_estacion="Atucha",fecha_emision=fecha_emision,cero=-0.53)
    plotFinal(df_Atucha_Obs,df_Atucha_sim2,nameout='productos/Prono_Lima.png',ydisplay=0,text_xoffset=(-2,-5),xytext=(-320,-200),ylim=(-0.5,3.5),fecha_emision=fecha_emision,nombre_estacion="Lima",cero=-0.53)

    if output_csv:
        outputcsv(df_Atucha_sim2,"productos_res/Prono_Atucha.csv")






    """# Prediccion en CAMPANA ##############################

    Carga los datos desde la api de Alerta
    """

    df_Campana_sim, fecha_emision = getLastProno(288,{'estacion_id': '41', 'var_id': 2})

    ###### Correccion 1 el simulado en campana adelanta 1 hora
    df_Campana_sim.index = df_Campana_sim.index + timedelta(hours=1) # - timedelta(hours=2)

    ## Carga Observados
    f_inicio = df_Campana_sim.index.min()
    f_fin = df_Campana_sim.index.max()
    Estaciones = {41:'Campana',}
    idDest = 41
    idSerieOrigen = 41
    df_Campana_Obs = getObs(idSerieOrigen,f_inicio,f_fin)

    X_input = df_Campana_sim[['h_sim',]].values

    # Prediccion
    df_Campana_sim['Y_predic'] = lr.predict(X_input)

    horas_plot = 24*10
    df_Campana_sim = df_Campana_sim[-horas_plot:]
    # df_simulado['e_pred_05'] = df_simulado['Y_predic'] + quant_Err[0.05]
    # df_simulado['e_pred_95'] = df_simulado['Y_predic'] + quant_Err[0.95]
    df_Campana_sim['e_pred_01'] = df_Campana_sim['Y_predic'] + quant_Err[0.001]
    df_Campana_sim['e_pred_99'] = df_Campana_sim['Y_predic'] + quant_Err[0.999]


    series.append(prono2serie(df_Campana_sim,main_colname="Y_predic",members={'e_pred_01':'p01','e_pred_99':'p99'},series_id=3405))

    # PLOT FINAL
    plotFinal(df_Campana_Obs,df_Campana_sim,nameout='productos/Prono_Campana.png',markersize=10,ydisplay=-0.75,text_xoffset=(-2,-5),xytext=(-320,-200),ylim=(-1.,2),obsLine=False,nombre_estacion="Campana", cero=0.03, fecha_emision=fecha_emision)

    if output_csv:
        outputcsv(df_Campana_sim,"productos/Prono_Campana.csv")
    

    """# Prediccion en ESCOBAR ###############################################

    Carga los datos desde la api de Alerta
    """
    df_Escobar_sim, fecha_emision = getLastProno(288,{'estacion_id': '42', 'var_id': 2})

    ###### Correccion 1 el simulado en zarate adelanta 2 horas 
    df_Escobar_sim.index = df_Escobar_sim.index + timedelta(hours=1) # - timedelta(hours=2)

    ## Carga Observados
    f_inicio = df_Escobar_sim.index.min()
    f_fin = df_Escobar_sim.index.max()
    Estaciones = {42:'Escobar',}
    idDest = 42
    seriesIdOrigen = 42
    df_Escobar_Obs = getObs(seriesIdOrigen,f_inicio,f_fin)

    X_input = df_Escobar_sim[['h_sim',]].values

    # Prediccion
    df_Escobar_sim['Y_predic'] = lr.predict(X_input)-0.2

    horas_plot = 24*10
    df_Escobar_sim = df_Escobar_sim[-horas_plot:]
    # df_simulado['e_pred_05'] = df_simulado['Y_predic'] + quant_Err[0.05]
    # df_simulado['e_pred_95'] = df_simulado['Y_predic'] + quant_Err[0.95]
    df_Escobar_sim['e_pred_01'] = df_Escobar_sim['Y_predic'] + quant_Err[0.001]
    df_Escobar_sim['e_pred_99'] = df_Escobar_sim['Y_predic'] + quant_Err[0.999]


    # 2series

    series.append(prono2serie(df_Escobar_sim,main_colname="Y_predic",members={'e_pred_01':'p01','e_pred_99':'p99'},series_id=3398))

    # PLOT FINAL
    plotFinal(df_Escobar_Obs,df_Escobar_sim,nameout='productos/Prono_Escobar.png',markersize=10,ydisplay=-0.75,text_xoffset=(-2,-5),xytext=(-320,-200),ylim=(-1.,3),obsLine=False,nombre_estacion="Escobar",fecha_emision=fecha_emision)
    
    ## UPLOAD PRONOSTICO
    if upload:
        uploadPronoSeries(series,cal_id=446,forecast_date=fecha_emision,outputfile="productos/prono_atucha.json",responseOutputFile="productos/pronoresponse.json")
    else:
        data = prono2json(series,forecast_date=fecha_emision)
        outputjson(data,"productos/prono_atucha.json")

    if output_csv:
        outputcsv(df_Escobar_sim,"productos/Prono_Escobar.csv")


def corrigeRosario(plots=False,upload=True, output_csv=False):
    ## Carga Simulados en Rosario
    df_Rosario_sim, fecha_emision = getLastProno(288,{'estacion_id': '5893','var_id':2})

    ## Carga Observados
    f_inicio = df_Rosario_sim.index.min()
    f_fin = df_Rosario_sim.index.max()
    Estaciones = {5893:'Rosario',}
    df_Rosario_Obs = getObs(29435,f_inicio,f_fin)

    # Elimina saltos
    df_Rosario_Obs=eliminaSaltos(df_Rosario_Obs,0.25)

    ###### Correccion 1 el simulado en Rosario adelanta 1 hora 
    df_Rosario_sim.index = df_Rosario_sim.index + timedelta(hours=1) # - timedelta(hours=2)

    ## Union
    df_Rosario = df_Rosario_sim.join(df_Rosario_Obs, how = 'outer')
    df_Rosario['h_sim'] = df_Rosario['h_sim'].interpolate(method='linear',limit=4)
    #df_Rosario['h_sim_Mavg'] = df_Rosario['h_sim'].rolling(4, min_periods=1).mean()

    if plots:
        plotObsVsSim(df_Rosario)

    """# Modelo RL"""

    train0 = df_Rosario.copy()
    train0['Error0'] = train0['h_sim'] - train0['h_obs']
    train0['dif_F1'] = train0['h_sim'].diff(periods=1).abs()
    train0['dif_F2'] = train0['h_sim'].diff(periods=2).abs()
    train0['dif_F3'] = train0['h_sim'].diff(periods=3).abs()

    train0['dif_B1'] = train0['h_sim'].diff(periods=-1).abs()
    train0['dif_B2'] = train0['h_sim'].diff(periods=-2).abs()
    train0['dif_B3'] = train0['h_sim'].diff(periods=-3).abs()

    train0['dif_F'] = train0['dif_F1'].rolling(3, min_periods=1).mean()
    train0['dif_B'] = train0['dif_B1'].rolling(3, min_periods=1).mean()

    
    ## Modelo
    train = train0[:].copy()
    train = train.dropna()

    var_obj = 'h_obs'
    #var_obj = 'Error0'
    covariav = ['h_sim','dif_F1','dif_B1','dif_F2','dif_B2','dif_F3','dif_B3']
    covariav = ['h_sim','dif_F1','dif_B1','dif_F2','dif_B2']
    covariav = ['h_sim','dif_F','dif_B']
    covariav = ['h_sim',]

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
    #train['Y_predictions'] = train['h_sim'] - train['Y_predictions'] 

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
    quant_Err = train['Error_pred'].quantile([.001,.05,.95,.999])

    if plots:
        plotObsVsSimVsPred(df_Rosario,train)
    
    """# Predicción  en Rosario

    Predice sobre todos los datos de Rosario.  Se vuelve a calcular porque al entrenar el modelo hay datos que se eliminan por no tener un observado para comparar. Al predecir si se tiene estos datos en cuenta y se completa la serie.
    """

    df_Rosario_sim2 = df_Rosario_sim.copy()
    X_input = df_Rosario_sim2[['h_sim',]].values

    # Prediccion
    df_Rosario_sim2['Y_predic'] = lr.predict(X_input)

    horas_plot = 24*10
    df_Rosario_sim2 = df_Rosario_sim2[-horas_plot:]
    # df_simulado['e_pred_05'] = df_simulado['Y_predic'] + quant_Err[0.05]
    # df_simulado['e_pred_95'] = df_simulado['Y_predic'] + quant_Err[0.95]
    df_Rosario_sim2['e_pred_01'] = df_Rosario_sim2['Y_predic'] + quant_Err[0.001]
    df_Rosario_sim2['e_pred_99'] = df_Rosario_sim2['Y_predic'] + quant_Err[0.999]

    # 2 series

    series =  [
        prono2serie(df_Rosario_sim2,main_colname="Y_predic",members={'e_pred_01':'p01','e_pred_99':'p99'},series_id=29542)
    ]

    # PLOT FINAL
    plotFinal(df_Rosario_Obs,df_Rosario_sim2,nameout='productos/Prono_Rosario.png',ydisplay=-0.75,text_xoffset=(0.5,0),ylim=(-1,6), obsLine=False,nombre_estacion="Rosario", cero=2.92,fecha_emision=fecha_emision)
   
    """# Prediccion en Rosario_vPNAobs"""

    ## Carga Observados

    f_inicio = df_Rosario_sim.index.min()
    f_fin = df_Rosario_sim.index.max()
    Estaciones = {34:'Rosario',}
    idDest = 34
    df_RosarioPNA_Obs = getObs(idDest,f_inicio,f_fin)

    # PLOT FINAL
    plotFinal(df_RosarioPNA_Obs,df_Rosario_sim2,nameout='productos/Prono_RosarioPNA.png',ydisplay=1.5,text_xoffset=(0,0),xytext=(-320,-200),ylim=(-1,6),obs_label='Nivel Observado PNA',extraObs=df_Rosario_Obs,extraObsLabel="Nivel Observado BDHI",fecha_emision=fecha_emision,obsLine=False,nombre_estacion="Rosario")

    if output_csv:
        outputcsv(df_Rosario_sim2,"productos/Prono_Rosario.csv")

    """# Prediccion en Timbues"""
    df_Timbues_sim, fecha_emision = getLastProno(288,{'estacion_id': '1770','var_id':2})

    ###### Correccion 1 el simulado en zarate adelanta 1 hora
    df_Timbues_sim.index = df_Timbues_sim.index + timedelta(hours=1) # - timedelta(hours=2)

    X_input = df_Timbues_sim[['h_sim',]].values

    # Prediccion
    df_Timbues_sim['Y_predic'] = lr.predict(X_input)

    horas_plot = 24*10
    df_Timbues_sim = df_Timbues_sim[-horas_plot:]
    # df_simulado['e_pred_05'] = df_simulado['Y_predic'] + quant_Err[0.05]
    # df_simulado['e_pred_95'] = df_simulado['Y_predic'] + quant_Err[0.95]
    df_Timbues_sim['e_pred_01'] = df_Timbues_sim['Y_predic'] + quant_Err[0.001]
    df_Timbues_sim['e_pred_99'] = df_Timbues_sim['Y_predic'] + quant_Err[0.999]

    # 2 series
    series.append(prono2serie(df_Timbues_sim,main_colname="Y_predic",members={'e_pred_01':'p01','e_pred_99':'p99'},series_id=3389))

    # PLOT FINAL
    plotFinal(None,df_Timbues_sim,nameout='productos/Prono_Timbues.png',ydisplay=3.4,text_xoffset=(0,0),ylim=(2,7),nombre_estacion="Timbúes",fecha_emision=fecha_emision)
    
    ## UPLOAD PRONOSTICO
    if upload:
        uploadPronoSeries(series,cal_id=441,forecast_date=fecha_emision,outputfile="productos/prono_rosario.json",responseOutputFile="productos/pronoresponse.json")
    else:
        data = prono2json(series,forecast_date=fecha_emision)
        outputjson(data,"productos/prono_rosario.json")
    
    if output_csv:
        outputcsv(df_Timbues_sim,"productos/Prono_Timbues.csv")


def corrigeVCons(plots=False,upload=True, output_csv=False):
        ## Carga Simulados en VConstitucion
    df_VConstitucion_sim, fecha_emision = getLastProno(288,{'estacion_id': '5905','var_id':2})

    ## Carga Observados
    f_inicio = df_VConstitucion_sim.index.min()
    f_fin = df_VConstitucion_sim.index.max()
    df_VConstitucion_Obs = getObs(35,f_inicio,f_fin) # getObs(29436,f_inicio,f_fin)  <-- edit 2/2/2023, estacion sat2 caída, cambio a convencional

    # Elimina saltos
    df_VConstitucion_Obs = eliminaSaltos(df_VConstitucion_Obs,0.25)

    ###### Correccion 1 el simulado en VConstitucion adelanta 1 hora 
    df_VConstitucion_sim.index = df_VConstitucion_sim.index + timedelta(hours=1) # - timedelta(hours=2)

    ## Union
    df_VConstitucion = df_VConstitucion_sim.join(df_VConstitucion_Obs, how = 'outer')
    df_VConstitucion['h_sim'] = df_VConstitucion['h_sim'].interpolate(method='linear',limit=4)
    #df_VConstitucion['h_sim_Mavg'] = df_VConstitucion['h_sim'].rolling(4, min_periods=1).mean()

    if plots:
        plotObsVsSim(df_VConstitucion)

    """# Modelo RL"""

    train0 = df_VConstitucion.copy()
    train0['Error0'] = train0['h_sim'] - train0['h_obs']
    train0['dif_F1'] = train0['h_sim'].diff(periods=1).abs()
    train0['dif_F2'] = train0['h_sim'].diff(periods=2).abs()
    train0['dif_F3'] = train0['h_sim'].diff(periods=3).abs()

    train0['dif_B1'] = train0['h_sim'].diff(periods=-1).abs()
    train0['dif_B2'] = train0['h_sim'].diff(periods=-2).abs()
    train0['dif_B3'] = train0['h_sim'].diff(periods=-3).abs()

    train0['dif_F'] = train0['dif_F1'].rolling(3, min_periods=1).mean()
    train0['dif_B'] = train0['dif_B1'].rolling(3, min_periods=1).mean()

    ## Modelo
    train = train0[:].copy()
    train = train.dropna()

    var_obj = 'h_obs'
    #var_obj = 'Error0'
    covariav = ['h_sim','dif_F1','dif_B1','dif_F2','dif_B2','dif_F3','dif_B3']
    covariav = ['h_sim','dif_F1','dif_B1','dif_F2','dif_B2']
    covariav = ['h_sim','dif_F','dif_B']
    covariav = ['h_sim',]

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
    #train['Y_predictions'] = train['h_sim'] - train['Y_predictions'] 

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
    quant_Err = train['Error_pred'].quantile([.001,.05,.95,.999])

    if plots:
        plotObsVsSimVsPred(df_VConstitucion,train)

    """# Predicción  en VConstitucion

    Predice sobre todos los datos de VConstitucion.  Se vuelve a calcular porque al entrenar el modelo hay datos que se eliminan por no tener un observado para comparar. Al predecir si se tiene estos datos en cuenta y se completa la serie.
    """

    df_VConstitucion_sim2 = df_VConstitucion_sim.copy()
    X_input = df_VConstitucion_sim2[['h_sim',]].values

    # Prediccion
    df_VConstitucion_sim2['Y_predic'] = lr.predict(X_input)

    horas_plot = 24*10
    df_VConstitucion_sim2 = df_VConstitucion_sim2[-horas_plot:]
    df_VConstitucion_sim2['e_pred_05'] = df_VConstitucion_sim2['Y_predic'] + quant_Err[0.05]
    df_VConstitucion_sim2['e_pred_95'] = df_VConstitucion_sim2['Y_predic'] + quant_Err[0.95]
    df_VConstitucion_sim2['e_pred_01'] = df_VConstitucion_sim2['Y_predic'] + quant_Err[0.001]
    df_VConstitucion_sim2['e_pred_99'] = df_VConstitucion_sim2['Y_predic'] + quant_Err[0.999]

    series = [
        prono2serie(df_VConstitucion_sim2,main_colname="Y_predic",members={'e_pred_01':'p01','e_pred_05':'p05','e_pred_95':'e95','e_pred_99':'p99'},series_id=29538)
    ]

    # PLOT FINAL
    plotFinal(df_VConstitucion_Obs,df_VConstitucion_sim2,nameout='productos/Prono_VConstitucion.png',ydisplay=-0.75,text_xoffset=(0.5,0),xytext=(-300,-100),ylim=(-1,6),bandaDeError=('e_pred_05','e_pred_95'),nombre_estacion="Villa Constitución",niveles_alerta={"aguas_bajas":1.9}, cero=1.98, fecha_emision=fecha_emision)

    """# Prediccion en VConstitucion_vPNAobs"""

    ## Carga Observados
    f_inicio = df_VConstitucion_sim.index.min()
    f_fin = df_VConstitucion_sim.index.max()
    Estaciones = {35:'VConstitucion',}
    df_VConstitucionPNA_Obs = getObs(35,f_inicio,f_fin)

    # PLOT FINAL
    plotFinal(df_VConstitucionPNA_Obs,df_VConstitucion_sim2,nameout='productos/Prono_VConstitucionPNA.png',ydisplay=1.5,text_xoffset=(-.5,.5),xytext=(-320,-200),ylim=(-1.,6),bandaDeError=('e_pred_05','e_pred_95'),extraObs=df_VConstitucion_Obs,extraObsLabel='Nivel Observado BDHI',obs_label='Nivel Observado PNA',fecha_emision=fecha_emision,obsLine=False,nombre_estacion="Villa Constitución",niveles_alerta={"aguas_bajas":1.9}, cero=1.98)

    if output_csv:
        outputcsv(df_VConstitucion_sim2,"productos/Prono_VConstitucion.csv")


    """# Prediccion en SanNicolas"""

    df_SanNicolas_sim, fecha_emision = getLastProno(288,{'estacion_id': '36','var_id':2})

    ###### Correccion 1 el simulado en zarate adelanta 1 hora
    df_SanNicolas_sim.index = df_SanNicolas_sim.index + timedelta(hours=1) # - timedelta(hours=2)

    ## Carga Observados
    f_inicio = df_SanNicolas_sim.index.min()
    f_fin = df_SanNicolas_sim.index.max()
    df_SanNicolas_Obs = getObs(36,f_inicio,f_fin)

    X_input = df_SanNicolas_sim[['h_sim',]].values

    # Prediccion
    df_SanNicolas_sim['Y_predic'] = lr.predict(X_input)+0.25

    horas_plot = 24*10
    df_SanNicolas_sim = df_SanNicolas_sim[-horas_plot:]
    df_SanNicolas_sim['e_pred_05'] = df_SanNicolas_sim['Y_predic'] + quant_Err[0.05]
    df_SanNicolas_sim['e_pred_95'] = df_SanNicolas_sim['Y_predic'] + quant_Err[0.95]
    df_SanNicolas_sim['e_pred_01'] = df_SanNicolas_sim['Y_predic'] + quant_Err[0.001]
    df_SanNicolas_sim['e_pred_99'] = df_SanNicolas_sim['Y_predic'] + quant_Err[0.999]

    # 2 series
    series.append(prono2serie(df_SanNicolas_sim,main_colname="Y_predic",members={'e_pred_01':'p01','e_pred_05':'p05','e_pred_95':'e95','e_pred_99':'p99'},series_id=3414))
    
    # PLOT FINAL
    plotFinal(df_SanNicolas_Obs,df_SanNicolas_sim,nameout='productos/Prono_SanNicolas.png',ydisplay=-0.75,text_xoffset=(0.5,0.5),ylim=(-1.,6),bandaDeError=('e_pred_05','e_pred_95'),markersize=10,obsLine=False,nombre_estacion="San Nicolás",niveles_alerta={"aguas_bajas":1.8}, cero=1.91,fecha_emision=fecha_emision)

    if output_csv:
        outputcsv(df_SanNicolas_sim,"productos/Prono_SanNicolas.csv")


    """# Prediccion en Ramallo"""
    df_Ramallo_sim, fecha_emision = getLastProno(288,{'estacion_id': '37','var_id':2})
    ###### Correccion 1 el simulado en zarate adelanta 1 hora
    df_Ramallo_sim.index = df_Ramallo_sim.index + timedelta(hours=1) # - timedelta(hours=2)

    ## Carga Observados
    f_inicio = df_Ramallo_sim.index.min()
    f_fin = df_Ramallo_sim.index.max()
    df_Ramallo_Obs = getObs(37,f_inicio,f_fin)

    X_input = df_Ramallo_sim[['h_sim',]].values

    # Prediccion
    df_Ramallo_sim['Y_predic'] = lr.predict(X_input)

    horas_plot = 24*10
    df_Ramallo_sim = df_Ramallo_sim[-horas_plot:]
    df_Ramallo_sim['e_pred_05'] = df_Ramallo_sim['Y_predic'] + quant_Err[0.05]
    df_Ramallo_sim['e_pred_95'] = df_Ramallo_sim['Y_predic'] + quant_Err[0.95]
    df_Ramallo_sim['e_pred_01'] = df_Ramallo_sim['Y_predic'] + quant_Err[0.001]
    df_Ramallo_sim['e_pred_99'] = df_Ramallo_sim['Y_predic'] + quant_Err[0.999]

    # 2 series
    series.append(prono2serie(df_Ramallo_sim,main_colname="Y_predic",members={'e_pred_01':'p01','e_pred_05':'p05','e_pred_95':'e95','e_pred_99':'p99'},series_id=3415))
    
    # PLOT FINAL
    plotFinal(df_Ramallo_Obs,df_Ramallo_sim,nameout='productos/Prono_Ramallo.png',ydisplay=-0.75,text_xoffset=(0.5,0.5),ylim=(-1.,5),bandaDeError=('e_pred_05','e_pred_95'),markersize=10,obsLine=False,nombre_estacion="Ramallo",niveles_alerta={"aguas_bajas":1.8}, cero=1.91,fecha_emision=fecha_emision)

    if output_csv:
        outputcsv(df_Ramallo_sim,"productos/Prono_Ramallo.csv")

        
    ## UPLOAD PRONOSTICO
    if upload:
        uploadPronoSeries(series,cal_id=442,forecast_date=fecha_emision,outputfile="productos/prono_vconstitucion.json",responseOutputFile="productos/pronoresponse.json")
    else:
        data = prono2json(series,forecast_date=fecha_emision)
        outputjson(data,"productos/prono_vconstitucion.json")


def corrigeCarabelas(plots=False,upload=True,output_csv=False):
    ## Carga Simulados
    df_simulado, fecha_emision = getLastProno(288,{'estacion_id': '5876','var_id':2})

    ## Carga Observados
    f_inicio = df_simulado.index.min()
    f_fin = df_simulado.index.max()
    df_Obs = getObs(26206,f_inicio,f_fin)
    if (not len(df_Obs)):
        print("Abortando corrigeCarabelas")
        return    
    ## Union
    # df_simulado.set_index(df_simulado['fecha'], inplace=True)
    # df_simulado.index = df_simulado.index.tz_convert(None)
    # del df_simulado['fecha']

    ###### Correccion 1
    df_simulado.index = df_simulado.index + timedelta(hours=1) # - timedelta(hours=2) #

    # df_Obs.set_index(df_Obs['fecha'], inplace=True)
    # df_Obs.index = df_Obs.index.tz_convert(None)
    # del df_Obs['fecha']

    df_union = df_simulado.join(df_Obs, how = 'outer')
    df_union['h_sim'] = df_union['h_sim'].interpolate(method='linear',limit=4)
    #df_union['h_sim_Mavg'] = df_union['h_sim'].rolling(4, min_periods=1).mean()

    df_union = df_union.dropna()

    ## Plot
    if plots:
        plotObsVsSim(df_union)
  
    ## Modelo
    train = df_union[:].copy()
    var_obj = 'h_obs'
    covariav = ['h_sim',]

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
    quant_Err = train['Error_pred'].quantile([.001,.05,.95,.999])
    if plots:
        plotObsVsSimVsPred(df_union,train)
 
    X_input = df_simulado[['h_sim',]].values
    df_simulado['Y_predic'] = lr.predict(X_input)

    horas_plot = 24*7
    df_simulado = df_simulado[-horas_plot:]

    if plots:
        fig = plt.figure(figsize=(17, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(df_simulado.index, df_simulado['Y_predic'], 'r-',label='Prediccion')
        ax.plot(df_Obs.index, df_Obs['h_obs'], 'b-.',label='Observado')

        plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
        plt.tick_params(axis='both', labelsize=18)
        plt.xlim(df_simulado.index.min(),df_simulado.index.max())
        plt.xlabel('Fecha', size=18)
        plt.ylabel('Nivel [m]', size=18)
        plt.legend(prop={'size':20},loc=0)
        plt.tight_layout()
        plt.show()

    # df_simulado['e_pred_05'] = df_simulado['Y_predic'] + quant_Err[0.05]
    # df_simulado['e_pred_95'] = df_simulado['Y_predic'] + quant_Err[0.95]
    df_simulado['e_pred_01'] = df_simulado['Y_predic'] + quant_Err[0.001]
    df_simulado['e_pred_99'] = df_simulado['Y_predic'] + quant_Err[0.999]

    # PLOT FINAL
    plotFinal(df_Obs,df_simulado,ydisplay=3.4,text_xoffset=(0.5,0.5),ylim=(-0.5,3.5),nameout='productos/prono_carabelas.png',nombre_estacion="Carabelas",cero=None,fecha_emision=fecha_emision)

    if output_csv:
        outputcsv(df_simulado,"productos/prono_carabelas.csv")

    series = [
        prono2serie(df_simulado,main_colname="Y_predic",members={'e_pred_01':'p01','e_pred_99':'p99'},series_id=29536)
    ]

    ## UPLOAD PRONOSTICO
    if upload:
        uploadPronoSeries(series,cal_id=439,forecast_date=fecha_emision,outputfile="productos/prono_carabelas.json",responseOutputFile="productos/pronoresponse.json")
    else:
        data = prono2json(series,forecast_date=fecha_emision)
        outputjson(data,"productos/prono_carabelas.json")

def corrigeNuevaPalmira(plots=False,upload=True,output_csv=False):
  conn_string = "dbname='" + config["database"]["dbname"] + "' user='" + config["database"]["user"] + "' host='" + config["database"]["host"] + "' port='" + str(config["database"]["port"]) + "'"
  try:
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
  except:
    print( "No se ha podido establecer conexion.")
    return


  plotea = False

  ahora = datetime.now()
  fecha_emision = ahora.replace(hour=0, minute=0, second=0, microsecond=0)
  
  f_inicio = datetime(2021,1, 1, 00, 0, 0, 0)
  f_fin = ahora

#   iDest = tuple([1699,80])
  Est_Uruguay = {3280:'NuevaPalmira',80:'Colon'}

  # OBSERVADOS
  df_obs_npal = getObs(3280,f_inicio,f_fin)
  df_obs_npal["series_id"] = 3280
  df_obs_col = getObs(80,f_inicio,f_fin)
  df_obs_col["series_id"] = 80
  df_obs = pd.concat([df_obs_npal,df_obs_col])
  df_Obs = pd.pivot_table(df_obs, values='h_obs', index=['fecha'], columns=['series_id'])
  #Parametros para la consulta SQL a la BBDD
#   paramH0 = (f_inicio, f_fin,iDest)  
#   sql_query = ('''SELECT unid as id, timestart as fecha, valor as h_obs
#                 FROM alturas_all
#                 WHERE  timestart BETWEEN %s AND %s AND unid IN %s;''')  #Consulta SQL
#   df_Obs0 = pd.read_sql_query(sql_query, conn, params=paramH0)              #Toma los datos de la BBDD
#   df_Obs0['fecha'] = pd.to_datetime(df_Obs0['fecha'])#.round('min')

  # Estaciones en cada columnas
#   df_Obs = pd.pivot_table(df_Obs0, values='h_obs', index=['fecha'], columns=['id'])
  print("Resample horario")
  df_Obs_H = df_Obs.resample('H').mean()

  # Nombres a los calumnas
  df_Obs_H = df_Obs_H.rename(columns=Est_Uruguay)

  # tz-aware to tz-naive
  df_Obs_H.index = pd.to_datetime(df_Obs_H.index).tz_convert('-03:00').tz_localize(None)
  ###############################
  # df_astro = getObs(6046,f_inicio,f_fin)
  # df_meteo = getObs(6059,f_inicio,f_fin)


  paramH1 = (f_inicio, f_fin)

  sql_query = ('''SELECT  timestart as fecha, fecha_emision, 
                          altura_meteo, altura_astro, altura_suma, altura_suma_corregida
                  FROM alturas_marea_full 
                  WHERE  timestart BETWEEN %s AND %s AND estacion_id = 1843
                  AND timestart >= fecha_emision and timestart <= fecha_emision + interval '5 hour';''')#WHERE  
  df_Sim0 = pd.read_sql_query(sql_query, conn, params=paramH1)	
  df_Sim0['fecha'] =  pd.to_datetime(df_Sim0['fecha'])#, format='%Y-%m-%d')                     #Convierte a formato fecha la columna [fecha]
  df_Sim0['fecha_emision'] =  pd.to_datetime(df_Sim0['fecha_emision'])

  # Calcula la cantidad de horas de pronóstico (anticipo): fecha del dato menos fecha de emision
  df_Sim0['horas_prono'] = (df_Sim0['fecha'] - df_Sim0['fecha_emision']).astype('timedelta64[h]')
  # Hora del pronostico
  df_Sim0['hora'] = df_Sim0.apply(lambda row: row['fecha'].hour,axis=1)

  # Lleva las salidas del modelo en cota IGN al cero local
  # df_Sim0['altura_astro'] = df_Sim0['altura_astro'].add(0.53)

  df_Sim0['Cat_anticipo'] = np.nan
  df_Sim0.loc[df_Sim0['horas_prono'].isin([0,1,2,3,4,5]), 'Cat_anticipo'] = 'A01'
  df_Sim0.loc[df_Sim0['horas_prono'].isin([6,7,8,9,10,11]), 'Cat_anticipo'] = 'A02'
  df_Sim0.loc[df_Sim0['horas_prono'].isin([12,13,14,15,16,17]), 'Cat_anticipo'] = 'A03'
  df_Sim0.loc[df_Sim0['horas_prono'].isin([18,19,20,21,22,23]), 'Cat_anticipo'] = 'A04'
  df_Sim0.loc[df_Sim0['horas_prono'].isin([24,25,26,27,28,29]), 'Cat_anticipo'] = 'A05'
  df_Sim0.loc[df_Sim0['horas_prono'].isin([30,31,32,33,34,35]), 'Cat_anticipo'] = 'A06'
  df_Sim0.loc[df_Sim0['horas_prono'].isin([36,37,38,39,40,41]), 'Cat_anticipo'] = 'A07'
  df_Sim0.loc[df_Sim0['horas_prono'].isin([42,43,44,45,46,47]), 'Cat_anticipo'] = 'A08'
  df_Sim0.loc[df_Sim0['horas_prono'].isin([48,49,50,51,52,53]), 'Cat_anticipo'] = 'A09'
  df_Sim0.loc[df_Sim0['horas_prono'].isin([54,55,56,57,58,59]), 'Cat_anticipo'] = 'A10'
  df_Sim0.loc[df_Sim0['horas_prono'].isin([60,61,62,63,64,65]), 'Cat_anticipo'] = 'A11'
  df_Sim0.loc[df_Sim0['horas_prono'].isin([66,67,68,69,70,71]), 'Cat_anticipo'] = 'A12'
  df_Sim0.loc[df_Sim0['horas_prono'].isin([72,73,74,75,76,77]), 'Cat_anticipo'] = 'A13'
  df_Sim0.loc[df_Sim0['horas_prono'].isin([78,79,80,81,82,83]), 'Cat_anticipo'] = 'A14'
  df_Sim0.loc[df_Sim0['horas_prono'].isin([84,85,86,87,88,89]), 'Cat_anticipo'] = 'A15'
  df_Sim0.loc[df_Sim0['horas_prono'].isin([90,91,92,93,94,95]), 'Cat_anticipo'] = 'A16'

  df_NP_PronoCat = pd.pivot_table(df_Sim0, 
                        values=['altura_meteo'],#, 'altura_suma', 'altura_suma_corregida'], 'fecha_emision','altura_astro'
                        index=['fecha','altura_astro'],
                        columns=['Cat_anticipo'], aggfunc=np.sum)

  df_NP_PronoCat.columns = df_NP_PronoCat.columns.get_level_values(1)
  df_NP_PronoCat = df_NP_PronoCat.reset_index(level=[1,])

  l_cat = ['A01','A02','A03','A04','A05','A06','A07','A08','A09','A10','A11','A12','A13','A14','A15','A16']
  #for cat in l_cat: print('NaN : '+cat+' '+str(df_NP_PronoCat[cat].isna().sum()))

  ### PRIMERA CORRECCION
  df_NP_PronoCat.index = df_NP_PronoCat.index + timedelta(hours=1)

  ## Union #######################################################
  #Crea DF base:
  indexUnico = pd.date_range(start=df_Sim0['fecha'].min(), end=df_Sim0['fecha'].max(), freq='H')	    #Fechas desde f_inicio a f_fin con un paso de 5 minutos
  df_base = pd.DataFrame(index = indexUnico)								#Crea el Df con indexUnico
  df_base.index.rename('fecha', inplace=True)							    #Cambia nombre incide por Fecha
  df_base.index = df_base.index.round("H")

  # Une obs y sim
  df_base = df_base.join(df_NP_PronoCat[['A01','altura_astro']], how = 'left')
  df_base = df_base.join(df_Obs_H[['Colon','NuevaPalmira']], how = 'left')

  #print(df_base.head())

  df_base = df_base.dropna().copy()

  ###########################
  df_base['CUru_dly'] = df_base['Colon'].shift(24*4)
  df_base['CUru_dly'] = df_base['CUru_dly'].interpolate(limit_direction="forward")

  df_base['h_met_Mavg'] = df_base['A01'].rolling(24, min_periods=1).mean()

  del df_base['Colon']
  df_base = df_base.dropna()

  #print('Cantidad de datos de entrenamiento:',len(df_base))
  #print(df_base.tail(5))
  ###########################


  ## Modelo
  train = df_base[:].copy()
  var_obj = 'NuevaPalmira'
  covariav = ['altura_astro','A01','h_met_Mavg','CUru_dly']
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
  
  ## Pronóstico 
  '''Conecta con BBDD'''

  # Plot final
  ahora = datetime.now()
  DaysMod = 10
  f_fin = ahora
  f_inicio = (f_fin - timedelta(days=DaysMod)).replace(hour=0, minute=0, second=0)
  f_fin = (f_fin + timedelta(days=5))

  # iDest = tuple([1699,2232])
  Est_Uruguay = {3280:'NuevaPalmira',80:'Colon'}
  
  # OBSERVADOS
  df_obs_npal = getObs(3280,f_inicio,f_fin)
  df_obs_npal["series_id"] = 3280
  df_obs_col = getObs(80,f_inicio,f_fin)
  df_obs_col["series_id"] = 80
  df_obs = pd.concat([df_obs_npal,df_obs_col])
  df_Obs = pd.pivot_table(df_obs, values='h_obs', index=['fecha'], columns=['series_id'])
  #Parametros para la consulta SQL a la BBDD
  #paramH0 = (f_inicio, f_fin,iDest)  
  #sql_query = ('''SELECT unid as id, timestart as fecha, valor as h_obs
  #              FROM alturas_all
  #              WHERE  timestart BETWEEN %s AND %s AND unid IN %s;''')  #Consulta SQL
  #df_Obs0 = pd.read_sql_query(sql_query, conn, params=paramH0)              #Toma los datos de la BBDD
  #df_Obs0['fecha'] = df_Obs0['fecha'].dt.round('min') # pd.to_datetime(df_Obs0['fecha']).round('min')

  # Estaciones en cada columnas
  #df_Obs = pd.pivot_table(df_Obs0, values='h_obs', index=['fecha'], columns=['id'])
  print("Resample horario")
  df_Obs_H = df_Obs.resample('H').mean()
  df_Obs_H.index = df_Obs_H.index.tz_convert("America/Argentina/Buenos_Aires")

  # Nombres a los calumnas
  df_Obs_H = df_Obs_H.rename(columns=Est_Uruguay)

  # SIMULADO
  paramH1 = (f_inicio, f_fin)         #Parametros para la consulta SQL a la BBDD
  sql_query = ('''SELECT unid as id, timestart as fecha,
                  altura_astronomica_ign as h_ast_ign, altura_meteorologica as h_met,
                  timeupdate as fecha_emision
                  FROM alturas_marea_suma
                  WHERE  timestart BETWEEN %s AND %s AND unid = 1843;''')              #Consulta SQL
  df_Sim = pd.read_sql_query(sql_query, conn, params=paramH1)								#Toma los datos de la BBDD	
  df_Sim['fecha'] =  pd.to_datetime(df_Sim['fecha'])#, format='%Y-%m-%d')                     #Convierte a formato fecha la columna [fecha]

  keys =  pd.to_datetime(df_Sim['fecha'])#, format='%Y-%m-%d')                     #Convierte a formato fecha la columna [fecha]
  df_Sim.set_index(keys, inplace=True)

  df_Sim.index = df_Sim.index + timedelta(hours=1)
  df_Sim.index = df_Sim.index.tz_localize("America/Argentina/Buenos_Aires")

  ## Union
  indexUnico = pd.date_range(start=df_Sim['fecha'].min(), end=df_Sim['fecha'].max(), freq='H',tz="America/Argentina/Buenos_Aires")	    #Fechas desde f_inicio a f_fin con un paso de 5 minutos
  df_base = pd.DataFrame(index = indexUnico)								#Crea el Df con indexUnico
  df_base.index.rename('fecha', inplace=True)							    #Cambia nombre incide por Fecha
  df_base.index = df_base.index.round("H")

  df_base = df_base.join(df_Sim[['h_met','h_ast_ign']], how = 'left')
  df_base = df_base.join(df_Obs_H['Colon'], how = 'left')

  df_base['CUru_dly'] = df_base['Colon'].shift(24*4)
  df_base['CUru_dly'] = df_base['CUru_dly'].interpolate(limit_direction="forward")

  df_base['h_met_Mavg'] = df_base['h_met'].rolling(24, min_periods=1).mean()

  del df_base['Colon']

  df_base = df_base.dropna()

  covariav = ['h_ast_ign','h_met','h_met_Mavg','CUru_dly']
  prediccion = lr.predict(df_base[covariav].values)
  df_base['Y_predic'] = prediccion

  #df_base['e_pred_05'] = df_base['Y_predic'] + quant_Err[0.05]
  df_base['e_pred_01'] = df_base['Y_predic'] + quant_Err[0.01]
  df_base['e_pred_99'] = df_base['Y_predic'] + quant_Err[0.99]
  #df_base['e_pred_95'] = df_base['Y_predic'] + quant_Err[0.95]

  df_Obs = df_Obs_H[['NuevaPalmira',]]
  df_Obs = df_Obs.rename(columns={'NuevaPalmira':'h_obs'})
  df_sim = df_base[['Y_predic','e_pred_01','e_pred_99']]
  df_sim1 = df_sim.copy()
  df_sim1.index = df_sim1.index.to_pydatetime()

  # PLOT FINAL
  plotFinal(df_Obs,df_sim1,ydisplay=3.4,text_xoffset=(0.5,0.5),ylim=(-0.5,3.5),nameout='productos/Prono_NPalmira.png',nombre_estacion="NuevaPalmira",cero=None,fecha_emision=fecha_emision)

  if output_csv:
      outputcsv(df_sim,"productos/prono_NuevaPalmira.csv")

  series = [
      prono2serie(df_sim,main_colname="Y_predic",members={'e_pred_01':'p01','e_pred_99':'p99'},series_id=26203)
  ]

  ## UPLOAD PRONOSTICO
  if upload:
      uploadPronoSeries(series,cal_id=433,forecast_date=fecha_emision,outputfile="productos/prono_NuevaPalmira.json",responseOutputFile="productos/pronoresponse.json")
  else:
      data = prono2json(series,forecast_date=fecha_emision)
      outputjson(data,"productos/prono_NuevaPalmira.json")

def corrigeBrazoLargo(plots=False,upload=True,output_csv=False):
    ## Carga Simulados
    df_simulado, fecha_emision = getLastProno(288,{'estacion_id': '2830','var_id':2})

    ## Carga Observados
    f_inicio = df_simulado.index.min()
    f_fin = df_simulado.index.max()
    df_Obs = getObs(9442,f_inicio,f_fin)
 
    ## Union
    # df_simulado.set_index(df_simulado['fecha'], inplace=True)
    # df_simulado.index = df_simulado.index.tz_convert(None)
    # del df_simulado['fecha']

    ###### Correccion 1
    df_simulado.index = df_simulado.index + timedelta(hours=1) # - timedelta(hours=2) #

    # df_Obs.set_index(df_Obs['fecha'], inplace=True)
    # df_Obs.index = df_Obs.index.tz_convert(None)
    # del df_Obs['fecha']

    df_union = df_simulado.join(df_Obs, how = 'outer')
    df_union['h_sim'] = df_union['h_sim'].interpolate(method='linear',limit=4)
    #df_union['h_sim_Mavg'] = df_union['h_sim'].rolling(4, min_periods=1).mean()

    df_union = df_union.dropna()

    ## Plot
    if plots:
        plotObsVsSim(df_union)
  
    ## Modelo
    train = df_union[:].copy()
    var_obj = 'h_obs'
    covariav = ['h_sim',]

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
    quant_Err = train['Error_pred'].quantile([.001,.05,.95,.999])
    if plots:
        plotObsVsSimVsPred(df_union,train)
 
    X_input = df_simulado[['h_sim',]].values
    df_simulado['Y_predic'] = lr.predict(X_input)

    horas_plot = 24*7
    df_simulado = df_simulado[-horas_plot:]

    if plots:
        fig = plt.figure(figsize=(17, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(df_simulado.index, df_simulado['Y_predic'], 'r-',label='Prediccion')
        ax.plot(df_Obs.index, df_Obs['h_obs'], 'b-.',label='Observado')

        plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
        plt.tick_params(axis='both', labelsize=18)
        plt.xlim(df_simulado.index.min(),df_simulado.index.max())
        plt.xlabel('Fecha', size=18)
        plt.ylabel('Nivel [m]', size=18)
        plt.legend(prop={'size':20},loc=0)
        plt.tight_layout()
        plt.show()

    # df_simulado['e_pred_05'] = df_simulado['Y_predic'] + quant_Err[0.05]
    # df_simulado['e_pred_95'] = df_simulado['Y_predic'] + quant_Err[0.95]
    df_simulado['e_pred_01'] = df_simulado['Y_predic'] + quant_Err[0.001]
    df_simulado['e_pred_99'] = df_simulado['Y_predic'] + quant_Err[0.999]

    # PLOT FINAL
    plotFinal(df_Obs,df_simulado,ydisplay=3.4,text_xoffset=(0.5,0.5),ylim=(-1.5,2.5),nameout='productos/prono_BrazoLargo.png',nombre_estacion="BrazoLargo",cero=None,fecha_emision=fecha_emision)

    if output_csv:
        outputcsv(df_simulado,"productos/BrazoLargo.csv")

    series = [
        prono2serie(df_simulado,main_colname="Y_predic",members={'e_pred_01':'p01','e_pred_99':'p99'},series_id=29586)
    ]

    ## UPLOAD PRONOSTICO
    if upload:
        uploadPronoSeries(series,cal_id=445,forecast_date=fecha_emision,outputfile="productos/prono_BrazoLargo.json",responseOutputFile="productos/pronoresponse.json")
    else:
        data = prono2json(series,forecast_date=fecha_emision)
        outputjson(data,"productos/BrazoLargo.json")

def runPlan(plan,plots=True,upload=False,output_csv=False):
    planes = {
        "zarate": corrigeZarate,
        "atucha": corrigeAtucha,
        "rosario": corrigeRosario,
        "vcons": corrigeVCons,
        "carabelas": corrigeCarabelas,
        "npalmira": corrigeNuevaPalmira,
        "blargo": corrigeBrazoLargo
    }
    if plan in ("all","All","ALL"):
        for key in planes:
            planes[key](plots,upload,output_csv)
    elif plan in planes:
        planes[plan](plots,upload,output_csv)
    else:
        print("Plan " + str(plan) + "not found")

"""# controller x línea de comando"""

def main(argv):
    plots=False
    upload=False
    output_csv=False
    plan="all"
    try:
        opts, args = getopt.getopt(argv,"hpucP:",["plots","update","output_csv","plan="])
    except getopt.GetoptError:
        print('pronos_delta.py -h -p -u -c -P <plan>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('pronos_delta.py -h -p -u -c -P <plan>')
            sys.exit()
        elif opt in ("-p", "--plots"):
            plots = True
        elif opt in ("-u", "--update"):
            upload = True
        elif opt in ("-P", "--plan"):
            plan = arg
        elif opt in ("-c", "--output_csv"):
            output_csv = True
    print({"plots":plots, "upload":upload, "output_csv":output_csv, "plan":plan})
    runPlan(plan,plots,upload,output_csv)

if __name__ == "__main__":
   main(sys.argv[1:])


# corrigeZarate(plots=True,upload=False)
# corrigeAtucha(plots=True,upload=False)
# corrigeRosario(plots=True,upload=False)
# corrigeVCons(plots=True,upload=False)
# corrigeCarabelas(plots=True,upload=False)
