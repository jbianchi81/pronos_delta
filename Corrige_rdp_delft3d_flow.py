import pandas as pd
import numpy as np
from datetime import timedelta,datetime
import matplotlib.pyplot as plt

import pytz
from pytz import timezone
localtz = timezone('America/Argentina/Buenos_Aires')
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from Funciones import Consulta_id_corridas, cargasim, readSerie
from Funciones import observacionesListToDataFrame, plotFinal
from Funciones import outputcsv, prono2serie, uploadPronoSeries, prono2json,outputjson

# import json
# import datetime
# import requests
# import dateutil
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from matplotlib.dates import DateFormatter

Dic_Estaciones = {'id_modelo':443,
                  'id_modelo_corr':543,
                  'DiasCorreccion':10,
                  'Estaciones': {'San Fernando': {'estacion_id':52,'obs_id': 52,'Nombre_out':'SanFernando'},
                                  'Nueva Palmira': {'estacion_id':1699,'obs_id': 31555,'Nombre_out':'NuevaPalmira'}}}

def corrigeSalidaModelo(est,Dic_Estaciones,plots=False,upload=True,output_csv=True,PlotControl=True):
    id_modelo = Dic_Estaciones['id_modelo']
    id_modelo_corr = Dic_Estaciones['id_modelo_corr']
    
    print('Modelo Original - cal_id: ', id_modelo,
          'Modelo Corregido - cal_id: ', id_modelo_corr,)

    estacion_id = Dic_Estaciones['Estaciones'][est]['estacion_id']
    obs_id = Dic_Estaciones['Estaciones'][est]['obs_id']
    Nombre_out = Dic_Estaciones['Estaciones'][est]['Nombre_out']

    ## Consulta id de las corridas
    DiasCorreccion = Dic_Estaciones['DiasCorreccion']
    ahora = datetime.now(pytz.timezone('America/Argentina/Buenos_Aires')).replace(minute=0, second=0, microsecond=0)
    fecha_pronos_calib = ahora - timedelta(days=DiasCorreccion) # Corrige con los ultimos x días

    ##################### Conulta Simulado #####################
    # 1 - Consulta ID Corridas
    json_res = Consulta_id_corridas(id_modelo)
    print('Cantidad total de corridas: ',len(json_res))
    
    lst_corridas = []
    lst_pronoday = []
    for corridas in range(len(json_res)):
        lst_corridas.append(json_res[corridas]['cor_id'])
        lst_pronoday.append(json_res[corridas]['forecast_date'])
    df_id = pd.DataFrame(lst_pronoday, index =lst_corridas,columns=['forecast_date',])
    df_id['forecast_date'] = pd.to_datetime(df_id['forecast_date'])
    df_id = df_id[df_id['forecast_date']>fecha_pronos_calib]            # Filtra las corridas viejas
    print('Cantidad filtrada de corridas: ',len(df_id))

    # 2 - Consulta Corridas
    df_base_unico = pd.DataFrame(columns=['h_sim','cor_id'])
    for index, row in df_id.iterrows():                         # Carga las series simuladas usnado los ID de las corridas
        df_corr_i = cargasim(id_modelo,index,estacion_id)
        if df_corr_i.empty: continue
        df_corr_i.set_index(df_corr_i['fecha'], inplace=True)

        df_corr_i.index = df_corr_i.index.tz_convert(localtz) # - timedelta(hours=3)
        # df_corr_i['fecha'] = df_corr_i.index
        # df_corr_i.reset_index(drop=True)
        #df_base = pd.concat([df_base, df_corr_i], ignore_index=True)
        df_base_unico.update(df_corr_i)                         # Actualiza los pronos 
        df_base_unico = pd.concat([df_base_unico, df_corr_i[~df_corr_i.index.isin(df_base_unico.index)]])   # guarda los nuevos
    

    ##################### Conulta Observado #####################

    f_inicio = df_base_unico.index.min()
    f_fin = df_base_unico.index.max()

    df_obs = readSerie(obs_id,f_inicio,f_fin)
    df_obs = observacionesListToDataFrame(df_obs["observaciones"])
    df_obs["series_id"] = obs_id
    df_obs = df_obs.rename(columns={'valor':'h_obs'})

    ### PRIMERA CORRECCION
    #df_base_unico.index = df_base_unico.index + timedelta(hours=2)
    # df_base['CUru_dly'] = df_base['Colon'].shift(24*4)
    # df_base['CUru_dly'] = df_base['CUru_dly'].interpolate(limit_direction="forward")
    # df_base['h_met_Mavg'] = df_base['A01'].rolling(24, min_periods=1).mean()
    # del df_base['Colon']
    # df_base = df_base.dropna()    

    # Une obs y sim
    df_base_unico = df_base_unico.join(df_obs, how = 'outer')
    df_base_unico['h_obs'] = df_base_unico['h_obs'].interpolate(limit=2)
    df_base_unico = df_base_unico[['h_sim', 'h_obs']]

    df_base_unico = df_base_unico.dropna().copy()
    # print(df_base_unico.head())

    print('Cantidad de datos de entrenamiento:',len(df_base_unico))
    #print(df_base_unico.tail(5))
    ###########################

    ## Modelo
    # print(df_base_unico.head())
    # print(df_base_unico.tail())

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

    df_last_prono = pd.DataFrame(columns=['h_sim','cor_id'])
    df_id_last = df_id[-4:]

    for index, row in df_id_last.iterrows():                         # Carga las series simuladas usnado los ID de las corridas
        df_corr_i = cargasim(id_modelo,index,estacion_id)
        if df_corr_i.empty: continue
        df_corr_i.set_index(df_corr_i['fecha'], inplace=True)
        df_corr_i.index = df_corr_i.index.tz_convert(localtz) # - timedelta(hours=3)
        # df_corr_i['fecha'] = df_corr_i.index
        # df_corr_i.reset_index(drop=True)
        #df_base = pd.concat([df_base, df_corr_i], ignore_index=True)

        df_last_prono.update(df_corr_i)                         # Actualiza los pronos 
        df_last_prono = pd.concat([df_last_prono, df_corr_i[~df_corr_i.index.isin(df_last_prono.index)]])   # guarda los nuevos
    
    #print(df_last_prono.cor_id.unique())
    #df_last_prono.index = df_last_prono.index + timedelta(hours=2)
    df_last_prono = df_last_prono[['h_sim',]]
    # print(df_obs_npal.head())
    # print(df_last_prono.head())

    covariav = ['h_sim',]
    prediccion = lr.predict(df_last_prono[covariav].values)
    df_last_prono['Y_predic'] = prediccion
    
    if PlotControl:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(df_base_unico.index, df_base_unico['h_obs'],'.',label='Nueva Palmira')    
        ax.plot(train.index, train['Y_predictions'],'-',label='Y_predictions')
        ax.plot(df_base_unico.index, df_base_unico['h_sim'],label='h_sim')
        ax.plot(df_last_prono.index, df_last_prono['h_sim'],label='Last Prono')
        ax.plot(df_last_prono.index, df_last_prono['Y_predic'],label='Last Prono Corregido')

        plt.grid(True,axis='y', which='both', color='0.75', linestyle='-.',linewidth=0.3)
        plt.tick_params(axis='y', labelsize=14)
        plt.tick_params(axis='x', labelsize=14,rotation=20)
        plt.xlabel('Mes', size=18)
        plt.ylabel('Nivel [m]', size=18)# 'Caudal [m'+r'$^3$'+'/s]'
        plt.legend(prop={'size':16},loc=0,ncol=1)
        plt.show()
        plt.close()

    df_last_prono['e_pred_01'] = df_last_prono['Y_predic'] + quant_Err[0.01]
    df_last_prono['e_pred_99'] = df_last_prono['Y_predic'] + quant_Err[0.99]
    
    df_Obs = df_obs[['h_obs',]]    # df_Obs = df_Obs.rename(columns={'NuevaPalmira':'h_obs'})
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
        uploadPronoSeries(series,cal_id=id_modelo_corr,forecast_date=fecha_emision,outputfile="productos/prono_"+Nombre_out+".json",responseOutputFile="productos/pronoresponse.json")
    else:
        data = prono2json(series,forecast_date=fecha_emision)
        outputjson(data,"productos/prono_"+Nombre_out+".json")


for est in Dic_Estaciones['Estaciones'].keys():
    print(est)
    corrigeSalidaModelo(est,Dic_Estaciones,plots=False,upload=False,output_csv=True)

