import sys, getopt
import requests#, psycopg2
import pandas as pd
import numpy as np
import logging
import json

from pytz import timezone

import datetime
from datetime import timedelta,datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

localtz = timezone('America/Argentina/Buenos_Aires')

import locale

# Configurar el idioma en español
locale.setlocale(locale.LC_TIME, 'es_ES.utf8')
                 
from Funciones_V2 import getLastProno, getObs, eliminaSaltos, prono2serie
from Funciones_V2 import uploadPronoSeries, prono2json, outputjson, outputcsv

file_config_estaciones = 'Estaciones.json'

with open(file_config_estaciones, 'r', encoding="utf-8") as archivo:
        config_estaciones = json.load(archivo)

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

def plotFinal(df_obs,
              df_sim,
              nameout='productos/plot_final.png',
              text_xoffset=(-2,-5),
              ylim=(-1,2.5),
              obs_label='Nivel Observado',
              extraObs=None,
              extraObsLabel='Nivel Observado 2', 
              fecha_emision = None, 
              bandaDeError=('e_pred_01','e_pred_99'),
              obsLine=False,
              nombre_estacion="Estación",
              niveles_alerta={}, 
              cero=0):
    
    df_sim.index = df_sim.index.tz_convert(tz="America/Argentina/Buenos_Aires")

    if not isinstance(df_obs,type(None)):
        df_obs.index = df_obs.index.tz_convert(tz="America/Argentina/Buenos_Aires")
    
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(1, 1, 1)

    # ax.title('Previsión de niveles a corto plazo en el puerto')

    ################## Nivel Simulado / Pronosticado
    ax.plot(df_sim.index, df_sim['Y_predic'], '-',color='b',label='Nivel Pronosticado (*)',linewidth=3)

    # Bandas de error
    ax.plot(df_sim.index, df_sim[bandaDeError[0]],'-',color='k',linewidth=0.5,alpha=0.75,label='_nolegend_')
    ax.plot(df_sim.index, df_sim[bandaDeError[1]],'-',color='k',linewidth=0.5,alpha=0.75,label='_nolegend_')
    ax.fill_between(df_sim.index,df_sim[bandaDeError[0]], df_sim[bandaDeError[1]],alpha=0.1,label='Banda de error')

    ################## Nivel Observado
    if not isinstance(df_obs, type(None)):
        ax.scatter(df_obs.index, df_obs['h_obs'],color='k',label=obs_label)

    if not isinstance(extraObs,type(None)):
        #if nombre_estacion != 'NuevaPalmira': extraObs.index = extraObs.index.tz_convert("America/Argentina/Buenos_Aires")
        # ax.plot(extraObs.index, extraObs['h_obs'],'o',color='grey',label=extraObsLabel,linewidth=3,alpha=0.5)
        # ax.plot(extraObs.index, extraObs['h_obs'],'-',color='grey',linewidth=1,alpha=0.5)
        ax.scatter(extraObs.index, extraObs['h_obs'],color='k',label=extraObsLabel)
    
    ################## Niveles de alerta
    xmin=df_sim.index.min()
    xmax=df_sim.index.max()  

    if niveles_alerta.get("aguas_bajas"):
        plt.hlines(niveles_alerta["aguas_bajas"], xmin, xmax, colors='y', linestyles='-.', label='Aguas Bajas',linewidth=1.5)
    if niveles_alerta.get("alerta"):
        plt.hlines(niveles_alerta["alerta"], xmin, xmax, colors='y', linestyles='-.', label='Alerta',linewidth=1.5)
    if niveles_alerta.get("evacuacion"):
        plt.hlines(niveles_alerta["evacuacion"], xmin, xmax, colors='r', linestyles='-.', label='Evacuación',linewidth=1.5)
    
    ################## Fecha emision
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
    offset = 10
    xdisplay = ahora + timedelta(days=1.0)

    ydisplay = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    ydisplay_fech = ylim[0] + (ylim[1] - ylim[0]) * 0.1

    ax.annotate('Pronóstico a 4 días',
        xy=(xdisplay, ydisplay), xytext=(text_xoffset[0]*offset, -offset), textcoords='offset points',
        bbox=bbox, fontsize=18)#arrowprops=arrowprops
    xdisplay = ahora - timedelta(days=2)
    ax.annotate('Días pasados',
        xy=(xdisplay, ydisplay), xytext=(text_xoffset[1]*offset, -offset), textcoords='offset points',
        bbox=bbox, fontsize=18)
    ax.annotate('Fecha de emisión',
        xy=(ahora, ydisplay_fech),fontsize=15, xytext=(ahora+timedelta(days=0.3), ydisplay_fech), arrowprops=dict(facecolor='black',shrink=0.05))
    
    fig.subplots_adjust(bottom=0.2,right=0.8)
    plt.figtext(0,0,'          (*) Esta previsión surge de aplicar el Modelo Matemático del Delta del Programa de Hidráulica Computacional (PHC) de la Subgerencia \n          del Laboratorio de Hidráulica (SLH) del Instituto Nacional del Agua (INA), forzado por el caudal pronosticado del río Paraná de acuerdo \n          al Sistema de Información y Alerta Hidrológico (SIyAH-INA) y por el nivel del Río de la Plata en el arco San Fernando - Nueva Palmira \n          pronosticado por el Servicio de Hidrografía Naval (SHN) y el Servicio Meteorológico Nacional (SMN). Es una herramienta preliminar \n          de pronóstico para utilizar en la emergencia hídrica, que se irá ajustando en el tiempo para generar información más confiable. \n \n',fontsize=12,ha="left")
    
    if cero is not None:
        plt.figtext(0,0,'          (**) El cero de la escala de ' + nombre_estacion 
                    + ' corresponde a ' + str(cero+0.53) 
                    +' mMOP / '+ str(cero) 
                    +' mIGN \n',fontsize=12,ha="left")
    
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
    
    df_prono['Y_predic'] = df_prono['Y_predic'].round(2)
    df_prono['Hora'] = df_prono['Hora'].astype(str)
    df_prono['Hora'] = df_prono['Hora'].replace('0', '00')
    df_prono['Hora'] = df_prono['Hora'].replace('6', '06')
    df_prono['Dia'] = df_prono['Dia'].astype(str)
    df_prono['Fechap'] = df_prono['Dia']+' '+df_prono['Hora']+'hrs'
    df_prono = df_prono[['Fechap','Y_predic',]]

    cell_text = df_prono.values.tolist()
    columns = ('Fecha','Nivel',)
    table = plt.table(cellText=cell_text,
                      colLabels=columns,
                      bbox = (1.08, 0, 0.2, 0.5))
    table.set_fontsize(12)

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

def GeneraGrafico(DicEst,plots=False, upload=True, output_csv=False):

    nombre = DicEst['nombre']

    CeroEst = DicEst['cero']
    estacion_id = DicEst['estacion_id']
    serie_obs = DicEst['serie_obs']
    umbral_salto = DicEst['umbral_salto']
    name_file = DicEst['name_file']
    serie_upload = DicEst['serie_upload']

    fig_ylim= tuple(DicEst['fig_ylim'])
    DicNiveles = DicEst['niveles_alerta']
    cal_id_sim = DicEst['cal_id_sim']
    cal_id_save = DicEst['cal_id']

    ## Carga Simulados
    try:
        df_sim, fecha_emision = getLastProno(cal_id_sim,{'estacion_id':estacion_id ,
                                                         'var_id': 2})
    except Exception as e:
        logging.error(e)
        return
    
    
    ## Carga Observados
    f_inicio = df_sim.index.min()
    f_fin = df_sim.index.max()

    # Estaciones a consultar
    df_Obs = getObs(serie_obs,f_inicio,f_fin)

    # Elimina saltos
    df_Obs = eliminaSaltos(df_Obs,umbral_salto)

    ## Union
    df = df_sim.join(df_Obs, how = 'outer')
    df['h_sim'] = df['h_sim'].interpolate(method='linear',limit=4)

    ##Plot
    if plots:
        plotObsVsSim(df)

    df['Y_predic'] = df['h_sim']
    df['Error_pred'] =  df['Y_predic']  - df['h_obs']
    quant_Err = df['Error_pred'].quantile([.001,.05,.95,.999])

    horas_plot = 24*10
    df = df[-horas_plot:]

    # Ajustar los límites según la naturaleza de los errores
    if quant_Err[0.001] > 0 and quant_Err[0.999] > 0:
        limite_inferior = max(quant_Err[0.001], 0.15)
        limite_superior = max(quant_Err[0.999], 0.15)

        df['e_pred_01'] = df['Y_predic'] - limite_inferior  # Restamos el mínimo para evitar que ambos límites estén arriba
        df['e_pred_99'] = df['Y_predic'] + limite_superior
    elif quant_Err[0.001] < 0 and quant_Err[0.999] < 0:
        limite_inferior = min(quant_Err[0.001], -0.15)
        limite_superior = min(quant_Err[0.999], -0.15)

        df['e_pred_01'] = df['Y_predic'] + limite_inferior  # Si ambos errores son negativos, sumamos el menor
        df['e_pred_99'] = df['Y_predic'] - limite_superior  # Restamos el mayor (que es negativo)
    else:
        # Caso normal cuando hay mezcla de errores positivos y negativos
        limite_inferior = min(quant_Err[0.001], -0.15)
        limite_superior = max(quant_Err[0.999], 0.15)
        df['e_pred_01'] = df['Y_predic'] + limite_inferior
        df['e_pred_99'] = df['Y_predic'] + limite_superior



    if quant_Err[0.001] > 0 and quant_Err[0.999] > 0:
        df['e_pred_01'] = df['Y_predic'] - quant_Err[0.001]  # Restamos el cuantil más bajo
        df['e_pred_99'] = df['Y_predic'] + quant_Err[0.999]  # Sumamos el cuantil más alto
    elif quant_Err[0.001] < 0 and quant_Err[0.999] < 0:
        df['e_pred_01'] = df['Y_predic'] + quant_Err[0.001]  # Ambos errores son negativos, se suma el menor
        df['e_pred_99'] = df['Y_predic'] - quant_Err[0.999]  # Se resta el mayor (que es negativo)
    else:
        # Caso normal cuando hay mezcla de errores positivos y negativos
        df['e_pred_01'] = df['Y_predic'] + quant_Err[0.001]
        df['e_pred_99'] = df['Y_predic'] + quant_Err[0.999]

    # SERIE upload 
    series = [prono2serie(df,main_colname="Y_predic",
                          members={'e_pred_01':'p01','e_pred_99':'p99'},
                          series_id=serie_upload)]

    # PLOT FINAL
    plotFinal(df_Obs,
              df,
              nameout='productos/'+name_file+'.png',
              ylim=fig_ylim,
              nombre_estacion=nombre,
              niveles_alerta=DicNiveles,
              cero=CeroEst,
              fecha_emision=fecha_emision)
    
    ## UPLOAD PRONOSTICO
    if upload:
        uploadPronoSeries(series,
                          cal_id=cal_id_save,
                          forecast_date=fecha_emision,
                          outputfile="productos/"+name_file+".json",
                          responseOutputFile="productos/pronoresponse.json")
    else:
        data = prono2json(series,forecast_date=fecha_emision)
        outputjson(data,"productos/"+name_file+".json")
    
    if output_csv:
        outputcsv(df,"productos/"+name_file+".csv")

def runPlan(plan,plots=True,upload=False,output_csv=False):
    planes = {
        # "zarate": GeneraGrafico,
        # "atucha": GeneraGrafico,
        # "lima": GeneraGrafico,
        # "campana": GeneraGrafico,
        # "escobar": GeneraGrafico,
        # "rosario": GeneraGrafico,
        # "vcons": GeneraGrafico,
        # "SanNicolas": GeneraGrafico,
        # "ramallo": GeneraGrafico,
        "SanLorenzo": GeneraGrafico,
        # #"carabelas": GeneraGrafico,
        # "blargo": GeneraGrafico,
    }

    if plan in ("all","All","ALL"):
        for key in planes:
            print(key)
            planes[key](config_estaciones[key],plots,upload,output_csv)
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
