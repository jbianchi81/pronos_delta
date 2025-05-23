# -*- coding: utf-8 -*-

"""Prono_SanFernando.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fgX7obwikorAVqUiU-WF1SKAS98wYH6v
"""

import psycopg2
import datetime
from datetime import timedelta
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')  # comentar para versiones más nuevas de matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import requests
import json

import locale

# set locale for date formatting
locale.setlocale(locale.LC_ALL,"es_AR.utf8")

with open("config.json") as f:
	config = json.load(f)

'''Conecta con BBDD'''
try:
   conn_string = "dbname='%s' user='%s' host='%s' port='%d'" % (config["database"]["dbname"], config["database"]["user"], config["database"]["host"], config["database"]["port"])
   conn = psycopg2.connect(conn_string)
   cur = conn.cursor()
except:
   print( "No se ha podido establecer conexion.")
   exit(1)

ahora = datetime.datetime.now()
DaysMod = 15   
f_fin = ahora
f_inicio = (f_fin - timedelta(days=DaysMod)).replace(hour=0, minute=0, second=0)

plotea = False

unid_margen_derecha = 52 # 85

# OBSERVADOS
#Parametros para la consulta SQL a la BBDD
paramH0 = (f_inicio, f_fin, str(unid_margen_derecha)) 
sql_query = ('''SELECT timestart as fecha, valor as h_obs
              FROM alturas_all
              WHERE  timestart BETWEEN %s AND %s AND unid = %s;''')  #Consulta SQL
df_sferObs = pd.read_sql_query(sql_query, conn, params=paramH0)             #Toma los datos de la BBDD
keys =  pd.to_datetime(df_sferObs['fecha'])
df_sferObs.set_index(keys, inplace=True)

# SIMULADO
paramH1 = (f_inicio, f_fin)
sql_query = ('''SELECT unid as id, timestart as fecha,
                altura_astronomica_ign as h_ast_ign, altura_meteorologica as h_met
                FROM alturas_marea_suma 
                WHERE  timestart BETWEEN %s AND %s AND unid = 1838;''')              #Consulta SQL
df_sferSim = pd.read_sql_query(sql_query, conn, params=paramH1)								#Toma los datos de la BBDD	
keys =  pd.to_datetime(df_sferSim['fecha'])#, format='%Y-%m-%d')                     #Convierte a formato fecha la columna [fecha]
df_sferSim.set_index(keys, inplace=True)
df_sferSim['h_ast_ign'] = df_sferSim['h_ast_ign'] + 0.53
df_sferSim['h_sim0'] = df_sferSim['h_ast_ign'] + df_sferSim['h_met']

indexUnico = pd.date_range(start=df_sferSim['fecha'].min(), end=df_sferSim['fecha'].max(), freq='15min')	    #Fechas desde f_inicio a f_fin con un paso de 5 minutos
df_base = pd.DataFrame(index = indexUnico)								#Crea el Df con indexUnico
df_base.index.rename('fecha', inplace=True)							    #Cambia nombre incide por Fecha
df_base.index = df_base.index.round("15min")

df_base = df_base.join(df_sferSim[['h_ast_ign','h_met','h_sim0']], how = 'left')
df_base = df_base.join(df_sferObs['h_obs'], how = 'left')
df_base['h_obs'] = df_base['h_obs'].interpolate(limit=2)

df_base = df_base.dropna()

if plotea:
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df_base.index, df_base['h_obs'],'o',label='h_obs',linewidth=1)
    ax.plot(df_base.index, df_base['h_obs'],'-',color='k',linewidth=1)
    
    ax.plot(df_base.index, df_base['h_sim0'],'-',label='h_sim: solo Suma',linewidth=2)
    ax.plot(df_base.index, df_base['h_ast_ign'],'-',label='h_ast_ign',linewidth=0.8)
    ax.plot(df_base.index, df_base['h_met'],'-',label='h_met',linewidth=0.8)
    
    plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.7)
    plt.tick_params(axis='both', labelsize=16)
    plt.xlabel('Fecha', size=18)
    plt.ylabel('Nivel [m]', size=18)
    plt.legend(prop={'size':16},loc=2,ncol=2 )
    plt.show()
    plt.close()

## Modelo
train = df_base[:].copy()
var_obj = 'h_obs'
covariav = ['h_ast_ign','h_met']
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
print('Coefficients: \n', lr.coef_)
# The mean squared error
mse = mean_squared_error(Y_test, Y_predictions)
print('Mean squared error: %.5f' % mse)
# The coefficient of determination: 1 is perfect prediction
coefDet = r2_score(Y_test, Y_predictions)
print('r2_score: %.5f' % coefDet)
train['Error_pred'] =  train['Y_predictions']  - train[var_obj]


if plotea:
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    # Plot outputs
    plt.scatter(Y_predictions, Y_test,  label='A01')
    plt.xlabel('h_sim', size=12)
    plt.ylabel('h_obs', size=12)
    plt.legend(prop={'size':16},loc=2,ncol=2 )
    plt.show()

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    # Plot outputs
    plt.scatter(train['Y_predictions'], train['Error_pred'], label='Error')
    plt.xlabel('H Sim', size=12)
    plt.ylabel('Error', size=12)
    plt.legend(prop={'size':16},loc=2,ncol=2 )
    plt.show()


quant_Err = train['Error_pred'].quantile([0.05,.25, .75,0.95])

# Plot final
DaysMod = 1
f_fin = ahora
f_inicio = (f_fin - timedelta(days=DaysMod)).replace(hour=0, minute=0, second=0)
f_fin = (f_fin + timedelta(days=5))

# OBSERVADOS
#Parametros para la consulta SQL a la BBDD
paramH0 = (f_inicio, f_fin, str(unid_margen_derecha)) 
sql_query = ('''SELECT timestart as fecha, valor as altura
                FROM alturas_all
                WHERE  timestart BETWEEN %s AND %s AND unid = %s;''')  #Consulta SQL
df_sferObs = pd.read_sql_query(sql_query, conn, params=paramH0)             #Toma los datos de la BBDD
df_sferObs['fecha'] =  pd.to_datetime(df_sferObs['fecha'])


# SIMULADO
paramH1 = (f_inicio, f_fin)         #Parametros para la consulta SQL a la BBDD
# ~ sql_query = ('''SELECT unid as id, timestart as fecha, 
                # ~ altura_astronomica_ign as h_ast_ign, altura_meteorologica as h_met,
                # ~ timeupdate as fecha_emision
                # ~ FROM alturas_marea_suma_corregida 
                # ~ WHERE  timestart BETWEEN %s AND %s AND unid = 1838;''')              #Consulta SQL
sql_query = ('''SELECT unid as id, timestart as fecha, 
                altura_astronomica_ign as h_ast_ign, altura_meteorologica as h_met,
                timeupdate as fecha_emision
                FROM alturas_marea_suma 
                WHERE  timestart BETWEEN %s AND %s AND unid = 1838;''')              #Consulta SQL
df_simulado = pd.read_sql_query(sql_query, conn, params=paramH1)								#Toma los datos de la BBDD	
df_simulado['fecha'] =  pd.to_datetime(df_simulado['fecha'])#, format='%Y-%m-%d')                     #Convierte a formato fecha la columna [fecha]

# Solo astronomica a cero local SFernando
df_simulado['h_ast_ign'] = df_simulado['h_ast_ign'] + 0.53 


# Guarda en BBDD LOCAL
if False:
    df_sim_guarda = df_simulado[df_simulado['fecha']>ahora].copy()
    mes = str(ahora.month)
    dia = str(ahora.day)
    hora = str(ahora.hour)
    cod = ''.join([mes, dia, hora])

    df_sim_guarda['emision'] = cod
    conn2 = sqlite3.connect('PronostSanFer.sqlite')
    df_sim_guarda.to_sql('Simulado', con = conn2, if_exists='append',index=False)


# Pronostico
covariav = ['h_ast_ign','h_met']
prediccion = lr.predict(df_simulado[covariav].values)
df_simulado['h_sim'] = prediccion

df_simulado['e_pred_05'] = df_simulado['h_sim'] + quant_Err[0.05]
df_simulado['e_pred_25'] = df_simulado['h_sim'] + quant_Err[0.25]
df_simulado['e_pred_75'] = df_simulado['h_sim'] + quant_Err[0.75]
df_simulado['e_pred_95'] = df_simulado['h_sim'] + quant_Err[0.95]



# PLOT
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(1, 1, 1)


ax.plot(df_simulado['fecha'], df_simulado['h_sim'], '-',color='b',label='Nivel Pronosticado (*)',linewidth=3)

ax.plot(df_sferObs['fecha'], df_sferObs['altura'],'o',color='k',label='Nivel Observado',linewidth=3)
ax.plot(df_sferObs['fecha'], df_sferObs['altura'],'-',color='k',linewidth=1,label="_altura")


ax.plot(df_simulado['fecha'], df_simulado['e_pred_05'],'-',color='k',linewidth=0.5,alpha=0.75,label="_nolegend_")
ax.plot(df_simulado['fecha'], df_simulado['e_pred_95'],'-',color='k',linewidth=0.5,alpha=0.75,label="_nolegend_")
ax.fill_between(df_simulado['fecha'],df_simulado['e_pred_05'], df_simulado['e_pred_95'],alpha=0.1,label='Banda de error')


# Lineas: 1 , 1.5 y 2 mts
xmin=df_simulado['fecha'].min()
xmax=df_simulado['fecha'].max()


plt.hlines(3.5, xmin, xmax, colors='r', linestyles='-.', label='Evacuación',linewidth=1.5)
plt.hlines(3, xmin, xmax, colors='y', linestyles='-.', label='Alerta',linewidth=1.5)


# fecha emision
plt.axvline(x=df_simulado['fecha_emision'].max(),color="black", linestyle="--",linewidth=2)#,label='Fecha de emisión')

bbox = dict(boxstyle="round", fc="0.7")
arrowprops = dict(
    arrowstyle="->",
    connectionstyle="angle,angleA=0,angleB=90,rad=10")
offset = 10


xdisplay = ahora + timedelta(days=1.0)
ax.annotate('Pronóstico a 4 días',
    xy=(xdisplay, -0.8), xytext=(-8*offset, -offset), textcoords='offset points',
    bbox=bbox, fontsize=18)#arrowprops=arrowprops

xdisplay = ahora - timedelta(days=0.8)
ax.annotate('Días pasados',
    xy=(xdisplay, -0.8), xytext=(-8*offset, -offset), textcoords='offset points',
    bbox=bbox, fontsize=18)


ax.annotate('Fecha de emisión',
    xy=(df_simulado['fecha_emision'].max() - timedelta(days=0.01), -0.35),fontsize=15, xytext=(df_simulado['fecha_emision'].max() + timedelta(days=0.45), -0.30), arrowprops=dict(facecolor='black',shrink=0.05))


nombre_estacion = 'San Fernando'
cero = -0.53
ylim=(-1,4)


fig.subplots_adjust(bottom=0.205,right=0.7)
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
plt.legend(prop={'size':18},loc=2,ncol=2 )
plt.title('Previsión de niveles a corto plazo en ' + nombre_estacion,fontsize=20)


#### TABLA
h_resumne = [0,6,12,18]
df_prono = df_simulado[df_simulado['fecha'] > ahora ].copy()
df_prono.set_index(df_prono['fecha'], inplace=True)
df_prono['Hora'] = df_prono.index.hour
df_prono['Dia'] = df_prono.index.day
df_prono = df_prono[df_prono['Hora'].isin(h_resumne)].copy()

#print(df_prono)
df_prono['Y_predic'] = df_prono['h_sim'].round(2)
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
    #print(cell_text)

columns = ('Fecha','Nivel',)
table = plt.table(cellText=cell_text,
                  colLabels=columns,
                  bbox = (1.08, 0, 0.2, 0.5))
table.set_fontsize(12)
#table.scale(2.5, 2.5)  # may help


date_form = DateFormatter("%Hhrs \n %d-%b")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_minor_locator(mdates.HourLocator((0,6,12,18,))) #3,9,15,21,)))


## FRANJAS VERTICALES
df_simulado['horas'] =  df_simulado['fecha'].dt.hour
list0hrs = df_simulado[df_simulado['horas']==0]['fecha'].tolist()
ax.axvspan(list0hrs[0], list0hrs[1], alpha=0.1, color='grey')
if len(list0hrs) >= 4:
    ax.axvspan(list0hrs[2], list0hrs[3], alpha=0.1, color='grey')


#plt.show()

########################################################################################################################
nameout = 'productos/Prono_SanFernando.png'
plt.savefig(nameout, format='png')# , dpi=200, facecolor='w', edgecolor='w',bbox_inches = 'tight', pad_inches = 0
plt.close()

# pone timezone a fecha
for i in df_simulado.index:
  df_simulado.at[i,"fecha"] = df_simulado.at[i,"fecha"].tz_localize("America/Argentina/Buenos_Aires")

# df para UPSERT
df_para_upsert = df_simulado[['fecha','h_sim']].rename(columns = {'fecha':'timestart', 'h_sim':'valor'},inplace = False)
df_para_upsert['qualifier'] = 'main'
df_para_upsert = pd.concat([df_para_upsert, df_simulado[['fecha','e_pred_05']].rename(columns = {'fecha':'timestart', 'e_pred_05': 'valor'})], ignore_index=True)
df_para_upsert['qualifier'].fillna(value='p05',inplace=True)
df_para_upsert = pd.concat([df_para_upsert, df_simulado[['fecha','e_pred_25']].rename(columns = {'fecha':'timestart', 'e_pred_25': 'valor'})], ignore_index=True)
df_para_upsert['qualifier'].fillna(value='p25',inplace=True)
df_para_upsert = pd.concat([df_para_upsert, df_simulado[['fecha','e_pred_75']].rename(columns = {'fecha':'timestart', 'e_pred_75': 'valor'})], ignore_index=True)
df_para_upsert['qualifier'].fillna(value='p75',inplace=True)
df_para_upsert = pd.concat([df_para_upsert, df_simulado[['fecha','e_pred_95']].rename(columns = {'fecha':'timestart', 'e_pred_95': 'valor'})], ignore_index=True)
df_para_upsert['qualifier'].fillna(value='p95',inplace=True)
df_para_upsert['timeend'] = df_para_upsert['timestart']  # .map(lambda a : a.isoformat())
# ~ df_para_upsert['timestart'] = df_para_upsert['timestart'].map(lambda a : a.isoformat())
# ~ print(df_para_upsert)
para_upsert = {'forecast_date':df_simulado['fecha_emision'].max().isoformat(),
			 'series': [
				{
					'series_table': 'series',
					'series_id': 26202,
					'pronosticos': json.loads(df_para_upsert.to_json(orient='records',date_format='iso'))
				}
			]}
# ~ print(para_upsert)

# UPSERT Simulado

def uploadProno(data,cal_id,responseOutputFile):
    response = requests.post(
        config["api"]["url"] + '/sim/calibrados/' + str(cal_id) + '/corridas',
        data=json.dumps(data),
        headers={'Authorization': 'Bearer ' + config["api"]["token"], 'Content-type': 'application/json'},
    )
    print("prono upload, response code: " + str(response.status_code))
    print("prono upload, reason: " + response.reason)
    if(response.status_code == 200):
        if(responseOutputFile):
            outresponse = open(responseOutputFile,"w")
            outresponse.write(json.dumps(response.json()))
            outresponse.close()

uploadProno(para_upsert,432,"productos/prono_sanFernando.json")


