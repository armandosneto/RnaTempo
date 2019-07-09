from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

print("IA que Através de alguns dados meteriológicos, afirma se irá chover ou não\n")
X=[0,0,0,0,0,0,0]
X[0] = input('Temperatura na hora da medição: ')
X[1] = input('Umidade: ')
X[2] = input('Pressão atmosférica: ')
X[3] = input('Velocidade do vento: ')
X[4] = input('Direção do Vento: ')
X[5] = input('Nebulosidade: ')
X[6] = input('Temperatura mínima do dia: ')

data = {'temp_instantanea':[X[0]],
'umidade': [X[1]],'pressao': [X[2]],
'velocidade_vento': [X[3]],
'direcao_vento': [X[4]],
'nebulosidade': [X[5]],
'temperatura_min_dia':[X[6]]}

Y = pd.read_csv("dados.csv", sep =';')
exemplo = Y.iloc[:,0:7]
df=pd.DataFrame(data, columns=['temp_instantanea','umidade','pressao','velocidade_vento','direcao_vento','nebulosidade','temperatura_min_dia'])

scaler = StandardScaler().fit(exemplo)
df = scaler.transform(df)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("REDE NEURAL CARREGADA DO DISCO0")

y_pred = loaded_model.predict(df)

if y_pred >0.5:
    print('\n\n\nVai chover!')
else:
    print('\n\n\nNão Chove hoje!')

