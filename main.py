import os
import csv 
from keras.models import Sequential
from keras.layers import Dense 
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import json 

print('############')
def list_files_and_directories(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')
            
model_dir = os.environ['SM_MODEL_DIR']

with open(model_dir + '/output_model.txt', 'w') as f:
    f.write('Ciao sono il modello')
    
 # ( 1 )
output_dir = os.environ['SM_MODEL_DIR']
with open(output_dir + '/output.txt', 'w') as f:
    f.write('Ciao sono i log del terminale')
    
input_dir = os.environ['SM_INPUT_DIR']
list_files_and_directories(input_dir)

with open(input_dir + "/data/training/card_transdata.csv", 'r') as fp:
    lines = len(fp.readlines())
    print('######### Total Number of lines:', lines)
    with open(output_dir + "/output_lines.txt", 'w') as f:
        f.write(f'CIao le line sono: {lines}')
        
model = Sequential(
        layers=(
            Dense(units=3,activation='relu',input_dim = 7),
            Dense(units=2,activation='relu'),
            Dense(units=1,activation='sigmoid')
        ),
        name = 'PrimoModello'
    )

model.summary()

df = pd.read_csv(input_dir + "/data/training/card_transdata.csv")
y = df.fraud.values
y = y.astype(int)
print(y.shape)
print(y.dtype)

x = df.drop('fraud', axis=1)
x = x.values
print(x.shape)
print(x.dtype)

x_train ,x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3)

model.compile(optimizer='adam', loss='binary_crossentropy',metrics = ['accuracy'])

print("############### " + os.environ['SM_HPS'])
epoche = json.loads(os.environ['SM_HPS'])
print("############### " + json.dumps(epoche))

print('Fase di training')
history = model.fit(x_train,y_train,batch_size= 128, epochs=epoche['epochs'],validation_split=0.2)


model.save(model_dir + '/output_model.keras')

print('Fase di test')
y_hat = model.predict(x_test)
y_hat = y_hat.reshape(-1)
threshold = 0.6
y_hat[y_hat>= threshold] = 1
y_hat[y_hat<1]=0
y_hat = y_hat.astype(int)

#y_hat = np.round(y_hat).astype(int)
# print(y_hat.shape)
# print(y_test.shape)

# print(f'1° campione di test: {x_test[0]}')
# print(f'Capione di previsione: {y_test[0]}')
# print(f'Classe prevista: {int(round(y_hat[0],0))}')
# print(f'Classe reale: {y_test[0]}')

df_finale = pd.DataFrame({       
    'campione': y_test,
    'previsione': y_hat,
    'reale': y_test
})

print(df_finale)
