import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



db = pd.read_csv("db.csv")
db.drop('education', axis=1)


#Tratamento dos dados#

db.dropna(axis=0, inplace=True) #Remove as linhas que contenham valores nulos

#Definindo variaveis para regressão
'''Como uma regressão logica funciona basicamente como uma função y = a + b1.x1 + b2.x2 + ... + bn.xn, 
   onde b é uma constante e x são os paramentros (colunas do dataset), precisamos definir estas variaveis.'''

x = db.drop('TenYearCHD',axis=1) #
y = db['TenYearCHD']


'''Aqui definimos a divisão do dataset para treinamento (6% p/ teste e 94% para treinamento)
   O random_state é opcional e serve apenas para manter a constancia na seed do treinamento.'''

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.06, random_state=42);



'''Definimos o modelo de treinamento e o número máximo de iterações e o preenchemos 
   com o x_train e y_train (objetos retornados da função train_test_split)  '''

model = LogisticRegression(max_iter=21200)
model.fit(x_train,y_train)

#Simples verificação de precisão
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", round(accuracy,4)*100,"%")
#Fim da verificação
isRunning = True

while(isRunning):
    new_data = []
    sex = input("Sexo: (0 = Feminino; 1 = Masculino)")
    new_data.append(float(sex))
    age = input("Idade: ")
    new_data.append(float(age))
    education = input("Grau de escolaridade: (1-min, 4-max) ")
    new_data.append(float(education))
    currentSmoker = input("É fumante? (0-Não, 1-Sim) ")
    new_data.append(float(currentSmoker))
    cigsPerDay= input("Quantos cigarros por dia? ")
    new_data.append(float(cigsPerDay))
    bpMeds = input("Usa médicamento para pressão? (0-não, 1-sim) ")
    new_data.append(float(bpMeds))
    prevalentStroke = input("Já teve ataque cardiaco? (0-não, 1-sim) ")
    new_data.append(float(prevalentStroke))
    prevalentHyp = input("É hipertenso? (0-não, 1-sim) ")
    new_data.append(float(prevalentHyp))
    diabetes = input("Possui diabetes? (0-não, 1-sim) ")
    new_data.append(float(diabetes))
    totChol = input("Nível de colesterol: ")
    new_data.append(float(totChol))
    sysBP = input("Pressão sistólica: ")
    new_data.append(float(sysBP))
    diaBP = input("Pressão diastólica: ")
    new_data.append(float(diaBP))
    imc = input("Qual o seu IMC? ")
    new_data.append(float(imc))
    heartRate = input("Qual a sua frequência cardiaca? ")
    new_data.append(float(heartRate))
    glicose = input("Qual o nível de glicose? ")
    new_data.append(float(glicose))

    info = np.array(new_data).reshape(1,-1)

    prediction = model.predict(info)

    if prediction == 0: result = "Não há risco de doença cardiaca nos próximos 10 anos."
    elif prediction > 0 : result = "Há risco de doença cardiaca nos próximos 10 anos"

    print(result)

    choice = input("Deseja continuar? (s/n)")
    if choice == "s" : new_data.clear()
    if choice == "n" :  isRunning = False