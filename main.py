import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_excel('./base/dataset.xlsx')

# verificando object types
# print(data.dtypes)

#Criando dict passando coluna Portatil com chave
alterando = {'Portatil': {'Smartphone': 1, 'Tablet': 2}}

# Inserir as alterações no data
data.replace(alterando, inplace=True)


y = data.Portatil
X = data.drop(columns= ['Portatil'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
# o computador vai aprender
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

#prediter ( 0 computador será testado)
resp_pc = clf.predict(X_test)
gabarito = y_test

print(f'{resp_pc}')
print(f'{gabarito.values}')

print(f'Precisão: {str(metrics.precision_score(gabarito, resp_pc))}')



