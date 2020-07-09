
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from database.utils.pipe_modules import *

import warnings

warnings.filterwarnings("ignore")

# Modelo Random Forest

df_rf = pd.read_csv("clean_dataset.csv", dtype={"cnae": str})

# Los codigos de la columna del CNAE se expresa en terminos de las clases A, B, C, ... a la que pertenecen
df_rf = df_rf.dropna(subset = ['cnae'])
df_rf['cnae'] = df_rf['cnae'].astype(int)

def conditions(df_rf):
    if df_rf['cnae'] < 510:
        return 'A'
    elif df_rf['cnae'] >= 510 and df_rf['cnae'] < 1011:
        return 'B'
    elif df_rf['cnae'] >= 1011 and df_rf['cnae'] < 3512:
        return 'C'
    elif df_rf['cnae'] >= 3512 and df_rf['cnae'] < 3600:
        return 'D'
    elif df_rf['cnae'] >= 3600 and df_rf['cnae'] < 4110:
        return 'E'
    elif df_rf['cnae'] >= 4110 and df_rf['cnae'] < 4511:
        return 'F'
    elif df_rf['cnae'] >= 4511 and df_rf['cnae'] < 4910:
        return 'G'
    elif df_rf['cnae'] >= 4910 and df_rf['cnae'] < 5510:
        return 'H'
    elif df_rf['cnae'] >= 5510 and df_rf['cnae'] < 5811:
        return 'I'
    elif df_rf['cnae'] >= 5811 and df_rf['cnae'] < 6411:
        return 'J'
    elif df_rf['cnae'] >= 6411 and df_rf['cnae'] < 6810:
        return 'K'
    elif df_rf['cnae'] >= 6810 and df_rf['cnae'] < 6910:
        return 'L'
    elif df_rf['cnae'] >= 6910 and df_rf['cnae'] < 7711:
        return 'M'
    elif df_rf['cnae'] >= 7711 and df_rf['cnae'] < 8411:
        return 'N'
    elif df_rf['cnae'] >= 8411 and df_rf['cnae'] < 8510:
        return 'O'
    elif df_rf['cnae'] >= 8510 and df_rf['cnae'] < 8610:
        return 'P'
    elif df_rf['cnae'] >= 8610 and df_rf['cnae'] < 9001:
        return 'Q'
    elif df_rf['cnae'] >= 9001 and df_rf['cnae'] < 9411:
        return 'R'
    elif df_rf['cnae'] >= 9411 and df_rf['cnae'] < 9700:
        return 'S'
    elif df_rf['cnae'] >= 9700 and df_rf['cnae'] < 9900:
        return 'T'
    elif df_rf['cnae'] >= 9900:
        return 'S'
    else:
        return 'Unknown'
    
df_rf['cnae_reduced'] = df_rf.apply(conditions, axis=1)

# Se hace un get_dummies de la columna 'cnae_reduced' y se elimina la columna 'cnae'

df_rf = pd.get_dummies(df_rf, prefix='cnae', prefix_sep='_')

df_rf = df_rf.drop(['cnae'], axis=1)

# Se crean los conjuntos de datos de entrenamiento y de test
X = df_rf.drop(['target_status'], axis=1)
y = df_rf['target_status']
X_train, X_test, y_train, y_test = train_test_split( X, y , test_size = 0.2 , random_state = 42, stratify = y)

# Clasificador Random Forest
rf_class = RandomForestClassifier(n_estimators=50, random_state = 1234, class_weight={0:0.1, 1:0.9}, n_jobs=-1)

model = rf_class.fit(X_train, y_train)

y_pred = model.predict(X_train) 
y_pred_test = model.predict(X_test) 

# Métricas para el modelo Random Forest

from sklearn.metrics import accuracy_score

print("Accuracy train: {0}".format(accuracy_score(y_pred, y_train)))

print("Accuracy test: {0}".format(accuracy_score(y_pred_test, y_test)))

from sklearn.metrics import classification_report

print("classification report for train")
print(classification_report(y_train, y_pred))

print("classification report for test")
print(classification_report(y_test, y_pred_test))

# AUC

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_test)
print("AUC: {0: .4f}".format(metrics.auc(fpr, tpr)))

print ("SAVING THE PERSISTENT MODEL...")
from joblib import dump#, load
dump(model, 'database/Rating_EnhancedModel.joblib')