from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from database.utils.pipe_modules import *

import warnings

warnings.filterwarnings("ignore")


df_rf = pd.read_csv("clean_dataset.csv", dtype={"cnae": str})

X = df_rf.drop(['target_status'], axis=1)
y = df_rf['target_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Categrical features to pass down the categorical pipeline
categorical_features = ["sector"]

# Numerical features to pass down the numerical pipeline
#numerical_features = X.select_dtypes(['int', 'float']).columns.tolist()
numerical_features = ["p40100_mas_40500_h1","p10000_h1","p20000_h1"]

preprocessing = Pipeline([("CNAE_Transformer", CNAE_Transformer()), ("Mean_Imputer", Mean_Imputer()),
                          ])
categorical_pipeline = Pipeline(steps=[('preprocessing', preprocessing),
                                       ('cat_selector', FeatureSelector(categorical_features)),
                                       ('one_hot_encoder', OneHotEncoder(sparse=False, handle_unknown='ignore'))])

numerical_pipeline = Pipeline(steps=[('preprocessing', preprocessing),
                                     ('cat_selector', FeatureSelector(numerical_features)),
                                     ('standardScaler', StandardScaler())
                                     ])

full_pipeline = FeatureUnion(transformer_list=[('categorical_pipeline', categorical_pipeline),

                                               ('numerical_pipeline', numerical_pipeline)])

full_pipeline_m = Pipeline(steps=[('full_pipeline', full_pipeline),

                                  ('model', RandomForestClassifier(n_estimators=50, random_state=1234,
                                                                   class_weight={0: 0.1, 1: 0.9}, n_jobs=-1))])

full_pipeline_m.fit(X_train, y_train)

y_pred = full_pipeline_m.predict(X_train)
y_pred_test = full_pipeline_m.predict(X_test)

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


print("SAVING THE PERSISTENT MODEL...")
from joblib import dump  # , load

dump(full_pipeline_m, 'database/Rating_EnhancedModel.joblib')
