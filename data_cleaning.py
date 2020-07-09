import pandas as pd
import glob
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.pipeline import Pipeline

big_df = pd.DataFrame()

for filename in glob.glob("database/Rating_*.xlsx"):

    print("READING DATABASE: " + filename)

    df = pd.read_excel(open(filename, 'rb'), sheet_name="Resultados",
                       header=None)  # Reading SABI Export without index
    df.columns = ['id', 'nombre_x', 'nif', 'nombre', 'provincia', 'calle', 'telefono', 'web', 'desc_actividad', 'cnae',
                  'cod_consolidacion', 'rating_grade_h2', 'rating_grade_h1', 'rating_grade_h0', 'rating_numerico_h2',
                  'rating_numerico_h1', 'rating_numerico_h0', 'modelo_propension_h2', 'modelo_propension_h1',
                  'modelo_propension_h0', 'guo_nombre', 'guo_id_bvd', 'guo_pais', 'guo_tipo', 'estado_detallado',
                  'fecha_cambio_estado', 'fecha_constitucion', 'p10000_h0', 'p10000_h1', 'p10000_h2', 'p20000_h0',
                  'p20000_h1', 'p20000_h2', 'p31200_h0', 'p31200_h1', 'p31200_h2', 'p32300_h0', 'p32300_h1',
                  'p32300_h2', 'p40100_mas_40500_h0', 'p40100_mas_40500_h1', 'p40100_mas_40500_h2', 'p40800_h0',
                  'p40800_h1', 'p40800_h2', 'p49100_h0', 'p49100_h1', 'p49100_h2']
    df['h0_anio'] = 2017
    df = df.fillna('')
    df = df.drop(df.index[0])  # Dropping SABI variable names.
    df['nif'] = df.nif.str.upper()  # CONVERTING cif INTO UPPERCASE

    for partida in ['p10000_h0', 'p10000_h1', 'p10000_h2', 'p20000_h0', 'p20000_h1', 'p20000_h2', 'p31200_h0',
                    'p31200_h1', 'p31200_h2', 'p32300_h0', 'p32300_h1', 'p32300_h2', 'p40100_mas_40500_h0',
                    'p40100_mas_40500_h1', 'p40100_mas_40500_h2', 'p40800_h0', 'p40800_h1', 'p40800_h2', 'p49100_h0',
                    'p49100_h1', 'p49100_h2']:
        df[partida] = pd.to_numeric(df[partida], errors='coerce').fillna(0) - 0.005

        df['nif_normalizado'] = df['nif'].str[-8:]
    big_df = big_df.append(df, ignore_index=True)

df = big_df
df['target_status'] = [0 if i in ['Activa', ''] else 1 for i in df['estado_detallado']]  # 0 si Activa, 1 si algo raro!

df = df[['cnae','p49100_h1','p40800_h1','p40100_mas_40500_h1','p31200_h1','p32300_h1', 'p10000_h1', 'p20000_h1',
         'target_status']]

df.loc[:,'p49100_mas_40800_h1'] = df['p49100_h1'] + df['p40800_h1']

df = df[(df['p40100_mas_40500_h1'] != 0) | (df['p49100_mas_40800_h1'] != 0) | (df['p20000_h1'] !=0 )]

df.drop(columns='p49100_mas_40800_h1', inplace=True)

df_h2 = big_df

df_h2.loc[:,'p49100_mas_40800_h1'] = df_h2['p49100_h1'] + df_h2['p40800_h1']
df_h2.loc[:,'p49100_mas_40800_h2'] = df_h2['p49100_h2'] + df_h2['p40800_h2']

df_h2 = df_h2[((df_h2['p40100_mas_40500_h1'] == 0) & (df_h2['p40100_mas_40500_h2'] != 0))
               | ((df_h2['p40100_mas_40500_h1'] == 0) & (df_h2['p49100_mas_40800_h2'] != 0))
               | ((df_h2['p20000_h1'] == 0) & (df_h2['p20000_h2'] != 0))]

df_h2 = df_h2[['cnae','p49100_h2','p40800_h2','p40100_mas_40500_h2','p31200_h2','p32300_h2', 'p10000_h2', 'p20000_h2',
         'target_status']]

df_h2.columns = ['cnae','p49100_h1','p40800_h1','p40100_mas_40500_h1','p31200_h1','p32300_h1', 'p10000_h1', 'p20000_h1',
         'target_status']

df_global = pd.concat([df, df_h2], ignore_index=True)


# Aplicacion de la clase SMOTENC para balancear la clase minoritaria en el conjunto de datos.

over = SMOTENC(sampling_strategy=1,k_neighbors=8, categorical_features=[0], n_jobs=-1)
under = RandomUnderSampler(sampling_strategy=1)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

X = df_global.drop(['target_status'], axis=1)
y = df_global['target_status']

X_res, y_res = pipeline.fit_resample(X, y)

df_smotenc = pd.concat([X_res, y_res], axis=1)

# Se filtra el nuevo dataset, descartando aquellos valores de CNAE que son '' (string vacio)

df_smotenc = df_smotenc[df_smotenc['cnae'] != '']

# El dataset se almacena en el fichero 'clean_dataset'.

df_smotenc.to_csv('./clean_dataset.csv', index = False)
