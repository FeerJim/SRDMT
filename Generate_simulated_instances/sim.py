import pandas as pd
import math

# Reading of data

def read_data(file_name):
    ''' 
    - Input:
    file_name: example.csv

    - Output:
    df: a DataFrame object
    '''
    df = pd.read_csv(file_name)
    return df

# --------------------------------------------------
# Instances with equal processing time (geometric mean)
# --------------------------------------------------

def inst_new_duration(instancias,file = None):
    for inst in instancias:
        file_name = '../instancias/'+inst+'.csv' # object with file name saved in the folder instancias
        df = read_data(file_name) # read file
        # Calculate the new duration
        df.Duracion = round(df.Duracion.mean()/60,0)*60
        df.Minutos = round(df.Minutos,0)
        df.loc[df["Minutos"] == 60.0,'Horas'] = df[df['Minutos']==60.0].Horas + 1
        df.loc[df["Minutos"] == 60.0,'Minutos'] = 0.0
        df.to_csv('../instancias/'+'p_'+inst+'.csv') # save the new instance

# --------------------------------------------------
# Summary of the instances
# --------------------------------------------------

def summary_inst(instancias,file = None):
    data_output = [inst for inst in instancias]
    df_output = pd.DataFrame(data_output,columns = ['Instance'])
    df_output['numJobs'] = 0
    df_output['numPeriods'] = 0
    df_output['meanProcessingTime'] = 0
    for inst in instancias:
        file_name = '../instancias/'+inst+'.csv' # definimos el nombre
        df = read_data(file_name) # leemos el archivo

        # Guardamos parametros de la solucion
        df_output.loc[df_output.Instance == inst,['numJobs']] = len(df)
        df_output.loc[df_output.Instance == inst,['numPeriods']] = math.ceil(max(df.Horas+df.Minutos/60+df.Duracion/3600 + 20/60))-min(df.Horas)
        df_output.loc[df_output.Instance == inst,['meanProcessingTime']] = round(df.Duracion.mean(),0)

    if file == None:
        df_output.to_csv('Summary.csv')
    else:
        df_output.to_csv(file +'.csv')