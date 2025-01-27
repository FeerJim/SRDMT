import pandas as pd
import os

# Folders to Save Results

try:
    os.makedirs('logs')
except OSError:
    if not os.path.isdir('logs'):
        Raise
try:
    os.makedirs('Results')
except OSError:
    if not os.path.isdir('Results'):
        Raise
try:
    os.makedirs('Solution')
except OSError:
    if not os.path.isdir('Solution'):
        Raise

# Data Reading

def read_file(file_name):
    ''' 
    - Input:
        
        file_name: A file in CSV format.

    - Output:
    
    df: A database in DataFrame format.
    '''
    df = pd.read_csv(file_name)
    return df

# Function to Calculate Vectors S and F for Customer Service Start and End Times

def calculate_SF(dfm, _p = None, L = 60, T = None):
    ''' 
    - Input:
    
    dfm: A DataFrame containing the data obtained from the reading process.
    
    T: If T != None, we change L to the maximum ending time to processe all jobs.
    
    L: lenght of a period
    
    _p: by default None, if change to 0 use equal duration for each job (mean of durations). Use _p = int_value to specify the duration in minutes.

    - Output:
    
    S: A nested dictionary indexed by customers and waiting times, using possible service start times as values.
    
    F: A nested dictionary indexed by customers and waiting times, using possible service end times as values.
    
    P_s: A list of start times for each period in hour format.
    
    S_r: A list of start times for each period in minute format.
    '''
    # ID of customers
    n,_ = dfm.shape
    dfm['ID'] = range(n)
    N = list(dfm['ID'])
    
    # Maximum Waiting Time
    m = 20
    
    # Release time
    if _p != None:
        if _p == 0:
            p = round(dfm.Duracion.mean()/60,0)
        else:
            p = _p
        s = {i: float(dfm[dfm['ID'] == i].Horas*60+dfm[dfm['ID'] == i].Minutos) for i in N}
        f = {i: float(dfm[dfm['ID'] == i].Horas*60+dfm[dfm['ID'] == i].Minutos+p) for i in N}
    else:
        s = {i: float(dfm[dfm['ID'] == i].Horas*60+dfm[dfm['ID'] == i].Minutos) for i in N}
        f = {i: float(dfm[dfm['ID'] == i].Horas*60+dfm[dfm['ID'] == i].Minutos+dfm[dfm['ID'] == i].Duracion/60) for i in N}

    # Dictionaries S and F
    S = {i: {j: s[i]+j for j in range(m+1)} for i in N}
    F = {i: {j: f[i]+j for j in range(m+1)} for i in N}
    
    # Starts time of periods    
    P_s = list(range(int(min([max(v.values()) for ind,v in S.items() ])/60),int((max([max(v.values()) for ind,v in F.items() ]))/60)+1))
    S_r = [ p*60 for p in P_s]
    if T != None:
        L = max(S_r)
    S_r = list(range(min(S_r),max(S_r)+1,L))
    P_s = [p/60 for p in S_r]
    
    return S,F,P_s,S_r


# Parameters required for the EJIAM formulation

def sets_EJIAM(dfm,_p = None,L = 60):
    ''' 
    - Input:
    
    dfm: A DataFrame containing the data obtained from the reading process.
    
    p: Processing duration (constant for all customers)
    
    L: Length of each period

    - Output:
    
    r: dictionary indexed by customers using releases time as values.
    
    d: dictionary indexed by customers using as values the deadlines.
    
    t: list with the periods (hours)
    
    P: list with the start times of each period in minutes format.
    
    H: all possible start or end times for processing a job (customers)
    
    I: dictionary whose values are the extremes of the intervals formed by the times in H.
    '''
    # Customers ID
    n,_ = dfm.shape
    dfm['ID'] = range(n)
    N = list(dfm['ID'])
    
    # Maximum Waiting Time
    gamma = 20
    
    # Calculate the average duration (by default 10 minutes)
    if _p != None:
        p = round(dfm.Duracion.mean()/60,0)
    else:
        p = 10

    # Releases times and Deadlines
    r = {i: float(dfm[dfm['ID'] == i].Horas*60+dfm[dfm['ID'] == i].Minutos) for i in N}
    r_aux = {r[i] for i in r.keys()}
    d = {i: float(dfm[dfm['ID'] == i].Horas*60+dfm[dfm['ID'] == i].Minutos+gamma+p) for i in N}
    
    # Beginning of periods
    P = list(range(int(min(r.values())/60),int(max(d.values())/60)+2 ))
    t = [ q*60 for q in P]
    t = list(range(min(t),max(t)+L+1,L))
    P = [ q/60 for q in t]
    
    # Intervals
    Z = max(gamma,L)+1
    H = {i + k*p for i in r_aux for k in range(-Z-1,Z) if i + k*p >= min(r.values()) and i + (k+1)*p <= min(max(r.values()) + 2*n*p,max(d.values()))}
    H |= {t[i] + k*p for i in range(len(t)-1) for k in range(-Z-1,Z) if t[i] + k*p >= min(r.values()) and t[i] + (k+1)*p <= min(max(r.values()) + 2*n*p,max(d.values()))}
    H = set(sorted(H))
    I = {}
    aux3 = 1
    for i in sorted(H):
        I[aux3] = pd.Interval(i,i+p,closed = 'left')
        aux3 += 1
    
    return r,d,t,P,H,I