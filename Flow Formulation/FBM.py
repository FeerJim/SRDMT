import time
from   gurobipy   import *
import numpy  as np
import pandas as pd
import math as math
import os

# Folders for save the results
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

# --------------------------------------------------
# FBM Formulation
# --------------------------------------------------

def FBM_model(file_name,K,output = 0, T_limit = None, focus = None, presolve = None, method = None, log_name = None, seed = None, cutsPlanes = None, concurrentMIP = None, heuristics = None, cliqueCuts = None, flowCoverCuts = None, zeroHalfCuts = None, mirCuts = None, relaxLiftCuts = None, strongCGCuts = None, infProofCuts = None):
    '''
    - Input:
    
    file_name: (string) The name of the file in CSV format
    
    K: (int) Number of available machines
    
    The following parameters are optional and active or desactive the parameters defined in the class Gurobi:
    
    output -> work similar to the parameter OutputFlag 
    T_limit -> work similar to the parameter change the parameter TimeLimit 
    focus -> work similar to the parameter change the parameter MIPFocus 
    presolve -> work similar to the parameter Presolve
    method -> work similar to the parameter Method
    seed -> work similar to the parameter Seed
    cutsPlanes -> work similar to the parameter Cuts
    concurrentMIP -> work similar to the parameter ConcurrentMIP
    heuristics -> work similar to the parameter Heuristics
    cliqueCuts -> work similar to the parameter CliqueCuts
    flowCoverCuts -> work similar to the parameter FlowCoverCuts
    zeroHalfCuts -> work similar to the parameter ZeroHalfCuts
    mirCuts -> work similar to the parameter MIRCuts
    relaxLiftCuts -> work similar to the parameter RelaxLiftCuts
    strongCGCuts -> work similar to the parameter StrongCGCuts
    infProofCuts -> work similar to the parameter infProofCuts

    - Output:
    m: Gurobi model object
    timpo_pre: Preprocessing time before starting the model
    tiempo_IP: Optimization time
    base: DataFrame object with the results
    '''
    # --------------------------------------------------
    # Data reading
    # --------------------------------------------------
    t1 = time.time()
    base = pd.read_csv(file_name)
    
    # Sort the data
    base.sort_values(['Horas', 'Minutos','Duracion'], ascending=[True, True,True], inplace=True,ignore_index=True)
    base = base[['Ticket_ID','Horas','Minutos','Duracion']]
    
    n = len(base) # Database size  
    N = list(range(n)) # List of jobs
    
    t_espera = 20 # maximum waiting time (gamma)
    
    hora_llegada=np.array(base.Horas) # releasing time (r_j)
    
    minuto_llegada=np.array(base.Minutos)
    a=[hora_llegada[i]*60 + minuto_llegada[i] for i in N]
    
    tiempo_atencion=np.array(base.Duracion) # Processing time (p_j)
    t=[tiempo_atencion[i]/60 for i in N]
    
    J = [j for j in range(t_espera+1)] # List of all possible waiting times 
    
    # --------------------------------------------------
    # Construction of sets T and T_i
    # --------------------------------------------------
    
    # T_i is a dictionary where the keys are the jobs, and each job has a list that contains all possible service intervals
    T_i = {i:[(a[i]+j,a[i]+j+t[i]) for j in J] for i in N}
    
    # T is a list of ordered pairs, each one represents the start and end points of the possible service intervals for all jobs (intervals may be repeated).
    T = [(h,k) for i in N for (h,k) in T_i[i]]
    
    # --------------------------------------------------
    # Construction of sets H and V
    # --------------------------------------------------
    
    H_p = {i for (i,_) in T} | {i for (_,i) in T}

    a_min=int(min([hora_llegada[i] for i in N])) # the earliest start time
    bt_max=int(max([hora_llegada[i] + ((minuto_llegada[i]+t[i]+t_espera)/60) for i in N]))+1 # the last processing completion time

    I = [i*60 for i in range(a_min,bt_max+1)] # List of all the periods of a workday

    CI=set(I)
    OH = list(H_p | set(I))
    OH.sort()

    # DH is a dictionary whose keys will be the nodes of the graph, and each node will be associated with its corresponding value from the list OH
    DH = {i:OH[i-1] for i in range(1,len(OH)+1)}
    H = list(DH.keys())
    
    H_n = [i for i in range(1,len(H))]
    H_n.sort()
    
    H_ultimo = max(H)
    o = H_ultimo+1 # start node o
    d = H_ultimo+2 # end node d
    
    V=H.copy()
    V.append(o)
    V.append(d)
    V.sort
    
    DHI={}
    contador_HI=1
    for i in OH:
        DHI[i]=contador_HI
        contador_HI += 1
    DHI_aux = list(DHI.keys())

    DT={**DHI}
    for i in DHI_aux:
        if i not in H_p:
            del DT[i]
    DT_claves = list(DT.keys())

    DCI={**DHI}
    for i in DHI_aux:
        if i not in CI:
            del DCI[i]
    DCI_claves = list(DCI.keys())
    
    # --------------------------------------------------
    # Construction of arcs and costs
    # --------------------------------------------------
    # Set A1
    A1=[(i,i+1) for i in H_n]
    A1.append((o,1))
    A1.append((H_ultimo,d))
    A1.append((o,d))

    # Set A2
    A2=[(DCI[i],DCI[j]) for i in DCI_claves for j in DCI_claves if i < j]

    # Costs and capacities on the arcs of set A1
    CA1=tupledict()
    for (i,j) in A1:
        if i==o and j==1:
            CA1[i,j]=(0,K)
        elif i==H_ultimo and j==d:
            CA1[i,j]=(0,K)
        elif i==o and j==d:
            CA1[i,j]=(0,K)
        elif DH[i] in I and j==i+1:
            CA1[i,j]=(1,K)
        else:
            CA1[i,j]=(0,K)
    # Costs and capacities on the arcs of set A2
    CA2=tupledict()
    for (i,j) in A2:
        CA2[i,j]=(0,K)
    
    # Costs and capacities on the arcs of set A3
    Ti={}
    for i in N:
        Ti[i]={}
        for (j,l) in T_i[i]:
            Ti[i][(DT[j],DT[l])]={}
            if j in I and l not in I:
                Ti[i][(DT[j],DT[l])]=(1+int(l/60)-int(j/60),1)
            elif l in I:
                Ti[i][(DT[j],DT[l])]=(int(l/60)-int(j/60)-1,1)
            else:
                Ti[i][(DT[j],DT[l])]=(int(l/60)-int(j/60),1)
    
    N_aux=range(len(N)+2)
    Ti[len(N)]={**CA1}
    Ti[len(N)+1]={**CA2}
    A=list(N_aux)
    c=list(N_aux)
    u=list(N_aux)
    
    # Set A with all the arcs, their costs, and capacities
    for i in N_aux:
        A[i],c[i],u[i]=multidict(Ti[i])
    t2 = time.time()
    tiempo_pre = t2 - t1
    
    # --------------------------------------------------
    # Linear Model with Gurobi
    # --------------------------------------------------
    t1 = time.time()
    
    m1 = Model('FBM_'+log_name)
    m1.setParam('OutputFlag', output)
    if log_name != None:
        m1.params.LogFile='logs/FBM_'+log_name+'.log'
    if T_limit != None:
        m1.setParam('TimeLimit', T_limit)
    if focus != None:
        m1.setParam('MIPFocus', focus)
    if presolve != None:
        m1.setParam('Presolve', presolve)
    if method != None:
        m1.setParam('Method', method)
    if heuristics != None:
        modelo.setParam('Heuristics', heuristics)
    if cutsPlanes != None:
        modelo.setParam('Cuts', cutsPlanes)
    if cliqueCuts != None:
        modelo.setParam('CliqueCuts', cliqueCuts)
    if flowCoverCuts != None:
        modelo.setParam('FlowCoverCuts', flowCoverCuts)
    if zeroHalfCuts != None:
        modelo.setParam('ZeroHalfCuts', zeroHalfCuts)
    if mirCuts != None:
        modelo.setParam('MIRCuts', mirCuts)
    if relaxLiftCuts != None:
        modelo.setParam('RelaxLiftCuts', relaxLiftCuts)
    if strongCGCuts != None:
        modelo.setParam('StrongCGCuts', strongCGCuts)
    if infProofCuts != None:
        modelo.setParam('InfProofCuts', infProofCuts)
    if concurrentMIP != None:
        modelo.setParam('ConcurrentMIP', concurrentMIP)
    if seed != None:
        m1.setParam('Seed', seed)

    SV=['x'+str(i) for i in N_aux]
    
    # Variables
    Variables={}
    for i in N_aux:
        Variables[SV[i]]= m1.addVars(A[i], vtype = GRB.INTEGER, name='x{}'.format(i),ub=u[i])

    # Branch Priority
    for l in N_aux:
        for i,j in A[l]:
            Variables['{}'.format(SV[l])][(i,j)].BranchPriority = 4            
    
    # Objective Function
    objetivo = quicksum(Variables['{}'.format(SV[i])].prod(c[i], '*')  for i in N_aux)
    m1.setObjective(objetivo, GRB.MINIMIZE)
    
    # Constraints
    m1.addConstrs((Variables['{}'.format(SV[i])].sum()==1 for i in N), "rest1")
    
    sum_izq={}
    sum_der={}
    for i in V:
        sum_izq[i]=[]
        sum_der[i]=[]
    for l in N_aux:
        for (j,k) in Ti[l]:
            sum_der[j].append((l,(j,k)))
            sum_izq[k].append((l,(j,k)))
    
    flujo= {}
    for i in V:
        flujo[i] = -K if i==o else (K if i==d else 0) 

    L_i=[0]*len(V)
    L_d=[0]*len(V)
    for i in V:
        for l in sum_der[i]:
            L_d[i-1] += Variables['{}'.format(SV[l[0]])][l[1]]
        for l in sum_izq[i]:
            L_i[i-1] += Variables['{}'.format(SV[l[0]])][l[1]]

    # Flow constraints
    m1.addConstrs( (L_i[i-1]- L_d[i-1] == flujo[i] for i in V), name="flujo")
    m1.update()

    m1.optimize()
    
    t2 = time.time()
    tiempo_IP = t2 - t1
    
    return m1, tiempo_pre, tiempo_IP


# --------------------------------------------------
#   Analysis of the results of the flow model
# --------------------------------------------------

def analysis_FBM(instances,K,file = None,output = 0, T_limit = None, focus = None, presolve = None,method = None,logs = False, seed = None, cutsPlanes = None, concurrentMIP = None, heuristics = None, cliqueCuts = None, flowCoverCuts = None, zeroHalfCuts = None, mirCuts = None, relaxLiftCuts = None, strongCGCuts = None, infProofCuts = None):
    '''
    - Input:
    
    instances: A list with the names of the instances to analyze from the folder "instances"
    
    file: (string) The name of the file in CSV format
    
    K: (int) Number of available machines
    
    logs: By default, it is False; if logs = True, a file will be created with the name of the instance in log format
    
    The optional parameters (output, T_limit, focus, presolve, method, seed, cutsPlanes, 
                            concurrentMIP, heuristics, cliqueCuts, flowCoverCuts, zeroHalfCuts, 
                            mirCuts, relaxLiftCuts, strongCGCuts and infProofCuts) 
    are the same parameters to the FBM_model.
    
    '''
    data_output = [('FBM_'+inst,inst) for inst in instances]
    df_output = pd.DataFrame(data_output,columns = ['Log','inst'])
    df_output['objIP'] = 0.0
    df_output['bestBound'] = 0.0
    df_output['MIPGap'] = 0.0
    df_output['numVars'] = 0
    df_output['numConstrs'] = 0
    df_output['TimePre'] = 0.0
    df_output['TimeOPT'] = 0.0
    df_output['TimeTotal'] = 0.0
    df_output['TimeIP'] = 0.0
    df_output['numNodos'] = 0
    for inst in instances:
        if logs == True:
            log_name = inst
        else:
            log_name = False
        file_name = '../instances/'+inst+'.csv' # file name
        # Solve the the instance file_name using the FBM model
        m, t3, t4 = FBM_model(file_name,K,output = output, T_limit = T_limit, focus = focus, presolve = presolve,method = method,log_name = log_name, seed = seed, cutsPlanes = cutsPlanes, concurrentMIP = concurrentMIP, heuristics = heuristics, cliqueCuts = cliqueCuts, flowCoverCuts = flowCoverCuts, zeroHalfCuts = zeroHalfCuts, mirCuts = mirCuts, relaxLiftCuts = relaxLiftCuts, strongCGCuts = strongCGCuts, infProofCuts = infProofCuts)

        # Parameters analyzed in the tables
        df_output.loc[df_output.Log == 'FBM_'+inst,['objIP']] = m.ObjVal
        df_output.loc[df_output.Log == 'FBM_'+inst,['bestBound']] = m.ObjBound
        df_output.loc[df_output.Log == 'FBM_'+inst,['TimePre']] = t3
        df_output.loc[df_output.Log == 'FBM_'+inst,['TimeOPT']] = m.Runtime
        df_output.loc[df_output.Log == 'FBM_'+inst,['TimeTotal']] = t3+t4
        df_output.loc[df_output.Log == 'FBM_'+inst,['TimeIP']] = t4
        df_output.loc[df_output.Log == 'FBM_'+inst,['numVars']] = m.NumVars
        df_output.loc[df_output.Log == 'FBM_'+inst,['numConstrs']] = m.NumConstrs
        df_output.loc[df_output.Log == 'FBM_'+inst,['numNodos']] = m.NodeCount
        df_output.loc[df_output.Log == 'FBM_'+inst,['MIPGap']] = m.MIPGap

    if file == None:
        df_output.to_csv('Results/FBM_table.csv')
    else:
        df_output.to_csv(file +'.csv')
