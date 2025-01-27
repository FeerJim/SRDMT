import numpy as np
from gurobipy import*
import pandas as pd
import math
import time
import preprocessing as pre
import algorithms_codes as alg

# Time limit using in callback function
tiempo_limite_cb = 60

# --------------------------------------------------
# Definition of the set J_t
# --------------------------------------------------

def J(t,S,F):
    '''
    - Input:
    
    t: Instant of time t at which conflicts are to be sought.
    
    S: Nested dictionary with the possible start times of service indexed by the job ID (i) and the waiting time (j).
    
    F: Nested dictionary with the possible end times of service indexed by the job ID (i) and the waiting time (j).

    - Output:
    
    A set with tuples (i, j) that corresponds to the index of job i and the waiting time j that are in conflict at the instant of time t.
    '''
    return {(i,j) for i in S.keys() for j in S[i].keys() if S[i][j] <= t < F[i][j]}

# --------------------------------------------------
# Definition of the set H_q
# --------------------------------------------------

def _H_q(r,S_r,S):
    '''
    - Input:
    
    r: Index of Period over which to search for the intervals.
    
    S_r: A list with the start times of each period in minutes.
    
    S: Nested dictionary with the possible start times of service indexed by the job ID (i) and the waiting time (j).

    - Output:
    
    A set with the start times of all jobs in period r plus the start time of period r.
    '''
    return {S_r[r]} | {S[i][j] for i in S.keys() for j in S[i].keys() if S_r[r] <= S[i][j] < S_r[r]+60}

# --------------------------------------------------
# Definition of the set I^r (J_1^q)
# --------------------------------------------------

def Ir(r,S_r,S,F):
    '''
    - Input:
    r: Index of the period in which to perform the search.
    
    S_r: A list of the start times of each period r.
    
    S: Nested dictionary with the possible start times of service indexed by the job ID (i) and the waiting time (j).
    
    F: Nested dictionary with the possible end times of service indexed by the job ID (i) and the waiting time (j).

    - Output:
    
    A set of clients that must be served in period r.
    '''
    return {i for i in S.keys() if min(S[i][j] for j in S[i].keys()) >= S_r[r] and max(F[i][j] for j in S[i].keys()) < S_r[r]+60}

# --------------------------------------------------
# Definition of the set J^r (J_3^q)
# --------------------------------------------------

def Jr(r,S_r,S,F):
    '''
    - Input:
    
    r: Index of the period in which to perform the search.
    
    S_r: A list of the start times of each period r.
    S: Nested dictionary with the possible start times of service indexed by the job ID (i) and the waiting time (j).
    
    F: Nested dictionary with the possible end times of service indexed by the job ID (i) and the waiting time (j).

    - Output:
    
    A set of jobs that must be served in period r and r+1.
    '''
    return {i for i in S.keys() if max(S[i][j] for j in S[i].keys()) < S_r[r] <= min(F[i][j] for j in S[i].keys()) <= S_r[r]+60}

# --------------------------------------------------
# Definition of the set I^p (J_2^q)
# --------------------------------------------------

def Ip(r,S_r,S,F):
    '''
    - Input:
    
    r: Index of the period in which to perform the search.
    
    S_r: A list of the start times of each period r.
    
    S: Nested dictionary with the possible start times of service indexed by the job ID (i) and the waiting time (j).
    
    F: Nested dictionary with the possible end times of service indexed by the job ID (i) and the waiting time (j).

    - Output:
    
    A set of jobs that must be served in period r.
    '''
    return {i for i in S.keys() if  S_r[r] <= min(S[i][j] for j in S[i].keys()) and max(S[i][j] for j in S[i].keys()) < S_r[r]+60}


# --------------------------------------------------
#   JIAM Formulation
# --------------------------------------------------

def JIAM(S_r,S,F,K,output = 0, T_limit = None, focus = None, BP=0, presolve = None,method = None,log_name = None, seed = None):
    '''
    - Input:
    
    S_r: List of the start times of the periods.
    
    S: Nested dictionary with the possible start times of service indexed by the job ID (i) and the waiting time (j).
    
    F: Nested dictionary with the possible end times of service indexed by the job ID (i) and the waiting time (j).
    
    K: Number of available machines.
    
    The following parameters are optional and active or desactive the parameters defined in the class Gurobi:
    
    output: To visualize the Gurobi output; by default, zero. If set to one, the output is displayed.
    T_limit: Parameter to modify the execution time limit; by default, None.
    focus: Parameter to modify the solver strategy:
            focus = 1 if there are no issues with solution quality and optimality is to be tested.
            focus = 2 if bounding is slow for optimality.
            focus = 3 if more bounds are to be found.
    BP: Parameter that allows us to decide the priority of the variable in branching; by default, it is zero, one toprioritize the variable, and two to prioritize variable x.
    presolve: Parameter to modify the level of the presolver:
            presolve = -1 by default.
            presolve = 0 off.
            presolve = 1 conservative.
            presolve = 2 aggressive.
    method: Parameter to modify the optimizer method:
            -1 = automatic.
            0 = primal simplex.
            1 = dual simplex.
            2 = barrier.
            3 = concurrent.
            4 = deterministic concurrent.
            5 = deterministic concurrent simplex.
    log_name: Name of the instance to save the .log file.
    Seed: Parameter to modify the solution path;
            0 = by default,
            int = any integer value.
        
    - Output:
    
    modelo: A Gurobi model class with the values of the variables.
    
    ind: List of indices (i, j, k) found in the solution of the SRDM-T model such that x[i, j, k] = 1.
    
    M_max: Number of machines used to serve the jobs.
    
    z_sol: Dictionary indexed by periods whose values are the number of machines per period.
    '''
    P = len(S_r)
    
    modelo = Model('JIAM Formulation')
    modelo.setParam('OutputFlag', output)
    if log_name != None:
        modelo.params.LogFile='logs/JIAM_'+log_name+'.log'
    if T_limit != None:
        modelo.setParam('TimeLimit', T_limit)
    if focus != None:
        modelo.setParam('MIPFocus', focus)
    if presolve != None:
        modelo.setParam('Presolve', presolve)
    if method != None:
        modelo.setParam('Method', method)
    if seed != None:
        modelo.setParam('Seed', seed)
    
    # Variables
    x = modelo.addVars(S.keys(), list(S.values())[0].keys(), range(K), vtype = GRB.BINARY, name="x")
    y = modelo.addVars(range(K), range(P), vtype = GRB.BINARY, name="y")

    # Objective Function
    obj = y.sum('*','*')

    # Sense of optimization
    modelo.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    # Processed jobs
    modelo.addConstrs((x.sum(i,'*','*') == 1 for i in S.keys()), "jobs") 

    # Conflicts between jobs
    modelo.addConstrs((quicksum(x[i,j,k] for (i,j) in J(t,S,F)) <= y[k,r] for k in range(K) for r in range(P) for t in _H_q(r,S_r,S)), "overlapping")
    
    # We can modify the priority in the selection of branching with the parameter BP
    if BP == 1:
        for k in range(K):
            for r in range(P):
                y[k,r].BranchPriority = 1
    if BP == 2:
        for i in range(n):
            for j in range(m):
                for k in range(K):
                    x[i,j,k].BranchPriority = 1
    if BP == 3:
        for k in range(K):
            for r in range(P):
                y[k,r].BranchPriority = K*P-k-r
    
    modelo.update()
    modelo.optimize()
    
    # Save Solution
    ind = [(i,j,k) for i in S.keys() for j in S[i].keys() for k in range(K) if x[i,j,k].x >= 0.5]
    z_sol = {r: round(sum(y[k,r].x for k in range(K))) for r in range(P)}
    M_max = max(z_sol[r] for r in range(P))

    return modelo, ind, M_max, z_sol

# --------------------------------------------------
# Function to convert the results into a data frame
# --------------------------------------------------

def df_JIAM(ind_sol,K,S,F,dfm,formato = False):
    '''
    - Input:
    
    ind: list of indices (i,j) found in the solution using JIAM such that x[i,j] = 1.
    
    K: Number of machines available to serve customers
    
    S: Nested dictionary with possible service start times indexed by customer ID (i) and waiting time (j).
    
    F: Nested dictionary with the possible end times of service indexed by the customer ID (i) and the waiting time (j).
    
    dfm: data frame containing the customers in the same order as they were created S and F.
    
    formato: a boolean parameter that allows us to save the results in DateTime format, by default the results are saved in minutes.

    - Output:
    
    dfm_sol: a DataFrame containing the results.
    '''
    
    dfm_sol = dfm.copy()
    dfm_sol['Machine'] = -1
    dfm_sol['WaitingTime'] = 0.0
    dfm_sol['StartTime'] = 0.0
    dfm_sol['EndTime'] = 0.0
    for k in range(K):
        for i,j,l in ind_sol:
            if l == k:
                dfm_sol.loc[dfm_sol.ID == i,['Machine']] = k+1
                dfm_sol.loc[dfm_sol.ID == i,['WaitingTime']] = j
                dfm_sol.loc[dfm_sol.ID == i,['StartTime']] = S[i][j]
                dfm_sol.loc[dfm_sol.ID == i,['EndTime']] = F[i][j]
    if formato == True:
        dfm_sol['StartTime'] = pd.TimedeltaIndex(dfm_sol['StartTime'], unit='m').round('S')
        dfm_sol['EndTime'] = pd.TimedeltaIndex(dfm_sol['EndTime'], unit='m').round('S')
        dfm_sol['Duracion'] = pd.TimedeltaIndex(dfm_sol['Duracion'], unit='m').round('S')
    dfm_sol.sort_values(by=['Machine','StartTime'])
    return dfm_sol

# --------------------------------------------------
#  Callback Functions
# --------------------------------------------------
def RN_callback(model, where):
    if where == GRB.Callback.MIP:
        runtime = model.cbGet(GRB.Callback.RUNTIME)

        if runtime > tiempo_limite_cb:
            model._changeParam_RN = True
            model.terminate()
# --------------------------------------------------
def RN_callback_cuts(model, where):
    if where == GRB.Callback.MIPSOL:
        objbnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        diferencia = abs(objbst - round(objbnd))
        _P = len(model._vars_z)
        if diferencia <= 1:
            x = model.cbGetSolution(model._vars)
            print('x:',[x[i] for i in range(len(x)) if 1 > x[i] > 0 or i in range(len(x),len(x) - _P-1,-1)])
        sol = model.cbGetSolution([model._vars_z[q] for q in range(_P)])
        cota_inferior = math.ceil((sum(model._F[i][0] - model._S[i][0] for i in model._S.keys()))/60)
        if sum(sol) < cota_inferior:
            model.cbLazy(quicksum(model._vars_z[q] for q in range(_P)) >= cota_inferior)
            print('callback_Lazy_added')
        for r in range(_P):
            J2 = Ip(r,model._S_r,model._S,model._F)
            cota_inferior = math.ceil(sum(model._F[i][0] - model._S[i][0] for i in J2 )/(max([0]+[model._F[i][j] for i in J2 for j in model._S[i].keys()]) - model._S_r[r]) )
            if len(J2) != 0 and sol[r] < cota_inferior:
                model.cbCut(model._vars_z[r] >= cota_inferior)
                print('callback_lazy_added_lb')
    elif where == GRB.Callback.MIPNODE:
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        objbnd = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
        objbst = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
        diferencia = abs(objbst - round(objbnd))
        if status == GRB.OPTIMAL:
            _P = len(model._vars_z)
            if diferencia <= 1:
                x = model.cbGetNodeRel(model._vars)
                print('x:',[x[i] for i in range(len(x)) if 1 > x[i] > 0 or i in range(len(x),len(x) - _P-1,-1)])
            cota_inferior = math.ceil((sum(model._F[i][0] - model._S[i][0] for i in model._S.keys()))/60)
            rel = model.cbGetNodeRel([model._vars_z[q] for q in range(_P)])
            if sum(rel) < cota_inferior:
                model.cbCut(quicksum(model._vars_z[q] for q in range(_P)) >= cota_inferior)
                print('callback_cuts_added')
            for r in range(_P):
                J2 = Ip(r,model._S_r,model._S,model._F)
                cota_inferior = math.ceil(sum(model._F[i][0] - model._S[i][0] for i in J2 )/(max([0]+[model._F[i][j] for i in J2 for j in model._S[i].keys()]) - model._S_r[r]) )
                if len(J2) != 0 and rel[r] < cota_inferior:
                    model.cbCut(model._vars_z[r] >= cota_inferior)
                    print('callback_cuts_added_lb')

# --------------------------------------------------
#   S-JIAM Formulation
# --------------------------------------------------

def S_JIAM(S_r,S,F,K,output = 0, T_limit = None, focus = None, BP=0, RN = None, presolve = None, method = None, log_name = None, seed = None, cutsPlanes = None, x_start = None, concurrentMIP = None, heuristics = None, cliqueCuts = None, flowCoverCuts = None, zeroHalfCuts = None, mirCuts = None, relaxLiftCuts = None, strongCGCuts = None, infProofCuts = None, callback = None, degenMoves = None):
    '''
    - Input:
    
    S_r: List of the start times of the periods.
    
    S: Nested dictionary with the possible start times of service indexed by the job ID (i) and the waiting time (j).
    
    F: Nested dictionary with the possible end times of service indexed by the job ID (i) and the waiting time (j).
    
    K: Number of available machines.
    
    x_start: dictionary indexed by the machines and whose values are the indices (i,j) = (customer, position)
    
    RN: Integer value to add new inequalities.
        1 = Bound of Proposition 2
        2 = Bound of Proposition 3
        3 = Maximum Bound Between the bounds in Proposition 2 and 3
        4 = Trivial Bound
        5 = Trivial Bound with Bound of Proposition 2
        6 = Trivial Bound with Bound of Proposition 3
        7 = Trivial Bound with Maximum Bound between Prop. 2 and 3
    
    The following parameters are optional and active or desactive the parameters defined in the class Gurobi:
    
    output: To visualize the Gurobi output; by default, zero. If set to one, the output is displayed.
    T_limit: Parameter to modify the execution time limit; by default, None.
    focus: Parameter to modify the solver strategy:
            focus = 1 if there are no issues with solution quality and optimality is to be tested.
            focus = 2 if bounding is slow for optimality.
            focus = 3 if more bounds are to be found.
    BP: Parameter that allows us to decide the priority of the variable in branching; by default, it is zero, one toprioritize the variable, and two to prioritize variable x.
    presolve: Parameter to modify the level of the presolver:
            presolve = -1 by default.
            presolve = 0 off.
            presolve = 1 conservative.
            presolve = 2 aggressive.
    method: Parameter to modify the optimizer method:
            -1 = automatic.
            0 = primal simplex.
            1 = dual simplex.
            2 = barrier.
            3 = concurrent.
            4 = deterministic concurrent.
            5 = deterministic concurrent simplex.
    log_name: Name of the instance to save the .log file.
    Seed: Parameter to modify the solution path;
            0 = by default,
            int = any integer value.
    heuristics: Fraction of time used to find solutions using a heuristic:
            heuristics = 0.05 by default
            heuristics <= 1
    cliqueCuts,flowCoverCuts,zeroHalfCuts: Parameter to modify the level of cut generation:
            ...Cuts = -1 by default
            ...Cuts = 0 disabled
            ...Cuts = 1 conservative
            ...Cuts = 2 aggressive
    cutsPlanes: Parameter to modify the level of aggressiveness of generate cutting planes:
            Cuts = -1 by default
            Cuts = 0 disabled
            Cuts = 1 conservative
            Cuts = 2 aggressive
            Cuts = 3 very aggressive
    ConcurrentMIP: By default 1, When the parameter is set to value n, the MIP solver performs n independent MIP solves in parallel, with different parameter settings for each.
    callback: Add constraints using the callback function.
            callback = None: Default model
            callback = 1: Callback modifying parameters and terminating the initial process
            callback = 2: Adding New Constraints (RN) as lazy constraints or cuts if they are violated.
    degenMoves: Degenerate simplex moves (INT) by default -1

    - Output:
    
    modelo: A Gurobi model class with the values of the variables.
    
    ind: A list of indices (i,j) found in the solution using S-JIAM, such that x[i,j] = 1.
    
    M_max: Number of machines used to serve customers.
    
    z_sol: A dictionary indexed by periods, with values representing the number of machines per period.
    '''
    P = len(S_r)
    
    modelo = Model('S-JIAM')
    modelo.setParam('OutputFlag', output)
    if log_name != None:
        modelo.params.LogFile='logs/S_JIAM_'+log_name+'.log'
    if T_limit != None:
        modelo.setParam('TimeLimit', T_limit)
    if focus != None:
        modelo.setParam('MIPFocus', focus)
    if presolve != None:
        modelo.setParam('Presolve', presolve)
    if method != None:
        modelo.setParam('Method', method)
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
    if degenMoves != None:
        modelo.setParam('DegenMoves', degenMoves)
    if concurrentMIP != None:
        modelo.setParam('ConcurrentMIP', concurrentMIP)
    if seed != None:
        modelo.setParam('Seed', seed)
    
    # Variables
    x = modelo.addVars(S.keys(), list(S.values())[0].keys(), vtype = GRB.BINARY, name="x")
    z = modelo.addVars(range(P), vtype = GRB.INTEGER, name="z", lb = 0, ub = K)
    
    # Add Initial solution
    if x_start != None:
        for k in x_start.keys():
            for (i,j) in x_start[k]:
                x[i,j].Start = 1

    # Objective Function
    obj = z.sum('*')

    # Sense of optimization
    modelo.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    # All jobs are processed
    modelo.addConstrs((x.sum(i,'*') == 1 for i in S.keys()), "jobs") 

    # overlapping
    modelo.addConstrs((quicksum(x[i,j] for (i,j) in J(t,S,F)) <= z[q] for q in range(P) for t in _H_q(q,S_r,S)), "overlapping")
    
    # New Constraints
    if RN == 4 or RN == 5 or RN == 6 or RN == 7:
        # Trivial bound
        modelo.addConstr((quicksum(z[q] for q in range(P)) >= math.ceil((sum(F[i][0] - S[i][0] for i in S.keys()))/60)), "newConstr")
    if RN == 1 or RN == 5:
        # Proposition 2
        for q in range(P):
            new_lb = math.ceil((sum(F[i][0] - S[i][0] for i in Ir(q,S_r,S,F) ))/60)
            z[q].setAttr("lb", new_lb)
    if RN == 2 or RN == 6:
        # Proposition 3
        for q in range(P):
            J2 = Ip(q,S_r,S,F)
            if len(J2) != 0:
                new_value = math.ceil(sum(F[i][0] - S[i][0] for i in J2 )/(max([0]+[F[i][j] for i in J2 for j in S[i].keys()]) - S_r[q]) )
                z[q].setAttr("lb", new_value)
    if RN == 3 or RN == 7:
        # Maximum bound between Proposition 2 and 3
        new_value2 = 0
        J2 = Ip(P-1,S_r,S,F)
        if len(J2) != 0:
            new_value2 = math.ceil(sum(F[i][0] - S[i][0] for i in J2 )/(max([0]+[F[i][j] for i in J2 for j in S[i].keys()]) - S_r[P-1]) )
        res = max(len( Jr(P-1,S_r,S,F) ), math.ceil((sum(F[i][0] - S[i][0] for i in Ir(P-1,S_r,S,F) ))/60), new_value2) - z[P-1]
        modelo.addConstr( res <= 0, name = 'res_N_'+str(P))
        for r in range(1,P):
            new_value2 = 0
            J2 = Ip(r-1,S_r,S,F)
            if len(J2) != 0:
                new_value2 = math.ceil(sum(F[i][0] - S[i][0] for i in J2)/(max([0]+[F[i][j] for i in J2 for j in S[i].keys()]) - S_r[r-1]) )
            r_nombre = 'res_N_'+str(r)
            res = max(len( Jr(r-1,S_r,S,F) ), len( Jr(r,S_r,S,F) ), math.ceil((sum(F[i][0] - S[i][0] for i in Ir(r-1,S_r,S,F) ))/60), new_value2) - z[r-1]
            modelo.addConstr( res <= 0, name = r_nombre)
    
    # Modified the priority in the branching Priority selection
    if BP == 1:
        for r in range(P):
            z[r].BranchPriority = 1
    if BP == 2:
        for i in S.keys():
            for j in S[i].keys():
                x[i,j].BranchPriority = 1
    if BP == 3:
        for r in range(P):
            z[r].BranchPriority = P-r
    
    modelo.update()

    # Optimization using callback functions
    if callback == None:
        modelo.optimize()
    elif callback == 1:
        modelo._changeParam_RN = False
        modelo.optimize(RN_callback)
        if modelo._changeParam_RN:
            modelo.addConstr((quicksum(z[r] for r in range(P)) >= math.ceil((sum(F[i][0] - S[i][0] for i in S.keys()))/60)), "newConstr")
            for r in range(P):
                J2 = Ip(r,S_r,S,F)
                if len(J2) != 0:
                    new_value = math.ceil(sum(F[i][0] - S[i][0] for i in J2 )/(max([0]+[F[i][j] for i in J2 for j in S[i].keys()]) - S_r[r]) )
                    z[r].setAttr("lb", new_value)
            modelo.update()
            modelo.optimize()
    elif callback == 2:
        modelo.Params.LazyConstraints = 1
        modelo._S_r = S_r
        modelo._S = S
        modelo._F = F
        modelo._vars = modelo.getVars()
        modelo._vars_z = [modelo.getVarByName('z['+str(q)+']') for q in range(P)]
        modelo.optimize(RN_callback_cuts)
        
    # Save Solution
    ind = [(i,j) for i in S.keys() for j in S[i].keys() if x[i,j].x >= 0.5]
    M_max = int(max([z[r].x for r in range(P)]))
    z_sol = {r: z[r].x for r in range(P)}

    return modelo, ind, M_max, z_sol

# --------------------------------------------------
# Function to convert the results into a data frame
# --------------------------------------------------

def df_S_JIAM(ind,K,S,F,dfm,formato = False):
    '''
    - Input:
    
    ind: list of indices (i,j) found in the solution using S-JIAM such that x[i,j] = 1.
    
    K: Number of machines available to serve customers
    
    S: Nested dictionary with possible service start times indexed by customer ID (i) and waiting time (j).
    
    F: Nested dictionary with the possible end times of service indexed by the customer ID (i) and the waiting time (j).
    
    dfm: data frame containing the customers in the same order as they were created S and F.
    
    formato: a boolean parameter that allows us to save the results in DateTime format, by default the results are saved in minutes.

    - Output:
    
    dfm_sol: a DataFrame containing the results.
    '''
    ind_sol = alg.generate_sol(S,F,ind, K)
    
    dfm_sol = dfm.copy()
    dfm_sol['Machine'] = -1
    dfm_sol['WaitingTime'] = 0.0
    dfm_sol['StartTime'] = 0.0
    dfm_sol['EndTime'] = 0.0
    for k in range(1,K+1):
        for i,j,l in ind_sol:
            if l == k:
                dfm_sol.loc[dfm_sol.ID == i,['Machine']] = k
                dfm_sol.loc[dfm_sol.ID == i,['WaitingTime']] = j
                dfm_sol.loc[dfm_sol.ID == i,['StartTime']] = S[i][j]
                dfm_sol.loc[dfm_sol.ID == i,['EndTime']] = F[i][j]
    if formato == True:
        dfm_sol['StartTime'] = pd.TimedeltaIndex(dfm_sol['StartTime'], unit='m').round('S')
        dfm_sol['EndTime'] = pd.TimedeltaIndex(dfm_sol['EndTime'], unit='m').round('S')
        dfm_sol['Duracion'] = pd.TimedeltaIndex(dfm_sol['Duracion'], unit='m').round('S')
    dfm_sol.sort_values(by=['Machine','StartTime'])
    return dfm_sol

# --------------------------------------------------
# Function to generate intervals
# --------------------------------------------------

def _I(H):
    '''
    - Input:
    
    H: Set of starting points of an interval r_i + kp // t_i + kp
        
    - Output:
    
    I: A dictionary with the intervals formed by the points of H
    '''
    I = {}
    ind = 1
    for i in H:
        I[ind] = pd.Interval(i,i+p,closed = 'left')
        ind += 1
    return I

# --------------------------------------------------
#  phi parameter (largest number of consecutive intervals that always intersect
# --------------------------------------------------

def _y(H,I):
    '''
    - Input:
    
    H: Set of starting points of an interval r_i + kp // t_i + kp
    
    I: A dictionary with the intervals formed by the points of H
        
    - Output:
    
    y: largest number of consecutive intervals that intersect without being empty
    '''
    aux = None
    for l in range(1,len(H)+1):
        for i in range(len(H)-l+1):
            A = [I[i+j] for j in range(1,l+1)]
            indices_A = pd.IntervalIndex(A)
            ind = indices_A.overlaps(I[i+1])
            if sum(ind) != len(ind):
                aux = sum(ind)
                break
        if aux != None:
            break
    y = aux
    return y


# --------------------------------------------------
# Function to check the intersections of intervals
# --------------------------------------------------

def _inter(I,t,i,k,q):
    '''
    - Input:
    
    I: A dictionary with the intervals formed by the points of H
    
    t: List with the start times of the periods
    
    i: index on which you want to check the intersections
    
    k: index for calculating the set of intervals to intersect
    
    q: index of period
        
    - Output:
    
    True/False: truth value about the intersection of the intervals and the period is different from the empty
    '''
    aux1 = pd.IntervalIndex([I[i+l] for l in range(1,k+1)])
    aux3 = aux1.overlaps(pd.Interval(t[q],t[q+1],closed = 'left'))
    if sum(aux3) == len(aux3):
        aux2 = aux1.overlaps(I[i+1])
        if sum(aux2) == len(aux2):
            aux_4 = True
        else:
            aux_4 = False
    else:
        aux_4 = False
    return aux_4

# --------------------------------------------------
#  EJIAM Formulation
# --------------------------------------------------
    
def EJIAM(r,d,t,H,I,M,output = 0, T_limit = None, focus = None, BP=0, presolve = None,method = None,log_name = None, seed = None):
    '''
    - Input:
    
    H: set with the start times of the intervals
    
    I: set of intervals
    
    t: List with the start times of the periods
    
    r: Nested dictionary with processing start times indexed by task ID (j).
    
    d: Nested dictionary containing processing completion times indexed by task ID (j).
    
    M: Number of machines
    
    The following parameters are optional and active or desactive the parameters defined in the class Gurobi:
    
    output: To visualize the Gurobi output; by default, zero. If set to one, the output is displayed.
    T_limit: Parameter to modify the execution time limit; by default, None.
    focus: Parameter to modify the solver strategy:
            focus = 1 if there are no issues with solution quality and optimality is to be tested.
            focus = 2 if bounding is slow for optimality.
            focus = 3 if more bounds are to be found.
    BP: Parameter that allows us to decide the priority of the variable in branching; by default, it is zero, one toprioritize the variable, and two to prioritize variable x.
    presolve: Parameter to modify the level of the presolver:
            presolve = -1 by default.
            presolve = 0 off.
            presolve = 1 conservative.
            presolve = 2 aggressive.
    method: Parameter to modify the optimizer method:
            -1 = automatic.
            0 = primal simplex.
            1 = dual simplex.
            2 = barrier.
            3 = concurrent.
            4 = deterministic concurrent.
            5 = deterministic concurrent simplex.
    log_name: Name of the instance to save the .log file.
    Seed: Parameter to modify the solution path;
            0 = by default,
            int = any integer value.
          
    - Output:
    
    modelo: A Gurobi model class with the values of the variables.
    '''
    P = len(t)-1
    y = _y(H,I)
    modelo = Model('EJIAM')
    modelo.setParam('OutputFlag', output)
    if log_name != None:
        modelo.params.LogFile='logs/EJIAM_'+log_name+'.log'
    if T_limit != None:
        modelo.setParam('TimeLimit', T_limit)
    if focus != None:
        modelo.setParam('MIPFocus', focus)
    if presolve != None:
        modelo.setParam('Presolve', presolve)
    if method != None:
        modelo.setParam('Method', method)
    if seed != None:
        modelo.setParam('Seed', seed)
    
    # Variables
    x = modelo.addVars(r.keys(), I.keys(), vtype = GRB.BINARY, name="x")
    z = modelo.addVars(range(1,P+1), vtype = GRB.INTEGER, name="z", lb = 0, ub = M)

    # Objective Function
    obj = z.sum('*')

    # Sense of optimization
    modelo.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    # jobs processed
    modelo.addConstrs((x.sum(j,'*') == 1 for j in r.keys()), "jobs") 

    # Conflicts
    for q in range(P):
        for k in range(y,0,-1):
            aux_y = 0
            for i in range(len(I) - k +1):
                if _inter(I,t,i,k,q) == True:
                    modelo.addConstr(quicksum(x[j,i+l] for j in r.keys() for l in range(1,k+1)) <= z[q+1], name = "machines")
                    aux_y = 1
            if aux_y == 1:
                break
    
    # Eliminating invalid intervals
    modelo.addConstrs( (x[j,i] <= 0.0 for j in r.keys() for i in I.keys() if I[i].left < r[j]), "compatibility_left")
    modelo.addConstrs( (x[j,i] <= 0.0 for j in r.keys() for i in I.keys() if I[i].right > d[j]), "compatibility_right")
    
    
    # Modified the priority in the branching selection
    if BP == 1:
        for q in range(1,P+1):
            z[q].BranchPriority = 1
    if BP == 2:
        for j in r.keys():
            for i in range(len(H)):
                x[j,i].BranchPriority = 1
    
    modelo.update()
    modelo.optimize()
    
    # Save solution
    ind = [(j,i) for j in r.keys() for i in I.keys() if x[j, i].x >= 0.5]
    M_max = int(max([z[q].x for q in range(1,P+1)]))
    z_sol = {q: z[q].x for q in range(1,P+1)}
    
    return modelo,ind,M_max,z_sol

# --------------------------------------------------
# Function to convert the results into a data frame
# --------------------------------------------------

def df_EJIAM(ind,M,r,I,dfm,formato = False):
    '''
    - Input:
    
    ind: list of indices (i,j) found in the solution using EJIAM such that x[i,j] = 1.
    
    M: Number of machines available to serve customers
    
    r: Dictionary with possible service start times
    
    I: Set of intervals
    
    dfm: data frame containing the customers in the same order as they were created in r.
    
    formato: a boolean parameter that allows us to save the results in DateTime format, by default the results are saved in minutes.

    - Output:
    
    dfm_sol: a DataFrame containing the results.
    '''
    p = next(iter(I.items()))[1].length
    ind_sol = alg.generate_sol_p(r,p,I,ind, M)
    
    dfm_sol = dfm.copy()
    dfm_sol['Machine'] = -1
    dfm_sol['WaitingTime'] = 0.0
    dfm_sol['StartTime'] = 0.0
    dfm_sol['EndTime'] = 0.0
    for k in range(1,M+1):
        for i,j,l in ind_sol:
            if l == k:
                dfm_sol.loc[dfm_sol.ID == i,['Machine']] = k
                dfm_sol.loc[dfm_sol.ID == i,['WaitingTime']] = j
                dfm_sol.loc[dfm_sol.ID == i,['StartTime']] = r[i]+j
                dfm_sol.loc[dfm_sol.ID == i,['EndTime']] = r[i]+j+p
    if formato == True:
        dfm_sol['StartTime'] = pd.TimedeltaIndex(dfm_sol['StartTime'], unit='m').round('S')
        dfm_sol['EndTime'] = pd.TimedeltaIndex(dfm_sol['EndTime'], unit='m').round('S')
        dfm_sol['Duracion'] = pd.TimedeltaIndex(dfm_sol['Duracion'], unit='m').round('S')
    dfm_sol.sort_values(by=['Machine','StartTime'])
    return dfm_sol

# --------------------------------------------------
# Linear relaxation of EJIAM
# --------------------------------------------------
    
def EJIAM_LR(r,d,t,H,I,M,output = 0, T_limit = None, focus = None, BP=0, presolve = None,method = None,log_name = None, seed = None):
    '''
    - Input:
    
    H: set with the start times of the intervals
    
    I: set of intervals
    
    t: List with the start times of the periods
    
    r: Nested dictionary with processing start times indexed by task ID (j).
    
    d: Nested dictionary containing processing completion times indexed by task ID (j).
    
    M: Number of machines
    
    The following parameters are optional and active or desactive the parameters defined in the class Gurobi:
    
    output: To visualize the Gurobi output; by default, zero. If set to one, the output is displayed.
    T_limit: Parameter to modify the execution time limit; by default, None.
    focus: Parameter to modify the solver strategy:
            focus = 1 if there are no issues with solution quality and optimality is to be tested.
            focus = 2 if bounding is slow for optimality.
            focus = 3 if more bounds are to be found.
    BP: Parameter that allows us to decide the priority of the variable in branching; by default, it is zero, one toprioritize the variable, and two to prioritize variable x.
    presolve: Parameter to modify the level of the presolver:
            presolve = -1 by default.
            presolve = 0 off.
            presolve = 1 conservative.
            presolve = 2 aggressive.
    method: Parameter to modify the optimizer method:
            -1 = automatic.
            0 = primal simplex.
            1 = dual simplex.
            2 = barrier.
            3 = concurrent.
            4 = deterministic concurrent.
            5 = deterministic concurrent simplex.
    log_name: Name of the instance to save the .log file.
    Seed: Parameter to modify the solution path;
            0 = by default,
            int = any integer value.
          
    - Output:
    
    modelo: A Gurobi model class with the values of the variables.
    '''
    P = len(t)-1
    y = _y(H,I)
    p = next(iter(I.items()))[1].length
    modelo = Model('EJIAM-Linear-Relaxation')
    modelo.setParam('OutputFlag', output)
    if log_name != None:
        modelo.params.LogFile='logs/EJIAM_LR_'+log_name+'.log'
    if T_limit != None:
        modelo.setParam('TimeLimit', T_limit)
    if focus != None:
        modelo.setParam('MIPFocus', focus)
    if presolve != None:
        modelo.setParam('Presolve', presolve)
    if method != None:
        modelo.setParam('Method', method)
    if seed != None:
        modelo.setParam('Seed', seed)
    
    # Variables
    x = modelo.addVars(r.keys(), I.keys(), vtype = GRB.CONTINUOUS, name="x", lb = 0.0)
    z = modelo.addVars(range(1,P+1), vtype = GRB.CONTINUOUS, name="z", lb = 0.0, ub = float(M))

    # Objective Function
    obj = z.sum('*')

    # Sense of optimization
    modelo.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    # jobs processed
    modelo.addConstrs((x.sum(j,'*') == p for j in r.keys()), "jobs") 

    # conflicts between jobs
    modelo.addConstrs((quicksum(x[j,i+l] for j in r.keys() for l in range(1,k+1)) <= p*z[q+1] for q in range(P) for k in range(1,y+1) for i in range(len(I)-k+1) if _inter(I,t,i,k,q) == True), "machines")
    
    # Eliminating invalid intervals
    modelo.addConstrs( (x[j,i] <= 0.0 for j in r.keys() for i in I.keys() if I[i].left < r[j]), "compatibility_left")
    modelo.addConstrs( (x[j,i] <= 0.0 for j in r.keys() for i in I.keys() if I[i].right > d[j]), "compatibility_right")
    
    # Modified the priority in the branching selection
    if BP == 1:
        for q in range(1,P+1):
            z[q].BranchPriority = 1
    if BP == 2:
        for j in r.keys():
            for i in range(len(H)):
                x[j,i].BranchPriority = 1
    
    modelo.update()
    modelo.optimize()
    
    # Save solution
    ind = {(j,i): round(x[j,i].x,3) for j in r.keys() for i in I.keys() if x[j, i].x >= 1e-3}
    z_sol = {q: z[q].x for q in range(1,P+1)}

    return modelo, ind,z_sol


# --------------------------------------------------
#    Algorithm 1 (based in Linear Programming) - LPB
# --------------------------------------------------

def LPB(x,M,r,d,I):
    '''
    - Input:
    
    modelo: A Gurobi model class with the solution using the linear relaxation of EJIAM
    
    r: Nested dictionary with processing start times indexed by task ID (j).
    
    d: Nested dictionary containing processing completion times indexed by task ID (j).
    
    I: Set of intervals

    - Output:
    
    alg_sol: a set with indices {j:i} that correspond to the assignment of job j in the interval i.
    '''
    # interval duration
    p = next(iter(I.items()))[1].length
    
    # define v(I)
    v_I = {}
    for k in sorted(I.keys()):
        v_I[k] = sum([x[(j,i)] for (j,i) in x.keys() if i <= k])
    
    # create 'mu' copies of v(I)
    mu = M
    v_mu = {}
    v_mu_I = {}
    aux_mu_ind = 0
    for l in I.keys():
        for k in range(M):
            v_mu[aux_mu_ind] = v_I[l]
            v_mu_I[aux_mu_ind] = l
            aux_mu_ind += 1
    
    # Select n intervals of I
    v_I_sol = {}
    for j in range(1,int(len(r.keys()))+1):
        for l in v_mu.keys():
            if v_mu[l] > (j-1) * p:
                v_I_sol[(j,v_mu_I[l])] = v_mu[l]
                break
    
    # Restricted interval
    r_sol = {}
    d_sol = {}
    I_rest = {}
    for j in r.keys():
        aux1 = [I[i].left for (_,i) in x.keys() if _ == j ]
        aux2 = [I[i].right for (_,i) in x.keys() if _ == j ]
        r_sol[j] = min(aux1)
        d_sol[j] = max(aux2)
    
    # Assign the jobs each interval
    alg_sol = {}
    aux = v_I_sol.copy()
    aux = dict(sorted(aux.items(), key=lambda item: item[0]))
    aux_tareas = r.copy()
    for (_,l) in aux.keys():
        d_aux = max(d_sol.values())+1
        j_aux = -1
        for j in aux_tareas.keys():
            if I[l].left >= r_sol[j] and I[l].right <= d_sol[j] and d_aux > d_sol[j]:
                d_aux = d_sol[j]
                j_aux = j
        if j_aux == -1:
            continue
        alg_sol[j_aux] = l
        aux_tareas.pop(j_aux)

    return alg_sol

# --------------------------------------------------
#    Generate a table with the results of JIAM
# --------------------------------------------------

def table_JIAM(instancias,K,file = None, T = None, guardar = False, output = 1, T_limit = 3600, p = None, L = 60,focus = None, BP=0, presolve = None, method = None, seed = None):
    '''
    - Input:
    
    instancias: a list with the name of the instances
    
    K: Number of machines
    
    file: Name of the file to export the csv file
    
    guardar: Save the solution of each job in a csv file
    
    p: by default None (use durations defined in data) if p is different of None meaning that data use equal durations for all jobs.
    
    L: lenght of each period
    
    The following parameters are optional and active or desactive the parameters defined in the class Gurobi:
    
    output: To visualize the Gurobi output; by default, zero. If set to one, the output is displayed.
    T_limit: Parameter to modify the execution time limit; by default, None.
    focus: Parameter to modify the solver strategy:
            focus = 1 if there are no issues with solution quality and optimality is to be tested.
            focus = 2 if bounding is slow for optimality.
            focus = 3 if more bounds are to be found.
    BP: Parameter that allows us to decide the priority of the variable in branching; by default, it is zero, one toprioritize the variable, and two to prioritize variable x.
    presolve: Parameter to modify the level of the presolver:
            presolve = -1 by default.
            presolve = 0 off.
            presolve = 1 conservative.
            presolve = 2 aggressive.
    method: Parameter to modify the optimizer method:
            -1 = automatic.
            0 = primal simplex.
            1 = dual simplex.
            2 = barrier.
            3 = concurrent.
            4 = deterministic concurrent.
            5 = deterministic concurrent simplex.
    Seed: Parameter to modify the solution path;
            0 = by default,
            int = any integer value.

    - Output:
    
    export a file in csv format
    '''
    
    data_output = [('JIAM_'+inst,inst) for inst in instancias]
    df_output = pd.DataFrame(data_output,columns = ['Log','inst'])
    df_output['objIP'] = 0.0
    df_output['MIPGap'] = 0.0
    df_output['numVars'] = 0
    df_output['numConstrs'] = 0
    df_output['TimePre'] = 0.0
    df_output['TimeOPT'] = 0.0
    df_output['TimeTotal'] = 0.0
    df_output['TimeIP'] = 0.0
    df_output['numNodos'] = 0
    for inst in instancias:
        if p == None:
            p_name = '_'+str(0) # default durations in the data.
        else:
            p_name = '_'+str(1) # durations equal to the average transaction time.
        file_name = '../instances/'+inst+'.csv' # define name of the file with data
        df = pre.read_file(file_name) # read the file
        t1 = time.time()
        S,F,P_s,S_r = pre.calculate_SF(df,p,L,T) # calculate the parameters to JIAM formulation
        t2 = time.time()
        t3 = t2 - t1

        # Solve the instance using the JIAM formulation
        t1 = time.time()
        m, ind, M_max, z_sol = JIAM(S_r,S,F,K,output = output, T_limit = T_limit, log_name = inst+p_name, focus = focus, BP=BP, RN = RN, presolve = presolve,method = method, seed = seed)
        t2 = time.time()
        t4 = t2-t1

        # save the solution in a .csv file
        if guardar == True:
            df_SRDMT = df_JIAM(ind,K,S,F,df,formato = True)
            df_SRDMT.to_csv('Solution/Sol_JIAM_'+inst+p_name+'.csv')

        # Columns of the table
        df_output.loc[df_output.Log == 'JIAM_'+inst,['objIP']] = m.ObjVal
        df_output.loc[df_output.Log == 'JIAM_'+inst,['TimePre']] = t3
        df_output.loc[df_output.Log == 'JIAM_'+inst,['TimeOPT']] = m.Runtime
        df_output.loc[df_output.Log == 'JIAM_'+inst,['TimeTotal']] = t3+t4
        df_output.loc[df_output.Log == 'JIAM_'+inst,['TimeIP']] = t4
        df_output.loc[df_output.Log == 'JIAM_'+inst,['numVars']] = m.NumVars
        df_output.loc[df_output.Log == 'JIAM_'+inst,['numConstrs']] = m.NumConstrs
        df_output.loc[df_output.Log == 'JIAM_'+inst,['numNodos']] = m.NodeCount
        df_output.loc[df_output.Log == 'JIAM_'+inst,['MIPGap']] = m.MIPGap

    if file == None:
        df_output.to_csv('Results/Table_Results_JIAM'+p_name+'.csv')
    else:
        df_output.to_csv(file +'.csv')

# --------------------------------------------------
#    Generate a table with the results of S-JIAM
# --------------------------------------------------

def table_S_JIAM(instancias,K,file = None, T = None, guardar = False,output = 1, T_limit = 3600, p = None, L = 60,focus = None, BP=0, RN = None, presolve = None,method = None, seed = None, cutsPlanes = None, x_inicial_method = None, concurrentMIP = None, heuristics = None, cliqueCuts = None, flowCoverCuts = None, zeroHalfCuts = None, mirCuts = None, relaxLiftCuts = None, strongCGCuts = None, infProofCuts = None, callback = None, degenMoves = None):
    '''
    - Input:
    
    instancias: a list with the name of the instances
    
    K: Number of machines
    
    file: Name of the file to export the csv file
    
    guardar: Save the solution of each job in a csv file
    
    p, L and T are the same parameters defined in function calculate_SF() in the preprocessing file.
    
    x_inicial_method: 
        x_inicial_method = str() -> AG, MGA, GBF
    
    The other parameters are optional defined in the function S_JIAM()

    - Output:
    
    export a file in csv format
    '''
    data_output = [('S_JIAM_'+inst,inst) for inst in instancias]
    df_output = pd.DataFrame(data_output,columns = ['Log','inst'])
    df_output['objIP'] = 0.0
    df_output['MIPGap'] = 0.0
    df_output['numVars'] = 0
    df_output['numConstrs'] = 0
    df_output['TimePre'] = 0.0
    df_output['TimeOPT'] = 0.0
    df_output['TimeTotal'] = 0.0
    df_output['TimeIP'] = 0.0
    df_output['numNodos'] = 0
    for inst in instancias:
        if p == None:
            p_name = '_'+str(0) # default durations in the data.
        else:
            p_name = '_'+str(1) # durations equal to the average transaction time
        file_name = '../instances/'+inst+'.csv' # define name of the file with data
        df = pre.read_file(file_name) # read file
        t1 = time.time()
        S,F,P_s,S_r = pre.calculate_SF(df,p,L,T) # calculate the parameters to S-JIAM formulation
        t2 = time.time()
        t3 = t2 - t1
        
        # Initial solution using a Greedy algorithm
        if x_inicial_method != None:
            if x_inicial_method == 'AG':
                t1 = time.time()
                x_start, k_sol, V_sol = alg.GA(S,F,S_r)
                t2 = time.time()
                t5 = t2 - t1
            elif x_inicial_method == 'MGA':
                t1 = time.time()
                x_start, k_sol, V_sol = alg.MGA(S,F,S_r)
                t2 = time.time()
                t5 = t2 - t1
            elif x_inicial_method == 'GBF':
                t1 = time.time()
                I, sol_z, sol_z_q = alg.GBF(S,F,S_r)
                x_start = {1:I}
                t2 = time.time()
                t5 = t2-t1
            else:
                print('Initial solution method not found')
                x_start = None
                t5 = 0.0
        else:
            x_start = None
            t5 = 0.0

        # Solve the instance using the S-JIAM formulation
        t1 = time.time()
        m, ind, M_max, z_sol = S_JIAM(S_r,S,F,K,output = output, T_limit = T_limit, log_name = inst+p_name, focus = focus, BP=BP, RN = RN, presolve = presolve,method = method, seed = seed, cutsPlanes = cutsPlanes, x_start = x_start, concurrentMIP = concurrentMIP, heuristics = heuristics, cliqueCuts = cliqueCuts, flowCoverCuts = flowCoverCuts, zeroHalfCuts = zeroHalfCuts, mirCuts = mirCuts, relaxLiftCuts = relaxLiftCuts, strongCGCuts = strongCGCuts, infProofCuts = infProofCuts, callback = callback, degenMoves = degenMoves)
        t2 = time.time()
        t4 = t2-t1

        # save the solution in a .csv file
        if guardar == True:
            try:
                df_SRDMT = df_S_JIAM(ind,K,S,F,df,formato = True)
                df_SRDMT.to_csv('Solution/Sol_S_JIAM_'+inst+p_name+'.csv')
            except:
                print('Solution cant be printed')

        # Columns of the table
        df_output.loc[df_output.Log == 'S_JIAM_'+inst,['objIP']] = m.ObjVal
        df_output.loc[df_output.Log == 'S_JIAM_'+inst,['TimePre']] = t3+t5
        df_output.loc[df_output.Log == 'S_JIAM_'+inst,['TimeOPT']] = m.Runtime
        df_output.loc[df_output.Log == 'S_JIAM_'+inst,['TimeTotal']] = t3+t4
        df_output.loc[df_output.Log == 'S_JIAM_'+inst,['TimeIP']] = t4
        df_output.loc[df_output.Log == 'S_JIAM_'+inst,['numVars']] = m.NumVars
        df_output.loc[df_output.Log == 'S_JIAM_'+inst,['numConstrs']] = m.NumConstrs
        df_output.loc[df_output.Log == 'S_JIAM_'+inst,['numNodos']] = m.NodeCount
        df_output.loc[df_output.Log == 'S_JIAM_'+inst,['MIPGap']] = m.MIPGap

    if file == None:
        df_output.to_csv('Results/Table_Results_S_JIAM'+p_name+'.csv')
    else:
        df_output.to_csv(file +'.csv')


# --------------------------------------------------
#     Generate a table with the results of EJIAM
# --------------------------------------------------

def table_EJIAM(instancias,M,p = None, L = 60,file = None, guardar = False, output = 1, T_limit = 3600, focus = None, BP=0, presolve = None, method = None, seed = None):
    '''
    - Input:
    
    instancias: a list with the name of the instances
    
    M: Number of machines
    
    file: Name of the file to export the csv file
    
    guardar: Save the solution of each job in a csv file
    
    p, L and T are the same parameters defined in function calculate_SF() in the preprocessing file.
    
    The other parameters are optional defined in the function EJIAM()

    - Output:
    
    export a file in csv format
    '''
    data_output = [('EJIAM_'+inst,inst) for inst in instancias]
    df_output = pd.DataFrame(data_output,columns = ['Log','inst'])
    df_output['objIP'] = 0.0
    df_output['MIPGap'] = 0.0
    df_output['numVars'] = 0
    df_output['numConstrs'] = 0
    df_output['TimePre'] = 0.0
    df_output['TimeOPT'] = 0.0
    df_output['TimeTotal'] = 0.0
    df_output['TimeIP'] = 0.0
    df_output['numNodos'] = 0
    
    for inst in instancias:
        if p == None:
            p_name = '_'+str(0) # durations equal to 10 seconds for all jobs
        else:
            p_name = '_'+str(1) # durations equal to the average transaction time.
        file_name = '../instances/'+inst+'.csv' # define name of the file with data
        df = pre.read_file(file_name) # read file
        t1 = time.time()
        r,d,t,P,H,I = pre.sets_EJIAM(df,p,L) # calculate the parameters to EJIAM formulation 
        t2 = time.time()
        t3 = t2-t1

        # Solve the instance using the EJIAM formulation
        t1 = time.time()
        m,ind_sol, nM, z_sol = EJIAM(r,d,t,H,I,M,output = output, T_limit = T_limit,log_name = inst+p_name, focus = focus, BP= BP, presolve = presolve, method = method, seed = seed)
        t2 = time.time()
        t4 = t2-t1

        # save the solution in a .csv file
        if guardar == True:
            df_SRDMT = df_EJIAM(ind_sol,M,r,I,df,formato = True)
            df_SRDMT.to_csv('Solution/Sol_EJIAM_'+inst+p_name+'.csv')

        # Columns of the table
        df_output.loc[df_output.Log == 'EJIAM_'+inst,['objIP']] = round(m.ObjVal,0)
        df_output.loc[df_output.Log == 'EJIAM_'+inst,['TimePre']] = t3
        df_output.loc[df_output.Log == 'EJIAM_'+inst,['TimeOPT']] = m.Runtime
        df_output.loc[df_output.Log == 'EJIAM_'+inst,['TimeTotal']] = t3+t4
        df_output.loc[df_output.Log == 'EJIAM_'+inst,['TimeIP']] = t4
        df_output.loc[df_output.Log == 'EJIAM_'+inst,['numVars']] = m.NumVars
        df_output.loc[df_output.Log == 'EJIAM_'+inst,['numConstrs']] = m.NumConstrs
        df_output.loc[df_output.Log == 'EJIAM_'+inst,['numNodos']] = m.NodeCount
        df_output.loc[df_output.Log == 'EJIAM_'+inst,['MIPGap']] = m.MIPGap
    
    if file == None:
        df_output.to_csv('Results/Table_Results_EJIAM'+p_name+'.csv')
    else:
        df_output.to_csv(file +'.csv')

# --------------------------------------------------------
#     Generate a table with the results of LPB algorithm
# --------------------------------------------------------

def table_LPB(instancias,M,p = None,L = 60,file = None,guardar = False,output = 1, T_limit = 3600, focus = None, BP=0, presolve = None,method = None, seed = None):
    '''
    - Input:
    
    instancias: a list with the name of the instances
    
    M: Number of machines
    
    file: Name of the file to export the csv file
    
    guardar: Save the solution of each job in a csv file
    
    L: lenght of each period (int) in minutes
    
    p: Processing duration (constant for all customers)
    
    The other parameters are optional defined in the function LPB()

    - Output:
    
    export a file in csv format
    '''
    data_output = [('LPB_'+inst,inst) for inst in instancias]
    df_output = pd.DataFrame(data_output,columns = ['Log','inst'])
    df_output['obj'] = 0.0
    df_output['objLP'] = 0.0
    df_output['numVars'] = 0
    df_output['numConstrs'] = 0
    df_output['TimePre'] = 0.0
    df_output['TimeOPT'] = 0.0
    df_output['TimeTotal'] = 0.0
    df_output['TimeLP'] = 0.0
    df_output['numNodos'] = 0
    
    for inst in instancias:
        if p == None:
            p_name = '_'+str(0) # durations equal to 10 seconds for all jobs
        else:
            p_name = '_'+str(1) # durations equal to the average transaction time.
        file_name = '../instances/'+inst+'.csv' # define name of the file with data
        df = pre.read_file(file_name) # read file
        t1 = time.time()
        r,d,t,P,H,I = pre.sets_EJIAM(df,p,L) # calculate the parameters to LPB algorithm 
        t2 = time.time()
        t3 = t2-t1

        # solve the instances using the EJIAM linear relaxation
        t1 = time.time()
        m,ind,z_sol_LP = EJIAM_LR(r,d,t,H,I,M,output = output, T_limit = T_limit,log_name = inst+p_name, focus = focus, BP= BP, presolve = presolve, method = method, seed = seed)
        t2 = time.time()
        t4 = t2-t1
        
        # Run the algorithm with the solution above
        t1 = time.time()
        alg_sol = LPB(ind,M,r,d,I)
        t2 = time.time()
        t5 = t2 - t1

        # save the solution in a .csv file
        if guardar == True:
            ind_sol = [(j,alg_sol[j]) for j in alg_sol.keys()]
            df_SRDMT = df_EJIAM(ind_sol,M,r,I,df,formato = True)
            df_SRDMT.to_csv('Solution/Sol_LPB_'+inst+p_name+'.csv')

        # Round the linear solution
        aux = 0
        for v in m.getVars():
            if 'z' in v.varName:
                aux += np.ceil(v.x)
        
        # Columns of the table
        df_output.loc[df_output.Log == 'LPB_'+inst,['obj']] = aux
        df_output.loc[df_output.Log == 'LPB_'+inst,['objLP']] = m.ObjVal
        df_output.loc[df_output.Log == 'LPB_'+inst,['TimePre']] = t3
        df_output.loc[df_output.Log == 'LPB_'+inst,['TimeOPT']] = m.Runtime
        df_output.loc[df_output.Log == 'LPB_'+inst,['TimeTotal']] = t3+t4+t5
        df_output.loc[df_output.Log == 'LPB_'+inst,['TimeLP']] = t4
        df_output.loc[df_output.Log == 'LPB_'+inst,['numVars']] = m.NumVars
        df_output.loc[df_output.Log == 'LPB_'+inst,['numConstrs']] = m.NumConstrs
    
    if file == None:
        df_output.to_csv('Results/Table_Results_LPB'+p_name+'.csv')
    else:
        df_output.to_csv(file +'.csv')

# ------------------------------------------------------
#    Generate a table with the results of algorithm GA
# ------------------------------------------------------

def table_AG(instancias,file = None,guardar = False, p = None, L = 60):
    '''
    - Input:
    
    instancias: a list with the name of the instances
    
    file: Name of the file to export the csv file
    
    guardar: by default False, if guardar = True a solution is saved in a csv file
    
    L: lenght of each period (int) in minutes
    
    p: Processing duration (constant for all customers)

    - Output:
    
    export a file in csv format
    '''
    data_output = [('GA_'+inst,inst) for inst in instancias]
    df_output = pd.DataFrame(data_output,columns = ['Log','inst'])
    df_output['objSRDM'] = 0.0 # objective value to the SRDM problem
    df_output['objSRDMT'] = 0.0 # objective value to the SRDMT problem
    df_output['TimePre'] = 0.0 # time to generate the parameters of algorithm
    df_output['TimeAlg'] = 0.0 # time to solve de instance using the algorithm
    df_output['TimeTotal'] = 0.0 # total time -> TimePre + TimeAlg
    for inst in instancias:
        if p == None:
            p_name = '_'+str(0) # durations by default in data
        else:
            p_name = '_'+str(1) # durations equal to the average transaction time.
        file_name = '../instances/'+inst+'.csv' # define the name file
        df = pre.read_file(file_name) # read the file
        t1 = time.time()
        S,F,P_s,S_r = pre.calculate_SF(df,p,L) # calculate the parameters
        t2 = time.time()
        t3 = t2 - t1
        
        # Run the greedy algorithm (GA)
        t1 = time.time()
        G_sol, k_sol, V_sol = alg.GA(S,F,S_r)
        t2 = time.time()
        t4 = t2-t1

        # save the solution in a .csv file
        if guardar == True:
            df_SRDMT = alg.df_GA(G_sol,k_sol,S,F,df,formato = True)
            df_SRDMT.to_csv('Solution/Sol_GA_'+inst+p_name+'.csv')

        # Columns of the table
        df_output.loc[df_output.Log == 'GA_'+inst,['objSRDM']] = len(V_sol)
        df_output.loc[df_output.Log == 'GA_'+inst,['objSRDMT']] = sum(len(V_sol[v]) for v in V_sol)
        df_output.loc[df_output.Log == 'GA_'+inst,['TimePre']] = t3
        df_output.loc[df_output.Log == 'GA_'+inst,['TimeAlg']] = t4
        df_output.loc[df_output.Log == 'GA_'+inst,['TimeTotal']] = t3+t4

    if file == None:
        df_output.to_csv('Results/Table_Results_GA'+p_name+'.csv')
    else:
        df_output.to_csv(file +'.csv')        

# ------------------------------------------------------
#    Generate a table with the results of algorithm MGA
# ------------------------------------------------------

def table_MGA(instancias,file = None,guardar = False, p = None, L = 60):
    '''
    - Input:
    
    instancias: a list with the name of the instances
    
    file: Name of the file to export the csv file
    
    guardar: by default False, if guardar = True a solution is saved in a csv file
    
    L: lenght of each period (int) in minutes
    
    p: Processing duration (constant for all customers)

    - Output:
    
    export a file in csv format
    '''
    data_output = [('MGA_'+inst,inst) for inst in instancias]
    df_output = pd.DataFrame(data_output,columns = ['Log','inst'])
    df_output['objSRDM'] = 0.0 # objective value to the SRDM problem
    df_output['objSRDMT'] = 0.0 # objective value to the SRDMT problem
    df_output['TimePre'] = 0.0 # time to generate the parameters of algorithm
    df_output['TimeAlg'] = 0.0 # time to solve de instance using the algorithm
    df_output['TimeTotal'] = 0.0 # total time -> TimePre + TimeAlg
    for inst in instancias:
        if p == None:
            p_name = '_'+str(0) # durations by default in data
        else:
            p_name = '_'+str(1) # durations equal to the average transaction time.
        file_name = '../instancies/'+inst+'.csv' # define where is the file saved
        df = pre.read_file(file_name) # read the file
        t1 = time.time()
        S,F,P_s,S_r = pre.calculate_SF(df,p,L) # calculate the parameters
        t2 = time.time()
        t3 = t2 - t1
        
        # Run the modify greedy algorithm (MGA)
        t1 = time.time()
        G_sol, k_sol, V_sol = alg.MGA(S,F,S_r)
        t2 = time.time()
        t4 = t2-t1

        # save the solution in a .csv file
        if guardar == True:
            df_SRDMT = alg.df_GA(G_sol,k_sol,S,F,df,formato = True)
            df_SRDMT.to_csv('Solution/Sol_MGA_'+inst+p_name+'.csv')

        # Columns of the table
        df_output.loc[df_output.Log == 'MGA_'+inst,['objSRDM']] = len(V_sol)
        df_output.loc[df_output.Log == 'MGA_'+inst,['objSRDMT']] = sum(len(V_sol[v]) for v in V_sol)
        df_output.loc[df_output.Log == 'MGA_'+inst,['TimePre']] = t3
        df_output.loc[df_output.Log == 'MGA_'+inst,['TimeAlg']] = t4
        df_output.loc[df_output.Log == 'MGA_'+inst,['TimeTotal']] = t3+t4

    if file == None:
        df_output.to_csv('Results/Table_Results_MGA'+p_name+'.csv')
    else:
        df_output.to_csv(file +'.csv')

# ---------------------------------------------------------
#    Generate a table with the results of algorithm GBF
# ---------------------------------------------------------

def table_GBF(instancias,file = None, guardar = False, p = None, L = 60):
    '''
    - Input:
    
    instancias: a list with the name of the instances
    
    file: Name of the file to export the csv file
    
    L: lenght of each period (int) in minutes
    
    p: Processing duration (constant for all customers)
    
    guardar: by default False, if guardar = True a solution is saved in a csv file

    - Output:
    
    export a file in csv format
    '''
    data_output = [('GBF_'+inst,inst) for inst in instancias]
    df_output = pd.DataFrame(data_output,columns = ['Log','inst'])
    df_output['objSRDM'] = 0.0
    df_output['objSRDMT'] = 0.0
    df_output['TimePre'] = 0.0
    df_output['TimeAlg'] = 0.0
    df_output['TimeTotal'] = 0.0
    for inst in instancias:
        if p == None:
            p_name = '_'+str(0) # durations by default in data
        else:
            p_name = '_'+str(1) # durations equal to the average transaction time.
        file_name = '../instancias/'+inst+'.csv' # define where is the file saved
        df = pre.read_file(file_name) # read  the file
        t1 = time.time()
        S,F,P_s,S_r = pre.calculate_SF(df,p,L) # calculate the parameters
        t2 = time.time()
        t3 = t2 - t1
        
        # Run the greedy best fit algorithm (GBF)
        t1 = time.time()
        I, sol_z, sol_z_q = alg.GBF(S,F,S_r)
        t2 = time.time()
        t4 = t2-t1

        # save the solution in a .csv file
        if guardar == True:
            df_SRDMT = df_S_JIAM(I,sol_z_q,S,F,df,formato = True)
            df_SRDMT.to_csv('Solution/Sol_GBF_'+inst+p_name+'.csv')

        # columns of the table
        df_output.loc[df_output.Log == 'GBF_'+inst,['objSRDM']] = sol_z
        df_output.loc[df_output.Log == 'GBF_'+inst,['objSRDMT']] = sol_z_q
        df_output.loc[df_output.Log == 'GBF_'+inst,['TimePre']] = t3
        df_output.loc[df_output.Log == 'GBF_'+inst,['TimeAlg']] = t4
        df_output.loc[df_output.Log == 'GBF_'+inst,['TimeTotal']] = t3+t4

    if file == None:
        df_output.to_csv('Results/Table_Results_GBF'+p_name+'.csv')
    else:
        df_output.to_csv(file +'.csv')