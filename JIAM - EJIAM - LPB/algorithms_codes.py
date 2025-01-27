import pandas as pd
# --------------------------------
#       Greedy algorithm
# --------------------------------
def GA(s,f,Sr):
    ''' 
    - Input:

    s: nested dictionary indexed by customers and waiting time, using as values ​​the possible start times of service.

    f: nested dictionary indexed by customers and waiting time, using as values ​​the possible end times of service.

    S_r: a list with the start times of each period in minutes format.

    - Output:

    G: a dictionary with the order of customer service indexed by machines

    k: number of machines used for customer service

    V: dictionary indexed by machines and whose values ​​are the periods in which each machine is used.
    '''
    n = len(s)
    S = {i: {j: s[i][j] for j in s[i].keys()} for i in s.keys() }
    G = {}
    V = {}
    P = len(Sr)
    s_max = max([max(v.values()) for ind,v in S.items() ])
    k = 0
    while k < n:
        T = 0
        G[k] = set()
        V[k] = set()
        while s_max >= T:
            aux = [f[i][j] for i in S.keys() for j in S[i].keys() if S[i][j] >= T]
            if aux == []:
                break
            f_min = min(aux)
            i,j = [(i,j) for i in S.keys() for j in S[i].keys() if f[i][j] == f_min and S[i][j] >= T][0]
            G[k] |= {(i,j)}
            V[k] |= {r for r in range(P) if (S[i][j] >= Sr[r] and S[i][j] < Sr[r]+60) or (f[i][j] >= Sr[r] and f[i][j] < Sr[r]+60) }
            S.pop(i)
            T = f[i][j]
            if S == {}:
                break
            s_max = max([max(v.values()) for ind,v in S.items() ])
        k += 1
        if S == {}:
            break
    return G,k,V
# ----------------------------------------------------------------------
#           Function to convert the results into a data frame
# ----------------------------------------------------------------------
def df_GA(G,k,S,F,dfm,formato = False):
    '''
    - Input:

    G : dictionary indexed by the machines and its values ​​are sets with the clients served by the machine

    k : number of machine used
    
    Note: G and k are the parameters ​​returned in the function GA()

    S: Nested dictionary with possible service start times indexed by customer ID (i) and waiting time (j).
    
    F: Nested dictionary with the possible end times of service indexed by the customer ID (i) and the waiting time (j).

    dfm : the data frame that contains the clients in the same order that S and F were created.

    formato : a boolean parameter that allows us to save the results in DateTime format, by default the results are saved in minutes.

    - Output:

    dfm_sol: a DataFrame that contains the results.
    '''
    dfm_sol = dfm.copy()
    dfm_sol['Machine'] = 0
    dfm_sol[['WaitingTime','ReleaseTime', 'EndTime']] = 0.0
    for k in G.keys():
        for i,j in G[k]:
            dfm_sol.loc[dfm_sol.ID == i,['Machine']] = k+1
            dfm_sol.loc[dfm_sol.ID == i,['WaitingTime']] = j
            dfm_sol.loc[dfm_sol.ID == i,['ReleaseTime']] = S[i][j]
            dfm_sol.loc[dfm_sol.ID == i,['EndTime']] = F[i][j]
    if formato == True:
        dfm_sol['WaitingTime'] = pd.TimedeltaIndex(dfm_sol['WaitingTime'], unit='m').round('S')
        dfm_sol['ReleaseTime'] = pd.TimedeltaIndex(dfm_sol['ReleaseTime'], unit='m').round('S')
        dfm_sol['EndTime'] = pd.TimedeltaIndex(dfm_sol['EndTime'], unit='m').round('S')
    return dfm_sol

# --------------------------------------------------
# Create a class Graph
# --------------------------------------------------
class Graph:
    # Constructor
    def __init__(self, V, E):
        self.V = V
        self.E = E
        self.veci = { _ : [] for _ in V}
 
        # add edges to the undirected graph
        for (s, t) in E:
            self.veci[s].append(t)
            self.veci[t].append(s)
 
# --------------------------------------------------
# Function to color the nodes of a graph
# --------------------------------------------------
def greedy_coloring(G,colores,ind,result = {}):
    '''
    - Input: 
    
    G: a graph created with the graph class
    
    colores: a list of available colors
    
    result: a dictionary indexed by the nodes and with its value the assigned color, by default it is empty
    
    ind: a dictionary indexed by the nodes and with its value the waiting time
    
    - Output: 
    
    ind_sol: a list with the triplet (i,j,k) = (customer, waiting time, machine)
    
    result: the dictionary with the colors assigned to the nodes.
    '''
    
    # assign a color to each node
    for u in G.V:
        if u in result.keys():
            continue
        
        # extract the colors used in the neighboring nodes of u
        c_usados = set([result.get(i) for i in G.veci[u] if i in result])
        
        # look for the first available color
        color = 1
        for c in c_usados:
            if color != c:
                break
            color += 1
        if color not in colores:
            print('Error de coloreo')
            break
 
        # coloring the node u with the first available color
        result[u] = color
 
    ind_sol = { (v,ind[v], result[v]) for v in G.V }
    return ind_sol,result

# ---------------------------------------------------------------------------
#     Function to assign customers to machines using S-JIAM formulation
# ---------------------------------------------------------------------------
def generate_sol(S,F,ind_sol, z):
    '''
    - Input: 
    
    S: Nested dictionary with possible service start times indexed by customer ID (i) and waiting time (j).
    
    F: Nested dictionary with the possible end times of service indexed by the customer ID (i) and the waiting time (j).
    
    ind_sol: a list with the indices (i,j) = (customer, waiting time)
    
    z: number of machines
    
    - Output: 
    
    ind_reco_sol: a list with the triplet (i,j,k) = (customer, waiting time, window)
    '''
    H = {S[i][j] for (i,j) in ind_sol}
    ind_reco_sol = set()
    result = {}
    colores = list(range(1,z+1))
    for t in sorted(H):
        aux = { (i,j) for (i,j) in ind_sol if S[i][j] <= t < F[i][j] }
        ind = {i: j for (i,j) in aux}
        V = sorted(set(ind.keys()))
        E = { (u,v) for u in V for v in V if u < v }
        G = Graph(V,E)
        ind_reco_sol_new,result_new = greedy_coloring(G,colores,ind,result)
        ind_reco_sol |= ind_reco_sol_new
        result.update(result_new)
    return ind_reco_sol

# ----------------------------------------------------------------------
#   Function to assign customers to machines using EJIAM formulation
# ----------------------------------------------------------------------
def generate_sol_p(r,p,I,ind_x, M):
    '''
    - Input: 
    
    r: Dictionary with possible service start times
    
    I: Set of intervals
    
    ind_x: a list with the indices (j,i) = (customer, interval)
    
    M: number of machines
    
    - Output: 
    
    ind_reco_sol: a list with the triple (j,i,k) = (customer, waiting time, machine)
    '''
    ind_sol = []
    for i in I.keys():
        for j in r.keys():
            if (j,i) in ind_x:
                ind_sol.append((j,I[i].left - r[j]))
    H = {r[j]+i for (j,i) in ind_sol}
    ind_reco_sol = set()
    result = {}
    colores = list(range(1,M+1))
    for t in sorted(H):
        aux = { (j,i) for (j,i) in ind_sol if r[j]+i <= t < r[j]+i+p }
        ind = {j: i for (j,i) in aux}
        V = sorted(set(ind.keys()))
        E = { (u,v) for u in V for v in V if u < v }
        G = Graph(V,E)
        ind_reco_sol_new,result_new = greedy_coloring(G,colores,ind,result)
        ind_reco_sol |= ind_reco_sol_new
        result.update(result_new)
    return ind_reco_sol

# --------------------------------
#       Modify Greedy Algorithm
# --------------------------------
def MGA(s,f,Sr):
    ''' 
    - Input:

    s: List whose values ​​are the lists of possible service times for customers.

    f: List whose values ​​are the lists of possible end times for customer service.

    S_r: a list of the start times for each period in minutes format.

    - Output:

    G: a dictionary with the order of customer service indexed by machines

    k: Number of machines used for customer service

    V: dictionary indexed by machines and whose values ​​are the periods in which each machine is used.
    '''
    n = len(s)
    S = {i: {j: s[i][j] for j in s[i].keys()} for i in s.keys() }
    G = {}
    V = {}
    P = len(Sr)
    s_max = max([max(v.values()) for ind,v in S.items() ])
    k = 0
    while k < n:
        T = 0
        G[k] = set()
        V[k] = set()
        while s_max >= T:
            aux = [s[i][j]-T for i in S.keys() for j in S[i].keys() if S[i][j] >= T]
            if aux == []:
                break
            s_min = min(aux)
            aux2 = [(i,j) for i in S.keys() for j in S[i].keys() if s[i][j]-T == s_min and S[i][j] >= T]
            min_d = min(f[i][j] - s[i][j] for i,j in aux2)
            i,j = [(i,j) for i,j in aux2 if f[i][j] - s[i][j] == min_d][0]
            G[k] |= {(i,j)}
            V[k] |= {r for r in range(P) if (S[i][j] >= Sr[r] and S[i][j] < Sr[r]+60) or (f[i][j] >= Sr[r] and f[i][j] < Sr[r]+60) }
            S.pop(i)
            T = f[i][j]
            if S == {}:
                break
            s_max = max([max(v.values()) for ind,v in S.items() ])
        k += 1
        if S == {}:
            break
    return G,k,V


# -----------------------------------------------------------
#       Overlap measurement function of interval I at point x
# -----------------------------------------------------------
def _h(S,F,I,x):
    aux = 0
    for (i,j) in I:
        if S[i][j] <= x < F[i][j]:
            aux += 1
    return aux

# -----------------------------------------------------------
#       Job sorting function by duration
# -----------------------------------------------------------
def sort_p(S,F):
    p = {i: F[i][0] - S[i][0] for i in S.keys()}
    return dict(sorted(p.items(), key=lambda item: item[1]))

# -----------------------------------------------------------
#   Job sorting function by machine size
# -----------------------------------------------------------
def sort_delta(S,F):
    delta = {i: F[i][max([j for j in S[i].keys()])] - S[i][0] for i in S.keys()}
    return dict(sorted(delta.items(), key=lambda item: item[1]))

# --------------------------------
#      Greedy Best Fit Algoritm
# --------------------------------
def GBF(S,F,Sr,od = None):
    ''' 
    - Input:

    S: A nested dictionary indexed by customers and waiting times, using possible service start times as values.
    
    F: A nested dictionary indexed by customers and waiting times, using possible service end times as values.

    - Output:
    
    G: a dictionary with the order of customer service indexed by machines
    
    k: Number of machines used for customer service
    '''
    I = set()
    P = len(Sr)
    H = {int(S[i][j]) for i in S.keys() for j in S[i].keys()} | {int(Sr[r]) for r in range(P)}
    if od == None:
        od = sort_p(S,F)
    else:
        od = sort_delta(S,F)
    
    for i in od.keys():
        h_max = {}
        for j in S[i].keys():
            h_max[j] = max([_h(S,F,I,x) for x in set(range(int(S[i][j]),int(F[i][j])+1)).intersection(H)  ])
        h_min = min([h_max[j] for j in S[i].keys()])
        j_min = min([j for j in S[i].keys() if h_max[j] == h_min])
        I |= {(i,j_min)}
    aux = 0
    aux_z = 0
    aux_z_q = 0
    for r in range(P):
        aux = max([_h(S,F,I,x) for x in set(range(int(Sr[r]),Sr[r]+60)).intersection(H)  ])
        aux_z = max(aux,aux_z)
        aux_z_q += aux
        
    return I,aux_z,aux_z_q