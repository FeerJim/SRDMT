import random
import simpy
import numpy as np
import pandas as pd
import pyreadr

# Function to read a file in formart .arg
def read_arg(file_name):
    result = pyreadr.read_r(file_name)
    print(result.keys())
    Arg = result["Arg"]
    return Arg

class servers(object):
    """This class generate a server (machine) to process the customers (jobs)
    """
    def __init__(self, env, num_servers):
        self.env = env
        self.server = simpy.Resource(env, num_servers)

    def process_time(self, customer, scale, shape):
        """Calculate processing time"""
        yield self.env.timeout(round(float(scale * np.random.weibull(shape, 1))*60,0)/60)

def customer(env, name, Servers,scale,shape,R,start,end):
    """
    Generate the parameters of a customer (r_j, d_j, p_j)
    """
    aux1 = env.now # release time
    R[name] = aux1
    with Servers.server.request() as request:
        yield request
        aux2 = env.now # start processing time
        start[name] = aux2
        yield env.process(Servers.process_time(name,scale,shape))
        aux3 = env.now
        end[name] = aux3 # ending time

def setup(env, num_servers, _lambda, scale, shape,R,start,end):
    """Generate a server and how it process the customers."""
    server = servers(env, num_servers)
    i = 0
    # Generate customers while the server is open
    while True:
        yield env.timeout(round(float(np.random.exponential(1/_lambda, 1))*60,0)/60)
        i += 1
        env.process(customer(env, 'Customer %d' % i, server,scale,shape,R,start,end))

def gen_data(RANDOM_SEED, SIM_TIME, NUM_SERVERS, LAMBDA, SCALE, SHAPE,R,start,end,HOUR):
    # Setup and start the simulation
    random.seed(RANDOM_SEED)  # This helps reproducing the results

    # Create an environment and start the setup process
    env = simpy.Environment()
    env.process(setup(env, NUM_SERVERS, LAMBDA, SCALE, SHAPE,R,start,end))

    # Execute!
    env.run(until=SIM_TIME)

    # Save data
    df = pd.DataFrame([R,start,end]).transpose()
    df.columns = ['Minutos','Inicio_tramite','Fin_tramite']
    df['Duracion'] = df['Fin_tramite'] - df['Inicio_tramite']
    df['Horas'] = HOUR
    df['Ticket_ID'] = df.index +'_'+ str(HOUR)
    df = df[df.Minutos < 60]
    df['Duracion'] = df['Duracion'].fillna(float(SCALE * np.random.weibull(SHAPE, 1)))
    df['Duracion'] = round(df['Duracion'] * 60,0)
    df['Minutos'] = round(df['Minutos'] * 60,0)/60
    return df