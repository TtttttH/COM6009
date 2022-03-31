import numpy as np
import numba
import matplotlib
from ecolab.agents import Rabbit, RHD_Status, Gender, AgentType
import matplotlib.pyplot as plt


@numba.jit  
def run_ecolab(env, agents, Niteration=[0, 365], max_density = 20, earlystop=True):

    #record=[] 
    sus = []
    infected = []
    immune = []
    infant = []
    total =[]
    
    preg_prob = {11: 11/365,
                 0: 35/365,
                 1: 59/365,
                 2: 82/365,
                 3: 59/365,
                 4: 35/365,
                 5: 11/365,
                 6: 5/365,
                 7: 5/365,
                 8: 5/365,
                 9: 5/365,
                 10: 5/365}
    for it in range(Niteration[0], Niteration[1], 1):
        #print("iteration: %g" %it)
        month = (it / 30) % 12  ## 0~11 ->  Jan~Dec
        prob = preg_prob.get(int(month))
        for agent in agents:
            agent.other_daily_grow()
            if not agent.death:
                if agent.type == AgentType.Adults and agent.gender == Gender.Female and agent.pregnancy_days == -1:
                    agent.reproduct(agents,prob=prob)
                agent.move(env)
                agent.die()
            
        alive_agents = [a for a in agents if not a.death]
        death_in_90_days_agents = [a for a in agents if a.death and a.days_dead < 90]
        for agent in alive_agents:
            if agent.rhd_status == RHD_Status.Infected and agent.infected_days > 0:
                agent.infection(alive_agents)
            
            if agent.type == AgentType.Adults and agent.rhd_status == RHD_Status.Susceptible:
                agent.carcasses_infection(death_in_90_days_agents)
            newborn = agent.born_new_rabbit(agents, env, max_density)
            if newborn is not None:
                agents += newborn
        
        alive_agents = [a for a in agents if not a.death]
        #record.append({'susceptible agents': np.array([a.summary_vector() for a in alive_agents if a.type ==AgentType.Adults and a.rhd_status == RHD_Status.Susceptible]),
        #               'infected agents': np.array([a.summary_vector() for a in alive_agents if a.rhd_status == RHD_Status.Infected]),
        #              'immune agents': np.array([a.summary_vector() for a in alive_agents if a.rhd_status == RHD_Status.Recoverd_Immune])})   
        
        #print("the number of whole agents: ", len(agents))
        #print("the number of Female adults", len([a for a in agents if a.type == AgentType.Adults and a.gender == Gender.Female]))
        #print("the number of Male adults", len([a for a in agents if a.type == AgentType.Adults and a.gender == Gender.Male]))
        #print("the infants of the agents", len([a for a in agents if a.type == AgentType.Infants]))
        # print("the adults of the agents", len([a for a in agents if a.type == AgentType.Adults]))
        #print("the number of susceptible agents: ", len(record[it]['susceptible agents']))   
        #print("the number of infected agents: ", len(record[it]['infected agents']))
        #print("the number of immune agents:", len(record[it]['immune agents']))
        #print("===========================================")    
        sus.append(len([a for a in alive_agents if a.type == AgentType.Adults and a.rhd_status == RHD_Status.Susceptible]))
        infected.append(len([a for a in alive_agents if a.rhd_status == RHD_Status.Infected]))
        immune.append(len([a for a in alive_agents if a.rhd_status == RHD_Status.Recoverd_Immune]))
        total.append(len(alive_agents))
        infant.append(len([a for a in alive_agents if a.type == AgentType.Infants]))                 
        
        if earlystop:
            if len(agents)==0: break
    
    return sus, infected, immune, total, infant, agents
