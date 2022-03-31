import numpy as np
import numba
from ecolab.agents import RHD_Status, Gender, AgentType
from datetime import datetime


@numba.jit  
def run_ecolab(env, agents, Niteration=[0, 360], max_density = 40, earlystop=True):
    start = datetime.now()
    record = []
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
        print("iteration:", it)
        month = (it / 30) % 12  ## 0~11 ->  Jan~Dec -> preg_prob {month-1: rate}
        prob = preg_prob.get(int(month))
        alive_agents = [] ## collect alive agents
        alive_male_adults = [] ## collect alive male adult agents
        alive_female_adults = [] ## collect alive female adult agents
        death_in_90_days_agents = [] ## collect death_days < 90days dead agents
        for agent in agents:
            agent.other_daily_grow()
            if not agent.death:
                # if agent.type == AgentType.Adults and agent.gender == Gender.Female and agent.pregnancy_days == -1:
                #     agent.reproduct(agents,prob=prob)
                agent.move(env)
                agent.die()
            if not agent.death:
                alive_agents.append(agent)
                if agent.type == AgentType.Adults:
                    if agent.gender == Gender.Male:
                        alive_male_adults.append(agent)
                    else:
                        alive_female_adults.append(agent)
            elif agent.days_dead < 90:
                death_in_90_days_agents.append(agent)
            
        # alive_agents = [a for a in agents if not a.death]
        # death_in_90_days_agents = [a for a in agents if a.death and a.days_dead < 90]
        for agent in alive_male_adults:
            if agent.rhd_status == RHD_Status.Infected and agent.infected_days > 0:
                    agent.infection(alive_agents)
            if agent.rhd_status == RHD_Status.Susceptible:
                    agent.carcasses_infection(death_in_90_days_agents)
                    
        for agent in alive_female_adults:
            # print("alive agents num:", len(alive_agents))
            if agent.rhd_status == RHD_Status.Infected and agent.infected_days > 0:
                agent.infection(alive_agents)
            if agent.rhd_status == RHD_Status.Susceptible:
                agent.carcasses_infection(death_in_90_days_agents)
            if agent.pregnancy_days == -1:
                agent.reproduct(alive_male_adults,prob=prob)
            if agent.pregnancy_days > 30:
                agent.born_new_rabbit(agents, alive_agents, env, max_density)
                    # newborn = agent.born_new_rabbit(agents, env, max_density)
                    # print("agents num :", len(agents))
                    # print("alive agents num:", len(alive_agents))
            # if newborn is not None:
            #     agents += newborn
        
        # alive_agents = [a for a in agents if not a.death]
        
        # count susceptible_adult_agents, infected_agents, immune_agents, infants, and total agents number
        sus_num = infected_num = immune_num = infant_num = 0
        
        for alive in alive_agents:
            if alive.rhd_status == RHD_Status.Infected:
                record.append(np.array([alive.position[0], alive.position[1], 1]))
                infected_num += 1
            elif alive.rhd_status == RHD_Status.Recoverd_Immune:
                record.append(np.array([alive.position[0], alive.position[1], 2]))
                immune_num += 1
            elif alive.rhd_status == RHD_Status.Susceptible and alive.type == AgentType.Adults:
                record.append(np.array([alive.position[0], alive.position[1], 0]))
                sus_num += 1
            elif alive.type == AgentType.Infants:
                infant_num += 1

        sus.append(sus_num)
        infected.append(infected_num)
        immune.append(immune_num)
        infant.append(infant_num)  
        total.append(len(alive_agents))              
        
        if earlystop:
            if len(agents)==0: break
    end = datetime.now()
    time_cost = (end - start).seconds
    print("The running time of experiment:", time_cost, "seconds.")
    return record, sus, infected, immune, total, infant, agents