
"""
@author: Eurus.T
"""
from enum import Enum
import numpy as np
import numba
import random


INFANT_NATURE_MOTALITY_RATE = 0.0002
ADULT_NATURE_MOTALITY_RATE = 0.0013
CARCASS_INFECTION_PROB = 0.5

class RHD_Status(Enum):
    """
    RHD_Status
    """
    Susceptible = 'susceptible'
    Infected = 'infected'
    Recoverd_Immune = 'immune'
    
 
class AgentType(Enum):
     """
     The type of rabbit according to different age
     """   
     Infants = 'infants'
     Adults = 'adults'
     
class Gender(Enum):
    """
    """
    Male = 1
    Female = 0
    
class Rabbit:
    """
    Base class for all types of rabbit
    """

    maxage = 3650
    def __init__(self,position,age,infected=False):
        self.position = position
        self.age = age
        self.pregnancy_days = -1
        self.infected_days = -1
        self.days_dead = -1
        if np.random.randint(0,2) == 1: ## 1 is male, 0 is female
            self.gender = Gender.Male
        else:
            self.gender = Gender.Female
        self.death = False
        if not infected :
            self.rhd_status = RHD_Status.Susceptible
        else :
            self.rhd_status = RHD_Status.Infected
            self.infected_days = 0
        if age < 90:
            self.speed = 0
            self.type = AgentType.Infants
        else :
            self.speed = 5
            self.type = AgentType.Adults
            # if self.gender == Gender.Female:
            #     self.reproduct_ablity = True
    @numba.jit         
    def try_move(self, newposition, env):
        if env.check_position(newposition):
            self.position = newposition
    
    @numba.jit        
    def move(self, env):
        if self.speed > 0:
            d = np.random.rand()*2*np.pi
            delta = np.round(np.array([np.cos(d),np.sin(d)]) * np.random.randint(0, self.speed + 1)) 
            self.try_move(self.position + delta, env)   
    
    @numba.jit    
    def die(self):
        
        if self.rhd_status == RHD_Status.Infected and self.infected_days > 0: ##infected death begins on the 2nd infected day
            # self.death = (np.random.rand() > 0.1 * self.infected_days)
            self.death = (np.random.rand() < np.exp(-self.infected_days))
            
        # nature death
        if self.age > self.maxage and not self.death: 
            self.death = True
    
        if not self.death:
            if self.type == AgentType.Infants:
                self.death =  (np.random.rand() < INFANT_NATURE_MOTALITY_RATE)
            
            else:
                self.death = (np.random.rand() < ADULT_NATURE_MOTALITY_RATE)
                
        if self.death:
            self.days_dead = 0
    
    @numba.jit 
    def infection(self, alive_agents):
        cnt = 0
        for a in alive_agents:
            if (a.type == AgentType.Adults and a.rhd_status == RHD_Status.Susceptible and (a.position == self.position).all()):
                #print("one rabbbit get infected")
                a.infected_days = 0
                a.rhd_status = RHD_Status.Infected
                cnt += 1
            if cnt == 2: break          
             
    @numba.jit 
    def carcasses_infection(self, death_in_90_days_agents):
        nearby_dead_agents = [a for a in death_in_90_days_agents if ((a.position == self.position).all())]
        if len(nearby_dead_agents) > 0 and np.random.rand() <= CARCASS_INFECTION_PROB:
            #print("there was a dead rabbit")
            self.infected_days = 0 
            self.rhd_status = RHD_Status.Infected

    @numba.jit 
    def reproduct(self, agents, prob):
        same_grid_male = [a for a in agents if (not a.death and a.type == AgentType.Adults and a.gender == Gender.Male and (a.position == self.position).all())]
        if len(same_grid_male) > 0 and np.random.rand() <= prob:
            self.pregnancy_days = 0
            # print("one female adult rabbit get pregant")
    
    
    @numba.jit             
    def born_new_rabbit(self,agents,env,max_density):
        if self.pregnancy_days > 30:
            self.pregnancy_days = -1
            alive_num = len([a for a in agents if not a.death])
            if alive_num / (env.shape[0] * env.shape[1]) < max_density:
                litter_num = np.random.randint(2, 8)
                newborn =[]
                i = 0
                for i in range(litter_num):
                    newborn.append(Rabbit(position = self.position, age = 1))
                    i += 1
                 # print("here is newborn , number:", len(newborn))
                return newborn
        
        return None
            
    @numba.jit 
    def other_daily_grow(self): # run this function first in each day loop
        if self.death == True:
            self.days_dead += 1
        else:
            self.age +=1
            if self.pregnancy_days > -1:
                self.pregnancy_days += 1
        
            if self.rhd_status == RHD_Status.Infected:
                self.infected_days += 1

            if self.infected_days > 10:
            # print("here is a new immune rabbit!")
                self.rhd_status = RHD_Status.Recoverd_Immune
                self.infected_days = -1    
            
            if self.age > 90:
                self.speed = 5
                self.type = AgentType.Adults
            
    @numba.jit        
    def summary_vector(self):
        return [self.position[0], self.position[1], self.type, self.rhd_status]