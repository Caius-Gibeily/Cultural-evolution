#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import numpy as np
from random import random


import agentpy as ap
import math
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import IPython
from numpy.linalg import norm
import random as rd
from numpy.random import default_rng
import seaborn as sns
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
rng = default_rng()

#%%
def cos_similarity(lst1,lst2):
    """
    Calculate the cosine similarity between two input lists. 
    If either list is a float/int, it is converted into a 1D list. 
    If both input items are 1D lists, a zero list (n=20) is appended to them

    Parameters
    ----------
    lst1 : Firt list
    lst2 : Second list

    Returns
    -------
    cos_sim : Similarity between lists

    """
    if isinstance(lst1,(int,float)) and isinstance(lst2,(int,float)):
        lst1a = [lst1]
        for i in range(20): lst1a.append(i)
        lst2a = [lst2]
        for i in range(20): lst2a.append(i)
    elif isinstance(lst1,(int,float)):
        lst1a = [lst1]
        lst2a = lst2
    elif isinstance(lst2,(int,float)):
        lst1a = lst1
        lst2a = [lst2]
    else:
        lst1a = lst1
        lst2a = lst2
    lst1f = lst1a.copy()
    lst2f = lst2a.copy()

    if has_lists(lst1):
        lst1f = flatten_list(lst1f)

    if has_lists(lst2):
        lst2f = flatten_list(lst2f)

    if len(lst1f) > len(lst2f):
        for i in range(len(lst1f)-len(lst2f)):lst2f.append(0)
    elif len(lst2f) > len(lst1f):
        for i in range(len(lst2f)-len(lst1f)):lst1f.append(0)
    cos_sim = np.dot(lst1f,lst2f)/(norm(lst1f)*norm(lst2f))
    return cos_sim

#################### List manipulation ################
def nested_list_to_set(nested_list):

    if isinstance(nested_list, list):
        return tuple(nested_list_to_set(item) for item in nested_list)
    else:
        return nested_list

def set_to_nested_list(s):
    """
    Inverse operation nested_list_to_set()

    Parameters
    ----------
    s : Input set

    Returns
    -------
    Nested list

    """
    if isinstance(s, tuple):
        return [set_to_nested_list(x) for x in s]
    else:
        return s

def find_non_intersecting_elements(lst1, lst2):
    """
    Determines which if any of the elements between the two lists are present
    only in one or the other list

    Parameters
    ----------
    lst1 : First list
    lst2 : Second list

    Returns
    -------
    Non-intersecting elements. {} if none

    """
    if isinstance(lst1,(int,float)):
        lst1a = [lst1]
    else:
        lst1a = lst1
    set1 = {nested_list_to_set(x) for x in lst1a}
    set2 = {nested_list_to_set(x) for x in lst2}

    non_intersecting_elements = set1.symmetric_difference(set2)


    result = [set_to_nested_list(x) for x in non_intersecting_elements]
    return result

def find_intersection_elements(lst1, lst2):
    """
    Inverse of find_non_intersecting_elements

    Parameters
    ----------
    lst1 : First list
    lst2 : Second list

    Returns
    -------
    Intersecting elements

    """

    if isinstance(lst1, (int, float)):
        lst1a = [lst1]
    else:
        lst1a = lst1

    set1 = {nested_list_to_set(x) for x in lst1a}
    set2 = {nested_list_to_set(x) for x in lst2}

    intersection_elements = set1.intersection(set2)

    result = [set_to_nested_list(x) for x in intersection_elements]
    return result

def has_lists(input_list):
    """
    Check whether a given list itself contains sublists

    Parameters
    ----------
    input_list : Test list

    Returns
    -------
    bool
        

    """
    if isinstance(input_list,(int,float)):
        return False
    for item in input_list:
        if isinstance(item, list):
            return True

    return False

def flatten_list(nested_list):
    """
    Flatten a nested list 

    Parameters
    ----------
    nested_list : Nested list with any level recursion 

    Returns
    -------
    flattened_list : Flattened list preserves sequence order

    """
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list

def count_nested(lst,to_dict=False):
    """
    Counts the number of tokens in a given list, allowing for list entries to be nested

    Parameters
    ----------
    lst : Input list (typically the token set)
    to_dict : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    Counts per token
    """
    out = []
    for k in lst:
      output = sum([1 for i in lst if i==k])
      out.append(output)

    if to_dict:
      counts = {}
      for i,item in enumerate(lst):
        species = {str(item):out[i]}
        counts.update(species)
      return counts
    return out

def round_list(lst):
    """
    Returns the base tokens, b, of all unit tokens in the complex token.

    Parameters
    ----------
    lst : Input token or token set

    Returns
    -------
    Rounded token

    """
    input_list = lst
    if isinstance(input_list, list):
        return [round_list(item) for item in input_list]
    elif isinstance(input_list, float):
        return int(input_list)
    else:
        return input_list

def depth(lst):
    """
    Determine the depth (level of recursion) of the input list

    Parameters
    ----------
    lst : Input list, typically tokens 

    Returns
    -------

    """
    if not lst or not isinstance(lst,list):
        return 0
    else:
        lst2 = lst + [1]
        return isinstance(lst2, list) and max(map(depth, list(filter(None,lst2))))+1

def flat_nested(nested_list, flat_list):
    """
    Given a matching nested structure, transform a flattened list back into a nested list.

    Parameters
    ----------
    nested_list 
    flat_list :

    Raises
    ------
    ValueError
        If there is a mismatch between list sizes

    Returns
    -------
    The nested list output

    """
    flat_iter = iter(flat_list)

    def replace_recursive(nested_list):
        result = []
        for item in nested_list:
            if isinstance(item, list):
                result.append(replace_recursive(item))
            else:
                try:
                    result.append(next(flat_iter))
                except StopIteration:
                    raise ValueError("Flat list is shorter than the nested list.")
        return result

    return replace_recursive(nested_list)

################################ Information theory #######################################
def redundancy(lst):
    """
    Calculate the redundancy (proportion of total list comprising a given base token)

    Parameters
    ----------
    lst : input list

    Returns
    -------
    list
        redundancy_list

    """
    if not lst:
        return []  # Handle the case where the list is empty
    rounded_values = round_list(lst)
    unit_frequencies = count_nested(rounded_values)

    total_elements = len(lst)
    redundancy_list = [i / total_elements for i in unit_frequencies]

    return redundancy_list


def sdi(data):
    ## Source: https://gist.github.com/audy/783125
    """
    Calculate the Shannon Diversity Index
    """
    from math import log as ln

    def p(n, N):
        """ Relative abundance """
        if n == 0:
            return 0
        else:
            return (float(n)/N) * ln(float(n)/N)

    N = sum(data.values())

    return -sum(p(n, N) for n in data.values() if n != 0)



def self_information(values):
    """
    Calculates the self-information for a given input sequence

    Parameters
    ----------
    values : Input sequence

    Returns
    -------
    self_info_list : List of self information values per item
    red : Redundancy of each item in the list
    """
    # Calculate the frequencies of values with the same unit part
   #unit_frequencies = Counter(int(v) for v in values)
    rounded_values = round_list(values)
    unit_frequencies = count_nested(rounded_values)
    self_info_list = []  # Initialize a list to store self-information for each element

    # Calculate the self-information for each element
    for i,value in enumerate(values):
        #print(i)
        # Calculate the probability of occurrence within the same unit
        probability = unit_frequencies[i] / len(values)

        # Calculate self-information using Shannon entropy formula
        self_info = -np.log2(probability)

        self_info_list.append(self_info)  # Add self-information to the list

    red = redundancy(values)

    return self_info_list, red

def optimise(values,fitnesses,test_succ=False):
    """
    Follows the burden equation described in the main text, comparing token 
    fitness against its self information, redundancy and total redundancy of the token set

    Parameters
    ----------
    values : list of tokens
    fitnesses : their fitness values
    test_succ : TYPE, optional
        Select whether to compare burden against fitness. The default is False.

    Returns
    -------
    output : Burden values
    unit_frequencies : Frequencies of each element in the list

    """
    #print(values,"----")
    rounded_values = round_list(values)
    #print(rounded_values,"---")
    unit_frequencies = count_nested(rounded_values)

    self_info_list, red = self_information(values)
    total_red = sum(redundancy(flatten_list(values)))
    s = 0.8 ## scaling factor
    #count = [unit_frequencies[i] for i in rounded_values]
    output = [(self_info_list[i] + red[i]*unit_frequencies[i] + total_red) for i in range(len(values))]
    if test_succ:
        output = [fitnesses[i] - (self_info_list[i] + red[i]*unit_frequencies[i] + total_red) for i in range(len(values))]
    return output, unit_frequencies

#########################

def get_indices(lst,base):
    """
    Find indices for a target base token in the list. Used in consolidation

    Parameters
    ----------
    lst : token set
    base : target token

    Returns
    -------
    findices : Output indices corresponding to locations in the token set

    """
    findices = [i for i in range(len(lst)) if np.array_equal(np.array(round_list(lst)[i], dtype="object"), base)]
    return findices

def interpolate_y(x,mean,scale,upper):
    """
    calculate fitness value for a token variant given the information in "distros" on a base token's mean, scale and peak value

    Parameters
    ----------
    x : base value
    mean : corresponding mean value
    scale : scale of standard deviation
    upper : peak value
    Returns
    -------
    fitness value

    """
    mean_pdf = 1 / (scale * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((mean - mean) / scale) ** 2)
    pdf_value = ((1 / (scale * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / scale) ** 2))/(mean_pdf)) * upper
    return(pdf_value)


def find_fitness(lst):
    """
    Calculates the fitness value for a given input token. For complex tokens, 
    the sum of unit token fitnesses is calculated

    Parameters
    ----------
    lst : Input token

    Returns
    -------
    

    """
    global distros
    lst2 = lst.copy()
    if not lst2:
        return 0

    el = lst2.pop()
    if isinstance(el, list):
        return find_fitness(el) + find_fitness(lst2)

    else:
        t_class = math.floor(el)
        row = distros[np.where(distros[:,0]==t_class)[0][0]]

        el = interpolate_y(el,t_class+row[2],row[1],row[3])

        return el + find_fitness(lst2)

def mapords(lst):
    s = sorted(range(len(lst)), key=lambda i: lst[i])
    o = [-1] * len(lst)
    for i,j in enumerate(s):
        o[j] = i
    return o


def j_curve(x,scale=1.1):
    """
    Generate the mortality curve to be used to determine P(mortality)

    Parameters
    ----------
    x : ages matrix
    scale : α scaling factor

    Returns
    -------
    y : P(mortality)

    """
    y = []
    for i, v in enumerate(x):
        y.append(scale**x)
    y = y[0]/max(y[0])
    return y
x = np.arange(0,99,1)
mort_vector = j_curve(x)


def gumbel_learning(scale=1.2,plot=False):
    """
    Generates the learning curve to be used to determine the learning rate

    Parameters
    ----------
    scale : β scaling factor
    plot : plot learning curve, True/False

    Returns
    -------
    y : gumbel curve

    """
    from scipy.stats import gumbel_r
    mean, var, skew, kurt = gumbel_r.stats(moments='mvsk')
    x = np.linspace(gumbel_r.ppf(0.1),
                    gumbel_r.ppf(0.99), 100)
    y = gumbel_r.pdf(x,scale=scale)
    y = y/max(y)
    if plot:
        fig, ax = plt.subplots(1, 1)

        x = x/max(x)*80
        ax.plot(x+abs(min(x)), y,
               'r-', lw=5, alpha=0.6, label='gumbel_r pdxf')
    return y
gumbel = gumbel_learning()


def generate_gauss(n_int, variance,pos_skew):
    """
    Core function that generates the distros array that specifies the innovation
    space and fitness values per base token 

    Parameters
    ----------
    n_int : range of the innovation space
    variance : variance of the Gaussians 
    pos_skew : skew relative to the centre

    Returns
    -------
    distros : innovation space array

    """
    distros = np.zeros((n_int,4))
    upper = np.random.normal(0+pos_skew,size=n_int,scale=variance)
    mean = np.random.uniform(0.3,0.7,size=n_int)
    var = abs(np.random.normal(0,size=n_int))
    distros[:,0] = np.arange(-1 * n_int//2,n_int//2,1) + n_int//2 + 1
    distros[:,1],distros[:,2],distros[:,3] = var,mean,upper
    return distros

innovation_limit = 200
distros = generate_gauss(innovation_limit,2,2)
#%%

class person(ap.Agent):
    """
    Agent class
    """
    def setup(self):
        """
        Specify agent variables, including user-defined variables (self.p)

        Returns
        -------
        None.

        """
        self.buffer = self.p.buffer
        #self.capacity = self.p.capacity
        self.burden = 0
        self.tokens = []
        self.fitness = []
        self.success = 0
        self.exp_tokens = 1
        self.age = np.random.randint(0,40)
        self.state = "A"
        self.position = [0,0]
        self.nsuccess = []
        self.nodelist = []
        self.depth = 0
        self.ilearn = 0
        self.overim = 0
        self.ifalse = 0
        self.ofalse = 0
        self.homo = 0
    def calc_fitness(self):
        """
        Calculates fitness and success values for a given agent

        """
        self.fitness = []

        global distros
        for i,token in enumerate(self.tokens):

            self.fitness.append(find_fitness([token]))
            #self.fitness.append(interpolate_y(self.tokens[i],t_class+distros[t_class,2],distros[t_class,1],distros[t_class,3]))
        self.success = sum(self.fitness)
        return self.fitness,self.success
    def initialise_tokens(self,naive=False):
        """
        Initialise tokens for all agents before i = 0 or for naive agents following
        agent death

        Parameters
        ----------
        naive : TYPE, whether to return naive agents or not
            

        """
        self.position = [np.random.randint(0,self.p.lim),np.random.randint(0,self.p.lim)]
        if naive:
            self.age = 0
        else:
            global distros
            n_tokens = np.random.poisson(self.exp_tokens,1)[0] ## could be set to uniform value

            if n_tokens > 0:
                self.tokens = list(np.round(np.random.normal(np.mean(distros[:,0]),scale=self.p.scale,size=n_tokens),1))

                self.fitness,self.success = self.calc_fitness()



    def initiate_dunbar(self):
        """
        Initialise the social network for each agent, weighting learning edges 
        according to the normalised relative success of agents in the learnig radius, r

        """
        self.ilearn = 0
        self.overim = 0
        ### output nodelist of likely candidate models
        nlist = self.space.neighbors(self,self.p.radius).to_list() # visibility range

        nsuccess = [nlist[i].success for i in range(len(nlist))]
        sumsuccess = []

        if sum(nsuccess) > 0:
            sumsuccess = [1-round(i/sum(nsuccess),1) for i in nsuccess]
            
            #sumsuccess = [round(np.random.uniform(-1,1),1) for i in nsuccess]

            self.nodelist = [(self.node, nlist[i].node, {"weight": sumsuccess[i]}) for i in range(len(nlist))]

        else:
            self.nodelist = [(self.node, nlist[i].node, {"weight": 1}) for i in range(len(nlist))]
        self.network.graph.add_edges_from(self.nodelist)

        return self.nodelist


    def minimise(self):
        """
        Memory consolidation function, divided into forgetting, consolidation | Chunking

        """
        #self.tokens= get_unique(self.tokens)[0]
        self.fitness,self.success = self.calc_fitness()
        #self.fitness = list(filter(None, self.fitness))
        if len(self.tokens) > 0:
                output,count = optimise(self.tokens,self.fitness,test_succ=True)
                output = np.array(output,dtype="object")
                self.burden = sum(output)
                cthreshold = self.p.cprob
                prob2 = np.random.rand()
                ## Forget #
                
                if len(output[output <= 0]) >= 1:
                    findex = np.where((output <= 0))[0]
                    if len(findex) > 1:
                        findex = [np.random.choice(findex)]

                    del self.tokens[findex[0]]
                    del self.fitness[findex[0]]
                    self.success = sum(self.fitness)
                    
                ## Consolidate #
                output,count = optimise(self.tokens,self.fitness)
                output = np.array(output,dtype="object")
                if prob2 < cthreshold and any(i > 1 for i in count):

                    bases = np.array(self.tokens,dtype="object")[np.array(count) > 1].tolist(); rbases = round_list(bases)
                    base = [rbases[i] for i in np.random.randint(0,len(rbases),1)]
                    rounded_tokens = round_list(self.tokens)
                    findices = get_indices(rounded_tokens,base[0])
                    rindices= np.setdiff1d(np.arange(0,len(self.tokens),1),findices).astype(int).tolist()
                    dups = np.random.choice(findices,2,replace=False)
                    self.token = np.array(self.tokens,dtype="object")
                    self.fitnes = np.array(self.fitness,dtype="object")
                    if isinstance(self.token[dups],np.ndarray):
                        vals = self.token[dups].tolist()
                    else:
                        vals = self.token[dups]
                    if not has_lists(vals):
                        consolidated = round(np.mean(vals),1)
                    else:
                        consolidated = self.token[dups][0]
                        if len(consolidated) == 1 and not has_lists(consolidated):
                            consolidated = consolidated[0]

                    if isinstance(consolidated,np.ndarray):
                        consolidated = consolidated.tolist()
                    self.tokens = np.delete(self.token,dups).tolist()
                    self.tokens.append(consolidated)
                    self.fitness = np.delete(self.fitnes,dups).tolist()
                    if isinstance(consolidated,(float,int)):
                        consolidat = [consolidated]
                    else:
                        consolidat = consolidated


                    self.fitness.append(find_fitness(consolidat))
                    self.success = sum(self.fitness)

                ## Chunk #
                elif prob2 > cthreshold and len(self.tokens) >= 2:
                    n_chunking = np.random.poisson(2); n_chunking = np.clip(n_chunking,2,len(self.tokens))
                    chunk_ind = rng.choice(len(self.tokens),size=n_chunking,replace=False)
                    chunk = [self.tokens[i] for i in chunk_ind]
                    for ele in sorted(chunk_ind, reverse = True):
                        del self.tokens[ele]
                        del self.fitness[ele]

                    self.tokens.append(chunk)
                    self.fitness.append(find_fitness(chunk))

        self.success = sum(self.fitness)

    def innovate(self):
        """
        Innovation, returning a random token within the innovation space

        """
        ## novel class
        tokens = self.tokens.copy()
        self.tokens = tokens.copy()
        discovery_rate = self.p.p_innov
        if np.random.rand() <= discovery_rate:
            token = round(np.random.choice(np.arange(distros[0,0],distros[-1,0],1)) + round(np.random.rand(),1),1)

            self.tokens.append(token)
            self.fitness,self.success = self.calc_fitness()


    def inductive_learning(self,learn_tokens):
        ### Iterate over tokens in learn list.against items in long-term buffer ###
        ### Perform matching for chunk/unitary tokens to calculate net cognitive load. Effective priors will minimise the load and support complex
        ### learning.
        #learn_tokens = [[9.6,1,3],[9,3],1]
        #selflist = [[1,3],9]
        learned_list = [False for i in range(len(learn_tokens))]

        threshold = self.p.learn_thresh
        #threshold = 0.1
        tokenlist = self.tokens.copy()

        sim = []

        for i, ltoken in enumerate(learn_tokens):
            t = self.buffer
            test_list = tokenlist.copy()
            while 0 < t <= self.buffer:
                similarity = []
                for j,stoken in enumerate(test_list):

                    similarity.append(cos_similarity(stoken,ltoken))
                    #similarity.append(hamming_distance(stoken,ltoken))
                if max(similarity) > threshold:
                    top = [x for _, x in sorted(zip(mapords(similarity),test_list),reverse=True)]

                    #print(top,"!!!")
                    if isinstance(top[0],(int,float)):
                        learned_list[i] = top[0]
                        sim.append(max(similarity))
                        self.ilearn += 1
                        break
                    else:

                        if len(flatten_list(top[0])) == len(flatten_list(ltoken)):

                            learned_list[i] = flat_nested(ltoken,flatten_list(top[0]))

                        #learned_list[i] = top
                            sim.append(max(similarity))
                            self.ilearn +=1
                            break

                    #print(learned_list[i],"***")
                    sim.append(max(similarity))

                    break

                top_hits = [x for _, x in sorted(zip(mapords(similarity),test_list),reverse=True)]
                for j,val in enumerate(top_hits):
                    if isinstance(val,(int,float)):
                        top_hits[j] = [top_hits[j]]

                selflist = tokenlist.copy()
                for j,val in enumerate(selflist):
                    if isinstance(val,(int,float)):
                        selflist[j] = [selflist[j]]

                test_list = [i + j for i in top_hits[:3] for j in selflist]
                t -= 1
                if t == 0:
                  sim.append(max(similarity))
                  self.ifalse += 1
        return learned_list,sim


    def imitate(self,token):
        imitation_thresh = self.p.p_overimitate
        im = np.random.rand()
        if im < imitation_thresh:
            if isinstance(token,(int,float)):
                token = [token]
            selfi = sum(self_information(flatten_list(token))[0])

            if selfi < self.buffer:
                self.overim += 1
                return True
            else:
                self.ofalse += 1
                return False

    def learn(self):
        """
        Run learn function. Calculates agent social networks, selects a 
        model, selects a number of tokens to learn with a target teaching strategy
        (random, low-high ordered or high-low ordered) and allows agents to attempt learning inductively


        """
        global gumbel
        learn = np.random.uniform(0,1)
        if learn <= gumbel[self.age]:

            self.initiate_dunbar()
            ## select test model
            weights = [int(d["weight"]*10) for (u,v,d) in self.nodelist]
            models= [v for (u,v,d) in self.nodelist]

            choices = [elem for n, elem in zip(weights, models) for i in range(n)]
            if len(choices) == 0:
                return

            mimic = np.random.choice(choices)
            mimic = self.model.agents.select(self.model.agents.node == mimic)
            mtokens = mimic.tokens[0]

            if len(mtokens) == 0:
                return
            num_learn = np.clip(np.random.poisson(1,size=1),1,len(mtokens))

            learn_subset = [mtokens[i] for i in rng.choice(len(mtokens),size=num_learn,replace=False)]
            ordered = []
            for i,token in enumerate(learn_subset):
                if isinstance(token,(float,int)):
                    ordered.append(1)
                else:
                    ordered.append(len(flatten_list(token)))
            ## learn simplest to most complex ##

            learn_subset = [x for _, x in sorted(zip(mapords(ordered),learn_subset),reverse=False)]

            if len(self.tokens) > 0:

                learned_list,sim = self.inductive_learning(learn_subset)

                for i,token in enumerate(learned_list):

                    if token==False:

                        outcome = self.imitate(learn_subset[i])

                        if outcome:
                            learned_list[i] = mtokens[i]

                learned_list = list(filter(bool,learned_list))
                for i,item in enumerate(learned_list):
                    if isinstance(item,list) and len(item) == 1 and not has_lists(item):

                        learned_list[i] = item[0]
                a = find_non_intersecting_elements(learned_list + self.tokens, self.tokens)

                for i in a:
                    if i:
                        self.tokens.append(i)


                self.fitness,self.success = self.calc_fitness()
                #print(self.tokens)

            else:
                for i in learn_subset:
                    outcome = self.imitate(i)
                    if outcome:
                        if isinstance(i,(float,int)):
                            test = [i]
                        else:
                            test = i
                        b = find_non_intersecting_elements(test + self.tokens, self.tokens)
                        if b:
                            self.tokens.append(i)

                        self.fitness,self.success = self.calc_fitness()

            #print(self.tokens,"!!")
        #self.depth = depth(self.tokens)
            #weight = self.network.graph[self.node][mimic.node[0]]["weight"]
            #self.network.graph.add_edge(self.node,mimic.node[0],weight=weight+5)

    def die(self):
        """
        Mark the agent for death during the update step

        """
        self.model.agents.remove(self)

    def ageing(self):
        """
        Increase the age of the agent by 1. Also update other self values
        (homo - SDI, burden)

        Returns
        -------
        None.

        """
        self.age += 1
        #global mort_vector
        x = np.arange(0,99,1)
        mort_vector = j_curve(x,self.p.scale_death)
        if np.random.uniform(0,1) <  mort_vector[self.age]:
            self.state = "D"
        outcome,count = optimise(self.tokens,self.fitness)
        self.burden = sum(outcome)
        self.homo = sdi(count_nested(flatten_list(self.tokens),True))
        ## select agent imitation pool
        ## of imitation pool, select agent to copy
        ## run through inference algorithm
        ## update model


###################################

def mirror(x,**kwargs):
    """
    Dummy function for overcoming Agentpy reporting bug. Returns an exact copy of the input object
    """

    if isinstance(x,dict):
        y = {}
        y = {k:v for (k,v) in zip(x.keys(), x.values())}
    elif isinstance(x,list):
        a = [i for i in x]
        b = nx.DiGraph()
        b.add_nodes_from(kwargs["nodes"])
        b.add_edges_from(a)
        y = b
    else:
        y = x*1
    return y

class CE_model(ap.Model):
    """
    The model class. It is divided into get_pos, setup, step and update functions
    """
    def get_pos(self):
        """
        Get agent positions from node data


        """
        pos_set = {}
        for i in self.network.nodes:
            ag = self.agents.select(self.agents.node == i)
            pos_set[i] = list(ag.position)[0]
        return pos_set
    def setup(self):
        """
        Initialise the model, setting up the graph object, n agents and the spatial field

        """
        # Prepare a small-world network
        graph = nx.empty_graph(
        self.p.population)
        self.graph = nx.DiGraph(graph)

        # Create agents and network
        self.agents = ap.AgentList(self, self.p.population, person)
        self.agents.initialise_tokens(naive=False)

        self.network = self.agents.network = ap.Network(self, self.graph)
        self.agents.node = self.network.nodes

        self.network.add_agents(self.agents, self.network.nodes)

        pos_set = self.get_pos()
        nx.set_node_attributes(self.network.graph, pos_set, 'pos')

        poslist = []
        for a in self.agents:
            poslist.append(a.position)
        self.space = self.agents.space = ap.Space(self,shape=[self.p.lim,self.p.lim])
        self.space.add_agents(self.agents,positions=poslist)
        self.i = 0

    def step(self):
        """
        Specifies what is performed during each iteration (i). 

        Returns
        -------
        None.

        """
        for i in range(self.p.period):
            self.agents.innovate()
            self.agents.learn()
            self.agents.minimise()
        ### Population statistics ###
        self.agents.ageing()
        self.remove = self.agents.select(self.agents.state == "D")
        self.space.remove_agents(self.remove)

        #self.network.remove_node(self.remove.node)
        #self.network.remove_agents(self.remove)
        for i in self.remove:
            self.network.graph.remove_node(i.node)

        agents_to_remove = [agent for agent in self.agents if agent.state == "D"]
        for agenta in agents_to_remove:
            agenta.die()

        diff = self.p.population - len(self.model.agents)
        if diff > 0:
            self.list_of_new_agents = ap.AgentList(self, diff, person)
            self.list_of_new_agents.initialise_tokens(naive=True)

            self.agents.extend(self.list_of_new_agents)

            self.network.add_agents(self.list_of_new_agents)
            self.agents.network  = self.network


            self.agents.node = self.network.nodes


            pos_set = self.get_pos()
            nx.set_node_attributes(self.network.graph, pos_set, "pos")

            #self.space.add_agents(self.list_of_new_agents,random=True)
            poslist = []
            for a in self.agents:
                poslist.append(a.position)
            self.space = self.agents.space = ap.Space(self,shape=[self.p.lim,self.p.lim])
            self.space.add_agents(self.agents,positions=poslist)

        print(np.mean([self.agents.success]))
        #print(self.ages)
        ##############################

    def update(self):
        """
        Updates the selected variables below


        """
        self.record("ages", mirror(self.agents.age))
        self.record("positions", mirror(nx.get_node_attributes(self.network.graph,"pos")))
        self.record("successes",mirror(self.agents.success))
        self.record("networks", mirror(list(self.network.graph.edges(data=True)),nodes=list(self.network.graph.nodes(data=True))))
        self.record("mean_success",mirror(np.mean([self.agents.success])))
        self.record("tokens",mirror(self.agents.tokens))
        self.record("burden",mirror(self.agents.burden))
        self.record("overim",mirror(np.sum([self.agents.overim])))
        self.record("inductive",mirror(np.sum([self.agents.ilearn])))
        self.record("homo",mirror(self.agents.homo))
        self.record("ifalse",mirror(np.sum([self.agents.ifalse])))
        self.record("ofalse",mirror(np.sum([self.agents.ofalse])))
        self.record("inductive_self",mirror(self.agents.ilearn))
        self.record("overim_self",mirror(self.agents.overim))
    
        self.agents.ilearn = 0
        self.agents.overim = 0
        self.agents.ofalse = 0
        self.agents.ifalse = 0
    def end(self):
        self.agents.record('tokens')

        self.agents.record('fitness')
        self.agents.record('age')
        self.agents.record('success')
        self.agents.record('burden')
        self.agents.record('depth')

        #self.report("success",np.mean([self.agents.success]))

#%%
def run_exp(parameters,resolution,n_reps):
    """
    Runs an experiment for a given parameter set and parameter range

    Parameters
    ----------
    parameters : Dict object of model parameters
    resolution : Specifies the sampling interval within a given parameter range
    n_reps : Number of replicates to perform

    Returns
    -------
    results : ap.DataDict object

    """
    sample = ap.Sample(parameters, n=resolution)
    exp = ap.Experiment(CE_model, sample, iterations=n_reps, record=True)
    results = exp.run()
    return results

def run_trial(parameters):
    """
    Run a single trial with the input parameter list specified 

    Parameters
    ----------
    parameters : Dict object

    Returns
    -------
    results : ap.DataDict object

    """
    model = CE_model(parameters)
    results = model.run()
    return results

