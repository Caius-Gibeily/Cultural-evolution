#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:15:08 2023
"""

import os
import numpy as np
import pandas as pd
import pickle 
import matplotlib.pyplot as plt
os.chdir("/home/caiusgibeily/Documents/Emory/MBC 501/Report/Model/")
import CE_model as model
import math
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import networkx as nx
########### Data collection ###############
########### One place #####################


#%% Parameter search analyses

from scipy import interpolate
def run_search(param1,range1,param2,range2,resolution,n_reps,xlabel,ylabel,**kwargs):
    def param_search(param1,range1, param2, range2, resolution,n_reps):
        p1,p2 = resolution
    
        area = np.zeros([p1,p2])
    
        space = pd.DataFrame(area)
    
        parameters = {'population': 100, 'steps': 70, 'period': 1,'lim':300,'radius':60, 'buffer': 7, 'p_innov': 1/10000, 'p_overimitate':1,'learn_thresh':0.999999999999,"scale_death":1.1,"scale":1} ## Default parameters
    #999
        parameters[param1] = model.ap.Range(range1[0],range1[1])
        #val_range = np.logspace(range2[0],range2[1],p2)
        parameters[param2] = model.ap.Range(range2[0],range2[1])
        if "steps" in kwargs:
            parameters["steps"] = kwargs["steps"]
        results = model.run_exp(parameters,resolution[0])
        
        fname = param1 + "_" + str(range1[0]) + "-" + str(range1[1]) + 'w_' + param2 + str(range2[0]) + "-" + str(range2[1]) + "_rep" + str(n_reps)
        space.index = np.unique(results["parameters"]["sample"].iloc[:,0])
        space.columns = np.unique(results["parameters"]["sample"].iloc[:,1])
    
        return results,space,fname

    results,space,fname = param_search(param1,range1,param2,range2,resolution,n_reps)

    def calc_output(results,fill,output,fname,**kwargs):
    
        fill2 = fill.copy()
        results2 = results.copy()
        var = results2["variables"]["CCE_model"]
        if "time" in kwargs:
            data = var.iloc[var.index.get_level_values('t') == kwargs["time"]]
        else:
            data = var.iloc[var.index.get_level_values('t') == var.index[-1][2]]
    
        ids = results2["parameters"]["sample"]
        for i in range(len(ids)):
    
            it = ids.iloc[i]
            ## average across iterations
            #lengths = []
            #for j in range(len(ids)):
            #  lengths.append(len(results2["variables"]["person"]["tokens"].loc[i]))
            #out = np.mean(lengths)
            out = np.mean(data[output][i])
            #out = np.mean(results3["variables"]["person"].loc[i]["burden"])
            fill2.loc[it[0],it[1]] = out
        if "save" in kwargs and kwargs["save"] == True:
    
            with open(fname +'.pkl', 'wb') as f:
                pickle.dump(results, f)
            np.savetxt("distro_" + fname + ".csv",model.distros)
        return fill2
    
    
    def create_mesh(z_grid,**kwargs):
        x = np.linspace(z_grid.index[0], z_grid.index[-1], len(z_grid))
        y = np.linspace(z_grid.columns[0], z_grid.columns[-1], len(z_grid))
        x_grid, y_grid = np.meshgrid(x, y)
    
        # Create a higher-resolution meshgrid
        x_highres = np.linspace(z_grid.index[0], z_grid.index[-1], 15)
        y_highres = np.linspace(z_grid.columns[0], z_grid.columns[-1], 15)
        x_grid_highres, y_grid_highres = np.meshgrid(x_highres, y_highres)
    
        z_grid2 = z_grid.T.to_numpy()
        # Interpolate the values onto the higher-resolution grid
        z_grid_highres = interpolate.griddata((x_grid.flatten(), y_grid.flatten()), z_grid2.flatten(), (x_grid_highres, y_grid_highres), method='cubic')
    
        # Create the heatmap
        plt.figure(figsize=(12,10))
        plt.pcolormesh(x_grid_highres, y_grid_highres, z_grid_highres, cmap="viridis",vmin=0,vmax=60)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        if "ctitle" in kwargs:
            plt.colorbar(label=kwargs["ctitle"])
        else:
            plt.colorbar(label="Mean success")
        if "xlabel" in kwargs or "ylabel" in kwargs:
            plt.xlabel(kwargs["xlabel"],fontsize=40)
            plt.ylabel(kwargs["ylabel"],fontsize=40)
        if "title" in kwargs:
            plt.title(kwargs["title"],fontsize=70)
        plt.show()

    fill_array = calc_output(results,space,"mean_success",fname)
    create_mesh(fill_array,xlabel=xlabel,ylabel=ylabel,ctitle="Mean success")
    return results


#%% Phylogram analysis
def equalise(lst):
    lst2 = lst.copy()
    maxlen = max([len(i) for i in lst2])
    for i in lst2:
        diff = maxlen - len(i)
        for j in range(diff):
            i.append(0)
    return lst2
def phylo_analysis(results,index,title,thresh=100):
    large_set = results["variables"]["CCE_model"]["tokens"].loc[index]
    def listicate(results):
      large = []
      for i in large_set:
          for j in i:
              large.append(j)
      large2 = []
      for i in large:
          if isinstance(i,(float,int)):
              i = [i]
          large2.append(model.flatten_list(i))
      return large2

    # Calculate the distance between the lists
    def get_unique(lst,counts=True):
        c = []
        n=0
        indices = []
        if counts:
            count = []
            counta = model.count_nested(lst)
        #lst2 = np.array(lst,dtype="object")
        for i, tokens in enumerate(lst):
    
            if tokens not in lst[i+1:]:
               n += 1
               c.append(tokens)
               indices.append(i)
               if counts:
                   count.append(counta[i])
        return c,count
    
    
    large2 = listicate(large_set)
    large3 = equalise(large2)
    large4 = get_unique(large3)[0]
    distances = pdist(large4)
    
    # Compute hierarchical clustering
    large4,counta = get_unique(large3,large3)
    linkage_matrix = linkage(distances, "ward")

    # Plot the dendrogram
    
    plt.figure(figsize=(15,20))
    dendrogram(linkage_matrix,color_threshold=1000,orientation='left',labels=counta)
    plt.xlabel('Lists')
    
    plt.ylabel('Tokens',fontsize=30)
    plt.xlabel("Distance",fontsize=30)
    plt.xticks(fontsize=30)
    #plt.title("Dendrogram of token sequence similarity",fontsize=50)
    plt.savefig(title + ".svg")
    plt.yticks(fontsize=15)
    clusters = fcluster(linkage_matrix, t=thresh,criterion='distance')
    output = np.concatenate([np.reshape(clusters,[len(clusters),1]),large4],axis=1)
    return output
    
#%% 1D parameter search
def do_1D(param1,range1, resolution,n_reps,flo=True,**kwargs):
    def param_search_1D(param1,range1, resolution,n_reps,flo):
        parameters = {'population': 100, 'steps': 70, 'period': 1,'lim':300,'radius':60, 'buffer': 7, 'p_innov': 1/10000, 'p_overimitate':1,'learn_thresh':0.999999999,"scale_death":1.1,"scale":5,"cprob":0.98} # Default
        if not flo:
            parameters[param1] = model.ap.IntRange(range1[0],range1[1])
        else:
            parameters[param1] = model.ap.Range(range1[0],range1[1])
        if "steps" in kwargs:
            parameters["steps"] = kwargs["steps"]
            
        if "lim" in kwargs:
            parameters["lim"] = kwargs["lim"]
        results = model.run_exp(parameters,resolution,n_reps)
        fname = param1 + "_" + str(range1[0]) + "-" + str(range1[1]) + 'w_' + "-" + "_rep" + str(n_reps)
        return results,fname
    results,fname = param_search_1D(param1,range1,resolution,n_reps,flo)
    return results

def plot_1D(results,variable,length):
    for i in range(length):
        plt.plot(results["variables"]["CCE_model"][variable])
#%% Curve fitting
from matplotlib.colors import Normalize

def get_depths(tokens,length,do_max=True):
    trajectory = []
    for k in range(length+1):
      a = []
      for l in tokens.iloc[k]:
        a.append(model.depth(l))
      if do_max:  
          trajectory.append(np.max(a))
      else:
          trajectory.append(np.mean(a))
    return trajectory

def get_depths2(tokens,length):
    trajectory = []
    for k in range(length+1):
      a = []
      for l in tokens.iloc[k]:
          for m in l:
              a.append(model.depth(l))
        
      trajectory.append(np.mean(a))
    return trajectory

def get_lengths(tokens,length):
    trajectory = []
    for k in range(length+1):
      a = []
      for l in tokens.iloc[k]:
         a.append(len(l))
        
      trajectory.append(np.mean(a))
    return trajectory

def get_burdens(burdens,length):
    import ast
    trajectory = []
    for k in range(length+1):
        trajectory.append(np.mean(list(burdens[k])))
    return trajectory
def do_curve_fitting(results,length,n_reps,var="mean_success",depth=False,lens=False,clabel="Working memory buffer",ylabel="Mean success",do_max=True):
    def fit_curve(data,length):
      x = np.concatenate([np.arange(0,length,1)])
      y = np.concatenate([data])
    
      def hyperbolic_function(x, a, b):
          return a*x / (b+ x)
      params, covariance = curve_fit(hyperbolic_function, x, y, maxfev=10000)
      a_fit, b_fit = params
      x_fit = np.linspace(0, len(x), 1000)
      y_fit = hyperbolic_function(x_fit, a_fit, b_fit)
      return x_fit, y_fit
    # Plot the fitted logistic curve
    
    def get_series(results,length,n_reps,clabel="Working memory buffer",var1="mean_success",ylabel="Mean success",depth=False,lens=False,do_max=True):
        var = results["variables"]["CCE_model"][var1]
        ids = results["parameters"]["sample"]
        var_tok = results["variables"]["CCE_model"]["tokens"]
    
        means = {}
        colors = plt.cm.viridis(np.linspace(0, 1, len(ids)))
        norm = Normalize(vmin=0,vmax=20)
        figure = plt.figure(figsize=(12,12))
        for i in range(len(ids)):
        
          means[str(i)] = np.zeros([length+1,n_reps])
    
          for j in range(n_reps):
            ## burden
            #data_mean = [np.mean(list(lst), axis=0) for lst in var.iloc[var.index.get_level_values('iteration') == j].loc[i]]
            #means[str(i)][:,j] = data_mean
            ## success
            if not depth and not lens:
                if var1 == "burden":
                    data_mean = get_burdens(var[i][j],length)
                    means[str(i)][:,j] = data_mean
                else:
                    means[str(i)][:,j] = list(var.iloc[var.index.get_level_values('iteration') == j].loc[i])
            elif depth:
            ## token depth
                depths = get_depths(var_tok.iloc[var.index.get_level_values("iteration") == j].loc[i],length,do_max=do_max) 

                means[str(i)][:,j] = depths
            elif lens:
                lengths = get_lengths(var_tok.iloc[var.index.get_level_values("iteration") == j].loc[i],length) 
                means[str(i)][:,j] = lengths
                
          means[str(i)] = np.insert(means[str(i)],0,np.mean(means[str(i)],axis=1),axis=1)
          
          plt.plot(means[str(i)][:,0],c=colors[i],linewidth=2)
          #if i != 0:
          if depth:
            if i!=0:
                x_fit, y_fit = fit_curve(means[str(i)][:,0],length+1)
                plt.plot(x_fit,y_fit,c=colors[i],linewidth=5,alpha=1)
    
            for j in range(n_reps):
                plt.plot(means[str(i)][:,j],c=colors[i])
            ## average across
        sm = plt.cm.ScalarMappable(cmap="viridis",norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, label=clabel,ticks=np.arange(0,21,2))
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(clabel, fontsize=20)
        plt.xlabel("Iteration",fontsize=30)
        plt.ylabel(ylabel,fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        return means
    
    means = get_series(results,length,n_reps,depth=depth,var1=var,ylabel=ylabel,clabel=clabel,lens=lens,do_max=do_max)
    return means

#%% Network analysis
def visualise_network(results,index):
    def animation_plot(m,it,axs):
        import matplotlib.colors as mc
        from matplotlib.colors import Normalize
        # Plot network on second axis
        import ast
    
        cmap = plt.cm.viridis
        norm = mc.Normalize(vmin=1.5, vmax=4.5)
    
        #color_dict = {0:'b', 1:'r', 2:'g'}
        #colors = [color_dict[c] for c in m.agents.success]
        #G = m.network.graph
        #unconnected_nodes = [node for node in G.nodes() if not any(G.has_edge(node, neighbor) for neighbor in G.nodes())]
        #G.remove_nodes_from(unconnected_nodes)
    
    
        G = m["variables"]["CCE_model"]["networks"].iloc[it]
        G.remove_edges_from([(source, target) for source, target in G.edges() if source == target])
        age = m["variables"]["CCE_model"]["ages"].iloc[it]
        pos = m["variables"]["CCE_model"]["positions"].iloc[it]
        success = np.multiply(m["variables"]["CCE_model"]["successes"].iloc[it],2)
        print(success)
        print(pos,"**")
        widths = nx.get_edge_attributes(G, 'weight')
        widths2 = {edge: weight*7 for edge, weight in widths.items()}
        print(widths2)
        for edge, width in widths.items():
          G[edge[0]][edge[1]]['weight'] = width
    
        nx.draw_networkx(G,node_color=age,cmap=cmap,
                         node_size=success+10 ,with_labels=False,ax=axs,pos=pos,width=list(widths2.values()))
        norm = Normalize(vmin=0,vmax=80)
        sm = plt.cm.ScalarMappable(cmap="viridis",norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, label="Age",ticks=np.arange(0,80,10))
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label("Age", fontsize=20)
    #animation_plot(model,ax)
    
    fig, ax = plt.subplots(figsize=(8,6))
    animation_plot(results,index,ax)
    

#%% Similarity analysis

def find_intersection_elements(lst1, lst2):
    def nested_list_to_set(lst):
        if isinstance(lst, list):
            return tuple(nested_list_to_set(x) for x in lst)
        else:
            return lst

    if isinstance(lst1, (int, float)):
        lst1a = [lst1]
    else:
        lst1a = lst1

    set1 = {nested_list_to_set(x) for x in lst1a}
    set2 = {nested_list_to_set(x) for x in lst2}

    intersection_elements = set1.intersection(set2)

    def set_to_nested_list(s):
        if isinstance(s, tuple):
            return [set_to_nested_list(x) for x in s]
        else:
            return s

    result = [set_to_nested_list(x) for x in intersection_elements]
    return result

def find_union_elements(lst1, lst2):
    def nested_list_to_set(lst):
        if isinstance(lst, list):
            return tuple(nested_list_to_set(x) for x in lst)
        else:
            return lst

    if isinstance(lst1, (int, float)):
        lst1a = [lst1]
    else:
        lst1a = lst1

    set1 = {nested_list_to_set(x) for x in lst1a}
    set2 = {nested_list_to_set(x) for x in lst2}

    union_elements = set1.union(set2)

    def set_to_nested_list(s):
        if isinstance(s, tuple):
            return [set_to_nested_list(x) for x in s]
        else:
            return s

    result = [set_to_nested_list(x) for x in union_elements]
    return result

def jaccard_similarity(lst1,lst2):
  intersect = find_intersection_elements(lst1,lst2)
  union = find_union_elements(lst1,lst2)
  J = float(len(intersect))/float(len(union))
  return J

def quant_similarity(result,it,lim,sample,comparison="flatten"):
  tokens = result["variables"]["CCE_model"]["tokens"].iloc[it]

  positions = list(result["variables"]["CCE_model"]["positions"].iloc[it].values())
  G = result["variables"]["CCE_model"]["networks"].iloc[it]
  quadrants = [[[0,lim/2],[0,lim/2]],[[lim/2,lim],[0,lim/2]],[[0,lim/2],[lim/2,lim]],[[lim/2,lim],[lim/2,lim]]]
  corr_matrix = np.zeros([1,2])
  for val,i in enumerate(quadrants):
    node_sel = []
    token_sel = []
    pos_sel = []
    t = 0
    for j in range(len(G.nodes)):

      if positions[j][0] in range(int(i[0][0]),int(i[0][1])) and positions[j][1] in range(int(i[1][0]),int(i[1][1])) and t != sample:
        if len(tokens[j]) != 0:
          node_sel.append(list(G.nodes())[j])
          token_sel.append(tokens[j])
          pos_sel.append(positions[j])
          t += 1

    for k,token in enumerate(token_sel):

      values = []
      for index,l in enumerate(tokens):
        if comparison == "flatten":
              sim = jaccard_similarity(model.flatten_list(token),model.flatten_list(l))
        elif comparison == "round":
              sim = jaccard_similarity(model.round_list(token),model.round_list(l))
        values.append(sim)

        distance = math.dist(tuple(pos_sel[k]),tuple(positions[index]))
        if sim > 0.2:
          corr_matrix = np.append(corr_matrix,np.reshape([distance,sim],[1,2]),axis=0)
      for node, value in zip(G.nodes(), values):
        nx.set_node_attributes(G, {node: {str(val)+str(k): value}})

      corr_matrix = corr_matrix[1:,:]
  return corr_matrix

############### Plot similarity-distance ###################

from scipy.optimize import curve_fit
def fit_curve_distance(corrs,length):
    def fit_curve(data,length):
      x = data[:,0]
      y = data[:,1] - np.min(data[:,1])
    
      def exponential(x, a,b,c):
          return c**(a-b*x)
      
      params, covariance = curve_fit(exponential, x, y, maxfev=20000)
      a_fit,b_fit,c_fit = params
      x_fit = np.linspace(0, np.max(x), 1000)
      y_fit = exponential(x_fit, a_fit,b_fit,c_fit)
      return x_fit, y_fit + np.min(data[:,1])
    
    x,y = fit_curve(corrs,length)
    return x,y

def plt_dist_sim(corr,color,x,y,title,add=False):
    if not add:
        plt.figure(figsize=(12,12))
    if x != "" and y != "":
        plt.plot(x,y,linewidth=5,color=color)
    plt.scatter(corr[:,0],corr[:,1],s=200,color=color)
    
    plt.xlabel("Cartesian distance",fontsize=30)
    plt.ylabel("Jaccard similarity",fontsize=30)
    plt.legend(title=title,fontsize=20,title_fontsize=20)
    plt.xticks(fontsize=20,rotation=90)
    plt.yticks(fontsize=30)

#################### Plot similarity network ################

def sim_plot(m,it,lim=300,sample=5,agent="00",comparison="flatten"):
    corr = quant_similarity(m,it,lim,sample,comparison=comparison)
    import matplotlib.colors as mc
    from matplotlib.colors import Normalize
    # Plot network on second axis

    cmap = plt.cm.viridis
    norm = mc.Normalize(vmin=1.5, vmax=4.5)

    #color_dict = {0:'b', 1:'r', 2:'g'}
    #colors = [color_dict[c] for c in m.agents.success]
    #G = m.network.graph
    #unconnected_nodes = [node for node in G.nodes() if not any(G.has_edge(node, neighbor) for neighbor in G.nodes())]
    #G.remove_nodes_from(unconnected_nodes)


    G = m["variables"]["CCE_model"]["networks"].iloc[it]
    G.remove_edges_from([(source, target) for source, target in G.edges() if source == target])
    pos = m["variables"]["CCE_model"]["positions"].iloc[it]
    success = np.multiply(m["variables"]["CCE_model"]["successes"].iloc[it],2)

    widths = nx.get_edge_attributes(G, 'weight')
    widths2 = {edge: weight*7 for edge, weight in widths.items()}

    for edge, width in widths.items():
      G[edge[0]][edge[1]]['weight'] = width
    
    fig, ax = plt.subplots(figsize=(12,12))

    nx.draw_networkx(G,node_color=list(nx.get_node_attributes(G,agent).values()),cmap=cmap,
                     node_size=((success))+10 ,with_labels=False,ax=ax,pos=pos,width=list(widths2.values()))
    norm = Normalize(vmin=0,vmax=1)
    sm = plt.cm.ScalarMappable(cmap="viridis",norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, label="Jaccard similarity",ticks=np.arange(0,1.1,0.2))
    return corr
#%% Token visualisation

def plot_tokens(results,index,sample,title):
    def leaf_distances(lst):
        def is_leaf(elem):
            return not isinstance(elem, list)
    
        def get_leaves(node):
            leaves = []
            if is_leaf(node):
                return [node]
            for child in node:
                leaves.extend(get_leaves(child))
            return leaves
    
        def calculate_distance(leaf1, leaf2):
            return abs(float(leaf1) - float(leaf2))
    
        leaves = get_leaves(lst)
        num_leaves = len(leaves)
    
        # Initialize a matrix to store distances between leaves
        distance_matrix = [[0.0] * num_leaves for _ in range(num_leaves)]
    
        # Populate the distance matrix
        for i in range(num_leaves):
            for j in range(i + 1, num_leaves):
                distance_matrix[i][j] = calculate_distance(leaves[i], leaves[j])
    
        return distance_matrix, leaves
    
    token = results["variables"]["CCE_model"]["tokens"][index][sample]
    distances, leaves = leaf_distances(token)
    return distances
    # Perform hierarchical clustering
    linkage_matrix = linkage(distances)
    plt.figure(figsize=(16,12))
    # Plot the dendrogram
    dendrogram(linkage_matrix, labels=leaves)
    #plt.title('Dendrogram')
    plt.xlabel("Unit tokens",fontsize=30)
    plt.ylabel("Distance",fontsize=30)
    plt.xticks(fontsize=20,rotation=90)
    plt.yticks(fontsize=30)
    plt.savefig(title + ".svg")
    plt.show()

#%% Plot any two variables

def plot_vars(results,var1,var2,xlabel="",ylabel="",exp=False,color="black",add=False,label=None,**kwargs):
    if add != True:
        plt.figure(figsize=(12,12))

    if exp:
        output1 = results[var1] ### specify experiment in input
        if var2 != "":
            output2 = results[var2]
    else:
        print("Hi")
        output1 = results["variables"]["CCE_model"][var1]
        if var2 != "":
            output2 = results["variables"]["CCE_model"][var2]
        
    if "index" in kwargs:
        if var2 != "":
            plt.scatter(list(output1.iloc[kwargs["index"]]),list(output2.iloc[kwargs["index"]]),s=140,color=color,label=label)
        else:
            plt.plot(list(output1.loc[kwargs["index"]]),linewidth=5,color=color,label=label)            
    else:
        if var2 != "":
            for i in range(len(output1)):
                plt.scatter(list(output1.loc[i]),list(output2.loc[i]),color=color,s=140,label=label)

        else:
            plt.plot(output1,color=color,linewidth=5,label=label)
    plt.xlabel(xlabel=xlabel,fontsize=30)
    plt.ylabel(ylabel=ylabel,fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    
def fit_curve_2D(x,y,length):
  def hyperbolic_function(x, a, b):
      return a*x / (b+ x)
  params, covariance = curve_fit(hyperbolic_function, x, y, maxfev=10000)
  a_fit, b_fit = params
  x_fit = np.linspace(0, len(x), 1000)
  y_fit = hyperbolic_function(x_fit, a_fit, b_fit)
  return x_fit, y_fit 
