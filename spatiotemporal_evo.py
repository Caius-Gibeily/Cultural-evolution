#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:45:46 2023

@author: caiusgibeily
"""

import numpy as np
import seaborn as sns
def get_IDs(lst):
    db = []
    ids = [row.copy() for row in lst]
    count = []
    for i, row in enumerate(lst):
        for j, token in enumerate(row):
            found = False
            for k, entry in enumerate(db):
                if np.array_equal(entry, token):
                    ids[i][j] = k + 1
                    count[k] += 1
                    found = True
                    break
            
            if not found:
                db.append(token)
                count.append(1)
                ids[i][j] = len(db)

        a = np.array(count); a[a < 0] = 0
        print(i)
    return ids,count,db


def to_list(plst,rounded=False):
    olst = []

    for i in plst:
        if rounded:
            a = []
            for j in i:
                a.append(an.model.round_list(j))
            olst.append(a)
        else:
            olst.append(list(i))
    return olst
# Example usage:
import copy

def flatten_list(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list

def plot_continuity(lst,count,results,it,rep,get_ranges=True):
  ids = max(np.ndarray.flatten(np.array(lst)))
  plt_array = np.zeros([ids,len(lst)])
  for i,item in enumerate(lst):
    for j in item:
      plt_array[j-1,i] = j
  plt_array[plt_array==0] = np.nan
  perc = int(np.percentile(count,99))
  colors = ac.get_colors(perc)
  fig, ax1 = plt.subplots(figsize=(12,12))
  ax1.set_xlabel("Iteration",fontsize=0) 
  ax1.set_ylabel("Unique token ID", color = "black",fontsize=30) 
  ax1.tick_params(axis ='y', labelcolor = "black") 
   
  for i,item in enumerate(plt_array):
      if count[i] >= perc:
          c = colors[-1]
      else: c = colors[count[i]]
      ax1.plot(item,color=c)
      
  ac.add_xy_color("Iterations", "Unique token ID", "Count", perc,sample_range=[0,49,8])
  ax2 = ax1.twinx() 
  ax2.plot(results["variables"]["CCE_model"]["mean_success"][it][rep], linewidth=5,color = "black") 
  ax2.tick_params(axis ='y', labelcolor = "black",labelsize=15)
  

  ax2.set_ylim(0,1000)
  
  plt_array[np.isnan(plt_array)] = 0
  if get_ranges:
    ranges = np.zeros([len(plt_array),2])
    for i,row in enumerate(plt_array):
        prominence = np.where(row != 0)[0]
        
        ind_min = np.min(prominence); ind_max = np.max(prominence)
        ranges[i] = [ind_min,ind_max]
    ranges = np.append(np.reshape(count,[len(count),1]),ranges,axis=1)
    return plt_array,ranges
  return plt_array


############# Objective 1: Temporal - determine the longevity of individual tokens and return a count + index ranges
olst = to_list(rad0_100_p100["variables"]["CCE_model"]["tokens"],rounded=False)
slst = copy.deepcopy(olst)
s8,count_s8,atlas_s8 = get_IDs(slst)
rad3_arr,ranges3 = plot_continuity(s8,count_s8,saeculum_8)

############ Determine the spatial locations of agents carrying the token at all timepoints within its prominence
def mini_listicate(array):
    large = []
    for i in array:
      if isinstance(i,(float,int)):
          i = [i]
      large.append(an.model.flatten_list(i))
    return large

def equalise(lst):
    lst2 = lst.copy()
    maxlen = max([len(i) for i in lst2])
    for i in lst2:
        diff = maxlen - len(i)
        for j in range(diff):
            i.append(0)
    return lst2


def get_coords(results,it,time,rep):
    large = np.zeros([1,3])

    for j,iteration in enumerate(results["variables"]["CCE_model"]["tokens"][it][rep]):
        if j in time:
            positions = list(results["variables"]["CCE_model"]["positions"][it][rep][j].values())
            for k, agent in enumerate(iteration):
                x,y = positions[k]
                for l,num in enumerate(agent):
                    if isinstance(num,(float,int)):
                        tok_data = [x,y,num]
                    else:
                        tok_data = [x,y,an.model.flatten_list(num)]
                    large = np.append(large,np.reshape(tok_data,[1,3]),axis=0)

        large = large[1:,:]
    return large

def get_specific_coords(results,it,time,rep,token):
    large = np.zeros([1,3])
    in_token = token[token!=0]
    for j,iteration in enumerate(results["variables"]["CCE_model"]["tokens"][it][rep]):
        if j in time:
            positions = list(results["variables"]["CCE_model"]["positions"][it][rep][j].values())
            for k, agent in enumerate(iteration):
                x,y = positions[k]
                for l,num in enumerate(agent):
                    if len(flatten_list([num])) == len(in_token):
                        if np.equal(in_token,flatten_list([num]),where=True).all():
                         tok_data = [x,y,an.model.flatten_list([num])]
                         large = np.append(large,np.reshape(tok_data,[1,3]),axis=0)
    large = large[1:,:]
    return large

def match_clusters(clusters,token_coords):
    clust_arr = np.zeros([len(token_coords),1])
    for i,item in enumerate(token_coords):
        bool_mask = np.equal(clusters[:,1:],item[2:],where=True).all(1)

        clust_arr[i] = clusters[bool_mask][0][0]
    coords = copy.deepcopy(token_coords)
    coords = np.append(clust_arr,coords,axis=1).astype("float")
    return coords

def get_coords_tokens(results,it,time,rep):
    token_coords = get_coords(results,it,time,rep)
    tokens = mini_listicate(token_coords[:,2]); tokens = np.array(equalise(tokens))
    token_coords2 = token_coords[:,:-1]
    token_coords2 = np.append(token_coords2,tokens,axis=1)
    
    
    phylo = ac.get_single_exp(results,it,it=rep)
    clust = an.phylo_analysis(phylo,time[0],thresh=200,title="")
    coords_tokens = match_clusters(clust,token_coords2)

    return coords_tokens

def select_prominent(ranges,id_arr,atlas,results,it,rep,mode="prominent"):
    """

    Parameters
    ----------
    ranges : TYPE
        DESCRIPTION.
    token_coords2 : TYPE
        DESCRIPTION.

    Returns
    -------
    An array containing the most prominent tokens and their spatial location
    """
    fatlas = np.array(equalise(mini_listicate(atlas)))


    if mode == "prominent":
        perc = np.percentile(ranges[:,0],90)
        prominents = ranges[ranges[:,0]>=perc]
        ids = id_arr[ranges[:,0]>=perc]
        
    elif mode == "depths":
        depths = []
        for i in range(len(atlas)):
            depths.append(an.model.depth(atlas[i]))
        perc = np.percentile(depths,70)
        prominents = ranges[depths>=perc]
        ids = id_arr[depths>=perc]

    tokens = np.zeros([len(prominents),fatlas.shape[1]+2])
    coords_tokens = np.zeros([1,5])
    print(ids.shape)
    for i,token in enumerate(ids):
        if len(np.unique(token)) == 1:
            index = int(np.unique(token)[0]) - 1
        else:
            index = np.unique(token)[1]
            index = int(index) - 1
        tokens[i,2:] = fatlas[index]
        tokens[i,:2] = ranges[index,1:]
        
        for j in range(int(ranges[index,1]),int(ranges[index,2])):
            all_instances = get_specific_coords(results, it, [j], rep, fatlas[index])
            ids = np.reshape(np.repeat(index,len(all_instances)),[len(all_instances),1])
            time = np.reshape(np.repeat(j-ranges[index,1],len(all_instances)),[len(all_instances),1])
            all_instances = np.append(ids,all_instances,axis=1)
            all_instances = np.append(time,all_instances,axis=1)  
            coords_tokens = np.append(coords_tokens,all_instances,axis=0)
    tokens = tokens[1:,:]
    coords_tokens = coords_tokens[1:,:]
    return coords_tokens

a = select_prominent(ranges3, rad3_arr, atlas3, rad0_100_p100, 3, 0,mode="prominent")

for i in range(0,126):
    plot = a[a[:,0]==float(i)]
    plot_heatmap(plot[plot[:,1]==293],size=10,mode="a")
    
    
import matplotlib.animation as animation
def plot_heatmap(coords,size,sel=1,mode="taxa"):
    if mode == "taxa":
        coords_sel = coords[coords[:,0] == sel]
        # Create a grid for the heatmap
    else:
        coords_sel = coords
    grid_size = size # Adjust this based on your preference
    x_bins = np.linspace(0, 300, grid_size)
    y_bins = np.linspace(0, 300, grid_size)

    # Create a 2D histogram to represent the spatial distribution
    if mode == "taxa":
        heatmap, xedges, yedges = np.histogram2d(coords_sel[:,1], coords_sel[:,2], bins=[x_bins, y_bins])
    elif mode == "temporal":
        heatmap, xedges, yedges = np.histogram2d(coords_sel[:,2], coords_sel[:,3], bins=[x_bins, y_bins])

    # Transpose the heatmap to match the orientation of the plot
    heatmap = heatmap.T

    # Create a DataFrame from the heatmap
    heatmap_df = pd.DataFrame(heatmap,index=np.round(xedges[1:],0),columns=np.round(yedges[1:],0))

    # Plot the hotspot heatmap for target sequences
    

    a = sns.heatmap(heatmap_df,cmap="viridis",cbar=False)
    plt.ylabel("Binned y coordinates",fontsize=30)
    plt.xlabel("Binned x coordinates",fontsize=30)
    plt.xticks(rotation=90)
    plt.gca().invert_yaxis()
    

# Create an animation
fig = plt.figure(figsize=(10, 8))
ac.add_xy_color("X coordinate", "Y coordinate", "Density", 11,[0,3.1,0.5])

plt.xticks(fontsize=30)
ani = animation.FuncAnimation(fig, animate, frames=16)

# Save the animation as an AVI file
fig = plt.figure(figsize=(10, 8))
i = 0
j = 5
plot = a[a[:, 0] == i]

heat = plot_heatmap(plot[plot[:, 1] == j], size=10, mode="a")
plt.yticks(rotation=0)
ac.add_xy_color("X coordinate", "Y coordinate", "Density", 11,[0,3.1,0.5])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


def output_movies(prominents,mdir):
    for j in np.unique(prominents[:,1]):
        token = prominents[prominents[:,1]==j]
        duration = len(np.unique(token[:,0]))
        
        def animate(i):  # Clear the current figure
            
            plot = a[a[:, 0] == i]
            heat = plot_heatmap(plot[plot[:, 1] == j], size=10, mode="a")
            title = "Iteration " + str(i) 
            plt.title(title,fontsize=30)
        
        fig = plt.figure(figsize=(10, 8))
        ac.add_xy_color("X coordinate", "Y coordinate", "Density", 11,[0,3.1,0.5])a
        plt.text(1,1,str(j))
        plt.xticks(fontsize=30)
        print(duration)
        plt.clf()
        ani = animation.FuncAnimation(fig, animate, frames=duration)
        
        ani.save(mdir + "token_ID-" + str(j) + ".avi", writer='ffmpeg', fps=2)
        plt.show()

output_movies(a,os.getcwd() + "/films/")
    




for i in np.unique(coords_tokens[:,0]):
    plot_heatmap(coords_tokens, 10,mode="taxa",sel=i)
    ac.add_xy_color("X coordinate", "Y coordinate", "Density", 11,[0,3.1,0.5])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.show()
    
rad_b =  ac.get_single_exp(rad0_100_p100,4,it=1)
an.visualise_network(saeculum_8,100)
zac.add_xy_color("","","Age",8,[0,71,10])
a = an.phylo_analysis(rad_b,10,thresh=1000,title="rad4b_10")



def list_to_empty_nested_tuple(nested_list):
    if isinstance(nested_list, list):
        return tuple(list_to_empty_nested_tuple(item) for item in nested_list)
    else:
        return ()


def plot_token(results,it,iteration,agent,token=0,rep=0):
    plt.figure(figsize=(14,10))
    nested_tuple = list_to_empty_nested_tuple(results["variables"]["CCE_model"]["tokens"][it][rep][iteration][agent][token])
    T = nx.from_nested_tuple(nested_tuple,sensible_relabeling=True)
    T = nx.convert_node_labels_to_integers(T)
    pos = graphviz_layout(T,prog="dot")
    pos = {int(k):v for k,v in pos.items()}
    nx.draw_networkx(T,pos=pos,with_labels=False,node_size=200,width=8,node_color="grey")
    plt.gca().invert_yaxis()
        
    plt.show()

plot_token(rad0_100_p100,4,10,50,0,rep=1)
plot_token(rad0_100_p100,4,50,80,2,rep=1)
plot_token(rad0_100_p100,4,100,57,0,rep=1)
plot_token(rad0_100_p100,4,100,58,0,rep=1)



for i in range(100):    
        
        for j,row in enumerate(clusters[:,3:]):
            if len(rad0_100_p100["variables"]["CCE_model"]["tokens"][4][1][149][i]) != 0:
                flattened = flatten_list(rad0_100_p100["variables"]["CCE_model"]["tokens"][4][1][149][i][0])
    
                if len(row[row!=0]) ==  len(flattened):
    
                    if np.equal(row[row!=0],flattened).all():
                        
                        plt.text(1,1,str(coords_tokens[j,0]))
        if len(rad0_100_p100["variables"]["CCE_model"]["tokens"][4][1][149][i]) != 0:
            plot_token(rad0_100_p100,4,149,i)
            clusters = coords_tokens
            plt.show()

print(nested_tuple)

token_vars = np.zeros([1,coord_tokens[2:,:].shape[1]])



coords_tokens = get_coords_tokens(rad0_100_p100,4,[60],1)
for i in coords_tokens[:,2:]:    
    comp = np.zeros([1,len(i)])
    count = 0
    for j in coords_tokens[:,2:]:
                if np.equal(an.model.round_list(list(i)),an.model.round_list(list(j))).all():
                    comp = np.append(comp,np.reshape(an.model.round_list(j),[1,len(j)]),axis=0)
                    count +=1
                    print(count)
    comp = comp[1:,:]
    var = np.subtract(comp,an.model.round_list(list(i)))
    plt.plot(var[var!=0].T)
    plt.show()
    
        if len(rad0_100_p100["variables"]["CCE_model"]["tokens"][4][1][149][i]) != 0:
            plot_token(rad0_100_p100,4,149,i)
            clusters = coords_tokens
            plt.show()






