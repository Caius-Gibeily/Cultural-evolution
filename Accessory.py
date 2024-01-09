#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 19:57:09 2023
"""
import numpy as np
import networkx as nx
import pandas as pd
import os,glob
import ast

def save_networks(results,n_samples,n_reps,length,path,fname):
    for i in range(n_samples):
        for j in range(n_reps):
            for k in range(length+1): 
                G = results["variables"]["CCE_model"]["networks"][i][j][k]
                G2 = nx.convert_node_labels_to_integers(G,first_label=0)
                nx.write_gml(G2,path + fname +"-" + str(i) + "_" + str(j) + "_" + str(k) + ".gml")
            



def return_literal(results,n_samples,n_reps,length,var,return_graph=True,path=os.getcwd()):
    if return_graph:
        os.chdir(path)
        files = glob.glob("*.gml")
    it = 0
    for i in range(n_samples):
        for j in range(n_reps):
            for k in range(length+1):                    
                for l,v in enumerate(var):
                    results["variables"]["CCE_model"][v][i][j][k] = ast.literal_eval(results["variables"]["CCE_model"][v][i][j][k])
                if return_graph:
                    H = nx.read_gml(files[it])
                    results["variables"]["CCE_model"]["networks"][i][j][k] = H       
                    results["variables"]["CCE_model"]["positions"][i][j][k] = nx.get_node_attributes(H,"pos")    
                    print(i,j,k)
                it += 1
        print(i)
    os.chdir("..")
    return results

def get_single_exp(exp,sample,it="all"):
    """
    Parameters
    ----------
    exp : Experiment (1D)
    sample : parameter value.
    it : time index

    Returns
    -------
    results : single result

    """
    results = {}
    results["variables"] = {}
    if it != "all":
        results["variables"]["CCE_model"] = exp["variables"]["CCE_model"].loc[sample].loc[it]
    else:        
        results["variables"]["CCE_model"] = exp["variables"]["CCE_model"].loc[sample]
    return results

def replace_trial(results,length,trial):
    results2 = results.copy()
    variables = ['ages', 'positions', 'successes', 'networks', 'mean_success', 'tokens',
           'burden', 'overim', 'inductive', 'homo']
    for i,var in enumerate(variables):
        for j in range(length):
            print(trial["variables"]["CCE_model"][var].iloc[j])
            print(results2[var].iloc[j])
            results2[var].iloc[j] = list(trial["variables"]["CCE_model"][var].iloc[j])
    return results2
#################### Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def get_colors(n_samples):
    colors = plt.cm.viridis(np.linspace(0, 1, n_samples))
    return colors 

def add_xy_color(xlabel,ylabel,clabel,n_samples,sample_range=""):
    if sample_range=="":
        sample_range = [0,n_samples,1]
        norm = Normalize(vmin=0,vmax=20)
    else:
        norm = Normalize(vmin=sample_range[0],vmax=sample_range[1])
    plt.xlabel(xlabel,fontsize=30)
    plt.ylabel(ylabel,fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    sm = plt.cm.ScalarMappable(cmap="viridis",norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, label=clabel,ticks=np.arange(sample_range[0],sample_range[1],sample_range[2]))
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(clabel, fontsize=20)
