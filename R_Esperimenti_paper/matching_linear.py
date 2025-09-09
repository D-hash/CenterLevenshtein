# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 10:49:00 2025

@author: oya
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import networkx as nx
import itertools
import time

# Define the problem parameters
#n = 10  # Number of positions (adjust based on the actual model)
#m = 10  # Number of input strings (adjust as needed)

# Generate random binary sigma vectors (m x n)
#np.random.seed(42)  # Set seed for reproducibility
#sigma = np.random.randint(0, 2, size=(m, n))  # Random 0-1 vectors


#num Instances
I = 5

#Set result file
with open("result_matching_linear.txt", "a") as file:
    file.write(f"Instance,Seed,Best Incumbent,BestBound,SolutionTime,GAP,Nodes,SimplexIterations,Surrogate,All_cuts,only_root_frac_cuts,use_pool,#cuts,#clique_cuts,#no_del_add_cuts,#no_crossing,#continuity_cuts,#heur_cliques,#lin_incomp_cuts,\n")

print("### STARTING MATCHING LINEAR ###")


for stringlength in range(10,60,10):
    for stringnumber in range(10,110,10):
        for it in range(I):
            for seed in [2025, 111, 923821]:

                # Initialize variables
                n, m, sigma = None, None, None
                
                # Read the file
                with open(f"I_{stringlength}_{stringnumber}_{it}.txt", "r") as f:
                    lines = f.readlines()
                
                # Extract n and m
                n = int(lines[0].split("=")[1].strip())
                m = int(lines[1].split("=")[1].strip())
                
                # Extract sigma (from line index 3 onwards)
                sigma = np.array([list(map(int, line.split())) for line in lines[3:]])
                
                # Print to verify
                print(f"n = {n}")
                print(f"m = {m}")
                print("sigma =\n", sigma)

                
                median = (np.sum(sigma, axis=0) >= sigma.shape[0] / 2).astype(int)
                print("Heuristic Median String:", median.tolist())
                total_distance = np.sum(np.abs(sigma - median))
                print("Total Hamming Distance:", total_distance)
                optimal_x = {j: int(median[j]) for j in range(n)}
                print("Optimal Median String:", optimal_x)

                
                # Create a new Gurobi model
                model = gp.Model("Branch-and-Cut")

                #OPTIONS
                only_root_frac_cuts = 1 #1 to add fractional cuts only at root node
                all_cuts = 1 #1 for adding all violated inequalities; 0 for the most violated one
                surrogate = 1 #1 surrogate of no-crossing constraints; 0 full formulation
                heur_cliques = 0 #1 for heuristically separate (43); 0 no (43) cuts
                use_pool = 0 #1 all cuts are computed beforehand (slow down a lot); 0 cut visited on the fly 
                model.Params.LazyConstraints = 1
                model.setParam(GRB.Param.TimeLimit, 600)
                #model.Params.PreCrush = 1

                if(heur_cliques):
                    #Create graph for clique computation
                    G = nx.Graph()
                    for i in range(n):
                        for j in range(n):
                            if(j >= i + math.ceil(n/2) or i >= j + math.ceil(n/2) # reduce the size by max number of matching edges
                            ):
                                continue
                            else:
                                G.add_edge(i, j) 
                                
                    #Incompatibility graph
                    H = nx.Graph()
                    edges = list(G.edges())
                    for edge in edges:
                        H.add_node(edge)
                    for e1, e2 in itertools.combinations(H.nodes, 2):
                        i1, j1 = e1
                        i2, j2 = e2
                    
                        # Incompatibility rules
                        if(
                            j1 == j2 or
                            i1 == i2 or
                            (i2 < i1 and j2 > j1) or
                            (i2 > i1 and j2 < j1)
                        ):
                            H.add_edge(e1, e2)

                    def build_heur_clique_pool(H, m):
                        for clique in nx.find_cliques(H):
                            if len(clique) > 2:
                                for k in range(m):
                                    expr = gp.LinExpr()
                                    expr_vector = []
                                    for (i, j) in clique:
                                        var = model._mk[k, i, j]
                                        expr += var
                                        expr_vector.append(var)
                                    if expr.size() > 0:
                                        heur_clique_pool[k].append(expr_vector)
                
                
                # Decision variables
                mk = model.addVars(m, n, n, vtype=GRB.BINARY, name="mk")  # Matching variables
                fk = model.addVars(m, n, n, vtype=GRB.BINARY, name="fk")  # Free Matching variables
                dk = model.addVars(m,n, vtype=GRB.BINARY, name="dk") #Deletion matching variables L side bipartite
                ak = model.addVars(m,n, vtype=GRB.BINARY, name="ak") #Addition matching variables R side bipartite
                x = model.addVars(n, vtype=GRB.BINARY, name="x")           # Optimal string variables
                
                # Set warm start values for x
                for j in range(n):
                    x[j].start = optimal_x[j]  # Provide the initial value from the previous model
                # Set warm start values for mk
                for k in range(m):
                    for i in range(n):
                        if(sigma[k,i] == optimal_x[i]):
                            fk[k, i, i].start = 1;
                        else:
                            mk[k, i, i].start = 1;
                            
                # Objective function (as defined in the paper's model)
                model.setObjective(
                    gp.quicksum(mk[k, i, j] for i in range(n) for j in range(n) for k in range(m)) + gp.quicksum(dk[k,i] for i in range(n) for k in range(m)) + gp.quicksum(ak[k,i] for i in range(n) for k in range(m)),
                    GRB.MINIMIZE
                )

                #Linearization constraints
                for k in range(m):
                    for i in range(n):
                        model.addConstr(gp.quicksum(mk[k, i, j] + fk[k, i, j] for j in range(n)) + dk[k,i] == 1, name=f"match_row_{k}_{i}")
                        for j in range(n):
                            model.addConstr(gp.quicksum(mk[k, i, j] + fk[k, i, j] for i in range(n)) + ak[k,j] == 1, name=f"match_col_{k}_{j}")
                            model.addConstr(fk[k, i, j] <= sigma[k,i]*x[j] + (1-sigma[k,i])*(1-x[j]), name=f"linearization_{k}_{i}_{j}")
                            
                            
                            if (surrogate == 0):
                                for i_prime in range(i + 1, n):  # i' > i
                                    for j_prime in range(j):  # j' < j
                                        model.addConstr(mk[k, i, j] + fk[k, i, j] + mk[k, i_prime, j_prime] + fk[k, i_prime, j_prime] <= 1, name=f"constraint_33_{k}_{i}_{j}_{i_prime}_{j_prime}")
                            else:
                                model.addConstr(
                                    n*(mk[k,i, j]+fk[k, i, j]) + gp.quicksum(mk[k, i_prime, j_prime] + fk[k, i_prime, j_prime] for i_prime in range(i) for j_prime in range(j + 1, n) ) +
                                    gp.quicksum(mk[k, i_prime, j_prime] + fk[k, i_prime, j_prime] for j_prime in range(j) for i_prime in range(i + 1, n) ) <= n,
                                    name=f"constraint2_{k}_{i}_{j}_surr"
                                )
                ##Implications based on number of matching edges
                    model.addConstr(gp.quicksum(mk[k, i, j] + fk[k, i, j] for i in range(n) for j in range(n)) <= n, name=f"max_num_match_{k}")
                    model.addConstr(gp.quicksum(mk[k, i, j] + fk[k, i, j] for i in range(n) for j in range(n)) >= math.ceil((n+1)/2), name=f"min_num_match_{k}")
                    model.addConstr(gp.quicksum(mk[k, i, j] + fk[k, i, j] for i in range(n) for j in range(i+math.ceil(n/2),n)) + \
                                    gp.quicksum(mk[k, i, j] + fk[k, i, j] for j in range(n) for i in range(j+math.ceil(n/2),n)) == 0, name=f"fix_zero_match_{k}")

                   
                
                def find_new_nocross_cuts(m_vals, f_vals, m, n):
                    #Always all cuts
                    cuts = []
                    for k in range(m):
                        for i in range(n):
                            for j in range(n):
                                l1 = max(min(i - 1, n - j - 1), 0)
                                l2 = max(min(n - i - 1, j - 1), 0)
                                lhs_value1 = m_vals[k, i, j] + f_vals[k, i, j] + sum(m_vals[k, i - q, j + q] + f_vals[k, i - q, j + q] for q in range(1, l1 + 1)) + sum(m_vals[k, i + q, j - q] + f_vals[k, i + q, j - q] for q in range(1, l2 + 1)) +\
                                sum(m_vals[k, 0, j + q] + f_vals[k, 0, j + q] for q in range(l1 + 1, n - j)) + \
                                sum(m_vals[k, i - q, n - 1] + f_vals[k, i - q, n - 1] for q in range(l1 + 1, i + 1)) + \
                                sum(m_vals[k, i + q, 0] + f_vals[k, i + q, 0] for q in range(l2 + 1, n - i)) + \
                                sum(m_vals[k, n - 1, j - q] + f_vals[k, n - 1, j - q] for q in range(l2 + 1, j + 1))
                                if lhs_value1 -1 > 0.0001:
                                    cut_expr = mk[k, i, j] + fk[k, i, j] + \
                                    gp.quicksum(mk[k, i - q, j + q] + fk[k, i - q, j + q] for q in range(1, l1 + 1)) + \
                                    gp.quicksum(mk[k, i + q, j - q] + fk[k, i + q, j - q] for q in range(1, l2 + 1)) + \
                                    gp.quicksum(mk[k, 0, j + q] + fk[k, 0, j + q] for q in range(l1 + 1, n - j)) + \
                                    gp.quicksum(mk[k, i - q, n - 1] + fk[k, i - q, n - 1] for q in range(l1 + 1, i + 1)) + \
                                    gp.quicksum(mk[k, i + q, 0] + fk[k, i + q, 0] for q in range(l2 + 1, n - i)) + \
                                    gp.quicksum(mk[k, n - 1, j - q] + fk[k, n - 1, j - q] for q in range(l2 + 1, j + 1)) <= 1
                                    cuts.append(cut_expr)

                    return cuts
                    
                def find_linear_incomp_cuts(f_vals, m, n):
                    #Always all cuts
                    cuts = []

                    for k in range(m):
                        for k_prime in range(k+1, m):
                            for j in range(n):
                                lhs_value1 = sum(sigma[k,i]*f_vals[k, i, j] for i in range(n)) + sum((1-sigma[k_prime,i])*f_vals[k_prime, i, j] for i in range(n)) - 1
                                lhs_value2 = sum((1-sigma[k,i])*f_vals[k, i, j] for i in range(n)) + sum(sigma[k_prime,i]*f_vals[k_prime, i, j] for i in range(n)) - 1
                                if lhs_value1 > 0:
                                    cut_expr = gp.quicksum(sigma[k,i]*model._fk[k, i, j] for i in range(n)) + gp.quicksum((1-sigma[k_prime,i])*model._fk[k_prime, i, j] for i in range(n))   <= 1
                                    cuts.append(cut_expr)
                                if lhs_value2 > 0:
                                    cut_expr = gp.quicksum((1-sigma[k,i])*model._fk[k, i, j] for i in range(n)) + gp.quicksum(sigma[k_prime,i]*model._fk[k_prime, i, j] for i in range(n))   <= 1
                                    cuts.append(cut_expr)

                    return cuts
                
                def build_no_cross_pool(n, m):
                    for k in range(m):
                        for i in range(n):
                            for j in range(n):
                                for i_prime in range(i + 1, n):  # i' > i
                                    for j_prime in range(j):  # j' < j
                                        expr = gp.LinExpr()
                                        expr_vector = []
                                        expr = model._mk[k, i, j] + model._mk[k, i_prime, j_prime] + model._fk[k, i, j] + model._fk[k, i_prime, j_prime]
                                        expr_vector.append(model._mk[k, i, j])
                                        expr_vector.append(model._mk[k, i_prime, j_prime])
                                        expr_vector.append(model._fk[k, i, j])
                                        expr_vector.append(model._fk[k, i_prime, j_prime])
                                        if expr.size() > 0:
                                            no_cross_pool.append(expr_vector)

                
                def find_no_crossing_cuts_from_pool(type):
                    cuts = []
                    for idx, cut in enumerate(model._nocrosspool):
                        violation = 0
                        for var_element in cut:
                            if(type == 0):
                                violation += model.cbGetSolution(var_element)
                            else:
                                violation += model.cbGetNodeRel(var_element)
                        if violation - 1 > 0.0001:
                            cut_expr = gp.quicksum(cut) <= 1
                            cuts.append(cut_expr)
                    return cuts
                
                def find_no_crossing_cuts(m_vals, f_vals, m, n):
                    #Always all cuts
                    cuts = []

                    for k in range(m):
                        for i in range(n):
                            for j in range(n):
                                for i_prime in range(i + 1, n):  # i' > i
                                    for j_prime in range(j):  # j' < j
                                        lhs_value1 = m_vals[k, i, j] + m_vals[k, i_prime, j_prime] + f_vals[k, i, j] + f_vals[k, i_prime, j_prime] - 1
                                        if lhs_value1 > 0:
                                                best_i1, best_j1 = i, j
                                                best_iprime, best_jprime = i_prime, j_prime
                                                cut_expr = model._mk[k, best_i1, best_j1] + model._mk[k, best_iprime, best_jprime] + model._fk[k, best_i1, best_j1] + model._fk[k, best_iprime, best_jprime]   <= 1
                                                cuts.append(cut_expr)
                    return cuts

                def build_special_cliques_pool(n, m):
                    for k in range(m): 
                        for i in range(n):
                            for j in range(n):

                                expr1 = gp.LinExpr()
                                expr1_vector = []
                                expr2 = gp.LinExpr()
                                expr2_vector = []
                                
                                expr1 += model._mk[k, i, j]
                                expr1_vector.append(model._mk[k, i, j])
                                expr2 += model._mk[k, i, j]
                                expr2_vector.append(model._mk[k, i, j])
                                expr1 += model._fk[k, i, j]
                                expr1_vector.append(model._fk[k, i, j])
                                expr2 += model._fk[k, i, j]
                                expr2_vector.append(model._fk[k, i, j])
                                # Sum over elements to the right in the row
                                for j_prime in range(j + 1, n):
                                    expr1 += model._mk[k, i, j_prime]
                                    expr1_vector.append(model._mk[k, i, j_prime])
                                    expr1 += model._fk[k, i, j_prime]
                                    expr1_vector.append(model._fk[k, i, j_prime])
                                # Sum over elements below in the column
                                for i_prime in range(i + 1, n):
                                    expr1 += model._mk[k, i_prime, j]
                                    expr1_vector.append(model._mk[k, i_prime, j])
                                    expr1 += model._fk[k, i_prime, j]
                                    expr1_vector.append(model._fk[k, i_prime, j])
                                # Sum over elements to the right in the row
                                for j_prime in range(j):
                                    expr2 += model._mk[k, i, j_prime]
                                    expr2_vector.append(model._mk[k, i, j_prime])
                                    expr2 += model._fk[k, i, j_prime]
                                    expr2_vector.append(model._fk[k, i, j_prime])
                                # Sum over elements below in the column
                                for i_prime in range(i):
                                    expr2 += model._mk[k, i_prime, j]
                                    expr2_vector.append(model._mk[k, i_prime, j])
                                    expr2 += model._fk[k, i_prime, j]
                                    expr2_vector.append(model._fk[k, i_prime, j])
                                if expr1.size() > 0:
                                    special_cliques_pool.append(expr1_vector)
                                if expr2.size() > 0:
                                    special_cliques_pool.append(expr2_vector)
                
                def find_special_clique_cuts_from_pool(type):

                    cuts = []
                    for idx, cut in enumerate(model._specialcliquepool):
                        violation = 0
                        for var_element in cut:
                            if(type == 0):
                                violation += model.cbGetSolution(var_element)
                            else:
                                violation += model.cbGetNodeRel(var_element)
                        if violation - 1 > 0.0001:
                            cut_expr = gp.quicksum(cut) <= 1
                            cuts.append(cut_expr)
                    return cuts
                
                    
                def find_most_violated_clique_cuts(m_vals, f_vals, m, n, all_cuts):
                    cuts = []
                    #For first constr clique ineq
                    best_i1 = -1
                    best_j1 = -1
                
                    #For second constr clique ineq
                    best_i2 = -1
                    best_j2 = -1
                
                    #Clique ineq. separator
                    for k in range(m):
                        max_violation1 = 0
                        max_violation2 = 0
                        
                        for i in range(n):
                            for j in range(n):
                                lhs_value1 = m_vals[k, i, j] + f_vals[k, i, j]   # Current value of m_ij
                                lhs_value2 = m_vals[k, i, j] + f_vals[k, i, j]  # Current value of m_ij

                                #new_nocross_cuts 
                                l1 = max(min(i - 1, n - j - 1), 0)
                                l2 = max(min(n - i - 1, j - 1), 0)
                                lhs_value3 = m_vals[k, i, j] + f_vals[k, i, j] + sum(m_vals[k, i - q, j + q] + f_vals[k, i - q, j + q] for q in range(1, l1 + 1)) + sum(m_vals[k, i + q, j - q] + f_vals[k, i + q, j - q] for q in range(1, l2 + 1)) +\
                                sum(m_vals[k, 0, j + q] + f_vals[k, 0, j + q] for q in range(l1 + 1, n - j)) + \
                                sum(m_vals[k, i - q, n - 1] + f_vals[k, i - q, n - 1] for q in range(l1 + 1, i + 1)) + \
                                sum(m_vals[k, i + q, 0] + f_vals[k, i + q, 0] for q in range(l2 + 1, n - i)) + \
                                sum(m_vals[k, n - 1, j - q] + f_vals[k, n - 1, j - q] for q in range(l2 + 1, j + 1))
                                
                                
                                # Sum over elements to the right in the row
                                for j_prime in range(j + 1, n):
                                    lhs_value1 += m_vals[k, i, j_prime] + f_vals[k, i, j_prime]
                                
                                # Sum over elements below in the column
                                for i_prime in range(i + 1, n):
                                    lhs_value1 += m_vals[k, i_prime, j] + f_vals[k, i_prime, j]
                                    
                                # Sum over elements to the right in the row
                                for j_prime in range(j):
                                    lhs_value2 += m_vals[k, i, j_prime] + f_vals[k, i, j_prime]
                                
                                # Sum over elements below in the column
                                for i_prime in range(i):
                                    lhs_value2 += m_vals[k, i_prime, j] + f_vals[k, i_prime, j]
                                violation1 = lhs_value1 - 1  # Amount by which it is violated
                                violation2 = lhs_value2 - 1  # Amount by which it is violated
                                
                                if(all_cuts):
                                    if violation1 > 0.0001:
                                        best_i1, best_j1 = i, j
                                        cut_expr = model._mk[k, best_i1, best_j1] + model._fk[k, best_i1, best_j1] + gp.quicksum(model._mk[k, best_i1, j_prime] + model._fk[k, best_i1, j_prime] for j_prime in range(best_j1 + 1, n)) + \
                                        gp.quicksum(model._mk[k, i_prime, best_j1] + model._fk[k, i_prime, best_j1] for i_prime in range(best_i1 + 1, n)) <= 1
                                        cuts.append(cut_expr)

                                    if violation2 > 0.0001:
                                        best_i2, best_j2 = i, j
                                        cut_expr = model._mk[k, best_i2, best_j2] + model._fk[k, best_i2, best_j2] + gp.quicksum(model._mk[k, best_i2, j_prime] + model._fk[k, best_i2, j_prime] for j_prime in range(best_j2)) + \
                                        gp.quicksum(model._mk[k, i_prime, best_j2] + model._fk[k, i_prime, best_j2] for i_prime in range(best_i2)) <= 1
                                        cuts.append(cut_expr)

                                    if lhs_value3 -1 > 0.0001:
                                        cut_expr = mk[k, i, j] + fk[k, i, j] + \
                                        gp.quicksum(mk[k, i - q, j + q] + fk[k, i - q, j + q] for q in range(1, l1 + 1)) + \
                                        gp.quicksum(mk[k, i + q, j - q] + fk[k, i + q, j - q] for q in range(1, l2 + 1)) + \
                                        gp.quicksum(mk[k, 0, j + q] + fk[k, 0, j + q] for q in range(l1 + 1, n - j)) + \
                                        gp.quicksum(mk[k, i - q, n - 1] + fk[k, i - q, n - 1] for q in range(l1 + 1, i + 1)) + \
                                        gp.quicksum(mk[k, i + q, 0] + fk[k, i + q, 0] for q in range(l2 + 1, n - i)) + \
                                        gp.quicksum(mk[k, n - 1, j - q] + fk[k, n - 1, j - q] for q in range(l2 + 1, j + 1)) <= 1
                                        cuts.append(cut_expr)
                            
                                else:
                                    if violation1 > max_violation1:
                                        max_violation1 = violation1
                                        best_i1, best_j1 = i, j
                                    
                                    if violation2 > max_violation2:
                                        max_violation2 = violation2
                                        best_i2, best_j2 = i, j
                                
                        if(all_cuts == 0):
                            if max_violation1 > 0.0001:
                                cut_expr = model._mk[k, best_i1, best_j1] + model._fk[k, best_i1, best_j1] + gp.quicksum(model._mk[k, best_i1, j_prime] + model._fk[k, best_i1, j_prime] for j_prime in range(best_j1 + 1, n)) + \
                                    gp.quicksum(model._mk[k, i_prime, best_j1] + model._fk[k, i_prime, best_j1] for i_prime in range(best_i1 + 1, n)) <= 1
                                cuts.append(cut_expr)
                            if max_violation2 > 0.0001:
                                cut_expr = model._mk[k, best_i2, best_j2] +  model._fk[k, best_i2, best_j2] + gp.quicksum(model._mk[k, best_i2, j_prime] + model._fk[k, best_i2, j_prime] for j_prime in range(best_j2)) + \
                                    gp.quicksum(model._mk[k, i_prime, best_j2] + model._fk[k, i_prime, best_j2] for i_prime in range(best_i2)) <= 1
                                cuts.append(cut_expr)
                
                    return cuts

                def build_no_del_add_pool(n, m):
                    for k in range(m): 
                        for i in range(n):
                            expr = gp.LinExpr()
                            expr_vector = []
                            for b in range(n):
                                expr += model._mk[k, i, b]
                                expr_vector.append(model._mk[k, i, b])
                                expr += model._fk[k, i, b]
                                expr_vector.append(model._fk[k, i, b])
                            for j in range(n):
                                for b in range(n):
                                    expr += model._mk[k, b, j]
                                    expr_vector.append(model._mk[k, b, j])
                                    expr += model._fk[k, b, j]
                                    expr_vector.append(model._fk[k, b, j])
                                for i_prime in range(i):
                                    for j_prime in range(j + 1, n):
                                        expr += model._mk[k, i_prime, j_prime]
                                        expr_vector.append(model._mk[k, i_prime, j_prime])
                                        expr += model._fk[k, i_prime, j_prime]
                                        expr_vector.append(model._fk[k, i_prime, j_prime])
                                for i_prime in range(i+1, n):
                                    for j_prime in range(j):
                                        expr += model._mk[k, i_prime, j_prime]
                                        expr_vector.append(model._mk[k, i_prime, j_prime])
                                        expr += model._fk[k, i_prime, j_prime]
                                        expr_vector.append(model._fk[k, i_prime, j_prime])
                                if expr.size() > 0:
                                    no_del_add_pool.append(expr_vector)

                def find_no_del_add_cuts_from_pool(type):
                    cuts = []
                    for idx, cut in enumerate(model._nodeladdpool):
                        violation = 0
                        for var_element in cut:
                            if(type == 0):
                                violation += model.cbGetSolution(var_element)
                            else:
                                violation += model.cbGetNodeRel(var_element)
                        if violation - 1 < -0.0001:
                            cut_expr = gp.quicksum(cut) >= 1
                            cuts.append(cut_expr)
                    return cuts
                
                
                def find_most_violated_no_del_add_cuts(m_vals, f_vals, m, n, all_cuts):
                    cuts = []
                
                    #No del/add ineq. separator
                    for k in range(m):
                        max_violation = 0
                        best_i = -1
                        best_j = -1
                        
                        for i in range(n):
                            rhs = sum(m_vals[k, i, b] + f_vals[k, i, b] for b in range(n)) 
                            for j in range(n):
                                rhs += sum(m_vals[k, b, j] + f_vals[k, b, j] for b in range(n)) 
                                rhs += sum(m_vals[k, i_prime, j_prime] + f_vals[k, i_prime, j_prime] for i_prime in range(i) for j_prime in range(j + 1, n))
                                rhs += sum(m_vals[k, i_prime, j_prime] + f_vals[k, i_prime, j_prime] for i_prime in range(i+1, n) for j_prime in range(j))

                                if(all_cuts):
                                    if 1 - rhs > 0:
                                        best_i, best_j = i, j
                                        cut = 1 - gp.quicksum(model._mk[k, best_i, b] + model._fk[k, best_i, b] for b in range(n)) - gp.quicksum(model._mk[k, b, best_j] + model._fk[k, b, best_j] for b in range(n)) -\
                                        gp.quicksum(model._mk[k, i_prime, j_prime] + model._fk[k, i_prime, j_prime] for i_prime in range(best_i) for j_prime in range(best_j + 1, n)) -\
                                        - gp.quicksum(model._mk[k, i_prime, j_prime] + model._fk[k, i_prime, j_prime]  for i_prime in range(best_i + 1, n) for j_prime in range(best_j)) <= 0
                                        cuts.append(cut)
                                else: 
                                    if 1 - rhs > max_violation: 
                                        max_violation = 1 - rhs
                                        best_i, best_j = i, j

                        
                        if(all_cuts == 0): 
                            if max_violation > 0:
                                cut = 1 - gp.quicksum(model._mk[k, best_i, b] + model._fk[k, best_i, b] for b in range(n)) - gp.quicksum(model._mk[k, b, best_j] + model._fk[k, b, best_j] for b in range(n)) -\
                                    gp.quicksum(model._mk[k, i_prime, j_prime] + model._fk[k, i_prime, j_prime] for i_prime in range(best_i) for j_prime in range(best_j + 1, n)) -\
                                    - gp.quicksum(model._mk[k, i_prime, j_prime] + model._fk[k, i_prime, j_prime] for i_prime in range(best_i + 1, n) for j_prime in range(best_j)) <= 0
                                cuts.append(cut)
                            
                    return cuts

                def find_most_violated_continuity_cuts(m_vals, f_vals, m, n, all_cuts):
                    cuts = []
                    for k in range(m):
                        for i in range(n-1):
                            for j in range(n-1):
                                lhs = m_vals[k, i, j] + f_vals[k, i, j]   
                                lhs -= sum(m_vals[k, i+1, b] + f_vals[k, i+1, b] for b in range(j+1,n)) 
                                lhs -= sum(m_vals[k, b, j+1] + f_vals[k, b, j+1] for b in range(i+1,n)) 

                                if lhs > 0:
                                    cut = model._mk[k, i, j] + model._fk[k, i, j] - gp.quicksum(model._mk[k, i+1, q] + model._fk[k, i+1, q] for q in range(j+1,n)) - gp.quicksum(model._mk[k, q, j+1] + model._fk[k, q, j+1] for q in range(i+1,n)) <= 0
                                    cuts.append(cut)
                        
                    return cuts
                                
                def separator_callback(model, where):

                    global all_cuts #1 for adding all the violated inequalities
                    global surrogate #1 use the surrogate of no-crossing constraints
                    global heur_cliques #1 heuristic generation of cliques
                    global time_heur_sep
                    global use_pool #1 use the pool of cuts computed beforehand
                    global only_root_frac_cuts
                    
                    if where == GRB.Callback.MIPSOL:
                        # During MIPSOL, get the solution of the decision variables
                        if(use_pool == 0):
                            m_vals = {(k, i, j): model.cbGetSolution(model._mk[k, i, j])
                                    for k in range(model._m)
                                    for i in range(model._n)
                                    for j in range(model._n)}
                            f_vals = {(k, i, j): model.cbGetSolution(model._fk[k, i, j])
                                    for k in range(model._m)
                                    for i in range(model._n)
                                    for j in range(model._n)}
                        
                            violated_cuts_clique = find_most_violated_clique_cuts(m_vals, f_vals, m, n, all_cuts)
                        else:
                            violated_cuts_clique = find_special_clique_cuts_from_pool(0)
                        for cut in violated_cuts_clique:
                            model.cbLazy(cut)  # Add the cut dynamically
                            cuts_added.append(cut)
                            clique_cuts_added.append(cut)
                            
                        if(surrogate):
                            if(use_pool == 0): 
                                violated_cuts_no_crossing = find_no_crossing_cuts(m_vals, f_vals, m, n)
                            else:
                                violated_cuts_no_crossing = find_no_crossing_cuts_from_pool(0)
                            for cut in violated_cuts_no_crossing:
                                model.cbLazy(cut)  # Add the cut dynamically
                                cuts_added.append(cut)
                                no_crossing_added.append(cut)
                        
                        violated_cuts_continuity = find_most_violated_continuity_cuts(m_vals, f_vals, m, n, all_cuts)
                        for cut in violated_cuts_continuity:
                            model.cbLazy(cut)  # Add the cut dynamically
                            cuts_added.append(cut)
                            continuity_cuts_added.append(cut)
                            
                        if(heur_cliques):
                            for k in range(m):
                                max_violation = 0.001
                                max_viol_clique = None
                                for idx, clique in enumerate(model._heurcliquepool[k]):
                                    violation = 0
                                    for var_element in clique:
                                        violation += model.cbGetSolution(var_element)
                                    if violation - 1 > max_violation:
                                        max_violation = violation - 1
                                        max_viol_clique = clique
                                if max_viol_clique is not None: 
                                    cut_expr = gp.quicksum(max_viol_clique)
                                    model.cbLazy(cut_expr <= 1)
                                    heur_cliques_added.append(cut_expr <= 1)
                                    cuts_added.append(cut_expr <= 1)              
                                        
                    elif where == GRB.Callback.MIPNODE:
                        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
                        if status == GRB.OPTIMAL:
                        #if model.cbGetRelaxedSolution():
                            if(only_root_frac_cuts):
                                nodecnt = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
                                if nodecnt > 0:
                                    return  # Not at root node
                        
                            # During MIPNODE, get the node relaxation of the decision variables
                            if(use_pool == 0): 
                                m_vals = {(k, i, j): model.cbGetNodeRel(model._mk[k, i, j])
                                        for k in range(model._m)
                                        for i in range(model._n)
                                        for j in range(model._n)}
                                f_vals = {(k, i, j): model.cbGetNodeRel(model._fk[k, i, j])
                                    for k in range(model._m)
                                    for i in range(model._n)
                                    for j in range(model._n)}
                            
                                violated_cuts_clique = find_most_violated_clique_cuts(m_vals, f_vals, m, n, all_cuts)
                            else:
                                violated_cuts_clique = find_special_clique_cuts_from_pool(1)
                            for cut in violated_cuts_clique:
                                model.cbCut(cut)  # Add the cut dynamically
                                cuts_added.append(cut)
                                clique_cuts_added.append(cut)

                            
                            if(surrogate):
                                if(use_pool == 0): 
                                    violated_cuts_no_crossing = find_no_crossing_cuts(m_vals, f_vals, m, n)
                                else:
                                    violated_cuts_no_crossing = find_no_crossing_cuts_from_pool(1)
                                for cut in violated_cuts_no_crossing:
                                    model.cbCut(cut)  # Add the cut dynamically
                                    cuts_added.append(cut)
                                    no_crossing_added.append(cut)


                            violated_cuts_continuity = find_most_violated_continuity_cuts(m_vals, f_vals, m, n, all_cuts)
                            for cut in violated_cuts_continuity:
                                model.cbCut(cut)  # Add the cut dynamically
                                cuts_added.append(cut)
                                continuity_cuts_added.append(cut)
             
                            
                            if(heur_cliques):
                                for k in range(m):
                                    max_violation = 0.001
                                    max_viol_clique = None
                                    for idx, clique in enumerate(model._heurcliquepool[k]):
                                        violation = 0
                                        for var_element in clique:
                                            violation += model.cbGetNodeRel(var_element)
                                        if violation - 1 > max_violation:
                                            max_violation = violation - 1
                                            max_viol_clique = clique
                                    if max_viol_clique is not None: 
                                        cut_expr = gp.quicksum(max_viol_clique)
                                        model.cbCut(cut_expr <= 1)
                                        heur_cliques_added.append(cut_expr <= 1)
                                        cuts_added.append(cut_expr <= 1)

                        else:
                            return
                
                    else:
                        return
                
                    
                
                # After optimization completes, print the cuts
                def print_cuts():
                    if cuts_added:
                        print("Cuts added during the optimization:")
                        print(f"Added {len(cuts_added)} cuts")
                        print(f"Added {len(clique_cuts_added)} clique cuts")
                        print(f"Added {len(no_del_add_cuts_added)} no deletion/addition cuts")
                        print(f"Added {len(no_crossing_added)} no crossing cuts")
                        print(f"Added {len(heur_cliques_added)} no huer clique cuts")
                        for cut in clique_cuts_added:
                            print(cut)
                        for cut in no_del_add_cuts_added:
                            print(cut)
                        for cut in no_crossing_added:
                            print(cut)
                        for cut in heur_cliques_added:
                            print(cut)
                    else:
                        print("No cuts were added.")
                
                
                model._mk = mk  # Store mk inside the model
                model._fk = fk  # Store fk inside the model
                model._x = x  # Store mk inside the model
                model._m = m    # Store m (number of layers) inside the model
                model._n = n    # Store n (number of elements) inside the model
                cuts_added = []  # List to store added cuts
                clique_cuts_added = []
                no_del_add_cuts_added = []
                no_crossing_added = []
                heur_cliques_added = []
                continuity_cuts_added = []
                lin_incomp_cuts_added = []
                new_nocross_cuts_added = []
                time_heur_sep = 0
                
                
                if(heur_cliques):
                    heur_clique_pool = [[] for _ in range(m)]
                    build_heur_clique_pool(H, m)
                    model._heurcliquepool = heur_clique_pool
                    #print(len(heur_clique_pool))

                if(use_pool):
                    no_cross_pool = []
                    build_no_cross_pool(n, m)
                    model._nocrosspool = no_cross_pool
                    special_cliques_pool = []
                    build_special_cliques_pool(n, m)
                    model._specialcliquepool = special_cliques_pool
                    no_del_add_pool = []
                    build_no_del_add_pool(n, m)
                    model._nodeladdpool = no_del_add_pool

            
                # Print the model
                #model.write("model_match_sep.lp")
                #model.optimize()
                model.setParam("Seed", seed)
                model.optimize(separator_callback)
                
                # Print results
                if model.status == GRB.OPTIMAL:
                    print("Optimal solution found!")
                else:
                    print("No optimal solution found.")
                
                # Print the cuts added during the process
                #print_cuts()
                
                with open("result_matching_linear.txt", "a") as file:
                    # Collect results
                    best_incumbent = model.ObjVal if model.SolCount > 0 else "NFS" #no feasible solution
                    best_bound = model.ObjBound if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else "N/A"
                    solution_time = model.Runtime
                    gap = model.MIPGap
                    nodes = model.NodeCount
                    simplexiters = model.IterCount
                    # Write results in a single row format
                    file.write(f"I_{stringlength}_{stringnumber}_{it}.txt,{seed},{best_incumbent},{best_bound},{solution_time},{gap},{nodes},{simplexiters},{surrogate},{all_cuts},{only_root_frac_cuts},{use_pool},{len(cuts_added)},{len(clique_cuts_added)} ; {len(no_del_add_cuts_added)},{len(no_crossing_added)},{len(continuity_cuts_added)},{len(heur_cliques_added)},{len(lin_incomp_cuts_added)}\n")
            
