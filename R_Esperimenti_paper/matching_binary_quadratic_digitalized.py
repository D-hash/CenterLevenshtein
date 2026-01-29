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
import sys 


I = 3
ecc = sys.argv[1]

#Set result file
with open(f"result_matching_quadratic_{ecc}.txt", "a") as file:
    file.write(f"Instance,Seed,BestIncumbent,BestBound,SolutionTime,GAP,Nodes,SimplexIterations,CenterString\n")

print("### STARTING MATCHING BINARY ###")
#for stringlength in [5,10,15,20]:
for stringnumber in [10, 20, 30, 40, 50]:
# for stringlength in [5]:
#     for stringnumber in [10]:
        for it in range(1,I+1):
            for seed in [2025]:

                # -------------------------------------------------
                # Read input as in your quadratic model
                # -------------------------------------------------
                n, m, sigma = None, None, None

                with open(f"pools/{ecc}_{stringnumber}_{it}.txt", "r") as f:
                    lines = f.readlines()

                m = int(lines[1].split("=")[1].strip())
                K = range(m)

                sigma = [list(map(int, line.strip())) for line in lines[3:]]

                n = 0
                for s in sigma:
                    n = max(n, len(s))
                n = 2*n
                print(f"n = {n/2}")
                print(f"m = {m}")
                print("sigma =\n", sigma)

                # Positions
                Lk = {k: range(1, len(sigma[k]) + 1) for k in K}  # left side (strings)
                R = range(1, n + 1)                               # right side (median positions)

                # -------------------------------------------------
                # Model
                # -------------------------------------------------
                model = gp.Model("Matching-Linear-Median")
                model.setParam(GRB.Param.TimeLimit, 600)

                # Median bits
                x = model.addVars(R, vtype=GRB.BINARY, name="x")

                # Length encoding: z_k = 1 if position k exists in the median
                z = model.addVars(R, vtype=GRB.BINARY, name="z")
                p = min(len(s) for s in sigma)
                for i in range(1,p+1):
                    z[i].lb = 1
                # Substitution arcs: free (F) and costly (C)
                #yF = {}  # free substitutions
                yC = {}  # costly substitutions

                for k in K:
                    for h in Lk[k]:
                        for j in R:
                            #yF[(k, h, j)] = model.addVar(vtype=GRB.BINARY, name=f"yF_{k}_{h}_{j}")
                            yC[(k, h, j)] = model.addVar(vtype=GRB.BINARY, name=f"yC_{k}_{h}_{j}")

               

                # -------------------------------------------------
                # Constraints
                # -------------------------------------------------

                # 1) Each position h in string k: matched (via F or C) or deleted
                for k in K:
                    for h in Lk[k]:
                        model.addConstr(
                            gp.quicksum(yC[(k, h, j)] for j in R) <= 1,
                            name=f"match_left_{k}_{h}"
                        )
                for k in K:
                    for h in Lk[k]:
                        model.addConstr(
                            gp.quicksum(yC[(k, h, j)] for j in R) + 
                            gp.quicksum(yC[(k, u, 1)] for u in range(h+1,len(Lk[k])+1)) +
                            gp.quicksum(yC[(k, u, len(R))] for u in range(1,h)) <= 1,
                            name=f"match_left_ii_{k}_{h}"
                        )

                # 2) Each median position j: matched or inserted, controlled by z_j
                for k in K:
                    for j in R:
                        model.addConstr(
                            gp.quicksum( yC[(k, h, j)] for h in Lk[k]) <= z[j],
                            name=f"match_right_{k}_{j}"
                        )
                for k in K:
                    for j in R:
                        model.addConstr(
                            gp.quicksum( yC[(k, h, j)] for h in Lk[k]) +
                            gp.quicksum( yC[(k, 1, v)] for v in range(j+1,len(R)+1)) +
                            gp.quicksum( yC[(k, len(Lk[k]), v)] for v in range(1,j))
                            <= 1,
                            name=f"match_right_ii_{k}_{j}"
                        )

                # 3) z monotonicity: z_{j+1} <= z_j (prefix of ones)
                for j in range(1, n):
                    model.addConstr(z[j+1] <= z[j], name=f"z_rank_{j}")

                for k in K:
                    for u in Lk[k]:
                        for v in R:
                            lhs = gp.LinExpr()
                            lhs += gp.quicksum( yC[(k, u, j)] for j in R if j != v)
                            lhs += gp.quicksum( yC[(k, h, v)] for h in Lk[k] if h != u)
                            for h in range(1, u):
                                for j in range(v+1, n+1):
                                    lhs +=  yC[(k, h, j)]
                            for h in range(u+1, len(Lk[k])+1):
                                for j in range(1,v):
                                    lhs +=  yC[(k, h, j)]
                            model.addConstr(
                                lhs <=
                                len(Lk[k]) * (1 - yC[(k, u, v)]),
                                name=f"nocross_l_{k}_{u}_{v}"
                            )
                                    


           
                lhs = gp.LinExpr()
                for k in K:
                    lhs += gp.quicksum(z[j] for j in range(1, n + 1))
                    lhs += len(sigma[k])

                    for i in range(1,len(sigma[k])+1):
                        for j in range(1,n+1):
                            lhs -= 2*yC[(k,i,j)]
                            lhs += sigma[k][i-1]*yC[(k,i,j)]*(1-x[j])
                            lhs += (1-sigma[k][i-1])*yC[(k,i,j)]*x[j]
                model.setObjective(lhs)


                # Print the model
                #model.write("model_match_sep.lp")
                model.setParam("Seed", seed)
                model.optimize()
                
                # Print results
                if model.status == GRB.OPTIMAL:
                    print("Optimal solution found!")
                else:
                    print("No optimal solution found.")
                # Compute median length from z-variables
                median_length = sum(int(z[k].X) for k in range(1, n+1))

                # Build the median string
                optstring = ""
                for k in range(1, median_length + 1):
                    optstring += str(int(x[k].X))

                print("Median string:", optstring)
                print("Median length:", median_length)

                # Print the cuts added during the process
                #print_cuts()
                print(optstring)
                with open(f"result_matching_quadratic_{ecc}.txt", "a") as file:
                    # Collect results
                    best_incumbent = model.ObjVal if model.SolCount > 0 else "NFS" #no feasible solution
                    best_bound = model.ObjBound if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else "N/A"
                    solution_time = model.Runtime
                    gap = model.MIPGap
                    nodes = model.NodeCount
                    simplexiters = model.IterCount
                    # Write results in a single row format
                    file.write(f"I_{ecc}_{stringnumber}_{it}.txt,{seed},{best_incumbent},{best_bound},{solution_time},{gap},{nodes},{simplexiters},{optstring}\n")
            
