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
from Levenshtein import median, median_improve, distance as lev_distance

def find_median_string(str_list, improve_steps=0):
    med = median(str_list)
    for _ in range(improve_steps):
        med = median_improve(med, str_list)
    return med

I = 3
#ecc = sys.argv[1]

#Set result file
with open(f"result_matching_digitalized_2n_loop_warm.txt", "a") as file:
    file.write(f"Instance,Seed,BestIncumbent,BestBound,SolutionTime,GAP,Nodes,SimplexIterations,CenterString,HeuTime\n")

print("### STARTING MATCHING BINARY ###")
for stringlength in [20]:
    for stringnumber in [10, 20, 30, 40, 50]:
# for stringlength in [5]:
#     for stringnumber in [10]:
        for it in range(0,I):
            for seed in [2025]:

                # -------------------------------------------------
                # Read input as in your quadratic model
                # -------------------------------------------------
                n, m, sigma = None, None, None

                with open(f"I_{stringlength}_{stringnumber}_{it}.txt", "r") as f:
                    lines = f.readlines()

                m = int(lines[1].split("=")[1].strip())
                K = range(m)

                sigma = [list(map(int, line.strip())) for line in lines[3:]]
                sigmas = ["".join(map(str, row)) for row in sigma]
                n = 0
                for s in sigma:
                    n = max(n, len(s))
                n = 2*n
                print(f"n = {n/2}")
                print(f"m = {m}")
                print("sigma =\n", sigma)
                heutime = -time.time()
                heumedian = find_median_string(sigmas, 100)
                heutime += time.time()
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
                for idx, c in enumerate(heumedian):
                    x[idx+1].Start = int(c)
                # Length encoding: z_k = 1 if position k exists in the median
                z = model.addVars(R, vtype=GRB.BINARY, name="z")
                p = min(len(s) for s in sigma)
                for i in range(1,p+1):
                    z[i].lb = 1
                # Substitution arcs: free (F) and costly (C)
                yF = {}  # free substitutions
                yC = {}  # costly substitutions

                for k in K:
                    for h in Lk[k]:
                        for j in R:
                            yF[(k, h, j)] = model.addVar(vtype=GRB.BINARY, name=f"yF_{k}_{h}_{j}")
                            yC[(k, h, j)] = model.addVar(vtype=GRB.BINARY, name=f"yC_{k}_{h}_{j}")

                # Deletion loops (on L^i)
                yDel = {}
                for k in K:
                    for h in Lk[k]:
                        yDel[(k, h)] = model.addVar(vtype=GRB.BINARY, name=f"yDel_{k}_{h}")

                # Insertion loops (on R)
                yIns = {}
                for k in K:
                    for j in R:
                        yIns[(k, j)] = model.addVar(vtype=GRB.BINARY, name=f"yIns_{k}_{j}")

                # -------------------------------------------------
                # Constraints
                # -------------------------------------------------

                # 1) Each position h in string k: matched (via F or C) or deleted
                for k in K:
                    for h in Lk[k]:
                        model.addConstr(
                            gp.quicksum(yF[(k, h, j)] + yC[(k, h, j)] for j in R) + yDel[(k, h)] == 1,
                            name=f"match_left_{k}_{h}"
                        )
                for k in K:
                    for h in Lk[k]:
                        model.addConstr(
                            gp.quicksum(yF[(k, h, j)] + yC[(k, h, j)] for j in R) + 
                            gp.quicksum(yF[(k, u, 1)] + yC[(k, u, 1)] for u in range(h+1,len(Lk[k])+1)) +
                            gp.quicksum(yF[(k, u, len(R))] + yC[(k, u, len(R))] for u in range(1,h)) <= 1,
                            name=f"match_left_ii_{k}_{h}"
                        )

                # 2) Each median position j: matched or inserted, controlled by z_j
                for k in K:
                    for j in R:
                        model.addConstr(
                            gp.quicksum(yF[(k, h, j)] + yC[(k, h, j)] for h in Lk[k]) + yIns[(k, j)]  == z[j],
                            name=f"match_right_{k}_{j}"
                        )
                for k in K:
                    for j in R:
                        model.addConstr(
                            gp.quicksum(yF[(k, h, j)] + yC[(k, h, j)] for h in Lk[k]) +
                            gp.quicksum(yF[(k, 1, v)] + yC[(k, 1, v)] for v in range(j+1,len(R)+1)) +
                            gp.quicksum(yF[(k, len(Lk[k]), v)] + yC[(k, len(Lk[k]), v)] for v in range(1,j))
                            <= 1,
                            name=f"match_right_ii_{k}_{j}"
                        )

                # 3) z monotonicity: z_{j+1} <= z_j (prefix of ones)
                for j in range(1, n):
                    model.addConstr(z[j+1] <= z[j], name=f"z_rank_{j}")



                
                # # 4) Planarity (no crossing arcs) on substitutions
                # #    For each string k, for any (h,j) and (u,v) that cross, enforce:
                # #    yF + yC <= 1
                # for k in K:
                #     for h in Lk[k]:
                #         for u in Lk[k]:
                #             for j in R:
                #                 for v in R:
                #                     # crossing if (h < u and j > v) or (h > u and j < v)
                #                     if (h < u and j > v) or (h > u and j < v):
                #                         model.addConstr(
                #                             yF[(k, h, j)] + yC[(k, h, j)] +
                #                             yF[(k, u, v)] + yC[(k, u, v)] <= 1,
                #                             name=f"nocross_{k}_{h}_{j}_{u}_{v}"
                #                         )
                # for k in K:
                #     for h in Lk[k]:
                #         for u in Lk[k]:
                #             if h < u: 
                #                 for v in R:
                #                     lhs = gp.LinExpr()
                #                     for j in R:
                #                         if j <= v: continue
                #                         lhs +=  yF[(k, h, j)] + yC[(k, h, j)]
                #                         # crossing if (h < u and j > v) or (h > u and j < v)  
                #                     model.addConstr(
                #                             lhs +
                #                             yF[(k, u, v)] + yC[(k, u, v)] <= 1,
                #                             name=f"nocross_{k}_{h}_{u}_{v}"
                #                         )
                #             if h > u:
                #                 for v in R:
                #                     lhs = gp.LinExpr()
                #                     for j in R:
                #                         if j >= v: continue
                #                         lhs +=  yF[(k, h, j)] + yC[(k, h, j)]
                #                         # crossing if (h < u and j > v) or (h > u and j < v)                                    
                #                     model.addConstr(
                #                             lhs +
                #                             yF[(k, u, v)] + yC[(k, u, v)] <= 1,
                #                             name=f"nocross_{k}_{h}_{u}_{v}"
                #                         )
                for k in K:
                    for u in Lk[k]:
                        for v in R:
                            lhs = gp.LinExpr()
                            lhs += gp.quicksum(yF[(k, u, j)] + yC[(k, u, j)] for j in R if j != v)
                            lhs += gp.quicksum(yF[(k, h, v)] + yC[(k, h, v)] for h in Lk[k] if h != u)
                            for h in range(1, u):
                                for j in range(v+1, n+1):
                                    lhs += yF[(k, h, j)] + yC[(k, h, j)]
                            for h in range(u+1, len(Lk[k])+1):
                                for j in range(1,v):
                                    lhs += yF[(k, h, j)] + yC[(k, h, j)]
                            model.addConstr(
                                lhs + yDel[(k,u)] <=
                                len(Lk[k]) * (1 - yF[(k, u, v)] - yC[(k, u, v)]),
                                name=f"nocross_l_{k}_{u}_{v}"
                            )
                            # model.addConstr(
                            #     lhs + yIns[(k, v)] <=
                            #     len(Lk[k]) * (1 - yF[(k, u, v)] - yC[(k, u, v)]),
                            #     name=f"nocross_r_{k}_{u}_{v}"
                            # )
                                    


                # 5) Activation constraints for free substitutions (F^i)
                #    y_e + (1 - 2*lambda_h) x_k <= 1 - lambda_h
                #    where lambda_h = sigma[k][h-1]
                for k in K:
                    for h in Lk[k]:
                        lam = sigma[k][h-1]
                        for j in R:
                            if lam == 1:
                                # yF + (1 - 2*1)*x_j <= 1 - 1  -> yF - x_j <= 0
                                model.addConstr(
                                    yF[(k, h, j)] - x[j] <= 0,
                                    name=f"act1_{k}_{h}_{j}"
                                )
                            else:
                                # yF + (1 - 0)*x_j <= 1 -> yF + x_j <= 1
                                model.addConstr(
                                    yF[(k, h, j)] + x[j] <= 1,
                                    name=f"act0_{k}_{h}_{j}"
                                )

                # -------------------------------------------------
                # Objective: sum of distances (linear)
                #   - deletions: cost 1
                #   - insertions: cost 1
                #   - costly substitutions (C): cost 1
                #   - free substitutions (F): cost 0
                # -------------------------------------------------
                obj = gp.LinExpr()

                for k in K:
                    # deletions
                    for h in Lk[k]:
                        obj += yDel[(k, h)]

                    # insertions
                    for j in R:
                        obj += yIns[(k, j)]

                    # costly substitutions
                    for h in Lk[k]:
                        for j in R:
                            obj += yC[(k, h, j)]

                model.setObjective(obj, GRB.MINIMIZE)


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
                with open(f"result_matching_digitalized_2n_loop_warm.txt", "a") as file:
                    # Collect results
                    best_incumbent = model.ObjVal if model.SolCount > 0 else "NFS" #no feasible solution
                    best_bound = model.ObjBound if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else "N/A"
                    solution_time = model.Runtime
                    gap = model.MIPGap
                    nodes = model.NodeCount
                    simplexiters = model.IterCount
                    # Write results in a single row format
                    file.write(f"I_{stringlength}_{stringnumber}_{it}.txt,{seed},{best_incumbent},{best_bound},{solution_time},{gap},{nodes},{simplexiters},{optstring},{heutime}\n")
            
