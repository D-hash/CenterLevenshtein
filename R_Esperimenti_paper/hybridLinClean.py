
from gurobipy import Model, GRB, quicksum
import numpy as np
import math

#num Instances
I = 5

#Set result file
with open("result_hybrid_lin_clean.txt", "a") as file:
    file.write(f"Instance,Seed,Best Incumbent,BestBound,SolutionTime,GAP,Nodes,SimplexIterations\n")

print("### STARTING HYBRID ###")


for stringlength in range(50,60,10):
    for stringnumber in range(20,30,10):
        for it in range(1):
            for seed in [2025]:
                # Initialize variables
                n, m, sigma = None, None, None
                
                # Read the file
                with open(f"I_{stringlength}_{stringnumber}_{it}.txt", "r") as f:  # Corrected
                    lines = f.readlines()
                
                # Extract n and m
                n = int(lines[0].split("=")[1].strip())
                m = int(lines[1].split("=")[1].strip())
                
                # Extract sigma (from line index 3 onwards)
                sigma = np.array([list(map(int, line.split())) for line in lines[3:]])
                
                # Print to verify
                print(f"Instance_{stringlength}_{stringnumber}_{it}.txt")
                print(f"n = {n}")
                print(f"m = {m}")
                print("sigma =\n", sigma)
                
                
                # Problem parameters
                P = [(i, j) for i in range(n) for j in range(n) if abs(i - j) < math.ceil(n / 2)]
                K = range(m)  # layers k
                
                # Source and destination
                source = {p: p[0] for p in P}
                dest = {p: p[1] for p in P}
                
                median = (np.sum(sigma, axis=0) >= sigma.shape[0] / 2).astype(int)
                print("Heuristic Median String:", median.tolist())
                total_distance = np.sum(np.abs(sigma - median))
                print("Total Hamming Distance:", total_distance)
                optimal_x = {j: int(median[j]) for j in range(n)}
                print("Optimal Median String:", optimal_x)
                
                
                # Define auxiliary parameters
                def one(k, p): return sigma[k, source[p]] == 1
                def zero(k, p): return sigma[k, source[p]] == 0

                #Define arcs
                arc = {
                    (p, q): 1
                    for p in P for q in P
                    if (source[p] < source[q] and dest[p] < dest[q])
                    and not (source[p] < source[q] - 1 and dest[p] < dest[q] - 1)
                    and not ((source[q] - source[p] + dest[q] - dest[p]) - 2 > n - math.ceil((n + 1) / 2))
                }
                #print(arc)

                cost = {
                    (p, q): (source[q] - source[p] + dest[q] - dest[p]) - 2 for (p, q) in arc
                } #if (source[q] - source[p] + dest[q] - dest[p]) - 2 < n/2
                    
                # Start and finish costs
                startcost = {p: source[p] + dest[p] for p in P} #-2 removed, P starts from 0
                finishcost = {p: 2*(n-1) - source[p] - dest[p] for p in P} #2n removed, P starts from 0

                
                # Model
                m = Model("hybrid_lin")
                print_lps = 0   #active only print of lps

                #OPTIONS
                use_ub = 0 #1 to enable upper bound on c^k_i variables
                heuristic = 0 #1 to enable preliminary primal heuristic
                
                # Variables
                x = {}
                for i in range(n):
                    name = f"x_{i}"
                    x[i] = m.addVar(vtype=GRB.BINARY, name=name)
                z = {}
                for k in K:
                    for (p, q) in arc.keys():
                        name = f"z_{k}_{p[0]}_{p[1]}_{q[0]}_{q[1]}"
                        z[k, (p, q)] = m.addVar(vtype=GRB.CONTINUOUS, name=name)
                start = {}
                finish = {}
                c = {}
                for k in K:
                    for i in range(n):
                        name = f"c_{k}_{i}"
                        c[k, i] = m.addVar(vtype=GRB.CONTINUOUS, name=name)
                for k in K:
                    for p in P:
                        #Upper bound constraints imposed in the definition
                        start_ub = 0.0 if not (source[p] == 0 or dest[p] == 0) else 1
                        finish_ub = 0.0 if not (source[p] == n - 1 or dest[p] == n - 1) else 1
                
                        name = f"start_{k}_{p[0]}_{p[1]}"
                        start[k, p] = m.addVar(vtype=GRB.CONTINUOUS, ub=start_ub, name=name)
                
                        name = f"finish_{k}_{p[0]}_{p[1]}"
                        finish[k, p] = m.addVar(vtype=GRB.CONTINUOUS, ub=finish_ub, name=name)
            
            

                #NON-REQUIRED CONSTRAINTS (CUTS) ARE con2 AND start_z_link_constr
                # Constraints
                con1 = {} 
                cut = {}
                con3 = {} 
                con4 = {} 

                #for heuristic setting
                if(heuristic):
                    min_num_match_edges = n - 4 #n - 6  #math.ceil((n+1)/2)
                    max_diff_s_d = 2 #3 #math.ceil(n/2)
                else:
                    min_num_match_edges = math.ceil((n+1)/2)
                    max_diff_s_d = math.ceil(n/2)
                
                for k in K:
                    con1[k] = m.addConstr(quicksum(z[k, (p, q)] for (p, q) in arc.keys() if (k, (p, q)) in z) >= min_num_match_edges - 1, name=f"con1_{k}")
                
                    for p in P:
                        valid_qs_out = [q for q in P if (p, q) in arc.keys()]  # Outgoing arcs
                        valid_qs_in = [q for q in P if (q, p) in arc.keys()]   # Incoming arcs
                        m.addConstr(
                            quicksum(z[k, (p, q)] for q in valid_qs_out if (k, (p, q)) in z)
                            - quicksum(z[k, (q, p)] for q in valid_qs_in if (k, (q, p)) in z)
                            == start[k, p] - finish[k, p], name=f"flow_{k}_{p[0]}_{p[1]}"
                        )
                        #The following two are cuts
                        m.addConstr(start[k, p] + finish[k, p] <= 1, name=f"con2_{k}_{p[0]}_{p[1]}")
                        m.addConstr(start[k, p] + quicksum(z[k, (q, p)] for q in valid_qs_in if (k, (q, p)) in z) <=1, name=f"start_z_link_{k}_{p[0]}_{p[1]}")

                        m.addConstr(start[k, p] <= (source[p] == 0 or dest[p] == 0), name=f"fix_start_{k}_{p[0]}_{p[1]}")
                        m.addConstr(finish[k, p] <= (source[p] == n-1 or dest[p] == n-1), name=f"fix_finish_{k}_{p[0]}_{p[1]}")
                    #Constraints linear version

                    for i in range(n):
                        m.addConstr(
                            c[k, i] >= quicksum(z[k, (q, p)] for (q, p) in arc.keys() if dest[p] == i if sigma[k, source[p]] == 1 if (k, (q, p)) in z) + \
                        quicksum(start[k, p] for p in P if dest[p] == i if sigma[k, source[p]] == 1) - x[i], name=f"con_sigma1_{k}_{i}")
                
                        m.addConstr(
                                c[k, i] >= quicksum(z[k, (q, p)] for (q, p) in arc.keys() if dest[p] == i if sigma[k, source[p]] == 0 if (k, (q, p)) in z) + \
                        quicksum(start[k, p] for p in P if dest[p] == i if sigma[k, source[p]] == 0) + x[i] - 1, name=f"con_sigma0_{k}_{i}")

                        #Upper bound on c[k,i]
                        if(use_ub):
                            m.addConstr(
                                c[k, i] <= quicksum(z[k, (q, p)] for (q, p) in arc.keys() if dest[p] == i if (k, (q, p)) in z) + \
                            quicksum(start[k, p] for p in P if dest[p] == i), name=f"con_ub_c_{k}_{i}")

                    con3[k] = m.addConstr(quicksum(start[k, p] for p in P if startcost[p] <= (n-min_num_match_edges) and (source[p] == 0 or dest[p] == 0)) == 1, name=f"con3_{k}") 
                    con4[k] = m.addConstr(quicksum(finish[k, p] for p in P if finishcost[p] <= (n-min_num_match_edges) and (source[p] == n - 1 or dest[p] == n - 1)) == 1, name=f"con4_{k}")
                    
                # Objective linear
                obj = quicksum(2*(n - 1 - quicksum(z[k, (p, q)] for (p, q) in arc.keys() if (k, (p, q)) in z)) for k in K) + quicksum(c[k,i] for k in K for i in range(n))
                m.setObjective(obj, GRB.MINIMIZE)

                m.setParam("Seed", seed)
                if(print_lps):
                    # Print the model
                    string = "hyb_lin_clean_" + "20_20_"+ str(it) + ".lp"
                    m.write(string)
                else:
                    # Set time limit to 600 seconds
                    m.setParam(GRB.Param.TimeLimit, 600)
                
                    m._c = c  
                    m._z = z    
                    m._start = start
                    m._x = x 
                    cuts_added = []  # List to store added cuts

                    
                    # Heuristic call
                    if(heuristic):
                        m.setParam(GRB.Param.TimeLimit, 60)
                        m.optimize()
                        min_num_match_edges = math.ceil((n+1)/2)
                        max_diff_s_d = math.ceil(n/2)
                        for k in K:
                            con1[k].RHS = min_num_match_edges - 1
                            m.remove(cut[k])
                            m.remove(con3[k])
                            m.remove(con4[k])
                            m.addConstr(quicksum(start[k, p] for p in P if abs(source[p] - dest[p]) >= (max_diff_s_d)) + quicksum(finish[k, p] for p in P if abs(source[p] - dest[p]) >= (max_diff_s_d)) +
                                quicksum(z[k, (q, p)] for (q, p) in arc if (k, (q, p)) in z if abs(source[p] - dest[p]) >= (max_diff_s_d)) == 0, name=f"cut_final_{k}")
                            m.addConstr(quicksum(start[k, p] for p in P if startcost[p] <= (n-min_num_match_edges) and (source[p] == 0 or dest[p] == 0)) == 1, name=f"con3_final_{k}") 
                            m.addConstr(quicksum(finish[k, p] for p in P if finishcost[p] <= (n-min_num_match_edges) and (source[p] == n - 1 or dest[p] == n - 1)) == 1, name=f"con4_final_{k}")
                    
                    
                    m.setParam(GRB.Param.TimeLimit, 3600)
                    m.optimize()

                    
                    # Print results
                    if m.status == GRB.OPTIMAL:
                        print("Optimal solution found!")
                        for i in range(n):
                            print(f"x[{i}] = {x[i].x}")
                        for k in K:
                            for (p, q) in arc.keys():
                                if((k, (p, q)) in z and z[k, (p,q)].x > 0): print(f"z[{k},{(p,q)}] = {z[k, (p,q)].x}")
                            #for p in P:
                                #if(start[k, p].x > 0): print(f"start[{k},{p[0]}, {p[1]}] = {start[k, p].x}")
                                #if(finish[k, p].x > 0): print(f"finish[{k},{p[0]}, {p[1]}] = {finish[k, p].x}")
                                #if(c[k,p]. x > 0): print(f"c[{k},{p}] = {c[k,p].x}")
                    else:
                        print("No optimal solution found.")
                
                    with open("result_hybrid_lin_clean.txt", "a") as file:
                        # Collect results
                        best_incumbent = m.ObjVal if m.SolCount > 0 else "NFS" #no feasible solution
                        best_bound = m.ObjBound if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else "N/A"
                        solution_time = m.Runtime
                        gap = m.MIPGap
                        nodes = m.NodeCount
                        simplexiters = m.IterCount
                        # Write results in a single row format
                        file.write(f"I_{stringlength}_{stringnumber}_{it}.txt,{seed},{best_incumbent},{best_bound},{solution_time},{gap},{nodes},{simplexiters}\n")
