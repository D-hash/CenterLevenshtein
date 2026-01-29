from gurobipy import Model, GRB, quicksum, LinExpr
import numpy as np
import math

#num Instances
I = 3

#Set result file
with open("result_hybrid_binary_general_length.txt", "a") as file:
    file.write(f"Instance,Seed,Best Incumbent,BestBound,SolutionTime,GAP,Nodes,SimplexIterations,Center\n")

print("### STARTING HYBRID ###")


for stringlength in [5, 10, 15, 20]:
    for stringnumber in [10, 20, 30, 40, 50]:
        for it in range(I):
            for seed in [2025]:

                # Read instance
                with open(f"I_{stringlength}_{stringnumber}_{it}.txt") as f:
                    lines = f.readlines()

                m = int(lines[1].split("=")[1].strip())
                sigma = [list(map(int, line.strip())) for line in lines[3:]]
                n = max(len(s) for s in sigma)

                # Helpers
                def s_of(p): return p[0]
                def d_of(p): return p[1]

                def is_one(k, s_idx):
                    return 1 <= s_idx <= len(sigma[k]) and sigma[k][s_idx-1] == 1

                def is_zero(k, s_idx):
                    return 1 <= s_idx <= len(sigma[k]) and sigma[k][s_idx-1] == 0


                # Build A^k and cost
                A = {}
                cost = {}

                for k in range(m):
                    n_k = len(sigma[k])
                    Pk = [(i, j) for i in range(n_k+2) for j in range(n+2)]
                    A_k = {}
                    cost_k = {}

                    for p in Pk:
                        s_p, d_p = s_of(p), d_of(p)
                        for q in Pk:
                            s_q, d_q = s_of(q), d_of(q)

                            if not (s_p < s_q and d_p < d_q):
                                continue

                            if q == (n_k+1, n+1) and s_p > n_k:
                                continue

                            A_k[(p, q)] = True
                            cost_k[(p, q)] = (s_q - s_p) + (d_q - d_p) - 2


                    A[k] = A_k
                    cost[k] = cost_k


                # Model
                model = Model("center_string_flow")
                model.setParam(GRB.Param.TimeLimit, 600)

                # Variables
                x = {i: model.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in range(1, n+1)}
                y = {(k, p, q): model.addVar(vtype=GRB.BINARY,
                                            name=f"y_{k}_{s_of(p)}_{d_of(p)}_{s_of(q)}_{d_of(q)}")
                    for k in range(m) for (p, q) in A[k]}

                z = {(k, i): model.addVar(vtype=GRB.BINARY, name=f"z_{k}_{i}")
                    for k in range(m) for i in range(1, n+1)}

                L = model.addVar(vtype=GRB.INTEGER, lb=1, ub=n, name="L")
                w = {k: model.addVar(vtype=GRB.INTEGER, lb=0, ub=n, name=f"w_{k}") for k in range(m)}

                model.update()


                # Flow + L + w constraints
                for k in range(m):
                    n_k = len(sigma[k])
                    Pk = [(i, j) for i in range(n_k+2) for j in range(n+2)]
                    start = (0, 0)
                    end = (n_k+1, n+1)

                    # Flow constraints
                    for p in Pk:
                        outgoing = [q for q in Pk if (p, q) in A[k]]
                        incoming = [q for q in Pk if (q, p) in A[k]]

                        rhs = 1 if p == start else -1 if p == end else 0

                        model.addConstr(
                            quicksum(y[(k, p, q)] for q in outgoing) -
                            quicksum(y[(k, q, p)] for q in incoming)
                            == rhs
                        )

                    # L and w^k
                    Lsum = quicksum(d_of(p) * y[(k, p, end)]
                                    for p in Pk if (p, end) in A[k])

                    model.addConstr(L >= Lsum)
                    model.addConstr(w[k] >= L - Lsum)


                # Mismatch constraints
                for k in range(m):
                    n_k = len(sigma[k])
                    for i in range(1, n+1):
                        sum1 = quicksum(y[(k, p, q)]
                                        for (p, q) in A[k]
                                        if d_of(q) == i and is_one(k, s_of(q)))

                        sum0 = quicksum(y[(k, p, q)]
                                        for (p, q) in A[k]
                                        if d_of(q) == i and is_zero(k, s_of(q)))

                        model.addConstr(z[(k, i)] >= sum1 - x[i])
                        model.addConstr(z[(k, i)] >= sum0 + x[i] - 1)


                # Objective
                obj = 0
                for k in range(m):
                    n_k = len(sigma[k])
                    end = (n_k+1, n+1)

                    for (p, q) in A[k]:
                        if q == end:
                            obj += (n_k - s_of(p)) * y[(k, p, q)]
                        else:
                            obj += cost[k][(p, q)] * y[(k, p, q)]

                    obj += quicksum(z[(k, i)] for i in range(1, n+1))
                    obj += w[k]

                model.setObjective(obj, GRB.MINIMIZE)



                model.update()
                model.write(f"lps/hybrid_{n}_{m}_{it}.lp")
                model.optimize()

                center = ""
                # Print results
                if model.status == GRB.OPTIMAL:
                    print("Optimal solution found!")
                    for i in range(1,n+1):
                        print(f"x[{i}] = {x[i].x}")
                        center += str(int(x[i].x))
                    # for k in K:
                    #     for (p, q) in arc.keys():
                    #         if((k, (p, q)) in y): print(f"y[{k},{(p,q)}] = {y[k, (p,q)].x}")
                        #for p in P:
                            #if(start[k, p].x > 0): print(f"start[{k},{p[0]}, {p[1]}] = {start[k, p].x}")
                            #if(finish[k, p].x > 0): print(f"finish[{k},{p[0]}, {p[1]}] = {finish[k, p].x}")
                            #if(c[k,p]. x > 0): print(f"c[{k},{p}] = {c[k,p].x}")
                else:
                    print("No optimal solution found.")
            
                with open("result_hybrid_binary_general_length.txt", "a") as file:
                    # Collect results
                    best_incumbent = model.ObjVal if model.SolCount > 0 else "NFS" #no feasible solution
                    best_bound = model.ObjBound if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else "N/A"
                    solution_time = model.Runtime
                    gap = model.MIPGap
                    nodes = model.NodeCount
                    simplexiters = model.IterCount
                    # Write results in a single row format
                    file.write(f"I_{stringlength}_{stringnumber}_{it}.txt,{seed},{best_incumbent},{best_bound},{solution_time},{gap},{nodes},{simplexiters},{str(center)}\n")
