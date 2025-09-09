import gurobipy as gb
from gurobipy import GRB
import numpy as np
from tqdm import tqdm

def generate_y(n, ray): # genero indici archi costosi
    y_vars = []
    for f_index in range(n+1):
        for s_index in range(n+1):
            if abs(f_index - s_index) >= ray:
                continue
            if f_index < n and abs(f_index + 1 - s_index) < ray:
                y_vars.append(((f_index, s_index), (f_index+1, s_index)))
            if s_index < n and abs(f_index - s_index - 1) < ray:
                y_vars.append(((f_index, s_index), (f_index, s_index+1)))
            if f_index < n and s_index < n:
                y_vars.append(((f_index, s_index), (f_index+1, s_index+1)))
    return y_vars


def generate_ye(word, n, ray): # genero indici archi gratis
    y_vars = []
    for f_index in range(n+1):
        for s_index in range(n+1):
            if abs(f_index - s_index) >= ray:
                continue
            if f_index < n and s_index < n:
                y_vars.append(((f_index, s_index), (f_index+1, s_index+1), word[f_index]))
    return y_vars


def generate_nodes(y): # genero indici nodi
    nodes = set()
    for arc in y:
        if arc[0] not in nodes:
            nodes.add(arc[0])
        if arc[1] not in nodes:
            nodes.add(arc[1])
    return list(nodes)

def find_violated_clique_cuts_from_pool(model, type, all_cuts):
                    cuts = []
                    if all_cuts == 0: #with all_cuts = 0 performs poorly
                        most_violated = -1
                        max_violated_value = -1

                        for idx, clique in enumerate(model._cliquepool):
                            violation = 0
                            for var_element in clique:
                                if(type == 0):
                                    violation += model.cbGetSolution(var_element)
                                else:
                                    violation += model.cbGetNodeRel(var_element)
                                if violation > max_violated_value:
                                    max_violated_value = violation
                                    most_violated = idx
                        if most_violated > -1:
                            cuts.append(gb.quicksum(model._cliquepool[most_violated]) <= 1)
                    else:
                        for idx, clique in enumerate(model._cliquepool):
                            violation = 0
                            for var_element in clique:
                                if(type == 0):
                                    violation += model.cbGetSolution(var_element)
                                else:
                                    violation += model.cbGetNodeRel(var_element)
                            if violation - 1 > 0.001:
                                cuts.append(gb.quicksum(clique) <= 1)
                    return cuts

def find_most_violated_clique_cuts(sol, all_cuts):
                    violated_cuts = []
                    for idx1, word1 in enumerate(stringpool):
                        z_pos_1 = set()
                        for ci, v in enumerate(word1):
                            if v == 0:
                                z_pos_1.add(ci)
                        for idx2, word2 in enumerate(stringpool[idx1 + 1:]):
                            z_pos_2 = set()
                            for ci, v in enumerate(word2):
                                if v == 0:
                                    z_pos_2.add(ci)
                            trueidx2 = idx1 + idx2 + 1
                            most_violated0 = None #for each pair of string, generate the most violated cut.
                            max_violated_value0 = 0 #the overall most violated cut performs poorly
                            most_violated1 = None
                            max_violated_value1 = 0
                            for column in range(string_length): 
                                aggregated_disjunction_0 = []
                                aggregated_disjunction_1 = []
                                for idxsa1, sa1 in enumerate(slant_arcs[idx1]):
                                    if sa1[0][1] == column:
                                        if sa1[0][0] in z_pos_1:
                                            aggregated_disjunction_0.append([idx1, idxsa1])
                                        else:
                                            aggregated_disjunction_1.append([idx1, idxsa1])
                                for idxsa2, sa2 in enumerate(slant_arcs[trueidx2]):
                                    if sa2[0][1] == column:
                                        if sa2[0][0] in z_pos_2:
                                            aggregated_disjunction_1.append([trueidx2, idxsa2])
                                        else:
                                            aggregated_disjunction_0.append([trueidx2, idxsa2])
                                lhs_0_val = 0
                                lhs_1_val = 0
                                for var in aggregated_disjunction_0:
                                    variable = slant_arcs_vars[var[0]][var[1]]
                                    lhs_0_val += sol[variable]
                                for var in aggregated_disjunction_1:
                                    variable = slant_arcs_vars[var[0]][var[1]]
                                    lhs_1_val += sol[variable]
                                if(all_cuts == 1):
                                    if lhs_0_val > 1:
                                        lhs_expr = gb.LinExpr()
                                        for var in aggregated_disjunction_0:
                                            variable = slant_arcs_vars[var[0]][var[1]]
                                            lhs_expr += variable
                                        violated_cuts.append(lhs_expr <= 1)
                                    if lhs_1_val > 1:
                                        lhs_expr = gb.LinExpr()
                                        for var in aggregated_disjunction_1:
                                            variable = slant_arcs_vars[var[0]][var[1]]
                                            lhs_expr += variable
                                        violated_cuts.append(lhs_expr <= 1)
                                else:
                                    if(lhs_0_val - 1 > max_violated_value0):
                                        max_violated_value0 = lhs_0_val - 1
                                        most_violated0 = aggregated_disjunction_0
                                    if(lhs_1_val - 1 > max_violated_value1):
                                        max_violated_value1 = lhs_1_val - 1
                                        most_violated1 = aggregated_disjunction_1
                            if(all_cuts == 0):
                                if(most_violated0 is not None):
                                    lhs_expr = gb.LinExpr()
                                    for var in most_violated0:
                                        variable = slant_arcs_vars[var[0]][var[1]]
                                        lhs_expr += variable
                                    violated_cuts.append(lhs_expr <= 1)
                                if(most_violated1 is not None):
                                    lhs_expr = gb.LinExpr()
                                    for var in most_violated1:
                                        variable = slant_arcs_vars[var[0]][var[1]]
                                        lhs_expr += variable
                                    violated_cuts.append(lhs_expr <= 1)
                        
                    
                    
                    return violated_cuts

def clique_separator_callback(model, where):
                    global use_pool
                    global all_cuts
                    global clique_cuts_added
                    if where == gb.GRB.Callback.MIPSOL:  # Called when an integer-feasible solution is found
                        if(use_pool == 0):
                            # Get current solution values
                            flat_vars = [var for sublist in model._slant_arcs_vars for var in sublist]  # Flatten list of lists
                            sol_vals  = model.cbGetSolution(flat_vars)
                            sol = dict(zip(flat_vars, sol_vals))
                            #print(sol)   
                            violated_cuts_clique = find_most_violated_clique_cuts(sol, all_cuts)
                        else:
                            violated_cuts_clique = find_violated_clique_cuts_from_pool(model, 0, all_cuts)
                        for cut in violated_cuts_clique:
                            model.cbLazy(cut)  # Add the cut dynamically
                            clique_cuts_added.append(cut)
                
                    elif where == GRB.Callback.MIPNODE:
                        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
                        if status == GRB.OPTIMAL:
                            if(only_root_frac_cuts):
                                nodecnt = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
                                if nodecnt > 0:
                                    return  # Not at root node
                        #if model.cbGetRelaxedSolution():
                            if(use_pool == 0):
                                # During MIPNODE, get the node relaxation of the decision variables
                                flat_vars = [var for sublist in model._slant_arcs_vars for var in sublist]  # Flatten list of lists
                                sol_vals  = model.cbGetNodeRel(flat_vars)
                                sol = dict(zip(flat_vars, sol_vals))
                                #print(sol)
                                violated_cuts_clique = find_most_violated_clique_cuts(sol, all_cuts)
                            else:
                                violated_cuts_clique = find_violated_clique_cuts_from_pool(model, 1, all_cuts)
                            for cut in violated_cuts_clique:
                                model.cbCut(cut)  # Add the cut dynamically
                                clique_cuts_added.append(cut)
                        else:
                            return
                    
                    else:
                        return

#num Instances
I = 5

#Set result file
with open("result_projected_rootcuts.txt", "a") as file:
    file.write(f"Instance,Seed,Best Incumbent,BestBound,SolutionTime,GAP,Nodes,SimplexIterations,full_lazy,b&c,only_root_frac_cuts,use_pool,all_cuts,#cliqueineq\n")

print("### STARTING PROJECTED ROOT CUTS ###")

for stringlength in range(10,40,10):
    for stringnumber in range(10,110,10):
        for it in range(I):
            for seed in [2025, 111, 923821]:
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

                string_length = n
                string_number = m
                branch = 1
                stringpool = sigma
                crlb = 0
                upper_bound = n
                obj = 1 #median

                full_lazy = 0 #1 use the clique inequalities as lazy (DO NOT SET full_lazy and branch_and_cut both to 1)
                branch_and_cut = 1 #1 use cutting plane for clique inequalities
                only_root_frac_cuts = 1 #1 to add fractional cuts only at root node, requires branch_and_cut = 1
                use_pool = 1 #1 compute beforehand the clique inequalities, 0 otherwise (need branch_and_cut = 1 to be effective)
                all_cuts = 1 #1 add all violated cuts, 0 only the most violated one

                projected = gb.Model()
                d = projected.addVar(vtype='C', name='d', lb=0, ub=n)
                median_lhs = gb.LinExpr()
                slant_arcs = []
                full_arcs = []
                slant_arcs_vars = []
                full_arcs_vars = []
                y_vars = generate_y(stringlength, stringlength // 2)
                nodes = generate_nodes(y_vars)
                arcs, costs = gb.multidict({arc: 1 for arc in y_vars})
                for idx, word in tqdm(enumerate(stringpool)):
                    #print("String", idx, ":", word)
                    diagonal_cost_y = generate_ye(word, string_length, string_length // 2)
                    ye, ye_costs = gb.multidict({arc: 0 for arc in diagonal_cost_y})
                    slant_arcs.append(ye)
                    full_arcs.append(arcs)
                    
                    flow = projected.addVars(arcs, name='flow' + str(idx), vtype='I')
                    
                    zero_cost_flow = projected.addVars(ye, name='ye' + str(idx), vtype='I')
                    slant_arcs_vars.append(list(zero_cost_flow.values()))
                    full_arcs_vars.append(list(flow.values()))

                    source = gb.LinExpr()
                    for arc in arcs:
                        if arc[0] == (0, 0):
                            source += flow[arc]
                    for arc in diagonal_cost_y:
                        if arc[0] == (0, 0):
                            source += zero_cost_flow[arc]

                    projected.addConstr(source == 1, name='source' + str(idx))
                    for i, j in nodes:
                        if (i, j) == (0, 0) or (i, j) == (string_length, string_length):
                            continue
                        outflow = gb.LinExpr()
                        inflow = gb.LinExpr()
                        for arc in arcs:
                            if arc[0] == (i, j):
                                outflow += flow[arc]
                            if arc[1] == (i, j):
                                inflow += flow[arc]
                        for arc in diagonal_cost_y:
                            if arc[0] == (i, j):
                                outflow += zero_cost_flow[arc]
                            if arc[1] == (i, j):
                                inflow += zero_cost_flow[arc]
                        projected.addConstr(outflow == inflow, name='conserv' + str(idx) + '-' + str(i) + str(j))
                    if obj:
                        ### Median problem ###
                        median_lhs += flow.prod(costs) + zero_cost_flow.prod(ye_costs)
                    else:
                        ### Radi\ problem ###
                        projected.addConstr(flow.prod(costs) + zero_cost_flow.prod(ye_costs) <= d, name="distance" + str(idx))
                    projected.update()
                    # for idx1, free_arc_1 in enumerate(zero_cost_flow.values()):
                    #     for idx2, free_arc_var_2 in enumerate(zero_cost_flow.values()):
                    #         if idx1 < idx2 and free_arc_1[0][1] == free_arc_var_2[0][1]: # if same column
                    #             projected.addConstr(zero_cost_flow[idx1] + zero_cost_flow[idx2] <= 1,
                    #                                 name="Edge_ineq_"+str(idx)+"_col_"+str(free_arc_1[0][1])+"_"+str(idx1)+"_"+str(idx2))
                    #         if idx1 < idx2 and free_arc_1[0][0] == free_arc_var_2[0][0]: # if same row
                    #             projected.addConstr(zero_cost_flow[idx1] + zero_cost_flow[idx2] <= 1,
                    #                                 name="Edge_ineq_"+str(idx)+"_row_"+str(free_arc_1[0][1])+"_"+str(idx1)+"_"+str(idx2))

                    #print('Work done for string', idx)
                if not obj:
                    projected.setObjective(d)
                else:
                    projected.setObjective(median_lhs)

                # edge_ineq_pool = []
                # print('\nAdding edge inequalities')
                # for i in range(string_number):
                #     for j in range(i + 1, string_number):
                #         for idx1, sav1 in enumerate(slant_arcs[i]):
                #                 for idx2, sav2 in enumerate(slant_arcs[j]):
                #                     if sav1[2] != sav2[2] and sav1[0][1] == sav2[0][1]: # different character in same column
                #                         #edge_ineq_pool.append([slant_arcs_vars[i][idx1],slant_arcs_vars[j][idx2]])
                #                         projected.addConstr(slant_arcs_vars[i][idx1] + slant_arcs_vars[j][idx2] <= 1,
                #                                             name="Edge_ineq_"+str(i)+"_"+str(j)+"_"+
                #                                                  str(sav1[0][0])+str(sav1[0][1])+"_"+str(sav2[0][0])+str(sav2[0][1]))
                                
                # if inequals == 2:
                #     print('\nAdding surrogate edge inequalities')
                #     for i in range(string_number):
                #         for j in range(i + 1, string_number):
                #             for idx1, sav1 in enumerate(slant_arcs[i]):
                #                 lhs = gb.LinExpr() 
                #                 for idx2, sav2 in enumerate(slant_arcs[j]):
                #                     if sav1[2] != sav2[2] and sav1[0][1] == sav2[0][1]: # different character in same column
                #                         lhs += slant_arcs_vars[j][idx2]
                #                 projected.addConstr(slant_arcs_vars[i][idx1]*lhs.size() + lhs <= lhs.size(), name="Edge_ineq_"+str(i)+"_"+str(j)+"_"+
                #                                                 str(sav1[0][0])+str(sav1[0][1]))                

                # if inequals == 1:
                #     print('\nAdding nodal inequalities')
                #     for k in range(string_length):
                #         for c in ['0', '1']:
                #             for idx1, word1 in enumerate(stringpool):
                #                 fbi = gb.LinExpr()
                #                 nbar = gb.LinExpr()
                #                 nodal_set = set()
                #                 for idxsa1, sa1 in enumerate(slant_arcs[idx1]):
                #                     if sa1[0][1] == k: # check if column is k
                #                         if sa1[2] == c: # check if slant arc sa1 imposes character c (that is, check if k-th character of word1 is c
                #                             fbi += slant_arcs_vars[idx1][idxsa1] # add to fbi the variable of the slant arc that imposes character c
                #                         else:
                #                             nbar += slant_arcs_vars[idx1][idxsa1]
                #                             nodal_set.add(idx1) # add to count nodal rank
                #                 if fbi.size() == 0:
                #                     continue
                #                 for idx2, word2 in enumerate(stringpool):
                #                     if idx1 == idx2:
                #                         continue
                #                     for idxsa2, sa2 in enumerate(slant_arcs[idx2]):
                #                         if sa2[0][1] == k:  # check if column is k
                #                             if sa2[2] != c:  # check if slant arc sa1 imposes character different from c (that is, check if k-th character of word1 is not c
                #                                 nbar += slant_arcs_vars[idx2][idxsa2]
                #                                 nodal_set.add(idx2)  # add to count nodal rank
                #                 # add nodal
                #                 constr = projected.addConstr(nbar + len(nodal_set) * fbi <= len(nodal_set),
                #                                             name='nodal_s' + str(idx1) + '_k' + str(k) + '_c' + c)
                #                 #constr.setAttr('Lazy', 1)

                #if inequals == 0:
                print('\nAdding clique inequalities')
                if(use_pool):
                    clique_cuts_added = []
                    cliquepool = []
                    for idx1, word1 in enumerate(stringpool):
                        z_pos_1 = set()
                        for ci, v in enumerate(word1):
                            if v == 0:
                                z_pos_1.add(ci)
                        for idx2, word2 in enumerate(stringpool[idx1 + 1:]):
                            z_pos_2 = set()
                            for ci, v in enumerate(word2):
                                if v == 0:
                                    z_pos_2.add(ci)
                            trueidx2 = idx1 + idx2 + 1
                            for column in range(string_length):
                                aggregated_disjunction_0 = []
                                aggregated_disjunction_1 = []
                                for idxsa1, sa1 in enumerate(slant_arcs[idx1]):
                                    if sa1[0][1] == column:
                                        if sa1[0][0] in z_pos_1:
                                            aggregated_disjunction_0.append([idx1, idxsa1])
                                        else:
                                            aggregated_disjunction_1.append([idx1, idxsa1])
                                for idxsa2, sa2 in enumerate(slant_arcs[trueidx2]):
                                    # col disjunctions
                                    if sa2[0][1] == column:
                                        if sa2[0][0] in z_pos_2:
                                            aggregated_disjunction_1.append([trueidx2, idxsa2])
                                        else:
                                            aggregated_disjunction_0.append([trueidx2, idxsa2])
                                lhs_0 = gb.LinExpr()
                                lhs_0_vector = [] 
                                lhs_1 = gb.LinExpr()
                                lhs_1_vector = []
                                for var in aggregated_disjunction_0:
                                    lhs_0 += slant_arcs_vars[var[0]][var[1]]
                                    #slant_arcs_vars[var[0]][var[1]].BranchPriority += 1
                                    lhs_0_vector.append(slant_arcs_vars[var[0]][var[1]])
                                for var in aggregated_disjunction_1:
                                    lhs_1 += slant_arcs_vars[var[0]][var[1]]
                                    #slant_arcs_vars[var[0]][var[1]].BranchPriority += 1
                                    lhs_1_vector.append(slant_arcs_vars[var[0]][var[1]])
                                if lhs_0.size():
                                    # constr = projected.addConstr(lhs_0 <= 1,
                                    #                             name="Clique_disj_s" + str(idx1) + "_s" + str(trueidx2) + "_k_" + str(
                                    #                                 column) + "_c_0")
                                    #constr.setAttr('Lazy', 3)
                                    cliquepool.append(lhs_0_vector)
                                if lhs_1.size():
                                    # constr = projected.addConstr(lhs_1 <= 1,
                                    #                             name="Clique_disj_s" + str(idx1) + "_s" + str(trueidx2) + "_k_" + str(
                                    #                                 column) + "_c_1")

                                    #constr.setAttr('Lazy', 3)
                                    cliquepool.append(lhs_1_vector)

                projected._slant_arcs = slant_arcs
                projected._slant_arcs_vars = slant_arcs_vars
                projected.update()
                
                
                #projected.setParam('NetworkAlg', 1)
                projected.Params.MIPGap = 1e-4
                projected._cliquepool = cliquepool
                #projected._edge_ineq_pool = edge_ineq_pool
                projected.params.TimeLimit = 600
                projected.params.Seed = seed
                projected.params.LazyConstraints = 1
                projected.optimize(clique_separator_callback)
                #projected.optimize()
                with open("result_projected_rootcuts.txt", "a") as file:
                        # Collect results
                        best_incumbent = projected.ObjVal if projected.SolCount > 0 else "NFS" #no feasible solution
                        best_bound = projected.ObjBound if projected.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else "N/A"
                        solution_time = projected.Runtime
                        gap = projected.MIPGap
                        nodes = projected.NodeCount
                        simplexiters = projected.IterCount
                        # Write results in a single row format
                        file.write(f"I_{stringlength}_{stringnumber}_{it}.txt,{seed},{best_incumbent},{best_bound},{solution_time},{gap},{nodes},{simplexiters},{full_lazy},{branch_and_cut},{only_root_frac_cuts},{use_pool},{all_cuts},{len(clique_cuts_added)}\n")