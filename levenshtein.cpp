//
// Created by andrea on 15/03/24.
//

#include "iostream"
#include "ilcplex/ilocplex.h"
#include <chrono>
#include <random>
#include <vector>
#include <map>
#include <set>
#include "algorithm"
#include "string"



using namespace std;
typedef IloArray<IloIntVarArray> IntVar2D;
typedef IloArray<IntVar2D> IntVar3D;
typedef IloArray<IntVar3D> IntVar4D;

typedef vector<bool> vb;
typedef vector<vb> vvb;
typedef unsigned int ui;
typedef vector<pair<ui, ui>> graph_nodes;
typedef pair<pair<ui, ui>, pair<ui, ui>> costly_arcs;
typedef pair<int,pair<int,int>> ti;
typedef vector<pair<pair<ui, ui>, pair<ui, ui>>> graph_arcs;
typedef pair<pair<pair<ui, ui>, pair<ui, ui>>, bool> free_arcs;
typedef vector<pair<pair<pair<ui, ui>, pair<ui, ui>>, bool>> graph_free_arcs;
template <typename OutputIt, typename Engine = std::mt19937>
void generate(OutputIt first, OutputIt last)
{
    static Engine engine;
    bernoulli_distribution distribution;

    while (first != last)
    {
        *first++ = distribution(engine);
    }
}

ui levenshtein (const vb &s1, const vb &s2){
    if(s1.size() > s2.size()) return levenshtein(s2, s1);
    if(s2.size() == 0) return s1.size();

    vector<ui> previous_row;
    previous_row.resize(s2.size() + 1);
    for(ui i = 0; i < previous_row.size(); i++){
        previous_row[i] = i;
    }
    for(ui i = 0; i < s1.size(); i++){
        vector<ui> current_row;
        current_row.push_back(i+1);
        for(ui j = 0; j < s2.size(); j++) {
            current_row.push_back(min(min(previous_row[j+1] + 1, current_row[j] + 1), previous_row[j] +
                                                                                      (s1[i] != s2[j] ? 1 : 0)));
        }
        previous_row = current_row;
    }

    return *previous_row.rbegin();
}

ui hamming( const vb &s1, const vb & s2){
    ui distance = 0;
    for(size_t i = 0; i <= s1.size(); i ++){
        distance += s1[i] != s2[i];
    }

    return distance;
}

void generate_y_and_nodes(ui n, ui ray, graph_arcs& y_vars_idx, graph_nodes& nodes){
    for(ui f_index = 0; f_index < n+1; f_index++){
        for(ui s_index = 0; s_index < n+1; s_index++){
            if(max(f_index,s_index) - min(f_index, s_index) >= ray) continue;
            if(f_index < n && max(f_index + 1, s_index) - min(f_index + 1, s_index) < ray) {
                y_vars_idx.push_back(make_pair(
                        make_pair(f_index, s_index), // primo nodo (coda) dell'arco
                        make_pair(f_index + 1, s_index) // secondo nodo (testa) dell'arco
                ));

                if (find(nodes.begin(), nodes.end(), make_pair(f_index, s_index)) == nodes.end())
                    nodes.push_back(make_pair(f_index, s_index));
                if (find(nodes.begin(), nodes.end(), make_pair(f_index + 1, s_index)) == nodes.end())
                    nodes.push_back(make_pair(f_index + 1, s_index));
            }
            if(s_index < n && max(f_index, s_index + 1) - min(f_index, s_index + 1) < ray) {
                y_vars_idx.push_back(make_pair(
                        make_pair(f_index, s_index), // primo nodo (coda) dell'arco
                        make_pair(f_index, s_index + 1) // secondo nodo (testa) dell'arco
                ));
                if (find(nodes.begin(), nodes.end(), make_pair(f_index, s_index)) == nodes.end())
                    nodes.push_back(make_pair(f_index, s_index));
                if (find(nodes.begin(), nodes.end(), make_pair(f_index, s_index + 1)) == nodes.end())
                    nodes.push_back(make_pair(f_index, s_index + 1));
            }
            if(f_index < n && s_index < n) {
                y_vars_idx.push_back(make_pair(
                        make_pair(f_index, s_index), // primo nodo (coda) dell'arco
                        make_pair(f_index + 1, s_index + 1) // secondo nodo (testa) dell'arco
                ));
                if (find(nodes.begin(), nodes.end(), make_pair(f_index, s_index)) == nodes.end())
                    nodes.push_back(make_pair(f_index, s_index));
                if (find(nodes.begin(), nodes.end(), make_pair(f_index + 1, s_index + 1)) == nodes.end())
                    nodes.push_back(make_pair(f_index + 1, s_index + 1));
            }
        }
    }
}

void generate_ye(ui n, ui ray, const vector<bool>& word, graph_free_arcs& y_vars_idx){
    for(ui f_index = 0; f_index < n+1; f_index++){
        for(ui s_index = 0; s_index < n+1; s_index++) {
            if(max(f_index,s_index) - min(f_index, s_index) >= ray) continue;
            if(f_index < n && s_index < n)
                y_vars_idx.push_back(make_pair(
                        make_pair(
                                make_pair(f_index,s_index), // primo nodo (coda) dell'arco
                                make_pair(f_index + 1, s_index + 1) // secondo nodo (testa) dell'arco
                        ),
                        word[f_index]) // carattere
                );
        }
    }
}


double JB_bound(const vector<vector<double>> & distances , ui n, ui m, const vector<int>& q){
    try
    {
        IloEnv env;
        IloModel Model(env);

        IloNumVarArray X(env, m);
        for(ui i = 0; i < m; i++) X[i] = IloNumVar(env, 0, n - q[i], "x");
        IloNumVar Z(env, 0, IloInfinity, "z");
        //IloExpr minsum(env);
        //minsum.clear();
        for(ui i = 0; i < m; i++){
            Model.add(Z >= X[i]);
            //minsum += X[i];
            for(ui j = 0; j < m; j++){
                if(i <= j) continue;
                Model.add(X[min(i,j)] + X[max(i,j)] >= distances[min(i,j)][max(i,j)]);
                Model.add(X[min(i,j)] + distances[min(i,j)][max(i,j)] >= X[max(i,j)]);
                Model.add(distances[min(i,j)][max(i,j)] + X[max(i,j)] >= X[min(i,j)]);
            }
        }

        Model.add(IloMinimize(env, Z));
        //Model.add(IloMinimize(env, minsum));

        IloCplex cplex(Model);
        cplex.setOut(env.getNullStream());
        //cplex.exportModel("bunke_lb.lp");
        if (!cplex.solve()) {
            env.error() << "Failed to optimize the SubProblem!!!" << endl;
            cout << "Status " << cplex.getStatus() << "\n";
            throw(-1);
        }

        double obj = cplex.getObjValue();

        // cout << "\n\nThe objective value is: " << obj << endl;
        return obj;
    }
    catch ( IloException& e )
    {
        std::cout << e << std::endl;
        e.end();
    }

}


double JB_bound_diameter(const vector<vector<double>> & distances , ui n, ui m, const vector<int>& q){
    try
    {
        IloEnv env;
        IloModel Model(env);

        IloNumVarArray X(env, m);
        for(ui i = 0; i < m; i++) X[i] = IloNumVar(env, 0, n - q[i], "x");
        IloNumVar Z(env, 0, IloInfinity, "z");

        for(ui i = 0; i < m; i++){
            Model.add(Z >= X[i]);
            for(ui j = 0; j < m; j++){
                if(i <= j) continue;
                Model.add(X[min(i,j)] + X[max(i,j)] >= distances[min(i,j)][max(i,j)]);
                Model.add(X[min(i,j)] + distances[min(i,j)][max(i,j)] >= X[max(i,j)]);
                Model.add(distances[min(i,j)][max(i,j)] + X[max(i,j)] >= X[min(i,j)]);
            }
        }

        Model.add(IloMinimize(env, Z));

        IloCplex cplex(Model);
        cplex.setOut(env.getNullStream());
        //cplex.exportModel("bunke_lb.lp");
        if (!cplex.solve()) {
            env.error() << "Failed to optimize the SubProblem!!!" << endl;
            cout << "Status " << cplex.getStatus() << "\n";
            throw(-1);
        }

        double obj = cplex.getObjValue();

        // cout << "\n\nThe objective value is: " << obj << endl;
        return obj;
    }
    catch ( IloException& e )
    {
        std::cout << e << std::endl;
        e.end();
    }

}

//ILOBRANCHCALLBACK4(BunkeBCB, IloNumVarArray, x, IloIntVar, d, const vector<vector<double>> &, distances, const vvb &, stringpool) {
//    if ( getBranchType() != BranchOnVariable )
//        return;
//
//    long k = getNbranches();
//    IloNumVarArray v;
//    v = IloNumVarArray(getEnv());
//    IloNumArray b;
//    b = IloNumArray(getEnv());
//    IloCplex::BranchDirectionArray dirs;
//    dirs = IloCplex::BranchDirectionArray(getEnv());
//    for(long i = 0; i < k; i++){
//        getBranch(v, b, dirs, i);
//        if(v[0].getId() == d.getId()) return;
//        if(dirs[0] == IloCplex::BranchDirection::BranchUp){
//            vector<int> q;
//            q.resize(stringpool.size(), 0);
//            // TODO: fare con AND bit a bit
//            for(int i = 0; i < x.getSize(); i ++){
//                for(int j = 0; j < stringpool.size(); j++) {
//                    q[j] += getValue(x[i]) == stringpool[j][i] ? 1 : 0;
//                }
//            }
//
//            double lb = JB_bound(distances, x.getSize(), distances.size(), q);
//            v.add(d);
//            b.add(std::ceil(lb));
//            dirs.add(IloCplex::BranchDirection::BranchUp);
//            makeBranch(v, b, dirs, getObjValue());
//        }
//        else{
//            makeBranch(v, b, dirs, getObjValue());
//        }
//    }
//
//}


ILOBRANCHCALLBACK2(BunkeBCB, std::vector<std::vector<IloNumVar>>, x0inc, std::vector<std::vector<IloNumVar>>, x1inc) {
    if ( getBranchType() != BranchOnVariable )
        return;

    long k = getNbranches();
    IloNumVarArray v;
    v = IloNumVarArray(getEnv());
    IloNumArray b;
    b = IloNumArray(getEnv());
    IloCplex::BranchDirectionArray dirs;
    dirs = IloCplex::BranchDirectionArray(getEnv());

    for(long i = 0; i < k; i++){
        getBranch(v, b, dirs, i);
        if(v[0].getId() > x0inc.size()) return;
        if(dirs[0] == IloCplex::BranchDirection::BranchUp){
            for(auto &y: x1inc[v[0].getId()]){
                v.add(y);
                b.add(0.0);
                dirs.add(IloCplex::BranchDirection::BranchDown);
            }
            makeBranch(v, b, dirs, getObjValue());
        }
        else{
            for(auto &y: x0inc[v[0].getId()]){
                v.add(y);
                b.add(0.0);
                dirs.add(IloCplex::BranchDirection::BranchDown);
            }
            makeBranch(v, b, dirs, getObjValue());
        }
        v.clear();
        b.clear();
        dirs.clear();
    }

}

ILOBRANCHCALLBACK1(BunkePrune, IloNum, lb){
//
//    if(floor(getObjValue()) < lb){
//        cout << "+++++++++++++++++ PRUNO +++++++++++++++\n";
//        prune();
//    }
};

ILOBRANCHCALLBACK5(HammingPrune, IloNumVarArray, x, const vector<vector<double>> &, distances, const vvb &, stringpool, IloNum, mhub, const vb &, mhs){

    vector<int> q;
    q.resize(stringpool.size(), 0);
    // TODO: fare con AND bit a bit
    vector<double> current_x;
    current_x.resize(x.getSize(), -1.0);
    for(int i = 0; i < x.getSize(); i ++){
        if(getLB(x[i]) == getUB(x[i])) {
            current_x[i] = getValue(x[i]);
            for (int j = 0; j < stringpool.size(); j++) {
                q[j] += getValue(x[i]) == stringpool[j][i] ? 1 : 0;
            }
        }
    }
    IloNum local_hd = 0.0;
    for(const auto & s: stringpool){
        int distance_from_median_hamming = 0;
        for(int i = 0; i < s.size(); i++){
            distance_from_median_hamming += current_x[i] != -1 ? current_x[i] != s[i] : mhs[i] != s[i];
        }
        local_hd += distance_from_median_hamming;
    }

    double lb = JB_bound(distances, x.getSize(), distances.size(), q);
    if(max(ceil(lb), ceil(getObjValue())) >= min(getIncumbentObjValue(), local_hd)){
        prune();
    }
};

ILOUSERCUTCALLBACK4(BunkeCB, IloNumVarArray, x, IloIntVar, d, const vector<vector<double>> &, distances, const vvb &, stringpool){

    vector<int> q;
    q.resize(stringpool.size(), 0);
    // TODO: fare con AND bit a bit
    for(int i = 0; i < x.getSize(); i ++){
        for(int j = 0; j < stringpool.size(); j++) {
            q[j] += getValue(x[i]) == stringpool[j][i] ? 1 : 0;
        }
    }

    double lb = JB_bound(distances, x.getSize(), distances.size(), q);

    if(getObjValue() < lb - 0.0001) {
        cout << "Current d value " << getValue(d) << " | Bunke dynamic LB " << lb << " | LocalBound " << getObjValue()
             << " | BestBound " << getBestObjValue() << "\n";
        addLocal(d >= lb);
    }
};


pair<double, double> extended_formulation_no_callback(ui n, ui m, const vvb & stringpool, ui lower_bound, ui upper_bound, const vector<vector<double>> & distances){
    try
    {
        IloEnv env;
        IloModel Model(env);

        IloNumVarArray x(env, n);
        for(ui i = 0; i < n; i++){
            string name = "x" + to_string(i);
            x[i] = IloIntVar(env, 0, 1, name.c_str());
        }
        //IloIntVar d(env, lower_bound, upper_bound, "d");
        IloExpr minsum(env);
        //minsum.clear();
        graph_nodes nodes;
        graph_arcs arcs;


        generate_y_and_nodes(n, n/2, arcs, nodes);

        for(ui index = 0; index < m; index++){
            graph_free_arcs diagonal_free_arcs;
            generate_ye(n, n/2, stringpool[index], diagonal_free_arcs);

            for(const auto & c: stringpool[index]) cout << c;

            std::map<costly_arcs, IloNumVar> y;
            std::map<free_arcs, IloNumVar> ye;
            for(const auto & arc: arcs){
                y.emplace(arc, IloIntVar(env, 0, 1));
                string name = "y_" +std::to_string(arc.first.first)
                              +"_"+std::to_string(arc.first.second)
                              +"_"+std::to_string(arc.second.first)
                              +"_"+std::to_string(arc.second.second)
                              +"_"+std::to_string(index);
                y[arc].setName(name.c_str());
            }
            for(const auto & arc: diagonal_free_arcs) {
                ye.emplace(arc, IloIntVar(env, 0, 1));
                string name = "ye_"+std::to_string(index)
                              +"_"+std::to_string(arc.first.first.first)
                              +"_"+std::to_string(arc.first.first.second)
                              +"_"+std::to_string(arc.first.second.first)
                              +"_"+std::to_string(arc.first.second.second)
                              +"_"+std::to_string(arc.second);
                ye[arc].setName(name.c_str());
            }
            IloExpr source(env);
            IloExpr distance(env);

            for(const auto & arc: diagonal_free_arcs){

                if(arc.second == 0)
                    Model.add(ye[arc] <= 1 - x[arc.first.first.second]);
                else
                    Model.add(ye[arc] <= x[arc.first.first.second]);

                if(arc.first.first.first == 0 && arc.first.first.second == 0)
                    source += ye[arc];
            }
            for(const auto & arc: arcs){
                distance += y[arc];
                if(arc.first.first == 0 && arc.first.second == 0)
                    source += y[arc];
            }
            Model.add(source == 1);
            //Model.add(distance <= d);

            minsum += distance;
            for(const auto & node: nodes){
                if((node.first == 0 && node.second == 0) || (node.first == n && node.second == n))
                    continue;
                IloExpr outflow(env);
                IloExpr inflow(env);
                for(const auto & arc: arcs){
                    if(arc.first.first == node.first && arc.first.second == node.second){
                        outflow += y[arc];
                    }
                    if(arc.second.first == node.first && arc.second.second == node.second){
                        inflow += y[arc];
                    }
                }
                for(const auto & arc: diagonal_free_arcs){
                    if(arc.first.first.first == node.first && arc.first.first.second == node.second){
                        outflow += ye[arc];
                    }
                    if(arc.first.second.first == node.first && arc.first.second.second == node.second){
                        inflow += ye[arc];
                    }
                }
                Model.add(outflow == inflow);
            }
        }
        // if(lower_bound > 0)
        //     Model.add(d >= static_cast<double>(lower_bound));
        // else cout << "\n@@@@@@@@@@@@@@ NO LOWER BOUND @@@@@@@@@@@@@\n";
        // if(upper_bound != n)
        //     Model.add(d <= static_cast<double>(upper_bound));
        // else cout << "\n@@@@@@@@@@@@@@ NO UPPER BOUND @@@@@@@@@@@@@\n";
        //Model.add(IloMinimize(env, d));
        Model.add(IloMinimize(env, minsum));

        IloCplex cplex(Model);

//        for(size_t i = 0; i < n; i++){
//            cplex.setPriority(x[i], 10);
//        }
        cplex.exportModel("levenshtein_extended.lp");
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        cplex.setParam(IloCplex::Param::MIP::Strategy::Search, 1);
        cplex.setParam(IloCplex::Param::TimeLimit, 3600);

        if (!cplex.solve()) {
            cplex.exportModel("levensh.lp");
            env.error() << "Failed to optimize the Master Problem!!!" << endl;
            throw(-1);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        double obj = cplex.getObjValue();
        cout << "\n\nThe objective value is: " << obj << endl;

        return make_pair(obj, std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
    }
    catch ( IloException& e )
    {
        std::cout << e << std::endl;
        e.end();
    }
}


pair<double, double> pop_extended_formulation(ui n, ui m, const vvb & stringpool, int lower_bound, ui upper_bound, const vector<vector<double>> & distances){
    try
    {
        IloEnv env;
        IloModel Model(env);

        IloNumVarArray x(env, n);
        for(ui i = 0; i < n; i++){
            string name = "x" + to_string(i);
            x[i] = IloIntVar(env, 0, 1, name.c_str());
        }
        IloIntVar d(env, lower_bound, upper_bound, "d");
        //IloExpr minsum(env);
        //minsum.clear();
        graph_nodes nodes;
        graph_arcs arcs;

        generate_y_and_nodes(n, n/2, arcs, nodes);

        std::vector<std::vector<IloIntVar>> l;
        l.resize(m);
        for(int i = 0; i < m; i++){
            l[i].resize(n+1);
            for(int j = lower_bound+1; j < n + 1; j++){
                string name = "l_"+std::to_string(i)
                              +"_"+std::to_string(j);

                l[i][j] = IloIntVar(env, 0, 1, name.c_str());
            }
            for(int j = 1; j < lower_bound + 1; j++){
                string name = "l_"+std::to_string(i)
                              +"_"+std::to_string(j);

                l[i][j] = IloIntVar(env, 0, 0, name.c_str());
            }
            string name = "l_"+std::to_string(i)
                          +"_0";
            l[i][0] = IloIntVar(env, 0, 0, name.c_str());

        }

        for(int i = 0; i < m; i ++){
            for(int k = 0; k < n; k++){
                Model.add(l[i][k] <= l[i][k+1]);
                if(i > 0){
                    Model.add(l[0][k] <= l[i][k]);
                }
            }
            if(i>0){
                Model.add(l[0][n] <= l[i][n]);
            }
        }

        std::vector<std::vector<IloIntVar>> g;
        g.resize(m);
        for(int i = 0; i < m; i++){
            g[i].resize(n+1);
            for(int j = lower_bound; j < n; j++){
                string name = "g_"+std::to_string(i)
                              +"_"+std::to_string(j);

                g[i][j] = IloIntVar(env, 0, 1, name.c_str());
            }
            for(int j = 0; j < lower_bound; j++){
                string name = "g_"+std::to_string(i)
                              +"_"+std::to_string(j);

                g[i][j] = IloIntVar(env, 0, 1, name.c_str());
            }
            string name = "g_"+std::to_string(i)
                          +"_"+std::to_string(n);
            g[i][n] = IloIntVar(env, 0, 0, name.c_str());
        }

        for(int i = 0; i < m; i ++){
            for(int k = 1; k < n; k++){
                Model.add(g[i][k] >= g[i][k+1]);
                if(i > 0){
                    Model.add(g[0][k] <= g[i][k]);
                }
            }
            if(i>0){
                Model.add(g[0][0] <= g[i][0]);
            }
        }

        for(int i = 0; i < m; i++){
            for(int j=0; j < n; j++){
                Model.add(g[i][j] + l[i][j+1] == 1);
            }
        }

        for(ui index = 0; index < m; index++){
            graph_free_arcs diagonal_free_arcs;
            generate_ye(n, n/2, stringpool[index], diagonal_free_arcs);

            for(const auto & c: stringpool[index]) cout << c;

            std::map<costly_arcs, IloNumVar> y;
            std::map<free_arcs, IloNumVar> ye;
            for(const auto & arc: arcs){
                y.emplace(arc, IloIntVar(env, 0, 1));
                string name = "y_" +std::to_string(arc.first.first)
                              +"_"+std::to_string(arc.first.second)
                              +"_"+std::to_string(arc.second.first)
                              +"_"+std::to_string(arc.second.second)
                              +"_"+std::to_string(index);
                y[arc].setName(name.c_str());
            }
            for(const auto & arc: diagonal_free_arcs) {
                ye.emplace(arc, IloIntVar(env, 0, 1));
                string name = "ye_"+std::to_string(index)
                              +"_"+std::to_string(arc.first.first.first)
                              +"_"+std::to_string(arc.first.first.second)
                              +"_"+std::to_string(arc.first.second.first)
                              +"_"+std::to_string(arc.first.second.second)
                              +"_"+std::to_string(arc.second);
                ye[arc].setName(name.c_str());
            }
            IloExpr source(env);
            IloExpr distance(env);
            IloExpr suml(env);
            IloExpr sumg(env);
            for(int k = 1; k < n + 1; k++){
                suml += l[index][k];
            }
            for(int k = 0; k < n; k++){
                sumg += g[index][k];
            }

            for(const auto & arc: diagonal_free_arcs){

                if(arc.second == 0)
                    Model.add(ye[arc] <= 1 - x[arc.first.first.second]);
                else
                    Model.add(ye[arc] <= x[arc.first.first.second]);

                if(arc.first.first.first == 0 && arc.first.first.second == 0)
                    source += ye[arc];
            }
            for(const auto & arc: arcs){
                distance += y[arc];
                if(arc.first.first == 0 && arc.first.second == 0)
                    source += y[arc];
            }
            Model.add(source == 1);
            Model.add(n - suml >= distance);
            Model.add(sumg <= distance);

            //minsum += distance;
            for(const auto & node: nodes){
                if((node.first == 0 && node.second == 0) || (node.first == n && node.second == n))
                    continue;
                IloExpr outflow(env);
                IloExpr inflow(env);
                for(const auto & arc: arcs){
                    if(arc.first.first == node.first && arc.first.second == node.second){
                        outflow += y[arc];
                    }
                    if(arc.second.first == node.first && arc.second.second == node.second){
                        inflow += y[arc];
                    }
                }
                for(const auto & arc: diagonal_free_arcs){
                    if(arc.first.first.first == node.first && arc.first.first.second == node.second){
                        outflow += ye[arc];
                    }
                    if(arc.first.second.first == node.first && arc.first.second.second == node.second){
                        inflow += ye[arc];
                    }
                }
                Model.add(outflow == inflow);
            }
        }
        IloExpr objfun(env);
        for(int k=0; k < n; k++){
            objfun += g[0][k];
        }
        Model.add(IloMinimize(env, objfun));
        //Model.add(IloMinimize(env, minsum));

        IloCplex cplex(Model);
        for(size_t i = 0; i < m; i++){
            for(size_t j = 0; j < n; j++){
                cplex.setPriority(g[i][j], 100);
            }
        }
        cplex.exportModel("levenshtein_extended_pop.lp");
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        cplex.setParam(IloCplex::Param::MIP::Strategy::Search, 1);

        if (!cplex.solve()) {
            cplex.exportModel("levensh.lp");
            env.error() << "Failed to optimize the Master Problem!!!" << endl;
            throw(-1);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        double obj = cplex.getObjValue();
        cout << "\n\nThe objective value is: " << obj << endl;

        return make_pair(obj, std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
    }
    catch ( IloException& e )
    {
        std::cout << e << std::endl;
        e.end();
    }
}

pair<double, double> assignment_extended_formulation(ui n, ui m, const vvb & stringpool, int lower_bound, ui upper_bound, const vector<vector<double>> & distances){
    try
    {
        IloEnv env;
        IloModel Model(env);

        IloNumVarArray x(env, n);
        for(ui i = 0; i < n; i++){
            string name = "x" + to_string(i);
            x[i] = IloIntVar(env, 0, 1, name.c_str());
        }
        //IloIntVar d(env, lower_bound, upper_bound, "d");
        IloExpr minsum(env);
        minsum.clear();
        graph_nodes nodes;
        graph_arcs arcs;

        generate_y_and_nodes(n, n/2, arcs, nodes);

        std::vector<std::vector<IloIntVar>> l;
        l.resize(m);
        for(int i = 0; i < m; i++){
            l[i].resize(n+1);
            for(int j = 0; j < n+1; j++){
                l[i][j] = IloIntVar(env, 0 , 1);
            }
            IloExpr assign(env);
            for(int j= 0; j < n+1; j++){
                assign += l[i][j];
            }
            Model.add(assign == 1);
        }

        for(ui index = 0; index < m; index++){
            graph_free_arcs diagonal_free_arcs;
            generate_ye(n, n/2, stringpool[index], diagonal_free_arcs);

            for(const auto & c: stringpool[index]) cout << c;

            std::map<costly_arcs, IloNumVar> y;
            std::map<free_arcs, IloNumVar> ye;
            for(const auto & arc: arcs){
                y.emplace(arc, IloIntVar(env, 0, 1));
                string name = "y_" +std::to_string(arc.first.first)
                              +"_"+std::to_string(arc.first.second)
                              +"_"+std::to_string(arc.second.first)
                              +"_"+std::to_string(arc.second.second)
                              +"_"+std::to_string(index);
                y[arc].setName(name.c_str());
            }
            for(const auto & arc: diagonal_free_arcs) {
                ye.emplace(arc, IloIntVar(env, 0, 1));
                string name = "ye_"+std::to_string(index)
                              +"_"+std::to_string(arc.first.first.first)
                              +"_"+std::to_string(arc.first.first.second)
                              +"_"+std::to_string(arc.first.second.first)
                              +"_"+std::to_string(arc.first.second.second)
                              +"_"+std::to_string(arc.second);
                ye[arc].setName(name.c_str());
            }
            IloExpr source(env);
            IloExpr distance(env);
            IloExpr suml(env);
            for(int k = 0; k < n+1; k++){
                suml += k*l[index][k];
            }
            for(const auto & arc: diagonal_free_arcs){

                if(arc.second == 0)
                    Model.add(ye[arc] <= 1 - x[arc.first.first.second]);
                else
                    Model.add(ye[arc] <= x[arc.first.first.second]);

                if(arc.first.first.first == 0 && arc.first.first.second == 0)
                    source += ye[arc];
            }
            for(const auto & arc: arcs){
                distance += y[arc];
                if(arc.first.first == 0 && arc.first.second == 0)
                    source += y[arc];
            }
            Model.add(source == 1);
            Model.add(suml >= distance);

            minsum += suml;
            for(const auto & node: nodes){
                if((node.first == 0 && node.second == 0) || (node.first == n && node.second == n))
                    continue;
                IloExpr outflow(env);
                IloExpr inflow(env);
                for(const auto & arc: arcs){
                    if(arc.first.first == node.first && arc.first.second == node.second){
                        outflow += y[arc];
                    }
                    if(arc.second.first == node.first && arc.second.second == node.second){
                        inflow += y[arc];
                    }
                }
                for(const auto & arc: diagonal_free_arcs){
                    if(arc.first.first.first == node.first && arc.first.first.second == node.second){
                        outflow += ye[arc];
                    }
                    if(arc.first.second.first == node.first && arc.first.second.second == node.second){
                        inflow += ye[arc];
                    }
                }
                Model.add(outflow == inflow);
            }
        }

        //Model.add(IloMinimize(env, objfun));
        Model.add(IloMinimize(env, minsum));

        IloCplex cplex(Model);

        for(size_t i = 0; i < n; i++){
            cplex.setPriority(x[i], 10);
        }
        cplex.exportModel("levenshtein_extended_pop.lp");
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        cplex.setParam(IloCplex::Param::MIP::Strategy::Search, 1);

        if (!cplex.solve()) {
            cplex.exportModel("levensh.lp");
            env.error() << "Failed to optimize the Master Problem!!!" << endl;
            throw(-1);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        double obj = cplex.getObjValue();
        cout << "\n\nThe objective value is: " << obj << endl;

        return make_pair(obj, std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
    }
    catch ( IloException& e )
    {
        std::cout << e << std::endl;
        e.end();
    }
}


pair<double, double> resource_extended_formulation(ui n, ui m, const vvb & stringpool, int lower_bound, ui upper_bound, const vector<vector<double>> & distances){
    try
    {
        IloEnv env;
        IloModel Model(env);

        IloNumVarArray x(env, n);
        for(ui i = 0; i < n; i++){
            string name = "x" + to_string(i);
            x[i] = IloIntVar(env, 0, 1, name.c_str());
        }
        //IloIntVar d(env, lower_bound, upper_bound, "d");
        //IloExpr minsum(env);
        //minsum.clear();
        graph_nodes nodes;
        graph_arcs arcs;

        generate_y_and_nodes(n, n/2, arcs, nodes);

        std::vector<IloIntVar> l;
        l.resize(n);
        for(int i = lower_bound+1; i < n+1; i++){
            string name = "l" + to_string(i);
            l[i] = IloIntVar(env, 0, 1, name.c_str());
        }
        for(int i = 1; i < lower_bound+1; i++){
            string name = "l" + to_string(i);
            l[i] = IloIntVar(env, 1, 1, name.c_str());
        }
        for(int i = 2; i < n+1; i++){
            Model.add(l[i-1]>= l[i]);
        }

        for(ui index = 0; index < m; index++){
            graph_free_arcs diagonal_free_arcs;
            generate_ye(n, n/2, stringpool[index], diagonal_free_arcs);

            for(const auto & c: stringpool[index]) cout << c;

            std::map<costly_arcs, IloNumVar> y;
            std::map<free_arcs, IloNumVar> ye;
            for(const auto & arc: arcs){
                y.emplace(arc, IloIntVar(env, 0, 1));
                string name = "y_" +std::to_string(arc.first.first)
                              +"_"+std::to_string(arc.first.second)
                              +"_"+std::to_string(arc.second.first)
                              +"_"+std::to_string(arc.second.second)
                              +"_"+std::to_string(index);
                y[arc].setName(name.c_str());
            }
            for(const auto & arc: diagonal_free_arcs) {
                ye.emplace(arc, IloIntVar(env, 0, 1));
                string name = "ye_"+std::to_string(index)
                              +"_"+std::to_string(arc.first.first.first)
                              +"_"+std::to_string(arc.first.first.second)
                              +"_"+std::to_string(arc.first.second.first)
                              +"_"+std::to_string(arc.first.second.second)
                              +"_"+std::to_string(arc.second);
                ye[arc].setName(name.c_str());
            }
            IloExpr source(env);
            IloExpr distance(env);
            IloExpr suml(env);
            for(int k = 1; k < n+1; k++){
                suml += l[k];
            }
            for(const auto & arc: diagonal_free_arcs){

                if(arc.second == 0)
                    Model.add(ye[arc] <= 1 - x[arc.first.first.second]);
                else
                    Model.add(ye[arc] <= x[arc.first.first.second]);

                if(arc.first.first.first == 0 && arc.first.first.second == 0)
                    source += ye[arc];
            }
            for(const auto & arc: arcs){
                distance += y[arc];
                if(arc.first.first == 0 && arc.first.second == 0)
                    source += y[arc];
            }
            Model.add(source == 1);
            Model.add(suml >= distance);

            //minsum += distance;
            for(const auto & node: nodes){
                if((node.first == 0 && node.second == 0) || (node.first == n && node.second == n))
                    continue;
                IloExpr outflow(env);
                IloExpr inflow(env);
                for(const auto & arc: arcs){
                    if(arc.first.first == node.first && arc.first.second == node.second){
                        outflow += y[arc];
                    }
                    if(arc.second.first == node.first && arc.second.second == node.second){
                        inflow += y[arc];
                    }
                }
                for(const auto & arc: diagonal_free_arcs){
                    if(arc.first.first.first == node.first && arc.first.first.second == node.second){
                        outflow += ye[arc];
                    }
                    if(arc.first.second.first == node.first && arc.first.second.second == node.second){
                        inflow += ye[arc];
                    }
                }
                Model.add(outflow == inflow);
            }
        }
        IloExpr objfun(env);
        for(int k=1; k < n+1; k++){
            objfun += l[k];
        }
        Model.add(IloMinimize(env, objfun));
        //Model.add(IloMinimize(env, minsum));

        IloCplex cplex(Model);

        for(size_t i = 0; i < n; i++){
            cplex.setPriority(x[i], 10);
        }
        cplex.exportModel("levenshtein_extended_pop.lp");
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        cplex.setParam(IloCplex::Param::MIP::Strategy::Search, 1);

        if (!cplex.solve()) {
            cplex.exportModel("levensh.lp");
            env.error() << "Failed to optimize the Master Problem!!!" << endl;
            throw(-1);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        double obj = cplex.getObjValue();
        cout << "\n\nThe objective value is: " << obj << endl;

        return make_pair(obj, std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
    }
    catch ( IloException& e )
    {
        std::cout << e << std::endl;
        e.end();
    }
}


pair<double, double> warm_extended_formulation(ui n, ui m, const vvb & stringpool, ui lower_bound, ui upper_bound,
                                               const vector<vector<double>> & distances, vector<double> xbound){
    try
    {
        IloEnv env;
        IloModel Model(env);

        IloNumVarArray x(env, n);
        for(ui i = 0; i < n; i++){
            string name = "x" + to_string(i);
            x[i] = IloIntVar(env, 0, 1, name.c_str());

        }
        //IloIntVar d(env, lower_bound, upper_bound, "d");
        IloExpr minsum(env);
        minsum.clear();
        graph_nodes nodes;
        graph_arcs arcs;


        generate_y_and_nodes(n, n/2, arcs, nodes);

        for(ui index = 0; index < m; index++){
            graph_free_arcs diagonal_free_arcs;
            generate_ye(n, n/2, stringpool[index], diagonal_free_arcs);

            for(const auto & c: stringpool[index]) cout << c;

            std::map<costly_arcs, IloNumVar> y;
            std::map<free_arcs, IloNumVar> ye;
            for(const auto & arc: arcs){
                y.emplace(arc, IloIntVar(env, 0, 1));
                string name = "y_" +std::to_string(arc.first.first)
                              +"_"+std::to_string(arc.first.second)
                              +"_"+std::to_string(arc.second.first)
                              +"_"+std::to_string(arc.second.second)
                              +"_"+std::to_string(index);
                y[arc].setName(name.c_str());
            }
            for(const auto & arc: diagonal_free_arcs) {
                ye.emplace(arc, IloIntVar(env, 0, 1));
                string name = "ye_"+std::to_string(index)
                              +"_"+std::to_string(arc.first.first.first)
                              +"_"+std::to_string(arc.first.first.second)
                              +"_"+std::to_string(arc.first.second.first)
                              +"_"+std::to_string(arc.first.second.second)
                              +"_"+std::to_string(arc.second);
                ye[arc].setName(name.c_str());
            }
            IloExpr source(env);
            IloExpr distance(env);

            for(const auto & arc: diagonal_free_arcs){

                if(arc.second == 0)
                    Model.add(ye[arc] <= 1 - x[arc.first.first.second]);
                else
                    Model.add(ye[arc] <= x[arc.first.first.second]);

                if(arc.first.first.first == 0 && arc.first.first.second == 0)
                    source += ye[arc];
            }
            for(const auto & arc: arcs){
                distance += y[arc];
                if(arc.first.first == 0 && arc.first.second == 0)
                    source += y[arc];
            }
            Model.add(source == 1);
            //Model.add(distance <= d);
            minsum += distance;
            for(const auto & node: nodes){
                if((node.first == 0 && node.second == 0) || (node.first == n && node.second == n))
                    continue;
                IloExpr outflow(env);
                IloExpr inflow(env);
                for(const auto & arc: arcs){
                    if(arc.first.first == node.first && arc.first.second == node.second){
                        outflow += y[arc];
                    }
                    if(arc.second.first == node.first && arc.second.second == node.second){
                        inflow += y[arc];
                    }
                }
                for(const auto & arc: diagonal_free_arcs){
                    if(arc.first.first.first == node.first && arc.first.first.second == node.second){
                        outflow += ye[arc];
                    }
                    if(arc.first.second.first == node.first && arc.first.second.second == node.second){
                        inflow += ye[arc];
                    }
                }
                Model.add(outflow == inflow);
            }
        }
        //Model.add(IloMinimize(env, d));
        Model.add(minsum >= static_cast<double>(lower_bound));

        Model.add(IloMinimize(env, minsum));

        IloCplex cplex(Model);
        IloNumArray xstart(env);
        for(int i = 0; i < xbound.size(); i++){
            xstart.add(xbound[i]);
        }
        for(size_t i = 0; i < n; i++){
            cplex.setPriority(x[i], 10);
        }
        cplex.addMIPStart(x, xstart);

        cplex.exportModel("levenshtein_extended.lp");
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        cplex.setParam(IloCplex::Param::MIP::Strategy::Search, 1);
        if (!cplex.solve()) {
            cplex.exportModel("levensh.lp");
            env.error() << "Failed to optimize the Master Problem!!!" << endl;
            throw(-1);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        double obj = cplex.getObjValue();
        cout << "\n\nThe objective value is: " << obj << endl;

        return make_pair(obj, std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
    }
    catch ( IloException& e )
    {
        std::cout << e << std::endl;
        e.end();
    }
}


pair<double,double> extended_formulation(ui n, ui m, const vvb & stringpool, int lower_bound, ui upper_bound,
                                         const vector<vector<double>> & distances, int mhub,
                                         const vb & median_hamming_string, vector<double>& xvars,
                                         vector<graph_arcs>& solution_ys,  vector<graph_free_arcs>& solution_yes){
    try
    {
        IloEnv env;
        IloModel Model(env);

        IloNumVarArray x(env, n);
        long max_x_id = 0;
        std::vector<IloExpr> dist_fo;
        dist_fo.clear();
        std::vector<std::vector<IloNumVar>> x_0_incompatibilities;
        std::vector<std::vector<IloNumVar>> x_1_incompatibilities;
        for(ui i = 0; i < n; i++){
            string name = "x" + to_string(i);
            x[i] = IloIntVar(env, 0, 1, name.c_str());
            max_x_id = std::max(max_x_id, x[i].getId());
        }
        //IloIntVar z = IloIntVar(env, 0, mhub, "z");
//        x_0_incompatibilities.resize(max_x_id + 1);
//        for(size_t t = 0; t < x_0_incompatibilities.size(); t++){
//            x_0_incompatibilities[t].clear();
//        }
//        x_1_incompatibilities.resize(max_x_id + 1);
//        for(size_t t = 0; t < x_1_incompatibilities.size(); t++){
//            x_1_incompatibilities[t].clear();
//        }
        IloIntVar d(env, lower_bound, upper_bound, "d");

        graph_nodes nodes;
        graph_arcs arcs;

        IloExpr minsum(env);
        minsum.clear();
        generate_y_and_nodes(n, n/2, arcs, nodes);
        vector<std::map<costly_arcs, IloNumVar>> yvars;
        vector<std::map<free_arcs, IloNumVar>> yevars;
        for(ui index = 0; index < m; index++){
            graph_free_arcs diagonal_free_arcs;
            generate_ye(n, n/2, stringpool[index], diagonal_free_arcs);

            for(const auto & c: stringpool[index]) cout << c;

            std::map<costly_arcs, IloNumVar> y;
            std::map<free_arcs, IloNumVar> ye;
            for(const auto & arc: arcs){
                y.emplace(arc, IloIntVar(env, 0, 1));
                string name = "y_" +std::to_string(arc.first.first)
                              +"_"+std::to_string(arc.first.second)
                              +"_"+std::to_string(arc.second.first)
                              +"_"+std::to_string(arc.second.second)
                              +"_"+std::to_string(index);
                y[arc].setName(name.c_str());
            }
            for(const auto & arc: diagonal_free_arcs) {
                ye.emplace(arc, IloIntVar(env, 0, 1));
                string name = "ye_"+std::to_string(index)
                              +"_"+std::to_string(arc.first.first.first)
                              +"_"+std::to_string(arc.first.first.second)
                              +"_"+std::to_string(arc.first.second.first)
                              +"_"+std::to_string(arc.first.second.second)
                              +"_"+std::to_string(arc.second);
                ye[arc].setName(name.c_str());
            }
            yvars.push_back(y);
            yevars.push_back(ye);
            IloExpr source(env);
            IloExpr distance(env);

            for(const auto & arc: diagonal_free_arcs){

                if(arc.second == 0) {
                    Model.add(ye[arc] <= 1 - x[arc.first.first.second]);
                    //x_1_incompatibilities[x[arc.first.first.second].getId()].push_back(ye[arc]);
                }
                else {
                    Model.add(ye[arc] <= x[arc.first.first.second]);
                    //x_0_incompatibilities[x[arc.first.first.second].getId()].push_back(ye[arc]);
                }

                if(arc.first.first.first == 0 && arc.first.first.second == 0)
                    source += ye[arc];
            }
            for(const auto & arc: arcs){
                distance += y[arc];
                if(arc.first.first == 0 && arc.first.second == 0)
                    source += y[arc];
            }
            //Model.add(source == 1);
            Model.add(distance <= d);
            //minsum += distance;
            dist_fo.push_back(distance);

            for(const auto & node: nodes){
                if((node.first == 0 && node.second == 0) || (node.first == n && node.second == n))
                    continue;
                IloExpr outflow(env);
                IloExpr inflow(env);

                for(const auto & arc: arcs){
                    if(arc.first.first == node.first && arc.first.second == node.second){
                        outflow += y[arc];
                    }
                    if(arc.second.first == node.first && arc.second.second == node.second){
                        inflow += y[arc];
                    }
                }
                for(const auto & arc: diagonal_free_arcs){
                    if(arc.first.first.first == node.first && arc.first.first.second == node.second){
                        outflow += ye[arc];
                    }
                    if(arc.first.second.first == node.first && arc.first.second.second == node.second){
                        inflow += ye[arc];
                    }
                }
                Model.add(outflow == inflow);
            }
        }
//        for(ui i = 0; i < m; i++){
//            for(ui j = 0; j < m; j++){
//                if(i <= j) continue;
//                Model.add(dist_fo[min(i,j)] + dist_fo[max(i,j)] >= distances[min(i,j)][max(i,j)]);
//                Model.add(dist_fo[min(i,j)] + distances[min(i,j)][max(i,j)] >= dist_fo[max(i,j)]);
//                Model.add(distances[min(i,j)][max(i,j)] + dist_fo[max(i,j)] >= dist_fo[min(i,j)]);
//            }
//        }
        //Model.add(minsum <= z);
        Model.add(IloMinimize(env, d));
        IloCplex cplex(Model);
        cplex.exportModel("levenshtein_extended.lp");
//        IloNumVarArray initvars(env);
//        IloNumArray initvalue(env);
//
        //for(size_t i = 0; i < n; i++){
            //cplex.setPriority(x[i], 10);
            //initvars.add(x[i]);
            //initvalue.add(xvars[i]);
        //}
//        for(size_t j = 0; j < m; j++){
//            for(const auto& arc: yvars[j]){
//                initvars.add(yvars[j][arc.first]);
//                if(std::find(solution_ys[j].begin(), solution_ys[j].end(),arc.first) != solution_ys[j].end()){
//                    initvalue.add(1.0);
//                } else {
//                    initvalue.add(0.0);
//                }
//            }
//            for(const auto& arc: yevars[j]){
//                initvars.add(yevars[j][arc.first]);
//                if(std::find(solution_yes[j].begin(), solution_yes[j].end(),arc.first) != solution_yes[j].end()){
//                    initvalue.add(1.0);
//                } else {
//                    initvalue.add(0.0);
//                }
//            }
//        }
        //cplex.addMIPStart(initvars, initvalue);
        //cplex.use(BunkeBCB(env, x_0_incompatibilities, x_1_incompatibilities));
        //cplex.use(BunkePrune(env, lower_bound));
        //cplex.use(HammingPrune(env,x, distances, stringpool, mhub, median_hamming_string));
        cplex.setParam(IloCplex::Param::MIP::Strategy::Search,
                       IloCplex::Traditional);
        cplex.setParam(IloCplex::Param::TimeLimit, 3600);
        //cplex.use(BunkeCB(env, x, d, distances, stringpool));
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        if (!cplex.solve()) {
            cplex.exportModel("levensh.lp");
            env.error() << "Failed to optimize the Master Problem!!!" << endl;
            throw(-1);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        double obj = cplex.getObjValue();
        cout << "\n\nThe objective value is: " << obj << endl;

        return make_pair(obj, std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
    }
    catch ( IloException& e )
    {
        std::cout << e << std::endl;
        e.end();
    }
}

double projected_hamming(ui n, ui m, const vvb & stringpool, double ub){
    try
    {
        IloEnv env;
        IloModel Model(env);

        long max_x_id = 0;
        std::vector<IloExpr> dist_fo;
        dist_fo.clear();

        graph_nodes nodes;
        graph_arcs arcs;
        vector<std::map<costly_arcs, IloIntVar>> yvars;
        vector<std::map<free_arcs, IloIntVar>> yevars;
        IloExpr minsum(env);
        minsum.clear();
        generate_y_and_nodes(n, n/2, arcs, nodes);
        //IloIntVar d(env, 0, n, "d");

        for(ui index = 0; index < m; index++){
            graph_free_arcs diagonal_free_arcs;
            generate_ye(n, n/2, stringpool[index], diagonal_free_arcs);

            std::map<costly_arcs, IloIntVar> y;
            std::map<free_arcs, IloIntVar> ye;
            for(const auto & arc: arcs){
                y.emplace(arc, IloIntVar(env, 0, 1));
                string name = "y_" +std::to_string(arc.first.first)
                              +"_"+std::to_string(arc.first.second)
                              +"_"+std::to_string(arc.second.first)
                              +"_"+std::to_string(arc.second.second)
                              +"_"+std::to_string(index);
                y[arc].setName(name.c_str());
            }
            for(const auto & arc: diagonal_free_arcs) {
                ye.emplace(arc, IloIntVar(env, 0, 1));
                string name = "ye_"+std::to_string(index)
                              +"_"+std::to_string(arc.first.first.first)
                              +"_"+std::to_string(arc.first.first.second)
                              +"_"+std::to_string(arc.first.second.first)
                              +"_"+std::to_string(arc.first.second.second)
                              +"_"+std::to_string(arc.second);
                ye[arc].setName(name.c_str());
            }
            yvars.push_back(y);
            yevars.push_back(ye);

            IloExpr source(env);
            IloExpr distance(env);

            for(const auto & arc: diagonal_free_arcs){

                if(arc.first.first.first == 0 && arc.first.first.second == 0)
                    source += ye[arc];
            }
            for(const auto & arc: arcs){
                distance += y[arc];
                if(arc.first.first == 0 && arc.first.second == 0)
                    source += y[arc];
            }
            Model.add(source == 1);
            //Model.add(distance <= d);
            minsum += distance;
            dist_fo.push_back(distance);

            for(const auto & node: nodes){
                if((node.first == 0 && node.second == 0) || (node.first == n && node.second == n))
                    continue;
                IloExpr outflow(env);
                IloExpr inflow(env);

                for(const auto & arc: arcs){
                    if(arc.first.first == node.first && arc.first.second == node.second){
                        outflow += y[arc];
                    }
                    if(arc.second.first == node.first && arc.second.second == node.second){
                        inflow += y[arc];
                    }
                }
                for(const auto & arc: diagonal_free_arcs){
                    if(arc.first.first.first == node.first && arc.first.first.second == node.second){
                        outflow += ye[arc];
                    }
                    if(arc.first.second.first == node.first && arc.first.second.second == node.second){
                        inflow += ye[arc];
                    }
                }
                Model.add(outflow == inflow);
            }
        }
//        std::cout << "\nAdding edge inequalities\n";
//
//        for(size_t i = 0; i < m; i++){
//            for(size_t j = i+1; j < m; j++){
//                for(const auto & [mapi, vari]: yevars[i]){
//                    for(const auto & [mapj, varj]: yevars[j]){
//                        if(mapi.second != mapj.second && mapi.first.first.second == mapj.first.first.second){
//                            Model.add(vari + varj <= 1);
//                        }
//                    }
//                }
//            }
//        }
        std::cout << "\nAdding clique inequalities\n";
        int idx = 0;
        for(const auto & str: stringpool){
            std::set<int> z_pos_i;
            int chidx = 0;
            for(const auto & ch: str){
                if(!ch){
                    z_pos_i.insert(chidx);
                }
                chidx++;
            }
            for(int jdx = idx +1; jdx < m; jdx ++){
                std::set<int> z_pos_j;
                for(int chjdx = 0; chjdx < stringpool[jdx].size(); chjdx++){
                    if(!stringpool[jdx][chjdx]){
                        z_pos_j.insert(chjdx);
                    }
                }
                for(int col = 0; col < n; col++){
                    std::vector<IloIntVar> aggregated_disjunction_0;
                    std::vector<IloIntVar> aggregated_disjunction_1;
                    for(const auto & [mapi, vari]: yevars[idx]){
                        if(mapi.first.first.second == col){
                            if(z_pos_i.find(mapi.first.first.first) != z_pos_i.end()){
                                aggregated_disjunction_0.push_back(vari);
                            }
                            else{
                                aggregated_disjunction_1.push_back(vari);
                            }
                        }
                    }
                    for(const auto & [mapj, varj]: yevars[jdx]){
                        if(mapj.first.first.second == col) {
                            if (z_pos_j.find(mapj.first.first.first) != z_pos_j.end()) {
                                aggregated_disjunction_1.push_back(varj);
                            } else {
                                aggregated_disjunction_0.push_back(varj);
                            }
                        }
                    }
                    IloExpr clique0(env);
                    IloExpr clique1(env);

                    for(const auto & var: aggregated_disjunction_0){
                        clique0 += var;
                    }
                    for(const auto & var: aggregated_disjunction_1){
                        clique1 += var;
                    }

                    if(!aggregated_disjunction_0.empty()){
                        Model.add(clique0 <= 1);
                    }
                    if(!aggregated_disjunction_1.empty()){
                        Model.add(clique1 <= 1);
                    }
                }
            }
            idx++;
        }
        Model.add(IloMinimize(env, minsum));
        IloCplex cplex(Model);
        cplex.exportModel("projected_hamming.lp");

        cplex.setParam(IloCplex::Param::MIP::Strategy::Search,
                       IloCplex::Traditional);
        cplex.setParam(IloCplex::Param::TimeLimit, 3600);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        if (!cplex.solve()) {
            cplex.exportModel("proj_hamm_error.lp");
            env.error() << "Failed to optimize the Master Problem!!!" << endl;
            throw(-1);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        double obj = cplex.getObjValue();
        cout << "Ojective value is: " << obj << " in time " <<
             std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
             << endl;


        return obj;
    }
    catch ( IloException& e )
    {
        std::cout << e << std::endl;
        e.end();
    }
}

double w_diagonal_bound(ui n, ui m, const vvb & stringpool, const vector<vector<double>> & distances, ui w,
                        vector<double>& xvars, vector<graph_arcs>& solution_ys,  vector<graph_free_arcs>& solution_yes,
                        double jbbound){
    try
    {
        IloEnv env;
        IloModel Model(env);

        IloIntVarArray x(env, n);
        long max_x_id = 0;
        std::vector<IloExpr> dist_fo;
        dist_fo.clear();
        std::vector<std::vector<IloNumVar>> x_0_incompatibilities;
        std::vector<std::vector<IloNumVar>> x_1_incompatibilities;
        for(ui i = 0; i < n; i++){
            string name = "x" + to_string(i);
            x[i] = IloIntVar(env, 0, 1, name.c_str());
            max_x_id = std::max(max_x_id, x[i].getId());
        }

        graph_nodes nodes;
        graph_arcs arcs;
        vector<std::map<costly_arcs, IloIntVar>> yvars;
        vector<std::map<free_arcs, IloIntVar>> yevars;
        IloExpr minsum(env);
        minsum.clear();
        generate_y_and_nodes(n, w, arcs, nodes);
        IloIntVar d(env, 0, n, "d");

        for(ui index = 0; index < m; index++){
            graph_free_arcs diagonal_free_arcs;
            generate_ye(n, w, stringpool[index], diagonal_free_arcs);

            for(const auto & c: stringpool[index]) cout << c;

            std::map<costly_arcs, IloIntVar> y;
            std::map<free_arcs, IloIntVar> ye;
            for(const auto & arc: arcs){
                y.emplace(arc, IloIntVar(env, 0, 1));
                string name = "y_" +std::to_string(arc.first.first)
                              +"_"+std::to_string(arc.first.second)
                              +"_"+std::to_string(arc.second.first)
                              +"_"+std::to_string(arc.second.second)
                              +"_"+std::to_string(index);
                y[arc].setName(name.c_str());
            }
            for(const auto & arc: diagonal_free_arcs) {
                ye.emplace(arc, IloIntVar(env, 0, 1));
                string name = "ye_"+std::to_string(index)
                              +"_"+std::to_string(arc.first.first.first)
                              +"_"+std::to_string(arc.first.first.second)
                              +"_"+std::to_string(arc.first.second.first)
                              +"_"+std::to_string(arc.first.second.second)
                              +"_"+std::to_string(arc.second);
                ye[arc].setName(name.c_str());
            }
            yvars.push_back(y);
            yevars.push_back(ye);

            IloExpr source(env);
            IloExpr distance(env);

            for(const auto & arc: diagonal_free_arcs){

                if(arc.second == 0) {
                    Model.add(ye[arc] <= 1 - x[arc.first.first.second]);
                    //x_1_incompatibilities[x[arc.first.first.second].getId()].push_back(ye[arc]);
                }
                else {
                    Model.add(ye[arc] <= x[arc.first.first.second]);
                    //x_0_incompatibilities[x[arc.first.first.second].getId()].push_back(ye[arc]);
                }

                if(arc.first.first.first == 0 && arc.first.first.second == 0)
                    source += ye[arc];
            }
            for(const auto & arc: arcs){
                distance += y[arc];
                if(arc.first.first == 0 && arc.first.second == 0)
                    source += y[arc];
            }
            Model.add(source == 1);
            Model.add(distance <= d);
            //minsum += distance;
            dist_fo.push_back(distance);

            for(const auto & node: nodes){
                if((node.first == 0 && node.second == 0) || (node.first == n && node.second == n))
                    continue;
                IloExpr outflow(env);
                IloExpr inflow(env);

                for(const auto & arc: arcs){
                    if(arc.first.first == node.first && arc.first.second == node.second){
                        outflow += y[arc];
                    }
                    if(arc.second.first == node.first && arc.second.second == node.second){
                        inflow += y[arc];
                    }
                }
                for(const auto & arc: diagonal_free_arcs){
                    if(arc.first.first.first == node.first && arc.first.first.second == node.second){
                        outflow += ye[arc];
                    }
                    if(arc.first.second.first == node.first && arc.first.second.second == node.second){
                        inflow += ye[arc];
                    }
                }
                Model.add(outflow == inflow);
            }
        }
//        for(ui i = 0; i < m; i++){
//            for(ui j = 0; j < m; j++){
//                if(i <= j) continue;
//                Model.add(dist_fo[min(i,j)] + dist_fo[max(i,j)] >= distances[min(i,j)][max(i,j)]);
//                Model.add(dist_fo[min(i,j)] + distances[min(i,j)][max(i,j)] >= dist_fo[max(i,j)]);
//                Model.add(distances[min(i,j)][max(i,j)] + dist_fo[max(i,j)] >= dist_fo[min(i,j)]);
//            }
//        }
        //Model.add(minsum >= lower_bound);
        //Model.add(IloMinimize(env, minsum));

        std::cout << "\nAdding clique inequalities\n";
        int idx = 0;
        for(const auto & str: stringpool){
            std::set<int> z_pos_i;
            int chidx = 0;
            for(const auto & ch: str){
                if(!ch){
                    z_pos_i.insert(chidx);
                }
                chidx++;
            }
            for(int jdx = idx +1; jdx < m; jdx ++){
                std::set<int> z_pos_j;
                for(int chjdx = 0; chjdx < stringpool[jdx].size(); chjdx++){
                    if(!stringpool[jdx][chjdx]){
                        z_pos_j.insert(chjdx);
                    }
                }

                for(int col = 0; col < n; col++){
                    std::vector<IloIntVar> aggregated_disjunction_0;
                    std::vector<IloIntVar> aggregated_disjunction_1;
                    for(const auto & [mapi, vari]: yevars[idx]){
                        if(mapi.first.first.second == col){
                            if(z_pos_i.find(mapi.first.first.first) != z_pos_i.end()){
                                aggregated_disjunction_0.emplace_back(vari);
                            }
                            else{
                                aggregated_disjunction_1.emplace_back(vari);
                            }
                        }
                    }

                    for(const auto & [mapj, varj]: yevars[jdx]){
                        if(mapj.first.first.second == col) {
                            if (z_pos_j.find(mapj.first.first.first) != z_pos_j.end()) {
                                aggregated_disjunction_1.emplace_back(varj);
                            } else {
                                aggregated_disjunction_0.emplace_back(varj);
                            }
                        }
                    }
                    IloExpr clique0(env);
                    IloExpr clique1(env);

                    for(const auto & var: aggregated_disjunction_0){
                        clique0 += var;
                    }
                    for(const auto & var: aggregated_disjunction_1){
                        clique1 += var;
                    }

                    if(!aggregated_disjunction_0.empty()){
                        Model.add(clique0 <= 1);
                    }
                    if(!aggregated_disjunction_1.empty()){
                        Model.add(clique1 <= 1);
                    }
                }
            }
            idx++;
        }

        Model.add(IloMinimize(env, d));
        IloCplex cplex(Model);
        cplex.exportModel("levenshtein.lp");


//        for(size_t i = 0; i < n; i++) {
//            cplex.setPriority(x[i], 10);
//        }
        //cplex.use(BunkePrune(env, jbbound));
        cplex.setParam(IloCplex::Param::MIP::Strategy::Search,
                       IloCplex::Traditional);
        cplex.setParam(IloCplex::Param::Threads, 1);
        cplex.setParam(IloCplex::Param::TimeLimit, 3600);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        if (!cplex.solve()) {
            cplex.exportModel("levensh.lp");
            env.error() << "Failed to optimize the Master Problem!!!" << endl;
            throw(-1);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        double obj = cplex.getObjValue();
        cout << w << "-diagonal objective value is: " << obj << " in time " <<
                                                                            std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                                                                            << endl;

        return obj;
    }
    catch ( IloException& e )
    {
        std::cout << e << std::endl;
        e.end();
    }
}

void readfile(const string& filename, vvb & stringpool){
    ifstream ifs(filename);
    if(!ifs){
        throw std::runtime_error("Error opening File ");
    }
    string line;
    int counter = 0;
    // read stringpool
    getline(ifs, line);
    while(true){
        if(ifs.eof()) break;
        getline(ifs, line);
        stringpool[counter].clear();
        for(const char& c: line){
            if(isspace(c)) continue;
            stringpool[counter].push_back(c == '0' ? 0 : 1);
        }
        counter++;
    }
}

double xor_hamming(ui n, ui m, const vvb & stringpool){
    IloEnv env;
    IloModel model(env);
    IloIntVarArray x(env, n);
    long max_x_id = 0;
    std::vector<IloExpr> dist_fo;

    dist_fo.clear();
    for(ui i = 0; i < n; i++){
        string name = "x" + to_string(i);
        x[i] = IloIntVar(env, 0, 1, name.c_str());
        max_x_id = std::max(max_x_id, x[i].getId());
    }
    IloIntVar d(env, 0, n, "d");

    for(ui index = 0; index < m; index++){
        IloExpr distance(env);
        for(ui j = 0; j < n; j++){
            distance += x[j] + static_cast<int>(stringpool[index][j]) - 2*static_cast<int>(stringpool[index][j])*x[j];
        }
    }

    model.add(IloMinimize(env, d));
    IloCplex cplex(model);
    cplex.exportModel("xor_hamming.lp");

    cplex.setParam(IloCplex::Param::MIP::Strategy::Search,
                   IloCplex::Traditional);
    cplex.setParam(IloCplex::Param::Threads, 1);
    cplex.setParam(IloCplex::Param::TimeLimit, 3600);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (!cplex.solve()) {
        env.error() << "Failed to optimize the Master Problem!!!" << endl;
        throw(-1);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double obj = cplex.getObjValue();
    return obj;

}

double dense_hamming(ui n, ui m, const vvb & stringpool){
    IloEnv env;
    IloModel model(env);
    IloNumVarArray x(env, n);
    long max_x_id = 0;
    std::vector<IloExpr> dist_fo;
    dist_fo.clear();
    for(ui i = 0; i < n; i++){
        string name = "x" + to_string(i);
        x[i] = IloIntVar(env, 0, 1, name.c_str());
        max_x_id = std::max(max_x_id, x[i].getId());
    }
    IloIntVar d(env, 0, n, "d");

    for(ui index = 0; index < m; index++){
        IloExpr distance(env);
        distance = 2*d;
        std::vector<int> zero_pos;
        std::vector<int> one_pos;
        for(int c = 0; c < stringpool[index].size(); c++){
            if(!stringpool[index][c]){
                zero_pos.push_back(c);
            } else {
                one_pos.push_back(c);
            }
        }
        for(const int & c: zero_pos){
            distance -= x[c];
        }
        for(const int & c: one_pos){
            distance += x[c];
        }
        distance -= static_cast<int>(one_pos.size());
        model.add(distance >= 0);
    }

    model.add(IloMinimize(env, d));
    IloCplex cplex(model);
    cplex.exportModel("dense_hamming.lp");

    cplex.setParam(IloCplex::Param::MIP::Strategy::Search,
                   IloCplex::Traditional);
    cplex.setParam(IloCplex::Param::TimeLimit, 3600);
    cplex.setParam(IloCplex::Param::Threads, 1);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (!cplex.solve()) {
        env.error() << "Failed to optimize the Master Problem!!!" << endl;
        throw(-1);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double obj = cplex.getObjValue();
    return obj*2;

}

pair<double, double> matching_formulation(ui n, ui m, const vvb & stringpool){
    try
    {
        IloEnv env;
        IloModel Model(env);

        IloNumVarArray x(env, n);
        for(ui i = 0; i < n; i++){
            string name = "x" + to_string(i);
            x[i] = IloIntVar(env, 0, 1, name.c_str());
        }
        std::map<ti, IloNumVar> mk;

        for(ui i= 0; i < m; i++) {
            for(ui j = 0; j < n; j++) {
                for(ui k = 0; k < n; k++) {
                    string name = "m_"+ to_string(i) + "_"+to_string(j) + "_"+to_string(k);
                    mk[make_pair(i, make_pair(j,k))] = IloIntVar(env, 0, 1, name.c_str());
                }
            }
        }

        for(ui k = 0; k < m; k++) {
            for(ui i = 0; i < n; i++) {
                IloExpr lhs(env);
                for(ui j = 0; j < n; j++) {
                    lhs += mk[make_pair(k, make_pair(i,j))];
                }
                Model.add(lhs <= 1);
            }
        }

        for(ui k = 0; k < m; k++) {
            for(ui j = 0; j < n; j++) {
                IloExpr lhs(env);
                for(ui i = 0; i < n; i++) {
                    lhs += mk[make_pair(k, make_pair(i,j))];
                }
                Model.add(lhs <= 1);
            }
        }
        for(ui k = 0; k < m; k++) {
            for(ui i = 0; i < n; i++) {
                for(ui j = 0; j < n; j++) {
                    for(ui ip = i + 1;  ip < n; ip++) {
                        for(ui jp = 0; jp < j; jp++) {
                            Model.add(mk[make_pair(k, make_pair(i, j))] +
                                mk[make_pair(k, make_pair(ip, jp))] <= 1);
                        }
                    }
                }
            }
        }

        IloExpr fo(env);

        for(ui k= 0; k < m; k++){
            IloExpr aux(env);
            for(ui i = 0; i < n; i++) {
                for(ui j = 0; j < n; j++) {
                    aux += mk[make_pair(k, make_pair(i, j))];
                }
            }
            fo += 2*(n - aux);
            for(ui i = 0; i < n; i++) {
                if(stringpool[k][i] == 1) {
                    for(ui j = 0; j < n; j++) {
                        fo += mk[make_pair(k, make_pair(i, j))]*(1-x[j]);
                    }
                }
                else {
                    for(ui j = 0; j < n; j++) {
                        fo += mk[make_pair(k, make_pair(i, j))]*x[j];
                    }
                }
            }
        }
        Model.add(IloMinimize(env, fo));
        IloCplex cplex(Model);
        cplex.exportModel("levenshtein_matching.lp");
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        cplex.setParam(IloCplex::Param::MIP::Strategy::Search, 1);
        cplex.setParam(IloCplex::Param::TimeLimit, 3600);

        if (!cplex.solve()) {
            cplex.exportModel("levenshmatching.lp");
            env.error() << "Failed to optimize the Master Problem!!!" << endl;
            throw(-1);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        double obj = cplex.getObjValue();
        cout << "\n\nThe objective value is: " << obj << endl;

        return make_pair(obj, std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
    }
    catch ( IloException& e )
    {
        std::cout << e << std::endl;
        e.end();
    }
}


pair<double, double> matching_formulation2(ui n, ui m, const vvb & stringpool){
    try
    {
        IloEnv env;
        IloModel Model(env);

        IloNumVarArray x(env, n);
        for(ui i = 0; i < n; i++){
            string name = "x" + to_string(i);
            x[i] = IloIntVar(env, 0, 1, name.c_str());
        }
        std::map<ti, IloNumVar> mk;

        for(ui i= 0; i < m; i++) {
            for(ui j = 0; j < n; j++) {
                for(ui k = 0; k < n; k++) {
                    string name = "m_"+ to_string(i) + "_"+to_string(j) + "_"+to_string(k);
                    mk[make_pair(i, make_pair(j,k))] = IloIntVar(env, 0, 1, name.c_str());
                }
            }
        }

        for(ui k = 0; k < m; k++) {
            for(ui i = 0; i < n; i++) {
                IloExpr lhs(env);
                for(ui j = 0; j < n; j++) {
                    lhs += mk[make_pair(k, make_pair(i,j))];
                }
                Model.add(lhs <= 1);
            }
        }

        for(ui k = 0; k < m; k++) {
            for(ui j = 0; j < n; j++) {
                IloExpr lhs(env);
                for(ui i = 0; i < n; i++) {
                    lhs += mk[make_pair(k, make_pair(i,j))];
                }
                Model.add(lhs <= 1);
            }
        }
        for(ui k = 0; k < m; k++) {
            for(ui i = 0; i < n; i++) {
                for(ui j = 0; j < n; j++) {
                    IloExpr sum1(env);
                    for(ui ip = 0;  ip < i; ip++) {
                        for(ui jp = j+1; jp < n; jp++) {
                            sum1 += mk[make_pair(k, make_pair(ip, jp))];
                        }
                    }
                  IloExpr sum2(env);
                    for(ui ip = i + 1;  ip < n; ip++) {
                        for(ui jp = 0; jp < j; jp++) {
                            sum2 += mk[make_pair(k, make_pair(ip, jp))];
                        }
                    }
                    Model.add(sum1 + sum2 <= n * (1-mk[make_pair(k, make_pair(i, j))]));
                }
            }
        }

        IloExpr fo(env);

        for(ui k= 0; k < m; k++){
            IloExpr aux(env);
            for(ui i = 0; i < n; i++) {
                for(ui j = 0; j < n; j++) {
                    aux += mk[make_pair(k, make_pair(i, j))];
                }
            }
            fo += 2*(n - aux);
            for(ui i = 0; i < n; i++) {
                if(stringpool[k][i] == 1) {
                    for(ui j = 0; j < n; j++) {
                        fo += mk[make_pair(k, make_pair(i, j))]*(1-x[j]);
                    }
                }
                else {
                    for(ui j = 0; j < n; j++) {
                        fo += mk[make_pair(k, make_pair(i, j))]*x[j];
                    }
                }
            }
        }
        Model.add(IloMinimize(env, fo));
        IloCplex cplex(Model);
        cplex.exportModel("levenshtein_matching.lp");
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        cplex.setParam(IloCplex::Param::MIP::Strategy::Search, 1);
        cplex.setParam(IloCplex::Param::TimeLimit, 3600);

        if (!cplex.solve()) {
            cplex.exportModel("levenshmatching.lp");
            env.error() << "Failed to optimize the Master Problem!!!" << endl;
            throw(-1);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        double obj = cplex.getObjValue();
        cout << "\n\nThe objective value is: " << obj << endl;

        return make_pair(obj, std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
    }
    catch ( IloException& e )
    {
        std::cout << e << std::endl;
        e.end();
    }
}


int main(int argc, char* argv[]) {
    ui n = atoi(argv[1]);
    ui m = atoi(argv[2]);
    ui w = atoi(argv[3]);
    string filename = argv[4];
    cout << "**************************************\n";
    cout << "********INSTANCE n " << n << " | m " <<  m << " **************\n";
    cout << "**************************************\n";
    vector<vector<double>> distances;
    vvb stringpool;
    stringpool.resize(m);
    vb median_hamming_string;
    median_hamming_string.clear();
    //readfile(filename, stringpool);

    for(int j = 0; j < m; j++){
        stringpool[j].resize(n);
        generate(stringpool[j].begin(), stringpool[j].end());
    }
    std::cout << "Strings\n";
    for(int j = 0; j < m; j++){
      for(int i = 0; i < n; i++){
        cout << stringpool[j][i];
      }
      std::cout << endl;
    }
//    for(int i = 0; i < n; i++){
//        int count_ones = 0;
//        for(int j = 0; j < m; j++){
//            count_ones += stringpool[j][i];
//        }
//        cout << count_ones << " " << ceil((double)m/2) << endl;
//        median_hamming_string.push_back(count_ones >= ceil((double)m/2));
//    }
//    int median_hamming_UB = 0;
//    for(const auto & s: stringpool){
//        cout << "'";
//        int distance_from_median_hamming = 0;
//        for(int i = 0; i < s.size(); i++){
//            cout << s[i];
//            distance_from_median_hamming += s[i] != median_hamming_string[i];
//        }
//        median_hamming_UB += distance_from_median_hamming;
//        cout << "',";
//    }
//    cout << "\n";
    //cout << "Hamming central median ";
    //for(const auto & c: median_hamming_string) cout << c;
    //cout << endl;
    //cout << "Median Hamming Upper Bound " << median_hamming_UB << "\n";
    distances.resize(m);
    for(ui i = 0; i < m; i++){
        distances[i].resize(m);
        for(ui j = i+1; j < m; j++){
            distances[i][j] = levenshtein(stringpool[i], stringpool[j]);
        }
    }
//    cout << "Distanze calcolate\n";
//    for(auto v: distances)
//        for(auto u: v)
//            cout << u << " ";
//    cout << "\n";
    vector<int> q;
    q.resize(m, 0);
    cout << "**************************************\n";
    cout << "********JUNG-BUNKE BOUND**************\n";
    //double JBbound = JB_bound(distances,n,m, q);
    //cout << "JB Bound " << JBbound << "\n";
    double JBbound = JB_bound(distances,n,m, q);
    cout << "JB Bound " << JBbound << "\n";
//    cout << "**************************************\n";
//    cout << "**************************************\n";
//    cout << "******** EXT-FORMULATION NO BOUNDS **************\n";
    //auto enc = extended_formulation_no_callback(n, m, stringpool, 0, n, distances);
//    cout << "**************************************\n";
//    cout << "**************************************\n";
//    cout << "********"<< w <<"-DIAG UPPER BOUND COMPUTATION **************\n";
    vector<double> xvars;
    vector<graph_arcs> solution_ys;
    vector<graph_free_arcs> solution_yes;
//    auto wdb = w_diagonal_bound(n, m, stringpool, distances, w, xvars, solution_ys, solution_yes, JBbound);
    //cout << "\n" << wdb<< "\n";
//    cout << "**************************************\n";
//    cout << "******** XOR HAMMING **************\n";
//    auto xorham = xor_hamming(n, m, stringpool);
//    cout << "\n" << xorham << "\n";
//    cout << "**************************************\n";

//    cout << "**************************************\n";
//    cout << "******** DENSE HAMMING **************\n";
//    auto denseham = dense_hamming(n, m, stringpool);
//    cout << "\n" << denseham << "\n";
//    cout << "**************************************\n";
    //cout << "Hamming median " << median_hamming_UB << endl;
    //cout << "lenght " << xvars.size() << "\n";
//    cout << "**************************************\n";
//    cout << "**************************************\n";
//    cout << "******** PROJ-FORMULATION **************\n";
//    double projhamming = projected_hamming(n, m, stringpool, JBbound);
//
//
//    cout << "**************************************\n";

    cout << "**************************************\n";
    cout << "**************************************\n";
    cout << "******** EXT-FORMULATION**************\n";
    auto enc2 = extended_formulation_no_callback(n, m, stringpool, ceil(JBbound), n, distances);
    cout << "**************************************\n";

//    cout << "**************************************\n";
//    cout << "**************************************\n";
//    cout << "******** RESOURCE-FORMULATION **************\n";
//    auto pop = resource_extended_formulation(n, m, stringpool, ceil(JBbound), n, distances);
//    cout << "**************************************\n";

//    cout << "**************************************\n";
//    cout << "**************************************\n";
//    cout << "******** ASS-FORMULATION **************\n";
//    auto ass = assignment_extended_formulation(n, m, stringpool, ceil(JBbound), n, distances);
//    cout << "**************************************\n";

    // cout << "**************************************\n";
    // cout << "**************************************\n";
    // cout << "******** POP-FORMULATION **************\n";
    // auto pop = pop_extended_formulation(n, m, stringpool, ceil(JBbound), n, distances);
    // cout << "**************************************\n";


    cout << "**************************************\n";
    cout << "**************************************\n";
    cout << "******** MATCHING-FORMULATION 1**************\n";
    auto mf = matching_formulation(n, m, stringpool);
    cout << "**************************************\n";


    cout << "**************************************\n";
    cout << "**************************************\n";
    cout << "******** MATCHING-FORMULATION 2**************\n";
    auto mf2 = matching_formulation2(n, m, stringpool);
    cout << "**************************************\n";
//    cout << "******** EXT-FORMULATION DIAGONAL UB ONLY **************\n";
    //auto enc3 = extended_formulation(n, m, stringpool, 0, static_cast<ui>(48), distances,
                                     //static_cast<ui>(48), median_hamming_string, xvars, solution_ys,
                                     //solution_yes);
//    cout << "**************************************\n";
//    cout << "**************************************\n";
//    cout << "**************************************\n";
//    cout << "******** EXT-FORMULATION BOTH LB AND UB **************\n";
//    auto enc4 = extended_formulation(n, m, stringpool, JBbound, static_cast<int>(48), distances,
//                                     static_cast<int>(48), median_hamming_string, xvars, solution_ys,
//                                     solution_yes);
//    cout << "**************************************\n";
//    cout << "**************************************\n";
//    cout << "********EXTENDED WARM**************\n";
    //auto ubef = warm_extended_formulation(n, m, stringpool, ceil(JBbound), ceil(wdb.first), distances, wdb.second);
    //cout << "**************************************\n";
    //cout << "**************************************\n";
//    cout << "******** EXT-FORMULATION **************\n";
    //auto bce = extended_formulation(n, m, stringpool, ceil(JBbound), n, distances, median_hamming_UB, median_hamming_string);
//    cout << "**************************************\n";
    //cout << "Extended value " << enc.first << " in time " << enc.second << "\n";
    //cout << "BC value " << bce.first << " in time " << bce.second << "\n";
    //cout << "Hamming bound " << median_hamming_UB << " | " << w << "-Diagonal bound " << wdb.first << " in time " << wdb.second << "\n";
    //cout << "JB Bound " << JBbound << "\n";
}