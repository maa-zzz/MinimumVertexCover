import pandas as pd
import numpy as np
import random
import math
import gc
import matplotlib.pylab as plt
import networkx as nx
import time
import os
import operator
import PySimpleGUI as sg

# # Define the layout of the GUI
# layout = [
#     [sg.Text("Enter the number of nodes:")],
#     [sg.Input(key="nodes")],
#     [sg.Text("Enter the edge probability:")],
#     [sg.Input(key="edge_prob")],
#     [sg.Button("OK")]
# ]

# # Create the GUI window
# window = sg.Window("My GUI", layout)

# # Start the GUI event loop
# while True:
#     event, values = window.read()
#     if event == sg.WINDOW_CLOSED:
#         break
#     elif event == "OK":
#         nodes = int(values["nodes"])
#         edge_prob = float(values["edge_prob"])
#         break

# # Close the window and exit the event loop
# window.close()

# # Print the values of the nodes and edge_prob variables
# print("Number of nodes:", nodes)
# print("Edge probability:", edge_prob)

sg.theme('LightBlue5')
nodes=0
edge_prob = 0.00
layout = [    [sg.Text('Enter the number of nodes:', font=('Helvetica', 16))],
    [sg.Input(key='nodes', font=('Helvetica', 16))],
    [sg.Text('Enter the edge probability:', font=('Helvetica', 16))],
    [sg.Input(key='edge_prob', font=('Helvetica', 16))],
    [sg.Button("OK", font=('Helvetica', 16), pad=(20, 20),button_color=('#0B2447', '#EBD8B2') )]
]
window = sg.Window('MINIMUM VERTEX COVER', layout, background_color='#b1f2ff')
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    elif event == "OK":
        nodes = int(values["nodes"])
        edge_prob = float(values["edge_prob"])
        break


event, values = window.read()

window.close()

fin_ans = 0

adjacency_matrix = np.zeros((nodes,nodes),dtype = int)
adjacency_matrix.shape

edge_probability = edge_prob
# print("Edge probability:", edge_probability)
# edge_probability = float(input('Desired edge probability: '))
# edge_probability = .0085

edges = []
edges_cnt = 0
for i in range(nodes):
    for j in range(i):
        prob = random.random()
        if prob < edge_probability:
            adjacency_matrix[i,j] = 1
            edges.append((i,j))
            edges_cnt += 1

# adjacency matrix showing whether there is an edge between the i-th row and j-th column
adjacency_matrix

# these are the edges in the graph that we have generated
print(edges[:10])
print("Number of edges {}".format(edges_cnt))

G=nx.Graph()
G.add_nodes_from(list(range(0,nodes)))
G.add_edges_from(edges)

plt.figure(figsize=(12,6))
nx.draw(G,node_color='r', node_size=18, alpha=0.8)
plt.show() # display

def initial_solution(G, cutoff,randSeed):
    # create initial solution via removing nodes with more connection (lower bound)
    random.seed(randSeed)
    start_time=time.time()
    _G=list(G.nodes())
    VC=sorted(list(zip(list(dict(G.degree(_G)).values()),_G)),reverse=False)
    i=0
    while(i<len(VC) and (time.time()-start_time) < cutoff):
        check = True
        for x in G.neighbors(VC[i][1]):
            if x not in _G:
                check = False
        if check:    
            _G.remove(VC[i][1])            
        i += 1
    return _G

def LS1_SA(G, cutoff,randSeed):
    T=0.8 
    random.seed(randSeed)
    S=initial_solution(G, cutoff,randSeed) 
    start_time=time.time()    
    S_ret=S.copy()
    S_best=[]
    sol=[]
    trace=""
    while((time.time() - start_time) < cutoff):
        T=0.95 * T 
        # looking if there exists some better solution with lesser number of nodes
        while not S_best:
            S_ret=S.copy()
            trace +=(str(time.time()-start_time) + ", " + str(len(S_ret)) + "\n")
            delete_v=random.choice(S)
            for v in G.neighbors(delete_v):
                if v not in S:
                    S_best.append(v)
                    S_best.append(delete_v)
            S.remove(delete_v)     
            
        # deleting node
        S_current=S.copy()
        uncovered_S=S_best.copy()
        delete_v=random.choice(S)
        for v in G.neighbors(delete_v):
            if v not in S:
                S_best.append(v)
                S_best.append(delete_v)            
        S.remove(delete_v)   


        # add node
        add_v=random.choice(S_best)
        S.append(add_v)
        for v in G.neighbors(add_v):
            if v not in S:
                S_best.remove(v)
                S_best.remove(add_v)

        # accept a new solution based on the probability which is proportional to the 
        # difference between the quality of the best solution and the current solution, and the temperature. 
        if len(uncovered_S) < len(S_best): 
            p=math.exp(float(len(uncovered_S) - len(S_best))/T)
            alpha=random.uniform(0, 1)
            if alpha > p:    
                S=S_current.copy()
                S_best=uncovered_S.copy()
    
    S_ret.sort()
    sol += str(len(S_ret)) + '\n' + ', '.join([str(v) for v in S_ret])

    return S_ret, trace

sol, trace = LS1_SA(G,30,10)

print(len(sol))
sol

#Coloring the graph
color_map = []
for i in range(nodes):
    color_map.append('blue')
for i in range(nodes):
    for j in sol :
        if i==j:
            color_map[i]='green'
nx.draw(G, node_color=color_map, with_labels=True)
plt.show() 


# HILL CLIMBING

G2=nx.Graph()
G2.add_nodes_from(list(range(0,nodes)))
G2.add_edges_from(edges)

def init(G, start_time, cutoff, trace_output):
    temp_G = G.nodes()
    VC = sorted(list(zip(list(dict(G.degree(temp_G)).values()), temp_G)))
    VC_sol = temp_G
    i=0
    uncovE=[]
    optvc_len = len(VC_sol)
    while(i < len(VC) and (time.time() - start_time) < cutoff):
        flag=True
        for x in G.neighbors(VC[i][1]):
            if x not in temp_G:
                flag = False
        if flag:	
            list(temp_G).remove(VC[i][1])
        i=i+1
    return VC_sol, trace_output	

# Helper method to add a vertex
def addV(G, VC, conf_check, dscores, edge_weights, uncovE, add):
    dscores[add] = -dscores[add]
    for x in G.neighbors(add):
        if x not in VC:
            uncovE.remove((add,x))
            uncovE.remove((x,add))
            conf_check[x] = 1
            dscores[x] -= edge_weights[add][x]
        else:
            dscores[x] += edge_weights[add][x]

# Helper method to remove a vertex
def removeV(G, VC, conf_check, dscores, edge_weights, uncovE, removed):
    dscores[removed] = -dscores[removed]
    conf_check[removed] = 0
    for x in G.neighbors(removed):
        if x not in VC:
            uncovE.append((removed,x))
            uncovE.append((x,removed))
            conf_check[x] = 1
            dscores[x] += edge_weights[removed][x]
        else:
            dscores[x] -= edge_weights[removed][x]

            
def check(Gx, VC):
    for v in VC:
        Gx.remove_node(v)
    if len(Gx.edges())>0:
        print("Checked: Graph is not a Vertex Cover")
    else:
        print("Checked: Graph is a Vertex Cover (VC)")

def Approx(G_):
   
    G = G_.copy() # Make a copy of the graph for modifying
    
    vertex_cover = []
    sol = ""
    start = time.time()

    while G.number_of_edges() > 0:
        node_degree = [x[1] for x in G.degree]
        min_degree_node = list(G.degree)[np.argmin(node_degree)][0]    

        min_degree_node_neighbors = G.neighbors(min_degree_node)
        G.remove_node(min_degree_node)

        for v in min_degree_node_neighbors:
            vertex_cover.append(v)
            G.remove_node(v)
            
    total_time = time.time() - start

    vertex_cover.sort()
    sol += str(len(vertex_cover)) + '\n' + ','.join([str(v) for v in vertex_cover])

    return sol, vertex_cover

def Hill(G, V, E, randSeed,cutoff):
    # Initialization

    #Computing Initial Solution
    vertex_coverx, vertex_cover = Approx(G)



    start_time = time.time()
    sol = ""
#     trace = ""

    random.seed(randSeed)

    threshold = .5*V
    reduction_factor = .3

    edge_weights = nx.convert.to_dict_of_dicts(G, edge_data=1)

    conf_check = [1]*(V+1)
    dscores = [0]*(V+1)
    uncovE=[]

    VC=list(vertex_cover)
    VC_sol = VC.copy()
    optvc_len = len(VC)
    avg_weight = 0
    delta_weight = 0
    Gi=G.copy()

    #andHereWeGoAgain

    while((time.time() - start_time) < cutoff):	

        # If it is a vertex cover: remove the max cost node		
        while not uncovE:
            if (optvc_len > len(VC)):
                total_time = time.time() - start_time			
                VC_sol = VC.copy()	
                optvc_len = len(VC)
#                 trace_output += str(total_time) + ', ' + str(optvc_len) + '\n'	
            max_improv = -float('inf')
            for x in VC:
                if dscores[x] > max_improv:
                    max_improv = dscores[x]
                    opt_rem = x
            VC.remove(opt_rem)
            removeV(G, VC, conf_check, dscores, edge_weights, uncovE, opt_rem)


        # remove max cost node from solution
        max_improv = -float('inf')
        for x in VC:
            if dscores[x] > max_improv:
                max_improv = dscores[x]
                opt_rem = x
        VC.remove(opt_rem)
        removeV(G, VC, conf_check, dscores, edge_weights, uncovE, opt_rem)



        # find node from random uncovered edge to add
        randEdge = random.choice(uncovE)
        if conf_check[randEdge[0]] == 0 and randEdge[1] not in VC: 
            better_add = randEdge[1]
        elif conf_check[randEdge[1]] == 0 and randEdge[0] not in VC:
            better_add = randEdge[0]
        else:
            if dscores[randEdge[0]] > dscores[randEdge[1]]:
                better_add = randEdge[0]
            else:
                better_add = randEdge[1]
        VC.append(better_add)
        addV(G, VC, conf_check, dscores, edge_weights, uncovE, better_add)

        # Update Edge Weights and score functions
        for x in uncovE:
            edge_weights[x[1]][x[0]] += 1				
            dscores[x[0]] += 1
        delta_weight += len(uncovE)/2

        # If average edge weights of graph above threshold then partially forget prior weighting decisions
        if delta_weight >= E:
            avg_weight +=1
            delta_weight -= E
        if avg_weight > threshold:
            dscores = [0]*(V+1)
            new_tot =0
            uncovE = []
            for x in G.edges():
                edge_weights[x[0]][x[1]] = int(reduction_factor*edge_weights[x[0]][x[1]])
                edge_weights[x[1]][x[0]] = int(reduction_factor*edge_weights[x[1]][x[0]])					
                new_tot += edge_weights[x[0]][x[1]]
                if not (x[0] in VC or x[1] in VC):
                    uncovE.append((x[1],x[0]))
                    uncovE.append((x[0],x[1]))		
                    dscores[x[0]] += edge_weights[x[0]][x[1]]
                    dscores[x[1]] += edge_weights[x[0]][x[1]]
                elif not (x[0] in VC and x[1] in VC):
                    if x[0] in VC:
                        dscores[x[0]] -= edge_weights[x[0]][x[1]]
                    else:
                        dscores[x[1]] -= edge_weights[x[0]][x[1]]
            avg_weight = new_tot/E
        VC = sorted(set(VC))		

    # Creating the solution and trace files
    vertex_cover=list(VC_sol.copy())
    vertex_cover.sort()
    sol += str(len(vertex_cover)) + '\n' + ','.join([str(v) for v in vertex_cover])

    G_=G

    check(G_,VC_sol.copy())

    return vertex_cover

soln2= Hill(G2,nodes,edges_cnt,2,0.1)
print(len(soln2))
soln2


color_map = []
for i in range(nodes):
    color_map.append('blue')
for i in range(nodes):
    for j in soln2 :
        if i==j:
            color_map[i]='green'
nx.draw(G, node_color=color_map, with_labels=True)
plt.show()   


G3=nx.Graph()
G3.add_nodes_from(list(range(0,nodes)))
G3.add_edges_from(edges)

visited = np.zeros(nodes)
cnt = 0
for e in edges:
    (u,v) = e
#     print(u,v)
    if ((visited[u]==0) & (visited[v]==0)):
        visited[u] = 1
        visited[v] = 1
        cnt+=2

approximation_algo_result = cnt
approximation_algo_result

n = nodes
pop_total = int(50 * max(1,round(n/5.0))) # max population allowed in the environment
pop_init = int(20 * max(1,round(n/5.0)))
max_iterate = int(7 * max(1,round(n/5.0)))
mutat_prob = 0.1

print("N = {}\nPopulation Total = {}\nPopulation Initial = {}\nMax Iteration = {}\n".format(n,pop_total,pop_init,max_iterate))

def chromosomes_gen(n,k,pop_init):
    lst = []
    for i in range(pop_init):
        chromosome = np.zeros(n,dtype= int)
        samples = random.sample(range(0,n), k = k)
        for j in range(k):
            chromosome[samples[j]] = 1
        lst.append(chromosome)
    return lst

def cost(cmbn,n,edges):
    obstacles = 0
    for e in edges:
        (u,v) = e
        if ((cmbn[u]==0) & (cmbn[v]==0)):
            obstacles += 1
    return obstacles

def selection(lst,pop_total,n,edges):
    score = []
    output_lst = []
    len_lst = len(lst)
    for i in range(len_lst):
        score.append(cost(lst[i],n,edges))
    sorted_index = np.argsort(score)
    cnt = 0
    for i in range(len_lst):
        output_lst.append(lst[sorted_index[i]])
        if((i+1) == pop_total):
            break
    lst = output_lst
    return lst,score[sorted_index[0]]

def helper_print(lst,n):
    res = []
    for i in range(n):
        if lst[i] == 1:
            res.append(i)
    print(res)

def cross_over_mutate_extended(lst,n,k,mutat_prob,pop_total,edges):
    new_lst = lst.copy()
    len_lst = len(lst)
    cross_over_prob = 0.50
    mutat_prob = 0.05
    variations = 1
    
    
    #Crossover
    for i in range(len_lst):
        for v in range(variations):
            tmp = lst[i].copy()

            mate_with = lst[int(random.uniform(0,len_lst))]

            tmp_unique = []
            mate_with_unique = []

            for j in range(n):
                if(tmp[j]==1):
                    tmp_unique.append(j)
                if(mate_with[j]==1):
                    mate_with_unique.append(j)

            tmp_unique = np.setdiff1d(tmp,mate_with)
            random.shuffle(tmp_unique)
            mate_with_unique = np.setdiff1d(mate_with,tmp)
            random.shuffle(mate_with_unique)

            swap = math.ceil(cross_over_prob * min(len(tmp_unique),len(mate_with_unique)))

            for j in range(swap):
                tmp[mate_with_unique[j]] = 1
                tmp[tmp_unique[j]] = 0

            # Mutation 
            zeroes = []
            ones = []
            for j in range(n):
                if tmp[j]==1:
                    ones.append(j)
                else:
                    zeroes.append(j)
            
            random.shuffle(ones)
            random.shuffle(zeroes)

            coin_toss = random.random()
            if(coin_toss <= 0.5):
                swaps = min(len(ones),len(zeroes))

                for j in range(swaps):
                    coin_toss2 = random.random()
                    if(coin_toss2 < mutat_prob):
                        tmp[ones[j]] = 0
                        tmp[zeroes[j]] = 1
                        #Swapping logic
                        dummy = ones[j]
                        ones[j] = zeroes[j]
                        zeroes[j] = dummy
            else:    
                
                mutate_lst = []
                for e in edges:
                    (u,v) = e
                    if((tmp[u] == 0) & (tmp[v] == 0)):
                        coin_toss2 = random.random()
                        if(coin_toss2 < mutat_prob):
                            coin_toss3 = random.random()
                            if(coin_toss3 <= 0.5):
                                if(u not in mutate_lst):
                                    mutate_lst.append(u)
                            else:
                                if(v not in mutate_lst):
                                    mutate_lst.append(v)

                random.shuffle(mutate_lst)
                sz = min(len(ones),len(mutate_lst))
                
                for j in range(sz):
                    tmp[ones[j]] = 0
                    tmp[mutate_lst[j]] = 1
                    #Swapping logic
                    dummy = ones[j]
                    ones[j] = mutate_lst[j]
                    mutate_lst[j] = dummy
                
            new_lst.append(tmp)
    
    return new_lst

def environment(n,k,mutat_prob,pop_init,pop_total,max_iterate,edges):
    lst = chromosomes_gen(n,k,pop_init)
    for it in range(max_iterate):
        lst = cross_over_mutate_extended(lst,n,k,mutat_prob,pop_total,edges)
#         return
        lst,cost_value = selection(lst,pop_total,n,edges)
        if (it%10)==9:
            print("k = {}, Iteration = {}, Cost = {}".format(k,it+1,cost_value))
        if cost_value==0:
            break
    result = []
    soln = lst[0]
    for j in range(len(soln)):
        if(soln[j] == 1):
            result.append(j)
    print("k = {}, Iteration = {}, Cost = {}\nSoln = {}".format(k,it,cost_value,result))
    return cost_value,result

def free_memory():
    gc.collect()

def mfind(n,mutat_prob,pop_init,pop_total,max_iterate,edges,start,end):
    result_dict = {}
    l = start
    h = end
    ans = {}
    while(l<=h):
        m = int((l+h)/2.0)
        cost_value,result = environment(n,m,mutat_prob,pop_init,pop_total,max_iterate,edges)
#         print("Cost is {} result is {}".format(cost_value,result))
        if(cost_value==0):
            ans = result
            result_dict[m] = result
            h = m-1
        else:
            l = m + 1
    return ans

# free_memory()
result = mfind(n,mutat_prob,pop_init,pop_total,max_iterate,edges,int(approximation_algo_result/2),n)

print(len(result))
result

color_map = []
for i in range(nodes):
    color_map.append('blue')
for i in range(nodes):
    for j in result :
        if i==j:
            color_map[i]='green'
nx.draw(G, node_color=color_map, with_labels=True)
plt.show()        
    
# BRANCH AND BOUND
G4=nx.Graph()
G4.add_nodes_from(list(range(0,nodes)))
G4.add_edges_from(edges)

def BnB(G, T):
        #RECORD START TIME
        start_time=time.time()
        end_time=start_time
        delta_time=end_time-start_time
        times=[]    #list of times when solution is found, tuple=(VC size,delta_time)

        # INITIALIZE SOLUTION VC SETS AND FRONTIER SET TO EMPTY SET
        OptVC = []
        CurVC = []
        Frontier = []
        neighbor = []

        # ESTABLISH INITIAL UPPER BOUND
        UpperBound = G.number_of_nodes()
        print('Initial UpperBound:', UpperBound)

        CurG = G.copy()  # make a copy of G
        # sort dictionary of degree of nodes to find node with highest degree
        v = find_maxdeg(CurG)
        print(v)
        # APPEND (V,1,(parent,state)) and (V,0,(parent,state)) TO FRONTIER
        Frontier.append((v[0], 0, (-1, -1)))  
        # tuples of node,state,(parent vertex,parent vertex state)
        Frontier.append((v[0], 1, (-1, -1)))


        while Frontier!=[] and delta_time<T:
            (vi,state,parent)=Frontier.pop() #set current node to last element in Frontier

            backtrack = False

            if state == 0:  # if vi is not selected, state of all neighbors=1
                neighbor = CurG.neighbors(vi)  # store all neighbors of vi
                for node in list(neighbor):
                    CurVC.append((node, 1))
                    CurG.remove_node(node)  # node is in VC, remove neighbors from CurG
            elif state == 1:  # if vi is selected, state of all neighbors=0
                CurG.remove_node(vi)  # vi is in VC,remove node from G
            else:
                pass

            CurVC.append((vi, state))
            CurVC_size = VC_Size(CurVC)

            if CurG.number_of_edges() == 0:  # end of exploring, solution found

                if CurVC_size < UpperBound:
                    OptVC = CurVC.copy()
                    print('Current Opt VC size', CurVC_size)
                    UpperBound = CurVC_size
                    times.append((CurVC_size,time.time()-start_time))
                backtrack = True

            else:   #partial solution
                CurLB = Lowerbound(CurG) + CurVC_size

                if CurLB < UpperBound:  # worth exploring
                    vj = find_maxdeg(CurG)
                    Frontier.append((vj[0], 0, (vi, state)))#(vi,state) is parent of vj
                    Frontier.append((vj[0], 1, (vi, state)))
                else:
                    # end of path, will result in worse solution,backtrack to parent
                    backtrack=True

            if backtrack==True:
                if Frontier != []:	#otherwise no more candidates to process
                    nextnode_parent = Frontier[-1][2]	#parent of last element in Frontier (tuple of (vertex,state))

                    # backtrack to the level of nextnode_parent
                    if nextnode_parent in CurVC:

                        id = CurVC.index(nextnode_parent) + 1
                        while id < len(CurVC):	#undo changes from end of CurVC back up to parent node
                            mynode, mystate = CurVC.pop()	#undo the addition to CurVC
                            CurG.add_node(mynode)	#undo the deletion from CurG

                            # find all the edges connected to vi in Graph G
                            # or the edges that connected to the nodes that not in current VC set.

                            curVC_nodes = list(map(lambda t:t[0], CurVC))
                            for nd in G.neighbors(mynode):
                                if (nd in CurG.nodes()) and (nd not in curVC_nodes):
                                    CurG.add_edge(nd, mynode)	#this adds edges of vi back to CurG that were possibly deleted

                    elif nextnode_parent == (-1, -1):
                        # backtrack to the root node
                        CurVC.clear()
                        CurG = G.copy()
                    else:
                        print('error in backtracking step')

            end_time=time.time()
            delta_time=end_time-start_time
            if delta_time>T:
                print('Cutoff time reached')

        return OptVC,times

def find_maxdeg(g):
        deglist = g.degree(list(G.nodes))
        print(deglist)
        v = [0,-1]
        for i in deglist :
            if v[1]<i[1]:
                v=i
        return v

def Lowerbound(graph):
        lb=graph.number_of_edges() / find_maxdeg(graph)[1]
        lb=ceil(lb)
        return lb

def ceil(d):
        #return the minimum integer that is bigger than d 
        if d > int(d):
            return int(d) + 1
        else:
            return int(d)
        
def VC_Size(VC):
        vc_size = 0
        for element in VC:
            vc_size = vc_size + element[1]
        return vc_size

Sol_VC,times = BnB(G4, 300)

result = Sol_VC
print(result)

color_map = []
for i in range(nodes):
    color_map.append('blue')
for i in range(nodes):
    for j in result :
        if i==j[0]:
            color_map[i]='green'
nx.draw(G, node_color=color_map, with_labels=True)
plt.show()        
    
    


