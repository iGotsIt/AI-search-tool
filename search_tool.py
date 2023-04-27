#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pydot
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from networkx.drawing.nx_pydot import graphviz_layout
from collections import deque
import queue
from copy import deepcopy
from celluloid import Camera
from IPython.display import HTML
from typing import List, Union, Optional
from typing import Callable, Dict, Iterator
import timeit


# In[2]:


"""
The Graph class is used to represent and visualize a graph. Graphs are stored as an adjacency list.
"""
class Graph:
    def __init__(self, vertices, edges, start, goal, weighted = False):
        """
        Initializes the Graph object.
        
        Parameters:
        - Vertices: A list of tuples containing the name of a node and its heuristic cost 
        Example: [(S, 1), (A, 2), ...]
        - Edges: A list of tuples containing the nodes that comprise and edge and the optional edge cost 
        Example: [(S, A, 1), (S, B, 2), ...]
        - Start: The starting node for the search algorithms.
        - Goal: The end node that the search algorithms are attempting to find a path to.
        - Weighted: A boolean denoting whether or not edges are weighted. If true, edge weights will be displayed on the visualization.
        """
        self.vertices, self.edges = {}, {}
        self.weighted = weighted
        self.visualization = edges
        self.start, self.goal = start, goal
        self.adjList = defaultdict(list)
        self.nx_graph = nx.Graph()
        self.initialize_params(vertices, edges)

    def visualize_graph(self):
        """
        Displays the networkx visualization of the Graph. If weighted, edge costs will be displayed.
        """
        if (self.weighted == True):
            costs = nx.get_edge_attributes(self.nx_graph, 'weight')
            pos = graphviz_layout(self.nx_graph, prog = "dot", root = self.start)
            nx.draw_networkx_edge_labels(self.nx_graph, pos, edge_labels = costs, font_color = 'red')
            nx.draw_networkx(self.nx_graph, pos, with_labels = "True", alpha = 0.9)
        else:
            pos = graphviz_layout(self.nx_graph, prog = "dot", root = self.start)
            nx.draw_networkx(self.nx_graph, pos, with_labels = "True")
        plt.show()
        
    def create_adj_list(self, edges):
        """
        Creates the adjacency list that represents the Graph. The list is a Python dictionary, where the key is a node
        and the value is a list of all nodes that shares an edge with the key node.
        
        Parameters:
        - Edges: A list of tuples containing the nodes that comprise and edge and the optional edge cost 
        Example: [(S, A, 1), (S, B, 2), ...] 
        """
        for edge in edges:
            source = edge[0]
            destination = edge[1]
            self.adjList[source].append(destination)
            self.adjList[destination].append(source)
            
    def print_adj_list(self):
        """
        Prints the adjacency list. Each row displays the node and the list of nodes that it shares an edge with.
        """
        for item in self.adjList.items():
            print(item)
    
    def create_node_dict(self, vertices):
        """
        Creates a dictionary of each vertex, where the key is the vertex name and the value is its heuristic cost.
        
        Parameters:
        - Vertices: A list of tuples containing the name of a node and its heuristic cost 
        Example: [(S, 1), (A, 2), ...]
        """
        for vertex in vertices:
            try: node, heuristic = vertex
            except: node, heuristic = vertex, 0
            self.vertices[node] = heuristic
    
    def create_edge_dict(self, edges):
        """
        Creates a dictionary of edges, where the key is the tuple of vertices and the value is the edge cost.
        
        Parameters:
        - Edges: A list of tuples containing the nodes that comprise and edge and the edge cost 
        Example: [(S, A, 1), (S, B, 2), ...]
        """
        for edge in edges:
            edge_name = (edge[0], edge[1])
            try:
                edge_cost = edge[2]
            except:
                edge_cost = 0
            self.edges[edge_name] = edge_cost
    
    def create_nx_graph(self):
        """
        Creates a networkx graph using the provided edges. 
        If weighted is True, edge costs will be displayed on the networkx graph.
        """
        if (self.weighted == True):
            self.nx_graph.add_weighted_edges_from(self.visualization)
        else:
            self.nx_graph.add_edges_from(self.visualization)
    
    def initialize_params(self, vertices, edges):
        """
        Initializes all class parameters using the provided vertices and edges.
        
        Parameters:
        - Vertices: A list of tuples containing the name of a node and its heuristic cost 
        Example: [(S, 1), (A, 2), ...]
        - Edges: A list of tuples containing the nodes that comprise and edge and the optional edge cost 
        Example: [(S, A, 1), (S, B, 2), ...]
        """
        self.create_adj_list(edges)
        self.create_node_dict(vertices)
        self.create_edge_dict(edges)
        self.create_nx_graph()
    


# In[3]:


def get_path(current_position, previous_node):
    """
    Updates the path being traversed during search.
    
    Parameters:
    - Current Position: The node that is currently being explored during search.
    - Previous Node: The node that was last explored during search
    
    Returns:
    - Path: a list of nodes containing the current and previous node.
    """
    path = []
    while current_position in previous_node.keys() and current_position != None:
        path.append(current_position)
        current_position = previous_node[current_position]
    return path


# In[4]:


def BFS(graph):
    """
    Executes a breadth-first search on the provided Graph object.
    
    Parameters:
    - Graph: An initialized object of the Graph data structure.
    
    Returns:
    - Frontier history: A list containing a queue of nodes in the frontier at each iteration.
    - Explore history: A list containing a queue of nodes explored at each iteration of the search.
    - Path history: A list containing a list of nodes in the path at each iteration of search.
    """
    frontier_steps = []
    explored_steps = []
    start = graph.start
    goal = graph.goal
    explored = deque([start])
    frontier = deque([start])
    previous_node = {start: None}
    paths_history = []
    frontier_steps.append(deepcopy(frontier))
    explored_steps.append(deepcopy(explored))
    paths_history.append([])
    while len(frontier) != 0:
        current_node = frontier[0]
        frontier.popleft()
        if current_node == goal:
            frontier_steps.append(deepcopy(frontier))
            explored_steps.append(deepcopy(explored))
            paths_history.append(get_path(current_node, previous_node))
            return frontier_steps, explored_steps, paths_history       
        for node in graph.adjList[current_node]:
            if node not in explored:
                frontier.append(node)
                explored.append(node)
                previous_node[node] = current_node
        frontier_steps.append(deepcopy(frontier))
        explored_steps.append(deepcopy(explored))
        paths_history.append(get_path(current_node, previous_node))


# In[5]:


def DFS(graph):
    """
    Executes a depth-first search on the provided Graph object.
    
    Parameters:
    - Graph: An initialized object of the Graph data structure.
    
    Returns:
    - Frontier history: A list containing a queue of nodes in the frontier at each iteration.
    - Explore history: A list containing a queue of nodes explored at each iteration of the search.
    - Path history: A list containing a list of nodes in the path at each iteration of search.
    """
    frontier_steps = []
    explored_steps = []
    start = graph.start
    goal = graph.goal
    explored = deque([start])
    frontier = deque([start])
    previous_node = {start: None}
    paths_history = []
    frontier_steps.append(deepcopy(frontier))
    explored_steps.append(deepcopy(explored))
    paths_history.append([])
    while len(frontier) != 0:
        current_node = frontier[-1]
        frontier.pop()
        if current_node == goal:
            frontier_steps.append(deepcopy(frontier))
            explored_steps.append(deepcopy(explored))
            paths_history.append(get_path(current_node, previous_node))
            return frontier_steps, explored_steps, paths_history       
        for node in graph.adjList[current_node]:
            if node not in explored:
                frontier.append(node)
                explored.append(node)
                previous_node[node] = current_node
        frontier_steps.append(deepcopy(frontier))
        explored_steps.append(deepcopy(explored))
        paths_history.append(get_path(current_node, previous_node))

#In[6]:
def a_star_search(graph: Graph):
    """
     Executes a breadth-first search on the provided Graph object with the given heuristics and costs
    
    Parameters:
    - Graph: An initialized object of the Graph data structure.
    
    Returns:
    - Frontier history: A list containing a list of nodes in the frontier at each iteration.
    - Explore history: A list containing a list of nodes explored at each iteration of the search.
    - Path history: A list containing a list of nodes in the path at each iteration of search.
    """
    frontier_steps = []
    explored_steps = []
    path_history = []
    frontier = queue.PriorityQueue()
    start = graph.start
    previous_node = {start: None}
    cost_of_node = {start: 0}
    frontier.put((graph.vertices[start],start))
    frontier_steps.append([node[1] for node in frontier.queue])
    path_history.append([])
    explored_steps.append(list(cost_of_node.keys()))
    while not frontier.empty():
        current_position = frontier.get()[1]
        if current_position == graph.goal:
            frontier_steps.append([node[1] for node in frontier.queue])
            path_history.append(get_path(current_position, previous_node))
            explored_steps.append(list(cost_of_node.keys()))
            return frontier_steps, explored_steps, path_history
        for new_position in graph.adjList[current_position]:
            try:
                edge = (current_position, new_position)
                new_position_cost = cost_of_node[current_position] + graph.edges[edge]
            except:
                edge = (new_position, current_position)
                new_position_cost = cost_of_node[current_position] + graph.edges[edge]
            if new_position not in cost_of_node or new_position_cost < cost_of_node[new_position]:
                cost_of_node[new_position] = new_position_cost
                priority = new_position_cost + graph.vertices[new_position]
                frontier.put((priority, new_position))
                previous_node[new_position] = current_position
        frontier_steps.append([node[1] for node in frontier.queue])
        path_history.append(get_path(current_position, previous_node))
        explored_steps.append(list(cost_of_node.keys()))

# In[7]:


def animate_search(graph, f_h, e_h, p_h):
    """
    Creates an animation of the searched graph, showing the state of the frontier, the explored nodes,
    and the solution path at each iteration.
    
    Parameters:
    - Graph: An initialized object of the Graph data structure.
    - Frontier history f_h: A list of nodes in the frontier at each iteration of search.
    - Explored history e_h: A list of nodes that were explored at each iteration of search.
    - Path history p_h: A list of nodes in the path at each iteration of search
    """
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(len(f_h)):
        color_map = []
        for node in graph.nx_graph:
            if node in p_h[i]: 
                color_map.append('green')
            elif node in f_h[i]:
                color_map.append('yellow')
            elif node in e_h[i]:
                color_map.append('purple')
            else:
                color_map.append('cyan')
        if graph.weighted == True:
            costs = nx.get_edge_attributes(graph.nx_graph, 'weight')
            pos = graphviz_layout(graph.nx_graph, prog = "dot", root = graph.start)
            nx.draw_networkx_edge_labels(graph.nx_graph, pos, edge_labels = costs, font_color = 'red')
            nx.draw_networkx(graph.nx_graph, pos, with_labels = "True", alpha = 0.9, node_color=color_map)
            plt.draw()
            camera.snap()
        else:
            pos = graphviz_layout(graph.nx_graph, prog = "dot", root = graph.start)
            nx.draw_networkx(graph.nx_graph, pos, with_labels = "True", node_color=color_map)
            plt.draw()
            camera.snap()
    animation = camera.animate(interval=500, repeat=True)
    return animation

def save_animation(animation, filename):
    """
    Saves the animation as the specified filename.
    
    Parameters:
    - Animation: The search animation that was output from the animate_search function.
    - Filename: The name and extension that the file should be saved as.
    """
    animation.save(filename)


# In[8]:


Point = Union[int, str]
Trace = List[Point]
def interative_deepening(graph: Graph) -> Optional[Trace]:
    """
    Uses the iterative deepening algorithm to search for a path from `graph.start` to `graph.goal`.

    Parameters:
    - graph: A Graph object representing the graph to search.

    Returns:
    - If a path is found, returns a Trace object containing the path from `graph.start` to `graph.goal`.
    - If no path is found, returns None.

    The function uses the depth-first search (DFS) algorithm to implement iterative deepening. Each iteration starts at `graph.start`
    and increases the depth limit until it reaches the depth limit of all nodes in the graph.
    For each iteration, the function calls the __dfs_with_limit helper function to perform a DFS search with a limited depth and store
    the search result in a trace list. If the search finds the goal node, it returns the trace list. Otherwise, the function increases
    the depth limit and continues to the next iteration until it finds the goal node or reaches the maximum depth limit.
    """
    # A helper function that performs a DFS search with a limited depth 
    # and returns a trace list of the search result.
    def __dfs_with_limit(cur_point: Point, deep: int, trace: Trace) -> Optional[Trace]: 
        trace.append(cur_point)
        neighbours = graph.adjList.get(cur_point, [])
        if cur_point == graph.goal: return trace
        if deep == 0 or len(neighbours) == 0: return None
        for n in neighbours: 
            if n in trace: continue
            found_trace = __dfs_with_limit(n, deep-1, trace.copy())
            if found_trace is not None: return found_trace
        return None
    # Initialize depth variables
    cur_deep, max_deep = 0, len(graph.vertices)
    # Iteratively perform DFS search with increasing depth limit
    while cur_deep != max_deep: 
        found = __dfs_with_limit(graph.start, cur_deep, list())
        if found is not None: return found
        cur_deep = cur_deep + 1
    # Return None if no path is found
    return None


# In[9]:


EvalFunc = Callable[[Graph, Point], int]
degree_eval: EvalFunc = lambda graph, point : len(graph.adjList.get(point, []))
euclidean_eval = lambda g, p: -1 # TODO(NEED HAVE COORDIATION)
manhattan_eval = lambda g, p: -1 # TODO(NEED HAVE COORDIATION)
def beam_search(graph: Graph, width: int = 2, eval_func: EvalFunc = degree_eval) -> Optional[Trace]: 
    """
    Uses the beam search algorithm to search for a path from `graph.start` to `graph.goal`.

    Parameters:
    - graph: A Graph object representing the graph to search.
    - width: An integer representing the beam width.

    Returns:
    - If a path is found, returns a Trace object containing the path from `graph.start` to `graph.goal`.
    - If no path is found, returns None.

    The function uses the beam search algorithm to search for the shortest path from `graph.start` to `graph.goal`. The beam width
    determines the maximum number of candidate nodes to consider at each level of the search tree. The evaluation function is based
    on the number of neighboring nodes of a node. The function starts with the `graph.start` node and iteratively expands the most
    promising nodes in the beam until it finds the `graph.goal` node or no path is found within the beam width.
    """

    # Define the evaluation function and parent dictionary
    eval_func: EvalFunc = lambda p : len(graph.adjList.get(p, []))
    parent_of: Dict[Point, Point] = {graph.start: None}

    # Perform the beam search
    current_beam = {graph.start}
    while current_beam and graph.goal not in parent_of:
        next_beam = set()
        for point in current_beam:
            neighbors = {n for n in  graph.adjList.get(point, []) if n not in parent_of}
            for n in neighbors: parent_of[n] = point
            next_beam.update(neighbors)
        next_beam = sorted(next_beam, key=eval_func, reverse=True)[:width]
        current_beam = next_beam

    # Return the result trace or None if no path is found
    if graph.goal not in parent_of:return None
    trace = [graph.goal]
    while trace[0] != graph.start: 
        trace.insert(0, parent_of[trace[0]])
    return trace


# In[10]:


def hill_climbing(graph: Graph, eval_func: EvalFunc = degree_eval, only_best: bool = True) -> Optional[Trace]:
    """
    Uses the Hill Climbing algorithm to search for a path from `graph.start` to `graph.goal`.

    Parameters:
    - graph: A Graph object representing the graph to search.
    - eval_func: An evaluation function to evaluate the quality of nodes (default is degree_eval).
    - only_best: A boolean indicating whether to consider only the best neighbour (default is True).
            If we following exactly the definition of the hill climbing, it will only find the best. 

    Returns:
    - If a path is found, returns a list containing the path from `graph.start` to `graph.goal`.
    - If no path is found, returns None.
    """
    def __dfs_with_best(cur_point: Point, trace: Trace) -> Optional[Trace]: 
        trace.append(cur_point)
        if cur_point == graph.goal: return trace
        neighbours = graph.adjList.get(cur_point, [])
        if len(neighbours) == 0: return None
        neighbours.sort(key=lambda p: eval_func(graph, p), reverse=True)
        if only_best: neighbours = [neighbours[0]]
        for n in neighbours: 
            if n in trace: continue
            found_trace = __dfs_with_best(n, trace.copy())
            if found_trace is not None: return found_trace
        return None
    return __dfs_with_best(graph.start, [])


# In[11]:

"""
The Benchmarking class is used time the execution of the algorithms and graph the results.
"""
class Benchmarking:
    """
    The constructor takes in the algorithms to be benchmarked, the inputs to be used, and creates the names of the algorithms.
    Inputs:
        - bfs: The breadth first search algorithm
        - dfs: The depth first search algorithm
        - a_star: The A* algorithm
        - iterative_deeping: The iterative deeping algorithm
        - beam: The beam algorithm
        - hill_climbing: The hill climbing algorithm
        - inputs: A list of lists of inputs to be used for each algorithm (have to be inputted in the same order as the algorithms)
    """
    def __init__(self, bfs, dfs, a_star, iterative_deeping, beam, hill_climbing, inputs):
        self.bfs = bfs
        self.dfs = dfs
        self.a_star = a_star
        self.iterative_deeping = iterative_deeping
        self.beam = beam
        self.hill_climbing = hill_climbing
        self.inputs = inputs
        self.list_of_functions = [self.bfs, self.dfs, self.a_star, self.iterative_deeping, self.beam, self.hill_climbing]
        self.names = ["BFS", "DFS", "A*", "Iterative Deeping", "Beam", "Hill Climbing"]
        self.exclusion_list = []
        self.execution_time = [-1, -1, -1, -1, -1, -1]
    """
    The update_exclusion_list function takes in a list of indices of algorithms to be excluded from the benchmarking.
    Inputs:
        - exclusion_list: A list of indices of algorithms to be excluded from the benchmarking.
    """
    def update_exclusion_list(self, exclusion_list):
        self.exclsion_list = exclusion_list
    """
    The function_timer runs the function with the given inputs and returns the time it took to run.
    Inputs:
        - function: The function to be timed
        - inputs: The inputs to be used for the function
        - number: The number of times to run the function (default is 1)
    Returns:
        - The time it took to run the function (in seconds)
    """    
    def function_timer(self, function, inputs, number=1):
        timer = timeit.Timer(lambda: function(*inputs))
        return timer.timeit(number=number)
    """
    The run_single_function function runs a single function and prints the time it took to run.
    Inputs:
        - function: The function to be timed
        - name: The name of the function
        - inputs: The inputs to be used for the function
        - i: The index of the function in the list of functions
    """
    def run_single_function(self, function, name, inputs, i):
        time = self.function_timer(function, inputs)
        print(f'{name} took {time} seconds to run')
        self.execution_time[i] = time
    """
    The reset_execution_time function resets the execution time to -1 for all algorithms.
    """    
    def reset_execution_time(self):
        self.execution_time = [-1, -1, -1, -1, -1, -1]
    """
    The benchmark function runs all the algorithms and prints the time it took to run. Also updates the execution_time list.
    """
    def benchmark(self):
        self.reset_execution_time()
        for i in range(len(self.list_of_functions)):
            if i not in self.exclusion_list:
                self.run_single_function(self.list_of_functions[i], self.names[i], self.inputs[i], i)
    """
    The graph_execution_times function graphs the execution times of the algorithms. Excludes the algorithms in the exclusion list.
    """                            
    def graph_execution_times(self):
        compared_names = [self.names[i] for i in range(len(self.names)) if i not in self.exclusion_list]
        compared_times = [self.execution_time[i] for i in range(len(self.execution_time)) if i not in self.exclusion_list]

        fig, ax1 = plt.subplots()
        blue_bars = ax1.barh(compared_names, compared_times, color='blue', height=0.5, alpha=0.5)
        ax1.set_xlabel('Execution Time (s)')

        plt.show()

