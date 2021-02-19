from abc import ABC, abstractmethod
from algo.dstruct import graph
from algo.dstruct.graph import EWDiGraph, EWGraph, FlowEdge, FlowNetwork
from collections import deque
import algo.dstruct as dstruct
from algo.dstruct import IndexedMinHeap
from algo.dstruct import WeightedQuickUnion
import warnings


__all__ = ["DFS", "UndirectedDFS", "DirectedDFS"]

# Algorithms related to DFS
class DFS(ABC):
    """Parent class for UndirectedDFS and DirectedDFS to inherit."""
    def __init__(self):
        self._reset()

    @abstractmethod
    def _reset(self):
        self._graph = None
        self._visited = None


    @abstractmethod
    def has_cycle(self, graph):
        """Given a graph, return if there is cycle in the graph."""
        pass
            

    def _dfs(self, s):
        if not self._visited[s]:
            self._visited[s] = True
            for v in self._graph.adj(s):
                self._dfs(v)

    def is_connected(self, graph, s, v):
        """Given a graph and two vertices return if the two vertices are connected."""
        self._graph = graph 
        self._visited = [False] * self._graph.V
        self._dfs(s)
        result = self._visited[v]
        self._reset()

        return result



class UndirectedDFS(DFS):
    def has_cycle(self, graph):
        """Given an undirected graph, return if there is at least a cycle in the graph."""
        self._graph = graph
        self._visited = [False] * self._graph.V
        result = False
        for s in range(self._graph.V):
            if not self._visited[s]:
                if self._has_cycle_dfs(s, None):
                    result = True
                    break
        self._reset()
        return result

    def _has_cycle_dfs(self, s, parent):
        self._visited[s] = True
        for v in self._graph.adj(s):
            if not self._visited[v]:
                if self._has_cycle_dfs(v, s):
                    return True
            elif v != parent:
                return True
        return False

    def cc(self, graph):
        """Given a graph, return a list of graph.V integers, 
        each specifying the group the corresponding vertex belongs to.
        """
        self._graph = graph
        self._group_id = -1
        self._groupings = [None] * self._graph.V
        self._visited = [False] * self._graph.V
        for s in range(self._graph.V):
            if not self._visited[s]:
                self._group_id += 1
                self._cc_dfs(s)
        groupings = self._groupings
        self._reset()
        return groupings

    
    def _cc_dfs(self, s):
        self._visited[s] = True
        self._groupings[s] = self._group_id
        for v in self._graph.adj(s):
            if not self._visited[v]:
                self._cc_dfs(v)

        
    def _reset(self):
        super()._reset()


class DirectedDFS(DFS):
    def has_cycle(self, graph):
        """Given a directed graph, return if there is at least one cycle present in the graph."""
        self._graph = graph
        self._visited = [False] * self._graph.V
        self._instack = [False] * self._graph.V
        result = False
        for s in range(self._graph.V):
            if not self._visited[s]:
                if self._has_cycle_dfs(s):
                    result = True
                    break
        self._reset()
        return result

    def _has_cycle_dfs(self, s):
        self._visited[s] = True
        self._instack[s] = True
        for v in self._graph.adj(s):
            if not self._visited[v]:
                if self._has_cycle_dfs(v):
                    return True
            elif self._instack[v]:
                return True
        self._instack[s] = False
        return False

    def topological_sort(self, graph, reverse = False):
        """Given a directed graph, return a list of graph.V integers,
        specifiying topological order.
        """
        if self.has_cycle(graph):
            warnings.warn("The graph is not acyclic, topological sort is not possible. \
                None is returned.")
            return None

        order = self._dfs_order(graph)
        if not reverse:
            order = list(reversed(order))
        return order

    def _dfs_order(self, graph):
        self._graph = graph
        self._visited = [False] * self._graph.V
        self._reverse_order = []
        for s in range(self._graph.V):
            if not self._visited[s]:
                self._topo_dfs(s)
        order = self._reverse_order
        self._reset()
        return order


    def _topo_dfs(self, s):
        self._visited[s] = True
        for v in self._graph.adj(s):
            if not self._visited[v]:
                self._topo_dfs(v)
        self._reverse_order.append(s)


    def scc(self, graph):
        """Given a directed graph, 
        return a list of Graph.V integers each representing the strongly connected component they belong to
        """
        order = self._dfs_order(graph.reverse())
        self._graph = graph
        self._visited = [False] * self._graph.V
        self._group_id = -1
        self._groupings = [None] * self._graph.V
        for s in list(reversed(order)):
            if not self._visited[s]:
                self._group_id += 1
                self._scc_dfs(s)
        groupings = self._groupings
        self._reset()
        return groupings


    def _scc_dfs(self, s):
        self._visited[s] = True
        self._groupings[s] = self._group_id
        for v in self._graph.adj(s):
            if not self._visited[v]:
                self._scc_dfs(v)

        

    def _reset(self):
        super()._reset()
        self._instack = None
        self._reverse_order = None
        self._group_id = None
        self._groupings = None



# Algorithm related to BFS
class BFS(ABC):
    """Parent class for DirectedBFS and UndirectedBFS to inherit."""
    def __init__(self):
        pass

    def is_connected(self, graph, s, v):
        """Check if v can be reached from s in the graph."""
        visited = [False] * graph.V
        Q = deque()
        Q.append(s)
        while Q:
            s = Q.popleft()
            for w in graph.adj(s):
                if visited[w]:
                    continue
                if w == v:
                    return True
                visited[w] = True
                Q.append(w)
        return False

    def _shortest_paths_from(self, graph, s):
        visited = [False] * graph.V
        pths = [None] * graph.V
        Q = deque()
        Q.append(s)
        pths[s] = s
        while Q:
            s = Q.popleft()
            for v in graph.adj(s):
                if visited[v]:
                    continue
                visited[v] = True
                Q.append(v)
                pths[v] = s
        return pths


    def shortest_path(self, graph, s, v):
        """Return the shortest path from s to v."""
        pths = self._shortest_paths_from(graph, s)
        if pths[v] is None:
            return None
        shortest_pth = []
        while v != s:
            shortest_pth.append(v)
            v = pths[v]
        shortest_pth.append(s)
        return list(reversed(shortest_pth))
                
class UndirectedBFS(BFS):
    """Breadth first search for undirected graph."""
    pass

class DirectedBFS(BFS):
    """Breadth first search for directed graph."""
    pass


class EWSingleSourceSP(ABC):
    """Parent class for single source shortest pass on edge weighted graph."""
    def __init__(self, graph, s):
        """Provide a graph of EWDiGraph or EWGraph and a source vertex."""
        self.reset(graph, s)

    @abstractmethod
    def path_to(self, v):
        """Return a list of edges, vertices and distance that correspond 
        to the shortest path between the given source vertex and vertex v.

        To use this method, the child class must have self._edge_to, self._dist defined.
        """
        if v == self.s:
            return [v], [None], 0
        if self._edge_to[v] is None:
            return None, None, float('inf')
        pths = [v]
        edges = []
        dist = self._dist[v]
        while self._edge_to[v] is not None:
            e = self._edge_to[v]
            w = e.other(v)
            assert w != v, "Algorithm error, as self loop should not be in the path."
            edges.append(e)
            pths.append(w)
            v = w
        pths = list(reversed(pths))
        edges = list(reversed(edges))
        return pths, edges, dist

    @abstractmethod
    def dist_to(self, v):
        """Return the length of the shortest path from source vertex s to vertext v."""
        return self._dist[v]

    @abstractmethod
    def reset(self, graph, s):
        pass
        


class DijkstraShortest(EWSingleSourceSP):
    """Dijkstra shortest path with indexed heap."""
    def _relax(self, e, s):
        v = e.other(s)
        new_dist = self._dist[s] + e.weight
        if new_dist < self._dist[v]:
            self._dist[v] = new_dist
            self._edge_to[v] = e
            if self._heap.contains_idx(v):
                self._heap.update(v, new_dist)
            else:
                self._heap.insert(v, new_dist)

    def _run(self):
        self._dist[self.s] = 0
        self._heap.insert(self.s, 0)
        while self._heap:
            _, s = self._heap.pop(return_idx=True)
            for e in self._graph.adj(s):
                self._relax(e, s)

    def path_to(self, v):
        """Return a list of edges, vertices and distance that correspond 
        to the shortest path between the given source vertex and vertex v.

        """
        return super().path_to(v)

    def dist_to(self, v):
        """Return the length of the shortest path from source vertex s to vertext v."""
        return super().dist_to(v)
        
    def reset(self, graph, s):
        """Rerun Dijkstra shortest path with the new EWGraph/EWDiGraph and new source vertex."""

        assert isinstance(graph, (dstruct.EWDiGraph, dstruct.EWGraph)),\
            "Graph needs to be EWDiGraph or EWGraph"
        assert s < graph.V, f"Source vertex s {s} is not smaller than graph.V {graph.V}"

        for e in graph.edges:
            assert e.weight >= 0, f"Edge {e} is negative."
        self.s = s
        self._graph = graph
        self._heap = IndexedMinHeap()
        self._edge_to = [None] * self._graph.V
        self._dist = [float("inf")] * self._graph.V

        self._run()



class BellmanFordShortest(EWSingleSourceSP):
    """Bellman-Ford algorithm for shortest path.
    
    This is an efficient implementation using queue to record all vertices that were changed
    during the last round of update, avoiding redundant checks for all vertices.
    Negative cycle detection that is reachable from the source vertex is also included.
    """
    def reset(self, graph, s):
        """Rerun Bellman Ford shortest path with the new EWDiGraph and new source vertex."""

        assert isinstance(graph, dstruct.EWDiGraph),\
            "Graph needs to be EWDiGraph or EWGraph"
        assert s < graph.V, f"Source vertex s {s} is not smaller than graph.V {graph.V}"
        self._graph = graph
        self.s = s
        self._queue = deque()
        self._edge_to = [None] * self._graph.V
        self._dist = [float("inf")] * self._graph.V
        self._in_queue = [False] * self._graph.V
        self._num_relax = 0
        self._neg_cycle = []

        self._run()

    def _run(self):
        self._queue.append(self.s)
        self._in_queue[self.s] = True
        self._dist[self.s] = 0
        while self._queue and not self._neg_cycle:
            s = self._queue.popleft()
            self._in_queue[s] = False
            for e in self._graph.adj(s):
                self._relax(e, s)

    def _relax(self, e, s):
        v = e.other(s)
        new_dist = self._dist[s] + e.weight
        if new_dist < self._dist[v]:
            self._dist[v] = new_dist
            self._edge_to[v] = e
            if not self._in_queue[v]:
                # Ensure at any time only one copy of the v is in the queue,
                # else the memory may blow off.
                # It does not affect the algorithm as if the v is already in the queue,
                # it means that v's out-going edges will be relaxed.
                # As its distance value is already the newest, the next update will sort
                # of fast-forward to use the most updated value (we do not need to allocate
                # one update for the worse-than-current value).
                self._queue.append(v)
                self._in_queue[v] = True
        
        self._num_relax += 1
        if (self._num_relax % self._graph.V == 0):
            # Note: if a negative cycle exists, self._queue will never be empty,
            # so this condition will always be reached.
            self._check_neg_cycle()
    
    def _check_neg_cycle(self):
        visited = [False] * self._graph.V
        stack_set = set()
        stack = []
        for s in range(self._graph.V):
            if not visited[s]:
                stack.append(s)
                stack_set.add(s)
                while stack:
                    curr = stack[-1]
                    e = self._edge_to[curr]
                    # If curr vertex is the source vertex
                    # and there is no negative cycle involving source vertex
                    if e is None:
                        stack_set.remove(stack.pop(-1))
                        continue

                    v = e.other(curr)
                    # if the new v has not been visited
                    # visit this vertex
                    if not visited[v]:
                        stack.append(v)
                        stack_set.add(v)
                        visited[v] = True
                    # else we have visited
                    # check if it is on the stack
                    # if it is, then we found a negative cycle
                    elif v in stack_set:
                        self._neg_cycle.append(v)
                        while stack[-1] is not v:
                            self._neg_cycle.append(stack.pop(-1))
                        self._neg_cycle.append(v)
                        return
                    # else we have finished the depth first search wrt curr
                    # we can pop curr
                    else:
                        stack_set.remove(stack.pop(-1))
                        
    def negative_cycle(self):
        """Return a list representing the negative cycle reachable from source vertex.
        
        Empty list is returned if no such cycle exists.
        """
        return list(self._neg_cycle)
    

    def path_to(self, v):
        """If no negative cycle is reachable from s,
        return a list of edges, vertices and distance that correspond 
        to the shortest path between the given source vertex and vertex v.

        Else, a warning is issued and None is returned.
        """
        if self._neg_cycle:
            warnings.warn("There is negative cycle in the group. \
                No path is generated. None is returned")
            return None
        return super().path_to(v)

    def dist_to(self, v):
        """Return the length of the shortest path from source vertex s to vertext v.
        
        If a negative cylce is reachable from source vertex s, None is returned.
        """
        if self._neg_cycle:
            warnings.warn("There is negative cycle in the group. \
                No path is generated. None is returned")
            return None
        return super().dist_to(v)

    def has_neg_cycle(self):
        """Return if there is any negative cycle reachable from source vertex s."""
        return len(self._neg_cycle) > 0




class EWAllPairsSP(ABC):
    def __init__(self, graph):
        self.reset(graph)

    @abstractmethod
    def path_to(self, s, v):
        """Return a list of edges, vertices and distance that correspond
        to the shortest path between the source vertex s and vertex v.

        To use this method, the child class must have self._edge_to, self._dist defined.
        """
        edge_to = self._edge_to[s]
        if v == s:
            return [v], [None], 0
        if edge_to[v] is None:
            return None, None, float('inf')
        pths = [v]
        edges = []
        dist = self._dist[s][v]
        while edge_to[v] is not None:
            e = edge_to[v]
            w = e.other(v)
            assert w != v, "Algorithm error, as self loop should not be in the path."
            edges.append(e)
            pths.append(w)
            v = w
        pths = list(reversed(pths))
        edges = list(reversed(edges))
        return pths, edges, dist


    @abstractmethod
    def dist_to(self, s, v):
        """Return the length of the shortest path from source vertex s to vertext v."""
        return self._dist[s][v]


    @abstractmethod
    def reset(self, graph):
        """Provide a new graph and re-run the shortest path algorithm.
        
        Subclass must call this class in their reset method for input checking.
        """
        assert isinstance(graph, dstruct.EWDiGraph),\
            "Graph needs to be EWDiGraph"


    @abstractmethod
    def has_neg_cycle(self):
        """Return if there is negative cycle in the graph."""
        pass




class FloydWarshallShortest(EWAllPairsSP):
    """Floyd Warshall all paires shortest paths algorithm."""
    def reset(self, graph):
        """Provide a new EWDiGraph and re-run the Floyd Warshall shortest path algorithm."""
        super().reset(graph)
        self._graph = graph
        # self._edge_to[s][v] is the edge going to v when source vertex is s
        self._edge_to = [[None] * self._graph.V for _ in range(self._graph.V)]
        # self._dist_to[s][v] is the shortest distance from s to v
        self._dist = [[float('inf')] * self._graph.V for _ in range(self._graph.V)]
        self._neg_vertices = []
        self._run()

    def _run(self):
        for s in range(self._graph.V):
            self._dist[s][s] = 0
            for e in self._graph.adj(s):
                v = e.other(s)
                self._dist[s][v] = e.weight
                self._edge_to[s][v] = e

        # only vertices that are smaller than or equal to k can be used
        # for intermediate connections
        for k in range(self._graph.V):
            # self.
            for s in range(self._graph.V):
                for v in range(self._graph.V):
                    old_dist = self._dist[s][v]
                    new_dist = self._dist[s][k] + self._dist[k][v]
                    if new_dist < old_dist:
                        self._dist[s][v] = new_dist
                        self._edge_to[s][v] = self._edge_to[k][v]
        
        for s in range(self._graph.V):
            if self._dist[s][s] < 0:
                self._neg_vertices.append(s)

    def has_neg_cycle(self):
        """Return if there is negative cycle in the graph.

        Please do not use this for the sole purpose of checking if negative cycle exists,
        as this is not efficient for checking negative cycle. 
        This method is used to indicate if the shortest paths are correctly generated.
        """
        return len(self._neg_vertices) > 0

    def path_to(self, s, v):
        """Return the length of the shortest path from source vertex s to vertext v."""
        
        if self._neg_vertices:
            warnings.warn("There is negative cycle in the group. \
                No path is generated. None is returned")
            return None
        return super().path_to(s, v)

    def dist_to(self, s, v):
        """Return the length of the shortest path from source vertex s to vertext v."""
        
        if self._neg_vertices:
            warnings.warn("There is negative cycle in the group. \
                No path is generated. None is returned")
            return None
        return super().dist_to(s, v)



class JohnsonShortest(EWAllPairsSP):
    """Johnson's algorithm for all pairs shortest paths."""
    def reset(self, graph):
        """Provide a new EWDiGraph and re-run the Johnson's shortest path algorithm."""
        super().reset(graph)

        # create a new graph that has one additional vertex
        self._graph = graph.empty_graph(graph.V + 1)
        for e in graph.edges:
            self._graph.add_edge(e.from_edge(e))

        # add an edge of weight 0 from the additional vertex to all the original
        # vertices
        for v in range(self._graph.V - 1):
            self._graph.add_edge_(self._graph.V - 1, v, 0)

        # self._edge_to[s][v] is the edge going to v when source vertex is s
        # to be filled up in _run
        self._edge_to = []
        # self._dist_to[s][v] is the shortest distance from s to v
        # to be filled up in _run
        self._dist = []
        self._neg_cycle = []
        self._run()

    def has_neg_cycle(self):
        """Return if there is at least a negative cycle in the graph.
        
        Please do not use this for the sole purpose of checking if negative cycle exists,
        as this is not efficient for checking negative cycle. 
        This method is used to indicate if the shortest paths are correctly generated.
        """
        return len(self._neg_cycle) > 0

    def negative_cycle(self):
        """Return a negative cycle that is detected in the graph."""
        return list(self._neg_cycle)

    def _reweight(self):
        bellman_ford = BellmanFordShortest(self._graph, self._graph.V - 1)
        self._neg_cycle = bellman_ford.negative_cycle()
        if self._neg_cycle:
            return

        self._weights = bellman_ford._dist

        graph = self._graph.empty_graph(self._graph.V - 1)
        for s in range(self._graph.V - 1):
            for e in self._graph.adj(s):
                e.weight += self._weights[e.source] - self._weights[e.sink]
                graph.add_edge(e)
        self._graph = graph


    def _run(self):
        self._reweight()
        # if no negative cycle detected by Bellman-Ford
        # use Dijkstra self._graph.V times
        if not self._neg_cycle:
            for s in range(self._graph.V):
                dijkstra = DijkstraShortest(self._graph, s)
                self._dist.append(dijkstra._dist)
                self._edge_to.append(dijkstra._edge_to)
        
            for s in range(self._graph.V):
                for v in range(self._graph.V):
                    self._dist[s][v] -= self._weights[s] - self._weights[v]

            for e in self._graph.edges:
                e.weight -= self._weights[e.source] - self._weights[e.sink]


    def dist_to(self, s, v):
        """Return the length of the shortest path from source vertex s to vertext v."""

        if self._neg_cycle:
            warnings.warn("There is negative cycle in the group. \
                No path is generated. None is returned")
            return None
        return super().dist_to(s, v)


    def path_to(self, s, v):
        """Return the length of the shortest path from source vertex s to vertext v."""

        if self._neg_cycle:
            warnings.warn("There is negative cycle in the group. \
                No path is generated. None is returned")
            return None
        return super().path_to(s, v)




class MSTTemplate(ABC):
    def __init__(self, graph, *args, **kargs):
        """Graph must be edge weighted undirected graph."""
        self.reset(graph, *args, **kargs)

    def edges(self):
        """Return a list of edges that are in the MST."""
        return self._edges

    def edges_(self, v):
        """Return edges that are connected to the given vertex in the MST."""
        return self._edges_[v]

    def dist(self):
        """Return the total length of the MST."""
        return self._dist

    @abstractmethod
    def reset(self, graph, *args, **kargs):
        assert isinstance(graph, dstruct.EWGraph),\
            "MST algorithms only operate on edge weighted undirected graph."
        self._graph = graph
        # self._edges is a list (of lists of) each vertices' edges that are in the MST
        self._edges_ = [[] for _ in range(self._graph.V)]
        # self.edges is a list of edges that are in the MST
        self._edges = []
        # self._dist is the
        self._dist = 0




class PrimMST(MSTTemplate):
    """Prim's algorithm for minimum spanning tree."""
    def _update(self, e, v):
        other = e.other(v)
        weight = e.weight
        if other in self._heap:
            if weight < self._heap.get(other):
                self._heap.update(other, weight)
                self._edges_to[other] = e


    def _run(self):
        for v in range(self._graph.V):
            self._heap.insert(v, float("inf"))
        # initialise the tree with a random vertex
        _, s = self._heap.pop(return_idx=True)
        for e in self._graph.adj(s):
            self._update(e, s)
        # prim's algo
        while self._heap:
            _, v = self._heap.pop(return_idx=True)
            e = self._edges_to[v]
            self._edges.append(e)
            self._edges_[e.v].append(e)
            self._edges_[e.w].append(e)
            self._dist += e.weight
            for e in self._graph.adj(v):
                self._update(e, v)


    def reset(self, graph):
        """Rerun Prim's MST algorithm on the new graph."""
        super().reset(graph)
        self._heap = IndexedMinHeap()
        # self._edges_to is a list of currently shortest edges connecting the vertices
        # to the tree we are building
        self._edges_to = [None] * self._graph.V
        self._run()


class KruskalMST(MSTTemplate):
    def _run(self, target):
        edges = list(self._graph.edges)
        edges.sort(key=lambda e: e.weight)
        edge_ptr = 0
        while True:
            e = edges[edge_ptr]
            v, w = e.v, e.w
            if not self._UF.connected(v, w):
                self._UF.union(v, w)
                self._edges.append(e)
                self._edges_[v].append(e)
                self._edges_[w].append(e)
                self._num_clusters -= 1
            
            if self._num_clusters <= target:
                break
            if edge_ptr == len(edges):
                raise Exception(
                    "Finished all edges but target number of clusters is not achieved.")
            edge_ptr += 1


    def reset(self, graph, num_clusters = 1):
        """Rerun Kruskal's MST algorithm on the new graph.
        When number of clusters is reached, Kruskal algorithm will be stopped.
        number_clusters = 1 is the standard minimum spanning tree.
        number_clusters > 1 is for maximum minimum distance clustering.
        """
        super().reset(graph)
        self._UF = WeightedQuickUnion(self._graph.V)
        self._num_clusters = self._graph.V
        self._run(num_clusters)

    def groupings(self):
        """Return the groups each vertex belongs to."""
        return list(self._UF.groupings())





class FordFulkerson:
    """Ford Fulkerson algorithm for s-t mincut/maxflow."""
    def __init__(self, graph, source, sink):
        """Provide a FlowNetwork graph, a source and a sink vertices.
        
        For the FlowNetwork, there should be no edge pointing towards the source vertex
        and no edge pointing away from the sink vertex.
        """
        self.reset(graph, source, sink)

    def has_augment_path(self):
        """Return whether there is an augmenting path from the given source vertex from the sink vertex."""
        self._visited = [False] * self.graph.V
        self._edge_to = [None] * self.graph.V
        queue = deque()
        queue.append(self.source)
        self._visited[self.source] = True
        while queue:
            s = queue.popleft()
            for edge in self.graph.adj(s):
                v = edge.other(s)
                if not self._visited[v] and edge.res_capacity_to(v) > 0:
                    self._edge_to[v] = edge
                    self._visited[v] = True
                    queue.append(v)
        return self._visited[self.sink]

    
    def run(self):
        """Run the Ford Fulkerson algorithm on the given flow network."""
        while (self.has_augment_path()):
            bottleneck = float("inf")
            v = self.sink
            while v != self.source:
                edge = self._edge_to[v]
                s = edge.other(v)
                if edge.res_capacity_to(v) < bottleneck:
                    bottleneck = edge.res_capacity_to(v)
                v = s

            v = self.sink
            while v != self.source:
                edge = self._edge_to[v]
                s = edge.other(v)
                edge.add_res_flow_to(v, bottleneck)
                v = s
        self._has_run = True


    def incut(self, v):
        """Return whether the vertex v can be reached from source vertex.
        
        i.e. given the computed min cut, whether v is in the same cut as the source vertex s.
        """
        if not self._has_run:
            self.run()
        return self._visited[v]


    def max_flow(self):
        """Return the value of max flow from source vertex to sink vertex."""
        if not self._has_run:
            self.run()

        if self._value is None:
            self._value = 0
            # compute the flow across cut
            for edge in self.graph.adj(self.sink):
                self._value += edge.flow
        
        return self._value

        


    def reset(self, graph, source, sink):
        """Re-initialise Ford Fulkenson algorithm with respect to the given graph and source/sink vertices."""
        assert isinstance(graph, FlowNetwork), \
            "Ford Fulkerson algorithm can only be applied to FlowNetwork"
        for edge in graph.adj(source):
            assert edge.source == source, \
                f"There should be no edge pointing towards source vertex {source}"
        
        for edge in graph.adj(sink):
            assert edge.sink == sink, \
                f"There should be no edge pointing from the sink vertex {sink}"
        
        self.graph = graph.from_graph(graph)
        self.source, self.sink = source, sink
        self._has_run = False
        self._value = None
        




if __name__ == "__main__":
    from algo.dstruct.graph import Graph, DiGraph, FlowNetwork

    # # DFS

    # # Undirected Graph
    # print("\n*Undirected Group Search:\n")
    # udfs = UndirectedDFS()
    # # checking has_cycle
    # adj_list = [{1, 3, 4}, {0, 2}, {1}, {0, }, {0, }]
    # ug = Graph.from_adj_list(adj_list)
    # print("Checking cycle (expect True):")
    # print(udfs.has_cycle(ug))

    # adj_list = [{1, 3, 4}, {0, 2}, {1}, {}, {}]
    # ug = Graph.from_adj_list(adj_list)
    # print("Checking cycle (expect False):")
    # print(udfs.has_cycle(ug))

    # # checking connected components
    # print("Checking connected components:")
    # adj_list = [{1, 2}, {0, 2}, {0, 1}, {3, 1}, {}, {6, 7}, {5}, {5}]
    # ug = Graph.from_adj_list(adj_list)
    # print("Expect 0,0,0,0,1,2,2,2")
    # print(udfs.cc(ug))


    # # Directed Graph
    # print("\n\n*Directed Group Search:\n")
    # ddfs = DirectedDFS()

    # # checking has_cycle
    # adj_list = [{1, 2}, {3, 4}, {0}, {}, {}]
    # dg = DiGraph.from_adj_list(adj_list)
    # print("Checking cycle (expect True):")
    # print(ddfs.has_cycle(dg))

    # adj_list = [{1, 2}, {3, 4}, {}, {}, {}]
    # dg = DiGraph.from_adj_list(adj_list)
    # print("Checking cycle (expect False):")
    # print(ddfs.has_cycle(dg))

    # # checking topological sort
    # adj_list = [{1, 2}, {2, 3, 5}, {}, {4}, {6}, {}, {}]
    # dg = DiGraph.from_adj_list(adj_list)
    # print("Checking topological sort:")
    # print(f"graph is \n{dg}")
    # print(ddfs.topological_sort(dg))

    # # checking strongly connected componenets
    # adj_list = [{2}, {0}, {1, 3}, {4}, {5}, {3, 6}, {5, 7}, {8}, {}]
    # dg = DiGraph.from_adj_list(adj_list)
    # print("Checking strongly connected components:")
    # print("Expecting 3,3,3,2,2,2,2,1,0")
    # print(ddfs.scc(dg))


    # # BFS
    

    # # Undirected Graph
    # adj_list = [[1, 2, 4], [0, 5], [0, 4], [2, 7], [0, 2, 6], [1, 6], [4, 5], [3], []]
    # ug = Graph.from_adj_list(adj_list)

    # ubfs = UndirectedBFS()
    # # is_connected
    # print("Checking is_connected")
    # print("Expect True: ", ubfs.is_connected(ug, 0, 6))
    # print("Expect False: ", ubfs.is_connected(ug, 8, 6))

    # # checking shortest path
    # print("Checking shortest path")
    # print("Expect [0, 4, 6]: ", ubfs.shortest_path(ug, 0, 6))
    # print("Expect None: ", ubfs.shortest_path(ug, 0, 8))


    # # # Dijkstra shortest path
    # from algo.dstruct import EWGraph, EWDiGraph
    # adj_list = [[(1, 1), (3, 8), (0, 0)], [(2, 6), (3, 5), (4, 2), (1, 0)], [(4, 3), (2, 0)], [(3, 0)], [(2, 3), (4, 0)]]
    # g = EWDiGraph.from_adj_list(adj_list)
    # dijsktra = DijkstraShortest(g, 0)
    # pth, edges, dist = dijsktra.path_to(4)
    # print(pth)
    # print(", ".join(str(e) for e in edges))
    # print(dist)

    # adj_list = [[(1, 1), (3, 4), (2, 5), (5, 7)], [(3, 2), (4, 5)], [(3, 1)], [(4, 1)], [(5, 2)], []]
    # g = EWGraph.from_adj_list(adj_list)
    # dijsktra.reset(g, 0)
    # pth, edges, dist = dijsktra.path_to(2)
    # print(pth)
    # print(", ".join(str(e) for e in edges))
    # print(dist)


    # # Prim's MST
    # print("Prim MST")
    # g = EWGraph.from_adj_list([[(1, 0.1), (2, 0.2), (0, 0.0)], \
    #     [(2, 0.7), (3, 0.6), (5, 0.9), (1, 0.0)], [(3, 0.4), (4, 0.3), (2, 0.0)], \
    #         [(4, 0.8), (5, 0.5)], [], []])
    # prim = PrimMST(g)
    # for e in prim.edges():
    #     print(e.weight)


    # # Kruskal's MST
    # print("Kruskal MST")
    # g = EWGraph.from_adj_list([[(1, 0.1), (2, 0.2), (0, 0.0)],
    #                            [(2, 0.7), (3, 0.6), (5, 0.9), (1, 0.0)], [
    #     (3, 0.4), (4, 0.3), (2, 0.0)],
    #     [(4, 0.8), (5, 0.5)], [], []])
    # kruskal = KruskalMST(g)
    # for e in kruskal.edges():
    #     print(e.weight)
    # kruskal.reset(g, 2)
    # for e in kruskal.edges():
    #     print(e.weight)
    # print(kruskal.groupings())


    # # Bellman-Ford shortest path
    # print("Bellman-Ford shortest path")
    # g = EWDiGraph.from_adj_list([[(1, 2), (2, 1)], [(3, 1), (5, 5)], \
    #     [(1, 8), (4, 6)], [(4, 2)], [(2, -6)], []])
    # bellman_ford = BellmanFordShortest(g, 0)
    # print(bellman_ford.negative_cycle())
    # pth, edges, dist = bellman_ford.path_to(2)
    # print(pth)
    # print(", ".join(str(e) for e in edges))
    # print(dist)

    # g = EWDiGraph.from_adj_list([[(1, 2), (2, 1)], [(3, 1), (5, 5)],
    #                              [(1, 2), (4, 6)], [(4, 2)], [(2, -6)], []])
    # bellman_ford = BellmanFordShortest(g, 0)
    # print(bellman_ford.negative_cycle())


    # # Floyd-Warshall
    # print("Floyd-Warshall")
    # g = EWDiGraph.from_adj_list([[(1, 2), (2, 1)], [(3, 1), (5, 5)],
    #                              [(1, 8), (4, 6)], [(4, 2)], [(2, -6)], []])
    # floyd_warshall = FloydWarshallShortest(g)
    # for s in range(g.V):
    #     bellman_ford = BellmanFordShortest(g, s)
    #     for v in range(g.V):
    #         assert floyd_warshall.dist_to(s, v) == bellman_ford.dist_to(v)

    # for s in range(g.V):
    #     bellman_ford = BellmanFordShortest(g, s)
    #     for v in range(g.V):
    #         pth_f, edges_f, dist_f = floyd_warshall.path_to(s, v)
    #         pth_b, edges_b, dist_b = bellman_ford.path_to(v)
    #         if edges_f is not None:
    #             print("Floyd  : ", pth_f,
    #                   " ".join([str(e) for e in edges_f]))
    #         else:
    #             print("Floyd  : ", pth_f,
    #                   edges_f)
    #         if edges_b is not None:
    #             print("Bellman: ", pth_b,
    #                   " ".join([str(e) for e in edges_b]))
    #         else:
    #             print("Bellman: ", pth_b,
    #                   edges_b)

    # g = EWDiGraph.from_adj_list([[(1, 2), (2, 1)], [(3, 1), (5, 5)],
    #                              [(1, 2), (4, 6)], [(4, 2)], [(2, -6)], []])
    # bellman_ford = BellmanFordShortest(g, 0)
    # floyd_warshall = FloydWarshallShortest(g)
    # print(bellman_ford.has_neg_cycle())
    # print(floyd_warshall.has_neg_cycle())
    

    # # Johnson's Algorithm
    # print("Johnson's algorithm")
    # g = EWDiGraph.from_adj_list([[(1, 2), (2, 1)], [(3, 1), (5, 5)],
    #                              [(1, 8), (4, 6)], [(4, 2)], [(2, -6)], []])
    # floyd_warshall = FloydWarshallShortest(g)
    # johnson = JohnsonShortest(g)
    # for s in range(g.V):
    #     for v in range(g.V):
    #         assert floyd_warshall.dist_to(s, v) == johnson.dist_to(s, v)

    # for s in range(g.V):
    #     for v in range(g.V):
    #         pth_f, edges_f, dist_f = floyd_warshall.path_to(s, v)
    #         pth_j, edges_j, dist_j = johnson.path_to(s, v)
    #         if edges_f is not None:
    #             print("Floyd  : ", pth_f,
    #                   " ".join([str(e) for e in edges_f]))
    #         else:
    #             print("Floyd  : ", pth_f,
    #                   edges_f)
    #         if edges_j is not None:
    #             print("Johnson: ", pth_j,
    #                   " ".join([str(e) for e in edges_j]))
    #         else:
    #             print("Johnson: ", pth_j,
    #                   edges_j)

    # g = EWDiGraph.from_adj_list([[(1, 2), (2, 1)], [(3, 1), (5, 5)],
    #                              [(1, 2), (4, 6)], [(4, 2)], [(2, -6)], []])
    # johnson = JohnsonShortest(g)
    # floyd_warshall = FloydWarshallShortest(g)
    # print(johnson.has_neg_cycle())
    # print(floyd_warshall.has_neg_cycle())


    # Ford Fulkerson
    print("Ford Fulkerson")
    flownet = FlowNetwork.from_adj_list([
        [(1, 10), (2, 5), (3, 15)],
        [(2, 4), (4, 9), (5, 15)],
        [(3, 4), (5, 8)],
        [(6, 16)],
        [(5, 15), (7, 10)],
        [(6, 15), (7, 10)],
        [(2, 6), (7, 10)],
        []
    ])
    print(flownet)
    ff = FordFulkerson(flownet, 0, 7)
    print(ff.max_flow())
    print(ff.graph)
    print(ff._visited)
    print(ff.incut(3))