from abc import ABC, abstractmethod


__all__ = ["Graph", "DiGraph", "EWGraph", "EWDiGraph", "Edge", "DiEdge", "FlowEdge", "FlowNetwork"]

class GraphTemplate(ABC):
    """Parent class for Graph and DiGraph to inherit."""
    def __init__(self, V):
        """Create a graph of V vertices."""
        self._V = V
        # adjacency list
        # Note: self._adj = [set()] * self._V  is a list of same id of the same set
        # it is shallow copy
        self._adj = [[] for _ in range(self._V)]
        self._edges = []
    
    @abstractmethod
    def _type(self):
        # This is meant to be a compulsory class attribute for subclass
        pass

    @property
    def type(self):
        """Return the type of the graph, either "directed" or "undirected"."""
        return self._type

    @property
    def V(self):
        """Return the number of vertices in the graph."""
        return self._V

    @property
    def E(self):
        """Return the number of edges in the graph."""
        return len(self._edges)
    
    @property
    def edges(self):
        """Return a list of strings of form "a-b", indicating an edge from a to b."""
        return self._edges

    @abstractmethod
    def add_edge(self, v1, v2):
        """Add an edge from v1 to v2."""
        pass
    
    def adj(self, v):
        """Return the adjacent list of vertex v."""
        return self._adj[v]


    @classmethod
    def from_adj_list(cls, adj_list):
        """Build and return a graph from the adjacent list.
        adj_list should be of [[1], [0, 1, 2], [1]] which indicats edges of
        0->1, 1->0, 1->1, 1->2, 2->1. For undirected graph, the adj_list should show
        reciprocal relationship i.e. if 1 is in the 0th list, 0 should be in the first list too.

        For undirected graph, only vertices that are larger or equal to the vertices
        represented by the list indices position is needed in the list. i.e. [[1], []] is the equivalent to, 
        (in fact, better than) than [[1], [0]] in this implementation.
        """
        graph = cls(len(adj_list))
        if cls._type == "undirected":
            for one_node, adj_nodes in enumerate(adj_list):
                for other_node in adj_nodes:
                    if other_node >= one_node:
                        graph.add_edge(one_node, other_node)

        else:
            for one_node, adj_nodes in enumerate(adj_list):
                for other_node in adj_nodes:
                    graph.add_edge(one_node, other_node)
        
        return graph

    @classmethod
    def from_graph(cls, graph):
        """Build and return an exact copy of the given graph."""
        new_graph = cls(graph.V)
        if cls._type == "undirected":
            for s in range(graph.V):
                for v in graph.adj(s):
                    if v >= s:
                        new_graph.add_edge(s, v)
        else:
            for s in range(graph.V):
                for v in graph.adj(s):
                    new_graph.add_edge(s, v)

        return new_graph

    def __str__(self):
        string = f"{self.type} graph of {self._V} nodes and {self.E} edges \n"
        string += "with edges:\n"
        temp_string = []
        for e in self._edges:
            temp_string.append(e)
            temp_string.append("\n")
        return string + "".join(temp_string)



class Graph(GraphTemplate):
    _type = "undirected"
    def add_edge(self, v1, v2):
        """Add an edge from v1 to v2."""
        assert v1 < self._V and v2 < self._V, \
            f"in add_edge: one of of the nodes {v1}, {v2} is not in the graph of {self._V} nodes"
        self._adj[v1].append(v2)
        self._adj[v2].append(v1)
        if v1 < v2:
            self._edges.append(f"{v1}-{v2}")
        else:
            self._edges.append(f"{v2}-{v1}")




class DiGraph(GraphTemplate):
    _type = "directed"
    def add_edge(self, v1, v2):
        """Add an edge from v1 to v2."""
        assert v1 < self._V and v2 < self._V, \
            f"in add_edge: one of of the nodes {v1}, {v2} is not in the graph of {self._V} nodes"
        self._adj[v1].append(v2)
        self._edges.append(f"{v1}-{v2}")



    def reverse(self):
        """Return a new graph with all the edges reversed."""
        graph = DiGraph(self._V)
        for one_node in range(self._V):
            adj_nodes = self._adj[one_node]
            for other_node in adj_nodes:
                graph.add_edge(other_node, one_node)
        return graph




class EdgeTemplate(ABC):
    """Parent class for Edge and DiEdge to inherit."""
    def __init__(self, v, w, weight = 0):
        """Create an Edge between v and w of weight.
        For DiEdge, the edge is pointing from v to w.
        """
        self._v, self._w, self._weight = v, w, weight

    @abstractmethod
    def _type(self):
        # This is meant to be a compulsory class attribute for subclass
        pass
    
    @property
    def type(self):
        """Indicating if the edge is directed or undirected."""
        return self._type

    @property
    def weight(self):
        """Weight of the edge."""
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight

    @classmethod
    def from_edge(cls, edge):
        """Clone an edge from the given edge."""
        return cls(edge._v, edge._w, edge._weight)

    def other(self, v):
        """Return the other vertex of the edge given this vertex."""
        if self._v == v:
            return self._w
        return self._v


    

class Edge(EdgeTemplate):
    _type = "undirected"
    @property
    def v(self):
        """One of the vertex that is different from self.w"""
        return self._v

    @property
    def w(self):
        """One of the vertex that is different from self.v"""
        return self._w

    def __str__(self):
        return f"{self._v}-{self._w} ({self._weight})"


class DiEdge(EdgeTemplate):
    _type = "directed"

    @property
    def source(self):
        """Source vertex of the directed edge."""
        return self._v
    
    @property
    def sink(self):
        """Sink vertex of the directed edge."""
        return self._w
    
    def __str__(self):
        return f"{self._v}->{self._w} ({self._weight})"



class EWGraphTemplate(ABC):
    """Parent class of EWGraph and EWDiGraph to inherit."""
    def __init__(self, V):
        self._V = V
        self._edges = []
        self._adj = [[] for _ in range(self._V)]

    @abstractmethod
    def _type(self):
        # This is meant to be a compulsory class attribute for subclass
        pass

    @property
    def type(self):
        return self._type

    @property
    def V(self):
        """Number of vertices in the graph."""
        return self._V

    @property
    def E(self):
        """Number of edges in the graph."""
        return len(self._edges)

    @property
    def edges(self):
        """Return all the Edge or DiEdge the graph contains in a list."""
        return self._edges

    def adj(self, v):
        """Return the adjacency list of vertex v."""
        return self._adj[v]

    @abstractmethod
    def add_edge(self, e):
        pass        

    @abstractmethod
    def add_edge_(self, v, w, weight):
        """A helper function for adding edge.
        It should create an edge of suitable type and use self.add_edge() to add it to the graph.
        """
        pass

    @abstractmethod
    def from_adj_list(cls, adj_list):
        """Create a new graph from the given adj_list."""
        pass

    @classmethod
    def from_graph(cls, graph):
        """Clone a graph from the given graph."""
        new_graph = cls(graph.V)
        for e in graph.edges:
            new_graph.add_edge(e.from_edge(e))
        return new_graph

    @classmethod
    def empty_graph(cls, V):
        """Create a new graph of the same class as this instance with V vertices."""
        return cls(V)

    def __str__(self):
        string = f"{self.type} edge-weighted graph of {self._V} nodes and {self.E} edges \n"
        string += "with edges:\n"
        temp_string = []
        for e in self._edges:
            temp_string.append(str(e))
            temp_string.append("\n")
        return string + "".join(temp_string)


class EWGraph(EWGraphTemplate):
    """Edge weighted graph"""

    _type = "undirected"
    def add_edge(self, e):
        """Add an edge e to the graph."""
        assert isinstance(e, Edge), "Edge e provided is of the wrong type."
        assert e.v < self.V and e.w < self.V, \
            f"Edge {e} 's vertices are larger than graph.V {self.V}"
        self._edges.append(e)
        self._adj[e.v].append(e)
        self._adj[e.w].append(e)
    
    def add_edge_(self, v, w, weight):
        """A helper function for adding edge.
        It creates an instance of Edge and use self.add_edge() to add it to the graph.
        """
        e = Edge(v, w, weight)
        self.add_edge(e)

    @classmethod
    def from_adj_list(cls, adj_list):
        """Build a new graph from adj_list.
        e.g. [[(1, 0.5)], [(0, 0.5)]] means a graph of 2 vertices with an edges of weight 0.5 between 
        vertex 0 and 1. In this implementation, only vertices that are larger or equal to the vertices
        represented by the list indices position is needed in the list. i.e. [[(1, 0.5)], []] is the equivalent to, 
        (in fact, better than) than [[(1, 0.5)], [(0, 0.5)]] in this implementation.
        """
        graph = cls(len(adj_list))
        for v, adj_ in enumerate(adj_list):
            for w, weight in adj_:
                if w >= v:
                    graph.add_edge_(v, w, weight)
        return graph


class EWDiGraph(EWGraphTemplate):
    """Edge weighted directed graph"""

    _type = "directed"

    def add_edge(self, e):
        """Add an edge e to the graph."""
        assert isinstance(e, DiEdge), "Edge e provided is of the wrong type."
        assert e.source < self.V and e.sink < self.V, \
             f"Edge {e} 's vertices are larger than graph.V {self.V}"
        self._edges.append(e)
        self._adj[e.source].append(e)

    def add_edge_(self, v, w, weight):
        """A helper function for adding edge.
        It creates an instance of DiEdge and use self.add_edge() to add it to the graph.
        """
        e = DiEdge(v, w, weight)
        self.add_edge(e)

    @classmethod
    def from_adj_list(cls, adj_list):
        """Build a new graph from adj_list.
        e.g. [[(1, 0.5)], [(2, 0.8)], [(0, 3.5)]] means a graph of 3 vertices with three edges 0->1 (0.5), 1->2 (0.8) and
        2->0 (3.5). 
        """
        graph = cls(len(adj_list))
        for v, adj_ in enumerate(adj_list):
            for w, weight in adj_:
                graph.add_edge_(v, w, weight)
        return graph


class FlowEdge:
    _type = "flow"
    def __init__(self, v, w, capacity, flow=0):
        """Create an Edge between v and w of weight.
        For DiEdge, the edge is pointing from v to w.
        """
        if flow > capacity:
            raise ValueError(f"Given flow {flow} is more than capacity {capacity}")
        self._v, self._w, self._capacity, self._flow = v, w, capacity, flow

    @property
    def type(self):
        """Indicating if the edge is directed or undirected."""
        return self._type

    @property
    def source(self):
        """Source vertex of the directed edge."""
        return self._v

    @property
    def sink(self):
        """Sink vertex of the directed edge."""
        return self._w

    @property
    def flow(self):
        """Return the flow in this edge."""
        return self._flow

    def res_capacity_to(self, v):
        """Return the residual capacity in the direction of the vertex.
        
        If v is the sink vertex, the remaining capacity of the edge is returned.
        if v is the source vertex, the edge's flow is returned.
        """
        if v == self.source:
            return self._flow
        elif v == self.sink:
            return self._capacity - self._flow
        else:
            raise ValueError(f"{v} is not an endpoint for {self}")

    def add_res_flow_to(self, v, delta):
        """Add residual flow in the direction of given vertex.
        
        If the given vertex is the sink vertex, delta amount of flow is added to the edge's flow.
        If the given vertex is the source vertex, deltat amount of flow is removed from the edge's flow.
        """
        if v == self.source:
            assert delta <= self._flow, \
                f"cannot add {delta} residual flow from {self._v} to {self._w} for {self}"
            self._flow -= delta
        elif v == self.sink:
            assert delta <= self._capacity - self._flow, \
                f"cannot add {delta} residual flow from {self._w} to {self._v} for {self}"
            self._flow += delta
        else:
            raise ValueError(f"{v} is not an endpoint for {self}")

    def __str__(self):
        return f"{self._v}->{self._w} ({self._flow}/{self._capacity})"

    @classmethod
    def from_edge(cls, edge):
        """Clone an edge from the given edge."""
        return cls(edge._v, edge._w, edge._capacity, edge._flow)

    def other(self, v):
        """Return the other vertex of the edge given this vertex."""
        if self._v == v:
            return self._w
        return self._v


class FlowNetwork(EWGraphTemplate):
    """Flow Network"""

    _type = "flow"

    def add_edge(self, e):
        """Add an edge e to the graph."""
        assert isinstance(e, FlowEdge), "Edge e provided is of the wrong type."
        assert e.source < self.V and e.sink < self.V, \
            f"Edge {e} 's vertices are larger than graph.V {self.V}"
        self._edges.append(e)
        self._adj[e.source].append(e)
        self._adj[e.sink].append(e)

    def add_edge_(self, v, w, capacity, flow = 0):
        """A helper function for adding edge.
        It creates an instance of FlowEdge and use self.add_edge() to add it to the graph.
        """
        e = FlowEdge(v, w, capacity, flow)
        self.add_edge(e)

    @classmethod
    def from_adj_list(cls, adj_list):
        """Build a new graph from adj_list.
        e.g. [[(1, 0.5)], [(2, 0.8, 0.1)], [(0, 3.5, 3.0)]] means a graph of 3 vertices 
        with three flow edges 0->1 (0.0/0.5), 1->2 (0.1/0.8) and 2->0 (3.0/3.5). 
        """
        graph = cls(len(adj_list))
        for v, adj_ in enumerate(adj_list):
            for edge_properties in adj_:
                graph.add_edge_(v, *edge_properties)
        return graph




if __name__ == "__main__":
    g = EWGraph.from_adj_list([[(1, 0.5)], [(2, 3.5), (3, 100)], [(3, 0.5)], []])
    new_g = g.from_graph(g)
    print(new_g)
    print(new_g is g)
    for e in g.adj(3):
        print(e)

    g = EWDiGraph.from_adj_list(
        [[(1, 0.2), (0, 0.1)], [(3, 0.6)], [(1, 0.4), (1, 0.3)], [(2, 0.5)]])
    new_g = g.from_graph(g)
    print(new_g)
    print(new_g is g)
    for e in g.adj(0):
        print(e)

    g = FlowNetwork.from_adj_list(
        [[(1, 0.2, 0.1), (0, 0.1)], [(3, 0.6, 0.2)], [(1, 0.4), (1, 0.3, 0.25)], [(2, 0.5)]]
    )
    print(g)
    new_g = g.from_graph(g)
    print(new_g)
