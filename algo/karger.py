from algo.dstruct.graph import Graph
import math
import random

def karger_mincut(graph, num_trials = None):
    """Given a undirected and unweighted (non-simple) graph, 
    return two lists of mutually exclusive vertices and the number of edges between the two partitions.
    """
    assert graph.type == "undirected", "Karger Mincut is only for undirected graph"
    assert graph.V > 1, "the given graph has less than 2 nodes"
    if num_trials is None:
        num_trials = math.ceil(graph.V ** 2 * math.log(graph.V))
    A, B = None, None
    num_edges = graph.E + 1
    for _ in range(num_trials):
        A_, B_, num_edges_ = _mincut(graph)
        if num_edges_ < num_edges:
            A, B, num_edges = A_, B_, num_edges_
    return A, B, num_edges


def _mincut(graph):
    edges = {}
    # Convert a list of edges to a dictionary of edges with weight (i.e. the number of edges between this two nodes)
    # and a list of nodes to a set of nodes
    # for the purpose of more efficient processing
    for e in graph.edges:
        edges[e] = edges.get(e, 0) + 1

    nodes = {i:[i] for i in range(graph.V)}
    adj = [set() for _ in range(graph.V)]
    for s in range(graph.V):
        for v in graph.adj(s):
            adj[s].add(v)

    # remove self edges
    for s, vs in enumerate(adj):
        vs.discard(s)
        edges.pop(f"{s}-{s}", None)

    while len(nodes) > 2:
        # we should randomly choose the edges with respect to their weights
        population, weights = list(zip(*edges.items()))
        e = random.choices(population, weights=weights, k = 1)[0]
        del edges[e]
        v1, v2 = e.split("-")
        v1, v2 = int(v1), int(v2)
        if len(adj[v1]) > len(adj[v2]):
            keep = v1
            rm = v2
        else:
            rm = v1
            keep = v2
        nodes[keep].extend(nodes[rm])
        del nodes[rm]

        adj[keep] = adj[keep] | adj[rm]
        adj[keep].discard(keep)
        adj[keep].remove(rm)
        for v in adj[rm]:
            if v == keep:
                continue
            adj[v].remove(rm)
            adj[v].add(keep)
            old_e = f"{rm}-{v}" if rm < v else f"{v}-{rm}"
            new_e = f"{keep}-{v}" if keep < v else f"{v}-{keep}"
            edges[new_e] = edges.get(new_e, 0) + edges[old_e]
            del edges[old_e]
        adj[rm] = set()
    A, B = [nodes[k] for k in nodes]
    num_edges = sum([v for _, v in edges.items()])
    return A, B, num_edges
        

if __name__ == "__main__":
    adj_list = [[1,2],[0,2], [0,1,3], {2,4,5,6},{3,5},{3,4, 6},{3,5}]
    ug = Graph.from_adj_list(adj_list)
    print(ug)
    A, B, num_edges = karger_mincut(ug)
    print("A: ", A)
    print("B: ", B)
    print("num_edges: ", num_edges)
