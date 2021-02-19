"""
This script contains some np-complete questions.

dp.py and local_search.py also contain some np-complete questions.
"""

from algo.dstruct import graph
from algo.dp import KnapsackVal

class VertexCover:
    """Vertex cover for generic graph."""
    def __init__(self, graph, k = None):
        """Given a Graph, find a set of <= k vertices such that all edges in the graph
        have at least one end point in the set.
        
        If k is None, find such a set with minimal number of vertices.
        """
        self.graph = graph
        edges = set(tuple(int(v) for v in e.split('-')) for e in graph.edges)
        adj_list = {}
        for s in range(self.graph.V):
            for v in self.graph.adj(s):
                neighbours = adj_list.get(s, set())
                neighbours.add(v)
                adj_list[s] = neighbours

        if k is not None:
            self._vertices = self._get_cover(edges, adj_list, k)
        else:
            self._vertices = list(range(self.graph.V))
            self._binary_search(0, self.graph.V, edges, adj_list)


    def _binary_search(self, beg, end, edges, adj_list):
        mid = (beg + end) // 2
        if mid < beg:
            return
        vertices = self._get_cover(edges, adj_list, k=mid)
        if vertices is not None:
            self._vertices = vertices
            self._binary_search(beg, mid - 1, edges, adj_list)
        else:
            self._binary_search(mid + 1, end, edges, adj_list)

    def _get_cover(self, edges, adj_list, k):
        if not edges:
            return []

        if len(edges) == 1:
            for e in edges:
                return [e[0]]

        if k == 0:
            return None

        if k == 1:
            for v in adj_list:
                found = True
                for v1, v2 in edges:
                    if v1 != v and v2 != v:
                        found = False
                        break
                if found:
                    return [v]
            return None

        
        v, w = edges.pop()
        for v_ in v, w:
            removed_edges = []
            neighbours = adj_list[v_]
            for x in neighbours:
                adj_list[x].remove(v_)
                if x < v_:
                    e = (x, v_)
                else:
                    e = (v_, x)
                if e == (v, w):
                    continue
                edges.remove(e)
                removed_edges.append(e) 
            del adj_list[v_]

            vertices = self._get_cover(edges, adj_list, k - 1)
            
            # put back the removed edges
            for e in removed_edges:
                edges.add(e)
            adj_list[v_] = neighbours
            for x in neighbours:
                adj_list[x].add(v_)
            
            if vertices is not None:
                edges.add((v, w))
                return vertices + [v]
        
        # if both cases fail
        edges.add((v, w))
        return None
            

    def vertices(self):
        return self._vertices



class Knapsack:
    """Each item has a non-negative value, as well as, a non-negative weight.
    Given a knapsack with a non-negative capacity, select a collection of items with max total value
    and a total weight no more than the knapsack capacity.

    This implementation gives answer in O(n^3 / eps) time complexity (where n is number of items) 
    with answer that is at least (1 - eps) the optimal max value.
    """
    def __init__(self):
        self.knapsack_val = KnapsackVal()
        
    def get_max_val(self, vals, weights, capacity, eps = 0.1, return_selection=False):
        """Return the max val of such selection.

        If return_selection is True, the indices of the selected items are returned as well.
        """
        m = eps * max(vals) / len(vals)
        int_vals = [int(v // m) for v in vals]
        _, selections = self.knapsack_val.get_max_val(int_vals, weights, capacity, return_selection = True)
        max_val = 0
        for i in selections:
            max_val += vals[i]
        if return_selection:
            return max_val, selections
        return max_val


class KnapsackBruteForce:
    """Each item has a non-negative value, as well as, a non-negative weight.
    Given a knapsack with a non-negative capacity, select a collection of items with max total value
    and a total weight no more than the knapsack capacity.

    This brute force implementation gives answer in O(2^n) time complexity (where n is the number of items).
    """
    def get_max_val(self, vals, weights, capacity, return_selection=False):
        """Return the max val of such selection.

        If return_selection is True, the indices of the selected items are returned as well.
        """
        self.vals = vals
        self.weights = weights
        self.capacity = capacity
        self.prefixes = []
        self._recur(0, 0, [])
        total_vals = [sum(self.vals[i] for i in prefix) for prefix in self.prefixes]
        max_val = 0
        selection = []
        for i, v in enumerate(total_vals):
            if v > max_val:
                max_val = v
                selection = self.prefixes[i]
        
        if return_selection:
            return max_val, selection
        return max_val



    def _recur(self, curr, curr_weight, prefix):
        if curr == len(self.vals):
            self.prefixes.append(prefix)
            return
        if curr_weight + self.weights[curr] <= self.capacity:
            self._recur(curr + 1, curr_weight + self.weights[curr], prefix + [curr])
        self._recur(curr + 1, curr_weight, prefix)



if __name__ == "__main__":
    # # vertex cover
    # from algo.dstruct import Graph
    # g = Graph.from_adj_list(
    #     [[1], [2, 3], [4, 5], [4], [], []]
    # )
    # vertex_cover = VertexCover(g)
    # print(vertex_cover.vertices())
    # g = Graph.from_adj_list(
    #     [[1, 2, 3, 4], [], [], [], []]
    # )
    # vertex_cover = VertexCover(g)
    # print(vertex_cover.vertices())
    # g = Graph.from_adj_list(
    #     [[2, 4],
    #     [2, 3, 5],
    #     [4, 5],
    #     [4],
    #     [5],
    #     []]
    # )
    # vertex_cover = VertexCover(g)
    # print(vertex_cover.vertices())

    # Knapsack
    import random
    n = 20
    vals = [random.uniform(0, 10) for _ in range(n)]
    weights = [random.uniform(0, 20) for _ in range(n)]
    capacity = random.uniform(100, 200)
    # print("vals: ", [(i, v) for i, v in enumerate(vals)])
    # print("weights: ", [(i, v) for i, v in enumerate(weights)])
    # print("capacity: ", capacity)

    knap_bf = KnapsackBruteForce()
    print(knap_bf.get_max_val(vals, weights, capacity, True))
    # print("vals: ", vals)
    # print("weights: ", weights)
    # print("capacity: ", capacity)

    knap = Knapsack()
    print(knap.get_max_val(vals, weights, capacity, 0.01, True))

    # knap_val = KnapsackVal()
    # print(knap_val.get_max_val(vals, weights, capacity, True))



