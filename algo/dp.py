"""Algorithm problems using dynamica programming."""

from algo.graph import DirectedBFS, DirectedDFS
from algo.dstruct.tree import BiNode, LLRedBlackTree
import itertools


class MIS:
    """Non adjacent set of numbers with maximum sum."""
    def get_max_sum(self, num_list, return_selection = False):
        """Compute the max sum of non adjacent numbers from the given list.
        If return_selection is True, the list of numbers used will also be returned.
        """
        self.num_list = num_list
        self.max_sum = [0] * len(num_list)
        for i, val in enumerate(self.num_list):
            prev = self.max_sum[i - 1] if i >= 1 else 0
            prev_2 = self.max_sum[i - 2] if i >= 2 else 0
            self.max_sum[i] = max(prev, prev_2 + val)
        if return_selection:
            selection = self._reconstruct()
            return self.max_sum[-1], selection
        return self.max_sum[-1]

    def _reconstruct(self):
        selection = []
        i = len(self.num_list) - 1
        while i >= 0:
            val = self.num_list[i]
            prev = self.max_sum[i - 1] if i >= 1 else 0
            prev_2 = self.max_sum[i - 2] if i >= 2 else 0
            if prev_2 + val > prev:
                # take current val and skip the prev val
                selection.append(val)
                i -= 2
            else:
                # skip current val
                i -= 1
        selection.reverse()
        return selection


class VertexCoverTree:
    """Vertex cover problem for tree."""
    def min_cost(self, tree, v_costs, return_vertices = False):
        """Return the minimal vertex costs for covering all edges in the tree graph.
        
        Parameters:
        tree (algo.dstruct.Digraph): A directed graph and outgoing edges pointing to the node's children.
        v_costs (list): A list of numbers representing the vertices' costs.
        return_vertices (bool): Whether returning the collection of vertices
        """
        from algo.graph import DirectedDFS
        self.tree = tree
        self.v_costs = v_costs
        dfs = DirectedDFS()

        # This order is from the leaves to root
        # It ensures that the child nodes have been processed
        # before the parent node
        orders = dfs.topological_sort(tree, reverse=True)
        # self._costs[i] will be the min cost to cover the tree
        # rooted at i
        self._costs = [None] * self.tree.V
        
        for root in orders:
            # decide whether to include root
            # if yes
            # we just inherit the minimal cost vertex cover of its children
            # if no
            # we need all the children to be included to cover the links between this root
            # its children. In this way, we can "recompute" the children's costs as 
            # now the children can all just inherit their children's (root's grand-children) min covers.
            # It is because previously, for some children, their optimal costs do not include themselves
            # so they cannot just inherit the min costs from the grand-children. Now we are assume the 
            # optimal costs for the current root will include the children, we can just inherit the
            # grand-children.
            root_in = self.v_costs[root]
            root_out = 0
            for child in self.tree.adj(root):
                root_in += self._costs[child]
                root_out += self.v_costs[child]
                for grandchild in self.tree.adj(child):
                    root_out += self._costs[grandchild]
            self._costs[root] = min(root_in, root_out)

        root = orders[-1]
        if return_vertices:
            vertices = self._reconstruct(orders[-1])
            return self._costs[root], vertices
        return self._costs[root]

        
    def _reconstruct(self, root):
        return self._recur(root, [])


    def _recur(self, root, prefix):
        root_in = self.v_costs[root]
        root_out = 0
        for child in self.tree.adj(root):
            root_in += self._costs[child]
            root_out += self.v_costs[child]
            for grandchild in self.tree.adj(child):
                root_out += self._costs[grandchild]

        # if root is in the optimal set
        # append root and recur on its children
        if root_in < root_out:
            prefix.append(root)
            for child in self.tree.adj(root):
                self._recur(child, prefix)
        # else all its children must be in the set
        # append all the children and recur on its grandchildren
        else:
            for child in self.tree.adj(root):
                prefix.append(child)
                for grandchild in self.tree.adj(child):
                    self._recur(grandchild, prefix)
        
        return prefix


class KnapsackWeight:
    """Each item has a non-negative value, as well as, a non-negative and integral weight.
    Given a knapsack with an integral capacity, select a collection of items with max total val
    and a total weight no more than the knapsack capacity.
    """
    def get_max_val(self, vals, weights, capacity, return_selection = False):
        """Return the max val of such selection.

        Parameters:
        vals (list): the list of items' values
        weights (list[int]): the list of items' weights corresponding to vals
        capacity (list[int]): the knapsack's capacity
        return_selection (bool): if True, return the list of selected items' indices as well

        """
        self.vals = vals
        self.weights = weights
        self.capacity = capacity

        self.max_val = [list([0] * (capacity + 1)) for _ in range(len(self.vals))]
        for i in range(len(self.vals)):
            for c in range(capacity + 1):
                # if we do not put in current item
                # then we can use all the capacity for the past items
                wo_curr = self.max_val[i - 1][c] if i >= 1 else 0
                # if we choose to put in current item
                # then we can use c - weight_i for the past items
                if c >= self.weights[i]: # check if current item fits
                    res_cap = c - self.weights[i]
                    with_curr = self.max_val[i - 1][res_cap] + self.vals[i] if (i >= 1 and res_cap >= 0) else 0
                else: # it is not possible to put current item
                    with_curr = 0 
                # the best choice would be the better of the two
                self.max_val[i][c] = max(wo_curr, with_curr)
        if return_selection:
            selection = self._reconstruct()
            return self.max_val[len(self.vals) - 1][self.capacity], selection
        return self.max_val[len(self.vals) - 1][self.capacity]


    def _reconstruct(self):
        selection = []
        i = len(self.vals) - 1
        capacity = self.capacity
        while i >= 0 and capacity >= 0:
            wo_curr = self.max_val[i - 1][capacity] if i >= 1 else 0
            if capacity >= self.weights[i]:  # check if current item fits
                res_cap = capacity - self.weights[i]
                with_curr = self.max_val[i - 1][res_cap] + \
                    self.vals[i] if (i >= 1 and res_cap >= 0) else 0
            else: # it is not possible to put current item
                with_curr = 0 

            if with_curr > wo_curr:
                # if we picked the current item
                selection.append(i)
                capacity -= self.weights[i]
            # else: we did not pick this item, we still have the same capacity
            i -= 1
        selection.reverse()
        return selection


class KnapsackVal:
    """Each item has a non-negative integral value, as well as, a non-negative weight.
    Given a knapsack with a non-negative capacity, select a collection of items with max total val
    and a total weight no more than the knapsack capacity.
    """
    def get_max_val(self, vals, weights, capacity, return_selection=False):
        """Return the max val of such selection.

        If return_selection is True, the indices of the selected items are returned as well.
        """
        self.vals = vals
        self.weights = weights
        self.capacity = capacity
        max_val = max(vals)

        self.total_weights = [list([0] * (len(self.vals) * max_val + 1))
                        for _ in range(len(self.vals))]
        
        for v in range(len(self.vals) * max_val + 1):
            self.total_weights[0][v] = self.weights[0] if v <= self.vals[0] else float('inf')
        
        for i in range(len(self.vals)):
            self.total_weights[i][0] = 0

        for i in range(1, len(self.vals)):
            for v in range(1, len(self.vals) * max_val + 1):
                excl_i = self.total_weights[i - 1][v]
                if v - self.vals[i] >= 0:
                    incl_i = self.total_weights[i - 1][v - self.vals[i]] + self.weights[i]
                else:
                    incl_i = self.weights[i]
                self.total_weights[i][v] = min(excl_i, incl_i)
        
        val = 0
        for v in range(len(self.vals) * max_val, -1, -1):
            if self.total_weights[-1][v] <= self.capacity:
                val = v
                break

        if return_selection:
            selection = self._reconstruct(val)
            return val, selection
        
        return val

    def _reconstruct(self, v):
        selection = []
        i = len(self.vals) - 1
        while i >= 0 and v >= 0:
            excl_i = self.total_weights[i - 1][v]
            if v - self.vals[i] >= 0:
                incl_i = self.total_weights[i -
                    1][v - self.vals[i]] + self.weights[i]
            else:
                incl_i = self.weights[i]
            if incl_i < excl_i:
                selection.append(i)
                v -= self.vals[i]
            i -= 1 
        if v > 0:
            selection.append(0) 
        selection.reverse()
        return selection          
        


class SequenceAlignment:
    """Provide the optimal sequence alignment with minimal Needleman-Wunsche score."""
    def __init__(self, p_gap=1, p_diff=1):
        """Initialise the object with p_gap (penalty for inserting a gap) and g_diff (for symbol mismatch)"""
        self.p_gap = p_gap
        self.p_diff = p_diff

    
    def align(self, seq_1, seq_2, return_align = False):
        """Given two sequences seq_1 and seq_2, return the min alignment penalty.

        If return_align is True, an additional tuple of corresponding alignments is returned.
        """
        self.seq_1 = seq_1
        self.seq_2 = seq_2
        # self.min_p[i][j] is the min penalty AFTER (i - 1)th char from seq_1 and (j - 1)th char from seq_2
        # are aligned
        # the additional row and column are added to cater to the empty strings
        self.min_p = [list([0] * (len(self.seq_2) + 1)) for _ in range(len(self.seq_1) + 1)]

        for i in range(len(self.seq_1) + 1):
            self.min_p[i][0] = i * self.p_gap
        
        for i in range(len(self.seq_2) + 1):
            self.min_p[0][i] = i * self.p_gap

        for i_1 in range(1, len(self.seq_1) + 1):
            for i_2 in range(1, len(self.seq_2) + 1):
                # suppose previously we aligned seq_2[i_2] with a gap
                # then we now should align seq_1[i_1] with a gap
                # the prefix score to use will be min_p[i_1 - 1][i_2]
                # because in this situation, we would just finished align seq_2[i_2] and seq_1[i_1 - 1]
                prefix_p = self.min_p[i_1 - 1][i_2]
                g_in_2 = prefix_p + self.p_gap
                # same argument
                prefix_p = self.min_p[i_1][i_2 - 1]
                g_in_1 = prefix_p + self.p_gap
                # similar argument, we now choose to match them
                # the situation only viable if neither of them is the very first char
                prefix_p = self.min_p[i_1 - 1][i_2 - 1]
                p = self.p_diff if self.seq_1[i_1 - 1] != self.seq_2[i_2 - 1] else 0
                match = prefix_p + p
                self.min_p[i_1][i_2] = min(g_in_1, g_in_2, match)

        if return_align:
            return self.min_p[-1][-1], self._reconstruct()
        return self.min_p[-1][-1]
        

    def _reconstruct(self):
        s1 = []
        s2 = []
        i_1 = len(self.seq_1)
        i_2 = len(self.seq_2)
        while i_1 >= 1 and i_2 >= 1:
            prefix_p = self.min_p[i_1 - 1][i_2]
            g_in_2 = prefix_p + self.p_gap
            # same argument
            prefix_p = self.min_p[i_1][i_2 - 1]
            g_in_1 = prefix_p + self.p_gap
            # similar argument, we now choose to match them
            # the situation only viable if neither of them is the very first char
            prefix_p = self.min_p[i_1 - 1][i_2 - 1]
            p = self.p_diff if self.seq_1[i_1 - 1] != self.seq_2[i_2 - 1] else 0
            match = prefix_p + p
            if g_in_2 < g_in_1:
                if match < g_in_2:
                    s1.append(self.seq_1[i_1 - 1])
                    i_1 -= 1
                    s2.append(self.seq_2[i_2 - 1])
                    i_2 -= 1
                else:
                    s2.append("-")
                    s1.append(self.seq_1[i_1 - 1])
                    i_1 -= 1
            else:
                if match < g_in_1:
                    s1.append(self.seq_1[i_1 - 1])
                    i_1 -= 1
                    s2.append(self.seq_2[i_2 - 1])
                    i_2 -= 1
                else:
                    s1.append("-")
                    s2.append(self.seq_2[i_2 - 1])
                    i_2 -= 1
        if i_1 >= 1:
            s1.extend("-" * (i_1 - 1))
        elif i_2 >= 1:
            s2.extend("-" * (i_2 - 1))

        s1.reverse()
        s2.reverse()
        return s1, s2
        

class OptimalBST:
    """Given a dictionary of keys and their occurance frequencies, build a tree with minimal weighted search cost."""
    def __init__(self, key_freq):
        self.keys = sorted(list(key_freq.keys()))
        self.freqs = [key_freq[v] for v in self.keys]
        self._compute()

    def _compute(self):
        self._costs = [[None] * len(self.keys) for _ in range(len(self.keys))]
        
        # diagnal entries i.e. the slice from i to i + 1 is just i itself
        # its cost should just be its frequency
        for i in range(len(self.keys)):
            self._costs[i][i] = self.freqs[i]

        # Note that s loop and i loop cannot switch,
        # else it will be accessing data that has not been filled up
        # s is the length of the slice that we are building tree for.
        # Suppose s is 3, we depending on the choice of the root, we will need
        # to access the minimal tree built for s = 1 and s = 2.
        # If s is the outer loop, we would have finished filling up the table cells for
        # all s = 1 and 2, so we can access them.
        # If i is the outer loop, we cannot ensure this condition.
        for s in range(1, len(self.keys) + 1):
            for i in range(0, len(self.keys) - s + 1):
                # select root
                temp_costs = [0] * s
                for r in range(i, i + s):
                    # if r is the beginning of the slice, there is no left subtree and cost is 0
                    cost_left = self._costs[i][r - 1] if r - 1 >= i else 0
                    # similarly
                    cost_right = self._costs[r + 1][i + s - 1] if r + 1 <= i + s - 1 else 0
                    temp_costs[r - i] = sum(self.freqs[i:i + s]) + cost_left + cost_right
                # the above is the same as conducting brute force search for which
                # root is optimal for the slice i: i + s of the vals
                # the optimal (min costs) is the cost for slice i:i + s 
                self._costs[i][i + s - 1] = min(temp_costs)


    def cost(self):
        """Return the cost that corresponds to the minimal cost tree."""
        return self._costs[0][len(self.keys) - 1]
    

    def _reconstruct(self):
        from algo.dstruct import BiNode
        self.tree = self._generate_tree(0, len(self.keys) - 1)
    
    def _generate_tree(self, beg, end):
        if beg > end:
            return None
        temp_costs = [None] * (end - beg + 1)
        for r in range(beg, end + 1):
            # if r is the beginning of the slice, there is no left subtree and cost is 0
            cost_left = self._costs[beg][r - 1] if r - 1 >= beg else 0
            # similarly
            cost_right = self._costs[r + 1][end] if r + 1 <= end else 0
            temp_costs[r - beg] = sum(self.freqs[beg:end + 1]) + \
                cost_left + cost_right
        # the above is the same as conducting brute force search for which
        # root is optimal for the slice i: i + s of the vals
        # the optimal (min costs) is the cost for slice i:i + s
        r = temp_costs.index(min(temp_costs)) + beg
        root = BiNode(self.keys[r], None)
        root.left = self._generate_tree(beg, r - 1)
        root.right = self._generate_tree(r + 1, end)
        return root

    def get_tree(self):
        """Return the root of the optimal tree."""
        if not hasattr(self, "tree"):
            self._reconstruct()
        return self.tree




class TSP:
    """Travelling salesman problem solved with dynamic programming.
    
    Complexity is O(n^2 * 2^n) for this approch.
    """
    def min_dist(self, graph, return_order=False):
        """Given a complete EWGraph that is complete, return the distance of the shortest
        cyclic trip that visits each vertex exactly once. 
        
        If return_order is True, return the order of the vertices on such trip.
        """
        self.graph = graph
        self.adj = [[None] * self.graph.V for _ in range(self.graph.V)]
        for e in self.graph.edges:
            assert self.adj[e.v][e.w] is None, f"There are paralle edges between vertices {e.v} and {e.w}"
            self.adj[e.v][e.w] = e.weight
            self.adj[e.w][e.v] = e.weight
        for i, row in enumerate(self.adj):
            for j in range(i + 1, len(row)):
                assert row[j] is not None, f"There is no edge between vertices {i} and {j}"
        
        self.num_combi = 2 ** (self.graph.V)
        # if self.graph.V = 3, self._dists[6] is the dimension where the Set = {1, 2} (0-based)
        # as 6 in binary number is 110
        # Technically all the even rows are not needed as the set must contain the 0th vertex.
        # self._dists[6][1] is the shortest distance traveled if the salesman was to visit
        # vertices 1 and 2 with vertex 1 visited last
        self._dists = [[None] * self.graph.V for _ in range(self.num_combi)]
        # Consider all the sets that have 0th vertices
        # if they are not the set {0}, then they should have inf distance
        # as there is no way to ensure visiting 0th vertex only once
        # i.e. now we are assuming that we are going out from vertex 0, and we aim to arrive back
        # at 0 at last, if the intermediate steps already have vertex 0 another time, then we
        # visited vertex 0 at least twice.
        for i in range(len(self._dists)):
            if i % 2:
                self._dists[i][0] = float("inf")
        # for the set that is {0}, the cost should be 0
        self._dists[1][0] = 0
        for m in range(2, self.graph.V + 1):
            size_m_with_0 = self._get_combi_gen(m, include=0)
            while True: # loop through all subsets satisfying the condition
                try:
                    row_number, subset = next(size_m_with_0)
                except:
                    break
                for j in subset:
                    temp_dists = []
                    if j == 0:
                        continue
                    # if this subset is formed from subset - {j} union j at last step, 
                    # what is the dist
                    temp_row_number = row_number - 2 ** j
                    for k in subset:
                        # brute force search j is visited from which vertex
                        if k == j:
                            continue
                        temp_dists.append(self._dists[temp_row_number][k] + self.adj[k][j])
                    self._dists[row_number][j] = min(temp_dists)
        
        # find the subsets that visited all the vertices and from there check the shortest path
        # back to vertex 0
        min_dist = min(self._dists[self.num_combi - 1][j] + self.adj[j][0] for j in range(1, self.graph.V))
        
        if return_order:
            return min_dist, self._reconstruct()
        return min_dist



    def _get_combi_gen(self, subset_size, exclude=None, include=None):
        """Create a generator of decimal numbers converted from all the possible
        binary numbers of length self.graph.V with subset_size of bits turned on, as well as, the
        list of the indices where the bits are 1.
        
        If exclude or include is specified to be an int, then that bit position will be fixed to be 0 or 1.
        """
        assert not ((exclude is not None) and (include is not None)), \
            "Both exclude and include are specified, while at most one of them can be non-None."

        if exclude is not None:
            combi_iter = itertools.combinations(
                range(self.graph.V - 1), subset_size)
            for combi in combi_iter:
                row_num = 0
                subset = []
                for v in combi:
                    if v >= exclude:
                        v += 1
                    row_num += 2 ** v
                    subset.append(v)
                yield row_num, subset

        elif include is not None:
            combi_iter = itertools.combinations(
                range(self.graph.V - 1), subset_size - 1)
            for combi in combi_iter:
                row_num = 0
                subset = []
                for v in combi:
                    if v >= include:
                        v += 1
                    row_num += 2 ** v
                    subset.append(v)
                row_num += 2 ** include
                subset.append(include)
                yield row_num, subset

        # else if we are just generating the combinations
        else:
            combi_iter = itertools.combinations(range(self.graph.V), subset_size)
            for combi in combi_iter:
                yield sum(2 ** v for v in combi), list(combi)


    def _reconstruct(self):
        pth = [0]
        subset = set(range(self.graph.V))
        row_number = self.num_combi - 1
        arr = [self._dists[row_number][j] + self.adj[j][0]
               for j in range(1, self.graph.V)]
        last_dest = min(range(len(arr)), key = lambda i: arr[i]) + 1
        pth.append(last_dest)
        row_number -= 2 ** last_dest
        subset.remove(last_dest)
        while row_number != 1:
            temp_dict = {}
            # find the subset that when adding last_dest to the set will generate the
            # min dist
            for j in subset:
                if j == 0:
                    continue
                # the dist reaching last_dest if the vertex before last_dest is j
                temp_dict[j] = self._dists[row_number][j] + self.adj[j][last_dest]
            last_dest = min(temp_dict, key = temp_dict.get)
            pth.append(last_dest)
            row_number -= 2 ** last_dest
            subset.remove(last_dest)
        pth.append(0)
        return pth

if __name__ == "__main__":
    # # mis
    # mis = MIS()
    # num_list = [3, 4, 9, 5, 6]
    # print(mis.get_max_sum(num_list, return_selection=True))

    # knapsack
    vals = [3, 2, 4, 4]
    weights = [4, 3, 2, 3]
    capacity = 6
    knapsack = KnapsackWeight()
    print(knapsack.get_max_val(vals, weights, capacity, return_selection=True))

    # knapsackII
    vals = [3, 2, 4, 4]
    weights = [4, 3, 2, 3]
    capacity = 6
    knapsack = KnapsackVal()
    print(knapsack.get_max_val(vals, weights, capacity, return_selection=True))


    # # string alignment
    # align = SequenceAlignment()
    # s1 = "AGGGCT"
    # s2 = "AGGCA"
    # print(align.align(s1, s2, True))
    # s1 = "GCATGCU"
    # s2 = "GATTACA"
    # print(align.align(s1, s2, True))

    # # optimal binary search tree
    # key_freq = {1: 0.25, 2: 0.2, 3: 0.05, 4: 0.2, 5: 0.3}
    # optimal_tree = OptimalBST(key_freq)
    # print(optimal_tree.cost()) # expect 2.1
    # # expect
    # #         2
    # #       /   \
    # #      1     5
    # #           /
    # #          4
    # #         /
    # #        3
    # root = optimal_tree.get_tree()
    # print(root.key)
    # print(root.left.key)
    # print(root.right.key)
    # print(root.right.left.key)
    # print(root.right.left.left.key)


    # # vertex cover for tree
    # from algo.dstruct import DiGraph
    # tree = DiGraph.from_adj_list([[3, 4], [2, 5], [], [6], [1, 7], [], [], []])
    # v_costs = [10, 50, 70, 30, 20, 80, 60, 40]
    # vertex_cover = VertexCoverTree()
    # print(vertex_cover.min_cost(tree, v_costs, return_vertices=True))


    # # travelling salesman problem
    # from algo.dstruct import EWGraph
    # tsp = TSP()
    # g = EWGraph.from_adj_list([[(1, 10), (3, 20), (2, 15)],
    #                                 [(2, 35), (3, 25)],
    #                                 [(3, 30)],
    #                                 []])
    # print(tsp.min_dist(g, return_order=True))
    # g = EWGraph.from_adj_list([[(1, 12), (2, 10), (3, 19), (4, 8)],
    #                             [(2, 3), (3, 7), (4, 2)],
    #                             [(3, 6), (4, 20)],
    #                             [(4, 4)],
    #                             []])
    # print(tsp.min_dist(g, return_order=True))
