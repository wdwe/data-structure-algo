import random
import math

class MaxCutLocalSearch:
    """Local search for max cut for unweighted graph.
    
    This implementation computes an at least 50% optimal max cut with O(nC2) time complexity.
    """
    def __init__(self, graph):
        """Providing an unweighted undirected Graph."""
        self.graph = graph
    
    def max_cut(self, num_trials = 1):
        """Run local search num_trials times and return two sets of vertices corresponding to
        the two cuts and number crossing that are the best among all trials.
        """
        best_A, best_B = set(), set()
        best_across = 0
        for _ in range(num_trials):
            size_A = round(random.random() * self.graph.V)
            vertices = list(range(self.graph.V))
            random.shuffle(vertices)
            A = set(vertices[:size_A])
            B = set(vertices[size_A:])
            A = {0, 1}
            B = {2, 3}
            modified = True
            while modified:
                modified = False
                for s in range(self.graph.V):
                    if s in A:
                        own_set = A
                        other_set = B
                    else:
                        own_set = B
                        other_set = A
                    within = 0
                    across = 0
                    for v in self.graph.adj(s):
                        if v in own_set:
                            within += 1
                        else:
                            across += 1
                    if within > across:
                        own_set.remove(s)
                        other_set.add(s)
                        modified = True
            across = 0                    
            for s in A:
                for v in self.graph.adj(s):
                    if v in B:
                        across += 1
            if across >= best_across:
                best_A, best_B = A, B
                best_across = across

        return best_A, best_B, best_across


class Papadimitriou2SAT:
    """Papadimitriou's algorithm for 2-SAT problem.
    
    The when repeats = log2(num_variables), this algorithm finds the satisfying assignment
    with probability of at least 1 - 1/num_variables.
    """
    def solve(self, num_variables, clauses, repeats = None):
        """Return the solution found or None.
        
        The clauses are [(-1, 2), (2, -num_varaibles)] means (not x_1 or x_2) and (x_2 or not x_{num_variables}).
        The variable idx are one-based for the ease of specifying the clauses.
        
        The algorithm fails with probability of 0.5^(repeats).
        """
        if repeats is None:
            repeats = math.ceil(math.log2(num_variables))
        for _ in range(repeats):
            variables = [random.random() > 0.5 for _ in range(num_variables)]
            for _ in range(2 * num_variables ** 2):
                unsatisfied = []
                for lit_1, lit_2 in clauses:
                    idx_1 = abs(lit_1) - 1
                    idx_2 = abs(lit_2) - 1
                    if lit_1 < 0:
                        var_1 = not variables[idx_1]
                    else:
                        var_1 = variables[idx_1]
                    if lit_2 < 0:
                        var_2 = not variables[idx_2]
                    else:
                        var_2 = variables[idx_2]
                    if not (var_1 or var_2):
                        unsatisfied.append((idx_1, idx_2))
                if not unsatisfied:
                    return variables
                to_flip = random.choice(random.choice(unsatisfied))
                variables[to_flip] = not variables[to_flip]
        return None





if __name__ == "__main__":
    # # max cut
    # print("max cut")
    # from algo.dstruct import Graph
    # g = Graph.from_adj_list([
    #     [1, 2, 3],
    #     [2, 3],
    #     [4, 5],
    #     [4, 5],
    #     [],
    #     []
    # ])
    # maxcut = MaxCut(g)
    # print(maxcut.max_cut(1))
    # g = Graph.from_adj_list([
    #     [1, 3],
    #     [2],
    #     [3],
    #     []
    # ])
    # maxcut = MaxCut(g)
    # print(maxcut.max_cut(1))

    # 2-sat
    print("2-sat")
    two_sat = Papadimitriou2SAT()
    clauses = [(1, 2), (-1, 3), (3, 4), (-2, -4)]
    print(two_sat.solve(4, clauses))
    clauses = [(1, 2), (2, 1), (1, 2)]
    print(two_sat.solve(2, clauses))
    print(two_sat.solve())
