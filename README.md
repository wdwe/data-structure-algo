# Python Implementation of Common Data Structures and Algorithms
## Introduction
Data structure and algorithm are so fundamental to everyday programming that I believe it is a "must-learn" for all serious programmers.<br>
I have taken a few courses from my undergraduate education and Coursera, but I too frequently forget the important concepts of these structures and algorithms. Finally, I decided to implement them myself to get a firmer grasp of them.<br>
Most of them are realisation of the concepts I learnt from [Princeton Algorithms](https://www.coursera.org/learn/algorithms-part1) and [Stanford Algorithms](https://www.coursera.org/specializations/algorithms) on Coursera. These are two really awesome courses that I do not hesitate to recommend to other programmers.

## Content Table
### Data Structures
Script | Function/Class | Data Structure
---|---|---
algo/dstruct/graph.py | Graph | Unweighted Undirected Graph
algo/dstruct/graph.py | DiGraph | Unweighted Directed Graph
algo/dstruct/graph.py | EWGraph | Edge Weighted Graph
algo/dstruct/graph.py | EWDiGraph | Edge Weighted Directed Graph
algo/dstruct/graph.py | FlowNetwork | Flow Network
algo/dstruct/heap.py | MinHeap | Min Heap 
algo/dstruct/heap.py | MaxHeap | Min Heap 
algo/dstruct/heap.py | IndexedMinHeap | Indexed Min Heap 
algo/dstruct/heap.py | IndexedMaxHeap | Indexed Min Heap 
algo/dstruct/tree.py | BinarySearchTree | BST with all symbol table operations
algo/dstruct/tree.py | LLRedBlackTree | Left Leaning Red Black Tree with all symbol table operations short of deletion
algo/dstruct/tree.py | RWayTrie | R-Way Trie with common string search operations
algo/dstruct/tree.py | TST | Ternary Search Trie with common string search operations
algo/dstruct/unionfind.py | WeightedQuickUnion | Weighted Quick Union with path compression


### Algorithms
Script | Function/Class | Algorithms
---|---|---
algo/graph.py | UndirectedDFS | **Unweighted Undirected Graph**<br>DFS <br> Connected Components <br> Cycle Detection <br> Has Path Between Two Vertices<br> |
algo/graph.py | DirectedDFS | **Unweighted Directed Graph**<br> DFS <br> Topological Sort <br> Cycle Detection <br> Has Path Between Two Vertices<br> Strongly Connected Components
algo/graph.py | DirectedBFS | **Unweighted Undirected Graph** <br> BFS <br> Has Path Between Two Vertices <br> Shortest Path
algo/graph.py | DirectedBFS | **Unweighted Directed Graph** <br> BFS <br> Has Path Between Two Vertices <br> Shortest Path
algo/graph.py | DijkstraShortest | Efficient Dijkstra shortest path with heap
algp/graph.py | BellmanFordShortest| Efficient Bellman-Ford shortest path with queue<br>Negative Cycle Detection
algo/graph.py | FloydWarshallShortest | Floyd-Warshall all pairs shortest path<br>with negative cycle indication
algo/graph.py | JohnsonShortest | Johnson's Algorithm for all pairs shortest path<br>with negative cycle indication (detection)
algo/graph.py | PrimMST | Prim's Minimum Spanning Tree |
algo/graph.py | KruskalMST | Kruskal's Minimum Spanning Tree<br>Max Min Distance Clustering
algo/graph.py | FordFulkerson | Ford Fulkenson's algorithm for s-t mincut/maxflow
algo/compress.py | Huffman | Huffman Encoding and Decoding
algo/sorting.py| mergesort | Mergesort
algo/sorting.py| quicksort | Two versions of in-place quicksorts
algo/sorting.py | counting_sort | Counting sort for ASCII characters
algo/sorting.py | LSD_radix_sort | Least significant digit first radix sort
algo/sorting.py | MSD_radix_sort | Most significant digit first radix sort 
algo/sorting.py | str_quicksort | 3-way string quicksort
algo/select_rank.py | random_select | Random partition algorithm for selecting the element of the specific rank from an unsorted list
algo/select_rank.py | deteministic_select | Deterministic partition algorithm for selecting the element of the specific rank from an unsorted list
algo/karger.py | karger_mincut | Karger mincut algorithm
algo/karatsuba.py | karatsuba_mult | Karatsuba multiplication for two super long integers represented by two lists
algo/dp.py | MIS | Maximum Independent Set<br> Given a list return the set of non-adjacent items that gives the maximum sum
algo/dp.py | KnapsackWeight | Knapsack Problem<br>(non-negative vals, non-negative integral weights and capacity)
algo/dp.py | KnapsackVal | Knapsack Problem<br>(non-negative integral vals, non-negative weights and capacity)
algo/dp.py | SequenceAlignment | Needleman–Wunsch algorithm <br> Find the best alignment of two different sequence (i.e. the one with minimal Needle-Wunsch penalty)
algo/dp.py | OptimalBST | Optimal BST<br>Given some keys and their frequencies of occurence, construct a binary search tree such that the expected cost of search is minimum
algo/dp.py | VertexCoverTree | Vertex cover for tree <br> Given a tree graph and costs for using each vertex, choose the set of vertices with minimum costs such that each edge has at least one vertex in this set
algo/dp.py | TSP | Travelling Salesman Problem <br> (with time complexity of O(n^2 * 2^n))
algo/p_np.py | VertexCover | Vertex Cover <br> Given a graph, find the smallest set of vertices such that all edges have least one endpoint in the set.
algo/p_np.py | Knapsack | Knapsack Problem (Approximation) <br> (non-negative vals, non-negative weights and capacity) <br> With O(n^3/eps) time complexity return an at least (1-eps) optimal value.
algo/p_np.py | KnapsackBruteForce | Knapsack Problem (Brute Force) <br> (non-negative vals, non-negative weights and capacity) <br> With O(2^n) time complexity return the optimal value.
algo/local_search.py | MaxCutLocalSearch | Max cut for unweighted graph using local search.
algo/local_search.py | Papadimitriou2SAT | Papadimitriou's algorithm for 2-SAT problem.
algo/string.py | KnuthMorrisPratt | Knuth Morris Pratt deterministic finite state automaton substring search
algo/string.py | BoyerMoore | Boyer Moore algorithm for substring search
algo/string.py | RabinKarp | Rabin Karp algorithm for substring search
./closestpair.py| get_closest_pair | Divide and conquer, an efficient method to find the two closest points in 2D space
./count_qsort_cmp | count_qsort_cmp | Count the number of comparisons made in quicksort
./inversion.py | count_inver | Count the number of inversions in a list of numbers

## Installation
To avoid the convoluted path system in python, I chose to write them as an installable package. If you are playing around with these codes, you need to install the `algo` package as below.
```bash
# cd to this parent directory
pip install -e .
```
It will install a package called `algo` in your environment. All the classes and functions from the algo folder can then be accessed anywhere.
