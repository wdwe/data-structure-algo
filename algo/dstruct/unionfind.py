__all__ = ["WeightedQuickUnion"]


class WeightedQuickUnion:
    """Weighted quick union with path compression."""
    def __init__(self, num):
        """Number of total vertices needs to be provided."""
        self.reset(num)


    def root(self, v):
        """Return the root of v.
        A second loop is used to compress the path of all visited vertices.
        """
        root = v
        while self.ids[root] != root:
            root = self.ids[root]
        # path compression
        while v != root:
            parent = self.ids[v]
            self.ids[v] = root
            v = self.ids[parent]
        return root

    def union(self, v1, v2):
        """Union v1 and v2.
        If v1 and v2 belong to two groups (trees), the root of the tree with smaller weight is
        attached to the root of the tree with larger weight.
        """
        root1 = self.root(v1)
        root2 = self.root(v2)
        if root1 == root2:
            return
        if self.weights[root1] > self.weights[root2]:
            self.ids[root2] = root1
            self.weights[root1] += self.weights[root2]
        else:
            self.ids[root1] = root2
            self.weights[root2] += self.weights[root1]
    
    def connected(self, v1, v2):
        """Return if v1 and v2 belong to the same group (connected)."""
        return self.root(v1) == self.root(v2)

    def groupings(self):
        """Return a list of ids that each vertices belong to."""
        groupings = [self.root(v) for v in self.ids]
        return groupings

    def reset(self, num):
        """Reset the union find completely."""
        self.num = num
        self.weights = [0] * self.num
        self.ids = [i for i in range(self.num)]

if __name__ == "__main__":
    UF = WeightedQuickUnion(10)
    UF.union(0, 5)
    UF.union(0, 0)
    UF.union(5, 6)
    UF.union(1, 2)
    UF.union(2, 7)
    UF.union(3, 8)
    UF.union(3, 4)
    UF.union(4, 9)
    print(UF.groupings())