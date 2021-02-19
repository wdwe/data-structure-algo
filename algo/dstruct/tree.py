from abc import ABC, abstractmethod
from functools import total_ordering
import random

__all__ = ["BinaryTreeNodeTemplate",
           "BiNode",
           "RBNode",
           "BSTTemplate",
           "BinarySearchTree",
           "LLRedBlackTree",
           "RTrieNode",
           "TSTNode",
           "RWayTrie",
           "TST"]


@total_ordering
class BinaryTreeNodeTemplate(ABC):
    """Parent class for other binary tree node classes to inherit."""
    def __init__(self, key, val, left = None, right = None):
        self.key, self.val, self.left, self.right = key, val, left, right
        self.count = 1

    def __eq__(self, other):
        if isinstance(other, BinaryTreeNodeTemplate):
            return self.key == other.key
        return self.key == other

    def __lt__(self, other):
        if isinstance(other, BinaryTreeNodeTemplate):
            return self.key < other.key
        return self.key < other
    

class BiNode(BinaryTreeNodeTemplate):
    """Node for the unbalance binary search tree"""
    pass

class RBNode(BinaryTreeNodeTemplate):
    """Node for red black tree."""
    def __init__(self, key, val, color = "red", left=None, right=None):
        super().__init__(key, val, left, right)
        self._color = color

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        assert color in ["black", "red"], "Color can only be 'black' or 'red'"
        self._color = color




class BSTTemplate(ABC):
    """Parent class for BinarySearchTree and LLRedBlackTree."""
    def __init__(self):
        self.root = None
    
    @abstractmethod
    def put(self, key, val):
        pass

    @abstractmethod
    def delete(self, key):
        pass

    @abstractmethod
    def delete_min(self):
        pass

    @abstractmethod
    def delete_max(self):
        pass

    def get(self, key):
        """Return the value that corresponds to the key.
        Raise KeyError if key is not in the tree.
        """
        root = self.root
        while root is not None:
            if key < root:
                root = root.left
            elif key > root:
                root = root.right
            else:
                return root.val
        raise KeyError(key)

    def __getitem__(self, key):
        return self.get(key)
    
    def __setitem__(self, key, val):
        self.put(key, val)

    def _predecessor(self, node, key):
        if node is None:
            return None
        if node == key:
            return node
        if key < node:
            return self._predecessor(node.left, key)
        if key > node:
            temp = self._predecessor(node.right, key)
            if temp is None:
                return node
            return temp


    def predecessor(self, key):
        """Return the largest key in the tree that is smaller or equal to the given key."""
        node = self._predecessor(self.root, key)
        return node.key


    def _succesor(self, node, key):
        if node is None:
            return None
        if node == key:
            return node
        if key > node:
            return self._succesor(node.right, key)
        if key < node:
            temp = self._succesor(node.left, key)
            if temp is None:
                return node
            return temp


    def successor(self, key):
        """Return the smallest key in the tree that is larger or equal to the given key."""
        node = self._succesor(self.root, key)
        return node.key

    def rank(self, key):
        """Return the number of keys that is smaller than the given key."""
        return self._rank(self.root, key)

    def _rank(self, node, key):
        if node is None:
            return 0
        if node == key:
            return self._count(node.left)
        if node > key:
            return self._rank(node.left, key)
        if node < key:
            return self._count(node.left) + self._rank(node.right, key) + 1

    def select(self, rank):
        """Return the key with the given rank."""
        assert rank < self.__len__(), \
            f"The given rank {rank} is not smaller than the total number of nodes {self.__len__()}"
        node = self.root
        while True:
            left_count = self._count(node.left)
            if left_count== rank:
                return node.key
            if left_count > rank:
                node = node.left
            elif left_count < rank:
                node = node.right
                rank = rank - left_count - 1

    def _count(self, node):
        if node is None:
            return 0
        return node.count


    def in_order_trav(self):
        """Return a sorted list of keys that are in the tree."""
        q = []
        self._in_order(self.root, q)
        return q

    def _in_order(self, node, q):
        if node is None:
            return
        self._in_order(node.left, q)
        q.append(node.key)
        self._in_order(node.right, q)


    def min_node(self, node = None):
        """Return the node in the tree that has the minimum key.
        If a node is specified, then that node is treated as the root.
        """
        if node is None:
            node = self.root
        if node is None:
            return None
        while node.left is not None:
            node = node.left
        return node

    def min(self):
        """Return the minimum key in the tree."""
        min_node = self.min_node()
        if min_node is None:
            return None
        return min_node.key

    def max_node(self, node = None):
        """Return the node in the tree that has the maximum key.
        If a node is specified, then that node is treated as the root.
        """
        if node is None:
            node = self.root
        if node is None:
            return None
        while node.right is not None:
            node = node.right
        return node
    
    def max(self):
        """Return the maximum key in the tree."""
        max_node = self.max_node()
        if max_node is None:
            return None
        return max_node.key

    def __contains__(self, x):
        try:
            self.get(x)
            return True
        except KeyError:
            return False

    def __len__(self):
        if self.root is None:
            return 0
        return self.root.count


class BinarySearchTree(BSTTemplate):
    """Unbalanced binary search tree."""
    def put(self, key, val):
        """Insert a key associated with val into the right location of the tree."""
        self.root = self._put(self.root, key, val)

    def _put(self, node, key, val):
        if node is None:
            return BiNode(key, val)
        if node == key:
            node.val = val
        elif key < node:
            node.left = self._put(node.left, key, val)
        else:
            node.right = self._put(node.right, key, val)
        node.count = self._count(node.left) + self._count(node.right) + 1
        return node
    
    def _delete_min(self, node):
        if node.left is None:
            return node.right
        node.left = self._delete_min(node.left)
        node.count = self._count(node.left) + self._count(node.right) + 1
        return node

    def delete_min(self):
        """Delete the node with the smallest key from the tree."""
        if self.root is not None:
            self._delete_min(self.root)

    def delete_max(self):
        """Delete the node with the largest key from the tree."""
        if self.root is not None:
            self._delete_max(self.root)


    def _delete_max(self, node):
        if node.right is None:
            return node.left
        node.right = self._delete_max(node.right)
        node.count = self._count(node.left) + self._count(node.right) + 1
        return node


    def delete(self, key):
        """Hibbard deletion with symmetric replacing nodes,
        i.e. replace the node to be deleted with either its predecessor or its successor with
        equal probability. 
        """
        self._delete(self.root, key)


    def _delete(self, node, key):
        if node is None:
            return None
        if key < node:
            node.left = self._delete(node.left, key)
        elif key > node:
            node.right = self._delete(node.right, key)
        else:
            if node.right is None:
                return node.left
            if node.left is None:
                return node.right
            temp = node
            # If we only delete from one direction, Hibbard deletion results in 
            # a very unbalanced tree. Therefore, we randomly replace the current node
            # with the node before it or after it.
            if random.random()>0.5:
                # replace with the node after it
                node = self.min_node(temp.right)
                node.right = self._delete_min(temp.right)
                node.left = temp.left
            else:
                # replace with the node before it
                node = self.max_node(temp.left)
                node.left = self._delete_max(temp.left)
                node.right = temp.right
        node.count = self._count(node.left) + self._count(node.right) + 1
        return node


class LLRedBlackTree(BSTTemplate):
    """Left leaning red black tree.
    This implementation is almost a direct translation from Robert Sedgewick and Kevin Wayne's
    java implementation in their Pricenton Algorithm course on Coursera.
    The rank(node.count) rectification is added in the rotation method.
    """
    def _is_red(self, node):
        if node is None:
            return False
        return node.color == "red"

    def _rotate_left(self, node):
        assert self._is_red(node.right),\
            "Given node's right child is not red in this left rotation."
        temp = node
        node = temp.right
        temp.right = node.left
        node.left = temp
        node.color = temp.color
        temp.color = "red"

        node.count = temp.count
        temp.count = self._count(temp.left) + self._count(temp.right) + 1

        return node

    def _rotate_right(self, node):
        assert self._is_red(node.left),\
            "Given node's left child is not red in this right rotation."
        temp = node
        node = temp.left
        temp.left = node.right
        node.right = temp
        node.color = temp.color
        temp.color = "red"

        node.count = temp.count
        temp.count = self._count(temp.left) + self._count(temp.right) + 1

        return node

    def _flip_color(self, node):
        assert not self._is_red(node), "Given node is red when flipping color."
        assert self._is_red(node.left), "Given node's left child is not red for flipping color."
        assert self._is_red(node.right), "Given node's right child is not red for flipping color."
        node.color = "red"
        node.left.color = "black"
        node.right.color = "black"

    def put(self, key, val):
        """Insert a key associated with val into the right location of the tree."""
        self.root = self._put(self.root, key, val)
        self.root.color = "black" # the root's color should always be black


    def _put(self, node, key, val):
        if node is None:
            return RBNode(key, val, "red")
        if key < node:
            node.left = self._put(node.left, key, val)
        elif key > node:
            node.right = self._put(node.right, key, val)
        else:
            node.val = val
        
        if (self._is_red(node.right) and not self._is_red(node.left)):
            node = self._rotate_left(node)
        if (self._is_red(node.left) and self._is_red(node.left.left)):
            node = self._rotate_right(node)
        if (self._is_red(node.left) and self._is_red(node.right)):
            self._flip_color(node)

        node.count = self._count(node.left) + self._count(node.right) + 1

        return node

    def delete(self, node):
        """Deletion for RedBlackTree is not implemented."""
        raise NotImplementedError

    def delete_max(self):
        """Deletion for RedBlackTree is not implemented."""
        raise NotImplementedError

    def delete_min(self):
        """Deletion for RedBlackTree is not implemented."""
        raise NotImplementedError





class RTrieNode:
    """R way trie node."""
    def __init__(self, R, val = None):
        self.R = R
        self.val = val
        self.next = [None] * R
        self._num_links = 0
    
    def __getitem__(self, idx):
        assert idx < self.R, f"{idx} is out of range of {self.R} way trie node."
        return self.next[idx]

    def __setitem__(self, idx, item):
        assert idx < self.R, f"idx, is out of range of {self.R} way trie node."
        if not (item is None or isinstance(item, RTrieNode)):
            raise ValueError("The item provided is not None or RTrieNode")
        if self.next[idx] == None and item is not None:
            self._num_links += 1
        elif self.next[idx] is not None and item is None:
            self._num_links -= 1
        self.next[idx] = item
    
    def __bool__(self):
        return self.val is not None or self._num_links != 0
    


class TrieTemplate(ABC):
    """Trie template."""
    def __setitem__(self, key, val):
        self.put(key, val)

    def __getitem__(self, key):
        return self.get(key)

    def __contains__(self, key):
        try:
            self.get(key)
            return True
        except KeyError:
            return False

    def __delitem__(self, key):
        self.delete(key)

    def __iter__(self):
        return iter(self.keys())

    @abstractmethod
    def put(self, key):
        """Insert a string key into the trie."""
        pass

    @abstractmethod
    def get(self, key):
        """Get the value corrsponding to the string key."""
        pass

    @abstractmethod
    def delete(self, key):
        """Delete the key from trie."""
        pass
    
    @abstractmethod
    def keys(self):
        """Get the list of keys in the trie."""
        pass


class RWayTrie(TrieTemplate):
    """R-way trie.
    
    This implementation is takes heavy reference from Robert Sedgewick and Kevin Wayne's
    Pricenton Algorithm course on Coursera.
    """
    def __init__(self, R = 256, char2idx = None):
        """Init the trie with a radis R and a char2idx mapping.
        
        char2idx is a dictionary that maps a single char string to a number that is
        smaller than the given R. ord(char) is used if it is not provided.
        """
        self.char2idx = char2idx
        if char2idx is not None:
            self.idx2char = {}
            for c, i in char2idx.items():
                self.idx2char[i] = c
        else:
            self.idx2char = None

        self.R = R
        self.root = self._node()
    

    def _to_idx(self, c):
        if self.char2idx is not None:
            return self.char2idx[c]
        return ord(c)

    def _to_char(self, idx):
        if self.idx2char is not None:
            return self.idx2char[idx]
        return chr(idx)

    def put(self, key, val):
        """Add a key and its associated value into the trie."""
        self.root = self._put(self.root, key, val, 0)

    def _put(self, node, key, val, d):
        if node is None:
            node = self._node()
        if d == len(key):
            node.val = val
            return node
        c = key[d]
        node[self._to_idx(c)] = self._put(node[self._to_idx(c)], key, val, d + 1)
        return node

    
    def get(self, key):
        """Get the value associated with the key in the trie."""
        node = self._get(self.root, key, 0)
        if node is None or node.val is None:
            raise KeyError(key)
        return node.val


    def _get(self, node, key, d):
        if node is None:
            return None
        if d == len(key):
            return node
        c = key[d]
        return self._get(node[self._to_idx(c)], key, d + 1)

    def _node(self, val = None):
        return RTrieNode(self.R, val)


    def delete(self, key):
        """Delete the key from trie."""
        self._delete(self.root, key, 0)

    
    def _delete(self, node, key, d):
        if node is None:
            raise KeyError
        if d == len(key):
            if node.val is None:
                raise KeyError
            node.val = None
            if node:
                return False
            return True
                
        c = key[d]
        if self._delete(node[self._to_idx(c)], key, d + 1):
            # print("node", node[self._to_idx(c)])
            node[self._to_idx(c)] = None
            if node:
                return False
            return True
        return False


    def keys(self):
        """Return a list of all keys in the trie."""
        prefixes = []
        self._collect(self.root, "", prefixes)
        return prefixes

    def _collect(self, node, prefix, prefixes):
        if node is None:
            return
        if node.val is not None:
            prefixes.append(prefix)
        for idx in range(self.R):
            self._collect(node[idx], prefix + self._to_char(idx), prefixes)

    def keys_with_prefix(self, prefix):
        """Return all key in the trie with the given prefix."""
        prefixes = []
        node = self._get(self.root, prefix, 0)
        self._collect(node, prefix, prefixes)
        return prefixes

    def longest_prefix_of(self, query):
        """Return the longest prefix of the query that is a key in the trie."""
        length = self._search(self.root, query, 0, 0)
        return query[:length]


    def _search(self, node, query, d, length):
        if node is None:
            return length
        if node.val is not None:
            length = d
        if d == len(query):
            return length
        c = query[d]
        return self._search(node[self._to_idx(c)], query, d, length)


@total_ordering
class TSTNode:
    """Ternary search trie node."""
    def __init__(self, char, val = None):
        self.char, self.val = char, val
        self.left, self.mid, self.right = None, None, None

    def __eq__(self, item):
        if isinstance(item, str):
            return self.char == item
        return self.char == item.char

    def __lt__(self, item):
        if isinstance(item, str):
            return self.char < item
        return self.char < item.char

    def __bool__(self):
        return self.left is not None \
                or self.right is not None \
                or self.mid is not None \
                or self.val is not None


class TST(TrieTemplate):
    """Ternary search trie."""
    def __init__(self):
        self.root = None

    def put(self, key, val):
        """Put the key and its associated value in the trie."""
        self.root = self._put(self.root, key, val, 0)

    def _put(self, node, key, val, d):
        c = key[d]
        if node is None:
            node = TSTNode(c)
            node.char = c
        if c < node:
            node.left = self._put(node.left, key, val, d)
        elif c > node:
            node.right = self._put(node.right, key, val, d)
        elif d < len(key) - 1:
            # only increment the depth if we are going down the middle
            # link
            node.mid = self._put(node.mid, key, val, d + 1)
        else:
            node.val = val
        return node

    def get(self, key):
        """Return the value that is associated with the given key."""
        node = self._get(self.root, key, 0)
        if node is None or node.val is None:
            raise KeyError(key)
        return node.val

    def _get(self, node, key, d):
        if node is None:
            return None
        c = key[d]
        if c < node:
            return self._get(node.left, key, d)
        elif c > node:
            return self._get(node.right, key, d)
        elif d < len(key) - 1: # if increment d if we matched the key
            return self._get(node.mid, key, d + 1)
        else: # if c == node.char and we reached the end of the string
            return node

    def delete(self, key):
        """Delete the given key from the trie."""
        if self._delete(self.root, key, 0):
            self.root = None

    def _delete(self, node, key, d):
        if node is None:
            raise KeyError(key)
        c = key[d]
        if c < node:
            if self._delete(node.left, key, d):
                node.left = None
                if node:
                    return False
                return True
        elif c > node:
            if self._delete(node.right, key, d):
                node.right = None
                if node:
                    return False
                return True
        elif d < len(key) - 1:
            if self._delete(node.mid, key, d + 1):
                node.mid = None
                if node:
                    return False
                return True
        else:
            if node.val is None:
                raise KeyError(key)
            node.val = None
            if node:
                return False
            return True

    def keys(self):
        """Return a list of all keys that are in the trie."""
        prefixes = []
        self._collect(self.root, "", prefixes)
        return prefixes


    def _collect(self, node, prefix, prefixes):
        if node is None:
            return
        if node.val is not None:
            prefixes.append(prefix)
        prefix += node.char
        for next_node in [node.left, node.mid, node.right]:
            self._collect(next_node, prefix, prefixes)

    def keys_with_prefix(self, prefix):
        """Return all keys that have the given prefix."""
        node = self._get(self.root, prefix, 0)
        prefixes = []
        self._collect(node, prefix, prefixes)
        return prefixes

    def longest_prefix_of(self, query):
        """Return the longest prefix of the query that is also a key in the trie."""
        length = self._search(self.root, query, 0, 0)
        return query[:length]


    def _search(self, node, query, d, length):
        if node is None:
            return length
        if node.val is not None:
            length = d
        if d == len(query):
            return length
        c = query[d]
        if c < node:
            return self._search(node.left, query, d, length)
        elif c > node:
            return self._search(node.right, query, d, length)
        else:
            return self._search(node.mid, query, d + 1, length)




if __name__ == "__main__":

    # # Testing unbalanced BST

    # tree = BinarySearchTree()
    # keys = [3, 2, 1, 4, 7, 5, 3, 9, 3.5, 3.3, 6]
    # for k in keys:
    #     tree.put(k, 0)

    # print(tree.max())
    # print(tree.min())
    # print(tree.predecessor(6.8))
    # print(tree.successor(4.5))
    # print(tree.rank(3.4))
    # print(len(tree))
    # print(tree.select(4))
    # tree.delete_max()
    # print(tree.max())
    # tree.delete(4)
    # print(len(tree))
    # print(tree.root.right.key)


    # # Testing left leaning red black tree

    # tree = LLRedBlackTree()
    # keys = [3, 2, 1, 4, 7, 5, 3, 9, 3.5, 3.3, 6]
    # for k in keys:
    #     tree.put(k, 0)
    #     print("root count:", tree.root.count)
    #     # print(len(tree))
    # print(tree.in_order_trav())
    # # print(tree.root.left.key)
    # # print(tree.root.left.left)

    # # Testing RWayTrie
    # print("R Way Trie")
    # trie = RWayTrie()
    # trie["adw"] = 3
    # # print(trie["adw"])
    # trie["ab"] = 2
    # trie["ac"] = 1
    # trie["adcd"] = 2
    # trie["adc"] = 2
    # trie["acc"] = 1
    # trie["bcsdafiow"] = 2
    # del trie["adc"]
    # del trie["adw"]
    # for key in trie:
    #     print(key)
    # print(trie.longest_prefix_of("acc"))
    # print(trie.keys_with_prefix("ac"))

    # Testing TST
    print("Ternary Search Trie")
    trie = TST()
    trie["adw"] = 3
    print(trie["adw"])
    trie["ab"] = 2
    trie["ac"] = 1
    trie["adcd"] = 2
    trie["adc"] = 2
    trie["acc"] = 1
    trie["bcsdafiow"] = 2
    del trie["adc"]
    del trie["adw"]
    print(trie["adcd"])
    for key in trie:
        print(key)
    print(trie.longest_prefix_of("acb"))
    print(trie.keys_with_prefix("ac"))

    
