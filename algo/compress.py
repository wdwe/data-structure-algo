from abc import ABC, abstractmethod
from algo.dstruct import MinHeap, BiNode, RWayTrie

class StrEncoder(ABC):
    @abstractmethod
    def encode(self):
        pass
    @abstractmethod
    def decode(self, mapping, text):
        pass


class Huffman(StrEncoder):
    """Huffman codes."""
    def _build_tree(self, sym_freq):
        heap = MinHeap()
        for sym, freq in sym_freq.items():
            heap.insert(BiNode(freq, sym))
        while len(heap) > 1:
            a = heap.pop()
            b = heap.pop()
            parent = BiNode(a.key + b.key, None, a, b)
            heap.insert(parent)
        return heap.pop()

    def _get_mapping(self, root, mapping, prefix):
        # if root is the leaf node
        if root.left is None and root.right is None:
            mapping[root.val] = "".join(prefix)
        else:
            if root.left is not None:
                prefix.append("0")
                self._get_mapping(root.left, mapping, prefix)
            if root.left is not None:
                prefix.append("1")
                self._get_mapping(root.right, mapping, prefix)
        del prefix[-1]
        return


    def get_mapping(self, sym_freq):
        """Given a dictionary of symbols and corresponding frequencies,
        map each symbol to a string of 0 and 1.
        """
        root = self._build_tree(sym_freq)
        mapping = {}
        self._get_mapping(root, mapping, prefix = [""])
        return mapping

    @staticmethod
    def encode(mapping, text):
        """Return a string of 0 and 1 corresponding to the given text."""
        code = []
        for sym in text:
            code.append(mapping[sym])
        
        return "".join(code)
    
    @staticmethod
    def decode(mapping, text):
        trie = RWayTrie(2, {"0": 0, "1": 1})
        for k, v in mapping.items():
            trie[v] = k
        curr_node = trie.root
        decoded_text = []
        for i, bit in enumerate(text):
            # traverse trie to search for the corresponding val
            # ideally we can search for the longest prefix
            # however, python implements slicing by copying, we
            # will be repetively copying the string if we do that
            if curr_node is None:
                raise ValueError(f"Unable to decode at position {i}")
            curr_node = curr_node[int(bit)]
            if curr_node.val is not None:
                decoded_text.append(curr_node.val)
                curr_node = trie.root
        return "".join(decoded_text)
        
if __name__ == "__main__":
    huffman = Huffman()
    sym_freq = {
        "A": 3,
        "B": 2,
        "C": 6,
        "D": 8,
        "E": 2,
        "F": 6
    }
    huff_map = huffman.get_mapping(sym_freq)
    print(huff_map)
    code = huffman.encode(huff_map, "AEAFBECD")
    print(code)
    decode = huffman.decode(huff_map, code)
    print(decode)

            
