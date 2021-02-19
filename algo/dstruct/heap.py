from abc import ABC, abstractmethod
from random import randrange
from typing import Pattern

__all__ = ["IndexedMinHeap", "IndexedMaxHeap", "MinHeap", "MaxHeap"]


class HeapTemplate(ABC):
    """Parent class for UnIndexedHeap and IndexedHeap to inherit."""
    def __init__(self):
        # heap is one-indexed
        self.reset()
    
    @abstractmethod
    def _type(self):
        # This is meant to be a compulsory class attribute for subclass
        pass

    @property
    def type(self):
        return self._type

    def parent(self, node):
        """Return the parent idx of the given node."""
        parent = node // 2
        if parent == 0:
            return None
        return parent
    
    def lchild(self, node):
        """Return the left child idx of the given node."""
        child = node * 2
        if child >= len(self._heap):
            return None
        return child
    
    def rchild(self, node):
        """Return the right child idx of the given node."""
        child = node * 2 + 1
        if child >= len(self._heap):
            return None
        return child

    def reset(self):
        """Empty the heap."""
        self._heap = [None]
    
    def __len__(self):
        return len(self._heap) - 1

    def __str__(self):
        return f"{self._type} of {self.__len__()} items"

    def __bool__(self):
        return len(self) != 0


class UnIndexedHeap(HeapTemplate):
    _type = "unindexedheap"

    @abstractmethod
    def _swim_cond(self, node_item, parent_item):
        """Given a node and its parent, return if this node should be swapped with its parent."""
        pass

    def _sink_cond(self, node_item, child_item):
        """Given a node and its child, return if this node should be swapped with its child."""
        return not self._swim_cond(node_item, child_item)
    
    def _swim(self, node):
        parent = self.parent(node)
        if parent is None:
            return
        if self._swim_cond(self._heap[node], self._heap[parent]):
            self._heap[node], self._heap[parent] = self._heap[parent], self._heap[node]
            self._swim(parent)
    
    def _sink(self, node):
        lchild = self.lchild(node)
        rchild = self.rchild(node)
        if lchild is None and rchild is None:
            return
        if lchild is None:
            child = rchild
        elif rchild is None:
            child = lchild
        elif self._sink_cond(self._heap[rchild], self._heap[lchild]):
            # for min heap, _sink_cond is a > b
            child = lchild
        else:
            child = rchild

        if self._sink_cond(self._heap[node], self._heap[child]):
            self._heap[node], self._heap[child] = self._heap[child], self._heap[node]
            self._sink(child)

    def delete(self, node):
        """Delete a node while maintaining the heap order."""
        # IMPORTANT: sometimes we need to swim up the tree!!!!!
        # for example, in a min heap, the biggest leaf in a sub tree can be smaller than some
        # node in another sub tree

        assert node >= 1
        self._heap[node] = self._heap[-1]
        del self._heap[-1]
        if node < len(self._heap):
            self._swim(node)
            self._sink(node)

    def insert(self, item):
        """Insert an item while maintaining the heap order."""
        self._heap.append(item)
        self._swim(len(self._heap) - 1)

    def pop(self):
        """Return the root node's item."""
        item = self._heap[1]
        self.delete(1)
        return item


class MinHeap(UnIndexedHeap):
    _type = "minheap"
    def _swim_cond(self, node_item, parent_item):
        """Given a node and its parent, return if this node should be swapped with its parent."""
        return node_item < parent_item

class MaxHeap(UnIndexedHeap):
    _type = "maxheap"
    def _swim_cond(self, node_item, parent_item):
        """Given a node and its parent, return if this node should be swapped with its parent."""
        return node_item > parent_item




class IndexedHeap(HeapTemplate):
    _type = "indexedheap"

    @abstractmethod
    def _swim_cond(self, node_item, parent_item):
        """Given a node and its parent, return if this node should be swapped with its parent."""
        pass

    def _sink_cond(self, node_item, child_item):
        """Given a node and its child, return if this node should be swapped with its child."""
        return not self._swim_cond(node_item, child_item)

    def _swim(self, node):
        parent = self.parent(node)
        if parent is None:
            return
        node_idx = self._heap[node]
        parent_idx = self._heap[parent]
        node_item = self._items[node_idx]
        parent_item = self._items[parent_idx]
        if self._swim_cond(node_item, parent_item):
            self._heap[parent], self._heap[node] = node_idx, parent_idx
            self._heap_idx[parent_idx], self._heap_idx[node_idx] = node, parent

            self._swim(parent)

    def _sink(self, node):
        lchild = self.lchild(node)
        rchild = self.rchild(node)
        if lchild is None and rchild is None:
            return
        if lchild is None:
            child = rchild
            child_idx = self._heap[child]
            child_item = self._items[child_idx]
        elif rchild is None:
            child = lchild
            child_idx = self._heap[child]
            child_item = self._items[child_idx]
        else:
            lchild_idx = self._heap[lchild]
            rchild_idx = self._heap[rchild]
            lchild_item = self._items[lchild_idx]
            rchild_item = self._items[rchild_idx]
            if self._sink_cond(rchild_item, lchild_item):
                child = lchild
                child_idx = lchild_idx
                child_item = lchild_item
            else:
                child = rchild
                child_idx = rchild_idx
                child_item = rchild_item
        node_idx = self._heap[node]
        node_item = self._items[node_idx]
        if self._sink_cond(node_item, child_item):
            self._heap[node], self._heap[child] = child_idx, node_idx
            self._heap_idx[node_idx], self._heap_idx[child_idx] = child, node
            self._sink(child)


    def insert(self, idx, item):
        """Insert an item and associat it with the idx given.
        We can modify the items by providing this idx with self.update method.
        New items inserted must not have the same idx as those that are still in the heap.
        """
        if len(self._items) < idx + 1:
            diff = idx + 1 - len(self._items)
            self._items.extend([None] * diff)
            self._heap_idx.extend([None] * diff)
        if self._items[idx] is not None:
            print(f"idx {idx} is taken, please only insert new item")
            print(f"to modify an item use other methods")
        self._items[idx] = item
        
        self._heap.append(idx)
        self._heap_idx[idx] = len(self._heap) - 1
        self._swim(len(self._heap) - 1)

    def delete_node(self, node):
        """Delete a node."""
        # delete the item from the currently active item
        # and its recorded heap index
        node_idx = self._heap[node]
        self._items[node_idx] = None
        self._heap_idx[node_idx] = None
        # copy the heap idx
        self._heap_idx[self._heap[-1]] = node
        self._heap[node] = self._heap[-1]
        del self._heap[-1]
        if node < len(self._heap):
            self._swim(node)
            self._sink(node)

    def delete_idx(self, idx):
        """Delete the node of the item with the specified idx."""
        self.delete_node(self._heap_idx[idx])

    def update(self, idx, item):
        """Update the item associted with the given idx.
        Heap order is maintained afterwards.
        """
        assert self.contains_idx(idx), f"idx {idx} has not been added."
        self._items[idx] = item
        # there is condition checking in swim and sink
        self._swim(self._heap_idx[idx])
        self._sink(self._heap_idx[idx])

    def pop(self, return_idx = False):
        """Return the root node's item.
        If return_idx is set to True, return item, idx.
        """
        idx = self._heap[1]
        item = self._items[idx]
        self.delete_node(1)
        if return_idx:
            return item, idx
        return item

    def reset(self):
        """Empty the heap."""
        super().reset()
        # super().reset() gives self._heap which stores the item's idx
        # self._heap_idx[3] gives the position in self._heap that correspond to idx 3
        # self._items[3] returns the item that are associated with idx 3
        self._items = []
        self._heap_idx = []
    
    def contains_idx(self, idx):
        """Return if the idx is in the heap."""
        return idx < len(self._heap_idx) and self._heap_idx[idx] is not None


    def get(self, idx):
        """Return the item associated with this idx."""
        assert idx in self
        return self._items[idx]

    def __contains__(self, idx):
        """Return if the idx is in the heap."""
        return self.contains_idx(idx)



class IndexedMinHeap(IndexedHeap):
    def _swim_cond(self, node_item, parent_item):
        """Given a node and its parent, return if this node should be swapped with its parent."""
        return node_item < parent_item

class IndexedMaxHeap(IndexedHeap):
    def _swim_cond(self, node_item, parent_item):
        """Given a node and its parent, return if this node should be swapped with its parent."""
        return node_item > parent_item




if __name__ == "__main__":
    import random

    # testing heap sort on unindexedheap
    min_heap = MinHeap()
    for _ in range(200):
        arr_length = 500
        arr = [random.randint(0, 999) for _ in range(arr_length)]
        for num in arr:
            min_heap.insert(num)
        sorted_arr = []
        while len(min_heap):
            sorted_arr.append(min_heap.pop())

        assert sorted(arr) == sorted_arr
        
    max_heap = MaxHeap()
    for _ in range(200):
        arr_length = 500
        arr = [random.randint(0, 999) for _ in range(arr_length)]
        for num in arr:
            max_heap.insert(num)
        sorted_arr = []
        while len(max_heap):
            sorted_arr.append(max_heap.pop())

        assert sorted(arr, reverse=True) == sorted_arr



    # testing heap sort on indexedheap
    idx_min_heap = IndexedMinHeap()
    for _ in range(200):
        arr_length = 500
        arr = [random.randint(0, 999) for _ in range(arr_length)]
        for i, num in enumerate(arr):
            idx_min_heap.insert(i, num)
        sorted_arr = []
        while len(idx_min_heap):
            sorted_arr.append(idx_min_heap.pop())

        assert sorted(arr) == sorted_arr


    idx_max_heap = IndexedMaxHeap()
    for _ in range(200):
        arr_length = 500
        arr = [random.randint(0, 999) for _ in range(arr_length)]
        for i, num in enumerate(arr):
            idx_max_heap.insert(i, num)
        sorted_arr = []
        while len(idx_max_heap):
            sorted_arr.append(idx_max_heap.pop())

        assert sorted(arr, reverse=True) == sorted_arr


    # testing update method and delete method
    idx_min_heap = IndexedMinHeap()
    arr_length = 500
    for _ in range(200):
        arr = [random.randint(0, 999) for _ in range(arr_length)]
        for i, num in enumerate(arr):
            idx_min_heap.insert(i, num)

        # randomly modify elements
        for _ in range(arr_length):
            idx = random.randint(0, len(arr) - 1)
            val = random.randint(0, 999)
            arr[idx] = val
            idx_min_heap.update(idx, val)

        # randomly delete
        delete_idc = set(random.randint(0, len(arr) - 1) for _ in range(50))
        for idx in delete_idc:
            idx_min_heap.delete_idx(idx)
        
        arr = [v for i, v in enumerate(arr) if i not in delete_idc]

        sorted_arr = []
        while len(idx_min_heap):
            sorted_arr.append(idx_min_heap.pop())
        assert sorted(arr) == sorted_arr


    idx_max_heap = IndexedMaxHeap()
    arr_length = 500
    for _ in range(200):
        arr = [random.randint(0, 999) for _ in range(arr_length)]
        for i, num in enumerate(arr):
            idx_max_heap.insert(i, num)

        # randomly modify elements
        for _ in range(arr_length):
            idx = random.randint(0, len(arr) - 1)
            val = random.randint(0, 999)
            arr[idx] = val
            idx_max_heap.update(idx, val)

        # randomly delete
        delete_idc = set(random.randint(0, len(arr) - 1) for _ in range(50))
        for idx in delete_idc:
            idx_max_heap.delete_idx(idx)

        arr = [v for i, v in enumerate(arr) if i not in delete_idc]

        sorted_arr = []
        while len(idx_max_heap):
            sorted_arr.append(idx_max_heap.pop())
        assert sorted(arr, reverse=True) == sorted_arr
