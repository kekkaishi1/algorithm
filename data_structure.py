#!/usr/bin/env python
# -*- coding:utf-8 -*-

__Author__ = 'Lin Xin'


class Stack:
    def __init__(self):
        self.s = []

    def add(self, e):
        self.s.append(e)

    def pop(self):
        return self.s.pop()

    def is_empty(self):
        return not bool(self.s)

    def get_top(self):
        return self.s[-1]


class Chain:
    def __init__(self, data=None):
        self.data = data
        self.next = None

    def _end(self):
        c_node = self
        while c_node.next:
            c_node = c_node.next
        return c_node

    def add_item(self, item):
        self._end().next = Chain(item)

    def reverse(self):
        n1 = self
        n2 = n1.next
        n3 = n2.next
        self.next = None
        while n3:
            n2.next = n1
            n1 = n2
            n2 = n3
            n3 = n3.next
        n2.next = n1
        return n2

    def show(self):
        n = self
        while n:
            print(n.data)
            n = n.next

    def pair_reverse(self, head):
        if head and head.next:
            next = head.next
            head.next = self.pair_reverse(next.next)
            next.next = head
            return next
        return head


class BinaryTree:
    def __init__(self, root=None, left=None, right=None):
        self.root = root
        self.left = left
        self.right = right

    def add_left(self, item):
        self.left = BinaryTree(item)

    def add_right(self, item):
        self.right = BinaryTree(item)

    def in_order(self):
        if self.root:
            if self.left:
                self.left.in_order()
            print(self.root)
            if self.right:
                self.right.in_order()

    def pre_order(self):
        if self.root:
            print(self.root)
            if self.left:
                self.left.pre_order()
            if self.right:
                self.right.pre_order()

    def last_order(self):
        if self.root:
            if self.left:
                self.left.pre_order()
            if self.right:
                self.right.pre_order()
            print(self.root)

    def __str__(self):
        return self.root

    def in_order_xiansuo(self, pre=None):
        if self.left:
            self.ltag = 0
            pre = self.left.inorder_xiansuo(pre)
        else:
            self.ltag = 1
            self.left = pre
        if pre and pre.rtag == 1:
            pre.right = self
        pre = self
        if self.right:
            self.rtag = 0
            pre = self.right.inorder_xiansuo(pre)
        else:
            self.rtag = 1
        return pre

    def get_depth(self):
        if self.root:
            if self.left or self.right:
                if self.left and self.right:
                    return max(self.left.get_depth(),self.right.get_depth())+1
                elif self.left:
                    return self.left.get_depth()+1
                else:
                    return self.right.get_depth()+1
            else:
                return 1
        else:
            return 0


class BinHeap:
    MODE = {0: lambda x, y: x > y, 1: lambda x, y: x < y}

    def __init__(self, mode=0):
        self.heaplist = [0]
        self.size = 0
        self._mode = mode  # 0 for small 1 for big
        self.mode = BinHeap.MODE[self._mode]

    def insert(self, item):
        self.size += 1
        self.heaplist[self.size] = item
        self.perc_up(self.size)

    def perc_up(self, move_element):
        p = move_element // 2
        while p > 0:
            if self.mode(self.heaplist[p], self.heaplist[move_element]):
                self.heaplist[p], self.heaplist[move_element] = self.heaplist[move_element], self.heaplist[p]
            move_element //= 2
            p = move_element // 2

    def pop(self):
        top = self.heaplist[1]
        self.heaplist[1] = self.heaplist[self.size]
        self.size -= 1
        self.perc_down(1)
        return top

    def top(self):
        return self.heaplist[self.size]

    def perc_down(self, move_element):
        while move_element <= self.size // 2:
            if 2 * move_element + 1 > self.size:
                child = 2 * move_element
            else:
                if self.mode(self.heaplist[2 * move_element], self.heaplist[2 * move_element + 1]):
                    child = 2 * move_element + 1
                else:
                    child = 2 * move_element
            if self.mode(self.heaplist[move_element], self.heaplist[child]):
                self.heaplist[move_element], self.heaplist[child] = self.heaplist[child], self.heaplist[move_element]
            move_element = child

    def build_from_list(self, alist):
        self.heaplist.extend(alist)
        self.size = len(alist)
        for i in range(self.size // 2, 0, -1):
            self.perc_down(i)

    def show(self):
        base_length = 2 ** (int(__import__('math').log2(self.size)) + 1)
        items = iter(self.heaplist[1:self.size + 1])
        i = 2
        while i <= base_length * 2:
            show_list = ['  '] * base_length
            for j in range(1, i, 2):
                try:
                    show_list[base_length // i * j] = str(next(items))
                except StopIteration:
                    break
            print(''.join(show_list))
            i *= 2

    def is_empty(self):
        return self.size == 0


class BinarySearchTree:

    def __init__(self):
        self.root = None
        self.size = 0

    def length(self):
        return self.size

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.root.__iter__()

    def put(self, key, value):
        if not self.root:
            self.root = TreeNode(key, value)
        else:
            self._put(key, value, self.root)

    def _put(self, key, value, current_node):
        if key < current_node.key:
            if current_node.leftChild:
                self._put(key, value, current_node.leftChild)
            else:
                current_node.leftChild = TreeNode(key, value)
        elif key == current_node.key:
            current_node.payload = value
        else:
            if current_node.rightChild:
                self._put(key, value, current_node.rightChild)
            else:
                current_node.rightChild = TreeNode(key, value)

    def __contains__(self, key):
        return self.get(key) and True

    def get(self, key):
        if self.root.key == key:
            return self.root.payload
        elif self.root.key > key:
            return self._get(key, self.root.leftChild) or print("get Nothing")
        else:
            return self._get(key, self.root.rightChild) or print("get Nothing")

    def _get(self, key, current_node):
        if current_node.key == key:
            return current_node.payload
        elif current_node.key > key:
            if current_node.leftChild:
                return self._get(key, current_node.leftChild)
            else:
                return False
        else:
            if current_node.rightChild:
                return self._get(key, current_node.rightChild)
            else:
                return False


class TreeNode:
    def __init__(self, key, val, left=None, right=None,
                 parent=None):
        self.key = key
        self.payload = val
        self.leftChild = left
        self.rightChild = right
        self.parent = parent

    def hasLeftChild(self):
        return self.leftChild

    def hasRightChild(self):
        return self.rightChild

    def isLeftChild(self):
        return self.parent and self.parent.leftChild == self

    def isRightChild(self):
        return self.parent and self.parent.rightChild == self

    def isRoot(self):
        return not self.parent

    def isLeaf(self):
        return not (self.rightChild or self.leftChild)

    def hasAnyChildren(self):
        return self.rightChild or self.leftChild

    def hasBothChildren(self):
        return self.rightChild and self.leftChild

    def replaceNodeData(self, key, value, lc, rc):
        self.key = key
        self.payload = value
        self.leftChild = lc
        self.rightChild = rc
