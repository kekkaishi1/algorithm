#!/usr/bin/env python
# -*- coding:utf-8 -*-

__Author__ = 'Lin Xin'

import unittest
import random
from algorithm import Algorithm
from data_structure import BinaryTree


class SortTest(unittest.TestCase):
    def setUp(self):
        self.al = Algorithm()
        a_list = []
        self.ls = []
        for i in range(10, 20):
            for j in range(i):
                a_list.append(random.uniform(0, 100))
            self.ls.append(a_list)

    def test_bubble_sort(self):
        for l in self.ls:
            l_sorted = sorted(l)
            self.al.bubble_sort(l)
            self.assertEqual(l, l_sorted)

    def test_quick_sort(self):
        for l in self.ls:
            self.assertEqual(self.al.quick_sort(l), sorted(l))

    def test_quick_sort_simple(self):
        for l in self.ls:
            l_sorted = sorted(l)
            self.al.quick_sort_simple(l)
            self.assertEqual(l, l_sorted)

    def test_merge_sort_no_slice(self):
        for l in self.ls:
            self.assertEqual(self.al.merge_sort_no_slice(l), sorted(l))

    def test_merge_sort(self):
        for l in self.ls:
            self.assertEqual(self.al.merge_sort(l), sorted(l))

    def test_insertion_sort(self):
        for l in self.ls:
            l_sorted = sorted(l)
            self.al.insertion_sort(l)
            self.assertEqual(l, l_sorted)

    def test_selection_sort(self):
        for l in self.ls:
            l_sorted = sorted(l)
            self.al.selection_sort(l)
            self.assertEqual(l, l_sorted)

    def test_short_bubble_sort(self):
        for l in self.ls:
            l_sorted = sorted(l)
            self.al.short_bubble_sort(l)
            self.assertEqual(l, l_sorted)

    def test_shell_sort(self):
        for l in self.ls:
            l_sorted = sorted(l)
            self.al.shell_sort(l)
            self.assertEqual(l, l_sorted)


class TreeTest(unittest.TestCase):
    def setUp(self):
        self.al = Algorithm()
        self.tree = BinaryTree(1)
        tree = self.tree
        tree.left = BinaryTree(2)
        tree.right = BinaryTree(3)
        tree = tree.left
        tree.left = BinaryTree(4)
        tree.right = BinaryTree(5)
        tree = self.tree
        tree = tree.right
        tree.left = BinaryTree(6)
        tree.right = BinaryTree(7)
        tree.right.right = BinaryTree(8)

    def test_pre_order(self):
        self.assertEqual(self.al.pre_order(self.tree), self.tree.pre_order())

    def test_in_order(self):
        self.assertEqual(self.al.in_order(self.tree), self.tree.in_order())

    def test_last_order(self):
        self.assertEqual(self.al.last_order(self.tree), self.tree.last_order())

    def test_mirror_tree(self):
        self.al.mirror_tree(self.tree)
        self.assertEqual(self.tree.pre_order(),[1,3,7,8,6,2,5,4])


if __name__ == "__main__":
    unittest.main()
