#!/usr/bin/env python
# -*- coding:utf-8 -*-

__Author__ = 'Lin Xin'

import unittest
import random
from algorithm import Algorithm


class AlgorithmTest(unittest.TestCase):
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
            assert l == l_sorted

    def test_quick_sort(self):
        for l in self.ls:
            l_sorted = sorted(l)
            self.al.bubble_sort(l)
            assert l == l_sorted

    def test_merge_sort_no_slice(self):
        for l in self.ls:
            assert self.al.merge_sort(l) == sorted(l)

    def test_merge_sort(self):
        for l in self.ls:
            assert self.al.merge_sort(l) == sorted(l)

    def test_insertion_sort(self):
        for l in self.ls:
            l_sorted = sorted(l)
            self.al.bubble_sort(l)
            assert l == l_sorted

    def test_selection_sort(self):
        for l in self.ls:
            l_sorted = sorted(l)
            self.al.bubble_sort(l)
            assert l == l_sorted

    def test_short_bubble_sort(self):
        for l in self.ls:
            l_sorted = sorted(l)
            self.al.bubble_sort(l)
            assert l == l_sorted

    def test_shell_sort(self):
        for l in self.ls:
            l_sorted = sorted(l)
            self.al.bubble_sort(l)
            assert l == l_sorted


if __name__ == "__main__":
    unittest.main()
