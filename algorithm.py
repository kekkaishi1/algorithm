#!/usr/bin/env python
# -*- coding:utf-8 -*-

__Author__ = 'Lin Xin'

"""
一些算法题记录
"""

from data_structure import Stack, BinHeap, Chain, BinaryTree


class Algorithm:

    # 1、硬币找零问题
    @staticmethod
    def change_num(coin_value_list, change, known_results):
        """
        使用备忘录方法递归实现
        :param coin_value_list: 零钱种类
        :param change: 找零总额
        :param known_results: 计算过的找零方案，默认为{}
        :return: 找零数目
        """
        if not known_results:
            known_results = {}
        least_num = change
        if change in coin_value_list:
            known_results[change] = 1
            return 1
        elif change in known_results:
            return known_results[change]
        else:
            for coin in [coin for coin in coin_value_list if coin <= change]:
                this_coin_num = Algorithm.change_num(coin_value_list, change - coin, known_results) + 1
                least_num = min(this_coin_num, least_num)
            known_results[change] = least_num
            return least_num

    @staticmethod
    def dp_change_num(coin_value_list, change):
        """
        使用动态规划实现
        :param coin_value_list: 零钱种类
        :param change: 找零总额
        :return: 找零数目,零钱使用情况
        """
        min_coins_num = list(range(max(coin_value_list)))
        min_coins = {}
        for ch in range(1, change + 1):
            for coin in [coin for coin in coin_value_list if coin <= ch]:
                if min_coins_num[ch] > 1 + min_coins_num[ch - coin]:
                    min_coins_num[ch] = 1 + min_coins_num[ch - coin]
                    min_coins[ch] = coin
        coin_used = []
        last = change
        while last > 0:
            coin_used.append(min_coins[last])
            last -= coin_used[-1]
        return min_coins_num[change], coin_used

    # 2、排序算法
    @staticmethod
    def bubble_sort(l):
        """
        冒泡排序：相邻元素比较后按顺序交换位置.交换位置操作多
        时间复杂度：O(n^2)
        :param l: 待排序列表
        :return: None
        """
        for i in range(len(l) - 1, 0, -1):
            for j in range(i):
                if l[j] > l[j + 1]:
                    l[j], l[j + 1] = l[j + 1], l[j]

    @staticmethod
    def short_bubble_sort(l):
        """
        短冒泡排序：若某次遍历无交换，则提前结束
        :param l: 待排序列表
        :return: None
        """
        for i in range(len(l) - 1, 0, -1):
            exchange = False
            for j in range(i):
                if l[j] > l[j + 1]:
                    exchange = True
                    l[j], l[j + 1] = l[j + 1], l[j]
            if not exchange:
                break

    @staticmethod
    def selection_sort(l):
        """
        选择排序：比较出最大的元素后进行位置交换
        时间复杂度：O(N^2)
        :param l: 待排序列表
        :return: None
        """
        for i in range(len(l) - 1, 0, -1):
            max_index = i
            for j in range(i):
                if l[j] > l[max_index]:
                    max_index = j
            l[max_index], l[i] = l[i], l[max_index]

    @staticmethod
    def insertion_sort(l):
        """
        插入排序：将列表前端认为是有序序列，依次将后续元素插入有序序列
        时间复杂度：O(N^2)
        :param l: 待排序列表
        :return: None
        """
        for i in range(1, len(l)):
            # 插入 l[i] 到 l[:i]
            temp = l[i]
            for j in range(i - 1, -1, -1):
                if l[j] > temp:
                    l[j + 1] = l[j]
                else:
                    l[j + 1] = temp
                    break
            else:
                l[0] = temp

    @staticmethod
    def shell_sort(l):
        """
        希尔排序：将列表按gap分组，对每组进行插入排序后组合为新列表并减小gap，最终进行一次插入排序
        :param l: 待排序列表
        :return: None
        """
        gap = len(l) // 2
        while gap > 0:
            for s in range(gap):
                for i in range(s + gap, len(l), gap):
                    temp = l[i]
                    for j in range(i - gap, s - 1, -gap):
                        if l[j] > temp:
                            l[j + gap] = l[j]
                        else:
                            l[j + gap] = temp
                            break
                    else:
                        l[s] = temp
            gap //= 2

    @staticmethod
    def merge_sort(l):
        """
        归并排序：采用分治法思想，递归地将列表分成两部分并排序后，将两部分有序子列表插入新列表中
        时间复杂度：O(n log n)
        :param l: 待排序列表
        :return: 排序完成列表
        """
        if len(l) == 1:
            return l
        i = j = 0
        half_length = len(l) // 2
        l_left = Algorithm.merge_sort(l[:half_length])
        l_right = Algorithm.merge_sort(l[half_length:])
        l_sorted = []
        while i != half_length and j != len(l) - half_length:
            if l_left[i] > l_right[j]:
                l_sorted.append(l_right[j])
                j += 1
                flag = 0
            else:
                l_sorted.append(l_left[i])
                i += 1
                flag = 1
        if flag:
            l_sorted += l_right[j:]
        else:
            l_sorted += l_left[i:]
        return l_sorted

    @staticmethod
    def merge_sort_no_slice(l):
        """
        归并排序 - 无切片
        :param l:待排序列表
        :return:排序列表
        """
        return Algorithm._merge_sort_no_slice(l, 0, len(l) - 1)

    @staticmethod
    def _merge_sort_no_slice(l, start, end):
        """
        归并排序 - 无切片  辅助函数
        :param l:待排序列表
        :param start:起始索引
        :param end:终止索引
        :return:排序列表
        """
        if start == end:
            return [l[start]]
        half_length = (end + start) // 2
        l_left = Algorithm._merge_sort_no_slice(l, start, half_length)
        l_right = Algorithm._merge_sort_no_slice(l, half_length + 1, end)
        l_sorted = []
        i = j = 0
        while i <= half_length - start and j <= end - half_length - 1:
            if l_left[i] > l_right[j]:
                l_sorted.append(l_right[j])
                j += 1
                flag = 0
            else:
                l_sorted.append(l_left[i])
                i += 1
                flag = 1
        if flag:
            l_sorted += l_right[j:]
        else:
            l_sorted += l_left[i:]
        return l_sorted

    @staticmethod
    def quick_sort(l):
        """
        快速排序
        :param l: 待排序列表
        :return: 排序列表
        """
        return Algorithm.quick_sort([i for i in l[1:] if i < l[0]]) + l[0] + Algorithm.quick_sort(
            [i for i in l[1:] if i >= l[0]])

    # 3、最长公共子串

    @staticmethod
    def lis(s1, s2):
        """
        寻找最长公共子串
        f(i,j)=f(i-1,j-1)+1 if s1[i]=s2[j] else 0
        :param s1:字符串1
        :param s2:字符串2
        :return:
        """
        lis = []
        max_length = 0
        dp = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                    if max_length < dp[i + 1][j + 1]:
                        max_length = dp[i + 1][j + 1]
                        lis = [(i + 1, j + 1)]
                    elif max_length == dp[i + 1][j + 1]:
                        lis.append((i + 1, j + 1))
                else:
                    dp[i + 1][j + 1] = 0
        return [s1[l[0] - max_length:l[0]] for l in lis], max_length

    # 4、最长公共子序列

    @staticmethod
    def lcs(s1, s2):
        """
        寻找最长公共子序列
        f(i,j)=f(i-1,j-1)+1 if s1[i]=s2[j] else max(f(i-1,j),f(i,j-1))
        :param s1:字符串1
        :param s2:字符串2
        :return:
        """
        lcs = []
        dp = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                    lcs.append(s1[i])
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
        return dp[len(s1)][len(s2)], ''.join(lcs)

    # 5、二分查找
    @staticmethod
    def binary_search(alist, find):
        start = 0
        end = len(alist)
        while start <= end:
            mid = (start + end) // 2
            if alist[mid] == find:
                return mid
            elif alist[mid] > find:
                end = mid - 1
            else:
                start = mid + 1
        return -1

    # 6、两个栈实现队列
    class QueueByStack:
        def __init__(self):
            self.stack_in = Stack()
            self.stack_out = Stack()

        def put(self, element):
            self.stack_in.add(element)

        def get(self):
            if not self.stack_out.is_empty():
                return self.stack_out.pop()
            elif not self.stack_in.is_empty():
                while not self.stack_in.is_empty():
                    self.stack_out.add(self.stack_in.pop())
                return self.stack_out.pop()
            else:
                print('queue is empty')
                return None

    # 7、无重复字符最长子串
    @staticmethod
    def max_substring(s):
        max_length = 1
        string_map = {s[0]: 0}
        start = i = j = 0
        while j < len(s) - 1:
            j += 1
            if s[j] in string_map:
                temp = string_map[s[j]]
                for x in s[i:string_map[s[j]]]:
                    del string_map[x]
                string_map[s[j]] = j
                i = temp + 1
            else:
                string_map[s[j]] = j
                if len(string_map) > max_length:
                    max_length += 1
                    start = i
        return s[start:max_length + start]

    # 8、斐波那契数列
    @staticmethod
    def fib():
        a, b = 1, 1
        yield a
        while 1:
            a, b = b, a + b
            yield a

    # 9、O（1）时间删除链表节点
    @staticmethod
    def del_node(node_list, del_node):
        del_node.data = del_node.next.data
        del_node.next = del_node.next.next

    # 10、求二进制中1的个数
    @staticmethod
    def one_count(num):
        count = 0
        while num != 0:
            num = num - 1 & num
            count += 1
        return count

    # 11、包含min函数的栈
    class StackWithMin(Stack):
        def __init__(self):
            super().__init__()
            self.min_stack = Stack()

        def add(self, e):
            super().add(e)
            if self.min_stack.is_empty():
                self.min_stack.add(e)
            else:
                if self.min_stack.get_top() > e:
                    self.min_stack.add(e)

        def pop(self):
            top = self.pop()
            if top == self.min_stack.get_top():
                self.min_stack.pop()
            return top

        def get_min(self):
            return self.min_stack.get_top()

    # 12、连续子数组最大值
    @staticmethod
    def max_sub_seq(alist):
        sum = 0
        sub = []
        for i in alist:
            if sum + i < i:
                sum = i
                sub = [i]
            else:
                sum += i
                sub.append(i)
        return sub, sum

    # 13、最小k个数
    @staticmethod
    def min_k(alist, k):
        heap = BinHeap(mode=1)
        heap.build_from_list(alist[:k])
        for i in alist[k:]:
            if heap.top() > i:
                heap.pop()
                heap.insert(i)
        while not heap.is_empty():
            print(heap.pop())

    # 14、数组中出现次数超过一半的数字
    @staticmethod
    def gt_half(alist):
        m = {}
        for i in alist:
            if i in m:
                m[i] += 1
            else:
                if m:
                    key = list(m.keys())[0]
                    m[key] -= 1
                    if m[key] == 0:
                        del m[key]
                else:
                    m[i] = 1
        return list(m.keys())[0]

    # 15、链表倒数第K个元素
    @staticmethod
    def get_last_k(chain, k):
        x = y = chain
        for i in range(k):
            y = y.next
        while y:
            x = x.next
            y = y.next
        return x

    # 16、重建二叉树
    # 输入某二叉树的前序遍历和中序遍历的结点，请重建出该二叉树
    @staticmethod
    def rebuild_binary_tree(pre_order, in_order):
        if len(pre_order) < 1:
            return BinaryTree()
        root_index = in_order.index(pre_order[0])
        left_in_order = in_order[:root_index]
        right_in_order = in_order[root_index + 1:]
        left_pre_order = pre_order[1:len(left_in_order) + 1]
        right_pre_order = pre_order[len(left_in_order) + 1:]
        return BinaryTree(pre_order[0], Algorithm.rebuild_binary_tree(left_pre_order, left_in_order),
                          Algorithm.rebuild_binary_tree(right_pre_order, right_in_order))

    # 17、权重可为负数的最短路径
    @staticmethod
    def bellman_ford(g, start, end):
        """
        求最短路径
        权值可为负数
        :param g: 有向图
                e.g.:
                    g = { 'A': {('B', -1), ('C', 4)},
                          'B': {('C', 3), ('D', 2), ('E', 2)},
                          'D': {('B', 1), ('C', 5)},
                          'C': {},
                          'E': {('D', -3)}
                        }
        :param start: 起始点
        :param end: 终止点
        :return: 最短路径，路径节点
        """
        result = {V: 1000 for V in g}  # 各点到起始点距离
        path = {}
        result[start] = 0
        for _ in range(len(g) - 1):  # 遍历n-1次（松弛）
            for v, es in g.items():  # 更新所有边
                for e in es:
                    if e[1] + result[v] < result[e[0]]:  # 起始点经过中间点到目标点距离小于原距离则更新
                        result[e[0]] = e[1] + result[v]
                        path[e[0]] = v
        for v, es in g.items():  # 再松弛一次，若结果变化，则存在负循环
            for e in es:
                if e[1] + result[v] < result[e[0]]:
                    return False
        v = [end]
        while v[0] != start:
            v.insert(0, path[v[0]])
        return result[end], v

    # 18、dijkstra算法求最短路径
    @staticmethod
    def dijkstra(g, start, end):
        """
        权值为非负数
        :param g: 有向图
                e.g.:
                    g = { 'A': {('B', 1), ('C', 4)},
                          'B': {('C', 3), ('D', 2), ('E', 2)},
                          'D': {('B', 1), ('C', 5)},
                          'C': {},
                          'E': {('D', 3)}
                        }
        :param start: 起始点
        :param end: 终止点
        :return: 最短路径，路径节点
        """
        result = {V: 1000 for V in g}  # 各点到起始点距离
        result[start] = 0
        s_spot = [start]  # 优化后的源点
        v_spot = [v for v in g.keys() if v != start]  # 未优化的点
        path = {}
        while end in v_spot:
            for es in g[s_spot[-1]]:  # 对最新加入源点所能连接到的点进行更新
                if es[1] + result[s_spot[-1]] < result[es[0]]:  # 起始点经过中间点到目标点距离小于原距离则更新
                    result[es[0]] = es[1] + result[s_spot[-1]]
                    path[es[0]] = s_spot[-1]
            new_spot = [v for v, e in result.items() if min([e for v, e in result.items() if v in v_spot]) == e and v in v_spot][0]
            s_spot.append(new_spot)
            v_spot.remove(new_spot)
        v = [end]
        while v[0] != start:
            v.insert(0, path[v[0]])
        return result[end], '-->'.join(v)

    # 19、求两个有序数组中位数
    @staticmethod
    def median(nums1, nums2):
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        m, n = len(nums1), len(nums2)
        if not m:
            if n % 2 == 1:
                return nums2[n // 2]
            else:
                return (nums2[n // 2] + nums2[n // 2 - 1]) / 2
        i_min, i_max = 0, len(nums1)
        i = (i_min + i_max) // 2 + 1
        j = (m + n) // 2 - i
        while i_min <= i_max:
            if i > 0 and nums2[j] < nums1[i - 1]:
                i_max = i - 1
            elif i < m and nums2[j - 1] > nums1[i]:
                i_min = i + 1
            else:
                # i ok and i is not 0 or m
                break
            i = (i_min + i_max) // 2
            j = (m + n) // 2 - i
        if (m + n) % 2 == 1:
            if i == m:
                return nums2[j]
            return min(nums1[i], nums2[j])
        else:
            if i == 0:
                if m != n:
                    return (min(nums1[i], nums2[j]) + nums2[j - 1]) / 2
                else:
                    return (nums1[i] + nums2[j - 1]) / 2
            if i == m:
                if m != n:
                    return (max(nums1[i - 1], nums2[j - 1]) + nums2[j]) / 2
                else:
                    return (nums1[i - 1] + nums2[j]) / 2
            return (min(nums1[i], nums2[j]) + max(nums1[i - 1], nums2[j - 1])) / 2

    # 20、求字典序
    # 从后向前找到第一组“顺序”的两个数字，将其中左边的数字与从右向左找第一个大于它的数字交换，随后，将后面的数排序
    @staticmethod
    def dict_sequence(s):
        for i in range(len(s) - 1, 0, -1):
            if s[i] > s[i - 1]:
                m = min([ss for ss in s[i:] if ss > s[i - 1]])
                s[s.index(m)], s[i - 1] = s[i - 1], m
                return s[:i] + sorted(s[i:])
        print('max already')

    # 21、螺旋数组
    @staticmethod
    def uzumaki(n):
        result = [[None for _ in range(n)] for _ in range(n)]
        nums = iter(range(1, n * n + 1))
        direction = 1
        row_col = 0
        path_length = n - 1
        pos = [0, n - 1]
        while path_length >= 0:
            for _ in range(4):
                for _ in range(path_length):
                    result[pos[0]][pos[1]] = next(nums)
                    pos[row_col] += direction
                row_col = 1 - row_col
                if row_col:
                    direction = -direction
            path_length -= 2
            pos[0] += 1
            pos[1] -= 1
        for row in result:
            print(row)
