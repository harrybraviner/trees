import unittest
from tree_node import TreeNode, uniquify_list, uniquify_index_list, CostTracker

class MiscTests(unittest.TestCase):
    def test_uniquify_empty_list(self):
        self.assertEqual([], uniquify_list([]))

    def test_uniquify_unique_list(self):
        self.assertEqual([1, 3, 4, 7], uniquify_list([1, 3, 4, 7]))

    def test_uniquify_redundant_list(self):
        self.assertEqual([1, 3, 4, 7], uniquify_list([1, 3, 3, 3, 3, 3, 3, 4, 7, 7]))

    def test_uniquify_empty_index_list(self):
        self.assertEqual([], uniquify_index_list([]))

    def test_uniquify_unique_index_list(self):
        self.assertEqual([(1.0, [3]), (2.0, [2]), (3.0, [1])], uniquify_index_list([(1.0, 3), (2.0, 2), (3.0, 1)]))

    def test_uniquify_redundant_index_list(self):
        self.assertEqual([(1.0, [3, 2]), (3.0, [1])], uniquify_index_list([(1.0, 3), (1.0, 2), (3.0, 1)]))

class CostTrackerTests(unittest.TestCase):
    def test_init(self):
        ct = CostTracker([1.0, 2.0, 3.0])
        self.assertEqual(0, ct.get_left_N())
        self.assertEqual(3, ct.get_right_N())
        self.assertEqual(None, ct.get_left_mean())
        self.assertEqual(2.0, ct.get_right_mean())
        self.assertEqual(0.0, ct.get_left_cost())
        self.assertEqual(2.0, ct.get_right_cost())
        self.assertEqual(2.0, ct.get_total_cost())

    def test_move_one(self):
        ct = CostTracker([1.0, 2.0, 3.0])
        ct.move_value_right_to_left(1.0)
        self.assertEqual(1, ct.get_left_N())
        self.assertEqual(2, ct.get_right_N())
        self.assertEqual(1.0, ct.get_left_mean())
        self.assertEqual(2.5, ct.get_right_mean())
        self.assertEqual(0.0, ct.get_left_cost())
        self.assertEqual(0.5, ct.get_right_cost())
        self.assertEqual(0.5, ct.get_total_cost())

    def test_move_two(self):
        ct = CostTracker([1.0, 2.0, 3.0])
        ct.move_value_right_to_left(1.0)
        ct.move_value_right_to_left(2.0)
        self.assertEqual(2, ct.get_left_N())
        self.assertEqual(1, ct.get_right_N())
        self.assertEqual(1.5, ct.get_left_mean())
        self.assertEqual(3.0, ct.get_right_mean())
        self.assertEqual(0.5, ct.get_left_cost())
        self.assertEqual(0.0, ct.get_right_cost())
        self.assertEqual(0.5, ct.get_total_cost())
