"""Test base data structures"""

import kkpandas
import numpy as np
import unittest
from kkpandas import Folded, Binned

class test1(unittest.TestCase):
    def test1(self):
        self.assertEqual(1,1)

class ConstructingFoldedFromArray(unittest.TestCase):
    def test1(self):
	"""Folded.from_flat, no overlap"""
        a = np.array([1, 4, 6])
        centers = np.array([2, 7])
        f = Folded.from_flat(a, centers=centers, dstart=-2, dstop=2,
            subtract_off_center=False)
        
	self.assertEqual(len(f), len(centers))
        self.assertTrue(np.all(f[0] == a[:1]))
        self.assertTrue(np.all(f[1] == a[2:]))

    def test2(self):
	"""Folded.from_flat, no overlap, no subtract off"""
        a = np.array([1, 4, 6])
        centers = np.array([2, 7])
        f = Folded.from_flat(a, centers=centers, dstart=-2, dstop=2)
	
	self.assertEqual(len(f), len(centers))
        self.assertTrue(np.all(f[0] == a[:1] - centers[0]))
        self.assertTrue(np.all(f[1] == (a[2:] - centers[1])))

    def testInclusivityLeftEdge(self):
	"""The left edge of window should be inclusive"""
        a = np.array([1, 5, 6])
        centers = np.array([2, 7])
        f = Folded.from_flat(a, centers=centers, dstart=-2, dstop=2)
	
	self.assertEqual(len(f), len(centers))
        self.assertTrue(np.all(f[0] == a[:1] - centers[0]))
        self.assertTrue(np.all(f[1] == (a[1:] - centers[1])))

    def testInclusivityRightEdge(self):
	"""The right edge of window should not be inclusive"""
        a = np.array([1, 5, 9])
        centers = np.array([2, 7])
        f = Folded.from_flat(a, centers=centers, dstart=-2, dstop=2)
	
	self.assertEqual(len(f), len(centers))
        self.assertTrue(np.all(f[0] == a[:1] - centers[0]))
        self.assertTrue(np.all(f[1] == (a[1:-1] - centers[1])))

if __name__ == '__main__':
    unittest.main()
