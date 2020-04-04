import unittest
from utils.csvUtils import *

class CSVUtilsMethods(unittest.TestCase):

    def testColumnLettersToNumbers(self):
        self.assertEqual(columnLettersToNumber("A", False), 1)
        self.assertEqual(columnLettersToNumber("B", False), 2)
        self.assertEqual(columnLettersToNumber("AA", False), 27)
        self.assertEqual(columnLettersToNumber("AB", False), 28)
        self.assertEqual(columnLettersToNumber("BA", False), 53)

        self.assertEqual(columnLettersToNumber("A", True), 0)
        self.assertEqual(columnLettersToNumber("B", True), 1)
        self.assertEqual(columnLettersToNumber("AA", True), 26)
        self.assertEqual(columnLettersToNumber("AB", True), 27)
        self.assertEqual(columnLettersToNumber("BA", True), 52)

    def testColumnLettersToNumbers(self):
        self.assertEqual(columnNumberToLetters(1, False), "A")
        self.assertEqual(columnNumberToLetters(2, False), "B")
        self.assertEqual(columnNumberToLetters(27, False), "AA")
        self.assertEqual(columnNumberToLetters(28, False), "AB")
        self.assertEqual(columnNumberToLetters(53, False), "BA")

        self.assertEqual(columnNumberToLetters(0, True), "A")
        self.assertEqual(columnNumberToLetters(1, True), "B")
        self.assertEqual(columnNumberToLetters(26, True), "AA")
        self.assertEqual(columnNumberToLetters(27, True), "AB")
        self.assertEqual(columnNumberToLetters(52, True), "BA")

if __name__ == '__main__':
    unittest.main()