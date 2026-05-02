"""
tests/test_sentiment.py

Basic tests for sentiment analysis.
"""

import unittest
from core.sentiment import SentimentEngine


class TestSentimentEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SentimentEngine()

    def test_analyse_positive(self):
        result = self.engine.analyse("I am so happy!")
        self.assertEqual(result["emotion"], "joy")
        self.assertGreater(result["compound"], 0)

    def test_analyse_negative(self):
        result = self.engine.analyse("I feel terrible")
        self.assertIn(result["emotion"], ["sad", "frustrated"])
        self.assertLess(result["compound"], 0)

    def test_analyse_neutral(self):
        result = self.engine.analyse("Hello world")
        self.assertEqual(result["emotion"], "neutral")


if __name__ == "__main__":
    unittest.main()