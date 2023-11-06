# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:40:56 2023

@author: 20793801
"""
import unittest
from view.panels.sample.sampleMonteCarloPanel import SampleOutputMonteCarloPanel

class TestSampleMonteCarloPanel(unittest.TestCase):
    def test_showNoDataPanel(self):
        # Create a SampleOutputMonteCarloPanel instance
        panel = SampleOutputMonteCarloPanel(controller=None, sample=None)

        # Test the _showNoDataPanel method
        panel._showNoDataPanel()

        # You can add assertions to check if the state of the panel is as expected

    def test_showDataPanel(self):
        # Create a SampleOutputMonteCarloPanel instance
        panel = SampleOutputMonteCarloPanel(controller=None, sample=None)

        # Test the _showDataPanel method
        panel._showDataPanel()

        # You can add assertions to check if the state of the panel is as expected

    # Add more test methods for other functionalities in SampleOutputMonteCarloPanel

if __name__ == '__main__':
    unittest.main()