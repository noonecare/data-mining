from unittest import TestCase
import numpy as np


# coding: utf-8
from neural_network.bp_algorithm import BP


class TestBP(TestCase):

    def test_update(self):
        sample = np.arange(100).reshape(25, 4)
        neural_network_model = BP(sample, 3, 4, 0.1)
        neural_network_model.update(np.array([1, 2, 3]), np.array([1]))

    def test_train(self):
        self.fail()

    def test_validate(self):
        self.fail()

    def test_predict(self):
        sample = np.arange(100).reshape(25, 4)
        neural_network_model = BP(sample, 3, 4, 0.1)
        print(neural_network_model.predict(np.array([1, 2, 3])))
