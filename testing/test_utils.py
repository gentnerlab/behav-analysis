from __future__ import absolute_import
from behav import utils


def test_stars():
    starsTest = []
    for val in [0.0009, 0.009, 0.04, 0.09, 1]:
        starsTest.append(utils.stars(val))
    assert starsTest == ['***', '**', '*', '.', 'n.s.']
