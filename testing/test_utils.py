import behav


def test_stars():
    starsTest = []
    for val in [0.0009, 0.009, 0.04, 0.09, 1]:
        starsTest.append(behav.utils.stars(val))
    assert starsTest == ['***', '**', '*', '.', 'n.s.']
"""
assert binP(3, 0.75, 5, 10) == 

q = 0.75/0.
k = 0.0 #increments if 
v = 1.0 #increment size
s = 0.0 #increments if k is between x1 and x2
tot = 0.0 #step counter

k = 0 tot = 1 """
