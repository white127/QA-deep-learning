import random

_list = [i for i in range(0, 10)]
_l1 = random.sample(_list, 2)
_l2 = random.sample(_list, 2)
print(_l1)
print(_l2)
for i in range(2, 2):
  print(i)

