a = [2, 3, 4]
print(a, type(a))  # [2, 3, 4] <class 'list'>
b = [0.1, 0.2, 0.5]
print(b, type(b))  # [0.1, 0.2, 0.5] <class 'list'>
c = zip(a, b)
print(c, type(c))  # <zip object at 0x7fc9ec0f5200> <class 'zip'>
d = list(zip(a, b))
print(d, type(d))  # [(2, 0.1), (3, 0.2), (4, 0.5)] <class 'list'>

for i in range(0, 3):
  print(d[i])
# (2, 0.1)
# (3, 0.2)
# (4, 0.5)

for a_item, b_item in c:
  print(a_item, b_item)
# 2 0.1
# 3 0.2
# 4 0.5
