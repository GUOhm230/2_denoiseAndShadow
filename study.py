a = -3
print(~a)  # 输出-（a+1)
b = True  # False输出-1.True输出 -2
print(~b)
c = [1,2,3,4,5]
d = [1,0,0,0,1]
d[d.index(0)] = 1
import numpy as np
# print(d!=0)
print(d)
e = [[98, 10], [93, 511], [450, 481], [451, 1]]
print(list(np.array(e)*2))