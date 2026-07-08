## 1. 퍼셉트론에서 가중치와 편향의 물리적 의미를 설명하라
- 가중치: 입력값이 출력값에 얼마나 큰 영향을 줄지 정하는 매개변수, 가중치가 클 수록 출력값에 미치는 영향이 크다.
- 편향 : 뉴런이 얼마나 쉽게 활성화될지 (1을 출력하는 것) 을 조정하는 매개변수이다. 

## 2. 퍼셉트론은 이진분류 알고리즘으로 볼수있다.  AND, OR문제를 이용하여 이진분류문제가 무엇인지 설명하라
``` python
import numpy as np
import matplotlib.pyplot as plt

def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
    
print(AND(0,0)) # 0
print(AND(1,0)) # 0
print(AND(0,1)) # 0 
print(AND(1,1)) # 1
```
