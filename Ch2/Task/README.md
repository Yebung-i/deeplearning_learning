## 1. 퍼셉트론에서 가중치와 편향의 물리적 의미를 설명하라
- 가중치: 입력값이 출력값에 얼마나 큰 영향을 줄지 정하는 매개변수, 가중치가 클 수록 출력값에 미치는 영향이 크다.
- 편향 : 뉴런이 얼마나 쉽게 활성화될지 (1을 출력하는 것) 을 조정하는 매개변수이다
  

## 2. 퍼셉트론은 이진분류 알고리즘으로 볼수있다.  AND, OR문제를 이용하여 이진분류문제가 무엇인지 설명하라
- AND
``` python
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

- OR
``` python
import numpy as np

def OR(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(w*x) + b

    if tmp <= 0:
        return 0
    else:
        return 1
    
print(OR(0,0)) # 0
print(OR(1,0)) # 1
print(OR(0,1)) # 1
print(OR(1,1)) # 1
```

- 이진분류란, 입력 데이터를 두 개의 클래스 (0 or 1)중 하나로 분류하는 문제로서, 입력값에 가중치와 편향을 적용한 후 임계값을 기준으로 0 또는 1을 출력한다.

## 3. 퍼셉트론은 입력이 2차원일때 2차원 평면에서 직선으로 분리가 가능한 입력 데이터들만 분류가 가능하다. 이때, 분류의 기준이 되는 경계선의 방정식을 유도하라
- z = w1x1 + w2x2 + b (여기서 z는 가중합을 의미)
- 이때 활성화 함수로 스텝 함수를 사용하여 y를 구한다. 
<img width="174" height="76" alt="image" src="https://github.com/user-attachments/assets/10f3ba8b-2ae5-4aa3-9591-1edf42a127e9" />

- 여기서 분류 결과가 나뉘는 경계는 z = 0 인 곳이다 (출력이 0에서 1로 바뀌는 지점) >> 결정 경계는 다음과 같이 된다.
  <img width="223" height="35" alt="image" src="https://github.com/user-attachments/assets/2d752f0f-d3d1-4c14-8f91-e035c4b4bd3e" />

- 또한 이를 x2에 대해 정리한다면
<img width="174" height="61" alt="image" src="https://github.com/user-attachments/assets/cd8a879c-1ac1-469d-89e3-1a8514e621a5" />
이 되기에, 2차원 입력을 사용하는 퍼셉트론의 결정 경계는 직선이다.



## 4. 퍼셉트론은 XOR문제를 풀수없다 이유를 설명하고 해결방안을 설명하라
|x1|x2|결과|
|---|---|---|
|0|0|0|
|1|0|1|
|0|1|1|
|1|1|0|

이를 2차원 좌표평면상에 표현하면

```text
            
                 ↑

        1     ● (0,1)         ○ (1,1)
              출력=1          출력=0

        0     ○ (0,0)         ● (1,0)
              출력=0          출력=1

              0------------------------→
                        0         1
```

- 과 같이 그려지는데, 보여지듯이 XOR은 두 클래스를 하나의 직선으로 나눌 수 없다. 때문에 직선 하나만을 결정 경계로 사용하 단층 퍼셉트론으로는 이를 완벽하게 분류해낼 수 없다.
- 따라서 중간에 은닉 층을 추가한 다층 퍼셉트론으로 해결 가능하며, 은닉층에 NAND 연산과 OR 연산을 담당하는 두 뉴런을 배치하고, 이 두 뉴런의 출력을 출력층에서 다시 AND로 결합하면 XOR와 동일한 결과를 얻을 수 있다.

<img width="732" height="584" alt="image" src="https://github.com/user-attachments/assets/660f5531-ded1-4db4-9dbc-3a5fd5e65e3c" />


## 5. 퍼셉트론에서 활성화함수로 스텝함수를 사용하는 이유를 설명하라
- 퍼셉트론은 이진 분류를 수행하는 모델이므로, 출력값을 0 또는 1로 구분해야 한다. 따라서 입력값이 임계값보다 크면 1, 아니면 0을 출력하기 때문에 퍼셉트론의 분류 목적에 적합하다.

## 6. . 퍼셉트론의 특성 파라미터인 가중치와 편향은 교재에서는 임의로 정했으나 실제로는 학습알고리즘을 이용하여 자동으로 결정한다. 프랭크 로젠블랫이 개발당시 사용한 퍼셉트론 학습알고리즘을 조사하고 AND 게이트 문제에 대하여 직접 구현하여 학습한후 AND게이트 문제를 잘 푸는지 검증해보라 matplotlib을 이용하여 오차그래프를 그려서 오차가 0으로 수렴하는지 확인하라 
1. 가중치와 편향을 임의의 값으로 초기화한다.
2. 입력 데이터를 이용하여 출력값을 계산한다.
3. 예측값과 정답을 비교하여 오차를 계산한다.
4. 오차를 이용하여 가중치와 편향을 수정한다.
5. 모든 학습 데이터에 대하여 반복 수행하며 오차가 0이 될 때까지 학습한다.
- 즉, 오차가 존재할 경우에만 가중치를 수정한다.

``` python
import numpy as np
import matplotlib.pyplot as plt

# 입력 데이터
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

# 정답
T = np.array([0,0,0,1])

# 초기 가중치와 편향
w = np.random.randn(2)
b = np.random.randn()

# 학습률 (가중치를 얼마나 크게 수정할 것인지)
lr = 0.1

# 전체 데이터를 몇 번 반복 학습할 것인지
epochs = 20

# epoch마다 발생한 오차 개수 저장
errors = []


# 퍼셉트론 학습
for epoch in range(epochs):
    error_count = 0

    # 학습 데이터를 하나씩 사용
    for x, t in zip(X, T):
        # 입력과 가중치를 곱하여 편향을 더함
        net = np.dot(x, w) + b

        # 활성화 함수, net이 0보다 크면 1, 아니면 0 반환
        if net > 0:
            y = 1
        else:
            y = 0

        # 오차 계산
        error = t - y

        # 가중치 수정, 맞게 예측했다면 수정 안 함
        w += lr * error * x

        # 편향 수정
        b += lr * error

        error_count += abs(error)

# 현재 epoh의 오차 저장
    errors.append(error_count)

# 학습 결과 출력
print("학습된 가중치 : ", w)
print("학습된 편향 : ", b)

print("==============================================")

for x in X:
    # 학습된 가중치와 편향을 이용하여 예측함
    net = np.dot(x,w)+b

    if net > 0:
        y = 1
    else:
        y = 0

    print(f"{x}->{y}")

# 오차 그래프 출력

plt.plot(errors,marker="o")

plt.title("Perception Learning Error")
plt.xlabel("Epoch")
plt.ylabel("Number of Errors")

# 격자 표시
plt.grid(True)
plt.show()

```

  <img width="635" height="558" alt="image" src="https://github.com/user-attachments/assets/21f2e4c3-e4ed-491e-b416-b0f64c8e6ba8" />


## 7. 수학적 관점에서 퍼셉트론의 입출력사이의 관계식 y=f(x)은 선형함수인가 비선형함인가? 또, 입력신호(독립변수)의 갯수는 몇개이고 가질수 있는 값(정의역)은 무엇인가, 출력신호(종속변수)는 몇개이고 가질수 있는 값(공역)은 무엇인가
-  $y = f(x)$는 비선형 함수 이다.
-  내부의 가중합  $z = w_1x_1 + w_2x_2 + b$ 자체는 선형이지만, 여기에 계단 함수 라고 하는 활성화 함수가 붙게 된다. ( 0을 넘으면 1, 아니면 0 을 씌워서 꺾이게 되니까 )
- 입력신호의 갯수는 특별히 정해진 것은 없다. 하지만 출력변수는 하나로 귀결된다.

## 8. 다음표는 2차원 평면상의 8개의 점의 x,y 좌표를 의미하고 label은 각점이 속한 클래스(0,1)를 의미한다.퍼셉트론의 입력은 2차원 좌표이고 출력은 0 또는 1의 값을 가져야한다
(1) 6번에서 구현한 학습알고리즘을 이용하여 아래데이터를 분류하는 퍼셉트론의 가중치와 편향값을 구하라
<img width="174" height="229" alt="image" src="https://github.com/user-attachments/assets/70323a32-eddc-4c2d-8ff7-008e511e715c" />


(2) matplotlib을 이용하여 오차그래프를 그려보고 오차가 0으로 수렴하는지 확인하라  
<img width="635" height="557" alt="image" src="https://github.com/user-attachments/assets/d126fc97-5331-47fe-8b99-322df9cb4fd9" />


(3) (1)번에서 구한 가중치와 편향값을 이용하여  0<=x<=500, 0<=y<=500 범위의 모든 정수 좌표에 대하여 퍼셉트론의 출력을 계산하고 0이면 빨간색으로 그리고 1이면 초록색으로 그려주시오 opencv를 이용하여 500x500 크기의 영상을 만들고 아래 결과그림처럼 그려주면됨, 결정경계는 곡선이 아니고 직선으로 나와야함

<img width="641" height="557" alt="image" src="https://github.com/user-attachments/assets/97f9001b-e1c0-4a1d-9aca-83bebe16c89a" />





``` python
import numpy as np
import matplotlib.pyplot as plt

# 입력 데이터
X = np.array([
    [150,200],
    [200,250],
    [100,250],
    [150,300],
    [350,100],
    [400,200],
    [400,300],
    [350,400]
])

# 정답
T = np.array([0,0,0,0,1,1,1,1])

# 초기 가중치와 편향
w1 = 0.0
w2 = 0.0
b = 0.0

# 학습률
lr = 0.001      # 입력값이 크므로 0.1보다 작게 설정

# 반복 횟수
epochs = 100

# Epoch마다 오차 저장
errors = []

# 퍼셉트론 학습
for epoch in range(epochs):

    error_count = 0

    for x, t in zip(X, T):

        x1 = x[0]
        x2 = x[1]

        # 순입력 계산
        net = w1 * x1 + w2 * x2 + b

        # 활성화 함수
        if net > 0:
            y = 1
        else:
            y = 0

        # 오차
        error = t - y

        # 가중치와 편향 갱신
        w1 += lr * error * x1
        w2 += lr * error * x2
        b += lr * error

        error_count += abs(error)

    errors.append(error_count)

    if error_count == 0:
        break

# 학습 결과
print("학습된 가중치")
print("w1 =", w1)
print("w2 =", w2)
print("b =", b)

print("========================")

# 검증
for x in X:

    x1 = x[0]
    x2 = x[1]

    net = w1 * x1 + w2 * x2 + b

    if net > 0:
        y = 1
    else:
        y = 0

    print(f"{x} -> {y}")

# 오차 그래프
plt.plot(range(1, len(errors)+1), errors, marker='o')
plt.title("Perceptron Learning Error")
plt.xlabel("Epoch")
plt.ylabel("Number of Errors")
plt.grid(True)
plt.show()
```

<img width="644" height="700" alt="image" src="https://github.com/user-attachments/assets/49e066cf-0526-4811-97b9-c81b72ed9024" />

