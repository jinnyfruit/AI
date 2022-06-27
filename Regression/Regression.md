# Regression 

> Supervised learning 은 크게 두가지로 나눌 수 있다. Regression과 Classification이다.  
> Regression은 임의의 어떤 숫자를 예측하는 것이다. data 사이의 상관관계를 통해서 수식을 세우고,  
> 이를 통해 임의의 수치값을 도출해내는 것이 Regression의 목적이다. 

## 1. Knn Regression
* K-nearest neighbor classification
  * 1) 예측하려는 샘플에 가장 가까운 K개의 데이터를 선택한다.
  * 2) 선택한 데이터가 어디에 속하는지 확인한 후 다수결을 통해 새로운 데이터가 어디에 속하는지 결정한다.
    
Knn Regression 또한 크게 다르지 않다. 분류가 아니라 예측한다는 점이 다르다.  
예측을 통해 임의의 수치를 도출해야 하기 때문에 데이터를 다루는 방식이 다르다. 그렇다면 어떻게 수치값을 구할 수 있을까?  
이웃하는 데이터의 수치를 이용해서 평균값을 구하면 그 값이 새로운 데이터의 예측값이 된다.  

*[한계]*  
새로운 데이터가 training dataset 의 범위를 벗어나게 되면 엉뚱한 값을 예측할 수 있다.  
ex) 수가 기하급수적으로 커지는데 여전히 낼 수 있는 최대치의 값을 똑같이 예측하는 경우 

## 2. Linear regression 

