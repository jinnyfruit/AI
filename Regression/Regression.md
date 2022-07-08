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
선형 회귀는 가장 많이 사용되는 회귀 알고리즘이다.  
선형은 말그대로 특성이 하나인 경우 어떤 직선을 학습하는 알고리즘을 의미한다.  
model 은 주로 sklearn.linear_model 패키지의 LinearRegression class를 사용한다. 

직선방정식은 주로 y = ax + b 형태를 띄고 있다. 이때 a과 b값은 model parameter에 해당한다.  
* weight (계수) = a  
* bias (가중치) = b  

앞서 나왔던 knn regression 과 다른 점이기도 하다.  
우리가 ML 알고리즘을 훈련함으로써 얻고자 하는 것은 최적의 model parameter이다.  
이 값은 model 훈련시 불러온 객체의 coef_ 와 intercept_ 에 저장되어있다. 

그러나 일차방정식만으로는 높은 정확도를 구현하기 힘든 경우들도 있다.  
이떄 사용되는 것이 다항식을 이용한 선형회귀이다. 

**[ polynomial regression (다항회귀) ]**  
이차 방정식의 형태를 띄고 있다.  
경우에 따라서 직선인 단순 선형 회귀 모델보다 더 나은 정확도를 구현할 수 있다.  
곡선형태를 띄고 있다. 직선모델로 데이터를 다 담을 수 없는 경우에는 좀 더 유연한 모델이 될 수 있다.  

>이때 드는 의문은 "선형" 모델인데 과연 곡선 형태인 다항모델을 선형 범주에 넣을 수 있는가 하는 것이다.   
> non-linear 한 것이 아닌가 생각할 수 있지만 2차식은 다른 변수로 치환가능하기 때문에 선형 관계를 띌 수 있다.  

## 3. Multiple Regression  
선형 회귀에서 특성이 하나일 때는 일차 함수의 형태를 띈 선형회귀를 사용했다.  
그러나 우리가 분석해야 할 데이터의 feature, 즉 특징이 두개 이상인 경우도 있다.   
이 경우에는 여러 feature들도 반영하여 regression을 진행하면 좀 더 괜찮은 모델이 될 수 있다.  
이런 여러 개의 feature를 반영한 회귀를 Multiple Regression이라고 부른다.  

Multiple Regression은 feature의 개수에 따라서  
y = ax1 + bx2 + cx3 ... + z 형태를 가지고 있다.  

**[ Feature engineering ]**  
이미 가진 feature들 사이에서 새로운 feature를 생성하는 작업을 의미.

> **[ Underfitting and Overfitting ]**  
> * Underfitting = 모델이 너무 단순한 나머지 정답률이 현저히 떨어지는 현상을 말한다.  
> * Overfitting = training dataset 에 너무 최적화된 나머지 training data가 아닌 다른 dataset에서는 예측률이 떨어지는 경향을 말한다. 

### Regularization
Machine Learning model이 train data에만 최적화하지 않도록 하는 역할을 한다.  
정규화를 하면 계수값의 크기가 너무 달라지지 않을 수 있도록 변환시킬 수 있다.  
sklean에서는 여러가지 scaler를 제공하고 있다.
###
**1. Standard Scaler**
  * 정규 분포로 변환
  * 모든 feature의 mean 값을 0, variation을 1로 변환  
###
**2. Normalizer**
  * 각 변수의 값을 원점으로부터 1만큼 떨어져 있는 범위 내로 변환  
###
**3. MinMaxScaler** 
  * 데이터의 value들을 0~1 사이의 값으로 변환
  * outlier에 민감
###
**4. Robust Scaler** 
  * 모든 feature가 같은 값을 가짐. 
  * mean과 variation이 아닌 median 과 IQR을 사용함 
###
Linear Regression + Regularization 을 한 모델은 두가지가 있다.  

**Ridge Regression**  
* 가중치를 0에 가깝게하고 feature들의 영향력이 감소함 
* feature의 중요도가 전반적으로 비슷함 
* non-sparse model

**Lasso Regression** 
* 가중치를 0으로 하고 featrue를 무력화 시킴
* 일부 featrue가 중요하다. 
* sparse model 
* feature selection 

모델을 만들 떄 alpha 매개변수를 사용하여 regulation 정도를 결정한다.  
alpha 값이 커지면, regulation 정도도 커지므로 이를 줄이기 위해서 계수 값을 더 줄인다.  



