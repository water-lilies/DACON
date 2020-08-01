# Catboost

> GBM(Gradient Boosting Algorithm) 을 구현해놓은 패키지 중 하나



영문 설명

https://www.kaggle.com/mitribunskiy/tutorial-catboost-overview

https://medium.com/@hanishsidhu/whats-so-special-about-catboost-335d64d754ae

https://towardsdatascience.com/https-medium-com-talperetz24-mastering-the-new-generation-of-gradient-boosting-db04062a7ea2

 

 

 

https://gentlej90.tistory.com/100

https://data-newbie.tistory.com/131

https://data-newbie.tistory.com/159

https://soobarkbar.tistory.com/34



 

**Catboost(Categorical Boosting)**



마지막으로 정리할 모델은 catboost입니다. 상대적으로 최근에 나온 알고리즘이며 범주형(categorical) 변수를 처리하는 데 유용한 알고리즘이라고 알려져 있습니다.



**1. Introduction**

\- Gradient Boosting은 매우 강력하지만 여전히 통계적 문제에 직면하고 있다.

\- 특히 **gradient boosting은 매 boosting round마다** **train****의 target data에 의존하여 잔차를 구하고 학습****하기 때문에 f(x_test) | x_test의 분포에서 f(x_train) | x_train 분포로 변화**를 만든다. 

\- 결국 **예측 결과 변화, prediction shift를 만들고 하나의 target leakage 문제이다.**

\- 또한, **기존 알고리즘들은 범주형 변수의 처리 문제가 있다**.

\- 이런 두가지 문제점을 **oredering priciple과 새로운 범주형 변수 처리 방법으로 해결**한다.



**기존 알고리즘들은 target data에 의존해 학습하여 taget leakage로 인한 overfitting과 범주형 변수 처리 문제가 있습니다.**



**2. Categorical Features**

\- 범주형 변수는 이산적인 값을 가지며 서로 비교할 수 없는 변수이다.

\- 범주형 변수 처리의 대표적인 기법은 One-hot encoding이다.

\- **One-hot encoding****은 각 범주의 특성들에 대해 이진변수로 만들기 때문에 feature의 수가 급격하게 증가하는 문제점**이 있을 수 있다.

\- 이러한 문제를 해결하고자 몇 개의 클러스터로 묶고 one-hot encoding을 하거나 **범주의 Target Statistcs(TS)를 추정하여 사용하는 방법**이 있다.

\- lightgbm의 경우 매 boosting round에서 범주형 변수를 gradient statistics를 활용하여 변환하지만 이는 계산이 오래 걸리며 많은 메모리를 사용해야 한다.

\- 일반적으로 **TS는 각 범주의 특성별로 가질 수 있는 y에 대한 기대값을 활용**한다.

\- **Greedy TS****의 경우 각 범주 특성의 y 평균값으로 추정하는데 이는 target leakage가 동일하게 발생하며 overfitting** 될 위험이 있다.

\- **Hold-out TS****는 이를 개선하여 train data를 a와 b로 나누고 a는 TS 추정에 b는 학습에 사용하여 overfitting을 줄일 수 있다. 하지만 데이터의 양이 줄어드는 문제점**이 있다.

\- Catboost에서는 **oredering priciple을 적용한 Ordered TS를 제안**한다.

\- 이는 **TS** **추정치를 observed history에서만 구하는 방식**이다.

\- **현 시점을 기준으로 과거 데이터로만 TS를 추정하기 위해서** **무작위 순열****,** **즉 인공적인 시간을 도입**한다.

\- 하지만 **하나의 무작위 순열로만 TS를 추정한다면 과거의 TS추정치는 그 이후의 추정치보다 분산이 높을 것이다.**

\- 이를 보정하기 위해 **각 단계마다 다른 무작위 순열**을 활용한다.



기존 범주형 변수를 수치형 변수로 변환하는 방법은 overfitting의 문제나 계산에서 문제가 있었습니다. 하지만 **catboost에서는 계산을 줄이고 target leakage를 최대한 줄일 수 있는 Ordered TS를 제안**합니다.



**3. Prediction Shift and Ordered Boosting**

\- **Target leakage, Prediction Shift****로 인한 overfitting에 대한 해결 방법으로 oredering principle를 제안**한다.

\- 매 boosting에서 **같은 데이터를 사용하여 잔차를 구하고 학습하기 때문에 이러한 문제가 발생**한다.

\- 이를 위해 **전체 데이터를 나누고 그에 맞는 다수의 모델을 유지**한다.

\- **전체 데이터를 나누기 위해 oredered TS에서 사용한 것과 마찬가지로 무작위 순열을 사용**한다.

\- **j****번째 샘플에 대한 잔차를 구하기 위해선 (j-1)번째까지 사용한 데이터로 학습한 모델을 사용**한다.

\- **동일한 데이터로 계속 학습한 모델로 잔차를 갱신하는 것이 아니라 다른 데이터로 학습한 모델로 잔차를 갱신하는 방법**이다.

\- ordered boosting 또한 마찬가지로 **하나의 무작위 순열이 아닌 몇 개의 무작위 순열을 사용하며 이는 oredered TS 추정에 사용한 순열과 동일하게 하는 것이 좋다.**



기존 Gradient Boosting이 가진 문제를 해결하고자 ordering priciple을 적용하여 prediction shift를 피하고자 고안된 방법입니다. 하지만 각기 다른 모델을 유지해야 하기 때문에 효율성 측면에서 떨어질 수 있다고 합니다.

이외에도 catboost에 사용되는 개념들이 있는데 **oblivious decision tree와 feature combination**을 보겠습니다.

**Oblivious decision tree,** **즉 망각 트리 방식은 트리를 분할할 때 동일한 분할 기준이 전체 트리 레벨에서 적용이 되는 것입니다. 이는 균형잡힌 트리들을 만들 수 있고 overfitting을 막아**줄 수 있습니다.

또한, **catboost****에서는 범주형 변수의 조합으로 새로운 변수 조합**을 만들어 낼 수 있습니다. Greedy 방식으로 이러한 조합들을 만들며 **트리를 분할할 때 이전에 사용된 데이터에서 조합을 찾아내고 TS로 변환하는 방식**입니다.



Catboost는 범주형 변수를 처리하는데 다른 알고리즘보다 효과적인 것으로 알려져 있습니다. 하지만 데이터 대부분이 수치형인 경우는 큰 효과를 못 볼 수 있으며 lightgbm에 비해 학습 속도가 느립니다.





**[****출처]** [[바람돌이/머신러닝\] 앙상블(Ensemble Learning)(4) - 부스팅(Boosting), XGBoost, CatBoost, LightGBM 이론](https://blog.naver.com/winddori2002/221931868686)|**작성자** [바람돌이](https://blog.naver.com/winddori2002)

 

**2. Catboost** **의 특징**

**2.1. Level-wise Tree**

XGBoost 와 더불어 Catboost 는 Level-wise 로 트리를 만들어나간다. (반면 Light GBM 은 Leaf-wise 다)
 Level-wise 와 Leaf-wise 의 차이는, 그냥 직관적으로 말하면 Level-wise 는 BFS 같이 트리를 만들어나가는 형태고, Leaf-wise 는 DFS 같이 트리를 만들어나가는 형태다. 물론 max_depth = -1 이면 둘은 같은 형태지만, 대부분의 부스팅 모델에서의 트리는 max_depth != -1 이기 때문에 이 둘을 구분하는 것이다.
 자세한 내용은 아래 링크를 참고하자.

https://datascience.stackexchange.com/questions/26699/decision-trees-leaf-wise-best-first-and-level-wise-tree-traverse



**2.2. Orderd Boosting**

Catboost 는 기존의 부스팅 과정과 전체적인 양상은 비슷하되, 조금 다르다.
 기존의 부스팅 모델이 일괄적으로 모든 훈련 데이터를 대상으로 잔차계산을 했다면, Catboost 는 일부만 가지고 잔차계산을 한 뒤, 이걸로 모델을 만들고, 그 뒤에 데이터의 잔차는 이 모델로 예측한 값을 사용한다.
 말로 들으면 좀 어려울 수 있는데, 아래 예를 보자.

![img](file:///C:/Users/Minji/AppData/Local/Temp/msohtmlclip1/01/clip_image001.png)

기존의 부스팅 기법은 모든 데이터(x1 ~ x10) 까지의 잔차를 일괄 계산한다.
 반면, Catboost 의 과정은 다음과 같다.

\1. 먼저 x1 의 잔차만 계산하고, 이를 기반으로 모델을 만든다. 그리고 x2 의 잔차를 이 모델로 예측한다.

\2. x1, x2 의 잔차를 가지고 모델을 만든다. 이를 기반으로 x3, x4 의 잔차를 모델로 예측한다.

\3. x1, x2, x3, x4 를 가지고 모델을 만든다. 이를 기반으로 x5, x6, z7, x8 의 잔차를 모델로 예측한다.

\4. ... 반복

이렇게 순서에 따라 모델을 만들고 예측하는 방식을 Ordered Boosting 이라 부른다.

**2.3. Random Permutation**

위에서 Ordered Boosting 을 할 때, 데이터 순서를 섞어주지 않으면 매번 같은 순서대로 잔차를 예측하는 모델을 만들 가능성이 있다. 이 순서는 사실 우리가 임의로 정한 것임으로, 순서 역시 매번 섞어줘야 한다. Catboost 는 이러한 것 역시 감안해서 데이터를 셔플링하여 뽑아낸다. 뽑아낼 때도 역시 모든 데이터를 뽑는게 아니라, 그 중 일부만 가져오게 할 수 있다. 이 모든 기법이 다 오버피팅 방지를 위해, 트리를 다각적으로 만들려는 시도이다.

**2.4. Ordered Target Encoding**

Target Encoding, Mean Encoding, Response Encoding 이라고 불리우는 기법 (3개 다 같은 말이다.)을 사용한다.
 범주형 변수를 수로 인코딩 시키는 방법 중, 비교적 가장 최근에 나온 기법인데, 간단한 설명을 하면 다음과 같다.

![img](file:///C:/Users/Minji/AppData/Local/Temp/msohtmlclip1/01/clip_image002.png)

위 데이터에서 time, feature1 으로 class_label 을 예측해야한다고 해보자.
 feature1 의 cloudy 는 다음과 같이 인코딩 할 수 있다.

cloudy = (15 +14 +20 + 25)/4 = 18.5

즉, cloudy 를 cloudy 를 가진 데이터들의 class_label 의 값의 평균으로 인코딩 하는 것이다.
 이 때문에 Mean encoding 이라 불리기도 한다.

그런데 위는 우리가 예측해야하는 값이 훈련 셋 피처에 들어가버리는 문제, 즉 Data Leakage 문제를 일으킨다. 이는 오버피팅을 일으키는 주 원인이자, Mean encoding 방법 자체의 문제이기도 하다.

그래서, Catboost 는 이에대한 해결책으로, 현재 데이터의 인코딩하기 위해 이전 데이터들의 인코딩된 값을 사용한다.
 예를 들면 다음과 같다.

\- Friday 에는, cloudy = (15+14)/2 = 15.5 로 인코딩 된다.

\- Saturday 에는, cloudy = (15+14+20)/3 = 16.3 로 인코딩 된다.

즉, 현재 데이터의 타겟 값을 사용하지 않고, 이전 데이터들의 타겟 값만을 사용하니, Data Leakage 가 일어나지 않는 것이다.
 범주형 변수를 수로 인코딩하는 할 때, 오버피팅도 막고 수치값의 다양성도 만들어 주는.. 참 영리한 기법이 아닐 수 없다.
 이러한 시도는 Smoothing, Expanding 등이 있어왔는데, 이 시도 역시 이런 종류 중 하나라고 볼 수 있겠다.
 (이에 대한 자세한 내용은 [Categorical Value Encoding 과 Mean Encoding](https://dailyheumsi.tistory.com/120) 참고)

**2.5. Categorical Feauture Combinations**

먼저 다음의 예부터 보자.

![img](file:///C:/Users/Minji/AppData/Local/Temp/msohtmlclip1/01/clip_image003.png)

country만 봐도 hair_color feature가 결정되기 때문에, class_label 을 예측하는데 있어, 두 feature 다 필요없이 이 중 하나의 feature 만 있으면 된다. Catboost 는 이렇게, information gain 이 동일한 두 feature 를 하나의 feature 로 묶어버린다. 결과적으로, 데이터 전처리에 있어 feature selection 부담이 조금 줄어든다고 할 수 있다.

**2.6. One-hot Encoding**

사실 범주형 변수를 항상 Target Encoding 하는 것은 아니다. Catboost 는 낮은 Cardinality 를 가지는 범주형 변수에 한해서, 기본적으로 One-hot encoding 을 시행한다. Cardinality 기준은 one_hot_max_size 파라미터로 설정할 수 있다.
 예를 들어, one_hot_max_size = 3 으로 준 경우, Cardinality 가 3이하인 범주형 변수들은 Target Encoding 이 아니라 One-hot 으로 인코딩 된다.
 아무래도 Low Cardinality 를 가지는 범주형 변수의 경우 Target Encoding 보다 One-hot 이 더 효율적이라 그런 듯 하다.

**2.7. Optimized Parameter tuning**

Catboost 는 기본 파라미터가 기본적으로 최적화가 잘 되어있어서, 파라미터 튜닝에 크게 신경쓰지 않아도 된다고 한다. (반면 xgboost 나 light gbm 은 파라미터 튜닝에 매우 민감하다.) 사실 대부분 부스팅 모델들이 파라미터 튜닝하는 이유는, 트리의 다형성과 오버피팅 문제를 해결하기 위함인데, Catboost 는 이를 내부적인 알고리즘으로 해결하고 있으니, 굳이.. 파라미터 튜닝할 필요가 없는 것이다.
 굳이 한다면 learning_rate, random_strength, L2_regulariser 과 같은 파라미터 튜닝인데, 결과는 큰 차이가 없다고 한다.

**3. Catboost** **의 한계**

다음과 같은 한계를 지닌다.

·    Sparse 한 Matrix 는 처리하지 못한다.

·    데이터 대부분이 수치형 변수인 경우, Light GBM 보다 학습 속도가 느리다. (즉 대부분이 범주형 변수인 경우 쓰라는 말)

**4. Catboost** **를 언제 사용해야할까?**

음.. 이거는 정말 케이스 바이 케이스가 참 많은데, 대략적으로만 말하자면 파라미터 조정을 통해 여기저기 적용 가능하다.
 사실 파라미터 수정해서 적용할 수 있는 상황이 많아서, 그냥 참고 링크 첫 번째 사이트를 들어가 보는게 나을 듯하다.
 (귀찮아서 더 안 적는거 아님...)


 
 출처: https://dailyheumsi.tistory.com/136 [하나씩 점을 찍어 나가며]

 

### Catboost란 무엇인가?

Catboost란 Yandex에서 개발된 오픈 소스 Machine Learning이다. 이 기술은 다양한 데이터 형태를 활용하여 기업이 직면한 문제들을 해결하는데 도움을 준다. 특히 분류 정확성에서 높은 점수를 제공한다.

Catboost는 Category와 Boosting을 합쳐서 만들어진 이름이다.
 여기에서 Boost는 Gradient boosting machine learnin algorithm에서 온 말인데 Gradient boosting은 추천 시스템, 예측 등 다양한 분야에서 활용되어지는 강력한 방법이고 Deep Learning과 달리 적은 데이터로도 좋은 결과를 얻을 수 있는 효율적인 방법이다.

### 왜 Catboost를 활용하는가?

#### 더 좋은 결과

Catboost는 Benchmark에서 더 좋은 결과를 얻었다.

![img](file:///C:/Users/Minji/AppData/Local/Temp/msohtmlclip1/01/clip_image005.jpg)[GBDT Algorithms Benchmark](https://towardsdatascience.com/https-medium-com-talperetz24-mastering-the-new-generation-of-gradient-boosting-db04062a7ea2)

#### Category features 사용의 편리성

Category features를 사용하기 위해서는 One-Hot-Encoding등 데이터를 전처리할 필요가 있었지만 Catboost에서는 사용자가 다른 작업을 하지 않아도 자동으로 이를 변환하여 사용한다. 이 분야를 공부한 경험이 있다면 이 기능이 얼마나 편리한지를 알 수 있을 것이다. 자세한 내용은 [document](https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages_cat-to-numberic-docpage/)를 통해 확인할 수 있다.

#### 빠른 예측

학습 시간이 다른 GBDT에 보다는 더 오래 걸리는 대신에 예측 시간이 13-16배 정도 더 빠르다.

![img](file:///C:/Users/Minji/AppData/Local/Temp/msohtmlclip1/01/clip_image007.jpg)[Left : CPU, Right : GPU](https://towardsdatascience.com/https-medium-com-talperetz24-mastering-the-new-generation-of-gradient-boosting-db04062a7ea2)

#### 더 나은 기능들

·    default parameters값으로 더 나은 성능
 hyper-parmeter tuning을 하지 않더라도 기본적인 세팅으로도 좋은 결과를 얻을 수 있어 활용성이 뛰어나다. 자세한 내용은 [document](https://tech.yandex.com/catboost/doc/dg/concepts/parameter-tuning-docpage/)를 통해 확인할 수 있다.

![img](file:///C:/Users/Minji/AppData/Local/Temp/msohtmlclip1/01/clip_image009.jpg)[GBDT Algorithms with default parameters Benchmark](https://towardsdatascience.com/https-medium-com-talperetz24-mastering-the-new-generation-of-gradient-boosting-db04062a7ea2)

·    feature interactions

![img](file:///C:/Users/Minji/AppData/Local/Temp/msohtmlclip1/01/clip_image011.jpg)[Catboost’s Feature Interactions](https://towardsdatascience.com/https-medium-com-talperetz24-mastering-the-new-generation-of-gradient-boosting-db04062a7ea2)

·    feature importances

![img](file:///C:/Users/Minji/AppData/Local/Temp/msohtmlclip1/01/clip_image013.jpg)[Catboost’s Feature Importance](https://towardsdatascience.com/https-medium-com-talperetz24-mastering-the-new-generation-of-gradient-boosting-db04062a7ea2)

·    object(row) importances

![img](file:///C:/Users/Minji/AppData/Local/Temp/msohtmlclip1/01/clip_image015.jpg)[Catboost’s Object Importance](https://towardsdatascience.com/https-medium-com-talperetz24-mastering-the-new-generation-of-gradient-boosting-db04062a7ea2)

·    the snapshot

### 결론

프로젝트 등을 수행하다보면 Catgory feature를 이용하는 것이 상당히 번거롭다는 것을 알 수 있을 것이다. 뿐만 아니라 예측 시간이 오래걸린다면 실제로 시스템에 적용하는데는 큰 문제점을 가지고 있음을 알고 있다.

다른 Maching Learning algorithms의 단점을 보완해주는 Catboost를 잘 활용한다면 좀 더 나은 시스템을 개발하는데 도움이 될 것이다.

https://databuzz-team.github.io/2018/10/24/Catboost/

 

 