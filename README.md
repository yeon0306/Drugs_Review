![header](https://capsule-render.vercel.app/api?type=soft&color=auto&height=300&section=header&text=Drug%20Review💊&fontSize=90)<br/>
## MobileBert를 활용한 의약품 사용자 리뷰 감성분석 프로젝트
<img src="https://img.shields.io/badge/PyTorch-E34F26?style=flat-square&logo=PyTorch&logoColor=white"/></a>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/></a>
## 1.개요 

### 1-1. 문제정의
UCI Drug dataset은 캘리포니아 어바인 대학에서 운영하는 UCI 머신러닝 저장소에서 제공하는 의약품 사용자 리뷰 데이터셋이다.
[drugs.com](https://www.drugs.com/support/about.html)의 소비자 약물 리뷰와 그에 따른 평점, 부작용 정보 등을 포함하고 있다. 
약물의 효과를 파악하기 위해서는 많은 수의 환자들이 복용한 후의 리뷰 데이터를 활용하여 특정 약물의 증상과 효과 등을 분석할 수 있다.
이 프로젝트에서는 각종 질환을 앓고 있는 환자들의 약물 리뷰, 평점 등 다양한 특징에 따라 긍정 또는 부정을 예측하는 인공지능 모델을 개발하고자 한다.

![UCI](https://user-images.githubusercontent.com/112537146/232968932-239bbea7-af8e-4f7e-b163-7f61e555e188.PNG)



### 1-2. 약물 리뷰의 영향력
미국에서 소비자와 의료 전문가에게 의약품 정보 및 리뷰 서비스를 제공하는 [drugs.com](https://www.drugs.com/support/about.html)은 보건의료산업에 큰 영향력을 주었다.
2010년 5월 미국 FDA는 Drugs.com 웹사이트 및 모바일 플랫폼에서 소비자 건강 업데이트를 배포하기 위해 Drugs.com과의 협력을 발표했다[[1]](https://en.wikipedia.org/wiki/Drugs.com)
또한 소비자 약물 리뷰를 통해 부작용이 발생하는 원인을 파악하고 다른 환자들의 실제 경험을 확인할 수 있으며 일반 소비자들도 쉽게 접근할 수 있기 때문에 소비자들은 자신의 권리를 적극적으로 주장할 수 있게 되고 의료 정보에 대한 이해도를 높일 수 있으며 의약품에 대한 불안감을 줄일 수 있다. 
실제 drugs.com 사용자들은 다른 사용자들의 리뷰로 의약품 지식과 부작용 등의 정보를 얻고 있다. [[2]](https://www.mouthshut.com/websites/Drugs-com-reviews-925769162)
<div> <img src = "https://user-images.githubusercontent.com/112537146/235820767-5d65e994-e282-4c5c-8cce-cd8076f51e4e.PNG" width="400"> <img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/6fcc8b7c-e8a1-4846-8ba2-38d63ab58f8c" width="400"> </div>

2016년 2월, comScore는 Drugs.com이 한 달 동안 약 2,300만 명의 방문자를 받는 6번째로 가장 인기 있는 건강 네트워크라고 발표했으며 [[3]](https://en.wikipedia.org/wiki/Drugs.com)
Searchmetrics는 Drugs.com을 검색 가시성 상위 100개 미국 웹사이트에 포함시켰다.
2017년 4월 The Harris Poll은 Drugs.com을 올해의 건강 정보 웹사이트 브랜드로 선정했다.  [[4]](https://en.wikipedia.org/wiki/Drugs.com)



## 2. 원시 데이터

[UCI-DRUG 데이터셋](https://www.kaggle.com/datasets/arpikr/uci-drug "UCI-DRUG")<br/>
[UCI 홈페이지](https://archive.ics.uci.edu/ml/index.php "UCI")


### 2-1. 데이터 구성

- 데이터명 

|uniqueID|drugName|condition|review|rating|date|usefulCount|
|--------|--------|--------|------|------|----|-----------|
|고유 식별자|의약품명|환자 병명|리뷰|평점|작성날짜(년월일)|유용한 리뷰추천수|


- 활용할 데이터 예시 

|-|uniqueID|drugName|condition|review|rating|date|usefulCount|
|-|--------|--------|--------|------|------|----|-----------|
|1|206461|Valsartan|Left Ventricular Dysfunction|"It has no side effect, I take it in combinati...|9|20-May-12|27|
|2|95260|Guanfacine|ADHD|	"My son is halfway through his fourth week of ...|8|27-Apr-10|192|
|3|92703|Lybrel|Birth Control|"I used to take another oral contraceptive, wh...|5|14-Dec-09|17|
|..|...|...|.....|.....|...|...|..|
|53764|130945|Levonorgestrel|Birth Control|"I'm married, 34 years old and I have no kids. Taking..."|8|15-Nov-10|7|
|53765|47656|Tapentadol|Pain|"I was prescribed Nucynta for severe neck/shoulder pain..."|1|28-Nov-11|20|
|53766|113712|Arthrotec|Sciatica|"It works!!!"|9|13-Sep-09|46|

데이터는 53766건이며 평점(rating)은 1점부터 10점까지 구성되어있다.<br/>

### 2-2. 데이터 부가정보 

**약물 개수** 

![약물 종류 111](https://github.com/yeon0306/Drugs_Review/assets/112537146/fe26ec16-124d-43d9-b9f2-d9e80e109930)<br/>
데이터의 약물종류(drugName)는 2637개이다.<br/>

**리뷰가 가장 많은 약물 Top10 (drugsName)** 
<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/fa86bc7d-103d-42e0-9318-40d3f8029ee3" width="1200"></div>

|순위|약물이름|  설명  |
|-|--------|--------|
|1|Levonorgestrel(레보노게스트렐)|응급 피임약|
|2|Etonogestrel (에토노게스트렐)|여성 피임약|
|3|Ethinyl estradiol/norethindrone(에티닐 에스트라디올/노르에틴드론 조합)|여성 복합 피임약|
|4|Nexplanon(넥스플래논)|여성 피임 임플란트, 팔에 성냥개비 크기의 플라스틱 막대의 임플란트를 이식하는 피임 시술|
|5|Ethinyl estradiol/norgestimate (에스트라디올/노르게스트 조합)|여성 피임약이지만 여드름 치료로도 사용|
|6|Ethinyl estradiol/levonorgestrel (에티닐에스트라디올/레보노르게스트렐)|복합 피임약,생리통,자궁 내막증 및 응급 피임약|
|7|Phentermine(펜터민)|비만,식욕 억제제|
|8|sertraline(설트랄린)|우울증, 강박증 치료제|
|9|Escitalopram(에스시탈로프람)|우울증, 공황장애, 사회불안장애 등 광범위하게 사용되는 항우울제|
|10|mirena (미레나)|자궁 내 피임 시술| 

약물 데이터의 전반을 이해하기 위해 리뷰가 가장 많은 약물 상위 10개 그래프를 그려 시각화하고 약물 이름과 설명을 표로 만들었다.
주로 여성 피임약과 피임 시술에 관련된 약물들인 것을 볼 수 있으며 그 외 비만과 우울증, 강박증 등의 항우울제 약물들이 리뷰가 가장 많은 약물 상위 10개에 포함되어 있다.

**환자질환 개수**

![환자 병명](https://github.com/yeon0306/Drugs_Review/assets/112537146/53090564-1f8d-407d-8dd6-711243f8deb3)<br/>
데이터의 환자가 앓고있는 질환(condition)의 종류는 709개이다.<br/>

**리뷰가 가장 많은 질환 Top10 (condition)**
<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/db49db21-2ea3-439b-923b-e01380de3463" width="800"></div>

|순위|질환|     |
|-|--------|--------|
|1|birth control|피임|
|2|Depression|우울증|
|3|Pain|통증|
|4|Anxiety|불안장애| 
|5|Acne|여드름|
|6|Bipolar Disorder|조울증| 
|7|Weight Loss|체중감소| 
|8|Insomnia|불면증| 
|9|Obesity|비만 |
|10|ADHD|주의력결핍 과잉행동장애|

리뷰가 가장 많은 질환은 약물 그래프에서 주를 차지했던 피임약들과 피임 시술들의 질환인 'Birth control'으로, 원시 데이터 53766건 중 9648건을 차지한다.
또한 이 그래프에서 알 수 있는 것은 Depression(우울증)과 Anxiety(불안장애),Bipolar Disorder(조울증), Insomnia(불면증), ADHD(주의력결핍 과잉행동장애)는 주로 항정신병약물을 사용하는 질환이기에 
약물 그래프 상위 10개 그래프의 들었던 항우울제인 약물과 관련이 있으므로 이는 두 그래프의 결과가 상관관계가 있다고 볼 수 있다.


### 2-3. 데이터 시각화

- 평점 분포

<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/337aa1f4-be0c-4512-a3e9-73b09eef13bb" width="600"></div>

1점부터 10점으로 구성되어있는 평점 분포표이다. 극단적인 점수들인 1점과 9~10점에 비해 2점부터 7점까지는 상당히 적게 분포되어 있다는 것을 볼 수 있으며
사람들이 일반적으로 정말 긍정적인 약물과 정말 부정적인 약물에 대한 리뷰를 쓴다는 점을 알 수 있다.

- 유용한 리뷰 추천수와 평점

<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/9b0e14c0-77ac-4e99-a751-cba3bac9c9f6" width="600"></div>

리뷰 추천수와 평점의 상관관계를 알아보기 위하여 리뷰 추천수와 평점 평균 산점도를 그려보았다. <br/>
평균적으로 평점 1점이 리뷰 추천수가 가장 낮고, 평점 10점이 리뷰 추천수를 가장 많은 것으로 보아 평점이 높을수록 리뷰 추천수가 높은 것을 볼 수 있다. <br/>


- 긍/부정 예측 

전체 약물 리뷰의 긍/부정 예측은 평점의 5점보다 크거나 같을때 긍정, 5점보다 작을때 부정으로 예측해보겠다.

![sentiment](https://user-images.githubusercontent.com/112537146/235824425-4d7374d5-c083-4117-9bb4-f9fb043a9193.PNG)

긍정인 리뷰 건수는 40269건, 부정인 리뷰 건수는 13497건으로 예측되었다. 

<div><img src = "https://user-images.githubusercontent.com/112537146/235824785-a88af42b-71b1-437b-9944-ba3c97f82781.PNG" width="600"></div>

약물 리뷰의 긍/부정 예측을 파이차트로 그려보았다. 긍정은 74.90%, 부정은 25.10% 으로, 긍정적인 리뷰가 압도적으로 많은 것을 확인할 수 있다.

- 리뷰 문장 길이 

<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/01940a74-943e-4998-9be4-4dd9fe536c2e" width="800"></div>

약물 리뷰 문장 길이 20자 이상 850자 이하 기준 그래프이다.<br/>
700자 중후반의 리뷰가 약 5천건으로 가장 많으며, 800자 이상의 리뷰가 가장 적게 나타난다. <br/> 

## 3. 데이터 전처리 

원시 데이터의 약물 종류가 2637개, 질환이 709개라는 점과 약물과 질환마다 리뷰에 쓰이는 의학 용어와 단어들이 매우 많아 데이터의 일관성을 위하여
위 데이터 시각화를 토대로 유의미 있는 데이터라고 판단된  **'Birth Control'** 의 리뷰 데이터를 추출하기로 했다.
추출한 Birth Control 데이터에서 클래스 불균형 문제를 보완하기 위해 긍정 부정 비율을 동일하게 추출하여 Birth Control 일부 데이터셋을 만들고, 일부 데이터로 학습한 모델로
전체 데이터와 Birth Control 데이터를 예측해 보기로 했다. 

### 3-1.전체 분석 데이터 
*positive 0 / negative 1* 

| |review|label|
|-|------|--------|
|1|"I've tried a few antidepressants over the years...|0|
|2|"My son has Crohn's disease and has done very well on the Asacol...|0|
|..|...|..|
|44162|"I'm married, 34 years old and I have no kids. Taking the pill was such...|0|
|44163|"I was prescribed Nucynta for severe neck/shoulder pain...|1|

기존 rating을 삭제하고 label 이라는 열을 새로 생성하여 평점이 8 ~ 10점으로 긍정적인 리뷰에는 0을, 평점이 1 ~ 3점인 부정적인 리뷰에는 1을 부여하였다. 
평점이 4~6점인 리뷰 데이터와 리뷰 문장 길이가 20자 미만, 800자 이상 데이터는 삭제하였고 'review', 'label' 데이터로 새로운 데이터셋을 생성하였다.

### 3-2.Birth Control 분석 데이터 

| |review|label|
|-|------|--------|
|1|"I have been on this birth control for one cycle ...|0|
|2|"I absolutely love this product and recommend ...|0|
|..|...|..|
|7227|"I first would like to thank all of you that posted comments...|1|
|7228|"I started taking Apri about 7 months ago. My breats got noticeably...|0|

가공된 전체 데이터에서 질환이 'Birth Control'인 데이터만 추출하였다.

### 3-3.Birth Control 학습 데이터 


| |review|label|
|-|------|--------|
|1|"I have been on this birth control for one cycle...|0|
|2|"I absolutely love this product and recommend...|0|
|..|...|..|
|1999|"I was hospitalized within the first 3 weeks of taking...|1|
|2000|"I'm 21 and have two babies. I had the Mirena...|1|

앞서 가공한 데이터의 긍부정 비율이 불균형적이기 때문에 학습 정확도를 높이기 위해
Birth Control의 긍정적인 리뷰와 부정적인 리뷰의 데이터를 각각 1000개씩 임의로 추출하여 2000개의 학습 데이터를 만들었다. 

## 4.결과

#### 개발환경  
<img src="https://img.shields.io/badge/pycharm 2022.3.3-000000?style=flat-square&logo=pycharm&logoColor=white"/> <img src="https://img.shields.io/badge/Python 3.9.0-3776AB?style=flat-square&logo=Python&logoColor=white"/>  
#### 패키지  
<img src="https://img.shields.io/badge/pandas 1.4.4-150458?style=flat-square&logo=pandas&logoColor=white"/> <img src="https://img.shields.io/badge/torch 1.12.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/> <img src="https://img.shields.io/badge/tensorflow 2.9.1-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/> <img src="https://img.shields.io/badge/numpy 1.24.2-013243?style=flat-square&logo=numpy&logoColor=white"/> <img src="https://img.shields.io/badge/transformers 4.21.2-81c147?style=flat-square&logo=transformers&logoColor=white"/> <img src="https://img.shields.io/badge/scikit-learn 1.2.2-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/> <img src="https://img.shields.io/badge/matplotlib 3.7.1-3776AB?style=flat-square&logo=matplot&logoColor=white"/>


### 4-1.MobileBERT 사용한 결과

<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/c8fc552f-6637-4fcf-a822-9fcc8a22a369" width="400"></div>

<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/fdd7bb5b-34e2-40ea-be5e-41862536c26c" width="700"><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/060d7b68-118c-4753-ad02-438de51095f3" width="700"> </div>

|step|0|1|2|3|
|:---:|:---:|:---:|:---:|:---:|
|loss|3.7182e+5|13.11|0.20|0.13|
|accuracy|0.9|0.93|0.94|0.95|

텐서보드를 활용하여 모델의 학습 단계를 시각화한 그래프이다.
초기 단계에서의 loss는 371,820으로 매우 높은 값을 나타내지만 훈련이 진행될 수록 loss가 감소하고 있으며 세번째 단계에서는 loss값이 0.13으로 매우 감소했다.
Accuracy(정확도)는 학습할 수록 증가했으며, 학습 데이터의 긍부정 예측 정확도는 0.95가 나왔다. 
즉 모델이 학습 데이터의 긍정과 부정을 분류하는 것을 올바르게 학습되었다는 것을 확인할 수 있다.

### 4-2.Birth control 분석 데이터에 적용한 결과값

<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/92578c84-d479-46f1-99ee-761fd37b72db" width="600"></div>
birth control 데이터를 학습시켜보았더니 Accuracy가 0.91이 나왔다. 


### 4-3.전체 분석 데이터에 적용한 결과값

<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/7ec24f1a-33d3-46e0-9be8-d7414b4898fa" width="600"></div>

전체 분석 데이터에 적용한 Accuracy 결과값은 0.79가 나왔다. </br>
예상했던 것보다는 정확도가 높게 나왔지만, Birth Control 분석 데이터의 결과값에 비하면 낮다고 볼 수 있다. </br>


## 5. 결론 및 배운 점 

약물 리뷰 데이터 특성상 2000개의 종류가 넘는 약물과 700여개의 질환마다 생소한 의학 용어들과 각기 다른 
표현 방식 때문에 데이터의 일관성이 부족하여 정확도 예측에 어려움이 있었다. 전체 데이터에서 랜덤으로 2000개의 긍부정 데이터를 
추출하여 학습한 모델의 Accuracy는 0.57이 나왔었기 때문에 전체 분석 데이터에 적용한 결과값도 크게 
다르지 않을 거라 예상했는데 일부 데이터로 학습시킨 모델을 전체에 적용한 결과값은 0.79로 꽤 높게 나와서 의외였었다. 
왜 그럴까 생각해보니 첫번째로 Birth Control 이라는 일관성이 있는 피임약 데이터들만 추출하였고,
두번째로는 피임약은 통상적으로 많은 사람들이 사용하는 대중적인 약품이라는 것이다. Birth Control의 리뷰 내용을 보니 
일반적으로 많이 사용하는 표현들이 많았고, 모델 토큰에 없을 만한 생소한 의학 용어들은 매우 적었다.
모든 약물 리뷰에 의학 용어만 사용된 것은 아니기 때문에 어느정도 피임약 리뷰 데이터에서 부정적인 경험, 긍정적인 경험을 표현하는 방식,
피임약과는 다른 약물이지만 다른 약물에서 나타나는 비슷한 부작용 사례(복통,어지럼증,두통 같은 통상적으로 많이 겪을 수 있는 부작용)를 고려해봤을 때 이런 경우의 수로 
예상보다는 높은 정확도가 나온 것 같다.
물론 어느정도 운의 요소도 따랐을 것이라고 생각되므로 온전히 일부 데이터를 학습 시킨 모델이 유의미하다고 결론을 내리기는 어려울 것 같다. 
배운 점은 Accuray를 높이기 위해서는 데이터의 전처리와 데이터의 일관성은 모델 학습에 매우 중요한 요소임을 깨달을 수 있었고
약물 같은 생소한 데이터를 다루면서 이 분야에 좀 더 관심이 있거나 지식이 많았더라면 더 수월하지 않았을까하는 생각이 든다.
이 프로젝트의 개선할 점은 약물과 질환을 종류대로 묶어서 데이터의 일관성을 더욱 갖추고, 약물 리뷰에 있는 의학 용어들을 토큰화 한다면 더욱 유의미한 약물 긍부정 예측 모델을 
만들 수 있지 않았을까하며 아쉬움이 남는다. 











