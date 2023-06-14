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
|0|206461|Valsartan|Left Ventricular Dysfunction|"It has no side effect, I take it in combinati...|9|20-May-12|27|
|1|95260|Guanfacine|ADHD|	"My son is halfway through his fourth week of ...|8|27-Apr-10|192|
|2|92703|Lybrel|Birth Control|"I used to take another oral contraceptive, wh...|5|14-Dec-09|17|
|3|138000|Ortho Evra|Birth Control|"This is my first time using any form of birth...|8|3-Nov-15|10|
|4|35696|Buprenorphine/naloxone|Opiate Dependence|"Suboxone has completely turned my life around...|9|27-Nov-16|37|

데이터는 53766건이며 2008년부터 2017년까지의 자료이다.<br/>
평점(rating)은 1점부터 10점까지 구성되어있다.<br/>
리뷰 작성 날짜(date)는 Year, month, day 로 나타내고 있다.<br/>


### 2-2. 데이터 부가정보 

**약물 개수** 

![약물 종류 111](https://github.com/yeon0306/Drugs_Review/assets/112537146/fe26ec16-124d-43d9-b9f2-d9e80e109930)<br/>
약물종류(drugName)는 2637개이다.<br/>

**리뷰가 가장 많은 약물 상위 10개를 뽑은 그래프** 
<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/fa86bc7d-103d-42e0-9318-40d3f8029ee3" width="1200"></div>

|순위|약물이름|  설명  |
|-|--------|--------|
|1|Levonorgestrel(레보노게스트렐)|응급 피임약|
|2|Etonogestrel (에토노게스트렐)|여성 피임약|
|3|Ethinyl estradiol/norethindrone(에티닐 에스트라디올/노르에틴드론 조합)|여성 복합 피임약|
|4|Nexplanon(넥스플래논)|여성 피임 임플란트, 팔에 성냥개비 크기의 플라스틱 막대의 임플란트를 이식하는 피임 시술|
|5|Ethinyl estradiol/norgestimate (에스트라디올/노르게스트 조합)|여성 피임약이지만 여드름 치료로도 사용|
|6|Ethinyl estradiol/levonorgestrel (에티닐에스트라디올/레보노르게스트렐)|복합 피임약,생리통,자궁 내막증 및 응급 피임약|
|7|Phentermine(펜터민)|비만 치료제|
|8|sertraline(설트랄린)|우울증, 강박증 치료제|
|9|Escitalopram(에스시탈로프람)|우울증, 공황장애, 사회불안장애 등 광범위하게 사용되는 항우울제|
|10|mirena (미레나)|자궁 내 피임 시술| 

그래프의 설명을 돕기 위해 약물 이름과 설명을 표로 만들었다. 여성 피임약과 피임 시술 리뷰가 가장 많다.

**환자병명 개수**

![환자 병명](https://github.com/yeon0306/Drugs_Review/assets/112537146/53090564-1f8d-407d-8dd6-711243f8deb3)<br/>
환자가 앓고있는 질환(condition)의 종류는 709개이다.<br/>

**리뷰가 가장 많은 질환 (condition)**
<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/db49db21-2ea3-439b-923b-e01380de3463" width="800"></div>

|순위|질환|  --  |
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

두 그래프의 결과가 상관관계가 있음을 알 수 있다.
리뷰가 가장 많은 약물 종류가 피임약인 만큼 리뷰가 가장 많은 질환이 피임이다.



### 2-3. 데이터 시각화

- 평점 분포

<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/337aa1f4-be0c-4512-a3e9-73b09eef13bb" width="800"></div>

1점부터 10점까지 구성되어있는 평점 분포표이다.<br/> 
사람들이 일반적으로 정말 좋아하는 약과 정말 싫어하는 약에 대한 리뷰를 쓴다는 것을 보여준다.<br/>
극단적인 점수들에 비해 중간 점수인 2점부터 7점까지는 상당히 적게 분포되어 있는것을 알 수 있다.<br/> 


- 유용한 리뷰 추천수와 평점

<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/9b0e14c0-77ac-4e99-a751-cba3bac9c9f6" width="600"></div>

리뷰 추천수와 평점의 상관관계를 알아보기 위하여 리뷰 추천수와 평점 평균 산점도를 그려보았다. <br/>
평균적으로 평점 1점이 리뷰 추천수가 가장 낮고, 평점 10점이 리뷰 추천수를 가장 많은 것을 알 수 있다. <br/> 
평점이 높을수록 리뷰 추천수가 높은 것을 볼 수 있다.  <br/>


- 긍/부정 예측 

약물 리뷰의 긍/부정 예측은 평점의 5점보다 크거나 같을때 긍정, 5점보다 작을때 부정으로 예측해보겠다.

![sentiment](https://user-images.githubusercontent.com/112537146/235824425-4d7374d5-c083-4117-9bb4-f9fb043a9193.PNG)

긍정인 리뷰 건수는 40269건, 부정인 리뷰 건수는 13497건으로 예측되었다. 

<div><img src = "https://user-images.githubusercontent.com/112537146/235824785-a88af42b-71b1-437b-9944-ba3c97f82781.PNG" width="600"></div>

약물 리뷰의 긍/부정 예측을 파이차트로 그려보았다. 긍정은 74.90%, 부정은 25.10% 이다.

- 리뷰 문장 길이 

<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/01940a74-943e-4998-9be4-4dd9fe536c2e" width="800"></div>

리뷰 문장 길이 20자 이상 850자 이하 기준 그래프이다.<br/>
700자 중후반의 리뷰가 약 5천건으로 가장 많다. <br/> 

### 3. 데이터 전처리 

원시 데이터의 약물 종류가 2637개, 질환이 709개라는 점과 약물, 질환마다 리뷰에 쓰이는 의학 용어와 단어들이 매우 많아 데이터의 일관성을 위하여 <br/> 
유의미 있는 데이터라고 판단된 리뷰가 가장 많은 질환 'Birth Control' 의 리뷰 데이터만 추출하여 학습하기로 했다. <br/>
원시 데이터 53766건 중 'Birth Control' 은 9648건이다. <br/>



*positive 0 / negative 1* 


|-|review|label|
|-|------|--------|
|0|"I have been on this birth control for one cycle ...|0|
|1|"I absolutely love this product and recommend ...|0|
|2|"honestly the measurement hurt more than the actual placement! And I'm... |0|
|3|"I'm 21 and have two babies. I had the Mirena inserted about ...|1|
|4|"Nexplanon was the WORST birth control I've had ...|1|
|5|"Had the Implanon for 4 months and I just got it removed today ...|1|

질환이 'Birth Control'인 리뷰 데이터만 추출하였다. <br/>
리뷰 문장 길이 20자 이하 데이터와 850자 이상 데이터는 삭제하였다.<br/>
긍/부정 예측을 정확하게 하기 위해 애매한 평점 4 ~ 6점의 데이터는 삭제하였고, 긍정은 8 ~ 10점 / 부정은 1 ~ 3점 기준으로 정하였다.<br/>
기존 rating 을 삭제하고 label을 추가하여 긍정일 경우 0, 부정일 경우 1 으로 기입하였다. <br/>


### 4. 데이터 학습 

birth control의 긍정/부정 데이터를 각각 1000건씩 추출하여 2000건을 학습했다.<br/>

<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/862e664f-80cc-4b06-b19b-acd701d58ea3" width="600"></div>
모델의 긍/부정 예측 정확도가 0.94로 높게 나왔다. 

<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/e6d00927-9616-4ed5-ba10-debd626918c5" width="400"></div>
모델의 loss 그래프 
학습할수록 loss가 떨어지고 있다.

<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/1df0340d-e41b-4aad-a2ca-dae2c0677a00" width="400"></div>
Accuracy 그래프 
학습할수록 정확도가 올라간다.

<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/92578c84-d479-46f1-99ee-761fd37b72db" width="600"></div>
birth control의 전체 데이터 7228건을 학습시켜보았더니 Accuracy가 0.91이 나왔다. 

### 5. 최종 결과

일부 데이터를 학습 시킨 모델에 전체 데이터 44159건을 예측해보았다. 
<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/7ec24f1a-33d3-46e0-9be8-d7414b4898fa" width="600"></div>

전체 데이터의 긍부정 예측 정확도는 0.79가 나왔다. </br>
drugs review 데이터 특성상 2000개의 종류가 넘는 약물과 700여개의 질환마다 각기 다른 의학 용어들과 문장 표현 방식때문에 정확도 예측에 어려움이 있었다.
전체 데이터를 섞어 2000개의 긍부정 데이터를 추출한 Accuracy가 0.57이 나왔었기 때문에 전체 데이터 예측 Accuracy는 더욱 낮을거라 예상했지만,
일관성을 위해 유의미한 데이터를 추출하여 일부 데이터로 학습시킨 모델은 전체 데이터의 예측 정확도가 0.79이기에     
꽤 높은 정확도를 보여주어 일부 데이터를 학습 시킨 모델이 유의미하다고 결론을 낼 수 있다. </br>



