![header](https://capsule-render.vercel.app/api?type=wave&color=auto&height=300&section=header&text=Drug%20Review💊&fontSize=90)<br/>
MobileBert를 활용한 의약품 사용자 리뷰 감성분석 프로젝트
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
또한 소비자 약물 리뷰를 통해 부작용이 발생하는 원인을 파악하고 다른 환자들의 실제 경험을 확인할 수 있으며 일반 소비자들도 쉽게 접근할 수 있기 때문에 소비자들은 자신의 권리를 적극적으로 
주장할 수 있게 되고 의료 정보에 대한 이해도를 높일 수 있으며 의약품에 대한 불안감을 줄일 수 있다.
2016년 2월, comScore는 Drugs.com이 한 달 동안 약 2,300만 명의 방문자를 받는 6번째로 가장 인기 있는 건강 네트워크라고 발표했으며[[2]](https://en.wikipedia.org/wiki/Drugs.com)
Searchmetrics는 Drugs.com을 검색 가시성 상위 100개 미국 웹사이트에 포함시켰다.
2017년 4월 The Harris Poll은 Drugs.com을 올해의 건강 정보 웹사이트 브랜드로 선정했다.[[3]](https://en.wikipedia.org/wiki/Drugs.com)

![DRUG](https://user-images.githubusercontent.com/112537146/235820767-5d65e994-e282-4c5c-8cce-cd8076f51e4e.PNG)


## 2. 데이터
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


### 2-2. 데이터 부가정보 

![부가정보](https://user-images.githubusercontent.com/112537146/232723675-458cba09-0021-43e7-9161-078fa9e4417c.PNG)

데이터는 53766건이며 2008년부터 2017년까지의 자료이다.<br/>
평점(rating)은 1점부터 10점까지 구성되어있다.<br/>
리뷰 작성 날짜(date)는 Year, month, day 로 나타내고 있다.<br/>

##### 약물 개수 
![약물 종류 111](https://github.com/yeon0306/Drugs_Review/assets/112537146/fe26ec16-124d-43d9-b9f2-d9e80e109930)<br/>
약물종류(drugName)는 2637개이다.<br/>

##### 환자병명 개수 
![환자 병명](https://github.com/yeon0306/Drugs_Review/assets/112537146/53090564-1f8d-407d-8dd6-711243f8deb3)<br/>
환자가 앓고있는 질환(condition)의 종류는 709개이다.<br/>


### 2-3. 데이터 시각화

- 연도별 리뷰 건수

<div><img src = "https://user-images.githubusercontent.com/112537146/232725924-ff4b00f3-a64f-48e9-8702-5cc9e6334693.png" width="800"></div>

연도별 리뷰 건수를 그래프로 나타내보았다. 가장 리뷰가 적은 연도는 2008년,가장 리뷰가 많은 연도는 2016년인 것을 알 수 있다.<br/>

- 평점 분포

<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/337aa1f4-be0c-4512-a3e9-73b09eef13bb" width="800"></div>

1점부터 10점까지 평점 분포표를 그려보았다.<br/> 
사람들이 일반적으로 정말 좋아하는 약과 정말 싫어하는 약에 대한 리뷰를 쓴다는 것을 보여준다.<br/>
극단적인 점수들에 비해 중간 점수인 2점부터 7점까지는 상당히 적게 분포되어 있는것을 알 수 있다.<br/> 


- 유용한 리뷰 추천수와 평점

<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/ee8a7ff9-a71d-4adf-8e3f-4ab3d813d670" width="600"></div>

리뷰 추천수와 평점의 상관관계를 알아보기 위하여 리뷰 추천수와 평점 산점도를 그려보았다. <br/>
평점이 높을수록 리뷰 추천수가 높은 것을 볼 수 있다.  


<div><img src = "https://github.com/yeon0306/Drugs_Review/assets/112537146/9b0e14c0-77ac-4e99-a751-cba3bac9c9f6" width="600"></div>

좀 더 직관적으로 알아보기 위한 리뷰 추천수와 평점 평균 산점도를 그려보았다. <br/>
평균적으로 평점 1점이 리뷰 추천수가 가장 낮고, 평점 10점이 리뷰 추천수를 가장 많은 것을 알 수 있다. <br/> 


- 긍/부정 예측 

약물 리뷰의 긍/부정 예측은 평점의 5점보다 크거나 같을때 긍정, 5점보다 작을때 부정으로 예측해보겠다.

![sentiment](https://user-images.githubusercontent.com/112537146/235824425-4d7374d5-c083-4117-9bb4-f9fb043a9193.PNG)

긍정인 리뷰 건수는 40269건, 부정인 리뷰 건수는 13497으로 예측되었다. 

<div><img src = "https://user-images.githubusercontent.com/112537146/235824785-a88af42b-71b1-437b-9944-ba3c97f82781.PNG" width="600"></div>

약물 리뷰의 긍/부정 예측을 파이차트로 그려보았다. 긍정은 74.90%, 부정은 25.10% 이다.


![ㅇㄹㅇㄹㅇ](https://user-images.githubusercontent.com/112537146/235827185-ea29de5a-fe5b-4e81-95bc-e215fa51fb13.PNG)

LightGBM을 사용한 형상 중요도 그림이다.
가장 중요한 특징은 평균 단어 길이와 그 이후 환자의 상태임을 추론할 수 있다.




