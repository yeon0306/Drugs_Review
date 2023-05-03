# Drug Review 💊
MobileBert를 활용한 의약품 사용자 리뷰 감성분석 프로젝트 <br/>
<br/>
<img src="https://img.shields.io/badge/PyTorch-E34F26?style=flat-square&logo=PyTorch&logoColor=white"/></a>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/></a>

## 1.개요 

### 1-1. 문제정의
[Drugs.com](https://www.drugs.com/) 은 미국에서 가장 큰 독립적인 의약품 정보 웹사이트이다.
소비자들이 제출한 약물 리뷰와 그에 따른 평점, 부작용 정보등을 포함하고 있다.
약물의 효과를 파악하기 위해서는 많은 수의 환자들이 복용한 후의 리뷰 데이터를 활용하여 특정 약물의 증상과 효과 등을 분석할 수 있다.
UCI Drug dataset은 캘리포니아 어바인 대학에서 운영하는 UCI 머신러닝 저장소에서 제공하는 의약품 사용자 리뷰 데이터셋이다.
이 프로젝트에서는 각종 질환을 앓고 있는 환자들의 약물 리뷰, 평점 등 다양한 특징에 따라 긍정 또는 부정을 예측하는 인공지능 모델을 개발하고자 한다.

![UCI](https://user-images.githubusercontent.com/112537146/232968932-239bbea7-af8e-4f7e-b163-7f61e555e188.PNG)



### 1-2. 약물 리뷰의 영향력
[drugs.com](https://www.drugs.com/support/about.html)은
미국에서 소비자와 의료 전문가에게 
약물 리뷰는 다양한 사회적, 산업적 영향을 미친다.
약물 리뷰를 통해 다수의 환자들의 경험을 취합하고 분석하면 의료기술 발전에 큰 도움이 된다.
약물 리뷰를 통해 부작용이 발생하는 원인을 파악하고 이를 토대로 부작용을 최소화하는 새로운 약물을 개발할 수 있다.
또한 약물 리뷰를 통해 다른 환자들의 실제 경험을 확인할 수 있다. 일반 소비자들도 쉽게 접근할 수 있기 때문에 소비자들은 자신의 권리를 적극적으로 
주장할 수 있게 되고 의료 정보에 대한 이해도를 높일 수 있으며 의약품에 대한 불안감을 줄일 수 있다.


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

    data = pd.read_csv('UCIdrug_test.csv')
    data.describe()

![부가정보](https://user-images.githubusercontent.com/112537146/232723675-458cba09-0021-43e7-9161-078fa9e4417c.PNG)

데이터는 53766건이며 2008년부터 2017년까지의 자료이다.<br/>
평점(rating)은 1점부터 10점까지 구성되어있다.<br/>
리뷰 작성 날짜(date)는 Year, month, day 로 나타내고 있다.<br/>


### 2-3. 데이터 시각화

![download](https://user-images.githubusercontent.com/112537146/232725924-ff4b00f3-a64f-48e9-8702-5cc9e6334693.png)

연도별 리뷰 건수를 그래프로 나타내보았다. 가장 리뷰가 적은 연도는 2008년,가장 리뷰가 많은 연도는 2016년인 것을 알 수 있다.
