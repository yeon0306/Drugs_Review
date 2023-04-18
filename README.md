# Drugs Review
UCI ML Drug Review dataset 를 활용한 약물 사용자 리뷰 감성분석 

# 1.개요 
## 1-1 문제정의
약물 리뷰는 소비자들이 의학적인 정보를 얻을 수 있는 중요한 수단이다.
UCI ML Drug Review 데이터셋은 대규모 약물 리뷰 데이터셋으로 소비자들이 제출한 약물 리뷰와 그에 따른 평점, 부작용 정보 등을 포함하고 있다.
약물의 효과를 파악하기 위해서는 많은 수의 환자들이 복용한 후의 리뷰 데이터를 활용하여 특정 약물의 증상과 효과 등을 분석할 수 있다.
이 프로젝트에서는 각종 질환을 앓고 있는 환자들의 약물 리뷰, 평점 등 다양한 특징에 따라 긍정 또는 부정을 예측하는 인공지능 모델을 개발하고자 한다.

## 1-2 약물 리뷰의 영향력 

약물 리뷰는 다양한 사회적, 산업적 영향을 미친다.
약물 리뷰를 통해 다수의 환자들의 경험을 취합하고 분석하면 의료기술 발전에 큰 도움이 된다. 약물 리뷰를 통해 부작용이 발생하는 원인을 파악하고
이를 토대로 부작용을 최소화하는 새로운 약물을 개발할 수 있다.
또한 약물 리뷰를 통해 다른 환자들의 실제 경험을 확인할 수 있다. 일반 소비자들도 쉽게 접근할 수 있기 때문에 소비자들은 자신의 권리를 적극적으로 주장할 수 있게 되고
의료 정보에 대한 이해도를 높일 수 있으며 의료 서비스에 대한 불안감을 줄일 수 있다.


# 2.데이터
[UCI-DRUG 데이터 출처](https://www.kaggle.com/datasets/arpikr/uci-drug "UCI-DRUG")


|-|uniqueID|drugName|condition|review|rating|date|usefulCount|
|-|--------|--------|---------|------|------|----|-----------|
|0|206461|Valsartan|Left Ventricular Dysfunction|"It has no side effect, I take it in combinati...|9|20-May-12|27|
|1|95260|Guanfacine|ADHD|	"My son is halfway through his fourth week of ...|8|27-Apr-10|192|
|2|92703|Lybrel|Birth Control|"I used to take another oral contraceptive, wh...|5|14-Dec-09|17|
|3|138000|Ortho Evra|Birth Control|"This is my first time using any form of birth...|8|3-Nov-15|10|
|4|35696|Buprenorphine/naloxone|Opiate Dependence|"Suboxone has completely turned my life around...|9|27-Nov-16|37|



<div><img src = "https://user-images.githubusercontent.com/112537146/232656257-a82044df-6a63-478d-a71d-3dbcadf2d427.png" width="300"></div>
