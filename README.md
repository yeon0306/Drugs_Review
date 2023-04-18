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
데이터 출처- https://www.kaggle.com/datasets/arpikr/uci-drug

![image](https://user-images.githubusercontent.com/112537146/232665068-f27ee67d-bdb7-43ff-86af-dae4201743e7.png) 약물 이름 

![image](https://user-images.githubusercontent.com/112537146/232665115-6da782f0-f64c-4848-af70-b5d573a09d25.png) 환자 병명 

![image](https://user-images.githubusercontent.com/112537146/232665143-4a2d4040-1876-448d-ae4a-07aeeb0b6ca1.png) 리뷰

![image](https://user-images.githubusercontent.com/112537146/232664054-175260bb-8498-443b-be61-551808a2885b.png) 평점

![image](https://user-images.githubusercontent.com/112537146/232664063-5706008d-ceaf-43e8-8c46-094ea6af259f.png) 리뷰 작성 날짜(년월일)

![image](https://user-images.githubusercontent.com/112537146/232665292-bca5a5df-6654-45ce-85d0-42072d109f40.png) 추천수 


|uniqueID|drugName|condition|review|rating|date|usefulCount|
|--------|--------|---------|------|------|----|-----------|
|1|테스트2|테스트3|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|



<div><img src = "https://user-images.githubusercontent.com/112537146/232656257-a82044df-6a63-478d-a71d-3dbcadf2d427.png" width="300"></div>
