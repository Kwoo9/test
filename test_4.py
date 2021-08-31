## mnist 이미지 분류 코드

import tensorflow as tf

#입력데이터 가져오기_keras의 mnist 데이터셋 로드
mnist = tf.keras.datasets.mnist

#데이터 전처리, 이미지 데이터를 0~1사이의 부동소수점으로
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

#모델링하기
model = tf.keras.models.Sequential([
    #입력층
    tf.keras.layers.Flatten(input_shape=(28,28)),

    #히든레이어(relu함수 사용)
    tf.keras.layers.Dense(128, activation = 'relu'),

    #히든과 출력층 사이의 에지에 작용하는 드롭아웃 기법, 20퍼센트 드롭아웃
    #이를 통해 오버피팅 줄임
    tf.keras.layers.Dropout(0.2),

    #출력층(softmax로 확률출력)
    tf.keras.layers.Dense(10, activation='softmax')  
    ])

#모델 컴파일, 계산그래프 마감, 위의 구조적 모델의 학습방식 설정
#optimizer : adam방식으로 학습, 손실함수 : 크로스 엔트로피 방식으로 학습, 반환되는 값의 이름은 accuracy
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

#학습_모델을 객체지향적 학습
#fit함수를 통해 학습
model.fit(x_train, y_train, epochs=5)

#evaluate로 검증

## mnist 이미지 분류 코드

import tensorflow as tf

#입력데이터 가져오기_keras의 mnist 데이터셋 로드
mnist = tf.keras.datasets.mnist

#데이터 전처리, 이미지 데이터를 0~1사이의 부동소수점으로
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

#모델링하기
model = tf.keras.models.Sequential([
    #입력층
    tf.keras.layers.Flatten(input_shape=(28,28)),

    #히든레이어(relu함수 사용)
    tf.keras.layers.Dense(128, activation = 'relu'),

    #히든과 출력층 사이의 에지에 작용하는 드롭아웃 기법, 20퍼센트 드롭아웃
    #이를 통해 오버피팅 줄임
    tf.keras.layers.Dropout(0.2),

    #출력층(softmax로 확률출력)
    tf.keras.layers.Dense(10, activation='softmax')  
    ])

#모델 컴파일, 계산그래프 마감, 위의 구조적 모델의 학습방식 설정
#optimizer : adam방식으로 학습, 손실함수 : 크로스 엔트로피 방식으로 학습, 반환되는 값의 이름은 accuracy
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

#학습_모델을 객체지향적 학습
#fit함수를 통해 학습
model.fit(x_train, y_train, epochs=5)

#evaluate로 검증
model.evaluate(x_test, y_test, verbose=2)