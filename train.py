"""
    Author: Alan Park (alanworks72@gmail.com)
    Date: Jul.09.2024
    File Name: train.py
    Version: 1.0
    Description: Model Trainer for Diabetes classifier
"""


import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data import dataloader
from proc import visualize


def train(dataset):
    """
    @brief 모델을 학습시키고 성능을 검증하는 함수
    
    @param dataset(tuple): 학습과 테스트를 위한 데이터셋 (x_train, x_test, y_train, y_test)
    """
    # 학습과 테스트를 위한 데이터 분리
    x_train, x_test, y_train, y_test = dataset

    # 랜덤 포레스트 분류기 학습 - 42번 랜덤 플래그를 사용하여 100회 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    # 모델 테스트
    y_pred = model.predict(x_test)

    # 테스트 성능 확인
    validation(y_test, y_pred)
    # 학습 결과 시각화
    visualize(model)

def validation(y_test, y_pred):
    """
    @brief 모델의 성능을 검증하는 함수
    
    @param y_test(np.array): 실제 레이블
    @param y_pred(np.array): 예측된 레이블
    """
    # 평가지표 출력 - Accuracy: 모델의 정확도, Confusion Matrix: 오검, 미검 확인, Classification Report: 분류 성능 종합 평가
    print("Metric Results")
    print("Accuracy: ", accuracy_score(y_test, y_pred),"\n")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred),"\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))

def run(path):
    """
    @brief 데이터 로드 후 모델 학습을 시작하는 함수
    
    @param path(str): 데이터셋 파일 경로
    """
    # 데이터 불러오기 및 전처리
    dataset = dataloader(path)
    # 모델 학습 시작
    train(dataset)

if __name__ == "__main__":
    # 데이터셋 경로 설정
    path = "./diabetes.csv"
    # 프로세스 시작
    run(path)