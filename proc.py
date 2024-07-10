"""
    Author: Alan Park (alanworks72@gmail.com)
    Date: Jul.10.2024
    File Name: proc.py
    Version: 1.0
    Description: Data Processor for Diabetes classifier
"""


import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preProc(df):
    """
    @brief 데이터 전처리 함수로, 이상치 처리, 특성 및 레이블 분리, 데이터셋 분할, 스케일링을 수행
    
    @param df(pd.DataFrame): 원본 데이터프레임
    
    @return x_train(np.array): 학습 데이터
    @return x_test(np.array): 테스트 데이터
    @return y_train(pd.Series): 학습 데이터 레이블
    @return y_test(pd.Series): 테스트 데이터 레이블
    """
    global column
    # 이상치 처리 - 값이 0인 항목들을 각 특성의 Median 값으로 대체
    df.replace({'Glucose': {0: df['Glucose'].median()},
                'BloodPressure': {0: df['BloodPressure'].median()},
                'SkinThickness': {0: df['SkinThickness'].median()},
                'Insulin': {0: df['Insulin'].median()},
                'BMI': {0: df['BMI'].median()}}, inplace=True)

    # 처리 결과 확인
    print("Processed Dataset\n", df.describe(), "\n")

    # 특성과 레이블 분리
    x = df.drop("Outcome", axis=1)
    column = x.columns
    y = df["Outcome"]

    # 데이터셋 분할 - 학습:테스트 = 7:3
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # 데이터 스케일링 - Standard Scaler는 특성 값을 평균 0, 분산 1로 조정함
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test

def visualize(model):
    """
    @brief 모델의 특성 중요도를 시각화하는 함수
    
    @param model(sklearn.ensemble.RandomForestClassifier): 학습된 모델
    """
    global column

    # 특성 중요도 추출 및 정렬 - 클래스 분류에 있어 어떤 특성이 얼마나 영향을 미쳤는지 확인
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # 특성 중요도 시각화
    plt.figure(figsize=(9,6))
    plt.title("Feature Importances")
    plt.bar(range(len(column)), importances[indices], align="center")
    plt.xticks(range(len(column)), [column[i] for i in indices], rotation=15, ha="right")
    plt.savefig("feature.png")