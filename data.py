"""
    Author: Alan Park (alanworks72@gmail.com)
    Date: Jul.09.2024
    Modified: Jul.10.2024
    File Name: data.py
    Version: 1.1
    Description: Data loader for Diabetes classifier
"""

import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn

from proc import preProc


""" FEATURE INFO
        Pregnancies: Number of times pregnant
        Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
        BloodPressure: Diastolic blood pressure (mm Hg)
        SkinThickness: Triceps skin fold thickness (mm)
        Insulin: 2-Hour serum insulin (mu U/ml)
        BMI: Body mass index (weight in kg/(height in m)^2)
        DiabetesPedigreeFunction: Diabetes pedigree function
        Age: Age (years)
        Outcome: Class variable (0 or 1)
"""

def dataloader(path):
    """
    @brief CSV 파일을 로드하고 전처리하는 함수
    
    @param path(str): CSV 파일 경로
    
    @return preProc(df): 전처리된 데이터프레임
    """
    # 데이터 불러오기 - dataframe 형식으로 불러와 데이터 처리에 용이
    df = pd.read_csv(path)

    # 데이터 통계
    print("Analysing Dataset\n", df.describe(), "\n")

    # 결측치 확인
    print("Null Check\n",df.isnull().sum(), "\n")
    print("NaN Check\n",df.isna().sum(), "\n")

    # 특성 간 상관관계 시각화
    plt.figure(figsize=(12, 10))
    seaborn.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.yticks(rotation=45, ha="right")
    plt.xticks(rotation=30, ha="right")
    plt.savefig("./corr.png")

    # 전처리 함수 호출 및 결과 반환
    return preProc(df)