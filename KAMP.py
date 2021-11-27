import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import seaborn as sns


# 이상치 측정 함수
def outlier(x):
    Q1 = x.quantile(1/4)
    Q3 = x.quantile(3/4)
    IQR = Q3 - Q1
    LL = Q1 - (1.5*IQR)
    UU = Q3 + (1.5*IQR)
    outlier = (x < LL) | (x > UU)
    return outlier

# 디렉토리 설정
data_dir = r"./injectionmoding_dataset/injectionmoding_dataset_unlabeled.csv"
init_df = pd.read_csv(data_dir)

# 디스플레이 옵션 설정
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.unicode.east_asian_width', True)

# 필요없는 데이터 제거
drop_df = init_df.drop_duplicates()
drop_df.drop("_id", axis=1, inplace=True)
drop_df.drop("Unnamed: 0", axis=1, inplace=True)
drop_df.drop("PART_FACT_PLAN_DATE", axis=1, inplace=True)
drop_df.drop(["Mold_Temperature_1", "Mold_Temperature_2", "Mold_Temperature_5", "Mold_Temperature_6", "Mold_Temperature_7", "Mold_Temperature_8", "Mold_Temperature_9", "Mold_Temperature_10","Mold_Temperature_11","Mold_Temperature_12",], axis=1, inplace=True)

# Datetime으로 변경
drop_df["TimeStamp"] = pd.to_datetime(drop_df['TimeStamp'], yearfirst=True)
# 타임테이블 인덱스 변경
# drop_result.set_index("TimeStamp", inplace=True)

# CN7_우진650톤 데이터 추출
CN7_filter = drop_df["PART_NAME"].str.contains("CN7")
Eq650_filter = drop_df["EQUIP_NAME"].str.contains("650")
CN7_650T_df = drop_df[CN7_filter & Eq650_filter]

# RH, LH 데이터를 더미 변수로 변환
label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

onehot_type = label_encoder.fit_transform(CN7_650T_df["PART_NAME"])
CN7_650T_df["PART_TYPE"] = onehot_type

# 데이터 분석 후 1차 가공
CN7_650T_1st_refine = CN7_650T_df.drop(["PART_FACT_SERIAL", "PART_NO", "PART_NAME", "EQUIP_CD", "EQUIP_NAME"], axis=1)

# 상관관계 시각화
plt.figure(figsize=(30, 30))
sns.heatmap(data=CN7_650T_1st_refine[CN7_650T_1st_refine["PART_TYPE"].isin([1])].corr(), annot=True, fmt='.2f', linewidths=.5, cmap='Blues')

# 히스토그램 시각화
visual_CN7_650T_1st_refine = CN7_650T_1st_refine.drop("TimeStamp", axis=1)
for index, value in enumerate(visual_CN7_650T_1st_refine):
    sub = plt.subplot(6, 5, index+1)
    sub.hist(visual_CN7_650T_1st_refine[value], facecolor=(50/255, 125/255, 200/255), linewidth=3, edgecolor='black')
    plt.title(value)

# 데이터 특성 상 RH, LH가 조금 다르므로 데이터 분리
CN7_650T_LH = CN7_650T_1st_refine[CN7_650T_1st_refine["PART_TYPE"].isin([0])]
CN7_650T_RH = CN7_650T_1st_refine[CN7_650T_1st_refine["PART_TYPE"].isin([1])]
CN7_650T_LH.drop("PART_TYPE", axis=1, inplace=True)
CN7_650T_RH.drop("PART_TYPE", axis=1, inplace=True)

# 잡음제거 오토인코더는 정상데이터로 학습시키므로 ERR기준으로 데이터 분리
CN7_650T_LH = CN7_650T_LH[CN7_650T_LH["ERR_FACT_QTY"] <= 10]
CN7_650T_RH = CN7_650T_RH[CN7_650T_RH["ERR_FACT_QTY"] <= 10]

# 학습을 위해 Time데이터 제거
CN7_650T_LH.drop("TimeStamp", axis=1, inplace=True)
CN7_650T_RH.drop("TimeStamp", axis=1, inplace=True)

# 학습을 위한 스케일링
scaler = preprocessing.MinMaxScaler()
CN7_650T_LH = scaler.fit_transform(CN7_650T_LH)
CN7_650T_RH = scaler.fit_transform(CN7_650T_RH)

# # 학습, 평가데이터 분리
# CN7_650T_LH_train, CN7_650T_LH_validation = train_test_split(CN7_650T_LH, train_size=0.8, test_size=0.2)
# CN7_650T_RH_train, CN7_650T_RH_validation = train_test_split(CN7_650T_RH, train_size=0.8, test_size=0.2)

def denoising_autoencoder(data):
    # Encoder
    dropout_encoder = Sequential([Dropout(0.3), Dense(20, activation="swish"), Dense(10, activation="swish"), Dense(3, activation="swish")])
    # Decoder
    dropout_decoder = Sequential([Dense(10, activation="swish"), Dense(20, activation="swish"), Dense(data.shape[1], activation="swish")])
    DAE = Sequential([dropout_encoder, dropout_decoder])
    DAE.compile(loss='mse', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

    return DAE


# 손실함수 옵티마이저 정의
DAE_LH = denoising_autoencoder(CN7_650T_LH)
DAE_RH = denoising_autoencoder(CN7_650T_RH)

# 모델 훈련
history_LH = DAE_LH.fit(CN7_650T_LH, CN7_650T_LH, batch_size=30, epochs=30, validation_split=0.2, callbacks=[EarlyStopping(monitor="val_loss", patience=7, mode='min')])
history_RH = DAE_RH.fit(CN7_650T_RH, CN7_650T_RH, batch_size=30, epochs=30, validation_split=0.2, callbacks=[EarlyStopping(monitor="val_loss", patience=7, mode='min')])


def training_visualize(model_his):
    sub = plt.subplot(2, 1, 1)
    # 훈련 결과 시각화
    sub.plot(model_his.history["loss"], label="Training Loss")
    sub.plot(model_his.history["val_loss"], label="Validation Loss")
    sub.legend()
    sub = plt.subplot(2, 1, 2)
    sub.plot(model_his.history["accuracy"], label="Training Acc")
    sub.plot(model_his.history["val_accuracy"], label="Validation Acc")
    sub.legend()


#
def defective_decision(DAE, train_data, test_data):
    # 학습 데이터 예측값
    pred = DAE.predict(train_data)
    # 학습 데이터 복원 오차
    train_loss = np.mean(np.square(pred-train_data), axis=1)
    #임계치 (3시그마-99.7%)
    threshold = np.mean(train_loss) + 3*np.std(train_loss)

    # 평가 데이터
    # 예측값
    pred_def = DAE.predict(test_data)
    # 복원오차
    test_loss = np.mean(np.square(pred_def - test_data), axis=1)

    defect = test_loss > threshold
    print("불량 개수: ", np.sum(defect))
    return defect