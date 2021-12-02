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
unlabeled_data_dir = r"./injectionmoding_dataset/injectionmoding_dataset_unlabeled.csv"
unlabeled_df = pd.read_csv(unlabeled_data_dir)

labeled_data_dir = r"./injectionmoding_dataset/경진대회용 사출성형기 데이터셋_labeled.csv"
labeled_df = pd.read_csv(labeled_data_dir)

# 디스플레이 옵션 설정
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.unicode.east_asian_width', True)


def useless_data_drop(dataframe):
    # 중복 데이터 제거
    dataframe = dataframe.drop_duplicates()
    # 1차 분석 후 불필요 데이터 제거
    dataframe.drop("_id", axis=1, inplace=True)
    dataframe.drop("PART_FACT_PLAN_DATE", axis=1, inplace=True)
    # 분석모델에 쓸모 없는 데이터 제거
    for column, value in dataframe.std().iteritems():
        if value == 0:
            dataframe.drop(f"{column}", axis=1, inplace=True)
    # Datetime으로 변경
    dataframe["TimeStamp"] = pd.to_datetime(dataframe['TimeStamp'], yearfirst=True)
    return dataframe

# 필요없는 데이터 제거
unlabeled_df = useless_data_drop(unlabeled_df)
unlabeled_df.drop("Unnamed: 0", axis=1, inplace=True)

labeled_df = useless_data_drop(labeled_df)

# 타임테이블 인덱스 변경
# drop_result.set_index("TimeStamp", inplace=True)

# 장비, 제품 별 분류 함수
def div_pa_eq(dataframe, part_name, equip_name):
    part_name_filter = dataframe["PART_NAME"].str.contains(part_name)
    equip_name_filter = dataframe["EQUIP_NAME"].str.contains(equip_name)

    return dataframe[part_name_filter & equip_name_filter]

# CN7_우진650톤 데이터 추출
CN7_filter = unlabeled_df["PART_NAME"].str.contains("CN7")
Eq650_filter = unlabeled_df["EQUIP_NAME"].str.contains("650")
ERR_filter = unlabeled_df["ERR_FACT_QTY"] <= 5  # 잡음제거 오토인코더는 정상데이터로 학습시키므로 ERR기준으로 데이터 분리
SOP_filter = unlabeled_df["Switch_Over_Position"] == 0  # Switch_Over_Position 0인지 아닌지에 따라 공정이 달라지는듯

CN7_650T_df = unlabeled_df[CN7_filter & Eq650_filter & ERR_filter & SOP_filter]
unlabeled_CN7_650T_train = CN7_650T_df.drop(["PART_FACT_SERIAL", "PART_NO", "PART_NAME", "EQUIP_CD", "EQUIP_NAME", "ERR_FACT_QTY", "TimeStamp", ], axis=1)

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

# 잡음제거 오토인코더는 정상데이터로 학습시키므로 ERR기준으로 데이터 분리
CN7_650T_1st_refine = CN7_650T_1st_refine[CN7_650T_1st_refine["ERR_FACT_QTY"] <= 10]
CN7_650T_1st_refine.drop("ERR_FACT_QTY", axis=1, inplace=True)
# 학습을 위해 Time데이터 제거
CN7_650T_1st_refine.drop("TimeStamp", axis=1, inplace=True)

# 데이터 특성 상 RH, LH가 조금 다르므로 데이터 분리
CN7_650T_LH = CN7_650T_1st_refine[CN7_650T_1st_refine["PART_TYPE"].isin([0])]
CN7_650T_RH = CN7_650T_1st_refine[CN7_650T_1st_refine["PART_TYPE"].isin([1])]
CN7_650T_LH.drop("PART_TYPE", axis=1, inplace=True)
CN7_650T_RH.drop("PART_TYPE", axis=1, inplace=True)

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
    # 손실함수 옵티마이저 정의
    DAE.compile(loss='mse', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    return DAE

# 각 데이터에 DAE 적용
DAE_LH = denoising_autoencoder(CN7_650T_LH)
DAE_RH = denoising_autoencoder(CN7_650T_RH)

# 모델 훈련
# history_LH = DAE_LH.fit(CN7_650T_LH, CN7_650T_LH, batch_size=30, epochs=30, validation_split=0.2, callbacks=[EarlyStopping(monitor="val_loss", patience=7, mode='min')])
# history_RH = DAE_RH.fit(CN7_650T_RH, CN7_650T_RH, batch_size=30, epochs=30, validation_split=0.2, callbacks=[EarlyStopping(monitor="val_loss", patience=7, mode='min')])
history_LH = DAE_LH.fit(CN7_650T_LH, CN7_650T_LH, batch_size=100, epochs=500, validation_split=0.2)
history_RH = DAE_RH.fit(CN7_650T_RH, CN7_650T_RH, batch_size=100, epochs=500, validation_split=0.2)


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

# 레이블 데이터 전 처리
test_df = div_pa_eq(labeled_df, "CN7", "650톤-우진2호기")
test_df = test_df.drop(["TimeStamp", "PART_FACT_SERIAL", "EQUIP_CD", "EQUIP_NAME", "Reason"], axis=1)

test_LH = test_df[test_df["PART_NAME"].str.contains("LH")]
test_LH.drop("PART_NAME", axis=1, inplace=True)
test_LH_Y = test_LH[test_LH["PassOrFail"].str.contains("Y")]
test_LH_Y.drop("PassOrFail", axis=1, inplace=True)
test_LH_N = test_LH[test_LH["PassOrFail"].str.contains("N")]
test_LH_N.drop("PassOrFail", axis=1, inplace=True)

test_RH = test_df[test_df["PART_NAME"].str.contains("RH")]
test_RH.drop("PART_NAME", axis=1, inplace=True)
test_RH_Y = test_RH[test_RH["PassOrFail"].str.contains("Y")]
test_RH_Y.drop("PassOrFail", axis=1, inplace=True)
test_RH_N = test_RH[test_RH["PassOrFail"].str.contains("N")]
test_RH_N.drop("PassOrFail", axis=1, inplace=True)

test_LH_Y = scaler.fit_transform(test_LH_Y)
test_LH_N = scaler.fit_transform(test_LH_N)

test_LH_Y = scaler.fit_transform(test_RH_Y)
test_RH_N = scaler.fit_transform(test_RH_N)