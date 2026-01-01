# K-League AI Competition - Track1: 좌표 예측

K리그-서울시립대 공개 AI 경진대회 Track1 알고리즘 부문: 축구 경기 이벤트 데이터를 활용한 최종 좌표 예측

## 프로젝트 개요

축구 경기 중 발생한 이벤트 시퀀스를 분석하여 마지막 이벤트의 종료 좌표(end_x, end_y)를 예측하는 프로젝트입니다.

## 접근 방법

### 1. 피처 엔지니어링
- **마지막 K개 이벤트 사용**: 각 episode의 마지막 20개 이벤트만 사용 (K=20)
- **Wide Format 변환**: 시계열 데이터를 넓은 형태(wide table)로 변환
- **시간/공간 피처**:
  - 시간 차이 (dt)
  - 이동량 (dx, dy)
  - 거리 (dist)
  - 속도 (speed)
  - 구역 분류 (x_zone, lane)
- **카테고리 인코딩**: type_name, result_name, team_id 등을 숫자로 변환

### 2. 모델링
- **AutoGluon TabularPredictor** 사용
- X, Y 좌표 각각 별도 모델 학습
- `best_quality` 프리셋으로 자동 앙상블
- 사용된 모델: CatBoost, LightGBM, XGBoost, Extra Trees, Random Forest 등

## 파일 구조

```
.
├── check1.ipynb                          # 메인 학습 및 추론 코드 (AutoGluon)
├── [Baseline,_Pytorch] LSTM 기반 학습 및 추론.ipynb  # LSTM 베이스라인
├── k_league의_좌표예측분석 코드.ipynb    # EDA 및 분석 코드
├── requirements.txt                      # 필요한 패키지 목록
└── README.md                            # 이 파일
```

## 사용 방법

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 데이터 준비
- `open_track1/` 폴더에 데이터 파일 배치
  - `train.csv`
  - `test.csv`
  - `match_info.csv`
  - `sample_submission.csv`
  - `test/` 폴더 (테스트 episode 파일들)

### 3. 학습 및 추론
`check1.ipynb` 노트북을 실행하세요.

주요 설정:
- `K = 20`: 마지막 K개 이벤트 사용
- `time_limit = 1800`: 학습 시간 제한 (30분)
- `presets = "best_quality"`: 최고 품질 모드

### 4. 결과 확인
두 번째 셀을 실행하여 모델 성능 및 제출 파일을 확인할 수 있습니다.

## 결과

- **제출 파일**: `submission_autogluon_lastK.csv`
- **모델 저장 경로**: `ag_models_x/`, `ag_models_y/`

## 주요 특징

1. **피처 직접 가공**: 시간/공간 피처를 직접 생성하여 모델에 제공
2. **AutoGluon 활용**: 복잡한 하이퍼파라미터 튜닝 없이 자동으로 최적 모델 선택 및 앙상블
3. **Wide Format**: 시계열을 넓은 형태로 변환하여 Tabular 모델에 적합하게 구성

## 참고사항

- 대용량 파일(모델, 데이터)은 `.gitignore`에 포함되어 저장소에 업로드되지 않습니다.
- 학습된 모델은 `ag_models_x/`, `ag_models_y/` 폴더에 저장됩니다.






