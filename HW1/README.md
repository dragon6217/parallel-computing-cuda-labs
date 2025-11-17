# HW1: 1D 필터 병렬화 (std::thread)

## 1. 과제 목표

단일 스레드(Serial) 환경에서 약 1.3~1.4초가 소요되는 1D Convolution 필터 연산을 **C++ `std::thread`**를 활용하여 1.0초 미만으로 성능을 개선합니다. (OpenMP 사용 금지)

## 2. 문제 분석: 1D Convolution

제공된 시리얼 코드는 `k = {0.25, 0.5, 0.25}` 커널을 사용하는 1D Convolution 필터입니다.

`output[i] = (input[i] * 0.25) + (input[i+1] * 0.5) + (input[i+2] * 0.25)`

이 연산의 핵심 특징은 각 `output[i]`의 계산이 다른 `output[j]`의 계산과 **데이터 종속성이 없는(independent)** 완벽한 **데이터 병렬(Data-Parallel)** 문제라는 점입니다. 따라서 별도의 Lock이나 동기화 없이 전체 작업을 나누어 처리할 수 있습니다.

## 3. 병렬화 전략: 정적 분할 (Static Partitioning)

1.  **작업 분할:** 총 `N-2`개의 독립적인 계산 작업을 `NT`개(과제 기본값: 32)의 스레드에게 균등하게 분배합니다.
2.  **작업자 함수:** `filter_worker` 함수를 정의하여, 각 스레드가 자신에게 할당된 범위(`start_index` ~ `end_index`)의 `output` 값만 계산하도록 합니다.
3.  **스레드 관리:** `std::vector<std::thread>`를 사용하여 `NT`개의 스레드를 생성하고, 메인 스레드는 `join()`을 호출하여 모든 작업자 스레드가 완료될 때까지 대기합니다.

## 4. 결과

과제 `README.md`에 명시된 샘플 출력을 기준으로, 성공적으로 성능 목표를 달성했습니다.

* **Serial (Before):** 1.337 sec
* **Parallel (After):** 0.433 sec
* **Speedup:** **약 3.09배** (목표 1.0초 미만 달성, 보너스 0.5초 미만 달성)

## 5. 핵심 학습 내용 (Key Takeaways)

* C++ `std::thread`를 사용한 기본적인 멀티스레딩 (스레드 생성, 인자 전달, `join` 대기) 방법을 익혔습니다.
* 데이터 병렬(Data-Parallel) 문제의 특성을 이해하고, 가장 기본적인 **정적 작업 분할(Static Partitioning)** 전략을 구현했습니다.
* 스레드 간 동기화가 필요 없는(Lock-Free) 문제의 경우, 스레드 생성 오버헤드를 제외하면 스레드 수에 비례하는 성능 향상(Linear Speedup)에 근접할 수 있음을 확인했습니다.