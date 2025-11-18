# Multicore & GPU Programming Labs

이 저장소는 멀티코어 CPU와 GPU 환경에서 **고성능 병렬 프로그래밍(High Performance Computing)** 기법을 학습하고 구현한 프로젝트 모음입니다.

**C++11**, **OpenMP**, **CUDA**를 사용하여 데이터 병렬성, 동시성 제어, 메모리 계층 최적화, GPGPU 가속 등 다양한 병렬 처리 문제를 해결하고 성능을 극대화했습니다.

## Projects Overview (핵심 성과)

총 6개의 프로젝트를 통해 **Naive 구현 대비 수십 배 이상의 성능 향상**을 달성했습니다.

| Project | 주제 (Topic) | 핵심 기술 (Key Tech) | 성과 (Performance) |
| :--- | :--- | :--- | :--- |
| **[HW1](./HW1)** | **1D Filter Parallelization** | `std::thread`, Data Partitioning | Serial 대비 **4.28배** 가속 (16-thread) |
| **[HW2](./HW2)** | **Concurrent Hash Table** | `std::mutex`, Striped Locking | Global Lock 대비 **13.1배** 가속 |
| **[HW3](./HW3)** | **Matrix Multiplication (CPU)** | `OpenMP`, Cache Tiling (L1) | Naive($O(N^3)$) 대비 **54.7배** 가속 |
| **[HW4](./HW4)** | **Table Join Optimization** | Parallel Sort-Merge Join | 100만 건 Join **0.14초** 달성 ($O(N^2) \to O(N \log N)$) |
| **[HW5](./HW5)** | **Matrix Multiplication (GPU)** | `CUDA`, Shared Memory Tiling | CPU 최적화 버전(HW3) 대비 **41.6배** 추가 가속 |
| **[HW6](./HW6)** | **PageRank (Graph Processing)** | `CUDA`, CSR Format, Pull-based | **6,900만 Edge** 그래프 2.75초 처리 (Laptop GPU) |

---

## Project Details

### 1. [HW1: 1D Image Filter](./HW1)
* **문제:** 대용량 1D 배열의 컨볼루션 연산 속도 저하.
* **해결:** 데이터 의존성이 없는 영역을 분할(Data Decomposition)하여 `std::thread`로 병렬 처리.
* **결과:** 16 스레드 환경에서 정적 작업 분할(Static Partitioning)만으로 4배 이상의 성능 향상 검증.

### 2. [HW2: Thread-safe Hash Table](./HW2)
* **문제:** 멀티 스레드 환경에서 해시 테이블 접근 시 발생하는 경쟁 상태(Race Condition)와 단일 락(Global Lock)의 병목 현상.
* **해결:** **Fine-Grained Locking (Striped Lock)** 기법을 도입하여 락의 범위를 버킷 단위로 최소화.
* **결과:** 락 경합(Contention)을 획기적으로 줄여 13배 이상의 처리량(Throughput) 증대.

### 3. [HW3: Optimized Matrix Multiplication (CPU)](./HW3)
* **문제:** 거대 행렬 곱셈 시 발생하는 심각한 Cache Miss와 메모리 대역폭 병목.
* **해결:** **Loop Tiling (Cache Blocking)** 기법을 적용하여 데이터를 L1 캐시 크기에 맞게 블록화하고, `OpenMP`로 병렬화.
* **결과:** 메모리 접근 효율 극대화로 Naive 알고리즘 대비 54배 성능 향상.

### 4. [HW4: Parallel Table Join](./HW4)
* **문제:** $O(N^2)$ 복잡도를 가진 Nested Loop Join의 비효율성.
* **해결:** 알고리즘을 **Sort-Merge Join** ($O(N \log N)$)으로 변경하고, 정렬 단계에 `__gnu_parallel::sort` (OpenMP 기반) 적용.
* **결과:** 100만 건 데이터 조인을 0.14초 만에 완료.

### 5. [HW5: GPU Matrix Multiplication](./HW5)
* **문제:** CPU의 연산 능력 한계 및 GPU Global Memory(VRAM)의 느린 접근 속도.
* **해결:** GPU의 **Shared Memory (On-chip)**를 활용한 Tiling 기법 구현.
* **결과:** HW3의 CPU 최적화 버전보다 41배 더 빠른 압도적인 연산 성능 증명.

### 6. [HW6: PageRank on GPU](./HW6)
* **문제:** 480만 노드, 6900만 엣지의 대규모 그래프 처리 속도.
* **해결:** **CSR(Compressed Sparse Row)** 포맷을 활용한 메모리 최적화 및 Atomic 연산 없는 **Pull-based** 커널 구현.
* **결과:** 모바일 GPU 환경에서도 대규모 그래프를 안정적이고 빠르게 처리 (Top-5 랭킹 정확도 100%).

---

## Environments & Tools

* **Languages:** C++11, CUDA C++
* **Libraries:** OpenMP, STL, GCC Parallel Extensions
* **Hardware Tested:**
    * **CPU:** AMD Ryzen 7 4800H (8C/16T)
    * **GPU:** NVIDIA GeForce GTX 1660 Ti (6GB)
* **OS:** Linux (Ubuntu 24.04 LTS via WSL2)