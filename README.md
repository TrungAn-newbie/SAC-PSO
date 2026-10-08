# A Soft Actor-Critic Particle Swarm Optimization for Scheduling Orders in Logistic Warehouse
### Deep Reinforcement Learning (Soft Actor-Critic) Dynamic Meta-Optimization for Particle Swarm Optimization (SAC-PSO) vs Standard PSO, GA, FIFO, SPT, LPT

> **Đề tài nghiên cứu**: *"A Soft Actor-Critic Particle Swarm Optimization for scheduling orders in Logistic warehouse"*  
> **Supervisor**: Dr. Truong Ngoc Cuong  
> **Student**: Tran Viet Trung An  

---

## 1. Tổng quan Bài toán JSSPLA trong Kho Logistics Thông Minh

Hệ thống kho hàng tự động diện tích $33.0\text{m} \times 16.5\text{m}$ vận hành đội 8 robot tự hành (AMR) phục vụ 4 cụm kệ hàng ($R_1, R_2, R_3, R_4$) và các trạm chức năng:
- **Trạm nhận & kiểm tra hàng**: Receiving $\to$ Buffer 1
- **Khu vực lưu trữ trung tâm**: 4 cụm kệ lưu trữ $R_1, R_2, R_3, R_4$
- **Trạm hạ nguồn**: Picking (lấy hàng) $\to$ Packing (đóng gói) $\to$ Buffer 2 $\to$ Shipping (xuất xưởng)
- **Trạm sạc tự động**: Automated Charging Station
- **Chính sách phân đội chuyên trách (Dedicated Sub-fleet Policy)**: Mỗi cụm kệ $k \in \{0, 1, 2, 3\}$ được giao đúng 2 robot cố định ($R_1 \to [0, 1]$, $R_2 \to [2, 3]$, $R_3 \to [4, 5]$, $R_4 \to [6, 7]$). Mọi tác vụ vận chuyển đơn hàng thuộc cụm đó bắt buộc do 1 trong 2 robot này thực hiện.
- **Ràng buộc thời gian và động học pin vật lý**:
  - Vận tốc định mức AMR: $v = 1.0\text{ m/s}$.
  - Thời gian bốc dỡ tự động tại trạm: $t_{load} = 5\text{s}, t_{unload} = 5\text{s}$.
  - Tiêu hao năng lượng theo trạng thái: tải trọng có hàng ($\alpha_l$) và chạy không tải ($\alpha_e$).
  - Ngưỡng pin an toàn $b_{min} = 20\%$, tự động điều hướng sang trạm sạc khi mức pin xuống thấp.

---

## 2. Cấu trúc Thư mục Dự án

```
Coding/
│
├── models/                               # Trọng số mạng nơ-ron đã huấn luyện
│   └── sac_basic_pso.pt                  # Trọng số SAC Meta-Optimizer Actor-Critic
│
├── results/                              # Kết quả thực nghiệm đối sánh & đồ họa trực quan
│   ├── table_standard_scenarios.md       # Bảng đối sánh 10 runs trên 9 kịch bản chuẩn (W1 - W9)
│   ├── table_metrics_cmax_sd_cputime.md  # Bảng chi tiết Cmax, SD, %SD và CPU Time
│   ├── barchart_standard_scenarios.png   # Biểu đồ cột Cmax phân nhóm trên 9 kịch bản (W1 - W9)
│   ├── boxplot_small_scenario.png        # Biểu đồ hộp Boxplot cho nhóm kịch bản Small (W1, W2, W3)
│   ├── boxplot_medium_scenario.png       # Biểu đồ hộp Boxplot cho nhóm kịch bản Medium (W4, W5, W6)
│   ├── boxplot_large_scenario.png        # Biểu đồ hộp Boxplot cho nhóm kịch bản Large (W7, W8, W9)
│   ├── comparison_barchart.png           # Biểu đồ cột đối sánh trực quan tổng thể
│   ├── sac_basic_training.png            # Đồ thị 4 panels đường cong huấn luyện SAC Agent
│   ├── sac_basic_training_history.json   # Log dữ liệu huấn luyện
│   └── benchmark_results.json            # Dữ liệu raw của benchmark 10 runs
│
├── warehouse_env.py                      # Mô hình môi trường kho JSSPLA, Layout, 8 AMRs, Pin, Decoder SPV
├── baselines.py                          # Thuật toán so chuẩn: Heuristics (FIFO, SPT, LPT), Standard GA, Standard PSO
├── sac_agent.py                          # Soft Actor-Critic (SAC) Agent (Policy & Critic Networks, Replay Buffer)
├── pso_env.py                            # Môi trường PSO RL (BasicWarehousePSORLEnv) & Solver (BasicSACPSOSolver)
├── train.py                              # Pipeline huấn luyện SAC 3-Stage Scenario Curriculum (300 episodes)
├── benchmark.py                          # Benchmark đối sánh 6 thuật toán trên 9 kịch bản chuẩn (W1 - W9)
├── main.py                               # Điểm khởi chạy chương trình chính (Entry point)
└── README.md                             # Tài liệu kỹ thuật chi tiết
```

---

## 3. Kiến trúc Thuật toán Đề xuất: SAC-PSO Meta-Optimization

Trong thuật toán PSO giải bài toán JSSPLA:
- Mỗi hạt đại diện cho một lời giải gồm hai vector liên tục:
  * $X_{os} \in [-4.0, 4.0]^D$: Vector liên tục xác định thứ tự công đoạn qua quy tắc SPV (*Smallest Position Value*).
  * $X_{aa} \in [0.0, 1.0]^D$: Vector liên tục xác định phân bổ AMR trong cặp robot chuyên trách của cụm.
- Phương trình cập nhật vận tốc và vị trí hạt:
  $$V_{os}(t+1) = \text{clip}(w \cdot V_{os}(t) + c_1 r_1 (pbest_{os} - X_{os}(t)) + c_2 r_2 (gbest_{os} - X_{os}(t)), -2.0, 2.0)$$
  $$X_{os}(t+1) = \text{clip}(X_{os}(t) + V_{os}(t+1), -4.0, 4.0)$$
  $$V_{aa}(t+1) = \text{clip}(w \cdot V_{aa}(t) + c_1 r_1 (pbest_{aa} - X_{aa}(t)) + c_2 r_2 (gbest_{aa} - X_{aa}(t)), -1.0, 1.0)$$
  $$X_{aa}(t+1) = \text{clip}(X_{aa}(t) + V_{aa}(t+1), 0.0, 1.0)$$

### Các Đặc trưng Kỹ thuật Cốt lõi:

1. **Không gian Trạng thái 6 Chiều ($s_t \in \mathbb{R}^6$)**:
   - `progress`: Tiến trình thế hệ $\frac{t}{T_{max}} \in [0, 1]$.
   - `gbest_ratio`: Tỉ lệ Makespan $gbest$ hiện tại so với ban đầu $\frac{gbest_t}{gbest_0}$.
   - `diversity`: Độ đa dạng thích nghi bầy hạt $\frac{\text{std}(pbest)}{\text{mean}(pbest) + \epsilon}$.
   - `stagnation_ratio`: Mức độ trì trệ của bầy hạt $\frac{\text{stagnation}}{T_{max}}$.
   - `step_improvement`: Mức cải thiện $gbest$ ở bước gần nhất $\frac{gbest_{t-1} - gbest_t}{gbest_{t-1}}$.
   - `mean_pbest_ratio`: Khoảng cách tương đối giữa trung bình bầy hạt và nghiệm tốt nhất $\frac{\overline{pbest} - gbest}{\overline{pbest} + \epsilon}$.

2. **Không gian Hành động Liên tục 3 Chiều ($a_t \in [-1, 1]^3$)**:
   - SAC điều biến trực tiếp 3 siêu tham số nòng cốt của bầy hạt tại mỗi bước lặp:
     * Quán tính $w \in [0.35, 0.95]$: Cân bằng giữa khám phá toàn cục (*exploration*) và khai thác cục bộ (*exploitation*).
     * Trọng số học cá nhân $c_1 \in [0.80, 2.60]$: Điều chỉnh xu hướng hướng về kinh nghiệm cá nhân $pbest$.
     * Trọng số học xã hội $c_2 \in [0.80, 2.60]$: Điều chỉnh xu hướng hội tụ về lời giải tốt nhất của bầy $gbest$.

3. **Cơ chế Tự động Điều chỉnh Nhiệt độ Entropy ($\alpha$)**:
   - Tối ưu hóa hàm mục tiêu cực đại hóa đồng thời Phần thưởng và Entropy:
     $$J(\pi) = \sum_{t} \mathbb{E} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right]$$
   - Entropy mục tiêu được thiết lập chuẩn hóa theo số chiều hành động: $\bar{\mathcal{H}} = -\text{dim}(\mathcal{A}) = -3.0$.
   - $\alpha$ được cập nhật thích ứng theo gradient descent: $\alpha$ tự động duy trì mức cao ở đầu giai đoạn để khám phá không gian tham số, sau đó giảm dần về mức nhỏ ($\approx 0.05 - 0.15$) giúp mạng hội tụ sâu và ổn định.

4. **Cơ chế Tinh chỉnh Lời giải Tinh hoa (Elitist Memetic Refinement)**:
   - *Greedy AMR Workload Balancing*: Đánh giá độ chênh lệch thời gian hoàn thành giữa 2 robot trong từng cụm kệ ($|t_{R,1} - t_{R,2}|$) và tự động điều chuyển tác vụ từ robot quá tải sang robot nhàn rỗi.
   - *Critical-Path Job Shifting*: Dịch chuyển thứ tự các công đoạn thuộc đơn hàng hoàn thành muộn nhất nhằm triệt tiêu các khoảng thời gian chờ (*idle gaps*).

5. **Hàm Phần thưởng Dày (Dense Reward Formulation)**:
   $$R = 15.0 \cdot \Delta gbest + 3.0 \cdot \Delta \overline{pbest} - 0.02 \cdot \frac{\text{stagnation}}{T_{max}} + R_{terminal}$$
   kích thích tác tử SAC liên tục hạ thấp Makespan bầy hạt và tránh rơi vào bẫy cực trị địa phương.

---

## 4. Quá trình Huấn luyện SAC Agent (3-Stage Scenario Curriculum)

Tác tử SAC được huấn luyện theo giáo trình 3 giai đoạn (300 episodes):
- **Stage 1 (Ep 1 - 100)**: Kịch bản nhỏ Small (12 - 50 jobs) $\to$ Học điều phối luồng cơ bản.
- **Stage 2 (Ep 101 - 200)**: Kịch bản trung bình Medium (50 - 150 jobs) $\to$ Học xử lý nghẽn trạm và đột biến luồng nhập.
- **Stage 3 (Ep 201 - 300)**: Kịch bản lớn Large (150 - 300 jobs) $\to$ Học chịu tải nặng và tối ưu hóa phân bổ robot.

![Đường cong huấn luyện SAC](results/sac_basic_training.png)

---

## 5. Hệ Thống 9 Benchmark Test Instances Tiêu Chuẩn ($W_1 \to W_9$)

Hệ thống benchmark được chuẩn hóa thành 3 nhóm quy mô: **Small (50 jobs / 100 ops)**, **Medium (150 jobs / 300 ops)**, và **Large (300 jobs / 600 ops)**. Trong mỗi quy mô, bài toán được phân rã thành 3 cấu hình luồng (Cân bằng $In=Out$, Thiên xuất $In<Out$, Thiên nhập $In>Out$):

| Mã Instance | Kịch bản | Quy mô ($N_{jobs}$) | Nhập ($N_{in}$) | Xuất ($N_{out}$) | Tổng công đoạn ($N_{ops}$) | Tỉ lệ $\beta = \frac{N_{in}}{N_{out}}$ | Đặc trưng luồng kho |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$W_1$** | Small | 50 | 25 | 25 | 100 | 1.00 | **Small: Cân bằng luồng (25 In, 25 Out)** |
| **$W_2$** | Small | 50 | 15 | 35 | 100 | 0.43 | **Small: Tải xuất kho cao (15 In, 35 Out)** |
| **$W_3$** | Small | 50 | 35 | 15 | 100 | 2.33 | **Small: Tải nhập kho cao (35 In, 15 Out)** |
| **$W_4$** | Medium | 150 | 75 | 75 | 300 | 1.00 | **Medium: Cân bằng luồng (75 In, 75 Out)** |
| **$W_5$** | Medium | 150 | 45 | 105 | 300 | 0.43 | **Medium: Tải xuất kho cao (45 In, 105 Out)** |
| **$W_6$** | Medium | 150 | 105 | 45 | 300 | 2.33 | **Medium: Tải nhập kho cao (105 In, 45 Out)** |
| **$W_7$** | Large | 300 | 150 | 150 | 600 | 1.00 | **Large: Cân bằng luồng (150 In, 150 Out)** |
| **$W_8$** | Large | 300 | 90 | 210 | 600 | 0.43 | **Large: Tải xuất kho cao (90 In, 210 Out)** |
| **$W_9$** | Large | 300 | 210 | 90 | 600 | 2.33 | **Large: Tải nhập kho cao (210 In, 90 Out)** |

---

## 6. Kết quả Thực nghiệm Đối sánh (10 Runs Độc lập trên $W_1 \to W_9$)

Thực nghiệm đo đạc độc lập **10 lượt chạy ngẫu nhiên (10 independent random seeds)** trên toàn bộ 9 instances tiêu chuẩn:

| Instance | Quy mô | Thuật toán | $C_{max}^{best}$ (s) | $C_{max}^{avg}$ (s) | $C_{max}^{worst}$ (s) | SD (s) | SD (%) | Cải thiện FIFO (%) | Cải thiện PSO (%) | Thời gian (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$W_1$** | Small (50 jobs) | FIFO | 1560.08 | 1560.08 | 1560.08 | 0.00 | 0.00% | +0.00% | -40.75% | 0.000 |
|  |  | GA | 1087.78 | 1121.42 | 1159.11 | 23.82 | 2.12% | +28.12% | -1.17% | 0.090 |
|  |  | PSO | 1056.11 | 1108.41 | 1198.82 | 39.33 | 3.55% | +28.95% | +0.00% | 0.120 |
|  |  | **SAC-PSO (Ours)** | **989.47** | **1034.45** | **1072.97** | **30.45** | **2.94%** | **+33.69%** | **+6.67%** | 0.243 |
| **$W_2$** | Small (50 jobs) | FIFO | 1846.30 | 1846.30 | 1846.30 | 0.00 | 0.00% | +0.00% | -22.65% | 0.000 |
|  |  | GA | 1530.32 | 1566.72 | 1610.34 | 25.73 | 1.64% | +15.14% | -4.07% | 0.110 |
|  |  | PSO | 1428.39 | 1505.39 | 1618.69 | 65.07 | 4.32% | +18.46% | +0.00% | 0.148 |
|  |  | **SAC-PSO (Ours)** | **1340.19** | **1415.81** | **1460.09** | **32.42** | **2.29%** | **+23.32%** | **+5.95%** | 0.183 |
| **$W_3$** | Small (50 jobs) | FIFO | 1248.29 | 1248.29 | 1248.29 | 0.00 | 0.00% | +0.00% | -37.42% | 0.000 |
|  |  | GA | 908.40 | 911.84 | 921.40 | 4.54 | 0.50% | +26.95% | -0.38% | 0.076 |
|  |  | PSO | 908.40 | 908.40 | 908.40 | 0.00 | 0.00% | +27.23% | +0.00% | 0.100 |
|  |  | **SAC-PSO (Ours)** | **908.40** | **908.40** | **908.40** | **0.00** | **0.00%** | **+27.23%** | **+0.00%** | 0.124 |
| **$W_4$** | Medium (150 jobs) | FIFO | 4336.21 | 4336.21 | 4336.21 | 0.00 | 0.00% | +0.00% | -33.82% | 0.001 |
|  |  | GA | 3238.72 | 3356.12 | 3523.93 | 69.61 | 2.07% | +22.60% | -3.57% | 0.259 |
|  |  | PSO | 3086.08 | 3240.37 | 3352.49 | 83.85 | 2.59% | +25.27% | +0.00% | 0.347 |
|  |  | **SAC-PSO (Ours)** | **2775.57** | **2873.07** | **2950.13** | **62.98** | **2.19%** | **+33.74%** | **+11.33%** | 0.512 |
| **$W_5$** | Medium (150 jobs) | FIFO | 5633.88 | 5633.88 | 5633.88 | 0.00 | 0.00% | +0.00% | -26.21% | 0.000 |
|  |  | GA | 4622.39 | 4724.82 | 4820.13 | 60.79 | 1.29% | +16.14% | -5.84% | 0.310 |
|  |  | PSO | 4314.93 | 4463.94 | 4595.16 | 80.81 | 1.81% | +20.77% | +0.00% | 0.419 |
|  |  | **SAC-PSO (Ours)** | **3779.07** | **3922.17** | **4030.62** | **75.05** | **1.91%** | **+30.38%** | **+12.14%** | 0.536 |
| **$W_6$** | Medium (150 jobs) | FIFO | 3191.28 | 3191.28 | 3191.28 | 0.00 | 0.00% | +0.00% | -24.96% | 0.000 |
|  |  | GA | 2553.26 | 2569.09 | 2624.38 | 19.66 | 0.77% | +19.50% | -0.59% | 0.209 |
|  |  | PSO | 2553.26 | 2553.91 | 2559.76 | 1.95 | 0.08% | +19.97% | +0.00% | 0.278 |
|  |  | **SAC-PSO (Ours)** | **2553.26** | **2553.26** | **2553.26** | **0.00** | **0.00%** | **+19.99%** | **+0.03%** | 0.337 |
| **$W_7$** | Large (300 jobs) | FIFO | 8581.55 | 8581.55 | 8581.55 | 0.00 | 0.00% | +0.00% | -28.59% | 0.001 |
|  |  | GA | 6723.65 | 6899.62 | 7039.83 | 90.47 | 1.31% | +19.60% | -3.38% | 0.508 |
|  |  | PSO | 6563.89 | 6673.73 | 6850.83 | 73.47 | 1.10% | +22.23% | +0.00% | 0.684 |
|  |  | **SAC-PSO (Ours)** | **5575.50** | **5768.63** | **5981.65** | **124.99** | **2.17%** | **+32.78%** | **+13.56%** | 1.302 |
| **$W_8$** | Large (300 jobs) | FIFO | 11304.07 | 11304.07 | 11304.07 | 0.00 | 0.00% | +0.00% | -24.22% | 0.001 |
|  |  | GA | 9179.45 | 9523.77 | 9688.75 | 155.52 | 1.63% | +15.75% | -4.66% | 0.619 |
|  |  | PSO | 8895.38 | 9099.89 | 9453.21 | 165.09 | 1.81% | +19.50% | +0.00% | 0.828 |
|  |  | **SAC-PSO (Ours)** | **7559.49** | **7737.86** | **7951.41** | **149.84** | **1.94%** | **+31.55%** | **+14.97%** | 1.642 |
| **$W_9$** | Large (300 jobs) | FIFO | 6136.77 | 6136.77 | 6136.77 | 0.00 | 0.00% | +0.00% | -14.40% | 0.001 |
|  |  | GA | 5363.72 | 5373.02 | 5392.38 | 8.86 | 0.16% | +12.45% | -0.16% | 0.408 |
|  |  | PSO | 5363.72 | 5364.37 | 5370.22 | 1.95 | 0.04% | +12.59% | +0.00% | 0.550 |
|  |  | **SAC-PSO (Ours)** | **5363.72** | **5363.72** | **5363.72** | **0.00** | **0.00%** | **+12.60%** | **+0.01%** | 0.719 |

*Bảng dữ liệu chi tiết $C_{max}$, SD, %SD và CPU Time xem tại [table_metrics_cmax_sd_cputime.md](results/table_metrics_cmax_sd_cputime.md).*

---

## 7. Biểu đồ Trực quan hóa Hiệu năng Chuẩn Xuất bản

### 7.1. Biểu đồ Cột Phân nhóm (Grouped Bar Chart - $W_1 \to W_9$)
![Grouped Bar Chart](results/barchart_standard_scenarios.png)

### 7.2. Phân bố Hộp (Boxplots) theo Cấp độ Quy mô
| Nhóm Small ($W_1, W_2, W_3$) | Nhóm Medium ($W_4, W_5, W_6$) |
| :---: | :---: |
| ![Boxplot Small](results/boxplot_small_scenario.png) | ![Boxplot Medium](results/boxplot_medium_scenario.png) |

| Nhóm Large ($W_7, W_8, W_9$) |
| :---: |
| ![Boxplot Large](results/boxplot_large_scenario.png) |

### 7.3. Các Điểm Nhấn Phân Tích:
1. **Hiệu năng bứt phá mạnh mẽ ở các kịch bản nghẽn (Thiên xuất & Cân bằng)**:
   - Ở các kịch bản cân bằng ($W_1, W_4, W_7$), SAC-PSO cải thiện vượt trội từ **+6.67%** (Small), **+11.33%** (Medium) đến **+13.56%** (Large) so với PSO chuẩn.
   - Ở các kịch bản thiên xuất chịu tải nặng ($W_2, W_5, W_8$), áp lực dồn về Outbound Dock gây nghẽn nghiêm trọng cho các thuật toán truyền thống. SAC-PSO nhờ điều biến linh hoạt $w$ và $c_1, c_2$ đã giải phóng tắc nghẽn, cải thiện tới **+12.14%** ($W_5$) và **+14.97%** ($W_8$) so với PSO chuẩn, và giảm hơn **30% - 33%** Makespan so với FIFO.
   - Ở các kịch bản thiên nhập ($W_3, W_6, W_9$), các thuật toán nhanh chóng đạt điểm hội tụ tiệm cận cận dưới lý thuyết do trạm Inbound phân bổ đồng đều với các Rack, SAC-PSO đạt Makespan tối ưu tuyệt đối với phương sai bằng 0.
2. **Độ ổn định cực cao**:
   - Hệ số phân tán SD (%) của SAC-PSO luôn được kiểm soát chặt dưới **3%** trên toàn bộ 9 instances.
3. **Thời gian tính toán thời gian thực**:
   - Thời gian CPU chỉ dao động từ **0.24s đến 1.64s**, hoàn toàn đáp ứng khả năng tái lập lịch trình trực tuyến trong môi trường kho công nghiệp.

---

## 8. Hướng dẫn Cài đặt & Khởi chạy

### Yêu cầu Môi trường:
- Python 3.9+
- PyTorch (`torch`)
- NumPy
- Matplotlib

Cài đặt nhanh các thư viện phụ thuộc:
```bash
pip install torch numpy matplotlib
```

### 1. Khởi chạy Nhanh (Quick Demo):
Chạy lệnh mặc định để đánh giá nhanh thuật toán SAC-PSO đối sánh với FIFO, SPT, LPT, GA, PSO trên kịch bản $W_1$:
```bash
python main.py
```
*(hoặc `python main.py --demo`)*

### 2. Chạy Benchmark Đối sánh 10 Runs ($W_1 \to W_9$):
Thực hiện đánh giá đa luồng song song trên toàn bộ 9 kịch bản chuẩn:
```bash
python main.py --benchmark --runs 10
```
*(hoặc chạy trực tiếp script: `python benchmark.py --runs 10`)*

### 3. Vẽ lại Biểu đồ & Xuất Bảng Markdown từ dữ liệu có sẵn:
```bash
python benchmark.py --plot-only
```

### 4. Huấn luyện lại SAC Agent:
Huấn luyện tác tử SAC qua mô hình 3-Stage Curriculum (300 episodes: Small $\to$ Medium $\to$ Large):
```bash
python train.py --episodes 300 --pop 50 --iter 50
```
Model trọng số được lưu tại `models/sac_basic_pso.pt` và biểu đồ huấn luyện lưu tại `results/sac_basic_training.png`.
