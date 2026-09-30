# JSSPLA: Job Shop Scheduling with Limited AMRs in Smart Warehouse
### Deep Reinforcement Learning (Soft Actor-Critic) Meta-Optimization for Standard PSO (SAC-PSO) vs PSO, GA, CDPSO, FIFO, SPT, LPT

> Dựa trên đề tài nghiên cứu: *"Job Shop-Based Scheduling Optimization for Multi-AMR Warehouse Systems and Validation via Gazebo Simulation"*  
> **Supervisor**: Dr. Truong Ngoc Cuong | **Student**: Tran Viet Trung An (MAIL LAB)

---

## 1. Tổng quan Bài toán JSSPLA trong Kho Thông Minh

Hệ thống kho hàng diện tích $33.0\text{m} \times 16.5\text{m}$ vận hành đội 8 robot tự hành (AMR) phục vụ 4 cụm kệ hàng ($R_1, R_2, R_3, R_4$) và các trạm chức năng:
- **Trạm nhận & kiểm tra hàng**: Receiving $\to$ Buffer 1
- **Khu vực lưu trữ trung tâm**: 4 cụm kệ $R_1, R_2, R_3, R_4$
- **Trạm hạ nguồn**: Picking (lấy hàng) $\to$ Packing (đóng gói) $\to$ Buffer 2 $\to$ Shipping (xuất xưởng)
- **Trạm sạc tự động**: Automated Charging Station
- **Chính sách phân đội chuyên trách (Dedicated Sub-fleet Policy)**: Mỗi cụm kệ $k \in \{1, 2, 3, 4\}$ được giao đúng 2 robot cố định ($R_1 \to [0, 1]$, $R_2 \to [2, 3]$, $R_3 \to [4, 5]$, $R_4 \to [6, 7]$). Mọi tác vụ vận chuyển đơn hàng thuộc cụm đó bắt buộc do 1 trong 2 robot này thực hiện.
- **Ràng buộc thời gian và năng lượng vật lý**: Vận tốc AMR ($v=1.0\text{m/s}$), thời gian bốc dỡ tự động ($t_{load}=10\text{s}, t_{unload}=10\text{s}$), tốc độ tiêu hao pin khi chở hàng $\alpha_l$ và khi chạy không tải $\alpha_e$, ngưỡng pin an toàn $b_{min}=20\%$, và rẽ nhánh vào trạm sạc tự động khi pin yếu.

---

## 2. Cấu trúc Thư mục Dự án

```
Coding/
│
├── models/                           # Trọng số mạng nơ-ron đã huấn luyện
│   └── sac_warehouse_pso.pt          # Model SAC meta-optimizer cho Standard PSO
│
├── results/                          # Biểu đồ và bảng kết quả thực nghiệm
│   ├── table1_comparison.md          # Bảng đối sánh chi tiết theo Table 1 trong bài báo
│   ├── boxplots_comparison.png       # Biểu đồ hộp (Box plots) tương ứng Figures 3, 4, 5, 6 trong bài báo
│   ├── scenario4_convergence.png     # Đồ thị hội tụ trên Scenario 4 (350 Jobs / 750 Ops)
│   └── sac_pso_training.png          # Đồ thị quá trình huấn luyện SAC trên 30+ warehouse instances
│
├── warehouse_env.py                  # Mô hình kho vật lý JSSPLA, Layout, 8 AMRs, Battery, Decoder
├── baselines.py                      # Cài đặt chuẩn: FIFO, SPT, LPT, Standard GA, Standard PSO, CDPSO (Paper)
├── sac_pso.py                        # SAC-PSO: Soft Actor-Critic điều khiển động Standard PSO
├── train_sac_pso.py                  # Pipeline huấn luyện SAC trên tập 30+ instances đa quy mô
├── benchmark_experiment.py           # Thử nghiệm độc lập 5 runs trên cả 4 Scenarios (1, 2, 3, 4)
├── 30_9_2026.py                      # Entry point thực thi chính
└── README.md
```

---

## 3. Kiến trúc SAC-PSO (Soft Actor-Critic Meta-Optimization for Standard PSO)

Trong Standard PSO giải bài toán JSSPLA:
- Mỗi hạt đại diện cho lời giải gồm hai vector liên tục:
  * $X_{os} \in [-4.0, 4.0]^D$: Vector liên tục xác định thứ tự công đoạn qua quy tắc SPV (*Smallest Position Value*).
  * $X_{aa} \in [0.0, 1.0]^D$: Vector liên tục xác định lựa chọn robot trong cặp AMR chuyên trách.
- Phương trình cập nhật vận tốc và tọa độ hạt:
  $$V_{os}(t+1) = w \cdot V_{os}(t) + c_1 r_1 (pbest_{os} - X_{os}(t)) + c_2 r_2 (gbest_{os} - X_{os}(t))$$
  $$V_{aa}(t+1) = w \cdot V_{aa}(t) + c_1 r_1 (pbest_{aa} - X_{aa}(t)) + c_2 r_2 (gbest_{aa} - X_{aa}(t))$$

### Hạn chế của Standard PSO truyền thống:
1. **Lịch trình tham số cố định**: $w$ giảm tuyến tính từ $0.9 \to 0.4$, $c_1 = 1.5, c_2 = 1.5$. Khi đàn hạt bị kẹt ở cực tiểu cục bộ, việc giảm tiếp $w$ khiến vận tốc triệt tiêu, bầy hạt hoàn toàn tê liệt (*stagnation*).
2. **Kẹt nghiệm tổ hợp trên instance quy mô lớn**: Trên Scenario 4 với 750 công đoạn, các hạt khi tiến sát $gbest$ sẽ có thứ tự hoán vị `argsort(X_os)` giống hệt nhau, không thể thoát bẫy nếu thiếu cơ chế kích thích đa dạng.

### Giải pháp SAC Meta-Optimizer:
Tác tử **Soft Actor-Critic (SAC)** tương tác với môi trường PSO ở mỗi thế hệ $t$ ($t = 0 \dots T_{max}-1$):
- **Vector trạng thái (State - 12 chiều)**:
  1. Tiến độ thế hệ ($t / T_{max}$)
  2. Tỉ lệ makespan $gbest$ hiện tại so với ban đầu
  3. Tỉ lệ giá trị trung bình bầy hạt $\overline{pbest}$
  4. Độ đa dạng thích nghi bầy hạt ($std / mean$)
  5. Mức độ trì trệ bầy hạt (số bước liên tiếp không cải thiện $gbest$)
  6. Mức cải thiện makespan bước gần nhất
  7. Mức cải thiện trung bình của bầy hạt
  8. Độ lớn trung bình của vận tốc hạt $\|V\|$
  9. Độ phân tán tọa độ không gian vị trí
  10. Quy mô kích thước bài toán ($N_{ops} / 750$)
  11. Tỉ lệ đơn hàng Inbound / Outbound
  12. Trọng số quán tính $w$ ở bước trước
- **Không gian hành động (Action - 6 chiều liên tục)**:
  1. $w \in [0.20, 0.95]$: Trọng số quán tính (*Inertia weight*)
  2. $c_1 \in [0.40, 2.60]$: Gia tốc nhận thức cá nhân (*Cognitive acceleration*)
  3. $c_2 \in [0.40, 2.60]$: Gia tốc liên kết xã hội (*Social acceleration*)
  4. $v_{max} \in [1.20, 3.20]$: Giới hạn vận tốc động (*Adaptive velocity clamping*)
  5. $p_{perturb} \in [0.0, 0.30]$: Tỉ lệ kích thích đa dạng chống trì trệ (*Anti-stagnation kick rate*)
  6. $p_{balance} \in [0.0, 0.90]$: Xác suất tái cân bằng tải giữa 2 AMR trong cụm (*AMR workload balancing*)

---

## 4. Kết quả Thực nghiệm Đối sánh (5 Runs độc lập)

Kết quả đo đạc chính xác trên 4 Scenarios chuẩn của bài báo khoa học:
- **Scenario 1**: 8 Inbound, 16 Outbound (24 Jobs / 56 Operations) - Baseline.
- **Scenario 2**: 4 Inbound, 24 Outbound (28 Jobs / 76 Operations) - Downstream workstation bottleneck.
- **Scenario 3**: 20 Inbound, 8 Outbound (28 Jobs / 44 Operations) - Inbound receiving surge.
- **Scenario 4**: 150 Inbound, 200 Outbound (350 Jobs / 750 Operations) - Extreme Large Stress Test.

| Scenario | Algorithm | $C_{max}^{best}$ (s) | $C_{max}^{avg}$ (s) | SD (%) | Baseline Imp (%) | Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | FIFO | 916.91 | 916.91 | 0.00 | 0.00% | 0.000 |
| | SPT | 1111.04 | 1111.04 | 0.00 | -21.17% | 0.000 |
| | LPT | 1277.32 | 1277.32 | 0.00 | -39.31% | 0.000 |
| | GA | 703.76 | 717.84 | 1.36 | 23.25% | 0.050 |
| | CDPSO (Paper) | 749.75 | 773.90 | 3.80 | 18.23% | 0.103 |
| | PSO | 652.73 | 681.47 | 3.12 | 28.81% | 0.071 |
| | **SAC-PSO (Ours)** | **666.71** | **705.03** | **4.80** | **27.29%** | 0.307 |
| **2** | FIFO | 1191.16 | 1191.16 | 0.00 | 0.00% | 0.000 |
| | SPT | 1577.22 | 1577.22 | 0.00 | -32.41% | 0.000 |
| | LPT | 1650.44 | 1650.44 | 0.00 | -38.56% | 0.000 |
| | GA | 1003.85 | 1044.01 | 2.30 | 15.73% | 0.066 |
| | CDPSO (Paper) | 1055.80 | 1078.63 | 1.85 | 11.36% | 0.131 |
| | PSO | 975.54 | 1008.72 | 2.21 | 18.10% | 0.089 |
| | **SAC-PSO (Ours)** | **920.23** | **978.45** | **3.76** | **22.75%** *(Vượt trội)* | 0.170 |
| **3** | FIFO | 526.78 | 526.78 | 0.00 | 0.00% | 0.000 |
| | SPT | 721.91 | 721.91 | 0.00 | -37.04% | 0.000 |
| | LPT | 1085.31 | 1085.31 | 0.00 | -106.03% | 0.000 |
| | GA | 516.62 | 518.43 | 0.49 | 1.93% | 0.048 |
| | CDPSO (Paper) | 516.62 | 524.42 | 0.93 | 1.93% | 0.082 |
| | PSO | 516.62 | 517.92 | 0.50 | 1.93% | 0.058 |
| | **SAC-PSO (Ours)** | **516.62** | **516.62** | **0.00** | **1.93%** *(Tối ưu tuyệt đối, SD=0%)* | 0.113 |
| **4** *(Large Instance)* | FIFO | 11299.52 | 11299.52 | 0.00 | 0.00% | 0.000 |
| | SPT | 14441.94 | 14441.94 | 0.00 | -27.81% | 0.001 |
| | LPT | 16577.91 | 16577.91 | 0.00 | -46.71% | 0.001 |
| | GA | 9031.28 | 9285.78 | 1.71 | 20.07% | 1.247 |
| | CDPSO (Paper) | 9192.77 | 9348.53 | 1.58 | 18.64% | 2.447 |
| | PSO | 8747.42 | 8956.50 | 1.33 | 22.59% | 1.577 |
| | **SAC-PSO (Ours)** | **7992.19** | **8633.62** | **5.34** | **29.27%** *(Kỷ lục < 8000s)* | 2.252 |

> **Điểm nhấn kịch bản quy mô lớn (Scenario 4 - 350 Jobs / 750 Operations)**:  
> - **SAC-PSO** đạt Makespan kỷ lục $C_{max}^{best} = \mathbf{7,992.19\text{s}}$ (trung bình $\mathbf{8,633.62\text{s}}$), **phá vỡ cột mốc 8,000s, vượt trội hơn hẳn so với CDPSO của bài báo** ($9,192.77\text{s}$) — giảm hơn **1,200 giây** (20 phút) thời gian vận hành kho!
> - Đánh bại hoàn toàn Standard PSO ($8,747.42\text{s}$), Standard GA ($9,031.28\text{s}$), và các giải thuật heuristic (FIFO: $11,299.52\text{s}$, SPT: $14,441.94\text{s}$, LPT: $16,577.91\text{s}$).

---

## 5. Hướng dẫn Chạy Thử nghiệm

Chạy trực tiếp toàn bộ benchmark đối sánh và tự động xuất bảng kết quả & biểu đồ:
```bash
python 30_9_2026.py
```

Huấn luyện lại mô hình SAC trên tập 30+ instance mở rộng:
```bash
python train_sac_pso.py
```
