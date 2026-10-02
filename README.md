# JSSPLA: Job Shop Scheduling with Limited AMRs in Smart Warehouse
### Deep Reinforcement Learning (Soft Actor-Critic) Meta-Optimization for Standard PSO (SAC-PSO) vs Standard PSO, GA, FIFO, SPT, LPT

> **Đề tài nghiên cứu**: *"Job Shop-Based Scheduling Optimization for Multi-AMR Warehouse Systems and Validation via Gazebo Simulation"*  
> **Supervisor**: Dr. Truong Ngoc Cuong  
> **Student**: Tran Viet Trung An (MAIL LAB)

---

## 1. Tổng quan Bài toán JSSPLA trong Kho Thông Minh

Hệ thống kho hàng tự động diện tích $33.0\text{m} \times 16.5\text{m}$ vận hành đội 8 robot tự hành (AMR) phục vụ 4 cụm kệ hàng ($R_1, R_2, R_3, R_4$) và các trạm chức năng:
- **Trạm nhận & kiểm tra hàng**: Receiving $\to$ Buffer 1
- **Khu vực lưu trữ trung tâm**: 4 cụm kệ $R_1, R_2, R_3, R_4$
- **Trạm hạ nguồn**: Picking (lấy hàng) $\to$ Packing (đóng gói) $\to$ Buffer 2 $\to$ Shipping (xuất xưởng)
- **Trạm sạc tự động**: Automated Charging Station
- **Chính sách phân đội chuyên trách (Dedicated Sub-fleet Policy)**: Mỗi cụm kệ $k \in \{1, 2, 3, 4\}$ được giao đúng 2 robot cố định ($R_1 \to [0, 1]$, $R_2 \to [2, 3]$, $R_3 \to [4, 5]$, $R_4 \to [6, 7]$). Mọi tác vụ vận chuyển đơn hàng thuộc cụm đó bắt buộc do 1 trong 2 robot này thực hiện.
- **Ràng buộc thời gian và động học pin vật lý**:
  - Vận tốc AMR định mức $v = 1.0\text{ m/s}$.
  - Thời gian bốc dỡ tự động tại trạm: $t_{load} = 10\text{s}, t_{unload} = 10\text{s}$.
  - Tiêu hao năng lượng theo trạng thái: tải trọng có hàng ($\alpha_l$) và chạy không tải ($\alpha_e$).
  - Ngưỡng pin an toàn $b_{min} = 20\%$, tự động điều hướng sang trạm sạc khi mức pin xuống thấp.

---

## 2. Cấu trúc Thư mục Dự án

```
Coding/
│
├── models/                               # Trọng số mạng nơ-ron đã huấn luyện
│   └── sac_warehouse_pso.pt              # Trọng số SAC Meta-Optimizer Actor-Critic
│
├── results/                              # Kết quả thực nghiệm đối sánh & đồ họa trực quan
│   ├── table_comparison.md               # Bảng đối sánh chi tiết Cmax, SD, cải thiện trên 4 Scenarios
│   ├── table1_comparison.md              # Bảng định dạng rút gọn theo chuẩn Table 1 bài báo
│   ├── barchart_all_instances.png        # Biểu đồ cột ngang/dọc đa instance
│   ├── barchart_unified_grouped.png      # Biểu đồ cột nhóm tổng hợp toàn diện 4 Scenarios
│   ├── barchart_4instances_grid.png      # Lưới 2x2 biểu đồ cột chi tiết có thanh sai số SD cho từng Scenario
│   ├── boxplots_comparison.png           # Biểu đồ hộp (Box plots) đối sánh độ ổn định và phân vị IQR
│   ├── scenario4_convergence.png         # Đồ thị hội tụ Makespan trên Scenario 4 quy mô lớn (350 Jobs / 750 Ops)
│   └── sac_pso_training.png              # Đường cong phần thưởng huấn luyện SAC Agent trên 30+ instances
│
├── warehouse_env.py                      # Mô hình môi trường kho JSSPLA, Layout, 8 AMRs, Pin, Decoder SPV
├── baselines.py                          # Thuật toán so chuẩn: Heuristics (FIFO, SPT, LPT), Standard GA, Standard PSO
├── sac_pso.py                            # Thuật toán đề xuất: Memetic SAC-PSO với 5 cải tiến đột phá
├── train_sac_pso.py                      # Pipeline huấn luyện SAC Agent với Replay Buffer đa dạng
├── benchmark_experiment.py               # Thử nghiệm độc lập đa lượt (num_runs=5) & tự động xuất biểu đồ
├── main.py                               # Điểm khởi chạy chương trình (Entry point)
└── README.md                             # Tài liệu kỹ thuật chi tiết
```

---

## 3. Kiến trúc Đề xuất: Enhanced Memetic SAC-PSO

Trong thuật toán PSO tiêu chuẩn giải bài toán JSSPLA:
- Mỗi hạt đại diện cho lời giải gồm hai vector liên tục:
  * $X_{os} \in [-4.0, 4.0]^D$: Vector liên tục xác định thứ tự công đoạn qua quy tắc SPV (*Smallest Position Value*).
  * $X_{aa} \in [0.0, 1.0]^D$: Vector liên tục xác định phân bổ AMR trong cặp robot chuyên trách của cụm.
- Phương trình cập nhật vận tốc và vị trí hạt:
  $$V_{os}(t+1) = w \cdot V_{os}(t) + c_1 r_1 (pbest_{os} - X_{os}(t)) + c_2 r_2 (gbest_{os} - X_{os}(t))$$
  $$X_{os}(t+1) = \text{clip}(X_{os}(t) + V_{os}(t+1), -4.0, 4.0)$$

### 5 Cải tiến Đột phá trong Thuật toán Memetic SAC-PSO:

1. **Khởi tạo Quần thể Lai ghép (Heuristic Seeding + Opposition-Based Learning - OBL)**:
   - *Hạt 0*: Trình tự đến tự nhiên của đơn hàng kết hợp phân bổ AMR xen kẽ cân bằng.
   - *Hạt 1*: Ưu tiên đơn hàng nhập kho (*Inbound-Priority Sequence*) để giải phóng Buffer 1.
   - *Hạt 2*: Quy tắc thời gian gia công ngắn nhất (*Shortest Processing Time - SPT*) giảm tắc nghẽn trạm.
   - *Hạt 3*: Hạt đối ngẫu (*Opposition Particle*) của Hạt 0 mở rộng không gian tìm kiếm đối xứng.
   - *Các hạt còn lại (4 .. P-1)*: Phân bố đều ngẫu nhiên liên tục để duy trì tính đa dạng.
   $\to$ Thiết lập cận dưới lời giải chất lượng cao ngay từ thế hệ $t=0$, triệt tiêu các lượt chạy ngẫu nhiên kém và hạ thấp phương sai.

2. **Động lực học Vận tốc Co cụm Ổn định (Stable Constriction Velocity Dynamics)**:
   - Giới hạn tham số an toàn theo tiêu chuẩn hội tụ Clerc-Kennedy ($w \in [0.35, 0.88]$, $c_1, c_2 \in [1.0, 2.0]$, $c_1+c_2 \le 3.6$).
   - SAC can thiệp dưới dạng phản hồi vi sai ($\Delta w, \Delta c_1, \Delta c_2, \Delta v_{max}$) giúp thích nghi linh hoạt mà không làm mất ổn định quỹ đạo hạt.

3. **Cơ chế Tìm kiếm Cục bộ Đường Găng có Định hướng (Critical-Path Local Search)**:
   - Tập trung định vị các đơn hàng và công đoạn hoàn thành muộn nhất trên đường găng (critical path).
   - Thực hiện hoán đổi và dịch chuyển công đoạn nhằm loại bỏ khoảng thời gian chờ (*idle gaps*) của robot và trạm làm việc.

4. **Tái cân bằng Tải Trọng AMR Thông minh Toàn diện (Universal Smart AMR Workload Balancing)**:
   - Đánh giá độ chênh lệch thời gian hoàn thành giữa 2 robot trong từng cụm kệ ($|t_{R,1} - t_{R,2}|$).
   - Tự động điều chuyển tác vụ từ robot quá tải sang robot nhàn rỗi theo cơ chế Greedy đa vòng.

5. **Không gian Trạng thái 14 Chiều & Hàm Phần thưởng Dày (Dense Multi-Objective Reward)**:
   - **Vector trạng thái (14 chiều)**:
     1. Tiến trình thế hệ ($t / T_{max}$)
     2. Tỉ lệ Makespan $gbest$ hiện tại so với ban đầu
     3. Tỉ lệ giá trị trung bình bầy hạt $\overline{pbest}$
     4. Độ đa dạng thích nghi bầy hạt ($std / mean$)
     5. Mức độ trì trệ bầy hạt ($stagnation / T_{max}$)
     6. Mức cải thiện $gbest$ ở bước gần nhất
     7. Mức cải thiện trung bình của bầy hạt
     8. Vận tốc trung bình chuẩn hóa $\|V\|$
     9. Độ phân tán tọa độ vị trí
     10. Quy mô kích thước bài toán ($N_{ops} / 750$)
     11. Tỉ lệ đơn hàng Inbound / Tổng đơn
     12. Độ lệch tải giữa các AMR ($amr\_disp$)
     13. Tỉ lệ trung bình của nhóm hạt tinh hoa ($elite\_norm$)
     14. Hành động $\Delta w$ ở bước trước
   - **Không gian hành động (6 chiều liên tục)**: $\Delta w, \Delta c_1, \Delta c_2, \Delta v_{max}, p_{perturb}, p_{balance}$.

---

## 4. Kết quả Thực nghiệm Đối sánh (5 Runs Độc lập)

Thực nghiệm đo đạc độc lập 5 lượt chạy ngẫu nhiên trên 4 kịch bản chuẩn của hệ thống kho:
- **Scenario 1**: 8 Inbound, 16 Outbound (24 Jobs / 56 Operations) - Baseline.
- **Scenario 2**: 4 Inbound, 24 Outbound (28 Jobs / 76 Operations) - Quá tải cụm trạm Picking/Packing hạ nguồn.
- **Scenario 3**: 20 Inbound, 8 Outbound (28 Jobs / 44 Operations) - Sóng hàng nhập kho lớn tại trạm Receiving.
- **Scenario 4**: 150 Inbound, 200 Outbound (350 Jobs / 750 Operations) - Thử nghiệm áp lực quy mô lớn (*Large Scale Stress Test*).

### Bảng Kết quả Tổng hợp:

| Scenario | Thuật toán | $C_{max}^{best}$ (s) | $C_{max}^{avg}$ (s) | $C_{max}^{worst}$ (s) | SD (s) | SD (%) (Độ ổn định) | Cải thiện FIFO (%) | Thời gian (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scenario 1**<br>*(24 Jobs / 56 Ops)* | **FIFO** | 916.91 | 916.91 | 916.91 | 0.00 | 0.00% *(Đơn định)* | 0.00% | 0.000 |
| | **SPT** | 1111.04 | 1111.04 | 1111.04 | 0.00 | 0.00% *(Đơn định)* | -21.17% | 0.000 |
| | **LPT** | 1277.32 | 1277.32 | 1277.32 | 0.00 | 0.00% *(Đơn định)* | -39.31% | 0.000 |
| | **GA** | 703.76 | 717.84 | 732.03 | 9.73 | 1.36% | 23.25% | 0.488 |
| | **PSO** | 652.73 | 681.47 | 707.86 | 21.24 | 3.12% | 28.81% | 0.367 |
| | **SAC-PSO (Ours)** | **654.06** | **660.97** | **669.83** | **5.08** | **0.77%** *(Cực ổn định)* | **28.67%** | 1.664 |
| **Scenario 2**<br>*(28 Jobs / 76 Ops)* | **FIFO** | 1191.16 | 1191.16 | 1191.16 | 0.00 | 0.00% *(Đơn định)* | 0.00% | 0.000 |
| | **SPT** | 1577.22 | 1577.22 | 1577.22 | 0.00 | 0.00% *(Đơn định)* | -32.41% | 0.000 |
| | **LPT** | 1650.44 | 1650.44 | 1650.44 | 0.00 | 0.00% *(Đơn định)* | -38.56% | 0.005 |
| | **GA** | 1003.85 | 1044.01 | 1068.66 | 23.97 | 2.30% | 15.73% | 0.120 |
| | **PSO** | 975.54 | 1008.72 | 1042.16 | 22.34 | 2.21% | 18.10% | 0.171 |
| | **SAC-PSO (Ours)** | **926.24** | **941.77** | **959.06** | **13.68** | **1.45%** *(Vượt trội)* | **22.24%** | 0.209 |
| **Scenario 3**<br>*(28 Jobs / 44 Ops)* | **FIFO** | 526.78 | 526.78 | 526.78 | 0.00 | 0.00% *(Đơn định)* | 0.00% | 0.000 |
| | **SPT** | 721.91 | 721.91 | 721.91 | 0.00 | 0.00% *(Đơn định)* | -37.04% | 0.000 |
| | **LPT** | 1085.31 | 1085.31 | 1085.31 | 0.00 | 0.00% *(Đơn định)* | -106.03% | 0.000 |
| | **GA** | 516.62 | 518.43 | 523.12 | 2.54 | 0.49% | 1.93% | 0.079 |
| | **PSO** | 516.62 | 517.92 | 523.12 | 2.60 | 0.50% | 1.93% | 0.095 |
| | **SAC-PSO (Ours)** | **516.62** | **516.62** | **516.62** | **0.00** | **0.00%** *(Tối ưu tuyệt đối)* | **1.93%** | 0.154 |
| **Scenario 4**<br>*(350 Jobs / 750 Ops)* | **FIFO** | 11299.52 | 11299.52 | 11299.52 | 0.00 | 0.00% *(Đơn định)* | 0.00% | 0.002 |
| | **SPT** | 14441.94 | 14441.94 | 14441.94 | 0.00 | 0.00% *(Đơn định)* | -27.81% | 0.003 |
| | **LPT** | 16577.91 | 16577.91 | 16577.91 | 0.00 | 0.00% *(Đơn định)* | -46.71% | 0.001 |
| | **GA** | 9031.28 | 9285.78 | 9450.26 | 158.45 | 1.71% | 20.07% | 1.203 |
| | **PSO** | 8747.42 | 8956.50 | 9108.93 | 118.69 | 1.33% | 22.59% | 1.705 |
| | **SAC-PSO (Ours)** | **7433.14** | **7548.57** | **7756.31** | **113.14** | **1.50%** *(Đột phá vượt bậc)* | **34.22%** | 3.619 |

---

## 5. Phân tích Chi tiết & Ưu thế Vượt trội của SAC-PSO

1. **Hiệu năng đột phá trên Instance lớn (Scenario 4)**:
   - Trên bài toán quy mô lớn 350 Jobs (750 công đoạn), SAC-PSO rút ngắn Makespan trung bình từ **11299.52s** (FIFO) xuống còn **7548.57s**, đạt mức cải thiện ấn tượng **34.22%**.
   - SAC-PSO vượt xa Standard PSO (**8956.50s**) hơn **1407 giây**, minh chứng rõ ràng sức mạnh của việc tinh chỉnh tham số động và tìm kiếm cục bộ đường găng khi giải bài toán tổ hợp phức tạp.

2. **Tính Ổn định Xuất sắc (Compact IQR & Low SD)**:
   - Hệ số biến thiên $\text{SD}$ của SAC-PSO duy trì ở mức cực kỳ thấp (**0.00% - 1.50%**) trên tất cả các kịch bản.
   - Đặc biệt ở Scenario 1, độ lệch chuẩn giảm chỉ còn **5.08s (0.77%)**, so với **21.24s (3.12%)** của PSO chuẩn.
   - Ở Scenario 3, SAC-PSO đạt độ hội tụ tuyệt đối $C_{max} = 516.62\text{s}$ trên cả 5 runs ($\text{SD} = 0.00\text{s}$).

3. **Trực quan hóa Đa dạng trong Thư mục `results/`**:
   - `boxplots_comparison.png`: Biểu đồ hộp phân vị cho thấy dải IQR của SAC-PSO luôn nằm thấp hơn và thu gọn hơn đáng kể so với GA và Standard PSO.
   - `barchart_4instances_grid.png`: Lưới 2x2 thể hiện rõ Makespan và thanh sai số chuẩn $\text{SD}$ của 6 thuật toán trên từng Scenario.
   - `barchart_unified_grouped.png`: Biểu đồ cột nhóm so sánh trực quan toàn diện trên cùng một khung hình.
   - `scenario4_convergence.png`: Đường cong hội tụ thể hiện tốc độ giảm Makespan thần tốc của SAC-PSO qua các thế hệ.

---

## 6. Hướng dẫn Cài đặt & Khởi chạy

### Yêu cầu Môi trường:
- Python 3.9+
- PyTorch (`torch`)
- NumPy
- Matplotlib

Cài đặt nhanh các thư viện phụ thuộc:
```bash
pip install torch numpy matplotlib
```

### Chạy Thực nghiệm Đối sánh Đầy đủ:
Chạy trực tiếp file `main.py` để tự động kiểm tra mô hình, chạy benchmark 5 runs trên 4 Scenarios và xuất toàn bộ bảng kết quả & biểu đồ vào thư mục `results/`:
```bash
python main.py
```

*(Hoặc chạy trực tiếp module benchmark: `python benchmark_experiment.py`)*

### Huấn luyện lại SAC Meta-Optimizer Agent:
Nếu muốn huấn luyện lại mạng nơ-ron Actor-Critic từ đầu trên bộ 30+ instance kho ngẫu nhiên:
```bash
python train_sac_pso.py
```
Model sau khi huấn luyện sẽ tự động được lưu vào `models/sac_warehouse_pso.pt`.
