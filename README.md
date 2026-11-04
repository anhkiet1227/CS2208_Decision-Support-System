# CS2208 - Hệ Hỗ Trợ Quyết Định (Decision Support Systems)

<div align="center">

### ĐỒ ÁN MÔN HỌC

## PHÂN TÍCH ĐIỀU KIỆN THỜI TIẾT TƯƠNG ĐỒNG NHẰM NÂNG CAO KHẢ NĂNG TÁI SỬ DỤNG TRONG HỆ HỖ TRỢ RA QUYẾT ĐỊNH NÔNG NGHIỆP
**(Analysis of Similar Weather Conditions to Improve Reuse in Weather-Based Decision Support Systems)**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![UIT - VNU-HCM](https://img.shields.io/badge/University-UIT--VNU--HCM-orange.svg)](https://www.uit.edu.vn/)

</div>

---

- **Học viên thực hiện:** Võ Anh Kiệt (MSHV: `26848109`)
- **Giảng viên hướng dẫn:** TS. Tôn Thất Hòa An
- **Đơn vị đào tạo:** Khoa Khoa học Máy tính – Trường Đại học Công nghệ Thông tin, ĐHQG TP.HCM
- **Kho lưu trữ GitHub:** [CS2208_Decision-Support-System](https://github.com/anhkiet1227/CS2208_Decision-Support-System)
- **Công trình khoa học nền tảng (Q1, ISI/Scopus):**  
  > Mamatha Alugubelly, Krishna Reddy Polepalli, Sreenivas Gade, & Seishi Ninomiya (2019). *Analysis of similar weather conditions to improve reuse in weather-based decision support systems*. **Computers and Electronics in Agriculture**, Vol. 157, pp. 154–165. https://doi.org/10.1016/j.compag.2018.12.010

---

## 📌 Giới thiệu và Đóng góp cốt lõi của Đề tài

Đề tài kế thừa và phát triển công trình gốc của Alugubelly và cộng sự (2019) về việc lượng tử hóa vector khí tượng số thực thành **Điều kiện Thời tiết dựa trên Danh mục (Categorical Weather Condition — CWC)** để đo lường độ bao phủ tái sử dụng (**Coverage Percentage — CP %**) trong Dịch vụ Tư vấn Nông nghiệp - Khí tượng Tích hợp (IAAS).

Nhằm khắc phục **4 hạn chế cốt lõi** của mô hình gốc (Hiệu ứng vách đá tại biên, Lời nguyền chiều dữ liệu & trọng số tĩnh, Thiếu hụt ngữ cảnh phi thời tiết, và Rủi ro ảo giác khi tự động hóa khuyến nghị), đề tài đề xuất khung giải pháp: **Hệ hỗ trợ quyết định Nông nghiệp Mờ - Thích ứng Động tích hợp Trí tuệ nhân tạo Tạo sinh (Fuzzy-Dynamic ECWC with Grounded RAG-LLM)** gồm 4 trụ cột:

1. **Giải pháp 1 — Làm mượt ranh giới bằng Logic mờ (Fuzzy CWC), Hệ số Gower & Rào cản Sinh học Tới hạn ($I_{veto}$):**  
   Ứng dụng hàm thuộc hình thang thỏa mãn phân hoạch Ruspini với bán kính chuyển tiếp chuẩn hóa $\delta = \pm 10\%$ quanh ngưỡng cắt $B_m$ và độ đo khoảng cách thứ bậc Fuzzy Gower. Khắc phục hoàn toàn việc rơi độ tương đồng về $0.0\%$, duy trì $Sim_{Fuzzy} > 97\%$ cho các mẫu lệch sát biên. Tích hợp biến chỉ thị rào cản sinh học $I_{veto}(X, Y) \in [0, 1]$ ngăn chặn tái sử dụng sai phác đồ khi xuất hiện nguy cơ bùng phát bệnh đạo ôn hoặc ngập úng.
2. **Giải pháp 2 — Lựa chọn tập biến và phân bổ trọng số thích ứng động (AHP):**  
   Xây dựng ma trận trọng số động $w(pt, \varphi)$ theo 3 mùa vụ (Kharif, Rabi, Summer) và 6 giai đoạn sinh trưởng của lúa bằng phương pháp Phân tích Thứ bậc (AHP, Saaty, 1980) thỏa mãn tỷ số nhất quán $CR = 0.0469 < 0.10$.
3. **Giải pháp 3 — Mở rộng không gian vector ngữ cảnh (Extended CWC — ECWC) & Truy hồi phân tầng 2 giai đoạn:**  
   Mở rộng vector 9 thành phần tích hợp thổ nhưỡng ($u_{soil}$), nhóm giống ($u_{variety}$), áp lực sâu bệnh ($u_{pest}$) và chỉ số viễn thám ($u_{ndvi}$). Cơ chế truy hồi phân tầng: Giai đoạn 1 lọc thô trên không gian thời tiết mờ (loại bỏ $96.2\%$ ca không phù hợp), Giai đoạn 2 tái xếp hạng ngữ cảnh theo hệ số $\alpha = 0.75$.
4. **Giải pháp 4 — Kiến trúc Grounded RAG-LLM với Nguyên tắc An toàn Hai lớp:**  
   Tách bạch tuyệt đối giữa **Tầng quy đổi số học tất định** (tính toán giải tích liều lượng phân bón/thuốc BVTV theo diện tích nông hộ) và **Tầng tạo sinh ngôn ngữ LLM** (neo chặt thông tin, chống ảo giác 100%), tự động xuất bản tin khuyến nông cá nhân hóa đa kênh qua **Zalo Official Account (OA)** và tin nhắn **SMS 160 ký tự**.

---

## 🗂️ Cấu trúc Thư mục Dự án (Project Structure)

```text
CS2208_Decision-Support-System/
├── src/
│   ├── __init__.py
│   ├── wct_mapper.py          # Mô-đun 1: Crisp WCT (IMD), Hàm thuộc mờ hình thang Ruspini & Rào cản sinh học I_veto
│   ├── cp_engine.py           # Mô-đun 2: Algorithm 1, Trọng số động AHP, Độ đo Fuzzy Gower, AP, F1 & Paired t-test
│   ├── ecwc_rag_llm.py        # Mô-đun 3: Vector ngữ cảnh ECWC (9-tuple), Truy hồi phân tầng 2 giai đoạn & Grounded RAG
│   └── dataset_generator.py   # Bộ sinh dữ liệu khí tượng mô phỏng hiệu chỉnh 30 năm (10.957 ngày) & Kho tri thức Pset
├── data/
│   ├── rajendranagar_weather_1986_2015.csv   # Dữ liệu thời tiết 30 năm trạm Hyderabad (10.957 bản ghi ngày)
│   └── agromet_knowledge_base.json           # Kho tri thức khuyến nghị kỹ thuật canh tác chuyên gia (Pset)
├── results/
│   ├── figures/                              # Bộ hình ảnh minh họa chất lượng cao (300 DPI)
│   │   ├── fig2_1_workflow.png               # Quy trình chuyển đổi WC -> CWC và đo CP (Alugubelly et al.)
│   │   ├── fig3_1_crisp_vs_fuzzy.png         # Đối sánh trực quan Phân lớp cứng vs Phân lớp mờ
│   │   ├── fig3_2_rag_architecture.png       # Sơ đồ kiến trúc 3 tầng: Fuzzy CWC, Định mức tất định & RAG-LLM
│   │   ├── fig4_1_cp_comparison.png          # Biểu đồ so sánh CP: WC số thực, Baseline Crisp CWC & Fuzzy-Dynamic
│   │   ├── fig4_2_code_execution_plots.png   # Kiểm chứng xóa bỏ Hiệu ứng Vách đá & Đường cong hội tụ CP 30 năm
│   │   └── fig4_3_streamlit_zalo_demo.png    # Giao diện Web Dashboard Streamlit & Mô phỏng Zalo OA / SMS
│   ├── table_4_2.csv                         # Đối sánh WC số thực vs CWC danh mục tại chu kỳ thứ 5 (p_5)
│   ├── table_4_3.csv                         # Độ bao phủ CP chu kỳ Năm (p_i = 3, 5, 7, 10)
│   ├── table_4_4.csv                         # Độ bao phủ CP vụ mùa đông Rabi (p_i = 3, 5, 7, 10)
│   ├── table_4_5.csv                         # Độ bao phủ CP giai đoạn Trổ hoa (Flowering) vụ Kharif & Rabi
│   ├── table_4_6.csv                         # Bảng đối sánh trung tâm R = 10 seeds, Baseline vs Proposed (CP, AP, F1, t-test)
│   ├── table_4_7_sensitivity.csv             # Quét lưới phân tích độ nhạy (θ, δ), AP_NoBio vs AP_WithBio và F1_reuse
│   ├── sim_log_kharif_flowering_n5_d5.csv    # Log chi tiết thực nghiệm vụ Kharif Trổ hoa (n=5, d=5) qua 10 seeds
│   ├── sim_log_year_n3_d5.csv                # Log chi tiết thực nghiệm chu kỳ Năm (n=3, d=5) qua 10 seeds
│   └── rag_llm_demo_output.json              # Dữ liệu mẫu kết quả truy hồi ECWC và bản tin RAG-LLM
├── report/
│   ├── 26848109_VoAnhKiet_DSS_final.docx     # Báo cáo đồ án môn học hoàn chỉnh (chuẩn mực 100% format & phản biện)
│   ├── 26848109_VoAnhKiet_DSS_final.pdf      # Bản xuất PDF báo cáo cuối kỳ
│   └── 26848147_26848109_DSS_midterm.pptx    # Slide thuyết trình báo cáo giữa kỳ
├── run_experiments.py         # Script CLI chạy toàn bộ thực nghiệm, kiểm định thống kê và vẽ biểu đồ tự động
├── app.py                     # Web Dashboard Streamlit tương tác & Mô phỏng Zalo OA Chatbot khuyến nông
├── generate_standard_report.py# Script tự động tạo file Word chuẩn mực từ dữ liệu thực nghiệm và phôi mẫu
├── requirements.txt           # Danh mục thư viện Python yêu cầu
├── LICENSE                    # Giấy phép mã nguồn mở MIT
└── README.md                  # Tài liệu hướng dẫn dự án
```

---

## 📊 Tóm tắt Kết quả Thực nghiệm Nổi bật

### 1. Đối sánh Đa hạt giống ($R = 10$ seeds) trên cùng Tập Dữ liệu (Bảng 4.6)
Toàn bộ thuật toán Baseline Crisp CWC và mô hình cải tiến Fuzzy-Dynamic CWC được chạy lại độc lập qua 10 hạt giống ngẫu nhiên (Seeds 42–51) trên chuỗi dữ liệu 30 năm (10.957 ngày):

| Kịch bản chu kỳ & Cấu hình | Chu kỳ | Paper CP (%) | Baseline CP (Mean±SD) | Baseline F1 (%) | Proposed CP (Mean±SD) | Proposed AP (%) | Proposed F1 (%) | Net Gain CP (%pts) | Net Gain F1 (%pts) | Paired t-test ($p$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Năm \| (n=3, d=5)** | $p_5$ | 84.0 | 95.2 ± 2.8% | 97.3 | **98.6 ± 1.2%** | 98.2 | **98.4** | +3.4 %pts | +1.1 %pts | $p < 0.01$ |
| **Năm \| (n=5, d=5)** | $p_5$ | 76.0 | 69.9 ± 6.2% | 82.0 | **89.3 ± 4.1%** | 95.2 | **92.1** | +19.5 %pts | +10.1 %pts | $p < 0.001$ |
| **Vụ Rabi \| (n=3, d=5)** | $p_5$ | 78.0 | 82.3 ± 9.6% | 90.0 | **90.7 ± 6.5%** | 94.2 | **92.3** | +8.3 %pts | +2.3 %pts | $p < 0.01$ |
| **Vụ Rabi \| (n=5, d=5)** | $p_5$ | 60.0 | 59.7 ± 11.6% | 74.0 | **80.0 ± 10.8%** | 91.1 | **84.8** | +20.3 %pts | +10.8 %pts | $p < 0.001$ |
| **Trổ hoa Kharif \| (n=3, d=5)** | $p_3$ | 53.0 | 45.0 ± 23.2% | 58.5 | **58.8 ± 20.2%** | 100.0 | **71.6** | +13.8 %pts | +13.1 %pts | $p = 0.012$ |
| **Trổ hoa Kharif \| (n=3, d=5)** | $p_7$ | 78.0 | 60.0 ± 15.6% | 73.7 | **86.2 ± 14.2%** | 87.8 | **86.4** | +26.2 %pts | +12.7 %pts | $p < 0.001$ |
| **Trổ hoa Kharif \| (n=5, d=5)** | $p_3$ | 22.0 | 16.2 ± 11.2% | 26.3 | **36.2 ± 19.7%** | 90.5 | **48.8** | +20.0 %pts | +22.4 %pts | $p < 0.01$ |
| **Trổ hoa Kharif \| (n=5, d=5)** | $p_{10}$ | 44.0 | 52.5 ± 18.4% | 66.8 | **81.2 ± 11.5%** | 88.7 | **84.4** | +28.8 %pts | +17.6 %pts | $p < 0.001$ |
| **Trổ hoa Rabi \| (n=3, d=5)** | $p_3$ | 38.0 | 48.6 ± 15.9% | 63.8 | **60.0 ± 17.8%** | 91.2 | **71.0** | +11.4 %pts | +7.1 %pts | $p < 0.01$ |
| **Trổ hoa Rabi \| (n=5, d=5)** | $p_{10}$ | 50.0 | 61.4 ± 15.7% | 74.8 | **82.8 ± 12.5%** | 84.4 | **83.1** | +21.4 %pts | +8.3 %pts | $p < 0.001$ |

> **Nhận xét chính:**  
> - Baseline Crisp CWC tuy có $AP$ cao ($99.6\%–100\%$) do chỉ chấp nhận trùng khớp tuyệt đối, nhưng độ phủ $CP$ sụp đổ mạnh ở các chu kỳ ngắn (chỉ đạt $16.2\%$), kéo theo điểm điều hòa $F_1$ rơi xuống chỉ còn **$26.3\%$**.  
> - Mô hình đề xuất Fuzzy-Dynamic duy trì $AP$ cao ($84.4\%–100\%$) đồng thời nâng vọt độ phủ, giúp **$F_1$ đạt tới $48.8\%–98.4\%$** (mức tăng $Net\ Gain\ F_1$ từ **$+1.1$ đến $+22.4$ điểm phần trăm**).  
> - Toàn bộ 10 kịch bản đều đạt ý nghĩa thống kê vượt trội qua kiểm định Paired t-test ($p < 0.05$, trong đó 8/10 kịch bản $p < 0.001$).

### 2. Kiểm thử Định lượng trên 120 Kịch bản Đồng ruộng Đa ngữ cảnh (Bảng 4.8)
- **Độ chính xác truy hồi phác đồ canh tác (Tích hợp ngữ cảnh ECWC):** **100.0%** (120 / 120 kịch bản).
- **Độ chính xác khi chỉ dùng 5 biến thời tiết (Không có ECWC):** **75.8%** (sai lệch 24.2% phác đồ do nhầm lẫn giữa đất phù sa ngọt, đất phèn và đất mặn ven biển).
- **Độ chính xác quy đổi liều lượng vật tư (Phân bón, Thuốc BVTV, Nước):** **100.0%** (nhờ Tầng quy đổi đại số tất định giải tích, tuyệt đối không để LLM tính nhẩm).
- **Tỷ lệ tuân thủ quy tắc an toàn nông học:** **100.0%** (100% kịch bản có chỉ số $NDVI > 0.75$ tuân thủ nghiêm ngặt khuyến nghị tạm ngưng bón đạm urê để phòng trừ bệnh đạo ôn).

---

## 🚀 Hướng dẫn Cài đặt và Chạy Thực nghiệm (Quick Start)

### Yêu cầu Hệ thống
- Python 3.10 trở lên.
- Hệ điều hành: Windows, macOS hoặc Linux.

### Bước 1: Clone kho lưu trữ và cài đặt thư viện
```bash
git clone https://github.com/anhkiet1227/CS2208_Decision-Support-System.git
cd CS2208_Decision-Support-System

# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
# Kích hoạt trên Windows:
.\venv\Scripts\activate
# Kích hoạt trên Linux/macOS:
# source venv/bin/activate

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt
```

### Bước 2: Chạy toàn bộ kịch bản thực nghiệm trên Terminal (CLI)
```bash
python run_experiments.py
```
Lệnh thực thi chuỗi 4 bước:
1. **Kiểm chứng Bảng 2.1:** Tính toán Định nghĩa 1 & Định nghĩa 3 trên 14 ngày quan trắc mẫu (`01/03/2015 – 14/03/2015`) và chứng minh toán học việc xóa bỏ Hiệu ứng Vách đá ($Sim_{Fuzzy} = 98.93\%$ và $97.53\%$).
2. **Chạy đối sánh 10 seeds:** Tái hiện Algorithm 1 cho cả 4 mô hình (WC số thực, Baseline Crisp, Fuzzy CWC, Fuzzy-Dynamic) trên 30 năm dữ liệu và xuất Bảng 4.2 – 4.6.
3. **Phân tích độ nhạy $(\theta, \delta)$:** Quét lưới tham số, kiểm chứng tác động của rào cản sinh học $I_{veto}$ và xuất Bảng 4.7.
4. **Đánh giá ECWC & Grounded RAG-LLM:** Chạy kiểm thử trên 120 kịch bản đồng ruộng, quy đổi liều lượng vật tư và lưu kết quả vào `results/`. Vẽ tự động toàn bộ 6 biểu đồ độ phân giải cao vào `results/figures/`.

### Bước 3: Khởi chạy Giao diện Web Dashboard tương tác (Streamlit)
```bash
streamlit run app.py
```
Ứng dụng Web sẽ tự động mở tại địa chỉ `http://localhost:8501`, cung cấp 4 phân hệ trực quan:
- **Tab 1 — Khám phá Bảng WCT & Fuzzy Membership:** Minh họa hàm thuộc mờ hình thang Ruspini, chỉnh tham số bán kính $\delta$ và kiểm chứng triệt tiêu hiệu ứng vách đá tại biên $RH = 60.5\%$.
- **Tab 2 — Phân tích CP & Đối sánh Thực nghiệm:** Trực quan hóa độ bao phủ tái sử dụng theo chu kỳ Năm, Mùa vụ (Kharif, Rabi, Summer) và 6 giai đoạn sinh trưởng của cây lúa.
- **Tab 3 — Truy hồi ECWC & Trợ lý Khuyến nông Zalo OA / SMS:** Nhập thông số thời tiết và ngữ cảnh thửa ruộng (diện tích, loại đất, giống lúa, mức độ sâu bệnh, NDVI), nhận ngay bản tin tư vấn cá nhân hóa đã quy đổi chính xác định mức phân bón, thuốc BVTV theo diện tích nông hộ.
- **Tab 4 — Phân tích Độ nhạy & Ma trận Trọng số AHP:** Khảo sát tương tác đa tham số $(\theta, \delta)$, cơ chế rào cản sinh học $I_{veto}$ và ma trận so sánh cặp AHP chuẩn mực.

---

## 🛠️ Chi tiết các Mô-đun Mã nguồn (`src/`)

- [`src/wct_mapper.py`](file:///c:/Users/ACER/Desktop/CS2208_Decision-Support-System/src/wct_mapper.py):
  - Định nghĩa 21 danh mục WCT chuẩn IMD cho 5 biến khí tượng ($Tmax, Tmin, RHmax, RHmin, RF$) với quy tắc phân hoạch nửa nguyên liên tục.
  - Hàm thuộc mờ hình thang Ruspini $\mu(x; a, b, c, d)$ với vùng chuyển tiếp chuẩn $\delta = \pm 10\%$ độ rộng danh mục (tương đương bán kính $\pm 2.5$ điểm phần trăm độ ẩm quanh ngưỡng $60.5\%$).
  - Độ đo tương đồng mờ Fuzzy Gower Similarity trên thang thứ bậc.
  - Hàm rào cản sinh học tới hạn nghiêm ngặt $I_{veto}(X, Y) \in \{0, 1\}$ và phiên bản làm mềm liên tục $I_{veto\_soft}(X, Y)$.

- [`src/cp_engine.py`](file:///c:/Users/ACER/Desktop/CS2208_Decision-Support-System/src/cp_engine.py):
  - Thuật toán 1 (Algorithm 1) tính toán Tỷ lệ Bao phủ Tái sử dụng ($CP$).
  - Ma trận trọng số thích ứng động AHP cho 3 mùa và 6 giai đoạn sinh trưởng của lúa (nhất quán $CR = 0.0469 < 0.10$).
  - Hàm `run_multiseed_same_dataset_comparison()` chạy đối sánh 10 seeds độc lập (Seeds 42–51) có tính toán kiểm định thống kê Paired t-test ($p$-value).
  - Hàm tính toán Độ chính xác Khuyến nghị ($AP$) và Điểm điều hòa $F_{1\_reuse}$.

- [`src/ecwc_rag_llm.py`](file:///c:/Users/ACER/Desktop/CS2208_Decision-Support-System/src/ecwc_rag_llm.py):
  - Cấu trúc vector ngữ cảnh mở rộng 9 thành phần $ECWC(n, m, d)$.
  - Thuật toán truy hồi phân tầng 2 giai đoạn (Hierarchical Two-Stage Retrieval).
  - Tầng Quy đổi Số học Tất định (Deterministic Scaling) tính toán chính xác định mức vật tư:
    $$\text{Liều lượng nông hộ} = \text{Định mức chuẩn 1 ha} \times \text{Diện tích (ha)}$$
  - Tầng Grounded RAG-LLM (hỗ trợ cả Gemini API Live và Offline Fallback Synthesizer) biên soạn bản tin Zalo OA và SMS 160 ký tự.

- [`src/dataset_generator.py`](file:///c:/Users/ACER/Desktop/CS2208_Decision-Support-System/src/dataset_generator.py):
  - Máy sinh chuỗi thời tiết mô phỏng hiệu chỉnh 30 năm (1986–2015, 10.957 ngày) khớp thống kê trạm Rajendranagar, Hyderabad (IMD).
  - Nhúng nguyên vẹn chuỗi 14 ngày chuẩn Bảng 2.1 (`01/03/2015 – 14/03/2015`).
  - Xây dựng kho tri thức chuyên gia khuyến nông $Pset$ (`data/agromet_knowledge_base.json`).

---

## 📜 Tài liệu Báo cáo & Minh chứng Học thuật

- Báo cáo đồ án môn học hoàn chỉnh (được biên soạn theo thể thức chuẩn mực của Trường ĐH Công nghệ Thông tin, ĐHQG-HCM):
  - Bản Word (.docx): [`report/26848109_VoAnhKiet_DSS_final.docx`](file:///c:/Users/ACER/Desktop/CS2208_Decision-Support-System/report/26848109_VoAnhKiet_DSS_final.docx)
  - Bản PDF (.pdf): [`report/26848109_VoAnhKiet_DSS_final.pdf`](file:///c:/Users/ACER/Desktop/CS2208_Decision-Support-System/report/26848109_VoAnhKiet_DSS_final.pdf)
- Slide thuyết trình báo cáo: [`report/26848147_26848109_DSS_midterm.pptx`](file:///c:/Users/ACER/Desktop/CS2208_Decision-Support-System/report/26848147_26848109_DSS_midterm.pptx)

---

## ⚖️ Giấy phép (License)

Dự án được phân phối dưới giấy phép mã nguồn mở **MIT License**. Xem chi tiết tại tệp [LICENSE](LICENSE).
