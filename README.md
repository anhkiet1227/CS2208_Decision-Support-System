# CS2208 - Hệ Hỗ Trợ Quyết Định (Decision Support Systems)

## Đề tài: Phân tích Điều kiện Thời tiết Tương đồng nhằm Nâng cao Khả năng Tái sử dụng trong Hệ Hỗ trợ Ra quyết định Nông nghiệp
**Tên tiếng Anh:** *Analysis of Similar Weather Conditions to Improve Reuse in Weather-Based Decision Support Systems*

- **Học viên thực hiện:** Võ Anh Kiệt (MSHV: `26848109`)
- **Giảng viên hướng dẫn:** TS. Tôn Thất Hòa An
- **Trường:** Đại học Công Nghệ Thông Tin - ĐHQG TP.HCM (UIT - VNU-HCM)
- **Kho lưu trữ mã nguồn (GitHub Repository):** https://github.com/anhkiet1227/CS2208_Decision-Support-System
- **Bài báo gốc (Q1, ISI/Scopus):** Mamatha Alugubelly, Krishna Reddy Polepalli, Sreenivas Gade, & Seishi Ninomiya (2019). *Analysis of similar weather conditions to improve reuse in weather-based decision support systems*. Computers and Electronics in Agriculture, Vol. 157, pp. 154–165.

---

## 1. Cấu trúc Thư mục Dự án (Project Structure)

```text
CS2208_Decision-Support-System/
├── src/
│   ├── __init__.py
│   ├── wct_mapper.py          # Mô-đun 1: Ánh xạ WC số thực, Crisp WCT (IMD) & Fuzzy Trapezoidal WCT
│   ├── cp_engine.py           # Mô-đun 2: Thuật toán 1 (Algorithm 1), Trọng số động & Độ đo mềm Gower
│   ├── ecwc_rag_llm.py        # Mô-đun 3: Vector ngữ cảnh mở rộng ECWC (9-tuple) & Grounded RAG-LLM
│   └── dataset_generator.py   # Bộ sinh dữ liệu 30 năm (1986-2015), Bảng 2.1 & Kho tri thức chuyên gia Pset
├── data/
│   ├── rajendranagar_weather_1986_2015.csv   # Chuỗi 10.957 bản ghi thời tiết ngày (1986-2015)
│   └── agromet_knowledge_base.json           # Kho tri thức khuyến nghị chuyên gia đã kiểm duyệt (Pset)
├── results/
│   ├── figures/
│   │   ├── fig4_2_code_execution_plots.png   # Biểu đồ kiểm chứng Hiệu ứng Vách đá & Hội tụ CP 30 năm
│   │   └── fig4_3_streamlit_zalo_demo.png    # Giao diện Hệ thống Khuyến nông & Zalo OA Chatbot
│   ├── table_4_2.csv ... table_4_6.csv       # Các bảng số liệu thực nghiệm xuất tự động
│   └── rag_llm_demo_output.json              # Log kết quả truy hồi ECWC và bản tin RAG-LLM
├── report/
│   ├── 26848109_VoAnhKiet_DSS_final.docx     # Báo cáo đồ án môn học hoàn chỉnh (đã tích hợp Code Demo Mục 4.4)
│   └── 26848147_26848109_DSS_midterm.pptx    # Slide báo cáo giữa kỳ
├── run_experiments.py         # Script CLI chạy toàn bộ thực nghiệm, kiểm chứng toán học & xuất biểu đồ
├── app.py                     # Ứng dụng Web Dashboard tương tác (Streamlit) & Mô phỏng Zalo OA Chatbot
└── requirements.txt           # Danh sách thư viện Python
```

---

## 2. Hướng dẫn Cài đặt và Thực thi (Quick Start)

### Bước 1: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy toàn bộ kịch bản thực nghiệm trên Terminal (CLI)
```bash
python run_experiments.py
```
Lệnh này thực hiện:
1. Kiểm chứng **Định nghĩa 1** và **Định nghĩa 3** trên 14 ngày dữ liệu chuẩn Bảng 2.1 (`01/03/2015 – 14/03/2015`) và chứng minh việc triệt tiêu **Hiệu ứng vách đá (Boundary Cliff Effect)** bằng hàm thuộc mờ hình thang $\mu(V)$.
2. Thực thi **Thuật toán 1 (Algorithm 1)** trên chuỗi dữ liệu 30 năm (`1986–2015`, 10.957 bản ghi ngày) và xuất toàn bộ bảng đối sánh **Bảng 4.2 – Bảng 4.6** vào thư mục `results/`.
3. Thực thi truy hồi **Extended CWC (9 thành phần)** kết hợp **Grounded RAG-LLM** để sinh bản tin khuyến nông cá nhân hóa (Zalo OA & SMS 160 ký tự) quy đổi sẵn định mức phân bón/thuốc BVTV theo diện tích canh tác.

### Bước 3: Khởi chạy Giao diện Web Demo tương tác (Streamlit)
```bash
streamlit run app.py
```
- Hỗ trợ cả **Chế độ Local Grounded RAG Synthesizer** (chạy mượt ngay lập tức không cần API Key) và **Chế độ Gemini API Live** (khi nhập `GEMINI_API_KEY` trên thanh công cụ).
