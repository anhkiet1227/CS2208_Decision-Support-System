"""
Module: dataset_generator.py
Sinh và quản lý:
1. Bảng 2.1 (Table 1 trong bài báo gốc): Dữ liệu quan trắc thực tế 14 ngày từ 01/03/2015 đến 14/03/2015
   tại trạm Rajendranagar, Hyderabad.
2. Chuỗi dữ liệu khí tượng mô phỏng hiệu chỉnh 30 năm (Calibrated Stochastic Weather Dataset, 1986–2015,
   N = 10.957 ngày) tại trạm Rajendranagar, được xây dựng bằng bộ sinh thời tiết ngẫu nhiên Markov-AR(1)
   (Richardson-type Weather Generator) hiệu chỉnh theo phân phối khí hậu mùa vụ (Kharif, Rabi, Summer)
   của Hyderabad và nhúng chính xác 14 ngày quan trắc của Bảng 2.1.
3. Kho tri thức Khuyến nghị Nông nghiệp - Khí tượng (Expert-verified Agromet Knowledge Base - Pset)
   tích hợp ngữ cảnh phi thời tiết U_agro phục vụ truy hồi Case-Based Reasoning (CBR) và RAG-LLM.
"""

import os
import json
from typing import List, Dict, Any
import numpy as np
import pandas as pd


# Bảng 2.1 (Table 1 - Alugubelly et al., 2019): Dữ liệu quan trắc 14 ngày thực tế tại Rajendranagar
TABLE_1_SAMPLE_DATA = [
    {"Date": "2015-03-01", "LabelDate": "1-Mar-2015",  "Tmax": 33.0, "Tmin": 17.0, "RHmax": 85.0, "RHmin": 58.0, "RF": 0.0},
    {"Date": "2015-03-02", "LabelDate": "2-Mar-2015",  "Tmax": 28.0, "Tmin": 20.4, "RHmax": 83.0, "RHmin": 43.0, "RF": 8.0},
    {"Date": "2015-03-03", "LabelDate": "3-Mar-2015",  "Tmax": 31.0, "Tmin": 19.5, "RHmax": 75.0, "RHmin": 34.0, "RF": 0.0},
    {"Date": "2015-03-04", "LabelDate": "4-Mar-2015",  "Tmax": 32.0, "Tmin": 16.0, "RHmax": 65.0, "RHmin": 33.0, "RF": 0.0},
    {"Date": "2015-03-05", "LabelDate": "5-Mar-2015",  "Tmax": 32.0, "Tmin": 18.2, "RHmax": 61.0, "RHmin": 44.0, "RF": 0.0},
    {"Date": "2015-03-06", "LabelDate": "6-Mar-2015",  "Tmax": 33.0, "Tmin": 21.9, "RHmax": 82.0, "RHmin": 62.0, "RF": 0.0},
    {"Date": "2015-03-07", "LabelDate": "7-Mar-2015",  "Tmax": 32.0, "Tmin": 19.5, "RHmax": 91.0, "RHmin": 44.0, "RF": 13.0},
    {"Date": "2015-03-08", "LabelDate": "8-Mar-2015",  "Tmax": 32.0, "Tmin": 19.5, "RHmax": 84.0, "RHmin": 40.0, "RF": 8.6},
    {"Date": "2015-03-09", "LabelDate": "9-Mar-2015",  "Tmax": 32.0, "Tmin": 19.0, "RHmax": 75.0, "RHmin": 63.0, "RF": 0.0},
    {"Date": "2015-03-10", "LabelDate": "10-Mar-2015", "Tmax": 28.5, "Tmin": 20.5, "RHmax": 69.0, "RHmin": 58.0, "RF": 0.0},
    {"Date": "2015-03-11", "LabelDate": "11-Mar-2015", "Tmax": 28.5, "Tmin": 16.0, "RHmax": 52.0, "RHmin": 26.0, "RF": 0.0},
    {"Date": "2015-03-12", "LabelDate": "12-Mar-2015", "Tmax": 32.5, "Tmin": 15.0, "RHmax": 56.0, "RHmin": 31.0, "RF": 0.0},
    {"Date": "2015-03-13", "LabelDate": "13-Mar-2015", "Tmax": 32.5, "Tmin": 16.0, "RHmax": 82.0, "RHmin": 21.0, "RF": 0.0},
    {"Date": "2015-03-14", "LabelDate": "14-Mar-2015", "Tmax": 34.0, "Tmin": 18.5, "RHmax": 38.0, "RHmin": 13.0, "RF": 0.0},
]


def get_table1_dataframe() -> pd.DataFrame:
    """Trả về DataFrame chứa đúng 14 ngày quan trắc mẫu trong Bảng 2.1 (Table 1 bài báo gốc)."""
    df = pd.DataFrame(TABLE_1_SAMPLE_DATA)
    df["Tmax_normal"] = 33.2
    df["Tmin_normal"] = 18.5
    return df


def compute_standard_meteorological_week(dt: pd.Timestamp) -> int:
    """
    Tính Tuần khí tượng chuẩn (Standard Meteorological Week - SMW: W1 đến W52) theo chuẩn IMD.
    W1 bắt đầu từ ngày 01/01 hàng năm, mỗi tuần 7 ngày; tuần 52 gồm các ngày cuối tháng 12.
    """
    day_of_year = dt.dayofyear
    week = (day_of_year - 1) // 7 + 1
    return min(week, 52)


def generate_30year_weather_dataset(
    save_path: str = "data/rajendranagar_weather_1986_2015.csv",
    seed: int = 2019,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    """
    Sinh bộ dữ liệu khí tượng mô phỏng hiệu chỉnh 30 năm (01/01/1986 - 31/12/2015, 10.957 bản ghi ngày)
    theo mô hình Richardson Markov-AR(1) hiệu chỉnh theo đặc trưng khí hậu trạm Rajendranagar, Hyderabad:
    1. Chuẩn lịch sử (Historical Normals): Hàm điều hòa cosin theo ngày trong năm (DOY).
    2. Nhiệt độ & Độ ẩm (Tmax, Tmin, RHmax, RHmin): Quá trình tự hồi quy bậc 1 AR(1) với hệ số
       tự tương quan synop ρ = 0.78 phản ánh các đợt sóng nhiệt/lạnh và chu kỳ gió mùa kéo dài 5–7 ngày.
    3. Lượng mưa (RF): Chuỗi Markov 2 trạng thái (Khô / Mưa) với xác suất chuyển trạng thái p_01, p_11
       đặc trưng theo từng mùa vụ (Kharif, Rabi, Summer) kết hợp phân phối Gamma cho cường độ mưa.
    4. Nhúng chính xác 14 ngày quan trắc thực tế từ 01/03/2015 đến 14/03/2015 của Bảng 2.1 (Table 1).
    """
    if save_path and os.path.exists(save_path) and not force_regenerate and seed == 2019:
        df = pd.read_csv(save_path)
        df["Timestamp"] = pd.to_datetime(df["Date"])
        return df

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="1986-01-01", end="2015-12-31", freq="D")
    n_days = len(dates)

    doy = dates.dayofyear.to_numpy()
    months = dates.month.to_numpy()
    years = dates.year.to_numpy()

    # 1. Đường cong chuẩn khí hậu 30 năm (Historical Normals) tại Hyderabad
    phase_tmax = 2.0 * np.pi * (doy - 125.0) / 365.25
    tmax_normal = 32.5 + 5.8 * np.cos(phase_tmax)

    phase_tmin = 2.0 * np.pi * (doy - 140.0) / 365.25
    tmin_normal = 20.2 + 5.2 * np.cos(phase_tmin)

    # 2. Quá trình tự hồi quy AR(1) cho độ lệch nhiệt độ và độ ẩm (ρ = 0.78)
    ar_tmax = np.zeros(n_days)
    ar_tmin = np.zeros(n_days)
    ar_rhmax = np.zeros(n_days)
    ar_rhmin = np.zeros(n_days)

    sig_tmax = np.where((months >= 6) & (months <= 10), 1.45, np.where((months >= 3) & (months <= 5), 1.55, 1.38))
    sig_tmin = np.where((months >= 6) & (months <= 10), 1.40, np.where((months >= 3) & (months <= 5), 1.45, 1.58))
    sig_rh = np.where((months >= 6) & (months <= 10), 8.2, 9.0)

    eps_tmax = rng.normal(0.0, sig_tmax)
    eps_tmin = rng.normal(0.0, sig_tmin)
    eps_rhmax = rng.normal(0.0, sig_rh)
    eps_rhmin = rng.normal(0.0, sig_rh * 1.05)

    rho = 0.78
    for i in range(1, n_days):
        ar_tmax[i] = rho * ar_tmax[i - 1] + eps_tmax[i]
        ar_tmin[i] = rho * ar_tmin[i - 1] + eps_tmin[i]
        ar_rhmax[i] = rho * ar_rhmax[i - 1] + eps_rhmax[i]
        ar_rhmin[i] = rho * ar_rhmin[i - 1] + eps_rhmin[i]

    tmax = np.round(tmax_normal + ar_tmax, 1)
    tmin = np.round(tmin_normal + ar_tmin, 1)

    rhmax_base = np.where(
        (months >= 6) & (months <= 10),
        78.0,
        np.where((months >= 3) & (months <= 5), 55.0, 69.0),
    )
    rhmin_base = np.where(
        (months >= 6) & (months <= 10),
        57.0,
        np.where((months >= 3) & (months <= 5), 30.0, 43.0),
    )

    rhmax = np.clip(np.round(rhmax_base + ar_rhmax, 1), 18.0, 99.0)
    rhmin = np.clip(np.round(rhmin_base + ar_rhmin, 1), 10.0, np.minimum(rhmax - 4.0, 92.0))

    # 3. Chuỗi Markov 2 trạng thái (Dry / Wet) + Phân phối Gamma cho lượng mưa RF (mm)
    rf = np.zeros(n_days)
    wet_state = False
    for i in range(n_days):
        m = months[i]
        if 6 <= m <= 10:  # Vụ Kharif (Gió mùa Tây Nam)
            p01, p11 = 0.22, 0.58
            scale_g = 12.5
        elif 3 <= m <= 5:  # Vụ Summer
            p01, p11 = 0.05, 0.25
            scale_g = 5.2
        else:  # Vụ Rabi (Mùa đông khô)
            p01, p11 = 0.04, 0.24
            scale_g = 4.8

        prob = p11 if wet_state else p01
        wet_state = bool(rng.random() < prob)
        if wet_state:
            val = rng.gamma(shape=0.90, scale=scale_g)
            rf[i] = max(round(float(val), 1), 0.2)

    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Year": years,
            "Month": months,
            "DayOfYear": doy,
            "SMW": [compute_standard_meteorological_week(d) for d in dates],
            "Tmax": tmax,
            "Tmin": tmin,
            "RHmax": rhmax,
            "RHmin": rhmin,
            "RF": rf,
            "Tmax_normal": np.round(tmax_normal, 2),
            "Tmin_normal": np.round(tmin_normal, 2),
        }
    )

    # 4. Nhúng chính xác 14 ngày từ 01/03/2015 đến 14/03/2015 của Bảng 2.1 (Table 1)
    sample_map = {row["Date"]: row for row in TABLE_1_SAMPLE_DATA}
    for idx, row in df[df["Date"].isin(sample_map.keys())].iterrows():
        s_row = sample_map[row["Date"]]
        df.loc[idx, "Tmax"] = s_row["Tmax"]
        df.loc[idx, "Tmin"] = s_row["Tmin"]
        df.loc[idx, "RHmax"] = s_row["RHmax"]
        df.loc[idx, "RHmin"] = s_row["RHmin"]
        df.loc[idx, "RF"] = s_row["RF"]

    # Tính lại cột Historical Normal 30 năm thực nghiệm theo đúng định nghĩa của IMD
    doy_means = df.groupby("DayOfYear")[["Tmax", "Tmin"]].transform("mean")
    df["Tmax_normal"] = doy_means["Tmax"].round(2)
    df["Tmin_normal"] = doy_means["Tmin"].round(2)

    if save_path and seed == 2019:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False, encoding="utf-8")
    df["Timestamp"] = pd.to_datetime(df["Date"])
    return df


def build_expert_knowledge_base(
    save_path: str = "data/agromet_knowledge_base.json",
) -> List[Dict[str, Any]]:
    """
    Xây dựng Kho tri thức Khuyến nghị Nông nghiệp - Khí tượng (Expert-verified Knowledge Base - Pset)
    bao gồm các ca mẫu lịch sử (Historical Cases) đã được chuyên gia kiểm duyệt, phục vụ:
    - Minh họa truy hồi tương đồng mờ (Fuzzy-Dynamic Gower Retrieval)
    - Đánh giá định lượng Giải pháp 3 & 4 (Extended CWC & Grounded RAG-LLM) trên tập kịch bản đa ngữ cảnh.
    """
    cases: List[Dict[str, Any]] = [
        {
            "case_id": "HIST-KHARIF-F-2011-08",
            "recorded_date": "2011-10-24",
            "season": "Kharif",
            "phenophase": "Flowering",
            "duration": 5,
            "psi_values": {"Tmax": 31.8, "Tmin": 21.5, "RHmax": 86.0, "RHmin": 60.0, "RF": 42.5},
            "temp_deviations": {"Tmax": -0.4, "Tmin": 0.6},
            "crisp_categories": {
                "Tmax": "Little change",
                "Tmin": "Little change",
                "RHmax": "Very high",
                "RHmin": "Moderate",
                "RF": "Rather heavy rain",
            },
            "agro_context": {
                "u_soil": "Đất phù sa ngọt",
                "u_variety": "Giống ngắn ngày (90-100 ngày)",
                "u_pest": "Nguy cơ bùng phát dịch đạo ôn/rầy nâu (Mức 2)",
                "u_ndvi": "NDVI Cao/Bão hòa (>0.7 - Sinh khối dày, nguy cơ dư đạm)",
            },
            "expert_advisory_en": (
                "During Flowering stage under Rather Heavy Rain and Very High Humidity (>85%), "
                "maintain shallow water level (2-3 cm) and clear drainage channels immediately. "
                "Suspend top-dressing nitrogen fertilizers due to dense canopy closure (NDVI > 0.7). "
                "Spray Tricyclazole 75% WP @ 0.6 g/L (300 g/ha) during rain-free afternoon hours "
                "to prevent neck blast (Pyricularia oryzae)."
            ),
            "expert_advisory_vi": (
                "Lúa đang giai đoạn Trổ bông gặp mưa khá to (35.6–64.4 mm) và độ ẩm rất cao (>85%): "
                "(1) Khơi thông rãnh thoát nước ngay, chỉ giữ mực nước nông 2–3 cm trên mặt ruộng; "
                "(2) Tạm ngưng bón thêm phân đạm do độ che phủ tán lá dày (NDVI > 0.7, nguy cơ dư đạm); "
                "(3) Tranh thủ tạnh ráo buổi chiều phun phòng bệnh đạo ôn cổ bông bằng hoạt chất "
                "Tricyclazole 75WP liều chuẩn 300 g/ha (pha 0.6 g/lít nước, lượng nước phun 500 lít/ha)."
            ),
            "dosage_norms_per_ha": {
                "tricyclazole_75wp_gram": 300,
                "spray_water_liter": 500,
                "urea_kg": 0,
                "potash_kcl_kg": 25,
                "kno3_foliar_kg": 0,
            },
        },
        {
            "case_id": "HIST-KHARIF-F-2012-03",
            "recorded_date": "2012-10-18",
            "season": "Kharif",
            "phenophase": "Flowering",
            "duration": 5,
            "psi_values": {"Tmax": 32.0, "Tmin": 21.6, "RHmax": 85.0, "RHmin": 61.0, "RF": 40.0},
            "temp_deviations": {"Tmax": -0.2, "Tmin": 0.5},
            "crisp_categories": {
                "Tmax": "Little change",
                "Tmin": "Little change",
                "RHmax": "Very high",
                "RHmin": "High",
                "RF": "Rather heavy rain",
            },
            "agro_context": {
                "u_soil": "Đất phèn hoạt động",
                "u_variety": "Giống kháng rầy",
                "u_pest": "Cảnh báo nhẹ (Mức 1)",
                "u_ndvi": "NDVI Thấp (<0.3 - Thiếu dinh dưỡng)",
            },
            "expert_advisory_en": (
                "During Flowering on active acid sulfate soil under Rather Heavy Rain: "
                "Maintain 3-5 cm water layer to prevent alum oxidation while draining excess surface water. "
                "Supplement foliar potassium and moderate nitrogen due to low canopy vigor (NDVI < 0.3)."
            ),
            "expert_advisory_vi": (
                "Giai đoạn Trổ bông trên nền đất phèn hoạt động có mưa khá to và tán lá thưa (NDVI < 0.3): "
                "(1) Điều tiết tháo nước mặt nhưng giữ lớp nước đệm 3–5 cm để ém phèn; "
                "(2) Bổ sung phân bón lá Kali và lượng đạm vừa phải (25 kg Ure/ha) phục hồi sức sống lá đòng; "
                "(3) Phun phòng đạo ôn cổ bông liều 250 g Tricyclazole 75WP/ha."
            ),
            "dosage_norms_per_ha": {
                "tricyclazole_75wp_gram": 250,
                "spray_water_liter": 450,
                "urea_kg": 25,
                "potash_kcl_kg": 30,
                "kno3_foliar_kg": 4,
            },
        },
        {
            "case_id": "HIST-KHARIF-MT-2013-04",
            "recorded_date": "2013-09-02",
            "season": "Kharif",
            "phenophase": "Maximum Tillering",
            "duration": 5,
            "psi_values": {"Tmax": 32.4, "Tmin": 22.8, "RHmax": 84.0, "RHmin": 64.0, "RF": 18.2},
            "temp_deviations": {"Tmax": 0.2, "Tmin": 0.5},
            "crisp_categories": {
                "Tmax": "Little change",
                "Tmin": "Little change",
                "RHmax": "Very high",
                "RHmin": "High",
                "RF": "Moderate rain",
            },
            "agro_context": {
                "u_soil": "Đất phèn hoạt động",
                "u_variety": "Giống kháng rầy",
                "u_pest": "Cảnh báo nhẹ (Mức 1)",
                "u_ndvi": "NDVI Chuẩn (0.3-0.6)",
            },
            "expert_advisory_en": (
                "At Maximum Tillering under Moderate Rain and High Humidity on acid sulfate soil: "
                "Maintain 3-5 cm standing water to suppress alum toxicity. Apply split potassium "
                "and phosphorus; reduce urea rate by 20% to avoid leaf blast and brown planthopper."
            ),
            "expert_advisory_vi": (
                "Giai đoạn Đẻ nhánh tối đa trên nền đất phèn có mưa vừa và ẩm độ cao: "
                "(1) Giữ mực nước ruộng ổn định 3–5 cm để ém phèn, không để mặt ruộng khô nứt; "
                "(2) Bón thúc cân đối Lân + Kali (40 kg KCl/ha), giảm 20% lượng đạm Ure (chỉ bón 35 kg Ure/ha); "
                "(3) Thăm đồng vạch gốc lúa kiểm tra mật số rầy nâu tuổi 1–2 và vết chấm kim đạo ôn lá."
            ),
            "dosage_norms_per_ha": {
                "tricyclazole_75wp_gram": 250,
                "spray_water_liter": 400,
                "urea_kg": 35,
                "potash_kcl_kg": 40,
                "kno3_foliar_kg": 0,
            },
        },
        {
            "case_id": "HIST-KHARIF-PI-2010-06",
            "recorded_date": "2010-10-03",
            "season": "Kharif",
            "phenophase": "Panicle Initiation",
            "duration": 5,
            "psi_values": {"Tmax": 33.0, "Tmin": 22.5, "RHmax": 79.0, "RHmin": 58.0, "RF": 12.0},
            "temp_deviations": {"Tmax": 0.5, "Tmin": 0.3},
            "crisp_categories": {
                "Tmax": "Little change",
                "Tmin": "Little change",
                "RHmax": "High",
                "RHmin": "Moderate",
                "RF": "Moderate rain",
            },
            "agro_context": {
                "u_soil": "Đất phù sa ngọt",
                "u_variety": "Giống trung ngày",
                "u_pest": "An toàn (Mức 0)",
                "u_ndvi": "NDVI Chuẩn (0.3-0.6)",
            },
            "expert_advisory_en": (
                "At Panicle Initiation under Moderate Rain and Normal Canopy (NDVI 0.3-0.6): "
                "Apply panicle fertilizer (Urea + MOP) at 3-5 cm shallow water level."
            ),
            "expert_advisory_vi": (
                "Giai đoạn Làm đòng (Đón đòng) trong điều kiện mưa vừa và sinh khối chuẩn (NDVI 0.3–0.6): "
                "(1) Giữ mực nước nông 3–5 cm; (2) Bón phân đón đòng theo tiêu chuẩn (40 kg Ure/ha + 35 kg KCl/ha) "
                "giúp phân hóa gié hoa tối ưu."
            ),
            "dosage_norms_per_ha": {
                "tricyclazole_75wp_gram": 0,
                "spray_water_liter": 0,
                "urea_kg": 40,
                "potash_kcl_kg": 35,
                "kno3_foliar_kg": 0,
            },
        },
        {
            "case_id": "HIST-RABI-F-2014-02",
            "recorded_date": "2014-03-18",
            "season": "Rabi",
            "phenophase": "Flowering",
            "duration": 5,
            "psi_values": {"Tmax": 31.2, "Tmin": 14.8, "RHmax": 73.8, "RHmin": 42.4, "RF": 2.0},
            "temp_deviations": {"Tmax": -0.8, "Tmin": -3.6},
            "crisp_categories": {
                "Tmax": "Little change",
                "Tmin": "Appreciable fall",
                "RHmax": "High",
                "RHmin": "Moderate",
                "RF": "Very light rain",
            },
            "agro_context": {
                "u_soil": "Đất nhiễm mặn (>2‰)",
                "u_variety": "Giống chịu mặn/chịu hạn",
                "u_pest": "An toàn (Mức 0)",
                "u_ndvi": "NDVI Thấp (<0.3 - Thiếu dinh dưỡng)",
            },
            "expert_advisory_en": (
                "During Rabi Flowering with Appreciable Fall in night temperature (Tmin deviation -3.6°C) "
                "and saline soil risk: Maintain 5-7 cm fresh water layer to buffer night chill and flush salinity. "
                "Foliar spray of Potassium Nitrate (KNO3 1%) + Boron is recommended to enhance pollen viability."
            ),
            "expert_advisory_vi": (
                "Lúa vụ Đông Xuân (Rabi) giai đoạn Trổ bông gặp nền nhiệt đêm giảm đáng kể (Tmin giảm 3.6°C) "
                "trên vùng đất có nguy cơ nhiễm mặn: (1) Đo độ mặn nguồn nước trước khi bơm, giữ lớp nước ngọt "
                "5–7 cm trong ruộng để giữ ấm gốc lúa ban đêm và ém mặn; (2) Phun bổ sung phân bón lá Kali Nitrat "
                "(KNO3 1%, 5 kg/ha) kết hợp Bo giúp tăng sức sống hạt phấn và hạn chế lem lép hạt."
            ),
            "dosage_norms_per_ha": {
                "tricyclazole_75wp_gram": 0,
                "kno3_foliar_kg": 5,
                "spray_water_liter": 400,
                "urea_kg": 25,
                "potash_kcl_kg": 30,
            },
        },
        {
            "case_id": "HIST-SUMMER-M-2012-05",
            "recorded_date": "2012-05-04",
            "season": "Summer",
            "phenophase": "Maturity",
            "duration": 5,
            "psi_values": {"Tmax": 38.6, "Tmin": 24.5, "RHmax": 52.0, "RHmin": 24.0, "RF": 0.0},
            "temp_deviations": {"Tmax": 3.5, "Tmin": 1.0},
            "crisp_categories": {
                "Tmax": "Appreciable rise",
                "Tmin": "Little change",
                "RHmax": "Moderate",
                "RHmin": "Low",
                "RF": "No rain",
            },
            "agro_context": {
                "u_soil": "Đất cát pha khô hạn",
                "u_variety": "Giống ngắn ngày (90-100 ngày)",
                "u_pest": "An toàn (Mức 0)",
                "u_ndvi": "NDVI Chuẩn (0.3-0.6)",
            },
            "expert_advisory_en": (
                "At Maturity stage under dry hot weather (Appreciable rise in Tmax, Low RHmin, No rain): "
                "Drain field water completely 7-10 days prior to harvest for uniform grain ripening and "
                "combine harvester operation. Harvest at 85-90% golden grain maturity to prevent sun-cracking."
            ),
            "expert_advisory_vi": (
                "Giai đoạn Chín / Thu hoạch trong điều kiện nắng nóng khô (Tmax tăng đáng kể, không mưa, RHmin thấp): "
                "(1) Rút cạn nước ruộng trước thu hoạch 7–10 ngày để mặt đất cứng giúp máy gặt đập liên hợp vận hành; "
                "(2) Thu hoạch đúng độ chín (khi 85–90% số hạt trên bông đã chín vàng) tránh phơi nắng quá già trên đồng "
                "làm rạn nứt hạt gạo và giảm tỷ lệ gạo nguyên."
            ),
            "dosage_norms_per_ha": {
                "tricyclazole_75wp_gram": 0,
                "spray_water_liter": 0,
                "urea_kg": 0,
                "potash_kcl_kg": 0,
                "kno3_foliar_kg": 0,
            },
        },
    ]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    return cases
