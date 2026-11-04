"""
Module 2: cp_engine.py
Độ đo Tỷ lệ Bao phủ (Coverage Percentage - CP), Độ chính xác Tái sử dụng Khuyến nghị (Advisory Precision - AP),
Thuật toán 1 (Algorithm 1 - Alugubelly et al., 2019) và Giải pháp Đề xuất 1 & 2
(Weighted Fuzzy Gower Similarity + Dynamic Feature Weighting + Biological Threshold Constraints).

Thực hiện:
- Định nghĩa 2: So khớp chính xác điều kiện thời tiết số thực WC(n, d).
- Định nghĩa 4: So khớp chính xác điều kiện thời tiết danh mục cứng Baseline CWC(n, d).
- Định nghĩa 5: Trích xuất tập chu kỳ S = <p_1, p_2, ..., p_k> theo Loại chu kỳ (Period Type - pt).
- Định nghĩa 6 & Thuật toán 1: Tính chuỗi CP(p_i / Pset) cho p_i = 2..k.
- Giải pháp đề xuất 1 & 2:
  + Hàm tương đồng mềm Gower kết hợp chỉ số cấp bậc mờ Rank_Fuzzy(X_k).
  + Ràng buộc không bù trừ bậc thứ tự (Non-compensatory Ordinal Veto I_veto) và
    Ràng buộc ngưỡng sinh học tới hạn (Critical Biological Regime Barrier I_bio).
  + Đánh giá chất lượng tái sử dụng bằng Độ chính xác Khuyến nghị (Advisory Precision - AP %)
    và điểm điều hòa F1_reuse = 2 * CP * AP / (CP + AP).
  + Thực nghiệm đa hạt giống ngẫu nhiên (R = 10 seeds) và phân tích độ nhạy tham số (θ, δ).
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from src.wct_mapper import (
    WCTMapper,
    WeatherCondition,
    CategoryWeatherCondition,
    WEATHER_VARS,
    BASELINE_VAR_COMBINATIONS,
)
from src.dataset_generator import generate_30year_weather_dataset


# Bảng 2.4 (Table 4 - Alugubelly et al., 2019): Tuần bắt đầu và kết thúc của 6 giai đoạn sinh trưởng lúa
RICE_PHENOPHASE_WEEKS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "Kharif": {
        "Sowing": (27, 32),             # 6 tuần
        "Transplanting": (32, 36),      # 5 tuần
        "Maximum Tillering": (35, 38),  # 4 tuần
        "Panicle Initiation": (39, 44), # 6 tuần
        "Flowering": (43, 48),          # 6 tuần
        "Maturity": (48, 52),           # 4 tuần
    },
    "Rabi": {
        "Sowing": (50, 1),              # 4 tuần (W50 năm trước đến W1 năm sau)
        "Transplanting": (1, 4),        # 4 tuần
        "Maximum Tillering": (5, 9),    # 5 tuần
        "Panicle Initiation": (9, 12),  # 4 tuần
        "Flowering": (12, 16),          # 5 tuần
        "Maturity": (16, 20),           # 5 tuần
    },
}


# Bảng 3.1: Ma trận lựa chọn biến chủ đạo V_active và phân bổ trọng số thích ứng động w(pt, φ)
# Được xác định thông qua quy trình Phân tích Thứ bậc (AHP, CR < 0.05) từ tài liệu sinh lý - bệnh học
# lúa (Mousanejad et al., 2009; Asseng et al., 2011; IMD Forecaster's Guide) kết hợp kiểm định độ nhạy.
# Lưu ý: Summer là chu kỳ mùa vụ độc lập (Bảng 2.3), trong khi Kharif và Rabi có đủ 6 giai đoạn (Bảng 2.4).
DYNAMIC_WEIGHT_MATRIX: Dict[str, Dict[str, Any]] = {
    "Year_Default": {
        "active_3vars": ["Tmax", "RHmax", "RF"],
        "weights": {"Tmax": 0.25, "Tmin": 0.15, "RHmax": 0.25, "RHmin": 0.10, "RF": 0.25},
    },
    "Kharif_Default": {
        "active_3vars": ["RF", "RHmax", "Tmax"],
        "weights": {"Tmax": 0.15, "Tmin": 0.05, "RHmax": 0.25, "RHmin": 0.10, "RF": 0.45},
    },
    "Kharif_Sowing": {
        "active_3vars": ["RF", "RHmax", "Tmax"],
        "weights": {"Tmax": 0.15, "Tmin": 0.05, "RHmax": 0.25, "RHmin": 0.05, "RF": 0.50},
    },
    "Kharif_Transplanting": {
        "active_3vars": ["RF", "RHmax", "Tmax"],
        "weights": {"Tmax": 0.15, "Tmin": 0.05, "RHmax": 0.25, "RHmin": 0.05, "RF": 0.50},
    },
    "Kharif_Maximum Tillering": {
        "active_3vars": ["RF", "RHmax", "RHmin"],
        "weights": {"Tmax": 0.10, "Tmin": 0.10, "RHmax": 0.30, "RHmin": 0.15, "RF": 0.35},
    },
    "Kharif_Panicle Initiation": {
        "active_3vars": ["RF", "RHmax", "RHmin"],
        "weights": {"Tmax": 0.10, "Tmin": 0.10, "RHmax": 0.30, "RHmin": 0.15, "RF": 0.35},
    },
    "Kharif_Flowering": {
        "active_3vars": ["RF", "Tmax", "RHmax"],
        "weights": {"Tmax": 0.25, "Tmin": 0.10, "RHmax": 0.25, "RHmin": 0.05, "RF": 0.35},
    },
    "Kharif_Maturity": {
        "active_3vars": ["RF", "Tmax", "RHmin"],
        "weights": {"Tmax": 0.30, "Tmin": 0.05, "RHmax": 0.10, "RHmin": 0.20, "RF": 0.35},
    },
    "Rabi_Default": {
        "active_3vars": ["Tmin", "Tmax", "RHmax"],
        "weights": {"Tmax": 0.22, "Tmin": 0.43, "RHmax": 0.20, "RHmin": 0.10, "RF": 0.05},
    },
    "Rabi_Sowing": {
        "active_3vars": ["Tmin", "RHmin", "Tmax"],
        "weights": {"Tmax": 0.20, "Tmin": 0.45, "RHmax": 0.15, "RHmin": 0.15, "RF": 0.05},
    },
    "Rabi_Transplanting": {
        "active_3vars": ["Tmin", "RHmin", "Tmax"],
        "weights": {"Tmax": 0.20, "Tmin": 0.45, "RHmax": 0.15, "RHmin": 0.15, "RF": 0.05},
    },
    "Rabi_Maximum Tillering": {
        "active_3vars": ["Tmin", "RHmin", "Tmax"],
        "weights": {"Tmax": 0.20, "Tmin": 0.45, "RHmax": 0.15, "RHmin": 0.15, "RF": 0.05},
    },
    "Rabi_Panicle Initiation": {
        "active_3vars": ["Tmin", "Tmax", "RHmax"],
        "weights": {"Tmax": 0.25, "Tmin": 0.40, "RHmax": 0.25, "RHmin": 0.05, "RF": 0.05},
    },
    "Rabi_Flowering": {
        "active_3vars": ["Tmin", "Tmax", "RHmax"],
        "weights": {"Tmax": 0.25, "Tmin": 0.40, "RHmax": 0.25, "RHmin": 0.05, "RF": 0.05},
    },
    "Rabi_Maturity": {
        "active_3vars": ["Tmax", "RHmin", "Tmin"],
        "weights": {"Tmax": 0.40, "Tmin": 0.20, "RHmax": 0.10, "RHmin": 0.20, "RF": 0.10},
    },
    "Summer_Default": {
        "active_3vars": ["Tmax", "RHmin", "RF"],
        "weights": {"Tmax": 0.45, "Tmin": 0.10, "RHmax": 0.10, "RHmin": 0.25, "RF": 0.10},
    },
}


def get_dynamic_weights_and_vars(
    period_type: str, phenophase: Optional[str] = None, n_vars: int = 5
) -> Tuple[List[str], Dict[str, float]]:
    """
    Trả về tập biến chủ đạo và vector trọng số động chuẩn hóa (∑ w_k = 1) theo Bảng 3.1.
    """
    key = f"{period_type}_{phenophase}" if phenophase else f"{period_type}_Default"
    if key not in DYNAMIC_WEIGHT_MATRIX:
        key = f"{period_type}_Default" if f"{period_type}_Default" in DYNAMIC_WEIGHT_MATRIX else "Year_Default"

    profile = DYNAMIC_WEIGHT_MATRIX[key]
    raw_weights: Dict[str, float] = profile["weights"]

    if n_vars == 3:
        selected_vars = list(profile["active_3vars"])
    elif n_vars < 5:
        sorted_vars = sorted(WEATHER_VARS, key=lambda v: raw_weights[v], reverse=True)
        selected_vars = sorted_vars[:n_vars]
    else:
        selected_vars = list(WEATHER_VARS)

    sub_total = sum(raw_weights[v] for v in selected_vars)
    norm_weights = {v: raw_weights[v] / sub_total for v in selected_vars}
    return selected_vars, norm_weights


def check_biological_regime_barrier(
    cwc_x: CategoryWeatherCondition,
    cwc_y: CategoryWeatherCondition,
    period_type: str = "Year",
    target_vars: Optional[List[str]] = None,
) -> bool:
    """
    Kiểm tra Ràng buộc Ngưỡng Sinh học Tới hạn I_bio(X, Y) ∈ {0, 1} (Mục 3.2):
    Ngăn chặn việc làm mượt ranh giới hoặc bù trừ tuyến tính xóa nhòa các ranh giới sinh học thực:
    1. Ngưỡng bùng phát nấm đạo ôn (Mousanejad et al., 2009): RHmax > 80.5% kết hợp có mưa RF > 2.45 mm.
    2. Ngưỡng ngập úng / tháo nước khẩn cấp: RF > 35.55 mm (Rather heavy rain trở lên).
    3. Ngưỡng mưa trái mùa vụ Rabi (khi w_RF = 0.05 rất nhỏ): Chặn ghép cặp ngày khô (RF <= 2.45 mm)
       với ngày mưa vừa/to trái mùa (RF > 7.55 mm).
    4. Ngưỡng sốc nhiệt nóng lúc trổ bông (ΔTmax > 2.5°C) và rét hại ban đêm vụ Đông Xuân (ΔTmin < -2.5°C).
    """
    vars_set = set(target_vars) if target_vars is not None else set(WEATHER_VARS)

    # 1. Ngưỡng sinh học bệnh đạo ôn (RHmax > 80.5% & RF > 2.45 mm)
    if "RHmax" in vars_set and "RF" in vars_set:
        blast_x = (cwc_x.psi_values.get("RHmax", 0.0) > 80.5) and (cwc_x.psi_values.get("RF", 0.0) > 2.45)
        blast_y = (cwc_y.psi_values.get("RHmax", 0.0) > 80.5) and (cwc_y.psi_values.get("RF", 0.0) > 2.45)
        if blast_x != blast_y:
            near_rh = (78.0 <= cwc_x.psi_values.get("RHmax", 0.0) <= 83.0) and (
                78.0 <= cwc_y.psi_values.get("RHmax", 0.0) <= 83.0
            )
            if not near_rh:
                return False

    # 2. Ngưỡng tháo nước mưa lớn (RF > 35.55 mm) và mưa trái mùa Rabi (RF > 7.55 mm)
    if "RF" in vars_set:
        rf_x = cwc_x.psi_values.get("RF", 0.0)
        rf_y = cwc_y.psi_values.get("RF", 0.0)
        if (rf_x > 35.55) != (rf_y > 35.55):
            if not ((32.5 <= rf_x <= 38.5) and (32.5 <= rf_y <= 38.5)):
                return False
        if period_type == "Rabi" and ((rf_x > 7.55) != (rf_y > 7.55)):
            if not ((6.9 <= rf_x <= 8.2) and (6.9 <= rf_y <= 8.2)):
                return False

    # 3. Ngưỡng sốc nhiệt nóng (ΔTmax > 2.5°C)
    if "Tmax" in vars_set:
        dt_x = cwc_x.temp_deviations.get("Tmax", 0.0)
        dt_y = cwc_y.temp_deviations.get("Tmax", 0.0)
        if (dt_x > 2.5) != (dt_y > 2.5):
            if not ((2.15 <= dt_x <= 2.85) and (2.15 <= dt_y <= 2.85)):
                return False

    # 4. Ngưỡng lạnh hại đêm (ΔTmin < -2.5°C)
    if "Tmin" in vars_set:
        dt_min_x = cwc_x.temp_deviations.get("Tmin", 0.0)
        dt_min_y = cwc_y.temp_deviations.get("Tmin", 0.0)
        if (dt_min_x < -2.5) != (dt_min_y < -2.5):
            if not ((-2.85 <= dt_min_x <= -2.15) and (-2.85 <= dt_min_y <= -2.15)):
                return False

    return True


def compute_fuzzy_gower_similarity(
    cwc_x: CategoryWeatherCondition,
    cwc_y: CategoryWeatherCondition,
    weights: Optional[Dict[str, float]] = None,
    subset_vars: Optional[List[str]] = None,
    enforce_non_compensatory: bool = False,
    primary_vars: Optional[List[str]] = None,
    max_primary_rank_diff: float = 0.85,
    max_secondary_rank_diff: float = 1.25,
    period_type: str = "Year",
    enforce_bio_barrier: bool = False,
) -> float:
    """
    Công thức Độ tương đồng mềm Gower kết hợp chỉ số cấp bậc mờ và Ràng buộc Không bù trừ (Mục 3.2):
        Sim_Fuzzy(X, Y) = I_veto(X, Y) * I_bio(X, Y) * ∑_{k=1..n} w_k * [ 1 - |Rank_Fuzzy(X_k) - Rank_Fuzzy(Y_k)| / (M_k - 1) ]

    Trong đó:
    - Nếu enforce_non_compensatory = False: Tính thuần túy trung bình có trọng số Gower theo công thức cơ sở.
    - Nếu enforce_non_compensatory = True:
      + I_veto(X, Y) = 0 nếu bất kỳ biến chủ đạo nào lệch quá `max_primary_rank_diff` (ngăn việc ghép 2 lõi
        danh mục khác nhau khi không nằm trong vùng chuyển tiếp mờ) hoặc biến phụ lệch quá `max_secondary_rank_diff`
        (ngăn hiện tượng bù trừ tuyến tính khi biến có trọng số thấp w_k = 0.05 lệch nhiều cấp).
      + I_bio(X, Y) = 0 nếu vi phạm ngưỡng sinh học tới hạn (khi enforce_bio_barrier = True).
    """
    target_vars = subset_vars if subset_vars is not None else cwc_x.variables
    if not target_vars:
        return 1.0

    if enforce_bio_barrier:
        if not check_biological_regime_barrier(cwc_x, cwc_y, period_type=period_type, target_vars=target_vars):
            return 0.0

    if weights is None:
        w_map = {v: 1.0 / len(target_vars) for v in target_vars}
    else:
        total_w = sum(weights.get(v, 1.0) for v in target_vars)
        w_map = {v: weights.get(v, 1.0) / total_w for v in target_vars}

    prim_set = set(primary_vars) if primary_vars is not None else set(target_vars)

    sim_score = 0.0
    for v in target_vars:
        rx = cwc_x.fuzzy_ranks[v]
        ry = cwc_y.fuzzy_ranks[v]
        diff = abs(rx - ry)

        if enforce_non_compensatory:
            limit = max_primary_rank_diff if v in prim_set else max_secondary_rank_diff
            if diff > limit:
                return 0.0

        max_r = max(cwc_x.max_ranks.get(v, 1), 1)
        var_sim = 1.0 - (diff / float(max_r))
        sim_score += w_map[v] * var_sim

    return round(float(sim_score), 4)


def derive_agronomic_advisory_oracle(
    cwc: CategoryWeatherCondition,
    subset_vars: Optional[List[str]] = None,
) -> Tuple[str, ...]:
    """
    Hàm chuyên gia đối chứng (Ground-truth Agronomic Decision Oracle A*(x)):
    Xác định vector quyết định khuyến nghị nông học chuẩn của mẫu thời tiết x dựa trên
    các ngưỡng sinh lý thực vật và bệnh học nông nghiệp thực tế (phục vụ tính chỉ số AP %):
    1. Quyết định quản lý nước (Water Action - phụ thuộc RF, RHmin)
    2. Quyết định phòng trừ bệnh hại (Disease Action - phụ thuộc RHmax, RF)
    3. Quyết định điều hòa nhiệt (Thermal Action - phụ thuộc Tmax, Tmin)
    """
    vars_set = set(subset_vars) if subset_vars is not None else set(cwc.variables)
    actions: List[str] = []

    # 1. Thermal Action
    if "Tmax" in vars_set:
        dt_max = cwc.temp_deviations.get("Tmax", 0.0)
        if dt_max > 2.5:
            actions.append("Heat_Stress_Protect")
        elif dt_max < -2.5:
            actions.append("Cool_Day_Regime")
        else:
            actions.append("Normal_Day_Temp")

    if "Tmin" in vars_set:
        dt_min = cwc.temp_deviations.get("Tmin", 0.0)
        if dt_min < -2.5:
            actions.append("Cold_Night_Water_Buffer")
        elif dt_min > 2.5:
            actions.append("Warm_Night_Resp_Watch")
        else:
            actions.append("Normal_Night_Temp")

    # 2. Humidity / Fungal Action
    if "RHmax" in vars_set:
        rhmax = cwc.psi_values.get("RHmax", 70.0)
        if rhmax > 80.5:
            actions.append("Spore_Alert_High_RH")
        elif rhmax > 60.5:
            actions.append("Moderate_RH_Watch")
        else:
            actions.append("Dry_Air_Regime")

    # 3. Water / Rainfall Action
    if "RF" in vars_set:
        rf = cwc.psi_values.get("RF", 0.0)
        if rf > 35.55:
            actions.append("Emergency_Drainage")
        elif rf > 7.55:
            actions.append("Suspend_Spray_Keep_Shallow")
        elif rf > 0.0:
            actions.append("Light_Moisture_Credit")
        else:
            actions.append("Irrigate_Schedule")

    return tuple(actions)


class CPEngine:
    """
    Bộ máy tính toán Tỷ lệ Bao phủ (Coverage Percentage - CP) và Độ chính xác Khuyến nghị (Advisory Precision - AP):
    1. Mô hình số thực WC(n, d) (Định nghĩa 2)
    2. Mô hình danh mục cứng Baseline Crisp CWC(n, d) (Thuật toán 1 - Alugubelly et al., 2019)
    3. Mô hình cải tiến Fuzzy CWC (Giải pháp 1: Làm mượt ranh giới bằng hàm hình thang + Gower đẳng trọng)
    4. Mô hình cải tiến toàn diện Fuzzy-Dynamic CWC (Giải pháp 1 + 2: Gower mờ + Trọng số động Bảng 3.1
       + Ràng buộc Không bù trừ & Ngưỡng sinh học tới hạn).
    """

    def __init__(self, mapper: Optional[WCTMapper] = None, theta: float = 0.85):
        self.mapper = mapper if mapper is not None else WCTMapper()
        self.theta = theta

    @staticmethod
    def split_periods(
        df: pd.DataFrame, period_type: str, phenophase: Optional[str] = None
    ) -> List[Tuple[str, pd.DataFrame]]:
        """
        Tách chuỗi dữ liệu 30 năm D thành dãy k chu kỳ S = <p_1, p_2, ..., p_k> theo Định nghĩa 5.
        """
        years = sorted(df["Year"].unique())
        periods: List[Tuple[str, pd.DataFrame]] = []

        if period_type == "Year":
            for y in years:
                sub = df[df["Year"] == y].reset_index(drop=True)
                if not sub.empty:
                    periods.append((f"Year_{y}", sub))

        elif period_type == "Kharif" and phenophase is None:
            for y in years:
                sub = df[(df["Year"] == y) & (df["Month"] >= 6) & (df["Month"] <= 10)].reset_index(drop=True)
                if not sub.empty:
                    periods.append((f"Kharif_{y}", sub))

        elif period_type == "Summer" and phenophase is None:
            for y in years:
                sub = df[(df["Year"] == y) & (df["Month"] >= 3) & (df["Month"] <= 5)].reset_index(drop=True)
                if not sub.empty:
                    periods.append((f"Summer_{y}", sub))

        elif period_type == "Rabi" and phenophase is None:
            for y in years[:-1]:
                cond_prev = (df["Year"] == y) & (df["Month"] >= 11)
                cond_next = (df["Year"] == y + 1) & (df["Month"] <= 3)
                sub = df[cond_prev | cond_next].reset_index(drop=True)
                if not sub.empty:
                    periods.append((f"Rabi_{y}_{y+1}", sub))

        elif phenophase is not None:
            w_start, w_end = RICE_PHENOPHASE_WEEKS[period_type][phenophase]
            if period_type == "Kharif":
                for y in years:
                    sub = df[
                        (df["Year"] == y) & (df["SMW"] >= w_start) & (df["SMW"] <= w_end)
                    ].reset_index(drop=True)
                    if not sub.empty:
                        periods.append((f"Kharif_{phenophase}_{y}", sub))
            elif period_type == "Rabi":
                for y in years[:-1]:
                    if w_start > w_end:
                        c1 = (df["Year"] == y) & (df["SMW"] >= w_start)
                        c2 = (df["Year"] == y + 1) & (df["SMW"] <= w_end)
                        sub = df[c1 | c2].reset_index(drop=True)
                    else:
                        sub = df[
                            (df["Year"] == y + 1) & (df["SMW"] >= w_start) & (df["SMW"] <= w_end)
                        ].reset_index(drop=True)
                    if not sub.empty:
                        periods.append((f"Rabi_{phenophase}_{y}_{y+1}", sub))
        else:
            raise ValueError(f"Không nhận diện được cấu hình chu kỳ: {period_type} / {phenophase}")

        return periods

    def run_algorithm_1_all_modes(
        self,
        df: pd.DataFrame,
        period_type: str,
        phenophase: Optional[str] = None,
        n_vars: int = 5,
        duration: int = 5,
        theta: Optional[float] = None,
        enforce_bio_constraints: bool = True,
        max_periods: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Thực thi đồng thời Thuật toán 1 (Algorithm 1) cho 4 chế độ trên cùng tập dữ liệu D,
        đồng thời đo lường cả Tỷ lệ Bao phủ (CP %) và Độ chính xác Khuyến nghị (Advisory Precision - AP %):
        1. Real_WC_CP: So khớp số thực chính xác WC(n, d) (Định nghĩa 2)
        2. Crisp_CWC_CP & Crisp_CWC_AP: So khớp danh mục cứng Baseline CWC(n, d) (Định nghĩa 4)
        3. Fuzzy_CWC_CP & Fuzzy_CWC_AP: So khớp mờ hình thang + Gower đẳng trọng trên baseline_vars (Giải pháp 1)
        4. Fuzzy_Dynamic_CWC_CP & Fuzzy_Dynamic_CWC_AP: So khớp mờ hình thang + Trọng số động Bảng 3.1
           + Ràng buộc Không bù trừ & Ngưỡng sinh học tới hạn (Giải pháp 1 + 2).
        """
        sim_threshold = theta if theta is not None else self.theta
        periods = self.split_periods(df, period_type, phenophase)
        if max_periods is not None:
            periods = periods[:max_periods]

        baseline_vars = BASELINE_VAR_COMBINATIONS[n_vars]
        _, raw_dyn_weights = get_dynamic_weights_and_vars(period_type, phenophase, 5)
        dyn_vars = list(baseline_vars)
        sub_w_sum = sum(raw_dyn_weights[v] for v in dyn_vars)
        dyn_weights = {v: raw_dyn_weights[v] / sub_w_sum for v in dyn_vars}

        # Xác định nhóm biến chủ đạo (Primary Active Variables) có trọng số mùa vụ cao nhất
        sorted_by_w = sorted(dyn_vars, key=lambda v: dyn_weights[v], reverse=True)
        n_primary = max(1, len(dyn_vars) - 1) if len(dyn_vars) <= 3 else 3
        primary_vars_active = sorted_by_w[:n_primary]
        equal_weights = {v: 1.0 / len(baseline_vars) for v in baseline_vars}

        # Ngưỡng sai lệch bậc tối đa phụ thuộc vào θ và độ rộng vùng chuyển tiếp δ:
        # Biến chủ đạo bắt buộc phải nằm trong cùng danh mục hoặc trong vùng chuyển tiếp mờ (|ΔRank| < 1.0).
        slack = max(0.02, 1.0 - sim_threshold)
        delta_val = self.mapper.transition_ratio
        tau_fuzzy_equal = min(0.94, 3.5 * slack + 1.2 * delta_val)
        tau_dyn_primary = min(0.96, 3.8 * slack + 1.4 * delta_val)
        tau_dyn_secondary = min(1.65, 0.65 + 2.0 * slack + 2.2 * delta_val)

        # Bước 1 & 2 (Lines 1-6 của Algorithm 1): Tính WC(p_i) và CWC(p_i) cho mọi p_i ∈ S
        period_cwcs: List[Tuple[str, List[CategoryWeatherCondition]]] = []
        cur_id = 1
        for p_name, p_df in periods:
            _, cwcs = self.mapper.extract_period_cwcs(
                period_df=p_df,
                duration=duration,
                variables=WEATHER_VARS,
                start_id=cur_id,
            )
            period_cwcs.append((p_name, cwcs))
            cur_id += len(cwcs)

        if len(period_cwcs) < 2:
            return pd.DataFrame()

        # Bước 3 (Lines 7-11 của Algorithm 1): Khởi tạo Pset = p_1 và tính CP(p_i / Pset) cho i = 2..k
        pset_real_sigs = set(c.real_signature(baseline_vars) for c in period_cwcs[0][1])
        pset_crisp_map: Dict[Tuple[str, ...], CategoryWeatherCondition] = {}
        pset_dyn_crisp_map: Dict[Tuple[str, ...], CategoryWeatherCondition] = {}
        for c in period_cwcs[0][1]:
            sig = c.crisp_signature(baseline_vars)
            if sig not in pset_crisp_map:
                pset_crisp_map[sig] = c
            dsig = c.crisp_signature(dyn_vars)
            if dsig not in pset_dyn_crisp_map:
                pset_dyn_crisp_map[dsig] = c
        pset_cwcs: List[CategoryWeatherCondition] = list(period_cwcs[0][1])

        records = []
        for idx in range(1, len(period_cwcs)):
            p_num = idx + 1  # p_2, p_3, ..., p_k
            p_name, current_cwcs = period_cwcs[idx]
            total_m = len(current_cwcs)
            if total_m == 0:
                continue

            real_hits = 0
            crisp_hits, crisp_correct = 0, 0
            fuzzy_hits, fuzzy_correct = 0, 0
            fuzzy_dyn_hits, fuzzy_dyn_correct = 0, 0

            for x in current_cwcs:
                oracle_x_base = derive_agronomic_advisory_oracle(x, baseline_vars)
                oracle_x_dyn = derive_agronomic_advisory_oracle(x, dyn_vars)

                # 1. Exact Real Match (Định nghĩa 2)
                if x.real_signature(baseline_vars) in pset_real_sigs:
                    real_hits += 1

                # 2. Baseline Crisp CWC Match (Định nghĩa 4)
                c_sig = x.crisp_signature(baseline_vars)
                is_crisp_match = c_sig in pset_crisp_map
                if is_crisp_match:
                    crisp_hits += 1
                    y_crisp = pset_crisp_map[c_sig]
                    if derive_agronomic_advisory_oracle(y_crisp, baseline_vars) == oracle_x_base:
                        crisp_correct += 1

                # 3. Fuzzy CWC (Giải pháp 1: Làm mượt biên trên tập biến baseline_vars, đẳng trọng)
                if is_crisp_match:
                    fuzzy_hits += 1
                    y_crisp = pset_crisp_map[c_sig]
                    if derive_agronomic_advisory_oracle(y_crisp, baseline_vars) == oracle_x_base:
                        fuzzy_correct += 1
                else:
                    best_f_sim = -1.0
                    best_f_y: Optional[CategoryWeatherCondition] = None
                    for y in pset_cwcs:
                        s_f = compute_fuzzy_gower_similarity(
                            x,
                            y,
                            weights=equal_weights,
                            subset_vars=baseline_vars,
                            enforce_non_compensatory=True,
                            primary_vars=baseline_vars,
                            max_primary_rank_diff=tau_fuzzy_equal,
                            max_secondary_rank_diff=tau_fuzzy_equal,
                            period_type=period_type,
                            enforce_bio_barrier=False,
                        )
                        if s_f > best_f_sim:
                            best_f_sim = s_f
                            best_f_y = y
                            if best_f_sim >= 0.99:
                                break

                    if best_f_sim >= sim_threshold and best_f_y is not None:
                        fuzzy_hits += 1
                        if derive_agronomic_advisory_oracle(best_f_y, baseline_vars) == oracle_x_base:
                            fuzzy_correct += 1

                # 4. Proposed Fuzzy-Dynamic CWC (Giải pháp 1 + 2: Trọng số động + Ngưỡng sinh học & Không bù trừ)
                d_sig = x.crisp_signature(dyn_vars)
                is_dyn_crisp_match = d_sig in pset_dyn_crisp_map
                if is_dyn_crisp_match and (
                    not enforce_bio_constraints
                    or check_biological_regime_barrier(x, pset_dyn_crisp_map[d_sig], period_type, dyn_vars)
                ):
                    fuzzy_dyn_hits += 1
                    y_dcrisp = pset_dyn_crisp_map[d_sig]
                    if derive_agronomic_advisory_oracle(y_dcrisp, dyn_vars) == oracle_x_dyn:
                        fuzzy_dyn_correct += 1
                else:
                    best_fd_sim = -1.0
                    best_fd_y: Optional[CategoryWeatherCondition] = None
                    for y in pset_cwcs:
                        s_fd = compute_fuzzy_gower_similarity(
                            x,
                            y,
                            weights=dyn_weights,
                            subset_vars=dyn_vars,
                            enforce_non_compensatory=True,
                            primary_vars=primary_vars_active,
                            max_primary_rank_diff=tau_dyn_primary,
                            max_secondary_rank_diff=tau_dyn_secondary,
                            period_type=period_type,
                            enforce_bio_barrier=enforce_bio_constraints,
                        )
                        if s_fd > best_fd_sim:
                            best_fd_sim = s_fd
                            best_fd_y = y
                            if best_fd_sim >= 0.99:
                                break

                    if best_fd_sim >= sim_threshold and best_fd_y is not None:
                        fuzzy_dyn_hits += 1
                        if derive_agronomic_advisory_oracle(best_fd_y, dyn_vars) == oracle_x_dyn:
                            fuzzy_dyn_correct += 1

            crisp_cp = round(crisp_hits * 100.0 / total_m, 1)
            fuzzy_cp = round(fuzzy_hits * 100.0 / total_m, 1)
            fuzzy_dyn_cp = round(fuzzy_dyn_hits * 100.0 / total_m, 1)

            crisp_ap = round(crisp_correct * 100.0 / crisp_hits, 1) if crisp_hits > 0 else 100.0
            fuzzy_ap = round(fuzzy_correct * 100.0 / fuzzy_hits, 1) if fuzzy_hits > 0 else 100.0
            fuzzy_dyn_ap = round(fuzzy_dyn_correct * 100.0 / fuzzy_dyn_hits, 1) if fuzzy_dyn_hits > 0 else 100.0

            f1_dyn = (
                round(2.0 * fuzzy_dyn_cp * fuzzy_dyn_ap / (fuzzy_dyn_cp + fuzzy_dyn_ap), 1)
                if (fuzzy_dyn_cp + fuzzy_dyn_ap) > 0
                else 0.0
            )

            records.append(
                {
                    "Period_Index": p_num,
                    "Period_Name": p_name,
                    "Total_CWCs": total_m,
                    "Real_WC_CP": round(real_hits * 100.0 / total_m, 1),
                    "Crisp_CWC_CP": crisp_cp,
                    "Fuzzy_CWC_CP": fuzzy_cp,
                    "Fuzzy_Dynamic_CWC_CP": fuzzy_dyn_cp,
                    "Crisp_CWC_AP": crisp_ap,
                    "Fuzzy_CWC_AP": fuzzy_ap,
                    "Fuzzy_Dynamic_CWC_AP": fuzzy_dyn_ap,
                    "Fuzzy_Dynamic_F1": f1_dyn,
                }
            )

            # Cập nhật Pset = Pset ∪ p_i (Line 10 của Algorithm 1)
            for x in current_cwcs:
                pset_real_sigs.add(x.real_signature(baseline_vars))
                c_sig = x.crisp_signature(baseline_vars)
                if c_sig not in pset_crisp_map:
                    pset_crisp_map[c_sig] = x
                d_sig = x.crisp_signature(dyn_vars)
                if d_sig not in pset_dyn_crisp_map:
                    pset_dyn_crisp_map[d_sig] = x
            pset_cwcs.extend(current_cwcs)

        return pd.DataFrame(records)


def run_multiseed_same_dataset_comparison(
    seeds: Optional[List[int]] = None,
    theta: float = 0.85,
    delta: float = 0.10,
) -> pd.DataFrame:
    """
    Thực thi đối sánh công bằng (Apples-to-Apples Comparison) giữa Baseline Crisp CWC, Fuzzy CWC
    và Fuzzy-Dynamic CWC trên CÙNG tập dữ liệu mô phỏng hiệu chỉnh qua R = 10 hạt giống ngẫu nhiên độc lập.
    """
    if seeds is None:
        seeds = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028]

    scenarios = [
        ("pt = Năm | (n=3, d=5)", "Year", None, 3, 5, 5, 84.0),
        ("pt = Năm | (n=5, d=5)", "Year", None, 5, 5, 5, 76.0),
        ("pt = Vụ Rabi | (n=3, d=5)", "Rabi", None, 3, 5, 5, 78.0),
        ("pt = Vụ Rabi | (n=5, d=5)", "Rabi", None, 5, 5, 5, 60.0),
        ("Trổ hoa Kharif | (n=3, d=5)", "Kharif", "Flowering", 3, 5, 3, 53.0),
        ("Trổ hoa Kharif | (n=3, d=5)", "Kharif", "Flowering", 3, 5, 7, 78.0),
        ("Trổ hoa Kharif | (n=5, d=5)", "Kharif", "Flowering", 5, 5, 3, 22.0),
        ("Trổ hoa Kharif | (n=5, d=5)", "Kharif", "Flowering", 5, 5, 10, 44.0),
        ("Trổ hoa Rabi | (n=3, d=5)", "Rabi", "Flowering", 3, 5, 3, 38.0),
        ("Trổ hoa Rabi | (n=5, d=5)", "Rabi", "Flowering", 5, 5, 10, 50.0),
    ]

    mapper = WCTMapper(transition_ratio=delta)
    engine = CPEngine(mapper=mapper, theta=theta)

    # Chạy từng seed một lần cho mỗi cấu hình (pt, ph, n, d)
    unique_configs = sorted(list({(s[1], s[2], s[3], s[4]) for s in scenarios}), key=lambda x: str(x))
    seed_results: Dict[Tuple[str, Optional[str], int, int, int], Dict[str, List[float]]] = {}

    for s_label, pt, ph, n, d, p_idx, _ in scenarios:
        seed_results[(pt, ph, n, d, p_idx)] = {
            "crisp_cp": [],
            "fuzzy_cp": [],
            "fdyn_cp": [],
            "crisp_ap": [],
            "fuzzy_ap": [],
            "fdyn_ap": [],
        }

    for sd in seeds:
        df_sd = generate_30year_weather_dataset(save_path="", seed=sd, force_regenerate=True)
        for pt, ph, n, d in unique_configs:
            res_df = engine.run_algorithm_1_all_modes(
                df=df_sd, period_type=pt, phenophase=ph, n_vars=n, duration=d, theta=theta, max_periods=10
            )
            for _, row in res_df.iterrows():
                p_i = int(row["Period_Index"])
                key = (pt, ph, n, d, p_i)
                if key in seed_results:
                    seed_results[key]["crisp_cp"].append(float(row["Crisp_CWC_CP"]))
                    seed_results[key]["fuzzy_cp"].append(float(row["Fuzzy_CWC_CP"]))
                    seed_results[key]["fdyn_cp"].append(float(row["Fuzzy_Dynamic_CWC_CP"]))
                    seed_results[key]["crisp_ap"].append(float(row["Crisp_CWC_AP"]))
                    seed_results[key]["fuzzy_ap"].append(float(row["Fuzzy_CWC_AP"]))
                    seed_results[key]["fdyn_ap"].append(float(row["Fuzzy_Dynamic_CWC_AP"]))

    rows = []
    for s_label, pt, ph, n, d, p_idx, paper_val in scenarios:
        k = (pt, ph, n, d, p_idx)
        c_cp_arr = np.array(seed_results[k]["crisp_cp"])
        f_cp_arr = np.array(seed_results[k]["fuzzy_cp"])
        fd_cp_arr = np.array(seed_results[k]["fdyn_cp"])
        c_ap_arr = np.array(seed_results[k]["crisp_ap"])
        fd_ap_arr = np.array(seed_results[k]["fdyn_ap"])

        c_mean = float(np.mean(c_cp_arr))
        c_sd = float(np.std(c_cp_arr))
        c_ap_mean = float(np.mean(c_ap_arr))
        c_ap_sd = float(np.std(c_ap_arr))
        c_f1_arr = 2.0 * (c_cp_arr * c_ap_arr) / (c_cp_arr + c_ap_arr + 1e-9)
        c_f1_mean = float(np.mean(c_f1_arr))

        f_mean = float(np.mean(f_cp_arr))
        f_sd = float(np.std(f_cp_arr))

        fd_mean = float(np.mean(fd_cp_arr))
        fd_sd = float(np.std(fd_cp_arr))
        fd_ap_mean = float(np.mean(fd_ap_arr))
        fd_ap_sd = float(np.std(fd_ap_arr))
        fd_f1_arr = 2.0 * (fd_cp_arr * fd_ap_mean) / (fd_cp_arr + fd_ap_mean + 1e-9)
        fd_f1_mean = float(np.mean(fd_f1_arr))

        t_stat, p_val = ttest_rel(fd_cp_arr, c_cp_arr)
        p_val_str = f"p < 0.001" if p_val < 0.001 else (f"p < 0.01" if p_val < 0.01 else f"p = {p_val:.3f}")

        net_gain = fd_mean - c_mean
        net_f1 = fd_f1_mean - c_f1_mean
        p_label = f"p_{p_idx} ({'Năm' if pt == 'Year' else 'Vụ'} {p_idx})"
        rows.append(
            {
                "Scenario": s_label,
                "Period": p_label,
                "Paper_Baseline_CP": round(paper_val, 1),
                "Baseline_Crisp_CP": f"{c_mean:.1f} ± {c_sd:.1f}%",
                "Baseline_Crisp_AP": f"{c_ap_mean:.1f}%",
                "Baseline_Crisp_F1": f"{c_f1_mean:.1f}%",
                "Fuzzy_CWC_CP": f"{f_mean:.1f} ± {f_sd:.1f}%",
                "Fuzzy_Dynamic_CP": f"{fd_mean:.1f} ± {fd_sd:.1f}%",
                "Proposed_AP": f"{fd_ap_mean:.1f}%",
                "Proposed_F1": f"{fd_f1_mean:.1f}%",
                "Net_Gain_CP": f"+{net_gain:.1f}%pts",
                "Net_Gain_F1": f"+{net_f1:.1f}%pts",
                "Paired_TTest_P": p_val_str,
            }
        )

    return pd.DataFrame(rows)


def run_sensitivity_analysis_theta_delta(
    seeds: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Phân tích độ nhạy (Sensitivity Analysis) theo ngưỡng tương đồng θ ∈ {0.75, 0.80, 0.85, 0.90, 0.95}
    và độ rộng vùng chuyển tiếp mờ δ ∈ {0.05, 0.10, 0.15, 0.20} cho giai đoạn Trổ hoa (Flowering, n=5, d=5),
    đo lường:
    - Độ bao phủ CP (%)
    - Độ chính xác khuyến nghị AP (%) (khi không có và khi có ràng buộc ngưỡng sinh học I_bio)
    - Điểm chất lượng điều hòa F1_reuse (%)
    """
    if seeds is None:
        seeds = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028]

    configs = [
        (0.75, 0.10),
        (0.80, 0.10),
        (0.85, 0.05),
        (0.85, 0.10),
        (0.85, 0.15),
        (0.85, 0.20),
        (0.90, 0.10),
        (0.95, 0.10),
    ]

    dfs = [generate_30year_weather_dataset(save_path="", seed=s, force_regenerate=True) for s in seeds]
    out_rows = []

    for theta, delta in configs:
        mapper = WCTMapper(transition_ratio=delta)
        eng = CPEngine(mapper=mapper, theta=theta)

        cp_list, ap_bio_list, ap_nobio_list = [], [], []
        for df_s in dfs:
            for season_pt in ("Kharif", "Rabi"):
                res_bio = eng.run_algorithm_1_all_modes(
                    df_s, season_pt, "Flowering", n_vars=5, duration=5, theta=theta, enforce_bio_constraints=True, max_periods=10
                )
                res_nobio = eng.run_algorithm_1_all_modes(
                    df_s, season_pt, "Flowering", n_vars=5, duration=5, theta=theta, enforce_bio_constraints=False, max_periods=10
                )
                # Lấy trung bình các mốc đánh giá p_7 và p_10 để giảm phương sai mẫu nhỏ
                sub_b = res_bio[res_bio["Period_Index"].isin([7, 10])]
                sub_nb = res_nobio[res_nobio["Period_Index"].isin([7, 10])]
                cp_list.append(float(sub_b["Fuzzy_Dynamic_CWC_CP"].mean()))
                ap_bio_list.append(float(sub_b["Fuzzy_Dynamic_CWC_AP"].mean()))
                ap_nobio_list.append(float(sub_nb["Fuzzy_Dynamic_CWC_AP"].mean()))

        cp_m = round(float(np.mean(cp_list)), 1)
        cp_sd = round(float(np.std(cp_list)), 1)
        ap_b_m = round(float(np.mean(ap_bio_list)), 1)
        ap_nb_m = round(float(np.mean(ap_nobio_list)), 1)
        f1_m = round(2.0 * cp_m * ap_b_m / (cp_m + ap_b_m), 1) if (cp_m + ap_b_m) > 0 else 0.0

        out_rows.append(
            {
                "Theta": theta,
                "Delta_Pct": f"±{int(delta * 100)}%",
                "CP_Mean": cp_m,
                "CP_SD": cp_sd,
                "AP_NoBio_Mean": ap_nb_m,
                "AP_WithBio_Mean": ap_b_m,
                "F1_Reuse": f1_m,
            }
        )

    return pd.DataFrame(out_rows)


def get_paper_benchmark_tables() -> Dict[str, pd.DataFrame]:
    """
    Trả về bộ bảng số liệu chuẩn từ thực nghiệm bài báo gốc (Alugubelly et al., 2019 - Tables 6–9).
    Lưu ý: Đối với Bảng 4.5, các tổ hợp biến lồng nhau V_1 ⊂ V_2 ⊂ V_3 ⊂ V_4 ⊂ V_5 tuân thủ tính đơn điệu
    không tăng của CP theo số biến n.
    """
    table_4_2 = pd.DataFrame(
        [
            {"n": 1, "Variables": "{Tmax}", "WC_d1": 98.0, "CWC_d1": 100.0, "CWC_d5": 98.0, "Gain_d1": "+2.0%"},
            {"n": 2, "Variables": "{Tmax, Tmin}", "WC_d1": 20.0, "CWC_d1": 99.0, "CWC_d5": 96.0, "Gain_d1": "+79.0%"},
            {"n": 3, "Variables": "{Tmax, Tmin, RHmax}", "WC_d1": 0.0, "CWC_d1": 96.0, "CWC_d5": 84.0, "Gain_d1": "+96.0%"},
            {"n": 4, "Variables": "{Tmax, Tmin, RHmax, RHmin}", "WC_d1": 0.0, "CWC_d1": 87.0, "CWC_d5": 87.0, "Gain_d1": "+87.0%"},
            {"n": 5, "Variables": "{Tmax, Tmin, RHmax, RHmin, RF}", "WC_d1": 0.0, "CWC_d1": 81.0, "CWC_d5": 76.0, "Gain_d1": "+81.0%"},
        ]
    )

    table_4_3 = pd.DataFrame(
        [
            {"Period": "p_3 (Năm 3)", "CWC(1,1)": 100, "CWC(2,1)": 98, "CWC(3,1)": 94, "CWC(4,1)": 82, "CWC(5,1)": 80, "CWC(1,5)": 100, "CWC(2,5)": 94, "CWC(3,5)": 83, "CWC(4,5)": 82, "CWC(5,5)": 60},
            {"Period": "p_5 (Năm 5)", "CWC(1,1)": 100, "CWC(2,1)": 99, "CWC(3,1)": 96, "CWC(4,1)": 87, "CWC(5,1)": 81, "CWC(1,5)": 98,  "CWC(2,5)": 96, "CWC(3,5)": 84, "CWC(4,5)": 87, "CWC(5,5)": 76},
            {"Period": "p_7 (Năm 7)", "CWC(1,1)": 100, "CWC(2,1)": 100, "CWC(3,1)": 99, "CWC(4,1)": 98, "CWC(5,1)": 90, "CWC(1,5)": 100, "CWC(2,5)": 97, "CWC(3,5)": 95, "CWC(4,5)": 90, "CWC(5,5)": 81},
            {"Period": "p_10 (Năm 10)", "CWC(1,1)": 100, "CWC(2,1)": 99, "CWC(3,1)": 99, "CWC(4,1)": 96, "CWC(5,1)": 88, "CWC(1,5)": 100, "CWC(2,5)": 97, "CWC(3,5)": 96, "CWC(4,5)": 91, "CWC(5,5)": 81},
        ]
    )

    table_4_4 = pd.DataFrame(
        [
            {"Period": "p_3 (Vụ 3)", "CWC(1,1)": 100, "CWC(2,1)": 90, "CWC(3,1)": 82, "CWC(4,1)": 74, "CWC(5,1)": 70, "CWC(1,5)": 95, "CWC(2,5)": 82, "CWC(3,5)": 76, "CWC(4,5)": 62, "CWC(5,5)": 50},
            {"Period": "p_5 (Vụ 5)", "CWC(1,1)": 100, "CWC(2,1)": 95, "CWC(3,1)": 90, "CWC(4,1)": 80, "CWC(5,1)": 74, "CWC(1,5)": 97, "CWC(2,5)": 85, "CWC(3,5)": 78, "CWC(4,5)": 70, "CWC(5,5)": 60},
            {"Period": "p_7 (Vụ 7)", "CWC(1,1)": 100, "CWC(2,1)": 98, "CWC(3,1)": 90, "CWC(4,1)": 90, "CWC(5,1)": 88, "CWC(1,5)": 98, "CWC(2,5)": 94, "CWC(3,5)": 88, "CWC(4,5)": 81, "CWC(5,5)": 78},
            {"Period": "p_10 (Vụ 10)", "CWC(1,1)": 100, "CWC(2,1)": 100, "CWC(3,1)": 100, "CWC(4,1)": 97, "CWC(5,1)": 97, "CWC(1,5)": 100, "CWC(2,5)": 94, "CWC(3,5)": 90, "CWC(4,5)": 81, "CWC(5,5)": 78},
        ]
    )

    table_4_5 = pd.DataFrame(
        [
            {"Season_Period": "Kharif - p_3", "CWC(1,1)": 82, "CWC(2,1)": 78, "CWC(3,1)": 58, "CWC(4,1)": 42, "CWC(5,1)": 38, "CWC(1,5)": 94, "CWC(2,5)": 62, "CWC(3,5)": 53, "CWC(4,5)": 40, "CWC(5,5)": 22},
            {"Season_Period": "Kharif - p_5", "CWC(1,1)": 100, "CWC(2,1)": 90, "CWC(3,1)": 76, "CWC(4,1)": 58, "CWC(5,1)": 40, "CWC(1,5)": 98, "CWC(2,5)": 82, "CWC(3,5)": 62, "CWC(4,5)": 44, "CWC(5,5)": 30},
            {"Season_Period": "Kharif - p_7", "CWC(1,1)": 100, "CWC(2,1)": 94, "CWC(3,1)": 86, "CWC(4,1)": 76, "CWC(5,1)": 61, "CWC(1,5)": 98, "CWC(2,5)": 90, "CWC(3,5)": 78, "CWC(4,5)": 62, "CWC(5,5)": 44},
            {"Season_Period": "Kharif - p_10", "CWC(1,1)": 100, "CWC(2,1)": 98, "CWC(3,1)": 93, "CWC(4,1)": 82, "CWC(5,1)": 78, "CWC(1,5)": 100, "CWC(2,5)": 94, "CWC(3,5)": 80, "CWC(4,5)": 68, "CWC(5,5)": 44},
            {"Season_Period": "Rabi - p_3", "CWC(1,1)": 100, "CWC(2,1)": 81, "CWC(3,1)": 61, "CWC(4,1)": 58, "CWC(5,1)": 42, "CWC(1,5)": 82, "CWC(2,5)": 62, "CWC(3,5)": 38, "CWC(4,5)": 22, "CWC(5,5)": 22},
            {"Season_Period": "Rabi - p_5", "CWC(1,1)": 100, "CWC(2,1)": 90, "CWC(3,1)": 78, "CWC(4,1)": 76, "CWC(5,1)": 76, "CWC(1,5)": 84, "CWC(2,5)": 62, "CWC(3,5)": 58, "CWC(4,5)": 54, "CWC(5,5)": 41},
            {"Season_Period": "Rabi - p_7", "CWC(1,1)": 100, "CWC(2,1)": 93, "CWC(3,1)": 82, "CWC(4,1)": 78, "CWC(5,1)": 78, "CWC(1,5)": 90, "CWC(2,5)": 78, "CWC(3,5)": 61, "CWC(4,5)": 60, "CWC(5,5)": 48},
            {"Season_Period": "Rabi - p_10", "CWC(1,1)": 100, "CWC(2,1)": 94, "CWC(3,1)": 84, "CWC(4,1)": 79, "CWC(5,1)": 76, "CWC(1,5)": 96, "CWC(2,5)": 83, "CWC(3,5)": 70, "CWC(4,5)": 60, "CWC(5,5)": 50},
        ]
    )

    return {
        "Table_4_2": table_4_2,
        "Table_4_3": table_4_3,
        "Table_4_4": table_4_4,
        "Table_4_5": table_4_5,
    }
