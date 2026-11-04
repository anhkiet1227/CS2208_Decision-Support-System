"""
Module 1: wct_mapper.py
Ánh xạ Điều kiện Thời tiết Số thực (WC), Danh mục Cứng (Crisp WCT) và Danh mục Mờ (Fuzzy WCT).

Thực hiện:
- Định nghĩa 1 (Weather Condition - WC): Tính thống kê ψ(V) cho d = 1 hoặc d = 5 ngày.
- Bảng 2.2 (Weather Category Table - WCT): Phân loại chuẩn Cục Khí tượng Ấn Độ (IMD) kèm quy tắc
  làm tròn / phân hoạch nửa nguyên liên tục (continuous half-integer cutoffs) không để khoảng hở.
- Định nghĩa 3 (Category-based Weather Condition - CWC): Ánh xạ ψ(V) sang nhãn danh mục cứng C(V).
- Giải pháp đề xuất 1 (Fuzzy CWC): Hàm thuộc mờ hình thang μ(x; a, b, c, d) thỏa mãn phân hoạch Ruspini
  với độ rộng vùng chuyển tiếp tham số hóa δ (transition_ratio) và chỉ số cấp bậc liên tục hóa
  Rank_Fuzzy(X_k) nhằm khắc phục Hiệu ứng vách đá (Boundary Cliff Effect).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


# Danh sách 5 biến thời tiết chuẩn theo bài báo Alugubelly et al. (2019)
WEATHER_VARS = ["Tmax", "Tmin", "RHmax", "RHmin", "RF"]

# Các tổ hợp biến V lồng nhau (nested subsets V_1 ⊂ V_2 ⊂ ... ⊂ V_5) theo Bảng 4.1
BASELINE_VAR_COMBINATIONS: Dict[int, List[str]] = {
    1: ["Tmax"],
    2: ["Tmax", "Tmin"],
    3: ["Tmax", "Tmin", "RHmax"],
    4: ["Tmax", "Tmin", "RHmax", "RHmin"],
    5: ["Tmax", "Tmin", "RHmax", "RHmin", "RF"],
}


# Thứ tự cấp bậc tự nhiên (Ordinal Ranks: 0..8, M_k = 9, M_k - 1 = 8) của độ lệch nhiệt độ ΔT
TEMP_CATEGORIES_ORDERED = [
    "Large fall",        # 0: ΔT < -6.5 °C (làm tròn <= -7 °C)
    "Marked fall",       # 1: -6.5 <= ΔT < -4.5 °C (-6 đến -5 °C)
    "Appreciable fall",  # 2: -4.5 <= ΔT < -2.5 °C (-4 đến -3 °C)
    "Fall",              # 3: -2.5 <= ΔT < -1.5 °C (-2 °C)
    "Little change",     # 4: -1.5 <= ΔT <= 1.5 °C (-1, 0, +1 °C)
    "Rise",              # 5: 1.5 < ΔT <= 2.5 °C (+2 °C)
    "Appreciable rise",  # 6: 2.5 < ΔT <= 4.5 °C (+3 đến +4 °C)
    "Marked rise",       # 7: 4.5 < ΔT <= 6.5 °C (+5 đến +6 °C)
    "Large rise",        # 8: ΔT > 6.5 °C (>= +7 °C)
]

# Danh mục chuẩn IMD trong Bảng 2.2 bài báo gốc (8 nhãn khi gộp Rise/fall)
IMD_TEMP_CATEGORIES = [
    "Large fall",
    "Marked fall",
    "Appreciable fall",
    "Little change",
    "Rise/fall",
    "Appreciable rise",
    "Marked rise",
    "Large rise",
]

# Thứ tự cấp bậc Độ ẩm tương đối RH (Ordinal Ranks: 0..3, M_k = 4, M_k - 1 = 3)
RH_CATEGORIES_ORDERED = [
    "Low",        # 0: 0 <= RH <= 30.5 % (làm tròn 0 - 30 %)
    "Moderate",   # 1: 30.5 < RH <= 60.5 % (làm tròn 31 - 60 %)
    "High",       # 2: 60.5 < RH <= 80.5 % (làm tròn 61 - 80 %)
    "Very high",  # 3: RH > 80.5 % (làm tròn >= 81 %)
]

# Thứ tự cấp bậc Lượng mưa RF (Ordinal Ranks: 0..7, M_k = 8, M_k - 1 = 7)
RF_CATEGORIES_ORDERED = [
    "No rain",               # 0: RF == 0.0 mm
    "Very light rain",       # 1: 0.0 < RF <= 2.45 mm (0.1 - 2.4 mm)
    "Light rain",            # 2: 2.45 < RF <= 7.55 mm (2.5 - 7.5 mm)
    "Moderate rain",         # 3: 7.55 < RF <= 35.55 mm (7.6 - 35.5 mm)
    "Rather heavy rain",     # 4: 35.55 < RF <= 64.45 mm (35.6 - 64.4 mm)
    "Heavy rain",            # 5: 64.45 < RF <= 124.45 mm (64.5 - 124.4 mm)
    "Very heavy rain",       # 6: 124.45 < RF <= 244.45 mm (124.5 - 244.4 mm)
    "Extremely heavy rain",  # 7: RF > 244.45 mm (>= 244.5 mm)
]


@dataclass
class WeatherCondition:
    """
    Định nghĩa 1: Điều kiện Thời tiết số thực WC_i(n, d) = <i, n, s, d, V, ψ(V)>
    """
    wc_id: int
    n: int
    start_date: str
    duration: int
    variables: List[str]
    psi_values: Dict[str, float]
    temp_deviations: Dict[str, float] = field(default_factory=dict)

    def to_tuple_str(self) -> str:
        vals_str = ", ".join(f"{self.psi_values[v]:.2f}" for v in self.variables)
        vars_str = ", ".join(self.variables)
        return f"<{self.wc_id}, {self.n}, {self.start_date}, {self.duration}, {{{vars_str}}}, {{{vals_str}}}>"


@dataclass
class CategoryWeatherCondition:
    """
    Định nghĩa 3 & Giải pháp 1: Điều kiện Thời tiết dựa trên Danh mục (Crisp CWC & Fuzzy CWC)
    Bao gồm:
    - categories: Nhãn danh mục cứng C(V) theo Bảng 2.2 (IMD WCT)
    - fuzzy_memberships: Vector độ thuộc mờ μ(V) cho từng biến
    - fuzzy_ranks: Giá trị cấp bậc liên tục hóa Rank_Fuzzy(X_k) ∈ [0, M_k - 1]
    - max_ranks: Khoảng cách cấp bậc tối đa (M_k - 1) của từng biến để chuẩn hóa khoảng cách Gower
    - temp_deviations: Độ lệch nhiệt độ ΔT so với chuẩn lịch sử 30 năm
    """
    cwc_id: int
    n: int
    start_date: str
    duration: int
    variables: List[str]
    psi_values: Dict[str, float]
    categories: Dict[str, str]
    fuzzy_memberships: Dict[str, Dict[str, float]] = field(default_factory=dict)
    fuzzy_ranks: Dict[str, float] = field(default_factory=dict)
    max_ranks: Dict[str, int] = field(default_factory=dict)
    temp_deviations: Dict[str, float] = field(default_factory=dict)

    def crisp_signature(self, subset_vars: Optional[List[str]] = None) -> Tuple[str, ...]:
        """Trả về tuple nhãn danh mục cứng C(V) để so khớp chính xác theo Định nghĩa 4."""
        target_vars = subset_vars if subset_vars is not None else self.variables
        return tuple(self.categories[v] for v in target_vars)

    def real_signature(self, subset_vars: Optional[List[str]] = None) -> Tuple[float, ...]:
        """Trả về tuple giá trị số thực ψ(V) để so khớp chính xác theo Định nghĩa 2."""
        target_vars = subset_vars if subset_vars is not None else self.variables
        return tuple(round(self.psi_values[v], 2) for v in target_vars)


def trapezoidal_mf(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Hàm thuộc hình thang (Trapezoidal Membership Function) μ(x; a, b, c, d) ∈ [0, 1].
    Công thức (Mục 3.2):
        μ(x; a, b, c, d) = max(0, min((x - a)/(b - a), 1, (d - x)/(d - c)))
    Hỗ trợ trường hợp suy biến (vai trái mở a == b hoặc vai phải mở c == d).
    """
    if a == b and x <= b:
        return 1.0
    if c == d and x >= c:
        return 1.0
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a) if b > a else 1.0
    if c < x < d:
        return (d - x) / (d - c) if d > c else 1.0
    return 0.0


def build_ruspini_trapezoids(
    labels: List[str],
    cutoffs: List[float],
    half_widths: List[float],
) -> List[Tuple[str, float, float, float, float]]:
    """
    Xây dựng danh sách các hình thang mờ liên tiếp thỏa mãn chính xác phân hoạch Ruspini:
        ∑_{m=1..M} μ_m(x) = 1.0  với mọi x ∈ ℝ.
    Tại mỗi ranh giới B_m = cutoffs[m] (giữa danh mục m và m+1) với bán kính vùng chuyển tiếp δ_m = half_widths[m]:
    - Danh mục m giảm tuyến tính từ 1 xuống 0 trên đoạn [B_m - δ_m, B_m + δ_m]
    - Danh mục m+1 tăng tuyến tính từ 0 lên 1 trên đúng đoạn [B_m - δ_m, B_m + δ_m]
    - Tại đúng điểm ranh giới x = B_m: μ_m(B_m) = 0.50 và μ_{m+1}(B_m) = 0.50.
    """
    assert len(labels) == len(cutoffs) + 1 == len(half_widths) + 1
    m_count = len(labels)
    trapezoids: List[Tuple[str, float, float, float, float]] = []

    for idx, label in enumerate(labels):
        if idx == 0:
            a, b = -1e6, -1e6
        else:
            a = cutoffs[idx - 1] - half_widths[idx - 1]
            b = cutoffs[idx - 1] + half_widths[idx - 1]

        if idx == m_count - 1:
            c, d = 1e6, 1e6
        else:
            c = cutoffs[idx] - half_widths[idx]
            d = cutoffs[idx] + half_widths[idx]

        trapezoids.append((label, float(a), float(b), float(c), float(d)))

    return trapezoids


class WCTMapper:
    """
    Bộ ánh xạ Bảng phân loại thời tiết WCT (Crisp IMD Standard & Proposed Fuzzy Trapezoidal WCT).
    Tham số `transition_ratio` (δ, mặc định = 0.10 tức ±10% độ rộng danh mục chuẩn) điều khiển
    độ rộng vùng chuyển tiếp mềm [B_m - δ_m, B_m + δ_m] quanh mỗi ngưỡng phân lớp B_m.
    """

    def __init__(self, transition_ratio: float = 0.10):
        """
        :param transition_ratio: Tỷ lệ độ rộng vùng chuyển tiếp mềm δ quanh ranh giới WCT (mặc định 0.10).
        """
        self.transition_ratio = float(transition_ratio)
        self._init_fuzzy_partitions()

    def _init_fuzzy_partitions(self) -> None:
        """
        Thiết lập các hình thang mờ liên tiếp thỏa mãn phân hoạch Ruspini: ∑_m μ_{k,m}(x) = 1.
        Khi transition_ratio = 0.10 (chuẩn):
        - Biến RH (%): Các ngưỡng liên tục B_m ∈ {30.5, 60.5, 80.5} với bán kính chuyển tiếp δ_RH = 2.5%
          tạo thành các vùng chuyển tiếp đối xứng [28.0%, 33.0%], [58.0%, 63.0%], [78.0%, 83.0%].
          + Tại RH = 60.5% (tâm ranh giới): μ_Moderate = 0.50, μ_High = 0.50.
          + Tại RH = 60.0%: μ_Moderate = (63.0 - 60.0)/5.0 = 0.60, μ_High = 0.40.
          + Tại RH = 60.8%: μ_Moderate = (63.0 - 60.8)/5.0 = 0.44, μ_High = 0.56.
        """
        scale = self.transition_ratio / 0.10 if self.transition_ratio > 0 else 0.001

        # 1. Biến Độ ẩm tương đối RH (%): Ngưỡng liên tục tại 30.5, 60.5, 80.5
        rh_cutoffs = [30.5, 60.5, 80.5]
        rh_base_deltas = [2.5, 2.5, 2.5]
        rh_deltas = [min(d * scale, 9.0) for d in rh_base_deltas]
        self.rh_boundaries = build_ruspini_trapezoids(
            RH_CATEGORIES_ORDERED, rh_cutoffs, rh_deltas
        )

        # 2. Biến Lượng mưa RF (mm): Ngưỡng liên tục tại 0.05, 2.45, 7.55, 35.55, 64.45, 124.45, 244.45
        rf_cutoffs = [0.05, 2.45, 7.55, 35.55, 64.45, 124.45, 244.45]
        rf_base_deltas = [0.04, 0.30, 0.65, 3.00, 4.00, 6.50, 12.50]
        rf_deltas = [
            min(d * scale, max((rf_cutoffs[i + 1] - rf_cutoffs[i]) * 0.45, 0.04))
            if i < len(rf_cutoffs) - 1
            else d * scale
            for i, d in enumerate(rf_base_deltas)
        ]
        rf_deltas[0] = min(0.04 * scale, 0.045)
        self.rf_boundaries = build_ruspini_trapezoids(
            RF_CATEGORIES_ORDERED, rf_cutoffs, rf_deltas
        )

        # 3. Biến Độ lệch Nhiệt độ ΔT (°C): Ngưỡng liên tục tại -6.5, -4.5, -2.5, -1.5, 1.5, 2.5, 4.5, 6.5
        temp_cutoffs = [-6.5, -4.5, -2.5, -1.5, 1.5, 2.5, 4.5, 6.5]
        temp_base_deltas = [0.45, 0.45, 0.35, 0.30, 0.30, 0.35, 0.45, 0.45]
        temp_deltas = [min(d * scale, 0.90) for d in temp_base_deltas]
        self.temp_dev_boundaries = build_ruspini_trapezoids(
            TEMP_CATEGORIES_ORDERED, temp_cutoffs, temp_deltas
        )

    # =========================================================================
    # CRISP WCT MAPPING (BẢNG 2.2 - CHUẨN IMD KÈM QUY TẮC LÀM TRÒN LIÊN TỤC)
    # =========================================================================
    @staticmethod
    def map_crisp_temp_deviation(dev: float) -> str:
        """
        Ánh xạ độ lệch nhiệt độ ΔT (°C) sang nhãn danh mục chuẩn IMD (Bảng 2.2).
        Quy tắc liên tục hóa không khoảng hở (half-integer rounding cutoffs):
        - ΔT < -6.5 (làm tròn <= -7)          -> Large fall
        - -6.5 <= ΔT < -4.5 (làm tròn -6, -5) -> Marked fall
        - -4.5 <= ΔT < -2.5 (làm tròn -4, -3) -> Appreciable fall
        - -2.5 <= ΔT < -1.5 hoặc 1.5 < ΔT <= 2.5 (làm tròn ±2) -> Rise/fall
        - -1.5 <= ΔT <= 1.5 (làm tròn -1,0,1) -> Little change
        - 2.5 < ΔT <= 4.5 (làm tròn 3, 4)     -> Appreciable rise
        - 4.5 < ΔT <= 6.5 (làm tròn 5, 6)     -> Marked rise
        - ΔT > 6.5 (làm tròn >= 7)            -> Large rise
        """
        if dev < -6.5:
            return "Large fall"
        elif dev < -4.5:
            return "Marked fall"
        elif dev < -2.5:
            return "Appreciable fall"
        elif dev < -1.5 or (1.5 < dev <= 2.5):
            return "Rise/fall"
        elif -1.5 <= dev <= 1.5:
            return "Little change"
        elif dev <= 4.5:
            return "Appreciable rise"
        elif dev <= 6.5:
            return "Marked rise"
        else:
            return "Large rise"

    @staticmethod
    def map_crisp_rh(rh: float) -> str:
        """
        Ánh xạ Độ ẩm tương đối RH (%) sang nhãn danh mục chuẩn IMD (Bảng 2.2).
        Sử dụng ngưỡng nửa nguyên liên tục (30.5%, 60.5%, 80.5%) tương ứng với quy tắc làm tròn
        về số nguyên [0–30], [31–60], [61–80], [>=81] của IMD:
        - RH <= 30.5%  -> Low
        - RH <= 60.5%  -> Moderate (VD: 60.0% -> Moderate; 60.8% -> High)
        - RH <= 80.5%  -> High
        - RH > 80.5%   -> Very high
        """
        if rh <= 30.5:
            return "Low"
        elif rh <= 60.5:
            return "Moderate"
        elif rh <= 80.5:
            return "High"
        else:
            return "Very high"

    @staticmethod
    def map_crisp_rf(rf: float) -> str:
        """
        Ánh xạ Lượng mưa RF (mm) sang nhãn danh mục chuẩn IMD (Bảng 2.2).
        Sử dụng ngưỡng liên tục hóa (0.0, 2.45, 7.55, 35.55, 64.45, 124.45, 244.45):
        - RF <= 0.0    -> No rain
        - RF <= 2.45   -> Very light rain (0.1 - 2.4 mm)
        - RF <= 7.55   -> Light rain (2.5 - 7.5 mm)
        - RF <= 35.55  -> Moderate rain (7.6 - 35.5 mm)
        - RF <= 64.45  -> Rather heavy rain (35.6 - 64.4 mm)
        - RF <= 124.45 -> Heavy rain (64.5 - 124.4 mm)
        - RF <= 244.45 -> Very heavy rain (124.5 - 244.4 mm)
        - RF > 244.45  -> Extremely heavy rain (>= 244.5 mm)
        """
        if rf <= 0.0:
            return "No rain"
        elif rf <= 2.45:
            return "Very light rain"
        elif rf <= 7.55:
            return "Light rain"
        elif rf <= 35.55:
            return "Moderate rain"
        elif rf <= 64.45:
            return "Rather heavy rain"
        elif rf <= 124.45:
            return "Heavy rain"
        elif rf <= 244.45:
            return "Very heavy rain"
        else:
            return "Extremely heavy rain"

    # =========================================================================
    # FUZZY WCT MAPPING (GIẢI PHÁP ĐỀ XUẤT 1 - MỤC 3.2)
    # =========================================================================
    @staticmethod
    def _compute_partition_memberships(
        x: float, boundaries: List[Tuple[str, float, float, float, float]]
    ) -> Tuple[Dict[str, float], float, int]:
        """
        Tính phân bố độ thuộc mờ μ_{k,m}(x) và chỉ số cấp bậc liên tục Rank_Fuzzy(X_k).
        Lưu ý: Dù đánh chỉ số m = 0..M_k-1 hay m = 1..M_k thì hiệu số |Rank_Fuzzy(X_k) - Rank_Fuzzy(Y_k)|
        là bất biến tuyệt đối vì ∑_m μ_{k,m}(x) = 1.
        Trả về: (dict_memberships, fuzzy_rank, max_rank = M_k - 1)
        """
        memberships: Dict[str, float] = {}
        total_mu = 0.0
        for label, a, b, c, d in boundaries:
            mu = trapezoidal_mf(x, a, b, c, d)
            memberships[label] = mu
            total_mu += mu

        # Chuẩn hóa Ruspini (đảm bảo tổng độ thuộc đúng bằng 1.0)
        if total_mu > 0:
            for k in memberships:
                memberships[k] = round(memberships[k] / total_mu, 4)

        fuzzy_rank = 0.0
        for idx, (label, *_) in enumerate(boundaries):
            fuzzy_rank += idx * memberships[label]

        max_rank = len(boundaries) - 1
        return memberships, round(fuzzy_rank, 4), max_rank

    def map_fuzzy_variable(
        self, var_name: str, val: float, temp_dev: Optional[float] = None
    ) -> Tuple[Dict[str, float], float, int]:
        """
        Tính độ thuộc mờ hình thang và Rank_Fuzzy cho một biến thời tiết.
        """
        if var_name in ("Tmax", "Tmin"):
            dev = temp_dev if temp_dev is not None else 0.0
            return self._compute_partition_memberships(dev, self.temp_dev_boundaries)
        elif var_name in ("RHmax", "RHmin"):
            return self._compute_partition_memberships(val, self.rh_boundaries)
        elif var_name == "RF":
            return self._compute_partition_memberships(val, self.rf_boundaries)
        else:
            raise ValueError(f"Biến thời tiết không hợp lệ: {var_name}")

    # =========================================================================
    # TẠO WC VÀ CWC TỪ CHUỖI DỮ LIỆU NGÀY (ĐỊNH NGHĨA 1 & ĐỊNH NGHĨA 3)
    # =========================================================================
    def compute_wc_and_cwc_from_window(
        self,
        wc_id: int,
        window_df: pd.DataFrame,
        variables: List[str],
        historical_normals: Optional[Dict[str, float]] = None,
    ) -> Tuple[WeatherCondition, CategoryWeatherCondition]:
        """
        Tính toán một cặp (WC, CWC) từ cửa sổ d ngày liên tiếp.
        - Theo Định nghĩa 1 (Mazumdar & Medha, 2008):
          + Tmax, Tmin, RHmax, RHmin: tính bằng trung bình cộng (mean) trong d ngày.
          + RF (lượng mưa): tính bằng tổng tích lũy (sum) trong d ngày.
        """
        d = len(window_df)
        start_date = str(window_df.iloc[0]["Date"])
        psi_values: Dict[str, float] = {}
        temp_devs: Dict[str, float] = {}

        for v in WEATHER_VARS:
            if v == "RF":
                psi_values[v] = float(window_df[v].sum())
            else:
                psi_values[v] = float(window_df[v].mean())

        # Tính độ lệch nhiệt độ ΔT so với Historical Normal 30 năm
        for tv in ("Tmax", "Tmin"):
            if f"{tv}_normal" in window_df.columns:
                normal_val = float(window_df[f"{tv}_normal"].mean())
            elif historical_normals and tv in historical_normals:
                normal_val = historical_normals[tv]
            else:
                normal_val = 32.0 if tv == "Tmax" else 18.5
            temp_devs[tv] = psi_values[tv] - normal_val

        wc = WeatherCondition(
            wc_id=wc_id,
            n=len(variables),
            start_date=start_date,
            duration=d,
            variables=list(variables),
            psi_values={v: round(psi_values[v], 2) for v in variables},
            temp_deviations={k: round(v, 2) for k, v in temp_devs.items()},
        )

        categories: Dict[str, str] = {}
        fuzzy_memberships: Dict[str, Dict[str, float]] = {}
        fuzzy_ranks: Dict[str, float] = {}
        max_ranks: Dict[str, int] = {}

        for v in variables:
            val = psi_values[v]
            if v in ("Tmax", "Tmin"):
                categories[v] = self.map_crisp_temp_deviation(temp_devs[v])
                f_mem, f_rank, m_rank = self.map_fuzzy_variable(v, val, temp_devs[v])
            elif v in ("RHmax", "RHmin"):
                categories[v] = self.map_crisp_rh(val)
                f_mem, f_rank, m_rank = self.map_fuzzy_variable(v, val)
            elif v == "RF":
                categories[v] = self.map_crisp_rf(val)
                f_mem, f_rank, m_rank = self.map_fuzzy_variable(v, val)
            else:
                raise ValueError(f"Unknown variable: {v}")

            fuzzy_memberships[v] = f_mem
            fuzzy_ranks[v] = f_rank
            max_ranks[v] = m_rank

        cwc = CategoryWeatherCondition(
            cwc_id=wc_id,
            n=len(variables),
            start_date=start_date,
            duration=d,
            variables=list(variables),
            psi_values={v: round(psi_values[v], 2) for v in variables},
            categories=categories,
            fuzzy_memberships=fuzzy_memberships,
            fuzzy_ranks=fuzzy_ranks,
            max_ranks=max_ranks,
            temp_deviations={k: round(v, 2) for k, v in temp_devs.items()},
        )
        return wc, cwc

    def extract_period_cwcs(
        self,
        period_df: pd.DataFrame,
        duration: int,
        variables: Optional[List[str]] = None,
        start_id: int = 1,
    ) -> Tuple[List[WeatherCondition], List[CategoryWeatherCondition]]:
        """
        Chia một chu kỳ p_r có D_p ngày thành ⌊D_p / d⌋ mẫu WC(n, d) và CWC(n, d).
        Tối ưu hóa bằng mảng NumPy để tăng tốc thực nghiệm đa hạt giống (Multi-seed evaluation).
        """
        if variables is None:
            variables = list(WEATHER_VARS)
        wcs: List[WeatherCondition] = []
        cwcs: List[CategoryWeatherCondition] = []

        n_rows = len(period_df)
        num_windows = n_rows // duration
        if num_windows == 0:
            return wcs, cwcs

        dates_arr = period_df["Date"].to_numpy()
        var_arrays = {v: period_df[v].to_numpy(dtype=float) for v in WEATHER_VARS}
        tmax_norm_arr = (
            period_df["Tmax_normal"].to_numpy(dtype=float)
            if "Tmax_normal" in period_df.columns
            else np.full(n_rows, 32.0)
        )
        tmin_norm_arr = (
            period_df["Tmin_normal"].to_numpy(dtype=float)
            if "Tmin_normal" in period_df.columns
            else np.full(n_rows, 18.5)
        )

        cur_id = start_id
        for w_idx in range(num_windows):
            s = w_idx * duration
            e = s + duration
            start_date = str(dates_arr[s])

            psi_values: Dict[str, float] = {}
            for v in WEATHER_VARS:
                if v == "RF":
                    psi_values[v] = float(var_arrays[v][s:e].sum())
                else:
                    psi_values[v] = float(var_arrays[v][s:e].mean())

            temp_devs = {
                "Tmax": psi_values["Tmax"] - float(tmax_norm_arr[s:e].mean()),
                "Tmin": psi_values["Tmin"] - float(tmin_norm_arr[s:e].mean()),
            }

            wc = WeatherCondition(
                wc_id=cur_id,
                n=len(variables),
                start_date=start_date,
                duration=duration,
                variables=list(variables),
                psi_values={v: round(psi_values[v], 2) for v in variables},
                temp_deviations={k: round(v, 2) for k, v in temp_devs.items()},
            )

            categories: Dict[str, str] = {}
            fuzzy_memberships: Dict[str, Dict[str, float]] = {}
            fuzzy_ranks: Dict[str, float] = {}
            max_ranks: Dict[str, int] = {}

            for v in variables:
                val = psi_values[v]
                if v in ("Tmax", "Tmin"):
                    categories[v] = self.map_crisp_temp_deviation(temp_devs[v])
                    f_mem, f_rank, m_rank = self.map_fuzzy_variable(v, val, temp_devs[v])
                elif v in ("RHmax", "RHmin"):
                    categories[v] = self.map_crisp_rh(val)
                    f_mem, f_rank, m_rank = self.map_fuzzy_variable(v, val)
                elif v == "RF":
                    categories[v] = self.map_crisp_rf(val)
                    f_mem, f_rank, m_rank = self.map_fuzzy_variable(v, val)
                else:
                    raise ValueError(f"Unknown variable: {v}")

                fuzzy_memberships[v] = f_mem
                fuzzy_ranks[v] = f_rank
                max_ranks[v] = m_rank

            cwc = CategoryWeatherCondition(
                cwc_id=cur_id,
                n=len(variables),
                start_date=start_date,
                duration=duration,
                variables=list(variables),
                psi_values={v: round(psi_values[v], 2) for v in variables},
                categories=categories,
                fuzzy_memberships=fuzzy_memberships,
                fuzzy_ranks=fuzzy_ranks,
                max_ranks=max_ranks,
                temp_deviations={k: round(v, 2) for k, v in temp_devs.items()},
            )
            wcs.append(wc)
            cwcs.append(cwc)
            cur_id += 1

        return wcs, cwcs
