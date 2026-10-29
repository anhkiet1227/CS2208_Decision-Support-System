"""
Module 1: wct_mapper.py
Ánh xạ Điều kiện Thời tiết Số thực (WC), Danh mục Cứng (Crisp WCT) và Danh mục Mờ (Fuzzy WCT).

Thực hiện:
- Định nghĩa 1 (Weather Condition - WC): Tính thống kê ψ(V) cho d = 1 hoặc d = 5 ngày.
- Bảng 2.2 (Weather Category Table - WCT): Phân loại chuẩn Cục Khí tượng Ấn Độ (IMD).
- Định nghĩa 3 (Category-based Weather Condition - CWC): Ánh xạ ψ(V) sang nhãn danh mục cứng C(V).
- Giải pháp đề xuất 1 (Fuzzy CWC): Hàm thuộc mờ hình thang μ(x; a, b, c, d) thỏa mãn phân hoạch Ruspini
  và chỉ số cấp bậc liên tục hóa Rank_Fuzzy(X_k) nhằm triệt tiêu Hiệu ứng vách đá (Boundary Cliff Effect).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


# Danh sách 5 biến thời tiết chuẩn theo bài báo Alugubelly et al. (2019)
WEATHER_VARS = ["Tmax", "Tmin", "RHmax", "RHmin", "RF"]

# Các tổ hợp biến V theo số lượng biến n = 1..5 (Bảng 4.1 / Table 5 bài báo gốc)
BASELINE_VAR_COMBINATIONS: Dict[int, List[str]] = {
    1: ["Tmax"],
    2: ["Tmax", "Tmin"],
    3: ["Tmax", "Tmin", "RHmax"],
    4: ["Tmax", "Tmin", "RHmax", "RHmin"],
    5: ["Tmax", "Tmin", "RHmax", "RHmin", "RF"],
}


# Thứ tự cấp bậc tự nhiên (Ordinal Ranks) của các danh mục chuẩn IMD (Bảng 2.2)
TEMP_CATEGORIES_ORDERED = [
    "Large fall",        # 0: <= -7 °C
    "Marked fall",       # 1: -6 to -5 °C
    "Appreciable fall",  # 2: -4 to -3 °C
    "Fall",              # 3: -2 °C (trong bài báo gốc gộp Rise/fall hoặc tách theo chiều)
    "Little change",     # 4: -1, 0, 1 °C
    "Rise",              # 5: +2 °C
    "Appreciable rise",  # 6: 3 to 4 °C
    "Marked rise",       # 7: 5 to 6 °C
    "Large rise",        # 8: > 7 °C
]

# Danh mục chuẩn IMD trong Bảng 2.2 bài báo gốc (8 nhãn nhiệt độ khi gộp Rise/fall)
IMD_TEMP_CATEGORIES = [
    "Large fall",        # 0: dev <= -6.5
    "Marked fall",       # 1: -6.5 < dev <= -4.5
    "Appreciable fall",  # 2: -4.5 < dev <= -2.5
    "Little change",     # 3: -1.5 <= dev <= 1.5
    "Rise/fall",         # 4: |dev| in (1.5, 2.5]
    "Appreciable rise",  # 5: 2.5 <= dev < 4.5
    "Marked rise",       # 6: 4.5 <= dev <= 6.5
    "Large rise",        # 7: dev > 6.5
]

RH_CATEGORIES_ORDERED = [
    "Low",        # 0: 0 - 30 %
    "Moderate",   # 1: 31 - 60 %
    "High",       # 2: 61 - 80 %
    "Very high",  # 3: >= 81 %
]

RF_CATEGORIES_ORDERED = [
    "No rain",               # 0: 0.0 mm
    "Very light rain",       # 1: 0.1 - 2.4 mm
    "Light rain",            # 2: 2.5 - 7.5 mm
    "Moderate rain",         # 3: 7.6 - 35.5 mm
    "Rather heavy rain",     # 4: 35.6 - 64.4 mm
    "Heavy rain",            # 5: 64.5 - 124.4 mm
    "Very heavy rain",       # 6: 124.5 - 244.4 mm
    "Extremely heavy rain",  # 7: >= 244.5 mm
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
    - max_ranks: Số bậc tối đa (M_k - 1) của từng biến để chuẩn hóa khoảng cách Gower
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
    Hỗ trợ trường hợp suy biến (vai trái a == b hoặc vai phải c == d).
    """
    if x <= a or x >= d:
        # Xử lý biên mở vô cùng ở hai đầu mút ngoài cùng
        if a == b and x <= b:
            return 1.0
        if c == d and x >= c:
            return 1.0
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a) if b > a else 1.0
    if c < x < d:
        return (d - x) / (d - c) if d > c else 1.0
    return 0.0


class WCTMapper:
    """
    Bộ ánh xạ Bảng phân loại thời tiết WCT (Crisp IMD Standard & Proposed Fuzzy Trapezoidal WCT).
    """

    def __init__(self, transition_ratio: float = 0.10):
        """
        :param transition_ratio: Độ rộng vùng chuyển tiếp mềm δ quanh ranh giới WCT (mặc định ±10%).
        """
        self.transition_ratio = transition_ratio
        self._init_fuzzy_partitions()

    def _init_fuzzy_partitions(self) -> None:
        """
        Thiết lập các hình thang mờ liên tiếp thỏa mãn phân hoạch Ruspini: ∑_m μ_{k,m}(x) = 1.
        Tại mỗi ranh giới cứng B_m giữa danh mục m và m+1, vùng chuyển tiếp là [B_m - δ_m, B_m + δ_m].
        """
        # 1. Biến Độ ẩm tương đối RH (%): Ngưỡng cứng IMD tại 30.5, 60.5, 80.5
        # Độ rộng vùng chuyển tiếp δ ≈ 2.5% - 3.0% quanh ngưỡng
        self.rh_boundaries = [
            ("Low", -1e6, -1e6, 28.0, 33.0),
            ("Moderate", 28.0, 33.0, 58.0, 63.0),
            ("High", 58.0, 63.0, 78.0, 83.0),
            ("Very high", 78.0, 83.0, 1e6, 1e6),
        ]

        # 2. Biến Lượng mưa RF (mm): Ngưỡng cứng IMD tại 0.05, 2.45, 7.55, 35.55, 64.45, 124.45, 244.45
        self.rf_boundaries = [
            ("No rain", -1e6, -1e6, 0.02, 0.15),
            ("Very light rain", 0.02, 0.15, 2.15, 2.75),
            ("Light rain", 2.15, 2.75, 6.90, 8.20),
            ("Moderate rain", 6.90, 8.20, 32.5, 38.5),
            ("Rather heavy rain", 32.5, 38.5, 60.5, 68.5),
            ("Heavy rain", 60.5, 68.5, 118.0, 131.0),
            ("Very heavy rain", 118.0, 131.0, 232.0, 257.0),
            ("Extremely heavy rain", 232.0, 257.0, 1e6, 1e6),
        ]

        # 3. Biến Độ lệch Nhiệt độ ΔT (°C so với Historical Normal):
        # Sắp xếp theo trục số thực từ lạnh (Large fall <= -7) đến nóng (Large rise > 7)
        self.temp_dev_boundaries = [
            ("Large fall", -1e6, -1e6, -7.0, -6.0),
            ("Marked fall", -7.0, -6.0, -5.0, -4.0),
            ("Appreciable fall", -5.0, -4.0, -3.0, -2.3),
            ("Fall", -3.0, -2.3, -1.8, -1.2),
            ("Little change", -1.8, -1.2, 1.2, 1.8),
            ("Rise", 1.2, 1.8, 2.3, 3.0),
            ("Appreciable rise", 2.3, 3.0, 4.0, 5.0),
            ("Marked rise", 4.0, 5.0, 6.0, 7.0),
            ("Large rise", 6.0, 7.0, 1e6, 1e6),
        ]

    # =========================================================================
    # CRISP WCT MAPPING (BẢNG 2.2 - CHUẨN IMD TRONG BÀI BÁO GỐC)
    # =========================================================================
    @staticmethod
    def map_crisp_temp_deviation(dev: float) -> str:
        """
        Ánh xạ độ lệch nhiệt độ ΔT (°C) sang nhãn danh mục chuẩn IMD (Bảng 2.2).
        -1, 0, 1 -> Little change
        2 or -2  -> Rise/fall
        3 - 4    -> Appreciable rise
        -3 to -4 -> Appreciable fall
        5 - 6    -> Marked rise
        -5 to -6 -> Marked fall
        > 7      -> Large rise
        <= -7    -> Large fall
        """
        r = round(dev)
        if r <= -7:
            return "Large fall"
        elif -6 <= r <= -5:
            return "Marked fall"
        elif -4 <= r <= -3:
            return "Appreciable fall"
        elif r == -2 or r == 2:
            return "Rise/fall"
        elif -1 <= r <= 1:
            return "Little change"
        elif 3 <= r <= 4:
            return "Appreciable rise"
        elif 5 <= r <= 6:
            return "Marked rise"
        else:  # r >= 7
            return "Large rise"

    @staticmethod
    def map_crisp_rh(rh: float) -> str:
        """
        Ánh xạ Độ ẩm tương đối RH (%) sang nhãn danh mục chuẩn IMD (Bảng 2.2).
        0 - 30   -> Low
        31 - 60  -> Moderate
        61 - 80  -> High
        >= 81    -> Very high
        """
        if rh <= 30.0:
            return "Low"
        elif rh <= 60.0:
            return "Moderate"
        elif rh <= 80.0:
            return "High"
        else:
            return "Very high"

    @staticmethod
    def map_crisp_rf(rf: float) -> str:
        """
        Ánh xạ Lượng mưa RF (mm) sang nhãn danh mục chuẩn IMD (Bảng 2.2).
        0.0          -> No rain
        0.1 - 2.4    -> Very light rain
        2.5 - 7.5    -> Light rain
        7.6 - 35.5   -> Moderate rain
        35.6 - 64.4  -> Rather heavy rain
        64.5 - 124.4 -> Heavy rain
        124.5 - 244.4-> Very heavy rain
        >= 244.5     -> Extremely heavy rain
        """
        if rf <= 0.0:
            return "No rain"
        elif rf <= 2.4:
            return "Very light rain"
        elif rf <= 7.5:
            return "Light rain"
        elif rf <= 35.5:
            return "Moderate rain"
        elif rf <= 64.4:
            return "Rather heavy rain"
        elif rf <= 124.4:
            return "Heavy rain"
        elif rf <= 244.4:
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
        Trả về: (dict_memberships, fuzzy_rank, max_rank)
        """
        memberships: Dict[str, float] = {}
        total_mu = 0.0
        for label, a, b, c, d in boundaries:
            mu = trapezoidal_mf(x, a, b, c, d)
            memberships[label] = round(mu, 4)
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

        # Tính độ lệch nhiệt độ so với Historical Normal 30 năm
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
        """
        if variables is None:
            variables = list(WEATHER_VARS)
        wcs: List[WeatherCondition] = []
        cwcs: List[CategoryWeatherCondition] = []

        n_rows = len(period_df)
        num_windows = n_rows // duration
        cur_id = start_id

        for w_idx in range(num_windows):
            sub_df = period_df.iloc[w_idx * duration : (w_idx + 1) * duration]
            wc, cwc = self.compute_wc_and_cwc_from_window(
                wc_id=cur_id,
                window_df=sub_df,
                variables=variables,
            )
            wcs.append(wc)
            cwcs.append(cwc)
            cur_id += 1

        return wcs, cwcs

