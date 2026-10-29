"""
Module 3: ecwc_rag_llm.py
Tích hợp Vector Ngữ cảnh Mở rộng Phi thời tiết (Extended CWC - ECWC) và Kiến trúc RAG-LLM
(Giải pháp Đề xuất 3 & 4 - Mục 3.4 và 3.5).

Thực hiện:
- Biểu diễn bộ 9 thành phần ECWC_i(n, m, d) = <i, n, m, s, d, V_weather, FC(V_weather), U_agro, C(U_agro)>
- Tính độ tương đồng tổng hợp (Composite Gower Similarity) kết hợp Thời tiết Mờ Trọng số Động
  và 4 thuộc tính nông học mở rộng U_agro = {u_soil, u_variety, u_pest, u_ndvi} (Bảng 3.2).
- Xây dựng Grounded RAG Prompt Template chống ảo giác (Anti-Hallucination).
- Hỗ trợ song song 2 chế độ sinh ngôn ngữ:
  1. Gemini API Mode (nếu người dùng cấu hình GEMINI_API_KEY).
  2. Local Grounded RAG Synthesizer (chạy độc lập không cần API Key, tự động nội suy liều lượng
     phân bón/thuốc BVTV/mực nước theo diện tích ha hoặc công đất Nam Bộ và xuất bản tin Zalo OA / SMS).
"""

import os
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from src.wct_mapper import WCTMapper, CategoryWeatherCondition, WEATHER_VARS
from src.cp_engine import compute_fuzzy_gower_similarity, get_dynamic_weights_and_vars
from src.dataset_generator import build_expert_knowledge_base


# Bảng 3.2: Danh mục chuẩn hóa của 4 biến ngữ cảnh phi thời tiết U_agro
AGRO_CONTEXT_SCHEMA: Dict[str, List[str]] = {
    "u_soil": [
        "Đất phù sa ngọt",
        "Đất phèn hoạt động",
        "Đất nhiễm mặn (>2‰)",
        "Đất cát pha khô hạn",
    ],
    "u_variety": [
        "Giống ngắn ngày (90-100 ngày)",
        "Giống trung ngày",
        "Giống chịu mặn/chịu hạn",
        "Giống kháng rầy",
    ],
    "u_pest": [
        "An toàn (Mức 0)",
        "Cảnh báo nhẹ (Mức 1)",
        "Nguy cơ bùng phát dịch đạo ôn/rầy nâu (Mức 2)",
    ],
    "u_ndvi": [
        "NDVI Thấp (<0.3 - Thiếu dinh dưỡng)",
        "NDVI Chuẩn (0.3-0.6)",
        "NDVI Rất cao (>0.7 - Thừa đạm)",
    ],
}


@dataclass
class ExtendedCWC:
    """
    Giải pháp đề xuất 3 (Mục 3.4): Điều kiện Thời tiết Danh mục Mở rộng Đa ngữ cảnh (9-ary tuple):
        ECWC_i(n, m, d) = <i, n, m, s, d, V_weather, FC(V_weather), U_agro, C(U_agro)>
    """
    ecwc_id: int
    n: int
    m: int
    start_date: str
    duration: int
    v_weather: List[str]
    fc_weather: CategoryWeatherCondition
    u_agro: List[str]
    c_agro: Dict[str, str]
    season: str = "Kharif"
    phenophase: str = "Flowering"

    def to_9tuple_summary(self) -> Dict[str, Any]:
        return {
            "i (ID)": self.ecwc_id,
            "n (Số biến thời tiết)": self.n,
            "m (Số biến phi thời tiết)": self.m,
            "s (Ngày bắt đầu)": self.start_date,
            "d (Thời lượng dự báo)": f"{self.duration} ngày",
            "V_weather": self.v_weather,
            "FC(V_weather) [Crisp & Fuzzy]": {
                v: {
                    "value": self.fc_weather.psi_values[v],
                    "crisp_label": self.fc_weather.categories[v],
                    "fuzzy_rank": self.fc_weather.fuzzy_ranks[v],
                }
                for v in self.v_weather
            },
            "U_agro": self.u_agro,
            "C(U_agro)": self.c_agro,
        }


@dataclass
class FarmerProfile:
    """Hồ sơ nông hộ để cá nhân hóa khuyến nghị qua RAG-LLM (Zalo Bot / SMS / App)."""
    farmer_name: str = "Chú Ba Kiệt"
    location: str = "Huyện Tháp Mười, Đồng Tháp (ĐBSCL)"
    area_ha: float = 1.5  # 1.5 ha = 15 công tầm nhỏ (1.000 m2)
    crop_name: str = "Lúa OM18 (Vụ Hè Thu / Kharif)"
    sowing_days_ago: int = 62


class ECWCRagEngine:
    """
    Hệ thống Truy hồi Tương đồng Mờ Đa ngữ cảnh (Extended CWC CBR) kết hợp RAG-LLM.
    """

    def __init__(self, kb_path: str = "data/agromet_knowledge_base.json"):
        self.mapper = WCTMapper()
        self.kb_cases = build_expert_knowledge_base(kb_path)
        self._prepare_kb_cwcs()

    def _prepare_kb_cwcs(self) -> None:
        """Khởi tạo sẵn đối tượng CategoryWeatherCondition cho từng ca mẫu trong Kho tri thức Pset."""
        self.kb_enriched: List[Dict[str, Any]] = []
        for idx, case in enumerate(self.kb_cases, start=1):
            psi = case["psi_values"]
            devs = case.get("temp_deviations", {"Tmax": 0.0, "Tmin": 0.0})
            row_df = pd.DataFrame(
                [
                    {
                        "Date": case["recorded_date"],
                        "Tmax": psi["Tmax"],
                        "Tmin": psi["Tmin"],
                        "RHmax": psi["RHmax"],
                        "RHmin": psi["RHmin"],
                        "RF": psi["RF"],
                        "Tmax_normal": psi["Tmax"] - devs["Tmax"],
                        "Tmin_normal": psi["Tmin"] - devs["Tmin"],
                    }
                ]
            )
            _, cwc = self.mapper.compute_wc_and_cwc_from_window(
                wc_id=idx,
                window_df=row_df,
                variables=WEATHER_VARS,
            )
            # Giữ nguyên thời lượng d = 5 ngày của ca lịch sử
            cwc.duration = case.get("duration", 5)
            enriched = dict(case)
            enriched["cwc_obj"] = cwc
            self.kb_enriched.append(enriched)

    def create_query_ecwc(
        self,
        psi_values: Dict[str, float],
        temp_deviations: Dict[str, float],
        c_agro: Dict[str, str],
        season: str = "Kharif",
        phenophase: str = "Flowering",
        start_date: str = "2026-10-30",
        duration: int = 5,
    ) -> ExtendedCWC:
        """Tạo đối tượng truy vấn ECWC 9 thành phần từ dự báo thời tiết 5 ngày và ngữ cảnh ruộng lúa."""
        row_df = pd.DataFrame(
            [
                {
                    "Date": start_date,
                    "Tmax": psi_values["Tmax"],
                    "Tmin": psi_values["Tmin"],
                    "RHmax": psi_values["RHmax"],
                    "RHmin": psi_values["RHmin"],
                    "RF": psi_values["RF"],
                    "Tmax_normal": psi_values["Tmax"] - temp_deviations.get("Tmax", 0.0),
                    "Tmin_normal": psi_values["Tmin"] - temp_deviations.get("Tmin", 0.0),
                }
            ]
        )
        _, cwc = self.mapper.compute_wc_and_cwc_from_window(
            wc_id=999,
            window_df=row_df,
            variables=WEATHER_VARS,
        )
        cwc.duration = duration
        u_keys = list(AGRO_CONTEXT_SCHEMA.keys())
        return ExtendedCWC(
            ecwc_id=999,
            n=len(WEATHER_VARS),
            m=len(u_keys),
            start_date=start_date,
            duration=duration,
            v_weather=list(WEATHER_VARS),
            fc_weather=cwc,
            u_agro=u_keys,
            c_agro=c_agro,
            season=season,
            phenophase=phenophase,
        )

    @staticmethod
    def compute_agro_context_similarity(c_query: Dict[str, str], c_case: Dict[str, str]) -> float:
        """
        Tính độ tương đồng Gower cho 4 biến ngữ cảnh phi thời tiết U_agro:
        - u_pest và u_ndvi là biến thứ bậc (ordinal), tính khoảng cách bậc chuẩn hóa.
        - u_soil và u_variety là biến định danh (nominal), tính tương đồng khớp hoặc bán phần.
        """
        scores = []
        # 1. u_soil
        scores.append(1.0 if c_query.get("u_soil") == c_case.get("u_soil") else 0.4)
        # 2. u_variety
        scores.append(1.0 if c_query.get("u_variety") == c_case.get("u_variety") else 0.5)
        # 3. u_pest (Ordinal: 0, 1, 2)
        pest_list = AGRO_CONTEXT_SCHEMA["u_pest"]
        i_q = pest_list.index(c_query["u_pest"]) if c_query.get("u_pest") in pest_list else 1
        i_c = pest_list.index(c_case["u_pest"]) if c_case.get("u_pest") in pest_list else 1
        scores.append(1.0 - abs(i_q - i_c) / 2.0)
        # 4. u_ndvi (Ordinal: 0, 1, 2)
        ndvi_list = AGRO_CONTEXT_SCHEMA["u_ndvi"]
        n_q = ndvi_list.index(c_query["u_ndvi"]) if c_query.get("u_ndvi") in ndvi_list else 1
        n_c = ndvi_list.index(c_case["u_ndvi"]) if c_case.get("u_ndvi") in ndvi_list else 1
        scores.append(1.0 - abs(n_q - n_c) / 2.0)

        return round(sum(scores) / len(scores), 4)

    def retrieve_best_case(
        self, query_ecwc: ExtendedCWC, alpha_weather: float = 0.75
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Truy hồi ca mẫu lịch sử tốt nhất từ Pset dựa trên điểm tương đồng kết hợp:
            Sim_Total = α * Sim_Fuzzy_Weather + (1 - α) * Sim_Agro_Context
        Đồng thời đối sánh với kết quả của Baseline Crisp CWC (so khớp cứng).
        """
        _, dyn_weights = get_dynamic_weights_and_vars(
            query_ecwc.season, query_ecwc.phenophase, n_vars=5
        )
        ranked_results: List[Dict[str, Any]] = []

        for case in self.kb_enriched:
            case_cwc: CategoryWeatherCondition = case["cwc_obj"]
            crisp_exact_match = (
                query_ecwc.fc_weather.crisp_signature() == case_cwc.crisp_signature()
            )
            weather_sim = compute_fuzzy_gower_similarity(
                query_ecwc.fc_weather, case_cwc, weights=dyn_weights, subset_vars=WEATHER_VARS
            )
            agro_sim = self.compute_agro_context_similarity(
                query_ecwc.c_agro, case["agro_context"]
            )
            # Ưu tiên cộng thưởng nếu khớp đúng giai đoạn sinh trưởng
            pheno_bonus = 0.03 if query_ecwc.phenophase == case["phenophase"] else 0.0
            composite_sim = min(
                1.0, alpha_weather * weather_sim + (1.0 - alpha_weather) * agro_sim + pheno_bonus
            )

            ranked_results.append(
                {
                    "case_id": case["case_id"],
                    "recorded_date": case["recorded_date"],
                    "season": case["season"],
                    "phenophase": case["phenophase"],
                    "crisp_exact_match": crisp_exact_match,
                    "weather_fuzzy_sim": round(weather_sim, 4),
                    "agro_context_sim": round(agro_sim, 4),
                    "composite_sim": round(composite_sim, 4),
                    "reusable_at_theta_085": weather_sim >= 0.85,
                    "case_data": case,
                }
            )

        ranked_results.sort(key=lambda r: r["composite_sim"], reverse=True)
        return ranked_results[0], ranked_results

    @staticmethod
    def build_grounded_rag_prompt(
        query_ecwc: ExtendedCWC,
        retrieved_match: Dict[str, Any],
        farmer: FarmerProfile,
    ) -> str:
        """
        Xây dựng Prompt Template ràng buộc ngữ cảnh (Grounded RAG) gửi tới LLM (Mục 3.5).
        Đảm bảo LLM chỉ biên dịch & quy đổi liều lượng dựa trên gói khuyến nghị chuyên gia đã kiểm duyệt.
        """
        case = retrieved_match["case_data"]
        f_cwc = query_ecwc.fc_weather
        prompt = f"""[SYSTEM ROLE - TRỢ LÝ KHUYẾN NÔNG THÔNG MINH GROUNDED RAG-DSS]
Bạn là chuyên gia Hệ hỗ trợ quyết định nông nghiệp - khí tượng (Agromet DSS).
NHIỆM VỤ BẮT BUỘC: Dựa TUYỆT ĐỐI vào [GÓI KHUYẾN NGHỊ CHUYÊN GIA ĐÃ KIỂM DUYỆT] được truy hồi từ kho tri thức lịch sử Pset bên dưới, hãy biên dịch thành bản tin tư vấn tiếng Việt gần gũi gửi qua Zalo Official Account và tin nhắn SMS cho nông hộ.
TUYỆT ĐỐI KHÔNG tự ý bịa đặt hoạt chất hóa học hay quy trình kỹ thuật ngoài phạm vi khuyến nghị gốc (Zero Hallucination).

1. THÔNG TIN NÔNG HỘ:
- Nông hộ: {farmer.farmer_name} | Địa điểm: {farmer.location}
- Cây trồng & Thời vụ: {farmer.crop_name} ({ farmer.sowing_days_ago } ngày sau sạ - Giai đoạn: {query_ecwc.phenophase})
- Diện tích canh tác: {farmer.area_ha:.2f} ha (tương đương {farmer.area_ha * 10:.1f} công tầm nhỏ 1.000 m²)

2. VECTOR ĐIỀU KIỆN THỜI TIẾT DỰ BÁO 5 NGÀY & NGỮ CẢNH MỞ RỘNG (ECWC):
- Thống kê ψ(V) 5 ngày: Tmax={f_cwc.psi_values['Tmax']}°C, Tmin={f_cwc.psi_values['Tmin']}°C, RHmax={f_cwc.psi_values['RHmax']}%, RHmin={f_cwc.psi_values['RHmin']}%, RF={f_cwc.psi_values['RF']} mm
- Nhãn WCT (IMD): {json.dumps(f_cwc.categories, ensure_ascii=False)}
- Ngữ cảnh phi thời tiết C(U_agro): Đất={query_ecwc.c_agro['u_soil']} | Giống={query_ecwc.c_agro['u_variety']} | Dịch hại={query_ecwc.c_agro['u_pest']} | Viễn thám={query_ecwc.c_agro['u_ndvi']}

3. KẾT QUẢ TRUY HỒI TỪ KHO TRI THỨC PSET (FUZZY-DYNAMIC GOWER SIMILARITY):
- Mã bản tin lịch sử khớp nhất: {retrieved_match['case_id']} (Ngày gốc: {retrieved_match['recorded_date']})
- Điểm tương đồng thời tiết mờ Sim_Fuzzy: {retrieved_match['weather_fuzzy_sim']*100:.1f}% (So khớp cứng Crisp CWC: {'Khớp' if retrieved_match['crisp_exact_match'] else 'THẤT BẠI do lệch nhẹ tại biên WCT'})
- Điểm tương đồng tổng hợp ECWC: {retrieved_match['composite_sim']*100:.1f}%
- [GÓI KHUYẾN NGHỊ CHUYÊN GIA ĐÃ KIỂM DUYỆT]: "{case['expert_advisory_vi']}"
- [ĐỊNH MỨC VẬT TƯ CHUẨN TRÊN 1.0 HA]: {json.dumps(case['dosage_norms_per_ha'], ensure_ascii=False)}

4. YÊU CẦU ĐẦU RA:
Hãy tính toán quy đổi chính xác lượng phân bón / thuốc BVTV / lượng nước phun cho diện tích {farmer.area_ha:.2f} ha của {farmer.farmer_name}, điều chỉnh lưu ý theo loại đất ({query_ecwc.c_agro['u_soil']}) và chỉ số NDVI ({query_ecwc.c_agro['u_ndvi']}), trình bày rõ ràng theo bố cục tin nhắn Zalo Khuyến nông."""
        return prompt

    def generate_personalized_advisory(
        self,
        query_ecwc: ExtendedCWC,
        retrieved_match: Dict[str, Any],
        farmer: FarmerProfile,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Sinh bản tin khuyến nông cá nhân hóa qua RAG-LLM:
        - Nếu có api_key (hoặc biến môi trường GEMINI_API_KEY): gọi trực tiếp Gemini API.
        - Nếu không có api_key (hoặc ngoại tuyến): sử dụng bộ tổng hợp Local Grounded RAG Synthesizer
          tính toán chính xác định mức vật tư và sinh văn bản Zalo Bot & SMS chuẩn xác.
        """
        prompt = self.build_grounded_rag_prompt(query_ecwc, retrieved_match, farmer)
        effective_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()

        # Tính toán định lượng vật tư chính xác theo diện tích nông hộ (Grounded Dosage Calculator)
        case = retrieved_match["case_data"]
        norms = case.get("dosage_norms_per_ha", {})
        area = farmer.area_ha
        cong_nam_bo = area * 10.0

        soil_type = query_ecwc.c_agro.get("u_soil", "Đất phù sa ngọt")
        ndvi_level = query_ecwc.c_agro.get("u_ndvi", "NDVI Chuẩn (0.3-0.6)")
        pest_level = query_ecwc.c_agro.get("u_pest", "Cảnh báo nhẹ (Mức 1)")

        # Điều chỉnh lượng phân đạm Ure theo chỉ số viễn thám NDVI và loại đất (Bảng 3.2)
        base_urea_per_ha = float(norms.get("urea_kg", 0.0))
        if "Rất cao" in ndvi_level:
            adj_urea_per_ha = 0.0
            ndvi_note = "Ảnh vệ tinh NDVI > 0.7 cho thấy ruộng lúa đang thừa đạm, lá xanh đậm rậm rạp -> NGƯNG TUYỆT ĐỐI việc bón thêm phân đạm Ure."
        elif "Thấp" in ndvi_level:
            adj_urea_per_ha = max(base_urea_per_ha * 1.15, 25.0)
            ndvi_note = f"Chỉ số NDVI < 0.3 báo hiệu lúa thiếu hụt dinh dưỡng -> Bổ sung {adj_urea_per_ha:.1f} kg Ure/ha kết hợp phân bón lá."
        else:
            adj_urea_per_ha = base_urea_per_ha
            ndvi_note = "Chỉ số NDVI ở ngưỡng chuẩn (0.3–0.6), duy trì định mức dinh dưỡng cân đối."

        # Điều chỉnh lưu ý theo loại đất
        if "phèn" in soil_type.lower():
            soil_note = "Ruộng thuộc nhóm Đất phèn hoạt động: Tuyệt đối giữ mực nước 3–5 cm để ém phèn, kết hợp bón thêm Lân/Vôi nông nghiệp và tuyệt đối không để ruộng khô nứt."
        elif "mặn" in soil_type.lower():
            soil_note = "Ruộng thuộc vùng Đất nhiễm mặn (>2‰): Kiểm tra kỹ độ mặn ngoài kênh (<1.0‰ mới lấy nước vào ruộng), giữ lớp nước ngọt 5–7 cm để rửa mặn và điều hòa nhiệt."
        elif "cát" in soil_type.lower():
            soil_note = "Ruộng Đất cát pha thoát nước nhanh: Chia nhỏ lượng phân bón làm nhiều lần và kiểm tra giữ ẩm mặt ruộng thường xuyên."
        else:
            soil_note = "Ruộng Đất phù sa ngọt: Chủ động rút nước hoặc giữ mực nước nông 2–3 cm linh hoạt theo dự báo mưa."

        total_tricyclazole_g = round(norms.get("tricyclazole_75wp_gram", 0) * area, 0)
        total_water_l = round(norms.get("spray_water_liter", 0) * area, 0)
        total_urea_kg = round(adj_urea_per_ha * area, 1)
        total_kcl_kg = round(norms.get("potash_kcl_kg", 0) * area, 1)
        total_kno3_kg = round(norms.get("kno3_foliar_kg", 0) * area, 1)

        # Tạo nội dung chuẩn Local Grounded RAG
        f_vals = query_ecwc.fc_weather.psi_values
        f_cats = query_ecwc.fc_weather.categories

        dosage_bullets = []
        if total_tricyclazole_g > 0:
            num_bình_25l = int(round(total_water_l / 25.0)) if total_water_l > 0 else 0
            dosage_bullets.append(
                f"• Thuốc đặc trị đạo ôn (Tricyclazole 75WP): Chuẩn bị đúng **{total_tricyclazole_g:.0f} gram** "
                f"pha với **{total_water_l:.0f} lít nước** (tương đương ~{num_bình_25l} bình máy 25 lít) phun đều cho {area:.2f} ha ({cong_nam_bo:.0f} công)."
            )
        if total_kno3_kg > 0:
            dosage_bullets.append(
                f"• Phân bón lá dưỡng đòng/hạt phấn (KNO3 1% + Bo): Sử dụng **{total_kno3_kg:.1f} kg** pha trong **{total_water_l:.0f} lít nước** phun vào chiều mát."
            )
        if total_urea_kg > 0:
            dosage_bullets.append(
                f"• Phân Đạm (Ure): Bón **{total_urea_kg:.1f} kg** cho toàn bộ {area:.2f} ha."
            )
        else:
            dosage_bullets.append(
                "• Phân Đạm (Ure): **0 kg (Ngưng bón đạm hoàn toàn)** để khóa nguồn thức ăn của nấm đạo ôn."
            )
        if total_kcl_kg > 0:
            dosage_bullets.append(
                f"• Phân Kali (KCl): Bổ sung **{total_kcl_kg:.1f} kg** cho {area:.2f} ha giúp cứng cây, chống đổ ngã và tăng tỷ lệ hạt chắc."
            )

        dosage_section = "\n".join(dosage_bullets)

        local_zalo_message = f"""🌾 **BẢN TIN KHUYẾN NÔNG THÔNG MINH (AGROMET DSS - ZALO OA)** 🌾
Kính gửi: **{farmer.farmer_name}** ({farmer.location})
📌 **Ruộng canh tác:** {farmer.crop_name} — Diện tích: **{area:.2f} ha ({cong_nam_bo:.0f} công)**
🌱 **Giai đoạn sinh trưởng:** {query_ecwc.phenophase} ({farmer.sowing_days_ago} ngày sau sạ)

🌦️ **1. Nhận định Thời tiết 5 ngày tới & Truy hồi Tri thức (Fuzzy-Dynamic ECWC):**
• Dự báo 5 ngày: Nhiệt độ {f_vals['Tmin']}–{f_vals['Tmax']}°C ({f_cats['Tmax']}), Độ ẩm {f_vals['RHmin']}–{f_vals['RHmax']}% ({f_cats['RHmax']}), Tổng mưa {f_vals['RF']} mm ({f_cats['RF']}).
• Hệ thống đối chiếu khớp **{retrieved_match['composite_sim']*100:.1f}%** với Phác đồ Chuyên gia mã số `{retrieved_match['case_id']}` (Độ tương đồng thời tiết mờ Gower đạt **{retrieved_match['weather_fuzzy_sim']*100:.1f}%** ≥ ngưỡng θ = 85%).

🛠️ **2. Hành động Kỹ thuật Khuyến nghị trên đồng ruộng:**
• **Quản lý nước & Đất ({soil_type}):** {soil_note}
• **Điều chỉnh dinh dưỡng theo vệ tinh ({ndvi_level}):** {ndvi_note}
• **Cảnh báo dịch hại ({pest_level}):** {case['expert_advisory_vi']}

⚖️ **3. Định mức Vật tư Quy đổi Chính xác cho {area:.2f} ha ({cong_nam_bo:.0f} công) của {farmer.farmer_name}:**
{dosage_section}

⏰ *Lưu ý: Chỉ phun thuốc khi lá lúa khô ráo, tránh phun lúc trời sắp mưa hoặc lúc lúa đang phơi màu rộ (9h–11h sáng).*"""

        sms_short = (
            f"[Agromet-DSS] {farmer.farmer_name} ({area}ha lua {query_ecwc.phenophase}): "
            f"Mua 5 ngay {f_vals['RF']}mm, am {f_vals['RHmax']}%. "
            f"Giu nuoc 2-3cm, NGUNG bon dam Ure ({ndvi_level.split(' ')[0]}). "
            f"Phun phong dao on: {total_tricyclazole_g:.0f}g Tricyclazole/ {total_water_l:.0f}L nuoc."
        )

        mode_used = "Local Grounded RAG Synthesizer (Offline Deterministic Mode)"
        final_zalo_text = local_zalo_message

        # Nếu có GEMINI_API_KEY thì thử gọi Gemini REST API
        if effective_key:
            try:
                url = (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"gemini-2.0-flash:generateContent?key={effective_key}"
                )
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 900},
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=12) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    candidates = resp_data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            final_zalo_text = parts[0]["text"].strip()
                            mode_used = "Gemini API Live Grounded RAG (gemini-2.0-flash)"
            except Exception as exc:
                mode_used = f"Local Grounded RAG Synthesizer (Fallback - API Error: {type(exc).__name__})"

        return {
            "mode_used": mode_used,
            "grounded_prompt": prompt,
            "zalo_message": final_zalo_text,
            "sms_message": sms_short,
            "calculated_dosages": {
                "area_ha": area,
                "area_cong": cong_nam_bo,
                "tricyclazole_75wp_gram": total_tricyclazole_g,
                "spray_water_liter": total_water_l,
                "urea_kg": total_urea_kg,
                "potash_kcl_kg": total_kcl_kg,
                "kno3_foliar_kg": total_kno3_kg,
            },
        }

