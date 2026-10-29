"""
Giao diện Web Thực nghiệm & Mô phỏng Hệ Hỗ Trợ Quyết Định Khuyến Nông Thông Minh (Streamlit App)
Chạy ứng dụng bằng lệnh:
    streamlit run app.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.wct_mapper import WCTMapper, WEATHER_VARS
from src.cp_engine import (
    CPEngine,
    RICE_PHENOPHASE_WEEKS,
    DYNAMIC_WEIGHT_MATRIX,
    compute_fuzzy_gower_similarity,
    get_dynamic_weights_and_vars,
    get_paper_benchmark_tables,
)
from src.dataset_generator import (
    get_table1_dataframe,
    generate_30year_weather_dataset,
    build_expert_knowledge_base,
)
from src.ecwc_rag_llm import ECWCRagEngine, FarmerProfile, AGRO_CONTEXT_SCHEMA


st.set_page_config(
    page_title="Agromet-DSS | Fuzzy-Dynamic ECWC & RAG-LLM",
    page_icon="🌾",
    layout="wide",
)


@st.cache_resource
def init_engines():
    mapper = WCTMapper()
    cp_engine = CPEngine(mapper=mapper, theta=0.85)
    df_30y = generate_30year_weather_dataset()
    build_expert_knowledge_base()
    rag_engine = ECWCRagEngine()
    return mapper, cp_engine, df_30y, rag_engine


def render_header():
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #0f172a 0%, #1e3a8a 100%);
                    padding: 1.2rem 1.6rem; border-radius: 10px; color: white; margin-bottom: 1.2rem;">
            <h2 style="margin:0; color:white;">🌾 Hệ Hỗ Trợ Ra Quyết Định Nông Nghiệp - Khí Tượng (Agromet-DSS v2.0)</h2>
            <p style="margin:0.35rem 0 0 0; opacity: 0.92; font-size: 0.96rem;">
                <b>Đồ án Môn học Hệ Hỗ Trợ Quyết Định (CS2208)</b> — Học viên: <b>Võ Anh Kiệt (MSHV: 26848109)</b> |
                GVHD: <b>TS. Tôn Thất Hòa An</b><br/>
                <i>Từ mô hình CWC gốc (Alugubelly et al., 2019) đến kiến trúc cải tiến Fuzzy-Dynamic ECWC + Grounded RAG-LLM</i>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    render_header()
    mapper, cp_engine, df_30y, rag_engine = init_engines()

    with st.sidebar:
        st.header("⚙️ Cấu hình Hệ thống DSS")
        theta_val = st.slider(
            "Ngưỡng tương đồng mềm Gower (θ)",
            min_value=0.70,
            max_value=0.98,
            value=0.85,
            step=0.01,
            help="Hai điều kiện thời tiết X và Y được coi là tương đồng có thể tái sử dụng khi Sim_Fuzzy(X, Y) >= θ.",
        )
        cp_engine.theta = theta_val
        api_key_input = st.text_input(
            "Gemini API Key (Tùy chọn cho RAG-LLM)",
            type="password",
            help="Nếu để trống, hệ thống tự động sử dụng chế độ Local Grounded RAG Synthesizer nội suy định mức chính xác.",
        )
        st.divider()
        st.markdown(
            """
            **Cấu trúc 3 Mô-đun Lõi (Bảng 4.7):**
            1. `src/wct_mapper.py`: Ánh xạ WC, Crisp WCT & Fuzzy Trapezoidal WCT.
            2. `src/cp_engine.py`: Thuật toán 1, Trọng số động & Độ đo Gower.
            3. `src/ecwc_rag_llm.py`: Ngữ cảnh mở rộng ECWC & Grounded RAG-LLM.
            """
        )

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Tab 1: Ánh xạ Crisp/Fuzzy WCT & Hiệu ứng Vách đá (wct_mapper.py)",
            "📈 Tab 2: Thực nghiệm Độ bao phủ CP 30 năm (cp_engine.py)",
            "🤖 Tab 3: Trợ lý Khuyến nông ECWC + RAG-LLM & Zalo Bot (ecwc_rag_llm.py)",
        ]
    )

    # =========================================================================
    # TAB 1: WCT MAPPER & BOUNDARY CLIFF EFFECT
    # =========================================================================
    with tab1:
        st.subheader("1.1. Kiểm chứng Định nghĩa 1 & Định nghĩa 3 trên dữ liệu chuẩn Bảng 2.1 (01–14/03/2015)")
        df_t1 = get_table1_dataframe()
        col_t1, col_wc = st.columns([1.25, 1.0])

        with col_t1:
            st.dataframe(
                df_t1[["LabelDate", "Tmax", "Tmin", "RHmax", "RHmin", "RF"]],
                use_container_width=True,
                height=290,
            )

        with col_wc:
            wc1, cwc1 = mapper.compute_wc_and_cwc_from_window(1, df_t1.iloc[0:1], WEATHER_VARS)
            wc2, cwc2 = mapper.compute_wc_and_cwc_from_window(2, df_t1.iloc[0:5], WEATHER_VARS)
            st.markdown("**Kết quả tính toán tự động từ `wct_mapper.py`:**")
            st.code(
                f"WC_1(5, 1) [d=1 ngày]:\n  {wc1.to_tuple_str()}\n  Crisp C(V): {cwc1.categories}\n  Fuzzy Rank: {cwc1.fuzzy_ranks}\n\n"
                f"WC_2(5, 5) [d=5 ngày]:\n  {wc2.to_tuple_str()}\n  Crisp C(V): {cwc2.categories}\n  Fuzzy Rank: {cwc2.fuzzy_ranks}",
                language="yaml",
            )

        st.divider()
        st.subheader("1.2. Mô phỏng Tương tác: Triệt tiêu Hiệu ứng Vách đá (Boundary Cliff Effect)")
        c_left, c_right = st.columns([1.0, 1.3])
        with c_left:
            st.markdown("**Điều chỉnh thông số Ngày lịch sử A (Pset) và Ngày dự báo B:**")
            rh_a = st.slider("Độ ẩm RHmin Ngày lịch sử A (%)", 45.0, 75.0, 60.0, 0.2)
            rh_b = st.slider("Độ ẩm RHmin Ngày dự báo B (%)", 45.0, 75.0, 60.8, 0.2)
            rf_a = st.slider("Lượng mưa RF Ngày lịch sử A (mm)", 20.0, 50.0, 35.5, 0.5)
            rf_b = st.slider("Lượng mưa RF Ngày dự báo B (mm)", 20.0, 50.0, 36.2, 0.5)

            df_a = pd.DataFrame([{"Date": "Day-A", "Tmax": 32.0, "Tmin": 21.5, "RHmax": 80.0, "RHmin": rh_a, "RF": rf_a, "Tmax_normal": 32.0, "Tmin_normal": 21.5}])
            df_b = pd.DataFrame([{"Date": "Day-B", "Tmax": 32.1, "Tmin": 21.5, "RHmax": 80.5, "RHmin": rh_b, "RF": rf_b, "Tmax_normal": 32.0, "Tmin_normal": 21.5}])
            _, cwc_a = mapper.compute_wc_and_cwc_from_window(101, df_a, WEATHER_VARS)
            _, cwc_b = mapper.compute_wc_and_cwc_from_window(102, df_b, WEATHER_VARS)

            crisp_ok = cwc_a.crisp_signature() == cwc_b.crisp_signature()
            _, dyn_w = get_dynamic_weights_and_vars("Kharif", "Flowering", 5)
            fuzzy_sim = compute_fuzzy_gower_similarity(cwc_a, cwc_b, weights=dyn_w)

            st.markdown(f"- **Nhãn cứng Ngày A:** `{cwc_a.categories}`")
            st.markdown(f"- **Nhãn cứng Ngày B:** `{cwc_b.categories}`")
            if crisp_ok:
                st.success("Baseline Crisp CWC: KHỚP (Cùng nhãn danh mục)")
            else:
                st.error("Baseline Crisp CWC: BẤT TƯƠNG ĐỒNG (0%) — Bị loại bỏ do vượt ranh giới cứng!")

            if fuzzy_sim >= theta_val:
                st.success(f"Proposed Fuzzy-Dynamic CWC: Sim_Fuzzy = {fuzzy_sim*100:.2f}% (>= θ={theta_val*100:.0f}%) -> TÁI SỬ DỤNG THÀNH CÔNG!")
            else:
                st.warning(f"Proposed Fuzzy-Dynamic CWC: Sim_Fuzzy = {fuzzy_sim*100:.2f}% (< θ={theta_val*100:.0f}%)")

        with c_right:
            fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=150)
            xs = np.linspace(20, 95, 400)
            mu_mod = [mapper.map_fuzzy_variable("RHmin", float(x))[0]["Moderate"] for x in xs]
            mu_high = [mapper.map_fuzzy_variable("RHmin", float(x))[0]["High"] for x in xs]
            ax.plot(xs, mu_mod, label="Moderate (31-60%)", lw=2.3, color="#2563eb")
            ax.plot(xs, mu_high, label="High (61-80%)", lw=2.3, color="#16a34a")
            ax.axvline(60.0, color="#dc2626", ls=":", lw=1.5, label="Ranh giới cứng IMD (60%)")
            ax.axvspan(58.0, 63.0, color="#fef08a", alpha=0.4, label="Vùng chuyển tiếp mờ [58%, 63%]")
            ax.scatter([rh_a, rh_b], [cwc_a.fuzzy_memberships["RHmin"]["Moderate"], cwc_b.fuzzy_memberships["RHmin"]["Moderate"]], color="#dc2626", s=70, zorder=5)
            ax.set_xlabel("Độ ẩm tương đối RHmin (%)")
            ax.set_ylabel("Độ thuộc mờ μ(RH)")
            ax.set_title("Hàm thuộc hình thang μ(RH) và vị trí hai mẫu A, B")
            ax.grid(True, ls="--", alpha=0.35)
            ax.legend(fontsize=8.5)
            st.pyplot(fig)

    # =========================================================================
    # TAB 2: 30-YEAR CP BENCHMARK & SIMULATION
    # =========================================================================
    with tab2:
        st.subheader("2.1. Bảng kết quả thực nghiệm chuẩn & Đối sánh Mô hình Cải tiến (Bảng 4.2 – 4.6)")
        benchmarks = get_paper_benchmark_tables()
        b_choice = st.selectbox(
            "Chọn bảng kết quả thực nghiệm cần xem:",
            [
                "Bảng 4.6: Đối sánh Baseline Crisp CWC vs. Fuzzy-Dynamic CWC (Cải tiến đề xuất)",
                "Bảng 4.2: So sánh WC số thực vs. CWC danh mục tại chu kỳ thứ 5 (pt = Năm)",
                "Bảng 4.3: Độ bao phủ CP (%) với chu kỳ pt = Năm (Table 6 bài báo gốc)",
                "Bảng 4.4: Độ bao phủ CP (%) với chu kỳ pt = Mùa vụ Rabi (Table 7 bài báo gốc)",
                "Bảng 4.5: Độ bao phủ CP (%) giai đoạn Trổ hoa lúa vụ Kharif & Rabi (Tables 8 & 9)",
            ],
        )
        key_map = {
            "Bảng 4.6": "Table_4_6",
            "Bảng 4.2": "Table_4_2",
            "Bảng 4.3": "Table_4_3",
            "Bảng 4.4": "Table_4_4",
            "Bảng 4.5": "Table_4_5",
        }
        selected_key = next(v for k, v in key_map.items() if b_choice.startswith(k))
        st.dataframe(benchmarks[selected_key], use_container_width=True)

        st.divider()
        st.subheader("2.2. Chạy trực tiếp Thuật toán 1 & Fuzzy-Dynamic CWC trên chuỗi 30 năm (1986–2015)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pt_sel = st.selectbox("Loại chu kỳ (Period Type - pt)", ["Kharif", "Rabi", "Summer", "Year"])
        with c2:
            if pt_sel in ("Kharif", "Rabi"):
                pheno_opts = ["(Toàn mùa vụ)"] + list(RICE_PHENOPHASE_WEEKS[pt_sel].keys())
                pheno_sel = st.selectbox("Giai đoạn sinh trưởng lúa (Phenophase)", pheno_opts)
                pheno_arg = None if pheno_sel == "(Toàn mùa vụ)" else pheno_sel
            else:
                st.selectbox("Giai đoạn sinh trưởng lúa", ["(Không áp dụng)"], disabled=True)
                pheno_arg = None
        with c3:
            n_sel = st.selectbox("Số biến thời tiết (n)", [1, 2, 3, 4, 5], index=4)
        with c4:
            d_sel = st.selectbox("Thời lượng dự báo d (ngày)", [1, 5], index=1)

        sim_df = cp_engine.run_algorithm_1_all_modes(
            df=df_30y,
            period_type=pt_sel,
            phenophase=pheno_arg,
            n_vars=n_sel,
            duration=d_sel,
            theta=theta_val,
        )

        col_chart, col_tbl = st.columns([1.25, 1.0])
        with col_chart:
            fig_cp, ax_cp = plt.subplots(figsize=(7.2, 4.2), dpi=150)
            ax_cp.plot(sim_df["Period_Index"], sim_df["Fuzzy_Dynamic_CWC_CP"], "o-", color="#15803d", lw=2.4, label="Proposed Fuzzy-Dynamic CWC")
            ax_cp.plot(sim_df["Period_Index"], sim_df["Fuzzy_CWC_CP"], "s--", color="#0284c7", lw=1.9, label="Fuzzy CWC (Equal Weights)")
            ax_cp.plot(sim_df["Period_Index"], sim_df["Crisp_CWC_CP"], "^-.", color="#ea580c", lw=1.9, label="Baseline Crisp CWC (Algorithm 1)")
            ax_cp.plot(sim_df["Period_Index"], sim_df["Real_WC_CP"], "x:", color="#64748b", lw=1.6, label="Real-valued WC")
            ax_cp.set_xlabel("Chu kỳ tích lũy p_i (2..k)")
            ax_cp.set_ylabel("Độ bao phủ CP (%)")
            ax_cp.set_ylim(-5, 105)
            ax_cp.set_title(f"Đường cong hội tụ CP(p_i / Pset) — pt={pt_sel} ({pheno_arg or 'Full'}), n={n_sel}, d={d_sel}")
            ax_cp.grid(True, ls="--", alpha=0.35)
            ax_cp.legend(loc="lower right", fontsize=8.5)
            st.pyplot(fig_cp)

        with col_tbl:
            st.dataframe(sim_df, use_container_width=True, height=340)

    # =========================================================================
    # TAB 3: EXTENDED CWC + RAG-LLM & ZALO OA SIMULATOR
    # =========================================================================
    with tab3:
        st.subheader("3.1. Thiết lập Dự báo Thời tiết 5 ngày, Ngữ cảnh Đồng ruộng (ECWC) & Hồ sơ Nông hộ")
        col_form1, col_form2, col_form3 = st.columns(3)

        with col_form1:
            st.markdown("**🌦️ Dự báo Thời tiết 5 ngày tới ψ(V):**")
            q_tmax = st.number_input("Nhiệt độ tối đa Tmax (°C)", 20.0, 45.0, 32.1, 0.2)
            q_tmin = st.number_input("Nhiệt độ tối thiểu Tmin (°C)", 10.0, 32.0, 21.8, 0.2)
            q_rhmax = st.number_input("Độ ẩm cực đại RHmax (%)", 30.0, 100.0, 87.5, 0.5)
            q_rhmin = st.number_input("Độ ẩm cực tiểu RHmin (%)", 10.0, 95.0, 60.8, 0.5)
            q_rf = st.number_input("Tổng lượng mưa 5 ngày RF (mm)", 0.0, 350.0, 44.0, 1.0)

        with col_form2:
            st.markdown("**🌱 Ngữ cảnh Nông học Mở rộng U_agro (Bảng 3.2):**")
            q_season = st.selectbox("Mùa vụ canh tác", ["Kharif", "Rabi", "Summer"], index=0)
            q_pheno = st.selectbox(
                "Giai đoạn sinh trưởng lúa",
                ["Flowering", "Maximum Tillering", "Panicle Initiation", "Maturity", "Sowing", "Transplanting"],
                index=0,
            )
            q_soil = st.selectbox("1. Đặc tính thổ nhưỡng (u_soil)", AGRO_CONTEXT_SCHEMA["u_soil"], index=0)
            q_var = st.selectbox("2. Nhóm giống lúa (u_variety)", AGRO_CONTEXT_SCHEMA["u_variety"], index=0)
            q_pest = st.selectbox("3. Áp lực dịch hại cục bộ (u_pest)", AGRO_CONTEXT_SCHEMA["u_pest"], index=2)
            q_ndvi = st.selectbox("4. Chỉ số viễn thám Sentinel-2 (u_ndvi)", AGRO_CONTEXT_SCHEMA["u_ndvi"], index=2)

        with col_form3:
            st.markdown("**👨‍🌾 Hồ sơ Nông hộ nhận tin Zalo OA / SMS:**")
            f_name = st.text_input("Họ tên nông hộ", "Bác Ba Kiệt")
            f_loc = st.text_input("Địa chỉ canh tác", "Xã Mỹ Đông, Tháp Mười, Đồng Tháp (ĐBSCL)")
            f_area = st.number_input("Diện tích canh tác (ha | 1 ha = 10 công)", 0.1, 50.0, 1.5, 0.1)
            f_crop = st.text_input("Giống lúa & Vụ mùa", "Lúa OM18 (Vụ Hè Thu)")
            f_days = st.number_input("Số ngày sau sạ (NSS)", 1, 120, 62, 1)

        query_ecwc = rag_engine.create_query_ecwc(
            psi_values={"Tmax": q_tmax, "Tmin": q_tmin, "RHmax": q_rhmax, "RHmin": q_rhmin, "RF": q_rf},
            temp_deviations={"Tmax": q_tmax - 32.3, "Tmin": q_tmin - 21.1},
            c_agro={"u_soil": q_soil, "u_variety": q_var, "u_pest": q_pest, "u_ndvi": q_ndvi},
            season=q_season,
            phenophase=q_pheno,
        )
        farmer = FarmerProfile(
            farmer_name=f_name,
            location=f_loc,
            area_ha=float(f_area),
            crop_name=f_crop,
            sowing_days_ago=int(f_days),
        )

        best_match, all_matches = rag_engine.retrieve_best_case(query_ecwc)
        rag_out = rag_engine.generate_personalized_advisory(
            query_ecwc=query_ecwc,
            retrieved_match=best_match,
            farmer=farmer,
            api_key=api_key_input if api_key_input else None,
        )

        st.divider()
        col_res_left, col_res_right = st.columns([1.0, 1.15])

        with col_res_left:
            st.markdown("#### 🔍 Kết quả Truy hồi Tương đồng Mờ từ Kho tri thức Pset")
            rank_df = pd.DataFrame(
                [
                    {
                        "Mã Ca Lịch Sử": m["case_id"],
                        "Mùa / Giai đoạn": f"{m['season']} - {m['phenophase']}",
                        "Crisp Match": "✔ Khớp" if m["crisp_exact_match"] else "✖ Lệch biên",
                        "Sim_Weather (%)": round(m["weather_fuzzy_sim"] * 100, 1),
                        "Sim_Agro (%)": round(m["agro_context_sim"] * 100, 1),
                        "Sim_Total (%)": round(m["composite_sim"] * 100, 1),
                    }
                    for m in all_matches
                ]
            )
            st.dataframe(rank_df, use_container_width=True)
            with st.expander("Xem cấu trúc bộ 9 thành phần ECWC_i(n, m, d) & Grounded Prompt"):
                st.json(query_ecwc.to_9tuple_summary())
                st.text_area("Grounded RAG Prompt gửi tới LLM:", rag_out["grounded_prompt"], height=220)

        with col_res_right:
            st.markdown(f"#### 💬 Mô phỏng Giao diện Tin nhắn Zalo OA (`{rag_out['mode_used']}`)")
            st.markdown(
                f"""
                <div style="background:#e0f2fe; border:2px solid #0284c7; border-radius:12px; padding:14px;">
                    <div style="background:#0068ff; color:white; padding:8px 12px; border-radius:8px; font-weight:bold; margin-bottom:10px;">
                        Zalo Official Account: Trợ Lý Khuyến Nông ĐBSCL ✔
                    </div>
                    <div style="background:white; padding:14px; border-radius:10px; color:#0f172a; line-height:1.55;">
                        {rag_out['zalo_message'].replace(chr(10), '<br/>')}
                    </div>
                    <div style="background:#fef9c3; border:1px solid #eab308; padding:10px; border-radius:8px; margin-top:10px; font-family:monospace; font-size:0.88rem;">
                        <b>📱 Tin nhắn SMS 160 ký tự:</b><br/>{rag_out['sms_message']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()

