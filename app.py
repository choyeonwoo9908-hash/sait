"""차세대 반도체 물질 발굴 플랫폼 (Materials Project 기반).

스크리닝 + 물리 시뮬레이션 + 조합 탐색을 하나의 연구용 대시보드로 통합한다.
"""
import itertools
import os
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import physics as phys
import data as datamod
import chatbot

PALETTE = ["#1f4e79", "#2e75b6", "#9dc3e6", "#c55a11", "#548235",
           "#7b5ea7", "#7f8c98", "#264653", "#9e480e", "#385723"]

st.set_page_config(page_title="메모리 반도체 소재 발굴 플랫폼", page_icon="◈", layout="wide")

# ── 전문 스타일 ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  html, body, [class*="css"] {
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
    font-feature-settings:"tnum"; }
  h1,h2,h3 { color:#14202b; font-weight:650; letter-spacing:-.2px; }
  .block-container { padding-top:4.2rem; }
  .app-header { border-bottom:2px solid #1f4e79; padding-bottom:.55rem; margin-bottom:1.1rem; }
  .app-title { font-size:1.55rem; font-weight:700; color:#14202b; letter-spacing:-.3px;
               margin:0; line-height:1.2; }
  .app-sub { color:#5b6b76; font-size:.84rem; margin-top:.3rem; }
  div[data-testid="stMetric"] { background:#ffffff; border:1px solid #d8dee4;
        border-radius:4px; padding:12px 16px; }
  div[data-testid="stMetric"] label p { color:#5b6b76; font-size:.7rem !important;
        text-transform:uppercase; letter-spacing:.5px; font-weight:600; }
  div[data-testid="stMetricValue"] { font-family:ui-monospace,"SF Mono",Menlo,monospace;
        font-size:1.45rem; color:#1f4e79; font-weight:600; }
  .stTabs [data-baseweb="tab-list"] { gap:.25rem; border-bottom:1px solid #d8dee4; }
  .stTabs [data-baseweb="tab"] { font-weight:600; font-size:.88rem; color:#5b6b76;
        padding:.4rem .85rem; }
  section[data-testid="stSidebar"] { background:#f3f5f8; border-right:1px solid #d8dee4; }
  section[data-testid="stSidebar"] h3 { font-size:.72rem; text-transform:uppercase;
        letter-spacing:.6px; color:#5b6b76; font-weight:700; margin-bottom:.2rem; }
  .badge { display:inline-block; background:#eaf0f6; color:#1f4e79; border:1px solid #cfe0ee;
        padding:2px 10px; border-radius:3px; font-size:.74rem; font-weight:600; margin-right:6px; }
  div[data-testid="stButton"] > button { border-radius:4px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-header">'
            '<div class="app-title">메모리 반도체 소재 발굴 플랫폼</div>'
            '<div class="app-sub">Materials Project DFT 데이터 · SRAM/DRAM 고유전율(high-k) '
            'κ·Eg · 강유전체(FeRAM/FeFET) · NAND 터널/블로킹 산화물 · 저항변화(RRAM) · '
            '조합 화학공간 탐색</div>'
            '</div>', unsafe_allow_html=True)


# ── 접근 비밀번호 게이트 ─────────────────────────────────────────────────────
# APP_PASSWORD가 secrets/env에 설정돼 있으면 일치해야 입장(공개 배포 시 API 키
# 무단 사용 방지). 미설정이면 공개 모드로 통과시키되 안내를 표시한다.
def _check_password() -> bool:
    try:
        configured = st.secrets["APP_PASSWORD"]
    except Exception:
        configured = os.environ.get("APP_PASSWORD")
    if not configured:
        return True                                  # 비밀번호 미설정 → 공개 모드
    if st.session_state.get("_authed"):
        return True
    with st.form("login_form"):
        st.markdown("#### 🔒 접근 비밀번호를 입력하세요")
        pw = st.text_input("비밀번호", type="password", label_visibility="collapsed")
        ok = st.form_submit_button("입장", type="primary")
    if ok:
        if pw == configured:
            st.session_state["_authed"] = True
            st.rerun()
        st.error("비밀번호가 올바르지 않습니다.")
    return False


if not _check_password():
    st.stop()


# ── 응용 프리셋 ─────────────────────────────────────────────────────────────
PRESETS = {
    "일반 탐색":              dict(bg=(0.3, 10.0), gap_type="전체", metal=True, hull=0.10, app="general"),
    "고유전율 게이트 (high-k)":  dict(bg=(4.0, 9.0),  gap_type="전체", metal=True, hull=0.10, app="highk"),
    "DRAM 커패시터 유전체":     dict(bg=(2.5, 9.0),  gap_type="전체", metal=True, hull=0.10, app="dram_cap"),
    "강유전체 (FeRAM/FeFET)":  dict(bg=(2.0, 7.0),  gap_type="전체", metal=True, hull=0.10, app="ferroelectric", include_any=["Hf", "Zr"]),
    "NAND 터널/블로킹 산화물":  dict(bg=(5.0, 10.0), gap_type="전체", metal=True, hull=0.10, app="nand_oxide"),
    "저항변화 메모리 (RRAM)":   dict(bg=(2.0, 6.0),  gap_type="전체", metal=True, hull=0.15, app="rram"),
}

st.sidebar.markdown("### 응용 프리셋")
preset = st.sidebar.selectbox("목적을 선택하면 조건이 자동 설정됩니다", list(PRESETS), key="preset")
if st.session_state.get("_last_preset") != preset:
    c = PRESETS[preset]
    st.session_state.bg_range = c["bg"]
    st.session_state.gap_type = c["gap_type"]
    st.session_state.exclude_metal = c["metal"]
    st.session_state.hull = c["hull"]
    st.session_state.app_key = c["app"]
    st.session_state.app_include_any = c.get("include_any")
    st.session_state._last_preset = preset

st.sidebar.divider()
st.sidebar.markdown("### 스크리닝 조건")

bg_min, bg_max = st.sidebar.slider("밴드갭 (eV)", 0.0, 10.0, step=0.1, key="bg_range")
gap_type = st.sidebar.radio("밴드갭 종류", ["전체", "직접갭만", "간접갭만"],
                            horizontal=True, key="gap_type")
exclude_metal = st.sidebar.checkbox("금속 제외 (반도체/절연체만)", key="exclude_metal")

st.sidebar.markdown("**안정성**")
hull = st.sidebar.slider("E above hull 최대 (eV/atom)", 0.0, 0.5, step=0.01, key="hull")
stable_only = st.sidebar.checkbox("열역학적 안정 물질만 (hull≈0)", value=False)
exp_only = st.sidebar.checkbox("실험적으로 알려진 물질만 (이론물질 제외)", value=False)

st.sidebar.markdown("**조성**")
include_elements = st.sidebar.text_input("포함 원소 (쉼표)", placeholder="예: Ga, N")
exclude_elements = st.sidebar.text_input("제외 원소 (쉼표)", placeholder="예: Pb, Hg, Cd")
nelements = st.sidebar.multiselect("원소 개수", [1, 2, 3, 4, 5],
                                   default=[], help="비우면 전체. 2=이원계, 3=삼원계 …")
nsites_max = st.sidebar.slider("최대 원자 수 (단위셀)", 1, 60, 30)

st.sidebar.markdown("**합성**")
ald_only = st.sidebar.checkbox(
    "ALD 합성 가능 물질만", value=False,
    help="ALD(원자층증착) 전구체가 확립된 양이온의 단순 산화물·질화물(이원~삼원계)만 표시. "
         "MP에 합성법 정보가 없어 적용하는 휴리스틱 필터입니다.")

with st.sidebar.expander("고급 물성 필터"):
    bulk_min = st.slider("부피탄성률 최소 (GPa)", 0, 400, 0)
    eps_min = st.slider("정적 유전율 최소 (ε)", 0.0, 50.0, 0.0, step=0.5)
    nonmag_only = st.checkbox("비자성 물질만", value=False)

max_results = st.sidebar.number_input("최대 결과 수", 20, 2000, 300, step=20)
run = st.sidebar.button("스크리닝 실행", type="primary", use_container_width=True)


# ── 시각화 레이아웃 ─────────────────────────────────────────────────────────
def fig_layout(fig, h=420):
    fig.update_layout(template="plotly_white", height=h,
                      margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(family="ui-monospace, monospace", size=12),
                      colorway=PALETTE)
    return fig


def kappa_eg_figure(dframe, title="κ–Eg 트레이드오프 지도", h=460):
    """high-k 핵심 차트: 유전율 κ vs 밴드갭 Eg.

    잘 알려진 게이트 유전체 기준점, 등(等)-FOM(κ·Eg) 보조선과 함께 후보를 표시.
    우상단(κ 큼 + Eg 큼)일수록 우수하지만 경험적 트레이드오프로 보통 반비례한다.
    """
    full = dframe[dframe.kappa.notna() & dframe.band_gap.notna()].copy()
    # 극단적 κ(격자 불안정 발산 아티팩트)는 산점도·축 스케일에서 제외
    sub = full[full.kappa <= phys.KAPPA_MAX_RELIABLE].copy()
    n_extreme = len(full) - len(sub)
    fig = go.Figure()

    ref_kappa_max = max(r[1] for r in phys.REFERENCE_DIELECTRICS)   # TiO2=80
    kdata = float(sub.kappa.max()) if not sub.empty else 90.0
    # 신뢰 κ 범위 안에서만 축을 잡아 한두 개 큰 값이 화면을 짓누르지 않게 함
    kmax = min(phys.KAPPA_MAX_RELIABLE, max(90.0, ref_kappa_max + 10.0, kdata * 1.05))
    eg_data = float(sub.band_gap.max()) if not sub.empty else 9.0
    ref_eg_max = max(r[2] for r in phys.REFERENCE_DIELECTRICS)
    ymax = max(10.5, eg_data * 1.05, ref_eg_max + 0.5)   # 밴드갭 스케일로 고정
    for prod, color in [(phys.EPS_SIO2 * phys.EG_SIO2, "#cfd8e0"), (150.0, "#b8c4cf"),
                        (300.0, "#a3b2bf")]:
        # 등-FOM 보조선을 보이는 y 범위로 클립 (y=prod/κ 가 ymax를 넘지 않게)
        kk = np.linspace(max(1.0, prod / ymax), kmax, 120)
        fig.add_trace(go.Scatter(
            x=kk, y=prod / kk, mode="lines",
            name=f"κ·Eg={prod:.0f}", line=dict(dash="dot", width=1, color=color),
            hoverinfo="skip", showlegend=True))

    rx = [r[1] for r in phys.REFERENCE_DIELECTRICS]
    ry = [r[2] for r in phys.REFERENCE_DIELECTRICS]
    rn = [r[0] for r in phys.REFERENCE_DIELECTRICS]
    fig.add_trace(go.Scatter(
        x=rx, y=ry, mode="markers+text", text=rn, name="기준 유전체",
        textposition="top center", textfont=dict(size=10, color="#9e480e"),
        marker=dict(symbol="diamond", size=11, color="#c55a11",
                    line=dict(width=0.5, color="#7a3608")),
        hovertemplate="%{text}<br>κ=%{x}<br>Eg=%{y} eV<extra></extra>"))

    if not sub.empty:
        fig.add_trace(go.Scatter(
            x=sub.kappa, y=sub.band_gap, mode="markers", name="후보 물질",
            marker=dict(size=8, color=sub.score, colorscale="Viridis", showscale=True,
                        colorbar=dict(title=dict(text="발굴점수", side="right"),
                                      thickness=12, len=0.92, x=1.0, xanchor="left"),
                        line=dict(width=0.4, color="#333")),
            text=sub.formula, customdata=sub.highk_fom,
            hovertemplate=("%{text}<br>κ=%{x:.1f}<br>Eg=%{y:.2f} eV"
                           "<br>κ·Eg=%{customdata:.1f}×SiO₂<extra></extra>")))

    if n_extreme:
        fig.add_annotation(
            text=f"κ&gt;{int(phys.KAPPA_MAX_RELIABLE)} {n_extreme}건 제외(DFPT 발산 가능)",
            xref="paper", yref="paper", x=0.99, y=0.02, xanchor="right",
            showarrow=False, font=dict(size=10, color="#999"))

    # 범례를 플롯 내부 우상단(트레이드오프상 비어있는 영역)으로 옮겨
    # 우측 색막대(colorbar)와 겹치지 않게 한다. 반투명 배경으로 점 가림 최소화.
    fig.update_layout(
        xaxis_title="정적 유전율 κ", yaxis_title="밴드갭 Eg (eV)", title=title,
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top",
                    bgcolor="rgba(255,255,255,0.72)", bordercolor="#d8dee4",
                    borderwidth=1, font=dict(size=10)))
    fig.update_xaxes(range=[0, kmax])
    fig.update_yaxes(range=[0, ymax])
    fig = fig_layout(fig, h)
    fig.update_layout(margin=dict(l=10, r=20, t=50, b=10))  # 우측 색막대 여백 확보
    return fig


# ── 실행 ────────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Materials Project 조회 및 물성 시뮬레이션 중…"):
        try:
            df_new = datamod.run_screening(
                bg_min=bg_min, bg_max=bg_max, hull=hull, nsites_max=nsites_max,
                max_results=int(max_results),
                include=[e.strip() for e in include_elements.split(",") if e.strip()] or None,
                exclude=[e.strip() for e in exclude_elements.split(",") if e.strip()] or None,
                nelements=nelements,
                app_key=st.session_state.get("app_key", "general"),
                include_any=st.session_state.get("app_include_any"),
                gap_type=gap_type, exclude_metal=exclude_metal, stable_only=stable_only,
                exp_only=exp_only, nonmag_only=nonmag_only, bulk_min=bulk_min, eps_min=eps_min,
                ald_only=ald_only)
        except Exception as e:
            st.error(f"API 오류: {e}")
            st.stop()
    st.session_state.df = df_new
    st.session_state.app_label = preset

# ── AI 어시스턴트 (팝업 모달 창) ─────────────────────────────────────────────
# 대시보드는 전체 폭으로 두고, 챗봇은 버튼으로 여는 분리된 모달 창으로 띄운다.
# 모달 안 스크리닝으로 st.session_state.df가 갱신되고, 창을 닫으면(전체 rerun)
# 아래 대시보드가 최신 결과로 다시 그려진다.
if st.button("AI 어시스턴트 열기", type="secondary",
             help="메모리 소재 발굴 대화형 어시스턴트입니다. 자연어 요청을 스크리닝 "
                  "조건으로 해석해 후보를 대시보드에 반영하고, 개념·물리 질문에는 "
                  "근거와 함께 답변합니다."):
    chatbot.open_chat_dialog()

tabs = st.tabs(["개요", "고유전율 (High-k)", "강유전체 (FeRAM/FeFET)",
                "NAND 산화물", "저항변화 (RRAM)",
                "화학공간 탐색", "추천 후보", "데이터"])

df = st.session_state.get("df")

if df is None or df.empty:
    for t in tabs:
        with t:
            st.info("사이드바에서 **스크리닝 실행**을 누르거나, 위의 "
                    "**AI 어시스턴트 열기** 버튼으로 자연어로 질문해 "
                    "메모리 소재 후보를 찾아보세요.")
    st.stop()

# 1) 개요
with tabs[0]:
    has_k = df[df.kappa.notna()]
    n_highk = int((df.kappa.fillna(0) >= 10).sum())
    st.markdown(f'<span class="badge">{st.session_state.get("app_label","")}</span> '
                f'<b>{len(df)}</b>개 후보', unsafe_allow_html=True)
    m = st.columns(6)
    m[0].metric("후보 물질", f"{len(df)}")
    m[1].metric("κ 데이터 보유", f"{len(has_k)}")
    m[2].metric("고유전율 κ≥10", f"{n_highk}")
    m[3].metric("ALD 가능", f"{int(df.is_ald.sum())}")
    m[4].metric("안정 물질", f"{int(df.is_stable.sum())}")
    m[5].metric("최고 발굴점수", f"{df.score.max():.0f}")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(kappa_eg_figure(df, "κ–Eg 트레이드오프 지도 (메모리 유전체 핵심)"),
                        use_container_width=True)
    with c2:
        fig = px.scatter(df, x="band_gap", y="e_above_hull", color="crystal_system",
                         size="nsites", hover_data=["formula", "material_id"],
                         labels={"band_gap": "밴드갭 (eV)", "e_above_hull": "E above hull (eV/atom)"},
                         title="안정성 vs 밴드갭")
        fig.update_traces(marker=dict(opacity=0.7))
        st.plotly_chart(fig_layout(fig), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        vc = df.crystal_system.value_counts().reset_index()
        vc.columns = ["결정계", "수"]
        st.plotly_chart(fig_layout(px.pie(vc, names="결정계", values="수",
                        title="결정계 분포", hole=0.4)), use_container_width=True)
    with c4:
        fig = px.histogram(df, x="band_gap", nbins=40, opacity=0.8,
                           labels={"band_gap": "밴드갭 (eV)"}, title="밴드갭 분포")
        st.plotly_chart(fig_layout(fig), use_container_width=True)

# 2) 고유전율 (High-k)
with tabs[1]:
    st.markdown("**고유전율(high-k) 게이트·커패시터 유전체** — 유전율 κ가 클수록 "
                "커패시턴스(전하저장)에 유리하지만 보통 밴드갭이 작아져 누설이 늘어납니다"
                "(κ–Eg 트레이드오프). 품질지수 **κ·Eg**(×SiO₂)와 5 nm 물리두께 기준 "
                "**EOT**로 SRAM/DRAM 적합성을 평가합니다.")
    hk_all = df[df.highk_fom.notna()].copy()
    hk = hk_all[hk_all.kappa_reliable].copy()   # 발산 아티팩트(κ>200) 배제
    n_drop = len(hk_all) - len(hk)
    if hk.empty:
        st.info("유전율(κ) 데이터가 있는 후보가 없습니다. 조건(밴드갭·원소·안정성)을 넓혀보세요.")
    else:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(kappa_eg_figure(hk_all, "κ–Eg 지도 · 색=발굴점수"),
                            use_container_width=True)
        with c2:
            st.markdown("**high-k 유망 Top 12** (발굴점수 순)")
            tp = hk.sort_values("score", ascending=False).head(12)
            st.dataframe(tp[["formula", "kappa", "band_gap", "highk_fom", "eot_5nm",
                             "is_ald", "score"]],
                         use_container_width=True, hide_index=True,
                         column_config={
                             "formula": st.column_config.TextColumn("물질"),
                             "kappa": st.column_config.NumberColumn("κ", format="%.1f"),
                             "band_gap": st.column_config.NumberColumn("Eg(eV)", format="%.2f"),
                             "highk_fom": st.column_config.NumberColumn("κ·Eg(×SiO₂)", format="%.1f"),
                             "eot_5nm": st.column_config.NumberColumn("EOT@5nm(nm)", format="%.2f"),
                             "is_ald": st.column_config.CheckboxColumn("ALD"),
                             "score": st.column_config.NumberColumn("발굴점수", format="%.0f")})
        cap = ("발굴점수는 높은 κ·Eg와 함께 누설 억제에 필요한 충분한 밴드갭(밴드오프셋)을 "
               "동시에 반영합니다. κ가 커도 Eg가 작으면 누설로 감점됩니다. "
               "EOT가 작을수록 같은 물리두께로 더 큰 커패시턴스(미세화 유리).")
        if n_drop:
            cap += f" ※ κ>{int(phys.KAPPA_MAX_RELIABLE)}인 {n_drop}건은 DFPT 발산 아티팩트로 보아 제외했습니다."
        st.caption(cap)

# 3) 강유전체 (FeRAM/FeFET)
with tabs[2]:
    st.markdown("**강유전체 메모리 후보 (FeRAM·FeFET)** — HfO₂/ZrO₂ 계열 산화물은 "
                "비휘발성 강유전 상(orthorhombic Pca2₁)을 형성해 차세대 메모리 모재로 주목받습니다. "
                "여기서는 Hf/Zr 함유 산화물을 추려 결정계·유전율·안정성으로 평가합니다.")
    fe = df[df.has_hf_zr & df.is_oxide].copy()
    if fe.empty:
        st.info("Hf/Zr 함유 산화물 후보가 없습니다. 사이드바 '포함 원소'에 Hf 또는 Zr를 넣거나 "
                "**강유전체 (FeRAM/FeFET)** 프리셋으로 다시 스크리닝해 보세요.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            vc = fe.crystal_system.value_counts().reset_index()
            vc.columns = ["결정계", "수"]
            fig = px.bar(vc, x="결정계", y="수",
                         title="Hf/Zr 산화물 결정계 (orthorhombic=강유전 상 관련)")
            st.plotly_chart(fig_layout(fig), use_container_width=True)
        with c2:
            fig = px.scatter(fe, x="kappa", y="e_above_hull", color="score",
                             size="nsites", hover_data=["formula", "crystal_system"],
                             color_continuous_scale="Viridis",
                             labels={"kappa": "유전율 κ", "e_above_hull": "hull",
                                     "score": "발굴점수"},
                             title="유전율 vs 안정성 (Hf/Zr 산화물)")
            st.plotly_chart(fig_layout(fig), use_container_width=True)
        st.markdown("**강유전 메모리 유망 Top 12** (발굴점수 순)")
        tp = fe.sort_values("score", ascending=False).head(12)
        st.dataframe(tp[["formula", "crystal_system", "band_gap", "kappa",
                         "e_above_hull", "score"]],
                     use_container_width=True, hide_index=True,
                     column_config={
                         "formula": st.column_config.TextColumn("물질"),
                         "crystal_system": st.column_config.TextColumn("결정계"),
                         "band_gap": st.column_config.NumberColumn("Eg(eV)", format="%.2f"),
                         "kappa": st.column_config.NumberColumn("κ", format="%.1f"),
                         "e_above_hull": st.column_config.NumberColumn("hull", format="%.3f"),
                         "score": st.column_config.NumberColumn("점수", format="%.0f")})
        st.caption("참고: 강유전성은 특정 준안정 orthorhombic 상에서 발현되므로, MP의 평형 "
                   "상(결정계)만으로 직접 판정되지 않습니다. 본 탭은 화학계 기반 1차 후보군입니다.")

# 4) NAND 산화물 (터널/블로킹)
with tabs[3]:
    st.markdown("**NAND·Flash 터널/블로킹 산화물** — 전하저장 구조에서 넓은 밴드갭은 높은 "
                "에너지 배리어(누설·전하손실 억제)를 뜻합니다. 산화물 후보를 밴드갭 중심으로 "
                "평가하며, 블로킹 산화물은 적당한 κ도 함께 고려합니다.")
    nd = df[df.is_oxide].copy()
    if nd.empty:
        st.info("산화물 후보가 없습니다. 조건을 넓혀보세요.")
    else:
        fig = px.scatter(nd, x="band_gap", y="kappa", color="score", size="nsites",
                         hover_data=["formula", "material_id"],
                         color_continuous_scale="Viridis",
                         labels={"band_gap": "밴드갭 Eg (eV)", "kappa": "유전율 κ",
                                 "score": "발굴점수"},
                         title="산화물 후보: 밴드갭(배리어) vs 유전율")
        fig.add_vrect(x0=5.0, x1=nd.band_gap.max() + 0.5, fillcolor="#9dc3e6", opacity=0.15,
                      line_width=0, annotation_text="넓은 배리어(Eg≥5)",
                      annotation_position="top left")
        st.plotly_chart(fig_layout(fig, 460), use_container_width=True)
        st.markdown("**넓은 밴드갭 산화물 Top 12** (Eg 순)")
        tp = nd.sort_values("band_gap", ascending=False).head(12)
        st.dataframe(tp[["formula", "band_gap", "kappa", "eot_5nm", "e_above_hull", "score"]],
                     use_container_width=True, hide_index=True,
                     column_config={
                         "formula": st.column_config.TextColumn("물질"),
                         "band_gap": st.column_config.NumberColumn("Eg(eV)", format="%.2f"),
                         "kappa": st.column_config.NumberColumn("κ", format="%.1f"),
                         "eot_5nm": st.column_config.NumberColumn("EOT@5nm(nm)", format="%.2f"),
                         "e_above_hull": st.column_config.NumberColumn("hull", format="%.3f"),
                         "score": st.column_config.NumberColumn("점수", format="%.0f")})

# 5) 저항변화 (RRAM)
with tabs[4]:
    st.markdown("**저항변화 메모리(RRAM) 후보** — 전이금속 산화물(HfO₂·TiO₂·Ta₂O₅ 등)에서 "
                "산소공공 필라멘트 생성/소멸로 저항이 스위칭됩니다. 전이금속 산화물 중 "
                "스위칭에 적합한 밴드갭(≈2~6 eV) 영역을 강조합니다.")
    rr = df[df.is_tm_oxide].copy()
    if rr.empty:
        st.info("전이금속 산화물 후보가 없습니다. **저항변화 메모리 (RRAM)** 프리셋으로 다시 "
                "스크리닝하거나 조건을 넓혀보세요.")
    else:
        fig = px.scatter(rr, x="band_gap", y="kappa", color="score", size="nsites",
                         hover_data=["formula", "material_id", "chemsys"],
                         color_continuous_scale="Viridis",
                         labels={"band_gap": "밴드갭 Eg (eV)", "kappa": "유전율 κ",
                                 "score": "발굴점수"},
                         title="전이금속 산화물: 밴드갭 vs 유전율")
        fig.add_vrect(x0=2.0, x1=6.0, fillcolor="#548235", opacity=0.12, line_width=0,
                      annotation_text="스위칭 적합대 (2~6 eV)", annotation_position="top left")
        st.plotly_chart(fig_layout(fig, 460), use_container_width=True)
        st.markdown("**RRAM 유망 Top 12** (발굴점수 순)")
        tp = rr.sort_values("score", ascending=False).head(12)
        st.dataframe(tp[["formula", "chemsys", "band_gap", "kappa", "e_above_hull", "score"]],
                     use_container_width=True, hide_index=True,
                     column_config={
                         "formula": st.column_config.TextColumn("물질"),
                         "chemsys": st.column_config.TextColumn("화학계"),
                         "band_gap": st.column_config.NumberColumn("Eg(eV)", format="%.2f"),
                         "kappa": st.column_config.NumberColumn("κ", format="%.1f"),
                         "e_above_hull": st.column_config.NumberColumn("hull", format="%.3f"),
                         "score": st.column_config.NumberColumn("점수", format="%.0f")})

# 6) 화학공간 탐색 (조합)
with tabs[5]:
    st.markdown("**조합 화학공간 탐색** — 후보군에서 어떤 원소·원소쌍이 유망한 메모리 소재를 형성하는지 분석합니다.")
    all_elems = list(itertools.chain.from_iterable(df.elements))
    freq = Counter(all_elems)
    c1, c2 = st.columns(2)
    with c1:
        fe = pd.DataFrame(freq.most_common(20), columns=["원소", "출현수"])
        st.plotly_chart(fig_layout(px.bar(fe, x="원소", y="출현수",
                        title="원소 출현 빈도 Top 20")), use_container_width=True)
    with c2:
        fig = px.scatter(df, x="band_gap", y="e_above_hull", color=df.nelements.astype(str),
                         size="nsites", hover_data=["formula", "chemsys"],
                         labels={"band_gap": "밴드갭 (eV)", "e_above_hull": "hull",
                                 "color": "원소수"},
                         title="원소수별 안정성-밴드갭 분포")
        fig.update_traces(marker=dict(opacity=0.7))
        st.plotly_chart(fig_layout(fig), use_container_width=True)

    st.markdown("**원소 동시출현 히트맵** — 함께 등장해 화합물을 이루는 원소쌍 (상위 15개 원소)")
    top_el = [e for e, _ in freq.most_common(15)]
    mat = pd.DataFrame(0, index=top_el, columns=top_el)
    for els in df.elements:
        present = [e for e in set(els) if e in top_el]
        for a, b in itertools.combinations(present, 2):
            mat.loc[a, b] += 1
            mat.loc[b, a] += 1
    fig = px.imshow(mat, color_continuous_scale="Teal", aspect="auto",
                    labels=dict(color="동시출현"), title="원소쌍 동시출현 횟수")
    st.plotly_chart(fig_layout(fig, 500), use_container_width=True)

    st.markdown("**유망 화학계(chemsys) Top 15** — 평균 발굴점수 기준")
    cs = (df.groupby("chemsys")
            .agg(물질수=("formula", "count"), 평균점수=("score", "mean"),
                 평균밴드갭=("band_gap", "mean"))
            .sort_values(["평균점수", "물질수"], ascending=False).head(15).reset_index())
    st.dataframe(cs, use_container_width=True, hide_index=True,
                 column_config={"평균점수": st.column_config.NumberColumn(format="%.1f"),
                                "평균밴드갭": st.column_config.NumberColumn(format="%.2f")})

# 7) 추천 후보
with tabs[6]:
    st.markdown(f"**종합 발굴점수 Top 20** — 안정성 + `{st.session_state.get('app_label','')}` "
                "적합도(유전율·밴드갭·화학계)로 산출한 휴리스틱 순위입니다.")
    best = df.sort_values("score", ascending=False).head(20).reset_index(drop=True)
    fig = px.bar(best, x="score", y="formula", orientation="h", color="band_gap",
                 color_continuous_scale="Viridis", custom_data=["material_id"],
                 labels={"score": "발굴점수", "formula": "", "band_gap": "Eg"},
                 title="추천 메모리 반도체 소재 후보 (막대 클릭 → MP 페이지)")
    fig.update_traces(hovertemplate="%{y}<br>발굴점수=%{x:.0f}"
                                    "<br><b>막대를 클릭하면 MP 링크</b><extra></extra>")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    event = st.plotly_chart(fig_layout(fig, 600), use_container_width=True,
                            on_select="rerun", selection_mode="points", key="rec_chart")

    # 클릭된 막대(물질)의 material_id로 Materials Project 링크 버튼을 띄운다.
    try:
        pts = event["selection"]["points"]
    except (TypeError, KeyError):
        pts = []
    picked = None
    if pts:
        cd = pts[0].get("customdata")
        if cd:
            picked = (pts[0].get("y"), cd[0])               # (화학식, material_id)
        else:
            idx = pts[0].get("point_index", pts[0].get("point_number"))
            if idx is not None and idx < len(best):
                picked = (best.formula[idx], best.material_id[idx])
    if picked:
        formula, mid = picked
        st.link_button(f"🔗 {formula} — Materials Project에서 열기",
                       f"https://materialsproject.org/materials/{mid}")
    else:
        st.caption("그래프의 막대를 클릭하면 해당 물질의 Materials Project 페이지로 가는 "
                   "링크 버튼이 여기에 나타납니다.")

# 8) 데이터
with tabs[7]:
    q = st.text_input("화학식/원소 검색", placeholder="예: HfO2, Hf, ZrO2")
    show = df[df.formula.str.contains(q, case=False, na=False)] if q else df
    st.dataframe(show.sort_values("score", ascending=False),
                 use_container_width=True, height=460)
    c1, c2 = st.columns(2)
    c1.download_button("CSV 다운로드", df.to_csv(index=False).encode("utf-8-sig"),
                       "memory_material_candidates.csv", "text/csv", use_container_width=True)
    c2.download_button("JSON 다운로드", df.to_json(orient="records", force_ascii=False).encode("utf-8"),
                       "memory_material_candidates.json", "application/json", use_container_width=True)
