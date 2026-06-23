"""실제 스크리닝 데이터를 캡처해 완전 오프라인 단일 파일 demo.py를 생성한다.

demo.py는 streamlit·plotly·pandas·numpy 만으로 동작(MP·Anthropic·네트워크 불필요).
실행: ./.venv/bin/python build_demo.py  →  demo.py 생성
"""
import json
import gzip
import base64

import data as d

COLS = ["material_id", "formula", "band_gap", "band_gap_corr", "kappa", "kappa_reliable",
        "highk_fom", "eot_5nm", "is_oxide", "has_hf_zr", "is_tm_oxide", "is_ald",
        "e_above_hull", "is_stable", "crystal_system", "spacegroup", "point_group",
        "is_polar", "score", "score_stability", "score_manu", "score_durability",
        "score_performance", "nsites", "nelements", "elements", "chemsys", "gap_kind",
        "is_metal"]


def cap(**kw):
    df = d.run_screening(**kw)
    df = df.sort_values("score", ascending=False).head(120)
    cols = [c for c in COLS if c in df.columns]
    return json.loads(df[cols].to_json(orient="records"))   # NaN→null 안전


print("스크리닝 캡처 중…")
datasets = {
    "고유전율 게이트 (high-k)": cap(bg_min=4.0, bg_max=9.0, hull=0.1, nsites_max=30,
        max_results=300, include=None, exclude=None, nelements=[], app_key="highk",
        exclude_metal=True),
    "강유전체 (FeRAM/FeFET)": cap(bg_min=2.0, bg_max=7.0, hull=0.1, nsites_max=30,
        max_results=300, include=None, exclude=None, nelements=[], app_key="ferroelectric",
        exclude_metal=True, include_any=["Hf", "Zr"]),
    "저항변화 메모리 (RRAM)": cap(bg_min=2.0, bg_max=6.0, hull=0.15, nsites_max=30,
        max_results=300, include=["O"], exclude=["Pb", "Hg", "Cd"], nelements=[],
        app_key="rram", exclude_metal=True),
    "일반 탐색": cap(bg_min=0.3, bg_max=10.0, hull=0.1, nsites_max=30, max_results=300,
        include=None, exclude=None, nelements=[], app_key="general", exclude_metal=True),
}
print("preset별 행수:", {k: len(v) for k, v in datasets.items()})
blob = base64.b64encode(gzip.compress(
    json.dumps(datasets, ensure_ascii=False).encode())).decode()
print("데이터 blob 길이:", len(blob))

TEMPLATE = '''"""메모리 반도체 소재 발굴 — 단일 파일 오프라인 데모.

네트워크/API 불필요. 미리 스크리닝한 데이터가 파일에 내장돼 있고, AI 답변은
사전 작성(canned)이라 어떤 환경에서도 즉시·안정적으로 동작한다.
필요 패키지: streamlit, plotly, pandas, numpy
실행: streamlit run demo.py
"""
import json, gzip, base64, itertools
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PALETTE = ["#1f4e79", "#2e75b6", "#9dc3e6", "#c55a11", "#548235",
           "#7b5ea7", "#7f8c98", "#264653", "#9e480e", "#385723"]

# ── 물리 상수/함수 (자체 포함) ──────────────────────────────────────────────
EPS_SIO2, EG_SIO2, EG_SI = 3.9, 9.0, 1.12
KAPPA_MAX_RELIABLE = 200.0
PBE_GAP_SCISSOR = 1.5
REFERENCE_DIELECTRICS = [
    ("SiO2", 3.9, 9.0), ("Si3N4", 7.0, 5.3), ("Al2O3", 9.0, 8.8), ("Y2O3", 15.0, 6.0),
    ("HfO2", 25.0, 5.8), ("ZrO2", 25.0, 5.8), ("Ta2O5", 25.0, 4.4), ("La2O3", 30.0, 6.0),
    ("TiO2", 80.0, 3.2)]
POLAR_POINT_GROUPS = {"1", "2", "m", "mm2", "4", "4mm", "3", "3m", "6", "6mm"}
ALD_PRECURSORS = {
    "Hf": ("TEMAHf / TDMAHf", 200, 300, 1.0), "Zr": ("TEMAZr / TDMAZr", 200, 300, 1.0),
    "Al": ("TMA", 150, 300, 1.1), "Ti": ("TiCl4 / TDMAT", 150, 300, 0.5),
    "Ta": ("PDMAT / Ta(OEt)5", 200, 300, 0.5), "La": ("La(iPrAMD)3", 200, 300, 0.4),
    "Y": ("Y(iPrCp)3 / Y(thd)3", 250, 350, 0.4), "Si": ("BDEAS / 3DMAS", 200, 400, 0.8),
    "Zn": ("DEZ", 120, 200, 1.8), "Sn": ("TDMASn", 150, 250, 0.5),
    "Nb": ("Nb(OEt)5 / NbCl5", 200, 300, 0.4), "V": ("VO(OiPr)3 / VCl4", 150, 250, 0.4),
    "Mo": ("MoF6", 200, 300, 0.4), "W": ("WF6 / W(CO)6", 200, 350, 0.4),
    "Sr": ("Sr(iPr3Cp)2", 200, 300, 0.6), "Ba": ("Ba(iPr3Cp)2", 250, 350, 0.5),
    "Ga": ("TMGa", 150, 300, 0.5), "In": ("InCp / TMIn", 150, 300, 0.5),
    "Mg": ("Mg(EtCp)2", 200, 300, 1.0), "Ce": ("Ce(iPrCp)3", 200, 300, 0.4),
    "Gd": ("Gd(thd)3", 200, 300, 0.4), "Ni": ("NiCp2", 200, 300, 0.4),
    "Co": ("CoCp2", 200, 300, 0.4), "Fe": ("FeCp2", 200, 350, 0.3),
    "Mn": ("Mn(EtCp)2", 150, 300, 0.4), "Cr": ("CrCp2", 200, 350, 0.3),
    "Cu": ("Cu amidinate", 150, 250, 0.3), "Ru": ("Ru(EtCp)2", 200, 350, 0.5),
    "Pt": ("MeCpPtMe3", 200, 300, 0.5), "Ir": ("Ir(acac)3", 200, 300, 0.4),
    "Bi": ("BiPh3", 150, 300, 0.4)}
_EPS0 = 8.854e-14


def _pos(*v):
    return all((x is not None and np.isfinite(x) and x > 0) for x in v)


def corrected_band_gap(eg):
    return float(eg) * PBE_GAP_SCISSOR if _pos(eg) else None


def is_polar(pg):
    return str(pg) in POLAR_POINT_GROUPS


def ald_recipe(elements):
    els = list(elements or [])
    is_ox, is_nit = ("O" in els), ("N" in els)
    cats = [e for e in els if e not in ("O", "N")]
    if not cats or not (is_ox or is_nit):
        return None
    rows, gpcs, lows, highs, missing = [], [], [], [], []
    for c in cats:
        info = ALD_PRECURSORS.get(c)
        if info:
            p, lo, hi, g = info
            rows.append({"cation": c, "precursor": p}); gpcs.append(g)
            lows.append(lo); highs.append(hi)
        else:
            rows.append({"cation": c, "precursor": "전구체 미등록 (문헌 확인)"}); missing.append(c)
    co = "NH3 또는 N2/H2 플라즈마" if (is_nit and not is_ox) else "H2O 또는 O3 (필요 시 O2 플라즈마)"
    if lows:
        lo, hi = max(lows), min(highs); ov = lo <= hi
        if not ov:
            lo, hi = min(lows), max(highs)
    else:
        lo = hi = None; ov = None
    return {"precursors": rows, "co_reactant": co, "t_lo": lo, "t_hi": hi, "t_overlap": ov,
            "gpc": (sum(gpcs) / len(gpcs)) if gpcs else None, "supercycle": len(cats) > 1,
            "missing": missing, "film": "질화물" if (is_nit and not is_ox) else "산화물"}


def ald_cycles(gpc_a, t_nm):
    return int(round(t_nm * 10.0 / gpc_a)) if _pos(gpc_a, t_nm) else None


def thickness_for_eot(eot, kap):
    return float(eot) * float(kap) / EPS_SIO2 if _pos(kap, eot) else None


def gate_cap_density(eot):
    return _EPS0 * EPS_SIO2 / (eot * 1e-7) * 1e6 if _pos(eot) else None


def band_offsets(eg):
    if eg is None or not np.isfinite(eg):
        return None
    delta = max(0.0, float(eg) - EG_SI)
    return 0.6 * delta, 0.4 * delta


def gate_leakage(t_nm, barrier, m_eff=0.4, j0=1.0e6):
    if not _pos(t_nm, barrier):
        return None
    m0, q, hbar = 9.109e-31, 1.602e-19, 1.055e-34
    kap = np.sqrt(2.0 * m_eff * m0 * q * float(barrier)) / hbar
    return float(j0 * np.exp(-2.0 * kap * (t_nm * 1e-9)))


# ── 번들 데이터 ─────────────────────────────────────────────────────────────
_DATA_B64 = "__DATA_BLOB__"


@st.cache_data(show_spinner=False)
def load_data():
    raw = json.loads(gzip.decompress(base64.b64decode(_DATA_B64)).decode())
    return {k: pd.DataFrame(v) for k, v in raw.items()}


DATASETS = load_data()

# 사전 작성 AI 답변(canned) — 프리셋 매핑
CANNED = {
    "ALD로 합성 가능한 high-k 게이트 유전체 후보": ("고유전율 게이트 (high-k)",
        "### ALD 합성 가능 high-k 게이트 유전체 후보\\n\\n"
        "Materials Project DFT 데이터에서 **넓은 밴드갭(4~9 eV)·금속 제외·안정** 조건으로 "
        "스크리닝하고, **ALD 전구체가 확립된 단순 산화물**만 추렸습니다.\\n\\n"
        "- 품질지수 **κ·Eg**(×SiO₂)와 5 nm 기준 **EOT**로 SRAM/DRAM 적합성을 평가\\n"
        "- 각 후보는 Materials Project 원본 데이터에 링크(감사 가능)\\n"
        "- **ALD 합성 가능**한 것만 → 실제 양산성 고려\\n\\n"
        "→ 결과를 대시보드에 반영했습니다. *고유전율·ALD 공정·소자 시뮬레이션* 탭에서 확인하세요."),
    "HfO2 ZrO2 계열 강유전 메모리(FeRAM) 소재": ("강유전체 (FeRAM/FeFET)",
        "### HfO₂·ZrO₂ 계열 강유전 메모리(FeRAM/FeFET) 후보\\n\\n"
        "Hf 또는 Zr 함유 산화물을 추리고, 강유전성의 **결정학적 필요조건인 극성(polar) 점군** "
        "여부로 후보를 한 단계 정교화했습니다(HfO₂ 강유전상은 orthorhombic Pca2₁).\\n\\n"
        "- 화학식만이 아니라 **극성 공간군**으로 1차 선별\\n"
        "- 극성 점군은 필요조건(충분조건 아님) — 1차 우선 검토 대상\\n\\n"
        "→ *강유전체* 탭에서 극성 후보를 확인하세요."),
    "RRAM용 전이금속 산화물 (산소공공 스위칭)": ("저항변화 메모리 (RRAM)",
        "### RRAM용 전이금속 산화물 후보\\n\\n"
        "산소공공 필라멘트 스위칭에 적합한 **전이금속 산화물**을, 스위칭에 유리한 "
        "**밴드갭 2~6 eV** 영역 중심으로 추렸습니다(HfO₂·TiO₂·Ta₂O₅ 계열).\\n\\n"
        "→ *저항변화(RRAM)* 탭에서 확인하세요."),
}

st.set_page_config(page_title="메모리 반도체 소재 발굴 (데모)", page_icon="◈", layout="wide")
st.markdown("""
<style>
  h1,h2,h3 { color:#14202b; font-weight:650; letter-spacing:-.2px; }
  .block-container { padding-top:3.2rem; }
  .app-header { border-bottom:2px solid #1f4e79; padding-bottom:.55rem; margin-bottom:1rem; }
  .app-title { font-size:1.5rem; font-weight:700; color:#14202b; margin:0; }
  .app-sub { color:#5b6b76; font-size:.84rem; margin-top:.3rem; }
  div[data-testid="stMetricValue"] { font-family:ui-monospace,monospace; color:#1f4e79; font-weight:600; }
  .badge { display:inline-block; background:#eaf0f6; color:#1f4e79; border:1px solid #cfe0ee;
        padding:2px 10px; border-radius:3px; font-size:.74rem; font-weight:600; }
  .demo { display:inline-block; background:#fde8c8; color:#9e480e; border:1px solid #f0c178;
        padding:2px 10px; border-radius:3px; font-size:.72rem; font-weight:700; margin-left:8px; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="app-header"><div class="app-title">메모리 반도체 소재 발굴 플랫폼'
            '<span class="demo">OFFLINE DEMO</span></div>'
            '<div class="app-sub">Materials Project DFT 데이터 · high-k κ·Eg · 강유전체(극성) · '
            'RRAM · ALD 공정 레시피 · 소자(TCAD) 시뮬레이션 — 네트워크/API 없이 동작하는 데모</div>'
            '</div>', unsafe_allow_html=True)


def fig_layout(fig, h=420):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(family="ui-monospace, monospace", size=12), colorway=PALETTE)
    return fig


def kappa_eg_figure(dframe, title, h=460):
    full = dframe[dframe.kappa.notna() & dframe.band_gap.notna()].copy()
    sub = full[full.kappa <= KAPPA_MAX_RELIABLE].copy()
    fig = go.Figure()
    kmax = min(KAPPA_MAX_RELIABLE, max(90.0, (float(sub.kappa.max()) if not sub.empty else 90) * 1.05))
    ymax = max(10.5, (float(sub.band_gap.max()) if not sub.empty else 9) * 1.05)
    for prod, color in [(EPS_SIO2 * EG_SIO2, "#cfd8e0"), (150.0, "#b8c4cf"), (300.0, "#a3b2bf")]:
        kk = np.linspace(max(1.0, prod / ymax), kmax, 120)
        fig.add_trace(go.Scatter(x=kk, y=prod / kk, mode="lines", name=f"κ·Eg={prod:.0f}",
                      line=dict(dash="dot", width=1, color=color), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[r[1] for r in REFERENCE_DIELECTRICS],
                  y=[r[2] for r in REFERENCE_DIELECTRICS], mode="markers+text",
                  text=[r[0] for r in REFERENCE_DIELECTRICS], name="기준 유전체",
                  textposition="top center", textfont=dict(size=10, color="#9e480e"),
                  marker=dict(symbol="diamond", size=11, color="#c55a11")))
    if not sub.empty:
        fig.add_trace(go.Scatter(x=sub.kappa, y=sub.band_gap, mode="markers", name="후보 물질",
                      marker=dict(size=8, color=sub.score, colorscale="Viridis", showscale=True,
                                  colorbar=dict(title="발굴점수", thickness=12)),
                      text=sub.formula,
                      hovertemplate="%{text}<br>κ=%{x:.1f}<br>Eg=%{y:.2f} eV<extra></extra>"))
    fig.update_layout(xaxis_title="정적 유전율 κ", yaxis_title="밴드갭 Eg (eV)", title=title,
                      legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top",
                                  bgcolor="rgba(255,255,255,0.72)"))
    fig.update_xaxes(range=[0, kmax]); fig.update_yaxes(range=[0, ymax])
    return fig_layout(fig, h)


# ── AI 어시스턴트 (사전 작성 답변) ───────────────────────────────────────────
@st.dialog("AI 어시스턴트 (데모)", width="large")
def open_chat():
    st.caption("메모리 소재를 자연어로 요청하면 조건을 해석해 스크리닝하고 근거와 함께 답합니다. "
               "(데모: 사전 준비된 시나리오로 즉시·오프라인 응답)")
    sel = st.session_state.get("_canned_sel")
    if sel:
        preset, answer = CANNED[sel]
        with st.chat_message("user"):
            st.markdown(sel)
        with st.chat_message("assistant"):
            st.markdown(answer)
        n = len(DATASETS[preset])
        if st.button(f"📊 결과 보기 · 창 닫기  ({n}개 후보)", type="primary",
                     use_container_width=True):
            st.session_state.preset = preset
            st.session_state._canned_sel = None
            st.rerun()
        if st.button("다른 질문", use_container_width=True):
            st.session_state._canned_sel = None
            st.rerun()
    else:
        st.markdown("**예시 질문 (클릭하면 실행)**")
        for q in CANNED:
            if st.button(q, use_container_width=True, key="cq_" + q[:8]):
                st.session_state._canned_sel = q
                st.rerun()


# ── 사이드바: 프리셋(=번들 데이터셋) ─────────────────────────────────────────
st.sidebar.markdown("### 응용 프리셋")
preset = st.sidebar.selectbox("목적을 선택하면 해당 후보군이 표시됩니다",
                              list(DATASETS), key="preset")
st.sidebar.caption("데모 모드: 미리 스크리닝한 결과(MP DFT 데이터)를 즉시 표시합니다. "
                   "실제 버전은 사이드바 조건으로 라이브 스크리닝합니다.")
st.sidebar.divider()
st.sidebar.markdown("이 데모는 **네트워크·API 없이** 동작합니다. "
                    "AI·스크리닝·ALD·소자 시뮬레이션 모두 내장 데이터로 즉시 반응합니다.")

if st.button("AI 어시스턴트 열기", type="secondary",
             help="자연어 질문으로 스크리닝(데모: 사전 시나리오)"):
    open_chat()

df = DATASETS[preset].copy()

tabs = st.tabs(["개요", "고유전율 (High-k)", "강유전체 (FeRAM/FeFET)", "저항변화 (RRAM)",
                "추천 후보", "ALD 공정", "소자 시뮬레이션", "데이터"])

# 1) 개요
with tabs[0]:
    has_k = df[df.kappa.notna()]
    st.markdown(f'<span class="badge">{preset}</span> <b>{len(df)}</b>개 후보',
                unsafe_allow_html=True)
    m = st.columns(5)
    m[0].metric("후보 물질", f"{len(df)}")
    m[1].metric("κ 데이터 보유", f"{len(has_k)}")
    m[2].metric("고유전율 κ≥10", f"{int((df.kappa.fillna(0) >= 10).sum())}")
    m[3].metric("ALD 가능", f"{int(df.is_ald.sum())}")
    m[4].metric("최고 발굴점수", f"{df.score.max():.0f}")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(kappa_eg_figure(df, "κ–Eg 트레이드오프 지도"),
                        use_container_width=True)
    with c2:
        fig = px.histogram(df, x="band_gap", nbins=40, opacity=0.9,
                           color="crystal_system",
                           color_discrete_sequence=px.colors.qualitative.Set2,
                           labels={"band_gap": "밴드갭 (eV)", "crystal_system": "결정계"},
                           title="밴드갭 분포 (결정계별)")
        fig.update_layout(yaxis_title="물질 수", legend_title_text="결정계")
        st.plotly_chart(fig_layout(fig), use_container_width=True)

# 2) 고유전율
with tabs[1]:
    st.markdown("**고유전율(high-k) 게이트·커패시터 유전체** — κ가 클수록 커패시턴스에 유리하나 "
                "보통 Eg가 작아져 누설↑(κ–Eg 트레이드오프). κ·Eg와 EOT로 적합성 평가.")
    hk = df[df.highk_fom.notna() & df.kappa_reliable].copy()
    if hk.empty:
        st.info("이 후보군에는 κ 데이터가 있는 high-k 후보가 적습니다. 다른 프리셋을 보세요.")
    else:
        st.plotly_chart(kappa_eg_figure(df[df.highk_fom.notna()], "κ–Eg 지도 · 색=발굴점수"),
                        use_container_width=True)
        st.caption("⚠️ Eg는 DFT(PBE)라 실제보다 낮습니다('Eg보정'=×1.5 추정). 실제 누설 여유는 더 나은 편.")

# 3) 강유전체 (극성 공간군)
with tabs[2]:
    st.markdown("**강유전체 메모리 (FeRAM·FeFET)** — 강유전성의 결정학적 필요조건인 "
                "**극성(polar) 점군** 여부로 Hf/Zr 산화물 후보를 정교화합니다.")
    fe = df[df.has_hf_zr & df.is_oxide].copy()
    if fe.empty:
        st.info("이 후보군엔 Hf/Zr 산화물이 적습니다. '강유전체' 프리셋을 선택하세요.")
    else:
        n_polar = int(fe.is_polar.sum())
        c = st.columns(3)
        c[0].metric("Hf/Zr 산화물", f"{len(fe)}")
        c[1].metric("극성 상(강유전 후보)", f"{n_polar}")
        c[2].metric("극성 비율", f"{100 * n_polar / max(len(fe),1):.0f}%")
        fep = fe.assign(극성=fe.is_polar.map({True: "극성", False: "비극성"}))
        fig = px.scatter(fep, x="kappa", y="e_above_hull", color="극성", symbol="극성",
                         size="nsites", hover_data=["formula", "spacegroup", "point_group"],
                         color_discrete_map={"극성": "#c55a11", "비극성": "#7f8c98"},
                         labels={"kappa": "유전율 κ", "e_above_hull": "hull"},
                         title="유전율 vs 안정성 (색=극성 여부)")
        st.plotly_chart(fig_layout(fig), use_container_width=True)
        tp = fe.sort_values(["is_polar", "score"], ascending=[False, False]).head(12)
        st.dataframe(tp[["formula", "spacegroup", "point_group", "is_polar", "band_gap",
                         "kappa", "e_above_hull", "score"]], use_container_width=True,
                     hide_index=True, column_config={
                         "is_polar": st.column_config.CheckboxColumn("극성"),
                         "band_gap": st.column_config.NumberColumn("Eg", format="%.2f"),
                         "kappa": st.column_config.NumberColumn("κ", format="%.1f"),
                         "e_above_hull": st.column_config.NumberColumn("hull", format="%.3f"),
                         "score": st.column_config.NumberColumn("점수", format="%.0f")})
        st.caption("극성 점군은 강유전의 필요조건(충분조건 아님). 극성 후보를 1차 우선 검토 대상으로.")

# 4) RRAM
with tabs[3]:
    st.markdown("**저항변화 메모리(RRAM)** — 전이금속 산화물에서 산소공공 필라멘트 스위칭. "
                "스위칭 적합 밴드갭(2~6 eV) 강조.")
    rr = df[df.is_tm_oxide].copy()
    if rr.empty:
        st.info("이 후보군엔 전이금속 산화물이 적습니다. 'RRAM' 프리셋을 선택하세요.")
    else:
        fig = px.scatter(rr, x="band_gap", y="kappa", color="score", size="nsites",
                         hover_data=["formula", "chemsys"], color_continuous_scale="Viridis",
                         labels={"band_gap": "밴드갭 Eg", "kappa": "유전율 κ", "score": "발굴점수"},
                         title="전이금속 산화물: 밴드갭 vs 유전율")
        fig.add_vrect(x0=2.0, x1=6.0, fillcolor="#548235", opacity=0.12, line_width=0,
                      annotation_text="스위칭 적합대 (2~6 eV)", annotation_position="top left")
        st.plotly_chart(fig_layout(fig, 460), use_container_width=True)
        tp = rr.sort_values("score", ascending=False).head(12)
        st.dataframe(tp[["formula", "chemsys", "band_gap", "kappa", "e_above_hull", "score"]],
                     use_container_width=True, hide_index=True)

# 5) 추천 후보 (클릭 → MP)
with tabs[4]:
    st.markdown(f"**종합 발굴점수 Top 20 — 왜 이 점수인가** · `{preset}` 기준. "
                "각 막대는 **안정성·상용성·내구성·성능** 4축 기여로 분해되며, "
                "막대 길이가 곧 발굴점수입니다. (막대 클릭 → 아래 상세·MP 링크)")
    _PRESET_APP = {"고유전율 게이트 (high-k)": "highk", "강유전체 (FeRAM/FeFET)": "ferroelectric",
                   "저항변화 메모리 (RRAM)": "rram", "일반 탐색": "general"}
    _WEIGHTS = {
        "highk": dict(stability=22, manufacturability=28, durability=12, performance=38),
        "general": dict(stability=22, manufacturability=28, durability=12, performance=38),
        "ferroelectric": dict(stability=22, manufacturability=28, durability=10, performance=40),
        "rram": dict(stability=22, manufacturability=28, durability=10, performance=40)}
    best = df.sort_values("score", ascending=False).head(20).reset_index(drop=True)
    AXES = [("score_stability", "stability", "안정성", "🟦", "#4C78A8"),
            ("score_manu", "manufacturability", "상용성", "🟧", "#F58518"),
            ("score_durability", "durability", "내구성", "🟩", "#54A24B"),
            ("score_performance", "performance", "성능", "🟥", "#E45756")]
    cmap = {k: c for _, _, k, _, c in AXES}
    longdf = best.melt(id_vars=["formula", "material_id", "score"],
                       value_vars=[col for col, *_ in AXES], var_name="_ax", value_name="기여")
    longdf["축"] = longdf["_ax"].map({col: k for col, _, k, _, _ in AXES})
    fig = px.bar(longdf, x="기여", y="formula", color="축", orientation="h",
                 color_discrete_map=cmap,
                 category_orders={"축": [k for _, _, k, _, _ in AXES],
                                  "formula": best.formula.tolist()},
                 custom_data=["material_id"],
                 labels={"기여": "발굴점수 기여", "formula": "", "축": "점수 축"},
                 title="추천 소재 후보 · 점수 구성 (막대 클릭 → 상세)")
    fig.update_layout(barmode="stack", yaxis=dict(autorange="reversed"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1, title_text=""))
    fig.update_traces(hovertemplate="%{y}<br>%{fullData.name} 기여=%{x:.1f}점<extra></extra>")
    ev = st.plotly_chart(fig_layout(fig, 620), use_container_width=True,
                         on_select="rerun", selection_mode="points", key="rec")
    try:
        pts = ev["selection"]["points"]
    except (TypeError, KeyError):
        pts = []
    if pts and pts[0].get("y") in set(best.formula):
        st.session_state["rec_sel"] = pts[0]["y"]
    st.divider()
    sel = st.selectbox("상세 분석할 물질 (레이더 차트)", best.formula.tolist(), key="rec_sel")
    r0 = best[best.formula == sel].iloc[0]
    w = _WEIGHTS.get(_PRESET_APP.get(preset, "general"), _WEIGHTS["general"])
    theta = [k for _, _, k, _, _ in AXES]
    rvals = [(getattr(r0, col) / w[wk] * 100.0 if w.get(wk) else 0.0) for col, wk, *_ in AXES]
    cL, cR = st.columns([5, 4])
    with cL:
        st.metric(f"{sel} · 발굴점수", f"{r0.score:.1f} / 100")
        for col, wk, k, emoji, _ in AXES:
            v = getattr(r0, col); frac = (v / w[wk] * 100.0) if w.get(wk) else 0.0
            st.write(f"{emoji} **{k}** — {v:.1f}점 ({frac:.0f}% · 만점 {w.get(wk, 0)})")
        st.link_button(f"🔗 {sel} — Materials Project에서 열기",
                       f"https://materialsproject.org/materials/{r0.material_id}")
    with cR:
        rfig = go.Figure(go.Scatterpolar(r=rvals + [rvals[0]], theta=theta + [theta[0]],
                                         fill="toself", line_color="#4C78A8",
                                         fillcolor="rgba(76,120,168,0.35)"))
        rfig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], ticksuffix="%")),
                           showlegend=False, title=f"{sel} · 축별 달성도(%)")
        st.plotly_chart(fig_layout(rfig, 360), use_container_width=True)

# 6) ALD 공정
with tabs[5]:
    st.markdown("**스크리닝 → 합성: ALD 공정 레시피** — **산화물** 후보를 골라 전구체·공반응물·온도창·"
                "**사이클 수**까지 1차 설계. 발굴에서 *어떻게 만들지*로. "
                "(산소(O)와 화합물을 이루는 ALD 산화물만 표시)")
    ald = df[df.is_ald & df.is_oxide].sort_values("score", ascending=False)
    if ald.empty:
        st.info("이 후보군엔 ALD 가능 산화물 후보가 적습니다.")
    else:
        opts = {f"{r.formula} · {r.material_id} (점수 {r.score:.0f})": r.material_id
                for _, r in ald.head(40).iterrows()}
        pk = st.selectbox("후보 선택 (ALD 산화물)", list(opts), key="ald_pick")
        row = ald[ald.material_id == opts[pk]].iloc[0]
        rec = ald_recipe(list(row.elements))
        if rec:
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(f"#### {row.formula} — ALD {rec['film']} 공정")
                pdf = pd.DataFrame(rec["precursors"]).rename(
                    columns={"cation": "양이온", "precursor": "대표 전구체"})
                st.dataframe(pdf, hide_index=True, use_container_width=True)
                st.markdown(f"- **공반응물:** {rec['co_reactant']}")
                if rec["t_lo"] is not None:
                    st.markdown(f"- **권장 증착 온도:** {rec['t_lo']}–{rec['t_hi']} °C")
                if rec["gpc"]:
                    st.markdown(f"- **GPC(평균):** ~{rec['gpc']:.2f} Å/cycle")
                if rec["supercycle"]:
                    st.markdown("- **다성분 → 슈퍼사이클**(비율 보정) 필요.")
                if rec["missing"]:
                    st.markdown(f"- ⚠️ 전구체 미등록: {', '.join(rec['missing'])}")
            with c2:
                st.markdown("**🎯 두께 → 사이클 계산**")
                kap = float(row.kappa) if pd.notna(row.kappa) and row.kappa > 0 else None
                if kap:
                    eot = st.slider("목표 EOT (nm)", 0.5, 3.0, 1.0, 0.1, key="ald_eot")
                    tp = thickness_for_eot(eot, kap)
                    cc = st.columns(2)
                    cc[0].metric("필요 물리두께", f"{tp:.1f} nm" if tp else "—")
                    cc[1].metric("ALD 사이클 수", f"{ald_cycles(rec['gpc'], tp)}" if tp else "—")
                    st.caption(f"κ={kap:.1f} 기준.")
                else:
                    tt = st.slider("목표 물리두께 (nm)", 1.0, 20.0, 5.0, 0.5, key="ald_t")
                    st.metric("ALD 사이클 수", f"{ald_cycles(rec['gpc'], tt)}")
        st.caption("※ MP 조성 + ALD 문헌 일반값 결합한 1차 설계 가이드(실 공정은 최적화 필요). "
                   "전구체 참고: [Atomic Limits ALD Database](https://www.atomiclimits.com/alddatabase/) "
                   "(DOI:10.6100/ALDDatabase).")

# 7) 소자 시뮬레이션 (TCAD 방향)
with tabs[6]:
    st.markdown("**소자 시뮬레이션 (TCAD 연동 방향)** — κ·Eg·EOT에서 게이트 스택의 "
                "**커패시턴스·누설·밴드정렬**을 1차 예측. 풀 TCAD 연동의 *방향* 예시.")
    hk = df[df.highk_fom.notna() & df.kappa_reliable].sort_values("highk_fom", ascending=False)
    if hk.empty:
        st.info("κ 데이터가 있는 게이트 유전체 후보가 적습니다.")
    else:
        opts = {f"{r.formula} · κ={r.kappa:.1f} · Eg={r.band_gap:.2f}eV": r.material_id
                for _, r in hk.head(40).iterrows()}
        pk = st.selectbox("게이트 유전체 후보 선택", list(opts), key="dev_pick")
        row = hk[hk.material_id == opts[pk]].iloc[0]
        kap, eg = float(row.kappa), float(row.band_gap)
        dEc, dEv = band_offsets(eg) or (0.0, 0.0)
        dEc_si = (band_offsets(EG_SIO2) or (0.0, 0.0))[0]
        floor = lambda j: max(j, 1e-9) if j else 1e-9
        eot = st.slider("목표 EOT (nm)", 0.5, 3.0, 1.0, 0.1, key="dev_eot")
        tphys = thickness_for_eot(eot, kap)
        m = st.columns(4)
        m[0].metric("게이트 C_ox", f"{gate_cap_density(eot):.2f} µF/cm²")
        m[1].metric("물리두께", f"{tphys:.1f} nm")
        m[2].metric("ΔEc (배리어)", f"~{dEc:.1f} eV")
        m[3].metric("추정 누설", f"{floor(gate_leakage(tphys, dEc)):.0e} A/cm²")
        c1, c2 = st.columns(2)
        with c1:
            eots = np.linspace(0.5, 3.0, 30)
            jm = [floor(gate_leakage(thickness_for_eot(e, kap), dEc)) for e in eots]
            js = [floor(gate_leakage(e, dEc_si)) for e in eots]
            f = go.Figure()
            f.add_trace(go.Scatter(x=eots, y=js, name="SiO₂", line=dict(color="#c55a11", width=2)))
            f.add_trace(go.Scatter(x=eots, y=jm, name=f"{row.formula} (고-κ)",
                                   line=dict(color="#1f4e79", width=2)))
            f.update_yaxes(type="log", title="게이트 누설 J (A/cm², 예시)", exponentformat="power")
            f.update_xaxes(title="EOT (nm)")
            f.update_layout(title="EOT–누설: EOT↓ 시 SiO₂는 폭증, 고-κ는 억제",
                            legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top",
                                        bgcolor="rgba(255,255,255,0.6)"))
            st.plotly_chart(fig_layout(f, 380), use_container_width=True)
        with c2:
            ec_si, ev_si = 0.0, -EG_SI
            ec_d, ev_d = dEc, dEc - eg
            b = go.Figure()
            b.add_vrect(x0=1, x1=2.3, fillcolor="#9dc3e6", opacity=0.15, line_width=0)
            b.add_trace(go.Scatter(x=[0, 1], y=[0, 0], mode="lines",
                                   line=dict(color="#7f8c98", width=4)))
            b.add_trace(go.Scatter(x=[1, 1, 2.3, 2.3, 3.5], y=[0, ec_d, ec_d, ec_si, ec_si],
                                   mode="lines", line=dict(color="#1f4e79", width=3)))
            b.add_trace(go.Scatter(x=[1, 2.3, 2.3, 3.5], y=[ev_d, ev_d, ev_si, ev_si],
                                   mode="lines", line=dict(color="#548235", width=3)))
            b.add_annotation(x=1.65, y=ec_d + 0.5, text=f"{row.formula}", showarrow=False,
                             font=dict(size=11, color="#1f4e79"))
            b.update_yaxes(title="에너지 (eV)")
            b.update_xaxes(tickvals=[0.5, 1.65, 2.9], ticktext=["금속", "유전체", "Si"])
            b.update_layout(title="게이트 스택 밴드 정렬 (모식도)", showlegend=False)
            st.plotly_chart(fig_layout(b, 380), use_container_width=True)
        st.caption("※ 해석적 1차 모델(예시): C_ox 정확, 누설은 직접터널링 경향(절대값 미보정), "
                   "밴드오프셋은 PBE Eg 기반 추정. 풀 TCAD로 C–V/이동도/신뢰성/I–V 정밀화 가능.")

# 8) 데이터
with tabs[7]:
    q = st.text_input("화학식 검색", placeholder="예: HfO2, ZrO2")
    show = df[df.formula.str.contains(q, case=False, na=False)] if q else df
    st.dataframe(show.sort_values("score", ascending=False), use_container_width=True, height=460)
    st.download_button("CSV 다운로드", df.to_csv(index=False).encode("utf-8-sig"),
                       "demo_candidates.csv", "text/csv")
'''

with open("demo.py", "w") as f:
    f.write(TEMPLATE.replace("__DATA_BLOB__", blob))
print("demo.py 생성 완료:", len(TEMPLATE), "chars (blob 제외)")
