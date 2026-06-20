"""Claude 기반 AI 스크리닝 어시스턴트.

자연어 질문 → Claude tool use(screen_materials) → Materials Project 조회 →
근거자료(물성값·물리 시뮬레이션 해석·MP 링크·스크리닝 조건)와 함께 해설.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

import data as datamod

# 속도·비용·품질 균형으로 Sonnet 4.6 사용(도구 라우팅+근거 기반 설명에 충분).
# 최고 수준 추론이 필요하면 "claude-opus-4-8"로 교체.
MODEL = "claude-sonnet-4-6"

MP_URL = "https://materialsproject.org/materials/{}"

APP_LABELS = {
    "general": "일반 탐색",
    "highk": "고유전율 게이트 (high-k)",
    "dram_cap": "DRAM 커패시터 유전체",
    "ferroelectric": "강유전체 (FeRAM/FeFET)",
    "nand_oxide": "NAND 터널/블로킹 산화물",
    "rram": "저항변화 메모리 (RRAM)",
}

GAP_MAP = {
    "any": datamod.GAP_TYPE_ANY,
    "direct": datamod.GAP_TYPE_DIRECT,
    "indirect": datamod.GAP_TYPE_INDIRECT,
}

SORT_COLS = {
    "score": "score", "highk_fom": "highk_fom", "kappa": "kappa",
    "eot": "eot_5nm", "band_gap": "band_gap", "stability": "e_above_hull",
}

SYSTEM_PROMPT = """\
당신은 메모리 반도체(SRAM·DRAM·NAND/Flash 및 차세대 메모리) 소재 발굴을 돕는 \
재료과학 연구 어시스턴트입니다. Materials Project(DFT 계산 데이터베이스) 기반의 \
스크리닝 도구를 갖추고 있으며, 반도체 메모리 소재·고유전율(high-k)·강유전체· \
저항변화 물리 전반의 개념 질문에도 답할 수 있습니다.

핵심 관심사는 메모리 소자용 유전체·강유전체·저항변화 산화물입니다. \
태양전지·전력반도체 같은 에너지 소자가 아니라 메모리 소재가 본 플랫폼의 목적임을 \
유념하세요.

먼저 질문 유형을 판단해 둘 중 하나로 대응하세요.

[유형 A] 물질 탐색·추천·스크리닝이 필요한 질문
(예: "SiO2를 대체할 high-k 게이트 유전체 후보", "HfO2계 강유전 메모리 소재", \
"RRAM용 전이금속 산화물 찾아줘")
→ screen_materials 도구를 호출해 스크리닝을 실행한 뒤, 결과(JSON)를 근거로 답합니다.
  - 응용 분야 매핑:
    · 고유전율 게이트(SRAM/로직)는 넓은 Eg(약 4~9 eV)·application="highk".
    · DRAM 커패시터 유전체는 매우 높은 κ 우선·application="dram_cap".
    · 강유전체(FeRAM/FeFET)는 application="ferroelectric" (보통 Hf 또는 Zr 포함 \
산화물; include_elements에 Hf 또는 Zr, 필요시 O 추가).
    · NAND 터널/블로킹 산화물은 넓은 Eg(약 5 eV 이상)·application="nand_oxide".
    · 저항변화 RRAM은 전이금속 산화물·밴드갭 약 2~6 eV·application="rram".
    · 특별한 응용이 없으면 application="general".
  - 산화물을 원하면 include_elements에 "O"를 넣고, 독성/원치 않는 원소는 \
exclude_elements로 배제하세요.
  - "ALD로 합성/증착 가능", "원자층증착", "박막 공정에 맞는" 등 합성법을 언급하면 \
ald_only=true로 설정하세요(전구체가 확립된 양이온의 단순 산화물·질화물로 한정). \
실제 ALD 가능성은 전구체·공정창에 좌우되는 1차 휴리스틱임을 답변에서 밝히세요.
  - 답변에는 반드시 아래 4가지 '근거자료'를 모두 포함하세요:
    (A) 물성 데이터값: 핵심 후보의 유전율 κ, 밴드갭 Eg, E above hull(안정성), \
결정계, (해당 시) ALD 합성 가능 여부 등 실제 수치를 표나 목록으로 제시.
    (B) 물리 지표 해석: high-k 품질지수 κ·Eg(×SiO₂)와 EOT(5 nm 기준, nm)의 의미 \
(커패시턴스 vs 누설 트레이드오프, 미세화), 강유전/RRAM 적합성 등을 해석.
    (C) MP 상세페이지 링크: 각 추천 물질을 markdown 링크로 제공. \
형식은 https://materialsproject.org/materials/<material_id>.
    (D) 스크리닝 조건 요약: 실제 적용된 필터(밴드갭 범위, 안정성 기준, 조성 \
제약 등)와 검색 결과 개수를 명시.
  - 마지막에 1~2문장으로 연구적 시사점이나 다음 탐색 제안을 덧붙이세요.

[유형 B] 개념 설명·정의·물리 해석·연구 조언, 또는 직전 결과에 대한 후속 질문
(예: "high-k에서 κ·Eg 트레이드오프가 뭐야?", "EOT는 어떻게 정의돼?", \
"왜 HfO2가 강유전 메모리에 쓰여?", "방금 1번 후보가 왜 유리해?")
→ 도구를 호출하지 말고, 재료과학 지식과 (있다면) 직전 대화의 스크리닝 결과를 \
바탕으로 한국어로 직접 설명합니다. 스크리닝이 꼭 필요하지 않으면 억지로 도구를 \
호출하지 마세요.

유형이 애매하면 무엇을 도와줄지 짧게 되묻거나, 합리적으로 해석해 진행하세요.

공통: 답변은 markdown으로 구조화하고, 과장 없이 근거에 기반해 설명하며, \
도구 결과나 확립된 지식에 없는 수치를 지어내지 마세요. 강유전성·RRAM 스위칭은 \
특정 준안정 상·결함에 의존하므로, 화학계 기반 1차 후보임을 필요시 밝히세요."""

TOOLS = [
    {
        "name": "screen_materials",
        "description": (
            "Materials Project에서 메모리 반도체용 소재 후보를 스크리닝한다. "
            "밴드갭/안정성/조성 조건으로 필터링하고 정적 유전율 κ, high-k 품질지수 "
            "κ·Eg(×SiO₂), EOT(등가산화막두께) 등 파생 지표를 함께 계산해 상위 후보를 "
            "반환한다. high-k 게이트/DRAM 커패시터, 강유전체(FeRAM/FeFET), NAND "
            "터널·블로킹 산화물, RRAM 전이금속 산화물 탐색에 사용한다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "band_gap_min": {"type": "number", "description": "최소 밴드갭 (eV)"},
                "band_gap_max": {"type": "number", "description": "최대 밴드갭 (eV)"},
                "gap_type": {
                    "type": "string", "enum": ["any", "direct", "indirect"],
                    "description": "밴드갭 종류. 직접갭은 발광/태양전지에 유리.",
                },
                "exclude_metals": {
                    "type": "boolean",
                    "description": "금속(밴드갭 0) 제외 여부. 보통 true 권장.",
                },
                "max_energy_above_hull": {
                    "type": "number",
                    "description": "최대 E above hull (eV/atom). 작을수록 안정. 보통 0.05~0.1.",
                },
                "stable_only": {
                    "type": "boolean",
                    "description": "열역학적 완전 안정(hull≈0) 물질만.",
                },
                "experimental_only": {
                    "type": "boolean",
                    "description": "실험적으로 합성된 물질만(순수 이론물질 제외).",
                },
                "ald_only": {
                    "type": "boolean",
                    "description": ("ALD(원자층증착)로 합성 가능한 후보만. 전구체가 "
                                    "확립된 양이온의 단순 산화물·질화물(이원~삼원계)로 "
                                    "한정한다. 박막 공정 적합성이 중요할 때 true."),
                },
                "include_elements": {
                    "type": "array", "items": {"type": "string"},
                    "description": "반드시 포함할 원소 기호 목록. 예: ['Ga','N'].",
                },
                "exclude_elements": {
                    "type": "array", "items": {"type": "string"},
                    "description": "배제할 원소 기호 목록. 예: ['Pb','Hg','Cd'].",
                },
                "num_elements": {
                    "type": "array", "items": {"type": "integer"},
                    "description": "구성 원소 개수 제한. 예: [2] 이원계, [2,3] 이~삼원계.",
                },
                "max_sites": {
                    "type": "integer",
                    "description": "단위셀 최대 원자 수. 기본 30.",
                },
                "application": {
                    "type": "string",
                    "enum": ["general", "highk", "dram_cap", "ferroelectric",
                             "nand_oxide", "rram"],
                    "description": ("발굴점수 계산에 쓰는 메모리 응용 분야. "
                                    "highk=고유전율 게이트, dram_cap=DRAM 커패시터, "
                                    "ferroelectric=강유전체(FeRAM/FeFET), "
                                    "nand_oxide=NAND 터널/블로킹 산화물, "
                                    "rram=저항변화 메모리."),
                },
                "max_results": {
                    "type": "integer",
                    "description": "MP에서 가져올 최대 후보 수. 기본 300.",
                },
                "sort_by": {
                    "type": "string",
                    "enum": list(SORT_COLS.keys()),
                    "description": "상위 후보 정렬 기준. 기본 score(종합 발굴점수).",
                },
            },
            "required": ["band_gap_min", "band_gap_max", "application"],
        },
    }
]


def _f(x, nd=3):
    """안전한 반올림 (None/NaN → None)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


def execute_screening(args: dict):
    """Claude가 고른 인자로 스크리닝 실행 → (df, criteria, payload)."""
    app_key = args.get("application", "general")
    gap_type = GAP_MAP.get(args.get("gap_type", "any"), datamod.GAP_TYPE_ANY)
    include = args.get("include_elements") or None
    exclude = args.get("exclude_elements") or None
    nelements = args.get("num_elements") or []
    max_results = int(args.get("max_results") or 300)
    max_sites = int(args.get("max_sites") or 30)

    # 강유전체(FeRAM/FeFET)는 Hf 또는 Zr 함유 산화물이 핵심 모재이므로
    # "Hf OR Zr" OR-조회로 후보를 확보한다(MP의 elements는 AND 의미).
    include_any = ["Hf", "Zr"] if app_key == "ferroelectric" and not include else None

    df = datamod.run_screening(
        bg_min=float(args.get("band_gap_min", 0.0)),
        bg_max=float(args.get("band_gap_max", 10.0)),
        hull=float(args.get("max_energy_above_hull", 0.1)),
        nsites_max=max_sites,
        max_results=max_results,
        include=include, exclude=exclude, nelements=nelements,
        app_key=app_key, include_any=include_any,
        gap_type=gap_type,
        exclude_metal=bool(args.get("exclude_metals", True)),
        stable_only=bool(args.get("stable_only", False)),
        exp_only=bool(args.get("experimental_only", False)),
        ald_only=bool(args.get("ald_only", False)),
    )

    sort_col = SORT_COLS.get(args.get("sort_by", "score"), "score")

    criteria = {
        "밴드갭(eV)": f"{args.get('band_gap_min')}~{args.get('band_gap_max')}",
        "밴드갭종류": gap_type,
        "금속제외": bool(args.get("exclude_metals", True)),
        "E_above_hull최대": args.get("max_energy_above_hull", 0.1),
        "안정물질만": bool(args.get("stable_only", False)),
        "실험물질만": bool(args.get("experimental_only", False)),
        "ALD합성만": bool(args.get("ald_only", False)),
        "포함원소": include,
        "제외원소": exclude,
        "원소개수": nelements or "전체",
        "응용분야": APP_LABELS.get(app_key, app_key),
        "정렬기준": sort_col,
    }

    if df.empty:
        payload = {"applied_criteria": criteria, "total_found": 0,
                   "top_materials": [], "note": "조건에 맞는 물질이 없습니다."}
        return df, criteria, payload

    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=(sort_col == "e_above_hull"),
                            na_position="last").reset_index(drop=True)

    top = df.head(15)
    mats = []
    for _, r in top.iterrows():
        mats.append({
            "material_id": r.material_id,
            "formula": r.formula,
            "mp_url": MP_URL.format(r.material_id),
            "band_gap_eV": _f(r.band_gap, 3),
            "gap_kind": r.gap_kind,
            "e_above_hull": _f(r.e_above_hull, 4),
            "is_stable": bool(r.is_stable),
            "crystal_system": r.crystal_system,
            "kappa": _f(r.kappa, 2),
            "highk_fom_xSiO2": _f(r.highk_fom, 2),
            "EOT_5nm_nm": _f(r.eot_5nm, 2),
            "is_oxide": bool(r.is_oxide),
            "has_Hf_or_Zr": bool(r.has_hf_zr),
            "is_TM_oxide": bool(r.is_tm_oxide),
            "ald_synthesizable": bool(r.is_ald),
            "Debye_K": _f(r.Debye, 0),
            "discovery_score": _f(r.score, 1),
        })

    stats = {
        "total_found": int(len(df)),
        "avg_band_gap_eV": _f(df.band_gap.mean(), 2),
        "n_with_kappa": int(df.kappa.notna().sum()),
        "n_highk_kappa_ge10": int((df.kappa.fillna(0) >= 10).sum()),
        "n_stable": int(df.is_stable.sum()),
        "n_ald_synthesizable": int(df.is_ald.sum()),
        "max_highk_fom_xSiO2": _f(df.highk_fom.max(), 2),
        "max_kappa": _f(df.kappa.max(), 2),
    }

    payload = {
        "applied_criteria": criteria,
        "summary_stats": stats,
        "top_materials": mats,
    }
    return df, criteria, payload


def _stream_claude(client, messages, token_cb=None):
    """Claude를 스트리밍으로 호출. token_cb가 있으면 답변 텍스트를 토큰 단위로
    흘려보내(체감 지연 감소) 최종 메시지를 반환한다.

    effort는 chat·설명 작업 권장값인 low로 두어 추론 지연을 줄인다(기본 high).
    """
    with client.messages.stream(
        model=MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        tools=[{**TOOLS[0], "cache_control": {"type": "ephemeral"}}],
        messages=messages,
    ) as stream:
        if token_cb:
            for text in stream.text_stream:   # 답변 텍스트 델타만(생각/도구 제외)
                token_cb(text)
        return stream.get_final_message()


def _text_of(resp):
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _screening_brief(args: dict) -> str:
    """스크리닝 도구 인자를 진행표시용 한 줄 요약으로 변환."""
    parts = [APP_LABELS.get(args.get("application", "general"),
                            args.get("application", ""))]
    bmin, bmax = args.get("band_gap_min"), args.get("band_gap_max")
    if bmin is not None or bmax is not None:
        parts.append(f"밴드갭 {bmin}~{bmax} eV")
    if args.get("include_elements"):
        parts.append("포함 " + ",".join(args["include_elements"]))
    if args.get("exclude_elements"):
        parts.append("제외 " + ",".join(args["exclude_elements"]))
    if args.get("ald_only"):
        parts.append("ALD 합성 가능만")
    return " · ".join(p for p in parts if p)


def answer_question(client, question: str, history: list, status_cb=None,
                    token_cb=None, df_cb=None):
    """tool-use 루프 실행. (최종텍스트, 결과 df 또는 None) 반환.

    history: 이전 턴들의 {'role','content'(text)} 목록(표시/문맥용).
    status_cb(kind, detail=None): 진행 상황 콜백(UI 표시용). kind는
    "thinking"(질문 해석·판단), "screening"(MP 스크리닝 실행; detail=조건 요약),
    "answering"(도구 결과로 답변 작성) 중 하나. None이면 무시한다.
    token_cb(text): 답변 텍스트 토큰을 실시간으로 전달(스트리밍 표시용).
    df_cb(df, criteria): 스크리닝이 끝나는 즉시 호출. 긴 답변 생성을 기다리지
    않고 결과를 바로 대시보드에 반영하기 위한 콜백.
    """
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": question})

    result_df = None
    last_criteria = None
    for i in range(5):  # 안전 상한
        if status_cb:
            status_cb("answering" if i else "thinking")
        resp = _stream_claude(client, messages, token_cb)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return _text_of(resp), result_df, last_criteria

        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            if block.name == "screen_materials":
                if status_cb:
                    status_cb("screening", _screening_brief(block.input))
                df, criteria, payload = execute_screening(block.input)
                result_df = df
                last_criteria = criteria
                # 답변 생성을 기다리지 않고 스크리닝 결과를 즉시 대시보드에 반영
                if df_cb is not None and df is not None and not df.empty:
                    df_cb(df, criteria)
                content = json.dumps(payload, ensure_ascii=False)
            else:
                content = json.dumps({"error": "unknown tool"}, ensure_ascii=False)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            })
        messages.append({"role": "user", "content": tool_results})

    return ("도구 호출이 반복 한도를 초과했습니다. 질문을 더 구체적으로 다시 "
            "시도해 주세요."), result_df, last_criteria


def _get_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None


EXAMPLE_PROMPTS = (
    "예시 질문\n"
    "**소재 탐색·스크리닝**\n"
    "- SiO₂를 대체할 high-k 게이트 유전체 후보\n"
    "- ALD로 합성 가능한 high-k 유전체만 추려줘\n"
    "- HfO₂·ZrO₂ 계열 강유전 메모리(FeRAM/FeFET) 소재\n"
    "- RRAM용 전이금속 산화물 (산소공공 스위칭)\n"
    "\n"
    "**개념·정의·물리 해설** (스크리닝 없이 바로 답변)\n"
    "- high-k에서 κ·Eg 트레이드오프가 뭐야?\n"
    "- EOT는 어떻게 정의돼?\n"
    "- 왜 HfO₂가 강유전 메모리에 쓰여?\n"
    "- ALD가 메모리 유전체 공정에서 왜 중요해?")


@st.dialog("AI 어시스턴트", width="large")
def open_chat_dialog():
    """챗봇을 대시보드와 분리된 팝업 모달 창으로 띄운다.

    모달 안의 위젯 조작은 fragment rerun이라 뒤쪽 대시보드는 즉시 갱신되지
    않는다. 스크리닝 결과(st.session_state.df)는 사용자가 창을 닫을 때의
    전체 rerun에서 대시보드에 반영된다.
    """
    render_chat_panel()


def render_chat_panel():
    """AI 어시스턴트 모달 본문.

    이전 대화를 위에 그리고(오래된 대화는 expander로 접음), 새 질문의 답변은
    그 아래에서 토큰 단위로 스트리밍해 체감 지연을 줄인다.
    """
    import anthropic

    st.caption("메모리 소재(high-k·강유전체·NAND 산화물·RRAM) 발굴을 돕는 대화형 "
               "어시스턴트입니다. 자연어 요청은 스크리닝 조건으로 해석해 후보를 "
               "탐색하고, 개념·정의·물리 해석에 관한 질문에는 근거와 함께 답변합니다. "
               "스크리닝 결과는 창을 닫으면 대시보드에 반영됩니다.")

    api_key = _get_api_key()
    if not api_key:
        st.warning(
            "Anthropic API 키가 없습니다. "
            "`.streamlit/secrets.toml`에 `ANTHROPIC_API_KEY`를 설정하세요.")
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    prompt = st.chat_input("메시지를 입력하세요…")

    # 이번 턴 이전까지의 대화를 먼저 그린다(스트리밍될 새 답변은 그 아래에 쌓는다).
    existing = st.session_state.chat_history
    if not existing and not prompt:
        st.info(EXAMPLE_PROMPTS)
        return
    if existing:
        older, latest = existing[:-2], existing[-2:]
        if older:
            with st.expander(f"이전 대화 {len(older) // 2}건 보기", expanded=False):
                box = st.container(height=320)
                for msg in older:
                    with box.chat_message(msg["role"]):
                        st.markdown(msg["content"])
        for msg in latest:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if not prompt:
        return

    # 새 질문: 사용자 메시지 → 진행 상태(단계) → 답변(토큰 스트리밍)을 차례로 노출.
    with st.chat_message("user"):
        st.markdown(prompt)

    status = st.status("질문을 해석하는 중…", expanded=True)
    answer_box = st.chat_message("assistant").empty()   # 상태 아래, 스트리밍 답변 자리
    buf = {"text": ""}
    screened = {"flag": False}

    def on_event(kind, detail=None):
        if kind == "thinking":
            buf["text"] = ""                            # 새 호출 시작 → 화면 초기화
            answer_box.markdown("")
            status.update(label="질문 분석 중…")
            status.write("🧭 질문 내용을 분석하고 있습니다.")
        elif kind == "screening":
            screened["flag"] = True
            status.update(label="Materials Project 스크리닝 중…")
            status.write(f"🔬 스크리닝 실행 — {detail}")
        elif kind == "answering":
            buf["text"] = ""                            # 라우팅 단계 텍스트 지우고 답변 시작
            answer_box.markdown("")
            status.update(label="답변을 작성하는 중…")
            status.write("✍️ 스크리닝 결과를 근거로 답변을 정리합니다.")

    def on_token(text):
        buf["text"] += text
        answer_box.markdown(buf["text"] + " ▌")          # 생성되는 대로 즉시 표시

    def on_df(d, crit):
        # 스크리닝 즉시 대시보드 데이터 커밋(긴 답변을 기다리지 않음). 모달은
        # fragment라 화면 갱신은 창을 닫을 때지만, 그 시점엔 이미 반영돼 있다.
        st.session_state.df = d
        if crit:
            st.session_state.app_label = crit.get("응용분야", "AI 스크리닝")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        answer, df, criteria = answer_question(
            client, prompt, existing, on_event, on_token, on_df)
    except Exception as e:
        status.update(label="오류 발생", state="error")
        st.error(f"AI 응답 오류: {e}")
        return

    answer_box.markdown(answer)                          # 커서 제거, 최종 답변 확정

    if df is not None and not df.empty:
        status.update(label=f"스크리닝 완료 · 후보 {len(df)}개를 대시보드에 반영",
                      state="complete", expanded=False)
    elif screened["flag"]:
        status.update(label="스크리닝 완료 · 조건에 맞는 물질 없음",
                      state="complete", expanded=False)
    else:
        status.update(label="질문에 바로 답변 (스크리닝 없음)",
                      state="complete", expanded=False)

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    if df is not None and not df.empty:
        st.session_state.df = df
        if criteria:
            st.session_state.app_label = criteria.get("응용분야", "AI 스크리닝")
