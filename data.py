"""Materials Project 조회 + 파생 물성 계산 공용 모듈.

app.py(사이드바 스크리닝)와 chatbot.py(AI 어시스턴트)가 동일한 조회/필터
로직을 공유하도록 분리했다.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from mp_api.client import MPRester

import phys_models as phys

_BUILTIN_MP_KEY = "jQlXCiEJQLNOjMZiV7es4Fgc7FIyyOOZ"  # MP 무료 API 기본 키(env/secrets의 MP_API_KEY가 우선)


def _mp_api_key():
    """Materials Project API 키를 환경변수 → Streamlit secrets 순으로 읽는다.

    MP는 무료 API라 기본 키를 내장한다(_BUILTIN_MP_KEY). env 또는 secrets에
    MP_API_KEY가 있으면 그 값이 우선한다.
    """
    key = os.environ.get("MP_API_KEY")
    if key:
        return key
    try:
        if st.secrets.get("MP_API_KEY"):
            return st.secrets["MP_API_KEY"]
    except Exception:
        pass
    return _BUILTIN_MP_KEY


API_KEY = _mp_api_key()

FIELDS = [
    "material_id", "formula_pretty", "band_gap", "is_gap_direct", "is_metal",
    "energy_above_hull", "formation_energy_per_atom", "is_stable",
    "nsites", "nelements", "elements", "chemsys", "volume", "density",
    "symmetry", "theoretical", "bulk_modulus", "shear_modulus",
    "homogeneous_poisson", "universal_anisotropy",
    "e_total", "e_electronic", "n", "is_magnetic", "ordering",
    "total_magnetization", "weighted_work_function",
]

GAP_TYPE_ANY = "전체"
GAP_TYPE_DIRECT = "직접갭만"
GAP_TYPE_INDIRECT = "간접갭만"


def _vrh(x):
    return x.get("vrh") if isinstance(x, dict) else x


@st.cache_data(show_spinner=False, ttl=3600)
def fetch(params: dict):
    q = dict(
        band_gap=(params["bg_min"], params["bg_max"]),
        energy_above_hull=(0, params["hull"]),
        nsites=(1, params["nsites_max"]),
        # 안정성순(hull 오름차순)으로 가져와, max_results 상한 안에서 가장 안정한
        # = 실제로 흔히 쓰이는 소재가 먼저 포함되게 한다(단순 산화물 누락 방지).
        _sort_fields=["energy_above_hull"],
        fields=FIELDS, num_chunks=1, chunk_size=params["max_results"],
    )
    if params["include"]:
        q["elements"] = params["include"]
    if params["exclude"]:
        q["exclude_elements"] = params["exclude"]
    if params["nelements"]:
        q["num_elements"] = (min(params["nelements"]), max(params["nelements"]))
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(**q)

    rows = []
    for d in docs:
        sym = d.symmetry
        cs = sym.crystal_system if sym else None
        rows.append(dict(
            material_id=str(d.material_id),
            formula=d.formula_pretty,
            band_gap=d.band_gap,
            gap_kind="직접" if d.is_gap_direct else "간접",
            is_metal=bool(d.is_metal),
            e_above_hull=d.energy_above_hull,
            e_form=d.formation_energy_per_atom,
            is_stable=bool(d.is_stable),
            nsites=d.nsites,
            nelements=d.nelements,
            elements=[str(e) for e in d.elements],
            chemsys=d.chemsys,
            volume=d.volume,
            density=d.density,
            crystal_system=cs.value if cs else "N/A",
            point_group=str(sym.point_group) if sym and sym.point_group else "N/A",
            spacegroup=str(sym.symbol) if sym and sym.symbol else "N/A",
            spacegroup_no=int(sym.number) if sym and sym.number else None,
            theoretical=bool(d.theoretical),
            bulk=_vrh(d.bulk_modulus),
            shear=_vrh(d.shear_modulus),
            poisson=d.homogeneous_poisson,
            eps_total=d.e_total,
            eps_elec=d.e_electronic,
            refractive=d.n,
            is_magnetic=bool(d.is_magnetic),
            ordering=str(d.ordering) if d.ordering else "N/A",
            magnetization=d.total_magnetization,
            work_function=d.weighted_work_function,
        ))
    return rows


def build_df(rows, app_key):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # ── 메모리 소재 핵심 지표 ────────────────────────────────────────────
    df["kappa"] = df["eps_total"]                         # 정적 유전율 κ (별칭)
    df["kappa_reliable"] = df["kappa"].apply(phys.kappa_reliable)  # DFPT 발산 아티팩트 배제용
    df["highk_fom"] = df.apply(
        lambda r: phys.highk_fom(r.eps_total, r.band_gap), axis=1)   # κ·Eg (×SiO2)
    df["eot_5nm"] = df.apply(
        lambda r: phys.eot(r.eps_total, 5.0), axis=1)    # 5 nm 물리두께 기준 EOT (nm)

    # 화학계 플래그 (강유전체/RRAM 카테고리 판별)
    df["is_oxide"] = df["elements"].apply(lambda es: "O" in es)
    df["has_hf_zr"] = df["elements"].apply(
        lambda es: ("Hf" in es) or ("Zr" in es))
    df["is_tm_oxide"] = df.apply(
        lambda r: r.is_oxide and any(e in phys.TRANSITION_METALS for e in r.elements),
        axis=1)
    df["is_ald"] = df["elements"].apply(phys.ald_synthesizable)   # ALD 합성 유망(휴리스틱)
    df["band_gap_corr"] = df["band_gap"].apply(phys.corrected_band_gap)  # PBE 과소평가 보정 추정 Eg
    df["is_polar"] = df["point_group"].apply(phys.is_polar)       # 극성 점군(강유전 1차 선별)

    # ── 박막 열·기계 안정성 보조 지표 ───────────────────────────────────
    df["Pugh"] = df.apply(lambda r: phys.pugh_ratio(r.bulk, r.shear), axis=1)
    df["Hv"] = df.apply(lambda r: phys.vickers_hardness(r.bulk, r.shear), axis=1)
    df["Debye"] = df.apply(
        lambda r: phys.debye_temperature(r.bulk, r.shear, r.density, r.nsites, r.volume),
        axis=1)

    # ── 응용별 종합 발굴 점수 ───────────────────────────────────────────
    df["score"] = df.apply(
        lambda r: phys.memory_score(
            band_gap=r.band_gap, e_above_hull=r.e_above_hull, kappa=r.eps_total,
            is_oxide=r.is_oxide, has_hf_zr=r.has_hf_zr, is_tm_oxide=r.is_tm_oxide,
            is_ald=r.is_ald, experimental=(not r.theoretical), is_polar=r.is_polar,
            application=app_key),
        axis=1)
    return df


def run_screening(*, bg_min, bg_max, hull, nsites_max, max_results,
                  include, exclude, nelements, app_key,
                  gap_type=GAP_TYPE_ANY, exclude_metal=False, stable_only=False,
                  exp_only=False, nonmag_only=False, bulk_min=0, eps_min=0.0,
                  ald_only=False, include_any=None):
    """조회 + 파생물성 + 후처리 필터를 한 번에 수행해 정제된 DataFrame을 반환.

    include_any: 주어지면 "이 중 하나라도 포함"(OR) 의미로 원소별 개별 조회를
    수행해 material_id 기준으로 합친다. MP의 elements 필터는 AND 의미라서
    강유전체(Hf 또는 Zr) 같은 OR 조건은 이렇게 처리한다. include(고정 AND
    원소)와 함께 쓰면 각 OR 후보 원소에 include가 AND로 결합된다.
    """
    base = dict(
        bg_min=bg_min, bg_max=bg_max, hull=hull, nsites_max=nsites_max,
        max_results=max_results, include=include, exclude=exclude,
        nelements=nelements,
    )
    if include_any:
        rows, seen = [], set()
        for el in include_any:
            p = dict(base, include=(list(include) if include else []) + [el])
            for r in fetch(p):
                mid = r["material_id"]
                if mid not in seen:
                    seen.add(mid)
                    rows.append(r)
    else:
        rows = fetch(base)
    df = build_df(rows, app_key)
    if not df.empty:
        if gap_type == GAP_TYPE_DIRECT:
            df = df[df.gap_kind == "직접"]
        elif gap_type == GAP_TYPE_INDIRECT:
            df = df[df.gap_kind == "간접"]
        if exclude_metal:
            df = df[~df.is_metal]
        if stable_only:
            df = df[df.is_stable]
        if exp_only:
            df = df[~df.theoretical]
        if nonmag_only:
            df = df[~df.is_magnetic]
        if bulk_min > 0:
            df = df[df.bulk.fillna(0) >= bulk_min]
        if eps_min > 0:
            df = df[df.eps_total.fillna(0) >= eps_min]
        if ald_only:
            df = df[df.is_ald]
    return df.reset_index(drop=True)
