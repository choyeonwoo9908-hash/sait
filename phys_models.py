"""물리 기반 시뮬레이션 / 유도 물성 계산 모듈 (메모리 반도체 소재 중심).

Materials Project summary 데이터에서 얻는 1차 물성(밴드갭, 정적 유전율 κ,
탄성계수, 밀도, 부피 등)으로부터 SRAM/DRAM/NAND·Flash 및 차세대 메모리
(FeRAM/FeFET, RRAM) 소재 스크리닝에 쓰이는 2차 지표를 계산한다.
모든 계산은 외부 데이터 없이 자체적으로 수행된다.
"""
from __future__ import annotations

import numpy as np
from scipy import constants

# 물리 상수 (SI)
K_B = constants.k          # J/K
Q_E = constants.e          # C
H_PL = constants.h         # J·s

# 기준 유전체 (SiO2) — high-k 지표 정규화 기준
EPS_SIO2 = 3.9            # SiO2 정적 유전율
EG_SIO2 = 9.0            # SiO2 밴드갭 (eV)
EG_SI = 1.12            # Si 밴드갭 (eV) — 밴드오프셋 적정성 판단 기준

# DFT(PBE-GGA) 밴드갭은 실제보다 체계적으로 작게 나온다(보통 30~50% 과소평가).
# κ·Eg FOM·EOT·누설 판단이 낙관적으로 치우치는 것을 막기 위한 경험적 scissor
# 계수(산화물 평균치 ~1.5). 정밀값은 물질별로 다르며 HSE/실험값이 정답이다.
PBE_GAP_SCISSOR = 1.5

# DFPT 정적 유전율 신뢰 상한.
# Materials Project의 정적 유전율은 격자 불안정(소프트 포논) 근처에서 발산해
# κ가 수백~수천에 이르는 비물리적 값이 나올 수 있다(예: 강유전 불안정 직전).
# 이런 값은 실제 게이트/커패시터 유전체로 활용 불가하므로 스크리닝에서는
# 이 상한을 넘는 κ를 '신뢰 불가'로 표시해 순위·차트에서 배제한다.
# 상한은 알려진 최고 실용 유전체(TiO2 κ≈80, 페로브스카이트 일부 ~200)를
# 여유 있게 포함하도록 잡았다.
KAPPA_MAX_RELIABLE = 200.0

# 잘 알려진 게이트 유전체 (κ, Eg[eV]) — κ–Eg 트레이드오프 차트 기준점
REFERENCE_DIELECTRICS = [
    ("SiO2", 3.9, 9.0),
    ("Si3N4", 7.0, 5.3),
    ("Al2O3", 9.0, 8.8),
    ("Y2O3", 15.0, 6.0),
    ("HfO2", 25.0, 5.8),
    ("ZrO2", 25.0, 5.8),
    ("Ta2O5", 25.0, 4.4),
    ("La2O3", 30.0, 6.0),
    ("TiO2", 80.0, 3.2),
]

# 전이금속(+란타넘) — RRAM/강유전 산화물 화학계 판별용
TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "La",
}

# ALD(원자층증착) 전구체가 문헌상 잘 확립된 양이온(금속/준금속) 원소.
# 반도체 박막 공정에서 ALD로 실증된 산화물·질화물의 금속 성분을 모았다.
# (Al2O3·HfO2·ZrO2·TiO2·Ta2O5·La2O3·Y2O3·ZnO·TiN·TaN·AlN 등)
ALD_CATIONS = {
    "Al", "Si", "Ti", "Zr", "Hf", "Ta", "Nb", "V", "W", "Mo",
    "Zn", "Sn", "Ga", "In", "Ge", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Y", "La", "Ce", "Pr", "Nd", "Gd", "Dy", "Er", "Yb", "Lu",
    "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Ru", "Rh", "Pd", "Pt", "Ir",
    "B", "Sb", "Bi",
}

# ALD 전구체 지식 베이스 (양이온 → (대표 전구체, T_lo°C, T_hi°C, GPC[Å/cycle])).
# 값은 표준 ALD 문헌 및 Atomic Limits ALD Database(DOI:10.6100/ALDDatabase,
# https://www.atomiclimits.com/alddatabase/)의 범위와 교차 검증한 대표값이다. 1차 공정
# 설계용이며 실제 GPC·온도창은 전구체·장비·기판에 따라 보정이 필요하다(공정별 정밀값은
# 위 DB 참조). 공반응물은 막종(산화물=O계열/질화물=N계열)으로 결정.
ALD_PRECURSORS = {
    "Hf": ("TEMAHf / TDMAHf", 200, 300, 1.0),
    "Zr": ("TEMAZr / TDMAZr", 200, 300, 1.0),
    "Al": ("TMA (트리메틸알루미늄)", 150, 300, 1.1),
    "Ti": ("TiCl₄ / TDMAT", 150, 300, 0.5),
    "Ta": ("PDMAT / Ta(OEt)₅", 200, 300, 0.5),
    "La": ("La(iPrAMD)₃", 200, 300, 0.4),
    "Y":  ("Y(iPrCp)₃ / Y(thd)₃", 250, 350, 0.4),
    "Si": ("BDEAS / 3DMAS", 200, 400, 0.8),
    "Zn": ("DEZ (디에틸아연)", 120, 200, 1.8),
    "Sn": ("TDMASn", 150, 250, 0.5),
    "Nb": ("Nb(OEt)₅ / NbCl₅", 200, 300, 0.4),
    "V":  ("VO(OiPr)₃ / VCl₄", 150, 250, 0.4),
    "Mo": ("MoF₆ / Mo(thd) 계열", 200, 300, 0.4),
    "W":  ("WF₆ / W(CO)₆", 200, 350, 0.4),
    "Sr": ("Sr(iPr₃Cp)₂", 200, 300, 0.6),
    "Ba": ("Ba(iPr₃Cp)₂", 250, 350, 0.5),
    "Ga": ("TMGa (트리메틸갈륨)", 150, 300, 0.5),
    "In": ("InCp / TMIn", 150, 300, 0.5),
    "Mg": ("Mg(EtCp)₂", 200, 300, 1.0),
    "Ce": ("Ce(iPrCp)₃ / Ce(thd)₄", 200, 300, 0.4),
    "Gd": ("Gd(thd)₃ / Gd(iPrCp)₃", 200, 300, 0.4),
    "Ni": ("NiCp₂ / Ni 아미디네이트", 200, 300, 0.4),
    "Co": ("CoCp₂ / Co 아미디네이트", 200, 300, 0.4),
    "Fe": ("FeCp₂ / Fe(thd)₃", 200, 350, 0.3),
    "Mn": ("Mn(EtCp)₂", 150, 300, 0.4),
    "Cr": ("CrCp₂ / CrO₂Cl₂", 200, 350, 0.3),
    "Cu": ("Cu 아미디네이트 / Cu(hfac)₂", 150, 250, 0.3),
    "Ru": ("Ru(EtCp)₂", 200, 350, 0.5),
    "Pt": ("MeCpPtMe₃", 200, 300, 0.5),
    "Ir": ("Ir(acac)₃ / Ir(EtCp)(COD)", 200, 300, 0.4),
    "Bi": ("BiPh₃ / Bi(OtBu)₃", 150, 300, 0.4),
}

# 극성(polar) 점군 — 강유전성의 결정학적 필요조건. 자발 분극이 가능한 10개 점군.
# 강유전체는 반드시 이 중 하나이지만 충분조건은 아니다(스위칭 가능한 준안정 극성
# 상이어야 함; 예: HfO2 강유전상은 orthorhombic Pca2₁ = 공간군 #29, 점군 mm2).
POLAR_POINT_GROUPS = {"1", "2", "m", "mm2", "4", "4mm", "3", "3m", "6", "6mm"}

# 원자번호 순 원소 기호 — 포함/제외 원소 선택 UI(검색 가능 multiselect)용.
ELEMENT_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]


def _pos(*vals):
    """모든 값이 유한하고 양수인지 검사."""
    for v in vals:
        if v is None or not np.isfinite(v) or v <= 0:
            return False
    return True


def corrected_band_gap(eg_ev, scissor=PBE_GAP_SCISSOR):
    """PBE 과소평가를 보정한 추정 밴드갭(eV) = eg×scissor(경험적).

    물질별로 정확도가 다르므로 1차 추정치로만 쓰고 원시 PBE 값과 함께 제시한다.
    """
    if eg_ev is None or not np.isfinite(eg_ev) or eg_ev <= 0:
        return None
    return float(eg_ev) * float(scissor)


def is_polar(point_group):
    """점군이 극성(polar)인지 — 강유전 후보의 1차 결정학적 선별 기준."""
    return str(point_group) in POLAR_POINT_GROUPS


# ──────────────────────────────────────────────────────────────────────────
# 1. 고유전율(high-k) 게이트/커패시터 지표
# ──────────────────────────────────────────────────────────────────────────
def kappa_reliable(kappa):
    """DFPT κ가 스크리닝에 신뢰할 만한 범위(유한·양수·상한 이하)인지 여부.

    상한(KAPPA_MAX_RELIABLE)을 넘는 값은 격자 불안정 근처의 발산 아티팩트로
    보고 κ·Eg·EOT 등 파생 지표 산정과 순위에서 배제하는 데 쓴다.
    """
    return bool(_pos(kappa)) and float(kappa) <= KAPPA_MAX_RELIABLE


def highk_fom(kappa, eg_ev):
    """High-k 품질지수 κ·Eg를 SiO2(=1) 대비 상대값으로.

    유전율 κ가 클수록 커패시턴스(전하저장)에 유리하지만, 경험적으로 κ가
    커지면 밴드갭이 작아져 누설이 증가한다(Robertson tradeoff). κ·Eg는 이
    상충을 한 지표로 묶은 대표적 품질지수다.
    """
    if not _pos(kappa, eg_ev):
        return None
    return float(kappa * eg_ev) / (EPS_SIO2 * EG_SIO2)


def eot(kappa, t_phys_nm=5.0):
    """주어진 물리두께(nm)에 대한 등가산화막두께 EOT (nm).

    EOT = t_phys · (κ_SiO2 / κ). 같은 물리두께라도 κ가 클수록 EOT가 작아져
    더 얇은 SiO2와 동등한 커패시턴스를 낸다(미세화에 유리, 작을수록 좋음).
    """
    if not _pos(kappa) or t_phys_nm <= 0:
        return None
    return float(t_phys_nm) * EPS_SIO2 / float(kappa)


def ald_synthesizable(elements, allow_nitride=True):
    """ALD(원자층증착) 박막 합성이 유망한지에 대한 휴리스틱 판정.

    Materials Project에는 합성법 정보가 없으므로 다음 경험 규칙으로 1차 선별한다
    (모두 충족해야 True):
      1) 산화물(O 포함) 또는 질화물(N 포함) — ALD 박막의 주류 화학.
      2) O/N을 뺀 모든 양이온이 ALD 전구체가 확립된 ALD_CATIONS에 속함.
      3) 단순 조성(이원~삼원계, 고유 원소 ≤3) — 사원계 이상 ALD는 드물다.

    실제 ALD 가능 여부는 전구체·온도창·기판에 따라 달라지므로 후보 선별용
    휴리스틱일 뿐 합성 보장이 아니다. 금속·칼코겐 ALD도 존재하나, 본 메모리
    유전체 플랫폼의 초점에 맞춰 산화물·질화물만 ALD 후보로 본다.
    """
    els = set(elements or [])
    anions = {"O", "N"} if allow_nitride else {"O"}
    if not (els & anions):                 # 산화물/질화물이 아니면 제외
        return False
    cations = els - {"O", "N"}
    if not cations:                        # 순수 O/N 화합물 등 제외
        return False
    if not cations.issubset(ALD_CATIONS):  # 미확립 양이온이 있으면 제외
        return False
    return len(els) <= 3                    # 이원~삼원계까지만


def ald_recipe(elements):
    """조성에 대한 ALD 공정 레시피(1차 추정)를 만든다.

    각 양이온의 대표 전구체, 공반응물(산화물=O계열/질화물=N계열), 권장 증착
    온도창(양이온 온도창의 교집합), 평균 GPC(Å/cycle)를 모은다. 미등록 양이온이
    있으면 그 목록도 반환한다. 다성분(이원계 이상)은 슈퍼사이클이 필요하다.
    산화물/질화물이 아니거나 양이온이 없으면 None.
    """
    els = list(elements or [])
    is_ox, is_nit = ("O" in els), ("N" in els)
    cations = [e for e in els if e not in ("O", "N")]
    if not cations or not (is_ox or is_nit):
        return None
    rows, gpcs, lows, highs, missing = [], [], [], [], []
    for c in cations:
        info = ALD_PRECURSORS.get(c)
        if info:
            prec, lo, hi, gpc = info
            rows.append({"cation": c, "precursor": prec})
            gpcs.append(gpc); lows.append(lo); highs.append(hi)
        else:
            rows.append({"cation": c, "precursor": "전구체 미등록 (문헌 확인 필요)"})
            missing.append(c)
    co_reactant = ("NH₃ 또는 N₂/H₂ 플라즈마" if (is_nit and not is_ox)
                   else "H₂O 또는 O₃ (필요 시 O₂ 플라즈마)")
    if lows and highs:
        t_lo, t_hi = max(lows), min(highs)
        overlap = t_lo <= t_hi
        if not overlap:                       # 양이온별 창이 겹치지 않음
            t_lo, t_hi = min(lows), max(highs)
    else:
        t_lo = t_hi = None
        overlap = None
    return {
        "precursors": rows,
        "co_reactant": co_reactant,
        "t_lo": t_lo, "t_hi": t_hi, "t_overlap": overlap,
        "gpc": (sum(gpcs) / len(gpcs)) if gpcs else None,
        "supercycle": len(cations) > 1,
        "missing": missing,
        "film": "질화물" if (is_nit and not is_ox) else "산화물",
    }


def ald_cycles(gpc_a, thickness_nm):
    """목표 물리두께(nm)를 얻기 위한 ALD 사이클 수 = 두께/GPC (nm→Å 환산)."""
    if not _pos(gpc_a, thickness_nm):
        return None
    return int(round(thickness_nm * 10.0 / gpc_a))


def thickness_for_eot(eot_nm, kappa):
    """목표 EOT(nm)를 만족하는 high-k 물리두께(nm) = EOT·κ/κ_SiO2."""
    if not _pos(kappa, eot_nm):
        return None
    return float(eot_nm) * float(kappa) / EPS_SIO2


# ──────────────────────────────────────────────────────────────────────────
# 1b. 게이트 스택 소자 1차 예측 (TCAD 연동 방향 — illustrative)
#   스크리닝/공정 파라미터(κ·Eg·EOT)에서 커패시턴스·누설·밴드정렬을 해석적으로
#   추정한다. 풀 TCAD(Sentaurus/Silvaco)로 가는 입력·방향을 보여주는 예시 모델.
# ──────────────────────────────────────────────────────────────────────────
_EPS0_F_PER_CM = 8.854e-14   # 진공 유전율 (F/cm)


def gate_cap_density(eot_nm):
    """게이트 단위면적 커패시턴스 C_ox (µF/cm²) = ε0·κ_SiO2 / EOT. (정확)"""
    if not _pos(eot_nm):
        return None
    return _EPS0_F_PER_CM * EPS_SIO2 / (eot_nm * 1e-7) * 1e6


def band_offsets(eg_ev):
    """유전체–Si 밴드오프셋(ΔEc, ΔEv) 1차 추정(illustrative).

    Eg 차이를 전도대/가전자대에 ~60/40으로 배분(산화물 경험칙). 실제 오프셋은
    전자친화도·계면 쌍극자에 좌우되므로 정밀값은 측정/계산이 필요하다.
    """
    if eg_ev is None or not np.isfinite(eg_ev):
        return None
    delta = max(0.0, float(eg_ev) - EG_SI)
    d_ec = 0.6 * delta
    return d_ec, delta - d_ec


def gate_leakage(t_phys_nm, barrier_ev, m_eff=0.4, j0=1.0e6):
    """게이트 직접터널링 누설 J(A/cm²) 1차 추정 = J0·exp(−2κt) (illustrative).

    κ=√(2 m* q Φb)/ħ. 고-κ가 같은 EOT에서 더 두꺼운 물리두께로 누설을 낮추는
    '경향'을 보여주는 해석 모델로, 절대값은 보정용 J0에 의존(정밀 TCAD 아님).
    """
    if not _pos(t_phys_nm, barrier_ev):
        return None
    m0, q, hbar = 9.109e-31, 1.602e-19, 1.055e-34
    kap = np.sqrt(2.0 * m_eff * m0 * q * float(barrier_ev)) / hbar     # 1/m
    return float(j0 * np.exp(-2.0 * kap * (t_phys_nm * 1e-9)))


# ──────────────────────────────────────────────────────────────────────────
# 2. 탄성/열적 물성 (박막 열 안정성 보조 지표; B=부피, G=전단 탄성률)
# ──────────────────────────────────────────────────────────────────────────
def pugh_ratio(bulk_gpa, shear_gpa):
    """B/G. >1.75 연성(ductile), <1.75 취성(brittle)."""
    if not _pos(bulk_gpa, shear_gpa):
        return None
    return bulk_gpa / shear_gpa


def vickers_hardness(bulk_gpa, shear_gpa):
    """Chen(2011) 모델 Vickers 경도 Hv = 2(k²G)^0.585 − 3 (GPa), k=G/B."""
    if not _pos(bulk_gpa, shear_gpa):
        return None
    k = shear_gpa / bulk_gpa
    hv = 2.0 * (k ** 2 * shear_gpa) ** 0.585 - 3.0
    return max(0.0, hv)


def debye_temperature(bulk_gpa, shear_gpa, density_gcm3, nsites, volume_a3):
    """탄성파 속도로부터 Debye 온도 θ_D (K). 열적 안정성·열전도 잠재력 지표."""
    if not _pos(bulk_gpa, shear_gpa, density_gcm3, nsites, volume_a3):
        return None
    b = bulk_gpa * 1e9
    g = shear_gpa * 1e9
    rho = density_gcm3 * 1000.0                          # kg/m³
    if (b + 4.0 / 3.0 * g) <= 0:
        return None
    v_l = np.sqrt((b + 4.0 / 3.0 * g) / rho)            # 종파
    v_t = np.sqrt(g / rho)                              # 횡파
    v_m = (1.0 / 3.0 * (1.0 / v_l ** 3 + 2.0 / v_t ** 3)) ** (-1.0 / 3.0)
    n_density = nsites / (volume_a3 * 1e-30)            # 원자수밀도 (1/m³)
    theta = (H_PL / K_B) * (3.0 * n_density / (4.0 * np.pi)) ** (1.0 / 3.0) * v_m
    return float(theta)


# ──────────────────────────────────────────────────────────────────────────
# 3. 메모리 소재 종합 발굴 점수 (0~100)
# ──────────────────────────────────────────────────────────────────────────
def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def _sat(x, x0):
    """포화 곡선 0→0, ∞→1 (x0에서 ≈0.63). 계단 없이 큰 값일수록 완만히 가산."""
    return float(1.0 - np.exp(-max(0.0, x) / x0)) if x0 > 0 else 0.0


def _bell(x, center, width):
    """가우시안 종형(중심 1.0). '적정 구간'을 절벽 없이 선호하게 한다."""
    return float(np.exp(-((float(x) - center) / width) ** 2))


# ── 발굴점수 4축 부분점수(각 0~1, 연속) ─────────────────────────────────────
def _stability_subscore(eah):
    """안정성: e_above_hull이 0에 가까울수록 1 (지수 감쇠, 0.03 eV에서 ≈0.37)."""
    eah = eah if eah is not None else 0.5
    return float(np.exp(-max(0.0, float(eah)) / 0.03))


def _manufacturability_subscore(experimental, is_ald):
    """상용성: DFT 안정 등재(기본 0.20) + 실험적 합성(+0.50) + ALD 가능(+0.30)."""
    return 0.20 + (0.50 if experimental else 0.0) + (0.30 if is_ald else 0.0)


def _durability_subscore(debye, hardness):
    """내구성(박막 열·기계 안정성): Debye 온도·Vickers 경도 정규화 가중 평균.

    탄성 데이터가 없으면 중립값 0.5(순위에 유불리 없음).
    """
    parts, wts = [], []
    if _pos(debye):
        parts.append(_clip01(float(debye) / 600.0)); wts.append(0.6)
    if _pos(hardness):
        parts.append(_clip01(float(hardness) / 20.0)); wts.append(0.4)
    if not parts:
        return 0.5
    return float(sum(p * w for p, w in zip(parts, wts)) / sum(wts))


def _performance_subscore(application, *, eg, eg_corr, kappa,
                          is_oxide, has_hf_zr, is_tm_oxide, is_polar):
    """응용별 성능 적합성(0~1, 연속). 모두 부드러운 포화/종형 함수로 변별력 확보."""
    egc = float(eg_corr) if (eg_corr and eg_corr > 0) else (
        float(eg) * PBE_GAP_SCISSOR if eg else 0.0)        # 보정 Eg(밴드오프셋용)
    k = float(kappa) if _pos(kappa) else 0.0
    fom_rel = (k * eg) / (EPS_SIO2 * EG_SIO2) if (k and eg) else 0.0   # κ·Eg (SiO2=1)
    fom_n = _sat(fom_rel, 3.0)                              # 포화: HfO2≈0.75
    if application in ("highk", "general"):
        offset_n = _clip01((egc - 3.0) / 4.0)              # 누설 억제 밴드오프셋 여유
        return 0.6 * fom_n + 0.4 * offset_n
    if application == "dram_cap":
        offset_n = _clip01((egc - 2.0) / 4.0)              # 커패시터: κ 비중↑
        return 0.7 * fom_n + 0.3 * offset_n
    if application == "nand_oxide":
        wide = _clip01((egc - 4.0) / 5.0)                  # 블로킹/전하트랩: 넓은 Eg
        return 0.75 * wide + 0.25 * fom_n
    if application == "ferroelectric":
        s = (0.25 if is_oxide else 0.0) + (0.35 if has_hf_zr else 0.0) \
            + (0.30 if is_polar else 0.0)
        s += 0.10 * _bell(egc, 5.8, 1.6)                   # HfO2계 강유전 적정 Eg창
        return _clip01(s)
    if application == "rram":
        gate = 1.0 if is_tm_oxide else (0.3 if is_oxide else 0.0)
        window = _bell(eg, 3.5, 1.6)                       # 스위칭 적정 Eg(PBE 기준)
        return gate * (0.4 + 0.6 * window)
    return 0.0


# 응용별 4축 가중치(합 100). 상용성 비중을 높게 유지해 실제 소재가 상위에 오게 한다.
_SCORE_WEIGHTS = {
    "highk":         dict(stability=22, manufacturability=28, durability=12, performance=38),
    "general":       dict(stability=22, manufacturability=28, durability=12, performance=38),
    "dram_cap":      dict(stability=20, manufacturability=28, durability=12, performance=40),
    "nand_oxide":    dict(stability=22, manufacturability=28, durability=15, performance=35),
    "ferroelectric": dict(stability=22, manufacturability=28, durability=10, performance=40),
    "rram":          dict(stability=22, manufacturability=28, durability=10, performance=40),
}


def score_components(*, band_gap, e_above_hull, kappa=None,
                     is_oxide=False, has_hf_zr=False, is_tm_oxide=False,
                     is_ald=False, experimental=False, is_polar=False,
                     band_gap_corr=None, debye=None, hardness=None,
                     application="general"):
    """발굴점수의 4축 구성요소(가중 적용 점수)와 합계를 dict로 반환.

    축: 안정성(hull) · 상용성(실험/ALD) · 내구성(Debye·경도) · 응용성능(κ·Eg·Eg창 등).
    각 축은 0~1 연속 부분점수에 응용별 가중치(_SCORE_WEIGHTS, 합 100)를 곱한다.
    이 분해를 그대로 노출하면 '왜 이 점수인지'를 설명할 수 있다.
    """
    if band_gap is None:
        return dict(stability=0.0, manufacturability=0.0,
                    durability=0.0, performance=0.0, total=0.0)
    w = _SCORE_WEIGHTS.get(application, _SCORE_WEIGHTS["general"])
    sub = dict(
        stability=_clip01(_stability_subscore(e_above_hull)),
        manufacturability=_clip01(_manufacturability_subscore(experimental, is_ald)),
        durability=_clip01(_durability_subscore(debye, hardness)),
        performance=_clip01(_performance_subscore(
            application, eg=float(band_gap), eg_corr=band_gap_corr, kappa=kappa,
            is_oxide=is_oxide, has_hf_zr=has_hf_zr,
            is_tm_oxide=is_tm_oxide, is_polar=is_polar)),
    )
    pts = {k: round(w[k] * sub[k], 1) for k in w}
    pts["total"] = round(min(100.0, sum(pts.values())), 1)
    return pts


def memory_score(*, band_gap, e_above_hull, kappa=None,
                 is_oxide=False, has_hf_zr=False, is_tm_oxide=False,
                 is_ald=False, experimental=False, is_polar=False,
                 band_gap_corr=None, debye=None, hardness=None,
                 application="general"):
    """메모리 응용별 종합 발굴 점수(0~100) = score_components(...)["total"]."""
    return score_components(
        band_gap=band_gap, e_above_hull=e_above_hull, kappa=kappa,
        is_oxide=is_oxide, has_hf_zr=has_hf_zr, is_tm_oxide=is_tm_oxide,
        is_ald=is_ald, experimental=experimental, is_polar=is_polar,
        band_gap_corr=band_gap_corr, debye=debye, hardness=hardness,
        application=application)["total"]
