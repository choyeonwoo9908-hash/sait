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


def memory_score(*, band_gap, e_above_hull, kappa=None,
                 is_oxide=False, has_hf_zr=False, is_tm_oxide=False,
                 application="general"):
    """메모리 응용별 휴리스틱 발굴 점수 (0~100).

    공통: 열역학적 안정성(hull) 0~40점.
    응용별: high-k/DRAM은 κ·Eg와 밴드오프셋 적정성, NAND 산화물은 넓은 Eg
    (배리어), 강유전체는 Hf/Zr 산화물 적합성, RRAM은 전이금속 산화물·스위칭
    적합 Eg를 평가한다. 화학·데이터가 맞지 않아도 안정성 점수는 유지된다.
    """
    if band_gap is None:
        return 0.0
    eg = float(band_gap)

    # 안정성 (0~40): hull=0 → 40, ≥0.1 eV/atom → 0
    eah = e_above_hull if e_above_hull is not None else 0.5
    stab = 40.0 * _clip01(1.0 - min(eah, 0.1) / 0.1)

    k = float(kappa) if _pos(kappa) else None
    fom = (k * eg) if k else 0.0          # κ·Eg (절대값)

    app_pts = 0.0
    bonus = 0.0

    if application in ("highk", "dram_cap", "general"):
        # κ·Eg가 크되, 누설 억제를 위한 밴드오프셋(≈넓은 Eg)도 갖춰야 함
        offset = _clip01((eg - (2.0 if application == "dram_cap" else 3.0)) / 2.0)
        app_pts = 45.0 * _clip01(fom / 250.0) * offset
        bonus = 15.0 * _clip01((k or 0.0) / 40.0)            # 고-κ 가산
    elif application == "nand_oxide":
        # 터널/블로킹 산화물: 넓은 밴드갭(높은 배리어)이 핵심
        app_pts = 45.0 * _clip01((eg - 3.0) / 5.0)
        bonus = 10.0 * _clip01((k or 0.0) / 15.0) if is_oxide else 0.0
    elif application == "ferroelectric":
        # HfO2/ZrO2 계열 산화물 우대 (강유전 orthorhombic 상 형성 모재)
        if is_oxide and has_hf_zr:
            app_pts = 45.0
        elif is_oxide:
            app_pts = 12.0
        bonus = 15.0 * _clip01((k or 0.0) / 30.0)
    elif application == "rram":
        # 산소공공 기반 저항변화: 전이금속 산화물·스위칭 적합 Eg(≈2~6 eV)
        if is_tm_oxide and 2.0 <= eg <= 6.0:
            app_pts = 45.0
        elif is_tm_oxide:
            app_pts = 20.0
        bonus = 10.0 * _clip01((k or 0.0) / 30.0)

    return round(min(100.0, stab + app_pts + bonus), 1)
