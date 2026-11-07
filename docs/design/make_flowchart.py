"""PyESD 한반도 적용 설계도(흐름도) 생성 — 전체도 1장 + 단계별 상세도 7장.

  pip install matplotlib koreanize-matplotlib
  python docs/design/make_flowchart.py          # docs/design/*.png, *.svg

이 저장소에 실제 구현된 두 체계(EC46 주별 1개월, SEAS5 월별 3개월)를 그대로 그린다.
상자마다 입력·출력 자료, 해상도, 담당 코드 모듈을 적는다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flowlib import C, INK2, MUTED, canvas, node, edge, band, step_header, legend, footer, save, NODES

STEPS = ["자료 입수", "전처리", "예측인자 생성", "전이함수 학습·검증", "모델 투영", "앙상블 통계·검증", "게시"]
FOOT = "nalssi89/downscale_by_pyesd · docs/design/make_flowchart.py"


# ───────────────────────────── 전체도 ─────────────────────────────
def overview():
    W, H = 26.2, 13.55
    fig, ax = canvas(W, H, "PyESD 한반도 적용 전체 흐름도",
                     "Perfect-Prognosis 경험적 통계 다운스케일링 · 예측대상 0.25° 412격자(북한 포함) + ASOS 66지점 · "
                     "두 체계 분리 운영: EC46 주별 1개월(매일) / SEAS5 월별 3개월(매월 14일)")
    X0, CW, G = 0.45, 3.28, 0.32
    cx = [X0 + i * (CW + G) for i in range(7)]
    for i, s in enumerate(STEPS):
        step_header(ax, cx[i], 1.12, CW, i + 1, s, color="#33424d")

    # 레인 바탕
    band(ax, 0.25, 1.72, W - 0.5, 2.25, "공통 · 예측대상", C["common"], "ERA5 0.25°·ASOS 관측 → 편차, 두 체계가 함께 사용")
    band(ax, 0.25, 4.15, W - 0.5, 3.45, "① EC46 주별 1개월 체계", C["ec46"], "ec46_weekly/ · Open-Meteo 지상 4변수 · 능형회귀")
    band(ax, 0.25, 7.8, W - 0.5, 4.05, "② SEAS5 월별 3개월 체계", C["seas5"], "seas5_monthly/ · CDS 모델 자료 · PyESD 원본형 ARD")

    # 공통 레인
    c = C["common"]
    node(ax, "t1", cx[0], 2.1, CW, "관측·재분석 입수", [
        "ERA5 0.25° 매시 → KST 일자료 (ARCO Zarr)",
        "  33–43.5°N, 124–132°E, 1991~2026",
        "ASOS 66지점 일자료·일평년 (기상청 API허브)",
        "  내륙 62 + 제주 4 (키는 환경변수)"],
        mod="training/get_era5_kor.py · get_asos*.py", color=c)
    node(ax, "t2", cx[1], 2.1, CW, "예측대상 구성", [
        "격자: 남·북한 경계 폴리곤 육지비율 ≥ 20 %",
        "  → 0.25° 412격자 (신의주 등 해안 포함)",
        "지점: ASOS 66 (남한 평균은 제주 제외 62)",
        "조화함수 4항 기후값(1991–2020) 제거, 강수 √"],
        mod="training/prep.py → target_grid/asos.nc", color=c)
    node(ax, "tx", cx[2], 2.1, CW, "확장 후보 (미적용)", [
        "300번 이하 기준망 ~85지점 (간격 ~26 km)",
        "AWS 보강망: 이웃 ASOS 계수 쪽 수축 회귀",
        "5 km 격자: ASOS·AWS 내삽 목표자료",
        "→ 학습 대상만 바뀌고 흐름은 동일"], kind="ext")
    edge(ax, "t1", "t2", color=c)
    edge(ax, "t2", "tx", color=C["ext"], ls=(0, (3, 2)))

    # EC46 레인
    c = C["ec46"]
    y = 4.6
    node(ax, "e1", cx[0], y, CW, "입수", [
        "학습: ERA5 1.5° 247점 일자료 (ARCO)",
        "  28–46°N, 114–141°E · t2m·d2m·msl·tp",
        "예보: Open-Meteo EC46 51멤버 일자료",
        "  같은 247점 좌표 · 46일 · 매일"],
        mod="get_era5_pred.py · openmeteo.py", color=c)
    node(ax, "e2", cx[1], y, CW, "편차·보정", [
        "일 → 7일(주)·달력월 평균",
        "ERA5 조화 기후값 기준 편차, 강수 √",
        "EC46 계통오차: 모델 평년 역산 후 ERA5로",
        "  (기온·이슬점·기압 가법, 강수 승법)"],
        mod="pyesd/esd.py · training/prep.py → pred.nc", color=c)
    node(ax, "e3", cx[2], y, CW, "예측인자", [
        "변수별 EOF PC (주 6개 · 월 3개)",
        "  표준화 편차장 SVD, 4변수",
        "대상점별 반경 250 km 가우스 평균 4개",
        "→ 설계행렬 주 28열 · 월 16열"],
        mod="esd.EOFPredictors · radius_weights", color=c)
    node(ax, "e4", cx[3], y, CW, "능형회귀 학습", [
        "점별 · 달별(±1개월 창) · 열 표준화",
        "α: 연도 5묶음 CV (1·10·100·1000)",
        "검증: leave-one-year-out + 독립 2023–26",
        "CV 상관 격자 T 0.97/0.95 · P 0.92/0.90"],
        mod="training/train.py → export_models.py", color=c)
    node(ax, "e5", cx[4], y, CW, "멤버별 투영", [
        "EC46 51멤버 × 247점 편차",
        "→ 학습 EOF 로딩에 투영 + 지역평균",
        "→ 412격자·66지점 멤버별 편차",
        "1~4주 + 달력월"],
        mod="ec46_weekly/forecast.py", color=c)
    node(ax, "e6", cx[5], y, CW, "확률 산출", [
        "멤버 μk + CV 잔차 σ 정규혼합",
        "3분위·중앙값 초과·극한(p10/p90)",
        "임계값 확률, p10~p90 예측범위",
        "남한 62지점 평균 south_mean"],
        mod="pyesd/products.py → downscaling.json", color=c)
    node(ax, "m4", cx[3], 6.5, CW, "학습 모형", ["models/{grid,asos}_{T,P}_{week,month}.npz",
                                                   "달별 EOF·계수·σ·분위 (numpy만으로 운영)"],
         kind="data", color=c)
    for a, b in [("e1", "e2"), ("e2", "e3"), ("e3", "e4"), ("e5", "e6")]:
        edge(ax, a, b, color=c)
    edge(ax, "e4", "m4", color=c)
    edge(ax, "m4", "e5", sa="r", sb="b", color=c, route="l")

    # SEAS5 레인
    c = C["seas5"]
    y = 8.25
    node(ax, "s1", cx[0], y, CW, "CDS 입수 (1건씩 순차)", [
        "ERA5 1° 월평균 1991~현재",
        "  지상 7 · 500/850 hPa z·t·q·u·v",
        "  + 200 hPa · 인도양 SST · 북반구 Z500",
        "SEAS5 v51 10월 초기화, 리드 1~4",
        "  재예보 1993–2016 25멤버 · 실황 51멤버"],
        mod="seas5_monthly/get_cds.py · seq_cds.py", color=c)
    node(ax, "s2", cx[1], y, CW, "편차·모형기후 보정", [
        "ERA5: 1991–2020 달별 기후값 편차",
        "SEAS5: 재예보(초기화·리드별) 기후 기준",
        "  + (ERA5 1993–16 − 1991–20) 보정",
        "단위 통일 → ERA5 1° 격자로 보간",
        "기후값 저장 clim_init10.nc"],
        mod="predictors.py era5_anoms · seas5_anoms", color=c)
    node(ax, "s3", cx[2], y, CW, "예측인자 (PyESD 원본형)", [
        "지역평균 12변수 (반경 250 km)",
        "기후지수 7~16 (자료 따라 자동 선택)",
        "  Niño3.4 WNPSH KH OKH KSST WP PJ SH",
        "  + CGT SAH EAJ SRP DMI NAO AO EU",
        "EOF PC 9 (z500·msl·t2m × 3)"],
        mod="predictors.py INDEX_DEFS · PATTERN_DEFS", color=c)
    node(ax, "s4", cx[3], y, CW, "ARD 회귀 학습", [
        "점별 · 달별(±1개월) · 열 표준화",
        "ARDRegression: 무관 인자 계수 → 0",
        "5년 블록 교차검증 1991–2025",
        "  EOF·패턴·회귀 폴드마다 재적합",
        "CV 잔차 σ · 3분위 경계 저장"],
        mod="seas5_monthly/train.py", color=c)
    node(ax, "s5", cx[4], y, CW, "멤버별 투영", [
        "실황 51멤버: 리드 2·3·4 = 다음 3개월",
        "재예보 25멤버 × 24년: 리드 1~4",
        "같은 지수·EOF 정의로 특징표 구성",
        "→ 학습 평균·표준편차로 표준화 → W"],
        mod="seas5_monthly/apply.py · forecast.py", color=c)
    node(ax, "s6", cx[5], y, CW, "앙상블 통계", [
        "멤버 + σ 정규혼합 → 3분위 확률",
        "10·25·50·75·90 백분위 예측범위",
        "임계값·극한 확률, south_mean",
        "격자 412 + ASOS 66"],
        mod="products.py → downscaling.json", color=c)
    node(ax, "s6v", cx[5], 10.4, CW, "재예보 검증 (1993–2016)", [
        "ACC · RPSS(3분위) · 10~90 % 포함률",
        "신뢰도표, 원 SEAS5 지역평균과 비교"],
        mod="hindcast_verify.py → docs/seas5/", color=c)
    node(ax, "m5", cx[3], 10.4, CW, "학습 모형", ["models/{grid,asos}_{T,P}.npz",
                                                    "EOF·패턴·희소 계수 W·σ·분위"],
         kind="data", color=c)
    for a, b in [("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s5", "s6")]:
        edge(ax, a, b, color=c)
    edge(ax, "s4", "m5", color=c)
    edge(ax, "m5", "s5", sa="r", sb="b", color=c, route="l", tb=0.75)
    edge(ax, "s5", "s6v", sa="b", sb="l", route="l", color=c, ta=0.2, label="재예보")

    # 공통 → 학습
    edge(ax, "t2", "e4", sa="b", sb="t", route="v", mid=4.4, color=C["common"], ta=0.75, label="예측대상 편차")
    edge(ax, "t2", "s4", sa="r", sb="t", route="h", mid=cx[3] - 0.17, color=C["common"], ta=0.88, tb=0.12)

    # 게시 (두 레인에 걸침)
    c = C["pub"]
    node(ax, "p1", cx[6], 4.6, CW, "EC46 자동 실행", ["GitHub Actions 매일 09:30 UTC",
                                                       "→ out-ec46 브랜치"], mod="ec46_weekly.yml", color=c)
    node(ax, "p2", cx[6], 8.25, CW, "SEAS5 자동 실행", ["매월 14일 03:00 UTC (공개 다음 날)",
                                                        "실황만 CDS에서 받음 (secret CDSAPI_KEY)",
                                                        "→ out-seas5 브랜치"], mod="seas5_monthly.yml", color=c)
    node(ax, "p3", cx[6], 6.12, CW, "사이트 자료 변환·게시", ["grid.json · map_1m/3m.json",
                                                            "stations.json · points/ · stations/",
                                                            "→ gh-pages (Leaflet 지도)"],
         mod="site.yml · site/build_site_data.py", color=c)
    edge(ax, "e6", "p1", color=C["ec46"])
    edge(ax, "s6", "p2", color=C["seas5"])
    edge(ax, "p1", "p3", color=c)
    edge(ax, "p2", "p3", sa="t", sb="b", color=c)
    node(ax, "web", cx[6], 10.4, CW, "nalssi89.github.io/downscale_by_pyesd", [
        "objective_climate_outlook 사이트 형식",
        "영역(남한·한반도) · 1개월/3개월 · 요소",
        "기간 · 3분위·평년편차·임계값·스프레드"], kind="note", color=c, title_fs=8.8)
    edge(ax, "p3", "web", sa="l", sb="l", route="h", mid=cx[6] - 0.16, color=c, ta=0.85, tb=0.2)

    # 규칙 상자
    node(ax, "rule", 0.45, 12.1, 10.6, "자료 입수 규칙", [
        "CDS 요청은 계정 전체에서 1건씩 — 완료·수신 후 다음 요청. 거절 건 재전송 금지, 받은 파일은 건너뜀, 대기 건 취소 금지",
        "ERA5 시간·일자료는 ARCO Zarr 우선, 월평균·SEAS5만 CDS. SEAS5는 넓은 영역으로 받아 잘라 저장. 이어받기 가능하게 실행"],
        kind="note", color=MUTED)
    node(ax, "excl", 11.4, 12.1, 7.1, "제외한 기후인자", [
        "MJO·BSISO: 일 단위 진동이라 월평균 체계에 부적합",
        "DRY(선행 토양건조): 토양수분 자료 미입수"], kind="note", color=MUTED)
    legend(ax, 18.9, 12.4, items=("proc", "data", "note", "ext"))
    footer(ax, W, H, FOOT)
    save(fig, "flow_overview")


# ─────────────────────────── 1. 자료 입수 ───────────────────────────
def s1_data():
    W, H = 20.5, 14.1
    fig, ax = canvas(W, H, "1단계 · 자료 입수", "자료원별 최적 경로: ERA5 시간·일자료는 ARCO Zarr, 월평균·SEAS5는 CDS API(계정 전체 1건씩 순차), "
                     "관측은 기상청 API허브, EC46은 Open-Meteo")
    cs, cm, cp, cd = C["common"], C["ec46"], C["seas5"], C["dec"]
    # 자료원
    xs, ws = 0.4, 3.6
    node(ax, "arco", xs, 1.45, ws, "ECMWF ARCO ERA5 Zarr", ["arco.datastores.ecmwf.int", "geoChunked 0.25° 매시간", "인증 불필요, 직접 읽기"], kind="data", color=cs)
    node(ax, "kma", xs, 4.2, ws, "기상청 API허브", ["kma_sfcdd3 일자료, 일평년값", "KMA_AUTHKEY (환경변수만)"], kind="data", color=cs)
    node(ax, "om", xs, 5.65, ws, "Open-Meteo Seasonal API", ["ECMWF EC46 51멤버 일자료", "지상 4변수만 제공"], kind="data", color=cm)
    node(ax, "cds", xs, 7.45, ws, "C3S Climate Data Store", ["cds.climate.copernicus.eu/api", "CDSAPI_KEY (환경변수만)", "ERA5 월평균 · SEAS5 v51"], kind="data", color=cp)
    # 처리
    xp, wp = 4.7, 4.9
    node(ax, "g_kor", xp, 1.15, wp, "한반도 0.25° 일자료", ["33–43.5°N, 124–132°E, 시간 청크 경계 단위 읽기", "KST(UTC+9) 일평균 기온 · 일강수량(mm)"],
         mod="training/get_era5_kor.py", color=cs)
    node(ax, "g_pred", xp, 2.6, wp - 0.0, "대규모 예측인자 1.5° 점표본", ["28–46°N, 114–141°E (13×19 = 247점)", "t2m·d2m(°C) · msl(hPa) · tp(mm/일)"],
         mod="training/get_era5_pred.py", color=cm)
    node(ax, "g_asos", xp, 4.05, wp, "ASOS 일자료", ["내륙 62 + 제주 4 (184·185·188·189), 1991~2026", "tm2를 최근으로 두고 받아 옛 자료 누락 방지"],
         mod="training/get_asos.py · get_asos_jeju.py", color=cs)
    node(ax, "g_om", xp, 5.5, wp, "EC46 앙상블 요청", ["247점 쉼표 묶음 요청, 429·5xx 지수 백오프", "<변수>, <변수>_member01…50 → (member, time, pt)"],
         mod="ec46_weekly/openmeteo.py", color=cm)
    # CDS 순차 처리 흐름
    node(ax, "q", xp, 7.15, wp, "요청 목록 생성", ["ERA5: 비용 한도(20000)로 연도 구간 분할 → parts/", "SEAS5: 초기화 달 × (재예보 | 실황) × 장 묶음",
                                                  "영역은 넓게: 지상 60N–10N, 60–180E 등"],
         mod="get_cds.py era5_requests · seas5_requests", color=cp)
    node(ax, "d1", xp + 0.85, 8.95, 3.2, "이미 받은 파일?", [], kind="dec", h=0.9)
    node(ax, "d2", xp + 0.85, 10.15, 3.2, "CDS에 같은 요청이", ["대기·성공 상태로 있나?"], kind="dec", h=1.0)
    node(ax, "sub", xp, 11.5, wp, "1건 제출 → 완료까지 대기 → 내려받기", ["계정 전체에서 동시에 1건만 (데이터셋 묶음별 순차)"],
         mod="seq_cds.py run_lane", color=cp)
    node(ax, "skip", 10.25, 8.9, 2.35, "건너뜀", ["다음 요청으로"], color=MUTED)
    node(ax, "resume", 10.25, 10.15, 2.35, "이어받기", ["새로 내지 않음"], color=MUTED)
    node(ax, "d3", 10.15, 11.55, 2.55, "Rejected?", [], kind="dec", h=0.85)
    node(ax, "rej", 10.25, 12.75, 2.35, "재전송 금지", ["조건을 바꿔 보고"], color="#b23b3b")
    edge(ax, "cds", "q", sa="r", sb="l", color=cp)
    edge(ax, "q", "d1", color=cp)
    edge(ax, "d1", "skip", label="예", color=MUTED)
    edge(ax, "d1", "d2", label="아니오", color=cp)
    edge(ax, "d2", "resume", label="예", color=MUTED)
    edge(ax, "d2", "sub", label="아니오", color=cp)
    edge(ax, "sub", "d3", color=cp)
    edge(ax, "d3", "rej", label="예", color="#b23b3b")
    edge(ax, "arco", "g_kor", sa="r", sb="l", ta=0.35, color=cs)
    edge(ax, "arco", "g_pred", sa="r", sb="l", ta=0.8, color=cm)
    edge(ax, "kma", "g_asos", color=cs)
    edge(ax, "om", "g_om", color=cm)
    # 산출 파일
    xo, wo = 13.3, 6.8
    node(ax, "o_kor", xo, 1.15, wo, "kor_YYYY.nc", ["ERA5 0.25° 한반도 일 t2m·tp (→ 예측대상 격자)"], kind="data", color=cs)
    node(ax, "o_pred", xo, 2.4, wo, "pred_daily (247점 × 4변수)", ["EC46 체계 학습용 대규모 예측인자, 1991~2026"], kind="data", color=cm)
    node(ax, "o_asos", xo, 3.65, wo, "asos 일자료 · 일평년값", ["66지점, 지점정보(위경도·고도), 1991–2020 평년"], kind="data", color=cs)
    node(ax, "o_om", xo, 4.9, wo, "EC46 예보 Dataset (51, 46일, 247)", ["매일 GitHub Actions에서 받음 (컨테이너에선 차단)"], kind="data", color=cm)
    node(ax, "o_era5", xo, 6.25, wo, "ERA5 1° 월평균 (CDS_DIR)", [
        "era5_sfc.nc  t2m d2m msl tp u10 v10 sst   60N–10N, 60–180E",
        "era5_pl.nc   z t q u v @500·850           60N–0N, 60–180E",
        "era5_sst_nino34.nc  Niño3.4 상자 SST",
        "확장: era5_pl200.nc (z·u·v 200 hPa) · era5_sst_io.nc (인도양)",
        "      era5_z500_nh.nc (북반구 20N 이북 Z500)"], kind="data", color=cp)
    node(ax, "o_s5", xo, 8.15, wo, "SEAS5 system 51 월평균 (10월 초기화)", [
        "seas5_hind_10_*.nc  재예보 1993–2016 · 25멤버 · 리드 1~4",
        "seas5_fc2026_10_*.nc  실황 · 51멤버 · 리드 1~4",
        "장 묶음: _sfc · _pl · _pl200 · _sst_io · _z500_nh · _sst_nino34",
        "넓게 받은 뒤 필요한 영역으로 잘라 저장"], kind="data", color=cp)
    edge(ax, "g_kor", "o_kor", color=cs); edge(ax, "g_pred", "o_pred", color=cm)
    edge(ax, "g_asos", "o_asos", color=cs, tb=0.5); edge(ax, "g_om", "o_om", color=cm)
    edge(ax, "d3", "o_era5", sa="r", sb="l", label="성공", color=cp, route="h", mid=12.95, tb=0.6, lpos=0.5)
    edge(ax, "d3", "o_s5", sa="r", sb="l", color=cp, route="h", mid=12.95)
    node(ax, "n", xo, 10.6, wo, "운영 시 입수 (매월 14일)", [
        "get_cds.py --only seas5 --no-hind : 실황 파일만 받음",
        "ERA5·재예보 기후값은 models/clim_init{mm}.nc 로 미리 저장돼 있음",
        "다운로드는 재시작 후 이어받기, 장시간 작업은 약 1시간마다 보고"], kind="note", color=MUTED)
    legend(ax, 0.45, 13.65, items=("proc", "data", "dec", "note"))
    footer(ax, W, H, FOOT)
    save(fig, "flow_s1_data")


# ─────────────────────────── 2. 전처리 ───────────────────────────
def s2_prep():
    W, H = 20.5, 9.85
    fig, ax = canvas(W, H, "2단계 · 전처리 (편차·계통오차 보정)",
                     "모든 장을 같은 기준(ERA5 1991–2020 기후)의 편차로 맞춘다 — 학습(ERA5)과 예보(모델) 쪽이 같은 분포를 갖게 하는 Perfect-Prognosis의 전제")
    cs, cm, cp = C["common"], C["ec46"], C["seas5"]
    band(ax, 0.25, 1.15, W - 0.5, 2.0, "예측대상", cs)
    band(ax, 0.25, 3.35, W - 0.5, 2.0, "EC46 예측인자", cm)
    band(ax, 0.25, 5.55, W - 0.5, 3.75, "SEAS5 예측인자", cp)
    y = 1.6
    node(ax, "a0", 0.45, y, 3.0, "입력", ["kor_YYYY.nc (0.25°)", "ASOS 66 일자료"], kind="data", color=cs)
    node(ax, "a1", 3.85, y, 3.4, "육지 마스크", ["korea.geojson 남·북 폴리곤", "셀 9×9 부분표본 육지비율", "≥ 20 % → 412격자"], mod="prep.korea_mask", color=cs)
    node(ax, "a2", 7.65, y, 3.0, "기간 집계", ["일 → 주(7일) · 월", "기온 평균, 강수 합/일평균"], mod="esd.aggregate", color=cs)
    node(ax, "a3", 11.05, y, 2.9, "변환", ["강수 √(mm/일)", "기온 그대로"], mod="esd.transform", color=cs)
    node(ax, "a4", 14.35, y, 2.8, "기후값 제거", ["조화함수 4항 일별 기후", "(1991–2020) → 편차"], mod="esd.harmonic_clim", color=cs)
    node(ax, "a5", 17.55, y, 2.6, "출력", ["target_grid.nc (412)", "target_asos.nc (66)", "jeju 표시 열"], kind="data", color=cs)
    for a, b in [("a0", "a1"), ("a1", "a2"), ("a2", "a3"), ("a3", "a4"), ("a4", "a5")]:
        edge(ax, a, b, color=cs)
    y = 3.8
    node(ax, "b0", 0.45, y, 3.0, "입력", ["ERA5 247점 일자료", "EC46 51멤버 일자료"], kind="data", color=cm)
    node(ax, "b1", 3.85, y, 3.4, "주·월 블록", ["예보 시작일부터 7일 묶음", "또는 완전히 포함된 달력월"], mod="training/apply.py", color=cm)
    node(ax, "b2", 7.65, y, 3.0, "ERA5 편차", ["조화 기후값(물리단위)", "predictors.npz 에 저장"], mod="export_models.py", color=cm)
    node(ax, "b3", 11.05, y, 2.9, "EC46 계통오차", ["모델 주·월 편차로", "모델 평년값 역산"], mod="forecast.py", color=cm)
    node(ax, "b4", 14.35, y, 2.8, "ERA5 기후로 교체", ["t2m·d2m·msl: 가법", "tp: 승법 (비율 유지)"], mod="forecast.py", color=cm)
    node(ax, "b5", 17.55, y, 2.6, "출력", ["pred.nc (학습)", "멤버별 편차 (예보)"], kind="data", color=cm)
    for a, b in [("b0", "b1"), ("b1", "b2"), ("b2", "b3"), ("b3", "b4"), ("b4", "b5")]:
        edge(ax, a, b, color=cm)
    y = 6.0
    node(ax, "c0", 0.45, y, 3.0, "입력", ["ERA5 1° 월평균", "SEAS5 재예보·실황"], kind="data", color=cp)
    node(ax, "c1", 3.85, y, 3.4, "변수 이름·단위 통일", ["2t→t2m, tprate→tp …", "z: m, msl: hPa, tp: mm/일", "sst: °C, 200 hPa 층 분리"], mod="predictors._std · _units", color=cp)
    node(ax, "c2", 7.65, y, 3.0, "ERA5 편차", ["1991–2020 달별 기후값", "→ 월 편차"], mod="era5_clim · era5_anoms", color=cp)
    node(ax, "c3", 7.65, 7.7, 3.0, "재예보 모형기후", ["초기화 달 · 리드별", "1993–2016 멤버 평균"], mod="hind_clim", color=cp)
    node(ax, "c4", 11.05, 6.85, 2.9, "SEAS5 편차", ["실황 − 재예보 기후", "+ (ERA5 1993–16", "   − ERA5 1991–20)"], mod="seas5_anoms", color=cp)
    node(ax, "c5", 14.35, 6.85, 2.8, "격자 맞춤", ["ERA5 1° 격자로", "선형 보간"], mod="predictors.regrid", color=cp)
    node(ax, "c6", 17.55, 6.0, 2.6, "출력", ["ERA5 편차 (학습)", "SEAS5 멤버 편차"], kind="data", color=cp)
    node(ax, "c7", 17.55, 7.55, 2.6, "clim_init10.nc", ["ERA5·재예보 기후값 (save_clims)", "변수별 그룹 ~18 MB", "운영 시 재사용"], kind="data", color=cp)
    edge(ax, "c0", "c1", color=cp)
    edge(ax, "c1", "c2", ta=0.4, color=cp)
    edge(ax, "c1", "c3", sa="r", sb="l", ta=0.85, color=cp)
    edge(ax, "c2", "c4", color=cp, tb=0.3)
    edge(ax, "c3", "c4", color=cp, tb=0.75)
    edge(ax, "c4", "c5", color=cp)
    edge(ax, "c5", "c6", color=cp, tb=0.6)
    edge(ax, "c5", "c7", color=cp, tb=0.4)
    footer(ax, W, H, FOOT)
    legend(ax, 0.45, 9.45, items=("proc", "data"))
    save(fig, "flow_s2_prep")


# ─────────────────────────── 3. 예측인자 생성 ───────────────────────────
def s3_pred():
    W, H = 20.5, 12.4
    fig, ax = canvas(W, H, "3단계 · 예측인자 생성",
                     "PyESD의 Predictor·RegionalAverage(지역평균)와 teleconnections.eof_analysis(원격상관 EOF)를 재구현 — 대상점마다 설계행렬 X 구성")
    cm, cp = C["ec46"], C["seas5"]
    band(ax, 0.25, 1.15, W - 0.5, 3.4, "EC46 주별 1개월", cm)
    band(ax, 0.25, 4.75, W - 0.5, 6.95, "SEAS5 월별 3개월", cp)
    node(ax, "e0", 0.45, 1.65, 3.2, "편차장", ["247점 × 4변수", "t2m·d2m·msl·tp", "(주 또는 월)"], kind="data", color=cm)
    node(ax, "e1", 4.1, 1.55, 4.6, "원격상관 EOF", ["변수별 표준화 편차장 SVD", "상위 PC: 주 6개 · 월 3개 (× 4변수)",
                                                  "달별(±1개월 창) 적합"], mod="esd.EOFPredictors", color=cm)
    node(ax, "e2", 4.1, 3.1, 4.6, "지역평균 (RegionalAverage)", ["대상점 중심 반경 250 km", "가우스 가중 평균 4변수"],
         mod="esd.radius_weights", color=cm)
    node(ax, "e3", 9.2, 2.2, 4.0, "설계행렬", ["X = [PC | 지역평균]", "주 24 + 4 = 28열", "월 12 + 4 = 16열"], mod="esd.design", color=cm)
    node(ax, "e4", 13.7, 2.2, 3.1, "열 표준화", ["MonthlyStandardizer", "역할 (학습 평균·SD)"], color=cm)
    node(ax, "e5", 17.3, 2.2, 2.85, "→ 4단계", ["점별 능형회귀"], kind="data", color=cm)
    edge(ax, "e0", "e1", color=cm, tb=0.5); edge(ax, "e0", "e2", sa="r", sb="l", color=cm, ta=0.85)
    edge(ax, "e1", "e3", color=cm, tb=0.3); edge(ax, "e2", "e3", color=cm, tb=0.75)
    edge(ax, "e3", "e4", color=cm); edge(ax, "e4", "e5", color=cm)

    node(ax, "p0", 0.45, 5.3, 3.2, "편차장 (1°)", ["지상 6: t2m d2m msl tp", "  u10 v10 (+ sst)", "기압면 6: z500 z850",
                                                     "  t850 q850 u850 v850", "확장: z·u·v200, 인도양",
                                                     "  SST, 북반구 Z500"], kind="data", color=cp)
    node(ax, "p1", 4.1, 5.2, 4.6, "지역평균 12변수", ["대상점 반경 250 km 가우스 가중", "지상 6 + 기압면 6 (sst는 지수로만)"],
         mod="predictors.LOCAL", color=cp)
    node(ax, "p2", 4.1, 6.65, 4.6, "상자·점 기후지수 (INDEX_DEFS)", [
        "Niño3.4  열대 동태평양 SST",
        "WNPSH  Z850 15–25°N, 115–150°E",
        "KH  Z500 30–45°N, 120–135°E (한반도 고기압)",
        "OKH  Z500 50–60°N, 130–150°E",
        "KSST  SST 25–40°N, 120–135°E",
        "WP  Z500 (60N,155E) − (30N,155E)",
        "SH  SLP 40–60°N, 80–120°E (시베리아고기압)",
        "EAJ  u200 40–50N − 30–37.5N (없으면 u500)",
        "CGT · SAH (Z200) · DMI (인도양) · EU (Z500)"],
        mod="predictors.indices", color=cp)
    node(ax, "p3", 4.1, 9.55, 4.6, "EOF 패턴 지수 (PATTERN_DEFS)", [
        "PJ  850 hPa 와도 EOF, 10–60N 100–160E",
        "NAO · AO  북반구 Z500 EOF",
        "SRP  v200 EOF, 20–60N 30–130E",
        "1991–2020 전체 달로 1회 적합, 부호 규칙 고정"], mod="predictors.PATTERN_DEFS", color=cp)
    node(ax, "p4", 9.2, 9.75, 4.0, "대규모 EOF PC 9", ["z500 · msl · t2m 각 3개", "달별(±1개월) · 폴드별 재적합"],
         mod="EOF_VARS", color=cp)
    node(ax, "pd", 9.25, 6.0, 3.9, "지수에 필요한 장이", ["자료에 있나?"], kind="dec", h=1.05)
    node(ax, "px", 9.2, 7.55, 4.0, "자동 제외", ["예: u200 없으면 EAJ 대신 EAJ500", "인도양 SST 없으면 DMI 제외"],
         color=MUTED)
    node(ax, "p5", 13.7, 6.65, 3.1, "특징표", ["지역평균 12", "+ 기후지수 7~16", "+ EOF PC 9",
                                               "= 28~37열"], mod="feature_table", color=cp)
    node(ax, "p6", 13.7, 8.65, 3.1, "열 표준화", ["학습 평균·SD 저장", "예보에 그대로 적용"], mod="train.standardize", color=cp)
    node(ax, "p7", 17.3, 6.65, 2.85, "→ 4단계", ["점별 ARD 회귀"], kind="data", color=cp)
    node(ax, "pn", 13.7, 10.25, 6.45, "제외 인자", ["MJO·BSISO: 일 단위 진동 · OLR 미입수 → 월평균 체계 제외",
                                                  "DRY(선행 토양건조): 토양수분 미입수"], kind="note", color=MUTED)
    edge(ax, "p0", "p1", color=cp, sa="r", sb="l", ta=0.12)
    edge(ax, "p0", "p2", color=cp, sa="r", sb="l", ta=0.5, tb=0.3)
    edge(ax, "p0", "p3", color=cp, sa="r", sb="l", ta=0.9, tb=0.4)
    edge(ax, "p2", "pd", color=cp, ta=0.25)
    edge(ax, "pd", "p5", label="예", color=cp, tb=0.35)
    edge(ax, "pd", "px", label="아니오", color=MUTED)
    edge(ax, "p1", "p5", sa="r", sb="t", route="l", color=cp)
    edge(ax, "p3", "p4", color=cp, ta=0.6)
    edge(ax, "p4", "p5", sa="r", sb="l", color=cp, tb=0.85, route="h", mid=13.45)
    edge(ax, "p5", "p6", color=cp)
    edge(ax, "p5", "p7", color=cp)
    legend(ax, 0.45, 11.95, items=("proc", "data", "dec", "note"))
    footer(ax, W, H, FOOT)
    save(fig, "flow_s3_predictors")


# ─────────────────────────── 4. 전이함수 학습·검증 ───────────────────────────
def s4_train():
    W, H = 20.5, 9.3
    fig, ax = canvas(W, H, "4단계 · 전이함수 학습·검증",
                     "대상(격자 412 | ASOS 66) × 요소(T | P) × 달(1~12)[× 주·월] 마다 점별 회귀 — EOF까지 폴드 안에서 다시 계산해 정보 누설 방지")
    cs, cm, cp = C["common"], C["ec46"], C["seas5"]
    node(ax, "in1", 0.45, 1.4, 3.3, "예측대상 편차", ["target_grid.nc (412)", "target_asos.nc (66)"], kind="data", color=cs)
    node(ax, "in2", 0.45, 2.75, 3.3, "예측인자", ["EC46: pred.nc 247점", "SEAS5: ERA5 1° 편차장"], kind="data", color=C["ext"])
    node(ax, "loop", 4.2, 1.4, 4.0, "반복 단위", ["kind ∈ {grid, asos}", "var ∈ {T, P}", "month 1~12 (±1개월 창 표본)",
                                               "EC46은 scale ∈ {week, month}"], mod="esd.season_window", color=INK2)
    edge(ax, "in1", "loop", color=cs); edge(ax, "in2", "loop", sa="r", sb="l", tb=0.75, color=INK2)
    # EC46
    band(ax, 8.55, 1.1, 11.7, 3.4, "EC46 · 능형회귀", cm)
    node(ax, "e1", 8.75, 1.6, 3.6, "교차검증", ["leave-one-year-out", "폴드마다 EOF 재적합", "+ 독립 검증 2023–2026"],
         mod="training/train.py", color=cm)
    node(ax, "e2", 12.6, 1.6, 3.6, "능형회귀 (배치)", ["열 표준화 후 점별 해", "α ∈ {1, 10, 100, 1000}", "연도 5묶음 CV로 α 선택"],
         mod="esd.fit_points · _ridge_batch", color=cm)
    node(ax, "e3", 16.45, 1.6, 3.6, "최종 모형", ["전 기간 재적합", "→ numpy 전용 npz", "EOF·계수·σ·q10/q50/q90"],
         mod="export_models.py", color=cm)
    node(ax, "e4", 12.6, 3.3, 7.45, None, ["CV 상관(재분석 입력) 격자 기온 주 0.97 · 월 0.95, 강수 0.92 · 0.90",
                                            "ASOS 기온 0.95 · 0.90, 강수 0.87 · 0.81",
                                            "모형 비교(ASOS 월 MSESS): ARD 1위 (T 0.87 · P 0.65)",
                                            "Lasso·Ridge 동급 → 운영은 계산이 빠른 Ridge"],
         kind="note", color=cm)
    for a, b in [("e1", "e2"), ("e2", "e3")]:
        edge(ax, a, b, color=cm)
    # SEAS5
    band(ax, 8.55, 4.7, 11.7, 3.55, "SEAS5 · ARD (PyESD 원본형)", cp)
    node(ax, "s1", 8.75, 5.2, 3.6, "5년 블록 교차검증", ["1991–2025, 7개 블록", "폴드마다 EOF·패턴·회귀", "모두 재적합"],
         mod="seas5_monthly/train.py", color=cp)
    node(ax, "s2", 12.6, 5.2, 3.6, "ARDRegression", ["점별 병렬 (joblib)", "관련 없는 인자 계수 → 0", "= 자동 예측인자 선택"],
         mod="train.fit_points", color=cp)
    node(ax, "s3", 16.45, 5.2, 3.6, "최종 모형", ["희소 계수 W (pack_W)", "EOF·패턴 로딩, 표준화 값", "σ, 3분위 경계, q10~q90"],
         mod="models/{kind}_{var}.npz", color=cp)
    node(ax, "s4", 8.75, 7.15, 11.3, None, ["재분석 입력 CV 점수(상관·MSESS)와 선택된 인자 빈도를 cv_seas5_{kind}_{var}.nc 에 저장",
                                            "모델 예보 기술(리드별 ACC·RPSS)은 6단계 재예보 검증에서 따로 평가"], kind="note", color=cp)
    for a, b in [("s1", "s2"), ("s2", "s3")]:
        edge(ax, a, b, color=cp)
    edge(ax, "loop", "e1", sa="r", sb="l", route="h", mid=8.4, ta=0.3, color=cm)
    edge(ax, "loop", "s1", sa="r", sb="l", route="h", mid=8.4, ta=0.7, color=cp)
    # 산출
    node(ax, "o1", 0.45, 5.2, 7.6, "교차검증 산출 (공통)", [
        "CV 예측 시계열 → 상관·MSESS·기준선(기후값) 비교",
        "CV 잔차 σ (점·달별) → 6단계 확률의 퍼짐",
        "관측 편차 3분위 q33·q67 (1991–2020) → 확률 경계",
        "q10·q50·q90 → 극한·중앙값 초과 확률",
        "그림: docs/figures/fig02~05, fig_s4·s5"], kind="data", color=INK2)
    edge(ax, "e1", "o1", sa="b", sb="t", route="v", mid=4.6, ta=0.15, tb=0.7, color=cm)
    edge(ax, "s1", "o1", color=cp, ta=0.5, tb=0.35)
    footer(ax, W, H, FOOT)
    legend(ax, 0.45, 8.85, items=("proc", "data", "note"))
    save(fig, "flow_s4_train")


# ─────────────────────────── 5. 모델 투영 ───────────────────────────
def s5_project():
    W, H = 20.5, 9.35
    fig, ax = canvas(W, H, "5단계 · 모델 투영 (예측)",
                     "학습에 쓴 EOF 로딩·지수 정의·표준화 값·계수를 그대로 모델 편차장에 적용 — 멤버마다 따로 투영해 앙상블 분포를 보존")
    cm, cp, cs = C["ec46"], C["seas5"], C["common"]
    band(ax, 0.25, 1.15, W - 0.5, 2.6, "EC46", cm)
    band(ax, 0.25, 3.95, W - 0.5, 4.75, "SEAS5", cp)
    node(ax, "e0", 0.45, 1.65, 3.0, "EC46 멤버 편차", ["51 × 블록 × 247점", "ERA5 기후로 보정됨"], kind="data", color=cm)
    node(ax, "e1", 3.85, 1.6, 3.5, "EOF 투영", ["편차장 · 학습 EOF 로딩", "→ 멤버별 PC"], mod="training/apply.py", color=cm)
    node(ax, "e2", 3.85, 2.95, 3.5, "지역평균", ["저장된 250 km 가중치"], color=cm)
    node(ax, "e3", 7.8, 2.2, 3.4, "계수 적용", ["달별 계수 × 표준화 X", "→ 478점 × 51 편차"], mod="esd.predict_points", color=cm)
    node(ax, "e4", 11.6, 2.2, 3.4, "역변환", ["강수 √ → 제곱", "+ 기후값 → 절대값"], color=cm)
    node(ax, "e5", 15.4, 2.0, 4.75, "멤버별 상세예측", ["412격자 + 66지점 × 51멤버", "1~4주(7일) · 달력월", "+ 원 모델과의 보정량 기록"],
         kind="data", color=cm)
    edge(ax, "e0", "e1", color=cm, ta=0.35); edge(ax, "e0", "e2", sa="r", sb="l", color=cm, ta=0.8)
    edge(ax, "e1", "e3", color=cm, tb=0.3); edge(ax, "e2", "e3", color=cm, tb=0.75)
    edge(ax, "e3", "e4", color=cm); edge(ax, "e4", "e5", color=cm)

    node(ax, "s0", 0.45, 4.45, 3.0, "SEAS5 편차 (1°)", ["실황 51멤버", "  리드 2·3·4", "재예보 25 × 24년", "  리드 1~4"], kind="data", color=cp)
    node(ax, "s1", 3.85, 4.4, 3.5, "특징표 재구성", ["지역평균 12", "기후지수 (같은 정의)", "패턴 지수: 저장 EOF 투영",
                                                    "EOF PC: 학습 로딩 투영"], mod="apply.model_eofs · feature_table", color=cp)
    node(ax, "s2", 7.8, 4.4, 3.4, "표준화 → W", ["학습 평균·SD로 표준화", "희소 계수 W 곱", "(unpack_W)"], mod="apply.model_W", color=cp)
    node(ax, "s3", 11.6, 4.4, 3.4, "역변환", ["강수 √ → 제곱", "월 일수로 누적 mm"], mod="forecast.downscale", color=cp)
    node(ax, "s4", 15.4, 4.3, 4.75, "멤버별 상세예측", ["실황: 478점 × 51 × 3개월", "  (10월 초기화 → 11·12·1월)",
                                                       "재예보: 478점 × 25 × 24년 × 4리드"], kind="data", color=cp)
    node(ax, "sv", 15.4, 7.4, 4.75, "원 SEAS5 기준선", ["같은 반경 가중 지역평균 t2m·√tp", "멤버 분포 → 3분위 (6단계 비교용)"],
         kind="data", color=C["ext"])
    edge(ax, "s0", "s1", color=cp); edge(ax, "s1", "s2", color=cp); edge(ax, "s2", "s3", color=cp); edge(ax, "s3", "s4", color=cp)
    edge(ax, "s0", "sv", sa="b", sb="l", route="l", color=C["ext"], ta=0.5, label="재예보·실황")
    node(ax, "mm", 7.8, 6.3, 3.4, "학습 모형 npz", ["4단계 산출"], kind="data", color=cp)
    edge(ax, "mm", "s2", sa="t", sb="b", color=cp)
    footer(ax, W, H, FOOT)
    legend(ax, 0.45, 8.95, items=("proc", "data"))
    save(fig, "flow_s5_project")


# ─────────────────────────── 6. 앙상블 통계·검증 ───────────────────────────
def s6_stats():
    W, H = 20.5, 9.75
    fig, ax = canvas(W, H, "6단계 · 앙상블 통계·검증",
                     "멤버 투영값 μk 와 교차검증 잔차 σ 의 정규혼합 분포로 확률·백분위를 계산 (회귀가 줄인 분산을 σ로 되살림)")
    cm, cp = C["ec46"], C["seas5"]
    node(ax, "i0", 0.45, 1.5, 3.4, "입력", ["멤버별 편차 μk (k = 1…K)", "CV 잔차 σ (점·달별)", "관측 q33·q67, q10~q90",
                                          "평년값 (기온 °C, 강수 mm)"], kind="data", color=INK2)
    node(ax, "f1", 4.3, 1.4, 5.2, "정규혼합 분포", ["F(x) = 1/K Σ Φ((x − μk) / σ)",
                                                   "P(낮음) = F(q33), P(높음) = 1 − F(q67)",
                                                   "멤버당 40개 표본으로 백분위 계산"], mod="pyesd/products.cdf · fields", color=INK2)
    node(ax, "f2", 10.0, 1.15, 5.0, "표출 항목 (field)", [
        "below / near / above  3분위 확률",
        "gt_median  평년 중앙값 초과 확률",
        "lt_p10 / gt_p90  극한 확률",
        "thr  기온 편차 ±1·±2 °C",
        "     강수 평년비 50·150·200 %",
        "     누적 mm (주 10~100, 월 50~300)",
        "p10 · p25 · p50 · p75 · p90 예측범위",
        "b10~b90 평년 분포 경계"], mod="products.fields · to_jsonable", color=INK2)
    node(ax, "f3", 15.5, 1.4, 4.65, "지역 집계", ["south_mean: 제주 제외 62지점", "  (jeju == 0) 평균",
                                                "T_p10/T_p90 · P_p10/P_p90"], color=INK2)
    node(ax, "o1", 15.5, 3.3, 4.65, "EC46 산출", ["out/ec46_weekly/downscaling.json", "asos66_downscaled.csv · map_week.png"],
         kind="data", color=cm)
    node(ax, "o2", 15.5, 4.6, 4.65, "SEAS5 산출", ["out/seas5_monthly/downscaling.json", "격자·지점 p10~p90 · map png"],
         kind="data", color=cp)
    edge(ax, "i0", "f1", color=INK2); edge(ax, "f1", "f2", color=INK2); edge(ax, "f2", "f3", color=INK2, ta=0.2, tb=0.4)
    edge(ax, "f3", "o1", color=cm); edge(ax, "f3", "o2", sa="r", sb="r", route="h", mid=W - 0.12, color=cp, ta=0.6)
    # 검증
    band(ax, 0.25, 5.75, W - 0.5, 3.4, "SEAS5 재예보 검증 (1993–2016, 25멤버, 리드 1~4)", cp)
    node(ax, "v0", 0.45, 6.25, 3.4, "재예보 투영값", ["478점 × 25 × 24년 × 4리드", "+ 원 SEAS5 기준선", "+ ERA5·ASOS 관측 편차"],
         kind="data", color=cp)
    node(ax, "v1", 4.3, 6.2, 3.6, "ACC", ["앙상블 평균과 관측의", "시간상관 (점별)"], mod="hindcast_verify.acc", color=cp)
    node(ax, "v2", 8.2, 6.2, 3.6, "RPSS", ["3분위 확률 순위확률점수", "기준: 기후확률 1/3"], mod="hindcast_verify.rpss", color=cp)
    node(ax, "v3", 4.3, 7.65, 3.6, "10~90 % 포함률", ["관측이 p10~p90 안에", "든 비율 (목표 80 %)"], color=cp)
    node(ax, "v4", 8.2, 7.65, 3.6, "신뢰도표", ["above·below 확률 구간별", "예보 확률 vs 관측 빈도"], mod="hindcast_verify.reliability", color=cp)
    node(ax, "v5", 12.3, 6.65, 3.4, "비교", ["상세화 vs 원 SEAS5", "리드·달·점별"], color=cp)
    node(ax, "v6", 16.1, 6.3, 4.05, "검증 산출", ["docs/seas5/hindcast_summary.csv", "hindcast_{kind}_{var}_init10.nc",
                                                  "fig_hindcast_init10.png", "models/clim_init10.nc"], kind="data", color=cp)
    edge(ax, "v0", "v1", color=cp, ta=0.3); edge(ax, "v0", "v3", sa="r", sb="l", color=cp, ta=0.85)
    edge(ax, "v1", "v2", color=cp); edge(ax, "v3", "v4", color=cp)
    edge(ax, "v2", "v5", color=cp, tb=0.3); edge(ax, "v4", "v5", sa="r", sb="l", color=cp, tb=0.8)
    edge(ax, "v5", "v6", color=cp)
    footer(ax, W, H, FOOT)
    legend(ax, 0.45, 9.35, items=("proc", "data"))
    save(fig, "flow_s6_stats")


# ─────────────────────────── 7. 게시 ───────────────────────────
def s7_publish():
    W, H = 20.5, 8.6
    fig, ax = canvas(W, H, "7단계 · 자동 실행·게시",
                     "GitHub Actions가 산출을 결과 브랜치에 올리면 사이트 게시 워크플로가 이어서 변환해 gh-pages로 배포 (Pages: gh-pages / root)")
    cm, cp, cu = C["ec46"], C["seas5"], C["pub"]
    node(ax, "w1", 0.45, 1.4, 4.4, "ec46_weekly.yml", ["매일 09:30 UTC (18:30 KST)", "pip install -r requirements.txt",
                                                       "forecast.py --maps", "(numpy·scipy만, 학습 npz 사용)"], color=cm)
    node(ax, "w2", 0.45, 4.0, 4.4, "seas5_monthly.yml", ["매월 14일 03:00 UTC", "(SEAS5 13일 공개 다음 날)", "초기화 달·연도 결정",
                                                         "get_cds.py --only seas5 --no-hind", "forecast.py --leads 2,3,4 --maps"],
         mod="secret: CDSAPI_KEY", color=cp)
    node(ax, "b1", 5.3, 1.75, 3.1, "out-ec46", ["단일 커밋 브랜치", "downscaling.json"], kind="data", color=cm)
    node(ax, "b2", 5.3, 4.6, 3.1, "out-seas5", ["단일 커밋 브랜치", "downscaling.json"], kind="data", color=cp)
    node(ax, "s", 8.9, 2.6, 4.3, "site.yml (사이트 게시)", ["workflow_run: 두 워크플로 완료 시", "최신 산출 가져오기",
                                                           "(없는 쪽은 건너뜀)"], color=cu)
    node(ax, "bd", 8.9, 4.7, 4.3, "build_site_data.py", ["downscaling.json → 사이트 형식", "기온 °C 평균 · 강수 mm 누적",
                                                       "확률은 정수 %"], color=cu)
    node(ax, "o", 13.7, 1.4, 6.45, "site/data (생성)", [
        "grid.json  0.25° 412격자 step·lats·lons·fill_index",
        "latest/manifest.json  발표 시각·기간 목록",
        "map_1m.json (EC46 1~4주) · map_3m.json (SEAS5 3개월)",
        "stations.json  ASOS 66 field + 남한 62지점 평균",
        "points/<i>.json · stations/<stn>.json  지점 상세"], kind="data", color=cu)
    node(ax, "gp", 13.7, 4.35, 6.45, "gh-pages → GitHub Pages", ["https://nalssi89.github.io/downscale_by_pyesd/",
                                                                  "index.html · point.html · js/ · css/ · vendor/leaflet"],
         color=cu)
    node(ax, "ui", 13.7, 6.05, 6.45, "화면 (objective_climate_outlook 형식)", [
        "영역: 남한(기본) · 한반도  |  전망: 1개월 · 3개월",
        "요소: 기온 · 강수  |  기간: 주 · 월",
        "표출: 3분위 우세범주 · 중앙값 초과 · 앙상블 평균",
        "  평년편차 · 임계값 · 극한 확률 · 스프레드, 지점 상세 창"], kind="note", color=cu)
    edge(ax, "w1", "b1", color=cm, ta=0.42); edge(ax, "w2", "b2", color=cp, ta=0.4)
    edge(ax, "b1", "s", color=cm, tb=0.25, mid=8.65); edge(ax, "b2", "s", sa="r", sb="l", route="h", mid=8.65, color=cp, tb=0.75)
    edge(ax, "s", "bd", color=cu); edge(ax, "bd", "o", sa="r", sb="l", color=cu, tb=0.75)
    edge(ax, "o", "gp", color=cu); edge(ax, "gp", "ui", color=cu)
    node(ax, "n", 0.45, 7.35, 12.75, "운영에 필요한 것", [
        "저장소 secret CDSAPI_KEY (SEAS5) — 코드·파일에 키를 쓰지 않음 · EC46은 키 불필요 (Open-Meteo)",
        "학습 모형 npz와 clim_init{mm}.nc 는 학습 환경에서 만들어 저장소에 둠 (재학습은 수동)"], kind="note", color=MUTED)
    footer(ax, W, H, FOOT)
    save(fig, "flow_s7_publish")


if __name__ == "__main__":
    overview(); s1_data(); s2_prep(); s3_pred(); s4_train(); s5_project(); s6_stats(); s7_publish()
