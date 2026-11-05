# ERA5·SEAS5 자료 입수 규칙

## 원칙 (사용자 지시)
- **CDS 요청은 전체에서 한 번에 1건.** 앞 건이 `successful`이 되고 내려받은 뒤에만 다음 건을 넣는다. 1건씩 넣으면 바로 running이 되지만, 여러 건을 넣으면 대기열에 묶이고 사용자당 대기 약 3건을 넘으면 `Maximum number queued requests for this dataset`로 거절된다.
- 인증키는 환경변수 `CDSAPI_KEY`로만 읽는다.
- 요청하기 전에 출력 폴더에 같은 이름의 파일이 있는지 보고, 있으면 건너뛴다. CDS에 같은 조건의 요청이 이미 있으면 새로 넣지 않고 이어받는다.
- 거절된 요청을 같은 조건으로 곧바로 다시 보내지 않는다. 대기열 한도 때문에 거절됐다면 10분 뒤 다시 넣고, 비용 한도 때문이면 요청을 나눈다.

## 자료별 경로
| 자료 | 경로 |
|---|---|
| ERA5 시간별 지상, 좁은 영역의 긴 기간 | ECMWF ARCO SFC geoChunked (`arco.datastores.ecmwf.int`, Bearer CDS 키) |
| ERA5 시간별 지상·기압면, 넓은 영역의 짧은 기간 | ECMWF ARCO SFC/PL timeChunked |
| ERA5 일평균 대규모장 1959~2022 | WeatherBench2 (gs, 익명) |
| OLR 등 ECMWF ARCO에 없는 변수 | Google ARCO-ERA5 (gs, 익명) |
| ERA5 월평균 | CDS API 순차 처리 (또는 ARCO 시간별 자료를 직접 평균) |
| SEAS5 재예보·실황 | CDS API 순차 처리 |

- ARCO는 `xr.open_zarr(..., chunks=None)`으로 연다. dask 청크로 열면 메모리 부족(OOM)이 난다.
- 비용 한도는 `ecmwf.datastores.Client.estimate_costs`로 미리 본다. 확인된 한도는 ERA5 지상 월평균 20000, ERA5 기압면 월평균 120000, seasonal 10000이다.
- 같은 데이터셋에서 영역만 다른 요청은 넓은 영역 하나로 묶어 받은 뒤 잘라 쓴다(`seas5_monthly/seas5_combined.py`).

## 구현
- `seas5_monthly/get_cds.py`: 요청 사양
- `seas5_monthly/seq_cds.py`: 순차 처리기. 파일 건너뛰기, 기존 요청 이어받기, zip 응답 병합, 대기열 거절 시 재시도를 맡는다. 실행은 `python seas5_monthly/seq_cds.py OUTDIR --init 10 --fc-year 2026`.
- 클라우드 컨테이너가 재시작되면 백그라운드 작업이 사라진다. 다시 실행하면 이어서 받는다.
