from EM import SPMSM
from EM import StarOfSlots
from EM import MMF
import numpy as np
import matplotlib.pyplot as plt


def B_g_FFT_myMachine():
    p = 7

    # 형상 치수
    R_i = 10.00 * 1e-3
    R_r = 12.05 * 1e-3
    R_m = 13.05 * 1e-3   # 영구자석 외경
    R_s = 13.55 * 1e-3   # 고정자 보어 내경
    R_so = 14.0 * 1e-3
    alpha_p = 0.9

    # -------------------------------
    # 재질 사양
    B_r = 1.12
    mu_r1 = 3000
    mu_r2 = 1.05
    mu_r3 = 1
    mu_r4 = 3000

    info = {
        "R_i": R_i,
        "R_r": R_r,
        "R_m": R_m,
        "R_s": R_s,
        "R_so": R_so,
        "p": p,
        "alpha_p": alpha_p,
        "B_r": B_r,
        "mu_r1": mu_r1,
        "mu_r2": mu_r2,
        "mu_r3": mu_r3,
        "mu_r4": mu_r4
    }

    # -----------------------------------
    # 공극자속밀도 파형 계산
    spmsm = SPMSM(info=info)
    theta, B_r, B_t = spmsm.airgap_fluxdensity(10, 1024) # 홀수고조파 10개,

    # FFT 계산
    N = len(B_r)  # 데이터 길이
    fft_result = np.fft.fft(B_r)  # FFT 계산
    fft_magnitude = np.abs(fft_result) / N  # FFT 크기 정규화
    one_side_magnitude = fft_magnitude[:N // 2] * 2  # One-sided 스펙트럼
    filtered_magnitudes = one_side_magnitude[(np.arange(len(one_side_magnitude)) % p == 0)]

    return filtered_magnitudes

def main():
    # -------------------------------
    # 모터 사양
    p = 7               # 극쌍수
    Q = 12              # 슬롯수
    beta = 30           # 전류위상각 [degE]

    print('-----------------------------------------------------')
    print('Compute on Number of poles: {}, Number of Slots: {}'.format(p*2, Q))
    print('-----------------------------------------------------')

    # -----------------------------------
    # Star of Slots 생성
    ss = StarOfSlots(p, Q)
    yq = ss.suggestYq

    if not ss.feasible:
        print('Not feasible')
        return

    print('Suggested Coil Throw:', yq)
    print('Zero Mutual Inductance:', ss.zeroMutual)
    print('Single Layer Winding:', ss.validateSingleLayerWinding)

    # -----------------------------------
    # 코일 패턴 출력
    patterns = ss.getPatterns()
    print('Winding patterns: {}'.format(patterns))

    # -----------------------------------
    # Star of Slots 출력
    ss.plotStarOfSlots()

    # -----------------------------------
    # 권선계수 출력 - 서브하모닉 고려됨
    # 기본파에 대한 권선계수는 주어진 pp에서 확인해야 함
    k_wd = ss.calculateDistributeFactor(p)
    k_wp = ss.calculateShortPitchFactor(yq, p)
    if k_wd is not None:
        k_w = np.abs(k_wd * k_wp)
        rounded_k_w = np.round(k_w, 6)
        print('Winding Coefficient Kw: {}'.format(rounded_k_w))

    # -----------------------------------
    # MMF 출력
    current = np.array([np.cos(np.radians(beta)), np.cos(np.radians(beta - 120)), np.cos(np.radians(beta - 240))])
    #current = np.array([0, 0.00, 1.00])
    mmf = MMF(ss, current, yq)

    # 합성기자력 공간 파형 플롯팅:
    mmf.plotMMF()

    # 합성기자력 고조파 크기 플롯팅: 극당상당 슬롯수로 하모닉 크기를 표준화시킴
    mmf.plotHarmonics()

    # -----------------------------------
    # EMF 출력
    # alpha = 0.7 from FEA
    B_g_FFT1 = np.array(
        [0.0,
         0.711071209,
         0.0,
         0.032913966,
         0.0,
         0.066593151,
         0.0,
         0.045979726,
         0.0,
         0.015706418,
         0.0,
         0.01222255,
         0.0,
         0.005297453,
         0.0,
         0.007201919,
         0.0,
         0.001709152,
         0.0,
         0.002668821
         ]
    )

    # alpha = 1 from FEA
    B_g_FFT2 = np.array(
        [0,
         0.761751336,
         0,
         0.201739701,
         0,
         0.09244109,
         0,
         0.044334382,
         0,
         0.028503203,
         0,
         0.02263106,
         0,
         0.010959666,
         0,
         0.006547525,
         0,
         0.007012817,
         0,
         0.005146179,
         0,
         0.00585453,
         0,
         0.005939372,
         0,
         0.004157365,
         0,
         0.002026747,
         0,
         0.000471535,
         0,
         0.001452851
         ]
    )

    # 해석모델로부터 공극자속밀도를 구함
    B_g_FFT3 = B_g_FFT_myMachine()

    print('THD of the backEMF:', mmf.THDforBackEMF(B_g_FFT1))
    mmf.plotBackEMF(B_g_FFT1, True, False)

    # -------------------------------------
    # Vibration mode to check
    check_modes = np.array([1,2,3,4,5]) # 검토할 진동모드
    mode_results_1 = mmf.vibrationModeBySubharmonics(check_modes)
    print('Checking Vibration Mode by Sub-harmonics')
    for mode, result in mode_results_1:
        print(f"  Mode: {mode}: {result}")

    with_mmf_harmonics = np.array([1, 5, 7, 11, 13])    # 모드계산에 사용될 기자력의 공간 고조파 차수
    with_pole_harmonics = np.array([1, 3, 5, 7])   # 모드계산에 사용될 자극의 공간 고조파 차수
    mode_results_2 = mmf.vibrationModeByHarmonics(B_g_FFT1, with_mmf_harmonics, with_pole_harmonics, check_modes)
    print('Checking Vibration Mode by Harmonics')
    for mode, result in mode_results_2:
        print(f"  Mode: {mode}: {result}")


if __name__ == '__main__':
    main()