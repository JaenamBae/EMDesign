from EM import SPMSM
from EM import StarOfSlots
from EM import MMF
import numpy as np
import matplotlib.pyplot as plt


def B_g_FFT_myMachine(n_harmonics:int = 20):
    p = 7

    # 형상 치수
    R_i = 10.00 * 1e-3
    R_r = 12.05 * 1e-3
    R_m = 13.05 * 1e-3   # 영구자석 외경
    R_s = 13.55 * 1e-3   # 고정자 보어 내경
    R_so = 15.0 * 1e-3
    alpha_p = 1.0

    g = R_s - R_m
    r = R_s - g*0.05

    # -------------------------------
    # 재질 사양
    B_r = 1.2
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
    spmsm.plotBg(n_harmonics)
    Bg_FFT = np.zeros(n_harmonics)
    for n in range(n_harmonics):
        B_rn, B_tn = spmsm.calculate_harmonic(r, n)
        Bg_FFT[n] = B_rn

    print(Bg_FFT)
    return Bg_FFT

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
    #mmf.plotMMF()

    # 합성기자력 고조파 크기 플롯팅: 극당상당 슬롯수로 하모닉 크기를 표준화시킴
    #mmf.plotHarmonics()

    # -----------------------------------
    # EMF 출력
    # alpha = 0.7 from FEA
    B_g_FFT1 = np.array(
        [0,
         0.941801512,
         0.00E+00,
         -0.215816784,
         0.00E+00,
         0.077892667,
         0.00E+00,
         -0.032732218,
         0.00E+00,
         0.01495327,
         0.00E+00,
         -0.007192753,
         0.00E+00,
         0.003582226,
         0.00E+00,
         -0.001830871,
         0.00E+00,
         0.000953092,
         0.00E+00,
         -0.000504592
         ]
    )

    # 해석모델로부터 공극자속밀도를 구함
    B_g_FFT3 = B_g_FFT_myMachine(40)

    B_g_FFT = B_g_FFT1

    print('THD of the backEMF:', mmf.THDforBackEMF(B_g_FFT))
    mmf.plotBackEMF(B_g_FFT, True, True)

    # -------------------------------------
    # Vibration mode to check
    check_modes = np.array([1,2,3,4,5]) # 검토할 진동모드
    mode_results_1 = mmf.vibrationModeBySubharmonics(check_modes)
    print('Checking Vibration Mode by Sub-harmonics')
    for mode, result in mode_results_1:
        print(f"  Mode: {mode}: {result}")

    with_mmf_harmonics = np.array([1, 5, 7, 11, 13])    # 모드계산에 사용될 기자력의 공간 고조파 차수
    with_pole_harmonics = np.array([1, 3, 5, 7])   # 모드계산에 사용될 자극의 공간 고조파 차수
    mode_results_2 = mmf.vibrationModeByHarmonics(B_g_FFT, with_mmf_harmonics, with_pole_harmonics, check_modes)
    print('Checking Vibration Mode by Harmonics')
    for mode, result in mode_results_2:
        print(f"  Mode: {mode}: {result}")


if __name__ == '__main__':
    main()