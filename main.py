from EM import StarOfSlots
from EM import MMF
import numpy as np


def main():
    # -----------------------------------
    phases = 3              # 상수
    pole_pair = 4           # 극쌍수
    Q = 12                  # 슬롯수
    polearc_ratio = 0.95    # 극호율
    beta = 30               # 전류위상각

    print('-----------------------------------------------------')
    print('Compute on Number of poles: {}, Number of Slots: {}'.format(pole_pair*2, Q))
    print('-----------------------------------------------------')

    # -----------------------------------
    # Star of Slots 생성
    ss = StarOfSlots(pole_pair, Q)
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
    k_wd = ss.calculateDistributeFactor(pole_pair)
    k_wp = ss.calculateShortPitchFactor(yq, pole_pair)
    if k_wd is not None:
        k_w = k_wd * k_wp
        rounded_k_w = np.round(k_w, 6)
        print('Winding Coefficient Kw: {}'.format(rounded_k_w))

    # -----------------------------------
    # MMF 출력
    current = np.array([np.cos(np.radians(beta)), np.cos(np.radians(beta - 120)), np.cos(np.radians(beta - 240))])
    #current = np.array([0, 0.00, 1.00])
    mmf = MMF(ss, current, yq)
    print('THD of the backEMF:', mmf.THDforBackEMF(polearc_ratio, 20))

    # 합성기자력 공간 파형 플롯팅:
    mmf.plotMMF()

    # 합성기자력 고조파 크기 플롯팅: 극당상당 슬롯수로 하모닉 크기를 표준화시킴
    mmf.plotHarmonics()

    # -------------------------------------
    # Vibration mode to check
    check_modes = np.array([1,2,3,4,5]) # 검토할 진동모드
    mode_results_1 = mmf.vibrationModeBySubharmonics(check_modes)
    print('Checking Vibration Mode by Sub-harmonics')
    for mode, result in mode_results_1:
        print(f"  Mode: {mode}: {result}")

    with_mmf_harmonics = np.array([1, 5, 7, 11, 13])    # 모드계산에 사용될 기자력의 공간 고조파 차수
    with_pole_harmonics = np.array([1, 3, 5, 7])   # 모드계산에 사용될 자극의 공간 고조파 차수
    mode_results_2 = mmf.vibrationModeByHarmonics(polearc_ratio, with_mmf_harmonics, with_pole_harmonics, check_modes)
    print('Checking Vibration Mode by Harmonics')
    for mode, result in mode_results_2:
        print(f"  Mode: {mode}: {result}")


if __name__ == '__main__':
    main()