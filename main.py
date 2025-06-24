from EM import StarOfSlots
from EM import MMF
import numpy as np
import matplotlib.pyplot as plt
import string

def plot_harmonic_magnitudes(coefficients, type, pp):
    """
    고조파별 푸리에 계수의 크기를 플롯팅합니다 (n = 0 포함, 양수 고조파만).

    Parameters:
        coefficients (np.ndarray): 푸리에 계수 c_n, n = 0부터 양의 정수까지 저장된 배열.
        type (np.ndarray): 하모닉차수의 공간분포 타입; 0: Unknown, 1: Same phase, 2: 120, 3: 240
        pp: 극쌍수
    """
    # n 값 생성: 0부터 len(coefficients)-1까지
    n_values = np.arange(len(coefficients))
    magnitudes = 2 * np.abs(coefficients)  # 크기 계산
    magnitudes[0] = np.abs(coefficients[0])

    # 플롯팅
    plt.figure(figsize=(10, 6))
    for harmonic, harmonic_type in zip(n_values, type):
        # 기본 플롯 생성
        stem = plt.stem([harmonic], [magnitudes[harmonic]], markerfmt='ko', linefmt='k-', basefmt='k')

        # `pp`의 배수인지 확인; 기본파와 고조파에 한해서는 마커 사이즈를 크게하여 표기함
        if harmonic % pp == 0 and harmonic != 0:  # 0은 제외
            stem[0].set_markersize(15)  # 마커 크기 설정
            stem[1].set_linewidth(5)    # 스템선의 굵기를 2로 설정

        # 고조파 타입의 확인; 고조파 타입은 마커 색상으로 표기함
        if harmonic_type == 1:  # 공간적 위상이 같음--> 그레이
            stem[0].set_color('gray')
            stem[1].set_color('gray')

        elif harmonic_type == 2:  # 공간적 위상이 120도 차이남 --> 파란색
            stem[0].set_color('blue')
            stem[1].set_color('blue')

        elif harmonic_type == 3:  # 공간적 위상이 240도 차이남 --> 빨간색
            stem[0].set_color('red')
            stem[1].set_color('red')

        else:  # 공간적 위상이 잘못됨 --> 검은색
            stem[0].set_color('k')
            stem[1].set_color('k')


    plt.title("Fourier Coefficients Magnitudes (n = 0 to positive harmonics)")
    plt.xlabel("Harmonic Number (n)")
    plt.ylabel("Magnitude (2|c_n|)")
    plt.grid(True)
    plt.show()

def create_mapping(max_value):
    letters = string.ascii_uppercase  # A-Z
    mapping = {}
    index = 1
    while index <= max_value:
        key = index
        # 동적 알파벳 생성 (AA, AB, ... 확장 포함)
        symbol = ""
        temp = index
        while temp > 0:
            temp -= 1
            symbol = letters[temp % 26] + symbol
            temp //= 26
        mapping[key] = symbol
        index += 1
    return mapping

def main():
    # -----------------------------------
    phases = 3  # 상수
    pole_pair = 7  # 극쌍수
    Q = 12  # 슬롯수
    polearc_ratio = 0.95

    # -----------------------------------
    # Star of Slots 생성
    ss = StarOfSlots(pole_pair, Q)
    yq = ss.suggestYq

    if not ss.feasible:
        print('Not feasible')
        return

    print('Suggested Coil Throw:', yq)
    print('Zero Mutual:', ss.zeroMutual)
    print('Validate Single Layer Winding:', ss.validateSingleLayerWinding)

    # -----------------------------------
    # 코일 패턴 출력
    # 패턴 매핑 사전 생성
    mapping = create_mapping(phases)

    # 변환된 결과 리스트
    winding_pattern = []

    # 1->A 와 같은 변환 작업
    for num in ss.pattern:
        if num > 0:
            winding_pattern.append(mapping[num])  # 양수는 그대로 매핑
        else:
            winding_pattern.append(f"-{mapping[abs(num)]}")  # 음수는 -를 붙임
    print('Winding patterns: {}'.format(winding_pattern))

    # -----------------------------------
    # 권선계수 출력 - 서브하모닉 고려됨
    # 기본파에 대한 권선계수는 주어진 pp에서 확인해야 함
    k_wd = ss.calculateDistributeFactor(pole_pair)
    k_wp = ss.calculateShortPitchFactor(yq, pole_pair)
    if k_wd is not None:
        k_w = k_wd * k_wp
        rounded_k_w = np.round(k_w, 6)
        print('Kw for {}-pole-pair, {}-slots is {}'.format(pole_pair, Q, rounded_k_w))

    # -----------------------------------
    # MMF 출력
    beta = 30
    current = np.array([np.cos(np.radians(beta)), np.cos(np.radians(beta-120)), np.cos(np.radians(beta-240))])
    #current = np.array([1.0, -0.5, -0.5])
    mmf = MMF(ss)
    print('THD of the backEMF:', mmf.THDforBackEMF(polearc_ratio, 20, yq))
    mmf.plotMMF(current, yq)
    mmf_coefficients, type = mmf.harmonicComponents(current, yq)

    # 극당상당 슬롯수로 하모닉 크기를 표준화시킴
    npp = Q / 3 / (2*pole_pair)
    plot_harmonic_magnitudes(mmf_coefficients/npp, type, pole_pair)

    # -------------------------------------
    # Vibration mode check
    modes = np.array([1,2,3,4])
    mode_results_1 = mmf.vibrationModeBySubharmonics(current, yq, modes)
    print('Vibration Mode Check by Sub-harmonics')
    for mode, result in mode_results_1:
        print(f"  Mode: {mode}, Result: {result}")

    mode_results_2 = mmf.vibrationModeByHarmonics(current, yq, polearc_ratio, np.array([1,5,7]), np.array([1,3,5]), modes)
    print('Vibration Mode Check by Harmonics')
    for mode, result in mode_results_2:
        print(f"  Mode: {mode}, Result: {result}")

if __name__ == '__main__':
    main()