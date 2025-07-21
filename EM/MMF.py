import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from .StarOfSlots import StarOfSlots

def generate_pulse_waveform(period: float, pulse_width: float, phase_shift: float=0, magnitude: float=1, num_points: int=1000):
    """
    Generates a pulse waveform with a given period, pulse width, and phase shift.

    Parameters:
        period (float): Period of the waveform.
        pulse_width (float): Pulse width.
        phase_shift (float): Phase shift in degrees.
        magnitude (float): Magnitude of the pulse.
        num_points (int): Number of points in the waveform.

    Returns:
        t (np.ndarray): Time array.
        waveform (np.ndarray): Generated pulse waveform.
    """
    t = np.linspace(0, period, num_points, endpoint=False)
    phase_shift_rad = np.radians(phase_shift)
    shifted_t = (t + phase_shift_rad * period / (2 * np.pi)) % period
    waveform = np.where(shifted_t < pulse_width, magnitude, 0.0)
    return t, waveform

def pulsewave_fourier_coefficients(a, w, period=360, n_terms=100):
    """
    Compute the Fourier coefficients c_n for a pulse wave. n >= 0

    Parameters:
        a (float): Start point of the pulse.
        w (float): Width of the pulse.
        period (float): Period of the wave (default: 360).
        n_terms (int): Number of Fourier terms to compute.

    Returns:
        dict: Fourier coefficients {n: c_n} for n in [0, ..., n_terms].
    """
    # Fundamental angular frequency
    omega = 2 * np.pi / period

    # Store coefficients in a dictionary
    coefficients = np.zeros(n_terms+1, dtype=np.complex128)

    # Compute c_0 (constant term)
    coefficients[0] = w / period

    # Compute c_n for n > 0 only
    for n in range(1, n_terms + 1):
        sin_term = np.sin(np.pi * n * w / period)
        phase_shift = np.exp(-1j * omega * n * (a + w / 2))
        c_n = (1 / (np.pi * n)) * sin_term * phase_shift
        coefficients[n] = c_n

    return coefficients

def trapezoidal_fourier_coefficients(w, num_terms=10):
    """
    사다리꼴파형의 푸리에 계수 계산 (주기 2*pi).

    Parameters:
        w (float): 상단 평평한 너비 (단위: radians, w < pi)
        num_terms (int): 계산할 푸리에 계수의 개수

    Returns:
        b_n (np.ndarray): 푸리에 계수 배열 (n = 1, 2, ..., num_terms)
    """
    b_n = np.zeros(num_terms)
    slope1 = 1 / (np.pi / 2 - w / 2)  # 구간 1의 기울기
    slope3 = -1 / (np.pi / 2 - w / 2)  # 구간 3의 기울기

    harmonic_order = range(1, num_terms + 1)
    for n in harmonic_order:
        # 구간 1: 선형 증가
        term1 = (1 / np.pi) * slope1 * (
                (-np.cos(n * (np.pi / 2 - w / 2)) + np.cos(n * (w / 2 - np.pi / 2))) / n
                + (np.sin(n * (np.pi / 2 - w / 2)) - np.sin(n * (w / 2 - np.pi / 2))) / n ** 2
        )

        # 구간 2: 상수
        term2 = (1 / np.pi) * (
                (-np.cos(n * (np.pi / 2 + w / 2)) + np.cos(n * (np.pi / 2 - w / 2))) / n / slope1
        )

        # 구간 3: 선형 감소
        term3 = (1 / np.pi) * slope3 * (
                (-np.cos(n * (3 * np.pi / 2 - w / 2)) + np.cos(n * (np.pi / 2 + w / 2))) / n
                + (np.sin(n * (3 * np.pi / 2 - w / 2)) - np.sin(n * (np.pi / 2 + w / 2))) / n ** 2
        )

        # 구간 4: 상수
        term4 = (1 / np.pi) * (
                (np.cos(n * (3 * np.pi / 2 + w / 2)) - np.cos(n * (3 * np.pi / 2 - w / 2))) / n / slope3
        )

        # 합산
        b_n[n - 1] = term1 + term2 + term3 + term4

    return b_n, harmonic_order

class MMF:
    def __init__(self, ss: StarOfSlots, current: np.array, yq: int=0) -> None:
        self._ss = ss
        self._current = current
        self._yq = yq
        if yq == 0:
            self._yq = ss.suggestYq

    def harmonicComponents(self, n_terms: int = 100) -> tuple[np.array, np.array]:
        nSlots = self._ss.nSlots
        nPhases = self._ss.nPhases
        current = self._current
        yq = self._yq

        slot_pitch = 360 / nSlots
        coil_pitch = slot_pitch * yq
        coefficients = np.zeros((nPhases,n_terms+1), dtype=np.complex128)

        pattern = self._ss.pattern
        for i, phase in enumerate(pattern):
            alpha = (-i * slot_pitch) % 360
            if phase > 0:
                coefficients[phase-1] += pulsewave_fourier_coefficients(alpha, coil_pitch, 360, n_terms)
            else:
                coefficients[-phase-1] -= pulsewave_fourier_coefficients(alpha, coil_pitch, 360, n_terms)

        sum_coefficient = np.sum(coefficients * current[:, np.newaxis], axis=0)

        phase_type = np.zeros(n_terms + 1)
        for j, column in enumerate(coefficients.T):
            phase = np.angle(column)
            phase_differences = np.diff(phase, append=phase[0])
            phase_differences = (phase_differences + 2 * np.pi) % (2 * np.pi)
            phase_deg = np.degrees(phase_differences)

            are_values_equal_000 = np.allclose(phase_deg, 000, atol=1e-1)
            are_values_equal_120 = np.allclose(phase_deg, 120, atol=1e-1)
            are_values_equal_240 = np.allclose(phase_deg, 240, atol=1e-1)

            if are_values_equal_000 :
                phase_type[j] = 1
            elif are_values_equal_120 :
                phase_type[j] = 2
            elif are_values_equal_240 :
                phase_type[j] = 3

        return sum_coefficient, phase_type

    def harmonicsForBackEMF(self, Bg_FFT: np.array) -> Union[np.array, None]:
        if not self._ss.feasible:
            return None

        emf_harmonic = np.zeros_like(Bg_FFT)
        pp = self._ss.nPolePairs
        for n, Bgn_FFT in enumerate(Bg_FFT):
            if n == 0: continue

            # 전기각 고려한 하모닉 차수
            pp_harmonic = int(n * pp)

            # 공극자속밀도의 고조파가 실제 쇄교자속에 미치는 영향은 고조파의 크기 및 고조파 권선계수의 곱으로 표현 가능함
            # 분포계수 구하기
            k_wd = self._ss.calculateDistributeFactor(pp_harmonic)

            # 단절계수 구하기
            k_wp = self._ss.calculateShortPitchFactor(self._yq, pp_harmonic)

            # 결합계수 구하기
            k_wc = self._ss.calculateCouplingFactor(pp_harmonic)

            # 쇄교자속에 대한 고조파의 영향도
            emf_harmonic[n] = k_wd * k_wp * k_wc * Bgn_FFT

            print('Winding factor for harmonic {}: K_wd: {}, K_wp: {}'.format(n, k_wd, k_wp))

        return emf_harmonic

    def THDforBackEMF(self, Bg_FFT: np.array) -> Union[float, None]:
        if not self._ss.feasible:
            return None

        # 홀수 고조파에 대한 계수만 계산된다
        harmonics_emf = self.harmonicsForBackEMF(Bg_FFT)
        thd_emf = 0
        emf_1 = 0
        for n, emf in enumerate(harmonics_emf):
            if n == 1:
                emf_1 = emf
            else:
                thd_emf += emf**2

        thd = np.abs(np.sqrt(thd_emf) / emf_1)
        return thd

    def vibrationModeBySubharmonics(self, check_mode: np.array=np.array([1,2,3,4]),
                                    threshold: float=0.1) -> tuple[np.array, bool]:
        """
        Check vibration mode by subharmonics.

        Parameters:
            check_mode (np.ndarray[int]): vibration mode for checking
            threshold (float): threshold ratio for magnitude of vibration mode based on fundamental harmonic

        Returns:
            dict: {mode number of vibration: validation results of the mode}
        """
        current = self._current
        yq = self._yq
        mmf_coefficients, type = self.harmonicComponents()
        result = np.zeros(check_mode.shape, dtype=bool)

        pp = self._ss.nPolePairs
        m1 = np.abs(mmf_coefficients[pp])
        normalized_coefficients = np.abs(mmf_coefficients) / m1

        for idx, mode in enumerate(check_mode):
            index = pp - mode
            if index >= 0:
                result[idx] |= (normalized_coefficients[index] > threshold)
            index = pp + mode
            if index < normalized_coefficients.shape:
                result[idx] |= (normalized_coefficients[index] > threshold)

        return zip(check_mode, result)

    def vibrationModeByHarmonics(self, Bg_FFT: np.array,
                                 with_mmf_harmonics: np.array, with_pole_harmonics: np.array,
                                 check_mode: np.array = np.array([1, 2, 3, 4]),
                                 threshold: float = 0.1) -> tuple[np.array, bool]:
        current = self._current
        yq = self._yq
        pp = self._ss.nPolePairs
        Q = self._ss.nSlots

        result = np.zeros(check_mode.shape, dtype=bool)

        # 계자 공극 자속밀도 분포의 FFT 계수 (극호율의 함수로만 표현함); with_pole_harmonics 성분만 계산
        pole_coefficients = Bg_FFT[with_pole_harmonics]

        # 기자력 FFT계수; with_mmf_harmonics 성분만 계산
        max_order = np.max(with_mmf_harmonics) * pp
        mmf_coeffs, _ = self.harmonicComponents(max_order)
        mmf1 = mmf_coeffs[pp]
        normalized_coefficients = mmf_coeffs / mmf1
        mmf_coefficients = normalized_coefficients[with_mmf_harmonics*pp]

        # 기자력과 계자 하모닉에 의한 진동모드를 고려한다
        for i, mmf_coeff in enumerate(mmf_coefficients):
            for j, pole_coeff in enumerate(pole_coefficients):
                case1 = abs(2 * mmf_coeff * pp - Q)     # MMF(슬롯 하모닉 포함)에 의한 진동모드
                case2 = abs(2 * pole_coeff * pp - Q)    # 자극(슬롯 하모닉 포함)에 의한 진동모드
                case3 = abs((mmf_coeff + pole_coeff) * pp - Q) # MMF와 자극의 상호작용에 의한 진동모드

                # 체크할 진동모드에 대해 판단하되, 진동모드가 존재하고 가진력의 크기가 일정크기 이상이면 진동가능서 있다고 판단
                for idx, mode in enumerate(check_mode):
                    if case1 == mode:
                        result[idx] |= (np.abs(mmf_coefficients[i]) > threshold)    
                    if case2 == mode:
                        result[idx] |= (np.abs(pole_coefficients[j]) > threshold)
                    if case3 == mode:
                        result[idx] |= (np.sqrt(np.abs(mmf_coefficients[i]) * np.abs(pole_coefficients[j])) > threshold)

        return zip(check_mode, result)

    def plotMMF(self) -> None:
        current = self._current
        yq = self._yq
        nSlots = self._ss.nSlots

        slot_pitch = 360 / nSlots
        coil_pitch = slot_pitch * yq

        # 파형 그리기 I - 직접 그리기
        t, summed_waveform = generate_pulse_waveform(360, 0, 0, 0)
        pattern = self._ss.pattern
        for i, phase in enumerate(pattern):
            i_ph = current[abs(phase) - 1]
            _, waveform = generate_pulse_waveform(360, coil_pitch, i*slot_pitch, i_ph)
            if phase > 0:
                summed_waveform += waveform
            else:
                summed_waveform -= waveform

        # 파형 그리기 II - 푸리에 계수를 활용한 신호 복원
        coefficients, _ = self.harmonicComponents()
        omega0 = 2 * np.pi / 360  # 기본 주파수
        signal = np.zeros_like(t, dtype=np.complex128)

        for n, cn in enumerate(coefficients):
            signal += cn * np.exp(1j * n * omega0 * t)

            # DC성분을 제외하고 켤레복소수에 대한 복원 작업을 진행함
            if n != 0:
                cn_conj = np.conj(cn)
                signal += cn_conj * np.exp(-1j * n * omega0 * t)

        # 신호의 실수 부분만 추출
        reconstructed_waveform = np.real(signal)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(t, reconstructed_waveform, label="MMF Waveform (fourier reconstructed)", color="blue", linewidth=3)
        plt.plot(t, summed_waveform, label="MMF Waveform", color="red", linewidth=2)
        plt.xlabel("Mechanical Angle (degrees)")
        plt.ylabel("MMF")
        plt.xlim([0, 360])
        plt.xticks(np.arange(0, 360, 60))
        plt.axhline(0, 0, 360, color='k', linestyle='-')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plotHarmonics(self) -> None:
        pp = self._ss.nPolePairs
        Q = self._ss.nSlots
        npp = Q / 3 / (2 * pp)
        coeffs, type = self.harmonicComponents()
        coefficients = coeffs / npp

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
                stem[1].set_linewidth(5)  # 스템선의 굵기를 2로 설정

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

        plt.title("Magnitudes of Harmonic Components (n = 0 to positive harmonics)")
        plt.xlabel("Harmonic Number (n)")
        plt.ylabel("Magnitude (2|c_n|)")
        plt.grid(True)
        plt.show()

    def plotBackEMF(self, Bg_FFT: np.array, even:bool, with_waveform=True, with_fft=True) -> None:
        # even 함수(cos 기반 함수)로 복원할 것임
        n_sample = 3600  # 샘플링 데이터 수
        theta = 360 * np.arange(n_sample) / n_sample
        emf = np.zeros(n_sample)

        coefficients = self.harmonicsForBackEMF(Bg_FFT)
        new_coefficients = np.zeros_like(coefficients)
        if even:  # 계수가 an인 경우 (코싸인 파형에 대한 계수인 경우)
            new_coefficients = coefficients
        else:   # 계수가 bn인 경우 (싸인 파형에 대한 계수인 경우)
            for n in range(1, len(coefficients)):
                sign = (-1) ** ((n - 1) // 2)
                new_coefficients[n] = coefficients[n] * sign

        for n, cn in enumerate(new_coefficients):
            # even (코싸인) 파형으로 복원함
            emf_n = cn * np.cos(np.radians(n * theta))
            emf = emf + emf_n

        # Plot
        if with_waveform:
            plt.figure(figsize=(12, 6))
            plt.plot(theta, emf, color="blue", linewidth=3)
            plt.title("back-EMF Waveform")
            plt.xlabel("Electric Angle (degrees)")
            plt.ylabel("back EMF [pu]")
            plt.xlim([0, 360])
            plt.xticks(np.arange(0, 360, 60))
            plt.grid(True)
            plt.show()


        if with_fft:
            # 하모닉 차수 생성 (0부터 시작)
            harmonic_order = np.arange(len(new_coefficients))

            # 크기 계산 (복소수인 경우 절댓값 사용)
            magnitudes = new_coefficients / new_coefficients[1]
            #print(magnitudes)

            # 그래프 그리기
            odd_mask = (harmonic_order % 2 == 1) & (harmonic_order <= 22)
            odd_orders = harmonic_order[odd_mask]
            odd_magnitudes = magnitudes[odd_mask]

            plt.figure(figsize=(8, 5))
            plt.stem(odd_orders, odd_magnitudes, basefmt="k-")

            # 막대 위에 퍼센트 값 표시 (소수점 둘째 자리까지)
            for x, y in zip(odd_orders, odd_magnitudes):
                percent = y * 100
                plt.text(x, y + 0.02, f"{percent:.2f}%", ha='center', va='bottom', fontsize=9)

            plt.title("back-EMF Harmonics")
            plt.xlabel("Harmonic Order")
            plt.ylabel("Magnitude [pu]")
            plt.grid(True)
            plt.xticks(odd_orders)  # ex) [1, 3, 5, 7, ...]

            plt.show()