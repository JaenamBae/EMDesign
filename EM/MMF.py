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

class MMF:
    def __init__(self, ss: StarOfSlots) -> None:
        self.ss = ss

    def harmonicComponents(self, current: np.array, yq: int=1, n_terms: int = 100) -> tuple[np.array, np.array]:
        nPoles = self.ss.nPoles
        nSlots = self.ss.nSlots
        nPhases = self.ss.nPhases

        slot_pitch = 360 / nSlots
        coil_pitch = slot_pitch * yq
        coefficients = np.zeros((nPhases,n_terms+1), dtype=np.complex128)

        pattern = self.ss.pattern
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

    def THDforBackEMF(self, polearc_ratio: float, n_terms: int, yq:int=1) -> Union[np.array, None]:
        if self.ss.pattern is None:
            return None

        # 단절계수 구하기
        k_wp = self.ss.calculateShortPitchFactor(yq)

        # 계자 자극의 크기(radE)
        w = np.pi * polearc_ratio

        # 계자 공극 자속밀도 분포의 FFT 계수 (극호율의 함수로만 표현함)
        # 홀수 차수만 고려한다
        harmonics = np.arange(1, n_terms * 2, 2)
        coefficients = 1. / (np.pi * harmonics) * (np.cos(harmonics*(np.pi - w)/2) - np.cos(harmonics*(np.pi + w)/2))

        thd_emf = np.zeros(self.ss.nPhases, dtype=float)
        emf_1 = np.zeros(self.ss.nPhases, dtype=float)
        pp = self.ss.nPolePairs
        for n, n_harmonic in enumerate(harmonics):
            # 홀수 고조파에 대해서만 다룬다
            pp_harmonic = int(n_harmonic * pp)

            # 공극자속밀도가 구형파 분포를 가진다 가정하면,
            # 공극자속밀도의 고조파 크기는 차수에 반비례함
            # 이러한 공극자속밀도의 고조파가 실제 쇄교자속에 미치는 영향은 고조파의 크기 및 고조파 권선계수의 곱으로 표현가능함
            k_wd = self.ss.calculateDistributeFactor(pp_harmonic)
            emf = k_wd * k_wp * coefficients[n]
            if n == 0:
                emf_1 = emf
            else:
                thd_emf += emf**2

        thd = np.sqrt(thd_emf) / abs(emf_1)
        return thd

    def vibrationModeBySubharmonics(self, current: np.array, yq: int,
                                    check_mode: np.array=np.array([1,2,3,4]),
                                    threshold: float=0.1) -> tuple[np.array, bool]:
        """
        Check vibration mode by subharmonics.

        Parameters:
            current (float): phase current for the 3-phase windings
            yq: coil throw
            check_mode (np.ndarray[int]): vibration mode for checking
            threshold (float): threshold ratio for magnitude of vibration mode based on fundamental harmonic

        Returns:
            dict: {mode number of vibration: validation results of the mode}
        """
        mmf_coefficients, type = self.harmonicComponents(current, yq)
        result = np.zeros(check_mode.shape, dtype=bool)

        pp = self.ss.nPolePairs
        m1 = mmf_coefficients[pp]
        normalized_coefficients = mmf_coefficients / m1

        for idx, mode in enumerate(check_mode):
            index = pp - mode
            if index >= 0:
                result[idx] |= (normalized_coefficients[index] > threshold)
            index = pp + mode
            if index < normalized_coefficients.shape:
                result[idx] |= (normalized_coefficients[index] > threshold)

        return zip(check_mode, result)

    def vibrationModeByHarmonics(self, current: np.array, yq: int, polearc_ratio: float,
                                 with_mmf_harmonics: np.array, with_pole_harmonics: np.array,
                                 check_mode: np.array = np.array([1, 2, 3, 4]),
                                 threshold: float = 0.1) -> tuple[np.array, bool]:
        pp = self.ss.nPolePairs
        Q = self.ss.nSlots

        result = np.zeros(check_mode.shape, dtype=bool)

        # 계자 공극 자속밀도 분포의 FFT 계수 (극호율의 함수로만 표현함); with_pole_harmonics 성분만 계산
        w = np.pi * polearc_ratio
        pole1 = 1. / np.pi * (np.cos((np.pi - w) / 2) - np.cos((np.pi + w) / 2))
        pole_coefficients = 1. / (np.pi * with_pole_harmonics) * (np.cos(with_pole_harmonics * (np.pi - w) / 2) - np.cos(
            with_pole_harmonics * (np.pi + w) / 2)) / pole1

        # 기자력 FFT계수; with_mmf_harmonics 성분만 계산
        max_order = np.max(with_mmf_harmonics) * pp
        mmf_coeffs, _ = self.harmonicComponents(current, yq, max_order)
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

    def plotMMF(self, current, yq: int = 1) -> None:
        nPoles = self.ss.nPoles
        nSlots = self.ss.nSlots
        nPhases = self.ss.nPhases

        slot_pitch = 360 / nSlots
        coil_pitch = slot_pitch * yq

        # 파형 그리기 I - 직접 그리기
        t, summed_waveform = generate_pulse_waveform(360, 0, 0, 0)
        pattern = self.ss.pattern
        for i, phase in enumerate(pattern):
            i_ph = current[abs(phase) - 1]
            _, waveform = generate_pulse_waveform(360, coil_pitch, i*slot_pitch, i_ph)
            if phase > 0:
                summed_waveform += waveform
            else:
                summed_waveform -= waveform

        # 파형 그리기 II - 푸리에 계수를 활용한 신호 복원
        coefficients, _ = self.harmonicComponents(current, yq)
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
        plt.plot(t, reconstructed_waveform, label="MMF Waveform (fourier reconstructed)", color="blue", linewidth=2)
        plt.plot(t, summed_waveform, label="MMF Waveform", color="red", linewidth=2)
        plt.xlabel("Angle (degrees)")
        plt.ylabel("MMF")
        plt.legend()
        plt.grid(True)
        plt.show()
