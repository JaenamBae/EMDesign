# EM/star_of_slots.py
import math
import cmath
from typing import Union
import numpy as np

class StarOfSlots:
    def __init__(self, pp: int, N_slots: int, N_phases: int = 3) -> None:
        self._pp = pp
        self._Q = N_slots
        self._m = N_phases

        self._t = 0
        self._feasible = False
        self._pattern = None
        self._zero_mutual = False
        self._signle_layer = False

        self._t = math.gcd(pp, N_slots)
        mt = self._m * self._t
        self._feasible = self._Q % mt == 0

        if (self._feasible):
            self._pattern = self.makeBasedPattern()

            # --------------------------------------
            # zero-Mutual Inductance 판별
            # 1. 코일 패턴에서 절대값 기준으로 그룹화 (같은 상 코일로 묶음)
            abs_vals = np.abs(self._pattern)
            unique_vals = np.unique(abs_vals)

            # 2. 각 절대값 그룹의 합 계산 (+코일, -코일 갯수가 같은지?)
            sums = np.array([self._pattern[abs_vals == val].sum() for val in unique_vals])

            # 3. 합산된 값이 모두 0인지 확인 (모든 상의 +-코일 갯수가 같으면 상호인덕턴스 0)
            self._zero_mutual = np.all(sums == 0)

            # --------------------------------------
            # Single Layer Winding 판별
            # 1. 코일피치가 1이 되어야 하고, 코일수가 짝수이어야 함
            suggested_yq = round(self._Q / (self._pp*2))
            self._signle_layer = (suggested_yq <= 1) and (self._Q % 2 == 0)

    @property
    def nPolePairs(self) -> int:
        return self._pp

    @property
    def nPoles(self) -> int:
        return self._pp * 2

    @property
    def nSlots(self) -> int:
        return self._Q

    @property
    def nPhases(self) -> int:
        return self._m

    @property
    def periodicity(self) -> int:
        return self._t

    @property
    def feasible(self) -> bool:
        return self._feasible

    @property
    def pattern(self):
        return self._pattern

    @property
    def zeroMutual(self) -> bool:
        return self._zero_mutual

    @property
    def validateSingleLayerWinding(self) -> bool:
        return self._signle_layer

    @property
    def suggestYq(self) -> int:
        return round(self._Q / (self._pp * 2))

    def makeBasedPattern(self) -> Union[np.ndarray, None]:
        if not self._feasible:
            return None

        # 고려할 슬롯 번호
        qq = np.arange(1, int(self._Q / self._t) + 1)

        # 주어진 극수에 맞는 페이저 (코일 패턴 생성을 위한 기준 페이저)
        based_pp = self._pp
        based_phasor = (360 / self._Q * (qq - 1) * based_pp) % 360

        # 코일 패턴 만들기 (짝수상에 대해서는 고려 안함)
        pattern = np.zeros(qq.size, dtype=int)
        section_angle = 180 / self._m
        half_angle = section_angle / 2

        for idx, angle in enumerate(based_phasor):
            for m in range(self._m):
                lower_limit = (2 * section_angle * m - half_angle) % 360
                upper_limit = (2 * section_angle * m + half_angle) % 360
                delta_lower = (angle - lower_limit) % 360
                delta_upper = (upper_limit - angle) % 360
                if delta_lower <= section_angle and delta_upper < section_angle:
                    pattern[idx] = (m+1)
                    break

                lower_limit = (2 * section_angle * m - half_angle + 180) % 360
                upper_limit = (2 * section_angle * m + half_angle + 180) % 360
                delta_lower = (angle - lower_limit) % 360
                delta_upper = (upper_limit - angle) % 360
                if delta_lower <= section_angle and delta_upper < section_angle:
                    pattern[idx] = -(m+1)
                    break
        
        # 대칭 평형 체크
        phase = np.zeros(self._m, dtype=complex)
        phase_count = np.zeros(self._m, dtype=int)
        for m, angle in zip(pattern, based_phasor):
            if m > 0:
                phase[m-1] += cmath.rect(1, np.radians(angle))
                phase_count[m-1] += 1
            else:
                phase[-m-1] += cmath.rect(1, np.radians(angle+180))
                phase_count[-m-1] += 1

        # 모든 상권선이 동일개의 코일로 이루어져 있는지 판별
        all_equal = np.all(phase_count == phase_count[0])
        if not all_equal:
            print('Unbalanced Windings!')
            return None

        # 각 상이 서로 동일한 각도만큼 차이 나는지 판단
        # 상권선의 페이저를 링으로 생각해서 각 상끼리의 위상차를 계산함
        phase_angle = np.angle(phase, deg=True)
        angle_differences = (phase_angle - np.roll(phase_angle, 1)) % 360
        #print('Differences:', angle_differences)

        phase_difference = 360 / self._m
        is_within_tolerance = np.abs(angle_differences - phase_difference) <= 0.001
        if not np.all(is_within_tolerance):
            print('Asymmetric Windings!')
            return None
        
        # 슬롯수에 맞게 주기수 만큼 복제
        return np.tile(pattern, self._t)

    def calculateDistributeFactor(self, pp: int = 0) -> Union[np.ndarray, None]:
        if self.pattern is None:
            #print('Not allowed pole({})-slot({}) combination'.format(self.P, self.Q))
            return None

        # 고려할 슬롯 번호
        qq = np.arange(1, self._Q + 1)
        pattern = self._pattern

        # 해당 극 기준 슬롯의 위상을 담을 변수
        if pp == 0: pp = self._pp
        phasor = (360 / self._Q * (qq - 1) * pp) % 360

        # 상권선별 분포계수 구하기 (밑작업)
        phase = np.zeros(self._m, dtype=complex)
        phase_count = np.zeros(self._m, dtype=int)
        for m, angle in zip(pattern, phasor):
            if m > 0:
                phase[m-1] += cmath.rect(1, np.radians(angle))
                phase_count[m-1] += 1
            else:
                phase[-m-1] += cmath.rect(1, np.radians(angle+180))
                phase_count[-m-1] += 1

        # 분포계수를 구함
        k_w = np.abs(phase) / phase_count
        return k_w

    def calculateShortPitchFactor(self, yq: int, pp: int = 0) -> Union[np.ndarray, None]:
        if self._pattern is None:
            #print('Not allowed pole({})-slot({}) combination'.format(self.P, self.Q))
            return None
        
        # 코일피치가 0일수는 없다
        if yq == 0:
            return None

        if pp == 0: pp = self._pp
        coil_pitch = 2 * pp / self._Q * abs(yq)
        k_wp = np.abs(np.sin(coil_pitch * np.pi / 2))
        return k_wp

    def THDforWindingFactor(self, harmonics: int, yq:int=1) -> Union[np.ndarray, None]:
        if self._pattern is None:
            return None

        # 단절계수 구하기
        k_wp = self.calculateShortPitchFactor(yq)

        thd_kw = np.zeros(self._m, dtype=float)
        k_w1 = np.zeros(self._m, dtype=float)
        pp = self.nPolePairs
        for n in range(harmonics):
            # 홀수 고조파에 대해서만 다룬다
            n_harmonics = 2*n + 1
            pp_harmonic = int(n_harmonics * pp)

            # 공극자속밀도가 구형파 분포를 가진다 가정하면,
            # 공극자속밀도의 고조파 크기는 차수에 반비례함
            # 이러한 공극자속밀도의 고조파가 실제 쇄교자속에 미치는 영향은 고조파의 크기 및 고조파 권선계수의 곱으로 표현가능함
            k_w = self.calculateDistributeFactor(pp_harmonic) * k_wp / n_harmonics
            if n == 0:
                k_w1 = k_w
            else:
                thd_kw += k_w**2

        thd = np.sqrt(thd_kw) / k_w1
        return thd