# EM/star_of_slots.py
import math
import cmath
from typing import Union
import numpy as np
import string
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch

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

class StarOfSlots:
    def __init__(self, pp: int, N_slots: int, N_phases: int = 3) -> None:
        self._pp = pp
        self._Q = N_slots
        self._m = N_phases

        self._t = 0
        self._feasible = False
        self._based_pattern = None
        self._based_phasor = None
        self._zero_mutual = False
        self._signle_layer = False

        self._t = math.gcd(pp, N_slots)
        mt = self._m * self._t
        self._feasible = self._Q % mt == 0

        if (self._feasible):
            self._based_pattern, self._based_phasor = self.makeBasedPattern()

            # --------------------------------------
            # zero-Mutual Inductance 판별
            # 1. 코일 패턴에서 절대값 기준으로 그룹화 (같은 상 코일로 묶음)
            abs_vals = np.abs(self.pattern)
            unique_vals = np.unique(abs_vals)

            # 2. 각 절대값 그룹의 합 계산 (+코일, -코일 갯수가 같은지?)
            sums = np.array([self.pattern[abs_vals == val].sum() for val in unique_vals])

            # 3. 합산된 값이 모두 0인지 확인 (모든 상의 +-코일 갯수가 같으면 상호인덕턴스 0)
            self._zero_mutual = np.all(sums == 0)

            # --------------------------------------
            # Single Layer Winding 판별
            # 1. 코일피치가 1이 되어야 하고, 코일수가 짝수이어야 함
            suggested_yq = self.suggestYq
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
        return np.tile(self._based_pattern, self._t)

    @property
    def basedPhasors(self):
        return self._based_phasor

    @property
    def zeroMutual(self) -> bool:
        return self._zero_mutual

    @property
    def validateSingleLayerWinding(self) -> bool:
        return self._signle_layer

    @property
    def suggestYq(self) -> int:
        return round(self._Q / (self._pp * 2) - 0.1)

    def makeBasedPattern(self) -> Union[np.ndarray, None]:
        if not self._feasible:
            return None

        # 고려할 슬롯 번호
        qq = np.arange(1, int(self._Q / self._t) + 1)

        # 주어진 극수에 맞는 페이저 (코일 패턴 생성을 위한 기준 페이저)
        based_pp = self._pp
        based_phasor = (360 / self._Q * (qq - 1) * based_pp) % 360

        # 코일 패턴 만들기 (짝수상에 대해서는 고려 안함)
        based_pattern = np.zeros(qq.size, dtype=int)
        section_angle = 180 / self._m
        half_angle = section_angle / 2

        for idx, angle in enumerate(based_phasor):
            for m in range(self._m):
                lower_limit = (2 * section_angle * m - half_angle) % 360
                upper_limit = (2 * section_angle * m + half_angle) % 360
                delta_lower = (angle - lower_limit) % 360
                delta_upper = (upper_limit - angle) % 360
                if delta_lower <= section_angle and delta_upper < section_angle:
                    based_pattern[idx] = (m+1)
                    break

                lower_limit = (2 * section_angle * m - half_angle + 180) % 360
                upper_limit = (2 * section_angle * m + half_angle + 180) % 360
                delta_lower = (angle - lower_limit) % 360
                delta_upper = (upper_limit - angle) % 360
                if delta_lower <= section_angle and delta_upper < section_angle:
                    based_pattern[idx] = -(m+1)
                    break
        
        # 대칭 평형 체크
        phase = np.zeros(self._m, dtype=complex)
        phase_count = np.zeros(self._m, dtype=int)
        for m, angle in zip(based_pattern, based_phasor):
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
        return based_pattern, based_phasor

    def calculateDistributeFactor(self, pp: int = 0) -> Union[np.ndarray, None]:
        if not self.feasible:
            #print('Not allowed pole({})-slot({}) combination'.format(self.P, self.Q))
            return None

        # 고려할 슬롯 번호
        qq = np.arange(1, self._Q + 1)
        pattern = self.pattern

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
        k_w = phase / phase_count
        return k_w

    def calculateShortPitchFactor(self, yq: int, pp: int = 0) -> Union[np.ndarray, None]:
        if not self.feasible:
            #print('Not allowed pole({})-slot({}) combination'.format(self.P, self.Q))
            return None
        
        # 코일피치가 0일수는 없다
        if yq == 0:
            return None

        if pp == 0: pp = self._pp
        coil_pitch = np.radians(360/self._Q * pp * abs(yq))
        k_wp = np.sin(coil_pitch / 2)
        return k_wp

    def getPatterns(self, yq: int=0):
        if not self.feasible:
            return None

        if yq == 0:
            yq = self.nPolePairs

        mapping = create_mapping(self._m)

        # 변환된 결과 리스트
        winding_pattern = {}

        # 1->A 와 같은 변환 작업
        for slotNo, phaseNo in enumerate(self.pattern):
            if phaseNo > 0:
                winding_pattern[slotNo + 1] = mapping[phaseNo]  # 양수는 그대로 매핑
            else:
                winding_pattern[slotNo + 1] = f"-{mapping[abs(phaseNo)]}"  # 음수는 -를 붙임

        return winding_pattern

    def plotStarOfSlots(self) -> None:
        phases = self.basedPhasors

        n_phasors = len(phases)
        radii = np.ones(n_phasors)  # 모든 페이저의 크기를 동일하게 1로 설정

        # 위상을 라디안으로 변환
        angles_rad = np.deg2rad(phases)

        # x, y 좌표 계산
        x_coords = radii * np.cos(angles_rad)
        y_coords = radii * np.sin(angles_rad)

        # 다이어그램 그리기
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.grid(False)  # 기본 격자 제거

        # 색상 리스트 (1, 2, 3 / 4, 5, 6은 대칭적으로 어두운 색상)
        base_colors = ['#FF5733', '#33FF57', '#3357FF']  # 1, 2, 3번 색상
        dark_colors = ['#CC4529', '#29CC45', '#2935CC']  # 4, 5, 6번 대칭 어두운 색상
        section_colors = base_colors + dark_colors

        for i in range(6):
            start_angle = i * 60 - 29  # 시작 각도
            end_angle = start_angle + 60  # 끝 각도
            wedge = Wedge(
                center=(0, 0), r=1.5, theta1=start_angle, theta2=end_angle,
                facecolor=section_colors[i], edgecolor=None, linewidth=0.5, alpha=0.2
            )
            ax.add_patch(wedge)

        # 단위 원 및 눈금자 그리기
        circle = plt.Circle((0, 0), 1.5, color='lightgray', fill=False, linestyle='dotted', linewidth=1)
        ax.add_artist(circle)
        for angle in range(0, 360, 15):  # 30도 간격으로 눈금 추가
            angle_rad = np.deg2rad(angle)
            x_tick = 1.6 * np.cos(angle_rad)
            y_tick = 1.6 * np.sin(angle_rad)
            ax.text(
                x_tick, y_tick,
                f"{angle}°", color='black', fontsize=10, ha='center', va='center'
            )

        # 페이저 그리기
        # 기본 색상 (red, green, blue)
        base_colors = ['#FF0000', '#0000FF', '#00FF00']  # 순서대로 빨강, 파랑, 초록

        # 대칭 어두운 색상 (red, green, blue)
        dark_colors = ['#AA0000', '#0000AA', '#00AA00']  # 순서대로 어두운 빨강, 파랑, 초록

        pattern = self._based_pattern
        for i in range(n_phasors):
            if pattern[i] > 0:
                color = base_colors[pattern[i]-1]
            else:
                color = dark_colors[abs(pattern[i]) - 1]

            ax.arrow(
                0, 0, x_coords[i], y_coords[i],
                head_width=0.05, head_length=0.1, fc=color, ec=color
            )
            ax.text(
                x_coords[i] * 1.2, y_coords[i] * 1.2,  # 화살표 끝점에서 더 떨어진 위치
                f"{i + 1}", color='k', fontsize=12, ha='center', va='center'
            )

        # 축 및 레이블 제거
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')  # x, y축 제거

        '''
        # 범례 추가
        legend_elements = [
            Patch(facecolor=base_colors[0], label='Phase A', edgecolor=None, alpha=0.5),
            Patch(facecolor=base_colors[1], label='Phase B', edgecolor=None, alpha=0.5),
            Patch(facecolor=base_colors[2], label='Phase C', edgecolor=None, alpha=0.5),
            Patch(facecolor=dark_colors[0], label='RePhaseA', edgecolor=None, alpha=0.5),
            Patch(facecolor=dark_colors[1], label='RePhaseB', edgecolor=None, alpha=0.5),
            Patch(facecolor=dark_colors[2], label='RePhaseC', edgecolor=None, alpha=0.5)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, title="Legend")
        '''
        plt.show()
