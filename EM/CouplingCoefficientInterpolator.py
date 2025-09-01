import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

class CouplingCoefficientInterpolator:
    """
    제공된 물리식을 기반으로 Flux Linkage 데이터로부터
    커플링 계수(k_wc)를 계산하고 보간하는 클래스입니다.

    Scipy의 RegularGridInterpolator를 사용합니다.
    """

    def __init__(self, file_path):
        """
        데이터를 로드하고 3D 격자를 생성한 뒤, k_wc 계수를 계산하여
        보간기를 초기화합니다.

        Args:
            file_path (str): 데이터 소스 CSV 파일 경로.
        """
        print("초기화 시작: 데이터를 로드하고 3D 격자를 생성합니다...")
        df = pd.read_csv(file_path)

        # 1. 보간을 위한 격자 좌표 생성
        self.h_coords = sorted(df["Harmonic_order"].unique())
        self.t_coords = sorted(df["tau_so"].unique())
        self.r_coords = sorted(df["SlotPitchRatio"].unique())

        # 2. 데이터를 3D 배열(flux_array)로 변환
        flux_array = np.full((len(self.h_coords), len(self.t_coords), len(self.r_coords)), np.nan)

        # DataFrame을 순회하며 3D 배열 채우기
        for index, row in df.iterrows():
            # 각 값에 해당하는 인덱스 찾기
            i = self.h_coords.index(row["Harmonic_order"])
            j = self.t_coords.index(row["tau_so"])
            k = self.r_coords.index(row["SlotPitchRatio"])
            flux_array[i, j, k] = row["FluxLinkage(Winding)"]

        print("k_wc 계수 계산 중...")
        # 3. 제공된 수식을 사용하여 k_wc 계수 계산 (벡터화 방식 사용)

        # 1차 고조파에 해당하는 flux 값 (기준값)
        base_values = flux_array[0, :, :]

        # 계산을 위해 차원 확장 (Broadcasting 활용)
        h_coords_bc = np.array(self.h_coords)[:, np.newaxis, np.newaxis]
        base_values_bc = base_values[np.newaxis, :, :]

        # 0으로 나누는 것을 방지하며 k_wc 계산
        with np.errstate(divide='ignore', invalid='ignore'):
            k_wc = (flux_array / base_values_bc) * h_coords_bc

        # base_value가 0 또는 NaN이었던 위치를 0으로 처리
        k_wc[np.isnan(k_wc)] = 0

        print("보간기 생성 중...")
        # 4. RegularGridInterpolator 보간기 생성
        self._interpolator = RegularGridInterpolator(
            (self.h_coords, self.t_coords, self.r_coords),
            k_wc,
            method='linear',
            bounds_error=False,  # 격자 밖의 점도 허용
            fill_value=None     #(격자 밖은 가장 가까운 점의 값으로 처리)
        )
        print("초기화 완료.")

    def interpolate(self, harmonic, tau_so, ratio):
        """
        주어진 (고조파, tau_so, 비율) 지점에서 k_wc 값을 보간합니다.

        Args:
            harmonic (int): 보간할 고조파 차수.
            tau_so (float): 보간할 tau_so 값.
            ratio (float): 보간할 SlotPitchRatio 값.

        Returns:
            float: 보간된 k_wc 값.
        """
        point = np.array([harmonic, tau_so, ratio])
        return self._interpolator(point).item()


class CouplingCoefficientInterpolator2:
    """
    CouplingCoefficients2.csv를 읽어 4D 격자를 구성하고,
    초기화 시 모든 고조파에 대해 1차 정규화 후 h를 곱한 k_wc 배열을 생성하는 클래스.
    """

    def __init__(self, file_path):
        """
        데이터를 로드하고 4D 격자를 생성한 뒤, k_wc 배열을 계산하여 보간기를 초기화합니다.

        Args:
            file_path (str): 데이터 소스 CSV 파일 경로.
        """
        print("초기화 시작: 데이터를 로드하고 4D 격자를 생성합니다...")
        df = pd.read_csv(file_path)

        # 1. 보간을 위한 격자 좌표 생성
        self.h_coords = sorted(df["Harmonic_order"].unique())  # 홀수만 존재
        self.tau_so_coords = sorted(df["tau_so"].unique())
        self.tau_g_coords = sorted(df["tau_g"].unique())
        self.r_coords = sorted(df["SlotPitchRatio"].unique())

        # 2. coupling_coefficient 4D 배열 생성
        coupling_array = np.full(
            (len(self.h_coords), len(self.tau_so_coords), len(self.tau_g_coords), len(self.r_coords)),
            np.nan
        )

        for _, row in df.iterrows():
            i = self.h_coords.index(row["Harmonic_order"])
            j = self.tau_so_coords.index(row["tau_so"])
            k = self.tau_g_coords.index(row["tau_g"])
            l = self.r_coords.index(row["SlotPitchRatio"])
            coupling_array[i, j, k, l] = row["coupling_coefficient"]

        # NaN → 0 처리
        coupling_array[np.isnan(coupling_array)] = 0

        print("k_wc 계산 중...")

        # 1차 고조파 값 (Harmonic_order=1에 해당하는 값 찾기)
        try:
            idx_fund = self.h_coords.index(1)
        except ValueError:
            raise ValueError("Harmonic_order=1 (기준값)이 데이터에 없습니다.")

        base_values = coupling_array[idx_fund, :, :, :]  # shape: (tau_so, tau_g, ratio)

        # Broadcasting 준비
        h_coords_bc = np.array(self.h_coords)[:, np.newaxis, np.newaxis, np.newaxis]
        base_values_bc = base_values[np.newaxis, :, :, :]

        # k_wc = (C(h)/C(1)) * h 계산 (분모 0 대비)
        with np.errstate(divide='ignore', invalid='ignore'):
            k_wc = (coupling_array / base_values_bc) * h_coords_bc

        # NaN → 0 처리
        k_wc[np.isnan(k_wc)] = 0

        print("보간기 생성 중...")
        self._interpolator = RegularGridInterpolator(
            (self.h_coords, self.tau_so_coords, self.tau_g_coords, self.r_coords),
            k_wc,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        print("초기화 완료.")

    def interpolate(self, harmonic, tau_so, tau_g, ratio):
        """
        주어진 (harmonic, tau_so, tau_g, ratio) 지점에서
        (C(h)/C(1))*h로 계산된 k_wc 값을 보간하여 반환.

        Args:
            harmonic (int): 고조파 차수(홀수).
            tau_so (float): tau_so 값.
            tau_g (float): tau_g 값.
            ratio (float): SlotPitchRatio 값.

        Returns:
            float: k_wc 값.
        """
        point = np.array([harmonic, tau_so, tau_g, ratio])
        return self._interpolator(point).item()

    def plot_harmonics(self, harmonics, tau_g, levels=20, cmap='viridis'):
        """
        선택한 하모닉들의 k_wc 값을 SlotPitchRatio vs tau_so 2D contour로 플로팅.

        Args:
            harmonics (list[int]): 플로팅할 고조파 리스트 (예: [1, 3, 5]).
            tau_g (float): 고정할 tau_g 값.
            levels (int): contour 레벨 수.
            cmap (str): 컬러맵.
        """
        fig, axes = plt.subplots(1, len(harmonics), figsize=(6*len(harmonics), 5), constrained_layout=True)

        # harmonics 개수가 1이면 axes를 리스트로 변환
        if len(harmonics) == 1:
            axes = [axes]

        tau_so_vals = np.array(self.tau_so_coords)
        ratio_vals = np.array(self.r_coords)
        X, Y = np.meshgrid(ratio_vals, tau_so_vals)  # X=SlotPitchRatio, Y=tau_so

        for ax, h in zip(axes, harmonics):
            Z = np.zeros_like(X)
            for i, tau_so in enumerate(tau_so_vals):
                for j, ratio in enumerate(ratio_vals):
                    Z[i, j] = self.interpolate(h, tau_so, tau_g, ratio)

            contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
            ax.set_title(f"Harmonic {h}, tau_g={tau_g}")
            ax.set_xlabel("SlotPitchRatio")
            ax.set_ylabel("tau_so")
            fig.colorbar(contour, ax=ax)

        plt.show()

    def plot_harmonics_3d(self, harmonics, tau_g, cmap='viridis'):
        """
        선택한 하모닉들의 k_wc 값을 SlotPitchRatio vs tau_so vs 값 3D surface로 플로팅.

        Args:
            harmonics (list[int]): 플로팅할 고조파 리스트 (예: [1, 3, 5]).
            tau_g (float): 고정할 tau_g 값.
            cmap (str): 컬러맵.
        """
        fig = plt.figure(figsize=(7*len(harmonics), 6))

        tau_so_vals = np.array(self.tau_so_coords)
        ratio_vals = np.array(self.r_coords)
        X, Y = np.meshgrid(ratio_vals, tau_so_vals)  # X=SlotPitchRatio, Y=tau_so

        for idx, h in enumerate(harmonics, 1):
            Z = np.zeros_like(X)
            for i, tau_so in enumerate(tau_so_vals):
                for j, ratio in enumerate(ratio_vals):
                    Z[i, j] = self.interpolate(h, tau_so, tau_g, ratio)

            ax = fig.add_subplot(1, len(harmonics), idx, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none')
            ax.set_title(f"Harmonic {h}, tau_g={tau_g}")
            ax.set_xlabel("SlotPitchRatio")
            ax.set_ylabel("tau_so")
            ax.set_zlabel("k_wc", rotation=90)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.15)

        plt.show()