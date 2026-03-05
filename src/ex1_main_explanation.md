# `ex1_main.py` 代码详细解释

本文档按照 `ex1_main.py` 中主函数（`if __name__ == "__main__":` 块）的实际运行顺序，对文件中每一行代码进行详细说明，涉及计算的部分同时给出对应的数学公式。

---

## 目录

1. [模块导入](#1-模块导入)
2. [辅助函数与类定义概览](#2-辅助函数与类定义概览)
3. [主函数入口](#3-主函数入口行1305-1360)
4. [网格创建函数 `create_grid`](#4-网格创建函数-create_grid行46-79)
5. [模型实例化 `Example1Model.__init__`](#5-模型实例化-example1model__init__行90-93)
6. [字段初始化 `_set_fields`](#6-字段初始化-_set_fields行1199-1234)
7. [仿真准备 `prepare_simulation`](#7-仿真准备-prepare_simulation行738-751)
   - 7.1 [网格设置 `create_grid`（方法）](#71-网格设置-create_grid方法行95-113)
   - 7.2 [时间参数 `_set_time_parameters`](#72-时间参数-_set_time_parameters行1154-1168)
   - 7.3 [岩石与流体参数 `_set_rock_and_fluid`](#73-岩石与流体参数-_set_rock_and_fluid行442-448)
   - 7.4 [初始条件 `_initial_condition`](#74-初始条件-_initial_condition行1112-1141)
   - 7.5 [参数设置 `_set_parameters`](#75-参数设置-_set_parameters)
   - 7.6 [离散化与线性求解器初始化](#76-离散化与线性求解器初始化)
8. [时间步进与牛顿迭代](#8-时间步进与牛顿迭代)
   - 8.1 [牛顿循环前处理 `before_newton_loop`](#81-牛顿循环前处理-before_newton_loop行753-755)
   - 8.2 [每次迭代前处理 `before_newton_iteration`](#82-每次迭代前处理-before_newton_iteration行757-782)
   - 8.3 [线性系统组装与求解 `assemble_and_solve_linear_system`](#83-线性系统组装与求解-assemble_and_solve_linear_system行796-817)
   - 8.4 [收敛性检验 `check_convergence`](#84-收敛性检验-check_convergence行819-948)
   - 8.5 [牛顿收敛后处理 `after_newton_convergence`](#85-牛顿收敛后处理-after_newton_convergence行786-794)
9. [数据导出与保存](#9-数据导出与保存)
   - 9.1 [导出步骤 `_export_step`](#91-导出步骤-_export_step行956-1021)
   - 9.2 [数据保存 `_save_data`](#92-数据保存-_save_data行1030-1110)
10. [辅助类 `Water` 与 `Granite`](#10-辅助类-water-与-granite行1249-1303)

---

## 1. 模块导入

```python
# 行 31-43
import logging
import os
import time

import numpy as np
import porepy as pp
import scipy.sparse.linalg as spla
from porepy.models.thm_model import THM

import thm_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

**说明：**

| 行号 | 说明 |
|------|------|
| 31 | 导入 `logging`，用于记录运行日志 |
| 32 | 导入 `os`，用于文件目录操作 |
| 33 | 导入 `time`，用于计时 |
| 35 | 导入 NumPy，用于数组和数值运算 |
| 36 | 导入 PorePy，多孔介质数值计算框架 |
| 37 | 导入 SciPy 稀疏线性代数模块，用于求解稀疏线性方程组 |
| 38 | 从 PorePy 导入 THM（热-水-力学耦合）模型基类 |
| 40 | 导入本地工具模块 `thm_utils` |
| 42 | 配置日志级别为 INFO |
| 43 | 创建当前模块的日志记录器 |

---

## 2. 辅助函数与类定义概览

文件中定义了以下主要内容（详细内容在后续章节按调用顺序展开）：

- **`create_grid(mesh_args)`**（行 46–79）：顶层函数，创建二维分形网格。
- **`Example1Model(THM)`**（行 82–1247）：示例模型类，继承自 PorePy 的 `THM` 基类，封装了所有参数设置、边界条件、数值方法等。
- **`Water`**（行 1249–1283）：流体相参数类。
- **`Granite(pp.Granite)`**（行 1286–1302）：固体相参数类，继承自 PorePy 的花岗岩类。

---

## 3. 主函数入口（行 1305–1360）

```python
# 行 1305-1360
if __name__ == "__main__":
    mesh_size = 0.8
    mesh_args = {
        "mesh_size_frac": mesh_size,
        "mesh_size_min": 0.5 * mesh_size,
        "mesh_size_bound": 3.6 * mesh_size,
    }
    params = {
        "nl_convergence_tol": 1e-8,
        "max_iterations": 50,
        "file_name": "thm",
        "dilation_angle": np.radians(5),
        "dilation_angle_constant_g": 0,
        "use_umfpack": True,
        "mesh_args": mesh_args,
    }

    root_folder = "ex1/"

    gmsh_file_base = root_folder + "mesh_"
    pickle_file_name = root_folder + "gblist"
    folder_base = root_folder + "convergence_study_"
    gb_list, models = [], []
    refinement_levels = np.arange(0, 6)

    if refinement_levels[0] > 1:
        gb_list = thm_utils.read_pickle(pickle_file_name)
        if gb_list is None:
            gb_list = []

    for i in refinement_levels:
        folder_name = folder_base + str(i)
        params["folder_name"] = folder_name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        msh_file = gmsh_file_base + str(i) + ".msh"
        grid_list = pp.fracs.simplex.triangle_grid_from_gmsh(file_name=msh_file)
        gb = pp.fracs.meshing.grid_list_to_grid_bucket(grid_list)

        m = Example1Model(params, gb)
        gb_list.append(m.gb)

        pp.run_time_dependent_model(m, params)

        m._export_pvd()
        thm_utils.write_fracture_data_txt(m)
        thm_utils.write_pickle(gb_list, pickle_file_name)

    if len(gb_list) > 1:
        for gb in gb_list[:-1]:
            pp.grids.match_grids.gb_refinement(gb, gb_list[-1])
    thm_utils.write_pickle(gb_list, pickle_file_name)
```

**说明：**

### 行 1306–1310：网格参数设置

```python
mesh_size = 0.8
mesh_args = {
    "mesh_size_frac": mesh_size,       # 裂缝附近的网格尺寸
    "mesh_size_min": 0.5 * mesh_size,  # 最小网格尺寸
    "mesh_size_bound": 3.6 * mesh_size,# 边界处的网格尺寸
}
```

定义用于 Gmsh 网格生成器的网格参数，网格尺寸为无量纲化后的长度（参考长度为模型内部的 `length_scale`）。

### 行 1312–1320：仿真参数设置

```python
params = {
    "nl_convergence_tol": 1e-8,       # 非线性迭代收敛容忍度
    "max_iterations": 50,              # 最大牛顿迭代次数
    "file_name": "thm",                # 输出文件名
    "dilation_angle": np.radians(5),   # 裂缝剪胀角，5°转为弧度
    "dilation_angle_constant_g": 0,    # 剪胀角全局常量（0表示不使用）
    "use_umfpack": True,               # 使用 UMFPACK 稀疏直接求解器
    "mesh_args": mesh_args,
}
```

- **剪胀角**（行 1316）：将 $5°$ 转换为弧度

$$\varphi = 5 \times \frac{\pi}{180} \approx 0.0873 \text{ rad}$$

### 行 1322–1327：路径与列表初始化

```python
root_folder = "ex1/"
gmsh_file_base = root_folder + "mesh_"
pickle_file_name = root_folder + "gblist"
folder_base = root_folder + "convergence_study_"
gb_list, models = [], []
refinement_levels = np.arange(0, 6)   # [0, 1, 2, 3, 4, 5]
```

定义输出路径，并准备对 6 个加密层次进行收敛性研究（`refinement_levels = [0, 1, 2, 3, 4, 5]`）。

### 行 1329–1333：条件加载已有网格

```python
if refinement_levels[0] > 1:
    gb_list = thm_utils.read_pickle(pickle_file_name)
    if gb_list is None:
        gb_list = []
```

若从中间加密层次开始（`refinement_levels[0] > 1`），则从 pickle 文件中加载之前已计算的网格桶列表。

### 行 1335–1360：主循环（对每个加密层次运行仿真）

```python
for i in refinement_levels:
    folder_name = folder_base + str(i)
    params["folder_name"] = folder_name
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    msh_file = gmsh_file_base + str(i) + ".msh"
    grid_list = pp.fracs.simplex.triangle_grid_from_gmsh(file_name=msh_file)
    gb = pp.fracs.meshing.grid_list_to_grid_bucket(grid_list)

    m = Example1Model(params, gb)
    gb_list.append(m.gb)

    pp.run_time_dependent_model(m, params)

    m._export_pvd()
    thm_utils.write_fracture_data_txt(m)
    thm_utils.write_pickle(gb_list, pickle_file_name)
```

对每个加密层次：
1. 创建输出文件夹（行 1337–1340）
2. 从预先生成的 `.msh` 文件读取三角网格（行 1343–1345），并构建"网格桶"（`GridBucket`，即混合维度网格）
3. 实例化模型 `Example1Model`（行 1347）
4. 调用 `pp.run_time_dependent_model` 执行时间步进仿真（行 1350）
5. 导出 PVD 文件（行 1353）、写入分形数据（行 1354）、保存 pickle 快照（行 1355）

### 行 1357–1360：计算网格映射用于误差分析

```python
if len(gb_list) > 1:
    for gb in gb_list[:-1]:
        pp.grids.match_grids.gb_refinement(gb, gb_list[-1])
thm_utils.write_pickle(gb_list, pickle_file_name)
```

将所有较粗网格映射到最细网格，以便在不同加密层次之间比较误差（收敛性分析）。

---

## 4. 网格创建函数 `create_grid`（行 46–79）

```python
def create_grid(mesh_args):
    frac_pts = np.array([
        [0.2, 0.7], [0.5, 0.7], [0.8, 0.65],
        [0.2, 0.3], [0.6, 0.2], [0.2, 0.15],
        [0.7, 0.4], [1.0, 0.4], [1.7, 0.85],
        [1.5, 0.65], [2.0, 0.55], [1, 0.3],
        [1.8, 0.4], [1.5, 0.05], [1.4, 0.25],
    ]).T
    frac_edges = np.array(
        [[0,1],[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14]]
    ).T

    box = {"xmin": 0, "ymin": 0, "xmax": 2, "ymax": 1}
    network = pp.FractureNetwork2d(frac_pts, frac_edges, domain=box)
    gb = network.mesh(mesh_args)
    return gb, network
```

**说明：**

- **行 51–69**：定义 15 个裂缝端点坐标（单位：m），转置为 $2 \times 15$ 的数组，每列为一个端点的 $(x, y)$ 坐标。
- **行 70–72**：定义 8 条裂缝的连接关系（每行为两端点的索引），构成 $2 \times 8$ 的数组。
- **行 74**：定义计算域 $\Omega = [0, 2] \times [0, 1]$（单位：m）。
- **行 76–78**：使用 PorePy 的 `FractureNetwork2d` 创建裂缝网络，并调用 Gmsh 生成混合维度网格桶。

该函数仅在主代码块中用作参考，实际使用的是从 `.msh` 文件读取的预生成网格。

---

## 5. 模型实例化 `Example1Model.__init__`（行 90–93）

```python
def __init__(self, params, gb):
    super().__init__(params)
    self._set_fields(params, gb)
```

**说明：**

- **行 91**：调用父类 `THM.__init__(params)`，初始化 THM 模型框架（变量名、参数关键字、缩放因子等）。父类 `THM` 来自 PorePy，其中定义了与热-水-力学（Thermo-Hydro-Mechanical）耦合问题相关的所有基础结构。
- **行 92**：调用 `_set_fields`，初始化案例特有的字段（详见 [第 6 节](#6-字段初始化-_set_fields行1199-1234)）。

---

## 6. 字段初始化 `_set_fields`（行 1199–1234）

```python
def _set_fields(self, params, gb=None):
    self.T_0_Kelvin = 300
    self.background_temp_C = self.T_0_Kelvin - 273
    if gb is not None:
        self.gb = gb

    self.scalar_scale = 1e9

    self.file_name = self.params["file_name"]
    self.folder_name = self.params["folder_name"]
    self.mechanics_parameter_key_from_t = "mechanics_from_t"

    self.export_fields = [
        "u_exp", "p_exp", "T_exp", "traction_exp",
        "aperture_exp", "u_global", "cell_centers", "du",
    ]
    self.initial_aperture = 5e-4 / self.length_scale

    self._dilation_angle = params.get("dilation_angle")
    self._dilation_angle_constant_g = params.get("dilation_angle_constant_g")
    self.full_g_coupling = params.get("full_g_coupling", True)
    self.mesh_args = params.get("mesh_args", None)
```

**说明：**

| 行号 | 字段 | 说明 |
|------|------|------|
| 1203 | `T_0_Kelvin = 300` | 参考温度 $T_0 = 300\,\text{K}$（约 $27°\text{C}$）|
| 1204 | `background_temp_C` | 背景温度（摄氏度）$= T_0 - 273 = 27°\text{C}$|
| 1209 | `scalar_scale = 1e9` | 压力无量纲化因子 $p_\text{scale} = 10^9\,\text{Pa} = 1\,\text{GPa}$|
| 1227 | `initial_aperture` | 初始裂缝开度 $a_0 = 5 \times 10^{-4}\,\text{m}$（除以 `length_scale` 做无量纲化）|
| 1230 | `_dilation_angle` | 剪胀角 $\varphi$（弧度）|
| 1231 | `_dilation_angle_constant_g` | 剪胀角全局常量（本例为 0，表示不对交叉点使用剪胀）|

---

## 7. 仿真准备 `prepare_simulation`（行 738–751）

```python
def prepare_simulation(self):
    self.create_grid()
    self._set_time_parameters()
    self._set_rock_and_fluid()
    self._initial_condition()
    self._set_parameters()
    self._assign_variables()
    self._assign_discretizations()
    self._discretize()
    self._initialize_linear_solver()
    self._export_step()
```

该方法按顺序完成所有仿真准备工作。下面逐一展开各子方法。

---

### 7.1 网格设置 `create_grid`（方法）（行 95–113）

```python
def create_grid(self):
    gb = self.gb
    self.box = {
        "xmin": 0, "ymin": 0,
        "xmax": 2 / self.length_scale,
        "ymax": 1 / self.length_scale,
    }
    pp.contact_conditions.set_projections(gb)
    self._Nd = gb.dim_max()
    self.n_frac = len(gb.grids_of_dimension(self._Nd - 1))
```

**说明：**

- **行 101**：获取构造函数中传入的网格桶 `gb`。
- **行 104–109**：定义无量纲化后的计算域边界框。原始物理域为 $[0,2]\,\text{m} \times [0,1]\,\text{m}$，除以 `length_scale` 后得到无量纲域。
- **行 110**：为网格桶中所有裂缝面（接口）设置切线-法线坐标系投影，供接触力学计算使用。
- **行 112**：获取最大维度（二维问题中 `_Nd = 2`）。
- **行 113**：统计裂缝数量（即 `_Nd - 1 = 1` 维网格的个数）。

---

### 7.2 时间参数 `_set_time_parameters`（行 1154–1168）

```python
def _set_time_parameters(self):
    self.time = self.params.get("time", -1e6)
    self.time_step = -self.time

    self.end_time = 10 * pp.HOUR
    self.max_time_step = self.end_time / 5
    self.phase_limits = np.array([self.time, 0, 3e1, self.end_time])
    self.phase_time_steps = np.array([self.time_step, 2e0, 2/3 * pp.HOUR, 1.0])
```

**说明：**

仿真被划分为四个阶段：

| 阶段 | 时间范围 | 描述 |
|------|----------|------|
| I（初始化） | $t \in [-10^6\,\text{s},\; 0]$ | 力学平衡阶段，施加位移边界条件 |
| II（流动） | $t \in [0,\; 30\,\text{s}]$ | 左侧施加压力梯度，开始流体流动 |
| III（热） | $t \in [30\,\text{s},\; 10\,\text{h}]$ | 左侧降温，耦合热效应 |

- **行 1160**：初始时间 $t_0 = -10^6\,\text{s}$（用于初始力学平衡计算）
- **行 1161**：初始时间步长 $\Delta t = -t_0 = 10^6\,\text{s}$（用一步完成初始化阶段）
- **行 1163**：结束时间 $t_\text{end} = 10\,\text{h} = 36000\,\text{s}$
- **行 1164**：最大时间步长 $\Delta t_\text{max} = t_\text{end}/5 = 7200\,\text{s}$
- **行 1167**：各阶段结束时刻数组 $[t_0,\; 0,\; 30,\; t_\text{end}]$
- **行 1168**：各阶段开始时使用的初始时间步长数组

---

### 7.3 岩石与流体参数 `_set_rock_and_fluid`（行 442–448）

```python
def _set_rock_and_fluid(self):
    self.rock = Granite()
    self.fluid = Water()
```

实例化 `Granite`（固体相）和 `Water`（流体相）两个参数类，参数包含弹性模量、热导率、热膨胀系数、渗透率、流体黏度等（详见 [第 10 节](#10-辅助类-water-与-granite行1249-1303)）。

---

### 7.4 初始条件 `_initial_condition`（行 1112–1141）

```python
def _initial_condition(self) -> None:
    for g, d in self.gb:
        d[pp.PARAMETERS] = pp.Parameters()
        d[pp.PARAMETERS].update_dictionaries([...])
    self._update_all_apertures(to_iterate=False)
    self._update_all_apertures()
    super()._initial_condition()

    for g, d in self.gb:
        d[pp.STATE]["cell_centers"] = g.cell_centers.copy()
        p0 = self._initial_scalar(g)
        T0 = self._initial_temperature(g)
        state = {
            self.scalar_variable: p0,
            self.temperature_variable: T0,
            "previous_u_exp": np.zeros((3, g.num_cells)),
        }
        iterate = {
            self.scalar_variable: p0,
            self.temperature_variable: T0,
        }
        pp.set_state(d, state)
        pp.set_iterate(d, iterate)
```

**说明：**

**初始压力**（行 1143–1149）：

```python
def _initial_scalar(self, g) -> np.ndarray:
    if getattr(self, "gravity_on", False):
        depth = self._depth(g.cell_centers)
    else:
        depth = np.zeros(g.num_cells)
    return self.fluid.hydrostatic_pressure(depth) / self.scalar_scale
```

本例不考虑重力（`gravity_on` 默认为 `False`），初始压力为大气压 $p_\text{atm}$，无量纲化后：

$$p_0 = \frac{p_\text{atm}}{p_\text{scale}} = \frac{101325\,\text{Pa}}{10^9\,\text{Pa}} \approx 1.01 \times 10^{-4}$$

**初始温度**（行 1151–1152）：

```python
def _initial_temperature(self, g) -> np.ndarray:
    return self.T_0_Kelvin * np.ones(g.num_cells)
```

$$T_0 = 300\,\text{K}$$

（注意：温度变量在 PorePy 内部已做无量纲化，此处存储的是 Kelvin 量级的值，与 `temperature_scale` 配合使用。）

**裂缝开度初始化**（行 1123–1124 调用 `_update_all_apertures`）：所有网格的初始开度按如下方法设置（详见 [第 7.5 节中的开度更新](#开度更新-_update_all_apertures行290-387)）。

---

### 7.5 参数设置 `_set_parameters`

`_set_parameters` 是父类方法，其内部会依次调用力学、流体、温度三个参数设置子方法。

#### 7.5.1 力学参数 `_set_mechanics_parameters`（行 450–521）

```python
def _set_mechanics_parameters(self):
    gb = self.gb
    for g, d in gb:
        if g.dim == self._Nd:
            rock = self.rock
            lam = rock.LAMBDA * np.ones(g.num_cells) / self.scalar_scale
            mu  = rock.MU    * np.ones(g.num_cells) / self.scalar_scale
            C = pp.FourthOrderTensor(mu, lam)

            bc = self._bc_type_mechanics(g)
            bc_values = self._bc_values_mechanics(g)
            ...
            coupling_coefficient = self._biot_alpha(g)
            pp.initialize_data(g, d, self.mechanics_parameter_key, {
                "bc": bc,
                "bc_values": bc_values,
                "fourth_order_tensor": C,
                "biot_alpha": coupling_coefficient,
                ...
            })
            pp.initialize_data(g, d, self.mechanics_temperature_parameter_key, {
                "biot_alpha": self._biot_beta(g),
                "p_reference": self.T_0_Kelvin * np.ones(g.num_cells),
                ...
            })
        elif g.dim == self._Nd - 1:
            pp.initialize_data(g, d, self.mechanics_parameter_key, {
                "friction_coefficient": self._set_friction_coefficient(g),
                "contact_mechanics_numerical_parameter": 1e1,
                "dilation_angle": self._dilation_angle,
                ...
            })
```

**力学弹性张量（行 460–462）：**

基质（`g.dim == _Nd`）中的各向同性线弹性刚度张量 $\mathbf{C}$，由拉梅参数 $\lambda$ 和 $\mu$ 描述：

$$\boldsymbol{\sigma} = \lambda \, (\nabla \cdot \mathbf{u}) \, \mathbf{I} + 2\mu \, \boldsymbol{\varepsilon}(\mathbf{u})$$

其中 $\boldsymbol{\varepsilon}(\mathbf{u}) = \frac{1}{2}(\nabla \mathbf{u} + (\nabla \mathbf{u})^T)$ 为应变张量，$\mathbf{I}$ 为单位张量。代码中 `lam` 和 `mu` 均除以 `scalar_scale`（$10^9\,\text{Pa}$）做无量纲化。

**力学边界条件（行 121–146）：**

- 北侧（`north`，即 $y = y_\text{max}$）和南侧（`south`，即 $y = 0$）施加 Dirichlet 位移边界条件：

```python
def _bc_values_mechanics(self, g) -> np.ndarray:
    _, _, _, north, south, _, _ = self._domain_boundary_sides(g)
    bc_values = np.zeros((g.dim, g.num_faces))
    x = 5e-4 / self.length_scale
    y = -2e-4 / self.length_scale
    bc_values[0, north] = x
    bc_values[1, north] = y
    return bc_values.ravel("F")
```

北侧位移 Dirichlet 值（无量纲化后存储）：

$$\mathbf{u}\big|_\text{north} = \left(\frac{5 \times 10^{-4}}{L_\text{scale}},\; \frac{-2 \times 10^{-4}}{L_\text{scale}}\right)$$

原始物理值为 $u_x = 5 \times 10^{-4}\,\text{m}$，$u_y = -2 \times 10^{-4}\,\text{m}$。南侧固定（零位移）。

**Biot 系数 $\alpha$（行 184–188）：**

```python
def _biot_alpha(self, g) -> np.ndarray:
    if g.dim == self._Nd:
        return 0.8
    else:
        return 1.0
```

$$\alpha = \begin{cases} 0.8 & \text{基质} \\ 1.0 & \text{裂缝} \end{cases}$$

Biot 系数 $\alpha$ 出现在动量方程中，描述孔隙流体压力对固体骨架的贡献：

$$\nabla \cdot \boldsymbol{\sigma} - \alpha \nabla p - \beta \nabla T = 0$$

**热-力学耦合系数 $\beta$（行 190–210）：**

```python
def _biot_beta(self, g):
    if g.dim == self._Nd:
        return self.rock.BULK_MODULUS * 3 * self.rock.THERMAL_EXPANSION
    else:
        _, T_k, _ = self._variable_increment(g, self.temperature_variable)
        beta = (T_k / self.T_0_Kelvin
                * self._fluid_density(g)
                * self.fluid.specific_heat_capacity())
        return beta
```

在基质中，热-力学耦合系数 $\beta$（即热膨胀引起的应力贡献）为：

$$\beta_s = K_s \cdot 3\alpha_{T,s}$$

其中 $K_s$ 为固体体积模量，$3\alpha_{T,s}$ 为体积热膨胀系数（线热膨胀系数乘以 3）。

在裂缝中，耦合系数与当前温度 $T^k$、流体密度 $\rho_f$ 及比热容 $c_f$ 相关：

$$\beta_f = \frac{T^k}{T_0} \cdot \rho_f \cdot c_f$$

**摩擦系数（行 1170–1176）：**

```python
def _set_friction_coefficient(self, g) -> float:
    return 0.5
```

裂缝接触力学中的库仑摩擦系数 $\mu_f = 0.5$，用于判断裂缝是否发生滑动（Coulomb 准则）：

$$|\mathbf{T}_\tau| \leq \mu_f |T_n|$$

其中 $\mathbf{T}_\tau$ 为切向接触力，$T_n$ 为法向接触力。

---

#### 7.5.2 流体参数 `_set_scalar_parameters`（行 523–577）

```python
def _set_scalar_parameters(self):
    for g, d in self.gb:
        a = self._aperture(g)
        specific_volumes = self._specific_volumes(g)
        bc = self._bc_type_scalar(g)
        bc_values = self._bc_values_scalar(g)
        biot_coefficient = self._biot_alpha(g)
        compressibility = self.fluid.COMPRESSIBILITY

        mass_weight = compressibility * self._porosity(g)
        if g.dim == self._Nd:
            mass_weight += (biot_coefficient - self._porosity(g)) / self.rock.BULK_MODULUS
        ...
        mass_weight *= self.scalar_scale * specific_volumes
        ...
        t2s_coupling = (
            self._temperature_to_scalar_coupling_coefficient(g)
            * specific_volumes * self.temperature_scale
        )
    self._set_permeability_from_aperture()
```

**流体边界条件（行 148–163）：**

```python
def _bc_type_scalar(self, g):
    _, east, west, *_ = self._domain_boundary_sides(g)
    return pp.BoundaryCondition(g, east + west, "dir")

def _bc_values_scalar(self, g):
    _, _, west, *_ = self._domain_boundary_sides(g)
    bc_values = np.zeros(g.num_faces)
    if self.time > self.phase_limits[1]:
        bc_values[west] = 4e7 / self.scalar_scale
    return bc_values
```

东侧和西侧施加 Dirichlet 压力边界：
- 东侧：$p_\text{east} = 0$（参考压力，除以 `scalar_scale` 后为 0）
- 西侧（阶段 II 之后）：$p_\text{west} = 4 \times 10^7\,\text{Pa}$（无量纲化后为 $4 \times 10^7 / 10^9 = 0.04$）

**质量权重（行 538–546）：**

质量方程中的时间导数项系数（压力方程中的压缩性项）：

$$M_p = \left(C_f \phi + \frac{\alpha - \phi}{K_s}\right) p_\text{scale} \cdot V$$

其中：
- $C_f = 10^{-10}\,\text{Pa}^{-1}$：流体压缩系数
- $\phi$：孔隙率（基质为 0.01，裂缝为 1）
- $\alpha$：Biot 系数
- $K_s$：固体体积模量
- $V$：单元比体积（`specific_volume`）

**温度-压力耦合系数（行 257–270，行 564–574）：**

```python
def _temperature_to_scalar_coupling_coefficient(self, g) -> float:
    b_f = self.fluid.thermal_expansion(0)
    if g.dim < self._Nd:
        coeff = -b_f
    else:
        b_s = 3 * self.rock.THERMAL_EXPANSION
        phi = self._porosity(g)
        coeff = -b_f * phi - (1 - phi) * b_s
    return coeff
```

在压力方程中，温度变化引起的体积膨胀项系数：

$$\xi_{T \to p} = \begin{cases} -\beta_f & \text{裂缝} \\ -\beta_f \phi - (1-\phi) \cdot 3\alpha_{T,s} & \text{基质} \end{cases}$$

其中 $\beta_f = 4 \times 10^{-4}\,\text{K}^{-1}$ 为流体体积热膨胀系数。

**渗透率设置（行 389–440）：**

```python
def _set_permeability_from_aperture(self):
    viscosity = self.fluid.dynamic_viscosity() / self.scalar_scale
    for g, d in gb:
        if g.dim < self._Nd:
            apertures = self._aperture(g, from_iterate=True)
            apertures_unscaled = apertures * self.length_scale
            k = np.power(apertures_unscaled, 2) / 12 / viscosity
            ...
        else:
            kxx = (self.rock.PERMEABILITY / viscosity * np.ones(g.num_cells)
                   / self.length_scale ** 2)
```

- **裂缝中（立方定律）**：裂缝中的渗透率由立方定律给出：

$$k_f = \frac{a^2}{12\mu}$$

其中 $a$ 为裂缝开度（物理值，单位 m），$\mu = 10^{-3}\,\text{Pa·s}$ 为流体动力黏度。

- **基质中**：使用花岗岩固有渗透率：

$$k_m = \frac{K_m}{\mu}$$

其中 $K_m = 10^{-15}\,\text{m}^2$（花岗岩典型渗透率）。

---

#### 7.5.3 温度参数 `_set_temperature_parameters`（行 611–704）

```python
def _set_temperature_parameters(self):
    div_T_scale = self.temperature_scale / self.length_scale**2 / self.T_0_Kelvin
    kappa_f = self.fluid.thermal_conductivity() * div_T_scale
    kappa_s = self.rock.thermal_conductivity() * div_T_scale
    heat_capacity_s = self.rock.specific_heat_capacity() * self.rock.DENSITY

    for g, d in self.gb:
        heat_capacity_f = self._fluid_density(g) * self.fluid.specific_heat_capacity()
        specific_volumes = self._specific_volumes(g)
        porosity = self._porosity(g)
        bc = self._bc_type_temperature(g)
        bc_values = self._bc_values_temperature(g)
        T_k = d[pp.STATE][pp.ITERATE][self.temperature_variable]

        effective_heat_capacity = (
            porosity * heat_capacity_f * (1 - T_k * self.fluid.thermal_expansion(0))
            + (1 - porosity) * heat_capacity_s * (1 - T_k * 3 * self.rock.THERMAL_EXPANSION)
        )
        mass_weight = (effective_heat_capacity * specific_volumes
                       * self.temperature_scale / self.T_0_Kelvin)

        effective_conductivity = porosity * kappa_f + (1 - porosity) * kappa_s
        ...
        advection_weight = heat_capacity_f * self.temperature_scale / self.T_0_Kelvin
        ...
        s2t_coupling = (
            self._scalar_to_temperature_coupling_coefficient(g)
            * specific_volumes * self.scalar_scale * T_k
        )
```

**温度边界条件（行 165–182）：**

```python
def _bc_values_temperature(self, g) -> np.ndarray:
    _, east, west, *_ = self._domain_boundary_sides(g)
    bc_values = np.zeros(g.num_faces)
    bc_values[east + west] = self.T_0_Kelvin / self.temperature_scale
    if self.time > self.phase_limits[2]:
        bc_values[west] = (self.T_0_Kelvin - 15) / self.temperature_scale
    return bc_values
```

- 东侧和西侧（初始）：$T_\text{east} = T_\text{west} = T_0 = 300\,\text{K}$
- 西侧（阶段 III 之后）：$T_\text{west} = T_0 - 15 = 285\,\text{K}$（降温 15 K）

**有效热容量（行 637–641）：**

$$(\rho c_p)_\text{eff} = \phi \rho_f c_f \left(1 - T^k \beta_f\right) + (1-\phi) \rho_s c_s \left(1 - T^k \cdot 3\alpha_{T,s}\right)$$

其中：
- $\phi$：孔隙率
- $\rho_f c_f$：流体体积热容（$\text{J}/(\text{m}^3 \cdot \text{K})$）
- $\rho_s c_s$：固体体积热容
- $\beta_f$：流体热膨胀系数
- $\alpha_{T,s}$：固体线热膨胀系数

**有效热导率（行 649–652）：**

$$\kappa_\text{eff} = \phi \kappa_f + (1 - \phi) \kappa_s$$

其中 $\kappa_f = 0.6\,\text{W}/(\text{m·K})$ 为流体热导率，$\kappa_s = 3.0\,\text{W}/(\text{m·K})$ 为固体（花岗岩）热导率。

**对流权重（行 654–656）：**

$$w_\text{adv} = \rho_f c_f \cdot \frac{T_\text{scale}}{T_0}$$

用于热平流项（流体流动携带热量）。

**压力-温度耦合系数（行 238–255，行 675–687）：**

```python
def _scalar_to_temperature_coupling_coefficient(self, g) -> float:
    c_f = self.fluid.specific_heat_capacity()
    rho_f = self._fluid_density(g)
    C_f = self.fluid.COMPRESSIBILITY
    if g.dim < self._Nd:
        coeff = c_f * rho_f * C_f
    else:
        c_s = self.rock.specific_heat_capacity()
        rho_s = self.rock.DENSITY
        K_s = self.rock.BULK_MODULUS
        phi = self._porosity(g)
        coeff = phi * c_f * rho_f * C_f + (1 - phi) * c_s * rho_s / K_s
    return coeff / self.T_0_Kelvin
```

在能量方程中，压力变化对温度（焓）的贡献系数：

$$\xi_{p \to T} = \begin{cases} \dfrac{c_f \rho_f C_f}{T_0} & \text{裂缝} \\[6pt] \dfrac{\phi c_f \rho_f C_f + (1-\phi) c_s \rho_s / K_s}{T_0} & \text{基质} \end{cases}$$

---

#### 开度更新 `_update_all_apertures`（行 290–387）

```python
def _update_all_apertures(self, to_iterate=True):
    gb = self.gb
    for g, d in gb:
        apertures = np.ones(g.num_cells)
        if g.dim == (self._Nd - 1):
            apertures *= self.initial_aperture
            g_h = gb.node_neighbors(g)[0]
            data_edge = gb.edge_props((g, g_h))
            if pp.STATE in data_edge:
                projection = d["tangential_normal_projection"]
                u_mortar_local = self.reconstruct_local_displacement_jump(
                    data_edge, projection, from_iterate=to_iterate
                )
                norm_u_n   = np.absolute(u_mortar_local[-1])
                norm_u_tau = np.linalg.norm(u_mortar_local[:-1], axis=0)
                apertures += (
                    norm_u_n + np.tan(self._dilation_angle_constant_g) * norm_u_tau
                )
    ...
    for g, d in gb:
        if g.dim < (self._Nd - 1):
            ...
            apertures = np.sum(parent_apertures, axis=0) / num_parents
            specific_volumes = np.power(apertures, self._Nd - g.dim)
```

**裂缝开度公式（行 314–320）：**

$$a = a_0 + \left|\llbracket u_n \rrbracket\right| + \tan(\varphi) \cdot \|\llbracket \mathbf{u}_\tau \rrbracket\|$$

其中：
- $a_0 = 5 \times 10^{-4}\,\text{m}$：初始开度
- $\llbracket u_n \rrbracket$：法向位移跳跃（裂缝张开量）
- $\llbracket \mathbf{u}_\tau \rrbracket$：切向位移跳跃（滑移量）
- $\varphi$：剪胀角（本例 `_dilation_angle_constant_g = 0`，即忽略剪胀效应）

**交叉点比体积（行 369–371）：**

对于维度更低的交叉点（$g.dim < _Nd - 1$），比体积为：

$$V = a^{N_d - \text{dim}}$$

其中 $N_d - \text{dim}$ 在二维中对交叉点（0 维）为 2。

---

#### 流体密度 `_fluid_density`（行 212–236）

```python
def _fluid_density(self, g, dp=None, dT=None) -> np.ndarray:
    ...
    rho_0 = 1e3 * (pp.KILOGRAM / pp.METER**3) * np.ones(g.num_cells)
    rho = rho_0 * np.exp(
        dp * self.fluid.COMPRESSIBILITY - dT * self.fluid.thermal_expansion(dT)
    )
    return rho
```

流体密度由状态方程给出（行 232–235）：

$$\rho_f = \rho_0 \exp\left(C_f \Delta p - \beta_f \Delta T\right)$$

其中：
- $\rho_0 = 1000\,\text{kg/m}^3$：参考密度（水）
- $C_f = 10^{-10}\,\text{Pa}^{-1}$：流体压缩系数
- $\beta_f = 4 \times 10^{-4}\,\text{K}^{-1}$：流体体积热膨胀系数
- $\Delta p = p - p_\text{atm}$：压力增量
- $\Delta T = T - T_0$：温度增量（被截断在 $[-T_0/3,\; T_0/3]$ 以保证收敛）

---

### 7.6 离散化与线性求解器初始化

```python
self._assign_variables()
self._assign_discretizations()
self._discretize()
self._initialize_linear_solver()
```

- **`_assign_variables`**：为每个子域（基质、裂缝、界面）分配自由度变量（位移 $\mathbf{u}$、压力 $p$、温度 $T$、接触牵引力 $\mathbf{T}_c$、界面位移 $\lambda$）。
- **`_assign_discretizations`**（行 722–736）：为每个子域和界面分配离散化方案（MPSA 有限体积法用于力学，MPFA/TPFA 用于流体和热传导）。对于耦合界面，还设置了通量缩放选项：

```python
def _assign_discretizations(self) -> None:
    super()._assign_discretizations()
    for e, d in self.gb.edges():
        d[pp.COUPLING_DISCRETIZATION][self.temperature_coupling_term][e][1].kinv_scaling = False
        d[pp.COUPLING_DISCRETIZATION][self.scalar_coupling_term][e][1].kinv_scaling = True
```

- **`_discretize`**：计算所有离散化矩阵（刚度矩阵、质量矩阵、耦合矩阵等），为全局系统组装做准备。
- **`_initialize_linear_solver`**：初始化线性求解器（本例使用 UMFPACK 直接求解器）。

---

## 8. 时间步进与牛顿迭代

`pp.run_time_dependent_model(m, params)` 调用 PorePy 框架的时间步进主循环，在每个时间步内执行牛顿迭代，直到收敛或达到最大迭代次数。

---

### 8.1 牛顿循环前处理 `before_newton_loop`（行 753–755）

```python
def before_newton_loop(self):
    super().before_newton_loop()
    self._iteration = 0
```

在每个时间步的牛顿迭代开始前，重置迭代计数器。

---

### 8.2 每次迭代前处理 `before_newton_iteration`（行 757–782）

```python
def before_newton_iteration(self):
    self._iteration += 1
    self.compute_fluxes()
    self._update_all_apertures(to_iterate=True)
    self._set_parameters()
    t_0 = time.time()
    term_list = [
        "!mpsa", "!stabilization", "!div_u", "!grad_p", "!diffusion",
    ]
    filt = pp.assembler_filters.ListFilter(term_list=term_list)
    self.assembler.discretize(filt=filt)
    for dim in range(self._Nd - 1):
        for g in self.gb.grids_of_dimension(dim):
            filt = pp.assembler_filters.ListFilter(
                term_list=["diffusion"], grid_list=[g]
            )
            self.assembler.discretize(filt=filt)
    logger.info("Rediscretized in {} s.".format(time.time() - t_0))
```

**说明：**

- **行 760**：递增迭代计数器 $k \leftarrow k + 1$。
- **行 761**：根据当前迭代解计算内部流量（用于非线性项，如对流）。
- **行 762**：用当前迭代位移更新裂缝开度 $a^{(k)}$（见开度公式）。
- **行 764**：重新设置所有参数（因为开度、密度等非线性参数依赖于当前迭代解）。
- **行 766–782**：选择性重新离散化：仅重新离散化随非线性解变化的项（质量项、对流项、耦合项），跳过耗时的 MPSA（力学）、扩散（固定线性项）等，以节省计算时间。"!"前缀表示排除该项。

---

### 8.3 线性系统组装与求解 `assemble_and_solve_linear_system`（行 796–817）

```python
def assemble_and_solve_linear_system(self, tol):
    A, b = self.assembler.assemble_matrix_rhs()
    use_umfpack = self.params.get("use_umfpack", True)
    if use_umfpack:
        A.indices = A.indices.astype(np.int64)
        A.indptr  = A.indptr.astype(np.int64)
    ...
    t_0 = time.time()
    x = spla.spsolve(A, b)
    logger.info("Solved in {} s.".format(time.time() - t_0))
    return x
```

**说明：**

组装全局稀疏线性系统：

$$\mathbf{A} \mathbf{x} = \mathbf{b}$$

其中 $\mathbf{A}$ 为全局雅可比矩阵（或刚度矩阵），$\mathbf{b}$ 为右端项，$\mathbf{x}$ 为当前牛顿迭代的解向量（包含所有子域的位移、压力、温度、接触力、界面位移增量）。

- **行 806–808**：当使用 UMFPACK 时，将 CSR 矩阵的整数索引转换为 `int64` 类型（UMFPACK 要求）。
- **行 815**：使用 `scipy.sparse.linalg.spsolve` 直接求解稀疏线性系统。

---

### 8.4 收敛性检验 `check_convergence`（行 819–948）

```python
def check_convergence(self, solution, prev_solution, init_solution, nl_params=None):
    g_max = self._nd_grid()

    if not self._is_nonlinear_problem():
        diverged = np.any(np.isnan(solution))
        converged = not diverged
        error = np.nan if diverged else 0
        return error, converged, diverged

    mech_dof   = self.dof_manager.dof_ind(g_max, self.displacement_variable)
    p_dof      = ...
    T_dof      = ...
    contact_dof = ...

    u_mech_now  = solution[mech_dof]
    u_mech_prev = prev_solution[mech_dof]
    ...
    difference_in_iterates_mech = np.sum((u_mech_now - u_mech_prev)**2)
    ...
    tol_convergence = nl_params["nl_convergence_tol"]

    # 绝对收敛准则
    if difference_in_iterates_mech < tol_convergence:
        converged_mech = True
    else:
        # 相对收敛准则
        u_norm = np.sum(u_mech_now**2)
        if difference_in_iterates_mech < tol_convergence * u_norm:
            converged_mech = True
        error_mech = difference_in_iterates_mech / u_norm

    converged = converged_mech and converged_T and converged_p and converged_contact
    return error_mech, converged, diverged
```

**说明：**

对力学位移、温度、压力、接触力分别检验收敛性，使用绝对与相对混合准则：

**绝对准则：**

$$\|\mathbf{x}^{(k)} - \mathbf{x}^{(k-1)}\|_2^2 < \varepsilon_\text{tol}$$

**相对准则：**

$$\frac{\|\mathbf{x}^{(k)} - \mathbf{x}^{(k-1)}\|_2^2}{\|\mathbf{x}^{(k)}\|_2^2} < \varepsilon_\text{tol}$$

其中 $\varepsilon_\text{tol} = 10^{-8}$（`nl_convergence_tol`），各变量分别进行检验，所有变量同时满足才认为收敛：

$$\text{converged} = \text{converged}_u \wedge \text{converged}_T \wedge \text{converged}_p \wedge \text{converged}_\lambda$$

---

### 8.5 牛顿收敛后处理 `after_newton_convergence`（行 786–794）

```python
def after_newton_convergence(self, solution, errors, iteration_counter):
    for g, d in self.gb:
        d[pp.STATE]["previous_u_exp"] = d[pp.STATE]["u_exp"].copy()
    super().after_newton_convergence(solution, errors, iteration_counter)
    self._update_all_apertures(to_iterate=False)
    self._update_all_apertures(to_iterate=True)
    self._export_step()
    self._adjust_time_step()
    self._save_data(errors, iteration_counter)
```

**说明：**

- **行 787–788**：将当前时间步的位移场保存为"上一时间步的位移场"，用于计算位移增量 $\Delta \mathbf{u}$（导出用）。
- **行 789**：调用父类方法，将迭代解写入状态（将 `ITERATE` 状态提升为时间步状态）。
- **行 790–791**：在时间步收敛后更新裂缝开度（先更新时间步状态，再更新迭代状态）。
- **行 792**：导出当前时间步结果（VTU 文件）。
- **行 793**：调整下一时间步的步长（见 `_adjust_time_step`）。
- **行 794**：保存当前时间步的统计数据。

#### 时间步长调整 `_adjust_time_step`（行 1178–1197）

```python
def _adjust_time_step(self):
    self.time_step = getattr(self, "time_step_factor", 1.0) * self.time_step

    for dt, lim in zip(self.phase_time_steps, self.phase_limits):
        diff = self.time - lim
        if diff < 0 and -diff <= self.time_step:
            self.time_step = -diff

        if np.isclose(self.time, lim):
            self.time_step = dt

    if self.time > 0:
        self.time_step = min(self.time_step, self.max_time_step)
```

时间步长调整策略：
1. 默认按固定因子（`time_step_factor`，默认 1）增加时间步长：$\Delta t^{n+1} = f \cdot \Delta t^n$
2. 确保精确到达各阶段边界时刻：若当前时间到阶段边界的距离小于下一步长，则令下一步长等于该距离
3. 到达阶段边界后，使用该阶段的初始时间步长（见 `phase_time_steps`）
4. 时间步长上限：$\Delta t \leq \Delta t_\text{max} = t_\text{end}/5$

---

## 9. 数据导出与保存

### 9.1 导出步骤 `_export_step`（行 956–1021）

```python
def _export_step(self):
    if "exporter" not in self.__dict__:
        self._set_exporter()
    for g, d in self.gb:
        if g.dim == self._Nd:
            pad_zeros = np.zeros((3 - g.dim, g.num_cells))
            u = d[pp.STATE][self.displacement_variable].reshape((self._Nd, -1), order="F")
            u_exp = np.vstack((u * self.length_scale, pad_zeros))
            d[pp.STATE]["u_exp"] = u_exp
            ...
        elif g.dim == (self._Nd - 1):
            ...
            u_exp = np.vstack((u_mortar_local * self.length_scale, pad_zeros))
            d[pp.STATE]["u_exp"] = u_exp
            traction = d[pp.STATE][self.contact_traction_variable].reshape(
                (self._Nd, -1), order="F"
            )
            d[pp.STATE]["traction_exp"] = np.vstack((traction, pad_zeros)) * self.scalar_scale
        ...
        d[pp.STATE]["aperture_exp"] = self._aperture(g) * self.length_scale
        d[pp.STATE]["p_exp"] = d[pp.STATE][self.scalar_variable] * self.scalar_scale
        d[pp.STATE]["T_exp"] = d[pp.STATE][self.temperature_variable] * self.temperature_scale
        d[pp.STATE]["du"] = d[pp.STATE]["u_exp"] - d[pp.STATE]["previous_u_exp"]
    self.exporter.write_vtu(self.export_fields, time_step=self.time)
    self.export_times.append(self.time)
```

**说明：**

将内部无量纲化变量还原为物理量并导出：

| 导出字段 | 还原公式 |
|----------|----------|
| `u_exp` | $\mathbf{u}_\text{phys} = \mathbf{u}_\text{nd} \times L_\text{scale}$（单位：m）|
| `p_exp` | $p_\text{phys} = p_\text{nd} \times p_\text{scale}$（单位：Pa）|
| `T_exp` | $T_\text{phys} = T_\text{nd} \times T_\text{scale}$（单位：K）|
| `traction_exp` | $\mathbf{T}_\text{phys} = \mathbf{T}_\text{nd} \times p_\text{scale}$（单位：Pa）|
| `aperture_exp` | $a_\text{phys} = a_\text{nd} \times L_\text{scale}$（单位：m）|
| `du` | $\Delta\mathbf{u} = \mathbf{u}_\text{exp}^n - \mathbf{u}_\text{exp}^{n-1}$（时间步位移增量）|

所有导出场都在 VTK/ParaView 中使用三分量向量（2D 问题补零至 3 维）。

---

### 9.2 数据保存 `_save_data`（行 1030–1110）

```python
def _save_data(self, errors, iteration_counter):
    n = self.n_frac
    ...
    self.iterations.append(iteration_counter + 1)

    for g, d in self.gb:
        ...
        if g.dim == self._Nd - 1:
            ...
            tangential_jump = np.linalg.norm(
                u_mortar_local[:-1] * self.length_scale, axis=0
            )
            normal_jump = u_mortar_local[-1] * self.length_scale
            vol = np.sum(g.cell_volumes)
            tangential_jump_norm = (
                np.sqrt(np.sum(tangential_jump**2 * g.cell_volumes)) / vol
            )
            normal_jump_norm = (
                np.sqrt(np.sum(normal_jump**2 * g.cell_volumes)) / vol
            )
            ...
```

**位移跳跃的体积加权均方根（行 1067–1072）：**

切向位移跳跃的体积加权 RMS 值：

$$\|\llbracket \mathbf{u}_\tau \rrbracket\|_\text{rms} = \frac{\sqrt{\sum_i |\llbracket \mathbf{u}_\tau \rrbracket_i|^2 \cdot V_i}}{\sum_i V_i}$$

法向位移跳跃的体积加权 RMS 值：

$$\|\llbracket u_n \rrbracket\|_\text{rms} = \frac{\sqrt{\sum_i |\llbracket u_n \rrbracket_i|^2 \cdot V_i}}{\sum_i V_i}$$

其中 $V_i$ 为第 $i$ 个裂缝单元的体积（面积），$\sum_i V_i$ 为裂缝总面积。

类似地，接触力切向和法向分量的体积加权 RMS 也按相同方式计算（行 1093–1100）。

---

## 10. 辅助类 `Water` 与 `Granite`（行 1249–1303）

### `Water` 类（行 1249–1283）

```python
class Water:
    def __init__(self, theta_ref=None):
        self.theta_ref = 20 * (pp.CELSIUS)  # 参考温度 20°C
        self.VISCOSITY = 1 * pp.MILLI * pp.PASCAL * pp.SECOND  # 动力黏度
        self.COMPRESSIBILITY = 1e-10 / pp.PASCAL                # 压缩系数
        self.BULK_MODULUS = 1 / self.COMPRESSIBILITY             # 体积模量

    def thermal_expansion(self, delta_theta):
        return 4e-4            # 体积热膨胀系数 β_f [m³/(m³·K)]

    def thermal_conductivity(self, theta=None):
        return 0.6             # 热导率 κ_f [W/(m·K)]

    def specific_heat_capacity(self, theta=None):
        return 4200            # 比热容 c_f [J/(kg·K)]

    def dynamic_viscosity(self, theta=None):
        return 0.001           # 动力黏度 μ [Pa·s]

    def hydrostatic_pressure(self, depth, theta=None):
        rho = 1e3 * (pp.KILOGRAM / pp.METER**3)
        return rho * depth * pp.GRAVITY_ACCELERATION + pp.ATMOSPHERIC_PRESSURE
```

**水静压力公式（行 1282–1283）：**

$$p_\text{hydrostatic}(z) = \rho_0 g z + p_\text{atm}$$

其中 $\rho_0 = 1000\,\text{kg/m}^3$，$g = 9.81\,\text{m/s}^2$，$p_\text{atm} \approx 101325\,\text{Pa}$。

### `Granite` 类（行 1286–1302）

```python
class Granite(pp.Granite):
    def __init__(self, theta_ref=None):
        super().__init__(theta_ref)
        self.BULK_MODULUS = pp.params.rock.bulk_from_lame(self.LAMBDA, self.MU)
        self.PERMEABILITY = 1e-15   # 渗透率 K_m [m²]

    def thermal_conductivity(self, theta=None):
        return 3.0    # 热导率 κ_s [W/(m·K)]

    def specific_heat_capacity(self, theta=None):
        return 790.0  # 比热容 c_s [J/(kg·K)]
```

花岗岩体积模量由拉梅参数计算（行 1293）：

$$K_s = \lambda + \frac{2}{3}\mu$$

其他参数（密度 $\rho_s$、热膨胀系数 $\alpha_{T,s}$、$\lambda$、$\mu$）从 PorePy 的 `pp.Granite` 基类继承。

---

## 汇总：物理参数表

| 参数 | 符号 | 数值 | 单位 |
|------|------|------|------|
| 计算域 | $\Omega$ | $[0,2] \times [0,1]$ | m² |
| 参考温度 | $T_0$ | 300 | K |
| 初始裂缝开度 | $a_0$ | $5 \times 10^{-4}$ | m |
| Biot 系数（基质） | $\alpha$ | 0.8 | — |
| Biot 系数（裂缝） | $\alpha$ | 1.0 | — |
| 摩擦系数 | $\mu_f$ | 0.5 | — |
| 剪胀角 | $\varphi$ | 5° | rad |
| 压力缩放 | $p_\text{scale}$ | $10^9$ | Pa |
| 流体黏度 | $\mu$ | $10^{-3}$ | Pa·s |
| 流体压缩系数 | $C_f$ | $10^{-10}$ | Pa⁻¹ |
| 流体热膨胀 | $\beta_f$ | $4 \times 10^{-4}$ | K⁻¹ |
| 流体热导率 | $\kappa_f$ | 0.6 | W/(m·K) |
| 流体比热容 | $c_f$ | 4200 | J/(kg·K) |
| 岩石渗透率 | $K_m$ | $10^{-15}$ | m² |
| 岩石热导率 | $\kappa_s$ | 3.0 | W/(m·K) |
| 岩石比热容 | $c_s$ | 790 | J/(kg·K) |
| 基质孔隙率 | $\phi$ | 0.01 | — |
| 西侧施加压力（阶段 II+） | $p_\text{west}$ | $4 \times 10^7$ | Pa |
| 西侧降温（阶段 III+） | $\Delta T_\text{west}$ | $-15$ | K |
| 北侧位移（$x$方向） | $u_x$ | $5 \times 10^{-4}$ | m |
| 北侧位移（$y$方向） | $u_y$ | $-2 \times 10^{-4}$ | m |
| 非线性收敛容忍度 | $\varepsilon_\text{tol}$ | $10^{-8}$ | — |
| 最大牛顿迭代次数 | $N_\text{max}$ | 50 | — |
| 仿真结束时间 | $t_\text{end}$ | 10 | h |
