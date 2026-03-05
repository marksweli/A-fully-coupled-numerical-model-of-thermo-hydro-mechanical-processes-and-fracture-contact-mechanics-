# `ex1_main.py` 代码详细解释

> 本文档按主函数（`if __name__ == "__main__":` 块）的运行顺序，对 `ex1_main.py` 中的每一行代码进行详细解释。  
> 涉及计算的代码给出对应的数学公式，行号在前，公式在后，分段对应。  
> 所有公式和行内符号均使用 LaTeX 渲染。

---

## 目录

1. [模块导入（第 31–43 行）](#1-模块导入第-31--43-行)
2. [主函数入口与参数配置（第 1305–1327 行）](#2-主函数入口与参数配置第-1305--1327-行)
3. [网格循环前的初始化（第 1327–1333 行）](#3-网格循环前的初始化第-1327--1333-行)
4. [细化等级主循环——建立网格（第 1335–1345 行）](#4-细化等级主循环建立网格第-1335--1345-行)
5. [模型实例化与字段初始化（第 1347 行 → `__init__` → `_set_fields`）](#5-模型实例化与字段初始化)
6. [仿真准备：`prepare_simulation`（第 1350 行触发，第 738–751 行）](#6-仿真准备-prepare_simulation)
   - [6.1 创建网格：`create_grid`（第 95–113 行）](#61-创建网格-create_grid)
   - [6.2 时间参数：`_set_time_parameters`（第 1154–1168 行）](#62-时间参数-_set_time_parameters)
   - [6.3 岩石与流体：`_set_rock_and_fluid`（第 442–448 行）](#63-岩石与流体-_set_rock_and_fluid)
   - [6.4 初始条件：`_initial_condition`（第 1112–1141 行）](#64-初始条件-_initial_condition)
   - [6.5 参数设置：`_set_parameters`](#65-参数设置-_set_parameters)
     - [6.5.1 力学参数：`_set_mechanics_parameters`（第 450–521 行）](#651-力学参数-_set_mechanics_parameters)
     - [6.5.2 流体参数：`_set_scalar_parameters`（第 523–577 行）](#652-流体参数-_set_scalar_parameters)
     - [6.5.3 温度参数：`_set_temperature_parameters`（第 611–704 行）](#653-温度参数-_set_temperature_parameters)
7. [时间推进与 Newton 迭代（第 1350 行内部循环）](#7-时间推进与-newton-迭代)
   - [7.1 Newton 迭代前准备：`before_newton_loop` / `before_newton_iteration`](#71-newton-迭代前准备)
   - [7.2 线性系统组装与求解：`assemble_and_solve_linear_system`（第 796–817 行）](#72-线性系统组装与求解)
   - [7.3 收敛性检验：`check_convergence`（第 819–948 行）](#73-收敛性检验)
   - [7.4 Newton 收敛后处理：`after_newton_convergence`（第 786–794 行）](#74-newton-收敛后处理)
8. [导出与数据保存（第 1353–1355 行）](#8-导出与数据保存)
9. [多网格精化后处理（第 1357–1360 行）](#9-多网格精化后处理)
10. [辅助类：`Water`（第 1249–1283 行）](#10-辅助类-water)
11. [辅助类：`Granite`（第 1286–1302 行）](#11-辅助类-granite)
12. [辅助函数说明](#12-辅助函数说明)

---

## 1. 模块导入（第 31–43 行）

```python
31  import logging
32  import os
33  import time
34
35  import numpy as np
36  import porepy as pp
37  import scipy.sparse.linalg as spla
38  from porepy.models.thm_model import THM
39
40  import thm_utils
41
42  logging.basicConfig(level=logging.INFO)
43  logger = logging.getLogger(__name__)
```

**说明：**

- **第 31–33 行**：导入 Python 标准库模块。`logging` 用于记录运行日志；`os` 用于创建目录等操作系统交互；`time` 用于计时。
- **第 35 行**：导入 `numpy`，提供多维数组与线性代数运算。
- **第 36 行**：导入 `porepy`，这是一个专门针对多孔介质中多物理场问题的开源 Python 库，涵盖网格生成、离散化、参数管理等。
- **第 37 行**：导入 `scipy.sparse.linalg`，提供稀疏线性方程组求解器（如 `spsolve`）。
- **第 38 行**：从 `porepy` 中导入 `THM`（Thermo-Hydro-Mechanical，热-水-力）模型基类。
- **第 40 行**：导入本地辅助模块 `thm_utils`，其中包含结果写入、读取及绘图函数。
- **第 42–43 行**：配置日志系统，设置默认输出级别为 `INFO`，并为当前模块创建日志记录器 `logger`。

---

## 2. 主函数入口与参数配置（第 1305–1327 行）

```python
1305  if __name__ == "__main__":
1306      mesh_size = 0.8
1307      mesh_args = {
1308          "mesh_size_frac": mesh_size,
1309          "mesh_size_min": 0.5 * mesh_size,
1310          "mesh_size_bound": 3.6 * mesh_size,
1311      }
1312      params = {
1313          "nl_convergence_tol": 1e-8,
1314          "max_iterations": 50,
1315          "file_name": "thm",
1316          "dilation_angle": np.radians(5),
1317          "dilation_angle_constant_g": 0,
1318          "use_umfpack": True,
1319          "mesh_args": mesh_args,
1320      }
1321
1322      root_folder = "ex1/"
1323
1324      gmsh_file_base = root_folder + "mesh_"
1325      pickle_file_name = root_folder + "gblist"
1326      folder_base = root_folder + "convergence_study_"
1327      gb_list, models = [], []
```

**说明：**

- **第 1305 行**：Python 标准保护，仅当脚本直接运行时才执行以下代码。
- **第 1306–1311 行**：设置 Gmsh 网格生成参数。`mesh_size_frac` 控制裂缝附近网格尺寸，`mesh_size_min` 为最小单元尺寸，`mesh_size_bound` 为边界附近单元尺寸。
  - 裂缝区最小单元尺寸为 \(h_{min} = 0.5 \times 0.8 = 0.4\)，边界区最大为 \(h_{bound} = 3.6 \times 0.8 = 2.88\)。
- **第 1312–1320 行**：设置求解器参数字典：
  - `nl_convergence_tol`：非线性（Newton）迭代的收敛容差，\(\varepsilon_{tol} = 10^{-8}\)；
  - `max_iterations`：最大迭代次数 50；
  - `dilation_angle`：剪胀角（裂缝开度随切向滑移增大的角度），\(\psi = 5°\)，转换为弧度：

\[
\psi = \frac{5 \times \pi}{180} \approx 0.08727 \text{ rad}
\]

  - `dilation_angle_constant_g`：点（交叉点）处剪胀角设为 0，即交叉点不参与剪胀计算；
  - `use_umfpack`：使用 UMFPACK 稀疏直接求解器（需要 64 位整数索引）。
- **第 1322–1327 行**：设置结果存储目录结构并初始化空列表 `gb_list`（存储各精化等级的 GridBucket）和 `models`（存储模型对象）。

---

## 3. 网格循环前的初始化（第 1328–1333 行）

```python
1328      refinement_levels = np.arange(0, 6)
1329
1330      if refinement_levels[0] > 1:
1331          gb_list = thm_utils.read_pickle(pickle_file_name)
1332          if gb_list is None:
1333              gb_list = []
```

**说明：**

- **第 1328 行**：创建整数数组 \([0, 1, 2, 3, 4, 5]\)，表示六个网格细化等级（从粗到细）。
- **第 1330–1333 行**：若起始细化等级大于 1（即从中间断点续算），则从磁盘上的 pickle 文件中读取已有的 GridBucket 列表，以便在已有结果基础上继续。若文件不存在或为空，则重置为空列表。

---

## 4. 细化等级主循环——建立网格（第 1335–1345 行）

```python
1335      for i in refinement_levels:
1336          # Set file names for saving of results etc.
1337          folder_name = folder_base + str(i)
1338          params["folder_name"] = folder_name
1339          if not os.path.exists(folder_name):
1340              os.makedirs(folder_name)
1341
1342          # Create grid bucket from existing mesh files for reproducibility
1343          msh_file = gmsh_file_base + str(i) + ".msh"
1344          grid_list = pp.fracs.simplex.triangle_grid_from_gmsh(file_name=msh_file)
1345          gb = pp.fracs.meshing.grid_list_to_grid_bucket(grid_list)
```

**说明：**

- **第 1337–1340 行**：为第 \(i\) 个细化等级设置结果文件夹路径（例如 `ex1/convergence_study_0`），并在磁盘上创建该目录（若不存在）。
- **第 1343 行**：构造预先由 Gmsh 生成的网格文件路径（如 `ex1/mesh_0.msh`）。使用预生成的固定网格文件是为了保证结果的可重复性（reproducibility）。
- **第 1344 行**：调用 PorePy 内部函数，从 Gmsh `.msh` 文件中读取三角网格并生成网格对象列表。
- **第 1345 行**：将网格对象列表转换为 PorePy 的 `GridBucket`——一个包含不同维度（2D 基质、1D 裂缝、0D 交叉点）子域及其接口（mortar grid）的数据容器，是整个多维度模拟框架的核心数据结构。

---

## 5. 模型实例化与字段初始化

```python
1347          m = Example1Model(params, gb)
```

这一行触发 `Example1Model.__init__`（第 90–93 行）：

```python
90      def __init__(self, params, gb):
91          super().__init__(params)
92          # Set additional case specific fields
93          self._set_fields(params, gb)
```

**`super().__init__(params)`**：调用 `THM` 基类（来自 PorePy）的初始化方法，设置通用物理量缩放系数（`length_scale`、`temperature_scale`）、变量名称、离散化方案名称等。

**`_set_fields(params, gb)`**（第 1199–1234 行）：

```python
1199      def _set_fields(self, params, gb=None):
1200          self.T_0_Kelvin = 300
1201          self.background_temp_C = self.T_0_Kelvin - 273
...
1209          self.scalar_scale = 1e9
...
1227          self.initial_aperture = 5e-4 / self.length_scale
1230          self._dilation_angle = params.get("dilation_angle")
1231          self._dilation_angle_constant_g = params.get("dilation_angle_constant_g")
```

**说明：**

- **第 1200 行**：设定参考温度 \(T_0 = 300\) K（27°C）。
- **第 1201 行**：换算为摄氏度 \(T_{0,C} = 300 - 273 = 27\)°C。
- **第 1209 行**：压力缩放系数 \(p_{scale} = 10^9\) Pa = 1 GPa，用于将方程无量纲化，改善线性系统条件数。
- **第 1227 行**：初始裂缝开度（aperture），以长度缩放系数（`length_scale`，通常为 1 m）归一化：

\[
a_0 = \frac{5 \times 10^{-4}}{l_{scale}} \quad [\text{m}]
\]

- **第 1230–1231 行**：存储剪胀角 \(\psi\)（单位 rad）及交叉点处剪胀角（取 0）。

---

## 6. 仿真准备：`prepare_simulation`

`pp.run_time_dependent_model(m, params)`（第 1350 行）首先调用 `m.prepare_simulation()`（第 738–751 行）：

```python
738      def prepare_simulation(self):
739          self.create_grid()
740          self._set_time_parameters()
741          self._set_rock_and_fluid()
742          self._initial_condition()
743          self._set_parameters()
744          self._assign_variables()
745          self._assign_discretizations()
746          self._discretize()
747          self._initialize_linear_solver()
748          ...
751          self._export_step()
```

### 6.1 创建网格：`create_grid`

```python
95      def create_grid(self):
101          gb = self.gb
104          self.box = {
105              "xmin": 0,
106              "ymin": 0,
107              "xmax": 2 / self.length_scale,
108              "ymax": 1 / self.length_scale,
109          }
110          pp.contact_conditions.set_projections(gb)
112          self._Nd = gb.dim_max()
113          self.n_frac = len(gb.grids_of_dimension(self._Nd - 1))
```

**说明：**

- **第 104–109 行**：以长度缩放系数归一化后的计算域为 \(\Omega = [0, 2] \times [0, 1]\)（单位：m）。
- **第 110 行**：`pp.contact_conditions.set_projections(gb)` 为每条裂缝计算切向–法向投影矩阵，用于后续接触力学约束的本地坐标系转换。
- **第 112 行**：`self._Nd = 2`（二维问题）。
- **第 113 行**：统计裂缝（1D 子域）数量，本例共 8 条裂缝。

---

### 6.2 时间参数：`_set_time_parameters`

```python
1154      def _set_time_parameters(self):
1160          self.time = self.params.get("time", -1e6)
1162          self.time_step = -self.time
1165          self.end_time = 10 * pp.HOUR
1166          self.max_time_step = self.end_time / 5
1167          self.phase_limits = np.array([self.time, 0, 3e1, self.end_time])
1168          self.phase_time_steps = np.array([self.time_step, 2e0, 2 / 3 * pp.HOUR, 1.0])
```

**说明：**

仿真分为四个阶段：

| 阶段 | 时间范围 | 驱动力 |
|------|----------|--------|
| I（平衡） | \(t = -10^6\) s → \(t = 0\) | 仅施加力学边界条件，一步跨越平衡 |
| II  | \(t = 0\) → \(t = 30\) s | 左边界附加压力梯度（\(p_{west} = 4 \times 10^7\) Pa） |
| III | \(t = 30\) s → \(t = 36000\) s（= 10 h） | 左边界温度降低 15°C（\(T_{west} = 285\) K） |

- **第 1160 行**：初始时间 \(t_0 = -10^6\) s（一个大的负时间步，用于先让力学平衡）。
- **第 1162 行**：初始时间步 \(\Delta t = -t_0 = 10^6\) s，一步跨越平衡阶段。
- **第 1165–1166 行**：总模拟时间 \(T_{end} = 10 \times 3600 = 36000\) s，最大时间步 \(\Delta t_{max} = T_{end}/5 = 7200\) s。

---

### 6.3 岩石与流体：`_set_rock_and_fluid`

```python
442      def _set_rock_and_fluid(self):
447          self.rock = Granite()
448          self.fluid = Water()
```

**说明：**

- 实例化花岗岩固体（`Granite`，继承自 PorePy 的内置类，见第 1286–1302 行）和水（`Water`，见第 1249–1283 行）。两个类的具体参数详见本文第 10–11 节。

---

### 6.4 初始条件：`_initial_condition`

```python
1112      def _initial_condition(self) -> None:
1113          for g, d in self.gb:
1114              d[pp.PARAMETERS] = pp.Parameters()
...
1123          self._update_all_apertures(to_iterate=False)
1124          self._update_all_apertures()
1125          super()._initial_condition()
1127          for g, d in self.gb:
1128              d[pp.STATE]["cell_centers"] = g.cell_centers.copy()
1129              p0 = self._initial_scalar(g)
1130              T0 = self._initial_temperature(g)
1131              state = {
1132                  self.scalar_variable: p0,
1133                  self.temperature_variable: T0,
1134                  "previous_u_exp": np.zeros((3, g.num_cells)),
1135              }
...
1140              pp.set_state(d, state)
1141              pp.set_iterate(d, iterate)
```

**说明：**

- **第 1113–1122 行**：为 GridBucket 中的每个子域初始化参数容器（`pp.Parameters`），为后续各物理场的参数存储做准备。
- **第 1123–1124 行**：分别将"上一时间步"与"当前迭代"状态的裂缝开度（aperture）初始化为初始值 \(a_0\)。
- **第 1125 行**：调用父类 THM 的初始条件方法，初始化位移 \(\mathbf{u} = \mathbf{0}\) 和接触牵引力 \(\boldsymbol{\lambda} = \mathbf{0}\) 等力学变量。
- **第 1129 行**：初始压力场（`_initial_scalar`，第 1143–1149 行）：

```python
1143      def _initial_scalar(self, g) -> np.ndarray:
1149          return self.fluid.hydrostatic_pressure(depth) / self.scalar_scale
```

当重力关闭时（本例默认），`depth = 0`，初始压力为大气压：

\[
p_0 = \frac{p_{atm}}{p_{scale}} = \frac{101325 \text{ Pa}}{10^9 \text{ Pa}} \approx 1.01 \times 10^{-4}
\]

- **第 1130 行**：初始温度场（`_initial_temperature`，第 1151–1152 行）：

\[
T_0 = \frac{T_{0,K}}{T_{scale}} = \frac{300 \text{ K}}{T_{scale}}
\]

其中 \(T_{scale}\) 为温度缩放系数（由 PorePy THM 基类指定，通常为 1 K）。

---

### 6.5 参数设置：`_set_parameters`

`_set_parameters` 依次调用以下三个子方法，分别设置力学、流体（标量）、温度参数。

---

#### 6.5.1 力学参数：`_set_mechanics_parameters`（第 450–521 行）

```python
450      def _set_mechanics_parameters(self):
455          gb = self.gb
456          for g, d in gb:
457              if g.dim == self._Nd:
459                  rock = self.rock
460                  lam = rock.LAMBDA * np.ones(g.num_cells) / self.scalar_scale
461                  mu = rock.MU * np.ones(g.num_cells) / self.scalar_scale
462                  C = pp.FourthOrderTensor(mu, lam)
```

**力学本构参数（四阶弹性张量）：**

弹性刚度张量 \(\mathbf{C}\) 由 Lamé 常数 \(\lambda\)（第一拉梅常数）和 \(\mu\)（剪切模量）描述：

\[
\sigma_{ij} = \lambda \varepsilon_{kk} \delta_{ij} + 2\mu \varepsilon_{ij}
\]

其中 \(\boldsymbol{\sigma}\) 为应力张量，\(\boldsymbol{\varepsilon}\) 为应变张量。代码中除以 `scalar_scale` 以归一化。

```python
464                  bc = self._bc_type_mechanics(g)
465                  bc_values = self._bc_values_mechanics(g)
```

**边界条件——力学（第 121–146 行）：**

```python
121      def _bc_type_mechanics(self, g) -> pp.BoundaryConditionVectorial:
125          _, _, _, north, south, _, _ = self._domain_boundary_sides(g)
126          bc = pp.BoundaryConditionVectorial(g, south + north, "dir")
```

- 在北（top，\(y = 1\) m）和南（bottom，\(y = 0\)）边界施加 Dirichlet 位移边界条件，东西边界自然为零牵引（Neumann）。

```python
133      def _bc_values_mechanics(self, g) -> np.ndarray:
141          x = 5e-4 / self.length_scale
142          y = -2e-4 / self.length_scale
144          bc_values[0, north] = x
145          bc_values[1, north] = y
```

北边界施加位移：

\[
u_x^{N} = \frac{5 \times 10^{-4}}{l_{scale}} \text{ m}, \quad u_y^{N} = \frac{-2 \times 10^{-4}}{l_{scale}} \text{ m}
\]

南边界位移固定为零（第 139 行初始化为零）：

\[
u_x^{S} = 0, \quad u_y^{S} = 0
\]

**Biot 耦合系数（第 184–188 行）：**

```python
184      def _biot_alpha(self, g) -> np.ndarray:
185          if g.dim == self._Nd:
186              return 0.8
187          else:
188              return 1.0
```

对于基质（2D），Biot-Willis 系数：

\[
\alpha = 0.8
\]

对于裂缝（1D），\(\alpha = 1.0\)（裂缝内流体与固体完全耦合）。

**热弹性耦合系数（第 190–210 行）：**

```python
190      def _biot_beta(self, g):
197              return self.rock.BULK_MODULUS * 3 * self.rock.THERMAL_EXPANSION
```

对于基质（2D），热弹性耦合系数 \(\beta\)（体积热膨胀导致的应力）：

\[
\beta = K_s \cdot 3\alpha_s
\]

其中 \(K_s\) 为岩石体积模量，\(\alpha_s\) 为线热膨胀系数（PorePy 中存储线膨胀系数，乘以 3 换算为体积膨胀系数）。

对于裂缝（1D，第 202–209 行）：

```python
202              _, T_k, _ = self._variable_increment(g, self.temperature_variable)
204              beta = (
205                  T_k
206                  / self.T_0_Kelvin
207                  * self._fluid_density(g)
208                  * self.fluid.specific_heat_capacity()
209              )
```

\[
\beta_{frac} = \frac{T_k}{T_0} \rho_f c_f
\]

其中 \(T_k\) 为当前迭代温度，\(\rho_f\) 为流体密度，\(c_f\) 为流体比热容。

**裂缝接触参数（第 499–511 行）：**

```python
499              elif g.dim == self._Nd - 1:
500                  pp.initialize_data(
501                      g, d, self.mechanics_parameter_key,
502                      {
503                          "friction_coefficient": self._set_friction_coefficient(g),
504                          "contact_mechanics_numerical_parameter": 1e1,
505                          "dilation_angle": self._dilation_angle,
506                      },
507                  )
```

- 摩擦系数 \(\mu_f = 0.5\)（第 1176 行）；
- 接触力学数值罚参数 \(c_\nu = 10\)；
- 剪胀角 \(\psi = 5°\)（弧度值）。

---

#### 6.5.2 流体参数：`_set_scalar_parameters`（第 523–577 行）

```python
523      def _set_scalar_parameters(self):
524          for g, d in self.gb:
525              a = self._aperture(g)
526              specific_volumes = self._specific_volumes(g)
527              bc = self._bc_type_scalar(g)
528              bc_values = self._bc_values_scalar(g)
529              biot_coefficient = self._biot_alpha(g)
530              compressibility = self.fluid.COMPRESSIBILITY
531              mass_weight = compressibility * self._porosity(g)
532              if g.dim == self._Nd:
533                  mass_weight += (
534                      biot_coefficient - self._porosity(g)
535                  ) / self.rock.BULK_MODULUS
```

**压力方程质量权重（storage 项）：**

对于基质（2D），压缩性质量权重 \(S_\varepsilon\)：

\[
S_\varepsilon = C_f \phi + \frac{\alpha - \phi}{K_s}
\]

其中 \(C_f\) 为流体压缩系数，\(\phi\) 为孔隙率，\(\alpha\) 为 Biot-Willis 系数，\(K_s\) 为岩石骨架体积模量。

```python
538          elif g.dim == 1 and (self._dilation_angle_constant_g > 1e-5):
539              biot_coefficient *= 0
540          mass_weight *= self.scalar_scale * specific_volumes
```

若剪胀角为零（本例），裂缝中不计 Biot 耦合（`biot_coefficient *= 0`）；对于裂缝（1D）：

\[
S_\varepsilon^{frac} = C_f \cdot \phi_{frac}
\]

实际存储质量权重乘以 \(p_{scale}\) 和比容 \(V\)：

\[
\tilde{S}_\varepsilon = S_\varepsilon \cdot p_{scale} \cdot V
\]

**流体边界条件（第 148–163 行）：**

```python
148      def _bc_type_scalar(self, g) -> pp.BoundaryCondition:
153          _, east, west, *_ = self._domain_boundary_sides(g)
154          return pp.BoundaryCondition(g, east + west, "dir")
```

东（\(x = 2\) m）和西（\(x = 0\)）边界施加 Dirichlet 压力，上下边界为零流量（Neumann）。

```python
157      def _bc_values_scalar(self, g) -> np.ndarray:
159          _, _, west, *_ = self._domain_boundary_sides(g)
160          bc_values = np.zeros(g.num_faces)
161          if self.time > self.phase_limits[1]:
162              bc_values[west] = 4e7 / self.scalar_scale
```

阶段 I 开始后，西边界施加超压：

\[
p_{west} = \frac{4 \times 10^7 \text{ Pa}}{p_{scale}} = \frac{4 \times 10^7}{10^9} = 0.04
\]

东边界保持 \(p_{east} = 0\)（参考值），即实际压力梯度：

\[
\Delta p = p_{west} - p_{east} = 4 \times 10^7 \text{ Pa}
\]

**温度→压力耦合系数（第 257–270 行）：**

```python
257      def _temperature_to_scalar_coupling_coefficient(self, g) -> float:
263          b_f = self.fluid.thermal_expansion(0)
264          if g.dim < self._Nd:
265              coeff = -b_f
266          else:
267              b_s = 3 * self.rock.THERMAL_EXPANSION
268              phi = self._porosity(g)
269              coeff = -b_f * phi - (1 - phi) * b_s
270          return coeff
```

温度对压力方程的耦合系数（热膨胀导致孔隙压力变化）：

基质（2D）：
\[
\gamma_{T \to p} = -\beta_f \phi - (1 - \phi) \beta_s
\]

裂缝（1D）：
\[
\gamma_{T \to p}^{frac} = -\beta_f
\]

其中 \(\beta_f = 4 \times 10^{-4}\) K\(^{-1}\) 为流体体积热膨胀系数，\(\beta_s = 3\alpha_s\) 为固体体积热膨胀系数（线膨胀系数的 3 倍）。

**渗透率（裂缝立方定律，第 389–440 行）：**

```python
389      def _set_permeability_from_aperture(self):
401          apertures = self._aperture(g, from_iterate=True)
402          apertures_unscaled = apertures * self.length_scale
403          k = np.power(apertures_unscaled, 2) / 12 / viscosity
```

裂缝渗透率采用立方定律（Cubic Law）：

\[
k_{frac} = \frac{a^2}{12\mu}
\]

其中 \(a\) 为裂缝开度（m），\(\mu\) 为流体动力黏度（Pa·s）。

基质渗透率（第 415–420 行）：

\[
k_{matrix} = \frac{K_{perm}}{\mu}
\]

其中 \(K_{perm} = 10^{-15}\) m\(^2\)（基质固有渗透率）。

法向渗透率（接口，第 425–440 行）：

\[
k_n = \frac{k_s}{a \cdot V / 2}
\]

其中分母代表沿法向取半个开度的梯度，\(k_s\) 为从裂缝继承的切向渗透率，\(V\) 为比容。

---

#### 6.5.3 温度参数：`_set_temperature_parameters`（第 611–704 行）

```python
611      def _set_temperature_parameters(self):
615          div_T_scale = self.temperature_scale / self.length_scale ** 2 / self.T_0_Kelvin
616          kappa_f = self.fluid.thermal_conductivity() * div_T_scale
617          kappa_s = self.rock.thermal_conductivity() * div_T_scale
```

热传导系数缩放：

\[
\kappa_{scaled} = \kappa \cdot \frac{T_{scale}}{l_{scale}^2 \cdot T_0}
\]

```python
621          heat_capacity_f = (
622              self._fluid_density(g) * self.fluid.specific_heat_capacity()
623          )
```

流体体积热容：

\[
(\rho c)_f = \rho_f \cdot c_f
\]

**流体密度（第 212–236 行）：**

```python
212      def _fluid_density(self, g, dp=None, dT=None) -> np.ndarray:
219          dp = p_k - pp.ATMOSPHERIC_PRESSURE
226          dT = T_k - self.T_0_Kelvin
232          rho_0 = 1e3 * (pp.KILOGRAM / pp.METER ** 3) * np.ones(g.num_cells)
233          rho = rho_0 * np.exp(
234              dp * self.fluid.COMPRESSIBILITY - dT * self.fluid.thermal_expansion(dT)
235          )
```

流体密度由状态方程（线性化 EOS）给出：

\[
\rho_f = \rho_0 \exp\!\left(C_f \Delta p - \beta_f \Delta T\right)
\]

其中：
- \(\rho_0 = 1000\) kg/m³（参考密度）
- \(C_f = 10^{-10}\) Pa\(^{-1}\)（流体压缩系数）
- \(\beta_f = 4 \times 10^{-4}\) K\(^{-1}\)（体积热膨胀系数）
- \(\Delta p = p_k - p_{atm}\)（超压）
- \(\Delta T = T_k - T_0\)（温度偏差），被截断在 \(\pm T_0/3\) 以确保收敛

**有效热容（第 637–647 行）：**

```python
637          effective_heat_capacity = porosity * heat_capacity_f * (
638              1 - T_k * self.fluid.thermal_expansion(0)
639          ) + (1 - porosity) * heat_capacity_s * (
640              1 - T_k * 3 * self.rock.THERMAL_EXPANSION
641          )
642          mass_weight = (
643              effective_heat_capacity
644              * specific_volumes
645              * self.temperature_scale
646              / self.T_0_Kelvin
647          )
```

有效体积热容（考虑热膨胀修正）：

\[
(\rho c)_{eff} = \phi \rho_f c_f \bigl(1 - T_k \beta_f\bigr)
+ (1 - \phi) \rho_s c_s \bigl(1 - 3 \alpha_s T_k\bigr)
\]

质量权重归一化：

\[
\tilde{m}_T = (\rho c)_{eff} \cdot V \cdot \frac{T_{scale}}{T_0}
\]

**有效热传导系数（第 649–652 行）：**

```python
649          effective_conductivity = porosity * kappa_f + (1 - porosity) * kappa_s
650          thermal_conductivity = pp.SecondOrderTensor(
651              effective_conductivity * specific_volumes
652          )
```

有效热传导系数（Voigt 平均）：

\[
\kappa_{eff} = \phi \kappa_f + (1 - \phi) \kappa_s
\]

**热对流权重（第 654–656 行）：**

```python
654          advection_weight = (
655              heat_capacity_f * self.temperature_scale / self.T_0_Kelvin
656          )
```

\[
w_{adv} = (\rho c)_f \cdot \frac{T_{scale}}{T_0}
\]

**压力→温度耦合系数（第 238–255 行，第 675–687 行）：**

```python
238      def _scalar_to_temperature_coupling_coefficient(self, g) -> float:
244          c_f = self.fluid.specific_heat_capacity()
245          rho_f = self._fluid_density(g)
246          C_f = self.fluid.COMPRESSIBILITY
247          if g.dim < self._Nd:
248              coeff = c_f * rho_f * C_f
249          else:
250              ...
251              coeff = phi * c_f * rho_f * C_f + (1 - phi) * c_s * rho_s / K_s
252          return coeff / self.T_0_Kelvin
```

压力对温度方程的耦合系数（Dufour 型压缩热效应）：

基质（2D）：
\[
\gamma_{p \to T} = \frac{1}{T_0}\left[\phi c_f \rho_f C_f + (1-\phi) \frac{c_s \rho_s}{K_s}\right]
\]

裂缝（1D）：
\[
\gamma_{p \to T}^{frac} = \frac{c_f \rho_f C_f}{T_0}
\]

**温度边界条件（第 165–182 行）：**

```python
165      def _bc_type_temperature(self, g) -> pp.BoundaryCondition:
171          return pp.BoundaryCondition(g, east + west, "dir")

174      def _bc_values_temperature(self, g) -> np.ndarray:
179          bc_values[east + west] = self.T_0_Kelvin / self.temperature_scale
180          if self.time > self.phase_limits[2]:
181              bc_values[west] = (self.T_0_Kelvin - 15) / self.temperature_scale
```

东西边界施加 Dirichlet 温度，初始均为参考温度 \(T_0 = 300\) K。阶段 II 开始后，西边界降温 15°C：

\[
T_{west} = T_0 - 15 \text{ K} = 285 \text{ K}
\]

---

### 6.6 其余 `prepare_simulation` 步骤

```python
744      self._assign_variables()
745      self._assign_discretizations()
746      self._discretize()
747      self._initialize_linear_solver()
751      self._export_step()
```

**说明：**

- **`_assign_variables`**：在 GridBucket 中注册所有未知量（位移 \(\mathbf{u}\)、压力 \(p\)、温度 \(T\)、接触牵引力 \(\boldsymbol{\lambda}\)、mortar 位移 \(\boldsymbol{\lambda}_m\)）。
- **`_assign_discretizations`**（第 722–736 行）：为每个子域与接口分配离散化方案。本例覆盖父类方法，对温度耦合接口关闭 `kinv_scaling`（第 732 行），对流体耦合接口开启（第 735 行），以改善刚度矩阵条件数。
- **`_discretize`**：运行全量离散化，计算各方程（动量方程 MPSA、Biot 散度项、Laplace 扩散等）的局部刚度矩阵并组装。
- **`_initialize_linear_solver`**：初始化线性求解器（UMFPACK 或默认 SuperLU）。
- **`_export_step`**（第 956–1021 行）：将初始状态导出为 VTU 文件，供 ParaView 可视化。

---

## 7. 时间推进与 Newton 迭代

`pp.run_time_dependent_model` 实现时间步进外循环和 Newton 内循环，依次调用以下方法。

---

### 7.1 Newton 迭代前准备

**`before_newton_loop`（第 753–755 行）：**

```python
753      def before_newton_loop(self):
754          super().before_newton_loop()
755          self._iteration = 0
```

每个时间步开始时重置迭代计数器 \(k = 0\)。

**`before_newton_iteration`（第 757–782 行）：**

```python
757      def before_newton_iteration(self):
760          self._iteration += 1
761          self.compute_fluxes()
762          self._update_all_apertures(to_iterate=True)
763          self._set_parameters()
...
775          self.assembler.discretize(filt=filt)
```

每次 Newton 迭代开始时：

1. **第 761 行**：由当前压力迭代值计算达西通量（对流项离散需要）；
2. **第 762 行**：根据当前位移跳量更新裂缝开度（`_update_all_apertures`，见下文）；
3. **第 763 行**：重新设定随解变化的参数（如密度、热容、渗透率等）；
4. **第 767–781 行**：对非线性项进行重离散化，过滤器 `filt` 跳过 MPSA 刚度、稳定化、div\_u 梯度、p 梯度和大区 Laplace 扩散（它们是线性的），只重算对流、质量项和裂缝扩散。

**裂缝开度更新（`_update_all_apertures`，第 290–387 行）：**

```python
302              if g.dim == (self._Nd - 1):
305                  apertures *= self.initial_aperture
...
315                  norm_u_n = np.absolute(u_mortar_local[-1])
316                  norm_u_tau = np.linalg.norm(u_mortar_local[:-1], axis=0)
317                  apertures += (
318                      norm_u_n + np.tan(self._dilation_angle_constant_g) * norm_u_tau
319                  )
```

裂缝开度由初始开度、法向位移跳量和切向位移跳量（乘以剪胀角的正切）组成：

\[
a = a_0 + |\llbracket u_n \rrbracket| + \tan(\psi) \|\llbracket \mathbf{u}_\tau \rrbracket\|
\]

其中：
- \(a_0 = 5 \times 10^{-4}\) m 为初始开度；
- \(\llbracket u_n \rrbracket\) 为法向位移跳量（正值表示张开）；
- \(\llbracket \mathbf{u}_\tau \rrbracket\) 为切向位移跳量向量；
- \(\psi\) 为剪胀角（本例 \(\psi = 5°\)，`_dilation_angle_constant_g = 0` 因此 \(\tan(\psi) = 0\) 对交叉点）。

对于交叉点（dim = 0，第 333–385 行），开度继承自相邻裂缝的平均开度，比容取开度的幂次：

\[
V_{0d} = a^{N_d - \text{dim}_{0d}}
\]

---

### 7.2 线性系统组装与求解

```python
796      def assemble_and_solve_linear_system(self, tol):
802          A, b = self.assembler.assemble_matrix_rhs()
803          use_umfpack = self.params.get("use_umfpack", True)
805          if use_umfpack:
806              A.indices = A.indices.astype(np.int64)
807              A.indptr = A.indptr.astype(np.int64)
815          x = spla.spsolve(A, b)
816          logger.info("Solved in {} s.".format(time.time() - t_0))
817          return x
```

**说明：**

- **第 802 行**：组装全局线性系统 \(\mathbf{A} \mathbf{x} = \mathbf{b}\)，其中 \(\mathbf{A}\) 为 THM 耦合雅可比矩阵（稀疏矩阵），\(\mathbf{b}\) 为残差向量。
- **第 805–807 行**：UMFPACK 要求 64 位整数的 CSR 索引，进行类型转换。
- **第 815 行**：调用 `scipy.sparse.linalg.spsolve` 直接求解线性方程组：

\[
\mathbf{A} \mathbf{x} = \mathbf{b} \quad \Rightarrow \quad \mathbf{x} = \mathbf{A}^{-1} \mathbf{b}
\]

\(\mathbf{x}\) 为所有未知量的 Newton 增量 \(\delta \mathbf{x}\)。

---

### 7.3 收敛性检验

```python
819      def check_convergence(self, solution, prev_solution, init_solution, nl_params=None):
...
855          u_mech_now = solution[mech_dof]
856          u_mech_prev = prev_solution[mech_dof]
...
865          difference_in_iterates_T = np.sum((T_now - T_prev) ** 2)
...
869          difference_in_iterates_p = np.sum((p_now - p_prev) ** 2)
...
871          difference_in_iterates_mech = np.sum((u_mech_now - u_mech_prev) ** 2)
872          difference_from_init_mech = np.sum((u_mech_now - u_mech_init) ** 2)
874          contact_norm = np.sum(contact_now ** 2)
875          difference_in_iterates_contact = np.sum((contact_now - contact_prev) ** 2)
```

**收敛准则：**

对每个物理场 \(\xi \in \{u, p, T, \boldsymbol{\lambda}\}\)，计算两种准则：

绝对准则（第 891 行）：
\[
\|\xi^k - \xi^{k-1}\|^2 < \varepsilon_{tol}
\]

相对准则（第 897–900 行）：
\[
\frac{\|\xi^k - \xi^{k-1}\|^2}{\|\xi^k\|^2} < \varepsilon_{tol}
\]

其中 \(\|\cdot\|^2\) 为向量 \(L^2\) 范数的平方（即分量平方和）。

接触力场采用放宽容差（第 921 行，\(10 \times \varepsilon_{tol}\)）。

全局收敛（第 932 行）：

\[
\text{converged} = \text{converged}_u \;\land\; \text{converged}_p \;\land\; \text{converged}_T \;\land\; \text{converged}_{\boldsymbol{\lambda}}
\]

---

### 7.4 Newton 收敛后处理

```python
786      def after_newton_convergence(self, solution, errors, iteration_counter):
787          for g, d in self.gb:
788              d[pp.STATE]["previous_u_exp"] = d[pp.STATE]["u_exp"].copy()
789          super().after_newton_convergence(solution, errors, iteration_counter)
790          self._update_all_apertures(to_iterate=False)
791          self._update_all_apertures(to_iterate=True)
792          self._export_step()
793          self._adjust_time_step()
794          self._save_data(errors, iteration_counter)
```

**说明：**

- **第 787–788 行**：保存当前时间步的位移导出值（用于下一步计算位移增量 \(\Delta\mathbf{u}_{exp}\)）。
- **第 789 行**：调用父类方法，将 Newton 收敛解 \(\mathbf{x}^k\) 更新到 `pp.STATE`（当前时间步解），并推进时间 \(t \leftarrow t + \Delta t\)。
- **第 790–791 行**：分别更新当前时间步和下一迭代的裂缝开度。
- **第 792 行**：将结果写入 VTU 文件（`_export_step`）。
- **第 793 行**：调整下一时间步步长（`_adjust_time_step`，第 1178–1197 行），保证恰好到达各阶段边界时间。
- **第 794 行**：保存各裂缝的位移跳量和接触力数据（`_save_data`，第 1030–1110 行）。

**`_adjust_time_step`（第 1178–1197 行）：**

```python
1185          self.time_step = getattr(self, "time_step_factor", 1.0) * self.time_step
1188          for dt, lim in zip(self.phase_time_steps, self.phase_limits):
1189              diff = self.time - lim
1190              if diff < 0 and -diff <= self.time_step:
1191                  self.time_step = -diff
1193              if np.isclose(self.time, lim):
1194                  self.time_step = dt
1197          self.time_step = min(self.time_step, self.max_time_step)
```

时间步自适应策略：每次收敛后步长扩大（第 1185 行），但在各阶段切换点前精确对齐（第 1190–1191 行），并在阶段切换点后按预设步长 \(\Delta t_{phase}\) 重置（第 1193–1194 行），且不超过最大步长 \(\Delta t_{max}\)。

**`_save_data`（第 1030–1110 行）——位移跳量数据存储：**

```python
1062              tangential_jump = np.linalg.norm(
1063                  u_mortar_local[:-1] * self.length_scale, axis=0
1064              )
1065              normal_jump = u_mortar_local[-1] * self.length_scale
1066              vol = np.sum(g.cell_volumes)
1067              tangential_jump_norm = (
1068                  np.sqrt(np.sum(tangential_jump ** 2 * g.cell_volumes)) / vol
1069              )
1070              normal_jump_norm = (
1071                  np.sqrt(np.sum(normal_jump ** 2 * g.cell_volumes)) / vol
1072              )
```

体积加权 \(L^2\) 范数形式的裂缝平均位移跳量：

\[
\|\llbracket \mathbf{u}_\tau \rrbracket\|_{vol} = \frac{\sqrt{\displaystyle\sum_i \|\llbracket \mathbf{u}_{\tau,i} \rrbracket\|^2 \cdot V_i}}{\displaystyle\sum_i V_i}
\]

\[
\|\llbracket u_n \rrbracket\|_{vol} = \frac{\sqrt{\displaystyle\sum_i \llbracket u_{n,i} \rrbracket^2 \cdot V_i}}{\displaystyle\sum_i V_i}
\]

其中 \(V_i\) 为第 \(i\) 个裂缝单元的体积（1D 长度），\(i\) 遍历裂缝网格的所有单元。

---

## 8. 导出与数据保存（第 1353–1355 行）

```python
1353          m._export_pvd()
1354          thm_utils.write_fracture_data_txt(m)
1355          thm_utils.write_pickle(gb_list, pickle_file_name)
```

**说明：**

- **第 1353 行**（`_export_pvd`，第 1023–1028 行）：将所有时间步 VTU 文件汇总写入 PVD 索引文件（ParaView 可读的时间序列文件）：

```python
1023      def _export_pvd(self):
1028          self.exporter.write_pvd(np.array(self.export_times))
```

- **第 1354 行**（`write_fracture_data_txt`，`thm_utils.py` 第 36–65 行）：将时间、Newton 迭代次数、切向/法向位移跳量、接触力等数据写入文本文件（`.txt`，空格分隔），便于后续绘图分析。
- **第 1355 行**（`write_pickle`，`thm_utils.py` 第 12–26 行）：将已处理的 GridBucket 列表序列化并存储到磁盘，以备后续从断点续算或后处理。

---

## 9. 多网格精化后处理（第 1357–1360 行）

```python
1357      if len(gb_list) > 1:
1358          for gb in gb_list[:-1]:
1359              pp.grids.match_grids.gb_refinement(gb, gb_list[-1])
1360      thm_utils.write_pickle(gb_list, pickle_file_name)
```

**说明：**

- **第 1357–1359 行**：当存在多个细化等级时，为每个粗网格 `gb` 建立到最细网格 `gb_list[-1]` 的映射（`gb_refinement`）。这些映射用于后续的收敛性分析（Richardson 外推法），将不同精化等级上的解插值到同一参考网格上进行误差估计。
- **第 1360 行**：将包含网格映射信息的更新后 GridBucket 列表再次写入磁盘，供可视化和误差分析使用。

---

## 10. 辅助类：`Water`（第 1249–1283 行）

```python
1249  class Water:
1254      def __init__(self, theta_ref=None):
1256          self.theta_ref = 20 * (pp.CELSIUS)
1259          self.VISCOSITY = 1 * pp.MILLI * pp.PASCAL * pp.SECOND
1260          self.COMPRESSIBILITY = 1e-10 / pp.PASCAL
1261          self.BULK_MODULUS = 1 / self.COMPRESSIBILITY
```

水的物性参数（20°C 时的典型值）：

| 参数 | 符号 | 值 | 单位 |
|------|------|----|------|
| 参考温度 | \(\theta_{ref}\) | 20 | °C |
| 动力黏度 | \(\mu\) | \(10^{-3}\) | Pa·s |
| 压缩系数 | \(C_f\) | \(10^{-10}\) | Pa\(^{-1}\) |
| 体积模量 | \(K_f = 1/C_f\) | \(10^{10}\) | Pa |
| 体积热膨胀系数 | \(\beta_f\) | \(4 \times 10^{-4}\) | K\(^{-1}\) |
| 热传导系数 | \(\kappa_f\) | 0.6 | W/(m·K) |
| 比热容 | \(c_f\) | 4200 | J/(kg·K) |

静水压力（第 1281–1283 行）：

\[
p_{hyd}(z) = \rho_0 g z + p_{atm}
\]

其中 \(\rho_0 = 1000\) kg/m³，\(g = 9.81\) m/s²，\(p_{atm} = 101325\) Pa。

---

## 11. 辅助类：`Granite`（第 1286–1302 行）

```python
1286  class Granite(pp.Granite):
1291      def __init__(self, theta_ref=None):
1292          super().__init__(theta_ref)
1293          self.BULK_MODULUS = pp.params.rock.bulk_from_lame(self.LAMBDA, self.MU)
1295          self.PERMEABILITY = 1e-15
```

花岗岩固体参数（继承自 PorePy 内置参数，并扩展）：

| 参数 | 符号 | 值 | 单位 |
|------|------|----|------|
| 第一拉梅常数 | \(\lambda\) | 由 PorePy 给出（典型值 \(\approx 1.6 \times 10^{10}\) Pa） | Pa |
| 剪切模量 | \(\mu\) | 由 PorePy 给出（典型值 \(\approx 1.6 \times 10^{10}\) Pa） | Pa |
| 体积模量 | \(K_s\) | 由拉梅常数推导 | Pa |
| 固有渗透率 | \(K_{perm}\) | \(10^{-15}\) | m² |
| 热传导系数 | \(\kappa_s\) | 3.0 | W/(m·K) |
| 比热容 | \(c_s\) | 790 | J/(kg·K) |

体积模量由拉梅常数计算（第 1293 行）：

\[
K_s = \lambda + \frac{2}{3}\mu
\]

---

## 12. 辅助函数说明

以下函数在主流程中多次被调用，在此统一说明。

### `_variable_increment`（第 1236–1246 行）

```python
1236      def _variable_increment(self, g, variable, scale=1, x0=None):
1241          if x0 is None:
1242              x0 = d[pp.STATE][variable] * scale
1244          x1 = d[pp.STATE][pp.ITERATE][variable] * scale
1245          dx = x1 - x0
1246          return dx, x1, x0
```

提取并计算变量的迭代增量：

\[
\Delta \xi = \xi^{(k)} - \xi^{(n)}
\]

其中 \(\xi^{(k)}\) 为当前 Newton 迭代值，\(\xi^{(n)}\) 为上一时间步值。返回三元组 \((\Delta \xi,\, \xi^{(k)},\, \xi^{(n)})\)。

### `_export_step`（第 956–1021 行）

```python
967          for g, d in self.gb:
968              if g.dim == self._Nd:
970                  pad_zeros = np.zeros((3 - g.dim, g.num_cells))
971                  u = d[pp.STATE][self.displacement_variable].reshape(
972                      (self._Nd, -1), order="F"
973                  )
973                  u_exp = np.vstack((u * self.length_scale, pad_zeros))
```

位移重塑为物理单位（乘以 `length_scale`），并填补零列至 3 分量（ParaView 向量场要求 3 分量）：

\[
\mathbf{u}_{exp} = l_{scale} \cdot \mathbf{u}_{scaled}
\]

```python
987                  displacement_jump_global_coord = (
988                      mg.mortar_to_secondary_avg(nd=self._Nd)
989                      * mg.sign_of_mortar_sides(nd=self._Nd)
990                      * mortar_u
991                  )
```

将 mortar 位移（定义在接口上）映射回裂缝单元的全局坐标系：

\[
\llbracket \mathbf{u} \rrbracket_{global} = \mathbf{P}_{m \to s} \cdot \mathbf{S}_{mortar} \cdot \boldsymbol{\lambda}_m
\]

其中 \(\mathbf{P}_{m \to s}\) 为 mortar→secondary 平均算子，\(\mathbf{S}_{mortar}\) 为 mortar 侧面符号矩阵。

---

> **总结**：`ex1_main.py` 实现了一个完整的 THM（热-水-力学）耦合数值模拟，包含 8 条裂缝的 2D 花岗岩域，模拟了三阶段加载过程（力学平衡 → 流体压力施加 → 温度冷却）。核心数值方法为 MPSA（多点应力近似）和 MPFA（多点通量近似），通过 Newton 迭代处理裂缝接触力学和非线性流体/热物性，并进行了六个细化等级的收敛性研究。
