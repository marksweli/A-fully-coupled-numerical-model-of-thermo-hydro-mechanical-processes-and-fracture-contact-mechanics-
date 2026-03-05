# `ex1_main.py` 代码逐行详解

> 本文按主函数（`if __name__ == "__main__":` 块）的实际运行顺序组织，依次说明每一个被调用的函数、方法或代码段。涉及计算的部分均给出对应数学公式。

---

## 目录

1. [模块顶部文档字符串](#1-模块顶部文档字符串)
2. [导入模块](#2-导入模块)
3. [日志系统初始化](#3-日志系统初始化)
4. [辅助函数与类定义（被主函数调用前已解析）](#4-辅助函数与类定义)
5. [主函数入口 `__main__`](#5-主函数入口-__main__)
6. [网格加载与模型构建](#6-网格加载与模型构建)
7. [`prepare_simulation()` — 仿真准备](#7-prepare_simulation--仿真准备)
   - 7.1 [`create_grid()`](#71-create_grid)
   - 7.2 [`_set_time_parameters()`](#72-_set_time_parameters)
   - 7.3 [`_set_rock_and_fluid()`](#73-_set_rock_and_fluid)
   - 7.4 [`_initial_condition()`](#74-_initial_condition)
   - 7.5 [`_set_parameters()`](#75-_set_parameters)
   - 7.6 [`_assign_variables()` 与 `_assign_discretizations()`](#76-_assign_variables-与-_assign_discretizations)
   - 7.7 [`_discretize()` 与 `_initialize_linear_solver()`](#77-_discretize-与-_initialize_linear_solver)
   - 7.8 [`_export_step()`（初始导出）](#78-_export_step初始导出)
8. [时间步进循环（由 `pp.run_time_dependent_model` 驱动）](#8-时间步进循环)
   - 8.1 [`before_newton_loop()`](#81-before_newton_loop)
   - 8.2 [`before_newton_iteration()`](#82-before_newton_iteration)
   - 8.3 [`assemble_and_solve_linear_system()`](#83-assemble_and_solve_linear_system)
   - 8.4 [`check_convergence()`](#84-check_convergence)
   - 8.5 [`after_newton_convergence()`](#85-after_newton_convergence)
9. [时间步结束后的数据保存与导出](#9-时间步结束后的数据保存与导出)
10. [辅助类 `Water` 与 `Granite`](#10-辅助类-water-与-granite)
11. [重要物理量与参数汇总](#11-重要物理量与参数汇总)

---

## 1. 模块顶部文档字符串

```python
"""
The simulation is set up for four phases:
    I   the system is allowed to reach equilibrium under the influence of the mechanical BCs.
    II  pressure gradient from left to right is added.
    III temperature reduced at left boundary.
...
"""
```

该多行字符串描述了数值模拟的整体物理场景：

| 阶段 | 描述 |
|------|------|
| Phase I  | 施加位移边界条件，系统在力学边界下达到平衡 |
| Phase II | 在左边界施加压力（压差从左到右），激活流体渗流 |
| Phase III| 左边界温度降低 15 K，引入热应力与热传导 |

计算域为 $(0,2)\times(0,1)$（单位 m），包含 **8 条裂缝**，其中两条形成 L 型交叉，两条形成 X 型交叉。模型采用全耦合 THM（热–水–力）模型。

---

## 2. 导入模块

```python
import logging      # Python 标准日志库
import os           # 操作系统接口（文件/目录操作）
import time         # 计时（用于性能记录）

import numpy as np              # 数值计算库
import porepy as pp             # 多孔介质数值模拟框架
import scipy.sparse.linalg as spla  # 稀疏线性方程组求解器
from porepy.models.thm_model import THM  # THM 耦合基类

import thm_utils    # 本地工具模块（序列化、绘图、数据导出）
```

- `numpy`：提供向量/矩阵运算。
- `porepy`：网格生成、有限体积离散化、多维耦合模型框架。
- `scipy.sparse.linalg`：提供稀疏矩阵直接求解器 `spsolve`（默认使用 UMFPACK）。
- `THM`：PorePy 中热–水–力全耦合模型的基类，包含动量平衡、质量守恒（达西流）和能量平衡方程的离散化逻辑。

---

## 3. 日志系统初始化

```python
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

- `logging.basicConfig(level=logging.INFO)`：配置全局日志，将输出级别设为 `INFO`（及以上级别的消息均会输出到控制台）。
- `logger = logging.getLogger(__name__)`：创建本模块的日志记录器，以模块名（`ex1_main`）为标识，方便多模块区分日志来源。

---

## 4. 辅助函数与类定义

Python 解释器在执行 `__main__` 块之前，会**顺序解析**以下定义（不立即执行函数体）：

| 对象 | 类型 | 作用 |
|------|------|------|
| `create_grid(mesh_args)` | 函数 | 创建含 8 条裂缝的二维断裂网络及混维网格 |
| `Example1Model` | 类（继承自 `THM`） | 封装算例的全部参数、边界条件和求解逻辑 |
| `Water` | 类 | 流体（水）物性参数 |
| `Granite` | 类（继承自 `pp.Granite`） | 固体（花岗岩）物性参数 |

---

## 5. 主函数入口 `__main__`

```python
if __name__ == "__main__":
```

Python 的标准守卫语句。当脚本被直接运行（而非作为模块被导入）时，执行以下缩进块中的代码。

### 5.1 网格参数设置

```python
mesh_size = 0.8
mesh_args = {
    "mesh_size_frac": mesh_size,        # 裂缝附近的目标网格尺寸
    "mesh_size_min": 0.5 * mesh_size,   # 全局最小网格尺寸
    "mesh_size_bound": 3.6 * mesh_size, # 外边界附近的目标网格尺寸
}
```

这三个参数传递给 Gmsh 网格生成器，控制网格密度分布。`mesh_size_frac=0.8` 表示裂缝周围的三角形单元边长约为 0.8 m。

### 5.2 模型参数设置

```python
params = {
    "nl_convergence_tol": 1e-8,         # 非线性迭代收敛容差
    "max_iterations": 50,               # Newton 迭代最大步数
    "file_name": "thm",                 # 导出文件名前缀
    "dilation_angle": np.radians(5),    # 剪胀角（弧度）
    "dilation_angle_constant_g": 0,     # 低维（交叉点）剪胀角（0 表示关闭）
    "use_umfpack": True,                # 使用 UMFPACK 直接求解器
    "mesh_args": mesh_args,             # 网格参数字典
}
```

**剪胀角（dilation angle）** $\psi = 5°$：裂缝剪切滑动时张开的角度，用于计算由切向位移跳跃引起的法向开度增量（见 [7.1 节](#71-create_grid) 开度更新公式）。

### 5.3 路径与列表初始化

```python
root_folder = "ex1/"
gmsh_file_base = root_folder + "mesh_"      # 预先生成的 .msh 文件前缀路径
pickle_file_name = root_folder + "gblist"   # 网格数据序列化存储路径
folder_base = root_folder + "convergence_study_"  # 各精度级别的结果目录前缀
gb_list, models = [], []                    # 初始化网格列表和模型列表
refinement_levels = np.arange(0, 6)        # 细化级别：0,1,2,3,4,5（共 6 套网格）
```

`refinement_levels` 用于收敛性研究：从 `mesh_0.msh` 到 `mesh_5.msh`，网格从粗到细，以验证数值解对网格的收敛性。

### 5.4 可选：从已保存文件加载粗网格

```python
if refinement_levels[0] > 1:
    gb_list = thm_utils.read_pickle(pickle_file_name)
    if gb_list is None:
        gb_list = []
```

若从非零级别开始（跳过前几个粗网格），则先从磁盘加载已有的网格对象，避免重复计算。

---

## 6. 网格加载与模型构建

```python
for i in refinement_levels:
```

主循环：依次处理 6 套由粗到细的网格，每次独立运行完整的 THM 仿真。

### 6.1 设置输出目录

```python
folder_name = folder_base + str(i)      # e.g. "ex1/convergence_study_0"
params["folder_name"] = folder_name
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
```

为每个细化级别创建独立的结果目录。`os.makedirs` 递归创建不存在的多级目录。

### 6.2 从 `.msh` 文件加载网格

```python
msh_file = gmsh_file_base + str(i) + ".msh"   # e.g. "ex1/mesh_0.msh"
grid_list = pp.fracs.simplex.triangle_grid_from_gmsh(file_name=msh_file)
gb = pp.fracs.meshing.grid_list_to_grid_bucket(grid_list)
```

- `triangle_grid_from_gmsh`：读取由 Gmsh 生成的 `.msh` 文件，将其转换为 PorePy 内部的三角形网格列表（包含二维基质网格和一维裂缝网格）。
- `grid_list_to_grid_bucket`：将各维度的网格组合为 `GridBucket`（混维网格容器），并生成连接二维基质与一维裂缝的 **mortar 网格**（用于界面耦合）。

`GridBucket` 是 PorePy 的核心数据结构，以图（Graph）形式管理：
- **节点**（nodes）：各维度的子域网格（$d=2$: 基质，$d=1$: 裂缝，$d=0$: 交叉点）。
- **边**（edges）：相邻子域之间的 mortar 网格，承载界面通量耦合。

### 6.3 构建模型对象

```python
m = Example1Model(params, gb)
```

调用 `Example1Model.__init__`（第 90 行）：

```python
def __init__(self, params, gb):
    super().__init__(params)     # 调用 THM 基类的构造函数（设置 PorePy 框架所需的基本属性）
    self._set_fields(params, gb) # 设置算例特定的物理参数和标量
```

`_set_fields` 内容（第 1199–1234 行）：

```python
self.T_0_Kelvin = 300                    # 参考温度 T₀ = 300 K
self.background_temp_C = 27              # 背景温度（摄氏度）
self.gb = gb                             # 存储网格对象
self.scalar_scale = 1e9                  # 压力量纲缩放因子：1 GPa（数值条件改善）
self.file_name = params["file_name"]     # "thm"
self.folder_name = params["folder_name"]
self.mechanics_parameter_key_from_t = "mechanics_from_t"  # 温度对力学的耦合键名
self.export_fields = [                   # VTU 导出字段列表
    "u_exp", "p_exp", "T_exp",
    "traction_exp", "aperture_exp",
    "u_global", "cell_centers", "du",
]
self.initial_aperture = 5e-4 / self.length_scale  # 初始裂缝开度 a₀ = 5×10⁻⁴ m
self._dilation_angle = np.radians(5)     # 剪胀角 ψ
self._dilation_angle_constant_g = 0     # 低维网格剪胀角（0 = 关闭）
```

`self.length_scale` 由 `THM` 基类设定（通常为 1），用于无量纲化空间坐标。

---

## 7. `prepare_simulation()` — 仿真准备

```python
pp.run_time_dependent_model(m, params)
```

`pp.run_time_dependent_model` 首先调用 `m.prepare_simulation()`（第 738 行）：

```python
def prepare_simulation(self):
    self.create_grid()              # 7.1
    self._set_time_parameters()     # 7.2
    self._set_rock_and_fluid()      # 7.3
    self._initial_condition()       # 7.4
    self._set_parameters()          # 7.5
    self._assign_variables()        # 7.6
    self._assign_discretizations()  # 7.6
    self._discretize()              # 7.7
    self._initialize_linear_solver()# 7.7
    self._export_step()             # 7.8
```

### 7.1 `create_grid()`

```python
def create_grid(self):
    gb = self.gb
    self.box = {
        "xmin": 0, "ymin": 0,
        "xmax": 2 / self.length_scale,
        "ymax": 1 / self.length_scale,
    }
    pp.contact_conditions.set_projections(gb)
    self._Nd = gb.dim_max()              # 最高维数（2D 问题中 = 2）
    self.n_frac = len(gb.grids_of_dimension(self._Nd - 1))  # 裂缝数量（= 8）
```

- `pp.contact_conditions.set_projections(gb)`：为每条裂缝在其 `GridBucket` 节点数据中存储**切向–法向投影矩阵**（`tangential_normal_projection`），后续用于将全局坐标系中的位移/力分解为切向和法向分量。

### 7.2 `_set_time_parameters()`

```python
def _set_time_parameters(self):
    self.time = self.params.get("time", -1e6)  # 起始"时间"= -1×10⁶ s（平衡阶段）
    self.time_step = -self.time                 # 第一步步长 = 1×10⁶ s（完成平衡）
    self.end_time = 10 * pp.HOUR               # 仿真结束时间 = 36000 s
    self.max_time_step = self.end_time / 5      # 最大时间步 = 7200 s
    self.phase_limits = np.array([self.time, 0, 3e1, self.end_time])
    # 各阶段边界时刻（s）:  -1e6,    0,    30,   36000
    self.phase_time_steps = np.array([self.time_step, 2e0, 2/3*pp.HOUR, 1.0])
    # 各阶段初始步长（s）:   1e6,    2,   2400,    1
```

**各仿真阶段说明**：

| 阶段 | 时间范围 | 初始步长 | 物理事件 |
|------|----------|----------|----------|
| Phase I  | $[-10^6, 0]$ s | $10^6$ s | 力学平衡（单步完成） |
| Phase II | $[0, 30]$ s | 2 s | 施加压差（东西边界） |
| Phase III| $[30, 36000]$ s | $\tfrac{2}{3} \times 3600 = 2400$ s | 左边界降温 15 K |

### 7.3 `_set_rock_and_fluid()`

```python
def _set_rock_and_fluid(self):
    self.rock = Granite()   # 固相：花岗岩（见第 10 节）
    self.fluid = Water()    # 液相：水（见第 10 节）
```

### 7.4 `_initial_condition()`

```python
def _initial_condition(self) -> None:
    for g, d in self.gb:
        d[pp.PARAMETERS] = pp.Parameters()
        d[pp.PARAMETERS].update_dictionaries([...])  # 初始化参数字典

    self._update_all_apertures(to_iterate=False)  # 设置初始裂缝开度（时间步状态）
    self._update_all_apertures()                  # 设置初始裂缝开度（迭代状态）
    super()._initial_condition()                  # THM 基类：初始化位移、接触力等变量为零
```

随后对每个子域 $g$：

```python
d[pp.STATE]["cell_centers"] = g.cell_centers.copy()
p0 = self._initial_scalar(g)         # 初始压力
T0 = self._initial_temperature(g)    # 初始温度
state = {
    self.scalar_variable: p0,
    self.temperature_variable: T0,
    "previous_u_exp": np.zeros((3, g.num_cells)),
}
pp.set_state(d, state)    # 存入"时间步"状态
pp.set_iterate(d, iterate) # 存入"迭代"状态
```

**初始压力**（`_initial_scalar`）：

$$p_0 = \rho_f \, g_{\text{grav}} \, z + p_{\text{atm}}$$

其中 $\rho_f = 10^3 \text{ kg/m}^3$，$g_{\text{grav}} = 9.81 \text{ m/s}^2$，$z$ 为深度（本算例不开启重力，$z=0$，所以 $p_0 = p_{\text{atm}}$）。

**初始温度**（`_initial_temperature`）：

$$T_0 = 300 \text{ K（均匀分布）}$$

**初始裂缝开度**（`_update_all_apertures`）：

对 $d=1$ 的裂缝子域：

$$a = a_0 + \|\boldsymbol{u}_n\| + \tan\psi \cdot \|\boldsymbol{u}_\tau\|$$

其中：
- $a_0 = 5 \times 10^{-4}$ m：初始（参考）开度
- $\boldsymbol{u}_n$：法向位移跳跃（开度变化量）
- $\boldsymbol{u}_\tau$：切向位移跳跃（剪切滑动量）
- $\psi = 5°$：剪胀角

初始时位移跳跃为零，故 $a = a_0$。

对 $d=0$ 的交叉点（intersections），开度取相交裂缝的平均值：

$$a_{\text{int}} = \frac{1}{N_{\text{par}}} \sum_{k} a_k^{\text{fracture}}$$

交叉点比容（specific volume）：

$$V_s = a^{N_d - d}$$

例如 $d=0$ 时 $V_s = a^2$（面积量纲）。

### 7.5 `_set_parameters()`

此方法由 `THM` 基类负责调用，依次执行：

- `_set_mechanics_parameters()`
- `_set_scalar_parameters()`
- `_set_temperature_parameters()`

#### 7.5.1 `_set_mechanics_parameters()`

**四阶弹性张量**（基质，$d = N_d = 2$）：

$$\mathbf{C} = 2\mu\,\mathbf{I}_4^{\text{sym}} + \lambda\,(\mathbf{I}\otimes\mathbf{I})$$

代码实现：

```python
lam = rock.LAMBDA * np.ones(g.num_cells) / self.scalar_scale  # 拉梅第一参数 λ / 量纲缩放
mu  = rock.MU    * np.ones(g.num_cells) / self.scalar_scale  # 剪切模量 μ / 量纲缩放
C   = pp.FourthOrderTensor(mu, lam)
```

**力学边界条件**（`_bc_type_mechanics` + `_bc_values_mechanics`）：

- 南侧（$y=0$）：Dirichlet，$\boldsymbol{u} = \boldsymbol{0}$（固定底部）
- 北侧（$y=y_{\max}$）：Dirichlet，$\boldsymbol{u} = (u_x, u_y)$

```python
x = 5e-4 / self.length_scale   # u_x = 5×10⁻⁴ m（水平位移）
y = -2e-4 / self.length_scale  # u_y = -2×10⁻⁴ m（竖向压缩）
```

- 东侧、西侧：Neumann（自由面力 = 0）
- 裂缝面（fracture faces）：Dirichlet（用于接触力学边界）

**Biot 系数**（压力–位移耦合，`_biot_alpha`）：

$$\alpha_B = \begin{cases} 0.8 & d = N_d \text{（基质）} \\ 1.0 & d < N_d \text{（裂缝/交叉点）} \end{cases}$$

**热–力耦合系数**（`_biot_beta`）：

对基质（$d = N_d$）：

$$\beta_s = K_{\text{bulk}} \cdot 3\beta_s^{\text{lin}}$$

其中 $K_{\text{bulk}}$ 为固体体积模量，$\beta_s^{\text{lin}}$ 为花岗岩线热膨胀系数。系数 3 将线膨胀转换为体积膨胀。

对裂缝（$d < N_d$）：

$$\beta_f = \frac{T_k}{T_0} \rho_f c_{p,f}$$

其中 $T_k$ 为当前迭代温度（K），$\rho_f$ 为流体密度，$c_{p,f}$ 为流体比热容。

**接触力学参数**（$d = N_d - 1$）：

```python
"friction_coefficient": 0.5,              # 摩擦系数 μ_f = 0.5（库仑摩擦）
"contact_mechanics_numerical_parameter": 1e1,  # 数值惩罚参数
"dilation_angle": np.radians(5),          # 剪胀角 ψ
```

库仑摩擦准则：$|\boldsymbol{t}_\tau| \leq \mu_f |\boldsymbol{t}_n|$（$\boldsymbol{t}_n \leq 0$ 为压力）。

#### 7.5.2 `_set_scalar_parameters()`（渗流参数）

**质量储存系数**（`mass_weight`）：

对基质（$d = N_d$）：

$$S = \phi C_f + \frac{\alpha_B - \phi}{K_{\text{bulk}}}$$

其中 $\phi$ 为孔隙度，$C_f$ 为流体压缩系数，$K_{\text{bulk}}$ 为固体体积模量。这是 Biot 储存系数。

最终存储参数（量纲缩放后）：

$$S_{\text{scaled}} = S \cdot p_{\text{scale}} \cdot V_s$$

其中 $V_s$ 为比容，$p_{\text{scale}} = 10^9$ Pa。

**温度对压力的耦合系数**（`_temperature_to_scalar_coupling_coefficient`）：

对基质：

$$\xi_{T \to p} = -\phi \beta_f - (1-\phi) \cdot 3\beta_s^{\text{lin}}$$

对裂缝：

$$\xi_{T \to p} = -\beta_f$$

其中 $\beta_f = 4\times10^{-4}$ K$^{-1}$ 为流体体热膨胀系数。

```python
t2s_coupling = xi_{T→p} * V_s * T_scale
```

**渗透率**（`_set_permeability_from_aperture`）：

对裂缝，使用**立方定律**（cubic law）：

$$k_f = \frac{a^2}{12 \mu_f / p_{\text{scale}}}$$

其中 $a$ 为裂缝开度，$\mu_f = 0.001$ Pa·s 为动力粘度。考虑截面积后的等效渗透率：

$$K_f = k_f \cdot V_s$$

无量纲化（除以 $l^2$，$l = $ `length_scale`）：

$$K_{f,\text{scaled}} = K_f / l^2$$

对基质，使用岩石固有渗透率：

$$k_s = \frac{K_{\text{perm}}}{\mu_f / p_{\text{scale}}} \cdot \frac{1}{l^2}$$

其中 $K_{\text{perm}} = 10^{-15}$ m$^2$（花岗岩渗透率）。

界面（mortar）法向渗透率 $k_n$（连接裂缝与基质）：

$$k_n = \frac{k_f}{a \cdot V_s / 2} \cdot V_{j}$$

其中 $V_j$ 为 mortar 单元在基质侧的体积权重。

**压力边界条件**（`_bc_type_scalar` + `_bc_values_scalar`）：

- 东侧（$x=x_{\max}$）：Dirichlet，$p = 0$（参考压力，相对大气压的增量为 0）
- 西侧（$x=0$）：Dirichlet，$p = 4\times10^7 / p_{\text{scale}}$（仅在 Phase II/III 开启）

$$p_{\text{west}} = \begin{cases} 0 & t \leq t_{\text{PhaseII}} \\ 4\times10^7 \text{ Pa} & t > t_{\text{PhaseII}} \end{cases}$$

- 南北侧：Neumann（无流动）

#### 7.5.3 `_set_temperature_parameters()`（热参数）

**有效热导率**（$\kappa_{\text{eff}}$）：

$$\kappa_{\text{eff}} = \phi \kappa_f + (1-\phi) \kappa_s$$

其中 $\kappa_f = 0.6$ W/(m·K)（水），$\kappa_s = 3.0$ W/(m·K)（花岗岩）。

量纲缩放（整个方程除以 $T_0$）：

$$\kappa_{\text{div}} = \frac{T_{\text{scale}}}{l^2 \cdot T_0} \kappa_{\text{eff}} V_s$$

**有效热容量**（`mass_weight`，等效热储存系数）：

$$(\rho c_p)_{\text{eff}} = \phi \rho_f c_{p,f}(1 - T_k \beta_f) + (1-\phi) \rho_s c_{p,s}(1 - T_k \cdot 3\beta_s^{\text{lin}})$$

（热膨胀修正项使热容与温度微弱相关）

量纲化：

$$m_T = (\rho c_p)_{\text{eff}} \cdot V_s \cdot T_{\text{scale}} / T_0$$

**对流权重**（advection weight，热对流项）：

$$w_{\text{adv}} = \rho_f c_{p,f} \cdot T_{\text{scale}} / T_0$$

**压力对温度的耦合系数**（`_scalar_to_temperature_coupling_coefficient`）：

对基质：

$$\xi_{p \to T} = \frac{\phi c_{p,f} \rho_f C_f + (1-\phi) c_{p,s} \rho_s / K_{\text{bulk}}}{T_0}$$

对裂缝：

$$\xi_{p \to T} = \frac{c_{p,f} \rho_f C_f}{T_0}$$

**温度边界条件**（`_bc_type_temperature` + `_bc_values_temperature`）：

- 东西两侧：Dirichlet
- 西侧温度（仅 Phase III 降温）：

$$T_{\text{west}} = \begin{cases} T_0 / T_{\text{scale}} & t \leq t_{\text{PhaseIII}} \\ (T_0 - 15) / T_{\text{scale}} & t > t_{\text{PhaseIII}} \end{cases}$$

- 东侧：$T_{\text{east}} = T_0 / T_{\text{scale}}$（保持参考温度）
- 南北侧：Neumann（绝热边界）

**界面法向热导率**：

$$\kappa_n = \frac{2}{a_{\text{mortar}}} \kappa_f \cdot V_j$$

其中 $a_{\text{mortar}}$ 为 mortar 网格上插值的裂缝开度。

### 7.6 `_assign_variables()` 与 `_assign_discretizations()`

- `_assign_variables()`（由 `THM` 基类实现）：为每个子域节点和 mortar 边注册自由度变量：
  - 基质（$d=2$）：位移 $\boldsymbol{u}$、压力 $p$、温度 $T$
  - 裂缝（$d=1$）：$p$、$T$
  - 交叉点（$d=0$）：$p$、$T$
  - mortar 边（基质↔裂缝）：mortar 位移 $\hat{\boldsymbol{u}}$、mortar 压力通量、mortar 热通量
  - 接触 mortar（裂缝内部）：接触力 $\boldsymbol{t}$

- `_assign_discretizations()`（第 722–736 行）：为每个方程分配离散化算子：
  - 动量方程：MPSA（多点应力近似）
  - 压力方程：MPFA（多点通量近似）
  - 温度方程：MPFA + 迎风对流
  - 界面流量：Robin 型（mortar）条件

  本算例在 `_assign_discretizations` 中覆盖了界面热扩散的缩放方式：

  ```python
  d[pp.COUPLING_DISCRETIZATION][self.temperature_coupling_term][e][1].kinv_scaling = False
  d[pp.COUPLING_DISCRETIZATION][self.scalar_coupling_term][e][1].kinv_scaling = True
  ```

  这意味着压力方程使用 $k_n^{-1}$ 缩放（改善较长时间步下的条件数），而温度方程不使用。

### 7.7 `_discretize()` 与 `_initialize_linear_solver()`

- `_discretize()`：调用各离散化算子的 `discretize()` 方法，预计算刚度矩阵中所有**与解无关**的静态部分（如 MPSA 重构矩阵）。
- `_initialize_linear_solver()`：初始化稀疏矩阵组装器（`Assembler`）。

### 7.8 `_export_step()`（初始导出）

```python
def _export_step(self):
    if "exporter" not in self.__dict__:
        self._set_exporter()   # 首次调用时创建 pp.Exporter
```

`_set_exporter` 初始化 PorePy 的 VTU 导出器：

```python
self.exporter = pp.Exporter(self.gb, self.file_name,
                            folder_name=self.viz_folder_name + "_vtu")
self.export_times = []   # 记录已导出的时刻
```

随后遍历所有子域，为每个子域准备导出字段：

**基质（$d=2$）**：
```python
u = d[pp.STATE][self.displacement_variable].reshape((self._Nd, -1), order="F")
u_exp = np.vstack((u * self.length_scale, pad_zeros))  # 恢复物理单位
d[pp.STATE]["u_exp"] = u_exp
d[pp.STATE]["p_exp"] = d[pp.STATE][self.scalar_variable] * self.scalar_scale
d[pp.STATE]["T_exp"] = d[pp.STATE][self.temperature_variable] * self.temperature_scale
```

**裂缝（$d=1$）**：

重建裂缝面的局部位移跳跃向量（切向 + 法向）：

$$\boldsymbol{u}_{\text{mortar,local}} = \mathbf{P}_{\text{tn}} \cdot \hat{\boldsymbol{u}}_{\text{mortar}}$$

其中 $\mathbf{P}_{\text{tn}}$ 为切向–法向投影矩阵，$\hat{\boldsymbol{u}}_{\text{mortar}}$ 为 mortar 位移变量。

全局坐标系中的位移跳跃：

$$\boldsymbol{u}_{\text{global}} = \mathbf{M}_{\text{mortar} \to \text{sec}} \cdot \mathbf{M}_{\text{sign}} \cdot \hat{\boldsymbol{u}}_{\text{mortar}}$$

孔径：

```python
d[pp.STATE]["aperture_exp"] = self._aperture(g) * self.length_scale
```

接触力（物理单位）：

```python
d[pp.STATE]["traction_exp"] = traction * self.scalar_scale
```

最后写入 VTU 文件并记录时刻：

```python
self.exporter.write_vtu(self.export_fields, time_step=self.time)
self.export_times.append(self.time)
```

---

## 8. 时间步进循环

`pp.run_time_dependent_model` 调用 `prepare_simulation` 后，进入时间步循环。每个时间步内执行 Newton 迭代，直到收敛（或超过最大迭代次数）。

### 8.1 `before_newton_loop()`

```python
def before_newton_loop(self):
    super().before_newton_loop()   # THM 基类：将迭代变量重置为时间步初始值
    self._iteration = 0            # 重置迭代计数器
```

### 8.2 `before_newton_iteration()`

每次 Newton 迭代开始前执行（第 757–782 行）：

```python
def before_newton_iteration(self):
    self._iteration += 1            # 迭代计数 +1
    self.compute_fluxes()           # 计算当前迭代解的 Darcy 通量（用于对流项）
    self._update_all_apertures(to_iterate=True)  # 更新裂缝开度（基于当前迭代位移）
    self._set_parameters()          # 用更新后的开度/密度重新设置所有参数
```

随后进行**选择性重离散化**（避免重新计算已有的静态项）：

```python
term_list = ["!mpsa", "!stabilization", "!div_u", "!grad_p", "!diffusion"]
filt = pp.assembler_filters.ListFilter(term_list=term_list)
self.assembler.discretize(filt=filt)
```

`!` 前缀表示排除这些项（仅重离散化其余项，即依赖于解的质量矩阵、对流项等）。

对低维（$d < N_d - 1$）子域，还需额外重离散化扩散项：

```python
for dim in range(self._Nd - 1):
    for g in self.gb.grids_of_dimension(dim):
        filt = pp.assembler_filters.ListFilter(term_list=["diffusion"], grid_list=[g])
        self.assembler.discretize(filt=filt)
```

计时并记录日志：

```python
logger.info("Rediscretized in {} s.".format(time.time() - t_0))
```

### 8.3 `assemble_and_solve_linear_system()`

```python
def assemble_and_solve_linear_system(self, tol):
    A, b = self.assembler.assemble_matrix_rhs()   # 组装全局刚度矩阵 A 和右端向量 b
    use_umfpack = self.params.get("use_umfpack", True)

    if use_umfpack:
        A.indices = A.indices.astype(np.int64)
        A.indptr  = A.indptr.astype(np.int64)
```

将 `CSR` 格式的稀疏矩阵索引转换为 `int64`，满足 UMFPACK 接口要求。

```python
    x = spla.spsolve(A, b)   # 求解线性方程组 Ax = b
    return x
```

求解**全耦合线性系统**：

$$\mathbf{A} \Delta\mathbf{x} = \mathbf{b}$$

其中 $\mathbf{A}$ 包含所有耦合项（力学、流体、热）的贡献，$\Delta\mathbf{x}$ 为当前 Newton 迭代的增量解向量（包含 $\Delta\boldsymbol{u}$、$\Delta p$、$\Delta T$、$\Delta\boldsymbol{t}$）。

日志输出矩阵条件相关信息：

```python
logger.debug("Max element in A {0:.2e}".format(np.max(np.abs(A))))
logger.info("Solved in {} s.".format(time.time() - t_0))
```

### 8.4 `check_convergence()`

（第 819–948 行）评估 Newton 迭代的收敛性。

#### 8.4.1 非线性问题（线性化处理）

```python
if not self._is_nonlinear_problem():
    diverged = np.any(np.isnan(solution))
    converged = not diverged
    error = np.nan if diverged else 0
    return error, converged, diverged
```

#### 8.4.2 提取各变量的自由度

```python
mech_dof    = self.dof_manager.dof_ind(g_max, self.displacement_variable)
p_dof       = self.dof_manager.dof_ind(g_max, self.scalar_variable)
T_dof       = self.dof_manager.dof_ind(g_max, self.temperature_variable)
contact_dof = ...   # 所有接触力变量的 DOF 索引
```

#### 8.4.3 计算误差范数

**力学误差**（当前迭代与前一迭代之差的 $l^2$ 范数平方）：

$$e_u = \|\Delta\boldsymbol{u}^{(k)} - \Delta\boldsymbol{u}^{(k-1)}\|_2^2$$

相对误差：

$$e_{u,\text{rel}} = \frac{e_u}{\|\Delta\boldsymbol{u}^{(k)}\|_2^2}$$

**收敛判据**（以力学变量为例）：

$$e_u < \varepsilon \quad \text{（绝对）} \qquad \text{或} \qquad e_u < \varepsilon \cdot \|\boldsymbol{u}^{(k)}\|_2^2 \quad \text{（相对）}$$

类似地对温度 $e_T$、压力 $e_p$、接触力 $e_c$ 均有相同判据，$\varepsilon = 10^{-8}$（`nl_convergence_tol`）。

接触力的容差放宽 10 倍：$\varepsilon_c = 10\varepsilon$。

**全局收敛**：

$$\text{converged} = \text{converged\_mech} \wedge \text{converged\_T} \wedge \text{converged\_p} \wedge \text{converged\_contact}$$

日志输出各误差：

```python
logger.info("Errors: contact force {:.2e}, matrix displacement {:.2e}, "
            "temperature {:.2e} and pressure {:.2e}".format(...))
```

返回值：`(error_mech, converged, diverged)`。

### 8.5 `after_newton_convergence()`

Newton 迭代收敛后调用（第 786–794 行）：

```python
def after_newton_convergence(self, solution, errors, iteration_counter):
    for g, d in self.gb:
        d[pp.STATE]["previous_u_exp"] = d[pp.STATE]["u_exp"].copy()  # 保存上一时间步的导出位移

    super().after_newton_convergence(...)  # 更新时间步状态：u, p, T, 接触力
    self._update_all_apertures(to_iterate=False)  # 更新时间步开度
    self._update_all_apertures(to_iterate=True)   # 同步迭代开度
    self._export_step()       # 导出当前时间步结果
    self._adjust_time_step()  # 动态调整下一步步长
    self._save_data(errors, iteration_counter)  # 记录统计数据
```

**时间步自适应调整**（`_adjust_time_step`，第 1178–1197 行）：

```python
self.time_step = getattr(self, "time_step_factor", 1.0) * self.time_step
```

默认保持当前步长不变（`time_step_factor = 1.0`）。同时，确保精确到达每个阶段的端点时刻：

```python
for dt, lim in zip(self.phase_time_steps, self.phase_limits):
    diff = self.time - lim
    if diff < 0 and -diff <= self.time_step:
        self.time_step = -diff   # 将步长截断到阶段边界
    if np.isclose(self.time, lim):
        self.time_step = dt      # 进入下一阶段，使用阶段初始步长
if self.time > 0:
    self.time_step = min(self.time_step, self.max_time_step)  # 上限控制
```

---

## 9. 时间步结束后的数据保存与导出

### 9.1 `_save_data()`（第 1030–1110 行）

记录各裂缝的位移跳跃和接触力，用于后处理与绘图。

**切向位移跳跃的体积加权范数**：

$$\|\boldsymbol{u}_\tau\|_{\text{avg}} = \frac{\sqrt{\sum_e |\boldsymbol{u}_{\tau,e}|^2 \cdot V_e}}{\sum_e V_e}$$

**法向位移跳跃的体积加权范数**：

$$\|u_n\|_{\text{avg}} = \frac{\sqrt{\sum_e u_{n,e}^2 \cdot V_e}}{\sum_e V_e}$$

**接触力分解**：

接触力向量按每个单元排列为 $[t_{\tau,1},\ldots,t_{\tau,N_d-1},t_n]$（切向分量在前，法向分量在后）：

```python
normal_indices    = np.arange(self._Nd - 1, contact_force.size, self._Nd)
tangential_indices = np.setdiff1d(np.arange(contact_force.size), normal_indices)
```

体积加权切向力：

$$\|\boldsymbol{t}_\tau\|_{\text{avg}} = \frac{\sqrt{\sum_e |\boldsymbol{t}_{\tau,e}|^2 \cdot V_e}}{\sum_e V_e}$$

所有裂缝的数据被追加到 `self.u_jumps_tangential`、`self.u_jumps_normal`、`self.force_tangential`、`self.force_normal` 数组中（行=时间步，列=裂缝编号）。

**阶段端点处特殊保存**：

```python
for i in np.arange(1, 4):
    if np.isclose(self.phase_limits[i], self.time):
        for var in ["p", "T", "u_exp", "contact_traction"]:
            d[pp.STATE]["{}_phase_{}".format(var, i)] = val
```

当模拟时刻恰好到达某阶段端点时，将该阶段的结果单独保存，以便后续比较。

### 9.2 仿真结束后的导出

```python
m._export_pvd()
```

```python
def _export_pvd(self):
    self.exporter.write_pvd(np.array(self.export_times))
```

将所有时间步的 VTU 文件路径写入一个 PVD（ParaView Data）索引文件，可直接用 ParaView 打开动画。

```python
thm_utils.write_fracture_data_txt(m)
```

将时间步、Newton 迭代次数、裂缝位移跳跃和接触力以文本格式（空格分隔）保存到 `folder_name/data_thm.txt`，各列依次为：

| 列 | 内容 |
|----|------|
| 1 | 时间（s） |
| 2 | Newton 迭代次数 |
| 3–10 | 各裂缝切向位移跳跃（m） |
| 11–18 | 各裂缝法向位移跳跃（m） |
| 19–26 | 各裂缝切向接触力（Pa） |
| 27–34 | 各裂缝法向接触力（Pa） |

```python
thm_utils.write_pickle(gb_list, pickle_file_name)
```

将已完成仿真的网格对象列表序列化存储，供后续收敛性分析复用。

### 9.3 收敛性分析（外层循环结束后）

```python
if len(gb_list) > 1:
    for gb in gb_list[:-1]:
        pp.grids.match_grids.gb_refinement(gb, gb_list[-1])
thm_utils.write_pickle(gb_list, pickle_file_name)
```

- `pp.grids.match_grids.gb_refinement(gb_coarse, gb_fine)`：在各粗网格与最细网格之间建立单元映射关系，用于后续计算离散误差（如 $L^2$ 误差）。
- 最终再次序列化 `gb_list`（此时含有网格映射信息）以供误差分析脚本使用。

---

## 10. 辅助类 `Water` 与 `Granite`

### 10.1 `Water`（第 1249–1283 行）

| 属性/方法 | 值/表达式 | 单位 |
|-----------|-----------|------|
| `VISCOSITY` | $10^{-3}$ | Pa·s |
| `COMPRESSIBILITY` | $10^{-10}$ | Pa$^{-1}$ |
| `BULK_MODULUS` | $10^{10}$ | Pa |
| `thermal_expansion(Δθ)` | $4\times10^{-4}$ | m³/(m³·K) |
| `thermal_conductivity()` | $0.6$ | W/(m·K) |
| `specific_heat_capacity()` | $4200$ | J/(kg·K) |
| `dynamic_viscosity()` | $0.001$ | Pa·s |
| `hydrostatic_pressure(depth)` | $\rho_f g z + p_{\text{atm}}$ | Pa |

**流体密度**（在 `_fluid_density` 中计算，第 212–236 行）：

$$\rho_f = \rho_0 \exp\!\left(C_f \Delta p - \beta_f \Delta T\right)$$

其中：
- $\rho_0 = 10^3$ kg/m³
- $\Delta p = p_k - p_{\text{atm}}$（超压）
- $\Delta T = T_k - T_0$（温差），限幅到 $[-T_0/3,\, T_0/3]$

```python
rho = rho_0 * np.exp(dp * self.fluid.COMPRESSIBILITY
                     - dT * self.fluid.thermal_expansion(dT))
```

### 10.2 `Granite`（第 1286–1302 行，继承自 `pp.Granite`）

| 属性/方法 | 值 | 单位 |
|-----------|-----|------|
| `LAMBDA` | （继承）$\approx 16.67\times10^9$ | Pa |
| `MU` | （继承）$\approx 16.67\times10^9$ | Pa |
| `BULK_MODULUS` | $K = \lambda + \tfrac{2}{3}\mu$ | Pa |
| `DENSITY` | （继承）$2700$ | kg/m³ |
| `THERMAL_EXPANSION` | （继承）$\approx 8\times10^{-6}$ | K$^{-1}$（线性） |
| `PERMEABILITY` | $10^{-15}$ | m² |
| `thermal_conductivity()` | $3.0$ | W/(m·K) |
| `specific_heat_capacity()` | $790$ | J/(kg·K) |

体积模量由 Lamé 参数计算：

$$K_{\text{bulk}} = \lambda + \frac{2}{3}\mu$$

---

## 11. 重要物理量与参数汇总

| 参数 | 符号 | 值 | 单位 |
|------|------|----|------|
| 计算域 | $\Omega$ | $(0,2)\times(0,1)$ | m² |
| 裂缝数量 | $N_f$ | 8 | — |
| 初始裂缝开度 | $a_0$ | $5\times10^{-4}$ | m |
| 剪胀角 | $\psi$ | 5° | — |
| 摩擦系数 | $\mu_f$ | 0.5 | — |
| Biot 系数（基质） | $\alpha_B$ | 0.8 | — |
| 孔隙度（基质） | $\phi$ | 0.01 | — |
| 孔隙度（裂缝） | $\phi_f$ | 1.0 | — |
| 参考温度 | $T_0$ | 300 | K |
| 北侧水平位移 | $u_x$ | $5\times10^{-4}$ | m |
| 北侧竖向位移 | $u_y$ | $-2\times10^{-4}$ | m |
| 西侧超压 | $\Delta p$ | $4\times10^7$ | Pa |
| 西侧温降 | $\Delta T$ | $-15$ | K |
| 压力量纲缩放 | $p_{\text{scale}}$ | $10^9$ | Pa |
| Newton 收敛容差 | $\varepsilon$ | $10^{-8}$ | — |
| 最大 Newton 迭代次数 | — | 50 | — |
| 仿真结束时间 | $t_{\text{end}}$ | $10\times3600 = 36000$ | s |

---

## 控制方程概述

本算例求解的全耦合 THM 方程组（各量均已量纲化）：

**动量平衡（力学）**：

$$\nabla\cdot\boldsymbol{\sigma} = \boldsymbol{0}, \qquad \boldsymbol{\sigma} = \mathbf{C}:\boldsymbol{\varepsilon}(\boldsymbol{u}) - \alpha_B p\,\mathbf{I} - \beta_s (T - T_0)\mathbf{I}$$

**质量守恒（渗流）**：

$$S \frac{\partial p}{\partial t} + \alpha_B \frac{\partial (\nabla\cdot\boldsymbol{u})}{\partial t} + \xi_{T\to p} \frac{\partial T}{\partial t} = \nabla\cdot\left(\frac{k}{\mu_f}\nabla p\right) + q_p$$

**能量守恒（热传导/对流）**：

$$(\rho c_p)_{\text{eff}} \frac{\partial T}{\partial t} + \xi_{p\to T} \frac{\partial p}{\partial t} + \beta \frac{\partial(\nabla\cdot\boldsymbol{u})}{\partial t} = \nabla\cdot(\kappa_{\text{eff}}\nabla T) - \nabla\cdot(\rho_f c_{p,f} T\boldsymbol{v}) + q_T$$

其中 $\boldsymbol{v} = -\tfrac{k}{\mu_f}\nabla p$ 为 Darcy 流速。

**接触力学（裂缝面）**：库仑摩擦准则

$$\begin{cases} t_n \leq 0 & \text{（压缩接触）} \\ |\boldsymbol{t}_\tau| \leq \mu_f |t_n| & \text{（Coulomb 摩擦）} \\ \text{若滑动}: |\boldsymbol{t}_\tau| = \mu_f |t_n| \end{cases}$$
