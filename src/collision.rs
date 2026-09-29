//! 二维碰撞与分离:玩家圆 vs 建筑 AABB / 车辆 AABB / 道具圆柱 + 世界边界。
//!
//! 不引入任何物理引擎(rust-standards §13.1 优先不新增第三方依赖),碰撞体
//! 全部**从资产 JSON 的 `bounds` 自动推导** —— 摆放蓝图(BUILDINGS /
//! PROPS / PALM_POSITIONS / VEHICLES)只给位置 + yaw + scale,包围盒由
//! [`placement_box`] 按 yaw 旋转资产本地 XZ 足迹得到,因此资产换了模型
//! 碰撞体自动跟着变,不存在第二份手写的魔法坐标表。
//!
//! 分离算法是最朴素的「圆心 → 形状最近点」推出法:对每个形状求圆心到
//! 形状的最短分离向量,把圆心沿该向量推出 `penetration` 的距离。圆形站在
//! AABB 角上时单次迭代可能残留一点重叠,所以固定迭代若干轮直到稳定。

use crate::r#type::{Vec2, Vec3};

/// 分离迭代轮数(圆心站在 AABB 角上时单轮残留一点,多轮收敛)。
const RESOLVE_ITERATIONS: usize = 4;

/// 判定「圆心在形状内部」的向量长度阈值(米)。
const INSIDE_EPSILON: f32 = 1e-5;

/// 车辆等效碰撞圆半径(米):比玩家大,车身也更宽。
pub const CAR_RADIUS: f32 = 1.25;

/// 相机当作球体扫描时的半径(米)。
///
/// 只用一条零半径射线会在墙角 / 屋檐这种掠射角上抖动 —— 命中距离随
/// yaw 的微小变化跳变。给探针一个实际尺寸(略大于一个肩宽)做球体推进,
/// 抖动幅度被压到可以忽略。它只被 [`ray_to_shapes`] 读到,所以定义在
/// 射线实现旁边而不是 `camera`。
pub const CAMERA_PROBE_RADIUS: f32 = 0.32;

/// 一个静态碰撞体(投影到世界 XZ 平面)。
#[derive(Clone, Copy, Debug)]
pub enum Shape {
    /// 轴对齐包围盒 —— 建筑 / 车辆 / 长椅这类方块状道具。
    Aabb {
        /// 盒中心的世界 XZ 坐标。
        center: Vec2,
        /// 盒的半尺寸(世界 XZ,米)。
        half: Vec2,
    },
    /// 竖直圆柱 —— 棕榈 / 垃圾桶 / 消防栓这类近似圆形的道具。
    Circle {
        /// 圆柱中心的世界 XZ 坐标。
        center: Vec2,
        /// 圆柱半径(米)。
        radius: f32,
    },
}

/// 静态碰撞世界:一组形状 + 世界边界 + 玩家圆半径。
#[derive(Clone, Debug)]
pub struct CollisionWorld {
    /// 全部静态碰撞体。
    shapes: Vec<Shape>,
    /// 玩家圆半径(米)。
    player_radius: f32,
    /// 世界半边长(X / Z 各一半,米),玩家不许走出这个范围。
    half_extent: Vec2,
}

impl CollisionWorld {
    /// 新建一个空的碰撞世界。
    ///
    /// # Returns
    ///
    /// - `Self` - 不含任何形状的碰撞世界。
    pub fn new() -> Self {
        Self {
            shapes: Vec::new(),
            player_radius: 0.35,
            half_extent: [48.0, 48.0],
        }
    }

    /// 静态碰撞体表的只读视图。
    ///
    /// # Returns
    ///
    /// - `&[Shape]` - 碰撞体表。
    pub fn get_shapes(&self) -> &[Shape] {
        &self.shapes
    }

    /// 静态碰撞体表的可变引用。
    ///
    /// # Returns
    ///
    /// - `&mut Vec<Shape>` - 碰撞体表。
    pub fn get_shapes_mut(&mut self) -> &mut Vec<Shape> {
        &mut self.shapes
    }

    /// 玩家圆半径。
    ///
    /// # Returns
    ///
    /// - `f32` - 半径(米)。
    pub fn get_player_radius(&self) -> f32 {
        self.player_radius
    }

    /// 覆盖玩家圆半径。
    ///
    /// # Arguments
    ///
    /// - `f32` - 新的半径(米)。
    pub fn set_player_radius(&mut self, value: f32) {
        self.player_radius = value;
    }

    /// 世界半边长。
    ///
    /// # Returns
    ///
    /// - `Vec2` - X / Z 各一半的边界(米)。
    pub fn get_half_extent(&self) -> Vec2 {
        self.half_extent
    }

    /// 覆盖世界半边长。
    ///
    /// # Arguments
    ///
    /// - `Vec2` - 新的 X / Z 半边长(米)。
    pub fn set_half_extent(&mut self, value: Vec2) {
        self.half_extent = value;
    }

    /// 追加一个轴对齐包围盒。
    ///
    /// # Arguments
    ///
    /// - `Vec2` - 盒中心的世界 XZ 坐标。
    /// - `Vec2` - 盒的半尺寸(米)。
    pub fn push_aabb(&mut self, center: Vec2, half: Vec2) {
        self.get_shapes_mut().push(Shape::Aabb { center, half });
    }

    /// 追加一个竖直圆柱。
    ///
    /// # Arguments
    ///
    /// - `Vec2` - 圆柱中心的世界 XZ 坐标。
    /// - `f32` - 圆柱半径(米)。
    pub fn push_circle(&mut self, center: Vec2, radius: f32) {
        self.get_shapes_mut().push(Shape::Circle { center, radius });
    }

    /// 把一个圆形碰撞体推出所有重叠的静态形状,并钳进世界边界。
    ///
    /// 这是纯函数式的:输入一个世界 XZ 坐标,输出分离后的坐标。多次调用
    /// 幂等 —— 已经贴住的圆再调一次不会抖。
    ///
    /// # Arguments
    ///
    /// - `Vec2` - 待分离的世界 XZ 坐标。
    ///
    /// # Returns
    ///
    /// - `Vec2` - 分离并钳进边界后的世界 XZ 坐标。
    pub fn resolve(&self, point: Vec2) -> Vec2 {
        let radius: f32 = self.get_player_radius();
        let mut current: Vec2 = point;
        for _ in 0..RESOLVE_ITERATIONS {
            let mut moved: bool = false;
            for shape in self.get_shapes() {
                let hit: Option<(Vec2, f32)> = match shape {
                    Shape::Aabb { center, half } => push_out_aabb(*center, *half, current, radius),
                    Shape::Circle {
                        center,
                        radius: other,
                    } => push_out_circle(*center, *other, current, radius),
                };
                if let Some((direction, depth)) = hit {
                    current[0] += direction[0] * depth;
                    current[1] += direction[1] * depth;
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
        let limits: Vec2 = self.get_half_extent();
        [
            current[0].clamp(-limits[0], limits[0]),
            current[1].clamp(-limits[1], limits[1]),
        ]
    }

    /// 用车辆半径做分离(玩家开车时的碰撞体更大)。
    ///
    /// 与 [`Self::resolve`] 同样的迭代分离,只是把「玩家圆半径」换成
    /// 车辆半径。车辆因此也穿不过建筑 / 道具,只是被推挤得更早。
    ///
    /// # Arguments
    ///
    /// - `Vec2` - 待分离的世界 XZ 坐标。
    /// - `f32` - 车辆的等效碰撞圆半径(米)。
    ///
    /// # Returns
    ///
    /// - `Vec2` - 分离并钳进边界后的世界 XZ 坐标。
    pub fn resolve_with_radius(&self, point: Vec2, radius: f32) -> Vec2 {
        let mut current: Vec2 = point;
        for _ in 0..RESOLVE_ITERATIONS {
            let mut moved: bool = false;
            for shape in self.get_shapes() {
                let hit: Option<(Vec2, f32)> = match shape {
                    Shape::Aabb { center, half } => push_out_aabb(*center, *half, current, radius),
                    Shape::Circle {
                        center,
                        radius: other,
                    } => push_out_circle(*center, *other, current, radius),
                };
                if let Some((direction, depth)) = hit {
                    current[0] += direction[0] * depth;
                    current[1] += direction[1] * depth;
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
        let limits: Vec2 = self.get_half_extent();
        [
            current[0].clamp(-limits[0], limits[0]),
            current[1].clamp(-limits[1], limits[1]),
        ]
    }

    /// 用车辆半径做分离 —— [`Self::resolve_with_radius`] 的固定版本。
    ///
    /// # Arguments
    ///
    /// - `Vec2` - 待分离的世界 XZ 坐标。
    ///
    /// # Returns
    ///
    /// - `Vec2` - 分离并钳进边界后的世界 XZ 坐标。
    pub fn resolve_car(&self, point: Vec2) -> Vec2 {
        self.resolve_with_radius(point, CAR_RADIUS)
    }

    /// 点是否落在某个形状内部(调试 / 自测用,不做分离)。
    ///
    /// # Arguments
    ///
    /// - `Vec2` - 世界 XZ 坐标。
    ///
    /// # Returns
    ///
    /// - `bool` - 圆心(零半径)与任一形状重叠时为 `true`。
    pub fn contains_point(&self, point: Vec2) -> bool {
        self.get_shapes().iter().any(|shape: &Shape| match shape {
            Shape::Aabb { center, half } => {
                (point[0] - center[0]).abs() <= half[0] && (point[1] - center[1]).abs() <= half[1]
            }
            Shape::Circle { center, radius } => {
                let dx: f32 = point[0] - center[0];
                let dy: f32 = point[1] - center[1];
                (dx * dx + dy * dy).sqrt() <= *radius
            }
        })
    }

    /// 一点到所有静态形状表面的最短距离(米)。
    ///
    /// 落在某个形状**内部**时该形状贡献 0,整体取最小值 —— 所以返回值
    /// 为 0 就等价于「这个点在某个碰撞体里」。相机遮挡回避用它判断眼点
    /// 是否已经扎进楼里,验收调试通道用它把眼点的实际位置报出来。
    ///
    /// # Arguments
    ///
    /// - `Vec2` - 世界 XZ 坐标。
    ///
    /// # Returns
    ///
    /// - `f32` - 到最近形状表面的距离(米);空碰撞世界时为 `f32::MAX`。
    pub fn nearest_surface_distance(&self, point: Vec2) -> f32 {
        let mut best: f32 = f32::MAX;
        for shape in self.get_shapes() {
            let candidate: f32 = distance_to_shape(shape, point);
            if candidate < best {
                best = candidate;
            }
        }
        best
    }
}

/// 一个点到单个静态形状表面的最短距离(米);点落在形状内部时为 0。
///
/// # Arguments
///
/// - `&Shape` - 静态碰撞体。
/// - `Vec2` - 世界 XZ 坐标。
///
/// # Returns
///
/// - `f32` - 最短距离(米)。
fn distance_to_shape(shape: &Shape, point: Vec2) -> f32 {
    match shape {
        Shape::Aabb { center, half } => {
            let dx: f32 = ((point[0] - center[0]).abs() - half[0]).max(0.0);
            let dz: f32 = ((point[1] - center[1]).abs() - half[1]).max(0.0);
            (dx * dx + dz * dz).sqrt()
        }
        Shape::Circle { center, radius } => {
            let dx: f32 = point[0] - center[0];
            let dz: f32 = point[1] - center[1];
            ((dx * dx + dz * dz).sqrt() - *radius).max(0.0)
        }
    }
}

/// 从 `origin` 沿单位方向 `dir`(XZ 分量)投射,求撞上第一个碰撞体的距离。
///
/// 用**球体推进**而不是零半径射线:把探针半径当成「相机本体的尺寸」,
/// 于是命中距离等于「中心还能走多远才让球面贴上墙」。掠射角(视线几乎
/// 平行墙面扫过)下零半径射线会给出剧烈跳变的命中距离,球体则平滑得多。
///
/// 建筑 / 长椅这类方块用**膨胀 AABB** 求交(把半径加到半尺寸上),
/// 圆形道具用**圆-圆求交**。两者都是闭式解,没有迭代。
///
/// # Arguments
///
/// - `&CollisionWorld` - 静态碰撞世界的只读引用。
/// - `Vec3` - 射线起点(世界坐标;只用 XZ 分量)。
/// - `Vec3` - 单位方向(世界坐标;只用 XZ 分量)。
///
/// # Returns
///
/// - `Option<(f32, Vec2)>` - `(命中距离, 命中点世界 XZ)`;未命中为 `None`。
pub fn ray_to_shapes(world: &CollisionWorld, origin: Vec3, dir: Vec3) -> Option<(f32, Vec2)> {
    let flat: f32 = (dir[0] * dir[0] + dir[2] * dir[2]).sqrt();
    // 视线完全竖直(pitch 接近 ±90°)时 XZ 投影退化,没有遮挡可言。
    if flat < 1.0e-5 {
        return None;
    }
    let (ux, uz): (f32, f32) = (dir[0] / flat, dir[2] / flat);
    let mut best: Option<(f32, Vec2)> = None;
    for shape in world.get_shapes() {
        let candidate: Option<(f32, Vec2)> = match shape {
            Shape::Aabb { center, half } => ray_into_box(
                [origin[0], origin[2]],
                [ux, uz],
                *center,
                [half[0] + CAMERA_PROBE_RADIUS, half[1] + CAMERA_PROBE_RADIUS],
            ),
            Shape::Circle { center, radius } => ray_into_circle(
                [origin[0], origin[2]],
                [ux, uz],
                *center,
                *radius + CAMERA_PROBE_RADIUS,
            ),
        };
        let Some((distance, at)) = candidate else {
            continue;
        };
        if distance < 0.0 {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|(current, _): &(f32, [f32; 2])| distance < *current)
        {
            best = Some((distance, at));
        }
    }
    best.map(|(distance, _): (f32, [f32; 2])| {
        (
            distance * flat,
            [
                origin[0] + ux * distance * flat,
                origin[2] + uz * distance * flat,
            ],
        )
    })
}

/// 单位方向 `dir` 的射线撞上轴对齐盒(半尺寸已含探针半径)时的参数 t。
///
/// 标准 slab 法:对 X / Z 两轴各求一次进出区间并取交集,交集的近端就是
/// 命中参数。起点在盒内时近端为 0(调用方按「已经贴住」处理)。
///
/// # Arguments
///
/// - `Vec2` - 射线起点 XZ。
/// - `Vec2` - XZ 平面上的单位方向。
/// - `Vec2` - 盒中心 XZ。
/// - `Vec2` - 盒半尺寸 XZ(已含探针半径)。
///
/// # Returns
///
/// - `Option<(f32, Vec2)>` - `(参数 t, 命中点 XZ)`;未命中为 `None`。
fn ray_into_box(origin: Vec2, dir: Vec2, center: Vec2, half: Vec2) -> Option<(f32, Vec2)> {
    let mut near: f32 = f32::NEG_INFINITY;
    let mut far: f32 = f32::INFINITY;
    for axis in 0..2 {
        // 方向分量接近 0 时该轴不构成约束:只有起点已经在板内才算命中。
        if dir[axis].abs() < 1.0e-6 {
            if (origin[axis] - center[axis]).abs() > half[axis] {
                return None;
            }
            continue;
        }
        let lo: f32 = (center[axis] - half[axis] - origin[axis]) / dir[axis];
        let hi: f32 = (center[axis] + half[axis] - origin[axis]) / dir[axis];
        let (enter, exit): (f32, f32) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        near = near.max(enter);
        far = far.min(exit);
        if near > far {
            return None;
        }
    }
    // 盒在射线**背后**时 `far` 为负:射线朝反方向走,永远不会撞上。
    // 少了这一句,任何背对建筑的相机都会拿到一个 `t = 0` 的假命中,
    // 回避逻辑随即把距离扣成负数(`res=-0.45`),相机被推到玩家背后。
    if far < 0.0 {
        return None;
    }
    let t: f32 = near.max(0.0);
    Some((t, [origin[0] + dir[0] * t, origin[1] + dir[1] * t]))
}

/// 单位方向 `dir` 的射线撞上圆柱(半径已含探针半径)时的参数 t。
///
/// 圆-圆求交:解 `|o + t·d - c|² = r²` 的二次方程取最小非负根。判别式
/// 小于 0 表示射线从旁边擦过,不相交。
///
/// # Arguments
///
/// - `Vec2` - 射线起点 XZ。
/// - `Vec2` - XZ 平面上的单位方向。
/// - `Vec2` - 圆柱中心 XZ。
/// - `f32` - 圆柱半径(已含探针半径)。
///
/// # Returns
///
/// - `Option<(f32, Vec2)>` - `(参数 t, 命中点 XZ)`;未命中为 `None`。
fn ray_into_circle(origin: Vec2, dir: Vec2, center: Vec2, radius: f32) -> Option<(f32, Vec2)> {
    let (ox, oz): (f32, f32) = (origin[0] - center[0], origin[1] - center[1]);
    let b: f32 = 2.0 * (ox * dir[0] + oz * dir[1]);
    let c: f32 = ox * ox + oz * oz - radius * radius;
    let discriminant: f32 = b * b - 4.0 * c;
    if discriminant < 0.0 {
        return None;
    }
    let root: f32 = discriminant.sqrt();
    let near: f32 = (-b - root) * 0.5;
    let far: f32 = (-b + root) * 0.5;
    // 起点在圆柱内:远端才是「穿出去」的那一侧,近端为负没有意义。
    let t: f32 = if near >= 0.0 { near } else { far };
    if t < 0.0 {
        return None;
    }
    Some((t, [origin[0] + dir[0] * t, origin[1] + dir[1] * t]))
}

impl Default for CollisionWorld {
    /// 返回空碰撞世界,与 `CollisionWorld::new` 等价。
    fn default() -> Self {
        CollisionWorld::new()
    }
}

/// 圆形 vs AABB 的分离量。
///
/// # Arguments
///
/// - `Vec2` - AABB 中心的世界 XZ 坐标。
/// - `Vec2` - AABB 半尺寸(米)。
/// - `Vec2` - 圆心的世界 XZ 坐标。
/// - `f32` - 圆半径(米)。
///
/// # Returns
///
/// - `Option<(Vec2, f32)>` - `(推出方向, 推出距离)`;不重叠时为 `None`。
fn push_out_aabb(center: Vec2, half: Vec2, point: Vec2, radius: f32) -> Option<(Vec2, f32)> {
    let closest: Vec2 = [
        point[0].clamp(center[0] - half[0], center[0] + half[0]),
        point[1].clamp(center[1] - half[1], center[1] + half[1]),
    ];
    let delta: Vec2 = [point[0] - closest[0], point[1] - closest[1]];
    let distance: f32 = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
    if distance > radius {
        return None;
    }
    if distance > INSIDE_EPSILON {
        return Some((
            [delta[0] / distance, delta[1] / distance],
            radius - distance,
        ));
    }
    // 圆心已经落在盒内:沿「离边界最近」的那个面推出去。
    let offset: Vec2 = [point[0] - center[0], point[1] - center[1]];
    let slack_x: f32 = half[0] - offset[0].abs();
    let slack_z: f32 = half[1] - offset[1].abs();
    if slack_x <= slack_z {
        let sign: f32 = if offset[0] >= 0.0 { 1.0 } else { -1.0 };
        Some(([sign, 0.0], radius + slack_x))
    } else {
        let sign: f32 = if offset[1] >= 0.0 { 1.0 } else { -1.0 };
        Some(([0.0, sign], radius + slack_z))
    }
}

/// 圆形 vs 圆的分离量。
///
/// # Arguments
///
/// - `Vec2` - 圆心 A 的世界 XZ 坐标。
/// - `f32` - 圆 A 的半径(米)。
/// - `Vec2` - 圆心 B(玩家圆心)的世界 XZ 坐标。
/// - `f32` - 圆 B 的半径(米)。
///
/// # Returns
///
/// - `Option<(Vec2, f32)>` - `(推出方向, 推出距离)`;不重叠时为 `None`。
fn push_out_circle(center: Vec2, other: f32, point: Vec2, radius: f32) -> Option<(Vec2, f32)> {
    let delta: Vec2 = [point[0] - center[0], point[1] - center[1]];
    let distance: f32 = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
    let reach: f32 = other + radius;
    if distance > reach {
        return None;
    }
    if distance > INSIDE_EPSILON {
        return Some(([delta[0] / distance, delta[1] / distance], reach - distance));
    }
    Some(([1.0, 0.0], reach))
}

/// 由资产 `bounds` 推导一个摆放实例在世界 XZ 平面上的 AABB。
///
/// 资产的 `bounds` 是**本地** Y-up 包围盒:建筑 / 车辆这类资产的长度在
/// 本地 X 上、宽度在本地 Z 上。摆放时绕 Y 轴旋转 `yaw` 并统一缩放
/// `scale`,所以：
///
/// 1. 本地足迹中心 = `((min.x + max.x) / 2, (min.z + max.z) / 2)` —— 资产
///    不保证足迹相对原点居中(例如 `bldg_deco_pink` 的 z 从 -5.08 到
///    6.49),不补这一步墙就会整体偏移半个身位。
/// 2. 本地足迹半尺寸 = `((max.x - min.x) / 2, (max.z - min.z) / 2)`。
/// 3. 用与 `Instance::new` 相同的旋转约定(本地 +X → 世界
///    `(cos yaw, 0, -sin yaw)`,本地 +Z → `(sin yaw, 0, cos yaw)`)把
///    中心偏移旋到世界,半尺寸取旋转后外接盒(yaw 为 90° 整数倍时是精确值)。
///
/// # Arguments
///
/// - `Vec3` - 资产 `bounds.min`(本地坐标)。
/// - `Vec3` - 资产 `bounds.max`(本地坐标)。
/// - `f32` - 绕 Y 轴的摆放朝向(弧度)。
/// - `f32` - 统一缩放系数。
/// - `Vec3` - 摆放位置(世界坐标)。
///
/// # Returns
///
/// - `(Vec2, Vec2)` - `(世界 XZ 盒中心, 世界 XZ 盒半尺寸)`。
pub fn placement_box(min: Vec3, max: Vec3, yaw: f32, scale: f32, position: Vec3) -> (Vec2, Vec2) {
    let (sin_yaw, cos_yaw): (f32, f32) = yaw.sin_cos();
    let local_center: Vec2 = [(min[0] + max[0]) * 0.5, (min[2] + max[2]) * 0.5];
    let local_half: Vec2 = [(max[0] - min[0]) * 0.5, (max[2] - min[2]) * 0.5];
    let center: Vec2 = [
        position[0] + (cos_yaw * local_center[0] + sin_yaw * local_center[1]) * scale,
        position[2] + (-sin_yaw * local_center[0] + cos_yaw * local_center[1]) * scale,
    ];
    let half: Vec2 = [
        (cos_yaw.abs() * local_half[0] + sin_yaw.abs() * local_half[1]) * scale,
        (sin_yaw.abs() * local_half[0] + cos_yaw.abs() * local_half[1]) * scale,
    ];
    (center, half)
}
