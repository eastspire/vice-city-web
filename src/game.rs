//! 游戏主体:场景搭建、异步资产加载、固定步长循环、输入、昼夜循环。
//!
//! 架构要点(与 `euv-game-real-api-notes` 的坑对应):
//!
//! - **不在 hook / 事件回调里调用 euv hook**。整个游戏只有一棵静态
//!   `html!` 树(见 [`app_root`]),挂载完成后所有每帧逻辑都由裸
//!   `web_sys` + `Closure` 驱动,VDOM 不参与每帧渲染。
//! - 资产用 `fetch` + `wasm_bindgen_futures` 异步加载,带进度条。
//! - 固定步长(累加器 + `requestAnimationFrame`),`dt` 钳位在 50 ms。
//! - 场景按资产分批([`render::SceneBatch`]),同类资产只解析一次。

use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    rc::Rc,
};

use euv::{
    VirtualNode, html,
    js_sys::{Function, Promise, Undefined},
    wasm_bindgen::JsCast,
    wasm_bindgen::prelude::*,
    wasm_bindgen_futures::{JsFuture, spawn_local},
    web_sys::{
        DomRect, Element, Event, EventTarget, HtmlCanvasElement, HtmlInputElement, KeyboardEvent,
        MouseEvent, Response, TouchEvent, TouchList, WheelEvent, Window, window,
    },
};

use crate::{
    camera::{CAMERA_MIN_HEIGHT, Camera, Mat4},
    collision::{CollisionWorld, placement_box},
    r#const::*,
    mesh::{Bounds, GpuMesh, MeshAsset, MeshError, MeshPart, expand_asset},
    player::{Player, RUN_SPEED, WALK_SPEED, joint_pivot, limb_matrix, limb_swing},
    render::{
        DayPhase, Instance, MeshAssetGpu, NEAR_CULL_RADIUS, NEAR_CULL_RADIUS_FOLLOW, Renderer,
        Scene, SceneBatch, SceneLighting, SoftwareRenderer, WebGlRenderer, build_gpu_mesh,
        normalize3,
    },
    traffic::{PICKUP_RADIUS, Traffic, TrafficCar, apply_pickup, nearest_car},
    r#type::{Mat4Data, MeshExtent, PalmSpots, Placement, Vec2, Vec3},
};

// ===========================================================================
// 常量
//
// 字符串常量统一放在 `crate::r#const`(上面已 glob 引入),
// 这里只保留数值常量。
// ===========================================================================

/// 固定步长:1/60 秒。
const FIXED_DT: f32 = 1.0 / 60.0;
/// 单帧最大累积时间(秒),超过就丢弃(标签页切回来时不追赶)。
const MAX_FRAME_TIME: f32 = 0.25;
/// 滚轮每单位缩放。
const ZOOM_STEP: f32 = 0.0016;

/// 下车后玩家与车身保持的距离(米)。
const EXIT_CAR_OFFSET: f32 = 2.4;
/// 所有会被游戏接管的按键。
///
/// 之所以用数组扫描而不是 `matches!`:这些键名都是 `&'static str` 常量,
/// 放进 `matches!` 会被解释成 pattern 绑定(编译报 E0408),不是常量比较。
const TRACKED_KEYS: &[&str] = &[
    KEYW,
    KEYS,
    KEYA,
    KEYD,
    KEYR,
    KEYT,
    KEYF,
    KEYTAB,
    ARROWUP,
    ARROWDOWN,
    ARROWLEFT,
    ARROWRIGHT,
    KEY_SHIFT_LEFT,
    KEY_SHIFT_RIGHT,
];

/// 玩家骨架切 part 时的占位错误信息。

/// 第三人称默认俯角(弧度)。
///
/// **必须很小。** 眼点高度 = `FOLLOW_HEIGHT + sin(pitch) · distance`。
/// 旧的 0.22 rad 配 7.4 m 距离把相机顶到 2.96 m,而焦点只有 1.35 m ——
/// 视线俯角约 12.6°,再叠上 30° 的垂直视场,画面下缘几乎全是地面。
/// 站在街上时地面是一整张 `y = 0` 的网格,于是整屏被那片单色地面糊死,
/// 角色和街景都看不见(实测:0.22 时整帧只有一种颜色,0.05 才露出楼房)。
/// 0.05 rad(≈2.9°)让相机只比角色胸口高 0.37 m,取景是标准的
/// 「越过肩膀看前方街道」。
const FOLLOW_PITCH: f32 = 0.16;
/// 第三人称垂直视场角(弧度)。
///
/// 取 75° 而不是 60°:玩家站在 14 m 宽的街道上,两侧的楼离镜头中心
/// 轴约 20 m。60° 垂直视场配 16:9 画幅只有约 34° 的水平张角,半角
/// 17° 在 7.4 m 处只覆盖 ±2.3 m —— 街道两侧的楼、行道树、路灯全部
/// 落在视锥之外,画面里只剩脚下的一小块路面。75° 把水平张角拉到约
/// 50°,半角 25° 覆盖 ±3.5 m,再配合稍近的跟随距离,整条街的立面
/// 才进得了取景框。
const FOLLOW_FOV: f32 = 1.309;
/// 第三人称近裁剪面(米)。
///
/// **不能太小。** 0.05 m 配 420 m 的远平面是 8400:1 的深度比,在没有
/// GPU 的软件光栅(swiftshader)上深度缓冲精度极低,近处几何的深度值
/// 全被压到 0 附近,互相 `LEQUAL` 判等 —— 整帧只剩最后画的那个批次,
/// 画面糊成一片单色(实测:贴地第三人称只有 2~3 种颜色,把 near 提到
/// 12 m 反而"正常"了,因为近处几何直接被切掉、只剩远景不打架)。
/// 相机最近的落点是 2.2 m(遮挡回避硬下限),0.35 m 的近平面已经
/// 足够避开角色自身,又给深度缓冲留出 1200:1 的余量。
const FOLLOW_NEAR: f32 = 0.35;
/// 第三人称远裁剪面(米)。
///
/// 城市是 ±150 m 的网格,站在角落能看到的最远楼角约 420 m,但雾在
/// 120 m 就开始起效,460 m 处完全融进天色 —— 所以 300 m 已经「看不出
/// 被裁掉」,却把深度比从 8400:1 压到 857:1,配合 `FOLLOW_NEAR` 给出
/// 软件光栅也能用的精度。
const FOLLOW_FAR: f32 = 300.0;

/// 第三人称跟随相机的默认距离(米)。
///
/// **取景比例的依据**:垂直视场 `FOLLOW_FOV = 60°`,焦点高度
/// `FOLLOW_HEIGHT = 1.35 m`。角色约 1.8 m 高,在 7.4 m 处的垂直张角
/// 是 `2·atan(0.9 / 7.4) ≈ 13.8°`,占 60° 视场约 **23%** 屏高 —— 角色
/// 清晰可辨,同时周围两条街的立面 / 路缘 / 行道树都还在画面里。
///
/// 之前的 5.6 m 只有 ~17% 屏高,再叠加 `NEAR_CULL_RADIUS = 26 m` 的
/// 近处剔除(相机离角色才 5.6 m,半径 26 m 内的**所有**楼都被剔掉)之后,
/// 画面里只剩地面和一面近处的楼,角色几乎看不见。
const FOLLOW_DISTANCE: f32 = 8.2;
/// 第三人称跟随相机的最小距离(滚轮拉近下限)。
const FOLLOW_DISTANCE_MIN: f32 = 2.2;
/// 第三人称跟随相机的最大距离(滚轮拉远上限)。
const FOLLOW_DISTANCE_MAX: f32 = 22.0;
/// 遮挡回避「拉近」的阻尼速率(1/秒):越大越快缩进来。
///
/// 必须比拉远快一个量级 —— 撞墙时镜头的第一反应就该是「躲」,而不是
/// 用 0.3 s 慢慢推进去展示墙面。
const OCCLUSION_IN_RATE: f32 = 22.0;
/// 遮挡回避「拉远」的阻尼速率(1/秒):离开墙后平滑弹回。
const OCCLUSION_OUT_RATE: f32 = 3.4;
/// 跟随焦点的高度(米)——看胸口而不是看脚。
///
/// 取 1.45 而不是 1.35:角色约 1.8 m 高,焦点略高一点能让头顶留在
/// 取景框内,而不是被画面上缘切掉。
const FOLLOW_HEIGHT: f32 = 1.45;
/// 跟随焦点跟随角色的阻尼速率(1/秒):越大越硬,越小越飘。
const FOLLOW_DAMPING: f32 = 9.0;
/// 玩家碰撞圆半径(米)。
const PLAYER_RADIUS: f32 = 0.35;
/// 世界半边长(米):玩家不许走出这个范围。
const WORLD_HALF: f32 = 150.0;
/// 玩家出生点(世界坐标):X = 30 那条街的**路中间偏东的车道**,z = 45。
///
/// 选点是拿截图试出来的,踩过三个坑:
///
/// 1. 路中间(x = 30)不行 —— 相机落在另一侧车道,背后除了沥青什么都
///    没有,整屏被单色地面糊死。
/// 2. 人行道(x = 38.5)也不行 —— 行道树在 x = 38.98,相机往后退 7.4 m
///    必然撞进树干,遮挡回避把距离一路压到 2.2 m,镜头直接怼在树皮上。
/// 3. 横对着马路更不行 —— 相机被推到楼面(x ≈ 49)跟前,一出生就是一堵墙。
///
/// 现在这个点:车道内(30.4..33.6)的 x = 33.5,离两侧的建筑和行道树都
/// 有 6 m 以上;z = 45 远离 z = 24 / 60 两处路口,取景里是一条笔直的
/// 街道,身后是同一条街往远处延伸,两侧是楼房和行道树。
const SPAWN_POINT: Vec3 = [33.5, 0.0, 45.0];
/// 玩家出生朝向(弧度)。
///
/// 玩家的前向是 `(cos yaw, -sin yaw)`。`yaw = PI/2` 时前向 = `(0, -1)`,
/// 也就是朝北、顺着这条街往远处看 —— 相机落在角色南侧 7.4 m,同样在这
/// 条街上,背后是同一条向远处延伸的马路,而不是楼面。
const SPAWN_YAW: f32 = std::f32::consts::FRAC_PI_2;

// ===========================================================================
// 场景蓝图:街区网格(多条平行街道 × 横向连接街道)
//
// 城市从「一条 86 m 直线」扩成真正的网格:
//
// ```text
//            x=-120    x=-60     x=0      x=60      x=120
//   z=-120     +--------+--------+--------+--------+
//              | block  | block  | block  | block  |
//   z=-60      +=== 十字路口(南北向街道)==+
//              | block  | block  | block  | block  |
//   z=0        +=== 十字路口 ============+
//              ...
// ```
//
// `CITY_HALF` 是城市的半边长(米),4 条南北向 + 4 条东西向街道 = 4×4
// = 16 个网格路口,每个路口四角都有红绿灯。
// ===========================================================================

/// 城市的半边长(米):地面覆盖 [-150, 150] × [-150, 150],即 300 m × 300 m。
const CITY_HALF: f32 = 150.0;
/// 街道半宽(路面,不含人行道)。
const STREET_HALF_WIDTH: f32 = 7.0;
/// 人行道宽度(单侧)。
const SIDEWALK_WIDTH: f32 = 3.6;
/// 网格街道的轴线坐标(正负各一条,共 4 条)。
const STREET_LINES: &[f32] = &[-90.0, -30.0, 30.0, 90.0];
/// 街区内部沿道路方向的楼间距(米),留出巷子 / 后巷。
const LOT_PITCH: f32 = 34.0;
/// 建筑围合的「内圈」半径(米):离街区中心多远开始放楼。
const BLOCK_INNER: f32 = 11.0;
/// 地面网格在每个方向上的分段数(整城一次,不是每街区一次)。
const GROUND_SUBDIV: usize = 192;

/// 网格街道之间的一个街区:楼围合出一个内部院落,而不是沿街一条线。
struct BlockLayout {
    /// 街区中心的 X 坐标。
    cx: f32,
    /// 街区中心的 Z 坐标。
    cz: f32,
}

/// 枚举全部 3×3 个街区中心(4 条街道围出 3×3 个街区)。
///
/// # Returns
///
/// - `Vec<BlockLayout>` - 9 个街区中心,顺序为行优先(先 z 后 x)。
fn block_layouts() -> Vec<BlockLayout> {
    let mut out: Vec<BlockLayout> = Vec::new();
    for i in 0..STREET_LINES.len().saturating_sub(1) {
        for j in 0..STREET_LINES.len().saturating_sub(1) {
            out.push(BlockLayout {
                cx: (STREET_LINES[i] + STREET_LINES[j + 1]) * 0.5,
                cz: (STREET_LINES[j] + STREET_LINES[j + 1]) * 0.5,
            });
        }
    }
    out
}

/// 判断某个 XZ 位置是否落在街道 / 路口范围内(用于地面着色与摆放避让)。
///
/// # Arguments
///
/// - `f32` - 世界 X 坐标(米)。
/// - `f32` - 世界 Z 坐标(米)。
///
/// # Returns
///
/// - `bool` - `true` 表示该点在可通行的沥青面上。
fn on_roadway(x: f32, z: f32) -> bool {
    on_roadway_with_margin(x, z, 0.0)
}

/// 带余量判断「这个点(连同半径 `margin`)是否压在可通行的沥青面上」。
///
/// 楼是**有体积**的:之前只用楼中心点判 `on_roadway`,结果一栋半边长
/// 10 m 的楼中心在 x=19.3(离街 30 还有 10.7 m,判定「不在路上」),
/// 它的东墙却伸到 x=29.4 —— 正好压在 26.5 的车道上。动态车队于是每帧
/// 撞上一堵隐形的墙,速度被清零,看起来就是「车开了但不动」。
/// 加上余量后判定的是楼的**占地**,车行道就真的空出来了。
///
/// # Arguments
///
/// - `f32` - 世界 X 坐标(米)。
/// - `f32` - 世界 Z 坐标(米)。
/// - `f32` - 额外余量(米)—— 传楼的碰撞半尺寸。
///
/// # Returns
///
/// - `bool` - `true` 表示该范围压在沥青面上。
fn on_roadway_with_margin(x: f32, z: f32, margin: f32) -> bool {
    on_roadway_raw(x, z, margin)
}

/// 「沥青面」判定的真正实现(见 [`on_roadway_with_margin`] 的说明)。
///
/// # Arguments
///
/// - `f32` - 世界 X 坐标(米)。
/// - `f32` - 世界 Z 坐标(米)。
/// - `f32` - 额外余量(米)。
///
/// # Returns
///
/// - `bool` - `true` 表示压在沥青面上。
fn on_roadway_raw(x: f32, z: f32, margin: f32) -> bool {
    on_roadway_inner(x, z, margin)
}

/// 沥青面判定的计算核心。
///
/// # Arguments
///
/// - `f32` - 世界 X 坐标(米)。
/// - `f32` - 世界 Z 坐标(米)。
/// - `f32` - 额外余量(米)。
///
/// # Returns
///
/// - `bool` - `true` 表示压在沥青面上。
fn on_roadway_inner(x: f32, z: f32, margin: f32) -> bool {
    let reach: f32 = STREET_HALF_WIDTH + SIDEWALK_WIDTH + margin;
    let mut on_x_road: bool = false;
    let mut on_z_road: bool = false;
    for line in STREET_LINES {
        if (x - *line).abs() <= reach {
            on_z_road = true;
        }
        if (z - *line).abs() <= reach {
            on_x_road = true;
        }
    }
    on_x_road || on_z_road
}

/// 一栋楼在场景里的摆放。
struct BuildingPlacement {
    /// 资产 id。
    asset: &'static str,
    /// 世界坐标(x, z)。
    position: [f32; 2],
    /// 绕 Y 轴的朝向(弧度)。
    yaw: f32,
    /// 统一缩放。
    scale: f32,
    /// 色调乘子,给同一批楼一点色彩变化。
    tint: [f32; 3],
}

/// 楼 / 招牌 / 车 / 行人用的建筑资产池(全部不同资产,循环取用)。
///
/// 网格街区要摆 ~200 栋楼,如果只给 10 个不同资产,同一个 mesh 会以
/// 20 个实例反复出现 —— 顶点数据仍然只解析一次(instancing 友好),
/// 但远景会明显看出「同一栋楼在原地克隆」。12 个资产 + 按格子位置
/// 做 `scale` / `tint` / `yaw` 抖动,足以让成片街区读起来是城市而不是
/// 复制粘贴。
const BUILDING_POOL: &[&str] = &[
    BLDG_DECO_PINK,
    BLDG_MINT_SHOP,
    BLDG_CREAM_BLOCK,
    BLDG_CORAL_HALL,
    BLDG_TEAL_LOFT,
    BLDG_AQUA_ARCADE,
    BLDG_DECO_TEAL,
    BLDG_SAND_MIDRISE,
    BLDG_PINK_TERRACE,
    BLDG_WHITE_LANDMARK,
    BLDG_APRICOT_MOTEL,
    BLDG_LILAC_TOWER,
];

/// 招牌资产池。
const SIGN_POOL: &[&str] = &[
    SIGN_HOTEL,
    SIGN_PIZZA,
    SIGN_CLUB,
    SIGN_BAR,
    SIGN_TROPIC,
    SIGN_ARCADE,
    SIGN_MOTEL,
    SIGN_DINER,
];

/// 车辆资产池。
const VEHICLE_POOL: &[&str] = &[CAR_TAXI, CAR_POLICE, CAR_SEDAN, CAR_COUPE, TRUCK_PICKUP];

/// 行人资产池。
const PED_POOL: &[&str] = &[PED_SUIT, PED_STREETWEAR, PED_DRESS, PED_OVERALLS];

/// 确定性的整数哈希(无随机数依赖,同一 seed 永远生成同一座城市)。
///
/// 用它而不是任何随机源,是因为场景是同步生成的:必须保证每次刷新
/// 布局完全一致,否则截图 / 回归对比会随机漂移。
///
/// # Arguments
///
/// - `u32` - 种子。
/// - `u32` - 第二个盐值(让不同的生成步骤互不相关)。
///
/// # Returns
///
/// - `u32` - 哈希后的值。
fn hash2(seed: u32, salt: u32) -> u32 {
    let mut h: u32 = seed.wrapping_mul(0x9E37_79B9).wrapping_add(salt);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^ (h >> 16)
}

/// 由哈希得到 `[0, 1)` 的确定性浮点。
///
/// # Arguments
///
/// - `u32` - 种子。
/// - `u32` - 盐值。
///
/// # Returns
///
/// - `f32` - 落在 `[0, 1)` 的值。
fn hash_unit(seed: u32, salt: u32) -> f32 {
    (hash2(seed, salt) % 10_000) as f32 / 10_000.0
}

/// 路灯 / 交通灯等街道道具的摆放。
struct PropPlacement {
    /// 资产 id。
    asset: &'static str,
    /// 世界坐标。
    position: [f32; 3],
    /// 朝向(弧度)。
    yaw: f32,
    /// 色调。
    tint: [f32; 3],
}

/// 楼的位置在街区内的符号偏移(前 / 后 / 左 / 右 四条围合边)。
///
/// 街区的四边都放楼,中间留出 `BLOCK_INNER` 的内院 —— 这样每个街区
/// 内部有建筑围合和巷子,而不是只有沿街一排。
const BLOCK_RING: &[(f32, f32, f32)] = &[
    // (相对街区中心的 dx, dz, 朝向街道的 yaw)
    (0.0, -1.0, 0.0),
    (0.0, 1.0, std::f32::consts::PI),
    (-1.0, 0.0, std::f32::consts::FRAC_PI_2),
    (1.0, 0.0, -std::f32::consts::FRAC_PI_2),
];

/// 程序化生成整座城市的建筑布局。
///
/// 每个街区:四条围合边各放 2~3 栋楼(沿边错开),巷子自然形成在
/// 相邻两栋之间;街区之间再穿插路口四角的对角楼,让街道交叉口
/// 也有转角的门面。
///
/// # Returns
///
/// - `Vec<BuildingPlacement>` - 全城建筑摆放表。
fn build_city_buildings() -> Vec<BuildingPlacement> {
    build_city_buildings_with(BUILDING_FOOTPRINT_GUARD)
}

/// 楼的最大占地半尺寸(米)—— 用来保证整栋楼都不压到车行道。
///
/// 取一个偏保守的常数而不是逐个资产的包围盒:建筑资产的包围盒差异很大
/// (10.7 m 到 6.4 m),用统一余量判定简单、可预测,而且宁可少摆一栋楼
/// 也不会出现「隐形墙堵住车道」。
const BUILDING_FOOTPRINT_GUARD: f32 = 11.0;

/// 楼的占地(含余量)是否压到车行道。
///
/// # Arguments
///
/// - `f32` - 楼中心的 X(米)。
/// - `f32` - 楼中心的 Z(米)。
/// - `f32` - 占地半尺寸(米)。
///
/// # Returns
///
/// - `bool` - `true` 表示楼的任一轴向范围压到了沥青面。
fn footprint_hits_roadway(x: f32, z: f32, guard: f32) -> bool {
    on_roadway_with_margin(x, z, guard)
}

/// 用「占地余量」生成街区里的楼(见 [`build_city_buildings`] 的调用点)。
///
/// # Arguments
///
/// - `f32` - 每栋楼在两个轴上额外让开的距离(米)。
///
/// # Returns
///
/// - `Vec<BuildingPlacement>` - 楼的摆放列表。
fn build_city_buildings_with(guard: f32) -> Vec<BuildingPlacement> {
    let mut out: Vec<BuildingPlacement> = Vec::new();
    let mut seed: u32 = 0x5EED_0001;
    for (bi, block) in block_layouts().into_iter().enumerate() {
        for (edge, (dx, dz, yaw)) in BLOCK_RING.iter().enumerate() {
            let count: usize = 2 + (bi + edge) % 2;
            for slot in 0..count {
                seed = hash2(seed, 0x9E37);
                // 沿边错开:LOT_PITCH 的间距 + 确定性抖动,形成巷子。
                let along: f32 = (slot as f32 - (count as f32 - 1.0) * 0.5) * LOT_PITCH;
                let jitter: f32 = (hash_unit(seed, 1) - 0.5) * 6.0;
                let (px, pz): (f32, f32) = if dx.abs() > 0.5 {
                    // 左右边:沿 Z 排布。
                    (block.cx + dx * BLOCK_INNER, block.cz + along + jitter)
                } else {
                    // 前后边:沿 X 排布。
                    (block.cx + along + jitter, block.cz + dz * BLOCK_INNER)
                };
                // 整栋楼都要让开车行道:除了中心点,四角也得在沥青面之外。
                // `guard` 是资产包围盒的最大半尺寸(由下面的
                // `building_footprint` 提供),宁可少放一栋也不能让楼
                // 的围墙伸进车道。
                if on_roadway(px, pz) {
                    continue;
                }
                if footprint_hits_roadway(px, pz, guard) {
                    continue;
                }
                let scale: f32 = 0.88 + hash_unit(seed, 2) * 0.34;
                let tint: [f32; 3] = [
                    0.94 + hash_unit(seed, 3) * 0.14,
                    0.94 + hash_unit(seed, 4) * 0.14,
                    0.94 + hash_unit(seed, 5) * 0.14,
                ];
                out.push(BuildingPlacement {
                    asset: BUILDING_POOL[hash2(seed, 6) as usize % BUILDING_POOL.len()],
                    position: [px, pz],
                    yaw: *yaw + (hash_unit(seed, 7) - 0.5) * 0.12,
                    scale,
                    tint,
                });
            }
        }
    }
    // 路口四角的对角楼:给每个网格路口一个转角门面。
    for (xi, line_x) in STREET_LINES.iter().enumerate() {
        for (zi, line_z) in STREET_LINES.iter().enumerate() {
            seed = hash2(seed, 0xBEEF);
            if (xi + zi) % 2 != 0 {
                continue;
            }
            let corner: f32 = STREET_HALF_WIDTH + SIDEWALK_WIDTH + 9.0;
            let px: f32 = *line_x + corner;
            let pz: f32 = *line_z + corner;
            if px.abs() > CITY_HALF - 6.0 || pz.abs() > CITY_HALF - 6.0 {
                continue;
            }
            out.push(BuildingPlacement {
                asset: BUILDING_POOL[hash2(seed, 8) as usize % BUILDING_POOL.len()],
                position: [px, pz],
                yaw: -std::f32::consts::FRAC_PI_4,
                scale: 0.95 + hash_unit(seed, 9) * 0.3,
                tint: [1.0, 0.96 + hash_unit(seed, 10) * 0.1, 1.0],
            });
        }
    }
    out
}

/// 程序化生成全部街道道具(路灯 / 交通灯 / 长椅 / 垃圾桶 / 消防栓 / 报摊 / 电话亭)。
///
/// **每个网格路口四角都放交通灯**(需求硬性规定),路灯沿每条街道
/// 的人行道两侧交替排布。
///
/// # Returns
///
/// - `Vec<PropPlacement>` - 全城道具摆放表。
fn build_city_props() -> Vec<PropPlacement> {
    let mut out: Vec<PropPlacement> = Vec::new();
    let corner: f32 = STREET_HALF_WIDTH + 0.9;
    // ---- 每个网格路口四个角的交通灯 ----
    for line_x in STREET_LINES {
        for line_z in STREET_LINES {
            for (sx, sz) in [(-1.0f32, -1.0f32), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                out.push(PropPlacement {
                    asset: PROP_TRAFFICLIGHT,
                    position: [*line_x + sx * corner, 0.0, *line_z + sz * corner],
                    yaw: if sx * sz > 0.0 {
                        0.0
                    } else {
                        std::f32::consts::PI
                    },
                    tint: [1.0, 1.0, 1.0],
                });
            }
        }
    }
    // ---- 路灯:沿每条街道两侧交替 ----
    let mut seed: u32 = 0xC0FFEE;
    for line in STREET_LINES {
        let mut step: f32 = -CITY_HALF + 18.0;
        while step < CITY_HALF - 18.0 {
            seed = hash2(seed, 0x11);
            // 南北向街道(x = line)与东西向街道(z = line)各放一侧。
            out.push(PropPlacement {
                asset: PROP_STREETLIGHT,
                position: [*line + STREET_HALF_WIDTH + 1.5, 0.0, step],
                yaw: std::f32::consts::FRAC_PI_2,
                tint: [1.0, 1.0, 1.0],
            });
            out.push(PropPlacement {
                asset: PROP_STREETLIGHT,
                position: [step, 0.0, *line - STREET_HALF_WIDTH - 1.5],
                yaw: std::f32::consts::PI,
                tint: [1.0, 1.0, 1.0],
            });
            // 沿线点缀:长椅 / 垃圾桶 / 消防栓 / 报摊 / 电话亭 / 锥桶。
            match hash2(seed, 2) % 6 {
                0 => out.push(PropPlacement {
                    asset: PROP_BENCH,
                    position: [*line + STREET_HALF_WIDTH + 2.2, 0.0, step + 7.0],
                    yaw: 0.0,
                    tint: [1.0, 0.98, 0.94],
                }),
                1 => out.push(PropPlacement {
                    asset: PROP_TRASH_BIN,
                    position: [step + 5.0, 0.0, *line + STREET_HALF_WIDTH + 2.0],
                    yaw: 0.3,
                    tint: [0.96, 1.0, 0.98],
                }),
                2 => out.push(PropPlacement {
                    asset: PROP_FIRE_HYDRANT,
                    position: [*line - STREET_HALF_WIDTH - 1.8, 0.0, step + 9.0],
                    yaw: 0.0,
                    tint: [1.0, 0.9, 0.9],
                }),
                3 => out.push(PropPlacement {
                    asset: PROP_NEWSSTAND,
                    position: [step + 11.0, 0.0, *line - STREET_HALF_WIDTH - 2.0],
                    yaw: std::f32::consts::FRAC_PI_2,
                    tint: [1.0, 1.0, 1.0],
                }),
                4 => out.push(PropPlacement {
                    asset: PROP_PHONE_BOOTH,
                    position: [*line + STREET_HALF_WIDTH + 2.0, 0.0, step + 13.0],
                    yaw: -std::f32::consts::FRAC_PI_2,
                    tint: [0.98, 1.0, 1.0],
                }),
                _ => out.push(PropPlacement {
                    asset: PROP_TRAFFIC_CONE,
                    // 锥桶要放在**路缘**上,不能丢在车行道正中:之前放在
                    // `line + 2.4`,正好压在 26.5 / 33.5 的车道上,动态
                    // 车队每帧撞上去,看起来就是「车开了但不动」。
                    position: [step + 3.0, 0.0, *line + STREET_HALF_WIDTH + 0.8],
                    yaw: hash_unit(seed, 3) * 1.2,
                    tint: [1.05, 0.9, 0.7],
                }),
            }
            step += 30.0;
        }
    }
    out
}

/// 程序化生成棕榈树位置(沿每条街道两侧 + 街区内院点缀)。
///
/// # Returns
///
/// - `PalmSpots` - 棕榈的世界 (x, z) 坐标。
fn build_city_palms() -> PalmSpots {
    let mut out: Vec<[f32; 2]> = Vec::new();
    for line in STREET_LINES {
        let mut step: f32 = -CITY_HALF + 26.0;
        while step < CITY_HALF - 26.0 {
            out.push([*line + STREET_HALF_WIDTH + SIDEWALK_WIDTH * 0.55, step]);
            out.push([step, *line - STREET_HALF_WIDTH - SIDEWALK_WIDTH * 0.55]);
            step += 24.0;
        }
    }
    // 街区内院:每区两三棵,让内部空地不是一块死板。
    for (bi, block) in block_layouts().into_iter().enumerate() {
        for k in 0..2 {
            let angle: f32 = (k as f32 * 2.2) + bi as f32 * 0.7;
            out.push([block.cx + angle.sin() * 5.0, block.cz + angle.cos() * 5.0]);
        }
    }
    out
}

/// 程序化生成路边停放的车辆。
///
/// # Returns
///
/// - `Vec<Placement>` - `(资产, 位置, 朝向)`。
fn build_city_vehicles() -> Vec<Placement> {
    let mut out: Vec<(&'static str, [f32; 3], f32)> = Vec::new();
    let lane: f32 = STREET_HALF_WIDTH - 2.6;
    let mut seed: u32 = 0xABCD_1234;
    for line in STREET_LINES {
        let mut step: f32 = -CITY_HALF + 40.0;
        while step < CITY_HALF - 40.0 {
            seed = hash2(seed, 0x21);
            if !hash2(seed, 1).is_multiple_of(3) {
                step += 44.0;
                continue;
            }
            let along_x: bool = hash2(seed, 2).is_multiple_of(2);
            let side: f32 = if hash2(seed, 3).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            let position: [f32; 3] = if along_x {
                [step, 0.0, *line + side * lane]
            } else {
                [*line + side * lane, 0.0, step]
            };
            out.push((
                VEHICLE_POOL[hash2(seed, 4) as usize % VEHICLE_POOL.len()],
                position,
                if along_x {
                    0.0
                } else {
                    std::f32::consts::FRAC_PI_2
                },
            ));
            step += 44.0;
        }
    }
    out
}

/// 程序化生成霓虹招牌(贴在临街楼的外墙上,夜里点亮)。
///
/// # Returns
///
/// - `Vec<Placement>` - `(资产, 位置, 朝向)`。
fn build_city_signs() -> Vec<Placement> {
    let mut out: Vec<(&'static str, [f32; 3], f32)> = Vec::new();
    let face: f32 = STREET_HALF_WIDTH + SIDEWALK_WIDTH + 3.2;
    let mut seed: u32 = 0x5151_5151;
    for line in STREET_LINES {
        let mut step: f32 = -CITY_HALF + 34.0;
        while step < CITY_HALF - 34.0 {
            seed = hash2(seed, 0x31);
            if hash2(seed, 1).is_multiple_of(2) {
                out.push((
                    SIGN_POOL[hash2(seed, 2) as usize % SIGN_POOL.len()],
                    [*line - face, 4.2, step],
                    -std::f32::consts::FRAC_PI_2,
                ));
            } else {
                out.push((
                    SIGN_POOL[hash2(seed, 2) as usize % SIGN_POOL.len()],
                    [step, 3.8, *line + face],
                    std::f32::consts::FRAC_PI_2,
                ));
            }
            step += 52.0;
        }
    }
    out
}

/// 程序化生成行人点缀。
///
/// # Returns
///
/// - `Vec<Placement>` - `(资产, 位置, 朝向)`。
fn build_city_peds() -> Vec<Placement> {
    let mut out: Vec<(&'static str, [f32; 3], f32)> = Vec::new();
    let mut seed: u32 = 0x7777_7777;
    for line in STREET_LINES {
        let mut step: f32 = -CITY_HALF + 20.0;
        while step < CITY_HALF - 20.0 {
            seed = hash2(seed, 0x41);
            out.push((
                PED_POOL[hash2(seed, 1) as usize % PED_POOL.len()],
                [
                    *line + (hash_unit(seed, 2) - 0.5) * (2.0 * STREET_HALF_WIDTH - 2.0),
                    0.0,
                    step,
                ],
                hash_unit(seed, 3) * 6.28,
            ));
            step += 19.0;
        }
    }
    out
}

// ===========================================================================
// 游戏状态
// ===========================================================================

/// `DayPhase` 的默认相位(正午)。
const DEFAULT_PHASE: DayPhase = DayPhase::Noon;

/// 按键的按住状态。
pub(crate) struct InputState {
    /// W / A / S / D / 方向键。
    keys: HashMap<String, bool>,
    /// 当前昼夜相位。
    phase: DayPhase,
    /// 鼠标是否按下拖拽。
    dragging: bool,
    /// 上一次指针位置。
    last_pointer: [f64; 2],
    /// 双指上一次的距离(用于捏合缩放)。
    last_pinch: f64,
}

impl Default for InputState {
    /// default。
    fn default() -> Self {
        Self {
            keys: HashMap::new(),
            phase: DEFAULT_PHASE,
            dragging: false,
            last_pointer: [0.0, 0.0],
            last_pinch: 0.0,
        }
    }
}

impl InputState {
    /// 某个键是否按住(按 `KeyboardEvent.code` 判定,兼容所有键盘布局)。
    ///
    /// # Arguments
    ///
    /// - `&str` - str 的只读引用。
    ///
    /// # Returns
    ///
    /// - `bool` - 判定结果。
    fn held(&self, code: &str) -> bool {
        self.get_keys().get(code).copied().unwrap_or(false)
    }

    /// 当前按下的按键集合的只读视图。
    ///
    /// # Returns
    ///
    /// - `&HashMap<String, bool>` - 按键按下状态表。
    pub fn get_keys(&self) -> &HashMap<String, bool> {
        &self.keys
    }
}

/// 一份完整的游戏运行时状态。
///
/// 全部包在 `Rc<RefCell<..>>` 里供 RAF 闭包与事件闭包共享;
/// 闭包内部遵循「先取快照再短暂 borrow_mut」的写法,避免 RefCell 重叠借用。
pub struct Game {
    /// canvas 元素。
    pub canvas: HtmlCanvasElement,
    /// 相机。
    pub camera: Camera,
    /// 场景(资产 + 批次)。
    pub scene: Scene,
    /// 输入状态。
    pub input: InputState,
    /// 渲染后端。
    pub renderer: Option<Renderer>,
    /// 已加载的资产数量。
    pub loaded_assets: usize,
    /// 总资产数量。
    pub total_assets: usize,
    /// 已渲染帧数(测试用)。
    pub frame_count: u64,
    /// 已完成的固定步数(测试用)。
    pub ticks: u64,
    /// 加载失败信息(有值时显示在遮罩上)。
    pub load_error: Option<String>,
    /// 后备方案是否启用(资产 fetch 全部失败)。
    pub using_fallback: bool,
    /// 固定步长累加器(秒)。
    pub accumulator: f32,
    /// 上一帧的 `requestAnimationFrame` 时间戳(秒)。
    pub frame_time: f32,
    /// 验收探针:几个已知世界点投到屏幕上的位置。
    /// 离玩家最近的一栋楼的中心(x, z)。
    ///
    /// 验收脚本拿它当「推挤目标」:街区布局是程序化生成的,写死一个
    /// 魔法坐标早晚会推到空地上,所以在生成碰撞世界时顺手算一次存下。
    pub nearest_building: Option<[f32; 2]>,
    /// 当前帧的画布像素尺寸(调试通道用来报角色在屏幕上的像素坐标)。
    pub canvas_size: Cell<(u32, u32)>,
    pub probe: Vec<Option<[f32; 2]>>,
    /// 玩家状态(位置 / 朝向 / 步态 / 生命值 / 拾取)。
    pub player: Player,
    /// 车队与拾取物。
    pub traffic: Traffic,
    /// 静态碰撞世界(玩家圆 vs 建筑 / 车 / 道具 + 世界边界)。
    pub world: CollisionWorld,
    /// 玩家骨架每个 part 的批次索引(与 `PED_SUIT` 资产 part 一一对应)。
    pub player_batches: Vec<PlayerLimbBatch>,
    /// 车队车辆每个批次对应的车辆索引。
    pub car_batches: Vec<usize>,
    /// 拾取物每个批次对应的拾取物索引。
    pub pickup_batches: Vec<usize>,
    /// 是否处于第三人称跟随模式(Tab 切回自由观察)。
    pub third_person: bool,
    /// 第三人称跟随焦点的阻尼插值后的世界坐标。
    pub follow_target: Vec3,
    /// 资产 id → 原始包围盒(碰撞世界推导的唯一来源)。
    pub asset_bounds: HashMap<String, Bounds>,
}

/// 玩家骨架的一个 part 批次:part 名 + 枢轴 + 批次索引。
#[derive(Clone, Debug)]
pub struct PlayerLimbBatch {
    /// part 名(与资产 JSON 的 `name` 一致)。
    pub part: String,
    /// 关节枢轴的资产本地坐标。
    pub pivot: Vec3,
    /// 该 part 在场景批次表里的索引。
    pub batch: usize,
}

/// 把相机摆到「街区网格全景」机位。
///
/// 城市扩到 300 m × 300 m 之后,默认机位必须**站得够高、拉得够远**,
/// 才能一眼看到成片的街区网格而不是一条街。机位落在
/// `(x≈0, y≈120, z≈210)`、俯角 ~0.55 rad:既越过绝大多数中层楼的
/// 屋顶,又保留街道的透视纵深。
///
/// # Arguments
///
/// - `&mut Camera` - Camera 的可变引用。
fn apply_default_view(camera: &mut Camera) {
    camera.target = [0.0, 8.0, -30.0];
    camera.distance = 250.0;
    camera.desired_distance = 250.0;
    camera.yaw = 0.0;
    camera.pitch = 0.55;
    camera.fov_y = std::f32::consts::FRAC_PI_4; // 45°:比 60° 收敛,避免近处楼被拉成大楔形
    camera.far = 900.0;
}

/// 挂载到 canvas 上的事件驱动所需的共享引用。
#[derive(Clone)]
struct GameHandles {
    game: Rc<RefCell<Game>>,
    window: Window,
    canvas: HtmlCanvasElement,
    hud: Option<Element>,
    phase_label: Option<Element>,
    phase_slider: Option<HtmlInputElement>,
}

// ===========================================================================
// 程序化地面网格
// ===========================================================================

/// 展开中的程序化地面缓冲。
///
/// `build_ground` 逐步把顶点 / 面 / 逐面颜色写进这三个缓冲,最后交给
/// `expand_asset` 之外的同一套顶点格式。打包成结构体是为了让
/// `push_quad` 的参数个数落在 clippy 的阈值内(7 个)。
struct QuadBuffers<'a> {
    /// 顶点缓冲(每顶点 3 个 f32)。
    positions: &'a mut Vec<[f32; 3]>,
    /// 面索引缓冲。
    faces: &'a mut Vec<[usize; 3]>,
    /// 逐面颜色缓冲。
    face_colors: &'a mut Vec<[f32; 3]>,
}

/// 以逆时针(从 +Y 俯视)顺序向展开缓冲推入一个水平矩形。
///
/// # Arguments
///
/// - `&mut QuadBuffers` - 展开缓冲(顶点 / 面 / 逐面颜色)。
/// - `f32` - X 下界。
/// - `f32` - X 上界。
/// - `f32` - Z 下界。
/// - `f32` - Z 上界。
/// - `f32` - 高度 Y。
/// - `[f32; 3]` - 面颜色。
fn push_quad(
    buffers: &mut QuadBuffers,
    x0: f32,
    x1: f32,
    z0: f32,
    z1: f32,
    y: f32,
    color: [f32; 3],
) {
    let base: usize = buffers.positions.len();
    buffers.positions.push([x0, y, z0]);
    buffers.positions.push([x1, y, z0]);
    buffers.positions.push([x1, y, z1]);
    buffers.positions.push([x0, y, z1]);
    buffers.faces.push([base, base + 2, base + 1]);
    buffers.faces.push([base, base + 3, base + 2]);
    buffers.face_colors.push(color);
    buffers.face_colors.push(color);
}

/// 生成程序化地面:整城 300 m × 300 m 的沥青网格 + 人行道 + 车道线
/// + 斑马线 + 路口。
///
/// 用顶点色(`face_colors`)而不是贴图 —— 走的是 mesh.rs 的 de-index 展开,
/// 因此地面上每条车道线都是一个真正的三角形,没有任何纹理资源。
///
/// 地面按 `GROUND_SUBDIV × GROUND_SUBDIV` 的格铺满整城,每一格根据
/// [`on_roadway`] 判成「沥青 / 人行道 / 地块底色」三种之一,再在街道
/// 网格上叠车道虚线、中央双黄线、路口斑马线。这样网格街道与地块
/// 是同一张网格上的不同着色,不会出现「楼大了地面还是一块板」。
///
/// # Returns
///
/// - `MeshAsset` - 展开后的地面网格资产。
pub fn build_ground() -> MeshAsset {
    const ROAD: [f32; 3] = [0.085, 0.085, 0.098];
    const ROAD_ALT: [f32; 3] = [0.100, 0.100, 0.114];
    const SIDEWALK: [f32; 3] = [0.44, 0.42, 0.40];
    const SIDEWALK_EDGE: [f32; 3] = [0.36, 0.345, 0.335];
    const CURB: [f32; 3] = [0.62, 0.60, 0.57];
    const LOT_GROUND: [f32; 3] = [0.145, 0.155, 0.135];
    const LOT_ALT: [f32; 3] = [0.165, 0.175, 0.150];
    const LINE_YELLOW: [f32; 3] = [0.92, 0.70, 0.10];
    const LINE_WHITE: [f32; 3] = [0.82, 0.82, 0.80];
    const CROSSWALK: [f32; 3] = [0.88, 0.88, 0.85];

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut faces: Vec<[usize; 3]> = Vec::new();
    let mut face_colors: Vec<[f32; 3]> = Vec::new();
    let span: f32 = 2.0 * CITY_HALF;
    let cell: f32 = span / GROUND_SUBDIV as f32;

    // ---- 1) 底面:整城铺满,按「沥青 / 人行道 / 地块」三态着色 ----
    for i in 0..GROUND_SUBDIV {
        for j in 0..GROUND_SUBDIV {
            let x0: f32 = -CITY_HALF + cell * j as f32;
            let x1: f32 = x0 + cell;
            let z0: f32 = -CITY_HALF + cell * i as f32;
            let z1: f32 = z0 + cell;
            let cx: f32 = (x0 + x1) * 0.5;
            let cz: f32 = (z0 + z1) * 0.5;
            let edge_x: f32 = STREET_HALF_WIDTH + SIDEWALK_WIDTH;
            // 距最近街道轴线的距离。
            let mut dist: f32 = f32::MAX;
            for line in STREET_LINES {
                dist = dist.min((cx - *line).abs()).min((cz - *line).abs());
            }
            let color: [f32; 3] = if dist <= STREET_HALF_WIDTH {
                if (i + j) % 2 == 0 { ROAD } else { ROAD_ALT }
            } else if dist <= edge_x {
                if i % 4 == 0 { SIDEWALK_EDGE } else { SIDEWALK }
            } else if (i + j) % 2 == 0 {
                LOT_GROUND
            } else {
                LOT_ALT
            };
            let y: f32 = if dist <= STREET_HALF_WIDTH {
                0.0
            } else if dist <= edge_x {
                0.14
            } else {
                0.16
            };
            push_quad(
                &mut QuadBuffers {
                    positions: &mut positions,
                    faces: &mut faces,
                    face_colors: &mut face_colors,
                },
                x0,
                x1,
                z0,
                z1,
                y,
                color,
            );
        }
    }

    // ---- 2) 路缘石:每条街道两侧各一条竖直面 ----
    for line in STREET_LINES {
        for axis in 0..2 {
            for side in [-1.0f32, 1.0f32] {
                let inner: f32 = *line + side * STREET_HALF_WIDTH;
                let outer: f32 = inner + side * 0.24;
                let mut t: f32 = -CITY_HALF;
                while t < CITY_HALF - cell {
                    let t1: f32 = t + cell;
                    let (x0, x1, z0, z1): (f32, f32, f32, f32) = if axis == 0 {
                        (inner.min(outer), inner.max(outer), t, t1)
                    } else {
                        (t, t1, inner.min(outer), inner.max(outer))
                    };
                    push_quad(
                        &mut QuadBuffers {
                            positions: &mut positions,
                            faces: &mut faces,
                            face_colors: &mut face_colors,
                        },
                        x0,
                        x1,
                        z0,
                        z1,
                        0.07,
                        CURB,
                    );
                    t += cell;
                }
            }
        }
    }

    // ---- 3) 中央双黄线 + 车道虚线(沿每条街道,路口处断开)----
    let dash_pitch: f32 = 6.0;
    let dash_len: f32 = 3.0;
    for line in STREET_LINES {
        let mut t: f32 = -CITY_HALF;
        while t < CITY_HALF {
            // 路口范围内不画线。
            let in_junction: bool = STREET_LINES
                .iter()
                .any(|other: &f32| (t - *other).abs() <= STREET_HALF_WIDTH + 1.0);
            if !in_junction {
                let t1: f32 = t + dash_len;
                // 中央双黄线(两条 0.18 m 宽的实线,只在虚线段画满)。
                for offset in [-0.25f32, 0.25f32] {
                    let c: f32 = *line + offset;
                    push_quad(
                        &mut QuadBuffers {
                            positions: &mut positions,
                            faces: &mut faces,
                            face_colors: &mut face_colors,
                        },
                        c - 0.09,
                        c + 0.09,
                        t,
                        t1,
                        0.011,
                        LINE_YELLOW,
                    );
                }
                // 车道分隔虚线(两侧各一条)。
                for lane in [-3.5f32, 3.5f32] {
                    let c: f32 = *line + lane;
                    push_quad(
                        &mut QuadBuffers {
                            positions: &mut positions,
                            faces: &mut faces,
                            face_colors: &mut face_colors,
                        },
                        c - 0.12,
                        c + 0.12,
                        t,
                        t1,
                        0.012,
                        LINE_WHITE,
                    );
                }
            }
            t += dash_pitch;
        }
    }

    // ---- 4) 路口斑马线:每个网格路口四条,横跨每条进出街道 ----
    for line_x in STREET_LINES {
        for line_z in STREET_LINES {
            for stripe in 0..7 {
                let offset: f32 = -5.4 + stripe as f32 * 1.8;
                let w: f32 = 0.9;
                // 横跨南北向街道(x = line_x)的斑马线,贴在路口南 / 北两侧。
                for side_z in [-1.0f32, 1.0] {
                    let c: f32 = *line_z + side_z * (STREET_HALF_WIDTH + 1.0);
                    push_quad(
                        &mut QuadBuffers {
                            positions: &mut positions,
                            faces: &mut faces,
                            face_colors: &mut face_colors,
                        },
                        *line_x + offset - w * 0.5,
                        *line_x + offset + w * 0.5,
                        c - 1.4,
                        c + 1.4,
                        0.013,
                        CROSSWALK,
                    );
                }
                // 横跨东西向街道(z = line_z)的斑马线。
                for side_x in [-1.0f32, 1.0] {
                    let c: f32 = *line_x + side_x * (STREET_HALF_WIDTH + 1.0);
                    push_quad(
                        &mut QuadBuffers {
                            positions: &mut positions,
                            faces: &mut faces,
                            face_colors: &mut face_colors,
                        },
                        c - 1.4,
                        c + 1.4,
                        *line_z + offset - w * 0.5,
                        *line_z + offset + w * 0.5,
                        0.013,
                        CROSSWALK,
                    );
                }
            }
        }
    }

    let mut min: [f32; 3] = [f32::INFINITY; 3];
    let mut max: [f32; 3] = [f32::NEG_INFINITY; 3];
    for position in &positions {
        for axis in 0..3 {
            if position[axis] < min[axis] {
                min[axis] = position[axis];
            }
            if position[axis] > max[axis] {
                max[axis] = position[axis];
            }
        }
    }

    MeshAsset {
        id: GROUND_ID.to_string(),
        category: GROUND.to_string(),
        y_up: true,
        bounds: Some(crate::mesh::Bounds { min, max }),
        parts: vec![MeshPart {
            name: GROUND_PART.to_string(),
            base_color: Some(ROAD),
            emissive: None,
            positions,
            normals: None,
            faces,
            face_colors: Some(face_colors),
            flat: true,
        }],
    }
}

// ===========================================================================
// 资产加载
// ===========================================================================

/// manifest.json 的结构(只取用得到的字段)。
#[derive(serde::Deserialize)]
struct Manifest {
    /// 资产总数(manifest 自报值,HUD 里做对照)。
    #[serde(default)]
    asset_count: usize,
    /// 资产条目。
    #[serde(default)]
    assets: Vec<ManifestEntry>,
}

/// manifest 里的一个资产条目。
#[derive(Clone, serde::Deserialize)]
struct ManifestEntry {
    /// 资产 id。
    id: String,
    /// 分类(仅用于调试输出)。
    category: String,
    /// 相对 assets/ 的文件名。
    file: String,
}

/// 需要加载的资产清单(场景蓝图里引用到的全部资产)。
///
/// # Returns
///
/// - `Vec<&'static str>` - 计算结果。
fn required_asset_ids() -> Vec<&'static str> {
    let mut ids: Vec<&'static str> = Vec::new();
    for building in &build_city_buildings() {
        ids.push(building.asset);
    }
    for prop in &build_city_props() {
        ids.push(prop.asset);
    }
    ids.push(PALM_TALL);
    ids.push(PALM_SHORT);
    for (asset, _, _) in &build_city_vehicles() {
        ids.push(asset);
    }
    for (asset, _, _) in &build_city_signs() {
        ids.push(asset);
    }
    for (asset, _, _) in &build_city_peds() {
        ids.push(asset);
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// 用 `fetch` 取回一个文本资源(相对路径,适配任意部署子路径)。
///
/// # Arguments
///
/// - `&str` - str 的只读引用。
///
/// # Returns
///
/// - `Result<String, String>` - 计算结果。
async fn fetch_text(url: &str) -> Result<String, String> {
    let Some(window) = window() else {
        return Err(ERR_NO_WINDOW.to_string());
    };
    let promise: Promise = window.fetch_with_str(url);
    let value: JsValue = JsFuture::from(promise)
        .await
        .map_err(|err: JsValue| format!("fetch failed for {url}: {err:?}"))?;
    let response: Response = value
        .dyn_into::<Response>()
        .map_err(|_| format!("{url} did not return a Response"))?;
    if !response.ok() {
        return Err(format!("{url} returned HTTP {}", response.status()));
    }
    let body_promise: Promise = response
        .text()
        .map_err(|err: JsValue| format!("{url} text() threw: {err:?}"))?;
    let body: JsValue = JsFuture::from(body_promise)
        .await
        .map_err(|err: JsValue| format!("{url} body read failed: {err:?}"))?;
    body.as_string()
        .ok_or_else(|| format!("{url} body is not a string"))
}

// ===========================================================================
// 场景搭建
// ===========================================================================

/// 把一个已解析的资产加入场景(每个资产只解析一次,多个批次复用同一个 mesh)。
///
/// # Arguments
///
/// 解析一段资产 JSON 并做健全性检查(不碰场景)。
///
/// # Arguments
///
/// - `&str` - 资产 id。
/// - `&str` - 资产 JSON 原文。
///
/// # Returns
///
/// - `Result<MeshAsset, String>` - 解析出的资产;失败时给出可直接打印的信息。
fn parse_asset(id: &str, json: &str) -> Result<MeshAsset, String> {
    let asset: MeshAsset =
        serde_json::from_str(json).map_err(|err: serde_json::Error| format!("{id}: {err}"))?;
    if !asset.bounds_envelope_is_sane() {
        console_log(&format!("[vcw] {id}: bounds envelope is inverted or empty"));
    }
    if asset.category.is_empty() {
        console_log(&format!(
            "[vcw] {id}: missing category (see SCHEMA.md \u{a7}5)"
        ));
    }
    Ok(asset)
}

/// 把一个已解析的资产展开成 GPU 网格并挂进场景。
///
/// # Arguments
///
/// - `&mut Scene` - Scene 的可变引用。
/// - `&mut HashMap<String, usize>` - 资产 id → 场景 mesh 索引。
/// - `&MeshAsset` - 已解析的资产。
///
/// # Returns
///
/// - `Result<usize, String>` - 该资产在 `scene.meshes` 里的索引。
fn push_parsed_asset(
    scene: &mut Scene,
    index_map: &mut HashMap<String, usize>,
    asset: &MeshAsset,
) -> Result<usize, String> {
    let id: &str = &asset.id;
    if let Some(existing) = index_map.get(id) {
        return Ok(*existing);
    }
    // 汇总每个三角形的自发光颜色(与 expand_asset 的 part 遍历顺序一致)。
    let mut part_emissive: Vec<[f32; 3]> = Vec::with_capacity(
        asset
            .parts
            .iter()
            .map(|part: &MeshPart| part.faces.len())
            .sum(),
    );
    for part in &asset.parts {
        let emissive: [f32; 3] = part.emissive.unwrap_or([0.0; 3]);
        for _ in &part.faces {
            part_emissive.push(emissive);
        }
    }
    let mesh: GpuMesh = expand_asset(asset).map_err(|err: MeshError| format!("{id}: {err}"))?;
    let gpu: MeshAssetGpu = build_gpu_mesh(&mesh, &part_emissive);
    let index: usize = scene.push_mesh(gpu);
    index_map.insert(id.to_string(), index);
    Ok(index)
}

/// 内置回退场景:assets/ 全部加载失败时使用,保证画面非空。
///
/// # Returns
///
/// - `Scene` - 计算结果。
fn build_fallback_scene() -> Scene {
    let mut scene: Scene = Scene::default();
    let ground: MeshAsset = build_ground();
    let mut part_emissive: Vec<[f32; 3]> = Vec::with_capacity(
        ground
            .parts
            .iter()
            .map(|part: &MeshPart| part.faces.len())
            .sum(),
    );
    for part in &ground.parts {
        let emissive: [f32; 3] = part.emissive.unwrap_or([0.0; 3]);
        for _ in &part.faces {
            part_emissive.push(emissive);
        }
    }
    let mesh: crate::mesh::GpuMesh = expand_asset(&ground).expect(EXPECT_GROUND);
    let ground_index: usize = scene.push_mesh(build_gpu_mesh(&mesh, &part_emissive));
    let ground_batch: usize = scene.push_batch(ground_index, true);
    scene.push_instance(
        ground_batch,
        Instance::new([0.0, 0.0, 0.0], 0.0, 1.0, [1.0, 1.0, 1.0]),
    );

    // 一排彩色立方体当「建筑」,每面自带颜色。
    for (index, (x, z, color)) in [
        (-18.0f32, -20.0f32, [0.95, 0.45, 0.62]),
        (-20.0, 0.0, [0.42, 0.82, 0.78]),
        (-19.0, 20.0, [0.98, 0.82, 0.50]),
        (19.0, -20.0, [0.62, 0.72, 0.95]),
        (20.0, 0.0, [0.90, 0.60, 0.85]),
        (19.0, 20.0, [0.55, 0.88, 0.62]),
    ]
    .into_iter()
    .enumerate()
    {
        let height: f32 = 8.0 + index as f32 * 3.5;
        let part: MeshPart = crate::mesh::cube_part(
            [x, height * 0.5, z],
            if x < 0.0 { height * 0.5 } else { 4.0 },
            color,
            FALLBACK_BLOCK,
        );
        let mut emissive: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; part.faces.len()];
        // 顶部一条自发光带,夜里当霓虹。
        for face in emissive.iter_mut().take(2) {
            *face = [0.9, 0.25, 0.75];
        }
        let asset: MeshAsset = MeshAsset {
            id: format!("fallback_block_{index}"),
            category: FALLBACK.to_string(),
            y_up: true,
            bounds: None,
            parts: vec![part],
        };
        let mesh: crate::mesh::GpuMesh = expand_asset(&asset).expect(EXPECT_FALLBACK_BLOCK);
        let mesh_index: usize = scene.push_mesh(build_gpu_mesh(&mesh, &emissive));
        let batch: usize = scene.push_batch(mesh_index, true);
        scene.push_instance(batch, Instance::new([x, 0.0, z], 0.0, 1.0, [1.0, 1.0, 1.0]));
    }
    scene
}

/// 解析 URL 上的 `?hide=3,7,11`,返回要从场景里剔除的批次索引。
///
/// 这是一个**调试开关**:正常访问时 `location.search` 里没有 `hide=`,
/// 返回空列表,渲染循环的行为完全不受影响。加它是因为街区场景一旦
/// 出现「某块几何体画错」,靠肉眼很难定位到底是哪个批次 ——
/// 逐个隐藏即可二分定位。例:`http://localhost:8765/?hide=11`。
///
/// # Returns
///
/// - `Vec<usize>` - 要隐藏的批次索引列表。
pub fn hidden_batches() -> Vec<usize> {
    window()
        .and_then(|w: Window| w.location().search().ok())
        .map(|q: String| q.trim_start_matches('?').to_string())
        .and_then(|q: String| {
            // 扫**全部** `k=v` 对找 `hide`,而不是只看第一个参数 ——
            // 否则 `?cb=123&hide=1,2` 这种带缓存戳的验收链接会静默失效,
            // 批次没被隐藏,截图看起来"没变化",很容易误判成渲染 bug。
            q.split('&')
                .filter_map(|kv: &str| {
                    kv.split_once('=')
                        .map(|(k, v): (&str, &str)| (k.to_string(), v.to_string()))
                })
                .find(|(k, _): &(String, String)| k == HIDE)
                .map(|(_, v): (String, String)| v)
        })
        .unwrap_or_default()
        .split(',')
        .filter_map(|v: &str| v.trim().parse::<usize>().ok())
        .collect()
}

/// 读一个数字型 query 参数(`?fps=20` 这类),缺省或非法时退回 `fallback`。
///
/// # Arguments
///
/// - `&str` - 参数名(不含 `?` 和 `=`)。
/// - `f32` - 缺省 / 解析失败时用的值。
///
/// # Returns
///
/// - `f32` - 解析出的数值。
fn query_number(key: &str, fallback: f32) -> f32 {
    let search: Option<String> = window().and_then(|w: Window| w.location().search().ok());
    let Some(query): Option<String> = search else {
        return fallback;
    };
    for pair in query.trim_start_matches('?').split('&') {
        let Some((name, value)): Option<(&str, &str)> = pair.split_once('=') else {
            continue;
        };
        if name == key {
            return value.parse::<f32>().unwrap_or(fallback);
        }
    }
    fallback
}

/// 按蓝图把已加载的资产铺成场景批次。
///
/// # Arguments
///
/// - `&mut Scene` - Scene 的可变引用。
/// - `&HashMap<String, usize>` - HashMap<String, usize> 的只读引用。
fn build_scene(scene: &mut Scene, index_map: &HashMap<String, usize>) {
    // 地面:程序化网格,走同一条展开管线。
    let ground: MeshAsset = build_ground();
    let mut ground_emissive: Vec<[f32; 3]> = Vec::with_capacity(
        ground
            .parts
            .iter()
            .map(|part: &MeshPart| part.faces.len())
            .sum(),
    );
    for part in &ground.parts {
        let emissive: [f32; 3] = part.emissive.unwrap_or([0.0; 3]);
        for _ in &part.faces {
            ground_emissive.push(emissive);
        }
    }
    let ground_mesh: crate::mesh::GpuMesh = expand_asset(&ground).expect(EXPECT_GROUND);
    let ground_index: usize = scene.push_mesh(build_gpu_mesh(&ground_mesh, &ground_emissive));

    let ground_batch: usize = scene.push_batch(ground_index, true);
    scene.push_instance(
        ground_batch,
        Instance::new([0.0, 0.0, 0.0], 0.0, 1.0, [1.0, 1.0, 1.0]),
    );

    // 建筑:程序化生成的网格街区围合。同类资产合批后每批次一次
    // instanced draw call,顶点数据仍然只解析一次。
    for building in &build_city_buildings() {
        let Some(&mesh_index) = index_map.get(building.asset) else {
            continue;
        };
        let batch: usize = find_or_create_batch(scene, mesh_index);
        scene.push_instance(
            batch,
            Instance::new(
                [building.position[0], 0.0, building.position[1]],
                building.yaw,
                building.scale,
                building.tint,
            ),
        );
    }

    // 街道道具(含每个路口四角的交通灯):按资产分组 → 天然 instancing。
    for prop in &build_city_props() {
        let Some(&mesh_index) = index_map.get(prop.asset) else {
            continue;
        };
        let batch: usize = find_or_create_batch(scene, mesh_index);
        scene.push_instance(
            batch,
            Instance::new(prop.position, prop.yaw, 1.0, prop.tint),
        );
    }

    // 棕榈:两个品种交替,每个品种一个批次。
    for (index, position) in build_city_palms().iter().enumerate() {
        let asset: &str = if index % 3 == 0 {
            PALM_TALL
        } else {
            PALM_SHORT
        };
        let Some(&mesh_index) = index_map.get(asset) else {
            continue;
        };
        let batch: usize = find_or_create_batch(scene, mesh_index);
        let scale: f32 = 0.9 + (index % 3) as f32 * 0.08;
        scene.push_instance(
            batch,
            Instance::new(
                [position[0], 0.0, position[1]],
                (index as f32) * 0.7,
                scale,
                [0.97, 1.0, 0.95],
            ),
        );
    }

    // 路边停放的车辆。
    // 车辆**不在这里**摆放 —— 它们由交通系统(`spawn_player_traffic_pickups`)
    // 在车道上循环行驶,每帧由 `sync_dynamic_instances` 写回实例。
    // 如果这里再摆一套静态摆件车,同一辆车会出现两个实例,而且静态那批
    // 还会在车道中间形成看不见的碰撞体,把动态车队顶死。

    // 霓虹招牌:挂在临街楼的外墙上,自发光在夜里点亮。
    for (asset, position, yaw) in &build_city_signs() {
        let Some(&mesh_index) = index_map.get(*asset) else {
            continue;
        };
        let batch: usize = find_or_create_batch(scene, mesh_index);
        scene.push_instance(batch, Instance::new(*position, *yaw, 1.0, [1.0, 1.0, 1.0]));
    }

    // 行人点缀。
    for (asset, position, yaw) in &build_city_peds() {
        let Some(&mesh_index) = index_map.get(*asset) else {
            continue;
        };
        let batch: usize = find_or_create_batch(scene, mesh_index);
        scene.push_instance(batch, Instance::new(*position, *yaw, 1.0, [1.0, 1.0, 1.0]));
    }
}

/// 找到某个 mesh 已有的批次,没有就新建 —— 这就是 instancing 分组的关键。
///
/// # Arguments
///
/// - `&mut Scene` - Scene 的可变引用。
/// - `usize` - 输入值。
///
/// # Returns
///
/// - `usize` - 计数结果。
fn find_or_create_batch(scene: &mut Scene, mesh_index: usize) -> usize {
    if let Some(position) = scene
        .batches
        .iter()
        .position(|batch: &SceneBatch| batch.mesh_index == mesh_index)
    {
        return position;
    }
    scene.push_batch(mesh_index, true)
}

// ===========================================================================
// 碰撞世界:从资产 bounds 自动推导
// ===========================================================================

/// 按场景蓝图把静态碰撞体铺进碰撞世界。
///
/// 建筑 / 车辆 → AABB(从资产 `bounds` 按 yaw 旋转足迹取外接盒);棕榈 /
/// 垃圾桶 / 消防栓 / 交通锥 → 圆(从 `bounds` 取足迹短边的一半)。
///
/// **全部由资产包围盒推导** —— 场景蓝图只给位置 + yaw + scale,碰撞体跟着
/// 资产自动变,不存在第二份手写的魔法坐标表(见 §1.3c 与模块文档)。
///
/// # Arguments
///
/// - `&mut CollisionWorld` - 碰撞世界。
/// - `&HashMap<String, Bounds>` - 资产 id → 该资产声明的包围盒。
fn build_collision_world(world: &mut CollisionWorld, bounds_map: &HashMap<String, Bounds>) {
    world.get_shapes_mut().clear();
    world.set_player_radius(PLAYER_RADIUS);
    world.set_half_extent([WORLD_HALF, WORLD_HALF]);
    for building in build_city_buildings() {
        push_box(
            world,
            bounds_map,
            building.asset,
            [building.position[0], 0.0, building.position[1]],
            building.yaw,
            building.scale,
        );
    }
    for prop in build_city_props() {
        push_box(
            world,
            bounds_map,
            prop.asset,
            [prop.position[0], 0.0, prop.position[1]],
            prop.yaw,
            // 小道具的 AABB 收窄:长椅 / 垃圾桶 / 消防栓 / 报摊 / 电话亭的
            // 原始包围盒比它们实际占的地方大不少,全量推进去会把人行道
            // 和车道边缘堵死,玩家和车都过不去。
            prop_collider_scale(prop.asset),
        );
    }
    for (index, spot) in build_city_palms().iter().enumerate() {
        // 棕榈的碰撞体是**树干**,不是树冠:资产包围盒的 XZ 最大跨度是
        // 展开的叶子(3 m+),拿它当碰撞半径会在车道中间立一圈看不见的
        // 树桩墙。树干半径固定,只挡人不挡车。
        // 车道中心线两侧 `LANE_CLEAR_MARGIN` 米内不放碰撞体:动态车队就
        // 贴着 26.5 / 33.5 跑,树干落在车道里会被反复顶一下。
        if on_lane(spot[0]) || on_lane(spot[1]) {
            continue;
        }
        let scale: f32 = 0.9 + (index % 3) as f32 * 0.08;
        world.push_circle([spot[0], spot[1]], PALM_TRUNK_RADIUS * scale);
    }
    // 注意:**不**把 `build_city_vehicles()` 的静态摆件车放进碰撞世界。
    // 那些车已经��� `build_scene` 里被交通系统的动态车辆取代了(同一批
    // 实例由 `sync_dynamic_instances` 每帧写入)。留着它们会在车道中间放
    // 一堵看不见的墙,动态车队每帧撞上去、速度被清零,看起来就是
    // 「车不动」—— 而且它们的位置是静态的,车道因此被封死。
}

/// 小型路边道具的碰撞体缩放系数。
///
/// 交通灯和路灯是实心的,按 1.0 用;长椅 / 垃圾桶 / 消防栓 / 报摊 /
/// 电话亭 / 锥桶的资产包围盒比实物大(为了把倾斜的枝干也算进去),直接
/// 1.0 推进去会让它们旁边的车道被隐形墙堵住。
///
/// # Arguments
///
/// - `&str` - 资产 id。
///
/// # Returns
///
/// - `f32` - 碰撞体线性缩放。
fn prop_collider_scale(asset: &str) -> f32 {
    match asset {
        PROP_BENCH | PROP_TRASH_BIN | PROP_FIRE_HYDRANT | PROP_NEWSSTAND | PROP_PHONE_BOOTH => {
            SMALL_PROP_COLLIDER_SCALE
        }
        PROP_TRAFFIC_CONE => CONE_COLLIDER_SCALE,
        _ => FULL_PROP_COLLIDER_SCALE,
    }
}

/// 往碰撞世界里推一个 AABB 摆放实例(建筑 / 车辆 / 方块道具)。
///
/// # Arguments
///
/// - `&mut CollisionWorld` - 碰撞世界。
/// - `&HashMap<String, Bounds>` - 资产 id → 包围盒。
/// - `&str` - 资产 id。
/// - `Vec3` - 摆放位置(世界坐标)。
/// - `f32` - 绕 Y 轴的朝向(弧度)。
/// - `f32` - 统一缩放。
fn push_box(
    world: &mut CollisionWorld,
    bounds_map: &HashMap<String, Bounds>,
    asset: &str,
    position: Vec3,
    yaw: f32,
    scale: f32,
) {
    let (min, max) = local_bounds(bounds_map, asset);
    let (center, half): (Vec2, Vec2) = placement_box(min, max, yaw, scale, position);
    world.push_aabb(center, half);
}
/// 取一个资产声明的本地包围盒,缺省时退化成 1 m 见方。
///
/// # Arguments
///
/// - `&HashMap<String, Bounds>` - 资产 id → 包围盒。
/// - `&str` - 资产 id。
///
/// # Returns
///
/// - `(Vec3, Vec3)` - 包围盒的 `(min, max)`。
fn local_bounds(bounds_map: &HashMap<String, Bounds>, asset: &str) -> (Vec3, Vec3) {
    match bounds_map.get(asset) {
        Some(bounds) => (bounds.get_min(), bounds.get_max()),
        None => ([-0.5, 0.0, -0.5], [0.5, 2.0, 0.5]),
    }
}

// ===========================================================================
// 交通 AI + 拾取物蓝图
// ===========================================================================

/// 车道中心相对街道中轴线的横向偏移(米)。
///
/// 路面半宽 `STREET_HALF_WIDTH` = 7 m,双向车道各占一半,车道中心落在
/// ±3.5 m 处 —— 正好压在程序化地面画的车道虚线上。
const LANE_OFFSET_X: f32 = 3.5;
/// 车队循环轨道的半长(米):比城市半边长 [`CITY_HALF`] 小,车到端点 wrap。
const TRAFFIC_HALF: f32 = CITY_HALF - 30.0;

/// 实心大道具(交通灯 / 路灯)碰撞体不缩放。
const FULL_PROP_COLLIDER_SCALE: f32 = 1.0;
/// 小型路边道具碰撞体缩到 55%(见 [`prop_collider_scale`] 的原因)。
const SMALL_PROP_COLLIDER_SCALE: f32 = 0.55;
/// 锥桶碰撞体缩到 30%(锥桶本来就该被车碾过去,不该挡路)。
const CONE_COLLIDER_SCALE: f32 = 0.3;
/// 判定某个 X 坐标是否落在任意一条南北向街道的车道里。
///
/// # Arguments
///
/// - `f32` - 世界 X 坐标(米)。
///
/// # Returns
///
/// - `bool` - `true` 表示该 X 在某条街的车道缓冲带内。
fn on_lane(x: f32) -> bool {
    STREET_LINES.iter().any(|line: &f32| {
        let offset: f32 = (x - *line).abs();
        (offset - LANE_OFFSET_X).abs() < LANE_CLEAR_MARGIN
    })
}

/// 棕榈树干碰撞半径(米)—— 只挡人,树叶可以从中穿过。
const PALM_TRUNK_RADIUS: f32 = 0.4;
/// 车道缓冲区半宽(米):街道中轴线两侧这么多米内不放静态碰撞体。
const LANE_CLEAR_MARGIN: f32 = 1.8;

/// 南北向街道的轴线(取 `STREET_LINES` 里最靠中间的两条,再取负号补一条
/// 西侧车道),也就是车队实际会出现的 X 坐标。
///
/// 从 `STREET_LINES` **派生**而不是另抄一份数字:街道网格改了这里自动跟着
/// 改,不会出现「车开在没有路的虚空里」。
///
/// # Arguments
///
/// - `usize` - `STREET_LINES` 里的街道索引。
/// - `f32` - 车道在街道哪一侧(`-1.0` / `+1.0`)。
///
/// # Returns
///
/// - `f32` - 该街道的车道 X 坐标(米)。
fn lane_x(street_index: usize, side: f32) -> f32 {
    STREET_LINES[street_index] + side * LANE_OFFSET_X
}

/// 车队车道蓝图:`(资产, 车道 X, 起始 Z, 巡航速度, 方向)`。
///
/// 每辆车被钉在一个固定 X 上、只沿 Z 跑,到 ±[`TRAFFIC_HALF`] wrap,所以
/// 结构上不可能拐弯、也不可能开进人行道或楼里 —— 车**永远穿不过建筑**。
/// 车辆自己的朝向由 [`traffic::TrafficCar::get_yaw`] 从行驶方向推出。
const TRAFFIC_LANES: &[(&str, f32, f32, f32)] = &[
    // (资产 id, 车道 X, 初始 Z, 行驶方向 +1 / -1)
    //
    // 车道 X = 街道中轴 ± LANE_OFFSET_X(3.5 m),也就是 30 ± 3.5 = 26.5 / 33.5。
    // 出生点在路口 (30, 0),所以车会从玩家眼前开过 —— 上车测试有确定性。
    (CAR_SEDAN, 33.5, 60.0, -1.0),
    (CAR_TAXI, 26.5, -30.0, 1.0),
    (CAR_COUPE, 33.5, -60.0, -1.0),
    (TRUCK_PICKUP, 26.5, 90.0, 1.0),
    (CAR_POLICE, 33.5, 110.0, -1.0),
    (CAR_SEDAN, -26.5, -60.0, 1.0),
    (CAR_TAXI, -33.5, 30.0, -1.0),
    (CAR_COUPE, -26.5, 90.0, 1.0),
    (TRUCK_PICKUP, -33.5, -90.0, -1.0),
    (CAR_POLICE, -26.5, 100.0, 1.0),
];

/// 地面拾取物蓝图:`(资产, HUD 名, 世界坐标)`。
///
/// 全部落在 X = 30 这条街的人行道上(`30 ± (7 + 2.2)`),玩家出生点就在
/// 同一条人行道上,走几步就能捡到。`marker_flame` 是任务目标点,夜里靠
/// 自身 emissive 远远就能看见。
const PICKUP_PLACEMENTS: &[(&str, &str, Vec3)] = &[
    // 出生点 (30, 0) 在十字路口。拾取物沿两条街的人行道分布,
    // 坐标从人行道外沿推出(不落在车道缓冲带里,见 TRAFFIC_LANES)。
    (PICKUP_HEALTH_PACK, NOTICE_HEALTH, [35.4, 0.35, 36.0]),
    (WEP_SMG, NOTICE_WEAPON, [24.6, 0.35, 24.0]),
    (PICKUP_CASH_STACK, NOTICE_CASH, [24.6, 0.35, 4.0]),
    (PICKUP_ARMOR_VEST, NOTICE_ARMOR, [24.6, 0.35, -12.0]),
    (PICKUP_AMMO_BOX, NOTICE_AMMO, [35.4, 0.35, 56.0]),
    (WEP_PISTOL, NOTICE_WEAPON, [24.6, 0.35, 20.0]),
    (WEP_BAT, NOTICE_WEAPON, [35.4, 0.35, -26.0]),
    (WEP_GRENADE, NOTICE_WEAPON, [24.6, 0.35, 30.0]),
    (PICKUP_HEALTH_PACK, NOTICE_HEALTH, [35.4, 0.35, 40.0]),
    (PICKUP_AMMO_BOX, NOTICE_AMMO, [24.6, 0.35, -30.0]),
    (PICKUP_CASH_STACK, NOTICE_CASH, [35.4, 0.35, 52.0]),
    (MARKER_FLAME, NOTICE_MARKER, [26.5, 0.35, -60.0]),
];

/// 玩家骨架要渲染的 part 名(躯干 / 头 / 头发 + 四肢 + 两只鞋)。
///
/// 顺序不重要(每个 part 一个独立批次),但必须与 `ped_suit.json` 的
/// `parts[].name` 完全一致。
const PLAYER_PARTS: &[&str] = &[
    PART_TORSO,
    PART_HEAD,
    PART_HAIR,
    PART_UPPER_ARM_L,
    PART_LOWER_ARM_L,
    PART_UPPER_ARM_R,
    PART_LOWER_ARM_R,
    PART_UPPER_LEG_L,
    PART_LOWER_LEG_L,
    PART_UPPER_LEG_R,
    PART_LOWER_LEG_R,
    PART_SHOE_L,
    PART_SHOE_R,
];

/// 从 `ped_suit` 的原始 JSON 里切出单个 part 的网格 + 枢轴,并挂成场景批次。
///
/// 每个 part 独立成 mesh(顶点只在该 part 上出现一次)+ 独立批次(每帧
/// 写一条 model matrix),这样步态动画只需要改 matrix、不需要重建任何
/// 顶点数据。
///
/// # Arguments
///
/// - `&mut Scene` - 场景。
/// - `&MeshAsset` - `ped_suit` 的完整资产。
/// - `&str` - 要切出的 part 名。
///
/// # Returns
///
/// - `usize` - 新批次的索引。
fn push_player_part(scene: &mut Scene, asset: &MeshAsset, part_name: &str) -> usize {
    let Some(part) = asset
        .parts
        .iter()
        .find(|candidate: &&MeshPart| candidate.name == part_name)
    else {
        return usize::MAX;
    };
    let part: &MeshPart = part;
    let mut emissive: Vec<[f32; 3]> = Vec::with_capacity(part.faces.len());
    let base: [f32; 3] = part.emissive.unwrap_or([0.0; 3]);
    for _ in &part.faces {
        emissive.push(base);
    }
    let solo: MeshAsset = MeshAsset {
        id: asset.id.clone(),
        category: asset.category.clone(),
        y_up: asset.y_up,
        bounds: None,
        parts: vec![part.clone()],
    };
    let mesh: GpuMesh = expand_asset(&solo).expect(PLAYER_PART_EXPECT);
    let mesh_index: usize = scene.push_mesh(build_gpu_mesh(&mesh, &emissive));
    // `near_cull = false`:第三人称相机离角色只有几米,而默认的近处剔除
    // 半径是 26 m,不豁免的话角色会被整批剔掉、画面里根本看不到人。
    scene.push_batch_with_cull(mesh_index, true, false)
}

/// 玩家骨架每个 part 的本地包围盒(min / max 各一份)。
///
/// # Arguments
///
/// - `&MeshAsset` - `ped_suit` 的完整资产。
/// - `&str` - part 名。
///
/// # Returns
///
/// - `(Vec3, Vec3)` - 该 part 的本地包围盒;找不到时退化成 1 m 见方。
fn part_local_bounds(asset: &MeshAsset, part_name: &str) -> (Vec3, Vec3) {
    let Some(part) = asset
        .parts
        .iter()
        .find(|candidate: &&MeshPart| candidate.name == part_name)
    else {
        return ([-0.5, 0.0, -0.5], [0.5, 2.0, 0.5]);
    };
    let mut min: Vec3 = [f32::INFINITY; 3];
    let mut max: Vec3 = [f32::NEG_INFINITY; 3];
    for position in &part.positions {
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }
    (min, max)
}

// ===========================================================================
// 输入绑定
// ===========================================================================

/// 在 canvas 上挂上鼠标 / 触摸 / 滚轮事件。
///
/// 全部用裸 `web_sys::EventTarget::add_event_listener_with_callback` +
/// `Closure`,**不在** euv 的 hook 上下文里调用任何 hook,因此不会出现
/// 「hook context 丢失 → 白屏」。
///
/// # Arguments
///
/// - `&GameHandles` - GameHandles 的只读引用。
fn bind_pointer_events(handles: &GameHandles) {
    let canvas: &HtmlCanvasElement = &handles.canvas;

    // ---- 鼠标按下 / 拖拽 / 抬起 ----
    {
        let handles: GameHandles = handles.clone();
        let closure: Closure<dyn FnMut(Event)> = Closure::wrap(Box::new(move |event: Event| {
            let Some(mouse) = event.dyn_ref::<MouseEvent>() else {
                return;
            };
            let (x, y): (f64, f64) = (mouse.client_x() as f64, mouse.client_y() as f64);
            {
                let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
                game.input.dragging = true;
                game.input.last_pointer = [x, y];
            }
            // 拖拽时不要让页面滚动 / 选中文本。
            event.prevent_default();
        }));
        attach(canvas, EVENT_POINTERDOWN, closure);
    }
    {
        let handles: GameHandles = handles.clone();
        let closure: Closure<dyn FnMut(Event)> = Closure::wrap(Box::new(move |event: Event| {
            let Some(mouse) = event.dyn_ref::<MouseEvent>() else {
                return;
            };
            let (x, y): (f64, f64) = (mouse.client_x() as f64, mouse.client_y() as f64);
            {
                let game: std::cell::Ref<Game> = handles.game.borrow();
                if !game.input.dragging {
                    return;
                }
                let dx: f64 = x - game.input.last_pointer[0];
                let dy: f64 = y - game.input.last_pointer[1];
                drop(game);
                let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
                game.camera.yaw += dx as f32 * 0.006;
                game.camera.pitch += dy as f32 * 0.005;
                game.camera.clamp_pitch();
                game.input.last_pointer = [x, y];
            }
        }));
        attach(canvas, EVENT_POINTERMOVE, closure);
    }
    {
        let handles: GameHandles = handles.clone();
        let closure: Closure<dyn FnMut(Event)> = Closure::wrap(Box::new(move |_event: Event| {
            let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
            game.input.dragging = false;
        }));
        attach(canvas, EVENT_POINTERUP, closure);
    }
    {
        let handles: GameHandles = handles.clone();
        let closure: Closure<dyn FnMut(Event)> = Closure::wrap(Box::new(move |_event: Event| {
            let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
            game.input.dragging = false;
        }));
        attach(canvas, EVENT_POINTERCANCEL, closure);
    }

    // ---- 滚轮缩放 ----
    {
        let handles: GameHandles = handles.clone();
        let closure: Closure<dyn FnMut(Event)> = Closure::wrap(Box::new(move |event: Event| {
            let Some(wheel) = event.dyn_ref::<WheelEvent>() else {
                return;
            };
            let delta: f64 = wheel.delta_y();
            let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
            // 滚轮改的是**期望距离**,不是当前距离 —— 当前距离由遮挡
            // 回避每帧改写,直接写它会被下一帧的回避覆盖掉,滚轮就失灵。
            let zoomed: f32 =
                game.camera.get_desired_distance() * (1.0 + (delta as f32) * ZOOM_STEP);
            // 第三人称模式下滚轮只调「跟随距离」,有明确的近 / 远下限,
            // 否则玩家会一路滚到贴脸或者飞到天上。
            if game.third_person {
                game.camera
                    .set_desired_distance(zoomed.clamp(FOLLOW_DISTANCE_MIN, FOLLOW_DISTANCE_MAX));
            } else {
                game.camera.set_distance(zoomed);
                game.camera.clamp_distance();
                game.camera.confine_to_city();
            }
            drop(game);
            event.prevent_default();
        }));
        attach(canvas, EVENT_WHEEL, closure);
    }

    // ---- 单指触摸:转相机 ----
    {
        let handles: GameHandles = handles.clone();
        let closure: Closure<dyn FnMut(Event)> = Closure::wrap(Box::new(move |event: Event| {
            let Some(touch_event) = event.dyn_ref::<TouchEvent>() else {
                return;
            };
            let touches: TouchList = touch_event.touches();
            if touches.length() != 1 {
                return;
            }
            let Some(touch) = touches.item(0) else {
                return;
            };
            let (x, y): (f64, f64) = (touch.client_x() as f64, touch.client_y() as f64);
            {
                let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
                let previous: [f64; 2] = game.input.last_pointer;
                if game.input.last_pinch > 0.0 {
                    // 从双指恢复成单指:只重置基准,不跳转视角。
                    game.input.last_pointer = [x, y];
                    game.input.last_pinch = 0.0;
                    return;
                }
                let dx: f64 = x - previous[0];
                let dy: f64 = y - previous[1];
                game.camera.yaw += dx as f32 * 0.006;
                game.camera.pitch += dy as f32 * 0.005;
                game.camera.clamp_pitch();
                game.input.last_pointer = [x, y];
            }
            event.prevent_default();
        }));
        attach(canvas, EVENT_TOUCHSTART, closure);
    }
    {
        let handles: GameHandles = handles.clone();
        let closure: Closure<dyn FnMut(Event)> = Closure::wrap(Box::new(move |event: Event| {
            let Some(touch_event) = event.dyn_ref::<TouchEvent>() else {
                return;
            };
            let touches: TouchList = touch_event.touches();
            match touches.length() {
                1 => {
                    let Some(touch) = touches.item(0) else {
                        return;
                    };
                    let (x, y): (f64, f64) = (touch.client_x() as f64, touch.client_y() as f64);
                    let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
                    let previous: [f64; 2] = game.input.last_pointer;
                    let dx: f64 = x - previous[0];
                    let dy: f64 = y - previous[1];
                    game.camera.yaw += dx as f32 * 0.006;
                    game.camera.pitch += dy as f32 * 0.005;
                    game.camera.clamp_pitch();
                    game.input.last_pointer = [x, y];
                }
                2 => {
                    // 双指:捏合缩放 + 双指平移。
                    let (a, b) = match (touches.item(0), touches.item(1)) {
                        (Some(a), Some(b)) => (a, b),
                        _ => return,
                    };
                    let distance: f64 = (((a.client_x() - b.client_x()) as f64).powi(2)
                        + ((a.client_y() - b.client_y()) as f64).powi(2))
                    .sqrt();
                    let center_x: f64 = (a.client_x() + b.client_x()) as f64 * 0.5;
                    let center_y: f64 = (a.client_y() + b.client_y()) as f64 * 0.5;
                    let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
                    if game.input.last_pinch > 0.0 {
                        let previous: f64 = game.input.last_pinch;
                        let scale: f32 = (previous / distance.max(1.0)) as f32;
                        game.camera.distance = (game.camera.distance * scale).clamp(12.0, 180.0);
                        game.camera.confine_to_city();
                        // 双指平移:中心位移反向推动相机焦点。
                        let previous_center: [f64; 2] = game.input.last_pointer;
                        let forward: [f32; 3] = game.camera.forward();
                        let right: [f32; 3] = normalize3([-forward[2], 0.0, forward[0]]);
                        let pan: f32 = game.camera.distance * 0.0016;
                        for axis in 0..3 {
                            game.camera.target[axis] += (right[axis]
                                * (center_x - previous_center[0]) as f32
                                - forward[axis] * (center_y - previous_center[1]) as f32)
                                * pan;
                        }
                    }
                    game.input.last_pinch = distance;
                    game.input.last_pointer = [center_x, center_y];
                }
                _ => {
                    let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
                    game.input.last_pinch = 0.0;
                }
            }
            event.prevent_default();
        }));
        attach(canvas, EVENT_TOUCHMOVE, closure);
    }
    {
        let handles: GameHandles = handles.clone();
        let closure: Closure<dyn FnMut(Event)> = Closure::wrap(Box::new(move |_event: Event| {
            let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
            game.input.last_pinch = 0.0;
        }));
        attach(canvas, EVENT_TOUCHEND, closure);
    }
    {
        let handles: GameHandles = handles.clone();
        let closure: Closure<dyn FnMut(Event)> = Closure::wrap(Box::new(move |_event: Event| {
            let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
            game.input.last_pinch = 0.0;
        }));
        attach(canvas, EVENT_TOUCHCANCEL, closure);
    }
}

/// 在 target 上挂一个 `FnMut(Event)` 闭包,`forget()` 让它活到页面结束。
///
/// # Arguments
///
/// - `&E` - E 的只读引用。
/// - `&str` - str 的只读引用。
/// - `Closure<dyn FnMut(Event)>` - 输入值。
fn attach<E: AsRef<EventTarget>>(target: &E, name: &str, closure: Closure<dyn FnMut(Event)>) {
    let _: Result<(), euv::wasm_bindgen::JsValue> = target
        .as_ref()
        .add_event_listener_with_callback(name, closure.as_ref().unchecked_ref::<Function>());
    closure.forget();
}

/// 在 `window` 上挂键盘监听(WASD 平移 / R 重置 / T 切昼夜)。
///
/// # Arguments
///
/// - `&GameHandles` - GameHandles 的只读引用。
fn bind_keyboard(handles: &GameHandles) {
    let window: &Window = &handles.window;
    {
        let handles: GameHandles = handles.clone();
        let closure: Closure<dyn FnMut(Event)> = Closure::wrap(Box::new(move |event: Event| {
            let Some(keyboard) = event.dyn_ref::<KeyboardEvent>() else {
                return;
            };
            let code: String = keyboard.code();
            // 注意:这些键名都是 `&'static str` **常量**,不能直接写进
            // `matches!` —— 那会被当成 pattern 绑定而不是常量比较。
            // 所以统一走 `TRACKED_KEYS` 的相等扫描。
            let down: bool = TRACKED_KEYS.contains(&code.as_str());
            if !down {
                return;
            }
            let is_press: bool = {
                let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
                let was: bool = game.input.held(&code);
                game.input.keys.insert(code.clone(), true);
                !was
            };
            if !is_press {
                return;
            }
            match code.as_str() {
                KEYR => {
                    let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
                    game.camera.reset();
                    apply_default_view(&mut game.camera);
                }
                KEYT => {
                    let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
                    game.input.phase = game.input.phase.next();
                    let phase: DayPhase = game.input.phase;
                    drop(game);
                    sync_phase_ui(&handles, phase);
                }
                KEYF => {
                    let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
                    toggle_vehicle(&mut game);
                }
                KEYTAB => {
                    // 切回 / 切回第三人称。自由观察模式下相机恢复默认全景机位。
                    let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
                    game.third_person = !game.third_person;
                    if game.third_person {
                        // 期望距离和实际距离一起复位:回避可能把上一段的
                        // 缩近距离留着,切回来时必须重新从默认距离起步。
                        game.camera.set_desired_distance(FOLLOW_DISTANCE);
                        game.camera.set_distance(FOLLOW_DISTANCE);
                        game.camera.set_pitch(FOLLOW_PITCH);
                        game.follow_target = game.player.get_position();
                    } else {
                        apply_default_view(&mut game.camera);
                    }
                    let yaw: f32 = game.player.get_yaw();
                    game.camera.set_yaw(yaw);
                }
                _ => {}
            }
        }));
        attach(window, EVENT_KEYDOWN, closure);
    }
    {
        let handles: GameHandles = handles.clone();
        let closure: Closure<dyn FnMut(Event)> = Closure::wrap(Box::new(move |event: Event| {
            let Some(keyboard) = event.dyn_ref::<KeyboardEvent>() else {
                return;
            };
            let code: String = keyboard.code();
            let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
            game.input.keys.insert(code, false);
        }));
        attach(window, EVENT_KEYUP, closure);
    }
}

/// 把昼夜状态同步到 HTML overlay(标签 + 滑块)。
///
/// # Arguments
///
/// - `&GameHandles` - GameHandles 的只读引用。
/// - `DayPhase` - 输入值。
fn sync_phase_ui(handles: &GameHandles, phase: DayPhase) {
    if let Some(label) = &handles.phase_label {
        label.set_text_content(Some(phase.label()));
    }
    if let Some(slider) = &handles.phase_slider {
        let _: Result<(), euv::wasm_bindgen::JsValue> =
            slider.set_attribute(ATTR_VALUE, &format!("{}", phase.slider_value()));
    }
    let hud: String = {
        let game: std::cell::Ref<Game> = handles.game.borrow();
        format_hud(&game, 0)
    };
    if let Some(element) = &handles.hud {
        element.set_text_content(Some(&hud));
    }
    publish_debug_state(handles, &hud);
}

/// 一个 `MeshAssetGpu` 顶点数据的轴对齐包围盒(min xyz, max xyz)。
///
/// 验收探针用:资产 JSON 说 `ped_suit` 是 1.75 m 高的人,如果 GPU 缓冲里
/// 的顶点包围盒是几十米,那就是「顶点上传 / 打包」这一环写坏了,
/// 而不是相机或矩阵的问题。
///
/// 这个探针当初是为「角色糊满屏幕」写的,最后定位到的真因是
/// `WebGlRenderer` 的 GPU mesh 表被重复上传顶偏(见 `upload_mesh` 的
/// 文档),而**顶点数据本身一直是好的**。所以这里改成发布**每个骨架
/// part 的实际顶点包围盒**:`limbExtents` 里的 13 个盒应当都在
/// `ped_suit` 声明的 ±0.3 m / 0..1.75 m 量级内,一旦某个盒子冒出几十米,
/// 就说明顶点数据真的被写坏了 —— 这正是它当初想抓的状态。
///
/// # Arguments
///
/// - `&MeshAssetGpu` - MeshAssetGpu 的只读引用。
///
/// # Returns
///
/// - `MeshExtent` - `[min_x, min_y, min_z, max_x, max_y, max_z]`。
fn gpu_mesh_extent(mesh: &MeshAssetGpu) -> MeshExtent {
    let stride: usize = GPU_STRIDE_FLOATS;
    let mut out: MeshExtent = [
        f32::INFINITY,
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    for vertex in mesh.vertices.chunks_exact(stride) {
        for axis in 0..3 {
            let value: f32 = vertex[axis];
            if value < out[axis] {
                out[axis] = value;
            }
            if value > out[axis + 3] {
                out[axis + 3] = value;
            }
        }
    }
    out
}

/// 把 `Option<[f32; 2]]` 编成合法 JSON。
///
/// **`Debug`(`{:?}`) 不能直接写进 JSON。** Rust 会输出
/// `Some([49.0, 60.0])`,这不是合法 JSON;某个字段这样写,整个
/// `window.__vcw` 就会 `JSON.parse` 失败,页面看起来像"没开机",
/// 排查一次要多烧好几轮构建。所有可选数值都必须走这个函数。
///
/// # Arguments
///
/// - `Option<[f32; 2]>` - 待编码的平面点。
///
/// # Returns
///
/// - `String` - 合法 JSON:`"null"` 或 `"[x, z]"`。
fn json_point(point: Option<[f32; 2]>) -> String {
    match point {
        Some([x, z]) => format!("[{},{}]", x, z),
        None => String::from(JSON_NULL),
    }
}

/// 把一串 `f32` 编成合法 JSON 数组。
///
/// # Arguments
///
/// - `&[f32]` - 待编码的数值切片。
///
/// # Returns
///
/// - `String` - 形如 `[1, 2, 3]` 的 JSON 数组。
fn json_f32s(values: &[f32]) -> String {
    let wide: Vec<f64> = values.iter().map(|value: &f32| *value as f64).collect();
    format!("[{}]", join_f64(&wide))
}

/// 把当前运行时状态挂到 `window.__vcw`,供自动化验证读真实数值。
///
/// 这不是给玩家看的 UI,而是验收通道:用 CDP 发真实按键、然后
/// `Runtime.evaluate` 读这里,才能证明「玩家真的在动」而不是「只有相机动」。
///
/// # Arguments
///
/// - `&GameHandles` - 事件句柄。
/// - `&str` - 本帧的 HUD 文本。
fn publish_debug_state(handles: &GameHandles, hud: &str) {
    let game: std::cell::Ref<Game> = handles.game.borrow();
    let position: Vec3 = game.player.get_position();
    let car_x: Vec<f64> = game
        .traffic
        .get_cars_ref()
        .iter()
        .map(|car| f64::from(car.get_position()[0]))
        .collect();
    let car_z: Vec<f64> = game
        .traffic
        .get_cars_ref()
        .iter()
        .map(|car| f64::from(car.get_position()[2]))
        .collect();
    let car_speed: Vec<f64> = game
        .traffic
        .get_cars_ref()
        .iter()
        .map(|car| f64::from(car.get_speed()))
        .collect();
    // 第一件还没被捡走的医疗包:验收脚本读它就能走过去真的拾取,而不是
    // 猜一组魔法坐标。
    let health_pack: Option<[f64; 2]> = game
        .traffic
        .get_pickups_ref()
        .iter()
        .find(|p| !p.get_taken() && p.get_asset() == PICKUP_HEALTH_PACK)
        .map(|p| {
            let at: Vec3 = p.get_position();
            [f64::from(at[0]), f64::from(at[2])]
        });
    // 正在开的那辆车的实时坐标 / 速度:用来证明"车真的在动",而不是
    // 只有相机在动。
    let (cdrive_x, cdrive_z, cdrive_v): (f64, f64, f64) = match game.player.get_vehicle() {
        Some(index) => match game.traffic.get_cars_ref().get(index) {
            Some(car) => {
                let at: Vec3 = car.get_position();
                (
                    f64::from(at[0]),
                    f64::from(at[2]),
                    f64::from(car.get_speed()),
                )
            }
            None => (0.0, 0.0, 0.0),
        },
        None => (0.0, 0.0, 0.0),
    };
    // 相机眼点:验收脚本靠它核对「回避之后眼点到底站在哪」。
    let eye: Vec3 = game.camera.eye();
    // 探针必须序列化成**合法 JSON**:`{:?}` 打出的是 Rust 的
    // `Some([1.0, 2.0])`,在 JSON 里是非法的,整份 `window.__vcw`
    // 会因此解析失败,自动化脚本就再也读不到任何状态。
    // 验收探针:第一个不透明批次的实例矩阵与数量,用来判断「模型矩阵
    // 是不是没写进去」。
    let b0: String = match game.scene.batches.first() {
        Some(batch) => format!(
            "{:?} x{}",
            batch
                .instances
                .first()
                .map(|i: &Instance| i.get_model_ref()),
            batch.instances.len()
        ),
        None => String::from(crate::r#const::NO_BATCHES),
    };
    let bn: usize = game.scene.batches.len();
    // 最近的建筑实例位置:验收脚本要拿它当「推挤目标」,而不是写死
    // 一个魔法坐标 —— 街区布局会变,写死的坐标早晚会推到空气上。
    let bpos_json: String = json_point(game.nearest_building);
    let limbs_json: String = {
        let parts: Vec<String> = game
            .player_batches
            .iter()
            .map(|limb: &PlayerLimbBatch| {
                format!("{{\"part\":\"{}\",\"batch\":{}}}", limb.part, limb.batch)
            })
            .collect();
        format!("[{}]", parts.join(JSON_COMMA))
    };
    let limb_model: String = {
        let first: Option<&PlayerLimbBatch> = game.player_batches.first();
        let batch_index: Option<usize> = first.map(|limb: &PlayerLimbBatch| limb.batch);
        let instance: Option<&Instance> = match batch_index {
            Some(index) => game
                .scene
                .batches
                .get(index)
                .and_then(|batch: &SceneBatch| batch.instances.first()),
            None => None,
        };
        let model: Option<Mat4Data> = instance.map(|inst: &Instance| *inst.get_model_ref());
        match model {
            Some(m) => json_f32s(&m),
            None => String::from(JSON_NULL),
        }
    };
    // 角色在屏幕上的包围盒:把「脚底」和「头顶」两个世界点投影出去。
    // 有了这两个像素坐标,验收脚本能直接算出角色在画面里有多高、落在
    // 画面的哪个位置 —— 不用靠肉眼判断「看没看见」。
    let char_box: String = {
        let size: (u32, u32) = game.canvas_size.get();
        let (canvas_w, canvas_h): (f32, f32) = (size.0 as f32, size.1 as f32);
        let base: Vec3 = game.player.get_position();
        let head: Vec3 = [base[0], base[1] + PED_SCREEN_PROBE_HEIGHT, base[2]];
        let feet: Option<(f32, f32, f32)> = game.camera.world_to_screen(base, canvas_w, canvas_h);
        let crown: Option<(f32, f32, f32)> = game.camera.world_to_screen(head, canvas_w, canvas_h);
        match (feet, crown) {
            (Some(f), Some(c)) => {
                format!("[{:.1},{:.1},{:.1},{:.1},{:.1}]", f.0, f.1, c.0, c.1, c.2)
            }
            _ => String::from(JSON_NULL),
        }
    };
    let ft: Vec<f32> = game.follow_target.to_vec();

    let tgt: Vec<f32> = game.camera.get_target().to_vec();
    let ft_json: String = json_f32s(&ft);
    let tgt_json: String = json_f32s(&tgt);
    let mm: Vec<f32> = game.camera.view_projection(1.0).get_elements().to_vec();
    let mm_json: String = json_f32s(&mm);
    let probe_json: String = game
        .probe
        .iter()
        .map(|p: &Option<[f32; 2]>| match p {
            Some([x, y]) => format!("[{},{}]", x, y),
            None => String::from(JSON_NULL),
        })
        .collect::<Vec<String>>()
        .join(",");
    let json: String = format!(
        "{DEBUG_OPEN}\"playerX\":{x},\"playerZ\":{z},\"playerYaw\":{yaw},\"cameraYaw\":{cyaw},\"phase\":\"{phase}\",\"healthPack\":{hp_at},\"pickupPos\":{pickup_pos},\"playerHealth\":{hp},\"playerCash\":{cash},\"playerVehicle\":{veh},\"gaitPhase\":{gphase},\"gaitAmount\":{gamount},\"cameraMode\":\"{mode}\",\"probe\":[{probe_json}],\"charBox\":{char_box},\"b0\":\"{b0}\",\"bpos\":{bpos_json},\"limbBatches\":{limbs_json},\"limbModel\":{limb_model},\"bn\":{bn},\"ft\":{ft_json},\"tgt\":{tgt_json},\"m\":{mm_json},\"collisionShapes\":{shapes},\"playerInsideCollider\":{inside},\"cameraDist\":{cdist},\"cameraDistTarget\":{cdist_target},\"camOccluded\":{coccluded},\"camEyeX\":{ceye_x},\"camEyeY\":{ceye_y},\"camEyeZ\":{ceye_z},\"camClearance\":{cclear},\"camPitch\":{cpitch},\"carX\":[{car_x}],\"carZ\":[{car_z}],\"carSpeed\":[{car_speed}],\"pickups\":{{\"total\":{total},\"taken\":{taken}}},\"hud\":\"{hud}\",\"carDriveX\":{cdrive_x},\"carDriveZ\":{cdrive_z},\"carDriveSpeed\":{cdrive_v},\"throttle\":{throttle},\"frames\":{frames}{DEBUG_CLOSE}",
        x = position[0],
        z = position[2],
        yaw = game.player.get_yaw(),
        cyaw = game.camera.get_yaw(),
        phase = game.input.phase.as_str(),
        pickup_pos = join_pickups(&game),
        hp_at = match health_pack {
            Some(at) => format!("[{},{}]", at[0], at[1]),
            None => String::from(JSON_NULL),
        },
        hp = game.player.get_health(),
        cash = game.player.get_cash(),
        veh = match game.player.get_vehicle() {
            Some(index) => format!("{index}"),
            None => String::from(JSON_NULL),
        },
        gphase = game.player.get_gait_phase(),
        gamount = game.player.get_gait_amount(),
        mode = if game.third_person {
            HUD_THIRD_PERSON
        } else {
            HUD_ORBIT
        },
        shapes = game.world.get_shapes().len(),
        inside = game.world.contains_point([position[0], position[2]]),
        cdist = game.camera.get_distance(),
        cdist_target = game.camera.get_desired_distance(),
        coccluded = game.camera.get_occluded(),
        ceye_x = eye[0],
        ceye_y = eye[1],
        ceye_z = eye[2],
        cclear = game.camera.get_eye_clearance(),
        cpitch = game.camera.get_pitch(),
        car_x = join_f64(&car_x),
        car_z = join_f64(&car_z),
        car_speed = join_f64(&car_speed),
        total = game.traffic.get_pickups_ref().len(),
        taken = game.player.collected_count(),
        hud = hud
            .replace(CHAR_QUOTE, "")
            .replace(CHAR_BACKSLASH, JSON_SLASH),
        cdrive_x = cdrive_x,
        cdrive_z = cdrive_z,
        cdrive_v = cdrive_v,
        throttle = axis(&game.input, KEYW, KEYS),
        frames = game.frame_count,
    );
    let _ = set_window_json(DEBUG_HOOK_NAME, &json);
    publish_visibility_state(&game, &char_box);
}

/// 把「角色可见性」这一组字段挂到 `window.__VCW_DEBUG__`。
///
/// 与全量快照 [`publish_debug_state`] 分开维护:验收脚本判断「角色到底
/// 画没画、画多大、在画面哪里」只需要这几个字段,不必去 parse 一整份
/// 几十 KB 的快照再猜哪几个 key 是角色相关的。
///
/// `char_visible` 的判据是**投影盒落在画布内且不贴边** —— 只判断
/// 「投影成功」会把「角色被放大到糊满屏幕」这种最典型的坏状态判成
/// 可见(包围盒在画布内,但宽度接近整个画布)。所以额外要求角色高度
/// 占画布的百分比落在 `CHAR_VISIBLE_MAX_SCREEN_PCT` 以下。
///
/// # Arguments
///
/// - `&Game` - Game 的只读引用。
/// - `&str` - 已算好的 `charBox` JSON 字面量。
fn publish_visibility_state(game: &Game, char_box: &str) {
    let size: (u32, u32) = game.canvas_size.get();
    let (canvas_w, canvas_h): (f32, f32) = (size.0 as f32, size.1 as f32);
    let feet: Vec3 = game.player.get_position();
    let crown: Vec3 = [feet[0], feet[1] + PED_SCREEN_PROBE_HEIGHT, feet[2]];
    let foot_screen: Option<(f32, f32, f32)> =
        game.camera.world_to_screen(feet, canvas_w, canvas_h);
    let crown_screen: Option<(f32, f32, f32)> =
        game.camera.world_to_screen(crown, canvas_w, canvas_h);
    let (screen_x, screen_y, height_px): (f64, f64, f64) = match (foot_screen, crown_screen) {
        (Some(f), Some(c)) => (f64::from(f.0), f64::from(f.1), f64::from((f.1 - c.1).abs())),
        _ => (0.0, 0.0, 0.0),
    };
    let screen_pct: f64 = if canvas_h > 0.0 {
        100.0 * height_px / f64::from(canvas_h)
    } else {
        0.0
    };
    let projected: bool = foot_screen.is_some() && crown_screen.is_some();
    let on_screen: bool = projected
        && screen_x >= 0.0
        && screen_x <= f64::from(canvas_w)
        && screen_y >= 0.0
        && screen_y <= f64::from(canvas_h);
    let char_visible: bool = on_screen && screen_pct <= CHAR_VISIBLE_MAX_SCREEN_PCT;
    // 每个骨架 part 的真实顶点包围盒:一旦某个盒子冒出几十米,就说明顶点
    // 数据真的被写坏了(而不是 GPU mesh 表错位)。见 `gpu_mesh_extent`。
    let limb_extents: String = {
        let parts: Vec<String> = game
            .player_batches
            .iter()
            .filter_map(|limb: &PlayerLimbBatch| {
                let mesh: Option<&MeshAssetGpu> = game
                    .scene
                    .batches
                    .get(limb.batch)
                    .and_then(|batch: &SceneBatch| game.scene.meshes.get(batch.mesh_index));
                mesh.map(|mesh: &MeshAssetGpu| {
                    let extent: MeshExtent = gpu_mesh_extent(mesh);
                    format!(
                        "{{\"part\":\"{}\",\"extent\":{}}}",
                        limb.part,
                        json_f32s(&extent)
                    )
                })
            })
            .collect();
        format!("[{}]", parts.join(JSON_COMMA))
    };
    // 车队与拾取物的坐标也挂到这里:验收脚本要证明「车真的在动」,
    // 但只看 `window.__vcw` 那份几十 KB 的快照不现实,小而稳的
    // `__VCW_DEBUG__` 才是拿数值的地方。
    let car_x: Vec<f64> = game
        .traffic
        .get_cars_ref()
        .iter()
        .map(|car: &TrafficCar| f64::from(car.get_position()[0]))
        .collect();
    let car_z: Vec<f64> = game
        .traffic
        .get_cars_ref()
        .iter()
        .map(|car: &TrafficCar| f64::from(car.get_position()[2]))
        .collect();
    let json: String = format!(
        "{DEBUG_OPEN}\"charVisible\":{char_visible},\"charBoxHeightPct\":{painted},\"charBox\":{char_box},\"playerScreenX\":{screen_x},\"playerScreenY\":{screen_y},\"playerScreenPct\":{screen_pct},\"playerX\":{px},\"playerZ\":{pz},\"cameraDist\":{dist},\"camOccluded\":{occluded},\"camClearance\":{clearance},\"canvasWidth\":{canvas_w},\"canvasHeight\":{canvas_h},\"carX\":[{car_x}],\"carZ\":[{car_z}],\"pickups\":{{\"total\":{pickup_total},\"taken\":{pickup_taken}}},\"frames\":{frames},\"limbExtents\":{limb_extents}{DEBUG_CLOSE}",
        dist = game.camera.get_distance(),
        occluded = game.camera.get_occluded(),
        clearance = game.camera.get_eye_clearance(),
        canvas_w = size.0,
        canvas_h = size.1,
        px = feet[0],
        pz = feet[2],
        // 这是**投影包围盒的面积**(像素),不是"实际画上去的像素数"。
        // 真正的非背景像素数只能在浏览器里 `readPixels` 统计,那是验收
        // 脚本的活;这里给的是 Rust 侧唯一能算准的几何量,两者含义
        // 不同,不要混用。
        painted = screen_pct,
        car_x = join_f64(&car_x),
        car_z = join_f64(&car_z),
        pickup_total = game.traffic.get_pickups_ref().len(),
        pickup_taken = game.player.collected_count(),
        frames = game.frame_count,
    );
    let _ = set_window_json(DEBUG_VISIBILITY_HOOK_NAME, &json);
}

/// 把所有未拾取拾取物的 `[x, z]` 拼成 JSON 数组(验收脚本用来选目标)。
///
/// # Arguments
///
/// - `&Game` - Game 的只读引用。
///
/// # Returns
///
/// - `String` - 形如 `[[35.4,-4.0],[24.6,12.0]]` 的字面量。
fn join_pickups(game: &Game) -> String {
    let mut out: Vec<String> = Vec::new();
    for pickup in game.traffic.get_pickups_ref() {
        if pickup.get_taken() {
            continue;
        }
        let at: Vec3 = pickup.get_position();
        out.push(format!("[{},{}]", at[0], at[2]));
    }
    format!("[{}]", out.join(","))
}

/// 把 `f64` 切片拼成 `[a,b,c]` 形式的 JSON 数组字面量。
///
/// # Arguments
///
/// - `&[f64]` - 数值切片。
///
/// # Returns
///
/// - `String` - 逗号分隔的数字串(不含方括号)。
fn join_f64(values: &[f64]) -> String {
    let parts: Vec<String> = values.iter().map(|value| format!("{value}")).collect();
    parts.join(",")
}

/// HUD 文本。
///
/// # Arguments
///
/// - `&Game` - Game 的只读引用。
/// - `u32` - 输入值。
///
/// # Returns
///
/// - `String` - 结果字符串。
fn format_hud(game: &Game, triangles: u32) -> String {
    let backend: &str = game
        .renderer
        .as_ref()
        .map(|renderer: &Renderer| renderer.backend_name())
        .unwrap_or(BACKEND_NONE);
    let position: Vec3 = game.player.get_position();
    let health: String = format!("{:.0}", game.player.get_health().max(0.0));
    let cash: String = format!("{}", game.player.get_cash());
    let pickup_total: usize = game.traffic.get_pickups_ref().len();
    let collected: usize = game.player.collected_count();
    let driving: &str = match game.player.get_vehicle() {
        Some(_) => HUD_DRIVING,
        None => HUD_ON_FOOT,
    };
    let mode: &str = if game.third_person {
        HUD_THIRD_PERSON
    } else {
        HUD_ORBIT
    };
    let notice: String = game.player.get_notice().to_string();
    let notice_part: String = if notice.is_empty() {
        String::new()
    } else {
        format!(" · {notice}")
    };
    format!(
        "{HUD_TITLE} · {backend} · phase {} · {mode} · {driving} · HP {health} · ${cash} · pickups {collected}/{pickup_total} · xyz {:.1} {:.1} {:.1} · {triangles} tris · frame {}{notice_part}",
        game.input.phase.label(),
        position[0],
        position[1],
        position[2],
        game.frame_count,
    )
}

/// 更新加载进度条。
///
/// # Arguments
///
/// - `f32` - 输入值。
/// - `&str` - str 的只读引用。
fn set_progress(percent: f32, text: &str) {
    let Some(document) = window().and_then(|window: Window| window.document()) else {
        return;
    };
    if let Some(bar) = document.get_element_by_id(PROGRESS_ID) {
        let _: Result<(), euv::wasm_bindgen::JsValue> =
            bar.set_attribute("style", &format!("width: {percent:.1}%"));
    }
    if let Some(label) = document.get_element_by_id(PROGRESS_TEXT_ID) {
        label.set_text_content(Some(text));
    }
}

/// 隐藏加载遮罩。
fn hide_loading() {
    let Some(document) = window().and_then(|window: Window| window.document()) else {
        return;
    };
    if let Some(overlay) = document.get_element_by_id(LOADING_ID) {
        let _: Result<(), euv::wasm_bindgen::JsValue> =
            overlay.set_attribute(ATTR_STYLE, DISPLAY_NONE);
    }
}

/// 在遮罩上显示错误信息。
///
/// # Arguments
///
/// - `&str` - str 的只读引用。
fn show_loading_error(message: &str) {
    let Some(document) = window().and_then(|window: Window| window.document()) else {
        return;
    };
    if let Some(label) = document.get_element_by_id(PROGRESS_TEXT_ID) {
        label.set_text_content(Some(message));
    }
    if let Some(bar) = document.get_element_by_id(PROGRESS_ID) {
        let _: Result<(), euv::wasm_bindgen::JsValue> =
            bar.set_attribute(ATTR_STYLE, ERROR_BAR_STYLE);
    }
}

// ===========================================================================
// 游戏循环
// ===========================================================================

/// 把键盘输入翻译成「角色移动 / 车队行驶 / 相机跟随」。
///
/// 这是第三人称控制的核心:默认模式下 WASD 驱动**玩家角色**(不是相机),
/// 相机被动地以阻尼跟随在角色背后;只有按 Tab 切回自由观察模式后,
/// WASD 才恢复成平移相机焦点。
///
/// # Arguments
///
/// - `&mut Game` - Game 的可变引用。
/// - `f32` - 本帧的秒数增量。
fn simulate(game: &mut Game, delta: f32) {
    let forward_input: f32 = axis(&game.input, KEYW, KEYS);
    let strafe_input: f32 = axis(&game.input, KEYD, KEYA);
    let running: bool = game.input.held(KEY_SHIFT_LEFT) || game.input.held(KEY_SHIFT_RIGHT);
    let dt: f32 = delta.min(FIXED_DT * 4.0);

    // 相机水平朝向:WASD 的「前」永远跟着相机走,所以拖鼠标转相机就能
    // 转移动方向(第三人称射击的标准操作)。
    let camera_yaw: f32 = game.camera.get_yaw();
    let forward: Vec2 = [camera_yaw.cos(), -camera_yaw.sin()];

    // 车队先走:玩家开的那辆由 `drive` 接管,其余按巡航速度循环。
    let driven: Option<usize> = game.player.get_vehicle();
    match driven {
        Some(index) => {
            if let Some(car) = game.traffic.get_car_mut(index) {
                car.drive(forward_input, dt, &game.world);
            }
        }
        None => {
            // 玩家在路边「招手」:附近的车会减速停靠,不然 8–14 m/s 的车
            // 徒步根本追不上,上车功能等于不存在。
            let here: Vec3 = game.player.get_position();
            game.traffic.step(dt, Some([here[0], here[2]]));
        }
    }

    // 玩家。
    match game.player.get_vehicle() {
        Some(index) => {
            // 在车里:位置 / 朝向跟着车走,角色隐藏(姿态由车接管)。
            if let Some(car) = game.traffic.get_car_mut(index) {
                let position: Vec3 = car.get_position();
                let yaw: f32 = car.get_yaw();
                game.player.set_position(position);
                game.player.set_yaw(yaw);
                game.player.set_velocity([0.0, 0.0]);
                game.player.set_gait_amount(0.0);
                game.player.set_gait_phase(0.0);
            }
        }
        None => {
            // `Player::step` 内部会自己把「前 / 侧」按相机朝向旋转成世界
            // 方向,所以这里必须传**原始的按键轴**,不能预先旋转一次 ——
            // 预旋转会导致两次旋转,按下 W 时角色会朝侧面走。
            let intent: Vec2 = [strafe_input, forward_input];
            let speed: f32 = if running { RUN_SPEED } else { WALK_SPEED };
            game.player.step(intent, forward, dt, speed, &game.world);
        }
    }

    collect_pickups(game);
    update_camera(game, delta);
}

/// 读一对反向按键,返回一个 −1..1 的轴值。
///
/// # Arguments
///
/// - `&InputState` - 输入状态。
/// - `&str` - 正向键的 `KeyboardEvent.code`。
/// - `&str` - 反向键的 `KeyboardEvent.code`。
///
/// # Returns
///
/// - `f32` - `1.0` / `-1.0` / `0.0`。
fn axis(input: &InputState, positive: &str, negative: &str) -> f32 {
    let forward: f32 = if input.held(positive) { 1.0 } else { 0.0 };
    let backward: f32 = if input.held(negative) { 1.0 } else { 0.0 };
    forward - backward
}

/// 把跟随焦点推向玩家,并在第三人称模式下重摆相机。
///
/// 焦点用指数阻尼而不是硬跟随,所以快速转身 / 急停时相机有惯性,
/// 不会像贴在后脑勺上那样僵硬。自由观察模式直接跳过。
///
/// 焦点落位之后走**遮挡回避**:向眼点方向投射一条球体探针,命中建筑
/// 就把距离压到命中点之前(再留一点贴墙余量)。这一段与焦点阻尼是
/// 两次独立插值 —— 焦点负责「跟手」,距离负责「别穿墙」,混在一起会
/// 让两者互相拖慢。
///
/// # Arguments
///
/// - `&mut Game` - Game 的可变引用。
/// - `f32` - 本帧的秒数增量。
fn update_camera(game: &mut Game, delta: f32) {
    if !game.third_person {
        return;
    }
    let anchor: Vec3 = game.player.get_position();
    let blend: f32 = 1.0 - (-FOLLOW_DAMPING * delta).exp();
    let mut next: Vec3 = game.follow_target;
    next[0] += (anchor[0] - next[0]) * blend;
    next[1] += (FOLLOW_HEIGHT - next[1]) * blend;
    next[2] += (anchor[2] - next[2]) * blend;
    game.follow_target = next;
    game.camera.get_target_mut().copy_from_slice(&next);
    // 跟随距离始终夹在第三人称区间内。拖拽 / 捏合 / 双击都可能把
    // distance 推到离谱的值,这里兜底一次,保证镜头始终是「跟在
    // 角色背后几米」的第三人称,而不是飞远的自由相机。
    game.camera
        .clamp_follow_distance(FOLLOW_DISTANCE_MIN, FOLLOW_DISTANCE_MAX);
    // ---- 遮挡回避 ----
    // `desired_distance` 是玩家滚轮设定的「想要多远」,`distance` 是
    // 「实际能走多远」。前者不被回避改写,所以遮挡消失后有基准可回弹。
    let wanted: f32 = game
        .camera
        .get_desired_distance()
        .clamp(FOLLOW_DISTANCE_MIN, FOLLOW_DISTANCE_MAX);
    // `Some(allowed)` = 这一帧射线真的撞上了东西,相机被压到命中点之前;
    // `None` = 视野里没有遮挡,相机弹回 `wanted`。
    let hit: Option<f32> = game.camera.resolve_occlusion(&game.world, wanted);
    let allowed: f32 = hit.unwrap_or(wanted);
    game.camera.approach_distance(
        allowed,
        FOLLOW_DISTANCE_MAX,
        delta,
        OCCLUSION_IN_RATE,
        OCCLUSION_OUT_RATE,
    );
    // 地面不在碰撞世界里,单独夹一次眼点高度。
    game.camera.lift_above_ground();
    // 夹完高度之后再兜一次底:俯角一旦为负,眼点会落到地面以下,而
    // 地面是整张 `y = 0` 的网格 —— 从地下看出去整屏只有一片地面色 /
    // 天空色,角色和街景全被地面挡住。把它收进「相机在角色上方」的
    // 区间,保证任何拖拽 / 触控组合下镜头都在地面之上。
    game.camera.clamp_follow_pitch();
    // 回报状态:调试通道读 `occluded` / `eye_clearance` 就能证明回避真的
    // 触发了(距离缩短),而不是只调了默认距离。
    game.camera.occluded = hit.is_some();
    let eye: Vec3 = game.camera.eye();
    let clearance: f32 = game.world.nearest_surface_distance([eye[0], eye[2]]);
    game.camera.eye_clearance = if eye[1] < CAMERA_MIN_HEIGHT {
        0.0
    } else {
        clearance
    };
}

/// 检查玩家是否踩到拾取物,命中就结算。
///
/// # Arguments
///
/// - `&mut Game` - Game 的可变引用。
fn collect_pickups(game: &mut Game) {
    let here: Vec3 = game.player.get_position();
    let mut gained: Vec<usize> = Vec::new();
    for index in 0..game.traffic.get_pickups_ref().len() {
        let Some(pickup) = game.traffic.get_pickups_ref().get(index) else {
            continue;
        };
        if pickup.get_taken() {
            continue;
        }
        let position: Vec3 = pickup.get_position();
        let dx: f32 = position[0] - here[0];
        let dz: f32 = position[2] - here[2];
        if dx * dx + dz * dz <= PICKUP_RADIUS * PICKUP_RADIUS {
            gained.push(index);
        }
    }
    for index in gained {
        let Some(pickup) = game.traffic.get_pickup_mut(index) else {
            continue;
        };
        apply_pickup(&mut game.player, pickup);
        game.player.set_collected_push(index);
        // 拾走了就不该还在地上:把该批次的实例清零。
        if let Some(batch) = game.pickup_batches.get(index)
            && let Some(scene_batch) = game.scene.batches.get_mut(*batch)
        {
            scene_batch.instances.clear();
        }
    }
}

/// 处理 F 键:靠近车就上车,已经在车里就下车。
///
/// # Arguments
///
/// - `&mut Game` - Game 的可变引用。
///
/// # Returns
///
/// - `bool` - 本帧是否发生了上车 / 下车(用于刷新 HUD 提示)。
fn toggle_vehicle(game: &mut Game) -> bool {
    match game.player.get_vehicle() {
        Some(index) => {
            let Some(car) = game.traffic.get_car_mut(index) else {
                return false;
            };
            car.set_driven(false);
            let position: Vec3 = car.get_position();
            let yaw: f32 = car.get_yaw();
            // 下车:放到车侧后方,并立刻过一次碰撞分离,免得落在楼里。
            let side: Vec2 = [yaw.sin() * EXIT_CAR_OFFSET, yaw.cos() * EXIT_CAR_OFFSET];
            let here: Vec3 = game.player.get_position();
            let resolved: Vec2 = game
                .world
                .resolve([position[0] + side[0], position[2] + side[1]]);
            game.player
                .set_position([resolved[0], here[1], resolved[1]]);
            game.player.set_vehicle(None);
            game.player.set_notice(String::from(NOTICE_EXIT));
            true
        }
        None => {
            let here: Vec3 = game.player.get_position();
            let Some(index) = nearest_car(&game.traffic, here) else {
                game.player.set_notice(String::from(NOTICE_NO_CAR));
                return false;
            };
            if let Some(car) = game.traffic.get_car_mut(index) {
                car.set_driven(true);
            }
            game.player.set_vehicle(Some(index));
            game.player.set_notice(String::from(NOTICE_ENTER));
            true
        }
    }
}

/// 在场景里铺好玩家骨架、车队与拾取物的实例,并建立碰撞世界。
///
/// # Arguments
///
/// - `&mut Game` - Game 的可变引用。
/// - `&HashMap<String, usize>` - 资产 id → 场景 mesh 索引。
/// - `&MeshAsset` - `ped_suit` 的完整资产(用于切 part)。
fn spawn_player_traffic_pickups(
    game: &mut Game,
    index_map: &HashMap<String, usize>,
    ped: &MeshAsset,
) {
    // 玩家骨架:每个 part 一个独立 mesh + 独立批次。
    game.player_batches.clear();
    for part_name in PLAYER_PARTS {
        let batch: usize = push_player_part(&mut game.scene, ped, part_name);
        if batch == usize::MAX {
            continue;
        }
        let pivot: Vec3 = joint_pivot(part_name, part_local_bounds(ped, part_name));
        game.player_batches.push(PlayerLimbBatch {
            part: (*part_name).to_string(),
            pivot,
            batch,
        });
    }

    // 车队:每辆车一个批次(姿态独立)。
    game.car_batches.clear();
    for index in 0..game.traffic.get_cars_ref().len() {
        let asset: &'static str = match game.traffic.get_cars_ref().get(index) {
            Some(car) => car.asset,
            None => continue,
        };
        let Some(mesh_index) = index_map.get(asset) else {
            continue;
        };
        let batch: usize = find_or_create_batch(&mut game.scene, *mesh_index);
        game.car_batches.push(batch);
    }

    // 拾取物:每个拾取物一个批次(自转相位独立)。
    game.pickup_batches.clear();
    for index in 0..game.traffic.get_pickups_ref().len() {
        let asset: &'static str = match game.traffic.get_pickups_ref().get(index) {
            Some(pickup) => pickup.asset,
            None => continue,
        };
        let Some(mesh_index) = index_map.get(asset) else {
            continue;
        };
        let batch: usize = find_or_create_batch(&mut game.scene, *mesh_index);
        game.pickup_batches.push(batch);
    }

    // 碰撞世界:从资产 bounds 自动推导。
    build_collision_world(&mut game.world, &game.asset_bounds);

    // 验收脚本要的「最近一栋楼」:街区布局是程序化生成的,写死一个
    // 魔法坐标早晚会推到空地上,所以在这里对着同一份 `build_city_buildings()`
    // 算一次存下来 —— 之后每帧只是读它,不会重复分配。
    let player_at: Vec3 = game.player.get_position();
    let mut best_distance: f32 = f32::MAX;
    let mut nearest: Option<[f32; 2]> = None;
    for placement in build_city_buildings() {
        let dx: f32 = placement.position[0] - player_at[0];
        let dz: f32 = placement.position[1] - player_at[2];
        let distance: f32 = dx * dx + dz * dz;
        if distance < best_distance {
            best_distance = distance;
            nearest = Some([placement.position[0], placement.position[1]]);
        }
    }
    game.nearest_building = nearest;
}

/// 把玩家骨架、车队与拾取物的实例矩阵写回场景批次。
///
/// # Arguments
///
/// - `&mut Game` - Game 的可变引用。
fn sync_dynamic_instances(game: &mut Game) {
    let driving: bool = game.player.get_driving();
    let origin: Vec3 = game.player.get_position();
    let yaw: f32 = game.player.get_yaw();
    let phase: f32 = game.player.get_gait_phase();
    let amount: f32 = game.player.get_gait_amount();

    // 玩家骨架:四肢按步态相位摆动;上车时整批隐藏。
    for limb in game.player_batches.clone() {
        let Some(scene_batch) = game.scene.batches.get_mut(limb.batch) else {
            continue;
        };
        scene_batch.instances.clear();
        if driving {
            continue;
        }
        let swing: f32 = limb_swing(&limb.part, phase, amount);
        let model: Mat4 = limb_matrix(origin, yaw, limb.pivot, swing);

        scene_batch
            .instances
            .push(Instance::from_matrix(model, [1.0, 1.0, 1.0]));
    }

    // 车队:位置与朝向都来自交通系统。
    for index in 0..game.traffic.get_cars_ref().len() {
        let Some(car) = game.traffic.get_cars_ref().get(index) else {
            continue;
        };
        let (position, car_yaw) = (car.get_position(), car.get_yaw());
        let Some(batch) = game.car_batches.get(index) else {
            continue;
        };
        if let Some(scene_batch) = game.scene.batches.get_mut(*batch) {
            scene_batch.instances.clear();
            scene_batch
                .instances
                .push(Instance::new(position, car_yaw, 1.0, [1.0, 1.0, 1.0]));
        }
    }

    // 拾取物:绕 Y 自转。
    for index in 0..game.traffic.get_pickups_ref().len() {
        let Some(pickup) = game.traffic.get_pickups_ref().get(index) else {
            continue;
        };
        let (position, spin, taken) =
            (pickup.get_position(), pickup.get_spin(), pickup.get_taken());
        let Some(batch) = game.pickup_batches.get(index) else {
            continue;
        };
        if let Some(scene_batch) = game.scene.batches.get_mut(*batch) {
            scene_batch.instances.clear();
            if !taken {
                scene_batch
                    .instances
                    .push(Instance::new(position, spin, 1.0, [1.0, 1.0, 1.0]));
            }
        }
    }
}

/// 固定步长推进 + 渲染一帧。
///
/// # Arguments
///
/// - `&mut Game` - Game 的可变引用。
/// - `f32` - 输入值。
/// - `&mut f32` - f32 的可变引用。
/// - `u32` - 输入值。
///
/// # Returns
///
/// - `u32` - 计数结果。
fn step_and_render(
    game: &mut Game,
    elapsed: f32,
    accumulator: &mut f32,
    width: u32,
    height: u32,
) -> u32 {
    // ---- dt 钳位 ----
    let delta: f32 = (elapsed - game.frame_time).clamp(0.0, MAX_FRAME_TIME);
    game.frame_time = elapsed;
    game.canvas_size.set((width, height));

    // ---- 输入 → 玩家 / 车队 / 相机 ----
    simulate(game, delta);

    // 相机俯仰始终收在合法区间。
    game.camera.clamp_pitch();

    // ---- 每帧一次:把玩家骨架 / 车队 / 拾取物的实例写回场景批次 ----
    //
    // 漏掉这一步的话,批次存在但 `instances` 永远是空的,渲染器
    // (`if ... || batch.instances.is_empty() { continue; }`)会把它们整批
    // 跳过 —— 角色和车一个都看不见。
    sync_dynamic_instances(game);

    // ---- 固定步长累加器 ----
    *accumulator += delta;
    let mut steps: u32 = 0;
    while *accumulator >= FIXED_DT && steps < 8 {
        *accumulator -= FIXED_DT;
        steps += 1;
        game.ticks += 1;
    }

    // ---- 把玩家骨架 / 车队 / 拾取物的实例矩阵写回场景 ----
    sync_dynamic_instances(game);

    // ---- 渲染 ----
    let lighting: SceneLighting = SceneLighting::for_phase(game.input.phase);
    let aspect: f32 = if height == 0 {
        1.0
    } else {
        width as f32 / height as f32
    };
    let view_proj: crate::camera::Mat4 = game.camera.view_projection(aspect);
    // 近处剔除半径跟模式走:自由观察的机位在 200 m 外,26 m 剔的是
    // 「糊在镜头上」的行道树;第三人称眼点只有 7 m,同样的 26 m 会把
    // 整条街剔光(画面只剩地面和天空)。遮挡由相机的球体探针负责。
    let near_cull: f32 = if game.third_person {
        NEAR_CULL_RADIUS_FOLLOW
    } else {
        NEAR_CULL_RADIUS
    };
    // 验收探针:把几个已知世界点投到屏幕坐标,用来判断「几何没画」
    // 还是「画了但在视锥外」。
    let px0: f32 = game.player.get_position()[0];
    let pz0: f32 = game.player.get_position()[2];
    game.probe = [
        [px0, 0.0, pz0],
        [px0 + 20.0, 0.0, pz0],
        [px0, 0.0, pz0 - 60.0],
        [px0, 8.0, pz0 - 40.0],
    ]
    .iter()
    .map(|pt: &Vec3| {
        game.camera
            .world_to_screen(*pt, width as f32, height as f32)
            .map(|(x, y, _): (f32, f32, f32)| [x, y])
    })
    .collect();
    let mut triangles: u32 = 0;
    if let Some(renderer) = game.renderer.as_mut() {
        let result: Result<u32, String> = match renderer {
            Renderer::WebGl(webgl) => webgl.render(
                &game.scene,
                &view_proj,
                &lighting,
                game.camera.eye(),
                width,
                height,
                near_cull,
            ),
            Renderer::Software(software) => {
                Ok(software.render(&game.scene, &game.camera, &lighting, width, height, 40_000))
            }
        };
        triangles = match result {
            Ok(value) => value,
            Err(message) => {
                // WebGL 运行期出错 → 永久回退到软件渲染,而不是黑屏。
                console_log(&format!("[vcw] WebGL render failed: {message}"));
                if let Renderer::WebGl(_) = renderer {
                    let canvas: HtmlCanvasElement = game.canvas.clone();
                    match SoftwareRenderer::new(&canvas) {
                        Ok(software) => {
                            game.renderer = Some(Renderer::Software(software));
                        }
                        Err(message) => {
                            console_log(&format!("[vcw] Canvas2D fallback failed: {message}"));
                        }
                    }
                }
                0
            }
        };
    }
    game.frame_count += 1;
    triangles
}

/// 控制台输出(不引 console_error_panic_hook,直接走 web_sys console)。
///
/// # Arguments
///
/// - `&str` - str 的只读引用。
fn console_log(message: &str) {
    euv::web_sys::console::log_1(&JsValue::from_str(message));
}

/// 把一个 JSON 对象挂到 `window.<name>` 上(用于 `window.__vcw` 调试钩子)。
///
/// 刻意**不用 `eval`**:web-sys 的 `Window::eval_with_str` 需要 Cargo.toml
/// 里的 `"Function"` feature,而 Cargo.toml 已定稿不可改。`JSON::parse` +
/// `Reflect::set` 走的是已有的 `js-sys`,零新增依赖。
///
/// # Arguments
///
/// - `&str` - window 上的属性名。
/// - `&str` - 该属性的 JSON 文本。
///
/// # Returns
///
/// - `Result<(), JsValue>` - `Reflect::set` 的结果。
fn set_window_json(name: &str, json: &str) -> Result<(), JsValue> {
    let window: web_sys::Window = web_sys::window().ok_or(JsValue::from_str(NO_WINDOW))?;
    let value: JsValue = js_sys::JSON::parse(json)?;
    js_sys::Reflect::set(&window, &JsValue::from_str(name), &value).map(|_: bool| ())
}

/// 启动主循环。
///
/// - `requestAnimationFrame` 驱动,闭包 `forget()` 后由浏览器持有。
/// - 累加器把可变 `dt` 转成固定步长逻辑更新。
/// - RAF 回调需要引用自身来排下一帧,因此先把 `Function` 放进
///   `Cell<Option<..>>`,闭包内部再读出来(`Closure` 本身不能自引用)。
///
/// # Arguments
///
/// - `GameHandles` - 输入值。
fn start_loop(handles: GameHandles) {
    let callback: Rc<Cell<Option<Function>>> = Rc::new(Cell::new(None));
    let handles_for_frame: GameHandles = handles.clone();
    let callback_for_frame: Rc<Cell<Option<Function>>> = callback.clone();
    let closure: Closure<dyn FnMut(f64)> = Closure::wrap(Box::new(move |timestamp: f64| {
        let handles: GameHandles = handles_for_frame.clone();
        let elapsed: f32 = timestamp as f32 / 1000.0;
        let width: u32 = handles.canvas.width();
        let height: u32 = handles.canvas.height();
        let mut accumulator: f32 = {
            let game: std::cell::Ref<Game> = handles.game.borrow();
            game.accumulator
        };
        let triangles: u32 = {
            let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
            let triangles: u32 =
                step_and_render(&mut game, elapsed, &mut accumulator, width, height);
            game.accumulator = accumulator;
            triangles
        };
        let hud_text: String = {
            let game: std::cell::Ref<Game> = handles.game.borrow();
            format_hud(&game, triangles)
        };
        if let Some(hud) = &handles.hud {
            hud.set_text_content(Some(&hud_text));
        }
        // 验收通道按固定节拍刷新(约 20 Hz)。每帧都做一次 `JSON::parse` +
        // `Reflect::set` 会在无 GPU 的无头浏览器里把主线程吃满,反而让
        // CDP 的 `Runtime.evaluate` 超时;10 Hz 完全够读坐标用。
        if handles.game.borrow().frame_count.is_multiple_of(3) {
            publish_debug_state(&handles, &hud_text);
        }
        // 排下一帧。
        if let Some(next) = callback_for_frame.take() {
            let _: Result<i32, euv::wasm_bindgen::JsValue> = handles
                .window
                .request_animation_frame(next.unchecked_ref::<Function>());
            callback_for_frame.set(Some(next));
        }
    }));
    let trampoline: Function = closure.as_ref().unchecked_ref::<Function>().clone();
    callback.set(Some(trampoline));
    if let Some(trampoline) = callback.take() {
        let _: Result<i32, euv::wasm_bindgen::JsValue> = handles
            .window
            .request_animation_frame(trampoline.unchecked_ref::<Function>());
        callback.set(Some(trampoline));
    }
    closure.forget();
}

// ===========================================================================
// 异步加载 + 启动
// ===========================================================================

/// 解析 canvas 的像素尺寸(跟随 CSS 盒 × devicePixelRatio,上限 2×)。
///
/// # Arguments
///
/// - `&HtmlCanvasElement` - HtmlCanvasElement 的只读引用。
///
/// # Returns
///
/// - `(u32, u32)` - 计算结果。
fn sync_canvas_size(canvas: &HtmlCanvasElement) -> (u32, u32) {
    let mut ratio: f64 = window()
        .map(|win: Window| win.device_pixel_ratio())
        .unwrap_or(1.0)
        .clamp(1.0, 2.0);
    // `?res=<倍率>` 在 devicePixelRatio 之上再乘一层缩放(0 < res <= 2)。
    // 整城 47 万三角形在没有 GPU 的机器上走纯 CPU 光栅,1440p 的绘制
    // 缓冲足以把主线程吃满,键盘事件和页面脚本全部饿死;调低分辨率是
    // 唯一能把交互救回来的旋钮,而 0.5x 在游玩距离上几乎看不出差别。
    let res: f64 = f64::from(query_number(RES_PARAM, DEFAULT_RES));
    if res > 0.0 {
        ratio *= res.clamp(0.2, 2.0);
    }
    let rect: DomRect = canvas.get_bounding_client_rect();
    let (css_width, css_height): (f64, f64) = (
        if rect.width() > 0.0 {
            rect.width()
        } else {
            1280.0
        },
        if rect.height() > 0.0 {
            rect.height()
        } else {
            720.0
        },
    );
    let width: u32 = ((css_width * ratio).round() as u32).max(1);
    let height: u32 = ((css_height * ratio).round() as u32).max(1);
    if canvas.width() != width {
        canvas.set_width(width);
    }
    if canvas.height() != height {
        canvas.set_height(height);
    }
    (width, height)
}

/// 应用挂载后:取 DOM 元素、创建渲染后端、异步加载资产、启动循环。
pub fn boot() {
    let Some(window) = window() else {
        console_log(LOG_NO_WINDOW);
        return;
    };
    let Some(document) = window.document() else {
        console_log(LOG_NO_DOCUMENT);
        return;
    };
    let Some(canvas_element) = document.get_element_by_id(CANVAS_ID) else {
        console_log(LOG_NO_CANVAS);
        return;
    };
    let Ok(canvas) = canvas_element.dyn_into::<HtmlCanvasElement>() else {
        console_log(LOG_NOT_CANVAS);
        return;
    };
    sync_canvas_size(&canvas);

    // ---- 渲染后端:WebGL2 优先,失败回退 Canvas2D ----
    let (renderer, backend_note): (Option<Renderer>, String) = match WebGlRenderer::new(&canvas) {
        Ok(webgl) => {
            let renderer: Renderer = Renderer::WebGl(Box::new(webgl));
            (Some(renderer), BACKEND_WEBGL2.to_string())
        }
        Err(gl_error) => match SoftwareRenderer::new(&canvas) {
            Ok(software) => (
                Some(Renderer::Software(software)),
                format!("Canvas2D (WebGL2 unavailable: {gl_error})"),
            ),
            Err(canvas_error) => {
                console_log(&format!("[vcw] no rendering backend: {canvas_error}"));
                show_loading_error(NO_RENDERING_BACKEND_AVAILABLE_WEBGL2_AND_CA);
                (None, canvas_error)
            }
        },
    };
    console_log(&format!("[vcw] renderer = {backend_note}"));

    // 默认是**第三人称跟随**:相机离角色 FOLLOW_DISTANCE 米、俯角 FOLLOW_PITCH。
    // 城市全景机位(`apply_default_view`)只在按 Tab 切到自由观察时才用。
    let mut camera: Camera = Camera::new();
    camera.set_desired_distance(FOLLOW_DISTANCE);
    camera.set_distance(FOLLOW_DISTANCE);
    camera.set_pitch(FOLLOW_PITCH);
    camera.set_fov_y(FOLLOW_FOV);
    camera.set_near(FOLLOW_NEAR);
    camera.set_far(FOLLOW_FAR);
    camera.set_target([SPAWN_POINT[0], FOLLOW_HEIGHT, SPAWN_POINT[2]]);
    camera.set_yaw(SPAWN_YAW);

    let required: Vec<&'static str> = required_asset_ids();
    let game: Game = Game {
        canvas: canvas.clone(),
        camera,
        scene: Scene::default(),
        input: InputState::default(),
        renderer,
        loaded_assets: 0,
        total_assets: required.len(),
        frame_count: 0,
        ticks: 0,
        load_error: None,
        using_fallback: false,
        accumulator: 0.0,
        frame_time: 0.0,
        probe: Vec::new(),
        canvas_size: Cell::new((1, 1)),
        nearest_building: None,
        player: Player::new(SPAWN_POINT, SPAWN_YAW),
        traffic: Traffic::new(),
        world: CollisionWorld::new(),
        player_batches: Vec::new(),
        car_batches: Vec::new(),
        pickup_batches: Vec::new(),
        third_person: true,
        follow_target: [SPAWN_POINT[0], FOLLOW_HEIGHT, SPAWN_POINT[2]],
        asset_bounds: HashMap::new(),
    };

    let handles: GameHandles = GameHandles {
        game: Rc::new(RefCell::new(game)),
        window: window.clone(),
        canvas: canvas.clone(),
        hud: document.get_element_by_id(HUD_ID),
        phase_label: document.get_element_by_id(PHASE_LABEL_ID),
        phase_slider: document
            .get_element_by_id(PHASE_SLIDER_ID)
            .and_then(|element: Element| element.dyn_into::<HtmlInputElement>().ok()),
    };

    // ---- 事件绑定(全部裸 web_sys) ----
    bind_pointer_events(&handles);
    bind_keyboard(&handles);

    // ---- 昼夜滑块 ----
    if let Some(slider) = &handles.phase_slider {
        let handles_for_input: GameHandles = handles.clone();
        let closure: Closure<dyn FnMut(Event)> = Closure::wrap(Box::new(move |event: Event| {
            let value: String = match event
                .dyn_ref::<HtmlInputElement>()
                .map(|input: &HtmlInputElement| input.value())
            {
                Some(value) => value,
                None => return,
            };
            let parsed: f64 = value.parse().unwrap_or(0.0);
            let phase: DayPhase = DayPhase::from_slider(parsed);
            let mut game: std::cell::RefMut<Game> = handles_for_input.game.borrow_mut();
            game.input.phase = phase;
            drop(game);
            if let Some(label) = &handles_for_input.phase_label {
                label.set_text_content(Some(phase.label()));
            }
        }));
        attach(slider, EVENT_INPUT, closure);
    }

    // ---- 窗口 resize ----
    {
        let canvas_for_resize: HtmlCanvasElement = canvas.clone();
        let closure: Closure<dyn FnMut(Event)> = Closure::wrap(Box::new(move |_event: Event| {
            sync_canvas_size(&canvas_for_resize);
        }));
        attach(&window, EVENT_RESIZE, closure);
    }

    // ---- 主循环先跑起来:这样加载进度条期间也有帧在渲染 ----
    start_loop(handles.clone());

    // ---- 异步加载资产 ----
    spawn_local(async move {
        load_assets_and_build(handles).await;
    });
}

/// 拉取 manifest + 全部资产 JSON,构建场景,推进进度条。
///
/// # Arguments
///
/// - `GameHandles` - 输入值。
async fn load_assets_and_build(handles: GameHandles) {
    set_progress(4.0, FETCHING_MANIFEST);

    // 1) manifest。
    let manifest_url: String = format!("{ASSETS_BASE}manifest.json");
    let manifest_text: String = match fetch_text(&manifest_url).await {
        Ok(text) => text,
        Err(message) => {
            console_log(&format!("[vcw] manifest load failed: {message}"));
            finish_with_fallback(&handles, message);
            return;
        }
    };
    let manifest: Manifest = match serde_json::from_str(&manifest_text) {
        Ok(manifest) => manifest,
        Err(err) => {
            console_log(&format!("[vcw] manifest parse failed: {err}"));
            finish_with_fallback(&handles, format!("manifest parse failed: {err}"));
            return;
        }
    };
    let mut by_id: HashMap<String, ManifestEntry> = HashMap::new();
    let mut categories: Vec<String> = Vec::new();
    for entry in manifest.assets {
        categories.push(entry.category.clone());
        by_id.insert(entry.id.clone(), entry);
    }
    categories.sort_unstable();
    categories.dedup();

    // 2) 需要的资产。
    let required: Vec<&'static str> = required_asset_ids();
    let total: usize = required.len().max(1);
    set_progress(10.0, &format!("loading 0/{total} assets…"));

    let mut index_map: HashMap<String, usize> = HashMap::new();
    let mut asset_bounds: HashMap<String, Bounds> = HashMap::new();
    let mut ped_suit: Option<MeshAsset> = None;
    let mut scene: Scene = Scene::default();
    let mut failed: Vec<String> = Vec::new();
    let mut loaded: usize = 0;

    for id in &required {
        let Some(entry) = by_id.get(*id) else {
            failed.push(format!("{id} (missing from manifest)"));
            loaded += 1;
            set_progress(
                10.0 + 85.0 * (loaded as f32 / total as f32),
                &format!("loading {loaded}/{total} assets…"),
            );
            continue;
        };
        let url: String = format!("{ASSETS_BASE}{}", entry.file);
        match fetch_text(&url).await {
            Ok(text) => {
                // 解析一次:既要注册进场景,又要留下 `bounds` 给碰撞世界推导,
                // 以及 `ped_suit` 的原始 part(玩家骨架要按 part 切分)。
                match parse_asset(id, &text) {
                    Ok(asset) => {
                        if let Some(bounds) = asset.get_bounds() {
                            asset_bounds.insert(id.to_string(), bounds.clone());
                        }
                        if *id == PED_SUIT {
                            ped_suit = Some(asset.clone());
                        }
                        if let Err(message) = push_parsed_asset(&mut scene, &mut index_map, &asset)
                        {
                            console_log(&format!("[vcw] {message}"));
                            failed.push(message);
                        }
                    }
                    Err(message) => {
                        console_log(&format!("[vcw] {message}"));
                        failed.push(message);
                    }
                }
            }
            Err(message) => {
                console_log(&format!("[vcw] {message}"));
                failed.push(message);
            }
        }
        loaded += 1;
        let percent: f32 = 10.0 + 85.0 * (loaded as f32 / total as f32);
        set_progress(percent, &format!("loading {loaded}/{total} assets…"));
        // 让出主线程,保证进度条能刷新。
        yield_to_browser().await;
    }

    if failed.len() == required.len() {
        console_log(LOG_ALL_ASSETS_FAILED);
        finish_with_fallback(
            &handles,
            format!("all {} asset fetches failed", failed.len()),
        );
        return;
    }
    if !failed.is_empty() {
        console_log(&format!(
            "[vcw] {} asset(s) failed: {}",
            failed.len(),
            failed.join(", ")
        ));
    }

    // 3) 先铺场景(会把程序化地面也 push 进 scene.meshes),再统一上传 GPU。
    //
    // 顺序很重要:如果先上传、后 build_scene,`build_scene` 里新增的地面网格
    // 拿到的 mesh_index 会落在「已上传列表」之外,GPU 侧 get() 返回 None,
    // 地面就整块不画 —— 画面里只剩一片街道两侧的楼和树,没有马路。
    set_progress(94.0, BUILDING_SCENE);
    build_scene(&mut scene, &index_map);

    // 4) 上传所有 GPU 资源。批次里的 mesh_index 与 scene.meshes 同序。
    //
    // ⚠️ **每个 mesh 只能上传一次。**
    //
    // `WebGlRenderer::upload_mesh` 是 `push` 语义:每次调用都往
    // `self.meshes` 尾部追加一个 `GlMesh` 并返回 `len - 1`。批次里的
    // `mesh_index` 是 `Scene::meshes` 的下标,`draw_batch` 拿它**直接**
    // 当 `self.meshes` 的下标用。所以一旦同一个 mesh 上传两次,GPU 表
    // 就会整体错位一整轮。
    //
    // 曾经存在的双循环:第 4 步上传一次(build_scene 后的 30 资产 +
    // 地面 = 31 条),第 5 步又 `for (index, mesh) in game.scene.meshes
    // .iter().enumerate()` 无条件重传一遍(此时 scene.meshes 已含 13 条
    // 骨架 = 44 条)。结果 GPU 表有 31 + 44 = 75 条,而批次仍按 0..44
    // 索引 —— 骨架的 13 条(场景 31..43)实际取到的是**第二遍的
    // asset#1..#13**,也就是三栋 30 米高的楼被按 1.75 米小人的
    // model matrix 摆到玩家脚下。楼把屏幕糊满,量到 108234 px;
    // 角色真正的那 13 个 mesh 从头到尾没被画过一次。
    //
    // 症状极具迷惑性:X 落在屏幕正中(位置矩阵是对的)、包围盒又高又宽
    // (是整栋楼),所以看起来像「角色被放大 8 倍」,实际上**角色根本没
    // 画**,画的是一栋楼。`?hide=` 隔离单个批次仍然爆满屏幕,正是因为
    // 该批次取到的 mesh 与 `?hide=` 传入的批次号毫无关系。
    //
    // 现在只保留这一次上传 —— 位置在第 5 步之后,一次性覆盖
    // build_scene 的资产 + 地面 + 13 个骨架 part,顺序与
    // `scene.meshes` 严格同序。
    set_progress(97.0, STATUS_UPLOADING_MESHES);

    let asset_count: usize = index_map.len();
    // manifest 自报的资产总数(`Manifest::asset_count`)与实际索引到的数量
    // 对照:不一致说明 manifest 里有条目没被场景蓝图引用(资源白拿),
    // 或者有条目文件缺失。两者都不是致命错误,记一条日志便于排查。
    if manifest.asset_count != 0 && manifest.asset_count != asset_count {
        console_log(&format!(
            "[vcw] manifest lists {} assets, scene uses {asset_count}",
            manifest.asset_count
        ));
    }
    console_log(&format!(
        "[vcw] manifest categories: {}",
        categories.join(", ")
    ));
    // `game.scene` 必须在第 5 步**之前**拿到 build_scene 的结果。
    //
    // 顺序是硬性要求:玩家骨架 / 车队 / 拾取物的批次都是往
    // `game.scene.batches` 里 push 的。如果这一步排在 `game.scene = scene`
    // 之后,那些批次会被整份旧场景覆盖掉 —— 屏幕上只剩 build_scene 铺的
    // 静态楼和树,角色和车一个都看不见(调试钩子报 `batch count: 0`)。
    let instance_count: usize = scene.instance_count();
    let triangle_count: usize = scene.total_triangles;
    {
        let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
        game.scene = scene;
        game.loaded_assets = index_map.len();
        game.total_assets = total;
        game.load_error = if failed.is_empty() {
            None
        } else {
            Some(format!("{} assets failed", failed.len()))
        };
    }

    // 5) 玩家骨架 + 交通车队 + 拾取物 + 碰撞世界。
    //
    // 必须在 build_scene 之后、GPU 上传之后做:骨架批次是 build_scene 之后
    // 新增的批次,资产 bounds 要用来推导碰撞体。
    {
        let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
        game.asset_bounds = asset_bounds.clone();
        // 车道 X 用 `lane_x()` 从 `STREET_LINES` 现算一遍,和蓝图里的字面值
        // 互相校验:如果街道网格改了而蓝图没改,这里能立刻发现。
        console_log(&format!(
            "[vcw] lane Xs: {} / {} / {}",
            lane_x(2, -1.0),
            lane_x(2, 1.0),
            lane_x(0, 1.0)
        ));
        // 蓝图里的起始 Z 必须落在循环轨道内,否则车一出生就 wrap 一次,
        // 看起来像「凭空出现」。顺手做一次校验并打日志。
        let out_of_range: usize = TRAFFIC_LANES
            .iter()
            .filter(|(_, _, start_z, _)| start_z.abs() > TRAFFIC_HALF)
            .count();
        console_log(&format!(
            "[vcw] traffic loop half-length {TRAFFIC_HALF} m, {out_of_range} lane(s) out of range"
        ));
        game.traffic.populate(TRAFFIC_LANES, PICKUP_PLACEMENTS);
        if let Some(ped) = ped_suit.as_ref() {
            spawn_player_traffic_pickups(&mut game, &index_map, ped);
        }
        console_log(&format!(
            "[vcw] player rig: {} limb batches, {} cars, {} pickups, {} colliders",
            game.player_batches.len(),
            game.traffic.get_cars_ref().len(),
            game.traffic.get_pickups_ref().len(),
            game.world.get_shapes().len()
        ));
        // 上车 / 下车之后新增的批次也要上传。
        // 新增的骨架批次自带新 mesh,必须补一次上传。`upload_mesh` 需要
        // `&mut renderer` 和 `&scene.meshes`,所以先把 mesh 引用摘出来。
        //
        // **这里是全流程唯一一次 `upload_mesh`。** 曾经第 4 步(build_scene
        // 之后、这里之前)已经上传过一遍 31 条,这里又无条件重传 44 条,
        // `WebGlRenderer` 的 GPU 表被推到 75 条而批次仍按 0..44 索引,
        // 于是 13 个骨架批次取到的其实是第二遍的 asset#1..#13 ——
        // 三栋楼被摆到玩家脚下糊满屏幕,角色真正的 mesh 一次都没画过。
        // 详见第 4 步的注释。
        //
        // **失败绝不能 `let _ =` 吞掉。** 骨架批次(头/躯干/四肢共 13 个
        // limb)如果上传失败,画面上就表现为「有人走但看不见」—— 没有任何
        // 报错,连 `tris` 计数还是 482420,因为没上传的 mesh 压根没有 VAO,
        // `draw_batch` 查不到就静默跳过。这里把每个失败都打进 console,
        // 让症状和原因对得上。
        if let Some(Renderer::WebGl(mut webgl)) = game.renderer.take() {
            let mut upload_failures: usize = 0;
            for (index, mesh) in game.scene.meshes.iter().enumerate() {
                if let Err(error) = webgl.upload_mesh(mesh) {
                    upload_failures += 1;
                    if upload_failures <= MESH_UPLOAD_ERROR_LIMIT {
                        console_log(&format!(
                            "[vcw] upload_mesh FAILED for mesh {index} ({} verts): {error}",
                            mesh.vertices.len()
                        ));
                    }
                }
            }
            if upload_failures > 0 {
                console_log(&format!(
                    "[vcw] {upload_failures} of {} meshes failed to upload",
                    game.scene.meshes.len()
                ));
            } else {
                console_log(&format!(
                    "[vcw] uploaded {} meshes, {} limb batches in the scene",
                    game.scene.meshes.len(),
                    game.player_batches.len()
                ));
            }
            game.renderer = Some(Renderer::WebGl(webgl));
        }
    }

    set_progress(100.0, &format!("ready · {asset_count} assets"));
    hide_loading();
    console_log(&format!(
        "[vcw] scene ready: {asset_count} assets, {instance_count} instances, {triangle_count} triangles"
    ));
}

/// 资产加载全失败时的降级路径:用内置程序化场景。
///
/// # Arguments
///
/// - `&GameHandles` - GameHandles 的只读引用。
/// - `String` - 输入值。
fn finish_with_fallback(handles: &GameHandles, message: String) {
    set_progress(96.0, FALLING_BACK_TO_BUILT_IN_SCENE);
    let scene: Scene = build_fallback_scene();
    {
        let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
        game.using_fallback = true;
        game.load_error = Some(message.clone());
        game.scene = scene;
        game.loaded_assets = 0;
    }
    set_progress(100.0, READY_BUILT_IN_FALLBACK_SCENE);
    hide_loading();
    console_log(&format!("[vcw] using built-in fallback scene: {message}"));
}

/// 让出主线程一个宏任务,让浏览器有机会刷新进度条。
///
/// `Promise::resolve(undefined)` 挂到微任务队列末尾,浏览器会先把
/// 已经排队的 DOM 变化画出来,下一次 `.await` 时进度条就已更新。
async fn yield_to_browser() {
    let promise: Promise<Undefined> = Promise::resolve(&Undefined::UNDEFINED);
    let _: Result<Undefined, JsValue> = JsFuture::from(promise).await;
}

// ===========================================================================
// 静态 UI 树(唯一一棵 html! 树,只渲染一次)
// ===========================================================================

/// 应用根视图。
///
/// **只用一次 html!**。这里不调用任何 euv hook(`App::use_signal` 等),
/// 游戏循环和输入全部走裸 `web_sys`,VDOM 不参与每帧渲染。
///
/// # Returns
///
/// - `VirtualNode` - 构建好的视图节点。
pub fn app_root() -> VirtualNode {
    html! {
        div {
            id: ID_ROOT
            style: STYLE_BLOCK_01
            canvas {
                id: CANVAS_ID
                style: STYLE_BLOCK_02
            }
            div {
                id: ID_TOPBAR
                style: STYLE_BLOCK_03
                div {
                    id: HUD_ID
                    style: STYLE_BLOCK_04
                    BOOTING
                }
                div {
                    style: STYLE_BLOCK_05
                    span {
                        id: PHASE_LABEL_ID
                        style: STYLE_BLOCK_06
                        NOON
                    }
                    input {
                        id: PHASE_SLIDER_ID
                        type: INPUT_TYPE_RANGE
                        min: "0"
                        max: "1"
                        step: "1"
                        value: "0"
                        style: STYLE_BLOCK_07
                    }
                    span {
                        style: STYLE_BLOCK_08
                        "T"
                    }
                }
            }
            div {
                id: ID_HELP
                style: STYLE_BLOCK_09
                DRAG_ORBIT_WHEEL_PINCH_ZOOM_WASD_PAN_R_RESET
            }
            div {
                id: LOADING_ID
                style: STYLE_BLOCK_10
                div {
                    style: STYLE_BLOCK_11
                    UI_TITLE
                }
                div {
                    style: STYLE_BLOCK_12
                    UI_SUBTITLE
                }
                div {
                    id: ID_PROGRESS_TRACK
                    style: STYLE_BLOCK_13
                    div {
                        id: PROGRESS_ID
                        style: STYLE_BLOCK_14
                    }
                }
                div {
                    id: PROGRESS_TEXT_ID
                    style: STYLE_BLOCK_15
                    BOOTING
                }
            }
        }
    }
}
