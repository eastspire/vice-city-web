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
    camera::Camera,
    r#const::*,
    mesh::{GpuMesh, MeshAsset, MeshError, MeshPart, expand_asset},
    render::{
        DayPhase, Instance, MeshAssetGpu, Renderer, Scene, SceneBatch, SceneLighting,
        SoftwareRenderer, WebGlRenderer, build_gpu_mesh, normalize3,
    },
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
/// 相机键盘平移速度(米/秒)。
const PAN_SPEED: f32 = 26.0;
/// 滚轮每单位缩放。
const ZOOM_STEP: f32 = 0.0016;

// ===========================================================================
// 场景蓝图:一条 Vice City 大街 + 十字路口
// ===========================================================================

/// 街道沿 Z 轴延伸,人行道在两侧,建筑在 ±X 方向。
const STREET_HALF_WIDTH: f32 = 7.0;
const SIDEWALK_WIDTH: f32 = 3.6;
/// 街区沿 Z 轴的长度(米)。
const BLOCK_LENGTH: f32 = 96.0;
/// 街道网格的细分密度(车道线 / 人行道压条需要足够的顶点)。
const GROUND_SUBDIV: usize = 48;

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

/// 街道两侧的建筑(南侧 5 栋 + 北侧 5 栋 = 10 栋,全部不同资产)。
const BUILDINGS: &[BuildingPlacement] = &[
    // 南侧(x = -),朝向街道(+X)
    BuildingPlacement {
        asset: BLDG_DECO_PINK,
        position: [-19.0, -30.0],
        yaw: std::f32::consts::FRAC_PI_2,
        scale: 1.0,
        tint: [1.0, 0.96, 1.02],
    },
    BuildingPlacement {
        asset: BLDG_MINT_SHOP,
        position: [-20.5, -12.0],
        yaw: std::f32::consts::FRAC_PI_2,
        scale: 1.05,
        tint: [0.98, 1.02, 1.0],
    },
    BuildingPlacement {
        asset: BLDG_CREAM_BLOCK,
        position: [-21.0, 6.0],
        yaw: std::f32::consts::FRAC_PI_2,
        scale: 0.92,
        tint: [1.03, 1.0, 0.94],
    },
    BuildingPlacement {
        asset: BLDG_CORAL_HALL,
        position: [-20.0, 24.0],
        yaw: std::f32::consts::FRAC_PI_2,
        scale: 1.0,
        tint: [1.02, 0.95, 0.97],
    },
    BuildingPlacement {
        asset: BLDG_TEAL_LOFT,
        position: [-23.0, 42.0],
        yaw: std::f32::consts::FRAC_PI_2,
        scale: 0.95,
        tint: [0.96, 1.02, 1.03],
    },
    // 北侧(x = +),朝向街道(-X)
    BuildingPlacement {
        asset: BLDG_AQUA_ARCADE,
        position: [19.0, -30.0],
        yaw: -std::f32::consts::FRAC_PI_2,
        scale: 1.0,
        tint: [0.97, 1.01, 1.04],
    },
    BuildingPlacement {
        asset: BLDG_DECO_TEAL,
        position: [19.5, -12.0],
        yaw: -std::f32::consts::FRAC_PI_2,
        scale: 1.0,
        tint: [1.0, 1.0, 1.0],
    },
    BuildingPlacement {
        asset: BLDG_SAND_MIDRISE,
        position: [20.0, 6.0],
        yaw: -std::f32::consts::FRAC_PI_2,
        scale: 0.95,
        tint: [1.04, 1.0, 0.92],
    },
    BuildingPlacement {
        asset: BLDG_PINK_TERRACE,
        position: [20.0, 24.0],
        yaw: -std::f32::consts::FRAC_PI_2,
        scale: 0.9,
        tint: [1.02, 0.94, 1.0],
    },
    BuildingPlacement {
        asset: BLDG_WHITE_LANDMARK,
        position: [21.5, 42.0],
        yaw: -std::f32::consts::FRAC_PI_2,
        scale: 0.9,
        tint: [1.0, 1.0, 1.02],
    },
];

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

const PROPS: &[PropPlacement] = &[
    // 路灯:沿两侧人行道交替排布
    PropPlacement {
        asset: PROP_STREETLIGHT,
        position: [-8.6, 0.0, -36.0],
        yaw: std::f32::consts::FRAC_PI_2,
        tint: [1.0, 1.0, 1.0],
    },
    PropPlacement {
        asset: PROP_STREETLIGHT,
        position: [8.6, 0.0, -24.0],
        yaw: -std::f32::consts::FRAC_PI_2,
        tint: [1.0, 1.0, 1.0],
    },
    PropPlacement {
        asset: PROP_STREETLIGHT,
        position: [-8.6, 0.0, -12.0],
        yaw: std::f32::consts::FRAC_PI_2,
        tint: [1.0, 1.0, 1.0],
    },
    PropPlacement {
        asset: PROP_STREETLIGHT,
        position: [8.6, 0.0, 0.0],
        yaw: -std::f32::consts::FRAC_PI_2,
        tint: [1.0, 1.0, 1.0],
    },
    PropPlacement {
        asset: PROP_STREETLIGHT,
        position: [-8.6, 0.0, 12.0],
        yaw: std::f32::consts::FRAC_PI_2,
        tint: [1.0, 1.0, 1.0],
    },
    PropPlacement {
        asset: PROP_STREETLIGHT,
        position: [8.6, 0.0, 24.0],
        yaw: -std::f32::consts::FRAC_PI_2,
        tint: [1.0, 1.0, 1.0],
    },
    PropPlacement {
        asset: PROP_STREETLIGHT,
        position: [-8.6, 0.0, 36.0],
        yaw: std::f32::consts::FRAC_PI_2,
        tint: [1.0, 1.0, 1.0],
    },
    // 交通灯:路口四角
    PropPlacement {
        asset: PROP_TRAFFICLIGHT,
        position: [-9.4, 0.0, -8.6],
        yaw: 0.0,
        tint: [1.0, 1.0, 1.0],
    },
    PropPlacement {
        asset: PROP_TRAFFICLIGHT,
        position: [9.4, 0.0, 8.6],
        yaw: std::f32::consts::PI,
        tint: [1.0, 1.0, 1.0],
    },
    // 长椅 + 垃圾桶:人行道上
    PropPlacement {
        asset: PROP_BENCH,
        position: [-9.2, 0.0, -17.0],
        yaw: 0.0,
        tint: [1.0, 0.98, 0.94],
    },
    PropPlacement {
        asset: PROP_BENCH,
        position: [9.2, 0.0, 17.0],
        yaw: std::f32::consts::PI,
        tint: [1.0, 0.98, 0.94],
    },
    PropPlacement {
        asset: PROP_TRASH_BIN,
        position: [-9.6, 0.0, -8.0],
        yaw: 0.3,
        tint: [0.96, 1.0, 0.98],
    },
    PropPlacement {
        asset: PROP_TRASH_BIN,
        position: [9.6, 0.0, 20.0],
        yaw: -0.5,
        tint: [0.96, 1.0, 0.98],
    },
    PropPlacement {
        asset: PROP_FIRE_HYDRANT,
        position: [-9.4, 0.0, 27.0],
        yaw: 0.0,
        tint: [1.0, 0.9, 0.9],
    },
    PropPlacement {
        asset: PROP_TRAFFIC_CONE,
        position: [3.2, 0.0, -2.4],
        yaw: 0.0,
        tint: [1.05, 0.9, 0.7],
    },
    PropPlacement {
        asset: PROP_TRAFFIC_CONE,
        position: [3.2, 0.0, -0.8],
        yaw: 0.4,
        tint: [1.05, 0.9, 0.7],
    },
    PropPlacement {
        asset: PROP_NEWSSTAND,
        position: [-10.2, 0.0, 4.0],
        yaw: std::f32::consts::FRAC_PI_2,
        tint: [1.0, 1.0, 1.0],
    },
    PropPlacement {
        asset: PROP_PHONE_BOOTH,
        position: [10.2, 0.0, -20.0],
        yaw: -std::f32::consts::FRAC_PI_2,
        tint: [0.98, 1.0, 1.0],
    },
];

/// 棕榈树:交替排布在两侧人行道。
const PALM_POSITIONS: &[[f32; 2]] = &[
    [-9.9, -44.0],
    [-9.9, -32.0],
    [-9.9, -20.0],
    [-9.9, 14.0],
    [-9.9, 30.0],
    [-9.9, 42.0],
    [9.9, -44.0],
    [9.9, -32.0],
    [9.9, 2.0],
    [9.9, 14.0],
    [9.9, 30.0],
    [9.9, 42.0],
];

/// 路边停放的车辆(5 辆)。
const VEHICLES: &[(&str, [f32; 3], f32)] = &[
    (CAR_TAXI, [-4.3, 0.0, -30.0], std::f32::consts::FRAC_PI_2),
    (CAR_POLICE, [-4.3, 0.0, 6.0], std::f32::consts::FRAC_PI_2),
    (CAR_SEDAN, [4.3, 0.0, -14.0], -std::f32::consts::FRAC_PI_2),
    (CAR_COUPE, [4.3, 0.0, 22.0], -std::f32::consts::FRAC_PI_2),
    (TRUCK_PICKUP, [-4.3, 0.0, 40.0], std::f32::consts::FRAC_PI_2),
];

/// 霓虹招牌:挂在建筑外墙或街边灯杆上。
const SIGNS: &[(&str, [f32; 3], f32)] = &[
    (SIGN_HOTEL, [-11.6, 4.2, -30.0], std::f32::consts::FRAC_PI_2),
    (SIGN_PIZZA, [-11.8, 3.4, -12.0], std::f32::consts::FRAC_PI_2),
    (SIGN_CLUB, [-12.2, 5.0, 6.0], std::f32::consts::FRAC_PI_2),
    (SIGN_BAR, [-11.8, 3.8, 24.0], std::f32::consts::FRAC_PI_2),
    (SIGN_TROPIC, [-11.6, 4.6, 42.0], std::f32::consts::FRAC_PI_2),
    (
        SIGN_ARCADE,
        [11.6, 4.2, -30.0],
        -std::f32::consts::FRAC_PI_2,
    ),
    (SIGN_MOTEL, [11.8, 3.4, -12.0], -std::f32::consts::FRAC_PI_2),
    (SIGN_DINER, [11.8, 3.8, 6.0], -std::f32::consts::FRAC_PI_2),
];

/// 行人(可选点缀,让街区不至于太空)。
const PEDS: &[(&str, [f32; 3], f32)] = &[
    (PED_SUIT, [-8.2, 0.0, -6.0], 1.2),
    (PED_STREETWEAR, [8.2, 0.0, 10.0], -0.8),
    (PED_DRESS, [-8.4, 0.0, 26.0], 2.4),
    (PED_OVERALLS, [8.4, 0.0, -34.0], -2.0),
];

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
}

/// 把相机摆到「街区全景」机位。
///
/// 建筑沿街道两侧(|x| >= 14)排布,高度 9–43 m。因此默认机位必须
/// **站在街道中轴线上方**、沿 -Z 俯视整条街,否则眼点会落进两侧建筑的
/// 内部,画面会被近裁剪面附近的墙面糊死(表现为大面积近黑色块)。
///
/// 具体取值让眼点落在 `(x≈9, y≈20, z≈33)`:高于绝大多数低层建筑,
/// 又在街道走廊内,一眼能同时看到路面标线、两侧立面和停放的车辆。
///
/// # Arguments
///
/// - `&mut Camera` - Camera 的可变引用。
fn apply_default_view(camera: &mut Camera) {
    camera.target = [0.0, 4.0, -14.0];
    camera.distance = 64.0;
    camera.yaw = 0.0;
    camera.pitch = 0.42;
    camera.fov_y = std::f32::consts::FRAC_PI_4; // 45°:比 60° 收敛,避免近处楼被拉成大楔形
    camera.far = 400.0;
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

/// 生成程序化地面:沥青路面 + 人行道 + 中央双黄线 + 车道虚线 + 人行道压条。
///
/// 用顶点色(`face_colors`)而不是贴图 —— 走的是 mesh.rs 的 de-index 展开,
/// 因此地面上每条车道线都是一个真正的三角形,没有任何纹理资源。
///
/// # Returns
///
/// - `MeshAsset` - 展开后的地面网格资产。
pub fn build_ground() -> MeshAsset {
    const ROAD: [f32; 3] = [0.20, 0.20, 0.23];
    const ROAD_ALT: [f32; 3] = [0.23, 0.23, 0.26];
    const SIDEWALK: [f32; 3] = [0.62, 0.60, 0.58];
    const SIDEWALK_EDGE: [f32; 3] = [0.50, 0.48, 0.47];
    const CURB: [f32; 3] = [0.80, 0.78, 0.74];
    const LINE_YELLOW: [f32; 3] = [0.92, 0.74, 0.16];
    const LINE_WHITE: [f32; 3] = [0.90, 0.90, 0.88];

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut faces: Vec<[usize; 3]> = Vec::new();
    let mut face_colors: Vec<[f32; 3]> = Vec::new();

    let half_len: f32 = BLOCK_LENGTH * 0.5;

    // 路面:分格铺装,相邻格颜色交替,产生轻微沥青纹理感。
    for i in 0..GROUND_SUBDIV {
        for j in 0..GROUND_SUBDIV {
            let z0: f32 = -half_len + (BLOCK_LENGTH * i as f32) / GROUND_SUBDIV as f32;
            let z1: f32 = -half_len + (BLOCK_LENGTH * (i + 1) as f32) / GROUND_SUBDIV as f32;
            let x0: f32 =
                -STREET_HALF_WIDTH + (2.0 * STREET_HALF_WIDTH * j as f32) / GROUND_SUBDIV as f32;
            let x1: f32 = -STREET_HALF_WIDTH
                + (2.0 * STREET_HALF_WIDTH * (j + 1) as f32) / GROUND_SUBDIV as f32;
            let color: [f32; 3] = if (i + j) % 2 == 0 { ROAD } else { ROAD_ALT };
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
                0.0,
                color,
            );
        }
    }

    // 人行道:两侧抬高 14 cm,带路缘石。
    for side in [-1.0f32, 1.0f32] {
        let inner: f32 = side * STREET_HALF_WIDTH;
        let outer: f32 = side * (STREET_HALF_WIDTH + SIDEWALK_WIDTH);
        for i in 0..GROUND_SUBDIV {
            let z0: f32 = -half_len + (BLOCK_LENGTH * i as f32) / GROUND_SUBDIV as f32;
            let z1: f32 = -half_len + (BLOCK_LENGTH * (i + 1) as f32) / GROUND_SUBDIV as f32;
            let (x0, x1): (f32, f32) = if side < 0.0 {
                (outer, inner)
            } else {
                (inner, outer)
            };
            let color: [f32; 3] = if i % 4 == 0 { SIDEWALK_EDGE } else { SIDEWALK };
            push_quad(
                &mut QuadBuffers {
                    positions: &mut positions,
                    faces: &mut faces,
                    face_colors: &mut face_colors,
                },
                x0.min(x1),
                x0.max(x1),
                z0,
                z1,
                0.14,
                color,
            );
        }
        // 路缘石(竖直面,略微倾斜以接住高差)。
        let curb: f32 = inner;
        let curb_outer: f32 = inner - side * 0.22;
        for i in 0..GROUND_SUBDIV {
            let z0: f32 = -half_len + (BLOCK_LENGTH * i as f32) / GROUND_SUBDIV as f32;
            let z1: f32 = -half_len + (BLOCK_LENGTH * (i + 1) as f32) / GROUND_SUBDIV as f32;
            push_quad(
                &mut QuadBuffers {
                    positions: &mut positions,
                    faces: &mut faces,
                    face_colors: &mut face_colors,
                },
                curb.min(curb_outer),
                curb.max(curb_outer),
                z0,
                z1,
                0.07,
                CURB,
            );
        }
    }

    // 中央双黄线:整条街道两条实线。
    push_quad(
        &mut QuadBuffers {
            positions: &mut positions,
            faces: &mut faces,
            face_colors: &mut face_colors,
        },
        -0.34,
        -0.16,
        -half_len,
        half_len,
        0.011,
        LINE_YELLOW,
    );
    push_quad(
        &mut QuadBuffers {
            positions: &mut positions,
            faces: &mut faces,
            face_colors: &mut face_colors,
        },
        0.16,
        0.34,
        -half_len,
        half_len,
        0.011,
        LINE_YELLOW,
    );

    // 车道分隔虚线:两条车道边界,每 6 m 一段 3 m。
    for dash in 0..16 {
        let z0: f32 = -half_len + dash as f32 * 6.0 + 0.8;
        let z1: f32 = z0 + 3.0;
        if z1 > half_len {
            break;
        }
        for lane_x in [-3.4f32, 3.4f32] {
            push_quad(
                &mut QuadBuffers {
                    positions: &mut positions,
                    faces: &mut faces,
                    face_colors: &mut face_colors,
                },
                lane_x - 0.12,
                lane_x + 0.12,
                z0,
                z1,
                0.012,
                LINE_WHITE,
            );
        }
    }

    // 路口斑马线:两条斑马线横跨街道(z ≈ 0)。
    for stripe in 0..9 {
        let x: f32 = -6.6 + stripe as f32 * 1.55;
        for offset in [-3.2f32, 3.2f32] {
            push_quad(
                &mut QuadBuffers {
                    positions: &mut positions,
                    faces: &mut faces,
                    face_colors: &mut face_colors,
                },
                x,
                x + 0.85,
                offset - 1.5,
                offset + 1.5,
                0.013,
                LINE_WHITE,
            );
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
    for building in BUILDINGS {
        ids.push(building.asset);
    }
    for prop in PROPS {
        ids.push(prop.asset);
    }
    for _ in PALM_POSITIONS {
        ids.push(PALM_TALL);
        ids.push(PALM_SHORT);
    }
    for (asset, _, _) in VEHICLES {
        ids.push(asset);
    }
    for (asset, _, _) in SIGNS {
        ids.push(asset);
    }
    for (asset, _, _) in PEDS {
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
/// - `&mut Scene` - Scene 的可变引用。
/// - `&mut HashMap<String, usize>` - HashMap<String, usize> 的可变引用。
/// - `&str` - str 的只读引用。
///
/// # Returns
///
/// - `Result<usize, String>` - 计算结果。
fn register_asset(
    scene: &mut Scene,
    index_map: &mut HashMap<String, usize>,
    id: &str,
    json: &str,
) -> Result<usize, String> {
    if let Some(existing) = index_map.get(id) {
        return Ok(*existing);
    }
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
    let mesh: GpuMesh = expand_asset(&asset).map_err(|err: MeshError| format!("{id}: {err}"))?;
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
            q.split('&')
                .find_map(|kv: &str| {
                    kv.split_once('=')
                        .map(|(k, v): (&str, &str)| (k.to_string(), v.to_string()))
                })
                .filter(|(k, _): &(String, String)| k == HIDE)
                .map(|(_, v): (String, String)| v)
        })
        .unwrap_or_default()
        .split(',')
        .filter_map(|v: &str| v.trim().parse::<usize>().ok())
        .collect()
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

    // 建筑:10 栋,每个资产一个批次(每栋楼只出现一次 → 每批次 1 个实例,
    // 但顶点数据仍然只解析一次;同一批资产的多实例会合并到同一批次)。
    for building in BUILDINGS {
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

    // 街道道具:按资产分组 → 天然 instancing(路灯 7 个实例只 draw 一次)。
    for prop in PROPS {
        let Some(&mesh_index) = index_map.get(prop.asset) else {
            continue;
        };
        let batch: usize = find_or_create_batch(scene, mesh_index);
        scene.push_instance(
            batch,
            Instance::new(prop.position, prop.yaw, 1.0, prop.tint),
        );
    }

    // 棕榈:三个品种交替,每个品种一个批次。
    for (index, position) in PALM_POSITIONS.iter().enumerate() {
        let asset: &str = match index % 3 {
            0 => PALM_TALL,
            1 => PALM_SHORT,
            _ => PALM_TALL,
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

    // 车辆:5 辆停在路边。
    for (asset, position, yaw) in VEHICLES {
        let Some(&mesh_index) = index_map.get(*asset) else {
            continue;
        };
        let batch: usize = find_or_create_batch(scene, mesh_index);
        scene.push_instance(batch, Instance::new(*position, *yaw, 1.0, [1.0, 1.0, 1.0]));
    }

    // 霓虹招牌:挂在建筑外墙上,自发光在夜里点亮。
    for (asset, position, yaw) in SIGNS {
        let Some(&mesh_index) = index_map.get(*asset) else {
            continue;
        };
        let batch: usize = find_or_create_batch(scene, mesh_index);
        scene.push_instance(batch, Instance::new(*position, *yaw, 1.0, [1.0, 1.0, 1.0]));
    }

    // 行人点缀。
    for (asset, position, yaw) in PEDS {
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
            game.camera.distance *= 1.0 + (delta as f32 * ZOOM_STEP);
            game.camera.clamp_distance();
            game.camera.confine_to_street();
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
                        game.camera.confine_to_street();
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
            let down: bool = matches!(
                code.as_str(),
                KEYW | KEYA
                    | KEYS
                    | KEYD
                    | KEYR
                    | KEYT
                    | ARROWUP
                    | ARROWDOWN
                    | ARROWLEFT
                    | ARROWRIGHT
                    | KEY_SHIFT_LEFT
                    | KEY_SHIFT_RIGHT
            );
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
    format!(
        "VICE CITY WEB · {} · {} tris · {} instances · {} assets · phase {} · frame {}",
        backend,
        triangles,
        game.scene.instance_count(),
        game.loaded_assets,
        game.input.phase.label(),
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

    // ---- 键盘平移 ----
    {
        let input: &InputState = &game.input;
        let forward: f32 = if input.held(KEYW) || input.held(ARROWUP) {
            1.0
        } else if input.held(KEYS) || input.held(ARROWDOWN) {
            -1.0
        } else {
            0.0
        };
        let strafe: f32 = if input.held(KEYD) || input.held(ARROWRIGHT) {
            1.0
        } else if input.held(KEYA) || input.held(ARROWLEFT) {
            -1.0
        } else {
            0.0
        };
        let boost: f32 = if input.held(KEY_SHIFT_LEFT) || input.held(KEY_SHIFT_RIGHT) {
            2.2
        } else {
            1.0
        };
        if forward != 0.0 || strafe != 0.0 {
            let speed: f32 = PAN_SPEED * boost * delta;
            let heading: [f32; 3] = game.camera.forward();
            let right: [f32; 3] = normalize3([-heading[2], 0.0, heading[0]]);
            for axis in 0..3 {
                game.camera.target[axis] +=
                    (heading[axis] * forward + right[axis] * strafe) * speed;
            }
        }
    }

    // 焦点平移后重新把眼点收回街道走廊,避免 WASD 把相机推进建筑里。
    game.camera.clamp_pitch();

    // ---- 固定步长累加器 ----
    *accumulator += delta;
    let mut steps: u32 = 0;
    while *accumulator >= FIXED_DT && steps < 8 {
        *accumulator -= FIXED_DT;
        steps += 1;
        game.ticks += 1;
    }

    // ---- 渲染 ----
    let lighting: SceneLighting = SceneLighting::for_phase(game.input.phase);
    let aspect: f32 = if height == 0 {
        1.0
    } else {
        width as f32 / height as f32
    };
    let view_proj: crate::camera::Mat4 = game.camera.view_projection(aspect);
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
        if let Some(hud) = &handles.hud {
            let text: String = {
                let game: std::cell::Ref<Game> = handles.game.borrow();
                format_hud(&game, triangles)
            };
            hud.set_text_content(Some(&text));
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
    let ratio: f64 = window()
        .map(|win: Window| win.device_pixel_ratio())
        .unwrap_or(1.0)
        .clamp(1.0, 2.0);
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

    let mut camera: Camera = Camera::new();
    apply_default_view(&mut camera);

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
                if let Err(message) = register_asset(&mut scene, &mut index_map, id, &text) {
                    console_log(&format!("[vcw] {message}"));
                    failed.push(message);
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
    set_progress(97.0, STATUS_UPLOADING_MESHES);
    {
        let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
        for mesh in &scene.meshes {
            if let Some(Renderer::WebGl(webgl)) = game.renderer.as_mut()
                && let Err(message) = webgl.upload_mesh(mesh)
            {
                console_log(&format!("[vcw] upload failed: {message}"));
            }
        }
    }

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
    let instance_count: usize = scene.instance_count();
    let triangle_count: usize = scene.total_triangles;
    {
        let mut game: std::cell::RefMut<Game> = handles.game.borrow_mut();
        game.scene = scene;
        game.loaded_assets = asset_count;
        game.total_assets = total;
        game.load_error = if failed.is_empty() {
            None
        } else {
            Some(format!("{} assets failed", failed.len()))
        };
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
