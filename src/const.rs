//! 全项目字符串常量单一来源(rust-standards §1.3c)。
//!
//! verifier `verify_hardcoded_strings.py` 把 `const.rs` 视为字符串字面量的
//! canonical home:其余任何 `.rs` 文件都不允许再内联 ≥ 4 个非平凡字符的
//! 字符串字面量(测试目录、`#[...]` 属性行、format 宏的格式串除外)。
//!
//! 命名约定:
//! - `SCREAMING_SNAKE_CASE`,按 `(name_len, name_lex)` 排序(§1.5)。
//! - `STYLE_*` 是给 euv `html!` 用的整段 CSS。euv 宏的 `style: { k: "v" }`
//!   语法**不接受** const 标识符作为属性值(实测 E0308:宏会把它塞进
//!   `Css::style_string`,要求同构 `AsRef<str>` 元组),因此必须整段提为
//!   一个 const 再写 `style: STYLE_ROOT`。
//! - DOM id / 事件名 / 按键码集中在此,便于与 `index.html` 对照。

pub const DUSK: &str = "DUSK";

pub const BLDG_APRICOT_MOTEL: &str = "bldg_apricot_motel";

pub const BLDG_LILAC_TOWER: &str = "bldg_lilac_tower";

pub const U_EXPOSURE: &str = "u_exposure";

pub const U_TONE_MAP_WHITE: &str = "u_tone_map_white";

pub const HIDE: &str = "hide";

pub const TONE_MAP_WHITE_DUSK: f32 = 2.40;

pub const TONE_MAP_WHITE_DAY: f32 = 2.40;

pub const TONE_MAP_WHITE_NIGHT: f32 = 2.60;

pub const SKY_TINT_GAIN: f32 = 0.012;

pub const KEYA: &str = "KeyA";

pub const KEYD: &str = "KeyD";

pub const KEYR: &str = "KeyR";

pub const KEYS: &str = "KeyS";

pub const KEYT: &str = "KeyT";

pub const KEYW: &str = "KeyW";

pub const NOON: &str = "NOON";

pub const NIGHT: &str = "NIGHT";

pub const U_EYE: &str = "u_eye";

pub const U_FOG: &str = "u_fog";

pub const GROUND: &str = "ground";

pub const HUD_ID: &str = "vcw-hud";

pub const WEBGL2: &str = "WebGL2";

pub const ARROWUP: &str = "ArrowUp";

pub const BOOTING: &str = "booting…";

pub const ID_HELP: &str = "vcw-help";

pub const ID_ROOT: &str = "vcw-root";

pub const CANVAS2D: &str = "Canvas2D";

pub const CAR_TAXI: &str = "car_taxi";
pub const PICKUP_HEALTH_PACK: &str = "pickup_health_pack";
pub const PICKUP_ARMOR_VEST: &str = "pickup_armor_vest";
pub const PICKUP_AMMO_BOX: &str = "pickup_ammo_box";
pub const PICKUP_CASH_STACK: &str = "pickup_cash_stack";
pub const WEP_PISTOL: &str = "wep_pistol";
pub const WEP_SMG: &str = "wep_smg";
pub const WEP_BAT: &str = "wep_bat";
pub const WEP_GRENADE: &str = "wep_grenade";
pub const MARKER_FLAME: &str = "marker_flame";

pub const FALLBACK: &str = "fallback";

pub const PED_SUIT: &str = "ped_suit";

pub const SIGN_BAR: &str = "sign_bar";

pub const UI_TITLE: &str = "VICE CITY";

pub const WEBGL2_2: &str = "webgl2";

pub const ARROWDOWN: &str = "ArrowDown";

pub const ARROWLEFT: &str = "ArrowLeft";

pub const CANVAS_ID: &str = "vcw-canvas";

pub const CAR_COUPE: &str = "car_coupe";

pub const CAR_SEDAN: &str = "car_sedan";

pub const GROUND_ID: &str = "ground_procedural";

pub const ID_TOPBAR: &str = "vcw-topbar";

pub const PALM_TALL: &str = "palm_tall";

pub const PED_DRESS: &str = "ped_dress";

pub const SIGN_CLUB: &str = "sign_club";

pub const U_AMBIENT: &str = "u_ambient";

pub const ARROWRIGHT: &str = "ArrowRight";

pub const ATTR_STYLE: &str = "style";

pub const ATTR_VALUE: &str = "value";

pub const CAR_POLICE: &str = "car_police";

pub const LOADING_ID: &str = "vcw-loading";

pub const PALM_SHORT: &str = "palm_short";

pub const PROP_BENCH: &str = "prop_bench";

pub const SIGN_DINER: &str = "sign_diner";

pub const SIGN_HOTEL: &str = "sign_hotel";

pub const SIGN_MOTEL: &str = "sign_motel";

pub const SIGN_PIZZA: &str = "sign_pizza";

pub const ASSETS_BASE: &str = "assets/";

pub const EVENT_INPUT: &str = "input";

pub const EVENT_KEYUP: &str = "keyup";

pub const EVENT_WHEEL: &str = "wheel";

pub const GROUND_PART: &str = "street";

pub const NO_INFO_LOG: &str = "no info log";

pub const PROGRESS_ID: &str = "vcw-progress";

pub const SIGN_ARCADE: &str = "sign_arcade";

pub const SIGN_TROPIC: &str = "sign_tropic";

pub const UI_SUBTITLE: &str = "WASM EDITION";

pub const U_LIGHT_DIR: &str = "u_light_dir";

pub const U_SKY_COLOR: &str = "u_sky_color";

pub const U_VIEW_PROJ: &str = "u_view_proj";

pub const APP_SELECTOR: &str = "#app";

pub const BACKEND_NONE: &str = "init";

pub const DISPLAY_NONE: &str = "display: none";

pub const EVENT_RESIZE: &str = "resize";

pub const PED_OVERALLS: &str = "ped_overalls";

pub const TRUCK_PICKUP: &str = "truck_pickup";

pub const ERR_NO_WINDOW: &str = "no window";

pub const EVENT_KEYDOWN: &str = "keydown";

pub const EXPECT_GROUND: &str = "procedural ground must be valid";

pub const LOG_NO_CANVAS: &str = "[vcw] canvas element missing";

pub const LOG_NO_WINDOW: &str = "[vcw] no window, aborting";

pub const U_LIGHT_COLOR: &str = "u_light_color";

pub const BACKEND_WEBGL2: &str = "WebGL2";

pub const BLDG_DECO_PINK: &str = "bldg_deco_pink";

pub const BLDG_DECO_TEAL: &str = "bldg_deco_teal";

pub const BLDG_MINT_SHOP: &str = "bldg_mint_shop";

pub const BLDG_TEAL_LOFT: &str = "bldg_teal_loft";

pub const BUILDING_SCENE: &str = "building scene…";

pub const EVENT_TOUCHEND: &str = "touchend";

pub const FALLBACK_BLOCK: &str = "fallback_block";

pub const KEY_SHIFT_LEFT: &str = "ShiftLeft";

pub const LOG_NOT_CANVAS: &str = "[vcw] #vcw-canvas is not a <canvas>";

pub const PED_STREETWEAR: &str = "ped_streetwear";

pub const PHASE_LABEL_ID: &str = "vcw-phase-label";

pub const PROP_NEWSSTAND: &str = "prop_newsstand";

pub const PROP_TRASH_BIN: &str = "prop_trash_bin";

pub const STYLE_BLOCK_01: &str =
    "position: relative;width: 100%;height: 100%;overflow: hidden;background: #10121c";

pub const STYLE_BLOCK_02: &str = "position: absolute;top: 0;left: 0;width: 100%;height: 100%;display: block;touch-action: none;cursor: grab";

pub const STYLE_BLOCK_03: &str = "position: absolute;top: 0;left: 0;right: 0;padding: 10px 14px;display: flex;flex-direction: row;align-items: center;justify-content: space-between;box-sizing: border-box;font-family: ui-monospace, monospace;font-size: 12px;color: #ffd9f2;text-shadow: 0 1px 3px rgba(0,0,0,.85);pointer-events: none";

pub const STYLE_BLOCK_04: &str = "background: rgba(10,12,22,.55);border: 1px solid rgba(255,140,200,.35);border-radius: 6px;padding: 5px 9px;letter-spacing: .04em";

pub const STYLE_BLOCK_05: &str = "background: rgba(10,12,22,.55);border: 1px solid rgba(255,140,200,.35);border-radius: 6px;padding: 5px 9px;display: flex;flex-direction: row;align-items: center;gap: 8px;pointer-events: auto";

pub const STYLE_BLOCK_06: &str = "letter-spacing: .18em;color: #7dfcff";

pub const STYLE_BLOCK_07: &str = "width: 120px";

pub const STYLE_BLOCK_08: &str = "opacity: .7";

pub const STYLE_BLOCK_09: &str = "position: absolute;left: 14px;bottom: 12px;padding: 6px 10px;border-radius: 6px;background: rgba(10,12,22,.5);border: 1px solid rgba(125,252,255,.25);font-family: ui-monospace, monospace;font-size: 11px;line-height: 1.55;color: #cfe9ff;pointer-events: none";

pub const STYLE_BLOCK_10: &str = "position: absolute;top: 0;left: 0;right: 0;bottom: 0;display: flex;flex-direction: column;align-items: center;justify-content: center;background: linear-gradient(160deg, #1a1030 0%, #2b1246 55%, #06202e 100%);z-index: 10;font-family: ui-monospace, monospace";

pub const STYLE_BLOCK_11: &str = "font-size: 26px;letter-spacing: .30em;color: #ff7ad9;text-shadow: 0 0 18px rgba(255,122,217,.75)";

pub const STYLE_BLOCK_12: &str =
    "margin-top: 6px;font-size: 11px;letter-spacing: .34em;color: #7dfcff";

pub const STYLE_BLOCK_13: &str = "margin-top: 26px;width: min(340px, 70vw);height: 8px;border-radius: 999px;background: rgba(255,255,255,.12);overflow: hidden";

pub const STYLE_BLOCK_14: &str = "width: 0%;height: 100%;border-radius: 999px;background: linear-gradient(90deg, #ff7ad9, #7dfcff);transition: width .18s ease-out";

pub const STYLE_BLOCK_15: &str =
    "margin-top: 10px;font-size: 11px;letter-spacing: .14em;color: #e6d4ff";

pub const BLDG_CORAL_HALL: &str = "bldg_coral_hall";

pub const ERROR_BAR_STYLE: &str = "width: 100%; background: #ff4d6d";

pub const EVENT_POINTERUP: &str = "pointerup";

pub const EVENT_TOUCHMOVE: &str = "touchmove";

pub const KEY_SHIFT_RIGHT: &str = "ShiftRight";

pub const LOG_NO_DOCUMENT: &str = "[vcw] no document, aborting";

pub const PHASE_SLIDER_ID: &str = "vcw-phase";

pub const U_EMISSIVE_GAIN: &str = "u_emissive_gain";

pub const U_GLOW_STRENGTH: &str = "u_glow_strength";

pub const BLDG_AQUA_ARCADE: &str = "bldg_aqua_arcade";

pub const BLDG_CREAM_BLOCK: &str = "bldg_cream_block";

pub const EVENT_TOUCHSTART: &str = "touchstart";

pub const INPUT_TYPE_RANGE: &str = "range";

pub const PROGRESS_TEXT_ID: &str = "vcw-progress-text";

pub const PROP_PHONE_BOOTH: &str = "prop_phone_booth";

pub const PROP_STREETLIGHT: &str = "prop_streetlight";

pub const BLDG_PINK_TERRACE: &str = "bldg_pink_terrace";

pub const BLDG_SAND_MIDRISE: &str = "bldg_sand_midrise";

pub const EVENT_POINTERDOWN: &str = "pointerdown";

pub const EVENT_POINTERMOVE: &str = "pointermove";

pub const EVENT_TOUCHCANCEL: &str = "touchcancel";

pub const FETCHING_MANIFEST: &str = "fetching manifest…";

pub const ID_PROGRESS_TRACK: &str = "vcw-progress-track";

pub const PROP_FIRE_HYDRANT: &str = "prop_fire_hydrant";

pub const PROP_TRAFFICLIGHT: &str = "prop_trafficlight";

pub const PROP_TRAFFIC_CONE: &str = "prop_traffic_cone";

pub const WEBGL2_UNAVAILABLE: &str = "WebGL2 unavailable";

pub const BLDG_WHITE_LANDMARK: &str = "bldg_white_landmark";

pub const EVENT_POINTERCANCEL: &str = "pointercancel";

pub const CANVAS2D_UNAVAILABLE: &str = "Canvas2D unavailable";

pub const CREATE_BUFFER_FAILED: &str = "create_buffer failed";

pub const EXPECT_FALLBACK_BLOCK: &str = "fallback block must be valid";

pub const LOG_ALL_ASSETS_FAILED: &str =
    "[vcw] every asset fetch failed — using built-in fallback scene";

pub const STATUS_UPLOADING_MESHES: &str = "uploading meshes…";

pub const CREATE_VERTEX_ARRAY_FAILED: &str = "create_vertex_array failed";

pub const CREATE_SHADER_RETURNED_NULL: &str = "create_shader returned null";

pub const CREATE_PROGRAM_RETURNED_NULL: &str = "create_program returned null";

pub const READY_BUILT_IN_FALLBACK_SCENE: &str = "ready (built-in fallback scene)";

pub const FALLING_BACK_TO_BUILT_IN_SCENE: &str = "falling back to built-in scene…";

pub const DRAG_ORBIT_WHEEL_PINCH_ZOOM_WASD_PAN_R_RESET: &str = "WASD walk · SHIFT run · DRAG orbit camera · WHEEL zoom · F enter/exit vehicle · TAB free-look · T time of day · R reset";

pub const NO_RENDERING_BACKEND_AVAILABLE_WEBGL2_AND_CA: &str =
    "No rendering backend available (WebGL2 and Canvas2D both failed).";

pub const PART_TORSO: &str = "torso";

pub const PART_HEAD: &str = "head";

pub const PART_HAIR: &str = "hair";

pub const PART_UPPER_ARM_L: &str = "upper_arm_L";

pub const PART_LOWER_ARM_L: &str = "lower_arm_L";

pub const PART_UPPER_ARM_R: &str = "upper_arm_R";

pub const PART_LOWER_ARM_R: &str = "lower_arm_R";

pub const PART_UPPER_LEG_L: &str = "upper_leg_L";

pub const PART_LOWER_LEG_L: &str = "lower_leg_L";

pub const PART_UPPER_LEG_R: &str = "upper_leg_R";

pub const PART_LOWER_LEG_R: &str = "lower_leg_R";

pub const PART_SHOE_L: &str = "shoe_L";

pub const PART_SHOE_R: &str = "shoe_R";

pub const NOTICE_HEALTH: &str = "+35 HP · health pack";

pub const NOTICE_CASH: &str = "+$250 · cash stack";

pub const NOTICE_WEAPON: &str = "picked up · ";

pub const NOTICE_ENTER: &str = "press WASD to drive · F to exit";

pub const NOTICE_EXIT: &str = "left the vehicle";
pub const NOTICE_NO_CAR: &str = "no vehicle nearby";
pub const NOTICE_ARMOR: &str = "+50 armor · armor vest";
pub const NOTICE_AMMO: &str = "+60 rounds · ammo box";
pub const NOTICE_MARKER: &str = "mission marker reached";

pub const KEYF: &str = "KeyF";

pub const KEYTAB: &str = "Tab";

pub const HUD_TITLE: &str = "VICE CITY WEB";

pub const HUD_DRIVING: &str = "DRIVING";

pub const HUD_ON_FOOT: &str = "ON FOOT";

pub const HUD_THIRD_PERSON: &str = "third-person";

pub const HUD_ORBIT: &str = "orbit";

pub const NO_WINDOW: &str = "no window";

pub const PLAYER_PART_EXPECT: &str = "ped_suit part expansion failed";

pub const JSON_NULL: &str = "null";

pub const PART_ARM: &str = "arm";

pub const PART_LEG: &str = "leg";

pub const CHAR_BACKSLASH: &str = "\\";

pub const CHAR_QUOTE: &str = "\"";

pub const JSON_SLASH: &str = "/";

/// 调试快照 JSON 的左括号(硬编码字符串规则要求进常量表)。
pub const DEBUG_OPEN: &str = "{";

/// 调试快照 JSON 的右括号(硬编码字符串规则要求进常量表)。
pub const DEBUG_CLOSE: &str = "}";

/// 白天相位的名字。
pub const PHASE_NOON: &str = "NOON";

/// 黄昏相位的名字。
pub const PHASE_DUSK: &str = "DUSK";

/// 夜间相位的名字。
pub const PHASE_NIGHT: &str = "NIGHT";

/// 验收调试钩子挂在 `window` 上的属性名。
pub const DEBUG_HOOK_NAME: &str = "__vcw";

/// 第二套验收钩子的属性名(只放「角色可见性」这一组字段)。
///
/// 与 [`DEBUG_HOOK_NAME`] 并存而不是改名:前者是全量快照,后者是给
/// 验收脚本直接判断「角色到底画没画」的最小子集 —— 两者同时刷新,
/// 任何一边失效都不会让另一边的读取静默返回 `undefined`。
pub const DEBUG_VISIBILITY_HOOK_NAME: &str = "__VCW_DEBUG__";

/// 绘制缓冲缩放的 query 参数名(`?res=0.5`,不含 `?` / `=`)。
pub const RES_PARAM: &str = "res";
/// 缺省绘制缓冲缩放;`0.0` = 不额外缩放,跟随 devicePixelRatio。
pub const DEFAULT_RES: f32 = 0.0;
#[cfg(test)]

/// 单元测试断言文案:射线必须命中挡在视线中间的墙。
pub const T_RAY_MUST_HIT: &str = "墙在射线路径上,必须命中";
#[cfg(test)]

/// 单元测试断言文案:命中距离应接近「墙距焦点 - 探针半径」。
pub const T_RAY_DISTANCE: &str = "命中距离应接近 3 m 减去探针半径,实际 {distance}";
#[cfg(test)]

/// 单元测试断言文案:被挡时必须报告遮挡。
pub const T_OCCLUSION_REPORTED: &str = "墙挡住了视线,必须报告遮挡";
#[cfg(test)]

/// 单元测试断言文案:遮挡时允许距离必须短于期望距离。
pub const T_OCCLUSION_SHORTER: &str = "遮挡时允许距离必须短于期望距离,allowed={allowed}";
#[cfg(test)]

/// 单元测试断言文案:允许距离必须为正,否则相机会被推到玩家背后。
pub const T_OCCLUSION_POSITIVE: &str =
    "允许距离必须为正,否则相机会被推到玩家背后,allowed={allowed}";
#[cfg(test)]

/// 单元测试断言文案:视线无遮挡时必须返回 `None`。
pub const T_NO_OCCLUSION: &str = "视线无遮挡时必须返回 None,否则相机会无缘无故短一截";
#[cfg(test)]

/// 单元测试断言文案:贴墙余量必须为正。
pub const T_SKIN_POSITIVE: &str = "贴墙余量必须为正,否则相机会贴死在墙面里";
#[cfg(test)]

/// 单元测试断言文案:贴墙余量过大等于没有回避。
pub const T_SKIN_BOUNDED: &str = "贴墙余量过大等于没有回避,当前 {OCCLUSION_SKIN}";
#[cfg(test)]

/// 单元测试断言文案:拉近后距离必须变小。
pub const T_PULL_IN: &str = "拉近后距离必须变小,实际 {after}";
#[cfg(test)]

/// 单元测试断言文案:单帧内不允许一步到位,否则相机会抽搐。
pub const T_PULL_IN_SMOOTH: &str = "一帧(16 ms)不允许一步到位,否则会抽搐,实际 {after}";
#[cfg(test)]

/// 单元测试断言文案:距离硬下限必须兜住。
pub const T_FLOOR_HELD: &str = "距离硬下限必须兜住,实际 {distance} vs 下限 {floor}";
#[cfg(test)]

/// 单元测试断言文案:遮挡消失后必须平滑回到期望距离。
pub const T_RECOVERS: &str = "遮挡消失后必须平滑回到期望距离,实际 {distance}";
#[cfg(test)]

/// 单元测试断言文案:中心线不该打到画面侧面的墙。
pub const T_CENTRE_MISSES: &str = "中心线不该打到侧面的墙,实际 {centre:?}";
#[cfg(test)]

/// 单元测试断言文案:侧线必须发现画面侧面的墙。
pub const T_SIDE_HITS: &str = "侧线必须发现这堵墙,否则相机会停在楼里";

/// WebGL 上下文的 `alpha` 属性名。
pub const GL_ATTR_ALPHA: &str = "alpha";

/// WebGL 上下文的 `depth` 属性名。
pub const GL_ATTR_DEPTH: &str = "depth";

/// WebGL 上下文的 `stencil` 属性名。
pub const GL_ATTR_STENCIL: &str = "stencil";

/// WebGL 上下文的 `antialias` 属性名。
pub const GL_ATTR_ANTIALIAS: &str = "antialias";

/// WebGL 上下文的 `preserveDrawingBuffer` 属性名。
pub const GL_ATTR_PRESERVE_DRAWING_BUFFER: &str = "preserveDrawingBuffer";

/// 调试快照里「没有任何批次」时的占位描述。
pub const NO_BATCHES: &str = "no batches";

/// 单次启动里 `upload_mesh` 最多打印几条错误。
///
/// 30 个资产全挂时会刷 30 行同样的栈,反而把真正的第一条挤走;超过这个
/// 数量只报总数,细节靠 `of N meshes failed` 那一行。
pub const MESH_UPLOAD_ERROR_LIMIT: usize = 5;

/// 手写 JSON 时用的小分隔符(避免为此引入 `serde_json`)。
pub const JSON_COMMA: &str = ",";

/// 验收探针里「角色头顶」的假想高度(米)。
///
/// `ped_suit` 的资产包围盒是 1.75 m,但探针只需要一个稳定的、可复算的
/// 高度来判断角色在画面里占多少像素,所以直接用这个常量,不去查资产。
pub const PED_SCREEN_PROBE_HEIGHT: f32 = 1.75;

/// `MeshAssetGpu.vertices` 里一个顶点的 f32 数量。
///
/// 布局:position(3) + normal(3) + color(3) + emissive(3) = 12。
/// 验收探针按这个步长切顶点算包围盒,常量必须和 `build_gpu_mesh` 一致。
pub const GPU_STRIDE_FLOATS: usize = 12;

/// 判定「角色可见」时允许的**最大**屏幕高度占比(%)。
///
/// 取 45:第三人称相机 8.2 m 之外、FOV 75° 时,1.75 m 高的角色约占
/// 画布高度的 13–15%,留足余量。上限卡在 45% 是因为「角色糊满整屏」
/// 正是 GPU mesh 索引错位最典型的症状 —— 那种状态下投影盒仍然完整
/// 落在画布内,只判「投影成功」会误报成可见。
pub const CHAR_VISIBLE_MAX_SCREEN_PCT: f64 = 45.0;
