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

pub const HIDE: &str = "hide";

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

pub const DRAG_ORBIT_WHEEL_PINCH_ZOOM_WASD_PAN_R_RESET: &str =
    "drag: orbit · wheel / pinch: zoom · WASD: pan · R: reset · T: day/night";

pub const NO_RENDERING_BACKEND_AVAILABLE_WEBGL2_AND_CA: &str =
    "No rendering backend available (WebGL2 and Canvas2D both failed).";
