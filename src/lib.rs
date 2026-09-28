//! vice-city-web
//!
//! NEON BAY / Vice City Web —— GTA Vice City 风格开放世界浏览器游戏。
//!
//! 技术栈:`euv`(VDOM + 宏)+ `euv-engine`(数学 / 3D 数学)+ 手写
//! WebGL2 / Canvas2D 渲染后端 + `wasm-bindgen` → GitHub Pages。
//!
//! 模块划分:
//! - [`const`] —— 全项目字符串常量的单一来源(§1.3c)。
//! - [`mesh`] —— 资产 JSON schema 解析 → GPU 顶点布局。
//! - [`camera`] —— 轨道相机、投影矩阵、背面剔除。
//! - [`render`] —— WebGL2 / Canvas2D 两个渲染后端 + 共享光照参数。
//! - [`game`] —— 场景蓝图、异步加载、固定步长循环、输入、昼夜循环。

mod r#camera;
mod r#const;
mod r#game;
mod r#mesh;
mod r#render;
mod r#type;

use euv::{App, wasm_bindgen::prelude::*};

/// 挂载静态视图树并启动游戏循环。
///
/// 挂载一棵**静态** `html!` 树(canvas + 加载进度条 + HUD + 帮助),
/// 挂载完成后游戏循环和全部输入都由裸 `web_sys` + `Closure` 驱动,
/// VDOM 不参与每帧渲染 —— 这是 canvas 游戏的正确做法
/// (在事件回调里调用 euv hook 会因 hook context 丢失而静默白屏)。
#[wasm_bindgen]
pub fn main() {
    console_error_panic_hook::set_once();
    App::mount(r#const::APP_SELECTOR, r#game::app_root);
    r#game::boot();
}
