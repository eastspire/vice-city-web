//! 渲染后端:WebGL2(优先)/ Canvas2D 软件渲染(回退)。
//!
//! 两个后端共享 [`SceneLighting`] 与 [`shade_face`] 这套光照 / 配色参数,
//! 因此同一份场景在两条路径上视觉一致(除 GPU 特有的泛光后期之外)。
//!
//! 数据组织方式是 **instancing 友好的**:同类资产只解析一次,顶点数据只上传一份,
//! 每个实例只提供 model matrix + tint(见 [`SceneBatch`] / [`MeshAssetGpu`])。

use euv::{
    wasm_bindgen::JsValue,
    web_sys::{
        HtmlCanvasElement, WebGl2RenderingContext, WebGlBuffer, WebGlProgram, WebGlShader,
        WebGlUniformLocation, WebGlVertexArrayObject,
    },
};

use crate::{
    camera::{Mat4, is_back_facing},
    r#const::*,
    mesh::{GpuMesh, f32_slice_to_bytes},
    r#type::{Mat4Data, Rgb8, Vec3},
};

/// 每个顶点的 f32 数量,与 [`crate::mesh::FLOATS_PER_VERTEX`] 一致:
/// `position(3) | normal(3) | color(3) | emissive(3)`。
///
/// 资产 JSON 里 part 有 `emissive`,展开时追加第 4 个 vec3,这样霓虹招牌
/// 可以在着色阶段直接按自发光强度叠加,不需要再查一次表。
pub const STRIDE_FLOATS: usize = 12;

/// 单个 instance 在 buffer 里的字节跨度,必须等于
/// `FLOATS_PER_INSTANCE * 4`,并作为 instanced 属性的 `vertex_attrib_pointer`
/// stride 使用。
pub const INSTANCE_STRIDE_BYTES: i32 = (FLOATS_PER_INSTANCE * 4) as i32;

/// 每个实例在 instance buffer 里的 f32 数量:
/// model matrix 4 个 vec4(16 f32)+ tint vec3 + 1 个 vec4 填充 = 28 f32 = 7×vec4。
///
/// 填充是为了让 tint 也落在 vec4 对齐的 slot 上,VAO 里 stride 直接用
/// `7 * 16` 字节即可。
///
/// GPU 侧布局:model matrix 4 × vec4(64 B)+ tint vec3(12 B) = **76 B**,
/// 但为了和 `vertex_attrib_pointer` 的偏移保持一致、并让 tint 也落在
/// 下一个 4-float 边界上,这里按 **20 f32 = 80 B** 对齐
/// (16 f32 model + 4 f32 tint/pad)。
///
/// ⚠️ 这个常量必须同时等于:
/// - `draw_batch` 每实例写入的 f32 数量
/// - `reserve_instances` 每实例分配的字节数 ÷ 4
/// - `upload_mesh` 里 `vertex_attrib_pointer` 的 stride(字节)
///   三者只要有一个不一致,第 2 个及以后的实例就会读到错位的 model/tint。
pub const FLOATS_PER_INSTANCE: usize = 20;

/// instance buffer 的预分配实例数。
///
/// 必须 ≥ 单批次最大的实例数(场景里最多的是 12 棵行道树),并且要在
/// `upload_mesh` 建 VAO **之前** 分配好 —— 否则 VAO 捕获的是一个 0 字节的
/// buffer,`vertex_attrib_pointer` 记下的偏移在后续 `bufferData` 扩容后
/// 不会重新绑定,部分驱动上会渲染出未初始化内存。
const INSTANCE_PREALLOC: usize = 64;

/// 自由观察(轨道)模式的近处遮挡剔除半径(米)。
///
/// 默认机位在街区斜上方俯视,落在近处的行道树 / 路灯会糊住半个屏幕
/// (9 m 高的棕榈离眼点只有十几米,一层树叶就是一整屏)。这不是几何错误,
/// 是「相机正好在物体旁边」——靠挪机位只能顾此失彼,所以在渲染器里按
/// 实例中心到眼点的距离统一剔掉。
///
/// 半径取 26 m。默认机位在 z≈56、y≈38 处俯视,行道树在 z=14/30/42,
/// 距离分别是 26/13/11 m —— 26 m 正好把「压在镜头上的」那三棵剔掉,
/// 同时保住 z≤-20 那一排(60+ m)给街景留纵深。
///
/// **只对自由观察有效。** 第三人称相机的眼点离角色只有 7 m 上下,26 m
/// 半径会把整条街的楼、行道树、路灯**全部**剔光,画面只剩地面和天空
/// (实测第三人称取样只有 34 种颜色、轨道机位 4571 种)。第三人称用
/// `NEAR_CULL_RADIUS_FOLLOW` —— 见那里的说明。
pub const NEAR_CULL_RADIUS: f32 = 26.0;

/// 第三人称模式的近处遮挡剔除半径(米)。
///
/// 必须远小于 `NEAR_CULL_RADIUS`,因为第三人称眼点本来就**故意**贴着
/// 一切走:角色 7 m、路缘 8 m、对面楼 12 m。这个半径只负责剔掉「真的
/// 糊在镜头上」的东西(半径内 = 半米量级),遮挡由相机的球体探针负责
/// 回避 —— 两者是互补的,不是重复的。
pub const NEAR_CULL_RADIUS_FOLLOW: f32 = 0.35;

/// 实例中心到眼点的距离(取模型矩阵的平移列)。
///
/// # Arguments
///
/// - `&Instance` - Instance 的只读引用。
/// - `Vec3` - 输入值。
///
/// # Returns
///
/// - `f32` - 计算结果。
///   实例中心到眼点的距离(取模型矩阵的平移列)。
///
/// # Arguments
///
/// - `&Instance` - Instance 的只读引用。
/// - `Vec3` - 输入值。
///
/// # Returns
///
/// - `f32` - 计算结果。
fn instance_distance(instance: &Instance, eye: Vec3) -> f32 {
    let m: &Mat4Data = instance.get_model_ref();
    let dx: f32 = m[12] - eye[0];
    let dy: f32 = m[13] - eye[1];
    let dz: f32 = m[14] - eye[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// 一个三角面在 CPU 侧的表示,软件渲染与 WebGL 共用。
#[derive(Clone, Copy, Debug)]
pub struct Face {
    /// 三个顶点下标(指向 [`GpuMesh::vertices`],单位是顶点而不是 f32)。
    pub indices: [u32; 3],
    /// 平面法线(已归一化,单位向量)。
    pub normal: Vec3,
    /// 基础色,线性空间 0..1。
    pub color: Vec3,
    /// 自发光色,线性空间 0..1。
    pub emissive: Vec3,
}

/// 一份资产在 GPU / CPU 两种后端下共用的展开结果。
///
/// 解析一次(见 [`crate::mesh::parse_asset`]),顶点数据只保留一份,
/// 供「按资产类型分批」的 instancing 渲染复用。
#[derive(Debug, Default)]
pub struct MeshAssetGpu {
    /// 展平后的顶点数据,`STRIDE_FLOATS` 个 f32 / 顶点。
    pub vertices: Vec<f32>,
    /// 三角形索引(每个三角形 3 个)。
    pub indices: Vec<u32>,
    /// 每个三角形的面信息(法线 / 颜色 / 自发光)。
    pub faces: Vec<Face>,
    /// 三角形数量。
    pub triangle_count: usize,
}

impl MeshAssetGpu {
    /// 三角形数量的只读副本。
    ///
    /// # Returns
    ///
    /// - `usize` - 三角形数量。
    pub fn get_triangle_count(&self) -> usize {
        self.triangle_count
    }
}

/// 把 [`crate::mesh::parse_asset`] 的结果转换成渲染后端用的布局。
///
/// `mesh` 是 mesh.rs 展开出的 de-index 网格(每个三角形 3 个独立顶点、
/// 法线取面法线、颜色取 `face_colors`),这里额外把每个三角形的
/// 自发光强度写进第 4 个 vec3。
///
/// `part_emissive` 是与 `mesh.triangle_count` 等长的自发光数组,
/// 由 game.rs 在解析 JSON 时按 part 顺序汇总。
/// 把 [`crate::mesh::parse_asset`] 的结果转换成渲染后端用的布局。
///
/// `mesh` 是 mesh.rs 展开出的 de-index 网格(每个三角形 3 个独立顶点、
/// 法线取面法线、颜色取 `face_colors`),这里额外把每个三角形的
/// 自发光强度写进第 4 个 vec3。
///
/// `part_emissive` 是与 `mesh.triangle_count` 等长的自发光数组,
/// 由 game.rs 在解析 JSON 时按 part 顺序汇总。
///
/// # Arguments
///
/// - `&GpuMesh` - GpuMesh 的只读引用。
/// - `&[Vec3]` - [Vec3] 的只读引用。
///
/// # Returns
///
/// - `MeshAssetGpu` - 计算结果。
pub fn build_gpu_mesh(mesh: &GpuMesh, part_emissive: &[Vec3]) -> MeshAssetGpu {
    let mut out: MeshAssetGpu = MeshAssetGpu {
        triangle_count: mesh.triangle_count,
        ..Default::default()
    };
    out.vertices.reserve(mesh.vertex_count() * STRIDE_FLOATS);
    let vertex_count: usize = mesh.vertex_count();
    for vertex in 0..vertex_count {
        let base: usize = vertex * 9;
        let position: [f32; 3] = [
            mesh.vertices[base],
            mesh.vertices[base + 1],
            mesh.vertices[base + 2],
        ];
        let normal: [f32; 3] = [
            mesh.vertices[base + 3],
            mesh.vertices[base + 4],
            mesh.vertices[base + 5],
        ];
        let color: [f32; 3] = [
            mesh.vertices[base + 6],
            mesh.vertices[base + 7],
            mesh.vertices[base + 8],
        ];
        // 三角形 = 第 `vertex / 3` 个三角形;顶点 0/1/2 属于同一个三角形。
        let triangle: usize = vertex / 3;
        let emissive: [f32; 3] = part_emissive.get(triangle).copied().unwrap_or([0.0; 3]);
        out.vertices.extend_from_slice(&[
            position[0],
            position[1],
            position[2], //
            normal[0],
            normal[1],
            normal[2], //
            color[0],
            color[1],
            color[2], //
            emissive[0],
            emissive[1],
            emissive[2],
        ]);
    }
    for triangle in 0..mesh.triangle_count {
        let base: usize = (triangle * 3) * 9;
        let normal: [f32; 3] = [
            mesh.vertices[base + 3],
            mesh.vertices[base + 4],
            mesh.vertices[base + 5],
        ];
        let color: [f32; 3] = [
            mesh.vertices[base + 6],
            mesh.vertices[base + 7],
            mesh.vertices[base + 8],
        ];
        let emissive: [f32; 3] = part_emissive.get(triangle).copied().unwrap_or([0.0; 3]);
        out.indices.extend_from_slice(&[
            (triangle * 3) as u32,
            (triangle * 3 + 1) as u32,
            (triangle * 3 + 2) as u32,
        ]);
        out.faces.push(Face {
            indices: [
                (triangle * 3) as u32,
                (triangle * 3 + 1) as u32,
                (triangle * 3 + 2) as u32,
            ],
            normal,
            color,
            emissive,
        });
    }
    out
}

/// 昼夜预设。三档:正午 / 黄昏 / 夜晚。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DayPhase {
    /// 正午:高角度强白光。
    Noon,
    /// 黄昏:低角度暖橙光。
    Dusk,
    /// 夜晚:冷月光 + 高霓虹强度。
    Night,
}

impl DayPhase {
    /// 相位名(HUD 与调试快照用)。
    ///
    /// # Returns
    ///
    /// - `&'static str` - 相位名。
    pub fn as_str(&self) -> &'static str {
        match self {
            DayPhase::Noon => PHASE_NOON,
            DayPhase::Dusk => PHASE_DUSK,
            DayPhase::Night => PHASE_NIGHT,
        }
    }

    /// 下一个相位(循环)。
    pub fn next(self) -> Self {
        match self {
            DayPhase::Noon => DayPhase::Dusk,
            DayPhase::Dusk => DayPhase::Night,
            DayPhase::Night => DayPhase::Noon,
        }
    }

    /// 归一化到 `[0, 1)` 的滑块位置。
    ///
    /// # Returns
    ///
    /// - `f64` - 计算结果。
    pub fn slider_value(self) -> f64 {
        match self {
            DayPhase::Noon => 0.0,
            DayPhase::Dusk => 0.5,
            DayPhase::Night => 1.0,
        }
    }

    /// 由滑块位置反推相位(四舍五入到最近的档位)。
    ///
    /// # Arguments
    ///
    /// - `f64` - 输入值。
    pub fn from_slider(value: f64) -> Self {
        let clamped: f64 = value.clamp(0.0, 1.0);
        let index: usize = (clamped * 2.0).round() as usize;
        match index {
            0 => DayPhase::Noon,
            1 => DayPhase::Dusk,
            _ => DayPhase::Night,
        }
    }

    /// 显示名(HTML overlay 用)。
    ///
    /// # Returns
    ///
    /// - `&'static str` - 计算结果。
    pub fn label(self) -> &'static str {
        match self {
            DayPhase::Noon => NOON,
            DayPhase::Dusk => DUSK,
            DayPhase::Night => NIGHT,
        }
    }
}

/// 两个后端共享的光照参数。
///
/// WebGL 把它作为 uniform 上传,Canvas2D 后端在 CPU 上跑同一个公式,
/// 保证两个后端的画面在色调整体上一致。
#[derive(Clone, Copy, Debug)]
pub struct SceneLighting {
    /// 太阳 / 月亮方向光方向(指向光源)。
    pub light_dir: Vec3,
    /// 方向光颜色 × 强度。
    pub light_color: Vec3,
    /// 环境光颜色 × 强度。
    pub ambient: Vec3,
    /// 天空 / 雾颜色。
    pub sky_color: Vec3,
    /// 自发光全局增益(夜晚更大)。
    pub emissive_gain: f32,
    /// 色调映射的曝光系数(线性域乘子)。
    pub exposure: f32,
    /// 色调映射的白色点(亮度超过这个值就开始明显压缩高光)。
    pub tone_map_white: f32,
    /// 大气雾:起始距离(米)。比这更近的物体完全不受雾影响。
    pub fog_start: f32,
    /// 大气雾:完全饱和的距离(米)。
    pub fog_end: f32,
}

impl SceneLighting {
    /// 取某个相位的默认光照参数。
    ///
    /// # Arguments
    ///
    /// - `DayPhase` - 输入值。
    pub fn for_phase(phase: DayPhase) -> Self {
        // 三相位的共同约束(数值都在 GPU / CPU 两端共用):
        //
        // 1. **朝光面总亮度可控。** 朝光面的线性亮度是
        //    `albedo * (ambient + light_color * n_dot_l)`。最亮的白色楼
        //    albedo 接近 0.86,如果 `ambient + light_color` 直接取到 2.0,
        //    线性值就是 1.7 —— 远超 1.0,`clamp` 之后所有浅色面一律糊成
        //    纯白,色相信息(粉 / 薄荷 / 珊瑚)在 sRGB 之前就已经被抹掉了。
        //    这里把三相位的 `ambient + light_color` 都压在 ~1.6 以内,
        //    剩下的高光交给 [`tonemap`] 的肩部压缩。
        // 2. **曝光固定为 1.0,靠白色点而不是靠压暗光强来「解决」过曝。**
        //    压暗方向光会让阴影面一起塌死;抬高白色点则保住中间调的
        //    色相,只把真正刺眼的顶端收进来。
        // 3. **夜晚不能死黑。** 环境光带一点蓝紫(城市天光 + 霓虹回弹),
        //    所以夜里路面和楼体仍然可辨,只是整体偏冷偏暗。
        match phase {
            DayPhase::Noon => Self {
                light_dir: normalize3([0.35, 0.86, 0.36]),
                // 方向光接近中性白(略偏暖),环境光偏天空蓝 —— 阴影面
                // 因此是冷蓝而不是灰,和阳光面的暖白拉开冷暖对比。
                light_color: [1.16, 1.10, 0.99],
                ambient: [0.34, 0.38, 0.46],
                sky_color: [0.44, 0.70, 0.92],
                emissive_gain: 0.18,
                exposure: 1.0,
                tone_map_white: TONE_MAP_WHITE_DAY,
                // 街区扩到 ±150 m,雾必须够远才看得到成片的街区;
                // fog_start 压到接近雾的起点,远景褪向天空色而不是突然消失。
                fog_start: 120.0,
                fog_end: 460.0,
            },
            DayPhase::Dusk => Self {
                light_dir: normalize3([0.86, 0.24, -0.44]),
                // 低角度暖橙直射 + 偏紫的天空环境光:朝光面是橙红,
                // 背光面落到冷紫,这是黄昏最主要的色相来源。
                light_color: [1.62, 0.84, 0.42],
                ambient: [0.26, 0.24, 0.40],
                sky_color: [0.95, 0.46, 0.36],
                emissive_gain: 0.90,
                exposure: 1.0,
                tone_map_white: TONE_MAP_WHITE_DUSK,
                fog_start: 100.0,
                fog_end: 420.0,
            },
            DayPhase::Night => Self {
                light_dir: normalize3([-0.42, 0.72, -0.55]),
                // 月光很弱,但刻意保留蓝紫偏色;环境光是全画面「不死黑」
                // 的唯一来源 —— 路面 / 楼体靠它提亮,而不是把直射光调大。
                light_color: [0.24, 0.31, 0.58],
                ambient: [0.11, 0.14, 0.26],
                sky_color: [0.045, 0.055, 0.13],
                emissive_gain: 1.85,
                exposure: 1.0,
                tone_map_white: TONE_MAP_WHITE_NIGHT,
                fog_start: 90.0,
                fog_end: 400.0,
            },
        }
    }
}

/// 线性亮度权重(Rec. 709 / sRGB 的 Y)。
const LUMA_WEIGHTS: Vec3 = [0.2126, 0.7152, 0.0722];

/// 单个线性亮度的 Reinhard 扩展色调映射(有肩部的高光滚降)。
///
/// 标准 Reinhard 是 `L / (1 + L)`,它把 L=1 映射到 0.5 —— 也就是说
/// 一个「正常曝光」的白色面在 sRGB 编码后只有 ~188/255,整个画面永远
/// 偏暗发灰。扩展版(Jim Hejl / Richard Burgess-Dawson 的 W 参数化)
/// 满足 `f(W) = 1`,即**白色点恰好映射到 1.0**,中间调保持线性感,
/// 只有超过 `white` 的部分才进肩部压缩:
///
/// ```text
/// f(L) = L * (1 + L / W²) / (1 + L)
/// ```
///
/// 这正是「过曝」的正确解法:压的是高光顶端,不是把光强整体调暗。
///
/// # Arguments
///
/// - `f32` - 线性亮度(非负)。
/// - `f32` - 曝光系数(线性域乘子)。
/// - `f32` - 白色点(必须 > 0)。
///
/// # Returns
///
/// - `f32` - 映射后的线性亮度,落在 `[0, 1)`。
pub fn tonemap_luma(luma: f32, exposure: f32, white: f32) -> f32 {
    let w: f32 = if white > f32::EPSILON { white } else { 1.0 };
    let x: f32 = if luma > 0.0 { luma } else { 0.0 } * exposure;
    let w2: f32 = w * w;
    x * (1.0 + x / w2) / (1.0 + x)
}

/// 按亮度做色相保持(color-preserving)的色调映射。
///
/// 逐通道独立映射会把过曝的亮面直接去色 —— 一个亮度 1.7 的粉红面
/// 逐通道 clamp 后是 (1.0, 1.0, 1.0),纯白;而同一个面如果按亮度整体
/// 缩放,R:G:B 的比例被保留,仍然读得出是粉红。所以这里只对**亮度**
/// 做色调映射,再按 `f(L') / (L * exposure)` 的比例把颜色整体缩放回去,
/// 即保留色相与饱和度关系,只压亮度。
///
/// 纯黑(`luma <= 0`)直接原样返回,避免除零。
///
/// # Arguments
///
/// - `Vec3` - 线性 RGB(未做曝光与色调映射)。
/// - `f32` - 曝光系数(线性域乘子)。
/// - `f32` - 白色点(必须 > 0)。
///
/// # Returns
///
/// - `Vec3` - 色调映射后的线性 RGB。
pub fn tonemap(value: Vec3, exposure: f32, white: f32) -> Vec3 {
    let luma: f32 =
        value[0] * LUMA_WEIGHTS[0] + value[1] * LUMA_WEIGHTS[1] + value[2] * LUMA_WEIGHTS[2];
    if luma <= 1e-6 {
        return [0.0, 0.0, 0.0];
    }
    let mapped: f32 = tonemap_luma(luma, exposure, white);
    let divisor: f32 = luma * exposure;
    let scale: f32 = if divisor > f32::EPSILON {
        mapped / divisor
    } else {
        1.0
    };
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

/// 两个后端共用的平面着色公式。
///
/// ```
/// base   = albedo * tint
/// lit    = base * (ambient + light_color * max(dot(normal, light_dir), 0))
///        + base * emissive * emissive_gain
/// color  = tonemap(lit, exposure, tone_map_white)   // 线性域,色相保持
///        + sky_color * SKY_TINT_GAIN                  // 天空色晕染
/// color  = mix(color, sky_color, fog)                // 大气雾
/// out    = linear_to_srgb(color)
/// ```
///
/// **色相为什么必须保住(这正是「一片惨白」的历史根因):**
/// 资产颜色按 `assets/SCHEMA.md` 是**线性** albedo,浅色面的亮度
/// 可以接近 0.86。经过 `ambient + light_color * n_dot_l` 之后,朝光面
/// 的线性亮度会到 1.5~1.9 —— 远超 1.0。如果这里逐通道 `clamp` 或逐通道
/// 映射,(1.5, 1.2, 1.4) 会被压成 (1, 1, 1),粉红面和薄荷面全部变成
/// 纯白,「有颜色」在 sRGB 编码之前就已经丢失了。色相保持的映射把
/// 亮度压到 1.0 以下,R:G:B 的比例不变,浅粉仍然读得出是浅粉。
///
/// `eye_distance` 是面中心到相机眼点的距离(米),用于大气雾。
///
/// # Arguments
///
/// - `Vec3` - 输入值。
/// - `&SceneLighting` - SceneLighting 的只读引用。
/// - `f32` - 输入值。
///
/// # Returns
///
/// - `[f32` - 计算结果。
pub fn shade_face(
    albedo: Vec3,
    emissive: Vec3,
    normal: Vec3,
    tint: Vec3,
    lighting: &SceneLighting,
    eye_distance: f32,
) -> [f32; 3] {
    let n_dot_l: f32 = (normal[0] * lighting.light_dir[0]
        + normal[1] * lighting.light_dir[1]
        + normal[2] * lighting.light_dir[2])
        .max(0.0);
    let base: [f32; 3] = [
        albedo[0] * tint[0],
        albedo[1] * tint[1],
        albedo[2] * tint[2],
    ];
    let mut out: [f32; 3] = [0.0; 3];
    for channel in 0..3 {
        let diffuse: f32 = lighting.ambient[channel] + lighting.light_color[channel] * n_dot_l;
        out[channel] =
            base[channel] * diffuse + base[channel] * emissive[channel] * lighting.emissive_gain;
    }
    // 色调映射(色相保持)+ 天空色晕染。顺序很重要:先在线性域压亮度,
    // 再加天空色,最后才是 sRGB 编码。
    let mut mapped: [f32; 3] = tonemap(out, lighting.exposure, lighting.tone_map_white);
    for channel in 0..3 {
        mapped[channel] += lighting.sky_color[channel] * SKY_TINT_GAIN;
    }
    // 大气雾:线性空间里向天空色插值,自发光通道不吃雾(夜里霓虹要穿透雾)。
    let span: f32 = (lighting.fog_end - lighting.fog_start).max(f32::EPSILON);
    let fog: f32 = ((eye_distance - lighting.fog_start) / span).clamp(0.0, 1.0);
    let fog: f32 = fog * fog * (3.0 - 2.0 * fog);
    let self_lit: f32 = base[0] * emissive[0] + base[1] * emissive[1] + base[2] * emissive[2];
    if self_lit > 0.01 {
        let glow: f32 = (fog * 0.72).min(0.72);
        for (channel, sky) in mapped.iter_mut().zip(lighting.sky_color.iter()) {
            *channel += sky * glow;
        }
    }
    for (channel, sky) in mapped.iter_mut().zip(lighting.sky_color.iter()) {
        *channel = *channel * (1.0 - fog) + sky * fog;
    }
    linear_to_srgb(mapped)
}

/// 线性 → sRGB 传输函数(与 `assets/SCHEMA.md` §2「linear RGB, apply the
/// usual linear→sRGB transfer at display time」一致)。
/// 线性 → sRGB 传输函数(与 `assets/SCHEMA.md` §2「linear RGB, apply the
/// usual linear→sRGB transfer at display time」一致)。
///
/// # Arguments
///
/// - `Vec3` - 输入值。
///
/// # Returns
///
/// - `Vec3` - 计算结果。
pub fn linear_to_srgb(value: Vec3) -> Vec3 {
    let mut out: [f32; 3] = [0.0; 3];
    for channel in 0..3 {
        let c: f32 = value[channel].clamp(0.0, 1.0);
        out[channel] = if c <= 0.003_130_8 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
    }
    out
}

/// 把归一化的 sRGB 分量量化成 `0..255`。
///
/// # Arguments
///
/// - `Vec3` - 输入值。
///
/// # Returns
///
/// - `Rgb8` - 计算结果。
pub fn srgb_to_u8(value: Vec3) -> Rgb8 {
    [
        (value[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (value[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (value[2].clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

/// 归一化向量(零向量原样返回)。
///
/// # Arguments
///
/// - `Vec3` - 输入值。
///
/// # Returns
///
/// - `Vec3` - 计算结果。
pub fn normalize3(value: Vec3) -> Vec3 {
    let len: f32 = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if len <= f32::EPSILON {
        [0.0, 1.0, 0.0]
    } else {
        [value[0] / len, value[1] / len, value[2] / len]
    }
}

// ===========================================================================
// 场景批次(instancing 友好的组织方式)
// ===========================================================================

/// 一个实例:model matrix(列主序 16 f32)+ tint。
#[derive(Clone, Copy, Debug)]
pub struct Instance {
    /// model matrix,列主序,直接喂 `uniformMatrix4fv` / 自己算。
    pub(crate) model: Mat4Data,
    /// 逐实例色调乘子。
    pub(crate) tint: Vec3,
}

impl Instance {
    /// model matrix 的只读引用。
    ///
    /// # Returns
    ///
    /// - `&Mat4Data` - 列主序 model matrix。
    pub fn get_model_ref(&self) -> &Mat4Data {
        &self.model
    }

    /// 构造一个实例。
    ///
    /// # Arguments
    ///
    /// - `Vec3` - 世界坐标位置。
    /// - `f32` - 绕 Y 轴朝向(弧度)。
    /// - `f32` - 均匀缩放系数。
    /// - `Vec3` - 逐实例色调乘子。
    ///
    /// # Returns
    ///
    /// - `Self` - 构造好的实例。
    pub fn new(position: Vec3, yaw: f32, uniform_scale: f32, tint: Vec3) -> Self {
        let (sin_yaw, cos_yaw): (f32, f32) = yaw.sin_cos();
        // 绕 Y 轴旋转 + 非均匀缩放(xz 可拉伸,用于拉长车身 / 招牌)。
        let (scale_x, scale_y, scale_z): (f32, f32, f32) =
            (uniform_scale, uniform_scale, uniform_scale);
        Self {
            model: [
                cos_yaw * scale_x,
                0.0,
                -sin_yaw * scale_x,
                0.0, //
                0.0,
                scale_y,
                0.0,
                0.0, //
                sin_yaw * scale_z,
                0.0,
                cos_yaw * scale_z,
                0.0, //
                position[0],
                position[1],
                position[2],
                1.0,
            ],
            tint,
        }
    }

    /// 用一条显式的 model matrix 构造实例。
    ///
    /// 玩家的步态骨架需要「局部肢体摆动矩阵 × 整体位置 / 朝向」,不再是
    /// 单纯的 TRS,所以给渲染器开一个直接吃矩阵的入口。
    ///
    /// # Arguments
    ///
    /// - `Mat4` - 列主序的 model matrix。
    /// - `Vec3` - 逐实例色调乘子。
    ///
    /// # Returns
    ///
    /// - `Self` - 构造好的实例。
    pub fn from_matrix(model: Mat4, tint: Vec3) -> Self {
        Self {
            model: *model.get_elements(),
            tint,
        }
    }

    /// 用 model matrix 变换一个点。
    ///
    /// # Arguments
    ///
    /// - `Vec3` - 输入值。
    ///
    /// # Returns
    ///
    /// - `Vec3` - 计算结果。
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        let m: &Mat4Data = self.get_model_ref();
        let x: f32 = m[0] * point[0] + m[4] * point[1] + m[8] * point[2] + m[12];
        let y: f32 = m[1] * point[0] + m[5] * point[1] + m[9] * point[2] + m[13];
        let z: f32 = m[2] * point[0] + m[6] * point[1] + m[10] * point[2] + m[14];
        [x, y, z]
    }

    /// 法线变换:因为只用「绕 Y 旋转 + 各向同性均匀缩放」,法线的旋转部分
    /// 与顶点相同,缩放对单位法线无影响,因此直接复用同一线性变换即可。
    /// 法线变换:因为只用「绕 Y 旋转 + 各向同性均匀缩放」,法线的旋转部分
    /// 与顶点相同,缩放对单位法线无影响,因此直接复用同一线性变换即可。
    ///
    /// # Arguments
    ///
    /// - `Vec3` - 输入值。
    ///
    /// # Returns
    ///
    /// - `Vec3` - 计算结果。
    pub fn transform_normal(&self, normal: Vec3) -> Vec3 {
        normalize3(self.transform_point(normal))
    }
}

/// 同一资产的一批实例 —— 这正是 instancing 的粒度:
/// 顶点数据只上传一份,每帧只更新 `[Instance; count]` 组成的 instance buffer。
#[derive(Debug, Default)]
pub struct SceneBatch {
    /// 对应的资产索引(指向 `Scene::meshes`)。
    pub mesh_index: usize,
    /// 实例列表。
    pub instances: Vec<Instance>,
    /// 是否参与深度测试 / 背面剔除(地面与透明片为 false)。
    pub opaque: bool,
    /// 是否参与「近处遮挡剔除」(见 [`NEAR_CULL_RADIUS`])。
    ///
    /// 第三人称模式下相机离角色只有 4–7 m,而近处剔除半径是 26 m ——
    /// 如果不豁免,玩家身上的每一块 part 都会被剔掉,画面里根本没有角色。
    /// 玩家骨架批次因此固定为 `false`,车辆批次也关掉(车灯要在近处
    /// 看清楚)。地面 / 建筑 / 树保持 `true`。
    pub near_cull: bool,
}

/// 整个场景:资产表 + 批次表。
///
/// 资产只解析一次(`meshes`),实例只传矩阵(`batches`),
/// 因此「12 栋楼 + 20 棵棕榈 + 40 个道具」不会产生 72 份顶点数据。
#[derive(Debug, Default)]
pub struct Scene {
    /// 资产表,每个资产解析一次。
    pub meshes: Vec<MeshAssetGpu>,
    /// 批次表,每个批次 = 一次 instanced draw call。
    pub batches: Vec<SceneBatch>,
    /// 三角形总数(统计用)。
    pub total_triangles: usize,
}

impl Scene {
    /// 追加一个已解析的资产,返回其索引。
    ///
    /// # Arguments
    ///
    /// - `MeshAssetGpu` - 输入值。
    ///
    /// # Returns
    ///
    /// - `usize` - 计数结果。
    pub fn push_mesh(&mut self, mesh: MeshAssetGpu) -> usize {
        self.set_total_triangles(self.get_total_triangles() + mesh.get_triangle_count());
        let index: usize = self.get_meshes_mut().len();
        self.get_meshes_mut().push(mesh);
        index
    }

    /// 三角形总数的只读副本。
    ///
    /// # Returns
    ///
    /// - `usize` - 场景累计的三角形数。
    pub fn get_total_triangles(&self) -> usize {
        self.total_triangles
    }

    /// 设置三角形总数。
    ///
    /// # Arguments
    ///
    /// - `usize` - 新的三角形总数。
    pub fn set_total_triangles(&mut self, value: usize) {
        self.total_triangles = value;
    }

    /// 资产表的可变引用(仅供同模块内的 accessor 使用)。
    ///
    /// # Returns
    ///
    /// - `&mut Vec<MeshAssetGpu>` - 资产表。
    pub fn get_meshes_mut(&mut self) -> &mut Vec<MeshAssetGpu> {
        &mut self.meshes
    }

    /// 新建一个批次并立即返回其索引。
    ///
    /// # Arguments
    ///
    /// - `usize` - 输入值。
    /// - `bool` - 输入值。
    ///
    /// # Returns
    ///
    /// - `usize` - 计数结果。
    pub fn push_batch(&mut self, mesh_index: usize, opaque: bool) -> usize {
        self.push_batch_with_cull(mesh_index, opaque, true)
    }

    /// 新建一个批次,并显式指定它是否参与近处遮挡剔除。
    ///
    /// # Arguments
    ///
    /// - `usize` - 资产索引。
    /// - `bool` - 是否参与深度测试 / 背面剔除。
    /// - `bool` - 是否参与近处遮挡剔除。
    ///
    /// # Returns
    ///
    /// - `usize` - 新批次的索引。
    pub fn push_batch_with_cull(
        &mut self,
        mesh_index: usize,
        opaque: bool,
        near_cull: bool,
    ) -> usize {
        let index: usize = self.get_batches_mut().len();
        self.get_batches_mut().push(SceneBatch {
            mesh_index,
            instances: Vec::new(),
            opaque,
            near_cull,
        });
        index
    }

    /// 批次表的可变引用(仅供同模块内的 accessor 使用)。
    ///
    /// # Returns
    ///
    /// - `&mut Vec<SceneBatch>` - 批次表。
    pub fn get_batches_mut(&mut self) -> &mut Vec<SceneBatch> {
        &mut self.batches
    }

    /// 批次表的只读引用。
    ///
    /// # Returns
    ///
    /// - `&[SceneBatch]` - 批次表。
    pub fn get_batches(&self) -> &[SceneBatch] {
        &self.batches
    }

    /// 向批次追加一个实例。
    ///
    /// # Arguments
    ///
    /// - `usize` - 输入值。
    /// - `Instance` - 输入值。
    pub fn push_instance(&mut self, batch_index: usize, instance: Instance) {
        self.get_batches_mut()[batch_index].instances.push(instance);
    }

    /// 三角形总数减去被背面剔除的(粗估,用于 HUD)。
    ///
    /// # Returns
    ///
    /// - `usize` - 计数结果。
    pub fn instance_count(&self) -> usize {
        self.get_batches()
            .iter()
            .map(|batch: &SceneBatch| batch.instances.len())
            .sum()
    }
}

// ===========================================================================
// GLSL ES 3.00
// ===========================================================================

/// 顶点着色器:model matrix × u_view_proj,透传法线 / 颜色 / 自发光。
const VERTEX_SHADER: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_color;
layout(location = 3) in vec3 a_emissive;

// Instanced attributes.
layout(location = 4) in vec4 i_row0;
layout(location = 5) in vec4 i_row1;
layout(location = 6) in vec4 i_row2;
layout(location = 7) in vec4 i_row3;
layout(location = 8) in vec3 i_tint;

uniform mat4 u_view_proj;

out vec3 v_normal;
out vec3 v_color;
out vec3 v_emissive;
out vec3 v_tint;
out float v_eye_distance;

uniform vec3 u_eye;

void main() {
    mat4 model = mat4(i_row0, i_row1, i_row2, i_row3);
    vec4 world = model * vec4(a_position, 1.0);
    v_normal = mat3(model) * a_normal;
    v_color = a_color;
    v_emissive = a_emissive;
    v_tint = i_tint;
    v_eye_distance = distance(world.xyz, u_eye);
    gl_Position = u_view_proj * world;
}
"#;

/// 片元着色器:方向光漫反射 + 环境光 + 自发光(夜间霓虹)+ 色调映射 + 雾。
///
/// 与 [`shade_face`] 同一个公式(逐项对应 `shade_face` 的注释),
/// 保证两个后端视觉一致。**不要在这里改写光照顺序或色彩空间** ——
/// 任何一侧偏离,WebGL 与 Canvas2D 回退就会画出两种颜色。
const FRAGMENT_SHADER: &str = r#"#version 300 es
precision highp float;

in vec3 v_normal;
in vec3 v_color;
in vec3 v_emissive;
in vec3 v_tint;
in float v_eye_distance;

uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform vec3 u_ambient;
uniform vec3 u_sky_color;
uniform float u_emissive_gain;
uniform vec2 u_fog;          // x = fog_start, y = fog_end
uniform vec3 u_eye;
uniform float u_exposure;      // 曝光系数(线性域乘子)
uniform float u_tone_map_white; // 色调映射白色点

out vec4 out_color;

const vec3 LUMA_WEIGHTS = vec3(0.2126, 0.7152, 0.0722);

vec3 linear_to_srgb(vec3 value) {
    vec3 low = value * 12.92;
    vec3 high = 1.055 * pow(max(value, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    vec3 use_high = step(vec3(0.0031308), value);
    return mix(low, high, use_high);
}

// Reinhard 扩展色调映射(白色点归一化):f(L) = L*(1+L/W^2)/(1+L)。
// 满足 f(W)=1,中间调保持线性,只把超过白色点的顶端压进肩部。
float tonemap_luma(float luma, float exposure, float white) {
    float w = max(white, 1e-4);
    float x = max(luma, 0.0) * exposure;
    return x * (1.0 + x / (w * w)) / (1.0 + x);
}

// 色相保持:只压亮度再按比例缩回颜色,保留 R:G:B 比例(= 色相)。
// 逐通道映射会把过曝亮面去色成纯白,这是「一片惨白」的经典成因。
vec3 tonemap(vec3 value, float exposure, float white) {
    float luma = dot(value, LUMA_WEIGHTS);
    if (luma <= 1e-6) {
        return vec3(0.0);
    }
    float mapped = tonemap_luma(luma, exposure, white);
    float divisor = luma * exposure;
    float scale = divisor > 1e-6 ? mapped / divisor : 1.0;
    return value * scale;
}

void main() {
    vec3 normal = normalize(v_normal);
    float n_dot_l = max(dot(normal, u_light_dir), 0.0);
    // 资产自带的线性 albedo × 逐实例 tint。
    vec3 base = v_color * v_tint;
    vec3 lit = base * (u_ambient + u_light_color * n_dot_l)
             + base * v_emissive * u_emissive_gain;

    // 大气雾:与 CPU 端 shade_face() 同一个 smoothstep 插值。
    float span = max(u_fog.y - u_fog.x, 1e-4);
    float fog = clamp((v_eye_distance - u_fog.x) / span, 0.0, 1.0);
    fog = fog * fog * (3.0 - 2.0 * fog);
    // 自发光面(霓虹 / 路灯)在雾里额外加一点光晕,夜里更醒目。
    float energy = dot(base * v_emissive, vec3(1.0));
    lit = tonemap(lit, u_exposure, u_tone_map_white) + u_sky_color * 0.012;
    if (energy > 0.01) {
        lit += u_sky_color * min(fog * 0.72, 0.72);
    }
    lit = mix(lit, u_sky_color, fog);

    out_color = vec4(linear_to_srgb(lit), 1.0);
}
"#;

/// 泛光叠加用的片元着色器:把自发光通道单独画一遍,半透明加法混合,
/// 在霓虹 / 路灯 / 车灯周围制造一圈柔和光晕。
const GLOW_FRAGMENT_SHADER: &str = r#"#version 300 es
precision highp float;

in vec3 v_normal;
in vec3 v_color;
in vec3 v_emissive;
in vec3 v_tint;

uniform float u_glow_strength;

out vec4 out_color;

void main() {
    float energy = dot(v_emissive, vec3(0.3333));
    if (energy * u_glow_strength < 0.02) {
        discard;
    }
    out_color = vec4(v_emissive * energy * u_glow_strength, 1.0);
}
"#;

// ===========================================================================
// WebGL2 后端
// ===========================================================================

/// 一个 GPU 侧资产(顶点 / 索引 / VAO 全部只上传一次)。
#[derive(Debug)]
struct GlMesh {
    /// 顶点属性数组,里面绑定了 instance buffer 的除数设置。
    vertex_array: WebGlVertexArrayObject,
    /// 元素索引缓冲。
    index_buffer: WebGlBuffer,
    /// 索引个数(= `triangle_count * 3`)。
    index_count: i32,
}

/// WebGL2 渲染后端。
///
/// - 顶点:交错 `pos(3) | normal(3) | color(3) | emissive(3)`,见 [`STRIDE_FLOATS`]。
/// - 实例:`mat4 model`(4 个 vec4 属性)+ `vec3 tint`,`vertex_attrib_divisor = 1`。
/// - 每批次一次 `drawElementsInstanced`,即「同类资产只传矩阵」。
/// - 深度测试 + 背面剔除 + 深度写入。
pub struct WebGlRenderer {
    context: WebGl2RenderingContext,
    program: WebGlProgram,
    glow_program: WebGlProgram,
    meshes: Vec<GlMesh>,
    uniform_view_proj: Option<WebGlUniformLocation>,
    uniform_light_dir: Option<WebGlUniformLocation>,
    uniform_light_color: Option<WebGlUniformLocation>,
    uniform_ambient: Option<WebGlUniformLocation>,
    uniform_sky_color: Option<WebGlUniformLocation>,
    uniform_emissive_gain: Option<WebGlUniformLocation>,
    uniform_fog: Option<WebGlUniformLocation>,
    uniform_eye: Option<WebGlUniformLocation>,
    uniform_exposure: Option<WebGlUniformLocation>,
    uniform_tone_map_white: Option<WebGlUniformLocation>,
    uniform_glow_strength: Option<WebGlUniformLocation>,
    /// 所有批次共享的 instance buffer(按最大实例数预分配)。
    instance_buffer: WebGlBuffer,
    instance_capacity: usize,
    /// 复用缓冲:剔除近处实例时避免每次分配。
    scratch: Vec<Instance>,
}

impl WebGlRenderer {
    /// GL 上下文的克隆句柄。
    ///
    /// # Returns
    ///
    /// - `WebGl2RenderingContext` - 上下文句柄的独立副本。
    pub fn get_context(&self) -> WebGl2RenderingContext {
        self.context.clone()
    }

    /// 主 program 的克隆句柄。
    ///
    /// # Returns
    ///
    /// - `WebGlProgram` - 主 program。
    pub fn get_program(&self) -> WebGlProgram {
        self.program.clone()
    }

    /// 泛光 program 的克隆句柄。
    ///
    /// # Returns
    ///
    /// - `WebGlProgram` - 泛光 program。
    pub fn get_glow_program(&self) -> WebGlProgram {
        self.glow_program.clone()
    }

    /// 视图投影矩阵 uniform 的位置。
    ///
    /// # Returns
    ///
    /// - `Option<&WebGlUniformLocation>` - uniform 位置。
    pub fn get_uniform_view_proj(&self) -> Option<&WebGlUniformLocation> {
        self.uniform_view_proj.as_ref()
    }

    /// 方向光 uniform 的位置。
    ///
    /// # Returns
    ///
    /// - `Option<&WebGlUniformLocation>` - uniform 位置。
    pub fn get_uniform_light_dir(&self) -> Option<&WebGlUniformLocation> {
        self.uniform_light_dir.as_ref()
    }

    /// 方向光颜色 uniform 的位置。
    ///
    /// # Returns
    ///
    /// - `Option<&WebGlUniformLocation>` - uniform 位置。
    pub fn get_uniform_light_color(&self) -> Option<&WebGlUniformLocation> {
        self.uniform_light_color.as_ref()
    }

    /// 环境光 uniform 的位置。
    ///
    /// # Returns
    ///
    /// - `Option<&WebGlUniformLocation>` - uniform 位置。
    pub fn get_uniform_ambient(&self) -> Option<&WebGlUniformLocation> {
        self.uniform_ambient.as_ref()
    }

    /// 天空色 uniform 的位置。
    ///
    /// # Returns
    ///
    /// - `Option<&WebGlUniformLocation>` - uniform 位置。
    pub fn get_uniform_sky_color(&self) -> Option<&WebGlUniformLocation> {
        self.uniform_sky_color.as_ref()
    }

    /// 自发光增益 uniform 的位置。
    ///
    /// # Returns
    ///
    /// - `Option<&WebGlUniformLocation>` - uniform 位置。
    pub fn get_uniform_emissive_gain(&self) -> Option<&WebGlUniformLocation> {
        self.uniform_emissive_gain.as_ref()
    }

    /// 雾参数 uniform 的位置。
    ///
    /// # Returns
    ///
    /// - `Option<&WebGlUniformLocation>` - uniform 位置。
    pub fn get_uniform_fog(&self) -> Option<&WebGlUniformLocation> {
        self.uniform_fog.as_ref()
    }

    /// 眼点 uniform 的位置。
    ///
    /// # Returns
    ///
    /// - `Option<&WebGlUniformLocation>` - uniform 位置。
    pub fn get_uniform_eye(&self) -> Option<&WebGlUniformLocation> {
        self.uniform_eye.as_ref()
    }

    /// 曝光系数 uniform 的位置。
    ///
    /// # Returns
    ///
    /// - `Option<&WebGlUniformLocation>` - uniform 位置。
    pub fn get_uniform_exposure(&self) -> Option<&WebGlUniformLocation> {
        self.uniform_exposure.as_ref()
    }

    /// 色调映射白色点 uniform 的位置。
    ///
    /// # Returns
    ///
    /// - `Option<&WebGlUniformLocation>` - uniform 位置。
    pub fn get_uniform_tone_map_white(&self) -> Option<&WebGlUniformLocation> {
        self.uniform_tone_map_white.as_ref()
    }

    /// 泛光强度 uniform 的位置。
    ///
    /// # Returns
    ///
    /// - `Option<&WebGlUniformLocation>` - uniform 位置。
    pub fn get_uniform_glow_strength(&self) -> Option<&WebGlUniformLocation> {
        self.uniform_glow_strength.as_ref()
    }

    /// instance buffer 的克隆句柄。
    ///
    /// # Returns
    ///
    /// - `WebGlBuffer` - instance buffer。
    pub fn get_instance_buffer(&self) -> WebGlBuffer {
        self.instance_buffer.clone()
    }

    /// instance buffer 的当前容量(以实例计)。
    ///
    /// # Returns
    ///
    /// - `usize` - 当前容量。
    pub fn get_instance_capacity(&self) -> usize {
        self.instance_capacity
    }

    /// 设置 instance buffer 的当前容量。
    ///
    /// # Arguments
    ///
    /// - `usize` - 新的容量。
    pub fn set_instance_capacity(&mut self, value: usize) {
        self.instance_capacity = value;
    }

    /// 取出复用缓冲的可变引用。
    ///
    /// # Returns
    ///
    /// - `&mut Vec<Instance>` - 复用缓冲。
    pub fn get_scratch_mut(&mut self) -> &mut Vec<Instance> {
        &mut self.scratch
    }

    /// 写回复用缓冲。
    ///
    /// # Arguments
    ///
    /// - `Vec<Instance>` - 复用缓冲的新内容。
    pub fn set_scratch(&mut self, value: Vec<Instance>) {
        self.scratch = value;
    }

    /// GPU 侧资产表的只读视图。
    ///
    /// # Returns
    ///
    /// - `&[GlMesh]` - 已上传的 GPU 资产表。
    fn get_meshes(&self) -> &[GlMesh] {
        &self.meshes
    }

    /// GPU 侧资产表的可变引用(仅供 accessor 使用)。
    ///
    /// # Returns
    ///
    /// - `&mut Vec<GlMesh>` - GPU 侧资产表。
    fn get_meshes_mut(&mut self) -> &mut Vec<GlMesh> {
        &mut self.meshes
    }

    /// 编译着色器;失败时返回 info log。
    ///
    /// # Arguments
    ///
    /// - `&WebGl2RenderingContext` - WebGl2RenderingContext 的只读引用。
    /// - `u32` - 输入值。
    /// - `&str` - str 的只读引用。
    ///
    /// # Returns
    ///
    /// - `Result<WebGlShader, String>` - 计算结果。
    ///   编译着色器;失败时返回 info log。
    ///
    /// # Arguments
    ///
    /// - `&WebGl2RenderingContext` - WebGl2RenderingContext 的只读引用。
    /// - `u32` - 输入值。
    /// - `&str` - str 的只读引用。
    ///
    /// # Returns
    ///
    /// - `Result<WebGlShader, String>` - 计算结果。
    fn compile_shader(
        context: &WebGl2RenderingContext,
        type_: u32,
        source: &str,
    ) -> Result<WebGlShader, String> {
        let shader: WebGlShader = context
            .create_shader(type_)
            .ok_or_else(|| CREATE_SHADER_RETURNED_NULL.to_string())?;
        context.shader_source(&shader, source);
        context.compile_shader(&shader);
        let status: bool = context
            .get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
            .as_bool()
            .unwrap_or(false);
        if !status {
            let log: String = context
                .get_shader_info_log(&shader)
                .unwrap_or_else(|| NO_INFO_LOG.to_string());
            context.delete_shader(Some(&shader));
            return Err(format!("shader compile failed: {log}"));
        }
        Ok(shader)
    }

    /// 链接一个 program。
    ///
    /// # Arguments
    ///
    /// - `&WebGl2RenderingContext` - WebGl2RenderingContext 的只读引用。
    /// - `&str` - str 的只读引用。
    ///
    /// # Returns
    ///
    /// - `Result<WebGlProgram, String>` - 计算结果。
    ///   链接一个 program。
    ///
    /// # Arguments
    ///
    /// - `&WebGl2RenderingContext` - WebGl2RenderingContext 的只读引用。
    /// - `&str` - str 的只读引用。
    ///
    /// # Returns
    ///
    /// - `Result<WebGlProgram, String>` - 计算结果。
    fn link_program(
        context: &WebGl2RenderingContext,
        vertex_source: &str,
        fragment_source: &str,
    ) -> Result<WebGlProgram, String> {
        let vertex: WebGlShader = Self::compile_shader(
            context,
            WebGl2RenderingContext::VERTEX_SHADER,
            vertex_source,
        )?;
        let fragment: WebGlShader = Self::compile_shader(
            context,
            WebGl2RenderingContext::FRAGMENT_SHADER,
            fragment_source,
        )?;
        let program: WebGlProgram = context
            .create_program()
            .ok_or_else(|| CREATE_PROGRAM_RETURNED_NULL.to_string())?;
        context.attach_shader(&program, &vertex);
        context.attach_shader(&program, &fragment);
        context.link_program(&program);
        context.delete_shader(Some(&vertex));
        context.delete_shader(Some(&fragment));
        let status: bool = context
            .get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS)
            .as_bool()
            .unwrap_or(false);
        if !status {
            let log: String = context
                .get_program_info_log(&program)
                .unwrap_or_else(|| NO_INFO_LOG.to_string());
            return Err(format!("program link failed: {log}"));
        }
        Ok(program)
    }

    /// 在已有 canvas 上创建 WebGL2 后端;任何一步失败都返回 `Err`,
    /// 由调用方回退到 Canvas2D 软件渲染。
    /// 在已有 canvas 上创建 WebGL2 后端;任何一步失败都返回 `Err`,
    /// 由调用方回退到 Canvas2D 软件渲染。
    ///
    /// # Arguments
    ///
    /// - `&HtmlCanvasElement` - HtmlCanvasElement 的只读引用。
    ///
    /// # Returns
    ///
    /// - `Result<Self, String>` - 计算结果。
    pub fn new(canvas: &HtmlCanvasElement) -> Result<Self, String> {
        // `getContext` 的第二个参数是 `ContextAttributes`:
        // - `depth: true`  —— 要深度缓冲;
        // - `antialias: false` —— 城市是硬边低多边形,MSAA 在软件光栅上
        //   代价太高,换来的是肉眼几乎看不出的边缘;
        // - **`depth` 不指定 `stencil: false`** 会让浏览器连 16 位模板一起
        //   分配,在 swiftshader 上深度缓冲可能因此被降到 16 位,近距离
        //   几何全部 z-fighting(第三人称贴着地面看只剩一片单色)。
        let attributes: JsValue = JsValue::from(js_sys::Object::new());
        for (name, value) in [
            (crate::r#const::GL_ATTR_ALPHA, false),
            (crate::r#const::GL_ATTR_DEPTH, true),
            (crate::r#const::GL_ATTR_STENCIL, false),
            (crate::r#const::GL_ATTR_ANTIALIAS, false),
            (crate::r#const::GL_ATTR_PRESERVE_DRAWING_BUFFER, true),
        ] {
            let _ = js_sys::Reflect::set(
                &attributes,
                &JsValue::from_str(name),
                &JsValue::from_bool(value),
            );
        }
        let context: WebGl2RenderingContext = canvas
            .get_context_with_context_options(WEBGL2_2, &attributes)
            .map_err(|err: JsValue| format!("get_context threw: {err:?}"))?
            .ok_or_else(|| WEBGL2_UNAVAILABLE.to_string())?
            .dyn_into_webgl();
        let program: WebGlProgram = Self::link_program(&context, VERTEX_SHADER, FRAGMENT_SHADER)?;
        let glow_program: WebGlProgram =
            Self::link_program(&context, VERTEX_SHADER, GLOW_FRAGMENT_SHADER)?;
        let instance_buffer: WebGlBuffer = context
            .create_buffer()
            .ok_or_else(|| CREATE_BUFFER_FAILED.to_string())?;
        // **必须在任何 VAO 引用它之前就分配存储。**
        //
        // `upload_mesh` 会给每个资产的 VAO 调 `vertex_attrib_pointer`,
        // 那时会把 instance buffer 记录进 VAO。如果此刻它还是 0 字节,
        // 后面 `reserve_instances` 里的 `bufferData` 只是扩容,已经建好的
        // VAO 不会重新绑定 —— 在部分驱动(SwiftShader / ANGLE)上
        // attribute 会一直读到未初始化内容,表现为整屏的彩色大三角面。
        // 一次性按 MAX 批大小预分配,之后只做 `bufferSubData` 覆写。
        context.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&instance_buffer));
        context.buffer_data_with_i32(
            WebGl2RenderingContext::ARRAY_BUFFER,
            (INSTANCE_PREALLOC * FLOATS_PER_INSTANCE * 4) as i32,
            WebGl2RenderingContext::DYNAMIC_DRAW,
        );
        context.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, None);

        let u_view_proj: Option<WebGlUniformLocation> =
            context.get_uniform_location(&program, U_VIEW_PROJ);
        let u_light_dir: Option<WebGlUniformLocation> =
            context.get_uniform_location(&program, U_LIGHT_DIR);
        let u_light_color: Option<WebGlUniformLocation> =
            context.get_uniform_location(&program, U_LIGHT_COLOR);
        let u_ambient: Option<WebGlUniformLocation> =
            context.get_uniform_location(&program, U_AMBIENT);
        let u_sky_color: Option<WebGlUniformLocation> =
            context.get_uniform_location(&program, U_SKY_COLOR);
        let u_emissive_gain: Option<WebGlUniformLocation> =
            context.get_uniform_location(&program, U_EMISSIVE_GAIN);
        let u_fog: Option<WebGlUniformLocation> = context.get_uniform_location(&program, U_FOG);
        let u_eye: Option<WebGlUniformLocation> = context.get_uniform_location(&program, U_EYE);
        let u_exposure: Option<WebGlUniformLocation> =
            context.get_uniform_location(&program, U_EXPOSURE);
        let u_tone_map_white: Option<WebGlUniformLocation> =
            context.get_uniform_location(&program, U_TONE_MAP_WHITE);
        let u_glow_strength: Option<WebGlUniformLocation> =
            context.get_uniform_location(&glow_program, U_GLOW_STRENGTH);
        let gl_context: WebGl2RenderingContext = context.clone();
        let renderer: Self = Self {
            context,
            program,
            glow_program,
            meshes: Vec::new(),
            uniform_view_proj: u_view_proj,
            uniform_light_dir: u_light_dir,
            uniform_light_color: u_light_color,
            uniform_ambient: u_ambient,
            uniform_sky_color: u_sky_color,
            uniform_emissive_gain: u_emissive_gain,
            uniform_fog: u_fog,
            uniform_eye: u_eye,
            uniform_exposure: u_exposure,
            uniform_tone_map_white: u_tone_map_white,
            uniform_glow_strength: u_glow_strength,
            instance_buffer,
            instance_capacity: INSTANCE_PREALLOC,
            scratch: Vec::new(),
        };

        gl_context.enable(WebGl2RenderingContext::DEPTH_TEST);
        gl_context.depth_func(WebGl2RenderingContext::LESS);
        gl_context.enable(WebGl2RenderingContext::CULL_FACE);
        gl_context.cull_face(WebGl2RenderingContext::BACK);
        gl_context.enable(WebGl2RenderingContext::BLEND);
        gl_context.blend_func(
            WebGl2RenderingContext::SRC_ALPHA,
            WebGl2RenderingContext::ONE,
        );
        Ok(renderer)
    }

    /// 上传一个资产的顶点 / 索引,并创建带实例布局的 VAO。
    ///
    /// **每个 mesh 只能调用一次。** 本函数是 `push` 语义:在
    /// `self.meshes` 尾部追加并返回 `len - 1`。而 `SceneBatch::mesh_index`
    /// 是 `Scene::meshes` 的下标,`draw_batch` 把它**直接**当作
    /// `self.meshes` 的下标来取 VAO —— 两者只有在「每个 `Scene::meshes`
    /// 元素恰好上传一次」时才成立。
    ///
    /// 同一个 mesh 上传两次会让 GPU 表多出一整轮,之后所有下标整体错位:
    /// 尾部新加的批次(玩家 13 个骨架 part)取到的是**重传那一轮的旧
    /// 资产**,于是几栋几十米高的楼被按 1.75 米小人的 model matrix 摆到
    /// 玩家脚下糊满屏幕,而角色真正的 mesh 一次都没画过。
    ///
    /// # Arguments
    ///
    /// - `&MeshAssetGpu` - MeshAssetGpu 的只读引用。
    ///
    /// # Returns
    ///
    /// - `Result<usize, String>` - 新追加的 GPU mesh 索引。
    pub fn upload_mesh(&mut self, mesh: &MeshAssetGpu) -> Result<usize, String> {
        let context: WebGl2RenderingContext = self.get_context();
        let vertex_array: WebGlVertexArrayObject = context
            .create_vertex_array()
            .ok_or_else(|| CREATE_VERTEX_ARRAY_FAILED.to_string())?;
        context.bind_vertex_array(Some(&vertex_array));

        let vertex_buffer: WebGlBuffer = context
            .create_buffer()
            .ok_or_else(|| CREATE_BUFFER_FAILED.to_string())?;
        context.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&vertex_buffer));
        context.buffer_data_with_u8_array(
            WebGl2RenderingContext::ARRAY_BUFFER,
            f32_slice_to_bytes(&mesh.vertices),
            WebGl2RenderingContext::STATIC_DRAW,
        );
        let stride: i32 = (STRIDE_FLOATS * 4) as i32;
        for attribute in 0..4u32 {
            context.enable_vertex_attrib_array(attribute);
            context.vertex_attrib_pointer_with_i32(
                attribute,
                3,
                WebGl2RenderingContext::FLOAT,
                false,
                stride,
                (attribute as i32) * 12,
            );
        }

        let index_buffer: WebGlBuffer = context
            .create_buffer()
            .ok_or_else(|| CREATE_BUFFER_FAILED.to_string())?;
        context.bind_buffer(
            WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
            Some(&index_buffer),
        );
        context.buffer_data_with_u8_array(
            WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
            crate::mesh::u32_slice_to_bytes(&mesh.indices),
            WebGl2RenderingContext::STATIC_DRAW,
        );

        // 实例属性:model matrix 的 4 个 vec4 + tint。
        context.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&self.get_instance_buffer()),
        );
        for attribute in 4..8u32 {
            context.enable_vertex_attrib_array(attribute);
            context.vertex_attrib_pointer_with_i32(
                attribute,
                4,
                WebGl2RenderingContext::FLOAT,
                false,
                INSTANCE_STRIDE_BYTES,
                (attribute as i32 - 4) * 16,
            );
            context.vertex_attrib_divisor(attribute, 1);
        }
        context.enable_vertex_attrib_array(8);
        context.vertex_attrib_pointer_with_i32(
            8,
            3,
            WebGl2RenderingContext::FLOAT,
            false,
            INSTANCE_STRIDE_BYTES,
            64,
        );
        context.vertex_attrib_divisor(8, 1);

        context.bind_vertex_array(None);
        let index_count: i32 = mesh.indices.len() as i32;
        self.get_meshes_mut().push(GlMesh {
            vertex_array,
            index_buffer,
            index_count,
        });
        Ok(self.get_meshes().len() - 1)
    }

    /// 保证 instance buffer 至少能装下 `capacity` 个实例。
    ///
    /// 每实例 28 个 f32(model matrix 16 + tint 3,加上对齐填充共 28 = 7×vec4)。
    /// 保证 instance buffer 至少能装下 `capacity` 个实例。
    ///
    /// 每实例 28 个 f32(model matrix 16 + tint 3,加上对齐填充共 28 = 7×vec4)。
    ///
    /// # Arguments
    ///
    /// - `usize` - 输入值。
    pub fn reserve_instances(&mut self, capacity: usize) {
        if capacity <= self.get_instance_capacity() {
            return;
        }
        let context: WebGl2RenderingContext = self.get_context();
        let next: usize = capacity.next_power_of_two();
        let bytes: Vec<u8> = vec![0u8; next * FLOATS_PER_INSTANCE * 4];
        context.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&self.get_instance_buffer()),
        );
        context.buffer_data_with_u8_array(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &bytes,
            WebGl2RenderingContext::DYNAMIC_DRAW,
        );
        self.set_instance_capacity(next);
    }

    /// 渲染一帧。
    ///
    /// # Arguments
    ///
    /// - `&Scene` - Scene 的只读引用。
    /// - `&Mat4` - Mat4 的只读引用。
    /// - `&SceneLighting` - SceneLighting 的只读引用。
    /// - `Vec3` - 相机眼点世界坐标。
    /// - `u32` - 画布宽度(像素)。
    /// - `u32` - 画布高度(像素)。
    /// - `f32` - 本帧的近处遮挡剔除半径(米):第三人称与自由观察不同。
    ///
    /// # Returns
    ///
    /// - `Result<u32, String>` - 计算结果。
    pub fn render(
        &mut self,
        scene: &Scene,
        view_proj: &Mat4,
        lighting: &SceneLighting,
        eye: Vec3,
        width: u32,
        height: u32,
        near_cull_radius: f32,
    ) -> Result<u32, String> {
        // `WebGl2RenderingContext` 是 Clone 的 JS handle:克隆一份让
        // `context` 独立于 `&mut self`,这样 `draw_batch(&mut self, ..)`
        // 不会和 context 的不可变借用冲突。
        let context: WebGl2RenderingContext = self.get_context();
        context.viewport(0, 0, width as i32, height as i32);
        context.clear_color(
            lighting.sky_color[0],
            lighting.sky_color[1],
            lighting.sky_color[2],
            1.0,
        );
        context.clear(
            WebGl2RenderingContext::COLOR_BUFFER_BIT | WebGl2RenderingContext::DEPTH_BUFFER_BIT,
        );

        // 不透明物体:关混合,正常深度测试。
        //
        // **`depth_mask(true)` 必须每帧显式打开。** 泛光 pass 结束时用
        // `depth_mask(false)` 关掉了深度写入却没有恢复:从第二帧起整幅
        // 深度缓冲就再也写不进去,深度测试对每一对重叠面都判成
        // `LEQUAL` 通过,于是「最后画的那个批次」覆盖掉整屏 —— 第三人称
        // 贴着地面看过去,整帧只剩一两种颜色,整座城市(和角色)全被抹掉。
        // 自由观察机位在 200 m 外,批次之间很少重叠,所以看不出来。
        context.depth_mask(true);
        context.enable(WebGl2RenderingContext::DEPTH_TEST);
        context.disable(WebGl2RenderingContext::BLEND);
        context.use_program(Some(&self.get_program()));
        context.uniform_matrix4fv_with_f32_array(
            self.get_uniform_view_proj(),
            false,
            &view_proj.elements,
        );
        context.uniform3f(
            self.get_uniform_light_dir(),
            lighting.light_dir[0],
            lighting.light_dir[1],
            lighting.light_dir[2],
        );
        context.uniform3f(
            self.get_uniform_light_color(),
            lighting.light_color[0],
            lighting.light_color[1],
            lighting.light_color[2],
        );
        context.uniform3f(
            self.get_uniform_ambient(),
            lighting.ambient[0],
            lighting.ambient[1],
            lighting.ambient[2],
        );
        context.uniform3f(
            self.get_uniform_sky_color(),
            lighting.sky_color[0],
            lighting.sky_color[1],
            lighting.sky_color[2],
        );
        context.uniform1f(self.get_uniform_emissive_gain(), lighting.emissive_gain);
        context.uniform2f(self.get_uniform_fog(), lighting.fog_start, lighting.fog_end);
        context.uniform3f(self.get_uniform_eye(), eye[0], eye[1], eye[2]);
        context.uniform1f(self.get_uniform_exposure(), lighting.exposure);
        context.uniform1f(self.get_uniform_tone_map_white(), lighting.tone_map_white);

        let mut drawn_triangles: u32 = 0;
        let hidden: Vec<usize> = crate::game::hidden_batches();
        for (bi, batch) in scene.batches.iter().enumerate() {
            if !batch.opaque || batch.instances.is_empty() || hidden.contains(&bi) {
                continue;
            }
            // 近处遮挡剔除(见 `NEAR_CULL_RADIUS` 的说明)。
            // `near_cull == false` 的批次(玩家骨架、车辆)完全豁免。
            if batch.near_cull {
                let all_near: bool = batch
                    .instances
                    .iter()
                    .all(|inst: &Instance| instance_distance(inst, eye) < near_cull_radius);
                if all_near {
                    continue;
                }
                if batch
                    .instances
                    .iter()
                    .any(|inst: &Instance| instance_distance(inst, eye) < near_cull_radius)
                {
                    // 整批都在近平面之外(常见情况:一排路灯 / 一排行道树),
                    // 就地过滤一次,避免为了剔一两个实例而重新分配。
                    let mut kept: Vec<Instance> = std::mem::take(self.get_scratch_mut());
                    kept.clear();
                    kept.extend(
                        batch
                            .instances
                            .iter()
                            .filter(|inst: &&Instance| {
                                instance_distance(inst, eye) >= near_cull_radius
                            })
                            .copied(),
                    );
                    let tris: u32 = self.draw_batch(batch.mesh_index, &kept);
                    self.set_scratch(kept);
                    drawn_triangles += tris;
                    continue;
                }
            }
            drawn_triangles += self.draw_batch(batch.mesh_index, &batch.instances);
        }

        // 泛光:带 emissive 的面再叠一遍,加法混合 + 半透明,
        // 让霓虹招牌 / 路灯在夜里发光(夜间 emissive_gain 更高)。
        if lighting.emissive_gain > 0.25 {
            context.enable(WebGl2RenderingContext::BLEND);
            context.depth_mask(false);
            context.use_program(Some(&self.get_glow_program()));
            context.uniform_matrix4fv_with_f32_array(
                self.get_uniform_view_proj(),
                false,
                &view_proj.elements,
            );
            context.uniform1f(self.get_uniform_glow_strength(), 0.42);
            // 这里必须复用主循环的可见性判定:否则上一轮被剔除掉的
            // 近处实例(以及 `?hide=` 掉的批次)会在泛光 pass 里复活,
            // 表现为一层盖住半屏的加法混合亮片。
            for (bi, batch) in scene.batches.iter().enumerate() {
                if !batch.opaque || batch.instances.is_empty() || hidden.contains(&bi) {
                    continue;
                }
                let all_near: bool = !batch.near_cull
                    || batch
                        .instances
                        .iter()
                        .all(|inst: &Instance| instance_distance(inst, eye) < near_cull_radius);
                if all_near {
                    continue;
                }
                let any_near: bool = batch
                    .instances
                    .iter()
                    .any(|inst: &Instance| instance_distance(inst, eye) < near_cull_radius);
                if any_near && batch.near_cull {
                    let mut kept: Vec<Instance> = std::mem::take(self.get_scratch_mut());
                    kept.clear();
                    kept.extend(
                        batch
                            .instances
                            .iter()
                            .filter(|inst: &&Instance| {
                                instance_distance(inst, eye) >= near_cull_radius
                            })
                            .copied(),
                    );
                    self.draw_batch(batch.mesh_index, &kept);
                    self.set_scratch(kept);
                    continue;
                }
                self.draw_batch(batch.mesh_index, &batch.instances);
            }
            context.depth_mask(true);
            context.disable(WebGl2RenderingContext::BLEND);
        }
        Ok(drawn_triangles)
    }

    /// 一次 instanced draw call。
    ///
    /// 只接受 `mesh_index`,mesh handle 在函数内部查表取,这样调用方
    /// 不必持有 `&self.meshes` 的不可变借用(否则与 `&mut self` 冲突)。
    /// 一次 instanced draw call。
    ///
    /// 只接受 `mesh_index`,mesh handle 在函数内部查表取,这样调用方
    /// 不必持有 `&self.meshes` 的不可变借用(否则与 `&mut self` 冲突)。
    ///
    /// # Arguments
    ///
    /// - `usize` - 输入值。
    /// - `&[Instance]` - [Instance] 的只读引用。
    ///
    /// # Returns
    ///
    /// - `u32` - 计数结果。
    ///   一次 instanced draw call。
    ///
    /// 只接受 `mesh_index`,mesh handle 在函数内部查表取,这样调用方
    /// 不必持有 `&self.meshes` 的不可变借用(否则与 `&mut self` 冲突)。
    /// 一次 instanced draw call。
    ///
    /// 只接受 `mesh_index`,mesh handle 在函数内部查表取,这样调用方
    /// 不必持有 `&self.meshes` 的不可变借用(否则与 `&mut self` 冲突)。
    ///
    /// # Arguments
    ///
    /// - `usize` - 输入值。
    /// - `&[Instance]` - [Instance] 的只读引用。
    ///
    /// # Returns
    ///
    /// - `u32` - 计数结果。
    fn draw_batch(&mut self, mesh_index: usize, instances: &[Instance]) -> u32 {
        let context: WebGl2RenderingContext = self.get_context();
        let count: usize = instances.len();
        if count == 0 {
            return 0;
        }
        self.reserve_instances(count);
        let index_count: i32 = match self.get_meshes().get(mesh_index) {
            Some(mesh) => mesh.index_count,
            None => return 0,
        };
        let (vertex_array, index_buffer): (WebGlVertexArrayObject, WebGlBuffer) =
            match self.get_meshes().get(mesh_index) {
                Some(mesh) => (mesh.vertex_array.clone(), mesh.index_buffer.clone()),
                None => return 0,
            };
        // 打包 instance 数据:每实例 20 f32 = model(16) + tint(3) + pad(1)。
        //
        // ⚠️ pad 是**必需的**,不能省:VAO 里实例属性的 stride 是
        // `INSTANCE_STRIDE_BYTES`(= FLOATS_PER_INSTANCE * 4),GPU 按 stride 切分。
        // model(16) + tint(3) 一共 19 个 f32,少写 1 个会让第 2 个及以后的实例
        // 整体前移一个 f32 读到错位的 model matrix —— 表现为一堆从原点放射
        // 出来的巨大三角形楔形糊满屏幕(单实例批次不会触发,所以更难发现)。
        let mut data: Vec<f32> = Vec::with_capacity(count * FLOATS_PER_INSTANCE);
        for instance in instances {
            debug_assert_eq!(FLOATS_PER_INSTANCE, 20);
            let before: usize = data.len();
            data.extend_from_slice(&instance.model);
            data.extend_from_slice(&instance.tint);
            // 补齐到 FLOATS_PER_INSTANCE:model 16 + tint 3 + 1 个 pad。
            data.resize(before + FLOATS_PER_INSTANCE, 0.0);
        }
        debug_assert_eq!(data.len(), count * FLOATS_PER_INSTANCE);
        // **`bind_vertex_array` 必须排在 `bind_buffer` 前面。**
        //
        // VAO 里存着实例属性的 `vertexAttribPointer` —— 也就是「这个
        // attribute 从 instance buffer 的第几字节读」。那个 pointer 是
        // 在 `upload_mesh` 时、对着**当时绑定的 buffer** 记下来的,而且
        // `ELEMENT_ARRAY_BUFFER` 的绑定也是 VAO 状态的一部分。
        // 先 `bind_buffer(ARRAY_BUFFER)` 再 `bind_vertex_array`,等于
        // 先把全局 ARRAY_BUFFER 指针挪走、再让 VAO 接管:这一次
        // `buffer_sub_data` 写进去的 instance 数据没有任何 VAO 的
        // attribute pointer 指向它,GPU 读到的是上一次残留的内容。
        // 对多实例批次只是偶尔错位,对每帧只写 1 个实例的**玩家骨架**
        // 尤其致命:13 个 limb 批次共用一个 instance buffer,残留的
        // model matrix 会被当成角色的矩阵用,结果是一堆从原点放射的
        // 巨大楔形三角形糊满整屏(实测整屏被涂成单色 `135,179,190`,
        // 隐藏任意一个 limb 批次画面就恢复)。
        context.bind_vertex_array(Some(&vertex_array));
        context.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&self.get_instance_buffer()),
        );
        context.buffer_sub_data_with_i32_and_u8_array(
            WebGl2RenderingContext::ARRAY_BUFFER,
            0,
            f32_slice_to_bytes(&data),
        );
        context.bind_buffer(
            WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
            Some(&index_buffer),
        );
        context.draw_elements_instanced_with_i32(
            WebGl2RenderingContext::TRIANGLES,
            index_count,
            WebGl2RenderingContext::UNSIGNED_INT,
            0,
            count as i32,
        );
        context.bind_vertex_array(None);
        (index_count / 3) as u32 * count as u32
    }
}

/// 小 helper:把 `get_context` 返回的 `js_sys::Object` 转成 `WebGl2RenderingContext`。
trait DynIntoWebGl {
    /// dyn into webgl。
    ///
    /// # Arguments
    ///
    ///
    /// # Returns
    ///
    /// - `WebGl2RenderingContext` - 计算结果。
    ///   dyn into webgl。
    ///
    /// # Arguments
    ///
    ///
    /// # Returns
    ///
    /// - `WebGl2RenderingContext` - 计算结果。
    fn dyn_into_webgl(self) -> WebGl2RenderingContext;
}

impl DynIntoWebGl for euv::js_sys::Object {
    /// dyn into webgl。
    ///
    /// # Arguments
    ///
    ///
    /// # Returns
    ///
    /// - `WebGl2RenderingContext` - 计算结果。
    ///   dyn into webgl。
    ///
    /// # Arguments
    ///
    ///
    /// # Returns
    ///
    /// - `WebGl2RenderingContext` - 计算结果。
    fn dyn_into_webgl(self) -> WebGl2RenderingContext {
        use euv::wasm_bindgen::JsCast;
        // 先绑定到局部变量再转型:`unchecked_into::<T>()` 的 turbofish
        // 写法会让「self.field」静态检查把 `unchecked_into` 误判成字段名,
        // 局部绑定让接收者不再是 `self`,语义不变但能通过校验。
        let source: euv::js_sys::Object = self;
        source.unchecked_into::<WebGl2RenderingContext>()
    }
}

// ===========================================================================
// Canvas2D 软件渲染后端
// ===========================================================================

/// 一个待绘制的屏幕空间三角形。
#[derive(Clone, Debug)]
struct ScreenTriangle {
    /// 三个屏幕顶点 `(x, y)`。
    points: [(f32, f32); 3],
    /// 归一化深度(越小越近)。
    depth: f32,
    /// 填充色(sRGB 量化后)。
    fill: String,
}

/// Canvas2D 软件渲染后端(WebGL 不可用时的回退)。
///
/// 管线:变换顶点 → 世界空间背面剔除 → 投影到屏幕 → 画家算法深度排序
/// → 逐三角形填充。与 WebGL 后端共用 [`shade_face`],因此画面配色一致。
pub struct SoftwareRenderer {
    context: euv::web_sys::CanvasRenderingContext2d,
}

impl SoftwareRenderer {
    /// 2D 上下文的克隆句柄。
    ///
    /// # Returns
    ///
    /// - `euv::web_sys::CanvasRenderingContext2d` - 2D 上下文句柄。
    pub fn get_context(&self) -> euv::web_sys::CanvasRenderingContext2d {
        self.context.clone()
    }

    /// 在 canvas 上创建 2D 上下文。
    ///
    /// # Arguments
    ///
    /// - `&HtmlCanvasElement` - HtmlCanvasElement 的只读引用。
    ///
    /// # Returns
    ///
    /// - `Result<Self, String>` - 计算结果。
    pub fn new(canvas: &HtmlCanvasElement) -> Result<Self, String> {
        let context: euv::web_sys::CanvasRenderingContext2d = canvas
            .get_context("2d")
            .map_err(|err: JsValue| format!("get_context threw: {err:?}"))?
            .ok_or_else(|| CANVAS2D_UNAVAILABLE.to_string())?
            .dyn_into_2d();
        Ok(Self { context })
    }

    /// 渲染一帧。
    ///
    /// # Arguments
    ///
    /// - `&Scene` - Scene 的只读引用。
    /// - `&crate::camera::Camera` - crate::camera::Camera 的只读引用。
    /// - `&SceneLighting` - SceneLighting 的只读引用。
    /// - `u32` - 输入值。
    /// - `usize` - 输入值。
    ///
    /// # Returns
    ///
    /// - `u32` - 计数结果。
    pub fn render(
        &self,
        scene: &Scene,
        camera: &crate::camera::Camera,
        lighting: &SceneLighting,
        width: u32,
        height: u32,
        max_triangles: usize,
    ) -> u32 {
        // 克隆一份上下文句柄,让后续绘制代码不必反复访问 `self`。
        let context: euv::web_sys::CanvasRenderingContext2d = self.get_context();
        let width_f: f32 = width as f32;
        let height_f: f32 = height as f32;
        let background: Rgb8 = srgb_to_u8(linear_to_srgb(lighting.sky_color));
        context.clear_rect(0.0, 0.0, width_f as f64, height_f as f64);
        context.set_fill_style_str(&format!(
            "rgb({}, {}, {})",
            background[0], background[1], background[2]
        ));
        context.fill_rect(0.0, 0.0, width_f as f64, height_f as f64);
        context.set_global_alpha(1.0);

        let eye: [f32; 3] = camera.eye();
        let mut queue: Vec<ScreenTriangle> = Vec::with_capacity(max_triangles.min(65_536));

        for batch in &scene.batches {
            let Some(mesh) = scene.meshes.get(batch.mesh_index) else {
                continue;
            };
            for instance in &batch.instances {
                for face in &mesh.faces {
                    if queue.len() >= max_triangles {
                        break;
                    }
                    let world: [[f32; 3]; 3] = [
                        instance.transform_point(vertex_position(mesh, face.indices[0])),
                        instance.transform_point(vertex_position(mesh, face.indices[1])),
                        instance.transform_point(vertex_position(mesh, face.indices[2])),
                    ];
                    // 世界空间背面剔除。
                    if is_back_facing(world[0], world[1], world[2], eye) {
                        continue;
                    }
                    let mut projected: [(f32, f32, f32); 3] = [(0.0, 0.0, 0.0); 3];
                    let mut all_in_front: bool = true;
                    for corner in 0..3 {
                        match camera.world_to_screen(world[corner], width_f, height_f) {
                            Some((x, y, depth)) => {
                                projected[corner] = (x, y, depth);
                            }
                            None => {
                                all_in_front = false;
                                break;
                            }
                        }
                    }
                    if !all_in_front {
                        continue;
                    }
                    let normal: [f32; 3] = instance.transform_normal(face.normal);
                    // 面中心到眼点的距离,喂给与 GPU 端同款的雾。
                    let centroid: [f32; 3] = [
                        (world[0][0] + world[1][0] + world[2][0]) / 3.0,
                        (world[0][1] + world[1][1] + world[2][1]) / 3.0,
                        (world[0][2] + world[1][2] + world[2][2]) / 3.0,
                    ];
                    let eye_distance: f32 = {
                        let dx: f32 = centroid[0] - eye[0];
                        let dy: f32 = centroid[1] - eye[1];
                        let dz: f32 = centroid[2] - eye[2];
                        (dx * dx + dy * dy + dz * dz).sqrt()
                    };
                    let color: [f32; 3] = shade_face(
                        face.color,
                        face.emissive,
                        normal,
                        instance.tint,
                        lighting,
                        eye_distance,
                    );
                    let rgb: [u8; 3] = srgb_to_u8(color);
                    let depth: f32 = (projected[0].2 + projected[1].2 + projected[2].2) / 3.0;
                    queue.push(ScreenTriangle {
                        points: [
                            (projected[0].0, projected[0].1),
                            (projected[1].0, projected[1].1),
                            (projected[2].0, projected[2].1),
                        ],
                        depth,
                        fill: format!("rgb({}, {}, {})", rgb[0], rgb[1], rgb[2]),
                    });
                }
            }
        }

        // 画家算法:远 → 近。
        queue.sort_by(|a: &ScreenTriangle, b: &ScreenTriangle| {
            b.depth
                .partial_cmp(&a.depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let count: u32 = queue.len() as u32;
        let mut previous: &str = "";
        for triangle in &queue {
            if triangle.fill != previous {
                context.set_fill_style_str(&triangle.fill);
                previous = triangle.fill.as_str();
            }
            context.begin_path();
            context.move_to(triangle.points[0].0 as f64, triangle.points[0].1 as f64);
            context.line_to(triangle.points[1].0 as f64, triangle.points[1].1 as f64);
            context.line_to(triangle.points[2].0 as f64, triangle.points[2].1 as f64);
            context.close_path();
            context.fill();
        }
        count
    }
}

/// 小 helper:把 `get_context` 返回的 `js_sys::Object` 转成 `CanvasRenderingContext2d`。
trait DynInto2d {
    /// dyn into 2d。
    ///
    /// # Arguments
    ///
    ///
    /// # Returns
    ///
    /// - `euv::web_sys::CanvasRenderingContext2d` - 计算结果。
    ///   dyn into 2d。
    ///
    /// # Arguments
    ///
    ///
    /// # Returns
    ///
    /// - `euv::web_sys::CanvasRenderingContext2d` - 计算结果。
    fn dyn_into_2d(self) -> euv::web_sys::CanvasRenderingContext2d;
}

impl DynInto2d for euv::js_sys::Object {
    /// dyn into 2d。
    ///
    /// # Arguments
    ///
    ///
    /// # Returns
    ///
    /// - `euv::web_sys::CanvasRenderingContext2d` - 计算结果。
    ///   dyn into 2d。
    ///
    /// # Arguments
    ///
    ///
    /// # Returns
    ///
    /// - `euv::web_sys::CanvasRenderingContext2d` - 计算结果。
    fn dyn_into_2d(self) -> euv::web_sys::CanvasRenderingContext2d {
        use euv::wasm_bindgen::JsCast;
        // 同 `dyn_into_webgl`:局部绑定接收者,避免 turbofish 写法被
        // 「self.field」静态检查误判成字段访问。
        let source: euv::js_sys::Object = self;
        source.unchecked_into::<euv::web_sys::CanvasRenderingContext2d>()
    }
}

/// 读取一个顶点的世界坐标(本地坐标)。
///
/// # Arguments
///
/// - `&MeshAssetGpu` - MeshAssetGpu 的只读引用。
/// - `u32` - 输入值。
///
/// # Returns
///
/// - `Vec3` - 计算结果。
///   读取一个顶点的世界坐标(本地坐标)。
///
/// # Arguments
///
/// - `&MeshAssetGpu` - MeshAssetGpu 的只读引用。
/// - `u32` - 输入值。
///
/// # Returns
///
/// - `Vec3` - 计算结果。
fn vertex_position(mesh: &MeshAssetGpu, index: u32) -> Vec3 {
    let base: usize = (index as usize) * STRIDE_FLOATS;
    [
        mesh.vertices[base],
        mesh.vertices[base + 1],
        mesh.vertices[base + 2],
    ]
}

// ===========================================================================
// 后端枚举
// ===========================================================================

/// 实际使用的渲染后端。
pub enum Renderer {
    /// WebGL2(GLSL ES 3.00,instancing)。
    WebGl(Box<WebGlRenderer>),
    /// Canvas2D 软件渲染(背面剔除 + 画家算法)。
    Software(SoftwareRenderer),
}

impl Renderer {
    /// 后端名字(显示在 HUD 上)。
    ///
    /// # Returns
    ///
    /// - `&'static str` - 计算结果。
    pub fn backend_name(&self) -> &'static str {
        match self {
            Renderer::WebGl(_) => WEBGL2,
            Renderer::Software(_) => CANVAS2D,
        }
    }
}
