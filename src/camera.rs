//! 相机模块:位置 / yaw / pitch / 透视投影 / 背面剔除。
//!
//! 世界坐标为 Y 轴向上、单位米。相机采用「球坐标环绕 + 可平移焦点」模型:
//! - `target` 是焦点(被注视的世界点)
//! - `distance` 是眼睛到焦点的距离
//! - `yaw`   绕 Y 轴方位角(0 = 看向 -Z 方向)
//! - `pitch` 仰角,限制在 ±(PI/2 - 0.01) 防止万向节翻转
//!
//! 视线矩阵用「右手系 look-at」构造,与 `Matrix4x4::look_at` 语义一致;
//! 投影矩阵是标准 WebGL 深度范围 `[-1, 1]` 的右手透视矩阵。

use crate::r#type::{Mat4Data, Vec3, Vec4};

/// 街道走廊半宽:路面 7 m + 人行道 3.6 m,取 9.5 m 作为安全边界。
pub const CORRIDOR_LIMIT: f32 = 9.5;

/// 一个列主序(mat4 GLSL 约定)的 4x4 矩阵。
///
/// 存储顺序为 `m[col * 4 + row]`,与 GLSL `mat4` 的 uniform 布局一致,
/// 因此可以直接把元素原样喂给 `uniformMatrix4fv`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    /// 16 个元素,列主序。
    pub elements: Mat4Data,
}

impl Mat4 {
    /// 由列主序 16 元数组构造。
    ///
    /// # Arguments
    ///
    /// - `Mat4Data` - 列主序排列的 16 个矩阵元素。
    ///
    /// # Returns
    ///
    /// - `Self` - 持有该元素数组的矩阵。
    pub const fn from_column_major(elements: Mat4Data) -> Self {
        Self { elements }
    }

    /// 返回底层元素数组的只读视图。
    ///
    /// # Returns
    ///
    /// - `&Mat4Data` - 列主序元素数组。
    pub fn get_elements(&self) -> &Mat4Data {
        &self.elements
    }

    /// 矩阵乘法 `self * other`(数学意义上的乘法,注意顺序)。
    ///
    /// # Arguments
    ///
    /// - `&Mat4` - 右乘的矩阵。
    ///
    /// # Returns
    ///
    /// - `Mat4` - 乘积矩阵。
    pub fn multiply(&self, other: &Mat4) -> Mat4 {
        let a: &[f32; 16] = self.get_elements();
        let b: &[f32; 16] = other.get_elements();
        let mut out: [f32; 16] = [0.0; 16];
        for col in 0..4 {
            for row in 0..4 {
                let mut sum: f32 = 0.0;
                for k in 0..4 {
                    sum += a[k * 4 + row] * b[col * 4 + k];
                }
                out[col * 4 + row] = sum;
            }
        }
        Mat4::from_column_major(out)
    }

    /// 右手系透视投影矩阵,深度映射到 WebGL 的 `[-1, 1]`。
    ///
    /// # Arguments
    ///
    /// - `f32` - 垂直视场角(弧度)。
    /// - `f32` - 宽高比(width / height)。
    /// - `f32` - 近裁剪面距离(正数)。
    /// - `f32` - 远裁剪面距离(正数)。
    ///
    /// # Returns
    ///
    /// - `Mat4` - 投影矩阵。
    pub fn perspective(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let f: f32 = 1.0 / (fov_y_rad * 0.5).tan();
        let range: f32 = 1.0 / (near - far);
        let sx: f32 = f / aspect;
        let sy: f32 = f;
        let sz: f32 = (far + near) * range;
        let sz_translate: f32 = 2.0 * far * near * range;
        Mat4::from_column_major([
            sx,
            0.0,
            0.0,
            0.0, //
            0.0,
            sy,
            0.0,
            0.0, //
            0.0,
            0.0,
            sz,
            -1.0, //
            0.0,
            0.0,
            sz_translate,
            0.0,
        ])
    }

    /// 右手系 look-at 视图矩阵(世界 → 相机)。
    ///
    /// 约定:相机看向 `-Z`,`+Y` 向上(与 WebGL 的 gl_Position 语义一致)。
    ///
    /// # Arguments
    ///
    /// - `Vec3` - 相机眼点世界坐标。
    /// - `Vec3` - 注视中心世界坐标。
    /// - `Vec3` - 上方向向量。
    ///
    /// # Returns
    ///
    /// - `Mat4` - 视图矩阵。
    pub fn look_at(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
        let f: [f32; 3] = [center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]];
        let len_f: f32 = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
        let inv_len: f32 = if len_f <= f32::EPSILON {
            0.0
        } else {
            1.0 / len_f.sqrt()
        };
        let s: [f32; 3] = [
            f[1] * up[2] - f[2] * up[1],
            f[2] * up[0] - f[0] * up[2],
            f[0] * up[1] - f[1] * up[0],
        ];
        let len_s: f32 = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        let inv_len_s: f32 = if len_s <= f32::EPSILON {
            0.0
        } else {
            1.0 / len_s.sqrt()
        };
        let s_norm: [f32; 3] = [s[0] * inv_len_s, s[1] * inv_len_s, s[2] * inv_len_s];
        let u: [f32; 3] = [
            s_norm[1] * f[2] * inv_len - s_norm[2] * f[1] * inv_len,
            s_norm[2] * f[0] * inv_len - s_norm[0] * f[2] * inv_len,
            s_norm[0] * f[1] * inv_len - s_norm[1] * f[0] * inv_len,
        ];
        Mat4::from_column_major([
            s_norm[0],
            u[0],
            -f[0] * inv_len,
            0.0, //
            s_norm[1],
            u[1],
            -f[1] * inv_len,
            0.0, //
            s_norm[2],
            u[2],
            -f[2] * inv_len,
            0.0, //
            -(s_norm[0] * eye[0] + s_norm[1] * eye[1] + s_norm[2] * eye[2]),
            -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]),
            f[0] * inv_len * eye[0] + f[1] * inv_len * eye[1] + f[2] * inv_len * eye[2],
            1.0,
        ])
    }

    /// 用矩阵变换一个点(返回 `[x, y, z, w]`,未做透视除法)。
    ///
    /// # Arguments
    ///
    /// - `Vec4` - 待变换的四维齐次坐标。
    ///
    /// # Returns
    ///
    /// - `Vec4` - 变换后的齐次坐标。
    pub fn transform_vec4(&self, v: Vec4) -> Vec4 {
        let a: &[f32; 16] = self.get_elements();
        let mut out: [f32; 4] = [0.0; 4];
        for row in 0..4 {
            out[row] = a[row] * v[0] + a[4 + row] * v[1] + a[8 + row] * v[2] + a[12 + row] * v[3];
        }
        out
    }
}

/// 一台第三人称轨道/跟随相机。
#[derive(Clone, Debug)]
pub struct Camera {
    /// 注视焦点(世界坐标)。
    pub target: Vec3,
    /// 相机与焦点的距离(米)。
    pub distance: f32,
    /// 绕 Y 轴方位角(弧度)。0 表示相机在 +Z 侧看向 -Z。
    pub yaw: f32,
    /// 仰角(弧度),正值表示相机在焦点上方。
    pub pitch: f32,
    /// 垂直视场角(弧度)。
    pub fov_y: f32,
    /// 近裁剪面。
    pub near: f32,
    /// 远裁剪面。
    pub far: f32,
}

impl Camera {
    /// 创建一台默认相机:街区全景视角。
    ///
    /// 街区沿 Z 轴延伸约 ±48 m,街道半宽 7 m,因此默认镜头要退到
    /// 能同时看到两侧建筑立面、路面和几辆停车的位置,而不是贴脸视角。
    pub fn new() -> Self {
        Self {
            target: [0.0, 4.0, -14.0],
            distance: 64.0,
            yaw: 0.0,
            pitch: 0.42,
            fov_y: std::f32::consts::FRAC_PI_4,
            near: 0.1,
            far: 400.0,
        }
    }

    /// 恢复出厂设置。
    pub fn reset(&mut self) {
        *self = Camera::new();
    }

    /// 返回注视焦点的只读副本。
    ///
    /// # Returns
    ///
    /// - `Vec3` - 注视焦点世界坐标。
    pub fn get_target(&self) -> Vec3 {
        self.target
    }

    /// 返回注视焦点的可变引用。
    ///
    /// # Returns
    ///
    /// - `&mut Vec3` - 注视焦点世界坐标的可变引用。
    pub fn get_target_mut(&mut self) -> &mut Vec3 {
        &mut self.target
    }

    /// 返回相机与焦点的距离。
    ///
    /// # Returns
    ///
    /// - `f32` - 眼点到焦点的距离(米)。
    pub fn get_distance(&self) -> f32 {
        self.distance
    }

    /// 覆盖相机与焦点的距离。
    ///
    /// # Arguments
    ///
    /// - `f32` - 新的距离(米)。
    pub fn set_distance(&mut self, value: f32) {
        self.distance = value;
    }

    /// 返回绕 Y 轴的方位角。
    ///
    /// # Returns
    ///
    /// - `f32` - 方位角(弧度)。
    pub fn get_yaw(&self) -> f32 {
        self.yaw
    }

    /// 覆盖绕 Y 轴的方位角。
    ///
    /// # Arguments
    ///
    /// - `f32` - 新的方位角(弧度)。
    pub fn set_yaw(&mut self, value: f32) {
        self.yaw = value;
    }

    /// 返回仰角。
    ///
    /// # Returns
    ///
    /// - `f32` - 仰角(弧度)。
    pub fn get_pitch(&self) -> f32 {
        self.pitch
    }

    /// 覆盖仰角。
    ///
    /// # Arguments
    ///
    /// - `f32` - 新的仰角(弧度)。
    pub fn set_pitch(&mut self, value: f32) {
        self.pitch = value;
    }

    /// 返回垂直视场角。
    ///
    /// # Returns
    ///
    /// - `f32` - 垂直视场角(弧度)。
    pub fn get_fov_y(&self) -> f32 {
        self.fov_y
    }

    /// 返回近裁剪面距离。
    ///
    /// # Returns
    ///
    /// - `f32` - 近裁剪面距离(米)。
    pub fn get_near(&self) -> f32 {
        self.near
    }

    /// 返回远裁剪面距离。
    ///
    /// # Returns
    ///
    /// - `f32` - 远裁剪面距离(米)。
    pub fn get_far(&self) -> f32 {
        self.far
    }

    /// 钳制 pitch,避免接近垂直时的万向节翻转。
    ///
    /// 同时把眼点收回街道走廊(见 `confine_to_street`),
    /// 因为拖拽 / 触摸改的正是 yaw + pitch,越界检查必须跟着走。
    pub fn clamp_pitch(&mut self) {
        let limit: f32 = std::f32::consts::FRAC_PI_2 - 0.02;
        let clamped: f32 = self.get_pitch().clamp(-limit, limit);
        self.set_pitch(clamped);
        self.confine_to_street();
    }

    /// 钳制距离,防止穿模或跑到无穷远。
    pub fn clamp_distance(&mut self) {
        let clamped: f32 = self.get_distance().clamp(12.0, 180.0);
        self.set_distance(clamped);
    }

    /// 把眼点约束回「街道走廊」内。
    ///
    /// 建筑沿街道两侧(|x| >= [`CORRIDOR_LIMIT`])排布,眼点一旦越过这条线
    /// 就会钻进墙体内部,画面被近处的墙面糊成一整片死色。因此这里把眼点
    /// 的 x 硬钳在走廊内,并把注视焦点也限制在街区长度范围内,
    /// 保证平移(WASD)与缩放(滚轮 / 捏合)之后视角始终成立。
    pub fn confine_to_street(&mut self) {
        // 焦点限制在街区长度内,避免平移到街区之外看到虚空。
        {
            let target: &mut [f32; 3] = self.get_target_mut();
            target[0] = target[0].clamp(-CORRIDOR_LIMIT, CORRIDOR_LIMIT);
            target[2] = target[2].clamp(-46.0, 46.0);
            target[1] = target[1].clamp(0.5, 24.0);
        }
        let eye: [f32; 3] = self.eye();
        if eye[0].abs() <= CORRIDOR_LIMIT {
            return;
        }
        // 眼点越界:把 yaw 往 0 收,直到眼点回到走廊内。
        let mut yaw: f32 = self.get_yaw();
        for _ in 0..16 {
            self.set_yaw(yaw);
            if self.eye()[0].abs() <= CORRIDOR_LIMIT {
                return;
            }
            // 朝中轴方向收:yaw 的符号始终指向越界那一侧。
            let sign: f32 = if eye[0] > 0.0 { 1.0 } else { -1.0 };
            yaw -= sign * 0.12;
            if yaw.abs() > std::f32::consts::FRAC_PI_2 {
                // 已经转到街道另一侧,直接钳住不再继续。
                self.set_yaw(sign * std::f32::consts::FRAC_PI_2);
                return;
            }
        }
        self.set_yaw(yaw.clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2));
    }

    /// 计算相机在世界空间中的眼位置。
    ///
    /// # Returns
    ///
    /// - `Vec3` - 眼点世界坐标。
    pub fn eye(&self) -> Vec3 {
        let (sp, cp): (f32, f32) = self.get_pitch().sin_cos();
        let (sy, cy): (f32, f32) = self.get_yaw().sin_cos();
        let target: [f32; 3] = self.get_target();
        let distance: f32 = self.get_distance();
        // 球坐标:从焦点沿 yaw/pitch 反方向退后 distance。
        [
            target[0] + sy * cp * distance,
            target[1] + sp * distance,
            target[2] + cy * cp * distance,
        ]
    }

    /// 相机的前向单位向量(眼 → 焦点)。
    ///
    /// # Returns
    ///
    /// - `Vec3` - 前向单位向量。
    pub fn forward(&self) -> Vec3 {
        let eye: [f32; 3] = self.eye();
        let target: [f32; 3] = self.get_target();
        let d: [f32; 3] = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
        let len: f32 = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if len <= f32::EPSILON {
            [0.0, 0.0, -1.0]
        } else {
            [d[0] / len, d[1] / len, d[2] / len]
        }
    }

    /// 视图矩阵。
    ///
    /// # Returns
    ///
    /// - `Mat4` - 世界坐标到相机坐标的视图矩阵。
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at(self.eye(), self.get_target(), [0.0, 1.0, 0.0])
    }

    /// 投影矩阵,`aspect` = 宽 / 高。
    ///
    /// # Arguments
    ///
    /// - `f32` - 宽高比(width / height)。
    ///
    /// # Returns
    ///
    /// - `Mat4` - 透视投影矩阵。
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective(self.get_fov_y(), aspect, self.get_near(), self.get_far())
    }

    /// 视图投影矩阵。
    ///
    /// # Arguments
    ///
    /// - `f32` - 宽高比(width / height)。
    ///
    /// # Returns
    ///
    /// - `Mat4` - 视图与投影的乘积矩阵。
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect).multiply(&self.view_matrix())
    }

    /// 把世界坐标点投影到屏幕像素坐标。
    ///
    /// 返回 `Some((x, y, depth))`;若点在相机后方(w <= 0)返回 `None`。
    /// `depth` 为归一化深度 `[0, 1]`(近平面 0、远平面 1),供软件渲染排序用。
    ///
    /// # Arguments
    ///
    /// - `Vec3` - 世界坐标点。
    /// - `f32` - 画布宽度(像素)。
    /// - `f32` - 画布高度(像素)。
    ///
    /// # Returns
    ///
    /// - `Option<(f32, f32, f32)>` - 屏幕 `(x, y)` 与归一化深度;点在相机后方时为 `None`。
    pub fn world_to_screen(&self, point: Vec3, width: f32, height: f32) -> Option<(f32, f32, f32)> {
        let vp: Mat4 = self.view_projection(width / height.max(f32::EPSILON));
        let clip: [f32; 4] = vp.transform_vec4([point[0], point[1], point[2], 1.0]);
        if clip[3] <= f32::EPSILON {
            return None;
        }
        let inv_w: f32 = 1.0 / clip[3];
        let ndc_x: f32 = clip[0] * inv_w;
        let ndc_y: f32 = clip[1] * inv_w;
        let ndc_z: f32 = clip[2] * inv_w;
        let x: f32 = (ndc_x * 0.5 + 0.5) * width;
        let y: f32 = (1.0 - (ndc_y * 0.5 + 0.5)) * height;
        let depth: f32 = (ndc_z * 0.5 + 0.5).clamp(0.0, 1.0);
        Some((x, y, depth))
    }
}

impl Default for Camera {
    /// 返回默认相机,与 `Camera::new` 等价。
    fn default() -> Self {
        Camera::new()
    }
}

/// 背面剔除测试:三角形是否朝向相机。
///
/// 约定:资产里的 `faces` 顶点顺序为「从外侧看逆时针(CCW)」。
/// 在右手系 + 屏幕 Y 向下的像素坐标下,CCW 三角形在屏幕上变成顺时针,
/// 因此可见面的有符号面积(按屏幕坐标)为 **负**。
///
/// 这里采用更稳健的等价判据:计算面法线与「面 → 眼睛」向量。
/// `dot > 0` 表示法线背离相机,判定为背面,应被剔除。
///
/// 退化三角形(零面积)同样剔除,避免画出一个点。
///
/// # Arguments
///
/// - `Vec3` - 三角形第一个顶点。
/// - `Vec3` - 三角形第二个顶点。
/// - `Vec3` - 三角形第三个顶点。
/// - `Vec3` - 相机眼点世界坐标。
///
/// # Returns
///
/// - `bool` - `true` 表示该面应被背面剔除。
pub fn is_back_facing(a: Vec3, b: Vec3, c: Vec3, eye: Vec3) -> bool {
    let edge1: [f32; 3] = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let edge2: [f32; 3] = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let normal: [f32; 3] = [
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0],
    ];
    let to_eye: [f32; 3] = [eye[0] - a[0], eye[1] - a[1], eye[2] - a[2]];
    let dot: f32 = normal[0] * to_eye[0] + normal[1] * to_eye[1] + normal[2] * to_eye[2];
    let area2: f32 = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if area2 <= 1e-12 {
        return true;
    }
    dot <= 0.0
}
