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

use crate::{
    collision::{CollisionWorld, ray_to_shapes},
    r#type::{Mat4Data, Vec3, Vec4},
};

/// 城市边界半长(米):地面覆盖 [-150, 150] × [-150, 150]。
pub const CITY_BOUNDS: f32 = 150.0;

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

    /// 平移矩阵。
    ///
    /// # Arguments
    ///
    /// - `Vec3` - 平移向量(米)。
    ///
    /// # Returns
    ///
    /// - `Mat4` - 列主序平移矩阵。
    pub fn translation(offset: Vec3) -> Mat4 {
        let mut out: Mat4Data = [0.0; 16];
        out[0] = 1.0;
        out[5] = 1.0;
        out[10] = 1.0;
        out[15] = 1.0;
        out[12] = offset[0];
        out[13] = offset[1];
        out[14] = offset[2];
        Mat4::from_column_major(out)
    }

    /// 绕 X 轴旋转矩阵(用于步态的肩摆 / 髋摆)。
    ///
    /// # Arguments
    ///
    /// - `f32` - 旋转角(弧度)。
    ///
    /// # Returns
    ///
    /// - `Mat4` - 列主序旋转矩阵。
    pub fn rotation_x(radians: f32) -> Mat4 {
        let (sine, cosine) = radians.sin_cos();
        Mat4::from_column_major([
            1.0, 0.0, 0.0, 0.0, 0.0, cosine, sine, 0.0, 0.0, -sine, cosine, 0.0, 0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// 绕 Y 轴旋转矩阵(与 `Instance::new` 的朝向约定一致)。
    ///
    /// # Arguments
    ///
    /// - `f32` - 旋转角(弧度)。
    ///
    /// # Returns
    ///
    /// - `Mat4` - 列主序旋转矩阵。
    pub fn rotation_y(radians: f32) -> Mat4 {
        let (sine, cosine) = radians.sin_cos();
        Mat4::from_column_major([
            cosine, 0.0, -sine, 0.0, 0.0, 1.0, 0.0, 0.0, sine, 0.0, cosine, 0.0, 0.0, 0.0, 0.0, 1.0,
        ])
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
    /// 玩家自定义的跟随距离(滚轮缩放结果),不受遮挡回避影响。
    ///
    /// 这是「想要多远」;`distance` 是「实际能走多远」。遮挡时
    /// `distance` 被压到 `distance` 射线命中点之前,遮挡消失后再阻尼
    /// 回到 `desired_distance`。分成两个字段是为了让回避**可逆**:
    /// 如果只有一个 `distance`,被拉近之后就没有基准可以回弹了。
    pub desired_distance: f32,
    /// 眼点(相机世界位置)到最近静态碰撞体表面的距离(米,0 = 扎在楼里)。
    pub eye_clearance: f32,
    /// 本帧相机是否被几何体遮挡(射线在到达 `desired_distance` 之前命中)。
    pub occluded: bool,
}

/// 自由观察(轨道)相机的最小距离(米)。
///
/// 24 m 是为「绕着整座 300 m 城市看」定的。**第三人称跟随不能用它**:
/// 之前 `confine_to_city` 无条件收尾调它,把跟随的 5.6 m 顶到 24 m,
/// 角色缩到几个像素、地面被挤出取景框。
pub const ORBIT_MIN_DISTANCE: f32 = 24.0;

/// 自由观察(轨道)相机的最大距离(米)。
pub const ORBIT_MAX_DISTANCE: f32 = 620.0;

/// 遮挡回避后相机与焦点的**硬下限**(米)。
///
/// 低于这个距离相机会钻进角色身体里,画面里只剩自己的后脑勺 —— 那比
/// 穿墙本身更糟。取 `FOLLOW_DISTANCE_MIN` 同一个值,保证「任何时候角色
/// 都在画面里」优先于「任何时候画面都没有墙」。
///
/// 这个下限之所以真的会触发:探针半径 `CAMERA_PROBE_RADIUS` 会把 AABB
/// 膨胀,玩家站到离墙 0.32 m 以内时射线起点就落在膨胀盒**内部**,求交
/// 返回 `t = 0`,相机被一路压到下限。此时正确的行为是「贴着角色站住、
/// 接受一点墙面穿帮」,而不是把镜头怼进模型里。
pub const OCCLUSION_MIN_DISTANCE: f32 = 2.2;

/// 遮挡回避命中后**额外**保留的贴墙余量(米)。
///
/// 射线命中点正好在墙面上,眼点贴上去之后近平面仍然会啃掉半面墙。
/// 往回退一点让眼点停在墙外,画面里才不会出现「一大片模型不展示」。
pub const OCCLUSION_SKIN: f32 = 0.45;

/// 遮挡检测的「视锥宽度系数」—— 射线不只打中心一条,而是打
/// **中心 + 左右各偏 `t · tan(FOV/2) · COVERAGE_WIDTH` 的三条**。
///
/// 只打中心线是不够的:玩家贴着一栋楼走、镜头从楼的**侧面**掠过去时,
/// 中心线可能完全畅通,但楼体照样会糊住半边画面(这正是用户报的
/// 「摄像头穿模导致大片模型不展示」)。左右两条侧线保证只要**画面里
/// 会有任何一部分**被挡住,相机就先退回来。
///
/// 取 0.62 而不是 1.0:侧线打到的是画面边缘附近,那里就算被挡住也只
/// 遮住一条窄边;要求整条侧线都畅通会让相机在楼群之间反复弹进弹出,
/// 比偶尔糊一条边更难受。
pub const OCCLUSION_COVERAGE_WIDTH: f32 = 0.62;

/// 相机眼点的最低离地高度(米)。
///
/// 地面是一整张 `y = 0` 的网格,碰撞体里没有它,所以线段-地面求交要
/// 单独算。不夹这一条,玩家从高处走向坡地时相机会先钻进地面。
pub const CAMERA_MIN_HEIGHT: f32 = 0.45;

impl Camera {
    /// 创建一台默认相机:街区全景视角。
    ///
    /// 街区沿 Z 轴延伸约 ±48 m,街道半宽 7 m,因此默认镜头要退到
    /// 能同时看到两侧建筑立面、路面和几辆停车的位置,而不是贴脸视角。
    pub fn new() -> Self {
        Self {
            target: [0.0, 4.0, -14.0],
            distance: 250.0,
            yaw: 0.0,
            pitch: 0.42,
            fov_y: std::f32::consts::FRAC_PI_4,
            near: 0.1,
            far: 900.0,
            desired_distance: 250.0,
            eye_clearance: f32::MAX,
            occluded: false,
        }
    }

    /// 恢复出厂设置。
    pub fn reset(&mut self) {
        *self = Camera::new();
    }

    /// 返回注视焦点的只读副本。
    ///
    /// 写入注视点。
    ///
    /// # Arguments
    ///
    /// - `Vec3` - 新的注视点。
    pub fn set_target(&mut self, value: Vec3) {
        self.target = value;
    }

    /// 写入垂直视场角。
    ///
    /// # Arguments
    ///
    /// - `f32` - 新的视场角(弧度)。
    pub fn set_fov_y(&mut self, value: f32) {
        self.fov_y = value;
    }

    /// 写入近裁剪面。
    ///
    /// # Arguments
    ///
    /// - `f32` - 新的近裁剪面距离(米)。
    pub fn set_near(&mut self, value: f32) {
        self.near = value;
    }

    /// 写入远裁剪面。
    ///
    /// # Arguments
    ///
    /// - `f32` - 新的远裁剪面距离(米)。
    pub fn set_far(&mut self, value: f32) {
        self.far = value;
    }

    /// 注视焦点世界坐标。
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

    /// 返回玩家期望的跟随距离(滚轮缩放的目标值)。
    ///
    /// # Returns
    ///
    /// - `f32` - 期望距离(米),遮挡回避不会改写它。
    pub fn get_desired_distance(&self) -> f32 {
        self.desired_distance
    }

    /// 覆盖期望跟随距离,并让 `distance` 跟上(未被遮挡时即刻生效)。
    ///
    /// 滚轮 / TAB 复位走这条路径。遮挡回避每帧写的 `distance` 不会经过
    /// 这里,所以「玩家想拉近一点」不会被回避逻辑覆盖掉。
    ///
    /// # Arguments
    ///
    /// - `f32` - 新的期望距离(米)。
    pub fn set_desired_distance(&mut self, value: f32) {
        self.desired_distance = value;
        if !self.get_occluded() {
            self.set_distance(value);
        }
    }

    /// 返回眼点到最近静态碰撞体表面的距离(米)。
    ///
    /// # Returns
    ///
    /// - `f32` - 净空距离(米);`0.0` 表示眼点落在某个碰撞体内部。
    pub fn get_eye_clearance(&self) -> f32 {
        self.eye_clearance
    }

    /// 本帧相机是否被几何体遮挡。
    ///
    /// # Returns
    ///
    /// - `bool` - `true` 表示回避射线在到达期望距离之前命中了碰撞体。
    pub fn get_occluded(&self) -> bool {
        self.occluded
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

    /// 第三人称俯仰的合法区间(弧度)。
    ///
    /// 下限保证**眼点永远在角色上方**:pitch 为负时相机在焦点下方,
    /// 而地面是整张 `y = 0` 的网格,从地下看出去整屏只有地面色 / 天空
    /// 色,角色和街景全被地面挡住。上限避免接近垂直时的万向节翻转。
    pub const FOLLOW_PITCH_MIN: f32 = -0.05;
    /// 第三人称俯仰的合法上限(弧度)。
    pub const FOLLOW_PITCH_MAX: f32 = 1.05;

    /// 把第三人称的 pitch 收进「相机在角色上方」的区间。
    pub fn clamp_follow_pitch(&mut self) {
        let clamped: f32 = self
            .get_pitch()
            .clamp(Self::FOLLOW_PITCH_MIN, Self::FOLLOW_PITCH_MAX);
        self.set_pitch(clamped);
    }

    /// 钳制 pitch,避免接近垂直时的万向节翻转。
    ///
    /// **只夹 pitch,不动距离。** 之前这里顺带调了 `confine_to_city()`,
    /// 而 `confine_to_city` 收尾会 `clamp_distance()` 把 distance 顶到轨道
    /// 相机的下限 24 m。游戏主循环每帧都调 `clamp_pitch`,于是第三人称
    /// 跟随的 5.6 m 每帧被顶回 24 m:角色缩成几个像素,相机越拉越高,
    /// 连地面都被挤出取景框(截图里只剩一片蓝天)。距离的钳制是
    /// `clamp_distance` / `clamp_follow_distance` 的职责,谁改距离谁夹。
    pub fn clamp_pitch(&mut self) {
        let limit: f32 = std::f32::consts::FRAC_PI_2 - 0.02;
        let clamped: f32 = self.get_pitch().clamp(-limit, limit);
        self.set_pitch(clamped);
    }

    /// 钳制距离,防止穿模或跑到无穷远。
    pub fn clamp_distance(&mut self) {
        // **自由观察的最小距离是 24 m** —— 这是为「绕着整座城市看」的
        // 轨道相机定的,第三人称跟随完全用不上。之前 `confine_to_city`
        // 无条件收尾调 `clamp_distance`,把第三人称的 5.6 m 硬生生顶到
        // 24 m:角色小到只剩几个像素,而且镜头越拉越远,连地面都被挤出
        // 取景框(截图里只剩一片蓝天)。
        let clamped: f32 = self
            .get_distance()
            .clamp(ORBIT_MIN_DISTANCE, ORBIT_MAX_DISTANCE);
        self.set_distance(clamped);
    }

    /// 第三人称跟随专用的距离钳制(允许贴近角色)。
    ///
    /// # Arguments
    ///
    /// - `f32` - 允许的最近距离(米)。
    /// - `f32` - 允许的最远距离(米)。
    pub fn clamp_follow_distance(&mut self, min: f32, max: f32) {
        let clamped: f32 = self.get_distance().clamp(min, max);
        self.set_distance(clamped);
    }

    /// 沿「焦点 → 眼点」方向扫描碰撞世界,求出相机不被穿模的最远距离。
    ///
    /// 这是**纯查询**,不写 `distance` —— 分成两步是因为回避的「命中多近」
    /// 和「实际走到多近」必须解耦:
    ///
    /// - 命中距离在镜头快速扫过一栋楼时会一帧一变(掠射角),直接用
    ///   距离赋值相机会疯狂抽搐;
    /// - 所以这里只返回**这一帧允许的目标距离**,由调用方做阻尼逼近。
    ///
    /// **返回 `None` 表示这一帧没有遮挡** —— 调试通道据此报告
    /// `camOccluded`,验收脚本读它才能证明「回避真的触发了」而不是
    /// 「每帧都被贴墙余量削掉 0.45 m」。
    ///
    /// 命中距离**超过**期望距离时同样返回 `None`:那说明挡在中间的东西
    /// 本来就在相机该待的位置之外,扣贴墙余量只会让相机无缘无故短一截
    /// (这正是「第三人称永远停在 6.95 m 而不是 7.4 m」那个 bug 的成因)。
    ///
    /// 焦点本身可能落在碰撞体里(玩家被挤进墙角的极端情况):此时任何
    /// 朝外的射线都立刻命中,回避无从判断方向,直接放弃,交给距离硬下限
    /// 兜底。
    ///
    /// # Arguments
    ///
    /// - `&CollisionWorld` - 静态碰撞世界的只读引用。
    /// - `f32` - 期望的最大距离(米)。
    ///
    /// # Returns
    ///
    /// - `Option<f32>` - 真正遮挡时为「本帧允许的最远距离(米)」;否则为 `None`。
    pub fn resolve_occlusion(&self, world: &CollisionWorld, max_distance: f32) -> Option<f32> {
        let target: Vec3 = self.get_target();
        if world.contains_point([target[0], target[2]]) {
            return None;
        }
        let forward: Vec3 = self.eye_direction();
        // 侧向 = 视线方向在 XZ 上的左法线,三条射线共用同一个起点。
        let flat: f32 = (forward[0] * forward[0] + forward[2] * forward[2]).sqrt();
        if flat < 1.0e-5 {
            // 视线完全竖直:地面之上没有东西能挡,交给 `lift_above_ground`。
            return None;
        }
        let side: Vec3 = [-forward[2] / flat, 0.0, forward[0] / flat];
        // 侧线在距离 `t` 处的横向偏移 = `t · tan(FOV/2) · 宽度系数`。
        let spread: f32 = (self.get_fov_y() * 0.5).tan() * OCCLUSION_COVERAGE_WIDTH;
        let mut tightest: Option<f32> = None;
        for sign in [-1.0_f32, 0.0, 1.0] {
            let dir: Vec3 = [
                forward[0] + side[0] * spread * sign,
                forward[1],
                forward[2] + side[2] * spread * sign,
            ];
            let Some((hit, _point)) = ray_to_shapes(world, target, dir) else {
                continue;
            };
            if hit >= max_distance {
                continue;
            }
            // `sign != 0` 的侧线更长(斜着走),要按投影折回焦点轴上,
            // 否则会高估遮挡范围、把相机缩得比需要更近。
            let along: f32 = if sign == 0.0 {
                hit
            } else {
                hit / (1.0 + spread * spread).sqrt()
            };
            tightest = Some(match tightest {
                Some(current) => current.min(along),
                None => along,
            });
        }
        tightest.map(|limit: f32| (limit - OCCLUSION_SKIN).max(0.0))
    }

    /// 眼点相对焦点的单位方向向量(焦点 → 眼点)。
    ///
    /// # Returns
    ///
    /// - `Vec3` - 单位方向。
    pub fn eye_direction(&self) -> Vec3 {
        let (sp, cp): (f32, f32) = self.get_pitch().sin_cos();
        let (sy, cy): (f32, f32) = self.get_yaw().sin_cos();
        // 与 [`Self::eye`] 同一套 yaw 约定:焦点 → 眼点 = 前向的反方向。
        [-cy * cp, sp, sy * cp]
    }

    /// 阻尼逼近一个目标距离,并把结果夹在 `[OCCLUSION_MIN_DISTANCE, max]`。
    ///
    /// **拉近比拉远快**:贴墙时相机必须立刻缩进来(否则仍有一两帧糊脸),
    /// 离开墙时慢慢弹回(否则镜头会像弹簧一样抖)。两个速率都是
    /// 指数阻尼 `1 - exp(-rate * dt)`,与帧率无关。
    ///
    /// # Arguments
    ///
    /// - `f32` - 本帧允许的目标距离(米)。
    /// - `f32` - 距离硬上限(米)。
    /// - `f32` - 本帧秒数增量。
    /// - `f32` - 拉近速率(1/秒)。
    /// - `f32` - 拉远速率(1/秒)。
    pub fn approach_distance(
        &mut self,
        target: f32,
        max: f32,
        dt: f32,
        in_rate: f32,
        out_rate: f32,
    ) {
        let current: f32 = self.get_distance();
        let rate: f32 = if target < current { in_rate } else { out_rate };
        let blend: f32 = (1.0 - (-rate * dt).exp()).clamp(0.0, 1.0);
        let stepped: f32 = current + (target - current) * blend;
        self.set_distance(stepped.clamp(OCCLUSION_MIN_DISTANCE, max));
    }

    /// 把眼点抬到地面之上,避免相机沉进 `y = 0` 的地面网格。
    ///
    /// 碰撞体里只有建筑 / 车辆 / 道具,没有地面,所以这一条是纯几何
    /// 修正。只在俯角为负(视线朝下)且焦点离地够低时才会真正缩短距离;
    /// 俯角为正(常规第三人称)时相机本来就在高处,直接返回。
    pub fn lift_above_ground(&mut self) {
        let target: Vec3 = self.get_target();
        let direction: Vec3 = self.eye_direction();
        // 方向朝上(y 分量 > 0)时相机只会更高,永远碰不到地面。
        if direction[1] > 0.0 {
            return;
        }
        // 眼点高度 = target.y + direction.y * distance,要 >= CAMERA_MIN_HEIGHT。
        let headroom: f32 = CAMERA_MIN_HEIGHT - target[1];
        let needed: f32 = if direction[1] < -1.0e-4 {
            headroom / -direction[1]
        } else {
            0.0
        };
        let allowed: f32 = self.get_distance().min(needed);
        self.set_distance(allowed.max(OCCLUSION_MIN_DISTANCE));
    }

    /// 把眼点约束回城市范围内。
    ///
    /// 城市是 300 m × 300 m 的开放网格,不像原来的单条街道那样有
    /// 「走廊」可钻。因此这里约束的是**焦点与眼点都落在城市边界内**:
    /// 焦点被钳在 `[-CITY_..., CITY_...]`,眼点单独钳一次,保证平移
    /// (WASD)、缩放(滚轮 / 捏合)、拖拽转视角之后视角始终成立 ——
    /// 既不会飞到城市外面看到虚空,也不会钻进楼群里被近处墙面糊死。
    pub fn confine_to_city(&mut self) {
        self.confine_follow(true)
    }

    /// 第三人称跟随版的边界约束:只夹焦点,不动距离下限。
    ///
    /// # Arguments
    ///
    /// - `bool` - `true` 表示自由观察(应用 24 m 的轨道距离下限)。
    fn confine_follow(&mut self, orbit: bool) {
        {
            let target: &mut [f32; 3] = self.get_target_mut();
            target[0] = target[0].clamp(-CITY_BOUNDS, CITY_BOUNDS);
            target[2] = target[2].clamp(-CITY_BOUNDS, CITY_BOUNDS);
            target[1] = target[1].clamp(0.5, 90.0);
        }
        // 眼点被 distance 拉到城外时,缩短 distance 直到眼点回到边界内。
        let eye: [f32; 3] = self.eye();
        if eye[0].abs() > CITY_BOUNDS || eye[2].abs() > CITY_BOUNDS {
            let mut distance: f32 = self.get_distance();
            for _ in 0..24 {
                let (_, cp): (f32, f32) = self.get_pitch().sin_cos();
                let (sy, cy): (f32, f32) = self.get_yaw().sin_cos();
                let target: [f32; 3] = self.get_target();
                let x: f32 = target[0] + sy * cp * distance;
                let z: f32 = target[2] + cy * cp * distance;
                if x.abs() <= CITY_BOUNDS && z.abs() <= CITY_BOUNDS {
                    break;
                }
                distance -= distance * 0.12;
            }
            self.set_distance(distance.max(ORBIT_MIN_DISTANCE));
        }
        if orbit {
            self.clamp_distance();
        }
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
        // 球坐标:从焦点沿 yaw/pitch 反方向退后 `distance`。
        //
        // yaw 的约定必须和「玩家朝向」完全一致:玩家的前向是
        // `(cos yaw, -sin yaw)`(见 `player::Player::step` 用的 `direction`),
        // 所以相机要退到**前向的反方向**,即
        // `(-cos yaw·cos pitch, sin pitch, +sin yaw·cos pitch)`。
        // 早先这里用的是 `(sin yaw, cos yaw)`,和玩家前向差了 90° ——
        // 角色明明朝西走,镜头却站在正南,于是出生时镜头正好怼在
        // 车道对面的行道树上,画面里根本看不到角色。
        [
            target[0] - cy * cp * distance,
            target[1] + sp * distance,
            target[2] + sy * cp * distance,
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

#[cfg(test)]
mod tests {
    use crate::camera::{Camera, OCCLUSION_MIN_DISTANCE, OCCLUSION_SKIN};
    use crate::collision::{CollisionWorld, ray_to_shapes};
    use crate::r#const::{
        T_CENTRE_MISSES, T_FLOOR_HELD, T_NO_OCCLUSION, T_OCCLUSION_POSITIVE, T_OCCLUSION_REPORTED,
        T_OCCLUSION_SHORTER, T_PULL_IN, T_PULL_IN_SMOOTH, T_RAY_DISTANCE, T_RAY_MUST_HIT,
        T_RECOVERS, T_SIDE_HITS, T_SKIN_BOUNDED, T_SKIN_POSITIVE,
    };
    use crate::r#type::{Vec2, Vec3};

    /// 把断言文案里的 `{名字}` 占位符替换成实际数值。
    ///
    /// 断言文案按 §1.3c 全部住在 `const.rs`,而 `assert!` 的格式参数必须是
    /// 字面量,所以这里先把文案填好再整体塞进 `"{}"`。
    fn fill(template: &str, args: &[(&str, &str)]) -> String {
        let mut out: String = template.to_string();
        let mut index: usize = 0;
        while index < args.len() {
            let key: &str = args[index].0;
            let value: &str = args[index].1;
            out = out.replace(&format!("{{{key}}}"), value);
            index += 1;
        }
        out
    }

    /// 游戏第三人称相机真实的垂直视场(弧度);`Camera::new()` 默认是轨道相机的 45°。
    const FOLLOW_FOV: f32 = 1.309;

    /// 造一个「玩家站在一堵墙正前方」的最小场景。
    ///
    /// 焦点在原点;`yaw = 0` 时眼点落在 -X 方向,所以墙立在 x = -4 m 处。
    fn world_with_wall() -> CollisionWorld {
        let mut world: CollisionWorld = CollisionWorld::new();
        world.push_aabb([-4.0, 0.0], [1.0, 6.0]);
        world
    }

    #[test]
    fn ray_hits_wall_between_focus_and_eye() {
        let world: CollisionWorld = world_with_wall();
        let origin: Vec3 = [0.0, 1.45, 0.0];
        let hit: Option<(f32, Vec2)> = ray_to_shapes(&world, origin, [-1.0, 0.0, 0.0]);
        let (distance, _point): (f32, Vec2) = hit.expect(T_RAY_MUST_HIT);
        // 墙前表面在 x = -3 m,射线从原点出发 -> 约 3 m 再减探针半径。
        assert!(
            (distance - 2.68).abs() < 0.35,
            "{}",
            fill(T_RAY_DISTANCE, &[("distance", &format!("{distance}"))])
        );
    }

    #[test]
    fn occlusion_shortens_camera_distance() {
        let world: CollisionWorld = world_with_wall();
        let mut camera: Camera = Camera::new();
        camera.set_target([0.0, 1.45, 0.0]);
        camera.set_pitch(0.0);
        camera.set_yaw(0.0);
        camera.set_fov_y(FOLLOW_FOV);
        camera.set_distance(9.5);
        camera.set_desired_distance(9.5);

        let allowed: f32 = camera
            .resolve_occlusion(&world, camera.get_desired_distance())
            .expect(T_OCCLUSION_REPORTED);
        assert!(
            allowed < camera.get_desired_distance(),
            "{}",
            fill(T_OCCLUSION_SHORTER, &[("allowed", &format!("{allowed}"))])
        );
        assert!(
            allowed > 0.5,
            "{}",
            fill(T_OCCLUSION_POSITIVE, &[("allowed", &format!("{allowed}"))])
        );
    }

    #[test]
    fn clear_line_of_sight_reports_no_occlusion() {
        let world: CollisionWorld = world_with_wall();
        let mut camera: Camera = Camera::new();
        camera.set_target([0.0, 1.45, 0.0]);
        camera.set_pitch(0.0);
        // 眼点朝 +X(墙的另一侧),中间没有东西。
        camera.set_yaw(std::f32::consts::PI);
        camera.set_fov_y(FOLLOW_FOV);
        camera.set_distance(9.5);
        camera.set_desired_distance(9.5);

        assert!(
            camera
                .resolve_occlusion(&world, camera.get_desired_distance())
                .is_none(),
            "{}",
            T_NO_OCCLUSION
        );
    }

    #[test]
    fn skin_keeps_camera_off_the_wall() {
        assert!(OCCLUSION_SKIN > 0.0, "{}", T_SKIN_POSITIVE);
        assert!(OCCLUSION_SKIN <= 0.6, "{}", T_SKIN_BOUNDED);
    }

    #[test]
    fn approach_distance_is_smooth_and_respects_floor() {
        let mut camera: Camera = Camera::new();
        camera.set_distance(7.4);
        // 一步 16 ms 拉向 1.0 m,不允许一帧跳完。
        camera.approach_distance(1.0, 22.0, 0.016, 9.0, 2.5);
        let after: f32 = camera.get_distance();
        assert!(
            after < 7.4,
            "{}",
            fill(T_PULL_IN, &[("after", &format!("{after}"))])
        );
        assert!(
            after > 1.0,
            "{}",
            fill(T_PULL_IN_SMOOTH, &[("after", &format!("{after}"))])
        );
    }

    #[test]
    fn approach_distance_never_goes_below_floor() {
        let mut camera: Camera = Camera::new();
        camera.set_distance(1.2);
        for _ in 0..400 {
            camera.approach_distance(0.05, 22.0, 0.016, 30.0, 6.0);
        }
        let distance: f32 = camera.get_distance();
        let floor: f32 = OCCLUSION_MIN_DISTANCE;
        assert!(
            distance >= floor - 1e-3,
            "{}",
            fill(
                T_FLOOR_HELD,
                &[
                    ("distance", &format!("{distance}")),
                    ("floor", &format!("{floor}"))
                ]
            )
        );
    }

    #[test]
    fn approach_distance_recovers_to_desired() {
        let mut camera: Camera = Camera::new();
        camera.set_distance(2.5);
        // 遮挡消失,慢慢回弹到 7.4 m,400 步(约 6.4 s)足够。
        let mut steps: usize = 0;
        while steps < 400 && camera.get_distance() < 7.3 {
            camera.approach_distance(7.4, 22.0, 0.016, 9.0, 2.5);
            steps += 1;
        }
        let distance: f32 = camera.get_distance();
        assert!(
            distance > 7.0,
            "{}",
            fill(T_RECOVERS, &[("distance", &format!("{distance}"))])
        );
    }

    #[test]
    fn side_ray_catches_off_axis_wall() {
        // `yaw = 0` 时中心线沿 -X 走;墙摆在 (-6, -3) —— 侧线在 t ≈ 6 m
        // 处正好经过 (-6, -2.85),中心线却离它 3 m 远。这正是「大片模型
        // 不展示」的真实成因:楼在画面边缘,不在视线正中。
        let mut world: CollisionWorld = CollisionWorld::new();
        world.push_aabb([-6.0, -3.0], [0.6, 1.6]);
        let mut camera: Camera = Camera::new();
        camera.set_target([0.0, 1.45, 0.0]);
        camera.set_pitch(0.0);
        camera.set_yaw(0.0);
        camera.set_fov_y(FOLLOW_FOV);
        camera.set_distance(9.5);
        camera.set_desired_distance(9.5);
        // 中心射线单独打:应当无命中(墙在正侧方)。
        let centre: Option<(f32, Vec2)> =
            ray_to_shapes(&world, camera.get_target(), camera.eye_direction());
        assert!(
            centre.is_none(),
            "{}",
            fill(T_CENTRE_MISSES, &[("centre", &format!("{centre:?}"))])
        );
        // 但整体判定必须认为视线被挡。
        let allowed: Option<f32> = camera.resolve_occlusion(&world, camera.get_desired_distance());
        assert!(allowed.is_some(), "{}", T_SIDE_HITS);
    }
}
