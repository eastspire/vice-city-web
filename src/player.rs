//! 玩家角色:第三人称移动、步态动画、跟随相机目标、生命值与拾取计数。
//!
//! 复用 `ped_suit` 资产(不新建模)。行人资产在 JSON 里是**分部件**的
//! (`torso` / `head` / `hair` / `upper_arm_L` … / `lower_leg_R` / `shoe_R`),
//! 所以渲染时把每个 part 拆成一个独立 mesh + 独立批次,每帧按行走相位
//! 给每个 part 算一条 model matrix —— 上臂 / 小臂 / 大腿 / 小腿绕各自
//! 关节的局部 X 轴摆动,鞋子随小腿一起摆,停下时相位衰减回 0、肢体归位。
//!
//! 关节枢轴点不是手写的魔法坐标,而是从**每个 part 自己的包围盒**推出来
//! 的:上臂枢轴 = 肩(盒顶面 Y + 靠近身体一侧的 X),大腿 = 胯,小腿 = 膝。
//! 换模型时只要 part 划分一致,骨架自动跟着长。

use crate::{
    camera::Mat4,
    collision::CollisionWorld,
    r#const::*,
    r#type::{Vec2, Vec3},
};

/// 步行速度(米/秒)。
pub const WALK_SPEED: f32 = 4.6;
/// 奔跑速度(米/秒,按住 Shift)。
pub const RUN_SPEED: f32 = 8.4;
/// 转身插值速度(1/秒):越小转身越「重」,不会瞬移式扭头。
pub const TURN_RATE: f32 = 11.0;
/// 行走相位推进速率(rad / 米):每走 1 米推进约 2.6 rad,约 2.4 步。
pub const GAIT_RATE_PER_METER: f32 = 2.6;
/// 肢体摆动幅度上限(弧度)。
pub const LIMB_SWING: f32 = 0.62;
/// 停止后相位回零的速率(1/秒)。
pub const GAIT_RELAX_RATE: f32 = 6.5;
/// 起步 / 停步时速度的指数趋近速率(1/秒)。
pub const SPEED_RAMP: f32 = 14.0;
/// 低于这个平面速度(米/秒)就认为「站住了」:不转身、不推进相位。
pub const PLANAR_EPSILON: f32 = 0.05;
/// 玩家初始生命值。
pub const MAX_HEALTH: f32 = 100.0;

/// 玩家的全部骨架 part(与 `ped_suit` 的 part 划分一一对应)。
///
/// 枢轴由 [`build_rig`] 从资产包围盒推导,这里只声明「哪些 part 怎么摆」。
pub const LIMB_PLAN: &[(&str, f32, f32)] = &[
    (PART_UPPER_ARM_L, GAIT_PHASE_L, LIMB_SWING),
    (PART_LOWER_ARM_L, GAIT_PHASE_L, LIMB_SWING * 0.8),
    (PART_UPPER_ARM_R, GAIT_PHASE_R, LIMB_SWING),
    (PART_LOWER_ARM_R, GAIT_PHASE_R, LIMB_SWING * 0.8),
    (PART_UPPER_LEG_L, GAIT_PHASE_L, LIMB_SWING),
    (PART_LOWER_LEG_L, GAIT_PHASE_L, LIMB_SWING * 0.9),
    (PART_UPPER_LEG_R, GAIT_PHASE_R, LIMB_SWING),
    (PART_LOWER_LEG_R, GAIT_PHASE_R, LIMB_SWING * 0.9),
];

/// 某个 part 在当前步态相位下的摆动角度(弧度)。
///
/// 上下臂同相位、上下腿同相位、左右反相,形成对角的步态;幅度随
/// `gait_amount` 缩放,所以停下时全部归零。
///
/// # Arguments
///
/// - `&str` - part 名。
/// - `f32` - 当前行走相位(弧度)。
/// - `f32` - 当前步态幅度 `0.0..1.0`。
///
/// # Returns
///
/// - `f32` - 该 part 的关节摆角(弧度)。
pub fn limb_swing(part: &str, phase: f32, amount: f32) -> f32 {
    let Some((_, offset, scale)) = LIMB_PLAN.iter().find(|(name, _, _)| *name == part) else {
        return 0.0;
    };
    scale.sin() * amount * (phase + offset)
}

/// 左腿相位:0。
pub const GAIT_PHASE_L: f32 = 0.0;
/// 右腿相位:与左腿反相。
pub const GAIT_PHASE_R: f32 = std::f32::consts::PI;

/// 一个玩家的运行时状态。
#[derive(Clone, Debug)]
pub struct Player {
    /// 脚下世界坐标(Y 恒为 0,踩在人行道高度上由场景统一处理)。
    position: Vec3,
    /// 朝向(绕 Y 轴弧度,0 = 面向 +X)。
    yaw: f32,
    /// 当前水平速度(X / Z,米/秒)。
    velocity: Vec2,
    /// 行走相位(弧度)。
    gait_phase: f32,
    /// 当前步态幅度 0..1,由速度平滑驱动。
    gait_amount: f32,
    /// 生命值。
    health: f32,
    /// 已拾取的现金。
    cash: f32,
    /// 当前驾驶的车辆索引(`None` = 在地面上)。
    vehicle: Option<usize>,
    /// 上一次收到的拾取提示(空串 = 无提示)。
    notice: String,
    /// 已被拾取掉的拾取物索引。
    collected: Vec<usize>,
}

impl Player {
    /// 脚下世界坐标。
    ///
    /// # Returns
    ///
    /// - `Vec3` - 当前世界坐标(Y 恒为 0)。
    pub fn get_position(&self) -> Vec3 {
        self.position
    }

    /// 写入脚下世界坐标。
    ///
    /// # Arguments
    ///
    /// - `Vec3` - 新世界坐标。
    pub fn set_position(&mut self, value: Vec3) {
        self.position = value;
    }

    /// 朝向(绕 Y 轴弧度)。
    ///
    /// # Returns
    ///
    /// - `f32` - 当前朝向。
    pub fn get_yaw(&self) -> f32 {
        self.yaw
    }

    /// 写入朝向。
    ///
    /// # Arguments
    ///
    /// - `f32` - 新朝向(弧度)。
    pub fn set_yaw(&mut self, value: f32) {
        self.yaw = value;
    }

    /// 当前水平速度。
    ///
    /// # Returns
    ///
    /// - `Vec2` - X / Z 速度分量(米/秒)。
    pub fn get_velocity(&self) -> Vec2 {
        self.velocity
    }

    /// 写入当前水平速度。
    ///
    /// # Arguments
    ///
    /// - `Vec2` - 新速度(米/秒)。
    pub fn set_velocity(&mut self, value: Vec2) {
        self.velocity = value;
    }

    /// 行走相位。
    ///
    /// # Returns
    ///
    /// - `f32` - 当前相位(弧度)。
    pub fn get_gait_phase(&self) -> f32 {
        self.gait_phase
    }

    /// 写入行走相位。
    ///
    /// # Arguments
    ///
    /// - `f32` - 新相位(弧度)。
    pub fn set_gait_phase(&mut self, value: f32) {
        self.gait_phase = value;
    }

    /// 当前步态幅度。
    ///
    /// # Returns
    ///
    /// - `f32` - `0.0` 站立,`1.0` 全速行走。
    pub fn get_gait_amount(&self) -> f32 {
        self.gait_amount
    }

    /// 写入当前步态幅度。
    ///
    /// # Arguments
    ///
    /// - `f32` - 新幅度。
    pub fn set_gait_amount(&mut self, value: f32) {
        self.gait_amount = value;
    }

    /// 生命值。
    ///
    /// # Returns
    ///
    /// - `f32` - 当前生命值。
    pub fn get_health(&self) -> f32 {
        self.health
    }

    /// 写入生命值。
    ///
    /// # Arguments
    ///
    /// - `f32` - 新生命值。
    pub fn set_health(&mut self, value: f32) {
        self.health = value;
    }

    /// 已拾取的现金。
    ///
    /// # Returns
    ///
    /// - `f32` - 当前现金。
    pub fn get_cash(&self) -> f32 {
        self.cash
    }

    /// 累加现金。
    ///
    /// # Arguments
    ///
    /// - `f32` - 增量(可以为负)。
    pub fn set_cash_add(&mut self, amount: f32) {
        self.cash += amount;
    }

    /// 当前驾驶的车辆索引。
    ///
    /// # Returns
    ///
    /// - `Option<usize>` - 在驾驶时为 `Some(索引)`,否则为 `None`。
    pub fn get_vehicle(&self) -> Option<usize> {
        self.vehicle
    }

    /// 写入当前驾驶的车辆索引。
    ///
    /// # Arguments
    ///
    /// - `Option<usize>` - 新车辆索引。
    pub fn set_vehicle(&mut self, value: Option<usize>) {
        self.vehicle = value;
    }

    /// 玩家是否坐在车里。
    ///
    /// # Returns
    ///
    /// - `bool` - 在车里时为 `true`。
    pub fn get_driving(&self) -> bool {
        self.vehicle.is_some()
    }

    /// 上一次收到的拾取提示。
    ///
    /// # Returns
    ///
    /// - `&str` - 提示文本,空串表示无提示。
    pub fn get_notice(&self) -> &str {
        &self.notice
    }

    /// 写入拾取提示。
    ///
    /// # Arguments
    ///
    /// - `String` - 新提示文本。
    pub fn set_notice(&mut self, value: String) {
        self.notice = value;
    }

    /// 已被拾取掉的拾取物索引。
    ///
    /// # Returns
    ///
    /// - `&Vec<usize>` - 已拾取的索引列表。
    pub fn get_collected_ref(&self) -> &Vec<usize> {
        &self.collected
    }

    /// 追加一个已拾取的拾取物索引。
    ///
    /// # Arguments
    ///
    /// - `usize` - 拾取物索引。
    pub fn set_collected_push(&mut self, index: usize) {
        self.collected.push(index);
    }

    /// 已拾取的数量。
    ///
    /// # Returns
    ///
    /// - `usize` - 已拾取的拾取物个数。
    pub fn collected_count(&self) -> usize {
        self.get_collected_ref().len()
    }

    /// 在给定位置创建一个满血玩家。
    ///
    /// # Arguments
    ///
    /// - `Vec3` - 初始世界坐标。
    /// - `f32` - 初始朝向(弧度)。
    ///
    /// # Returns
    ///
    /// - `Self` - 就绪的玩家状态。
    pub fn new(position: Vec3, yaw: f32) -> Self {
        Self {
            position,
            yaw,
            velocity: [0.0, 0.0],
            gait_phase: 0.0,
            gait_amount: 0.0,
            health: MAX_HEALTH,
            cash: 0.0,
            vehicle: None,
            notice: String::new(),
            collected: Vec::new(),
        }
    }

    /// 推进一个固定步长:移动、碰撞分离、转身、步态相位。
    ///
    /// # Arguments
    ///
    /// - `Vec2` - 期望的前 / 侧向输入(各分量 −1..1)。
    /// - `Vec2` - 相机「前」方向的世界 XZ 分量。
    /// - `f32` - 固定步长(秒)。
    /// - `f32` - 目标速度上限(米/秒)。
    /// - `&CollisionWorld` - 静态碰撞世界。
    pub fn step(
        &mut self,
        intent: Vec2,
        direction: Vec2,
        dt: f32,
        speed: f32,
        world: &CollisionWorld,
    ) {
        // 目标速度 = 前向分量 + 侧向分量,两者各自受 `speed` 上限约束,
        // 所以斜向移动不会比直线快(和真实操作手感一致)。
        let forward: f32 = intent[1];
        let strafe: f32 = intent[0];
        let target: Vec2 = [
            (direction[0] * forward + -direction[1] * strafe) * speed,
            (direction[1] * forward + direction[0] * strafe) * speed,
        ];
        let ramp: f32 = (1.0 - (-SPEED_RAMP * dt).exp()).clamp(0.0, 1.0);
        let current: Vec2 = self.get_velocity();
        self.set_velocity([
            current[0] + (target[0] - current[0]) * ramp,
            current[1] + (target[1] - current[1]) * ramp,
        ]);

        // 碰撞:把想要的位移交给分离函数,拿到被挡下来的位置。
        let velocity: Vec2 = self.get_velocity();
        let here: Vec3 = self.get_position();
        let wanted: Vec2 = [here[0] + velocity[0] * dt, here[2] + velocity[1] * dt];
        let resolved: Vec2 = world.resolve(wanted);
        self.set_position([resolved[0], here[1], resolved[1]]);

        // 速度取「**实际走出来的位移** / dt」,而不是把分离修正量加回去。
        // 后者会被 1/dt 放大几十倍,一次 1 cm 的分离就会注入 0.6 m/s 的
        // 侧向速度,角色会被「弹」着走。现在撞墙时朝墙的速度分量自然
        // 归零,不会一直顶着墙搓,也不会被弹飞。
        let inverse_dt: f32 = 1.0 / dt.max(f32::EPSILON);
        let achieved: Vec2 = [resolved[0] - here[0], resolved[1] - here[2]];
        self.set_velocity([
            (achieved[0] * inverse_dt).clamp(-speed, speed),
            (achieved[1] * inverse_dt).clamp(-speed, speed),
        ]);

        // 转身 / 步态用**碰撞前的目标速度** `velocity`,而不是被墙削过的
        // 实际速度:蹭着墙走时手脚照样摆,只是真的走不动 —— 这才符合直觉。
        let planar: f32 = (velocity[0] * velocity[0] + velocity[1] * velocity[1]).sqrt();
        if planar > PLANAR_EPSILON {
            let desired: f32 = -velocity[1].atan2(velocity[0]);
            let turning: f32 = wrap_angle(desired - self.get_yaw()) * TURN_RATE * dt;
            self.set_yaw(self.get_yaw() + turning);
        }

        // 步态:幅度跟随实际速度,相位按走过的距离推进。
        let ratio: f32 = (planar / speed).clamp(0.0, 1.0);
        let relaxed: f32 = (1.0 - (-GAIT_RELAX_RATE * dt).exp()).clamp(0.0, 1.0);
        if planar > PLANAR_EPSILON {
            let scaled: f32 = self.get_gait_amount() + (ratio - self.get_gait_amount()) * relaxed;
            self.set_gait_amount(scaled);
            self.set_gait_phase(
                (self.get_gait_phase() + planar * GAIT_RATE_PER_METER * dt)
                    % (2.0 * std::f32::consts::PI),
            );
        } else {
            let shrunk: f32 = self.get_gait_amount() - self.get_gait_amount() * relaxed;
            self.set_gait_amount(shrunk);
            self.set_gait_phase(self.get_gait_phase() * (1.0 - relaxed));
        }
        self.set_gait_amount(self.get_gait_amount().clamp(0.0, 1.0));
    }
}

/// 把角度折到 (−π, π],避免转身走远路。
///
/// # Arguments
///
/// - `f32` - 原始角度差(弧度)。
///
/// # Returns
///
/// - `f32` - 折回后的角度差(弧度)。
pub fn wrap_angle(angle: f32) -> f32 {
    let two_pi: f32 = 2.0 * std::f32::consts::PI;
    let mut value: f32 = angle % two_pi;
    if value > std::f32::consts::PI {
        value -= two_pi;
    } else if value < -std::f32::consts::PI {
        value += two_pi;
    }
    value
}

/// 从 part 的本地包围盒推出关节枢轴。
///
/// 手臂 / 腿:枢轴 = 盒的**顶面** Y(肩 / 胯),绕 X 轴摆动就是标准的
/// 肩摆 / 髋摆;横向取靠近身体中线的一侧。鞋:枢轴 = 踝(盒顶面 Y)。
///
/// # Arguments
///
/// - `&str` - part 名。
/// - `(Vec3, Vec3)` - 该 part 的本地包围盒 `(min, max)`。
///
/// # Returns
///
/// - `Vec3` - 关节枢轴的资产本地坐标。
pub fn joint_pivot(part: &str, bounds: (Vec3, Vec3)) -> Vec3 {
    let (min, max) = bounds;
    if part.contains(PART_ARM) || part.contains(PART_LEG) {
        return [min[0], max[1], (min[2] + max[2]) * 0.5];
    }
    if part == PART_SHOE_L || part == PART_SHOE_R {
        return [(min[0] + max[0]) * 0.5, max[1], (min[2] + max[2]) * 0.5];
    }
    [0.0, 0.0, 0.0]
}

/// 一个玩家肢体的 model matrix:整体位姿 × 绕枢轴的局部 X 轴摆动。
///
/// # Arguments
///
/// - `Vec3` - 玩家脚下世界坐标。
/// - `f32` - 玩家朝向(弧度)。
/// - `Vec3` - 关节枢轴的资产本地坐标。
/// - `f32` - 该关节的摆角(弧度)。
///
/// # Returns
///
/// - `Mat4` - 列主序 model matrix。
pub fn limb_matrix(origin: Vec3, yaw: f32, pivot: Vec3, swing: f32) -> Mat4 {
    let placement: Mat4 = Mat4::translation(origin).multiply(&Mat4::rotation_y(yaw));
    let joint: Mat4 = Mat4::translation(pivot)
        .multiply(&Mat4::rotation_x(swing))
        .multiply(&Mat4::translation([-pivot[0], -pivot[1], -pivot[2]]));
    placement.multiply(&joint)
}
