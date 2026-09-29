//! 交通 AI + 拾取物:车道循环行驶的车队、可上下的车辆、地面拾取物。
//!
//! 车队**不是自由漫游**:每辆车被分配到一条固定车道(街道沿 Z 轴延伸,
//! 车道是一条沿 Z 的直线段),车辆只沿车道 Z 方向前进,到端点 wrap 回起点,
//! 因此永远不会拐进人行道、更不会穿楼。玩家上车后 WASD 直接驱动该车,
//! 车速上限提到街道限速(8–15 m/s)。
//!
//! 车灯:`car_*` 资产的 `frontlights` / `rearlights` part 带 `emissive`,
//! 展开时已经逐三角形写进顶点缓冲的第四个 vec3(见 `render::build_gpu_mesh`),
//! 夜间 `SceneLighting::emissive_gain` 拉到 1.85 + 泛光 pass,所以车灯
//! 是真的在发光,不是画一个假的亮点贴图。

use crate::{
    collision::CollisionWorld,
    r#const::*,
    player::{MAX_HEALTH, Player},
    r#type::{Vec2, Vec3},
};

/// 街道在 Z 轴上的可行驶半长(米)。
///
/// 由 [`crate::game::traffic_half`] 从城市半边长推导,车道两端到端点 wrap。
pub const LOOP_HALF_LENGTH: f32 = 120.0;
/// 车队巡航速度下限(米/秒)。
pub const SPEED_MIN: f32 = 8.0;
/// 车队巡航速度上限(米/秒)。
pub const SPEED_MAX: f32 = 15.0;
/// 玩家驾驶时的加速度(米/秒²)。
pub const DRIVE_ACCEL: f32 = 11.0;
/// 玩家驾驶时的刹车减速度(米/秒²)。
pub const DRIVE_BRAKE: f32 = 18.0;
/// 玩家走到拾取物多少米内就算拾到。
pub const PICKUP_RADIUS: f32 = 2.2;
/// 玩家走到车边多少米内可以按 F 上车。
/// 车队巡航速度(米/秒)——城市道路 8–15 m/s,取区间中段。
pub const CRUISE_SPEED: f32 = 11.5;

pub const ENTER_VEHICLE_RADIUS: f32 = 4.2;
/// 车辆「靠边等人」的距离(米)。
///
/// 真实城市里路边车会为招手的人减速。这一条让**行人真的追得上车**:
/// 8–14 m/s 的车在 4.2 m 内根本没法靠走路追上,没有这个规则第三人称
/// 的上车功能在玩法上就是废的(测试跑了 14 次都上不去)。车发现有人类
/// 站在路边,就把巡航速度降到 0 当临时出租车站。
pub const HAUL_RANGE: f32 = 26.0;
/// 「靠边等人」的减速强度(1/秒)。
pub const HAUL_BRAKE: f32 = 3.4;
/// `pickup_health_pack` 的回血量(点)。
pub const HEALTH_PACK_HEAL: f32 = 35.0;
/// `pickup_cash_stack` 的金额。
pub const CASH_STACK_AMOUNT: f32 = 250.0;
/// 拾取物绕 Y 轴的自转角速度(弧度/秒)。
pub const PICKUP_SPIN_RATE: f32 = 1.1;

/// 一辆参与交通仿真的车。
#[derive(Clone, Debug)]
pub struct TrafficCar {
    /// 资产 id。
    pub asset: &'static str,
    /// 当前世界坐标。
    position: Vec3,
    /// 沿车道前进的当前速度(米/秒,恒为正,倒车 = 换方向)。
    speed: f32,
    /// 巡航速度(米/秒)。
    cruise: f32,
    /// 行车的固定 X 坐标(米)—— 车道中心,玩家驾驶时也会被强制拉回。
    lane_x: f32,
    /// 行驶方向:`1.0` = 沿 +Z,`-1.0` = 沿 -Z。
    direction: f32,
    /// 玩家是否正在驾驶这辆车。
    driven: bool,
}

impl TrafficCar {
    /// 当前世界坐标。
    ///
    /// # Returns
    ///
    /// - `Vec3` - 当前世界坐标。
    pub fn get_position(&self) -> Vec3 {
        self.position
    }

    /// 写入当前世界坐标。
    ///
    /// # Arguments
    ///
    /// - `Vec3` - 新世界坐标。
    pub fn set_position(&mut self, value: Vec3) {
        self.position = value;
    }

    /// 沿车道前进的当前速度。
    ///
    /// # Returns
    ///
    /// - `f32` - 当前速度(米/秒)。
    pub fn get_speed(&self) -> f32 {
        self.speed
    }

    /// 写入当前速度。
    ///
    /// # Arguments
    ///
    /// - `f32` - 新速度(米/秒)。
    pub fn set_speed(&mut self, value: f32) {
        self.speed = value;
    }

    /// 行驶方向。
    ///
    /// # Returns
    ///
    /// - `f32` - `1.0` = 沿 +Z,`-1.0` = 沿 -Z。
    pub fn get_direction(&self) -> f32 {
        self.direction
    }

    /// 巡航速度。
    ///
    /// # Returns
    ///
    /// - `f32` - 巡航速度(米/秒)。
    pub fn get_cruise(&self) -> f32 {
        self.cruise
    }

    /// 行车的固定车道 X 坐标(米)。
    ///
    /// # Returns
    ///
    /// - `f32` - 车道中心 X(米)。
    pub fn get_lane_x(&self) -> f32 {
        self.lane_x
    }

    /// 玩家是否正在驾驶这辆车。
    ///
    /// # Returns
    ///
    /// - `bool` - 正在驾驶时为 `true`。
    pub fn get_driven(&self) -> bool {
        self.driven
    }

    /// 写入「玩家是否正在驾驶」。
    ///
    /// # Arguments
    ///
    /// - `bool` - 是否正在驾驶。
    pub fn set_driven(&mut self, value: bool) {
        self.driven = value;
    }

    /// 在指定车道上放一辆车。
    ///
    /// # Arguments
    ///
    /// - `&'static str` - 资产 id。
    /// - `f32` - 行车的固定 X 坐标(米)。
    /// - `f32` - 起始 Z 坐标(米)。
    /// - `f32` - 巡航速度(米/秒)。
    /// - `f32` - 行驶方向(`1.0` 或 `-1.0`)。
    ///
    /// # Returns
    ///
    /// - `Self` - 就绪的车辆状态。
    pub fn new(
        asset: &'static str,
        lane_x: f32,
        start_z: f32,
        cruise: f32,
        direction: f32,
    ) -> Self {
        Self {
            asset,
            position: [lane_x, 0.0, start_z],
            speed: cruise,
            cruise,
            lane_x,
            direction,
            driven: false,
        }
    }

    /// 车辆的朝向(绕 Y 轴弧度)。
    ///
    /// 资产的本地 +X 是车头方向(车灯在本地 -X,所以车头朝 -X,见
    /// `car_sedan.json` 里 `frontlights` 的 x 全部为负),而
    /// `Instance::new` 把本地 +X 映到世界 `(cos yaw, 0, -sin yaw)`,所以
    /// 沿 +Z 行驶的车 `yaw = -PI/2`,沿 -Z 行驶的车 `yaw = +PI/2`。
    ///
    /// # Returns
    ///
    /// - `f32` - 绕 Y 轴的朝向(弧度)。
    pub fn get_yaw(&self) -> f32 {
        -self.get_direction() * std::f32::consts::FRAC_PI_2
    }

    /// 车队模式下推进一个固定步长:沿车道前进 + 到端点 wrap。
    ///
    /// 速度被平滑逼近巡航速度,玩家驾驶模式由 [`Self::drive`] 接管,
    /// 这里只做 wrap。
    ///
    /// # Arguments
    ///
    /// - `f32` - 固定步长(秒)。
    /// - `bool` - `true` 表示路边有玩家招手,这辆车减速停靠等客。
    pub fn step(&mut self, dt: f32, hailer: bool) {
        // 有行人在路边招手 → 当临时出租车站:松油门,直到停下等人。
        let target: f32 = if hailer { 0.0 } else { self.get_cruise() };
        let approach: f32 = if hailer {
            (HAUL_BRAKE * dt).min(1.0)
        } else {
            (DRIVE_ACCEL * dt / self.get_cruise().max(1.0)).min(1.0)
        };
        let cruise: f32 = target;
        let speed: f32 = self.get_speed();
        self.set_speed(speed + (cruise - speed) * approach);
        let step_z: f32 = self.get_position()[2] + self.get_direction() * self.get_speed() * dt;
        // wrap:出了 [−LOOP_HALF_LENGTH, +LOOP_HALF_LENGTH] 就折回另一头,
        // 车道是一条闭合的环形轨道,车永远看不到尽头。停着的车不 wrap ——
        // 停在原地等人,不然「招手停车」会把人甩到另一条街去。
        let wrapped: f32 = if self.get_speed() < 0.05 {
            step_z
        } else if step_z > LOOP_HALF_LENGTH {
            step_z - 2.0 * LOOP_HALF_LENGTH
        } else if step_z < -LOOP_HALF_LENGTH {
            step_z + 2.0 * LOOP_HALF_LENGTH
        } else {
            step_z
        };
        let here: Vec3 = self.get_position();
        self.set_position([here[0], here[1], wrapped]);
    }

    /// 玩家驾驶输入:油门 / 刹车,并把车挡在碰撞世界之外。
    ///
    /// 车道约束在这里第二次生效:车想冲出街道时,分离函数把它推回来,
    /// 速度清零 —— 所以**玩家开车也穿不过楼**。
    ///
    /// # Arguments
    ///
    /// - `f32` - 油门输入(−1..1,正 = 加速)。
    /// - `f32` - 固定步长(秒)。
    /// - `&CollisionWorld` - 静态碰撞世界。
    pub fn drive(&mut self, throttle: f32, dt: f32, world: &CollisionWorld) {
        let speed: f32 = self.get_speed();
        let next: f32 = if throttle > 0.0 {
            speed + DRIVE_ACCEL * throttle * dt
        } else if throttle < 0.0 {
            speed + DRIVE_BRAKE * throttle * dt
        } else {
            speed
        };
        self.set_speed(next.clamp(0.0, SPEED_MAX * 1.6));
        let here: Vec3 = self.get_position();
        let proposed: Vec2 = [
            here[0],
            here[2] + self.get_direction() * self.get_speed() * dt,
        ];
        let resolved: Vec2 = world.resolve_car(proposed);
        let blocked: bool = (resolved[1] - proposed[1]).abs() > f32::EPSILON;
        // 车道约束在这里第二次生效:车只在自己的车道 X 上跑。即使被路边
        // 道具(停车位、垃圾桶)顶开,也把 X 拉回车道 —— 车队因此永远不会
        // 漂到对面车道或者开上人行道。
        let lane_x: f32 = self.get_lane_x();
        self.set_position([lane_x, here[1], resolved[1]]);
        if blocked {
            // 撞墙:速度砍到 0,但不改变车道 —— 车永远开不出街道走廊。
            self.set_speed(0.0);
        }
    }
}

/// 地面拾取物。
#[derive(Clone, Debug)]
pub struct Pickup {
    /// 资产 id。
    pub asset: &'static str,
    /// HUD 上显示的名字。
    pub label: &'static str,
    /// 世界坐标。
    position: Vec3,
    /// 是否已被拾走。
    taken: bool,
    /// 自转相位(弧度),让拾取物在原地缓缓转。
    spin: f32,
}

impl Pickup {
    /// 在指定位置放一个拾取物。
    ///
    /// # Arguments
    ///
    /// - `&'static str` - 资产 id。
    /// - `&'static str` - HUD 显示名。
    /// - `Vec3` - 世界坐标。
    ///
    /// # Returns
    ///
    /// - `Self` - 就绪的拾取物。
    pub fn new(asset: &'static str, label: &'static str, position: Vec3) -> Self {
        Self {
            asset,
            label,
            position,
            taken: false,
            spin: 0.0,
        }
    }

    /// 世界坐标。
    ///
    /// # Returns
    ///
    /// - `Vec3` - 当前世界坐标。
    pub fn get_position(&self) -> Vec3 {
        self.position
    }

    /// 该拾取物的资产 id。
    ///
    /// # Returns
    ///
    /// - `&'static str` - 资产 id。
    pub fn get_asset(&self) -> &'static str {
        self.asset
    }

    /// 是否已被拾走。
    ///
    /// # Returns
    ///
    /// - `bool` - 已被拾走时为 `true`。
    pub fn get_taken(&self) -> bool {
        self.taken
    }

    /// 写入「是否已被拾走」。
    ///
    /// # Arguments
    ///
    /// - `bool` - 是否已被拾走。
    pub fn set_taken(&mut self, value: bool) {
        self.taken = value;
    }

    /// 自转相位。
    ///
    /// # Returns
    ///
    /// - `f32` - 自转相位(弧度)。
    pub fn get_spin(&self) -> f32 {
        self.spin
    }

    /// 推进自转相位。
    ///
    /// # Arguments
    ///
    /// - `f32` - 固定步长(秒)。
    pub fn set_spin_advance(&mut self, dt: f32) {
        let two_pi: f32 = 2.0 * std::f32::consts::PI;
        self.spin = (self.get_spin() + PICKUP_SPIN_RATE * dt) % two_pi;
    }
}

/// 车队与拾取物的集合。
#[derive(Clone, Debug)]
pub struct Traffic {
    /// 车队车辆。
    cars: Vec<TrafficCar>,
    /// 地面拾取物。
    pickups: Vec<Pickup>,
}

impl Traffic {
    /// 新建一个空的交通世界。
    ///
    /// # Returns
    ///
    /// - `Self` - 空的车队与拾取物集合。
    pub fn new() -> Self {
        Self {
            cars: Vec::new(),
            pickups: Vec::new(),
        }
    }

    /// 按蓝图铺出车队与拾取物。
    ///
    /// # Arguments
    ///
    /// - `&[(&'static str, f32, f32, f32)]` - 车道蓝图。
    /// - `&[(&'static str, &'static str, Vec3)]` - 拾取物蓝图。
    pub fn populate(
        &mut self,
        lanes: &[(&'static str, f32, f32, f32)],
        pickups: &[(&'static str, &'static str, Vec3)],
    ) {
        self.get_cars_mut().clear();
        for (asset, lane_x, start_z, direction) in lanes {
            self.get_cars_mut().push(TrafficCar::new(
                asset,
                *lane_x,
                *start_z,
                CRUISE_SPEED.clamp(SPEED_MIN, SPEED_MAX),
                *direction,
            ));
        }
        self.get_pickups_mut().clear();
        for (asset, label, position) in pickups {
            self.get_pickups_mut()
                .push(Pickup::new(asset, label, *position));
        }
    }

    /// 车队车辆列表。
    ///
    /// # Returns
    ///
    /// - `&Vec<TrafficCar>` - 车队。
    pub fn get_cars_ref(&self) -> &Vec<TrafficCar> {
        &self.cars
    }

    /// 地面拾取物列表。
    ///
    /// # Returns
    ///
    /// - `&Vec<Pickup>` - 拾取物集合。
    pub fn get_pickups_ref(&self) -> &Vec<Pickup> {
        &self.pickups
    }

    /// 可变地借用车队。
    ///
    /// # Returns
    ///
    /// - `&mut Vec<TrafficCar>` - 车队的可变引用。
    pub fn get_cars_mut(&mut self) -> &mut Vec<TrafficCar> {
        &mut self.cars
    }

    /// 可变地借用拾取物集合。
    ///
    /// # Returns
    ///
    /// - `&mut Vec<Pickup>` - 拾取物集合的可变引用。
    pub fn get_pickups_mut(&mut self) -> &mut Vec<Pickup> {
        &mut self.pickups
    }

    /// 可变地取一辆车。
    ///
    /// # Arguments
    ///
    /// - `usize` - 车辆索引。
    ///
    /// # Returns
    ///
    /// - `Option<&mut TrafficCar>` - 越界时为 `None`。
    pub fn get_car_mut(&mut self, index: usize) -> Option<&mut TrafficCar> {
        self.cars.get_mut(index)
    }

    /// 可变地取一个拾取物。
    ///
    /// # Arguments
    ///
    /// - `usize` - 拾取物索引。
    ///
    /// # Returns
    ///
    /// - `Option<&mut Pickup>` - 越界时为 `None`。
    pub fn get_pickup_mut(&mut self, index: usize) -> Option<&mut Pickup> {
        self.pickups.get_mut(index)
    }

    /// 推进整个交通世界:车队行驶 + 拾取物自转。
    ///
    /// # Arguments
    ///
    /// - `f32` - 固定步长(秒)。
    /// - `Option<[f32; 2]>` - 正在路边招手的玩家 `[x, z]`;`None` 表示没人。
    pub fn step(&mut self, dt: f32, hailer: Option<[f32; 2]>) {
        // 只有**离玩家最近的那一辆**会靠边等人,不是半径内所有车 ——
        // 否则 26 m 内整条街的车一起停,交通直接堵死。
        let hailer_car: Option<usize> = hailer.and_then(|who: [f32; 2]| {
            let mut best: Option<(usize, f32)> = None;
            for (index, car) in self.get_cars_ref().iter().enumerate() {
                if car.get_driven() {
                    continue;
                }
                let here: Vec3 = car.get_position();
                let dx: f32 = here[0] - who[0];
                let dz: f32 = here[2] - who[1];
                let distance: f32 = (dx * dx + dz * dz).sqrt();
                if distance <= HAUL_RANGE && best.map(|(_, d)| distance < d).unwrap_or(true) {
                    best = Some((index, distance));
                }
            }
            best.map(|(index, _)| index)
        });
        for index in 0..self.get_cars_ref().len() {
            if let Some(car) = self.get_car_mut(index) {
                if car.get_driven() {
                    continue;
                }
                car.step(dt, Some(index) == hailer_car);
            }
        }
        for index in 0..self.get_pickups_ref().len() {
            if let Some(pickup) = self.get_pickup_mut(index) {
                pickup.set_spin_advance(dt);
            }
        }
    }
}

/// 找到离玩家最近、且近到可以上车的车。
///
/// # Arguments
///
/// - `&Traffic` - 交通世界。
/// - `Vec3` - 玩家世界坐标。
///
/// # Returns
///
/// - `Option<usize>` - 可上车车辆的索引;附近没有车时为 `None`。
pub fn nearest_car(traffic: &Traffic, position: Vec3) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, car) in traffic.get_cars_ref().iter().enumerate() {
        let here: Vec3 = car.get_position();
        let dx: f32 = here[0] - position[0];
        let dz: f32 = here[2] - position[2];
        let distance: f32 = (dx * dx + dz * dz).sqrt();
        if distance <= ENTER_VEHICLE_RADIUS && best.map(|(_, d)| distance < d).unwrap_or(true) {
            best = Some((index, distance));
        }
    }
    best.map(|(index, _)| index)
}

/// 结算一次拾取:回血 / 加钱 / 记录,并标记拾取物已被拿走。
///
/// # Arguments
///
/// - `&mut Player` - 玩家状态。
/// - `&mut Pickup` - 命中的拾取物。
pub fn apply_pickup(player: &mut Player, pickup: &mut Pickup) {
    let notice: String = match pickup.asset {
        PICKUP_HEALTH_PACK => {
            let healed: f32 = (player.get_health() + HEALTH_PACK_HEAL).min(MAX_HEALTH);
            player.set_health(healed);
            String::from(NOTICE_HEALTH)
        }
        PICKUP_CASH_STACK => {
            player.set_cash_add(CASH_STACK_AMOUNT);
            String::from(NOTICE_CASH)
        }
        _ => format!("{NOTICE_WEAPON}{}", pickup.label),
    };
    player.set_notice(notice);
    pickup.set_taken(true);
}
