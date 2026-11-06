"""
Dodge-v0: 一个 32x32 像素的极简"躲避"环境。

设计意图（对应 Ha & Schmidhuber 2018 的 DoomTakeCover）：
  * 观测是纯像素，agent 看不到任何结构化状态 —— 逼迫我们学 V(视觉压缩)
  * 存活即得分，被砸到就结束 —— 稀疏而简单的信用分配
  * 方块的**下一次生成位置是随机的** —— 未来是多模态的，逼迫 M 必须是概率模型
    （这正是 MDN 而非普通 MSE 回归的存在理由）

动作: 0=左移, 1=不动, 2=右移
奖励: 每存活一步 +1
终止: 被方块砸到, 或达到 max_steps
"""
import numpy as np

H = W = 32
PADDLE_Y0, PADDLE_Y1 = 27, 30      # 玩家所在行区间 [y0, y1)
PADDLE_HALF = 3                    # 玩家半宽 -> 宽度 7
BLOCK = 6                          # 方块边长
FALL = 3                           # 每步下落像素


class DodgeEnv:
    def __init__(self, max_steps=300, seed=None):
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)
        self.action_dim = 3

    def _spawn(self):
        self.bx = self.rng.randint(0, W - BLOCK + 1)
        self.by = -BLOCK

    def reset(self):
        self.px = W // 2
        self._spawn()
        self.t = 0
        return self._obs()

    def _obs(self):
        o = np.zeros((H, W), dtype=np.float32)
        # 玩家
        x0 = max(0, self.px - PADDLE_HALF)
        x1 = min(W, self.px + PADDLE_HALF + 1)
        o[PADDLE_Y0:PADDLE_Y1, x0:x1] = 1.0
        # 方块
        y0 = max(0, self.by)
        y1 = min(H, self.by + BLOCK)
        if y1 > y0:
            o[y0:y1, self.bx:self.bx + BLOCK] = 0.6
        return o

    def step(self, a):
        if a == 0:
            self.px -= 3
        elif a == 2:
            self.px += 3
        self.px = int(np.clip(self.px, PADDLE_HALF, W - 1 - PADDLE_HALF))

        self.by += FALL
        done = False

        # 碰撞判定：方块与玩家矩形相交
        overlap_y = (self.by < PADDLE_Y1) and (self.by + BLOCK > PADDLE_Y0)
        px0, px1 = self.px - PADDLE_HALF, self.px + PADDLE_HALF + 1
        overlap_x = (self.bx < px1) and (self.bx + BLOCK > px0)
        if overlap_y and overlap_x:
            done = True

        if self.by >= H:                # 落地未命中 -> 重新生成
            self._spawn()

        self.t += 1
        if self.t >= self.max_steps:
            done = True
        return self._obs(), 1.0, done


def rollout_random(n_episodes, seed=0, max_steps=300, sticky=0.6):
    """
    用随机策略采集数据（Ha&Schmidhuber 第 1 步）。
    sticky: 以该概率重复上一个动作。纯 iid 随机策略在本环境里平均只活 24 步，
            数据覆盖太窄；"粘性"随机策略活得久一些、轨迹更平滑，
            让世界模型见到更多样的状态。这是实践中很重要的一步 ——
            **世界模型的能力上限由采集策略的状态覆盖决定**。
    """
    env = DodgeEnv(max_steps=max_steps, seed=seed)
    rng = np.random.RandomState(seed + 1)
    obs_eps, act_eps, done_eps = [], [], []
    for _ in range(n_episodes):
        o = env.reset()
        os_, as_, ds_ = [o], [], []
        a = rng.randint(3)
        while True:
            if rng.rand() > sticky:
                a = rng.randint(3)
            o, r, d = env.step(a)
            os_.append(o); as_.append(a); ds_.append(float(d))
            if d:
                break
        obs_eps.append(np.array(os_, dtype=np.float32))
        act_eps.append(np.array(as_, dtype=np.int64))
        done_eps.append(np.array(ds_, dtype=np.float32))
    return obs_eps, act_eps, done_eps


if __name__ == "__main__":
    env = DodgeEnv(seed=0)
    o = env.reset()
    tot = 0
    rng = np.random.RandomState(0)
    lens = []
    for ep in range(200):
        env.reset(); n = 0
        while True:
            o, r, d = env.step(rng.randint(3)); n += 1
            if d:
                break
        lens.append(n)
    print("random policy 平均存活步数:", np.mean(lens), "+-", np.std(lens))
