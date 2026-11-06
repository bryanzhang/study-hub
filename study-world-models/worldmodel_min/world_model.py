"""
最小可用世界模型 —— Ha & Schmidhuber (2018) 的完整复现骨架
=========================================================
V (ConvVAE)  : 像素 -> 低维隐变量 z            [压缩"当下"]
M (MDN-RNN)  : (z_t, a_t, h_t) -> P(z_{t+1})   [压缩"时间"，概率式]
C (线性)     : [z_t, h_t] -> a_t               [极小策略，用 CMA-ES 进化]

关键在于：C **只在 M 生成的梦里训练**，从未见过真实环境，然后直接迁移回真实环境。

运行:
    python3 world_model.py --stage all           # 全流程
    python3 world_model.py --stage all --quick   # 快速版（约 3-5 分钟 CPU）
"""
import argparse, os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env_dodge import DodgeEnv, rollout_random

torch.set_num_threads(max(1, os.cpu_count() or 1))
DEV = "cpu"
CKPT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 超参
Z_DIM      = 16     # latent 维度（论文 CarRacing 用 32，Doom 用 64）
RNN_HID    = 128    # LSTM 隐单元（论文 256 / 512）
K_MIX      = 5      # 高斯混合分量数（论文 K=5）
ACT_DIM    = 3
MAX_STEPS  = 300


# ================================================================
# V: 卷积 VAE
# ================================================================
class ConvVAE(nn.Module):
    """
    结构对齐论文：4 层 stride-2 卷积编码 + 4 层反卷积解码，输出 sigmoid。
    这里输入是 32x32x1（论文是 64x64x3），层数相应减为 3 层。
    """
    def __init__(self, z_dim=Z_DIM):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1), nn.ReLU(),    # 32 -> 16
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),   # 16 -> 8
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),   # 8  -> 4
        )
        self.fc_mu     = nn.Linear(64 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(64 * 4 * 4, z_dim)
        self.fc_dec    = nn.Linear(z_dim, 64 * 4 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),   # 4 -> 8
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),   # 8 -> 16
            nn.ConvTranspose2d(16, 1, 4, 2, 1),               # 16 -> 32
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 64, 4, 4)
        return torch.sigmoid(self.dec(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        # 重参数化技巧：z = mu + sigma * eps，让梯度绕开采样这一步
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """
    -ELBO = 重构项 + beta * KL 项
      重构项  E_q[-log p(x|z)]  这里用逐像素 BCE（等价 Bernoulli 似然）
      KL 项   KL(q(z|x) || N(0,I))  有解析式，见讲义附录 A.1
    """
    rec = F.binary_cross_entropy(recon, x, reduction="none").sum(dim=(1, 2, 3)).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    return rec + beta * kl, rec.item(), kl.item()


# ================================================================
# M: MDN-RNN
# ================================================================
class MDNRNN(nn.Module):
    """
    LSTM 输出经一个线性头，给出**每个 latent 维度独立**的 K 分量高斯混合参数
    （这与 hardmaru 官方实现一致：每一维一个 mixture，等价于对角协方差）
    额外输出一个 done logit —— 让梦知道什么时候该结束。
    """
    def __init__(self, z_dim=Z_DIM, act_dim=ACT_DIM, hid=RNN_HID, k=K_MIX):
        super().__init__()
        self.z_dim, self.k, self.hid = z_dim, k, hid
        self.lstm = nn.LSTM(z_dim + act_dim, hid, batch_first=True)
        self.head = nn.Linear(hid, 3 * k * z_dim + 1)   # logpi, mu, logsigma, done

    def forward(self, z, a_onehot, hc=None):
        out, hc = self.lstm(torch.cat([z, a_onehot], dim=-1), hc)
        y = self.head(out)
        B, T, _ = y.shape
        done_logit = y[..., -1]
        p = y[..., :-1].view(B, T, 3, self.k, self.z_dim)
        logpi   = F.log_softmax(p[:, :, 0], dim=2)          # (B,T,K,Z) 沿 K 归一化
        mu      = p[:, :, 1]
        logsig  = torch.clamp(p[:, :, 2], -7.0, 3.0)
        return logpi, mu, logsig, done_logit, out, hc


LOG_SQRT_2PI = float(np.log(np.sqrt(2 * np.pi)))


def mdn_nll(logpi, mu, logsig, target):
    """
    -log P(z_{t+1} | ...) = -sum_dim logsumexp_k [ log pi_k + log N(z | mu_k, sigma_k) ]
    见讲义 3.3 节推导
    """
    t = target.unsqueeze(2)                                  # (B,T,1,Z)
    logN = -0.5 * ((t - mu) / logsig.exp()).pow(2) - logsig - LOG_SQRT_2PI
    return -torch.logsumexp(logpi + logN, dim=2).sum(-1).mean()


def mdn_sample(logpi, mu, logsig, temperature=1.0):
    """
    温度采样。τ>1 -> 更随机（梦更"难"），τ<1 -> 更确定（梦更"甜"、更容易被 C 钻空子）
      * mixture logits 除以 τ
      * sigma 乘以 sqrt(τ)
    """
    logpi = logpi / temperature
    pi = F.softmax(logpi, dim=-2)                            # 沿 K
    B, Z = pi.shape[0], pi.shape[-1]
    # 每个 (batch, dim) 独立按 pi 抽一个分量
    flat = pi.permute(0, 2, 1).reshape(-1, pi.shape[-2])     # (B*Z, K)
    idx = torch.multinomial(flat, 1).view(B, Z, 1).permute(0, 2, 1)   # (B,1,Z)
    m = torch.gather(mu, 1, idx).squeeze(1)
    s = torch.gather(logsig, 1, idx).squeeze(1).exp() * np.sqrt(temperature)
    return m + s * torch.randn_like(m)


# ================================================================
# C: 线性控制器
# ================================================================
class Controller:
    """a = argmax(W [z; h] + b)。参数量 = (Z+H)*3 + 3，本例 435 个。"""
    n_params = (Z_DIM + RNN_HID) * ACT_DIM + ACT_DIM

    @staticmethod
    def act_batch(params_mat, params_b, feat):
        """向量化：同时为一整批（多个个体/多条轨迹）算动作"""
        logits = torch.einsum("bi,bio->bo", feat, params_mat) + params_b
        return logits.argmax(dim=-1)

    @staticmethod
    def unpack(theta):
        theta = np.asarray(theta, dtype=np.float32)
        W = theta[:(Z_DIM + RNN_HID) * ACT_DIM].reshape(Z_DIM + RNN_HID, ACT_DIM)
        b = theta[(Z_DIM + RNN_HID) * ACT_DIM:]
        return torch.from_numpy(W), torch.from_numpy(b)


# ================================================================
# 阶段 1：采数据
# ================================================================
def stage_collect(n_ep):
    print(f"[1/4] 随机策略采集 {n_ep} 条 rollout ...")
    obs, act, done = rollout_random(n_ep, seed=0, max_steps=MAX_STEPS)
    n_frames = sum(len(o) for o in obs)
    np.save(f"{CKPT}/frames.npy", np.concatenate([o for o in obs])[:, None])
    with open(f"{CKPT}/eps.npz", "wb") as f:
        np.savez(f, **{f"o{i}": o for i, o in enumerate(obs)},
                    **{f"a{i}": a for i, a in enumerate(act)},
                    **{f"d{i}": d for i, d in enumerate(done)},
                 n=np.array(len(obs)))
    print(f"      -> {n_frames} 帧, 平均 episode 长度 {n_frames/n_ep:.1f}")


# ================================================================
# 阶段 2：训 V
# ================================================================
def stage_vae(epochs, bs=256, lr=1e-3):
    print(f"[2/4] 训练 ConvVAE ({epochs} epochs) ...")
    X = torch.from_numpy(np.load(f"{CKPT}/frames.npy"))
    vae = ConvVAE().to(DEV)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = rec_s = kl_s = nb = 0
        for i in range(0, n, bs):
            xb = X[perm[i:i + bs]]
            recon, mu, logvar = vae(xb)
            loss, rec, kl = vae_loss(recon, xb, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); rec_s += rec; kl_s += kl; nb += 1
        print(f"      epoch {ep+1}: -ELBO={tot/nb:8.2f}  recon={rec_s/nb:8.2f}  KL={kl_s/nb:6.2f}")
    torch.save(vae.state_dict(), f"{CKPT}/vae.pt")


# ================================================================
# 阶段 3：训 M
# ================================================================
def _encode_episodes(vae):
    d = np.load(f"{CKPT}/eps.npz")
    n = int(d["n"])
    eps = []
    with torch.no_grad():
        for i in range(n):
            o = torch.from_numpy(d[f"o{i}"])[:, None]
            mu, logvar = vae.encode(o)
            eps.append((mu, logvar, torch.from_numpy(d[f"a{i}"]), torch.from_numpy(d[f"d{i}"])))
    return eps


def stage_rnn(epochs, seq=24, bs=64, lr=1e-3, done_weight=8.0):
    """
    两个容易踩的坑（第一版代码就踩了，见讲义 4.4 节）：
    (a) 若只用「长度 > seq」的 episode，会**系统性筛掉所有短命 episode**，
        而死亡事件恰恰集中在那里 -> done 头永远预测"不会死" -> 梦里永生。
        修法：所有 episode 都用，不足 seq 的做 padding + mask。
    (b) done 是极不平衡的标签（约 1/30） -> 给正类加权 pos_weight。
    """
    print(f"[3/4] 训练 MDN-RNN ({epochs} epochs, seq={seq}) ...")
    vae = ConvVAE(); vae.load_state_dict(torch.load(f"{CKPT}/vae.pt")); vae.eval()
    eps = [e for e in _encode_episodes(vae) if len(e[2]) >= 2]
    n_done = sum(float(e[3].sum()) for e in eps)
    n_step = sum(len(e[2]) for e in eps)
    print(f"      episode {len(eps)} 条 / {n_step} 步，其中死亡事件 {int(n_done)} 个 "
          f"({100*n_done/n_step:.1f}%) -> pos_weight={done_weight}")

    rnn = MDNRNN().to(DEV)
    opt = torch.optim.Adam(rnn.parameters(), lr=lr)
    rng = np.random.RandomState(0)
    steps_per_epoch = 200
    pw = torch.tensor(done_weight)

    for ep in range(epochs):
        nll_s = bce_s = 0.0
        for _ in range(steps_per_epoch):
            Z = torch.zeros(bs, seq + 1, Z_DIM)
            A = torch.zeros(bs, seq, dtype=torch.long)
            D = torch.zeros(bs, seq)
            M = torch.zeros(bs, seq)
            for j in range(bs):
                mu, logvar, a, dflag = eps[rng.randint(len(eps))]
                L = len(a)
                t0 = rng.randint(0, max(1, L - seq + 1))
                ln = min(seq, L - t0)
                m, lv = mu[t0:t0 + ln + 1], logvar[t0:t0 + ln + 1]
                # 论文附录的 trick：存 mu/logvar，每个 batch 重新采样 z，等价于数据增广
                Z[j, :ln + 1] = m + torch.exp(0.5 * lv) * torch.randn_like(m)
                A[j, :ln] = a[t0:t0 + ln]
                D[j, :ln] = dflag[t0:t0 + ln]
                M[j, :ln] = 1.0
            a1h = F.one_hot(A, ACT_DIM).float()
            logpi, mu_, logsig, dlogit, _, _ = rnn(Z[:, :-1], a1h)
            # 逐步 NLL 后按 mask 加权平均
            t = Z[:, 1:].unsqueeze(2)
            logN = -0.5 * ((t - mu_) / logsig.exp()).pow(2) - logsig - LOG_SQRT_2PI
            step_nll = -torch.logsumexp(logpi + logN, dim=2).sum(-1)
            nll = (step_nll * M).sum() / M.sum()
            bce_raw = F.binary_cross_entropy_with_logits(dlogit, D, pos_weight=pw, reduction="none")
            bce = (bce_raw * M).sum() / M.sum()
            loss = nll + 10.0 * bce
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
            opt.step()
            nll_s += nll.item(); bce_s += bce.item()
        print(f"      epoch {ep+1}: MDN-NLL={nll_s/steps_per_epoch:8.3f}  done-BCE={bce_s/steps_per_epoch:.4f}")
    torch.save(rnn.state_dict(), f"{CKPT}/rnn.pt")
    stage_diag()


# ================================================================
# 诊断：世界模型本身有多准？（在训 C 之前必须做这一步）
# ================================================================
def stage_diag():
    vae = ConvVAE(); vae.load_state_dict(torch.load(f"{CKPT}/vae.pt")); vae.eval()
    rnn = MDNRNN(); rnn.load_state_dict(torch.load(f"{CKPT}/rnn.pt")); rnn.eval()
    eps = _encode_episodes(vae)
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for mu, logvar, a, dflag in eps[:200]:
            _, _, _, dlogit, _, _ = rnn(mu[:-1].unsqueeze(0), F.one_hot(a, ACT_DIM).float().unsqueeze(0))
            pred = (torch.sigmoid(dlogit[0]) > 0.5).float()
            tp += float(((pred == 1) & (dflag == 1)).sum()); fp += float(((pred == 1) & (dflag == 0)).sum())
            fn += float(((pred == 0) & (dflag == 1)).sum()); tn += float(((pred == 0) & (dflag == 0)).sum())
    prec = tp / max(tp + fp, 1e-9); rec = tp / max(tp + fn, 1e-9)
    print(f"      [诊断] done 预测 precision={prec:.3f} recall={rec:.3f} "
          f"(tp={int(tp)} fp={int(fp)} fn={int(fn)})")
    if rec < 0.3:
        print("      ⚠ recall 太低：世界模型基本不会让 agent 死 -> 梦是可被随意钻空子的，"
              "训出来的 C 迁移必然失败")
    return prec, rec


# ================================================================
# 阶段 4：在梦里进化 C
# ================================================================
class DreamEnv:
    """
    把 M 本身包装成一个"环境"——这就是论文里的 DoomRNN。
    完全不渲染像素，直接在 latent 空间 step。整批轨迹并行推进，所以很快。
    """
    def __init__(self, rnn, init_z_pool, temperature=1.0, max_steps=MAX_STEPS):
        self.rnn, self.pool, self.T, self.max_steps = rnn, init_z_pool, temperature, max_steps

    def rollout_batch(self, thetas, seed=0):
        """thetas: (P, n_params) -> 每个个体的累计回报"""
        P = len(thetas)
        g = torch.Generator().manual_seed(seed)
        Ws = torch.stack([Controller.unpack(t)[0] for t in thetas])
        bs = torch.stack([Controller.unpack(t)[1] for t in thetas])

        idx = torch.randint(0, len(self.pool), (P,), generator=g)
        z = self.pool[idx].clone()
        h = torch.zeros(1, P, RNN_HID); c = torch.zeros(1, P, RNN_HID)
        alive = torch.ones(P)
        ret = torch.zeros(P)
        with torch.no_grad():
            for t in range(self.max_steps):
                feat = torch.cat([z, h[0]], dim=-1)
                a = Controller.act_batch(Ws, bs, feat)
                logpi, mu, logsig, dlogit, _, (h, c) = self.rnn(
                    z.unsqueeze(1), F.one_hot(a, ACT_DIM).float().unsqueeze(1), (h, c))
                z = mdn_sample(logpi[:, 0], mu[:, 0], logsig[:, 0], self.T)
                ret += alive
                done = (torch.sigmoid(dlogit[:, 0]) > 0.5).float()
                alive = alive * (1 - done)
                if alive.sum() == 0:
                    break
        return ret.numpy()


def eval_real(theta, vae, rnn, n_ep=32, seed=1234, max_steps=MAX_STEPS):
    """把梦里学到的 C 直接搬回真实环境"""
    W, b = Controller.unpack(theta)
    scores = []
    with torch.no_grad():
        for k in range(n_ep):
            env = DodgeEnv(max_steps=max_steps, seed=seed + k)
            o = env.reset()
            h = torch.zeros(1, 1, RNN_HID); c = torch.zeros(1, 1, RNN_HID)
            tot = 0.0
            while True:
                mu, _ = vae.encode(torch.from_numpy(o)[None, None])
                feat = torch.cat([mu, h[0]], dim=-1)
                a = int((feat @ W + b).argmax())
                _, _, _, _, _, (h, c) = rnn(mu.unsqueeze(1),
                                            F.one_hot(torch.tensor([a]), ACT_DIM).float().unsqueeze(1),
                                            (h, c))
                o, r, d = env.step(a)
                tot += r
                if d:
                    break
            scores.append(tot)
    return float(np.mean(scores)), float(np.std(scores))


def stage_controller(generations, temperature, pop=32, n_seeds=4):
    import cma
    print(f"[4/4] 在梦里用 CMA-ES 进化控制器 (τ={temperature}, pop={pop}, {generations} 代) ...")
    vae = ConvVAE(); vae.load_state_dict(torch.load(f"{CKPT}/vae.pt")); vae.eval()
    rnn = MDNRNN(); rnn.load_state_dict(torch.load(f"{CKPT}/rnn.pt")); rnn.eval()

    # 梦的初始状态池：从真实数据里取每条 episode 的开头几帧编码
    d = np.load(f"{CKPT}/eps.npz"); n = int(d["n"])
    with torch.no_grad():
        pool = torch.cat([vae.encode(torch.from_numpy(d[f"o{i}"])[:3, None])[0] for i in range(min(n, 300))])
    dream = DreamEnv(rnn, pool, temperature=temperature)

    es = cma.CMAEvolutionStrategy(Controller.n_params * [0.0], 0.5,
                                  {"popsize": pop, "seed": 3, "verbose": -9})
    best_theta, best_dream = None, -1e9
    hist = []
    for gen in range(generations):
        sols = es.ask()
        # fitness = 多个随机种子下梦中回报的均值（论文用 16 个种子）
        fits = np.mean([dream.rollout_batch(sols, seed=1000 * gen + s) for s in range(n_seeds)], axis=0)
        es.tell(sols, (-fits).tolist())          # cma 是最小化
        if fits.max() > best_dream:
            best_dream, best_theta = fits.max(), sols[int(fits.argmax())]
        if (gen + 1) % 5 == 0 or gen == 0:
            rm, rs = eval_real(np.array(es.result.xfavorite), vae, rnn, n_ep=16)
            hist.append({"gen": gen + 1, "dream": float(fits.mean()), "real": rm})
            print(f"      gen {gen+1:3d}: 梦中均值={fits.mean():6.1f} 梦中最好={fits.max():6.1f} "
                  f"| 真实环境={rm:6.1f} ± {rs:5.1f}")

    theta = np.array(es.result.xfavorite)
    np.save(f"{CKPT}/controller_tau{temperature}.npy", theta)

    print("\n===== 最终评估（100 个 episode）=====")
    rm, rs = eval_real(theta, vae, rnn, n_ep=100, seed=7777)
    dm = float(np.mean(dream.rollout_batch([theta] * 8, seed=999)))
    rng = np.random.RandomState(0); rand = []
    for k in range(100):
        env = DodgeEnv(max_steps=MAX_STEPS, seed=7777 + k); env.reset(); t = 0
        while True:
            _, _, dd = env.step(rng.randint(3)); t += 1
            if dd: break
        rand.append(t)
    print(f"随机策略          : {np.mean(rand):6.1f} ± {np.std(rand):5.1f}")
    print(f"梦中训练的 C（梦里）: {dm:6.1f}")
    print(f"梦中训练的 C（真实）: {rm:6.1f} ± {rs:5.1f}   <-- 零真实环境交互")
    json.dump({"temperature": temperature, "hist": hist,
               "final_real": rm, "final_dream": dm, "random": float(np.mean(rand))},
              open(f"{CKPT}/result_tau{temperature}.json", "w"), indent=2)
    return rm, dm


# ================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "collect", "vae", "rnn", "controller", "diag"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.5)   # 1.5 是本环境实测的甜点
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--vae-epochs", type=int, default=6)
    ap.add_argument("--rnn-epochs", type=int, default=10)
    ap.add_argument("--generations", type=int, default=40)
    args = ap.parse_args()
    if args.quick:
        args.episodes, args.vae_epochs, args.rnn_epochs, args.generations = 700, 4, 6, 20

    t0 = time.time()
    if args.stage in ("all", "collect"):    stage_collect(args.episodes)
    if args.stage in ("all", "vae"):        stage_vae(args.vae_epochs)
    if args.stage in ("all", "rnn"):        stage_rnn(args.rnn_epochs)
    if args.stage == "diag":           stage_diag()
    if args.stage in ("all", "controller"): stage_controller(args.generations, args.temperature)
    print(f"\n总耗时 {time.time()-t0:.0f}s")
