# /// notebook
# Recreation of Huber, Poon & Zhu (2026), "Soft-Noncrossing Bayesian Panel
# Quantile Regression for Measuring Climate Tail Risk" (arXiv:2608.04664),
# built on the GPJax v1.0 branch API + NumPyro.
#
# Run (from the GPJax repo root):
#   uv run --with marimo --with xlrd marimo edit notebooks/soft_noncrossing_panel_qr.py
# Headless:
#   uv run --with marimo --with xlrd marimo export html notebooks/soft_noncrossing_panel_qr.py -o /tmp/sncpqr.html
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Soft-Noncrossing Bayesian Panel Quantile Regression for Climate Tail Risk

    A didactic recreation of **Huber, Poon & Zhu (2026)**
    ([arXiv:2608.04664](https://arxiv.org/abs/2608.04664)) with
    **GPJax (v1.0 branch)** + **NumPyro**. Spec: repo issue #746.

    | Decision | Value |
    |---|---|
    | Quantile grid | paper's full 99 nodes, $\tau = 0.01, \dots, 0.99$ |
    | Bernstein degree | $M = 10$ |
    | Monotone cone | all 8 coefficient rows (intercept, 4 climate, 3 macro) |
    | Macro covariate scaling | per-country empirical rank transform |
    | Temperature data | ERA5, unweighted, country-level (gadm0), monthly |
    | Horizons | $h \in \{1, 4\}$ (paper: 1–10) |
    | Sampler | NUTS, 4 parallel chains × (2500 warmup + 1500 draws) |
    | Countries | all 33 GVAR economies |
    | Common-time-effect drivers $w_t$ | Δlog oil price, PPP-GDP-weighted global short rate |

    **Deviations from the paper, by decision** (details in `notebooks/docs/adr/`):

    - *Sampler* — NUTS on the joint posterior instead of the paper's bespoke
      Gibbs (Kozumi–Kobayashi augmentation, Botev truncated-MVN, precision
      sampler). Same model, different algorithm (ADR-0001).
    - *Prior measure on the monotone cone* — ordered-transformed Gaussian
      instead of a truncated MVN: identical support, different density
      (ADR-0002).
    - *Sample end* — the Figshare archive vintage of the climate data ends
      **2022Q4** (both ERA5 and CRU); the paper's sample runs to 2023Q3 via
      the live dashboard, which has no public programmatic access.

    **Caveats to keep in mind when comparing to the paper**: the working
    likelihood multiplies one asymmetric-Laplace term per (country, quarter,
    quantile node), so each observation is counted $L$ times — nominal
    credible bands are not calibrated frequentist intervals at *any* grid
    size (the paper's included). The Lemma-1 noncrossing guarantee covers
    the box of rank-transformed macro covariates at climate-zero (exactly
    the baseline scenario of the growth-at-risk exercise); where the signed
    climate shocks are negative, noncrossing is soft, not guaranteed.
    """)
    return


@app.cell
def _():
    import hashlib
    import io
    import json
    import os
    import struct
    import time
    import zipfile
    import zlib

    NUM_CHAINS = 4
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={NUM_CHAINS}"
    )

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import requests

    jax.config.update("jax_enable_x64", True)

    import gpjax as gpx
    import numpyro
    import numpyro.distributions as dist
    from jax.scipy.special import gammaln
    from numpyro.contrib.control_flow import scan as npscan
    from numpyro.distributions.transforms import OrderedTransform
    from numpyro.infer import MCMC, NUTS

    return (
        MCMC,
        NUM_CHAINS,
        NUTS,
        OrderedTransform,
        dist,
        gammaln,
        gpx,
        hashlib,
        io,
        jax,
        jnp,
        jr,
        json,
        mo,
        np,
        npscan,
        numpyro,
        os,
        pd,
        plt,
        requests,
        struct,
        time,
        zipfile,
        zlib,
    )


@app.cell
def _(np, os):
    # ---------------- configuration: every knob in one place ----------------
    SEED = 20260806
    TAUS = np.arange(1, 100) / 100.0          # paper's grid, L = 99
    M_BERN = 10                               # Bernstein degree (paper: unstated)
    HORIZONS = (1, 4)                         # paper: 1..10; default per spec
    NUM_WARMUP, NUM_SAMPLES = 2500, 1500
    TARGET_ACCEPT = 0.9
    RUNTIME_CAP_HOURS = 2.0                   # pilot stops-and-asks beyond this
    THIN_STORE = 4                            # cache thinning for posterior draws
    # NUM_CHAINS lives in the imports cell: XLA_FLAGS must be set before jax loads.

    CLIMATE_MEMBER = "WCD_CSV/gadm0/temperature/gadm0_era_tmp_un__monthly.csv"
    WCD_URL = "https://ndownloader.figshare.com/files/45222226"   # 2.7 GB archive (never fully downloaded)
    GVAR_URL = (
        "https://data.mendeley.com/public-files/datasets/kfp5fhgkvf/files/"
        "6a67972a-4dc1-46a4-8c15-e15550f05b5f/file_downloaded"
    )
    UA = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) sncpqr-notebook/1.0"
    }

    NB_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(NB_DIR, "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    # GVAR sheet code -> ISO3
    ISO3 = {
        "arg": "ARG", "austlia": "AUS", "austria": "AUT", "bel": "BEL",
        "bra": "BRA", "can": "CAN", "china": "CHN", "chl": "CHL", "fin": "FIN",
        "france": "FRA", "germ": "DEU", "india": "IND", "indns": "IDN",
        "italy": "ITA", "japan": "JPN", "kor": "KOR", "mal": "MYS",
        "mex": "MEX", "neth": "NLD", "nor": "NOR", "nzld": "NZL", "per": "PER",
        "phlp": "PHL", "safrc": "ZAF", "sarbia": "SAU", "sing": "SGP",
        "spain": "ESP", "swe": "SWE", "switz": "CHE", "thai": "THA",
        "turk": "TUR", "uk": "GBR", "usa": "USA",
    }
    COUNTRIES = sorted(ISO3.values())
    # EM membership per Sec 3.4's EM sample, extended to the full panel
    # ([INFERENCE] the paper's Table 3 partition did not render in any source).
    EMERGING = {"ARG", "BRA", "CHL", "CHN", "IDN", "IND", "KOR", "MEX", "MYS",
                "PER", "PHL", "SAU", "THA", "TUR", "ZAF"}
    SHOCK_NAMES = ("local temp", "global temp", "local temp vol", "global temp vol")
    return (
        CLIMATE_MEMBER,
        COUNTRIES,
        DATA_DIR,
        EMERGING,
        GVAR_URL,
        HORIZONS,
        ISO3,
        M_BERN,
        NUM_SAMPLES,
        NUM_WARMUP,
        RUNTIME_CAP_HOURS,
        SEED,
        SHOCK_NAMES,
        TARGET_ACCEPT,
        TAUS,
        THIN_STORE,
        UA,
        WCD_URL,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data

    Two public sources, downloaded and cached on first run:

    - **GVAR 2023 vintage** (Mohaddes & Raissi 2024, Mendeley,
      DOI [10.17632/kfp5fhgkvf.1](https://doi.org/10.17632/kfp5fhgkvf.1),
      CC BY 4.0): log real GDP `y`, inflation `Dp`, short rate `r`, real
      exchange rate `ep`, oil price `poil`, and PPP-GDP for the weights.
    - **Weighted Climate Dataset** (Gortan, Testa, Fagiolo & Lamperti 2024,
      *Scientific Data* 11:533): monthly country temperatures. The archive
      is 2.7 GB, so the single needed CSV is extracted by **HTTP-Range
      requests** — end-of-central-directory → central directory → one
      member — a ~0.5 MB transfer.
    """)
    return


@app.cell
def _(DATA_DIR, GVAR_URL, UA, os, pd, zipfile):
    def _download(url, path):
        # NB: Mendeley's WAF rejects `requests`' fingerprint; urllib + UA passes.
        import urllib.request

        if not os.path.exists(path):
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as fh:
                fh.write(r.read())
        return path

    gvar_zip = zipfile.ZipFile(_download(GVAR_URL, os.path.join(DATA_DIR, "gvar.zip")))
    _base = "GVAR Database (1979Q2-2023Q3)/"
    gvar_book = pd.ExcelFile(gvar_zip.open(_base + "GVAR_2023Q3.xls"))
    ppp_book = pd.ExcelFile(gvar_zip.open(_base + "PPP-GDP WDI (1990-2018).xls"))
    return gvar_book, ppp_book


@app.cell
def _(
    CLIMATE_MEMBER,
    DATA_DIR,
    UA,
    WCD_URL,
    io,
    os,
    pd,
    requests,
    struct,
    zlib,
):
    def _ranged(sess, url, start, end):
        r = sess.get(url, headers={"Range": f"bytes={start}-{end}", **UA}, timeout=120)
        r.raise_for_status()
        assert r.status_code == 206, "server did not honour Range"
        return r.content

    def fetch_archive_member(url, member, cache_path):
        """Extract one member from a remote zip without downloading the archive."""
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fh:
                return fh.read()
        sess = requests.Session()
        r0 = sess.get(url, headers={"Range": "bytes=0-0", **UA}, timeout=60, stream=True)
        r0.raise_for_status()
        if "Content-Range" not in r0.headers:
            raise RuntimeError("server no longer honours Range requests")
        size = int(r0.headers["Content-Range"].split("/")[-1])
        r0.close()
        tail = _ranged(sess, url, size - 65536, size - 1)
        eocd = tail.rfind(b"PK\x05\x06")
        cd_size, cd_off = struct.unpack("<II", tail[eocd + 12 : eocd + 20])
        loc = eocd - 20 if tail[eocd - 20 : eocd - 16] == b"PK\x06\x07" else -1
        if cd_off == 0xFFFFFFFF or loc >= 0:                     # zip64
            z64_off = struct.unpack("<Q", tail[loc + 8 : loc + 16])[0]
            z64 = _ranged(sess, url, z64_off, z64_off + 55)
            assert z64[:4] == b"PK\x06\x06", "bad zip64 EOCD"
            cd_size = struct.unpack("<Q", z64[40:48])[0]
            cd_off = struct.unpack("<Q", z64[48:56])[0]
        cd = _ranged(sess, url, cd_off, cd_off + cd_size - 1)
        pos = 0
        while pos < len(cd) and cd[pos : pos + 4] == b"PK\x01\x02":
            method = struct.unpack("<H", cd[pos + 10 : pos + 12])[0]
            csz, usz = struct.unpack("<II", cd[pos + 20 : pos + 28])
            nlen, elen, clen = struct.unpack("<HHH", cd[pos + 28 : pos + 34])
            lho = struct.unpack("<I", cd[pos + 42 : pos + 46])[0]
            name = cd[pos + 46 : pos + 46 + nlen].decode()
            extra = cd[pos + 46 + nlen : pos + 46 + nlen + elen]
            ep_ = 0
            while ep_ + 4 <= len(extra):                          # zip64 extras
                hid, hsz = struct.unpack("<HH", extra[ep_ : ep_ + 4])
                if hid == 1:
                    off8 = ep_ + 4
                    if usz == 0xFFFFFFFF:
                        usz = struct.unpack("<Q", extra[off8 : off8 + 8])[0]; off8 += 8
                    if csz == 0xFFFFFFFF:
                        csz = struct.unpack("<Q", extra[off8 : off8 + 8])[0]; off8 += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", extra[off8 : off8 + 8])[0]
                ep_ += 4 + hsz
            if name == member:
                lh = _ranged(sess, url, lho, lho + 29)
                n2, e2 = struct.unpack("<HH", lh[26:30])
                ds = lho + 30 + n2 + e2
                raw = _ranged(sess, url, ds, ds + csz - 1)
                blob = zlib.decompress(raw, -15) if method == 8 else raw
                with open(cache_path, "wb") as fh:
                    fh.write(blob)
                return blob
            pos += 46 + nlen + elen + clen
        raise KeyError(member)

    _blob = fetch_archive_member(
        WCD_URL, CLIMATE_MEMBER, os.path.join(DATA_DIR, os.path.basename(CLIMATE_MEMBER))
    )
    temp_monthly = pd.read_csv(io.BytesIO(_blob))
    return (temp_monthly,)


@app.cell
def _(COUNTRIES, HORIZONS, ISO3, gvar_book, np, pd, ppp_book, temp_monthly):
    # ---------------- shocks and estimation arrays (Sec 3.1-3.2) ----------------
    def _panel(sheet):
        df = gvar_book.parse(sheet).set_index("date").rename(columns=ISO3)[COUNTRIES]
        df.index = pd.PeriodIndex(df.index, freq="Q")
        return df

    Y_log, DP, EP, R_short = _panel("y"), _panel("Dp"), _panel("ep"), _panel("r")
    _poil = gvar_book.parse("poil").set_index("date")["poil_updated"]
    _poil.index = pd.PeriodIndex(_poil.index, freq="Q")

    _ppp = ppp_book.parse("WDI").set_index("Country Code")
    # paper Sec 3.1: omega_i from average PPP GDP over 2014-2016 (their choice, not ours)
    w_ppp = _ppp.loc[COUNTRIES, [2014, 2015, 2016]].astype(float).mean(axis=1)
    w_ppp = (w_ppp / w_ppp.sum()).values                      # omega_i, eq. (6)

    # deseasonalise monthly temps (month-dummy residuals), aggregate to quarters
    _tm = temp_monthly.set_index("Date")[COUNTRIES]
    _month = np.array([int(d[5:7]) for d in _tm.index])
    _des = _tm.values - np.stack(
        [_tm.values[_month == m].mean(0) for m in range(1, 13)]
    )[_month - 1]
    _qidx = pd.PeriodIndex(
        [f"{d[1:5]}Q{(int(d[5:7]) - 1) // 3 + 1}" for d in _tm.index], freq="Q"
    )
    T_q = pd.DataFrame(_des, index=_qidx, columns=COUNTRIES).groupby(level=0).mean()

    # eq. (4): AR filter with h=8, P=8, OLS per country over the full sample
    _H, _P = 8, 8
    _lags = range(_H, _H + _P + 1)
    _Tv, _rows, _start = T_q.values, T_q.shape[0], _H + _P
    _That = np.full_like(_Tv, np.nan)
    for _i in range(len(COUNTRIES)):
        _X = np.column_stack(
            [np.ones(_rows - _start)] + [_Tv[_start - l : _rows - l, _i] for l in _lags]
        )
        _b = np.linalg.lstsq(_X, _Tv[_start:, _i], rcond=None)[0]
        _That[_start:, _i] = _Tv[_start:, _i] - _X @ _b
    T_hat = pd.DataFrame(_That, index=T_q.index, columns=COUNTRIES)
    RV = np.sqrt((T_hat**2).rolling(4).mean())                # eq. (5)
    TG = T_hat @ w_ppp                                        # eq. (6)
    TL = T_hat.sub(TG, axis=0)                                # eq. (7)
    RVG = RV @ w_ppp                                          # eq. (8)
    RVL = RV.sub(RVG, axis=0)

    growth = 400 * Y_log.diff()
    drer = EP.diff()
    dpoil = _poil.diff()
    rstar = pd.Series(R_short.values @ w_ppp, index=R_short.index)

    def _rank01(df):                                           # per-country PIT
        return df.rank(axis=0).sub(0.5).div(df.notna().sum(axis=0), axis=1)

    _t0 = pd.Period("1979Q3", freq="Q")
    _clim_last = T_hat.dropna().index[-1]

    datasets = {}
    for _h in HORIZONS:
        _tt = pd.period_range(_t0, min(Y_log.index[-1] - _h, _clim_last), freq="Q")
        _y_h = pd.DataFrame(
            {c: (400 / _h) * (Y_log[c].shift(-_h) - Y_log[c]) for c in COUNTRIES}
        ).loc[_tt]                                             # eq. (3)
        _Xm = np.stack(
            [_rank01(growth.loc[_tt]).values, _rank01(DP.loc[_tt]).values,
             _rank01(drer.loc[_tt]).values], -1)
        _clim = np.stack(
            [TL.loc[_tt].div(TL.loc[_tt].std()).values,
             np.repeat((TG.loc[_tt] / TG.loc[_tt].std()).values[:, None], len(COUNTRIES), 1),
             RVL.loc[_tt].div(RVL.loc[_tt].std()).values,
             np.repeat((RVG.loc[_tt] / RVG.loc[_tt].std()).values[:, None], len(COUNTRIES), 1)],
            -1)                                                # standardised quartet
        _W = np.stack(
            [(dpoil.loc[_tt] / dpoil.loc[_tt].std()).values,
             ((rstar.loc[_tt] - rstar.loc[_tt].mean()) / rstar.loc[_tt].std()).values],
            -1)
        _R = np.concatenate(
            [np.ones((len(COUNTRIES), len(_tt), 1)),
             np.swapaxes(_clim, 0, 1), np.swapaxes(_Xm, 0, 1)], -1)
        assert not (np.isnan(_R).any() or np.isnan(_y_h.values).any() or np.isnan(_W).any())
        datasets[_h] = dict(t=_tt, Rmat=_R, y=_y_h.values.T, W=_W)

    sample_note = {
        h: f"{d['t'][0]}..{d['t'][-1]} (T={len(d['t'])})" for h, d in datasets.items()
    }
    return RVG, TG, datasets, sample_note


@app.cell(hide_code=True)
def _(RVG, TG, mo, plt, sample_note):
    _fig, _axes = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
    _tg = TG.dropna()
    for _ax, _s, _lab in zip(_axes, [_tg, RVG.dropna()], ["global temperature shock", "global temperature volatility"]):
        _ax.plot(_s.index.to_timestamp(), _s.values, lw=0.9, color="k")
        _ax.set_title(_lab, fontsize=9)
        for _a, _b in [(1982, 1984), (1997, 1999), (2002, 2004), (2009, 2011), (2015, 2017)]:
            _ax.axvspan(str(_a), str(_b), color="tab:red", alpha=0.12)
    _fig.suptitle("Recreation of paper Fig. 3 — El Niño episodes shaded", fontsize=10)
    _fig.tight_layout()
    mo.vstack([mo.md(f"Estimation samples: `{sample_note}`"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Model

    For each horizon $h$, the panel quantile local projection (eq. 9):

    $$Q_\tau(y_{i,t+h} \mid \mathcal{F}_t) = \underbrace{\alpha_{i,h}(\tau)}_{\Theta \text{ col } 0}
    + \underbrace{f_{t,h}}_{\text{common time effect}}
    + \sum_{s} \text{shock}^s_{it}\, \underbrace{\theta^s_{i,h}(\tau)}_{\Theta \text{ cols } 1\text{–}4}
    + \bm{z}_{it}'\, \underbrace{\bm{\beta}_{i,h}(\tau)}_{\Theta \text{ cols } 5\text{–}7}$$

    Each country's stacked **coefficient path** $\Theta_i \in \mathbb{R}^{pL}$ has the
    separable prior $\mathcal{N}\!\big(\text{vec}\,\mu,\; K_\lambda \otimes \Sigma\big)$:
    the **population path** $\mu(\tau) = \Gamma \Phi(\tau)$ is a Bernstein
    polynomial with every row nondecreasing (the monotone cone, via an
    ordered transform), $K_\lambda$ is the paper's exponential kernel —
    **GPJax `Matern12`** evaluated on the quantile grid — and $\Sigma$
    (LKJ) couples the $p$ coefficients within a node. **Unit deviations**
    around $\mu$ are small (HalfNormal(0.25) scales): that is what makes
    noncrossing *soft*. The **common time effect** is a sum-to-zero AR(1)
    driven by $w_t$. The **working likelihood** places one
    `AsymmetricLaplaceQuantile` term at every (country, quarter, node).
    """)
    return


@app.cell
def _(OrderedTransform, dist, gammaln, gpx, jnp, npscan, numpyro):
    def bernstein_basis(taus, M):
        m = jnp.arange(M + 1)
        binom = jnp.exp(gammaln(M + 1) - gammaln(m + 1) - gammaln(M - m + 1))
        return binom * taus[:, None] ** m * (1 - taus[:, None]) ** (M - m)

    def panel_qr_model(Rmat, y, W, taus, M):
        """Soft-noncrossing panel QR, eq. (9); priors per spec issue #746 (Q10)."""
        N, T, p = Rmat.shape
        L = taus.shape[0]
        Phi = bernstein_basis(taus, M)

        # monotone Bernstein population path: all p rows in the cone
        Gamma = numpyro.sample(
            "Gamma",
            dist.TransformedDistribution(
                dist.Normal(0.0, 2.0).expand([p, M + 1]).to_event(2),
                OrderedTransform(),
            ),
        )
        mu = Phi @ Gamma.T
        numpyro.deterministic("mu_path", mu)

        # separable GP prior across the quantile grid: chol(K (x) Sigma) directly
        lam = numpyro.sample("lambda_gp", dist.LogNormal(jnp.log(0.3), 0.5))
        K_lam = gpx.kernels.Matern12(
            lengthscale=lam, variance=jnp.array(1.0)
        ).gram(taus[:, None]).as_matrix() + 1e-8 * jnp.eye(L)
        sd = numpyro.sample("sigma_theta", dist.HalfNormal(0.25).expand([p]).to_event(1))
        L_corr = numpyro.sample("L_corr", dist.LKJCholesky(p, concentration=2.0))
        tril = jnp.kron(jnp.linalg.cholesky(K_lam), sd[:, None] * L_corr)
        Theta = numpyro.sample(
            "Theta",
            dist.MultivariateNormal(loc=mu.reshape(-1), scale_tril=tril).expand([N]),
        ).reshape(N, L, p)

        # quantile-invariant common time effect: sum-to-zero AR(1) with drivers
        gamma_f = numpyro.sample("gamma_f", dist.Uniform(-0.99, 0.99))
        xi = numpyro.sample("xi", dist.Normal(0.0, 1.0).expand([W.shape[1]]).to_event(1))
        sig_f = numpyro.sample("sigma_f", dist.HalfNormal(0.2))
        g0 = numpyro.sample("g0", dist.Normal(0.0, 1.0))

        def _step(g_prev, w_t):
            g_t = numpyro.sample("g_innov", dist.Normal(gamma_f * g_prev + w_t @ xi, sig_f))
            return g_t, g_t

        _, g = npscan(_step, g0, W)
        f = numpyro.deterministic("f", g - g.mean())

        # working likelihood: one ALD term per (i, t, tau); quantile-specific scales
        sig_tau = numpyro.sample("sigma_ald", dist.HalfNormal(2.0).expand([L]).to_event(1))
        q = jnp.einsum("ntp,nlp->ntl", Rmat, Theta) + f[None, :, None]
        numpyro.sample(
            "y",
            dist.AsymmetricLaplaceQuantile(loc=q, scale=sig_tau, quantile=taus).to_event(3),
            obs=jnp.broadcast_to(y[:, :, None], q.shape),
        )

    return (panel_qr_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Synthetic validation (the seam)

    Before touching real data, the *identical* model code is fit to a small
    simulated panel whose true quantile surface is known analytically
    ($y = a_i + f_t + \bm{c}'\bm{b}^c_i + \bm{z}'\bm{b}^m_i + (0.5 + 0.9 z_0)\,\varepsilon$,
    so true coefficient paths are $b + 0.9 z_0 \Phi^{-1}(\tau)$-shaped).
    Checks: NUTS divergences, population-path monotonicity in **every**
    draw, common-factor recovery, and per-node PIT calibration of the
    fitted quantile surface. At this panel size the hierarchical prior
    pools hard, so *unit-level* coefficients shrink toward the population
    path by design — the certifiable quantities are the surface and the
    population path, not per-unit coefficients.
    """)
    return


@app.cell
def _(DATA_DIR, MCMC, NUTS, SEED, jnp, jr, json, np, os, panel_qr_model):
    def run_synthetic_seam(seed=SEED, warmup=300, samples=300):
        Ns, Ts, Km, Kc = 6, 80, 3, 4
        kk = jr.split(jr.PRNGKey(seed), 8)
        Zs = jr.uniform(kk[0], (Ns, Ts, Km))
        Cs = jr.normal(kk[1], (Ns, Ts, Kc))
        Ws = jr.normal(kk[2], (Ts, 2))
        f_true = jnp.cumsum(0.25 * jr.normal(kk[3], (Ts,)))
        f_true = f_true - f_true.mean()
        a_true = 0.4 * jr.normal(kk[4], (Ns,))
        bc = jnp.array([-0.15, -0.30, -0.05, 0.10]) + 0.05 * jr.normal(kk[5], (Ns, Kc))
        bm = jnp.array([1.0, -0.6, 0.3]) + 0.2 * jr.normal(kk[6], (Ns, Km))
        het = 0.5 + 0.9 * Zs[..., 0]
        y = (
            a_true[:, None]
            + f_true[None, :]
            + jnp.einsum("ntk,nk->nt", Cs, bc)
            + jnp.einsum("ntk,nk->nt", Zs, bm)
            + het * jr.normal(kk[7], (Ns, Ts))
        )
        Rm = jnp.concatenate([jnp.ones((Ns, Ts, 1)), Cs, Zs], -1)
        taus_s = jnp.concatenate([jnp.arange(1, 6) / 100, jnp.arange(2, 20) / 20])

        mc = MCMC(
            NUTS(panel_qr_model, target_accept_prob=0.9),
            num_warmup=warmup,
            num_samples=samples,
            progress_bar=False,
        )
        mc.run(jr.PRNGKey(seed + 1), Rm, y, Ws, taus_s, 10, extra_fields=("diverging",))
        s = mc.get_samples()
        div = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
        mono = bool(jnp.all(jnp.diff(s["mu_path"], axis=1) >= -1e-12))
        fcorr = float(jnp.corrcoef(s["f"].mean(0), f_true)[0, 1])
        Th = np.asarray(s["Theta"]).reshape(-1, Ns, taus_s.shape[0], 8).mean(0)
        qf = (
            np.einsum("ntp,nlp->ntl", np.asarray(Rm), Th)
            + np.asarray(s["f"].mean(0))[None, :, None]
        )
        pit = {
            round(float(taus_s[l]), 2): round(
                float(np.mean(np.asarray(y)[:, :, None] <= qf[:, :, l : l + 1])), 3
            )
            for l in (0, 4, 11, 22)
        }
        return dict(
            divergences=div,
            mu_monotone_all_draws=mono,
            factor_corr=round(fcorr, 3),
            pit_by_node=pit,
        )

    _seam_path = os.path.join(DATA_DIR, f"seam_{SEED}_300_300.json")
    if os.path.exists(_seam_path):
        with open(_seam_path) as _fh:
            seam = json.load(_fh)  # replay: reopening never silently refits
            seam["pit_by_node"] = {
                float(k): v for k, v in seam["pit_by_node"].items()
            }  # JSON stringifies keys
    else:
        seam = run_synthetic_seam()
        with open(_seam_path, "w") as _fh:
            json.dump(seam, _fh)
    assert seam["divergences"] == 0, seam
    assert seam["mu_monotone_all_draws"], seam
    assert seam["factor_corr"] > 0.8, seam
    for _tau, _pit in seam["pit_by_node"].items():
        assert abs(_pit - _tau) < 0.05, seam  # binomial tolerance at 480 obs
    seam
    return (seam,)


@app.cell(hide_code=True)
def _(jnp, jr, mo, np, plt):
    # ------- Theorem 1, statically: crossing probability vs deviation scale -------
    _taus2 = jnp.array([0.25, 0.75])
    _lam, _p, _margin = 0.3, 8, 0.25
    _corr = float(jnp.exp(-jnp.abs(_taus2[1] - _taus2[0]) / _lam))
    _scales = np.linspace(0.01, 0.5, 30)
    _bound = _p * np.exp(-_margin**2 / (2 * _p**2 * 2 * (1 - _corr) * _scales**2))
    _k = jr.PRNGKey(0)
    _d = jr.normal(_k, (4000, len(_scales), _p)) * (
        np.sqrt(2 * (1 - _corr)) * _scales)[None, :, None]
    _z = jr.uniform(jr.PRNGKey(1), (64, _p - 1))
    _rz = jnp.concatenate([jnp.ones((64, 1)), _z], 1)
    _cross = ((_margin + _d @ _rz.T) < 0).any(-1).mean(0)
    _figT, _axT = plt.subplots(figsize=(6.5, 3))
    _axT.semilogy(_scales, np.minimum(_bound, 1.0), label="Theorem 1 union bound")
    _axT.semilogy(_scales, np.asarray(_cross) + 1e-5, label="simulated crossing freq.")
    _axT.set_xlabel(r"unit-deviation scale $\max_j \Sigma_{jj}^{1/2}$")
    _axT.set_ylabel(r"P(crossing between $\tau=.25, .75$)")
    _axT.legend(fontsize=8)
    _figT.tight_layout()
    mo.vstack([
        mo.md("**Soft noncrossing, visualised** — shrink the unit-deviation "
              "scale and crossings vanish exponentially (Theorem 1); the prior "
              "delivers noncrossing, no constraint needed at the unit level."),
        _figT,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Pilot, then the real fits

    The pilot runs a **truncated warmup in the exact production
    configuration** (4 parallel chains, full grid) and extrapolates the
    marginal late-warmup cost per draw. The projection is a **lower
    bound** — NUTS tree depth can still deepen after warmup. If it
    exceeds the runtime cap, the fit cells stop and ask (override
    checkbox below).
    """)
    return


@app.cell
def _(mo):
    pilot_button = mo.ui.run_button(label="Run pilot timing (~30-45 min)")
    fit_button = mo.ui.run_button(label="Run production fits (hours-scale)")
    cap_override = mo.ui.checkbox(label="I accept a projection above the runtime cap")
    mo.hstack([pilot_button, fit_button, cap_override], justify="start")
    return cap_override, fit_button, pilot_button


@app.cell
def _(
    MCMC,
    M_BERN,
    NUM_CHAINS,
    NUM_SAMPLES,
    NUM_WARMUP,
    NUTS,
    RUNTIME_CAP_HOURS,
    TARGET_ACCEPT,
    TAUS,
    datasets,
    jax,
    jnp,
    jr,
    mo,
    panel_qr_model,
    pilot_button,
    time,
):
    mo.stop(not pilot_button.value, mo.md("*Pilot not yet run — press the button above.*"))

    def _timed(warmup):
        d1 = datasets[1]
        t0 = time.time()
        mc = MCMC(NUTS(panel_qr_model, target_accept_prob=TARGET_ACCEPT),
                  num_warmup=warmup, num_samples=1, num_chains=NUM_CHAINS,
                  chain_method="parallel", progress_bar=False)
        mc.run(jr.PRNGKey(0), jnp.asarray(d1["Rmat"]), jnp.asarray(d1["y"]),
               jnp.asarray(d1["W"]), jnp.asarray(TAUS), M_BERN)
        # parallel chains dispatch asynchronously: block, or the clock lies
        jax.block_until_ready(mc.get_samples()["lambda_gp"])
        return time.time() - t0

    _ta = _timed(100)
    _tb = _timed(300)
    _per_draw = (_tb - _ta) / 200.0
    _proj_h = (_ta + _per_draw * (NUM_WARMUP + NUM_SAMPLES - 101)) * len(datasets) / 3600.0
    pilot_projection_hours = round(_proj_h, 2)
    mo.md(
        f"**Pilot**: late-warmup marginal cost ≈ `{_per_draw:.2f}` s/draw "
        f"(4 parallel chains, contention included). Projected total for "
        f"{len(datasets)} horizon fits: **≥ {pilot_projection_hours} h** "
        f"(lower bound; cap = {RUNTIME_CAP_HOURS} h)."
    )
    return (pilot_projection_hours,)


@app.cell
def _(
    DATA_DIR,
    MCMC,
    M_BERN,
    NUM_CHAINS,
    NUM_SAMPLES,
    NUM_WARMUP,
    NUTS,
    RUNTIME_CAP_HOURS,
    SEED,
    TARGET_ACCEPT,
    TAUS,
    THIN_STORE,
    cap_override,
    datasets,
    fit_button,
    hashlib,
    jnp,
    jr,
    json,
    mo,
    np,
    os,
    panel_qr_model,
    pilot_projection_hours,
    time,
):
    mo.stop(not fit_button.value, mo.md("*Production fits not started.*"))
    mo.stop(
        pilot_projection_hours > RUNTIME_CAP_HOURS and not cap_override.value,
        mo.md(f"**Stopped**: projection {pilot_projection_hours} h exceeds the "
              f"{RUNTIME_CAP_HOURS} h cap. Tick the override to proceed anyway."),
    )

    def _fit_cached(h):
        d = datasets[h]
        key = hashlib.sha1(
            json.dumps([h, list(map(float, TAUS)), M_BERN, NUM_WARMUP, NUM_SAMPLES,
                        NUM_CHAINS, SEED, TARGET_ACCEPT, THIN_STORE]).encode()
            + np.ascontiguousarray(d["y"]).tobytes()
            + np.ascontiguousarray(d["Rmat"]).tobytes()
            + np.ascontiguousarray(d["W"]).tobytes()
        ).hexdigest()[:12]
        path = os.path.join(DATA_DIR, f"posterior_h{h}_{key}.npz")
        if os.path.exists(path):
            z = np.load(path)
            return {k: z[k] for k in z.files}
        _t0 = time.time()
        mc = MCMC(NUTS(panel_qr_model, target_accept_prob=TARGET_ACCEPT),
                  num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                  num_chains=NUM_CHAINS, chain_method="parallel",
                  progress_bar=False)
        mc.run(jr.PRNGKey(SEED + h), jnp.asarray(d["Rmat"]), jnp.asarray(d["y"]),
               jnp.asarray(d["W"]), jnp.asarray(TAUS), M_BERN,
               extra_fields=("diverging",))
        s = mc.get_samples(group_by_chain=True)
        _rhat_max, _ess_min = _diag_extremes(s)
        out = {
            "Theta": np.asarray(s["Theta"], np.float32)[:, ::THIN_STORE].reshape(
                -1, *s["Theta"].shape[2:]),
            "mu_path": np.asarray(s["mu_path"], np.float32)[:, ::THIN_STORE].reshape(
                -1, *s["mu_path"].shape[2:]),
            "f": np.asarray(s["f"], np.float32)[:, ::THIN_STORE].reshape(
                -1, *s["f"].shape[2:]),
            "divergences": np.array(
                int(np.asarray(mc.get_extra_fields()["diverging"]).sum())),
            "rhat_max": np.array(_rhat_max),
            "ess_min": np.array(_ess_min),
            "wall_s": np.array(round(time.time() - _t0, 1)),
        }
        np.savez_compressed(path, **out)
        return out

    def _diag_extremes(s):
        # finite-filtered: at degenerate draw counts these emit NaN/inf noise
        from numpyro.diagnostics import effective_sample_size, gelman_rubin
        rmax, emin = 0.0, np.inf
        for k in ("mu_path", "f", "lambda_gp", "gamma_f"):
            if k in s:
                r = np.asarray(gelman_rubin(np.asarray(s[k]))).ravel()
                e = np.asarray(effective_sample_size(np.asarray(s[k]))).ravel()
                r, e = r[np.isfinite(r)], e[np.isfinite(e)]
                if r.size:
                    rmax = max(rmax, float(r.max()))
                if e.size:
                    emin = min(emin, float(e.min()))
        return rmax, emin

    posteriors = {h: _fit_cached(h) for h in datasets}
    mo.md("**Fits complete.** " + " · ".join(
        f"h={h}: div={int(p['divergences'])}, max R̂={float(p['rhat_max']):.3f}, "
        f"min ESS={float(p['ess_min']):.0f}, wall={float(p['wall_s'])/3600:.2f} h"
        for h, p in posteriors.items()))
    return (posteriors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Results — distributional impulse responses, growth-at-risk

    The **DIRF** is the climate coefficient path itself (eq. 11); the
    panel average (eq. 12) recreates the paper's Figure 4. **ΔGaR** at a
    node is the coefficient path evaluated there (one-s.d. shock, eq. 15);
    **ΔES(0.05)** averages the five tail nodes (eq. 17–18). The country
    selector below only re-indexes stored posterior draws.
    """)
    return


@app.cell
def _(COUNTRIES, mo, posteriors):
    country_pick = mo.ui.dropdown(options=COUNTRIES, value="USA", label="Country")
    horizon_pick = mo.ui.dropdown(
        options=[str(h) for h in posteriors], value=str(min(posteriors)), label="Horizon h")
    mo.hstack([country_pick, horizon_pick], justify="start")
    return country_pick, horizon_pick


@app.cell(hide_code=True)
def _(
    COUNTRIES,
    SHOCK_NAMES,
    TAUS,
    country_pick,
    horizon_pick,
    np,
    plt,
    posteriors,
):
    _h = int(horizon_pick.value)
    _P = posteriors[_h]
    _Theta = _P["Theta"]                                   # (draws, N, L*p) -> reshape
    _L = len(TAUS)
    _Theta = _Theta.reshape(_Theta.shape[0], len(COUNTRIES), _L, 8)
    _dirf_panel = _Theta[:, :, :, 1:5].mean(1)             # (draws, L, 4) panel average
    _i = COUNTRIES.index(country_pick.value)
    _dirf_ctry = _Theta[:, _i, :, 1:5]

    _fig, _axes = plt.subplots(2, 4, figsize=(11, 5), sharex=True)
    for _row, (_d, _ttl) in enumerate([(_dirf_panel, "panel average"),
                                       (_dirf_ctry, country_pick.value)]):
        for _k in range(4):
            _ax = _axes[_row, _k]
            _md = np.median(_d[:, :, _k], 0)
            for _lo, _hi, _al in [(5, 95, 0.15), (16, 84, 0.3)]:
                _ax.fill_between(TAUS, np.percentile(_d[:, :, _k], _lo, 0),
                                 np.percentile(_d[:, :, _k], _hi, 0),
                                 alpha=_al, color="tab:blue", lw=0)
            _ax.plot(TAUS, _md, color="tab:blue", lw=1.2)
            _ax.axhline(0, color="k", lw=0.5)
            if _row == 0:
                _ax.set_title(SHOCK_NAMES[_k], fontsize=9)
            if _k == 0:
                _ax.set_ylabel(f"DIRF, {_ttl}", fontsize=8)
            _ax.set_xlabel(r"$\tau$", fontsize=8)
    _fig.suptitle(f"Distributional impulse responses, h={_h} (median, 68/90% bands)",
                  fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(COUNTRIES, EMERGING, TAUS, mo, np, pd, posteriors):
    # -------- eq. (13) hypothesis + climate growth-at-risk table (h=1) --------
    _L = len(TAUS)
    _rows = []
    _h1 = min(posteriors)
    _T1 = posteriors[_h1]["Theta"].reshape(-1, len(COUNTRIES), _L, 8)
    _tail = TAUS <= 0.05
    _i05, _i10, _i50 = (int(np.argmin(np.abs(TAUS - t))) for t in (0.05, 0.10, 0.50))
    hyp13 = {}
    for _k, _nm in enumerate(("local temp vol", "global temp vol")):
        _dbar = _T1[:, :, :, 3 + _k].mean(1)
        hyp13[_nm] = round(float(np.mean(_dbar[:, _i10] < _dbar[:, _i50])), 3)
    _des = _T1[:, :, :, 1:5][:, :, _tail, :].mean(2)       # (draws, N, 4) = ΔES per shock
    _comb = _des.sum(-1)                                    # combined four-shock ΔES
    _gar05 = _T1[:, :, _i05, 1:5].sum(-1)                   # combined ΔGaR(0.05), eq. (15)
    _gar10 = _T1[:, :, _i10, 1:5].sum(-1)                   # combined ΔGaR(0.10)
    for _i, _c in enumerate(COUNTRIES):
        _rows.append({
            "country": _c,
            "group": "EM" if _c in EMERGING else "AE",
            **{f"dES {s}": round(float(np.median(_des[:, _i, _k])), 3)
               for _k, s in enumerate(("L,T", "G,T", "L,V", "G,V"))},
            "dGaR(.05) combined": round(float(np.median(_gar05[:, _i])), 3),
            "dGaR(.10) combined": round(float(np.median(_gar10[:, _i])), 3),
            "dES combined": round(float(np.median(_comb[:, _i])), 3),
        })
    gar_table = pd.DataFrame(_rows).set_index("country")
    _avg = gar_table.groupby("group")[
        ["dGaR(.05) combined", "dGaR(.10) combined", "dES combined"]
    ].mean().round(3)
    _panel_avg = round(float(gar_table["dES combined"].mean()), 3)
    mo.vstack([
        mo.md(
            f"**Eq. (13)** P(DIRF̄({0.10}) < DIRF̄({0.50})): `{hyp13}`  \n"
            f"**ΔES(0.05), h={_h1}, combined four shocks** — panel average "
            f"**{_panel_avg}** pp (paper: −0.460); EM/AE combined averages:  \n"
            f"`{_avg.to_dict('index')}`  \n"
            f"(paper ΔES: −0.530 / −0.395). ΔGaR at a single node is the DIRF "
            f"there (eq. 11/15); the table reports the combined four-shock "
            f"scenario. EM/AE split beyond §3.4's named sample is inferred."),
        mo.ui.table(gar_table.reset_index(), page_size=33),
    ])
    return


@app.cell(hide_code=True)
def _(COUNTRIES, TAUS, mo, np, posteriors):
    # -------- diagnostics: realized crossing rate on the Lemma-1 domain --------
    _L = len(TAUS)
    _msgs = []
    for _h, _P in posteriors.items():
        _Th = _P["Theta"].reshape(-1, len(COUNTRIES), _L, 8)[::10]   # thin for speed
        _corners = np.array([[float(b) for b in np.binary_repr(_j, 3)] for _j in range(8)])
        _q = _Th[:, :, :, 0][..., None] + np.einsum("dnlk,ck->dnlc", _Th[:, :, :, 5:8], _corners)
        _cross = float(np.mean(np.any(np.diff(_q, axis=2) < 0, axis=(2, 3))))
        _msgs.append(f"h={_h}: crossing rate {100*_cross:.1f}% "
                     f"(baseline scenario domain, z-box corners)")
    mo.md("**Soft noncrossing, realized** — " + " · ".join(_msgs))
    return


@app.cell(hide_code=True)
def _(jax, mo, np, seam):
    import gpjax as _gpx
    import numpyro as _npy
    mo.md(
        f"""
        ---
        **Reproducibility** — gpjax `{_gpx.__version__}` (v1.0 branch) ·
        numpyro `{_npy.__version__}` · jax `{jax.__version__}` ·
        numpy `{np.__version__}` · devices `{jax.local_device_count()}` ·
        synthetic seam: `{seam}`

        Data: Mohaddes & Raissi (2024), GVAR 2023 vintage (CC BY 4.0);
        Gortan, Testa, Fagiolo & Lamperti (2024), *Scientific Data* 11:533.
        """
    )
    return


if __name__ == "__main__":
    app.run()
