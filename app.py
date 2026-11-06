"""Experimentos de Cinesiologia — LEMOH · UFPA (versão Streamlit).

Rodar localmente:  streamlit run app.py
"""
import io
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.io import loadmat
from scipy.signal import butter, detrend, filtfilt, find_peaks

BASE = Path(__file__).parent
DATA = BASE / "data"
IMG = BASE / "img"


def find_file(name):
    """Procura o arquivo em data/, img/, na raiz do repositório ou em qualquer subpasta."""
    for d in (DATA, IMG, BASE):
        if (d / name).is_file():
            return d / name
    for p in BASE.rglob(name):
        if p.is_file() and ".git" not in p.parts:
            return p
    return None
C = ["#1F5FA8", "#C93A25", "#17806A"]  # azul, vermelho, verde
SEL = "rgba(201,58,37,0.14)"

st.set_page_config(page_title="Experimentos de Cinesiologia", page_icon="📈", layout="wide")

# --------------------------------------------------------------------------- configuração
EMG = {
    "Ombro": dict(
        title="RMS do ombro",
        intro="Eletromiografia do deltoide anterior e do deltoide posterior durante a flexão e a extensão do ombro.",
        muscles=["Deltoide anterior", "Deltoide posterior"],
        conds=[("Flexão", "ombro_flexao.txt", [1, 2]), ("Extensão", "ombro_extensao.txt", [2, 1])]),
    "Cotovelo": dict(
        title="RMS do cotovelo",
        intro="Eletromiografia do bíceps braquial com o antebraço em pronação, em supinação e no movimento combinado.",
        muscles=["Bíceps braquial"],
        conds=[("Prono", "cotovelo_prono.txt", [1]), ("Supino", "cotovelo_supino.txt", [1]),
               ("Combinado", "cotovelo_combinado.txt", [1])]),
    "Punho e mão": dict(
        title="RMS do punho e da mão",
        intro="Eletromiografia do flexor longo dos dedos e do extensor comum dos dedos durante a flexão e a extensão do punho.",
        muscles=["Flexor longo dos dedos", "Extensor comum dos dedos"],
        conds=[("Flexão", "punho_flexao.txt", [2, 1]), ("Extensão", "punho_extensao.txt", [2, 1])]),
    "Joelho": dict(
        title="RMS do joelho",
        intro="Eletromiografia do reto femoral (RF) e do semitendinoso (ST) durante a flexão e a extensão do joelho.",
        muscles=["Reto femoral (RF)", "Semitendinoso (ST)"],
        conds=[("Flexão", "joelho_flexao.txt", [1, 2]), ("Extensão", "joelho_extensao.txt", [1, 2])]),
    "Tornozelo": dict(
        title="RMS do tornozelo",
        intro="Eletromiografia do tibial anterior (TA) e do gastrocnêmio (GAS) no apoio unipodal no chão e sobre a prancha de equilíbrio.",
        muscles=["Tibial anterior (TA)", "Gastrocnêmio (GAS)"],
        conds=[("Unipodal no chão", "tornozelo_chao.txt", [1, 2]), ("Unipodal na prancha", "tornozelo_prancha.txt", [2, 1])]),
    "Cervical": dict(
        title="RMS da cervical",
        intro="Eletromiografia do esternocleidomastoideo esquerdo e direito (ECM E e ECM D) na flexão, na inclinação para a direita e na rotação para a esquerda.",
        muscles=["ECM esquerdo", "ECM direito"],
        conds=[("Flexão cervical", "cervical_flexao.txt", [1, 2]),
               ("Inclinação lado direito", "cervical_inclinacao_dir.txt", [1, 2]),
               ("Rotação lado esquerdo", "cervical_rotacao_esq.txt", [1, 2])]),
}
TABS = ["Planos e eixos", "Ombro", "Cotovelo", "Punho e mão", "Joelho", "Tornozelo", "Cervical", "Marcha", "Equilíbrio"]

# --------------------------------------------------------------------------- processamento
def nice_ceil(v):
    if not v or not np.isfinite(v) or v <= 0:
        return 1.0
    p = 10 ** math.floor(math.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * p >= v * 1.02:
            return m * p
    return 10 * p


def bandpass(x, hi, fs=100):
    """Butterworth 4ª ordem 0,1–hi Hz com filtfilt (padding igual ao MATLAB)."""
    b, a = butter(4, [0.1 / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x, padlen=3 * (max(len(a), len(b)) - 1))


def rms(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def win_mask(t, win):
    m = (t >= win[0]) & (t <= win[1])
    if not m.any():
        m[np.searchsorted(t, win[0]).clip(0, len(t) - 1)] = True
    return m


def fmt(v, d=2):
    if v is None or not np.isfinite(v):
        return "—"
    s = f"{v:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def minmax(t, y, n=1200):
    """Reduz pontos só para desenhar, preservando picos (envelope mín/máx)."""
    if len(t) <= 2 * n:
        return t, y
    k = len(t) // n
    m = len(t) // k * k
    tt = t[:m].reshape(-1, k)
    yy = y[:m].reshape(-1, k)
    lo, hi = np.nanargmin(yy, 1), np.nanargmax(yy, 1)
    a, b = np.minimum(lo, hi), np.maximum(lo, hi)
    r = np.arange(len(tt))
    return np.column_stack([tt[r, a], tt[r, b]]).ravel(), np.column_stack([yy[r, a], yy[r, b]]).ravel()


# --------------------------------------------------------------------------- leitura de arquivos
def parse_text(text):
    rows, header = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ";" in line:
            parts = [s.strip().replace(",", ".") for s in line.split(";")]
        elif "\t" in line:
            parts = line.split("\t")
        elif "," in line:
            parts = line.split(",")
        else:
            parts = line.split()
        parts = [s.strip() for s in parts if s.strip() != ""]
        try:
            nums = [float(s) for s in parts]
        except ValueError:
            if not rows:
                header = parts
            continue
        rows.append(nums)
    if not rows:
        raise ValueError("Não encontrei linhas numéricas no arquivo.")
    nc = max(len(r) for r in rows[:50])
    arr = np.full((len(rows), nc), np.nan)
    for i, r in enumerate(rows):
        arr[i, :min(nc, len(r))] = r[:nc]
    return header, [arr[:, j] for j in range(nc)]


def parse_mat(raw):
    m = loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)

    def find(v):
        if hasattr(v, "_fieldnames"):
            if "ML" in v._fieldnames and "AP" in v._fieldnames:
                return v
            for f in v._fieldnames:
                r = find(getattr(v, f))
                if r is not None:
                    return r
        if isinstance(v, np.ndarray) and v.dtype == object:
            for x in v.ravel():
                r = find(x)
                if r is not None:
                    return r
        return None

    s = find(m.get("dataToExport"))
    if s is None:
        for k, v in m.items():
            if not k.startswith("__"):
                s = find(v)
                if s is not None:
                    break
    if s is None and "ML" in m and "AP" in m:
        return {"ML": np.ravel(m["ML"]), "AP": np.ravel(m["AP"]), "Tempo": np.ravel(m.get("Tempo", []))}
    if s is None:
        raise ValueError("Não encontrei as variáveis ML e AP no .mat.")
    return {f: np.ravel(getattr(s, f)).astype(float) for f in s._fieldnames}


@st.cache_data(show_spinner=False)
def read_bytes(name, raw):
    if name.lower().endswith(".mat"):
        return {"mat": parse_mat(raw)}
    header, cols = parse_text(raw.decode("utf-8", errors="replace"))
    return {"header": header, "cols": cols}


def load(upload, example):
    """Retorna (dados, nome, é_do_usuário)."""
    if upload is not None:
        return read_bytes(upload.name, upload.getvalue()), upload.name, True
    p = find_file(example)
    if p is None:
        raise FileNotFoundError(f"O exemplo {example} não está no repositório. Suba esse arquivo (na raiz ou na pasta data/) ou carregue o seu.")
    return read_bytes(p.name, p.read_bytes()), p.name, False


def file_label(name, mine):
    return f":red[seu arquivo: `{name}`]" if mine else f"exemplo: `{name}`"


def side(eyebrow, title, intro, howto):
    st.caption(eyebrow.upper())
    st.markdown(f"## {title}")
    st.write(intro)
    st.markdown("\n".join(f"{i}. {h}" for i, h in enumerate(howto, 1)))


def results_block(df, fname, note=None):
    st.markdown("### Resultados")
    st.table(df.set_index(df.columns[0]))
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    if note:
        c1.caption(note)
    c2.download_button("Baixar tabela (.csv)", df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       file_name=fname, mime="text/csv", key="dl_" + fname, width="stretch")


def base_layout(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=36, b=10), showlegend=False,
                      hovermode="x unified", dragmode="zoom")
    return fig


def window_slider(t, key):
    t0, t1 = float(np.nanmin(t)), float(np.nanmax(t))
    return st.slider("Janela de análise (s)", t0, t1, (t0, t1), step=0.01, key=key,
                     help="Arraste as pontas para escolher o trecho. Com a janela toda, o RMS é do sinal inteiro.")


# --------------------------------------------------------------------------- EMG
def emg_tab(tab_name):
    cfg = EMG[tab_name]
    kid = re.sub(r"\W", "", tab_name)
    s_col, m_col = st.columns([1, 2.6], gap="large")
    conds = []
    with m_col:
        res_box = st.container(border=True)
        for i, (name, fname, cols) in enumerate(cfg["conds"]):
            box = st.container(border=True)
            with box:
                h1, h2 = st.columns([3, 2])
                up = h2.file_uploader("Carregar arquivo (.txt)", type=["txt", "csv", "dat"], key=f"{kid}_up{i}",
                                      label_visibility="collapsed")
                try:
                    d, fn, mine = load(up, fname)
                    data = d.get("cols")
                    if data is None or len(data) < 2:
                        raise ValueError("Use um arquivo de texto com colunas de tempo e EMG.")
                except Exception as e:  # noqa: BLE001
                    h1.markdown(f"### {name}")
                    st.error(str(e))
                    conds.append(None)
                    continue
                h1.markdown(f"### {name}\n{file_label(fn, mine)}")
                use = [min(c, len(data) - 1) for c in cols]
                if len(data) > 2:
                    sc = st.columns(len(cfg["muscles"]))
                    for k, m in enumerate(cfg["muscles"]):
                        use[k] = sc[k].selectbox(f"Coluna · {m}", list(range(1, len(data))), index=use[k] - 1,
                                                 format_func=lambda j: f"coluna {j + 1}", key=f"{kid}_c{i}{k}_{fn}")
                t = data[0]
                win = window_slider(t, f"{kid}_w{i}_{fn}")
                mask = win_mask(t, win)
                r = [rms(data[j][mask]) for j in use]
                conds.append(dict(name=name, t=t, ys=[data[j] for j in use], win=win, rms=r,
                                  full=win[0] <= t[0] + 1e-9 and win[1] >= t[-1] - 1e-9, slot=st.empty()))
        ok = [c for c in conds if c]
        if ok:
            ymax = nice_ceil(max(np.nanmax(np.abs(y)) for c in ok for y in c["ys"]))
            xr = [min(c["t"][0] for c in ok), max(c["t"][-1] for c in ok)]
            n = len(cfg["muscles"])
            for c in ok:
                fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.14 if n > 1 else 0.05,
                                    subplot_titles=[f"{m} · RMS {fmt(c['rms'][k], 1)}" for k, m in enumerate(cfg["muscles"])])
                for k, y in enumerate(c["ys"]):
                    tt, yy = minmax(c["t"], y)
                    fig.add_trace(go.Scatter(x=tt, y=yy, mode="lines", line=dict(color=C[k], width=1), name=cfg["muscles"][k]), k + 1, 1)
                    if not c["full"]:
                        fig.add_vrect(x0=c["win"][0], x1=c["win"][1], fillcolor=SEL, line_width=0, row=k + 1, col=1)
                fig.update_yaxes(range=[-ymax, ymax])
                fig.update_xaxes(range=xr)
                fig.update_xaxes(title_text="tempo (s)", row=n, col=1)
                c["slot"].plotly_chart(base_layout(fig, 190 * n + 50), width="stretch", key=f"{kid}_fig_{c['name']}")
    with s_col:
        side("Eletromiografia · RMS", cfg["title"], cfg["intro"], [
            "Os dados de exemplo já estão carregados. Para usar os seus, arraste o .txt (tempo, canal 1, canal 2…) para a caixa de cada condição.",
            "Ajuste a janela de análise no controle deslizante. Ela vale para todos os músculos da condição.",
            "Com a janela inteira, o RMS é do sinal inteiro.",
            "Todos os gráficos da aba usam a mesma escala de amplitude e de tempo, para comparar condições e músculos."])
        rows = []
        for (name, _, _), c in zip(cfg["conds"], conds):
            row = {"Condição": name, "Janela (s)": "—" if not c else ("inteiro" if c["full"] else f"{fmt(c['win'][0])}–{fmt(c['win'][1])}")}
            for k, m in enumerate(cfg["muscles"]):
                row[m] = fmt(c["rms"][k], 1) if c else "—"
            rows.append(row)
        with res_box:
          results_block(pd.DataFrame(rows), f"rms_{kid.lower()}.csv",
                      "RMS na unidade do arquivo de EMG, sem filtragem, como no script MATLAB.")


# --------------------------------------------------------------------------- Planos e eixos
def imu_tab():
    axes = [("Vertical (Y)", 2, C[1]), ("Anteroposterior (X)", 1, C[0]), ("Mediolateral (Z)", 3, C[2])]
    panels = [("Flexão de ombro · aceleração linear", "m/s²", "eixos_acc_flexao.csv"),
              ("Flexão de ombro · velocidade angular", "rad/s", "eixos_giro_flexao.csv"),
              ("Abdução de ombro · aceleração linear", "m/s²", "eixos_acc_ext.csv"),
              ("Abdução de ombro · velocidade angular", "rad/s", "eixos_giro_ext.csv")]
    s_col, m_col = st.columns([1, 2.6], gap="large")
    res = []
    with m_col:
        res_box = st.container(border=True)
        for i, (name, unit, fname) in enumerate(panels):
            with st.container(border=True):
                h1, h2 = st.columns([3, 2])
                up = h2.file_uploader("Carregar arquivo (.csv)", type=["csv", "txt"], key=f"imu_up{i}", label_visibility="collapsed")
                try:
                    d, fn, mine = load(up, fname)
                    cols = d.get("cols")
                    if cols is None or len(cols) < 4:
                        raise ValueError("O arquivo precisa de 4 colunas: tempo, X, Y, Z.")
                    t = cols[0]
                    if np.nanmedian(np.diff(t)) > 1:
                        t = t / 1000.0
                    f = [bandpass(detrend(cols[j]), 10) for _, j, _ in axes]
                except Exception as e:  # noqa: BLE001
                    h1.markdown(f"### {name}")
                    st.error(str(e))
                    res.append(None)
                    continue
                h1.markdown(f"### {name}\n{file_label(fn, mine)}")
                win = window_slider(t, f"imu_w{i}_{fn}")
                mask = win_mask(t, win)
                res.append(dict(name=name, unit=unit, t=t, f=f, win=win, rms=[rms(x[mask]) for x in f],
                                full=win[0] <= t[0] + 1e-9 and win[1] >= t[-1] - 1e-9, slot=st.empty()))
        for p in [p for p in res if p]:
            same = [q for q in res if q and q["unit"] == p["unit"]]
            ymax = nice_ceil(max(np.nanmax(np.abs(x)) for q in same for x in q["f"]))
            xmax = max(q["t"][-1] for q in same)
            fig = go.Figure()
            for (an, _, col), x, r in zip(axes, p["f"], p["rms"]):
                tx, yx = minmax(p["t"], x)
                fig.add_trace(go.Scatter(x=tx, y=yx, mode="lines", line=dict(color=col, width=1.6),
                                           name=f"{an} · RMS {fmt(r, 3)} {p['unit']}"))
            if not p["full"]:
                fig.add_vrect(x0=p["win"][0], x1=p["win"][1], fillcolor=SEL, line_width=0)
            fig.update_yaxes(range=[-ymax, ymax], title_text=p["unit"])
            fig.update_xaxes(range=[0, xmax], title_text="tempo (s)")
            base_layout(fig, 320).update_layout(showlegend=True, legend=dict(orientation="h", y=1.12, x=0))
            p["slot"].plotly_chart(fig, width="stretch", key=f"imu_fig_{p['name']}")
    with s_col:
        side("Sensores inerciais", "Planos e eixos de movimento",
             "Sensores inerciais do celular (acelerômetro e giroscópio) registram o movimento do ombro. Compare em qual eixo o sinal é maior em cada movimento para identificar o plano e o eixo de rotação.",
             ["Os dados de exemplo já estão carregados. Para usar os seus, arraste o .csv do app (tempo em ms; eixos X, Y, Z) para a caixa de cada registro.",
              "Os sinais passam por remoção de tendência e filtro passa-banda de 0,1–10 Hz, como no script original.",
              "O RMS é do trecho escolhido no controle deslizante (por padrão, o sinal inteiro).",
              "Registros da mesma grandeza usam a mesma escala, para comparar os movimentos."])
        df = pd.DataFrame([{"Registro": n, "Unidade": u, **{a[0]: (fmt(p["rms"][k], 3) if p else "—") for k, a in enumerate(axes)}}
                           for (n, u, _), p in zip(panels, res)])
        with res_box:
          results_block(df, "rms_planos_eixos.csv", "O maior RMS aponta o eixo com mais movimento em cada registro.")


# --------------------------------------------------------------------------- Marcha
def gait_tab():
    s_col, m_col = st.columns([1, 2.6], gap="large")
    with s_col:
        side("Plataforma de força", "Marcha",
             "Força de reação do solo vertical (Fz) registrada na plataforma de força durante um apoio na caminhada e na corrida.",
             ["Os dados de exemplo já estão carregados. Para usar os seus, arraste o .txt (tempo em s, Fz em N) para a caixa da condição.",
              "O tempo de apoio é calculado automaticamente enquanto Fz fica acima do limiar.",
              "Use o controle deslizante para medir manualmente a duração de qualquer trecho.",
              "Caminhada e corrida usam a mesma escala de força e janelas de tempo de mesma duração."])
        st.markdown("#### Parâmetros")
        thr = st.number_input("Limiar de contato (N)", 0.0, 500.0, 20.0, 1.0, key="g_thr")
        mass = st.number_input("Massa corporal (kg) — opcional", 0.0, 300.0, 0.0, 0.5, key="g_mass")
        full = st.toggle("Ver registro inteiro", key="g_full")
    conds = [("Caminhada", "marcha_caminhada.txt", 0.25), ("Corrida", "marcha_corrida.txt", 0.10)]
    out = []
    with m_col:
        res_box = st.container(border=True)
        for i, (name, fname, dist0) in enumerate(conds):
            with st.container(border=True):
                h1, h2 = st.columns([3, 2])
                up = h2.file_uploader("Carregar arquivo (.txt)", type=["txt", "csv"], key=f"g_up{i}", label_visibility="collapsed")
                try:
                    d, fn, mine = load(up, fname)
                    cols = d.get("cols")
                    if cols is None or len(cols) < 2:
                        raise ValueError("O arquivo precisa das colunas tempo e Fz.")
                except Exception as e:  # noqa: BLE001
                    h1.markdown(f"### {name}")
                    st.error(str(e))
                    out.append(None)
                    continue
                h1.markdown(f"### {name}\n{file_label(fn, mine)}")
                t, fz = cols[0], np.nan_to_num(cols[1])
                above = np.flatnonzero(fz > thr)
                contact = (t[above[0]], t[above[-1]]) if above.size else None
                dist = st.number_input("Distância mínima entre picos (s)", 0.0, 5.0, dist0, 0.01, key=f"g_d{i}")
                dt = float(np.nanmedian(np.diff(t)))
                pk, _ = find_peaks(fz, height=max(thr, 0.2 * fz.max()), distance=max(1, int(round(dist / dt))))
                mk0 = contact if contact else (float(t[0]), float(t[-1]))
                mark = st.slider("Trecho marcado (s)", float(t[0]), float(t[-1]), (float(mk0[0]), float(mk0[1])), 0.01, key=f"g_m{i}_{fn}")
                out.append(dict(name=name, t=t, fz=fz, contact=contact, peak=float(fz.max()), pk=pk, mark=mark, slot=st.empty()))
        ok = [c for c in out if c]
        if ok:
            ymax = nice_ceil(max(c["peak"] for c in ok))
            ymin = math.floor(min(0.0, min(float(c["fz"].min()) for c in ok)) / 50) * 50
            W = max((c["contact"][1] - c["contact"][0]) if c["contact"] else 0 for c in ok) + 0.6
            for c in ok:
                xr = [0, max(q["t"][-1] for q in ok)] if (full or not c["contact"]) else [c["contact"][0] - 0.3, c["contact"][0] - 0.3 + W]
                fig = go.Figure(go.Scatter(x=c["t"], y=c["fz"], mode="lines", line=dict(color=C[0], width=2), name="Fz"))
                fig.add_trace(go.Scatter(x=c["t"][c["pk"]], y=c["fz"][c["pk"]], mode="markers+text", marker=dict(color=C[1], size=9),
                                         text=[f"{fmt(v, 0)} N" for v in c["fz"][c["pk"]]], textposition="top center", name="picos"))
                fig.add_vrect(x0=c["mark"][0], x1=c["mark"][1], fillcolor=SEL, line_width=0)
                fig.update_yaxes(range=[ymin, ymax], title_text="Fz (N)")
                fig.update_xaxes(range=xr, title_text="tempo (s)")
                c["slot"].plotly_chart(base_layout(fig, 320), width="stretch", key=f"g_fig_{c['name']}")
    with res_box:
        bw = mass * 9.81 if mass > 0 else None
        rows = []
        for (name, _, _), c in zip(conds, out):
            row = {"Condição": name,
                   "Apoio (s)": fmt(c["contact"][1] - c["contact"][0]) if c and c["contact"] else "—",
                   "Pico Fz (N)": fmt(c["peak"], 0) if c else "—"}
            if bw:
                row["Pico (PC)"] = fmt(c["peak"] / bw) if c else "—"
            row["Marcado (s)"] = fmt(c["mark"][1] - c["mark"][0], 3) if c else "—"
            rows.append(row)
        results_block(pd.DataFrame(rows), "marcha.csv",
                      "Apoio: tempo com Fz acima do limiar. PC: múltiplos do peso corporal. Marcado: duração do trecho do controle deslizante.")


# --------------------------------------------------------------------------- Equilíbrio
def balance_tab():
    conds = [("OA", "Olhos abertos", "equilibrio_OA.txt"), ("OF", "Olhos fechados", "equilibrio_OF.txt"),
             ("UNI", "Unipodal", "equilibrio_UNI.txt")]
    s_col, m_col = st.columns([1, 2.6], gap="large")
    out = []
    with m_col:
        res_box = st.container(border=True)
        with st.container(border=True):
            st.markdown("### Arquivos")
            fc = st.columns(3)
            for i, (sig, long, fname) in enumerate(conds):
                with fc[i]:
                    st.markdown(f"**{sig} · {long}**")
                    up = st.file_uploader(f"Arquivo {sig}", type=["mat", "txt", "csv"], key=f"b_up{i}", label_visibility="collapsed")
                    try:
                        d, fn, mine = load(up, fname)
                        if "mat" in d:
                            m = d["mat"]
                            t, ml, ap = m.get("Tempo", np.arange(len(m["ML"])) / 100), m["ML"], m["AP"]
                        else:
                            cols, h = d["cols"], " ".join(d.get("header") or []).upper()
                            if len(cols) < 3:
                                raise ValueError("O arquivo precisa das colunas tempo, ML e AP.")
                            t = cols[0]
                            ml, ap = (cols[1], cols[3]) if ("ACC" in h and len(cols) >= 4) else (cols[1], cols[2])
                        t = np.asarray(t, float)
                        if len(t) != len(ml):
                            t = np.arange(len(ml)) / 100
                        if np.nanmedian(np.diff(t)) > 1:
                            t = t / 1000
                        t = t - t[0]
                        prep = lambda x: bandpass(detrend(x) - np.mean(detrend(x)), 4)  # noqa: E731
                        ml, ap = prep(np.asarray(ml, float)), prep(np.asarray(ap, float))
                        st.markdown(file_label(fn, mine))
                        out.append(dict(sig=sig, long=long, t=t, ml=ml, ap=ap))
                    except Exception as e:  # noqa: BLE001
                        st.error(str(e))
                        out.append(None)
        ok = [c for c in out if c]
        for c in ok:
            x, y = c["ml"], c["ap"]
            lam, vec = np.linalg.eigh(np.cov(x, y))
            A, B = 2.4478 * np.sqrt(np.maximum(lam[::-1], 0))
            v = vec[:, 1]
            ang = math.atan2(v[1], v[0])
            th = np.linspace(0, 2 * np.pi, 120)
            ex, ey = A * np.cos(th), B * np.sin(th)
            c["ell"] = (x.mean() + ex * math.cos(ang) - ey * math.sin(ang), y.mean() + ex * math.sin(ang) + ey * math.cos(ang))
            c["r"] = dict(tot=float(np.sum(np.hypot(x, y))), area=float(math.pi * A * B), rx=rms(x), ry=rms(y))
        with st.container(border=True):
            view = st.radio("Visualização", ["Estatocinesigrama", "Estabilograma"], horizontal=True, key="b_view")
            if ok:
                q = max(2, math.ceil(max(np.nanmax(np.abs(np.r_[c["ml"], c["ap"]])) for c in ok)))
                if view == "Estatocinesigrama":
                    st.caption("Eixo horizontal: ML (m/s²) · eixo vertical: AP (m/s²) · em vermelho, elipse de 95%.")
                    fig = make_subplots(rows=1, cols=len(ok), subplot_titles=[f"{c['sig']} · {c['long']}" for c in ok], horizontal_spacing=0.06)
                    for k, c in enumerate(ok, 1):
                        fig.add_trace(go.Scatter(x=c["ml"], y=c["ap"], mode="lines", line=dict(color="#5A6672", width=1)), 1, k)
                        fig.add_trace(go.Scatter(x=c["ell"][0], y=c["ell"][1], mode="lines", line=dict(color=C[1], width=2.5)), 1, k)
                        fig.update_xaxes(range=[-q, q], row=1, col=k, title_text="ML", constrain="domain")
                        fig.update_yaxes(range=[-q, q], row=1, col=k, scaleanchor=f"x{k if k > 1 else ''}", scaleratio=1, constrain="domain")
                    st.plotly_chart(base_layout(fig, 400), width="stretch", key="b_skg")
                else:
                    tmax = max(c["t"][-1] for c in ok)
                    fig = make_subplots(rows=len(ok), cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                        subplot_titles=[f"{c['sig']} · {c['long']}" for c in ok])
                    for k, c in enumerate(ok, 1):
                        fig.add_trace(go.Scatter(x=c["t"], y=c["ap"], mode="lines", line=dict(color=C[0], width=1.2), name="AP", showlegend=k == 1), k, 1)
                        fig.add_trace(go.Scatter(x=c["t"], y=c["ml"], mode="lines", line=dict(color=C[1], width=1.2), name="ML", showlegend=k == 1), k, 1)
                        fig.update_yaxes(range=[-q, q], row=k, col=1, title_text="m/s²")
                    fig.update_xaxes(range=[0, tmax])
                    fig.update_xaxes(title_text="tempo (s)", row=len(ok), col=1)
                    base_layout(fig, 170 * len(ok) + 80).update_layout(showlegend=True, legend=dict(orientation="h", y=1.08, x=0))
                    st.plotly_chart(fig, width="stretch", key="b_stb")
    with s_col:
        side("Posturografia", "Equilíbrio",
             "Oscilação postural em pé nas condições olhos abertos (OA), olhos fechados (OF) e apoio unipodal (UNI).",
             ["Os dados de exemplo já estão carregados. Para usar os seus, carregue um arquivo por condição: .mat exportado (dataToExport com ML, AP, Tempo) ou .txt/.csv com colunas tempo, ML, AP.",
              "Os sinais passam por remoção de tendência e da média e por filtro passa-banda de 0,1–4 Hz.",
              "Alterne entre o estatocinesigrama (ML × AP, com elipse de 95%) e o estabilograma (sinal no tempo). As três condições usam sempre a mesma escala."])
        names = [("Deslocamento total", "tot", 1), ("Área da elipse 95%", "area", 3), ("RMS ML", "rx", 3), ("RMS AP", "ry", 3)]
        df = pd.DataFrame([{"Variável": n, **{sig: (fmt(c["r"][k], d) if c else "—") for (sig, _, _), c in zip(conds, out)}} for n, k, d in names])
        with res_box:
          results_block(df, "equilibrio.csv",
                      "Sinais em m/s². Deslocamento total = soma das distâncias ao centro (Σ√(ML²+AP²)), como no script original. Área em (m/s²)².")


# --------------------------------------------------------------------------- página
st.markdown("""<style>
.block-container{padding-top:1.6rem}
h1{font-weight:800!important;letter-spacing:-.01em}
div[data-testid="stCaptionContainer"] p{letter-spacing:.02em}
button[data-baseweb="tab"] p{font-size:1rem;font-weight:600}
</style>""", unsafe_allow_html=True)
hc1, hc2, hc3 = st.columns([1.1, 0.45, 6], vertical_alignment="center")
for col, logo in ((hc1, "lemoh.png"), (hc2, "ufpa.png")):
    lp = find_file(logo)  # logos são opcionais: o app abre mesmo sem eles
    if lp:
        col.image(lp.read_bytes(), width="stretch")
hc3.markdown("# Experimentos de Cinesiologia\nLaboratório de Estudos da Motricidade Humana · Universidade Federal do Pará")
st.caption("Cada aba abre com os dados de exemplo do repositório. Para analisar uma coleta nova, use o botão **Upload** da condição; "
           "para voltar ao exemplo, remova o arquivo enviado (×).")

tabs = st.tabs(TABS)
with tabs[0]:
    imu_tab()
for i, name in enumerate(TABS[1:7], 1):
    with tabs[i]:
        emg_tab(name)
with tabs[7]:
    gait_tab()
with tabs[8]:
    balance_tab()
