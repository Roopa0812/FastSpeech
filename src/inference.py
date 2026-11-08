"""
FastSpeech 2 inference with variance-adaptor plots (English and Telugu).

Run from the root of the FastSpeech2 repo copy, for example:
  python inference.py --lang english --dataset Svarah --restore_step 20000 \
      --speaker S000 --out /content/drive/MyDrive/tts_project/Results/English
  python inference.py --lang telugu  --dataset Telugu --restore_step 20000 \
      --speaker TEF --sentences test_sentences_telugu.json \
      --out /content/drive/MyDrive/tts_project/Results/Telugu

For every test sentence it saves:
  <out>/test_generated_audios/<id>.wav
  <out>/variance_adaptor_plots/{duration,pitch,energy}/<id>.png
  <out>/mel_plots/<id>.png , <out>/variance_data/<id>.csv , <out>/phoneme_sequences.txt
"""
import argparse, csv, json, os, re, sys
import numpy as np

# ------------------------------------------------------------------ shared helpers
def clean_sp(phones):
    """collapse repeated pauses and drop leading/trailing ones (training alignments trim edge silence)"""
    out = []
    for p in phones:
        if p == "sp" and (not out or out[-1] == "sp"):
            continue
        out.append(p)
    while out and out[-1] == "sp":
        out.pop()
    return out

# ------------------------------------------------------------------ English front-end
DEFAULT_EN = [
    ("E1", "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."),
    ("E2a", "The orchestra's conduct during the ceremony was praised by the audience."),
    ("E2b", "The researchers will conduct an experiment to evaluate the proposed method."),
    ("E3", "The rhyth m of the orchestra changed unexpectedly."),
    ("E4", "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."),
]
MONTHS = "january february march april may june july august september october november december".split()
YEAR_CTX = set(MONTHS) | {"in", "of", "from", "since", "by", "till", "until", "during", "year", "to", "before", "after"}
AMPM = {"amtok": ["EY1", "EH1", "M"], "pmtok": ["P", "IY1", "EH1", "M"]}   # "A M" / "P M" spelled out

def normalize_en(text):
    """numbers, times, dates -> words. Keeps , . ; : ? ! so they can become pauses."""
    from num2words import num2words
    clean = lambda s: s.replace(",", "").replace("-", " ")
    card = lambda n: clean(num2words(int(n)))
    ordi = lambda n: clean(num2words(int(n), to="ordinal"))
    year = lambda n: clean(num2words(int(n), to="year"))
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text.strip())
    def _time(m):
        h, mi = int(m.group(1)), int(m.group(2))
        s = card(h) + ((" oh " + card(mi)) if 0 < mi < 10 else (" " + card(mi)) if mi else "")
        ap = m.group(3)
        return s + ((" amtok" if ap.lower() == "a" else " pmtok") if ap else "")
    t = re.sub(r"\b(\d{1,2}):(\d{2})(?:\s*([AaPp])\.?[Mm]\b)?", _time, t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1 percent", t)
    t = re.sub(r"(\d+)\s*\+", r"\1 plus", t)
    t = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: ordi(m.group(1)), t, flags=re.I)
    t = re.sub(r"\b(\d+)\.(\d+)\b", lambda m: card(m.group(1)) + " point " + " ".join(card(d) for d in m.group(2)), t)
    out, last = [], 0
    for m in re.finditer(r"\d+", t):
        prev = (re.findall(r"[a-z]+", t[:m.start()].lower()) or [""])[-1]
        n = int(m.group(0))
        s = year(n) if (len(m.group(0)) == 4 and 1100 <= n <= 2099 and prev in YEAR_CTX) else card(n)
        out.append(t[last:m.start()] + " " + s + " "); last = m.end()
    t = "".join(out) + t[last:]
    t = t.replace("&", " and ").replace("@", " at ").replace("+", " plus ")
    t = re.sub(r"[^a-z' ,.;:?!]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()

def en_frontend(text):
    """whole-sentence g2p_en (CMUdict + part-of-speech homograph handling, e.g. noun vs verb 'conduct')"""
    from g2p_en import G2p
    g2p = G2p()
    t = normalize_en(text)
    phones = []
    for seg in re.split(r"\b(amtok|pmtok)\b", t):
        if seg in AMPM:
            phones += AMPM[seg]; continue
        for tok in g2p(seg):
            if tok == " ":
                continue
            if re.fullmatch(r"[A-Z]+[0-2]?", tok):
                phones.append(tok)
            elif re.fullmatch(r"[,.;:?!]", tok):
                phones.append("sp")
    return t, clean_sp(phones)

# ------------------------------------------------------------------ Telugu front-end
IV = {0x0C05:"A",0x0C06:"AA",0x0C07:"I",0x0C08:"II",0x0C09:"U",0x0C0A:"UU",0x0C0B:"R U",0x0C60:"R UU",
      0x0C0E:"E",0x0C0F:"EE",0x0C10:"AI",0x0C12:"O",0x0C13:"OO",0x0C14:"AU"}
VS = {0x0C3E:"AA",0x0C3F:"I",0x0C40:"II",0x0C41:"U",0x0C42:"UU",0x0C43:"R U",0x0C44:"R UU",
      0x0C46:"E",0x0C47:"EE",0x0C48:"AI",0x0C4A:"O",0x0C4B:"OO",0x0C4C:"AU"}
CONS = {0x0C15:"K",0x0C16:"KH",0x0C17:"G",0x0C18:"GH",0x0C19:"NG",0x0C1A:"C",0x0C1B:"CH",0x0C1C:"J",
        0x0C1D:"JH",0x0C1E:"NY",0x0C1F:"TT",0x0C20:"TTH",0x0C21:"DD",0x0C22:"DDH",0x0C23:"NN",
        0x0C24:"T",0x0C25:"TH",0x0C26:"D",0x0C27:"DH",0x0C28:"N",0x0C2A:"P",0x0C2B:"PH",0x0C2C:"B",
        0x0C2D:"BH",0x0C2E:"M",0x0C2F:"Y",0x0C30:"R",0x0C31:"RR",0x0C32:"L",0x0C33:"LL",0x0C35:"V",
        0x0C36:"SH",0x0C37:"SS",0x0C38:"S",0x0C39:"H"}
VIRAMA, ANUSVARA, VISARGA = 0x0C4D, 0x0C02, 0x0C03
NASAL = {**dict.fromkeys(["P","PH","B","BH","M"], "M"), **dict.fromkeys(["K","KH","G","GH"], "NG"),
         **dict.fromkeys(["C","CH","J","JH"], "NY"), **dict.fromkeys(["TT","TTH","DD","DDH","NN"], "NN"),
         **dict.fromkeys(["T","TH","D","DH","N"], "N")}

def g2p_word(w):
    out, i, n = [], 0, len(w)
    while i < n:
        c = ord(w[i])
        if c in IV: out += IV[c].split(); i += 1
        elif c in CONS:
            out.append(CONS[c]); i += 1
            if i < n and ord(w[i]) == VIRAMA: i += 1
            elif i < n and ord(w[i]) in VS: out += VS[ord(w[i])].split(); i += 1
            else: out.append("A")
        elif c == ANUSVARA:
            nxt = CONS.get(ord(w[i+1])) if i + 1 < n else None
            out.append(NASAL.get(nxt, "M")); i += 1
        elif c == VISARGA: out.append("H"); i += 1
        else: raise ValueError(f"unhandled character U+{c:04X} in {w!r}")
    return out

# Telugu number words (DRAFT: please have a Telugu speaker verify before relying on it)
TE_SMALL = {0:"సున్నా",1:"ఒకటి",2:"రెండు",3:"మూడు",4:"నాలుగు",5:"ఐదు",6:"ఆరు",7:"ఏడు",8:"ఎనిమిది",9:"తొమ్మిది",
            10:"పది",11:"పదకొండు",12:"పన్నెండు",13:"పదమూడు",14:"పద్నాలుగు",15:"పదిహేను",16:"పదహారు",
            17:"పదిహేడు",18:"పద్దెనిమిది",19:"పంతొమ్మిది"}
TE_TENS = {2:"ఇరవై",3:"ముప్పై",4:"నలభై",5:"యాభై",6:"అరవై",7:"డెబ్బై",8:"ఎనభై",9:"తొంభై"}

def te_words(n):
    if n < 20: return TE_SMALL[n]
    if n < 100:
        t, r = divmod(n, 10)
        return TE_TENS[t] + ("" if r == 0 else " " + TE_SMALL[r])
    if n < 1000:
        h, r = divmod(n, 100)
        if h == 1: return "వంద" if r == 0 else "నూట " + te_words(r)
        return te_words(h) + (" వందలు" if r == 0 else " వందల " + te_words(r))
    th, r = divmod(n, 1000)
    if th == 1: return "వెయ్యి" if r == 0 else "వెయ్యి " + te_words(r)
    return te_words(th) + (" వేలు" if r == 0 else " వేల " + te_words(r))

def te_ordinal(n):
    w = te_words(n)
    return (w[:-1] if w[-1] in "ుి" else w) + "వ"

def te_normalize(text):
    t = text.replace("\ufeff", "").replace("\u200c", "").replace("\u200d", "").replace("o", "\u0C02")
    t = re.sub(r"(\d{1,2}):(\d{2})",
               lambda m: te_words(int(m.group(1))) + ("" if m.group(2) == "00" else " " + te_words(int(m.group(2)))), t)
    t = re.sub(r"(\d+)\u0C35", lambda m: te_ordinal(int(m.group(1))), t)
    t = re.sub(r"\d+", lambda m: te_words(int(m.group(0))), t)
    return re.sub(r"\s+", " ", t).strip()

def te_frontend(text):
    t = te_normalize(text)
    phones = []
    for tok in re.findall(r"[\u0C00-\u0C7F]+|[,.;:?!]", t):
        if tok in ",.;:?!": phones.append("sp")
        else: phones += g2p_word(tok)
    return t, clean_sp(phones)

# ------------------------------------------------------------------ plotting
def _edges(d_frames, hop, sr):
    return np.concatenate([[0], np.cumsum(d_frames)]) * hop / sr

def _label_axis(ax, phones, edges):
    centers = (edges[:-1] + edges[1:]) / 2
    keep = [i for i in range(len(phones)) if edges[i + 1] - edges[i] > 0]
    ax.set_xticks(centers[keep]); ax.set_xticklabels([phones[i] for i in keep], rotation=90, fontsize=7)
    for e in edges: ax.axvline(e, color="0.85", lw=0.5, zorder=0)

def plot_duration(phones, d_frames, hop, sr, title, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(7, 0.28 * len(phones)), 3.8))
    ax.bar(range(len(phones)), d_frames * hop / sr * 1000, color=["#bbb" if p == "sp" else "#2a6f97" for p in phones])
    ax.set_xticks(range(len(phones))); ax.set_xticklabels(phones, rotation=90, fontsize=7)
    ax.set_ylabel("predicted duration (ms)"); ax.set_title(title); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_curve(phones, d_frames, values, ylabel, title, path, hop, sr):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    edges = _edges(d_frames, hop, sr)
    fig, ax = plt.subplots(figsize=(max(7, 0.28 * len(phones)), 3.8))
    if len(values) == len(phones):                       # phoneme-level feature: step curve over time
        ax.stairs(values, edges, color="#c1121f", lw=1.6)
    else:                                                # frame-level feature
        ax.plot(np.arange(len(values)) * hop / sr, values, color="#c1121f", lw=1.2)
    _label_axis(ax, phones, edges)
    ax.set_xlabel("time (s), phoneme boundaries in grey"); ax.set_ylabel(ylabel); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_mel(mel, title, path, hop, sr):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.imshow(mel.T, origin="lower", aspect="auto", extent=[0, mel.shape[0] * hop / sr, 0, mel.shape[1]])
    ax.set_xlabel("time (s)"); ax.set_ylabel("mel bin"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["english", "telugu"], required=True)
    ap.add_argument("--dataset", required=True, help="config folder name: Svarah or Telugu")
    ap.add_argument("--restore_step", type=int, required=True)
    ap.add_argument("--speaker", required=True, help="speaker name from speakers.json (e.g. S000, TEF, TEM)")
    ap.add_argument("--sentences", default=None, help="JSON file: [{\"id\": \"T1\", \"text\": \"...\"}, ...]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch, yaml
    from scipy.io import wavfile
    sys.path.insert(0, os.getcwd())
    from utils.model import get_model, get_vocoder, vocoder_infer
    from text import text_to_sequence
    from text.symbols import symbols
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = lambda n: yaml.load(open(f"config/{a.dataset}/{n}.yaml"), Loader=yaml.FullLoader)
    pre, mod, trn = cfg("preprocess"), cfg("model"), cfg("train")
    model = get_model(a, (pre, mod, trn), device, train=False)
    vocoder = get_vocoder(mod, device)
    P = pre["path"]["preprocessed_path"]
    speakers = json.load(open(f"{P}/speakers.json")); stats = json.load(open(f"{P}/stats.json"))
    sid = speakers[a.speaker]
    sr = pre["preprocessing"]["audio"]["sampling_rate"]; hop = pre["preprocessing"]["stft"]["hop_length"]
    cleaners = pre["preprocessing"]["text"]["text_cleaners"]

    if a.sentences:
        sents = [(d["id"], d["text"]) for d in json.load(open(a.sentences, encoding="utf-8"))]
    elif a.lang == "english":
        sents = DEFAULT_EN
    else:
        raise SystemExit("Telugu needs --sentences test_sentences_telugu.json")
    front = en_frontend if a.lang == "english" else te_frontend

    for sub in ["test_generated_audios", "mel_plots", "variance_data",
                "variance_adaptor_plots/duration", "variance_adaptor_plots/pitch", "variance_adaptor_plots/energy"]:
        os.makedirs(f"{a.out}/{sub}", exist_ok=True)
    seq_log = open(f"{a.out}/phoneme_sequences.txt", "w", encoding="utf-8")

    for sid_name, text in sents:
        norm, phones = front(text)
        bad = [p for p in phones if "@" + p not in symbols]
        if bad: raise SystemExit(f"{sid_name}: phones unknown to the model: {bad}")
        seq_log.write(f"{sid_name}\nTEXT: {text}\nNORMALISED: {norm}\nPHONES: {' '.join(phones)}\n\n")
        seq = text_to_sequence("{" + " ".join(phones) + "}", cleaners)
        texts = torch.LongTensor(seq)[None].to(device)
        src_lens = torch.LongTensor([len(seq)]).to(device)
        spk = torch.LongTensor([sid]).to(device)
        with torch.no_grad():
            out = model(spk, texts, src_lens, len(seq))
        mel, post = out[0][0].cpu().numpy(), out[1][0].cpu().numpy()
        pit, ene = out[2][0].cpu().numpy(), out[3][0].cpu().numpy()
        dur = out[5][0].cpu().numpy(); mel_len = int(out[9][0])
        if pre["preprocessing"]["pitch"]["normalization"]:  pit = pit * stats["pitch"][3] + stats["pitch"][2]
        if pre["preprocessing"]["energy"]["normalization"]: ene = ene * stats["energy"][3] + stats["energy"][2]
        wav = vocoder_infer(out[1].transpose(1, 2), vocoder, mod, pre, lengths=[mel_len * hop])[0]
        wavfile.write(f"{a.out}/test_generated_audios/{sid_name}.wav", sr, wav)

        ttl = f"{a.lang} {sid_name}, speaker {a.speaker}, step {a.restore_step}"
        plot_duration(phones, dur, hop, sr, ttl + ": predicted duration", f"{a.out}/variance_adaptor_plots/duration/{sid_name}.png")
        plot_curve(phones, dur, pit, "predicted pitch (Hz)", ttl + ": predicted pitch (F0)", f"{a.out}/variance_adaptor_plots/pitch/{sid_name}.png", hop, sr)
        plot_curve(phones, dur, ene, "predicted energy", ttl + ": predicted energy", f"{a.out}/variance_adaptor_plots/energy/{sid_name}.png", hop, sr)
        plot_mel(post[:mel_len], ttl + ": mel-spectrogram", f"{a.out}/mel_plots/{sid_name}.png", hop, sr)
        with open(f"{a.out}/variance_data/{sid_name}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["phone", "duration_frames", "duration_ms", "pitch_hz_or_frame0", "energy"])
            level_p = len(pit) == len(phones); level_e = len(ene) == len(phones)
            for i, p in enumerate(phones):
                w.writerow([p, int(dur[i]), round(float(dur[i]) * hop / sr * 1000, 1),
                            round(float(pit[i]), 2) if level_p else "", round(float(ene[i]), 3) if level_e else ""])
        print(f"{sid_name}: {len(phones)} phones, {mel_len*hop/sr:.2f} s of audio")
    seq_log.close()

if __name__ == "__main__":
    main()
