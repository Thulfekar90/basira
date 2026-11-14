"""Audio robustness benchmark: simulated room recordings of text-to-speech, forged with
known ground truth, then re-encoded the way messaging apps and phone lines do it.

    python benchmarks/audio_robustness.py                 # everything (about 6 min)
    python benchmarks/audio_robustness.py --n 6 --only matrix
    python benchmarks/audio_robustness.py --set heldout   # other voices / languages / rooms

WHAT THE DATA IS -- read this before quoting a number
-----------------------------------------------------
There is no real microphone recording anywhere in this benchmark.

* Speech is produced offline by the macOS ``say`` command (needs macOS; nothing is
  downloaded). It has real phonetics, pauses and prosody, which the older benchmark's
  "speech-like" buzz (``benchmarks/synthetic_audio.py``) does not. But it is synthetic
  speech: band-limited at 6-11 kHz for most voices and rendered on digital silence.
* A "recording" is that speech convolved with a *synthetic* room impulse response, plus
  coloured Gaussian room noise whose level wanders, a few low-level background events,
  -- for every second recording only -- weak mains hum with harmonics, and -- for every
  third -- a simple automatic gain control that pumps the noise floor during pauses
  (``tests/synth_audio.py: simulate_recording``). Real rooms, real microphones, noise
  suppression, handling noise, overlapping speakers and music are all absent. Real
  recordings will be harder, and their false-positive rate is unknown.
* Forgeries are made by this script (sample-accurate joins, no cross-fade):
    cut       0.4-1.2 s removed at a random position (three different cuts per recording)
    insert    3 s of another recording (other voice, other room) inserted
    zeros     0.3 s of digital silence inserted
    voices    the recording switches to another recording (other voice and room) for good
* Conditions are applied to the finished clean / forged file: MP3, AAC, Opus and Vorbis
  through the bundled ffmpeg, resampling to 16 kHz, an 8 kHz telephone band
  (300-3400 Hz), and added white noise (20 dB below the speech level).

Scoring: "flagged" = manipulation score >= 0.45, the level at which the verdict becomes
"Suspicious". (The AI-generation score is reported separately: the speech here *is*
synthetic, so an AI-generation flag on a "clean" file is not a false alarm.) A forgery is
"located" when the file is flagged AND a reported segment of at most 6 s (plus the edit
length) overlaps the true edit within 0.5 s. "wrong place" counts flagged forgeries whose
segments are all somewhere else: a right verdict for the wrong reason.

Three sections:
  matrix    the robustness table described above
  tts       does the engine say anything about raw ``say`` output (per voice), after a
            simulated room, after MP3? (``audio.synthetic`` / ``audio.bandwidth``)
  nonspeech false positives on audio that ships with macOS (alert sounds, ring tones,
            ambient "background sounds", slideshow music). Local measurement only.

The ``dev`` set (default) is what the thresholds in ``basira/audio`` were tuned on;
``--set heldout`` uses voices, languages, texts and room seeds that were not looked at
during tuning.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import synth_audio as A  # noqa: E402

from basira.audio import analyze_audio  # noqa: E402
from basira.core.io import ffmpeg_exe  # noqa: E402

SR = 44100
DURATION = 30.0
TOL = 0.5
FLAG = 0.45            # category score at which the verdict becomes "Suspicious"
MAX_SEGMENT_S = 6.0

# --------------------------------------------------------------------------- #
# Speech material: text written for this benchmark, spoken by `say`
# --------------------------------------------------------------------------- #
TEXTS_EN = [
    "The meeting was moved to Thursday afternoon, because the delivery had not arrived. "
    "Nobody told the driver, so he waited outside the gate for almost an hour. When we "
    "finally called him, he laughed, and said the paperwork was still on his kitchen table.",
    "I never said that she took the money. What I said was that the envelope was on the "
    "desk in the morning, and that it was gone by lunch. There were four people in the "
    "office that day. You can ask any of them.",
    "We left the harbour at six. The sea was calm at first, but the wind turned before "
    "noon, and by two o'clock the waves were coming over the bow. The captain decided to "
    "turn back. I think that was the right decision, even though we lost a whole day.",
    "First, check that the valve is closed. Then open the cover, and remove the old "
    "filter. It may be stuck; do not force it. Clean the housing with a dry cloth, put "
    "the new filter in, and close the cover until you hear it click.",
    "My grandmother kept bees for forty years. She used to say that a hive tells you "
    "everything, if you are patient enough to listen. In spring she would sit beside "
    "them for an hour every morning, with a cup of tea, and never once wore gloves.",
    "The contract was signed on the ninth of March. Payment was due thirty days later. "
    "As of today, no payment has been received, and three reminders have gone "
    "unanswered. We therefore have no choice but to hand the matter over to our lawyers.",
    "Good evening. Rain is expected across the north tonight, heavy at times, with a risk "
    "of flooding near the river. Temperatures will fall to around four degrees. Tomorrow "
    "starts cloudy, but it should brighten up in the afternoon, with light winds.",
    "Honestly, I do not remember who called first. It was late, maybe eleven. He sounded "
    "tired. He asked whether I had seen the report, and I said no. That was the whole "
    "conversation. It lasted less than a minute, and we did not speak again that week.",
    "The village has one shop, one school and a bus that comes twice a day. Most of the "
    "young people have left for the city. Those who stayed grow olives, keep a few goats, "
    "and wait for the summer, when the families come back and the square is full again.",
    "Turn left at the second traffic light, then follow the road for about two "
    "kilometres. You will pass a petrol station and a small church. The house is the "
    "white one on the corner, with the blue door. If you reach the bridge, you have gone "
    "too far.",
    "The experiment failed twice before we understood the problem. The sensor was fine. "
    "The software was fine. What nobody had checked was the cable, which had been bent so "
    "often that it only worked when the table was not moving. We replaced it for two euros.",
    "She opened the letter slowly, read it once, and put it back in the envelope. Then "
    "she made coffee, sat down by the window, and watched the street for a long time. "
    "When the children came home, she told them that everything was going to be fine.",
]
TEXTS_OTHER = {
    "de": ["Die Sitzung wurde auf Donnerstag verschoben, weil die Lieferung nicht angekommen "
           "war. Niemand hatte dem Fahrer Bescheid gesagt, also wartete er fast eine Stunde "
           "vor dem Tor. Als wir ihn endlich anriefen, lachte er nur. Die Papiere lagen noch "
           "bei ihm zu Hause auf dem Tisch.",
           "Zuerst prüfen Sie, ob das Ventil geschlossen ist. Dann öffnen Sie den Deckel und "
           "nehmen den alten Filter heraus. Reinigen Sie das Gehäuse mit einem trockenen "
           "Tuch, setzen Sie den neuen Filter ein und schließen Sie den Deckel, bis er hörbar "
           "einrastet."],
    "fr": ["Nous avons quitté le port à six heures. La mer était calme au début, mais le vent "
           "a tourné avant midi, et vers deux heures les vagues passaient par-dessus le pont. "
           "Le capitaine a décidé de faire demi-tour. Je crois que c'était la bonne décision.",
           "Le contrat a été signé le neuf mars. Le paiement était dû trente jours plus tard. "
           "À ce jour, aucun paiement n'a été reçu, et trois rappels sont restés sans réponse. "
           "Nous n'avons donc pas d'autre choix que de transmettre le dossier à nos avocats."],
    "es": ["Mi abuela tuvo abejas durante cuarenta años. Decía que una colmena te lo cuenta "
           "todo, si tienes la paciencia de escuchar. En primavera se sentaba junto a ellas "
           "una hora cada mañana, con una taza de té, y nunca usó guantes.",
           "Gire a la izquierda en el segundo semáforo y siga la carretera unos dos "
           "kilómetros. Pasará una gasolinera y una iglesia pequeña. La casa es la blanca de "
           "la esquina, la de la puerta azul. Si llega al puente, se ha pasado."],
    "it": ["Il paese ha un negozio, una scuola e un autobus che passa due volte al giorno. "
           "Quasi tutti i giovani sono partiti per la città. Chi è rimasto coltiva olivi, "
           "tiene qualche capra e aspetta l'estate, quando le famiglie tornano e la piazza si "
           "riempie di nuovo.",
           "Sinceramente non ricordo chi ha chiamato per primo. Era tardi, forse le undici. "
           "Sembrava stanco. Mi ha chiesto se avevo visto il rapporto, e ho detto di no. "
           "Tutta la conversazione è durata meno di un minuto."],
    "pt": ["A experiência falhou duas vezes antes de percebermos o problema. O sensor estava "
           "bom. O programa estava bom. O que ninguém tinha verificado era o cabo, dobrado "
           "tantas vezes que só funcionava quando a mesa não se mexia.",
           "Boa noite. Espera-se chuva em todo o norte esta noite, por vezes forte, com risco "
           "de cheias junto ao rio. As temperaturas descem até aos quatro graus. Amanhã "
           "começa nublado, mas deve melhorar durante a tarde."],
    "nl": ["Ze maakte de brief langzaam open, las hem één keer en stopte hem terug in de "
           "envelop. Daarna zette ze koffie, ging bij het raam zitten en keek lang naar de "
           "straat. Toen de kinderen thuiskwamen, zei ze dat alles goed zou komen.",
           "Ik heb nooit gezegd dat zij het geld heeft gepakt. Ik heb gezegd dat de envelop "
           "'s ochtends op het bureau lag en dat hij voor de lunch weg was. Er waren die dag "
           "vier mensen op kantoor. U kunt het ze allemaal vragen."],
}

# Voices that ship with macOS (availability differs per machine; missing ones are skipped).
VOICES_DEV = ["Samantha", "Daniel", "Karen", "Moira", "Tessa", "Rishi", "Fred", "Kathy",
              "Ralph", "Eddy (English (US))", "Flo (English (US))", "Reed (English (US))",
              "Sandy (English (UK))", "Shelley (English (US))", "Rocko (English (UK))",
              "Grandma (English (US))", "Grandpa (English (UK))", "Aman", "Tara", "Albert",
              "Junior"]
VOICES_HELDOUT = [("Anna", "de"), ("Thomas", "fr"), ("Mónica", "es"), ("Alice", "it"),
                  ("Luciana", "pt"), ("Xander", "nl"), ("Eddy (German (Germany))", "de"),
                  ("Amélie", "fr"), ("Paulina", "es"), ("Flo (Italian (Italy))", "it"),
                  ("Joana", "pt"), ("Ellen", "nl"), ("Sandy (German (Germany))", "de"),
                  ("Jacques", "fr"), ("Reed (Spanish (Spain))", "es"),
                  ("Shelley (Italian (Italy))", "it"), ("Rocko (Portuguese (Brazil))", "pt"),
                  ("Grandma (French (France))", "fr"), ("Grandpa (German (Germany))", "de"),
                  ("Flo (Spanish (Mexico))", "es")]


def available_voices() -> List[str]:
    if not shutil.which("say"):
        return []
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    names = []
    for line in out.splitlines():
        head = line.split("#")[0].rstrip()
        parts = head.rsplit(None, 1)           # "<name>   <locale>"
        if len(parts) == 2:
            names.append(parts[0].strip())
    # `say -v Samantha` also selects "Samantha (English (US))": accept the short name too
    return sorted(set(names) | {n.split(" (")[0] for n in names})


def tts(voice: str, text: str, cache: str) -> np.ndarray:
    """Offline text-to-speech with `say` -> float64 mono at SR (cached on disk)."""
    os.makedirs(cache, exist_ok=True)
    key = hashlib.sha1(f"{voice}|{SR}|{text}".encode("utf-8")).hexdigest()[:16]
    path = os.path.join(cache, f"tts_{key}.wav")
    if not os.path.exists(path):
        tmp = path + ".part.wav"
        subprocess.run(["say", "-v", voice, "-o", tmp, f"--data-format=LEI16@{SR}", text],
                       check=True, capture_output=True, timeout=300)
        os.replace(tmp, path)
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    assert sr == SR, sr
    return x[:, 0]


def speakers(which: str, n: int) -> List[Tuple[str, str]]:
    """(voice, text) for n recordings; text = two paragraphs so it lasts >= DURATION."""
    have = set(available_voices())
    out: List[Tuple[str, str]] = []
    if which == "dev":
        voices = [v for v in VOICES_DEV if v in have]
        for i, v in enumerate(voices):
            out.append((v, TEXTS_EN[i % len(TEXTS_EN)] + " " + TEXTS_EN[(i + 5) % len(TEXTS_EN)]))
    else:
        voices2 = [(v, lang) for v, lang in VOICES_HELDOUT if v in have]
        for i, (v, lang) in enumerate(voices2):
            a, b = TEXTS_OTHER[lang]
            out.append((v, (a + " " + b) if i % 2 == 0 else (b + " " + a)))
    if not out:
        raise SystemExit("No usable `say` voices found: this benchmark needs macOS with the "
                         "standard system voices installed.")
    return [out[i % len(out)] for i in range(n)]


# --------------------------------------------------------------------------- #
# Recordings and forgeries
# --------------------------------------------------------------------------- #
def room_params(seed: int) -> Dict[str, object]:
    """Random room / device for one recording. Every second recording has no mains hum
    (a phone on battery, outdoors): the ENF test cannot apply there."""
    rng = np.random.default_rng(1000 + seed)
    return {
        "speech_db": float(rng.uniform(-26.0, -20.0)),
        "rt60": float(rng.uniform(0.2, 0.6)),
        "noise_db": float(rng.uniform(-62.0, -46.0)),
        "noise_tilt": float(rng.uniform(0.2, 0.9)),
        "hum_db": float(rng.uniform(-62.0, -52.0)) if seed % 2 == 0 else None,
        "hum_freq": 50.0 if (seed // 2) % 2 == 0 else 60.0,
        "events": int(rng.integers(0, 4)),
        "agc": seed % 3 == 2,               # every third device has automatic gain control
    }


def make_recording(voice: str, text: str, seed: int, cache: str) -> np.ndarray:
    dry = tts(voice, text, cache)
    n = int(DURATION * SR)
    if len(dry) < n:                      # short voices: repeat the text
        dry = np.tile(np.concatenate([dry, np.zeros(int(0.6 * SR))]), int(np.ceil(n / len(dry))) + 1)
    lead = np.zeros(int(0.4 * SR))
    dry = np.concatenate([lead, dry])[:n]
    return A.simulate_recording(dry, SR, seed, **room_params(seed))


def forge_all(x: np.ndarray, other: np.ndarray, seed: int) -> Dict[str, Tuple[np.ndarray, Tuple[float, float]]]:
    """All forgeries of one recording -> {name: (samples, (edit start, edit end))}."""
    rng = np.random.default_rng(5000 + seed)
    out = {}
    for name in ("cut", "cut_b", "cut_c"):        # three independent cuts: the outcome of
        at = float(rng.uniform(6, 22))            # one depends on the phase it happens to hit
        y, t = A.forge_cut(x, SR, at=at, remove=float(rng.uniform(0.4, 1.2)))
        out[name] = (y, (t, t))
    at = float(rng.uniform(6, 20))
    o0 = int(rng.uniform(4, 20) * SR)
    y, span = A.forge_insert(x, SR, at=at, seg=other[o0:o0 + 3 * SR])
    out["insert"] = (y, span)
    at = float(rng.uniform(6, 22))
    y, t = A.forge_zeros(x, SR, at=at, dur=0.3)
    out["zeros"] = (y, (t, t + 0.3))
    at = float(rng.uniform(9, 20))
    i0 = int(at * SR)
    out["voices"] = (np.concatenate([x[:i0], other[i0:]]), (at, at))
    return out


# --------------------------------------------------------------------------- #
# Conditions (what happens to the file after it was made)
# --------------------------------------------------------------------------- #
def _ffmpeg(src: str, dst: str, args: Sequence[str]) -> None:
    exe = ffmpeg_exe()
    r = subprocess.run([exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", src,
                        "-ac", "1", "-map_metadata", "-1", *args, dst],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])


def _encoders() -> str:
    exe = ffmpeg_exe()
    if not exe:
        return ""
    return subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True,
                          timeout=60).stdout


def conditions() -> List[Tuple[str, Callable[[np.ndarray, str], str]]]:
    """[(label, fn(samples, basepath) -> path of the file to analyse)]."""
    from scipy.signal import butter, resample_poly, sosfiltfilt

    def wav(x, base, sr=SR):
        p = base + ".wav"
        sf.write(p, A.quantize16(x), sr, subtype="PCM_16")
        return p

    def codec(ext, args):
        def run(x, base):
            src = wav(x, base + "_src")
            dst = base + ext
            _ffmpeg(src, dst, args)
            os.remove(src)
            return dst
        return run

    def rs16(x, base):
        return wav(resample_poly(x, 160, 441), base, 16000)

    def tel8(x, base):
        sos = butter(4, [300.0, 3400.0], btype="band", fs=SR, output="sos")
        return wav(resample_poly(sosfiltfilt(sos, x), 80, 441), base, 8000)

    def noisy(x, base):
        rng = np.random.default_rng(len(x))
        act = np.abs(x) > 0.05 * np.max(np.abs(x))
        lvl = np.sqrt(np.mean(x[act] ** 2)) * 10 ** (-20 / 20)
        return wav(x + lvl * rng.standard_normal(len(x)), base)

    enc = _encoders()
    conds: List[Tuple[str, Callable[[np.ndarray, str], str]]] = [("WAV 44.1k", wav)]
    if "libmp3lame" in enc:
        conds += [("MP3 128k", codec(".mp3", ["-c:a", "libmp3lame", "-b:a", "128k"])),
                  ("MP3 64k", codec(".mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]))]
    if " aac " in enc:
        conds.append(("AAC 96k", codec(".m4a", ["-c:a", "aac", "-b:a", "96k"])))
    if "libopus" in enc:
        conds.append(("Opus 24k", codec(".opus", ["-c:a", "libopus", "-b:a", "24k",
                                                 "-application", "voip"])))
    if "libvorbis" in enc:
        conds.append(("Vorbis q3", codec(".ogg", ["-c:a", "libvorbis", "-q:a", "3"])))
    conds += [("16 kHz", rs16), ("8 kHz tel.", tel8), ("+noise 20dB", noisy)]
    return conds


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def located(report, span: Tuple[float, float]) -> bool:
    t0, t1 = span
    return any(s.end >= t0 - TOL and s.start <= t1 + TOL
               and (s.end - s.start) <= MAX_SEGMENT_S + (t1 - t0)
               for s in report.fused_segments)


def firing(report) -> List[str]:
    return [e.detector for e in report.evidences
            if e.error is None and e.score >= 0.5 and e.confidence > 0
            and e.category.value != "provenance"]


def run_matrix(args, work: str) -> Dict[str, object]:
    spk = speakers(args.set, args.n)
    base_seed = 0 if args.set == "dev" else 500
    t0 = time.time()
    recs = [make_recording(v, txt, base_seed + i, args.cache) for i, (v, txt) in enumerate(spk)]
    print(f"[matrix] {len(recs)} recordings of {DURATION:.0f} s built in {time.time() - t0:.0f} s "
          f"({args.set} set: {', '.join(sorted({v for v, _ in spk}))})", flush=True)
    conds = conditions()
    fams = ["cut (hum)", "cut (no hum)", "insert", "zeros", "voices"]
    table: Dict[str, Dict[str, List[int]]] = {}
    blame: Dict[str, Dict[str, int]] = {}
    misses: List[str] = []
    for label, fn in conds:
        row = {"clean flagged": [0, 0], "clean any segment": [0, 0], "stray": [0, 0],
               "clean AI-flagged": [0, 0]}
        row.update({f: [0, 0] for f in fams})
        for i, x in enumerate(recs):
            seed = base_seed + i
            # the "other" recording alternates between one with and one without hum
            off = (len(recs) // 2) | 1
            other = recs[(i + off + (1 if i % 4 >= 2 else 0)) % len(recs)]
            if other is x:
                other = recs[(i + 1) % len(recs)]
            cases = {"clean": (x, None)}
            cases.update(forge_all(x, other, seed))
            for name, (sig, span) in cases.items():
                p = fn(sig, os.path.join(work, f"m_{i}_{name}"))
                r = analyze_audio(p)
                os.remove(p)
                flagged = r.scores["manipulation"] >= FLAG
                if name == "clean":
                    row["clean AI-flagged"][0] += int(r.scores["ai_generation"] >= FLAG)
                    row["clean AI-flagged"][1] += 1
                    row["clean flagged"][0] += int(flagged)
                    row["clean flagged"][1] += 1
                    row["clean any segment"][0] += int(bool(r.fused_segments))
                    row["clean any segment"][1] += 1
                    if flagged:
                        for d in firing(r):
                            blame.setdefault(label, {}).setdefault(d, 0)
                            blame[label][d] += 1
                        misses.append(f"FP {label} rec{i} ({spk[i][0]}): {firing(r)}")
                    continue
                fam = name
                if name.startswith("cut"):
                    fam = "cut (hum)" if room_params(seed)["hum_db"] is not None else "cut (no hum)"
                hit = flagged and located(r, span)
                row[fam][0] += int(hit)
                row[fam][1] += 1
                row["stray"][0] += int(flagged and not hit)
                row["stray"][1] += 1
        table[label] = row
        print(f"  {label:12s} " + "  ".join(f"{k}={v[0]}/{v[1]}" for k, v in row.items()),
              flush=True)
    cols = ["clean flagged", "clean any segment"] + fams + ["stray", "clean AI-flagged"]
    print()
    print("| condition | clean flagged | clean with any segment | "
          + " | ".join(f"{f} located" for f in fams)
          + " | forged flagged, wrong place | clean: AI-generation flagged |")
    print("|---|" + "---|" * len(cols))
    for label, row in table.items():
        print(f"| {label} | " + " | ".join(f"{row[c][0]} / {row[c][1]}" for c in cols) + " |")
    if blame:
        print("\nDetectors over threshold on flagged clean files:")
        for label, d in blame.items():
            print(f"  {label}: " + ", ".join(f"{k} x{v}" for k, v in sorted(d.items())))
    if args.verbose:
        print("\n".join(misses))
    return {"table": table, "blame": blame}


def run_tts(args, work: str) -> Dict[str, object]:
    """What the AI-generation cues say about real TTS output, per voice."""
    have = set(available_voices())
    if args.set == "dev":
        voices = [(v, TEXTS_EN[i % len(TEXTS_EN)]) for i, v in enumerate(VOICES_DEV) if v in have]
    else:
        voices = [(v, TEXTS_OTHER[lang][i % 2]) for i, (v, lang) in enumerate(VOICES_HELDOUT)
                  if v in have]
    conds = dict(conditions())
    variants = [("raw", "WAV 44.1k", False), ("MP3 64k", "MP3 64k", False),
                ("16 kHz", "16 kHz", False), ("room", "WAV 44.1k", True),
                ("room+MP3 64k", "MP3 64k", True)]
    variants = [v for v in variants if v[1] in conds]
    tally = {v[0]: [0, 0] for v in variants}
    manip = {v[0]: [0, 0] for v in variants}
    rows = []
    for k, (voice, text) in enumerate(voices):
        dry = tts(voice, text + " " + text, args.cache)
        dry = dry[: int(DURATION * SR)]
        cells = []
        for name, cond, room in variants:
            x = A.simulate_recording(dry, SR, 900 + k, **room_params(900 + 2 * k + 1)) if room else dry
            p = conds[cond](x, os.path.join(work, f"t_{k}"))
            r = analyze_audio(p)
            os.remove(p)
            bw, syn = r.get("audio.bandwidth"), r.get("audio.synthetic")
            ai = r.scores["ai_generation"]
            tally[name][0] += int(ai >= FLAG)
            tally[name][1] += 1
            manip[name][0] += int(r.scores["manipulation"] >= FLAG)
            manip[name][1] += 1
            cut = bw.details.get("cutoff_hz") if bw is not None else None
            cells.append(f"{ai:.2f} (bw {bw.score:.2f}"
                         + (f" @{cut / 1000:.1f}k" if cut else "")
                         + f", syn {syn.score:.2f})"
                         + ("!" if r.scores["manipulation"] >= 0.45 else ""))
        rows.append((voice, cells))
    print("\nTTS section: AI-generation score per voice (bw = audio.bandwidth score and "
          "cutoff, syn = audio.synthetic score; '!' = also flagged as manipulated)")
    print("| voice | " + " | ".join(v[0] for v in variants) + " |")
    print("|---|" + "---|" * len(variants))
    for voice, cells in rows:
        print(f"| {voice} | " + " | ".join(cells) + " |")
    print("| **AI-generation >= 0.45 (Suspicious)** | "
          + " | ".join(f"**{tally[v[0]][0]} / {tally[v[0]][1]}**" for v in variants) + " |")
    print("| flagged as *manipulated* (wrong) | "
          + " | ".join(f"{manip[v[0]][0]} / {manip[v[0]][1]}" for v in variants) + " |")
    return {"ai_flagged": tally, "manipulation_flagged": manip}


NONSPEECH_SOURCES = [
    ("alert sounds", "/System/Library/Sounds/*.aiff"),
    ("ambient background sounds",
     "/System/Library/PrivateFrameworks/HearingUtilities.framework/Versions/A/Resources/*.m4a"),
    ("slideshow music",
     "/System/Library/PrivateFrameworks/Slideshows.framework/Versions/A/Resources/Content/Audio/*.m4a"),
    ("ring tones",
     "/System/Library/PrivateFrameworks/ToneLibrary.framework/Versions/A/Resources/Ringtones/*.m4r"),
]


def run_nonspeech(args, work: str) -> Dict[str, object]:
    """False positives on non-speech audio that ships with macOS (read in place)."""
    out = {}
    print("\nNon-speech system audio (nothing here was edited by us; every flag is a false "
          "positive or a statement about how the sound was produced)")
    print("| material | files | manipulation >= 0.45 | AI-generation >= 0.45 | detectors over threshold |")
    print("|---|---|---|---|---|")
    for name, pattern in NONSPEECH_SOURCES:
        files = sorted(glob.glob(pattern))[: args.max_files]
        if not files:
            continue
        n = man = ai = 0
        dets: Dict[str, int] = {}
        for f in files:
            try:
                r = analyze_audio(f, max_seconds=120.0)
            except Exception as err:
                if args.verbose:
                    print("   cannot read", os.path.basename(f), err)
                continue
            n += 1
            m, a = r.scores["manipulation"] >= FLAG, r.scores["ai_generation"] >= FLAG
            man += int(m)
            ai += int(a)
            if m or a:
                for d in firing(r):
                    dets[d] = dets.get(d, 0) + 1
                if args.verbose:
                    print("   ", os.path.basename(f), firing(r))
        out[name] = {"files": n, "manipulation": man, "ai": ai, "detectors": dets}
        print(f"| {name} | {n} | {man} | {ai} | "
              + (", ".join(f"{k} x{v}" for k, v in sorted(dets.items())) or "-") + " |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=12, help="recordings in the matrix")
    ap.add_argument("--set", choices=["dev", "heldout"], default="dev")
    ap.add_argument("--only", choices=["matrix", "tts", "nonspeech"], action="append")
    ap.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "basira_audio_tts"),
                    help="where `say` output is cached")
    ap.add_argument("--max-files", type=int, default=60)
    ap.add_argument("--json", help="also write the raw counts to this file")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if not ffmpeg_exe():
        print("note: no ffmpeg found, codec conditions are skipped")
    sections = args.only or ["matrix", "tts", "nonspeech"]
    work = tempfile.mkdtemp(prefix="basira_audio_robust_")
    res: Dict[str, object] = {"set": args.set, "n": args.n}
    try:
        t0 = time.time()
        if "matrix" in sections:
            res["matrix"] = run_matrix(args, work)
        if "tts" in sections:
            res["tts"] = run_tts(args, work)
        if "nonspeech" in sections:
            res["nonspeech"] = run_nonspeech(args, work)
        print(f"\ntotal {time.time() - t0:.0f} s")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
