import re
from dataclasses import dataclass
from typing import List

@dataclass
class Subtitle:
    start: float
    end: float
    text: str

_TIME = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")

def _to_seconds(h, m, s, ms):
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0

def parse_srt(srt_text:str) -> List[Subtitle]:
    lines = srt_text.splitlines()
    subs = []
    i = 0
    while i < len(lines):
        if lines[i].strip().isdigit():
            i += 1
        if i >= len(lines): break
        m = _TIME.findall(lines[i])
        if len(m) >= 2:
            (h1,m1,s1,ms1),(h2,m2,s2,ms2) = m[0], m[1]
            start = _to_seconds(h1,m1,s1,ms1)
            end   = _to_seconds(h2,m2,s2,ms2)
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip()); i += 1
            subs.append(Subtitle(start, end, " ".join(text_lines)))
        while i < len(lines) and not lines[i].strip():
            i += 1
    return subs

def align_frames_to_subs(n_frames:int, fps:float, subs:List[Subtitle]):
    frame2sent = [-1]*n_frames
    for si, sub in enumerate(subs):
        s = int(sub.start * fps); e = int(sub.end * fps)
        s = max(0, min(n_frames-1, s)); e = max(0, min(n_frames, e))
        for f in range(s, e):
            frame2sent[f] = si
    return frame2sent
