import streamlit as st
import asyncio
import aiohttp
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from dotenv import load_dotenv
import time
import requests as sync_requests
from sklearn.cluster import KMeans

load_dotenv()

# ─── Provider Abstraction ────────────────────────────────────────────────────────

SQUAD_SIZE = 8  # target number of active models per provider
N_TIERS = 3     # price/intelligence tiers: cheap, mid, premium

@dataclass
class ModelInfo:
    id: str
    family: str
    prompt_price: float
    completion_price: float
    context_length: int
    tier: int = 0  # assigned by k-means

@dataclass
class Provider:
    name: str
    api_url: str
    api_key: str
    exclude_patterns: List[str] = field(default_factory=list)
    tier_strategy: str = "s-tier"  # 's-tier' = pick premium, 'a-tier' = pick budget/mid
    _catalog: Dict[str, ModelInfo] = field(default_factory=dict, repr=False)
    _all_models: List[str] = field(default_factory=list, repr=False)
    _active: List[str] = field(default_factory=list, repr=False)
    _bench: List[str] = field(default_factory=list, repr=False)
    _failed: set = field(default_factory=set, repr=False)
    _tier_labels: Dict[int, str] = field(default_factory=dict, repr=False)
    
    def discover(self) -> int:
        """Fetch all chat models, extract pricing, cluster by price/intelligence."""
        if not self.api_key:
            return 0
        try:
            url = self.api_url.replace("/chat/completions", "/models")
            r = sync_requests.get(
                url,
                headers={"Authorization": f"Bearer {self.api_key}", "Accept-Encoding": "identity"},
                timeout=10
            )
            if r.status_code != 200:
                return 0
            data = r.json()
            raw_models = data.get("data", [])
            
            for m in raw_models:
                mid = m["id"]
                if any(pat in mid.lower() for pat in self.exclude_patterns):
                    continue
                
                # Extract pricing (OpenRouter has it, Mammouth doesn't)
                pricing = m.get("pricing", {})
                prompt_p = float(pricing.get("prompt", 0) or 0)
                comp_p = float(pricing.get("completion", 0) or 0)
                ctx = m.get("context_length", 0) or 0
                if isinstance(ctx, dict):
                    ctx = 0
                
                # Infer family from model id
                base = mid.split("/")[-1] if "/" in mid else mid
                family = base.split("-")[0].lower()
                
                self._catalog[mid] = ModelInfo(
                    id=mid, family=family,
                    prompt_price=prompt_p, completion_price=comp_p,
                    context_length=int(ctx)
                )
            
            self._all_models = sorted(self._catalog.keys())
            self._cluster()
            self._pick_squad()
            return len(self._all_models)
        except Exception:
            pass
        return 0
    
    def _cluster(self):
        """K-means on (log_prompt_price, log_completion_price, log_context) → tiers."""
        models_with_price = [
            m for m in self._catalog.values()
            if m.prompt_price > 0 or m.completion_price > 0
        ]
        if len(models_with_price) < N_TIERS:
            # Not enough pricing data — all tier 0
            for m in self._catalog.values():
                m.tier = 0
            self._tier_labels = {0: "unknown"}
            return
        
        # Build feature matrix: log-scaled prices + context
        import math
        X = []
        ids = []
        for m in models_with_price:
            lp = math.log10(m.prompt_price + 1e-10)
            lc = math.log10(m.completion_price + 1e-10)
            lx = math.log10(m.context_length + 1) / 7  # normalize ~1M ctx
            X.append([lp, lc, lx])
            ids.append(m.id)
        
        X_arr = np.array(X)
        n_clusters = min(N_TIERS, len(X_arr))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(X_arr)
        
        # Sort clusters by centroid prompt price → tier 0=cheapest, 2=premium
        centroids = km.cluster_centers_
        sorted_clusters = sorted(range(n_clusters), key=lambda i: centroids[i][0])
        remap = {old: new for new, old in enumerate(sorted_clusters)}
        
        tier_names = ["budget", "mid", "premium"][:n_clusters]
        self._tier_labels = {i: tier_names[i] for i in range(n_clusters)}
        
        for mid, label in zip(ids, labels):
            self._catalog[mid].tier = remap[label]
        
        # Models without pricing get tier 1 (mid) by default
        for m in self._catalog.values():
            if m.id not in ids:
                m.tier = min(1, n_clusters - 1)
    
    # Preferred families — major frontier labs first
    PREFERRED_FAMILIES = [
        "gpt", "claude", "gemini", "mistral", "grok", "deepseek",
        "llama", "qwen", "kimi", "glm", "minimax", "nova",
    ]
    
    def _pick_squad(self):
        """Select a diverse squad: one per major family, spread across tiers."""
        available = [m for m in self._all_models if m not in self._failed]
        
        # Group by family
        families: Dict[str, List[str]] = {}
        for mid in available:
            info = self._catalog.get(mid)
            if not info:
                continue
            families.setdefault(info.family, []).append(mid)
        
        # Sort within each family based on tier strategy
        for fam in families:
            if self.tier_strategy == "s-tier":
                # Premium: biggest context (newest), highest tier
                families[fam].sort(
                    key=lambda m: (
                        -self._catalog[m].context_length,
                        -self._catalog[m].tier,
                        m
                    )
                )
            else:
                # A-tier: prefer mid (best value), then budget, avoid premium
                families[fam].sort(
                    key=lambda m: (
                        abs(self._catalog[m].tier - 1),   # mid=0, budget=1, premium=1
                        -self._catalog[m].context_length, # newest within tier
                        m
                    )
                )
        
        # Pick one from each preferred family first, then fill from remaining
        squad = []
        used_families = set()
        
        for fam in self.PREFERRED_FAMILIES:
            if fam in families and families[fam] and len(squad) < SQUAD_SIZE:
                squad.append(families[fam].pop(0))
                used_families.add(fam)
        
        # Fill remaining slots from other families (round-robin)
        remaining = {f: ms for f, ms in families.items() if f not in used_families and ms}
        rem_lists = list(remaining.values())
        idx = 0
        while len(squad) < SQUAD_SIZE and rem_lists:
            bucket = rem_lists[idx % len(rem_lists)]
            if bucket:
                squad.append(bucket.pop(0))
            if not bucket:
                rem_lists.pop(idx % len(rem_lists))
                if not rem_lists:
                    break
            else:
                idx += 1
        
        # If still short, pull more from preferred families
        if len(squad) < SQUAD_SIZE:
            for fam in self.PREFERRED_FAMILIES:
                if fam in families:
                    while families[fam] and len(squad) < SQUAD_SIZE:
                        squad.append(families[fam].pop(0))
        
        self._active = squad[:SQUAD_SIZE]
        self._bench = [m for m in available if m not in self._active]
    
    def get_active_models(self) -> List[str]:
        if not self._active and self._all_models:
            self._pick_squad()
        return self._active
    
    def swap_failed(self, failed_models: List[str]) -> List[str]:
        """Replace failed models with bench models. Returns list of swapped-in."""
        swapped = []
        for fm in failed_models:
            self._failed.add(fm)
            if fm in self._active:
                self._active.remove(fm)
            if self._bench:
                replacement = self._bench.pop(0)
                self._active.append(replacement)
                swapped.append(replacement)
        return swapped
    
    def model_tag(self, mid: str) -> str:
        """Short display tag: name + tier."""
        info = self._catalog.get(mid)
        if not info:
            return mid
        tier = self._tier_labels.get(info.tier, "?")
        short = mid.split("/")[-1] if "/" in mid else mid
        return f"{short} [{tier}]"
    
    @property
    def pool_size(self) -> int:
        return len(self._all_models)
    
    @property
    def tier_summary(self) -> str:
        """E.g. 'budget:3 mid:3 premium:2'"""
        counts: Dict[str, int] = {}
        for mid in self._active:
            info = self._catalog.get(mid)
            if info:
                label = self._tier_labels.get(info.tier, "?")
                counts[label] = counts.get(label, 0) + 1
        return " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))

# ─── Provider Instances ──────────────────────────────────────────────────────────

MAMMOUTH = Provider(
    name="Mammouth",
    api_url="https://api.mammouth.ai/v1/chat/completions",
    api_key=os.getenv("MAMMOUTH_API_KEY", ""),
    exclude_patterns=["embedding", "image", "codex", "codestral", "devstral", "sonar"],
    tier_strategy="s-tier",  # no pricing data — just pick the best
)

OPENROUTER = Provider(
    name="OpenRouter",
    api_url="https://openrouter.ai/api/v1/chat/completions",
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
    exclude_patterns=["embedding", "image", ":free", "lyria", "gemma", "whisper",
                       "tts", "moderation", "jamba", "~"],
    tier_strategy="a-tier",  # has pricing — optimize for value
)

PROVIDERS = [MAMMOUTH, OPENROUTER]

# Color palettes — warm tones for Mammouth, cool tones for OpenRouter
WARM_PALETTE = [
    (255, 60, 60), (255, 160, 0), (255, 220, 0), (200, 80, 200),
    (255, 100, 150), (220, 120, 50), (180, 200, 0), (200, 60, 100),
    (230, 90, 30), (170, 50, 180), (240, 180, 60), (190, 40, 40),
]

COOL_PALETTE = [
    (0, 160, 255), (0, 200, 120), (80, 80, 255), (0, 220, 220),
    (100, 200, 255), (0, 140, 80), (120, 120, 255), (0, 180, 180),
    (60, 120, 200), (0, 160, 160), (90, 180, 240), (40, 100, 180),
]

def get_provider_colors(provider: Provider, models: List[str]) -> Dict[str, Tuple]:
    palette = WARM_PALETTE if provider.name == "Mammouth" else COOL_PALETTE
    return {m: palette[i % len(palette)] for i, m in enumerate(models)}

def get_all_model_colors(providers_models: Dict[str, List[str]]) -> Dict[str, Tuple]:
    colors = {}
    for pname, models in providers_models.items():
        provider = next((p for p in PROVIDERS if p.name == pname), None)
        if provider:
            colors.update(get_provider_colors(provider, models))
    return colors

# The medium: all models speak braille, nothing else
SYSTEM_PROMPT = (
    "You communicate exclusively in 8-dot braille Unicode characters (U+2800 to U+28FF). "
    "Never use Latin text, numbers, punctuation, or any non-braille characters. "
    "Your entire response must consist only of braille characters. "
    "You are one of several parallel processes. Your braille output will be braided "
    "with outputs from other processes. Respond only in braille."
)

# 8-dot braille: dots numbered 1-8, mapped to bit positions in Unicode offset
# Dot 1=(0,0) Dot 2=(1,0) Dot 3=(2,0) Dot 4=(0,1) Dot 5=(1,1) Dot 6=(2,1) Dot 7=(3,0) Dot 8=(3,1)
BRAILLE_DOT_POSITIONS = [
    (0, 0),  # dot 1
    (1, 0),  # dot 2
    (2, 0),  # dot 3
    (0, 1),  # dot 4
    (1, 1),  # dot 5
    (2, 1),  # dot 6
    (3, 0),  # dot 7
    (3, 1),  # dot 8
]

MAX_ITERATIONS = 10
CONVERGENCE_THRESHOLD = 0.95  # 95% dot agreement = consensus
PLATEAU_WINDOW = 3            # If convergence doesn't improve for this many rounds, declare disagreement
PLATEAU_EPSILON = 0.02        # Minimum improvement to count as "still converging"

# ─── Braille ↔ ASCII Codec ─────────────────────────────────────────────────────

def ascii_to_braille(text: str) -> str:
    """Encode ASCII string as 8-dot braille. Each byte maps to U+2800+byte."""
    return "".join(chr(0x2800 + ord(ch)) for ch in text)

def braille_to_ascii(braille: str) -> str:
    """Decode 8-dot braille back to ASCII. Inverse of ascii_to_braille."""
    result = []
    for ch in braille:
        if 0x2800 <= ord(ch) <= 0x28FF:
            byte_val = ord(ch) - 0x2800
            if 0x20 <= byte_val <= 0x7E:  # printable ASCII only
                result.append(chr(byte_val))
            elif byte_val == 0x0A:  # newline
                result.append('\n')
            elif byte_val == 0x09:  # tab
                result.append('\t')
            else:
                result.append(f'\\x{byte_val:02x}')
    return "".join(result)

# Allowlist of safe commands for consensus-gated execution
SAFE_COMMAND_PREFIXES = [
    "echo ", "date", "pwd", "whoami", "uname", "ls", "cat ", "head ", "tail ",
    "wc ", "sort ", "grep ", "find ", "which ", "env", "printenv",
]

def is_safe_command(cmd: str) -> bool:
    """Check if a decoded command is in the safe allowlist."""
    cmd_stripped = cmd.strip()
    return any(cmd_stripped.startswith(prefix) for prefix in SAFE_COMMAND_PREFIXES)

def execute_consensus_command(cmd: str, timeout: int = 5) -> Dict:
    """Execute a consensus-gated command. Returns result dict."""
    import subprocess
    if not is_safe_command(cmd):
        return {
            "executed": False,
            "command": cmd,
            "reason": "Command not in safe allowlist",
            "stdout": "",
            "stderr": "",
            "returncode": -1,
        }
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {
            "executed": True,
            "command": cmd,
            "reason": "Consensus-gated execution",
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:500],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "executed": False,
            "command": cmd,
            "reason": "Timeout",
            "stdout": "",
            "stderr": "",
            "returncode": -1,
        }
    except Exception as e:
        return {
            "executed": False,
            "command": cmd,
            "reason": str(e),
            "stdout": "",
            "stderr": "",
            "returncode": -1,
        }

# ─── Braille Utilities ──────────────────────────────────────────────────────────

def is_braille_char(ch: str) -> bool:
    """Check if a character is in the 8-dot braille Unicode block."""
    return 0x2800 <= ord(ch) <= 0x28FF

def extract_braille(text: str) -> str:
    """Extract only braille characters from a response."""
    return "".join(ch for ch in text if is_braille_char(ch))

def validate_braille_response(text: str) -> Tuple[bool, str, float]:
    """
    Validate that a response is valid braille.
    Returns (is_valid, braille_only, purity_ratio).
    """
    braille = extract_braille(text)
    if len(text.strip()) == 0:
        return False, "", 0.0
    non_whitespace = text.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    if len(non_whitespace) == 0:
        return False, "", 0.0
    purity = len(braille) / len(non_whitespace)
    return purity >= 0.8, braille, purity  # Allow 80% purity (models may sneak in spaces)

def braille_to_dots(ch: str) -> List[bool]:
    """Convert a single braille Unicode character to its 8-dot pattern."""
    offset = ord(ch) - 0x2800
    return [(offset >> i) & 1 == 1 for i in range(8)]

def dots_to_braille(dots: List[bool]) -> str:
    """Convert an 8-dot pattern back to a braille Unicode character."""
    offset = sum((1 << i) for i, d in enumerate(dots) if d)
    return chr(0x2800 + offset)

def braille_string_to_dot_matrix(braille: str) -> List[List[bool]]:
    """Convert a braille string to a list of 8-dot patterns."""
    return [braille_to_dots(ch) for ch in braille if is_braille_char(ch)]

def compute_convergence(responses: List[str]) -> float:
    """
    Measure dot-level agreement across all valid braille responses.
    Returns 0.0 (no agreement) to 1.0 (perfect consensus).
    """
    braille_strings = [extract_braille(r) for r in responses]
    braille_strings = [b for b in braille_strings if len(b) > 0]
    
    if len(braille_strings) < 2:
        return 0.0
    
    # Pad/truncate to shortest length for comparison
    min_len = min(len(b) for b in braille_strings)
    if min_len == 0:
        return 0.0
    
    total_dots = 0
    agreed_dots = 0
    
    for pos in range(min_len):
        dot_patterns = [braille_to_dots(b[pos]) for b in braille_strings]
        for dot_idx in range(8):
            total_dots += 1
            dot_values = [p[dot_idx] for p in dot_patterns]
            if all(d == dot_values[0] for d in dot_values):
                agreed_dots += 1
    
    return agreed_dots / total_dots if total_dots > 0 else 0.0

def compute_majority_consensus(responses: List[str]) -> str:
    """
    Compute majority-vote braille string from all responses.
    For each cell position, for each dot, take the majority vote.
    """
    braille_strings = [extract_braille(r) for r in responses]
    braille_strings = [b for b in braille_strings if len(b) > 0]
    
    if not braille_strings:
        return ""
    
    min_len = min(len(b) for b in braille_strings)
    consensus = []
    
    for pos in range(min_len):
        dot_patterns = [braille_to_dots(b[pos]) for b in braille_strings]
        majority_dots = []
        for dot_idx in range(8):
            votes = sum(1 for p in dot_patterns if p[dot_idx])
            majority_dots.append(votes > len(dot_patterns) / 2)
        consensus.append(dots_to_braille(majority_dots))
    
    return "".join(consensus)

def braille_to_text_approx(braille: str) -> str:
    """
    Approximate braille-to-text decoding for Grade 1 braille.
    Maps common 6-dot braille patterns to ASCII letters.
    """
    BRAILLE_TO_CHAR = {
        '⠁': 'a', '⠃': 'b', '⠉': 'c', '⠙': 'd', '⠑': 'e',
        '⠋': 'f', '⠛': 'g', '⠓': 'h', '⠊': 'i', '⠚': 'j',
        '⠅': 'k', '⠇': 'l', '⠍': 'm', '⠝': 'n', '⠕': 'o',
        '⠏': 'p', '⠟': 'q', '⠗': 'r', '⠎': 's', '⠞': 't',
        '⠥': 'u', '⠧': 'v', '⠺': 'w', '⠭': 'x', '⠽': 'y',
        '⠵': 'z', '⠀': ' ',
        '⠼': '#',  # number indicator
        '⠂': '1', '⠆': '2', '⠒': '3', '⠲': '4', '⠢': '5',
        '⠖': '6', '⠶': '7', '⠦': '8', '⠔': '9', '⠴': '0',
    }
    result = []
    for ch in braille:
        if ch in BRAILLE_TO_CHAR:
            result.append(BRAILLE_TO_CHAR[ch])
        else:
            result.append(f'[{hex(ord(ch))}]')
    return "".join(result)

# ─── API Calls ──────────────────────────────────────────────────────────────────

async def call_model_braille(
    session: aiohttp.ClientSession,
    provider: Provider,
    model: str,
    messages: List[Dict]
) -> Dict:
    """Call a single model on a specific provider."""
    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        async with session.post(
            provider.api_url, headers=headers, json=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                data = await response.json()
                choices = data.get("choices") or []
                if not choices or not choices[0].get("message"):
                    return {
                        "model": model, "provider": provider.name,
                        "raw_response": "", "braille": "",
                        "is_valid_braille": False, "purity": 0.0,
                        "success": False, "error": "Empty choices"
                    }
                raw = choices[0]["message"].get("content") or ""
                is_valid, braille, purity = validate_braille_response(raw)
                return {
                    "model": model,
                    "provider": provider.name,
                    "raw_response": raw,
                    "braille": braille,
                    "is_valid_braille": is_valid,
                    "purity": purity,
                    "success": True
                }
            else:
                error_text = await response.text()
                return {
                    "model": model, "provider": provider.name,
                    "raw_response": "", "braille": "",
                    "is_valid_braille": False, "purity": 0.0,
                    "success": False,
                    "error": f"{response.status} - {error_text[:200]}"
                }
    except asyncio.TimeoutError:
        return {
            "model": model, "provider": provider.name,
            "raw_response": "", "braille": "",
            "is_valid_braille": False, "purity": 0.0,
            "success": False, "error": "Timeout"
        }
    except Exception as e:
        return {
            "model": model, "provider": provider.name,
            "raw_response": "", "braille": "",
            "is_valid_braille": False, "purity": 0.0,
            "success": False, "error": str(e)
        }

async def run_provider_round(
    session: aiohttp.ClientSession,
    provider: Provider,
    model_histories: Dict[str, List[Dict]],
    status_placeholder=None,
    last_user_msg: str = None,
) -> List[Dict]:
    """Run one round with auto-swap: failed models get replaced from the bench."""
    all_models = list(model_histories.keys())
    model_status = {m: "🤔" for m in all_models}
    model_braille_preview = {}  # model → braille snippet for display
    all_results = []
    
    # Assign colors for this provider's models
    color_map = get_provider_colors(provider, all_models)
    
    # Priority order: ✅ first (bright), 🤔 middle, ⚠️/❌ last (faded)
    STATUS_RANK = {"✅": 0, "🤔": 1, "⚠️": 2, "❌": 3}
    
    def render_status():
        if status_placeholder is None:
            return
        sorted_models = sorted(
            all_models,
            key=lambda m: STATUS_RANK.get(model_status[m], 1)
        )
        parts = []
        for m in sorted_models:
            s = model_status[m]
            tag = provider.model_tag(m)
            color = color_map.get(m, (128, 128, 128))
            braille = model_braille_preview.get(m, "")
            # Truncate braille to keep it compact
            braille_display = braille[:12] + ("…" if len(braille) > 12 else "") if braille else ""
            
            if s in ("❌", "⚠️"):
                parts.append(
                    f'<span style="opacity:0.3;font-size:0.85em">'
                    f'{s} <span style="color:rgb({color[0]},{color[1]},{color[2]})">{tag}</span>'
                    f'</span>'
                )
            elif s == "🤔":
                parts.append(
                    f'<span style="opacity:0.5;font-size:0.85em">'
                    f'{s} <span style="color:rgb({color[0]},{color[1]},{color[2]})">{tag}</span>'
                    f'</span>'
                )
            else:
                parts.append(
                    f'<span style="font-size:0.85em">{s} '
                    f'<span style="color:rgb({color[0]},{color[1]},{color[2]})">'
                    f'{tag} {braille_display}</span></span>'
                )
        status_placeholder.markdown(
            " &nbsp; ".join(parts),
            unsafe_allow_html=True
        )
    
    render_status()
    
    async def call_and_track(model):
        result = await call_model_braille(session, provider, model, model_histories[model])
        if result["success"] and result["is_valid_braille"]:
            model_status[model] = "✅"
            model_braille_preview[model] = result["braille"]
        elif result["success"]:
            model_status[model] = "⚠️"
        else:
            model_status[model] = "❌"
        render_status()
        return result
    
    # First wave
    models_to_call = list(model_histories.keys())
    tasks = [call_and_track(m) for m in models_to_call]
    results = await asyncio.gather(*tasks)
    all_results.extend(results)
    
    # Identify failures and swap in replacements
    failed = [r["model"] for r in results if not r["success"]]
    if failed:
        swapped_in = provider.swap_failed(failed)
        if swapped_in:
            # Initialize histories for new models with same context
            for new_model in swapped_in:
                model_histories[new_model] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                ]
                if last_user_msg:
                    model_histories[new_model].append(
                        {"role": "user", "content": last_user_msg}
                    )
                model_status[new_model] = "🤔"
                all_models.append(new_model)
            
            render_status()
            
            # Second wave — only the replacements
            retry_tasks = [call_and_track(m) for m in swapped_in]
            retry_results = await asyncio.gather(*retry_tasks)
            all_results.extend(retry_results)
    
    return all_results

async def health_check_model(
    session: aiohttp.ClientSession, provider: Provider, model: str
) -> Dict:
    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "⠓⠊"}],
        "max_tokens": 10
    }
    try:
        start_time = time.time()
        async with session.post(
            provider.api_url, headers=headers, json=payload,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            elapsed = time.time() - start_time
            return {
                "model": model, "provider": provider.name,
                "healthy": response.status == 200,
                "response_time": elapsed
            }
    except Exception:
        return {
            "model": model, "provider": provider.name,
            "healthy": False, "response_time": None
        }

async def health_check_all_providers() -> Dict[str, List[Dict]]:
    results = {}
    async with aiohttp.ClientSession(skip_auto_headers=["Accept-Encoding"]) as session:
        for provider in PROVIDERS:
            if not provider.api_key:
                continue
            models = provider.get_active_models()
            tasks = [health_check_model(session, provider, m) for m in models]
            results[provider.name] = await asyncio.gather(*tasks)
    return results

# ─── Visualization ──────────────────────────────────────────────────────────────

def render_braille_overlay(
    model_braille: Dict[str, str],
    color_map: Dict[str, Tuple],
    cell_size: int = 48,
    dot_radius: int = 8,
    max_cells: int = 60
) -> Image.Image:
    """Render braille overlay with provider-aware colors."""
    lengths = [len(b) for b in model_braille.values() if b]
    if not lengths:
        img = Image.new("RGB", (200, 60), "white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 20), "No valid braille", fill="gray")
        return img
    
    num_cells = min(max(lengths), max_cells)
    padding = 6
    cell_w = cell_size
    cell_h = int(cell_size * 2)
    img_w = num_cells * (cell_w + padding) + padding
    img_h = cell_h + padding * 2
    
    canvas = np.full((img_h, img_w, 3), 255, dtype=np.float64)
    active_models = sum(1 for b in model_braille.values() if b)
    
    for model, braille in model_braille.items():
        if not braille:
            continue
        color = color_map.get(model, (128, 128, 128))
        dots_list = braille_string_to_dot_matrix(braille)
        
        for cell_idx, dots in enumerate(dots_list[:num_cells]):
            cx = padding + cell_idx * (cell_w + padding)
            cy = padding
            
            for dot_idx, is_on in enumerate(dots):
                if not is_on:
                    continue
                row, col = BRAILLE_DOT_POSITIONS[dot_idx]
                dx = cx + int((col + 0.5) * cell_w / 2)
                dy = cy + int((row + 0.5) * cell_h / 4)
                
                for py in range(max(0, dy - dot_radius), min(img_h, dy + dot_radius + 1)):
                    for px in range(max(0, dx - dot_radius), min(img_w, dx + dot_radius + 1)):
                        if (px - dx) ** 2 + (py - dy) ** 2 <= dot_radius ** 2:
                            subtract = np.array([
                                255 - color[0],
                                255 - color[1],
                                255 - color[2]
                            ], dtype=np.float64) / active_models
                            canvas[py, px] -= subtract
    
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return Image.fromarray(canvas)

def render_convergence_chart(histories: Dict[str, List[float]]) -> None:
    """Display convergence for multiple clusters on one chart."""
    import pandas as pd
    max_len = max(len(h) for h in histories.values()) if histories else 0
    if max_len == 0:
        return
    data = {"Round": list(range(1, max_len + 1))}
    for label, hist in histories.items():
        padded = hist + [hist[-1]] * (max_len - len(hist)) if hist else [0.0] * max_len
        data[label] = padded
    df = pd.DataFrame(data)
    st.line_chart(df.set_index("Round"), height=200)

# ─── Codebook Analysis ─────────────────────────────────────────────────────────

def pairwise_dot_similarity(a: str, b: str) -> float:
    """Dot-level similarity between two braille strings (0.0–1.0)."""
    ba = extract_braille(a)
    bb = extract_braille(b)
    if not ba or not bb:
        return 0.0
    min_len = min(len(ba), len(bb))
    total = 0
    agreed = 0
    for i in range(min_len):
        da = braille_to_dots(ba[i])
        db = braille_to_dots(bb[i])
        for j in range(8):
            total += 1
            if da[j] == db[j]:
                agreed += 1
    return agreed / total if total > 0 else 0.0

def cluster_codebooks(
    model_braille: Dict[str, str],
    threshold: float = 0.85
) -> List[List[str]]:
    """
    Cluster models by encoding similarity.
    Models with ≥threshold dot agreement share a 'codebook'.
    Returns list of clusters (each cluster = list of model names).
    """
    valid = {m: b for m, b in model_braille.items() if b}
    models = list(valid.keys())
    if not models:
        return []
    
    # Build adjacency via pairwise similarity
    clusters = []
    assigned = set()
    
    for i, m1 in enumerate(models):
        if m1 in assigned:
            continue
        cluster = [m1]
        assigned.add(m1)
        for j in range(i + 1, len(models)):
            m2 = models[j]
            if m2 in assigned:
                continue
            sim = pairwise_dot_similarity(valid[m1], valid[m2])
            if sim >= threshold:
                cluster.append(m2)
                assigned.add(m2)
        clusters.append(cluster)
    
    # Sort: largest cluster first
    clusters.sort(key=len, reverse=True)
    return clusters

def render_codebook_map(
    model_braille: Dict[str, str],
    color_map: Dict[str, Tuple],
    clusters: List[List[str]],
    name: str = "",
) -> None:
    """
    Render the codebook divergence visualization.
    Shows: which models converged on which encoding, with the Bayesian framework.
    """
    if not clusters:
        return
    
    total_models = sum(len(c) for c in clusters)
    n_codebooks = len(clusters)
    
    # Header: the formal equation
    st.markdown(
        f'<div style="text-align:center;font-size:0.8em;opacity:0.7;margin-bottom:8px">'
        f'(ℓ̂, x̂) = argmax<sub>ℓ,x</sub> P(y|x,ℓ) · P(x|ℓ) · P(ℓ|context) '
        f'&nbsp;—&nbsp; {n_codebooks} codebook{"s" if n_codebooks != 1 else ""} detected'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # Render each cluster as a visual group
    for ci, cluster in enumerate(clusters):
        weight = len(cluster) / total_models
        is_dominant = ci == 0
        
        # Compute this cluster's consensus braille
        valid_in_cluster = [model_braille[m] for m in cluster if model_braille.get(m)]
        cluster_consensus = compute_majority_consensus(valid_in_cluster) if valid_in_cluster else ""
        cluster_decoded = braille_to_text_approx(cluster_consensus) if cluster_consensus else "?"
        
        # Build the model pills for this cluster
        pills = []
        for m in cluster:
            color = color_map.get(m, (128, 128, 128))
            short = m.split("/")[-1] if "/" in m else m
            # Truncate long names
            if len(short) > 20:
                short = short[:18] + "…"
            pills.append(
                f'<span style="display:inline-block;padding:2px 8px;margin:2px;'
                f'border-radius:12px;font-size:0.75em;'
                f'background:rgba({color[0]},{color[1]},{color[2]},0.2);'
                f'border:1px solid rgba({color[0]},{color[1]},{color[2]},0.6);'
                f'color:rgb({max(0,color[0]-40)},{max(0,color[1]-40)},{max(0,color[2]-40)})">'
                f'{short}</span>'
            )
        
        # Cluster box
        border_opacity = "1" if is_dominant else "0.4"
        bg_alpha = "0.06" if is_dominant else "0.02"
        label = "D<sub>ℓ̂</sub> — dominant codebook" if is_dominant else f"D<sub>ℓ{ci+1}</sub> — minority codebook"
        
        st.markdown(
            f'<div style="border:1px solid rgba(100,100,100,{border_opacity});'
            f'border-radius:8px;padding:10px 14px;margin:6px 0;'
            f'background:rgba(100,100,100,{bg_alpha})">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<span style="font-size:0.8em;opacity:0.7">{label}</span>'
            f'<span style="font-size:0.8em;font-weight:bold">'
            f'P(ℓ|y) ≈ {weight:.0%} · {len(cluster)}/{total_models} models</span>'
            f'</div>'
            f'<div style="margin:6px 0">{"".join(pills)}</div>'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
            f'<code style="font-size:1.2em;letter-spacing:2px">{cluster_consensus[:30]}</code>'
            f'<span style="font-size:0.85em;opacity:0.6">→ {cluster_decoded[:30]}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# ─── Cerebellar Loop ────────────────────────────────────────────────────────────

def detect_plateau(history: List[float]) -> bool:
    if len(history) < PLATEAU_WINDOW + 1:
        return False
    recent = history[-PLATEAU_WINDOW:]
    baseline = history[-(PLATEAU_WINDOW + 1)]
    max_improvement = max(r - baseline for r in recent)
    return max_improvement < PLATEAU_EPSILON

def collect_round_data(results: List[Dict]) -> Tuple[Dict[str, str], int]:
    """Extract model_braille map and valid count from results."""
    model_braille = {}
    valid_count = 0
    for r in results:
        if r["success"] and r["is_valid_braille"]:
            model_braille[r["model"]] = r["braille"]
            valid_count += 1
        else:
            model_braille[r["model"]] = ""
    return model_braille, valid_count

def apply_feedback(
    models: List[str],
    model_braille: Dict[str, str],
    model_histories: Dict[str, List[Dict]]
):
    """
    Braid outputs and feed back into all models' histories.
    Fully braille-native: feedback is encoded as braille so models
    never see Latin text after the system prompt. This makes the
    loop bidirectional — models decode braille input AND encode braille output.
    """
    # Separator: braille-encoded pipe character "|" = U+2800 + 0x7C = U+287C
    separator = chr(0x2800 + ord('|'))
    
    braid_parts = []
    for model in models:
        b = model_braille.get(model, "")
        if b:
            braid_parts.append(b)
    
    # Join with braille-encoded separator (not Latin space)
    braid = separator.join(braid_parts) if braid_parts else ""
    
    if braid:
        for model in models:
            own_braille = model_braille.get(model, "")
            if own_braille:
                model_histories[model].append(
                    {"role": "assistant", "content": own_braille}
                )
            model_histories[model].append(
                {"role": "user", "content": braid}
            )
            if len(model_histories[model]) > 13:
                model_histories[model] = (
                    model_histories[model][:1] +
                    model_histories[model][-12:]
                )

async def bbid_handshake(name: str, status_container):
    """
    BBID Handshake across all providers.
    Returns per-provider BBIDs, combined BBID, and model histories.
    """
    provider_histories = {}  # {provider_name: {model: [messages]}}
    provider_results = {}    # {provider_name: results}
    
    active_providers = [p for p in PROVIDERS if p.api_key]
    if not active_providers:
        return None, {}, {}, {}
    
    # Discover all available models dynamically
    for provider in active_providers:
        if not provider._all_models:
            count = provider.discover()
            status_container.caption(
                f"{provider.name}: {count} models → "
                f"squad {len(provider.get_active_models())} ({provider.tier_summary}) · "
                f"bench {len(provider._bench)}"
            )
    
    # Prepare histories and thinking placeholders
    # Encode the name as braille so models receive braille input AND produce braille output
    # The system prompt (Latin) is the only non-braille message — like DNA being the only
    # non-protein message in the cell. Everything after is braille-native.
    braille_name = ascii_to_braille(name)
    
    thinking_placeholders = {}
    for provider in active_providers:
        models = provider.get_active_models()
        histories = {}
        for model in models:
            histories[model] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": braille_name}
            ]
        provider_histories[provider.name] = histories
        status_container.write(f"**{provider.name}** — encoding...")
        thinking_placeholders[provider.name] = status_container.empty()
    
    # Fire all providers in parallel with auto-swap
    async with aiohttp.ClientSession(skip_auto_headers=["Accept-Encoding"]) as session:
        tasks = [
            run_provider_round(
                session, provider, provider_histories[provider.name],
                thinking_placeholders[provider.name],
                last_user_msg=name
            )
            for provider in active_providers
        ]
        gathered = await asyncio.gather(*tasks)
        for provider, results in zip(active_providers, gathered):
            provider_results[provider.name] = results
    
    # Per-provider consensus
    bbid_per_provider = {}
    agreement_per_provider = {}
    all_valid_braille = []
    all_model_braille = {}
    
    for pname, results in provider_results.items():
        provider = next(p for p in PROVIDERS if p.name == pname)
        mb, vc = collect_round_data(results)
        valid = [b for b in mb.values() if b]
        conv = compute_convergence(valid) if len(valid) >= 2 else 0.0
        
        colors = get_provider_colors(provider, list(mb.keys()))
        overlay = render_braille_overlay(mb, colors)
        status_container.write(f"**{pname}** — {vc}/{len(mb)} responded · {conv:.0%}")
        status_container.image(overlay)
        
        if valid:
            bbid_per_provider[pname] = compute_majority_consensus(valid)
            all_valid_braille.extend(valid)
        agreement_per_provider[pname] = conv
        all_model_braille.update(mb)
        
        # Record assistant responses in histories
        for r in results:
            if r["success"] and r["is_valid_braille"]:
                provider_histories[pname][r["model"]].append(
                    {"role": "assistant", "content": r["braille"]}
                )
    
    # Combined consensus
    combined_conv = compute_convergence(all_valid_braille) if len(all_valid_braille) >= 2 else 0.0
    combined_bbid = compute_majority_consensus(all_valid_braille) if all_valid_braille else ""
    
    all_colors = get_all_model_colors(
        {pname: list(provider_histories[pname].keys()) for pname in provider_histories}
    )
    combined_overlay = render_braille_overlay(all_model_braille, all_colors)
    status_container.write(f"**Combined** — {combined_conv:.0%} agreement")
    status_container.image(combined_overlay)
    
    # Codebook analysis — cluster models by encoding strategy
    clusters = cluster_codebooks(all_model_braille)
    render_codebook_map(all_model_braille, all_colors, clusters, name=name)
    
    return combined_bbid, {
        **agreement_per_provider,
        "combined": combined_conv,
    }, provider_histories, bbid_per_provider, all_model_braille

async def cerebellar_loop(
    prompt: str,
    provider_histories: Dict[str, Dict[str, List[Dict]]],
    status_container
):
    """
    Multi-provider cerebellar loop.
    Runs each provider cluster independently + tracks combined consensus.
    """
    active_providers = [p for p in PROVIDERS if p.name in provider_histories]
    
    # Append prompt to all models — encoded as braille for fully braille-native loop
    braille_prompt = ascii_to_braille(prompt)
    for pname in provider_histories:
        for model in provider_histories[pname]:
            provider_histories[pname][model].append(
                {"role": "user", "content": braille_prompt}
            )
    
    conv_histories = {p.name: [] for p in active_providers}
    conv_histories["Combined"] = []
    all_rounds = []
    outcome = "inconclusive"
    
    async with aiohttp.ClientSession(skip_auto_headers=["Accept-Encoding"]) as session:
        for iteration in range(1, MAX_ITERATIONS + 1):
            status_container.write(f"**Round {iteration}** ⚡")
            
            # Create thinking placeholders per provider
            thinking_placeholders = {}
            for provider in active_providers:
                thinking_placeholders[provider.name] = status_container.empty()
            
            # Fire all providers in parallel with auto-swap
            provider_results = {}
            all_tasks = []
            for provider in active_providers:
                task = run_provider_round(
                    session, provider, provider_histories[provider.name],
                    thinking_placeholders[provider.name],
                    last_user_msg=prompt
                )
                all_tasks.append((provider, task))
            
            gathered = await asyncio.gather(*[t for _, t in all_tasks])
            for (provider, _), results in zip(all_tasks, gathered):
                provider_results[provider.name] = results
            
            # Per-provider convergence
            round_data = {"iteration": iteration, "providers": {}}
            all_valid_this_round = []
            all_model_braille = {}
            
            for provider in active_providers:
                pname = provider.name
                results = provider_results[pname]
                mb, vc = collect_round_data(results)
                valid = [b for b in mb.values() if b]
                conv = compute_convergence(valid) if len(valid) >= 2 else 0.0
                conv_histories[pname].append(conv)
                
                round_data["providers"][pname] = {
                    "results": results,
                    "model_braille": mb,
                    "convergence": conv,
                    "valid_count": vc,
                    "total": len(mb)
                }
                all_valid_this_round.extend(valid)
                all_model_braille.update(mb)
            
            # Combined convergence
            combined_conv = compute_convergence(all_valid_this_round) \
                if len(all_valid_this_round) >= 2 else 0.0
            conv_histories["Combined"].append(combined_conv)
            round_data["combined_convergence"] = combined_conv
            all_rounds.append(round_data)
            
            # Display per-provider overlays side by side
            cols = st.columns(len(active_providers) + 1)
            for idx, provider in enumerate(active_providers):
                pname = provider.name
                pd_data = round_data["providers"][pname]
                colors = get_provider_colors(
                    provider, list(pd_data["model_braille"].keys())
                )
                overlay = render_braille_overlay(pd_data["model_braille"], colors)
                with cols[idx]:
                    st.caption(f"{pname}: {pd_data['convergence']:.0%}")
                    st.image(overlay)
            
            # Combined overlay in last column
            all_colors = get_all_model_colors(
                {p.name: list(round_data["providers"][p.name]["model_braille"].keys())
                 for p in active_providers}
            )
            combined_overlay = render_braille_overlay(all_model_braille, all_colors)
            with cols[-1]:
                st.caption(f"Combined: {combined_conv:.0%}")
                st.image(combined_overlay)
            
            # ─── Termination ─────────────────────────────────────────────
            if combined_conv >= CONVERGENCE_THRESHOLD:
                outcome = "consensus"
                status_container.success(
                    f"✅ Consensus — {combined_conv:.0%} after {iteration} rounds"
                )
                break
            
            if detect_plateau(conv_histories["Combined"]):
                outcome = "disagreement"
                status_container.warning(
                    f"⚡ Stable disagreement — plateaued at {combined_conv:.0%}"
                )
                break
            
            # ─── Feedback: braid within each provider cluster ────────────
            for provider in active_providers:
                pname = provider.name
                models = list(provider_histories[pname].keys())
                mb = round_data["providers"][pname]["model_braille"]
                apply_feedback(models, mb, provider_histories[pname])
    
    return all_rounds, conv_histories, outcome

# ─── Main App ───────────────────────────────────────────────────────────────────

def outcome_icon(conv: float) -> str:
    if conv >= CONVERGENCE_THRESHOLD:
        return "⬛"
    elif conv > 0.5:
        return "🌈"
    else:
        return "⏳"

def main():
    st.set_page_config(page_title="Cerebellar Braille Loop", page_icon="🧠", layout="wide")
    st.title("🧠 Cerebellar Braille Loop")
    
    # ─── Session state ───────────────────────────────────────────────────
    for key, default in [
        ("bbid", None), ("bbid_agreements", None), ("user_name", None),
        ("provider_histories", None), ("health", None),
        ("bbid_per_provider", None), ("bbid_model_braille", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default
    
    # ─── Phase 1: BBID Handshake ─────────────────────────────────────────
    if not st.session_state.bbid:
        st.markdown("### What's your name?")
        st.caption("All models across all providers will encode it in braille.")
        name_input = st.text_input("Name", key="name_input", label_visibility="collapsed")
        if st.button("Identify", type="primary"):
            if not name_input.strip():
                return
            
            name = name_input.strip()
            status = st.container()
            
            combined_bbid, agreements, provider_histories, bbid_per_provider, model_braille = \
                asyncio.run(bbid_handshake(name, status))
            
            if combined_bbid:
                st.session_state.user_name = name
                st.session_state.bbid = combined_bbid
                st.session_state.bbid_agreements = agreements
                st.session_state.provider_histories = provider_histories
                st.session_state.bbid_per_provider = bbid_per_provider
                st.session_state.bbid_model_braille = model_braille
                st.rerun()
            else:
                st.error("No models responded. Check API keys.")
        return
    
    # ─── Phase 2: Identified ─────────────────────────────────────────────
    name = st.session_state.user_name
    bbid = st.session_state.bbid
    agreements = st.session_state.bbid_agreements or {}
    bbid_per_provider = st.session_state.bbid_per_provider or {}
    combined_agr = agreements.get("combined", 0)
    
    # Ensure providers are discovered (state lost on rerun)
    for p in PROVIDERS:
        if p.api_key and not p._all_models:
            p.discover()
    
    # Show identity + pool info with tiers
    pool_info = " · ".join(
        f"{p.name}: {len(p.get_active_models())}/{p.pool_size} ({p.tier_summary})"
        for p in PROVIDERS if p.api_key
    )
    
    decoded = braille_to_text_approx(bbid)
    if combined_agr >= CONVERGENCE_THRESHOLD:
        st.success(f"BBID verified by consensus ({combined_agr:.0%}): `{bbid}` → {decoded}")
    else:
        st.info(f"BBID by majority ({combined_agr:.0%}): `{bbid}` → {decoded}")
    
    # Per-provider BBIDs
    for pname, pbbid in bbid_per_provider.items():
        pagr = agreements.get(pname, 0)
        st.caption(f"{outcome_icon(pagr)} {pname}: `{pbbid}` ({pagr:.0%})")
    
    st.caption(f"Models: {pool_info}")
    
    # Codebook map from handshake (persistent)
    model_braille = st.session_state.bbid_model_braille
    if model_braille:
        clusters = cluster_codebooks(model_braille)
        if clusters:
            with st.expander("Codebook Analysis", expanded=len(clusters) > 1):
                all_colors = get_all_model_colors(
                    {pname: list((st.session_state.provider_histories or {}).get(pname, {}).keys())
                     for pname in bbid_per_provider}
                )
                render_codebook_map(model_braille, all_colors, clusters, name=name)
    
    if st.button("Not me", key="reset_bbid"):
        for key in ["bbid", "bbid_agreements", "user_name",
                     "provider_histories", "bbid_per_provider", "bbid_model_braille"]:
            st.session_state[key] = None
        st.rerun()
    
    # ─── Health ──────────────────────────────────────────────────────────
    with st.expander("Endpoints", expanded=False):
        if st.button("⚡ Warm up"):
            with st.spinner("..."):
                st.session_state.health = asyncio.run(health_check_all_providers())
        
        if st.session_state.health:
            for pname, results in st.session_state.health.items():
                st.caption(f"**{pname}**")
                for r in results:
                    dot = "🟢" if r["healthy"] else "🔴"
                    t = f"{r['response_time']:.1f}s" if r["response_time"] else ""
                    st.caption(f"  {dot} {r['model']} {t}")
    
    # ─── Prompt ──────────────────────────────────────────────────────────
    col_prompt, col_mode = st.columns([4, 1])
    with col_prompt:
        prompt = st.text_input("Prompt")
    with col_mode:
        tool_mode = st.checkbox("🔧 Tool call", help="Models propose bash commands in braille. Only consensus commands execute.")
    
    if st.button("Go", type="primary"):
        if not prompt:
            st.error("Enter a prompt.")
            return
        
        status = st.container()
        
        # Use tool-call system prompt if in tool mode
        sys_prompt = SYSTEM_PROMPT
        if tool_mode:
            sys_prompt = (
                "You communicate exclusively in 8-dot braille Unicode characters (U+2800 to U+28FF). "
                "Each braille character encodes one ASCII byte: chr(0x2800 + byte_value). "
                "The user will ask you to perform a task. Respond with a single bash command "
                "encoded in this braille-ASCII mapping. Never use Latin text. "
                "Example: 'ls -la' = ⠇⠎⠀⠤⠇⠁. Respond only with the braille-encoded command."
            )
        
        provider_histories = st.session_state.provider_histories
        if not provider_histories:
            provider_histories = {}
            for p in PROVIDERS:
                if p.api_key:
                    if not p._all_models:
                        p.discover()
                    provider_histories[p.name] = {}
                    for m in p.get_active_models():
                        provider_histories[p.name][m] = [
                            {"role": "system", "content": sys_prompt},
                        ]
        
        all_rounds, conv_histories, outcome = asyncio.run(
            cerebellar_loop(prompt, provider_histories, status)
        )
        
        st.session_state.provider_histories = provider_histories
        
        # ─── Outcome ────────────────────────────────────────────────
        last_round = all_rounds[-1] if all_rounds else None
        if last_round:
            # Per-provider consensus
            for pname, pd_data in last_round["providers"].items():
                valid = [b for b in pd_data["model_braille"].values() if b]
                if valid:
                    pcons = compute_majority_consensus(valid)
                    decoded = braille_to_text_approx(pcons)
                    st.caption(
                        f"{outcome_icon(pd_data['convergence'])} "
                        f"**{pname}** ({pd_data['convergence']:.0%}): "
                        f"`{pcons[:40]}` → {decoded[:40]}"
                    )
            
            # Combined
            all_valid = []
            for pd_data in last_round["providers"].values():
                all_valid.extend(b for b in pd_data["model_braille"].values() if b)
            
            if all_valid:
                consensus = compute_majority_consensus(all_valid)
                decoded = braille_to_text_approx(consensus)
                
                if outcome == "consensus":
                    st.markdown("### ⬛ Consensus")
                elif outcome == "disagreement":
                    st.markdown("### 🌈 Disagreement")
                else:
                    st.markdown("### ⏳ Inconclusive")
                
                st.code(consensus, language=None)
                st.caption(f"Decoded: {decoded}")
                
                # ─── Tool Call Execution ─────────────────────────────────
                if tool_mode and outcome == "consensus":
                    ascii_cmd = braille_to_ascii(consensus)
                    st.markdown("#### 🔧 Tool Call")
                    st.code(ascii_cmd, language="bash")
                    
                    if is_safe_command(ascii_cmd):
                        exec_result = execute_consensus_command(ascii_cmd)
                        if exec_result["executed"]:
                            st.success(f"✅ Executed (exit {exec_result['returncode']})")
                            if exec_result["stdout"]:
                                st.code(exec_result["stdout"], language=None)
                            if exec_result["stderr"]:
                                st.error(exec_result["stderr"])
                        else:
                            st.warning(f"Not executed: {exec_result['reason']}")
                    else:
                        st.warning(
                            f"⚠️ Command `{ascii_cmd[:60]}` not in safe allowlist. "
                            f"Consensus reached but execution blocked."
                        )
                elif tool_mode and outcome != "consensus":
                    ascii_cmd = braille_to_ascii(consensus)
                    st.info(
                        f"🔧 No consensus — command not executed. "
                        f"Best guess: `{ascii_cmd[:80]}`"
                    )
                
                # Codebook analysis on final round
                all_model_braille = {}
                for pd_data in last_round["providers"].values():
                    all_model_braille.update(pd_data["model_braille"])
                all_colors = get_all_model_colors(
                    {pname: list(pd_data["model_braille"].keys())
                     for pname, pd_data in last_round["providers"].items()}
                )
                clusters = cluster_codebooks(all_model_braille)
                if len(clusters) > 1:
                    st.markdown("#### Codebook Divergence")
                    render_codebook_map(all_model_braille, all_colors, clusters)
            else:
                st.warning("No valid braille responses.")
        
        if conv_histories:
            render_convergence_chart(conv_histories)
        
        with st.expander("Details"):
            for rd in all_rounds:
                st.caption(f"**Round {rd['iteration']}** · Combined: {rd['combined_convergence']:.0%}")
                for pname, pd_data in rd["providers"].items():
                    st.caption(f"  {pname}: {pd_data['convergence']:.0%} · {pd_data['valid_count']}/{pd_data['total']}")
                    for r in pd_data["results"]:
                        if r["success"]:
                            icon = "✅" if r["is_valid_braille"] else "⚠️"
                            raw_preview = r["raw_response"][:80].replace("\n", " ")
                            st.caption(f"    {icon} {r['model']} ({r['purity']:.0%}): `{r['braille'][:50]}`")
                            if not r["is_valid_braille"]:
                                st.caption(f"      raw: {raw_preview}")
                        else:
                            st.caption(f"    ❌ {r['model']}: {r.get('error', '?')[:80]}")

if __name__ == "__main__":
    main()
