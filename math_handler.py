"""
Local math/arithmetic handler — no LLM needed for these queries.
detect_and_compute(transcript) -> spoken answer string | None

Handles: square root, factors, average/mean, basic arithmetic (+−×÷),
         powers, percentages. Number words ("one", "two") are normalised
         to digits before matching.
"""
from __future__ import annotations
import math
import re

_WORD_TO_NUM: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
}


def _words_to_digits(text: str) -> str:
    """Replace number words with digit strings."""
    def _replace(m: re.Match) -> str:
        w = m.group(0).lower()
        return str(_WORD_TO_NUM[w]) if w in _WORD_TO_NUM else m.group(0)
    pattern = r'\b(' + '|'.join(_WORD_TO_NUM) + r')\b'
    return re.sub(pattern, _replace, text, flags=re.IGNORECASE)


def _strip_wake(text: str) -> str:
    """Remove wake word and filler greeting."""
    return re.sub(
        r'\b(hi\s+)?(pinku|pinky|pink|pingu|hey\s+pinku)\b',
        '', text, flags=re.IGNORECASE,
    ).strip()


def _nice(n: float) -> str:
    """Format number for speech: no trailing zeros, no sci notation."""
    if n == int(n) and abs(n) < 1e15:
        return str(int(n))
    formatted = f"{n:.8f}".rstrip("0").rstrip(".")
    return formatted


def detect_and_compute(transcript: str) -> str | None:
    """
    Return a spoken answer if transcript is a simple math query, else None.
    Call this before any LLM call — instant responses for arithmetic.
    """
    t = _words_to_digits(_strip_wake(transcript)).lower().strip()

    # ── Square root ───────────────────────────────────────────────────────────
    m = re.search(r'square\s+root\s+of\s+([\d.]+)', t)
    if m:
        n = float(m.group(1))
        r = math.sqrt(n)
        label = _nice(n)
        return (f"The square root of {label} is {_nice(r)}."
                if r == int(r) else
                f"The square root of {label} is approximately {r:.4f}.")

    # ── Cube root ─────────────────────────────────────────────────────────────
    m = re.search(r'cube\s+root\s+of\s+([\d.]+)', t)
    if m:
        n = float(m.group(1))
        r = n ** (1 / 3)
        return f"The cube root of {_nice(n)} is approximately {r:.4f}."

    # ── Factors ───────────────────────────────────────────────────────────────
    m = re.search(r'factors?\s+of\s+(\d+)', t)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 1_000_000:
            factors = [i for i in range(1, n + 1) if n % i == 0]
            if len(factors) <= 20:
                return f"The factors of {n} are {', '.join(str(f) for f in factors)}."
            return (f"{n} has {len(factors)} factors. "
                    f"The smallest are {', '.join(str(f) for f in factors[:8])} and so on.")

    # ── Average / mean ────────────────────────────────────────────────────────
    m = re.search(r'(?:average|mean)\s+of\s+(.+)', t)
    if m:
        nums = re.findall(r'[\d.]+', m.group(1))
        if len(nums) >= 2:
            vals = [float(x) for x in nums]
            avg = sum(vals) / len(vals)
            listed = ', '.join(_nice(v) for v in vals)
            return f"The average of {listed} is {_nice(avg)}."

    # ── Basic arithmetic ──────────────────────────────────────────────────────
    m = re.search(
        r'(?:what\s+is\s+|calculate\s+|compute\s+)?'
        r'([\d.]+)\s*'
        r'(plus|added\s+to|minus|subtract(?:ed\s+from)?|times|multiplied\s+by'
        r'|divided\s+by|over|[+\-×x*/÷])\s*'
        r'([\d.]+)',
        t,
    )
    if m:
        a, op, b = float(m.group(1)), m.group(2).strip(), float(m.group(3))
        result: float
        if op in ('plus', 'added to', '+'):
            result = a + b
        elif op in ('minus', '-') or 'subtract' in op:
            result = a - b
        elif op in ('times', 'multiplied by', '×', 'x', '*'):
            result = a * b
        elif op in ('divided by', 'over', '÷', '/'):
            if b == 0:
                return "I can't divide by zero."
            result = a / b
        else:
            return None
        return f"{_nice(a)} {op} {_nice(b)} equals {_nice(result)}."

    # ── Power ─────────────────────────────────────────────────────────────────
    m = re.search(
        r'([\d.]+)\s+(?:to\s+the\s+(?:power\s+of\s+)?|raised\s+to\s+)([\d.]+)',
        t,
    )
    if m:
        base, exp = float(m.group(1)), float(m.group(2))
        try:
            result = base ** exp
            return f"{_nice(base)} to the power of {_nice(exp)} is {_nice(result)}."
        except OverflowError:
            return f"That number is too large to compute."

    # ── Percentage ────────────────────────────────────────────────────────────
    m = re.search(r'([\d.]+)\s*(?:percent|%)\s+of\s+([\d.]+)', t)
    if m:
        pct, total = float(m.group(1)), float(m.group(2))
        result = pct * total / 100
        return f"{_nice(pct)} percent of {_nice(total)} is {_nice(result)}."

    # ── LCM / GCD ─────────────────────────────────────────────────────────────
    m = re.search(r'(?:lcm|lowest\s+common\s+multiple)\s+of\s+(\d+)\s+and\s+(\d+)', t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        result = abs(a * b) // math.gcd(a, b)
        return f"The LCM of {a} and {b} is {result}."

    m = re.search(r'(?:gcd|hcf|greatest\s+common\s+(?:divisor|factor))\s+of\s+(\d+)\s+and\s+(\d+)', t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return f"The GCF of {a} and {b} is {math.gcd(a, b)}."

    # ── Calendar facts (Python knows these exactly — never ask an LLM) ────────
    import calendar as _cal
    from datetime import date as _date
    _today = _date.today()

    # Days in this/current year
    if re.search(r'(?:how\s+many\s+days\s+(?:are\s+there\s+)?in\s+(?:this|the\s+current)\s+year'
                 r'|days\s+in\s+(?:this|the\s+current)\s+year)', t):
        yr   = _today.year
        days = 366 if _cal.isleap(yr) else 365
        leap = " It's a leap year." if _cal.isleap(yr) else ""
        return f"{yr} has {days} days.{leap}"

    # Days in a specific year
    m = re.search(r'(?:how\s+many\s+days\s+(?:are\s+there\s+)?in\s+(?:the\s+year\s+)?|days\s+in\s+(?:year\s+)?)(\d{4})', t)
    if m:
        yr   = int(m.group(1))
        days = 366 if _cal.isleap(yr) else 365
        leap = " It's a leap year." if _cal.isleap(yr) else ""
        return f"{yr} has {days} days.{leap}"

    # Leap year check
    m = re.search(r'is\s+(\d{4})\s+a\s+leap\s+year', t)
    if m:
        yr = int(m.group(1))
        return (f"Yes, {yr} is a leap year." if _cal.isleap(yr)
                else f"No, {yr} is not a leap year.")

    # Days in a month
    m = re.search(
        r'(?:how\s+many\s+days\s+(?:are\s+there\s+)?in\s+)'
        r'(january|february|march|april|may|june|july|august|september|october|november|december)',
        t,
    )
    if m:
        month_name = m.group(1).capitalize()
        month_num  = list(_cal.month_name).index(month_name)
        _, days    = _cal.monthrange(_today.year, month_num)
        return f"{month_name} {_today.year} has {days} days."

    # ── Day of week ───────────────────────────────────────────────────────────
    from datetime import timedelta as _td

    if re.search(r'\b(what\s+day\s+is\s+(today|it)|today[\'s]*\s+day)\b', t):
        return f"Today is {_today.strftime('%A')}, {_today.strftime('%-d %B %Y')}."

    if re.search(r'\bwhat\s+(day\s+is\s+)?tomorrow\b', t):
        tom = _today + _td(days=1)
        return f"Tomorrow is {tom.strftime('%A')}, {tom.strftime('%-d %B %Y')}."

    if re.search(r'\bwhat\s+(day\s+is\s+)?yesterday\b', t):
        yest = _today - _td(days=1)
        return f"Yesterday was {yest.strftime('%A')}, {yest.strftime('%-d %B %Y')}."

    # "is today Sunday?" / "is today a weekday?"
    m = re.search(r'\bis\s+today\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|a\s+weekday|a\s+weekend)\b', t)
    if m:
        asked = m.group(1)
        actual = _today.strftime('%A').lower()
        if 'weekday' in asked:
            is_wd = _today.weekday() < 5
            return f"Yes, today is a weekday — it's {_today.strftime('%A')}." if is_wd else f"No, today is {_today.strftime('%A')}, a weekend."
        if 'weekend' in asked:
            is_we = _today.weekday() >= 5
            return f"Yes, today is {_today.strftime('%A')}, a weekend." if is_we else f"No, today is {_today.strftime('%A')}, a weekday."
        return (f"Yes, today is {_today.strftime('%A')}."
                if actual == asked else
                f"No, today is {_today.strftime('%A')}.")

    return None
