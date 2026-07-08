"""
Spoken access announcement.

Design decision — pyttsx3 vs. browser speech
--------------------------------------------
The brief suggested pyttsx3. pyttsx3 drives the *server's* OS speech engine
(SAPI5/NSSpeechSynthesizer/espeak). In a Streamlit web app the server and the
user are different machines, so pyttsx3 would speak on the server and the user
would hear nothing — and it needs an audio device / extra system packages that
are absent on most deployment hosts (e.g. Streamlit Community Cloud).

The clean, dependency-free choice for a web app is the browser's built-in
Web Speech API, which runs client-side, offline, and needs no pip package.
That is what we use below. A pyttsx3 fallback for purely-local use is included
in comments for completeness.
"""
from __future__ import annotations

import html

from . import config
from .compliance import ComplianceResult


def _spoken(item_name: str) -> str:
    return config.PPE_REQUIREMENTS.get(item_name, {}).get("spoken", item_name.lower())


def _natural_join(items: list[str]) -> str:
    """['a'] -> 'a'; ['a','b'] -> 'a and b'; ['a','b','c'] -> 'a, b, and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_announcement(result: ComplianceResult) -> str:
    """Compose the spoken decision, matching the LabPPE workflow script.

    Granted:
        "Inspection complete. All required personal protective equipment
         detected. Access granted. You may enter the laboratory. Have a safe day."
    Denied (reads every required item, then the instruction):
        "Inspection complete. Lab coat detected. Gloves not detected. ...
         Access denied. Please wear your gloves before entering the laboratory."
    """
    parts = ["Inspection complete."]
    if result.access_granted:
        parts.append("All required personal protective equipment detected.")
        parts.append("Access granted. You may enter the laboratory. Have a safe day.")
    else:
        for item in result.checklist:
            if not item.required:
                continue
            state = "detected" if item.present else "not detected"
            parts.append(f"{_spoken(item.name).capitalize()} {state}.")
        parts.append("Access denied.")
        missing = _natural_join([_spoken(m) for m in result.missing])
        parts.append(f"Please wear your {missing} before entering the laboratory.")
    return " ".join(parts)


def speak_html(text: str) -> str:
    """Return an HTML snippet that speaks `text` via the browser once rendered."""
    safe = html.escape(text).replace("\n", " ")
    js_string = safe.replace("'", "\\'")
    return f"""
    <script>
      (function() {{
        try {{
          const msg = new SpeechSynthesisUtterance('{js_string}');
          msg.rate = 1.0; msg.pitch = 1.0;
          window.speechSynthesis.cancel();
          window.speechSynthesis.speak(msg);
        }} catch (e) {{ /* speech not supported */ }}
      }})();
    </script>
    <div style="font-size:0.8rem;color:#888;">🔊 {safe}</div>
    """


# --- Optional local-only alternative (uncomment to use on your own machine) ---
# import pyttsx3
# def speak_local(text: str) -> None:
#     engine = pyttsx3.init()
#     engine.say(text)
#     engine.runAndWait()
