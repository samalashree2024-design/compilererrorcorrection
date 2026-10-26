import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import streamlit as st

st.set_page_config(page_title="Cross-Compiler Error Triangulation", page_icon="🧭", layout="wide")

DEFAULT_CODE = '''#include <stdio.h>

int main(void) {
    int number = 10
    printf("%d\\n", number);
    return 0;
}
'''
GCC_DEFAULT = r"C:\MinGW\bin\gcc.exe"
CLANG_DEFAULT = r"C:\Program Files\LLVM\bin\clang.exe"
CLANG_TARGET_FLAGS = ["--target=i686-w64-windows-gnu", "-isystem", r"C:\MinGW\include"]


@dataclass
class Diagnostic:
    compiler: str
    severity: str
    message: str
    line: Optional[int]
    column: Optional[int]
    category: str
    raw: str


def find_compiler(configured_path: str, executable_name: str) -> Optional[str]:
    path = (configured_path or "").strip().strip('"')
    if path and os.path.isfile(path):
        return path
    return shutil.which(executable_name)


def classify_message(message: str) -> str:
    m = (message or "").lower()

    # Specific semicolon messages must be checked before generic declaration errors.
    if any(s in m for s in (
        "expected ',' or ';' before",
        "expected ';' at end of declaration",
        "expected ';' after",
        "expected ';'",
        "expected ‘;’",
        "expected semicolon",
        "missing ';'",
    )):
        return "missing_semicolon"
    if "unused variable" in m or "unused-but-set variable" in m:
        return "unused_variable"
    if any(s in m for s in ("use of undeclared identifier", "undeclared", "not declared", "unknown identifier")):
        return "undeclared_variable"
    if "no such file or directory" in m or "file not found" in m:
        return "missing_header"
    if "redefinition" in m or "already defined" in m:
        return "redefinition"
    if "expected '}'" in m or "unterminated" in m:
        return "unclosed_construct"
    if "conflicting types" in m:
        return "type_error"
    if "expected expression" in m or "expected declaration" in m or "syntax error" in m:
        return "syntax_error"
    return "other"


def parse_diagnostics(compiler: str, output: str) -> list[Diagnostic]:
    pattern = re.compile(
        r"^(?P<file>.*?):(?P<line>\d+)(?::(?P<column>\d+))?:\s*"
        r"(?P<severity>fatal error|error|warning|note):\s*(?P<message>.*)$",
        re.IGNORECASE,
    )
    found = []
    for raw in (output or "").splitlines():
        match = pattern.match(raw.strip())
        if not match:
            continue
        message = match.group("message").strip()
        found.append(Diagnostic(
            compiler=compiler,
            severity=match.group("severity").lower(),
            message=message,
            line=int(match.group("line")),
            column=int(match.group("column")) if match.group("column") else None,
            category=classify_message(message),
            raw=raw,
        ))
    return found


def run_compiler(name: str, path: Optional[str], code: str) -> dict:
    if not path:
        return {"compiler": name, "available": False, "success": False, "returncode": None,
                "output": f"{name} executable not found. Check its path in the sidebar.",
                "diagnostics": [], "elapsed_ms": 0}
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="cross_compiler_") as temp_dir:
        source = os.path.join(temp_dir, "snippet.c")
        with open(source, "w", encoding="utf-8", newline="\n") as f:
            f.write(code)
        if name == "GCC":
            command = [path, "-x", "c", "-fsyntax-only", "-Wall", "-Wextra", source]
        else:
            command = [path, *CLANG_TARGET_FLAGS, "-x", "c", "-fsyntax-only", "-Wall", "-Wextra", source]
        try:
            result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=12, check=False)
            output = "\n".join(x for x in (result.stdout, result.stderr) if x).strip()
            return {
                "compiler": name, "available": True, "success": result.returncode == 0,
                "returncode": result.returncode,
                "output": output or ("No diagnostics." if result.returncode == 0 else "Compiler exited without diagnostic text."),
                "diagnostics": parse_diagnostics(name, output),
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
        except subprocess.TimeoutExpired:
            return {"compiler": name, "available": True, "success": False, "returncode": None,
                    "output": "Compilation timed out after 12 seconds.", "diagnostics": [],
                    "elapsed_ms": round((time.perf_counter() - started) * 1000)}
        except OSError as exc:
            return {"compiler": name, "available": False, "success": False, "returncode": None,
                    "output": f"Could not run compiler: {exc}", "diagnostics": [],
                    "elapsed_ms": round((time.perf_counter() - started) * 1000)}


def compile_both(code: str, gcc_path: Optional[str], clang_path: Optional[str]) -> tuple[dict, dict]:
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(run_compiler, "GCC", gcc_path, code)
        b = pool.submit(run_compiler, "Clang", clang_path, code)
        return a.result(), b.result()

def compile_selected(code: str, gcc_path: Optional[str], clang_path: Optional[str], mode: str):
    if mode == "GCC only":
        return run_compiler("GCC", gcc_path, code), None
    elif mode == "Clang only":
        return None, run_compiler("Clang", clang_path, code)
    else:
        return compile_both(code, gcc_path, clang_path)


def diagnostic_matches(a: Diagnostic, b: Diagnostic) -> bool:
    if a.category != b.category:
        return False
    return a.line is None or b.line is None or abs(a.line - b.line) <= 2


def triangulate(gcc: dict, clang: dict) -> tuple[list[dict], list[dict]]:
    gd = [d for d in gcc["diagnostics"] if d.severity in ("error", "fatal error")]
    cd = [d for d in clang["diagnostics"] if d.severity in ("error", "fatal error")]
    used_clang = set()
    findings = []
    cascades = []
    priority = {"missing_semicolon": 0, "unclosed_construct": 1, "missing_header": 2}
    gd.sort(key=lambda d: (priority.get(d.category, 10), d.line or 0))

    for g in gd:
        best, best_distance = None, 999
        for i, c in enumerate(cd):
            if i in used_clang or not diagnostic_matches(g, c):
                continue
            distance = abs((g.line or 0) - (c.line or 0))
            if distance < best_distance:
                best, best_distance = i, distance
        if best is not None:
            c = cd[best]
            used_clang.add(best)
            score = 95 if best_distance == 0 else 88 if best_distance == 1 else 80
            findings.append({"category": g.category, "gcc": g, "clang": c, "confidence": score,
                             "agreement": "Both compilers agree on the normalized error category."})
        else:
            findings.append({"category": g.category, "gcc": g, "clang": None, "confidence": 35,
                             "agreement": "GCC reports this error, but no matching Clang error was found."})

    for i, c in enumerate(cd):
        if i not in used_clang:
            findings.append({"category": c.category, "gcc": None, "clang": c, "confidence": 35,
                             "agreement": "Clang reports this error, but no matching GCC error was found."})

    roots = [f for f in findings if f["category"] in ("missing_semicolon", "syntax_error", "unclosed_construct")
             and f["gcc"] is not None and f["clang"] is not None]
    for result in (gcc, clang):
        for d in result["diagnostics"]:
            if d.severity != "warning" and d.category != "unused_variable":
                continue
            near = any(
                line is not None and d.line is not None and abs(d.line - line) <= 2
                for f in roots for line in (
                    f["gcc"].line if f["gcc"] else None,
                    f["clang"].line if f["clang"] else None,
                )
            )
            if near:
                cascades.append({"compiler": d.compiler, "line": d.line, "message": d.message,
                                 "reason": "Possible follow-on diagnostic near a shared syntax error; shown separately."})
    findings.sort(key=lambda f: min([d.line for d in (f["gcc"], f["clang"]) if d and d.line] or [999999]))
    return findings, cascades


def propose_semicolon_fix(code: str, findings: list[dict]) -> Optional[str]:
    if not any(f["category"] == "missing_semicolon" for f in findings):
        return None
    lines = code.splitlines()
    candidates = set()
    for finding in findings:
        if finding["category"] != "missing_semicolon":
            continue
        for diag in (finding.get("gcc"), finding.get("clang")):
            if not diag or not diag.line:
                continue
            if "before" in diag.message.lower() and diag.line > 1:
                candidates.add(diag.line - 1)
            else:
                candidates.add(diag.line)

    for n in sorted(candidates):
        if not 1 <= n <= len(lines):
            continue
        original = lines[n - 1]
        s = original.rstrip()
        if not s or s.lstrip().startswith(("#", "//", "/*", "*")):
            continue
        if s.endswith((";", "{", "}", ":", ",", "\\")):
            continue
        if re.match(r"^\s*(if|for|while|switch|else|do)\b.*\)\s*$", s):
            continue
        if re.match(r"^\s*(int|char|short|long|float|double|size_t|unsigned|signed|void|return)\b", s) or "=" in s or re.search(r"\b(printf|scanf|puts|putchar|fprintf|fputs)\s*\(", s):
            changed = lines.copy()
            changed[n - 1] = s + ";"
            return "\n".join(changed) + ("\n" if code.endswith("\n") else "")
    return None


def explain(category: str) -> str:
    messages = {
        "missing_semicolon": "A statement appears to be missing `;`. GCC may point at the next token while Clang points at the declaration end.",
        "unused_variable": "A variable is not used. Near a syntax error, this may be a follow-on symptom rather than the root cause.",
        "undeclared_variable": "A name is used that the compiler cannot find in the current scope.",
        "missing_header": "A header file could not be found. Check the include name and compiler include paths.",
        "redefinition": "A name appears to be defined more than once where that is not allowed.",
        "unclosed_construct": "A brace or another construct may be left unclosed.",
        "type_error": "The compiler found conflicting or incompatible type information.",
        "syntax_error": "The compiler could not parse this part of the code. Check nearby punctuation and statement structure.",
        "other": "This diagnostic did not match a known local explanation rule.",
    }
    return messages.get(category, messages["other"])


def diag_text(d: Optional[Diagnostic]) -> str:
    return "No matching diagnostic." if d is None else f"Line {d.line or '?'}: {d.message}"


st.title("🧭 Real-Time Compiler Error Correction")
st.caption("Cross-Compiler Triangulation (GCC vs. Clang) • Confidence-Scored Diagnostics")

with st.sidebar:
    st.header("Compiler settings")
    gcc_config = st.text_input("GCC executable", value=GCC_DEFAULT)
    clang_config = st.text_input("Clang executable", value=CLANG_DEFAULT)
    st.caption("Clang uses the MinGW target and include path for this Windows setup.")
    st.divider()
    st.caption("Scores are transparent heuristic agreement scores, not statistically calibrated probabilities.")

gcc_path = find_compiler(gcc_config, "gcc")
clang_path = find_compiler(clang_config, "clang")
a, b = st.columns(2)
with a:
    st.write("**GCC**")
    st.success(gcc_path if gcc_path else "Not found — check sidebar path.")
with b:
    st.write("**Clang**")
    st.success(clang_path if clang_path else "Not found — check sidebar path.")

sample = st.selectbox("Load a demonstration example", [
    "Current code", "Missing semicolon (recommended demo)", "Correct C program",
    "Undeclared variable", "Missing closing brace","Multiple errors in one program","Type mismatch"
])
samples = {
    "Missing semicolon (recommended demo)": DEFAULT_CODE,
    "Correct C program": '#include <stdio.h>\n\nint main(void) {\n    int number = 10;\n    printf("%d\\n", number);\n    return 0;\n}\n',
    "Undeclared variable": '#include <stdio.h>\n\nint main(void) {\n    printf("%d\\n", total);\n    return 0;\n}\n',
    "Missing closing brace": '#include <stdio.h>\n\nint main(void) {\n    puts("hello");\n    return 0;\n',
    "Multiple errors in one program": 'int main(void) { first = 10; second = 20; return 0; }',
    "Type mismatch":'int main(void) { int *ptr = "hello"; return 0; }'
}
if "code_buffer" not in st.session_state:
    st.session_state.code_buffer = DEFAULT_CODE
if sample != "Current code":
    st.session_state.code_buffer = samples[sample]
code = st.text_area("C code", key="code_buffer", height=310)



analysis_mode = st.radio(
    "Choose compiler",
    ["GCC only", "Clang only", "GCC + Clang"],
    horizontal=True,
    key="analysis_mode"
)
button_label = {
    "GCC only": "Analyze with GCC",
    "Clang only": "Analyze with Clang",
    "GCC + Clang": "Analyze with GCC + Clang",
}

analyze = st.button(
    button_label[analysis_mode],
    type="primary"
)


if analyze:
    if not code.strip():
        st.warning("Enter some C code first.")
    else:
        with st.spinner(f"Running {analysis_mode}..."):
            gcc_result, clang_result = compile_selected(
                code, gcc_path, clang_path, analysis_mode
            )

        findings, cascades = [], []

        if analysis_mode == "GCC + Clang":
            findings, cascades = triangulate(gcc_result, clang_result)

        # Show whichever compiler results are available
        st.subheader("Compiler results")

        results = [
            result
            for result in (gcc_result, clang_result)
            if result is not None
        ]

        columns = st.columns(len(results))

        for col, result in zip(columns, results):
            with col:
                st.markdown(f"**{result['compiler']}**")

                if not result["available"]:
                    st.error("Unavailable")
                elif result["success"]:
                    st.success(
                        f"Compilation passed • {result['elapsed_ms']} ms"
                    )
                else:
                    st.error(
                        f"Compilation failed • {result['elapsed_ms']} ms"
                    )

                with st.expander(
                    f"{result['compiler']} raw output",
                    expanded=not result["success"]
                ):
                    st.code(
                        result["output"] or "No output",
                        language="text"
                    )

        # Compare and match diagnostics only when both compilers run
        if analysis_mode == "GCC + Clang":
            st.subheader("Triangulated findings")

            if not findings:
                if gcc_result["success"] and clang_result["success"]:
                    st.success(
                        "Both compilers accepted the code. "
                        "No errors to triangulate."
                    )
                else:
                    st.info(
                        "No parseable error diagnostics found. "
                        "Check the raw output above."
                    )

            for i, finding in enumerate(findings, start=1):
                category = finding["category"]

                with st.container(border=True):
                    st.markdown(
                        f"**Finding {i}: "
                        f"{category.replace('_', ' ').title()}**"
                    )

                    st.progress(
                        finding["confidence"] / 100,
                        text=(
                            "Agreement confidence: "
                            f"{finding['confidence']}%"
                        )
                    )

                    st.write(explain(category))

                    x, y = st.columns(2)

                    with x:
                        st.caption("GCC")
                        st.write(diag_text(finding["gcc"]))

                    with y:
                        st.caption("Clang")
                        st.write(diag_text(finding["clang"]))

                    st.caption(finding["agreement"])

            if cascades:
                st.subheader("Possible cascading diagnostics")
                st.info(
                    "These messages are near a shared syntax error, "
                    "so they may be follow-on noise. They are shown "
                    "separately, not treated as independent root causes."
                )

                for item in cascades:
                    st.write(
                        f"**{item['compiler']} — "
                        f"line {item['line'] or '?'}:** "
                        f"{item['message']}"
                    )
                    st.caption(item["reason"])

            # Suggest and verify fixes using BOTH compilers
            candidate = propose_semicolon_fix(code, findings)

            st.subheader("Suggested fix and verification")

            if candidate and candidate != code:
                st.code(candidate, language="c")

                with st.spinner(
                    "Verifying the suggested fix with both compilers..."
                ):
                    fixed_gcc, fixed_clang = compile_both(
                        candidate, gcc_path, clang_path
                    )

                v1, v2 = st.columns(2)

                with v1:
                    st.write("**GCC verification**")
                    if not fixed_gcc["available"]:
                        st.error("Unavailable")
                    else:
                        st.success(
                            "Passed" if fixed_gcc["success"] else "Failed"
                        )

                with v2:
                    st.write("**Clang verification**")
                    if not fixed_clang["available"]:
                        st.error("Unavailable")
                    else:
                        st.success(
                            "Passed" if fixed_clang["success"] else "Failed"
                        )

                if (
                    fixed_gcc["available"]
                    and fixed_clang["available"]
                    and fixed_gcc["success"]
                    and fixed_clang["success"]
                ):
                    st.success(
                        "Verified: both compilers accept the suggested code."
                    )

                    st.download_button(
                        "Download corrected C file",
                        data=candidate,
                        file_name="corrected_snippet.c",
                        mime="text/x-c"
                    )
                else:
                    st.warning(
                        "The candidate was not accepted by both compilers, "
                        "so it is not marked verified."
                    )

                    with st.expander("Verification output"):
                        st.code(
                            "GCC:\n"
                            + fixed_gcc["output"]
                            + "\n\nClang:\n"
                            + fixed_clang["output"]
                        )

            elif gcc_result["success"] and clang_result["success"]:
                st.success(
                    "No correction needed: both compilers accepted "
                    "the current code."
                )
            else:
                st.info(
                    "No safe automatic fix was generated for this "
                    "diagnostic. An unverified suggestion is never "
                    "labelled correct."
                )

            st.caption(
                "Pipeline: run GCC and Clang concurrently → normalize "
                "and match diagnostics → score agreement → separate "
                "possible cascade noise → verify a conservative "
                "candidate with both compilers."
            )

        else:
            st.info(
                f"{analysis_mode} mode: showing that compiler's results "
                "only. Diagnostic matching, agreement confidence, and "
                "two-compiler fix verification are available in "
                "GCC + Clang mode."
            )

else:
    st.info(
        "Choose a sample or edit the code, then click Analyze."
    )
