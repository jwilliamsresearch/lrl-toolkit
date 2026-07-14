"""lrl-toolkit web dashboard (Streamlit).

Four tabs over the same config-driven pipeline the CLI drives:
  1. Wizard        — pick language/base model/compute -> generate a project.yaml
  2. Run & monitor — run stages, view the manifest and each stage's card
  3. Review        — accept/edit/reject conversational pairs before SFT
  4. Chat          — load the trained model and talk to it

Run with: ``lrl dashboard`` (or ``streamlit run dashboard/app.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from lrl_toolkit import __version__
from lrl_toolkit.config import STAGE_ORDER, load_project
from lrl_toolkit.manifest import Manifest
from lrl_toolkit.pipeline import run_pipeline, run_single_stage
from lrl_toolkit.registry import list_compute, list_languages, list_models
from lrl_toolkit.scaffold import scaffold_yaml

PROJECTS_DIR = Path("projects")
TRANSLATE_BACKENDS = ["nllb", "m2m100", "opusmt", "madlad", "teacher", "mock"]
SYNTH_PROVIDERS = ["ollama", "local", "mock"]

st.set_page_config(page_title="lrl-toolkit", page_icon="🗣️", layout="wide")
st.title("🗣️ lrl-toolkit")
st.caption(f"Build custom LLMs for low-resource languages · v{__version__}")


def list_projects() -> list[Path]:
    return sorted(PROJECTS_DIR.glob("*.yaml")) if PROJECTS_DIR.exists() else []


def pick_project(key: str):
    projects = list_projects()
    if not projects:
        st.info("No projects yet — create one in the **Wizard** tab.")
        return None
    choice = st.selectbox("Project", projects, format_func=lambda p: p.stem, key=key)
    return load_project(choice)


wizard, monitor, review, chat = st.tabs(["Wizard", "Run & monitor", "Review", "Chat"])

# --------------------------------------------------------------------------- #
# 1. Wizard
# --------------------------------------------------------------------------- #
with wizard:
    st.subheader("New project")
    name = st.text_input("Project name", value="my-welsh-model")
    c1, c2, c3 = st.columns(3)
    with c1:
        language = st.selectbox("Language", list_languages() or ["(add a profile)"])
    with c2:
        base_model = st.selectbox("Base model", list_models() or ["(add a profile)"])
    with c3:
        compute = st.selectbox("Compute", list_compute() or ["(add a profile)"])

    c4, c5, c6 = st.columns(3)
    with c4:
        translate_backend = st.selectbox("Translate backend", TRANSLATE_BACKENDS)
    with c5:
        synth_provider = st.selectbox("Synth teacher", SYNTH_PROVIDERS)
    with c6:
        review_flag = st.checkbox("Human review before SFT", value=True)

    yaml_str = scaffold_yaml(
        name, language, base_model, compute,
        translate_backend=translate_backend, synth_provider=synth_provider, review=review_flag,
    )
    st.code(yaml_str, language="yaml")
    if st.button("💾 Save project", type="primary"):
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        out = PROJECTS_DIR / f"{name}.yaml"
        out.write_text(yaml_str, encoding="utf-8")
        st.success(f"Saved {out}")

# --------------------------------------------------------------------------- #
# 2. Run & monitor
# --------------------------------------------------------------------------- #
with monitor:
    st.subheader("Pipeline")
    project = pick_project("monitor_project")
    if project:
        manifest = Manifest.load(project.workdir, project.name)
        cols = st.columns(len(STAGE_ORDER))
        for col, stage in zip(cols, STAGE_ORDER, strict=True):
            done = stage in manifest.stages
            col.metric(stage, "✅" if done else "—")

        run_all = st.button("▶ Run full pipeline", type="primary")
        stage_to_run = st.selectbox("…or run a single stage", STAGE_ORDER)
        run_one = st.button(f"▶ Run '{stage_to_run}'")

        try:
            if run_all:
                with st.spinner("Running pipeline…"):
                    outcomes = run_pipeline(project)
                st.success("Done: " + ", ".join(f"{o.stage}({o.status})" for o in outcomes))
            elif run_one:
                with st.spinner(f"Running {stage_to_run}…"):
                    outcome = run_single_stage(project, stage_to_run)
                st.success(f"{outcome.stage}: {outcome.status}")
        except Exception as exc:  # surface stage errors (e.g. license gate)
            st.error(f"{type(exc).__name__}: {exc}")

        st.markdown("#### Stage cards")
        for stage in STAGE_ORDER:
            for card in project.stage_dir(stage).glob("*_card.json"):
                with st.expander(f"{stage} · {card.name}"):
                    st.json(json.loads(card.read_text(encoding="utf-8")))

# --------------------------------------------------------------------------- #
# 3. Review
# --------------------------------------------------------------------------- #
with review:
    st.subheader("Conversational-pair review")
    project = pick_project("review_project")
    if project:
        queue_path = project.stage_dir("convdata") / "review_queue.jsonl"
        if not queue_path.exists():
            st.info("No review queue yet — run the **convdata** stage first.")
        else:
            lines = queue_path.read_text("utf-8").splitlines()
            items = [json.loads(line) for line in lines if line]
            st.write(f"{len(items)} pairs")
            decisions = ["pending", "accepted", "rejected"]
            with st.form("review_form"):
                updated = []
                for item in items[:50]:  # page the first 50
                    msgs = {m["role"]: m["content"] for m in item["messages"]}
                    st.markdown(f"**#{item['id']}** · _{item.get('source','')}_")
                    instr = st.text_area("User", msgs.get("user", ""), key=f"u{item['id']}")
                    resp = st.text_area(
                        "Assistant", msgs.get("assistant", ""), key=f"a{item['id']}"
                    )
                    status = st.radio(
                        "Decision", decisions,
                        index=decisions.index(item.get("status", "pending")),
                        horizontal=True, key=f"s{item['id']}",
                    )
                    item["messages"] = [
                        {"role": "user", "content": instr},
                        {"role": "assistant", "content": resp},
                    ]
                    item["status"] = status
                    updated.append(item)
                    st.divider()
                if st.form_submit_button("💾 Save reviews", type="primary"):
                    with queue_path.open("w", encoding="utf-8") as fh:
                        for it in updated:
                            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
                    st.success(f"Saved {len(updated)} reviews. Re-run finetune to apply.")

# --------------------------------------------------------------------------- #
# 4. Chat
# --------------------------------------------------------------------------- #
with chat:
    st.subheader("Chat with your model")
    project = pick_project("chat_project")
    if project:
        if st.button("Load model"):
            from lrl_toolkit.modeling import load_model_and_tokenizer, resolve_artifacts

            merged = project.stage_dir("export") / "merged"
            with st.spinner("Loading model…"):
                if merged.exists():
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    tok = AutoTokenizer.from_pretrained(str(merged))
                    model = AutoModelForCausalLM.from_pretrained(str(merged))
                else:
                    tok_dir, adapter, _ = resolve_artifacts(project)
                    model, tok = load_model_and_tokenizer(
                        project.model_profile.hf_id, tok_dir, adapter, merge=True
                    )
            st.session_state["chat_model"] = (model, tok)
            st.success("Model loaded.")

        if "chat_model" in st.session_state:
            prompt = st.chat_input("Say something…")
            if prompt:
                model, tok = st.session_state["chat_model"]
                st.chat_message("user").write(prompt)
                messages = [{"role": "user", "content": prompt}]
                if tok.chat_template:
                    inputs = tok.apply_chat_template(
                        messages, add_generation_prompt=True, return_tensors="pt"
                    )
                else:
                    inputs = tok(prompt, return_tensors="pt").input_ids
                out = model.generate(inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
                reply = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
                st.chat_message("assistant").write(reply)
