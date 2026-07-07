"""
Multi-Agent Disaster Response Dispatcher
SDG 11 - Sustainable Cities and Communities
Built with CrewAI + Streamlit + Groq (free LLM API)
"""

import os
import re
import json
import datetime
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
import os
from dotenv import load_dotenv

# --- FIX for CrewAI + Groq "cache_breakpoint" bug (GitHub issue #5886) ---
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 0. HARDCODED API KEY  <-- PASTE YOUR FREE GROQ KEY HERE
# ---------------------------------------------------------------------------


load_dotenv()
# load_dotenv puts it into os.environ automatically if it's in your .env file!
MODEL_NAME = "groq/llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# 1. BASE MOCK RESOURCE DATABASE
# ---------------------------------------------------------------------------
BASE_RESOURCES = {
    "trucks":                 {"available": 6,   "location": "Central Depot",     "lat": 28.6139, "lon": 77.2090},
    "boats":                  {"available": 3,   "location": "Riverside Station", "lat": 28.6300, "lon": 77.2200},
    "shelters":               {"available": 4,   "location": "Various",           "lat": 28.6050, "lon": 77.1950},
    "medical_teams":          {"available": 5,   "location": "City Hospital",     "lat": 28.6200, "lon": 77.2100},
    "medical_supplies_kits":  {"available": 150, "location": "Central Depot",     "lat": 28.6139, "lon": 77.2090},
    "helicopters":            {"available": 1,   "location": "Airbase",           "lat": 28.6400, "lon": 77.1800},
    "drinking_water_pallets": {"available": 40,  "location": "Central Depot",     "lat": 28.6139, "lon": 77.2090},
}
INCIDENT_LOCATION = {"lat": 28.6180, "lon": 77.2050}

# ---------------------------------------------------------------------------
# 2. SESSION STATE INITIALIZATION (feature: resource depletion + history log)
# ---------------------------------------------------------------------------
if "resources" not in st.session_state:
    st.session_state.resources = json.loads(json.dumps(BASE_RESOURCES))  # deep copy

if "history" not in st.session_state:
    st.session_state.history = []  # list of dispatch records


def resources_as_text() -> str:
    return json.dumps(
        {k: {"available": v["available"], "location": v["location"]}
         for k, v in st.session_state.resources.items()},
        indent=2,
    )


# ---------------------------------------------------------------------------
# 3. LLM CONFIGURATION
# ---------------------------------------------------------------------------
def get_llm():
    return LLM(model=MODEL_NAME, temperature=0.3)


# ---------------------------------------------------------------------------
# 4. AGENT & TASK FACTORY
# ---------------------------------------------------------------------------
def build_crew(scenario_text: str, llm):

    damage_assessor = Agent(
        role="Emergency Triage Specialist",
        goal=(
            "Read incoming SOS texts or alert feeds, determine severity "
            "(Low, Medium, High), identify critical needs, flag vulnerable "
            "populations, and state confidence/uncertainty in the assessment."
        ),
        backstory=(
            "A veteran emergency triage officer who has worked through "
            "hurricanes, floods, and wildfires, skilled at turning chaotic "
            "reports into crisp, actionable assessments. Always honest about "
            "what is confirmed versus estimated."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    assess_task = Task(
        description=(
            f"Analyze this incoming disaster report:\n\n\"\"\"{scenario_text}\"\"\"\n\n"
            "Produce a structured triage report with these exact labeled sections:\n"
            "1. A line starting EXACTLY with 'SEVERITY: High' or 'SEVERITY: Medium' "
            "or 'SEVERITY: Low', followed by a one-line justification\n"
            "2. 'AFFECTED: <number>' - estimated number of people affected\n"
            "3. 'VULNERABLE GROUPS:' - a bulleted list of any vulnerable groups "
            "mentioned or implied (elderly, children, disabled, hospitalized, "
            "infants). Write 'None specifically identified' if none.\n"
            "4. 'CRITICAL NEEDS:' - a bulleted list of needs (evacuation, medical "
            "aid, water, shelter, etc.)\n"
            "5. 'HAZARDS:' - hazards responders should be aware of\n"
            "6. 'CONFIDENCE NOTE:' - one sentence stating which numbers/facts are "
            "confirmed versus your best estimate given incomplete information"
        ),
        expected_output=(
            "A structured triage report with SEVERITY, AFFECTED, VULNERABLE GROUPS, "
            "CRITICAL NEEDS, HAZARDS, and CONFIDENCE NOTE sections."
        ),
        agent=damage_assessor,
    )

    resource_allocator = Agent(
        role="Logistics Coordinator",
        goal=(
            "Match critical needs against the available emergency resource "
            "inventory, prioritizing vulnerable populations first when supply "
            "is limited, and produce a realistic allocation plan."
        ),
        backstory=(
            "Manages the city's emergency logistics warehouse and fleet. "
            "Never over-allocates beyond stock, always flags shortfalls, and "
            "follows the principle that vulnerable groups (elderly, children, "
            "disabled, hospitalized) get first priority when resources are scarce."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    allocate_task = Task(
        description=(
            "Using the triage report above (including any VULNERABLE GROUPS "
            f"flagged) and this live inventory (JSON):\n\n{resources_as_text()}\n\n"
            "Create an allocation plan that:\n"
            "1. Assigns specific quantities of resources to each need, in this "
            "exact format wherever a quantity is assigned: '<N> <resource_name>' "
            "e.g. '2 trucks', '1 boats', '3 medical_teams'\n"
            "2. Never allocates more than what's available\n"
            "3. If vulnerable groups were flagged, explicitly states they are "
            "prioritized first for shelters and medical teams\n"
            "4. Flags shortfalls explicitly with a line starting 'SHORTFALL:'\n"
            "5. States dispatch origin/location for each resource"
        ),
        expected_output=(
            "An allocation list mapping needs to specific resource quantities "
            "(in '<N> <resource_name>' format) and locations, with shortfalls flagged."
        ),
        agent=resource_allocator,
        context=[assess_task],
    )

    comms_router = Agent(
        role="Dispatch Command Writer",
        goal=(
            "Turn the triage report and allocation plan into deployment "
            "instructions for rescue teams and a public safety broadcast."
        ),
        backstory=(
            "The voice of the emergency operations center — clear, calm, "
            "unambiguous communication that teams and the public can act on."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    comms_task = Task(
        description=(
            "Using the triage report and allocation plan, produce:\n\n"
            "1. 'DEPLOYMENT INSTRUCTIONS' - numbered, action-oriented steps "
            "for each team: where to go, what to bring, objective.\n\n"
            "2. 'PUBLIC SAFETY BROADCAST' - under 100 words, calm and clear, "
            "with immediate safety actions and where to get help. If vulnerable "
            "groups were flagged, include specific guidance for them."
        ),
        expected_output="Two labeled sections: 'DEPLOYMENT INSTRUCTIONS' and 'PUBLIC SAFETY BROADCAST'.",
        agent=comms_router,
        context=[assess_task, allocate_task],
    )

    crew = Crew(
        agents=[damage_assessor, resource_allocator, comms_router],
        tasks=[assess_task, allocate_task, comms_task],
        process=Process.sequential,
        verbose=False,
    )
    return crew, assess_task, allocate_task, comms_task


# ---------------------------------------------------------------------------
# 5. HELPERS
# ---------------------------------------------------------------------------
def extract_field(text: str, label: str) -> str:
    match = re.search(rf"{label}:\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else "Not specified"


def extract_severity(text: str) -> str:
    match = re.search(r"SEVERITY:\s*(High|Medium|Low)", text, re.IGNORECASE)
    return match.group(1).capitalize() if match else "Unknown"


def severity_badge(level: str):
    colors = {"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#2ecc71", "Unknown": "#888888"}
    color = colors.get(level, "#888888")
    st.markdown(
        f"""<div style="background-color:{color};color:white;padding:8px 16px;
        border-radius:8px;display:inline-block;font-weight:bold;font-size:18px;">
        SEVERITY: {level.upper()}</div>""",
        unsafe_allow_html=True,
    )


def apply_depletion(allocation_text: str, resources: dict) -> list:
    """
    Parses '<N> <resource_name>' patterns from the allocation plan and
    deducts them from session-state resources. Returns a list of warnings
    for any resource that was requested beyond what's available.
    """
    warnings = []
    normalized_text = allocation_text.replace("_", " ")
    for name, info in resources.items():
        pretty_name = name.replace("_", " ")
        pattern = re.compile(rf"(\d+)\s*{re.escape(pretty_name)}", re.IGNORECASE)
        total_requested = sum(int(m) for m in pattern.findall(normalized_text))
        if total_requested == 0:
            continue
        if total_requested > info["available"]:
            warnings.append(
                f"⚠️ Requested {total_requested} {pretty_name}, but only "
                f"{info['available']} were available. Allocating all remaining stock."
            )
            info["available"] = 0
        else:
            info["available"] -= total_requested
    return warnings


# ---------------------------------------------------------------------------
# 6. STREAMLIT UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Disaster Response Dispatcher", page_icon="🚨", layout="wide")
st.title("🚨 Multi-Agent Disaster Response Dispatcher")
st.caption("SDG 11 · Sustainable Cities and Communities — Agentic AI Prototype")

# --- Sidebar: live resources + history + analytics ---
with st.sidebar:
    st.header("📦 Live Emergency Resources")
    for name, info in st.session_state.resources.items():
        base_amt = BASE_RESOURCES[name]["available"]
        st.metric(
            label=name.replace("_", " ").title(),
            value=info["available"],
            delta=info["available"] - base_amt if info["available"] != base_amt else None,
        )
    if st.button("🔄 Reset Resources to Full Stock", use_container_width=True):
        st.session_state.resources = json.loads(json.dumps(BASE_RESOURCES))
        st.rerun()

    st.divider()
    st.subheader("📜 Past Incidents")
    if not st.session_state.history:
        st.caption("No incidents dispatched yet this session.")
    else:
        for record in reversed(st.session_state.history):
            with st.expander(f"{record['time']} — {record['severity']}"):
                st.write(record["scenario"][:150] + ("..." if len(record["scenario"]) > 150 else ""))

    st.divider()
    st.subheader("📊 Session Analytics")
    if st.session_state.history:
        sev_counts = {"High": 0, "Medium": 0, "Low": 0, "Unknown": 0}
        for r in st.session_state.history:
            sev_counts[r["severity"]] = sev_counts.get(r["severity"], 0) + 1
        st.bar_chart(sev_counts)

        total_incidents = len(st.session_state.history)
        st.metric("Total Incidents Handled", total_incidents)

        utilization = {}
        for name in BASE_RESOURCES:
            base = BASE_RESOURCES[name]["available"]
            current = st.session_state.resources[name]["available"]
            used_pct = round(((base - current) / base) * 100) if base > 0 else 0
            utilization[name.replace("_", " ").title()] = used_pct
        st.caption("Resource Utilization (%)")
        st.bar_chart(utilization)
    else:
        st.caption("Dispatch an incident to see analytics.")

# --- Main input ---
st.subheader("📨 Incoming Disaster Report")
sample = (
    "URGENT: Heavy monsoon rainfall has caused the river near Sector 7 to "
    "overflow. Water levels rising rapidly. Around 300 residents trapped in "
    "low-lying apartment blocks, several elderly and children among them. "
    "Two people reported with injuries. Main road access is flooded, only "
    "boat access possible. Power lines are down near the school."
)
scenario_text = st.text_area("Paste an SOS text or scenario description:", value=sample, height=150)
run_button = st.button("🚀 Dispatch Agents", type="primary", use_container_width=True)

if run_button:
    if not scenario_text.strip():
        st.error("Please enter a disaster scenario description.")
    elif not os.getenv("GROQ_API_KEY"):
        st.error("You still need to paste your free Groq API key into the GROQ_API_KEY variable at the top of app.py.")
    else:
        llm = get_llm()
        crew, assess_task, allocate_task, comms_task = build_crew(scenario_text, llm)

        try:
            with st.spinner("Agents are coordinating the response..."):
                crew.kickoff()
        except Exception as e:
            err = str(e).lower()
            if "quota" in err or "rate limit" in err or "429" in err:
                st.error(
                    "⚠️ The LLM provider says you've hit a rate limit or quota. "
                    "If using Groq's free tier, wait a minute and try again, "
                    "or check https://console.groq.com/settings/limits"
                )
            else:
                st.error(f"⚠️ Something went wrong calling the LLM: {e}")
            st.stop()

        assess_out = assess_task.output.raw if assess_task.output else ""
        allocate_out = allocate_task.output.raw if allocate_task.output else ""
        comms_out = comms_task.output.raw if comms_task.output else ""

        st.success("✅ Dispatch plan complete!")

        severity = extract_severity(assess_out)
        vulnerable = extract_field(assess_out, "VULNERABLE GROUPS")
        confidence_note = extract_field(assess_out, "CONFIDENCE NOTE")

        # --- Badges row ---
        badge_col1, badge_col2 = st.columns([1, 2])
        with badge_col1:
            severity_badge(severity)
        with badge_col2:
            if vulnerable and "none" not in vulnerable.lower():
                st.markdown(
                    f"""<div style="background-color:#5b6ee1;color:white;padding:8px 16px;
                    border-radius:8px;display:inline-block;font-weight:bold;">
                    🛡️ VULNERABLE GROUPS FLAGGED: {vulnerable}</div>""",
                    unsafe_allow_html=True,
                )

        if confidence_note and confidence_note != "Not specified":
            st.info(f"🔎 **Confidence Note:** {confidence_note}")

        # --- Apply resource depletion (feature: session-persisted stock) ---
        depletion_warnings = apply_depletion(allocate_out, st.session_state.resources)
        if depletion_warnings:
            st.warning("**Stock Warnings:**\n\n" + "\n".join(depletion_warnings))

        # --- Log to history ---
        st.session_state.history.append({
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "severity": severity,
            "scenario": scenario_text,
        })

        st.write("")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🧑‍⚕️ Damage Assessor")
            st.markdown(assess_out)
        with col2:
            st.markdown("### 🚚 Resource Allocator")
            st.markdown(allocate_out)
        with col3:
            st.markdown("### 📢 Comms Router")
            st.markdown(comms_out)
        st.divider()
        st.subheader("🗺️ Resource & Incident Map")
        import pandas as pd
        map_df = pd.DataFrame(
            [{"lat": v["lat"], "lon": v["lon"]} for v in st.session_state.resources.values()]
            + [{"lat": INCIDENT_LOCATION["lat"], "lon": INCIDENT_LOCATION["lon"]}]
        )
        st.map(map_df, size=50)
        st.caption("Dots = resource depots/stations; last point = incident location (mock coordinates).")