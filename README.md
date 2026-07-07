# 🚨 Multi-Agent Disaster Response Dispatcher

**SDG 11 — Sustainable Cities and Communities**

## Overview
An agentic AI prototype that simulates emergency dispatch coordination during
climate-related disasters (floods, wildfires, etc.). Three specialized AI
agents work in sequence to turn a raw SOS/alert text into a complete,
resource-verified response plan.

## Architecture
1. **Damage Assessor** (Emergency Triage Specialist) — reads the incoming
   report, determines severity (Low/Medium/High), flags vulnerable groups
   (elderly, children, disabled), and notes confidence/uncertainty.
2. **Resource Allocator** (Logistics Coordinator) — matches critical needs
   against a live mock inventory (trucks, boats, shelters, medical teams,
   etc.), prioritizing vulnerable groups, and flags shortfalls.
3. **Communications Router** (Dispatch Command Writer) — drafts deployment
   instructions for rescue teams and a public safety broadcast.

Built with **CrewAI** (sequential process) and **Groq** (free LLM inference),
wrapped in a **Streamlit** UI.

## Features
- Real-time multi-agent dispatch pipeline
- Severity + vulnerable-group badges
- Session-persisted resource depletion across incidents
- Incident history log
- Session analytics (severity distribution, resource utilization)
- Map view of resource depots and incident location

## Setup
```bash
pip install -r requirements.txt
```
Create a `.env` file in the project folder with `GROQ_API_KEY=your_free_groq_key_here` (get a free key from https://console.groq.com/keys). Do not commit this file.

## Run
```bash
streamlit run app.py
```

## Limitations
- Resource inventory and coordinates are mocked, not connected to real
  municipal data.
- Incident history resets when the app restarts (session-only, no database).
- Single sequential LLM pipeline — no retry/fallback if an agent's output is
  malformed.
- Allocation validation uses simple regex matching, not full NLP parsing.

## Tech Stack
Python · CrewAI · Streamlit · Groq (Llama 3.3 70B)
