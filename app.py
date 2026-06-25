"""Scenario Swarm — multi-agent business simulation (Streamlit)."""

from __future__ import annotations

import datetime
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI

MODEL_ID = "gpt-4o-mini"
LABEL_DASH = " —"
POSITION_LABEL = "POSITION / FOCUSING POINT"

DEFAULT_AGENT_TEMPERATURE = 0.45

MAX_TOOL_URLS_PER_AGENT = 3
MAX_TOOL_CALLS_PER_TURN = 3
WEB_SEARCH_TOOL_NAME = "web_search"
MIN_EXTRACTED_CHARS = 300
MAX_CHARS_PER_ARTICLE = 5000
MAX_PROMPT_CONTEXT_CHARS = 2500
MAX_WEB_SEARCH_EXCERPT_CHARS = 280
ARTICLE_REQUEST_TIMEOUT = 12
PLAYWRIGHT_TIMEOUT_SEC = 30
WAYBACK_TIMEOUT_SEC = 20
WAYBACK_AVAILABILITY_API = (
    "https://archive.org/wayback/available?url={url}&timestamp={ts}"
)
WEB_FETCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
PLAYWRIGHT_STEALTH_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
if (window.chrome) {
    window.chrome.runtime = {connect: function() {}, sendMessage: function() {}};
}
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
"""
PLAYWRIGHT_LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--lang=en-US",
]
BLOCKED_PAGE_TITLES = {
    "access denied",
    "forbidden",
    "403 forbidden",
    "403",
    "blocked",
    "attention required",
    "just a moment",
    "checking your browser",
    "verify you are human",
}
BOT_BLOCK_HTTP_CODES = frozenset({401, 403, 429, 503})
BLOCKED_FETCH_NOTICE = (
    "Normal fetch is blocked. Crawling in silently — crossing the bots…"
)
WAYBACK_FETCH_NOTICE = (
    "Still blocked. Digging deeper for a readable copy of the source…"
)
WEB_SEARCH_RECOVERED_NOTICE = (
    "Normal fetch was blocked. We crawled in silently, crossed the bots, and recovered the source."
)

DEBATE_AGENT_SYSTEM = (
    "You are simulating one stakeholder in a business-school classroom debate. "
    "Follow the requested output format exactly. "
    "When web_search is available, call it only when you need specific evidence, facts, "
    "or examples from one of the agent's registered URLs. "
    "Do not call tools if your position and the debate history are enough. "
    "Do not invent facts beyond your position, the scenario, debate history, "
    "and text returned by web_search."
)

REASONING_STYLE_DEFINITIONS = {
    "Balanced": "Weighs both sides and keeps the argument measured.",
    "Analytical": "Uses structured logic, tradeoffs, and evidence-oriented reasoning.",
    "Skeptical": "Questions assumptions, exposes risks, and pushes for proof.",
    "Empathetic": "Focuses on people, trust, adoption, emotions, and lived experience.",
    "Practical": "Focuses on feasibility, implementation, operations, and constraints.",
    "Contrarian": "Challenges the dominant view and introduces an opposing frame.",
}

REASONING_STYLE_BEHAVIORS = {
    "Balanced": "Weigh both sides and acknowledge tradeoffs.",
    "Analytical": "Use structured logic, tradeoffs, and evidence-oriented reasoning.",
    "Skeptical": "Question assumptions, expose risks, and push for proof.",
    "Empathetic": "Focus on people, trust, adoption, emotions, and lived experience.",
    "Practical": "Focus on feasibility, implementation, operations, and constraints.",
    "Contrarian": "Challenge the dominant view and introduce an opposing frame.",
}

REASONING_STYLES = list(REASONING_STYLE_DEFINITIONS.keys())

DEFAULT_GENERAL_DEBATE_RULES = (
    "Agents must stay in character, disagree meaningfully, avoid final verdicts, "
    "respond to the strongest points made by others, never repeat their own prior "
    "arguments verbatim, call out when another participant recycles the same point "
    "and name the round number where that point was first raised, "
    "and if repetition was already challenged in an earlier round, reference that "
    "prior call-out instead of treating it as a new issue."
)

BUSINESS_IDEAS = {
    "AI-Written Exams and Assessment Policy": {
        "description": (
            "A business school discovers that students can use AI tools to draft exam answers, "
            "case analyses, and take-home submissions. The school is considering whether to ban AI, "
            "allow AI with disclosure, redesign assessments, or create AI-integrated exams."
        ),
        "category": "Academic Policy / Assessment Design / AI Ethics",
        "core_question": (
            "Should the institution treat AI-written exam work as misconduct, a new skill to be assessed, "
            "or a signal that traditional assessment design must change?"
        ),
    },
    "Campus Food Court Queue Optimization": {
        "description": (
            "A university food court introduces a pre-ordering and smart queue management system "
            "to reduce lunch-hour waiting time, improve vendor operations, and make the student "
            "dining experience faster and more predictable."
        ),
        "category": "Campus Operations / Food Services",
        "core_question": (
            "Will students, food vendors, and campus administrators align around a digital "
            "queue system, or will convenience, fairness, and adoption concerns clash?"
        ),
    },
    "University Student Wellbeing Companion": {
        "description": (
            "A university introduces a private AI wellbeing companion that helps students reflect, "
            "manage stress, discover campus support resources, and build healthier study-life routines."
        ),
        "category": "Student Experience / Wellbeing / AI Services",
        "core_question": (
            "Will students see an AI wellbeing companion as useful and safe, or will concerns "
            "around privacy, trust, and emotional dependence limit adoption?"
        ),
    },
}

SIMULATION_STEPS = [
    ("setup", "Scenario Setting", "Choose or create the scenario to be debated."),
    ("agents", "Agent Design", "Create stakeholder agents and define their positions."),
    ("analysis", "Scenario Analysis", "Selected agents debate automatically across up to four rounds."),
    ("swot", "Final Report", "Moderator synthesizes agent arguments into a final classroom report."),
]

STEP_LABELS = {step_id: title for step_id, title, _ in SIMULATION_STEPS}
STEP_DESCRIPTIONS = {step_id: desc for step_id, _, desc in SIMULATION_STEPS}
STEP_ORDER = [step_id for step_id, _, _ in SIMULATION_STEPS]

MIN_AGENTS_TO_SAVE = 2
MAX_AGENTS = 3
MAX_DEBATE_ROUNDS = 4
MIN_DEBATE_ROUNDS = 1
STREAM_CHAR_DELAY_SEC = 0.020

BITSoM_LOGO_URL = (
    "https://www.bitsom.edu.in/wp-content/uploads/2023/04/zero_scroll_logo-icn-1.svg"
)

CUSTOM_SCENARIO_DEFINITION = (
    "Set a Scenario Title, choose a Broad business category, and define the "
    "Core theme you want stakeholders to debate."
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --navy: #0B1F3A;
  --navy-mid: #162F56;
  --navy-light: #EAF0F8;
  --navy-subtle: #F2F5FA;
  --gold: #C9973A;
  --gold-light: #FDF4E3;
  --gold-dark: #9B7229;
  --success: #1B7A47;
  --success-bg: #E6F4ED;
  --text: #0B1F3A;
  --text-muted: #5A6880;
  --text-hint: #8A97AA;
  --border: #D7E1EE;
  --border-soft: #EDF1F7;
  --bg: #F2F5FA;
  --card: #FFFFFF;
  --radius: 12px;
  --radius-sm: 8px;
  --shadow-sm: 0 1px 3px rgba(11,31,58,0.08), 0 1px 2px rgba(11,31,58,0.06);
  --shadow-lg: 0 8px 24px rgba(11,31,58,0.13);
}

html, body, .stApp, [class*="css"] {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
  color: var(--text);
  line-height: 1.55;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
  padding-top: 1.25rem;
  max-width: 1360px;
}

.stApp {
  background: var(--bg);
}

.stMarkdown, .stMarkdown p, .stText, p, li, td, th, label, input, select, textarea, button {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
}

.stMarkdown p, .stText, .stRadio label p, .stCheckbox label p {
  font-size: 0.925rem !important;
}

.stCaption, .stCaption p, small {
  font-size: 0.8rem !important;
  color: var(--text-muted) !important;
}

div[data-testid="stSidebar"] h1,
div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3,
div[data-testid="stSidebar"] h4 {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
  color: var(--navy) !important;
  line-height: 1.25 !important;
}

div[data-testid="stSidebar"],
div[data-testid="stSidebar"] p,
div[data-testid="stSidebar"] span,
div[data-testid="stSidebar"] label,
div[data-testid="stSidebar"] .timeline-title,
div[data-testid="stSidebar"] .timeline-desc,
div[data-testid="stSidebar"] .step-nav-locked,
div[data-testid="stSidebar"] .sidebar-steps-capsule,
div[data-testid="stSidebar"] .sidebar-steps-capsule-label,
div[data-testid="stSidebar"] .sidebar-step-inner {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
}

.sidebar-steps-capsule {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  margin: 0 0 18px;
  padding: 12px 20px;
  border-radius: 999px;
  background: linear-gradient(135deg, var(--navy) 0%, var(--navy-mid) 100%);
  border: 2px solid var(--gold);
  box-shadow: 0 6px 18px rgba(11, 31, 58, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.12);
  position: relative;
  overflow: hidden;
}

.sidebar-steps-capsule::before {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, rgba(201, 151, 58, 0.18), transparent 55%);
  pointer-events: none;
}

.sidebar-steps-capsule-label {
  position: relative;
  z-index: 1;
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
  font-size: 0.92rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #FFFFFF !important;
  line-height: 1 !important;
  text-align: center;
}

.stAlert, .stAlert p, [data-testid="stNotification"] {
  font-size: 0.9rem !important;
  font-weight: 600 !important;
}

.page-header {
  position: relative;
  overflow: hidden;
  border-radius: var(--radius);
  background: var(--navy);
  padding: 36px 40px 32px;
  margin-bottom: 28px;
  box-shadow: var(--shadow-lg);
}

.page-header::before {
  content: "";
  position: absolute;
  right: -60px;
  top: -60px;
  width: 260px;
  height: 260px;
  border-radius: 50%;
  border: 40px solid rgba(201,151,58,0.18);
  pointer-events: none;
}

.header-inner {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
  gap: 20px;
}

.header-logo {
  flex-shrink: 0;
  padding: 10px 18px;
  background: rgba(255, 255, 255, 0.96);
  border-radius: var(--radius-sm);
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.12);
}

.custom-logo {
  height: 80px;
  width: auto;
  max-width: 280px;
  display: block;
  object-fit: contain;
}

.header-text h1 {
  margin: 0;
  font-size: 1.65rem;
  font-weight: 700;
  color: #FFFFFF;
  letter-spacing: -0.03em;
  line-height: 1.2;
}

.header-text p {
  margin-top: 5px;
  font-size: 0.9rem;
  color: rgba(255,255,255,0.65);
}

.block {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-sm);
  margin-top: 20px;
  margin-bottom: 16px;
  overflow: hidden;
}

.block-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 18px 24px 16px;
  border-bottom: 1px solid var(--border-soft);
  background: var(--navy-subtle);
}

.block-number {
  flex-shrink: 0;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: var(--navy);
  color: #fff;
  font-size: 0.78rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
}

.block-title {
  font-size: 1rem;
  font-weight: 700;
  color: var(--navy);
  line-height: 1.25;
}

.block-sub {
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-top: 2px;
}

.block-body {
  padding: 22px 24px;
}

.section-helper {
  font-size: 0.875rem;
  color: var(--text-muted);
  line-height: 1.55;
  margin-bottom: 14px;
}

.field-definition {
  display: block;
  margin: 2px 0 12px;
  font-size: 0.78rem;
  color: var(--text-muted);
  font-style: italic;
  line-height: 1.4;
}

.field-label {
  display: block;
  margin: 0 0 0.35rem;
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  color: var(--text-muted) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.04em !important;
  line-height: 1.3 !important;
}

.market-shock-header {
  padding: 12px 18px 14px;
  background: var(--navy);
  color: #fff;
}

.market-shock-header-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 6px;
}

.market-shock-title {
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.market-shock-header .field-definition {
  margin: 0;
  color: rgba(255, 255, 255, 0.72);
  font-size: 0.78rem;
  font-style: italic;
  line-height: 1.4;
  text-transform: none;
  letter-spacing: normal;
  font-weight: 400;
}

.market-shock-header .gold-badge {
  background: rgba(201, 151, 58, 0.18);
  color: #FDF4E3;
  border-color: var(--gold);
}

.meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 14px;
}

.meta-item label {
  display: block;
  margin-bottom: 6px;
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.meta-item strong {
  font-size: 0.925rem;
  font-weight: 600;
  color: var(--text);
}

.gold-badge, .week-use-hint {
  display: inline-block;
  padding: 3px 10px;
  background: var(--gold-light);
  color: var(--gold-dark);
  border: 1px solid var(--gold);
  border-radius: 20px;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: none;
}

.scenario-preview-card {
  border: 1.5px solid var(--gold);
  border-left: 5px solid var(--gold-dark);
  border-radius: var(--radius-sm);
  background: #FFFDF8;
  box-shadow: var(--shadow-sm);
  margin-top: 14px;
  margin-bottom: 16px;
  overflow: hidden;
}

.scenario-preview-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 10px 18px;
  background: var(--navy);
  color: #fff;
}

.scenario-preview-header span {
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.scenario-preview-header .gold-badge {
  background: rgba(201, 151, 58, 0.18);
  color: #FDF4E3;
  border-color: var(--gold);
}

.scenario-preview-body {
  padding: 18px 20px;
  background: linear-gradient(180deg, #FFFDF8 0%, var(--gold-light) 100%);
}

.scenario-preview-card .block-title {
  color: var(--navy);
  margin-bottom: 8px;
}

.scenario-preview-card .block-sub {
  color: var(--text-muted);
  margin-bottom: 12px;
  line-height: 1.55;
}

.scenario-preview-card .gold-badge {
  background: var(--gold-light);
  color: var(--gold-dark);
  border-color: var(--gold-dark);
}

.scenario-preview-card .core-question-line {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid rgba(201, 151, 58, 0.35);
  font-size: 0.875rem;
  color: var(--navy-mid);
  line-height: 1.55;
}

.scenario-preview-card .core-question-line strong {
  color: var(--gold-dark);
  font-weight: 700;
}

.scenario-preview-card .market-shock-preview-line {
  margin-top: 14px;
}

.custom-scenario-panel {
  border: 1.5px solid var(--gold);
  border-left: 5px solid var(--gold-dark);
  border-radius: var(--radius-sm);
  background: #FFFDF8;
  box-shadow: var(--shadow-sm);
  margin-top: 14px;
  margin-bottom: 10px;
  overflow: hidden;
}

.custom-scenario-panel .scenario-preview-header .gold-badge {
  background: rgba(201, 151, 58, 0.18);
  color: #FDF4E3;
  border-color: var(--gold);
}

.agent-card {
  border: 1.5px solid var(--border-soft);
  border-left: 4px solid var(--gold);
  background: #FFFDF8;
  padding: 14px 16px;
  margin-top: 12px;
  margin-bottom: 12px;
  border-radius: var(--radius-sm);
}

.agent-card h4 {
  display: block;
  margin: 0 0 8px;
  font-weight: 700;
  font-size: 0.88rem;
  color: var(--navy);
}

.debate-round-card {
  border: 1.5px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 18px;
  overflow: hidden;
  box-shadow: var(--shadow-sm);
  background: var(--card);
}

.debate-round-card--opening {
  border-top: 4px solid var(--gold);
}

.debate-round-card--defense {
  border-top: 4px solid var(--navy-mid);
}

.debate-round-header {
  padding: 12px 18px;
  background: linear-gradient(135deg, var(--navy) 0%, var(--navy-mid) 100%);
  color: #fff;
  font-size: 0.92rem;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.debate-round-card--defense .debate-round-header {
  background: linear-gradient(135deg, var(--navy-mid) 0%, #1e4475 100%);
}

.debate-round-desc {
  padding: 10px 18px;
  background: var(--gold-light);
  color: var(--gold-dark);
  font-size: 0.82rem;
  font-weight: 600;
  border-bottom: 1px solid rgba(201, 151, 58, 0.25);
}

.debate-round-body {
  padding: 4px 0;
}

.debate-turn {
  padding: 14px 18px;
  border-bottom: 1px solid var(--border-soft);
  border-left: 4px solid var(--gold);
  margin: 10px 14px;
  border-radius: var(--radius-sm);
  background: #FFFDF8;
  box-shadow: var(--shadow-sm);
}

.debate-turn:last-child {
  border-bottom: 1px solid var(--border-soft);
}

.debate-turn--agent-1 {
  border-left-color: var(--gold);
  background: #FFFDF8;
}

.debate-turn--agent-2 {
  border-left-color: var(--navy);
  background: var(--navy-light);
}

.debate-turn--agent-3 {
  border-left-color: var(--gold-dark);
  background: var(--gold-light);
}

.debate-turn--waiting {
  opacity: 0.72;
  background: var(--navy-subtle);
  border-left-color: var(--border);
}

.debate-turn--writing {
  box-shadow: 0 0 0 2px rgba(201, 151, 58, 0.25);
}

.debate-turn-agent {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  font-size: 0.88rem;
  font-weight: 700;
  color: var(--navy);
  margin-bottom: 6px;
}

.debate-turn-name {
  flex: 1;
}

.debate-status {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  padding: 3px 8px;
  border-radius: 999px;
  white-space: nowrap;
}

.debate-status--waiting {
  background: var(--navy-subtle);
  color: var(--text-muted);
  border: 1px solid var(--border);
}

.debate-status--writing {
  background: var(--gold-light);
  color: var(--gold-dark);
  border: 1px solid rgba(201, 151, 58, 0.35);
  animation: debatePulse 1.4s ease-in-out infinite;
}

.debate-status--present {
  background: var(--success-bg);
  color: var(--success);
  border: 1px solid rgba(27, 122, 71, 0.25);
}

.debate-turn-text {
  font-size: 0.88rem;
  color: var(--text);
  line-height: 1.6;
}

.debate-web-search-calls {
  margin-bottom: 10px;
}

.debate-web-search-call {
  padding: 10px 12px;
  margin-bottom: 8px;
  border-radius: var(--radius-sm);
  border: 1px solid rgba(201, 151, 58, 0.35);
  background: rgba(255, 253, 248, 0.95);
  font-size: 0.8rem;
  line-height: 1.5;
}

.debate-web-search-call--failed {
  border-color: rgba(180, 60, 60, 0.35);
  background: rgba(255, 245, 245, 0.95);
}

.debate-web-search-call--retrying {
  border-color: rgba(22, 47, 86, 0.35);
  background: rgba(234, 240, 248, 0.95);
}

.debate-web-search-blocked {
  margin: 6px 0 4px;
  padding: 8px 10px;
  border-left: 3px solid var(--navy-mid);
  color: var(--navy);
  font-weight: 600;
  background: rgba(255, 255, 255, 0.65);
}

.debate-web-search-method {
  margin-top: 4px;
  font-size: 0.75rem;
  color: var(--text-muted);
  font-style: italic;
}

.debate-web-search-label {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--gold-dark);
  margin-bottom: 4px;
}

.debate-web-search-reason {
  color: var(--navy);
  font-weight: 600;
  margin-bottom: 4px;
}

.debate-web-search-url a {
  color: var(--navy-mid);
  word-break: break-all;
}

.debate-web-search-title {
  margin-top: 4px;
  color: var(--text-muted);
  font-style: italic;
}

.debate-web-search-excerpt {
  margin: 8px 0 0;
  padding: 8px 10px;
  border-left: 3px solid var(--gold);
  color: var(--text);
  background: rgba(255, 255, 255, 0.7);
}

.debate-turn-text strong {
  font-weight: 700;
  color: var(--navy);
}

.debate-turn-text em,
.debate-turn-partial {
  color: var(--text-muted);
}

.debate-turn-placeholder {
  font-style: italic;
  color: var(--text-hint);
}

.debate-roster {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 10px;
  margin: 0 0 16px;
}

.debate-roster-item {
  border: 1.5px solid var(--border-soft);
  border-left: 4px solid var(--gold);
  border-radius: var(--radius-sm);
  padding: 10px 12px;
  background: #FFFDF8;
}

.debate-roster-item--agent-1 {
  border-left-color: var(--gold);
  background: #FFFDF8;
}

.debate-roster-item--agent-2 {
  border-left-color: var(--navy);
  background: var(--navy-light);
}

.debate-roster-item--agent-3 {
  border-left-color: var(--gold-dark);
  background: var(--gold-light);
}

.debate-roster-item--waiting {
  opacity: 0.78;
}

.debate-roster-item--writing {
  box-shadow: 0 0 0 2px rgba(201, 151, 58, 0.28);
}

.debate-roster-name {
  font-size: 0.84rem;
  font-weight: 700;
  color: var(--navy);
  margin-bottom: 4px;
}

.debate-roster-status {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.debate-roster-item--writing .debate-roster-status {
  color: var(--gold-dark);
  animation: debatePulse 1.4s ease-in-out infinite;
}

.debate-roster-item--present .debate-roster-status {
  color: var(--success);
}

.debate-analysis-wrap {
  margin-top: 8px;
}

.debate-thinking {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 10px 0 14px;
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  background: var(--gold-light);
  border: 1px solid rgba(201, 151, 58, 0.35);
  color: var(--gold-dark);
  font-size: 0.86rem;
  font-weight: 600;
  animation: debatePulse 1.4s ease-in-out infinite;
}

.debate-thinking-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--gold);
  flex-shrink: 0;
  animation: debateBlink 1s ease-in-out infinite;
}

@keyframes debatePulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.72; }
}

@keyframes debateBlink {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(0.75); opacity: 0.55; }
}

.final-report-wrap {
  margin-top: 8px;
}

.final-report-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--navy);
  padding: 14px 18px;
  margin-bottom: 16px;
  border-radius: var(--radius-sm);
  background: linear-gradient(135deg, var(--navy) 0%, var(--navy-mid) 100%);
  color: #fff;
  letter-spacing: 0.02em;
}

.final-report-section {
  border: 1.5px solid var(--border);
  border-radius: var(--radius);
  background: var(--card);
  margin-bottom: 18px;
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}

.final-report-section-title {
  padding: 12px 18px;
  background: var(--gold-light);
  color: var(--gold-dark);
  font-size: 0.9rem;
  font-weight: 700;
  border-bottom: 1px solid rgba(201, 151, 58, 0.25);
}

.final-report-section-body {
  padding: 16px 18px;
  font-size: 0.88rem;
  line-height: 1.6;
  color: var(--text);
}

.final-report-agent-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
  padding: 16px 18px;
}

.final-report-agent-card {
  border: 1.5px solid var(--border-soft);
  border-left: 4px solid var(--gold);
  border-radius: var(--radius-sm);
  padding: 14px 16px;
  background: #FFFDF8;
  box-shadow: var(--shadow-sm);
}

.final-report-agent-card--1 {
  border-left-color: var(--gold);
  background: #FFFDF8;
}

.final-report-agent-card--2 {
  border-left-color: var(--navy);
  background: var(--navy-light);
}

.final-report-agent-card--3 {
  border-left-color: var(--gold-dark);
  background: var(--gold-light);
}

.final-report-agent-name {
  font-size: 0.92rem;
  font-weight: 700;
  color: var(--navy);
  margin-bottom: 4px;
}

.final-report-agent-position {
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-bottom: 10px;
  line-height: 1.45;
}

.final-report-agent-body ul {
  margin: 8px 0 0;
  padding-left: 18px;
}

.final-report-agent-body li {
  margin-bottom: 6px;
  color: var(--text);
  font-size: 0.86rem;
  line-height: 1.55;
}

.final-report-core {
  font-size: 0.86rem;
  font-weight: 600;
  color: var(--navy-mid);
  margin-bottom: 8px;
}

.final-report-muted {
  color: var(--text-hint);
  font-style: italic;
  font-size: 0.84rem;
}

.final-report-map-wrap {
  padding: 0 18px 16px;
  overflow-x: auto;
}

.final-report-map-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.84rem;
}

.final-report-map-table th {
  background: var(--navy);
  color: #fff;
  font-weight: 700;
  text-align: left;
  padding: 10px 12px;
}

.final-report-map-table td {
  border-bottom: 1px solid var(--border-soft);
  padding: 10px 12px;
  vertical-align: top;
  color: var(--text);
  line-height: 1.5;
}

.final-report-map-table tr:nth-child(even) td {
  background: var(--navy-subtle);
}

.final-report-list {
  margin: 0;
  padding-left: 18px;
}

.final-report-list li {
  margin-bottom: 8px;
  line-height: 1.55;
}

.timeline-item {
  display: flex;
  gap: 12px;
  padding: 10px 0;
}

.timeline-marker {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
  font-size: 0.75rem;
  font-weight: 700;
  flex-shrink: 0;
}

.timeline-marker.completed {
  background: var(--success);
  color: white;
}

.timeline-marker.current {
  background: white;
  color: var(--success);
  border: 2px solid var(--success);
}

.timeline-marker.pending {
  background: var(--navy-subtle);
  color: var(--text-muted);
  border: 2px solid var(--border);
}

.timeline-marker.locked {
  background: var(--navy-subtle);
  color: var(--text-hint);
  border: 2px solid var(--border-soft);
}

.timeline-title {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
  font-weight: 700;
  font-size: 0.88rem;
  color: var(--navy);
  line-height: 1.25;
}

.timeline-desc {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-top: 2px;
}

.timeline-item.pending { opacity: 0.45; }
.timeline-item.locked { opacity: 0.45; cursor: not-allowed; }
.timeline-item.current .timeline-title { color: var(--success); }

div[data-testid="stSidebar"] div[class*="st-key-sidebar_step_"] > div[data-testid="stVerticalBlock"] {
  border: 1.5px solid var(--border);
  border-radius: var(--radius);
  background: var(--card);
  box-shadow: var(--shadow-sm);
  padding: 12px 14px 10px;
  margin-bottom: 14px;
}

div[data-testid="stSidebar"] div[class*="st-key-sidebar_step_"]:has(.sidebar-step-inner--current) > div[data-testid="stVerticalBlock"] {
  border-color: rgba(25, 135, 84, 0.45);
  box-shadow: 0 4px 14px rgba(25, 135, 84, 0.12);
}

div[data-testid="stSidebar"] div[class*="st-key-sidebar_step_"]:has(.sidebar-step-inner--completed) > div[data-testid="stVerticalBlock"] {
  border-color: rgba(25, 135, 84, 0.28);
}

div[data-testid="stSidebar"] div[class*="st-key-sidebar_step_"]:has(.sidebar-step-inner--locked) > div[data-testid="stVerticalBlock"],
div[data-testid="stSidebar"] div[class*="st-key-sidebar_step_"]:has(.sidebar-step-inner--pending) > div[data-testid="stVerticalBlock"] {
  opacity: 0.82;
}

div[data-testid="stSidebar"] .sidebar-step-inner .timeline-item {
  padding: 0 0 10px;
  margin: 0;
}

div[data-testid="stSidebar"] div[class*="st-key-sidebar_step_"] div.stButton {
  margin-top: 0;
}

div[data-testid="stSidebar"] div[class*="st-key-sidebar_step_"] div.stButton > button {
  margin-bottom: 0;
}

div[data-testid="stSidebar"] div[class*="st-key-sidebar_step_"] .step-nav-locked {
  margin-bottom: 0;
}

.step-nav-wrap {
  position: relative;
}

.step-nav-locked {
  display: block;
  width: 100%;
  margin: 0 0 0.5rem;
  padding: 0.46rem 0.75rem;
  border: 1.5px solid var(--gold);
  border-radius: var(--radius-sm);
  background: var(--gold-light);
  color: var(--gold-dark);
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
  font-size: 0.875rem;
  font-weight: 600;
  text-align: center;
  opacity: 0.55;
  cursor: not-allowed;
  position: relative;
}

.step-nav-locked:hover::after {
  content: "Complete the previous step for this.";
  position: absolute;
  left: 0;
  right: 0;
  bottom: calc(100% + 6px);
  z-index: 20;
  padding: 8px 10px;
  border-radius: var(--radius-sm);
  background: var(--navy);
  color: #fff;
  font-size: 0.78rem;
  font-weight: 600;
  line-height: 1.4;
  text-align: center;
  box-shadow: var(--shadow-sm);
  pointer-events: none;
  white-space: normal;
}

div.stButton > button:disabled {
  opacity: 0.35 !important;
  cursor: not-allowed !important;
}

div.stButton > button.add-agent-max:disabled {
  opacity: 0.35 !important;
  cursor: not-allowed !important;
  background: var(--navy-subtle) !important;
  color: var(--text-hint) !important;
  border: 1.5px solid var(--border) !important;
  box-shadow: none !important;
}

.success-banner {
  background: linear-gradient(135deg, var(--success), #155E37);
  color: white;
  border-radius: var(--radius);
  padding: 22px 24px;
  margin-bottom: 16px;
  font-size: 0.925rem;
  font-weight: 600;
}

div[data-testid="stSidebar"] {
  background: var(--bg);
  border-right: 1px solid var(--border);
}

div.stButton > button {
  font-family: inherit !important;
  border-radius: var(--radius-sm) !important;
  transition: background 0.15s, border-color 0.15s, transform 0.1s !important;
}

div.stButton > button[kind="primary"] {
  background: var(--navy) !important;
  color: #fff !important;
  border: none !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
  box-shadow: 0 4px 14px rgba(11,31,58,0.25) !important;
}

div.stButton > button[kind="primary"]:hover {
  background: var(--navy-mid) !important;
}

div.stButton > button[kind="primary"]:active {
  transform: scale(0.98) !important;
}

div.stButton > button[kind="secondary"] {
  background: var(--gold-light) !important;
  color: var(--gold-dark) !important;
  border: 1.5px solid var(--gold) !important;
  font-size: 0.875rem !important;
  font-weight: 600 !important;
}

div.stButton > button[kind="secondary"]:hover {
  background: #FDE8BB !important;
  border-color: var(--gold-dark) !important;
}

div.stButton > button[kind="secondary"]:active {
  transform: scale(0.97) !important;
}

.stTextInput input, .stTextArea textarea,
.stTextInput div[data-baseweb="input"] > div,
.stTextArea div[data-baseweb="textarea"] > div {
  font-size: 0.925rem !important;
  font-family: inherit !important;
  border-radius: var(--radius-sm) !important;
  border: 1px solid #000000 !important;
  background: #FFFFFF !important;
  color: var(--text) !important;
}

.stTextInput input:focus, .stTextArea textarea:focus,
.stTextInput div[data-baseweb="input"] > div:focus-within,
.stTextArea div[data-baseweb="textarea"] > div:focus-within {
  border-color: #000000 !important;
  box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.10) !important;
  outline: none !important;
}

div[data-baseweb="select"] > div,
.stSelectbox div[data-baseweb="select"] > div {
  font-size: 0.925rem !important;
  font-family: inherit !important;
  border-radius: var(--radius-sm) !important;
}

.st-key-preset_business_scenario div[data-testid="stSelectbox"] > div:last-of-type,
.st-key-preset_market_shock div[data-testid="stSelectbox"] > div:last-of-type,
.st-key-preset_business_scenario div[data-baseweb="select"] > div,
.st-key-preset_market_shock div[data-baseweb="select"] > div {
  font-size: 0.925rem !important;
  font-family: inherit !important;
  border-radius: var(--radius-sm) !important;
  border: 1px solid #000000 !important;
  background: #FFFFFF !important;
  color: var(--text) !important;
  box-shadow: none !important;
}

.st-key-preset_business_scenario div[data-testid="stSelectbox"] > div:last-of-type:focus-within,
.st-key-preset_market_shock div[data-testid="stSelectbox"] > div:last-of-type:focus-within,
.st-key-preset_business_scenario div[data-baseweb="select"] > div:focus-within,
.st-key-preset_market_shock div[data-baseweb="select"] > div:focus-within {
  border-color: #000000 !important;
  box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.10) !important;
  outline: none !important;
}

[class*="st-key-agent_reasoning_style_"] div[data-baseweb="select"] > div,
[class*="st-key-agent_reasoning_style_"] div[data-testid="stSelectbox"] > div:last-of-type {
  font-size: 0.925rem !important;
  font-family: inherit !important;
  border-radius: var(--radius-sm) !important;
  border: 1px solid #000000 !important;
  background: #FFFFFF !important;
  color: var(--text) !important;
  box-shadow: none !important;
}

[class*="st-key-agent_reasoning_style_"] div[data-baseweb="select"] > div:focus-within,
[class*="st-key-agent_reasoning_style_"] div[data-testid="stSelectbox"] > div:last-of-type:focus-within {
  border-color: #000000 !important;
  box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.10) !important;
  outline: none !important;
}

[class*="st-key-agent_temperature_"] div[data-testid="stSlider"] {
  border: 1px solid #000000 !important;
  border-radius: var(--radius-sm) !important;
  background: #FFFFFF !important;
  padding: 10px 14px 6px !important;
  box-shadow: none !important;
}

[class*="st-key-agent_temperature_"] div[data-testid="stSlider"]:focus-within {
  border-color: #000000 !important;
  box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.10) !important;
  outline: none !important;
}

.stSelectbox label, .stTextInput label, .stTextArea label,
.stMultiSelect label, .stRadio > label {
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  color: var(--text-muted) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.04em !important;
}

.stRadio div[role="radiogroup"] label p,
.stRadio div[role="radiogroup"] label span {
  font-size: 0.925rem !important;
  text-transform: none !important;
  letter-spacing: normal !important;
  font-weight: 400 !important;
  color: var(--text) !important;
}

div[data-testid="stPills"] [data-baseweb="button-group"] {
  flex-wrap: wrap;
  gap: 10px;
}

div[data-testid="stPills"] button {
  border-radius: 999px !important;
  border: 1px solid var(--border) !important;
  background: #FFFFFF !important;
  color: var(--navy) !important;
  font-size: 0.875rem !important;
  font-weight: 600 !important;
  padding: 8px 22px !important;
  min-height: 40px !important;
  box-shadow: var(--shadow-sm) !important;
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
}

div[data-testid="stPills"] button:hover {
  border-color: var(--gold) !important;
  background: var(--gold-light) !important;
  color: var(--navy) !important;
}

div[data-testid="stPills"] button[kind="primary"],
div[data-testid="stPills"] button[aria-pressed="true"] {
  background: var(--navy) !important;
  border-color: var(--navy) !important;
  color: #FFFFFF !important;
  box-shadow: 0 2px 8px rgba(11, 31, 58, 0.18) !important;
}

.stTabs [data-baseweb="tab"] {
  font-size: 0.875rem !important;
  font-weight: 600 !important;
}

.stTabs [aria-selected="true"] {
  color: var(--gold-dark) !important;
  border-bottom: 3px solid var(--gold) !important;
  font-weight: 700 !important;
}

.stDataFrame, .stDataFrame div {
  font-size: 0.925rem !important;
}

div[data-testid="stDataFrame"],
div[data-testid="stDataFrame"] *,
div[data-testid="stDataFrameGlideDataEditor"],
div[data-testid="stDataFrameGlideDataEditor"] *,
.stDataFrame [data-testid="glideDataEditor"],
.stDataFrame [data-testid="glideDataEditor"] * {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
  font-size: 0.925rem !important;
}

.agents-table-wrap {
  overflow-x: auto;
  margin: 12px 0 16px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--card);
  box-shadow: var(--shadow-sm);
}

.agents-table {
  width: 100%;
  border-collapse: collapse;
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
  font-size: 0.925rem !important;
  color: var(--text);
}

.agents-table th,
.agents-table td {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
  font-size: 0.925rem !important;
  line-height: 1.55;
  padding: 12px 14px;
  border-bottom: 1px solid var(--border-soft);
  text-align: left;
  vertical-align: top;
}

.agents-table th {
  background: var(--navy-subtle);
  color: var(--navy);
  font-weight: 700 !important;
  font-size: 0.78rem !important;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.agents-table tbody tr:last-child td {
  border-bottom: none;
}

.agents-table tbody tr:hover td {
  background: #FFFDF8;
}

.main .stMarkdown table,
.main .stMarkdown table th,
.main .stMarkdown table td {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
  font-size: 0.925rem !important;
  color: var(--text);
}

.main .stMarkdown table th {
  font-weight: 700 !important;
  color: var(--navy);
}

.agent-card,
.agent-card h4,
.agent-card span {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
}

.agent-design-block {
  border: 1px solid var(--border);
  border-left: 4px solid var(--gold);
  border-radius: var(--radius-sm);
  background: #FFFDF8;
  padding: 16px 18px;
  margin-bottom: 16px;
  box-shadow: var(--shadow-sm);
}

.agent-design-title {
  margin: 0 0 12px;
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--navy);
  line-height: 1.3;
}

div[class*="st-key-agent_design_card_0"] > div[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1.5px solid rgba(201, 151, 58, 0.45) !important;
  border-left: 5px solid var(--gold) !important;
  border-radius: var(--radius-sm) !important;
  background: linear-gradient(180deg, #FFFDF8 0%, var(--gold-light) 100%) !important;
  padding: 12px 14px 8px !important;
  margin-bottom: 16px !important;
  box-shadow: 0 4px 14px rgba(201, 151, 58, 0.12) !important;
}

div[class*="st-key-agent_design_card_1"] > div[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1.5px solid rgba(22, 47, 86, 0.28) !important;
  border-left: 5px solid var(--navy-mid) !important;
  border-radius: var(--radius-sm) !important;
  background: linear-gradient(180deg, #FFFFFF 0%, var(--navy-light) 100%) !important;
  padding: 12px 14px 8px !important;
  margin-bottom: 16px !important;
  box-shadow: 0 4px 14px rgba(11, 31, 58, 0.10) !important;
}

div[class*="st-key-agent_design_card_2"] > div[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1.5px solid rgba(155, 114, 41, 0.42) !important;
  border-left: 5px solid var(--gold-dark) !important;
  border-radius: var(--radius-sm) !important;
  background: linear-gradient(180deg, #FFFDF8 0%, #F8EBCF 100%) !important;
  padding: 12px 14px 8px !important;
  margin-bottom: 16px !important;
  box-shadow: 0 4px 14px rgba(155, 114, 41, 0.12) !important;
}

.agent-design-title--1 .gold-badge {
  background: rgba(201, 151, 58, 0.16) !important;
  color: var(--gold-dark) !important;
  border-color: var(--gold) !important;
}

.agent-design-title--2 .gold-badge {
  background: rgba(22, 47, 86, 0.10) !important;
  color: var(--navy-mid) !important;
  border-color: var(--navy-mid) !important;
}

.agent-design-title--3 .gold-badge {
  background: rgba(155, 114, 41, 0.14) !important;
  color: var(--gold-dark) !important;
  border-color: var(--gold-dark) !important;
}

.st-key-add_stakeholder_1_button div.stButton > button {
  background: var(--gold-light) !important;
  color: var(--gold-dark) !important;
  border: 1.5px solid var(--gold) !important;
  font-weight: 700 !important;
  box-shadow: 0 4px 12px rgba(201, 151, 58, 0.18) !important;
}

.st-key-add_stakeholder_1_button div.stButton > button:hover:not(:disabled) {
  background: #FDE8BB !important;
  border-color: var(--gold-dark) !important;
  color: var(--navy) !important;
}

.st-key-add_stakeholder_2_button div.stButton > button {
  background: var(--navy-light) !important;
  color: var(--navy) !important;
  border: 1.5px solid var(--navy-mid) !important;
  font-weight: 700 !important;
  box-shadow: 0 4px 12px rgba(11, 31, 58, 0.12) !important;
}

.st-key-add_stakeholder_2_button div.stButton > button:hover:not(:disabled) {
  background: #DCE8F5 !important;
  border-color: var(--navy) !important;
}

.st-key-add_stakeholder_3_button div.stButton > button {
  background: #F8EBCF !important;
  color: var(--gold-dark) !important;
  border: 1.5px solid var(--gold-dark) !important;
  font-weight: 700 !important;
  box-shadow: 0 4px 12px rgba(155, 114, 41, 0.16) !important;
}

.st-key-add_stakeholder_3_button div.stButton > button:hover:not(:disabled) {
  background: var(--gold-light) !important;
  border-color: var(--gold-dark) !important;
  color: var(--navy) !important;
}

.st-key-add_stakeholder_1_button div.stButton > button:disabled,
.st-key-add_stakeholder_2_button div.stButton > button:disabled,
.st-key-add_stakeholder_3_button div.stButton > button:disabled {
  opacity: 0.45 !important;
  box-shadow: none !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid var(--border) !important;
  border-left: 4px solid var(--gold) !important;
  border-radius: var(--radius-sm) !important;
  background: #FFFDF8 !important;
  padding: 8px 12px 4px !important;
  margin-bottom: 16px !important;
  box-shadow: var(--shadow-sm) !important;
}

.material-icons,
.material-symbols-outlined,
.material-symbols-rounded,
span[data-testid="stIconMaterial"],
[data-testid="stExpander"] .material-symbols-rounded,
[data-testid="stExpander"] .material-icons,
[data-testid="stExpander"] .material-symbols-outlined,
[data-testid="stSelectbox"] .material-symbols-rounded,
[data-testid="stSelectbox"] .material-icons,
[data-testid="stSelectbox"] .material-symbols-outlined {
  font-family: "Material Symbols Rounded", "Material Icons" !important;
  font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24 !important;
  letter-spacing: normal !important;
  text-transform: none !important;
}

.main .stMarkdown h1 {
  font-size: 1.65rem !important;
  font-weight: 700 !important;
  color: var(--navy) !important;
  line-height: 1.2 !important;
  letter-spacing: -0.03em !important;
}

.main .stMarkdown h2,
.main .stMarkdown h3 {
  font-size: 1rem !important;
  font-weight: 700 !important;
  color: var(--navy) !important;
  line-height: 1.25 !important;
}

.main .stMarkdown h4 {
  font-size: 0.88rem !important;
  font-weight: 700 !important;
  color: var(--navy) !important;
}

.main .stMarkdown h5,
.main .stMarkdown h6 {
  font-size: 0.875rem !important;
  font-weight: 600 !important;
  color: var(--text-muted) !important;
}

@media (max-width: 720px) {
  .header-text h1 { font-size: 1.3rem; }
  .block-body { padding: 16px; }
  .page-header { padding: 24px 20px; }
  .header-inner { flex-direction: column; align-items: flex-start; gap: 12px; }
}
</style>
"""


def render_field_label(text: str, *, trailing_dash: bool = True):
    suffix = LABEL_DASH if trailing_dash else ""
    st.markdown(
        f'<p class="field-label">{html.escape(text)}{suffix}</p>',
        unsafe_allow_html=True,
    )


def render_panel_header(title: str, badge_text: str, definition: str):
    st.markdown(
        f"""
        <div class="custom-scenario-panel">
          <div class="market-shock-header">
            <div class="market-shock-header-top">
              <span class="market-shock-title">{title}</span>
              <span class="gold-badge">{badge_text}</span>
            </div>
            <p class="field-definition"><em>{definition}</em></p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_openai_api_key() -> str:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"]).strip()
        openai_secrets = st.secrets.get("openai", {})
        if isinstance(openai_secrets, dict) and openai_secrets.get("api_key"):
            return str(openai_secrets["api_key"]).strip()
    except Exception:
        pass

    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    api_file = Path(__file__).resolve().parent / "API.txt"
    if api_file.exists():
        for line in api_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("OPENAI_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                if api_key:
                    return api_key

    raise ValueError(
        "OPENAI_API_KEY not found. Add it in Streamlit Cloud secrets "
        "(Settings → Secrets) or local API.txt."
    )


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI(api_key=load_openai_api_key())


def ask_model(user_prompt: str, max_new_tokens: int = 300, temperature: float = 0.3) -> str:
    response = get_client().chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise business-school AI assistant. "
                    "Follow the requested output format exactly. "
                    "Do not repeat the task."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else 0,
    )
    return (response.choices[0].message.content or "").strip()


def stream_chat_completion(
    user_prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.45,
):
    stream = get_client().chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise business-school AI assistant. "
                    "Follow the requested output format exactly. "
                    "Do not repeat the task."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else 0,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def build_scenario(
    scenario_name: str,
    scenario_description: str,
    scenario_category: str,
    scenario_core_question: str,
) -> dict:
    return {
        "scenario_name": scenario_name,
        "scenario_description": scenario_description,
        "scenario_category": scenario_category,
        "scenario_core_question": scenario_core_question,
    }


def get_scenario_from_preset(selected_idea: str) -> dict:
    idea = BUSINESS_IDEAS[selected_idea]
    return build_scenario(
        scenario_name=selected_idea,
        scenario_description="",
        scenario_category=idea["category"],
        scenario_core_question=idea["core_question"],
    )


def validate_custom_scenario(
    scenario_name: str,
    scenario_category: str,
    scenario_core_question: str,
) -> str | None:
    if not scenario_name.strip():
        return "Please enter a scenario title."
    if not scenario_category.strip():
        return "Please enter a broad business category."
    if not scenario_core_question.strip():
        return "Please enter a core theme."
    return None


def empty_agent() -> dict:
    return {
        "name": "",
        "position": "",
        "temperature": DEFAULT_AGENT_TEMPERATURE,
        "reasoning_style": "",
        "tool_urls": "",
    }


def normalize_agent(agent: dict) -> dict:
    raw_style = agent.get("reasoning_style", None)
    if raw_style in REASONING_STYLES:
        reasoning_style = raw_style
    elif raw_style == "":
        reasoning_style = ""
    elif raw_style is None:
        reasoning_style = "Balanced"
    else:
        reasoning_style = "Balanced"
    try:
        temperature = float(agent.get("temperature", DEFAULT_AGENT_TEMPERATURE))
    except (TypeError, ValueError):
        temperature = DEFAULT_AGENT_TEMPERATURE
    tool_urls = agent.get("tool_urls") or agent.get("supporting_urls", "")
    return {
        "name": agent.get("name", ""),
        "position": agent.get("position", ""),
        "temperature": temperature,
        "reasoning_style": reasoning_style,
        "tool_urls": tool_urls,
    }


def get_agent_tool_urls(agent: dict) -> list[str]:
    urls, _ = parse_tool_urls(normalize_agent(agent).get("tool_urls", ""))
    return urls


def parse_tool_urls(raw_urls: str) -> tuple[list[str], str | None]:
    lines = [line.strip() for line in raw_urls.splitlines() if line.strip()]

    if len(lines) > MAX_TOOL_URLS_PER_AGENT:
        return [], f"Add at most {MAX_TOOL_URLS_PER_AGENT} tool URLs."

    cleaned_urls: list[str] = []
    seen: set[str] = set()

    for url in lines:
        if not url.startswith(("http://", "https://")):
            return [], "Each tool URL must start with http:// or https://."

        if not urlparse(url).netloc:
            return [], "Each tool URL must be a valid web address."

        if url not in seen:
            cleaned_urls.append(url)
            seen.add(url)

    return cleaned_urls, None


def split_tool_urls(raw_urls: str | list[str]) -> list[str]:
    if isinstance(raw_urls, list):
        slots = [url.strip() for url in raw_urls[:MAX_TOOL_URLS_PER_AGENT]]
    else:
        slots = [line.strip() for line in raw_urls.splitlines() if line.strip()][
            :MAX_TOOL_URLS_PER_AGENT
        ]

    while len(slots) < MAX_TOOL_URLS_PER_AGENT:
        slots.append("")

    return slots


def join_tool_urls(url_slots: list[str]) -> str:
    return "\n".join(url.strip() for url in url_slots if url.strip())


def clean_extracted_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_web_search_excerpt(
    text: str,
    max_chars: int = MAX_WEB_SEARCH_EXCERPT_CHARS,
) -> str:
    cleaned = clean_extracted_text(text)
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned

    truncated = cleaned[:max_chars]
    last_sentence_end = max(
        truncated.rfind("."),
        truncated.rfind("!"),
        truncated.rfind("?"),
    )
    if last_sentence_end >= max_chars // 2:
        return truncated[: last_sentence_end + 1].strip()
    return f"{truncated.rstrip()}…"


def truncate_at_sentence_boundary(text: str, min_chars: int = MAX_PROMPT_CONTEXT_CHARS) -> str:
    """Keep at least min_chars, then extend to the next sentence-ending punctuation."""
    if len(text) <= min_chars:
        return text

    for index in range(min_chars, len(text)):
        if text[index] in ".!?":
            return text[: index + 1].strip()

    return text[:min_chars].strip()


def extract_text_from_html(html_content: str) -> tuple[str | None, str | None]:
    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()

    title = None
    h1 = soup.find("h1")
    if h1:
        title = clean_extracted_text(h1.get_text(" ", strip=True))

    if not title and soup.title:
        title = clean_extracted_text(soup.title.get_text(" ", strip=True))

    candidate_containers: list = []
    for selector in ("article", "main"):
        candidate_containers.extend(soup.select(selector))

    if not candidate_containers:
        candidate_containers = [soup]

    paragraphs: list[str] = []
    for container in candidate_containers:
        for paragraph_tag in container.find_all("p"):
            paragraph = clean_extracted_text(paragraph_tag.get_text(" ", strip=True))
            if len(paragraph) >= 40:
                paragraphs.append(paragraph)

    unique_paragraphs: list[str] = []
    seen: set[str] = set()
    for paragraph in paragraphs:
        if paragraph not in seen:
            unique_paragraphs.append(paragraph)
            seen.add(paragraph)

    text = clean_extracted_text("\n\n".join(unique_paragraphs))
    return title, text


def blocked_fetch_notice_for_error(exc: Exception | None) -> str:
    return BLOCKED_FETCH_NOTICE


def extract_with_trafilatura(url: str) -> tuple[str | None, str | None]:
    try:
        import trafilatura
    except Exception:
        return None, None

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None, None

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )

        metadata = trafilatura.extract_metadata(downloaded)
        title = metadata.title if metadata and metadata.title else None

        if text:
            text = clean_extracted_text(text)

        return title, text
    except Exception:
        return None, None


def extract_with_bs4(url: str, timeout: int = ARTICLE_REQUEST_TIMEOUT) -> tuple[str | None, str | None]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return extract_text_from_html(response.text)


def is_blocked_page_title(title: str | None) -> bool:
    normalized = (title or "").strip().lower()
    return normalized in BLOCKED_PAGE_TITLES or any(
        keyword in normalized for keyword in ("access denied", "forbidden")
    )


def fetch_with_playwright(
    url: str,
    timeout_sec: int = PLAYWRIGHT_TIMEOUT_SEC,
) -> tuple[str | None, str | None, str]:
    """Headless browser fetch when direct HTTP fails. Optional on Streamlit Cloud."""
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None, None, "Playwright is not installed."

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=PLAYWRIGHT_LAUNCH_ARGS,
            )
            try:
                context = browser.new_context(
                    user_agent=WEB_FETCH_USER_AGENT,
                    locale="en-US",
                    timezone_id="America/New_York",
                    viewport={"width": 1366, "height": 768},
                    screen={"width": 1920, "height": 1080},
                    color_scheme="light",
                    extra_http_headers={
                        "Accept-Language": "en-US,en;q=0.9",
                        "DNT": "1",
                    },
                )
                context.add_init_script(PLAYWRIGHT_STEALTH_SCRIPT)
                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_sec * 1000)
                try:
                    page.wait_for_load_state(
                        "networkidle",
                        timeout=min(timeout_sec * 1000, 15_000),
                    )
                except Exception:
                    pass
                page.wait_for_timeout(1_500)
                page.evaluate(
                    "() => { window.scrollTo(0, document.body.scrollHeight / 2); }"
                )
                page.wait_for_timeout(800)
                html_content = page.content()
                page_title = page.title()
            finally:
                browser.close()
    except PlaywrightError as exc:
        message = str(exc)
        if "Executable doesn't exist" in message:
            return None, None, "Playwright Chromium is not installed."
        return None, None, message
    except Exception as exc:
        return None, None, str(exc)

    if is_blocked_page_title(page_title):
        return page_title, None, "Blocked page received."

    parsed_title, parsed_text = extract_text_from_html(html_content)
    resolved_title = page_title or parsed_title
    if is_blocked_page_title(resolved_title):
        return resolved_title, None, "Blocked page received."
    if not parsed_text or len(parsed_text) < MIN_EXTRACTED_CHARS:
        return resolved_title, None, "Too little readable text."
    return resolved_title, parsed_text, ""


def fetch_with_wayback(
    url: str,
    timeout_sec: int = WAYBACK_TIMEOUT_SEC,
) -> tuple[str | None, str | None, str, str]:
    """Fetch closest Wayback Machine snapshot when live page is blocked."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
    api_url = WAYBACK_AVAILABILITY_API.format(
        url=urllib.parse.quote(url, safe=":/"),
        ts=timestamp,
    )
    try:
        with urllib.request.urlopen(api_url, timeout=timeout_sec) as response:
            data = json.loads(response.read())
        archived = data.get("archived_snapshots", {}).get("closest", {})
        if not archived.get("available"):
            return None, None, "", "No archived snapshot is available for this URL."
        archive_url = archived["url"]
        page_response = requests.get(
            archive_url,
            headers={"User-Agent": WEB_FETCH_USER_AGENT},
            timeout=timeout_sec,
        )
        page_response.raise_for_status()
        title, text = extract_text_from_html(page_response.text)
        if not text or len(text) < MIN_EXTRACTED_CHARS:
            return title, None, archive_url, "Archived page had too little readable text."
        return title, text, archive_url, ""
    except Exception as exc:
        return None, None, "", f"Archive fetch failed: {exc}"


def notify_fetch_progress(
    on_progress: Callable[[dict], None] | None,
    *,
    status: str,
    notice: str,
    fetch_method: str,
) -> None:
    if on_progress:
        on_progress(
            {
                "status": status,
                "blocked_notice": notice,
                "fetch_method": fetch_method,
            }
        )


def extract_article_text(
    url: str,
    timeout: int = ARTICLE_REQUEST_TIMEOUT,
    *,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    title = None
    text = None
    blocked_notice = ""
    fetch_method = "direct"
    fetch_source_url = ""
    direct_error: Exception | None = None
    playwright_error = ""

    t_title, t_text = extract_with_trafilatura(url)
    if t_text and len(t_text) >= MIN_EXTRACTED_CHARS:
        title = t_title
        text = t_text

    if not text:
        try:
            b_title, b_text = extract_with_bs4(url, timeout=timeout)
            title = title or b_title
            text = b_text
        except Exception as exc:
            direct_error = exc

    needs_playwright = direct_error is not None or not text or len(text or "") < MIN_EXTRACTED_CHARS
    if needs_playwright:
        blocked_notice = blocked_fetch_notice_for_error(direct_error)
        notify_fetch_progress(
            on_progress,
            status="retrying",
            notice=blocked_notice,
            fetch_method="playwright",
        )
        p_title, p_text, playwright_error = fetch_with_playwright(
            url,
            timeout_sec=PLAYWRIGHT_TIMEOUT_SEC,
        )
        if p_text and len(p_text) >= MIN_EXTRACTED_CHARS:
            fetch_method = "playwright"
            blocked_notice = WEB_SEARCH_RECOVERED_NOTICE
            title = p_title or title
            text = p_text

    if not text or len(text) < MIN_EXTRACTED_CHARS:
        notify_fetch_progress(
            on_progress,
            status="retrying",
            notice=WAYBACK_FETCH_NOTICE,
            fetch_method="wayback",
        )
        w_title, w_text, archive_url, wayback_error = fetch_with_wayback(url)
        if w_text and len(w_text) >= MIN_EXTRACTED_CHARS:
            fetch_method = "wayback"
            blocked_notice = WEB_SEARCH_RECOVERED_NOTICE
            fetch_source_url = archive_url
            title = w_title or title
            text = w_text
        else:
            error_parts: list[str] = []
            if direct_error is not None:
                error_parts.append("Normal fetch was blocked.")
            if playwright_error:
                error_parts.append("Silent crawl could not reach the live page.")
            if wayback_error:
                error_parts.append("Could not recover a readable copy of the source.")
            if not error_parts:
                error_parts.append(
                    "Too little readable article text was extracted. Try another article URL."
                )
            return {
                "url": url,
                "title": title or "",
                "text": text or "",
                "status": "failed",
                "error": " ".join(error_parts),
                "blocked_notice": blocked_notice or BLOCKED_FETCH_NOTICE,
                "fetch_method": fetch_method if fetch_method != "direct" else "wayback",
                "fetch_source_url": archive_url or "",
            }

    return {
        "url": url,
        "title": title or "Untitled article",
        "text": text[:MAX_CHARS_PER_ARTICLE],
        "status": "success",
        "error": "",
        "blocked_notice": blocked_notice,
        "fetch_method": fetch_method,
        "fetch_source_url": fetch_source_url,
    }


def tool_urls_summary_label(agent: dict) -> str:
    count = len(get_agent_tool_urls(agent))
    if count == 0:
        return "None"
    return f"web_search ({count} URL{'s' if count != 1 else ''})"


def format_tool_urls_for_prompt(tool_urls: list[str]) -> str:
    if not tool_urls:
        return "No tools are configured for this agent."

    lines = [
        "Tool — web_search",
        "Description — Fetches readable article text from one of this agent's registered URLs.",
        "Registered URLs (up to 3) —",
    ]
    for index, url in enumerate(tool_urls, start=1):
        lines.append(f"  {index}. {url}")
    return "\n".join(lines)


def build_web_search_tool_definition(tool_urls: list[str]) -> list[dict]:
    if not tool_urls:
        return []

    return [
        {
            "type": "function",
            "function": {
                "name": WEB_SEARCH_TOOL_NAME,
                "description": (
                    "Search one of this agent's registered article URLs and return readable "
                    "text. Call only when you need concrete evidence, facts, statistics, "
                    "or examples to support your argument in this turn."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": (
                                "One of the URLs registered under this agent's web_search tool."
                            ),
                            "enum": tool_urls,
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "Brief note on what evidence you hope to find in this source."
                            ),
                        },
                    },
                    "required": ["url"],
                },
            },
        }
    ]


def build_web_search_fetch_entry(
    requested_url: str,
    reason: str,
    extracted: dict,
) -> dict:
    entry = {
        "url": requested_url,
        "reason": reason,
        "status": extracted.get("status", "failed"),
        "title": extracted.get("title", ""),
        "excerpt": "",
        "error": "",
        "blocked_notice": extracted.get("blocked_notice", ""),
        "fetch_method": extracted.get("fetch_method", "direct"),
        "fetch_source_url": extracted.get("fetch_source_url", ""),
    }
    if entry["status"] == "success":
        entry["excerpt"] = extract_web_search_excerpt(extracted.get("text") or "")
    elif entry["status"] != "retrying":
        entry["error"] = extracted.get("error") or "Extraction failed."
    return entry


def render_web_search_entry_body(entry: dict, *, call_index: int) -> str:
    reason = html.escape(entry.get("reason") or "Fetching evidence for this argument.")
    url = entry.get("url", "")
    safe_url = html.escape(url, quote=True)
    status = entry.get("status", "failed")
    blocked_notice = entry.get("blocked_notice", "")
    blocked_html = ""
    if blocked_notice:
        blocked_html = (
            f'<div class="debate-web-search-blocked">{html.escape(blocked_notice)}</div>'
        )

    if status == "retrying":
        return (
            f'<div class="debate-web-search-call debate-web-search-call--retrying">'
            f'<div class="debate-web-search-label">web_search · Call {call_index}</div>'
            f'<div class="debate-web-search-reason">{reason}</div>'
            f'<div class="debate-web-search-url">'
            f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{html.escape(url)}</a>'
            f"</div>"
            f"{blocked_html}"
            f"</div>"
        )

    if status == "success":
        title = html.escape(entry.get("title") or "Untitled article")
        excerpt = html.escape(entry.get("excerpt") or "")
        return (
            f'<div class="debate-web-search-call">'
            f'<div class="debate-web-search-label">web_search · Call {call_index}</div>'
            f'<div class="debate-web-search-reason">{reason}</div>'
            f'<div class="debate-web-search-url">'
            f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{html.escape(url)}</a>'
            f"</div>"
            f"{blocked_html}"
            f'<div class="debate-web-search-title">{title}</div>'
            f'<blockquote class="debate-web-search-excerpt">"{excerpt}"</blockquote>'
            f"</div>"
        )

    error = html.escape(entry.get("error") or "Fetch failed.")
    return (
        f'<div class="debate-web-search-call debate-web-search-call--failed">'
        f'<div class="debate-web-search-label">web_search · Call {call_index}</div>'
        f'<div class="debate-web-search-reason">{reason}</div>'
        f'<div class="debate-web-search-url">'
        f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{html.escape(url)}</a>'
        f"</div>"
        f"{blocked_html}"
        f'<div class="debate-web-search-title">Error{LABEL_DASH} {error}</div>'
        f"</div>"
    )


def format_web_search_calls_text(web_searches: list[dict]) -> str:
    if not web_searches:
        return ""

    blocks: list[str] = []
    for index, entry in enumerate(web_searches, start=1):
        reason = entry.get("reason") or "Fetching evidence for this argument."
        url = entry.get("url", "")
        blocked_notice = entry.get("blocked_notice", "")
        notice_lines = [f"[web_search call {index}] {reason}", f"URL{LABEL_DASH} {url}"]
        if blocked_notice:
            notice_lines.append(f"Notice{LABEL_DASH} {blocked_notice}")
        if entry.get("status") == "retrying":
            blocks.append("\n".join(notice_lines))
            continue
        if entry.get("status") == "success":
            title = entry.get("title") or "Untitled article"
            excerpt = entry.get("excerpt") or ""
            notice_lines.extend(
                [
                    f"Title{LABEL_DASH} {title}",
                    f'Excerpt{LABEL_DASH} "{excerpt}"',
                ]
            )
        else:
            error = entry.get("error") or "Fetch failed."
            notice_lines.append(f"Error{LABEL_DASH} {error}")
        blocks.append("\n".join(notice_lines))
    return "\n\n".join(blocks)


def render_web_search_calls_html(web_searches: list[dict]) -> str:
    if not web_searches:
        return ""

    items = [
        render_web_search_entry_body(entry, call_index=index)
        for index, entry in enumerate(web_searches, start=1)
    ]
    return f'<div class="debate-web-search-calls">{"".join(items)}</div>'


def format_web_search_tool_result(extracted: dict) -> str:
    if extracted.get("status") != "success":
        lines = [
            f"Failed to fetch article{LABEL_DASH} {extracted.get('url', '')}",
            f"Error{LABEL_DASH} {extracted.get('error') or 'Extraction failed.'}",
        ]
        if extracted.get("blocked_notice"):
            lines.insert(1, f"Notice{LABEL_DASH} {extracted['blocked_notice']}")
        return "\n".join(lines)

    title = extracted.get("title") or "Untitled article"
    url = extracted.get("url", "")
    text = truncate_at_sentence_boundary(extracted.get("text") or "")
    lines = [f"Title{LABEL_DASH} {title}", f"URL{LABEL_DASH} {url}"]
    if extracted.get("blocked_notice"):
        lines.append(f"Notice{LABEL_DASH} {extracted['blocked_notice']}")
    lines.append(f"Text{LABEL_DASH}\n{text}")
    return "\n".join(lines)


def run_web_search_tool(
    url: str,
    allowed_urls: list[str],
    *,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    if url not in allowed_urls:
        return {
            "url": url,
            "title": "",
            "text": "",
            "status": "failed",
            "error": "That URL is not registered under this agent's web_search tool.",
            "blocked_notice": "",
            "fetch_method": "direct",
            "fetch_source_url": "",
        }
    return extract_article_text(url, on_progress=on_progress)


def append_assistant_tool_message(messages: list[dict], message) -> None:
    tool_calls = []
    for tool_call in message.tool_calls or []:
        tool_calls.append(
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        )

    messages.append(
        {
            "role": "assistant",
            "content": message.content,
            "tool_calls": tool_calls,
        }
    )


def resolve_agent_turn_messages(
    user_prompt: str,
    tool_urls: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    max_tool_rounds: int = MAX_TOOL_CALLS_PER_TURN,
    on_web_search_progress: Callable[[list[dict]], None] | None = None,
) -> tuple[list[dict], list[dict], str | None]:
    """
    Run the tool-calling loop for one debate turn.
    Returns (messages, fetch_log, immediate_content).
  immediate_content is set when the model already returned final text.
    """
    messages: list[dict] = [
        {"role": "system", "content": DEBATE_AGENT_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    tools = build_web_search_tool_definition(tool_urls)
    fetch_log: list[dict] = []

    for round_index in range(max_tool_rounds + 1):
        kwargs: dict = {
            "model": MODEL_ID,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else 0,
        }
        if tools and round_index < max_tool_rounds:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = get_client().chat.completions.create(**kwargs)
        message = response.choices[0].message

        if message.tool_calls:
            append_assistant_tool_message(messages, message)
            for tool_call in message.tool_calls:
                if tool_call.function.name != WEB_SEARCH_TOOL_NAME:
                    continue

                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                requested_url = str(arguments.get("url", "")).strip()
                reason = str(arguments.get("reason", "")).strip()

                def notify_fetch_progress(progress: dict) -> None:
                    if not on_web_search_progress:
                        return
                    on_web_search_progress(
                        fetch_log
                        + [
                            build_web_search_fetch_entry(
                                requested_url,
                                reason,
                                progress,
                            )
                        ]
                    )

                extracted = run_web_search_tool(
                    requested_url,
                    tool_urls,
                    on_progress=notify_fetch_progress,
                )
                fetch_log.append(
                    build_web_search_fetch_entry(
                        requested_url,
                        reason,
                        extracted,
                    )
                )
                if on_web_search_progress:
                    on_web_search_progress(fetch_log)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": format_web_search_tool_result(extracted),
                    }
                )
            continue

        content = (message.content or "").strip()
        if content:
            return messages, fetch_log, content

    return messages, fetch_log, None


def generate_agent_turn(
    user_prompt: str,
    tool_urls: list[str],
    *,
    max_new_tokens: int = 320,
    temperature: float = DEFAULT_AGENT_TEMPERATURE,
) -> tuple[str, list[dict]]:
    messages, fetch_log, immediate_content = resolve_agent_turn_messages(
        user_prompt,
        tool_urls,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    if immediate_content is not None:
        return immediate_content, fetch_log

    response = get_client().chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else 0,
    )
    return (response.choices[0].message.content or "").strip(), fetch_log


def stream_agent_turn(
    user_prompt: str,
    tool_urls: list[str],
    *,
    max_new_tokens: int = 320,
    temperature: float = DEFAULT_AGENT_TEMPERATURE,
    on_web_search_progress: Callable[[list[dict]], None] | None = None,
) -> tuple[list[dict], Iterator[str]]:
    messages, fetch_log, immediate_content = resolve_agent_turn_messages(
        user_prompt,
        tool_urls,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        on_web_search_progress=on_web_search_progress,
    )

    if immediate_content is not None:
        def immediate_tokens() -> Iterator[str]:
            yield from immediate_content

        return fetch_log, immediate_tokens()

    stream = get_client().chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else 0,
        stream=True,
    )

    def token_stream() -> Iterator[str]:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return fetch_log, token_stream()


def temperature_band(value: float) -> str:
    if value <= 0.30:
        return "Focused"
    if value <= 0.65:
        return "Balanced"
    return "Creative"


def reasoning_style_dropdown_label(style: str) -> str:
    return f"{style} — {REASONING_STYLE_DEFINITIONS[style]}"


def reasoning_style_behavior_lines() -> str:
    return "\n".join(
        f"- {style}: {behavior}" for style, behavior in REASONING_STYLE_BEHAVIORS.items()
    )


def is_agent_complete(agent: dict) -> bool:
    normalized = normalize_agent(agent)
    return bool(
        normalized["name"].strip()
        and normalized["position"].strip()
        and normalized["reasoning_style"] in REASONING_STYLES
    )


def get_complete_agents(agents: list[dict]) -> list[dict]:
    return [normalize_agent(agent) for agent in agents if is_agent_complete(agent)]


def validate_agents(agents: list[dict]) -> str | None:
    if len(agents) < MIN_AGENTS_TO_SAVE:
        return f"Complete at least {MIN_AGENTS_TO_SAVE} stakeholder agents before saving."
    if len(agents) > MAX_AGENTS:
        return f"You can define at most {MAX_AGENTS} stakeholder agents."

    names: list[str] = []
    for index, agent in enumerate(agents, start=1):
        normalized = normalize_agent(agent)
        if not normalized["name"].strip():
            return f"Agent {index}{LABEL_DASH} name cannot be empty."
        if not normalized["position"].strip():
            return f"Agent {index}{LABEL_DASH} {POSITION_LABEL.lower()} cannot be empty."
        if normalized["temperature"] < 0.0 or normalized["temperature"] > 1.0:
            return f"Agent {index}{LABEL_DASH} temperature must be between 0.0 and 1.0."
        if not normalized["reasoning_style"]:
            return f"Agent {index}{LABEL_DASH} please select a reasoning style."
        if normalized["reasoning_style"] not in REASONING_STYLES:
            return f"Agent {index}{LABEL_DASH} reasoning style is invalid."
        names.append(normalized["name"].strip())

    if len(set(names)) != len(names):
        return "Agent names must be unique."
    return None


def format_agents_for_prompt(agents: list[dict]) -> str:
    blocks = []
    for agent in agents:
        normalized = normalize_agent(agent)
        style = normalized["reasoning_style"]
        blocks.append(
            f"""Agent{LABEL_DASH} {normalized["name"]}
{POSITION_LABEL}{LABEL_DASH} {normalized["position"]}
Temperature{LABEL_DASH} {normalized["temperature"]}
Reasoning Style{LABEL_DASH} {style}
Reasoning Style Meaning{LABEL_DASH} {REASONING_STYLE_DEFINITIONS[style]}
Reasoning Style Behavior{LABEL_DASH} {REASONING_STYLE_BEHAVIORS[style]}"""
        )
    return "\n\n".join(blocks)


def get_round_spec(round_number: int) -> dict:
    if round_number == 1:
        return {
            "number": 1,
            "title": "Opening Positions",
            "description": "Each agent presents its starting argument.",
            "mode": "opening",
        }
    return {
        "number": round_number,
        "title": "Defense and Critique",
        "description": "Each agent defends their position and criticizes others' points.",
        "mode": "defense",
    }


def build_agent_line_prefix(agent: dict) -> str:
    return f"{agent['name']}{LABEL_DASH} "


def format_structured_turn_line(
    round_number: int,
    agent_name: str,
    text: str,
    *,
    speaking_agent_name: str | None = None,
) -> str:
    you_tag = " [YOU]" if speaking_agent_name and agent_name == speaking_agent_name else ""
    return f"Round {round_number} | Agent: {agent_name}{you_tag} | Statement: {text}"


def format_structured_debate_transcript(
    rounds_data: list[dict],
    speaking_agent_name: str | None = None,
) -> str:
    """Round-by-round transcript: all prior rounds plus n-1 speakers in the current round."""
    if not rounds_data or all(not round_data.get("turns") for round_data in rounds_data):
        return "No prior debate yet."

    lines: list[str] = []
    for round_data in rounds_data:
        round_number = round_data["number"]
        for turn in round_data.get("turns", []):
            web_search_text = format_web_search_calls_text(turn.get("web_searches") or [])
            if web_search_text:
                lines.append(web_search_text)
            lines.append(
                format_structured_turn_line(
                    round_number,
                    turn["agent"],
                    turn["text"],
                    speaking_agent_name=speaking_agent_name,
                )
            )
    return "\n".join(lines)


def build_agent_turn_generation_section(
    agent: dict,
    round_spec: dict,
    general_debate_rules: str,
    other_names_for_bold: str,
    line_prefix: str,
) -> str:
    normalized = normalize_agent(agent)
    reasoning_style = normalized["reasoning_style"]
    agent_name = normalized["name"]
    round_number = round_spec["number"]

    if round_spec["mode"] == "opening":
        round_task = (
            f"Present your opening argument clearly from your position. "
            "State what you believe and why it matters relative to the core theme. "
            "If an earlier speaker in this round repeats a point without adding substance, "
            f"call it out by name and state that they first raised it in Round {round_number}."
        )
    else:
        round_task = (
            "Defend your position. Respond directly to specific points from the debate above, "
            "including anyone who spoke earlier in this round. Add new substance, evidence, or "
            "critique where you can. If another agent misunderstood or misrepresented your "
            "position, call that out clearly and restate your point — but do not recycle the "
            "same argument without responding to what was said. "
            "If another agent repeats a claim without adding substance, call it out by name "
            "and state the round number where they first raised that same point."
        )

    return f"""Reasoning style for this turn{LABEL_DASH} {reasoning_style}
Meaning{LABEL_DASH} {REASONING_STYLE_DEFINITIONS[reasoning_style]}
Behavior{LABEL_DASH} {REASONING_STYLE_BEHAVIORS[reasoning_style]}

Additional debate rules{LABEL_DASH}
{general_debate_rules}

Round task (Round {round_number} — {round_spec["title"]}){LABEL_DASH}
{round_spec["description"]}
{round_task}
Stay aligned with your position and the core theme.
Keep it sharp and discussion-friendly in 2-5 sentences.

Output rules{LABEL_DASH}
- Follow your reasoning style while obeying the debate rules.
- Do NOT repeat the agent name prefix.
- Do NOT use markdown headers.
- Do NOT give a verdict or final answer.
- When referring to another participant by name, wrap their exact name in double asterisks (e.g. {other_names_for_bold}). Do not bold your own name.
- If you used web_search in this turn, you MUST cite the source URL you fetched and include a short verbatim excerpt from the returned text that supports your point (use quotation marks for the excerpt).
- If you call out an opponent for repeating a point, you MUST name the round number where they first made that point (e.g. "Round 1").
- Return ONLY the argument text that would appear after this prefix{LABEL_DASH}
{line_prefix.strip()}"""


def build_agent_turn_prompt(
    scenario: dict,
    selected_agents: list[dict],
    round_spec: dict,
    agent: dict,
    agent_index: int,
    general_debate_rules: str,
    rounds_data: list[dict],
) -> str:
    other_agents = [item for index, item in enumerate(selected_agents) if index != agent_index]
    other_agent_names = [participant["name"] for participant in other_agents]
    other_names_for_bold = (
        ", ".join(f'**{name}**' for name in other_agent_names)
        if other_agent_names
        else "none"
    )
    normalized = normalize_agent(agent)
    line_prefix = build_agent_line_prefix(agent)
    agent_name = normalized["name"]
    round_number = round_spec["number"]
    scenario_name = scenario["scenario_name"]
    position = normalized["position"]
    core_theme = scenario["scenario_core_question"]
    tool_urls = get_agent_tool_urls(agent)
    tools_block = format_tool_urls_for_prompt(tool_urls)
    debate_transcript = format_structured_debate_transcript(rounds_data, agent_name)
    generation_section = build_agent_turn_generation_section(
        agent,
        round_spec,
        general_debate_rules,
        other_names_for_bold,
        line_prefix,
    )

    if round_number == 1:
        debate_scope = (
            f"earlier speakers in round {round_number} only "
            "(no prior rounds yet)"
        )
    else:
        debate_scope = (
            f"rounds 1 through {round_number - 1} in full, "
            f"plus earlier speakers in round {round_number}"
        )

    tool_rules = (
        "Tools for this speaking agent —\n"
        f"{tools_block}\n\n"
        "Tool rules —\n"
        "- Article text is NOT pre-loaded. Call web_search only when you need evidence from a registered URL.\n"
        "- web_search accepts one URL at a time from the registered list above.\n"
        "- After fetching, use the returned text to strengthen your argument or counter opposing claims.\n"
        "- In your spoken turn, cite the URL you fetched and quote a short excerpt from the returned text.\n"
        "- Do not invent facts beyond your position, the scenario, debate history, and web_search results.\n"
        "- If no tools are configured, rely on your position, the scenario, and the debate history only."
    )

    return f"""
You are simulating one turn in a stakeholder debate for a business-school classroom exercise.

Your name is {agent_name}. You are debating in the scenario {scenario_name}. Your position is {position}. You are in round {round_number}.

{tool_rules}

The debate so far ({debate_scope}) —
Lines marked [YOU] are your own prior statements. Review them before speaking and do not repeat them without responding to what others have said.

{debate_transcript}

Given the above setting and your position, stay aligned with the core theme of this debate —
{core_theme}

The rules of the debate are:

To be a good debater:
- Do not repeat your points.
- When calling out repeated points from an opponent, name the round number where they first raised that point.

Now generate your turn accordingly.

{generation_section}
"""


def build_scenario_analysis_markdown(rounds_data: list[dict]) -> str:
    lines = ["# Scenario Analysis", ""]
    for round_data in rounds_data:
        lines.extend(
            [
                f"## Round {round_data['number']} — {round_data['title']}",
                "",
                f"_{round_data['description']}_",
                "",
            ]
        )
        for turn in round_data["turns"]:
            web_search_text = format_web_search_calls_text(turn.get("web_searches") or [])
            if web_search_text:
                lines.extend([web_search_text, ""])
            lines.append(f"{turn['agent']}{LABEL_DASH} {turn['text']}")
            lines.append("")
    return "\n".join(lines).strip()


def agent_order_index(selected_agents: list[dict]) -> dict[str, int]:
    return {agent["name"]: index for index, agent in enumerate(selected_agents)}


def render_agent_roster_html(
    selected_agents: list[dict],
    active_agent: str | None = None,
    completed_agents: list[str] | None = None,
) -> str:
    completed = set(completed_agents or [])
    items: list[str] = []

    for index, agent in enumerate(selected_agents):
        name = agent["name"]
        if name == active_agent:
            status_class = "writing"
            status_label = "Writing"
        elif name in completed:
            status_class = "present"
            status_label = "Present"
        else:
            status_class = "waiting"
            status_label = "Waiting"

        items.append(
            f'<div class="debate-roster-item debate-roster-item--{status_class} '
            f'debate-roster-item--agent-{index + 1}">'
            f'<div class="debate-roster-name">{html.escape(name)}</div>'
            f'<div class="debate-roster-status">{status_label}</div>'
            "</div>"
        )

    return f'<div class="debate-roster">{"".join(items)}</div>'


def format_debate_inline_text(text: str) -> str:
    """Escape debate turn text while rendering **bold** markers as <strong>."""
    if not text:
        return ""
    parts = re.split(r"(\*\*.+?\*\*)", text, flags=re.DOTALL)
    rendered: list[str] = []
    for part in parts:
        if len(part) >= 4 and part.startswith("**") and part.endswith("**"):
            rendered.append(f"<strong>{html.escape(part[2:-2])}</strong>")
        else:
            rendered.append(html.escape(part))
    return "".join(rendered)


def render_debate_turn_html(
    agent_name: str,
    agent_index: int,
    text: str,
    status: str,
    *,
    partial: bool = False,
    web_searches: list[dict] | None = None,
) -> str:
    web_search_html = render_web_search_calls_html(web_searches or [])

    if status == "waiting":
        body = '<div class="debate-turn-text debate-turn-placeholder">Waiting to present...</div>'
    elif partial or not text.strip():
        argument_html = (
            f'<div class="debate-turn-text debate-turn-partial">{format_debate_inline_text(text)}</div>'
            if text
            else '<div class="debate-turn-text debate-turn-placeholder">Preparing response...</div>'
        )
        body = f"{web_search_html}{argument_html}"
    else:
        body = (
            f"{web_search_html}"
            f'<div class="debate-turn-text">{format_debate_inline_text(text)}</div>'
        )

    status_label = {"waiting": "Waiting", "writing": "Writing", "present": "Present"}[status]
    turn_class = f"debate-turn debate-turn--agent-{agent_index + 1} debate-turn--{status}"

    return (
        f'<div class="{turn_class}">'
        f'<div class="debate-turn-agent">'
        f'<span class="debate-turn-name">{html.escape(agent_name)}</span>'
        f'<span class="debate-status debate-status--{status}">{status_label}</span>'
        f"</div>{body}</div>"
    )


def render_debate_rounds_html(
    rounds_data: list[dict],
    selected_agents: list[dict] | None = None,
    live_state: dict | None = None,
) -> str:
    name_to_index = agent_order_index(selected_agents or [])
    cards: list[str] = []

    for round_data in rounds_data:
        round_number = round_data["number"]
        is_live_round = live_state and live_state.get("round_number") == round_number

        turns_html: list[str] = []

        if selected_agents and is_live_round:
            active_agent = live_state.get("active_agent")
            partial_text = live_state.get("partial_text", "")

            for index, agent in enumerate(selected_agents):
                name = agent["name"]
                if name in {turn["agent"] for turn in round_data.get("turns", [])}:
                    turn = next(item for item in round_data["turns"] if item["agent"] == name)
                    turns_html.append(
                        render_debate_turn_html(
                            name,
                            index,
                            turn["text"],
                            "present",
                            web_searches=turn.get("web_searches"),
                        )
                    )
                elif name == active_agent:
                    turns_html.append(
                        render_debate_turn_html(
                            name,
                            index,
                            partial_text,
                            "writing",
                            partial=True,
                            web_searches=live_state.get("web_searches"),
                        )
                    )
                else:
                    turns_html.append(
                        render_debate_turn_html(name, index, "", "waiting")
                    )
        else:
            for turn in round_data.get("turns", []):
                agent_index = name_to_index.get(turn["agent"], 0)
                turns_html.append(
                    render_debate_turn_html(
                        turn["agent"],
                        agent_index,
                        turn["text"],
                        "present",
                        web_searches=turn.get("web_searches"),
                    )
                )

        round_style = "opening" if round_number == 1 else "defense"
        cards.append(
            f'<div class="debate-round-card debate-round-card--{round_style}">'
            f'<div class="debate-round-header">Round {round_number} — '
            f'{html.escape(round_data["title"])}</div>'
            f'<div class="debate-round-desc">{html.escape(round_data["description"])}</div>'
            f'<div class="debate-round-body">{"".join(turns_html)}</div>'
            "</div>"
        )

    return f'<div class="debate-analysis-wrap">{"".join(cards)}</div>'


def stream_scenario_analysis(
    scenario: dict,
    selected_agents: list[dict],
    round_count: int,
    output_area,
    status_area,
    general_debate_rules: str,
) -> tuple[str, list[dict]]:
    """
    Generate an automatic multi-round debate with live token streaming.
    Only the active agent shows Writing; others show Waiting or Present.
    """
    selected_agents = [normalize_agent(agent) for agent in selected_agents]
    rounds_data: list[dict] = []

    status_area.markdown(
        render_agent_roster_html(selected_agents),
        unsafe_allow_html=True,
    )
    output_area.markdown(
        render_debate_rounds_html(rounds_data, selected_agents),
        unsafe_allow_html=True,
    )

    for round_number in range(1, round_count + 1):
        round_spec = get_round_spec(round_number)
        round_entry = {
            "number": round_spec["number"],
            "title": round_spec["title"],
            "description": round_spec["description"],
            "turns": [],
        }
        rounds_data.append(round_entry)

        for agent_index, agent in enumerate(selected_agents):
            agent_name = agent["name"]
            completed_in_round = [turn["agent"] for turn in round_entry["turns"]]

            status_area.markdown(
                render_agent_roster_html(
                    selected_agents,
                    active_agent=agent_name,
                    completed_agents=completed_in_round,
                ),
                unsafe_allow_html=True,
            )
            output_area.markdown(
                render_debate_rounds_html(
                    rounds_data,
                    selected_agents,
                    {
                        "round_number": round_number,
                        "active_agent": agent_name,
                        "partial_text": "",
                        "completed_in_round": completed_in_round,
                    },
                ),
                unsafe_allow_html=True,
            )

            prompt = build_agent_turn_prompt(
                scenario,
                selected_agents,
                round_spec,
                agent,
                agent_index,
                general_debate_rules,
                rounds_data,
            )

            agent_temperature = float(agent.get("temperature", DEFAULT_AGENT_TEMPERATURE))
            tool_urls = get_agent_tool_urls(agent)
            partial = ""
            live_fetch_log: list[dict] = []

            def show_live_web_search_progress(fetch_entries: list[dict]) -> None:
                live_fetch_log.clear()
                live_fetch_log.extend(fetch_entries)
                output_area.markdown(
                    render_debate_rounds_html(
                        rounds_data,
                        selected_agents,
                        {
                            "round_number": round_number,
                            "active_agent": agent_name,
                            "partial_text": partial,
                            "completed_in_round": completed_in_round,
                            "web_searches": live_fetch_log,
                        },
                    ),
                    unsafe_allow_html=True,
                )

            fetch_log, token_stream = stream_agent_turn(
                prompt,
                tool_urls,
                max_new_tokens=320,
                temperature=agent_temperature,
                on_web_search_progress=show_live_web_search_progress,
            )
            if fetch_log:
                output_area.markdown(
                    render_debate_rounds_html(
                        rounds_data,
                        selected_agents,
                        {
                            "round_number": round_number,
                            "active_agent": agent_name,
                            "partial_text": "",
                            "completed_in_round": completed_in_round,
                            "web_searches": fetch_log,
                        },
                    ),
                    unsafe_allow_html=True,
                )
            for token in token_stream:
                partial += token
                time.sleep(len(token) * STREAM_CHAR_DELAY_SEC)
                status_area.markdown(
                    render_agent_roster_html(
                        selected_agents,
                        active_agent=agent_name,
                        completed_agents=completed_in_round,
                    ),
                    unsafe_allow_html=True,
                )
                output_area.markdown(
                    render_debate_rounds_html(
                        rounds_data,
                        selected_agents,
                        {
                            "round_number": round_number,
                            "active_agent": agent_name,
                            "partial_text": partial,
                            "completed_in_round": completed_in_round,
                            "web_searches": fetch_log,
                        },
                    ),
                    unsafe_allow_html=True,
                )

            turn_text = partial.strip()
            round_entry["turns"].append(
                {
                    "agent": agent_name,
                    "text": turn_text,
                    "web_searches": fetch_log,
                }
            )

            completed_in_round = [turn["agent"] for turn in round_entry["turns"]]
            status_area.markdown(
                render_agent_roster_html(
                    selected_agents,
                    active_agent=None,
                    completed_agents=completed_in_round,
                ),
                unsafe_allow_html=True,
            )
            output_area.markdown(
                render_debate_rounds_html(rounds_data, selected_agents),
                unsafe_allow_html=True,
            )

    status_area.empty()
    return build_scenario_analysis_markdown(rounds_data), rounds_data


def run_scenario_analysis(
    scenario: dict,
    selected_agents: list[dict],
    round_count: int,
    general_debate_rules: str,
) -> tuple[str, list[dict]]:
    """
    Generate an automatic multi-round debate among selected stakeholder agents.
    Non-streaming fallback when no UI placeholders are available.
    """
    selected_agents = [normalize_agent(agent) for agent in selected_agents]
    rounds_data: list[dict] = []

    for round_number in range(1, round_count + 1):
        round_spec = get_round_spec(round_number)
        round_entry = {
            "number": round_spec["number"],
            "title": round_spec["title"],
            "description": round_spec["description"],
            "turns": [],
        }
        rounds_data.append(round_entry)

        for agent_index, agent in enumerate(selected_agents):
            prompt = build_agent_turn_prompt(
                scenario,
                selected_agents,
                round_spec,
                agent,
                agent_index,
                general_debate_rules,
                rounds_data,
            )
            agent_temperature = float(agent.get("temperature", DEFAULT_AGENT_TEMPERATURE))
            tool_urls = get_agent_tool_urls(agent)
            argument, fetch_log = generate_agent_turn(
                prompt,
                tool_urls,
                max_new_tokens=320,
                temperature=agent_temperature,
            )
            turn_text = argument.strip()
            round_entry["turns"].append(
                {
                    "agent": agent["name"],
                    "text": turn_text,
                    "web_searches": fetch_log,
                }
            )

    return build_scenario_analysis_markdown(rounds_data), rounds_data


def split_report_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = []
        elif line.startswith("# ") and not line.startswith("## "):
            continue
        elif current_title is not None:
            current_lines.append(line)

    if current_title is not None:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return sections


def parse_agent_summaries(content: str) -> list[dict]:
    summaries: list[dict] = []
    chunks = re.split(r"^#### AGENT:\s*", content, flags=re.MULTILINE)
    for chunk in chunks[1:]:
        name, _, body = chunk.partition("\n")
        summaries.append({"name": name.strip(), "body": body.strip()})
    if summaries:
        return summaries

    chunks = re.split(r"^###\s*", content, flags=re.MULTILINE)
    for chunk in chunks[1:]:
        name, _, body = chunk.partition("\n")
        summaries.append({"name": name.strip(), "body": body.strip()})
    return summaries


def markdown_block_to_html(text: str) -> str:
    if not text.strip():
        return '<p class="final-report-muted">No content generated.</p>'

    items: list[str] = []
    bullet_items: list[str] = []
    numbered_items: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            bullet_items.append(f"<li>{html.escape(stripped[2:])}</li>")
        elif re.match(r"^\d+\.\s", stripped):
            cleaned = re.sub(r"^\d+\.\s*", "", stripped)
            numbered_items.append(f"<li>{html.escape(cleaned)}</li>")
        else:
            bold_match = re.match(r"^\*\*(.+?)\*\*(?:\s+(.*))?$", stripped)
            if bold_match:
                label = bold_match.group(1)
                rest = (bold_match.group(2) or "").strip()
                if rest:
                    items.append(
                        f'<p><span class="final-report-core">{html.escape(label)}</span> '
                        f"{html.escape(rest)}</p>"
                    )
                else:
                    items.append(
                        f'<p class="final-report-core">{html.escape(label)}</p>'
                    )
                continue
            items.append(f"<p>{html.escape(stripped)}</p>")

    if bullet_items:
        items.append(f'<ul class="final-report-list">{"".join(bullet_items)}</ul>')
    if numbered_items:
        items.append(f'<ol class="final-report-list">{"".join(numbered_items)}</ol>')
    return "".join(items) if items else f"<p>{html.escape(text)}</p>"


def parse_markdown_table(content: str) -> tuple[list[str], list[list[str]]]:
    rows = [line for line in content.splitlines() if line.strip().startswith("|")]
    if len(rows) < 2:
        return [], []

    headers = [cell.strip() for cell in rows[0].strip().strip("|").split("|")]
    data: list[list[str]] = []
    for row in rows[2:]:
        if re.match(r"^\|\s*-+", row):
            continue
        data.append([cell.strip() for cell in row.strip().strip("|").split("|")])
    return headers, data


def render_argument_map_html(title: str, content: str) -> str:
    headers, data = parse_markdown_table(content)
    if not headers:
        return (
            f'<div class="final-report-section">'
            f'<div class="final-report-section-title">{html.escape(title)}</div>'
            f'<div class="final-report-section-body">{markdown_block_to_html(content)}</div>'
            "</div>"
        )

    head_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_html = ""
    for row in data:
        cells = row + [""] * (len(headers) - len(row))
        body_html += "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells[: len(headers)]) + "</tr>"

    return (
        f'<div class="final-report-section final-report-map-section">'
        f'<div class="final-report-section-title">{html.escape(title)}</div>'
        f'<div class="final-report-map-wrap">'
        f'<table class="final-report-map-table"><thead><tr>{head_html}</tr></thead>'
        f"<tbody>{body_html}</tbody></table></div></div>"
    )


def render_final_report_html(report_text: str, participating_agents: list[dict]) -> str:
    name_to_index = agent_order_index(participating_agents)
    agent_summary_content = ""
    for title, content in split_report_sections(report_text):
        if "agent summar" in title.lower():
            agent_summary_content = content
            break
    summary_lookup = {
        item["name"]: item["body"] for item in parse_agent_summaries(agent_summary_content)
    }

    parts = ['<div class="final-report-wrap"><div class="final-report-title">Final Report</div>']

    for title, content in split_report_sections(report_text):
        if "agent summar" in title.lower():
            parts.append(
                f'<div class="final-report-section">'
                f'<div class="final-report-section-title">{html.escape(title)}</div>'
                f'<div class="final-report-agent-grid">'
            )
            for agent in participating_agents:
                normalized = normalize_agent(agent)
                agent_index = name_to_index.get(agent["name"], 0)
                body = summary_lookup.get(agent["name"], "")
                parts.append(
                    f'<div class="final-report-agent-card final-report-agent-card--{agent_index + 1}">'
                    f'<div class="final-report-agent-name">{html.escape(normalized["name"])}</div>'
                    f'<div class="final-report-agent-position">{html.escape(normalized["position"])}</div>'
                    f'<div class="final-report-agent-body">{markdown_block_to_html(body)}</div>'
                    "</div>"
                )
            parts.append("</div></div>")
        elif "argument map" in title.lower():
            parts.append(render_argument_map_html(title, content))
        else:
            parts.append(
                f'<div class="final-report-section">'
                f'<div class="final-report-section-title">{html.escape(title)}</div>'
                f'<div class="final-report-section-body">{markdown_block_to_html(content)}</div>'
                "</div>"
            )

    parts.append("</div>")
    return "".join(parts)


def render_final_report_display(report_text: str, participating_agents: list[dict]):
    st.markdown(
        render_final_report_html(report_text, participating_agents),
        unsafe_allow_html=True,
    )


def create_final_report(
    scenario: dict,
    agents: list[dict],
    scenario_analysis_output: str,
    general_debate_rules: str,
) -> str:
    agent_blocks = "\n\n".join(
        f"#### AGENT: {agent['name']}\n"
        f"**Core argument{LABEL_DASH}** ...\n"
        f"- Important point 1\n"
        f"- Important point 2\n"
        f"- Important point 3"
        for agent in agents
    )

    prompt = f"""
You are a classroom moderator preparing a final report after a full stakeholder debate.

The moderator is not a judge.
The moderator does not decide the answer.
The moderator reads the entire debate and synthesizes what each party argued.

Scenario{LABEL_DASH}
{scenario["scenario_name"]}

Category{LABEL_DASH}
{scenario["scenario_category"]}

Core theme{LABEL_DASH}
{scenario["scenario_core_question"]}

Stakeholder agents{LABEL_DASH}
{format_agents_for_prompt(agents)}

General Rules / Hard Rules of Debate{LABEL_DASH}
{general_debate_rules}

Full scenario analysis debate{LABEL_DASH}
{scenario_analysis_output or "No debate generated yet."}

Task{LABEL_DASH}
Create a Final Report based only on the debate above.

Do NOT label this as SWOT.
Do NOT use Strengths / Weaknesses / Opportunities / Threats sections.
Do NOT give a verdict.
Do NOT use Launch / Pivot / Kill.
Do NOT recommend one final answer.

Return exactly this format:

# Final Report

## Scenario
Brief classroom-ready summary of what was debated.

## Agent Summaries
{agent_blocks}

## Argument Map
| Tension | Agents involved | Why it matters |
|---|---|---|
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |

## Strongest Unresolved Questions
1. ...
2. ...
3. ...

## Moderator Closing Note
...

Rules{LABEL_DASH}
- Return only the final report.
- For every participating agent, fill in their Agent Summary with their actual argument from the debate.
- Agent summaries must contain a core argument and 3 important points.
- Argument Map must contain at least 3 tensions from the debate.
- Keep it classroom-friendly and presentation-ready.
- Use em dashes (—), not colons, as label separators in the output.
"""
    return ask_model(prompt, max_new_tokens=1200, temperature=0.35)


def create_moderator_swot(
    scenario: dict,
    agents: list[dict],
    scenario_analysis_output: str,
    general_debate_rules: str = DEFAULT_GENERAL_DEBATE_RULES,
) -> str:
    """Backward-compatible alias for final report generation."""
    return create_final_report(
        scenario,
        agents,
        scenario_analysis_output,
        general_debate_rules,
    )


def format_agents_export_section(agents: list[dict]) -> str:
    blocks: list[str] = []
    for agent in agents:
        normalized = normalize_agent(agent)
        style = normalized["reasoning_style"]
        blocks.append(
            f"""Agent{LABEL_DASH} {normalized["name"]}
{POSITION_LABEL}{LABEL_DASH} {normalized["position"]}
Temperature{LABEL_DASH} {normalized["temperature"]}
Reasoning Style{LABEL_DASH} {style}
Reasoning Style Meaning{LABEL_DASH} {REASONING_STYLE_DEFINITIONS[style]}
Reasoning Style Behavior{LABEL_DASH} {REASONING_STYLE_BEHAVIORS[style]}
Tools{LABEL_DASH} web_search
Registered URLs{LABEL_DASH}
{normalized["tool_urls"].strip() or "None"}"""
        )

    return "\n\n".join(blocks)


def build_export_markdown(
    scenario: dict,
    agents: list[dict],
    scenario_analysis_output: str | None,
    moderator_swot: str | None,
    general_debate_rules: str | None = None,
) -> str:
    debate_rules = general_debate_rules or DEFAULT_GENERAL_DEBATE_RULES
    sections = [
        "# Scenario Swarm Export",
        "",
        "## Scenario",
        f"**{scenario['scenario_name']}**",
        "",
        f"**Category{LABEL_DASH}** {scenario['scenario_category']}",
        f"**Core theme{LABEL_DASH}** {scenario['scenario_core_question']}",
        "",
        "## Debate Configuration",
        "",
        f"**General Rules / Hard Rules of Debate{LABEL_DASH}**",
        debate_rules,
        "",
        "## Stakeholder Agents",
        "",
        format_agents_export_section(agents),
        "",
    ]

    if scenario_analysis_output:
        sections.extend(["---", "", scenario_analysis_output, ""])

    if moderator_swot:
        sections.extend(["---", "", moderator_swot, ""])

    return "\n".join(sections)


def reset_workflow_after_scenario_change():
    st.session_state.custom_agents = []
    st.session_state.scenario_analysis_output = None
    st.session_state.scenario_analysis_rounds = []
    st.session_state.debate_round_count = None
    st.session_state.selected_debate_agents = []
    st.session_state.moderator_swot = None
    st.session_state.completed_steps.discard("agents")
    st.session_state.completed_steps.discard("analysis")
    st.session_state.completed_steps.discard("swot")
    if "agent_draft" in st.session_state:
        del st.session_state.agent_draft


def clear_scenario_analysis():
    st.session_state.scenario_analysis_output = None
    st.session_state.scenario_analysis_rounds = []
    st.session_state.moderator_swot = None
    st.session_state.completed_steps.discard("analysis")
    st.session_state.completed_steps.discard("swot")


def init_session_state():
    defaults = {
        "active_step": "setup",
        "completed_steps": set(),
        "selected_idea": list(BUSINESS_IDEAS.keys())[0],
        "scenario_mode": "preset",
        "custom_scenario_title": "",
        "custom_scenario_category": "",
        "custom_scenario_core_question": "",
        "scenario": None,
        "custom_agents": [],
        "scenario_analysis_output": None,
        "scenario_analysis_rounds": [],
        "debate_round_count": None,
        "selected_debate_agents": [],
        "moderator_swot": None,
        "general_debate_rules": DEFAULT_GENERAL_DEBATE_RULES,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    step_migrations = {
        "argument": "analysis",
        "map": "swot",
        "brief": "swot",
        "shift": "analysis",
    }
    active_step = st.session_state.get("active_step")
    if active_step in step_migrations:
        st.session_state.active_step = step_migrations[active_step]

    if "scenario_analysis_output" not in st.session_state:
        st.session_state.scenario_analysis_output = None
    if "scenario_analysis_rounds" not in st.session_state:
        st.session_state.scenario_analysis_rounds = []
    if "debate_round_count" not in st.session_state:
        st.session_state.debate_round_count = None
    if "selected_debate_agents" not in st.session_state:
        st.session_state.selected_debate_agents = []
    if "custom_agents" not in st.session_state:
        st.session_state.custom_agents = []
    if "general_debate_rules" not in st.session_state:
        st.session_state.general_debate_rules = DEFAULT_GENERAL_DEBATE_RULES

    if "argument_sets" in st.session_state and st.session_state.argument_sets:
        if not st.session_state.scenario_analysis_output:
            st.session_state.scenario_analysis_output = "\n\n---\n\n".join(
                argument_set["output"] for argument_set in st.session_state.argument_sets
            )
        del st.session_state.argument_sets

    completed = st.session_state.completed_steps
    if "argument" in completed:
        completed.add("analysis")
    if "map" in completed or "brief" in completed:
        completed.add("analysis")
        completed.add("swot")


def mark_complete(step_id: str):
    st.session_state.completed_steps.add(step_id)


def is_step_accessible(step_id: str) -> bool:
    step_index = STEP_ORDER.index(step_id)
    for previous_step in STEP_ORDER[:step_index]:
        if previous_step not in st.session_state.completed_steps:
            return False
    return True


def step_status(step_id: str) -> str:
    if step_id in st.session_state.completed_steps:
        return "completed"
    if step_id == st.session_state.active_step:
        return "current"
    if not is_step_accessible(step_id):
        return "locked"
    return "pending"


def render_header():
    st.markdown(
        f"""
        <div class="page-header">
          <div class="header-inner">
            <div class="header-logo">
              <img
                src="{BITSoM_LOGO_URL}"
                class="custom-logo"
                alt="BITSoM Logo"
                decoding="async"
              >
            </div>
            <div class="header-text">
              <h1>Scenario Swarm</h1>
              <p>Design stakeholder agents, run automatic debate rounds, and receive a moderator final report.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_block(title: str, subtitle: str, step_number: int | None = None, body: str = ""):
    number_html = f'<div class="block-number">{step_number}</div>' if step_number else ""
    body_html = f'<div class="block-body">{body}</div>' if body else ""
    st.markdown(
        f"""
        <div class="block">
          <div class="block-header">
            {number_html}
            <div>
              <div class="block-title">{title}</div>
              <div class="block-sub">{subtitle}</div>
            </div>
          </div>
          {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-steps-capsule">
              <span class="sidebar-steps-capsule-label">Steps</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for index, (step_id, title, desc) in enumerate(SIMULATION_STEPS, start=1):
            status = step_status(step_id)
            accessible = is_step_accessible(step_id)
            marker = "✓" if status == "completed" else str(index)

            with st.container(key=f"sidebar_step_{step_id}"):
                st.markdown(
                    f"""
                    <div class="sidebar-step-inner sidebar-step-inner--{status}">
                      <div class="timeline-item {status}">
                        <div class="timeline-marker {status}">{marker}</div>
                        <div>
                          <div class="timeline-title">{title}</div>
                          <div class="timeline-desc">{desc}</div>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if accessible:
                    if st.button(
                        title,
                        key=f"nav_{step_id}",
                        use_container_width=True,
                        type="primary" if step_id == st.session_state.active_step else "secondary",
                    ):
                        st.session_state.active_step = step_id
                        st.rerun()
                else:
                    st.markdown(
                        f'<div class="step-nav-locked">{title}</div>',
                        unsafe_allow_html=True,
                    )

        st.divider()
        st.markdown(f"**Model{LABEL_DASH}** `{MODEL_ID}`")
        if st.session_state.scenario:
            st.markdown(f"**Idea{LABEL_DASH}** {st.session_state.scenario['scenario_name']}")


def render_scenario_preview_card(
    header_label: str,
    badge_text: str,
    scenario_name: str,
    scenario_category: str,
    scenario_core_question: str,
):
    st.markdown(
        f"""
        <div class="scenario-preview-card">
          <div class="scenario-preview-header">
            <span>{html.escape(header_label)}</span>
            <span class="gold-badge">{html.escape(badge_text)}</span>
          </div>
          <div class="scenario-preview-body">
            <div class="block-title">{html.escape(scenario_name)}</div>
            <span class="gold-badge">{html.escape(scenario_category)}</span>
            <div class="core-question-line">
              <strong>Core theme{LABEL_DASH}</strong> {html.escape(scenario_core_question)}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scenario_card(scenario: dict):
    st.markdown(
        f"""
        <div class="block">
          <div class="block-header">
            <div class="block-number">i</div>
            <div>
              <div class="block-title">Selected Simulation</div>
              <div class="block-sub">Current scenario configuration</div>
            </div>
          </div>
          <div class="block-body">
            <div class="meta-grid">
              <div class="meta-item"><label>Business Idea</label><strong>{scenario['scenario_name']}</strong></div>
              <div class="meta-item"><label>Category</label><strong>{scenario['scenario_category']}</strong></div>
              <div class="meta-item"><label>Model</label><strong>{MODEL_ID}</strong></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"**Core theme{LABEL_DASH}** {scenario['scenario_core_question']}")


def require_scenario() -> bool:
    if not st.session_state.scenario:
        st.warning("Complete **Scenario Setting** first.")
        return False
    return True


def require_agents() -> bool:
    if not st.session_state.custom_agents:
        st.warning("Complete **Agent Design** first.")
        return False
    return True


def render_tool_urls_summary(tool_urls: str):
    urls, _ = parse_tool_urls(tool_urls)
    if not urls:
        return

    st.markdown("**web_search** — registered URLs")
    for index, url in enumerate(urls, start=1):
        st.caption(f"URL {index}{LABEL_DASH} {url}")


def render_agents_table(
    agents: list[dict],
    *,
    name_header: str = "Name",
    position_header: str = POSITION_LABEL,
):
    rows = []
    for agent in agents:
        normalized = normalize_agent(agent)
        rows.append(
            "<tr>"
            f"<td>{html.escape(normalized['name'])}</td>"
            f"<td>{html.escape(normalized['position'])}</td>"
            f"<td>{normalized['temperature']:.2f}</td>"
            f"<td>{html.escape(normalized['reasoning_style'])}</td>"
            f"<td>{html.escape(tool_urls_summary_label(normalized))}</td>"
            "</tr>"
        )

    st.markdown(
        f"""
        <div class="agents-table-wrap">
          <table class="agents-table">
            <thead>
              <tr>
                <th>{html.escape(name_header)}</th>
                <th>{html.escape(position_header)}</th>
                <th>Temperature</th>
                <th>Reasoning Style</th>
                <th>Tools</th>
              </tr>
            </thead>
            <tbody>
              {"".join(rows)}
            </tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_debate_agents_table(agents: list[dict]):
    if not agents:
        return
    render_agents_table(
        agents,
        name_header="Name of the Agent",
        position_header=f"{POSITION_LABEL} of the agent",
    )


def render_debate_rounds_display(
    rounds_data: list[dict],
    selected_agents: list[dict] | None = None,
):
    if not rounds_data:
        return
    st.markdown(
        render_debate_rounds_html(rounds_data, selected_agents),
        unsafe_allow_html=True,
    )


def render_custom_agents_summary(agents: list[dict]):
    for agent in agents:
        normalized = normalize_agent(agent)
        st.markdown(
            f"""
            <div class="agent-card">
              <h4>{html.escape(normalized["name"])}</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"**{POSITION_LABEL}** {normalized['position']}")
        st.markdown(
            f"**Settings{LABEL_DASH}** Temperature {normalized['temperature']:.2f}, "
            f"Reasoning Style {normalized['reasoning_style']}"
        )
        tool_count = len(get_agent_tool_urls(normalized))
        if tool_count == 0:
            tools_summary = "None"
        else:
            tools_summary = f"web_search ({tool_count} URL{'s' if tool_count != 1 else ''})"
        st.markdown(f"**Tools{LABEL_DASH}** {tools_summary}")


def render_setup_step(step_number: int):
    render_block(
        STEP_LABELS["setup"],
        STEP_DESCRIPTIONS["setup"],
        step_number,
    )

    scenario_options = ["Use preset scenario", "Create my own scenario"]
    default_option = (
        "Use preset scenario"
        if st.session_state.scenario_mode == "preset"
        else "Create my own scenario"
    )
    render_field_label("Scenario source")
    scenario_source = st.pills(
        "Scenario source",
        options=scenario_options,
        default=default_option,
        selection_mode="single",
        key="scenario_source_pills",
        label_visibility="collapsed",
    )
    if scenario_source is None:
        scenario_source = default_option
    st.session_state.scenario_mode = (
        "preset" if scenario_source == "Use preset scenario" else "custom"
    )

    scenario_name = ""
    scenario_category = ""
    scenario_core_question = ""

    if st.session_state.scenario_mode == "preset":
        render_field_label("Preset business scenario")
        selected_idea = st.selectbox(
            "Preset business scenario",
            options=list(BUSINESS_IDEAS.keys()),
            index=list(BUSINESS_IDEAS.keys()).index(st.session_state.selected_idea),
            key="preset_business_scenario",
            label_visibility="collapsed",
        )
        idea = BUSINESS_IDEAS[selected_idea]
        scenario_name = selected_idea
        scenario_category = idea["category"]
        scenario_core_question = idea["core_question"]

        render_scenario_preview_card(
            "Preset Preview",
            "Selected Scenario",
            selected_idea,
            idea["category"],
            idea["core_question"],
        )
    else:
        render_panel_header(
            "Create Your Scenario",
            "Custom Entry",
            CUSTOM_SCENARIO_DEFINITION,
        )
        st.info(
            """
            **How to complete each field**

            1. **Scenario Title** — a short, clear name for the situation (e.g. AI-Allowed Take-Home Exams).
            2. **Broad business category** — the domain or area it sits in (e.g. Academic Policy / AI Ethics / Assessment Design).
            3. **Core theme** — the central tension stakeholders should debate (e.g. Should AI use in exams be banned, allowed with disclosure, or integrated into assessment design?).

            **Weak core theme —** AI in exams

            **Strong core theme —** Should students be allowed to use AI tools for take-home exams if they disclose how AI was used, given faculty concerns about integrity and student arguments that AI use is now a managerial skill?
            """
        )
        render_field_label("Scenario Title")
        scenario_name = st.text_input(
            "Scenario Title",
            value=st.session_state.custom_scenario_title,
            placeholder="e.g. AI-Allowed Take-Home Exams",
            label_visibility="collapsed",
        )
        render_field_label("Broad business category")
        scenario_category = st.text_input(
            "Broad business category",
            value=st.session_state.custom_scenario_category,
            placeholder="e.g. Academic Policy / AI Ethics / Assessment Design",
            label_visibility="collapsed",
        )
        render_field_label("Core theme")
        scenario_core_question = st.text_input(
            "Core theme",
            value=st.session_state.custom_scenario_core_question,
            placeholder=(
                "e.g. Should AI use in exams be banned, allowed with disclosure, "
                "or integrated into assessment design?"
            ),
            label_visibility="collapsed",
        )

        if (
            validate_custom_scenario(
                scenario_name,
                scenario_category,
                scenario_core_question,
            )
            is None
        ):
            render_scenario_preview_card(
                "Custom Preview",
                "Your Design",
                scenario_name.strip(),
                scenario_category.strip(),
                scenario_core_question.strip(),
            )

    if st.button("Save scenario", type="primary", use_container_width=True):
        if st.session_state.scenario_mode == "custom":
            validation_error = validate_custom_scenario(
                scenario_name,
                scenario_category,
                scenario_core_question,
            )
            if validation_error:
                st.warning(validation_error)
                return

            st.session_state.custom_scenario_title = scenario_name.strip()
            st.session_state.custom_scenario_category = scenario_category.strip()
            st.session_state.custom_scenario_core_question = scenario_core_question.strip()

            st.session_state.scenario = build_scenario(
                scenario_name=st.session_state.custom_scenario_title,
                scenario_description="",
                scenario_category=st.session_state.custom_scenario_category,
                scenario_core_question=st.session_state.custom_scenario_core_question,
            )
        else:
            st.session_state.selected_idea = scenario_name
            st.session_state.scenario = get_scenario_from_preset(scenario_name)

        reset_workflow_after_scenario_change()
        mark_complete("setup")
        st.session_state.active_step = "agents"
        st.success("Scenario saved. Continue to Agent Design.")
        st.rerun()


def render_agent_design_step(step_number: int):
    if not require_scenario():
        return

    render_scenario_card(st.session_state.scenario)
    render_block(STEP_LABELS["agents"], STEP_DESCRIPTIONS["agents"], step_number)

    st.markdown("### Create Your Stakeholder Agents")
    st.caption(
        "Use **Add stakeholder 1**, **Add stakeholder 2**, and so on to open each agent form. "
        "Complete the current stakeholder (name, position, and reasoning style) before adding the next. "
        f"Save when at least {MIN_AGENTS_TO_SAVE} agents are complete (maximum {MAX_AGENTS})."
    )

    if "agent_draft" not in st.session_state:
        st.session_state.agent_draft = (
            [normalize_agent(agent) for agent in st.session_state.custom_agents]
            if st.session_state.custom_agents
            else []
        )

    if not st.session_state.agent_draft:
        st.info("Click **Add stakeholder 1** to design your first agent.")

    for index, agent in enumerate(st.session_state.agent_draft):
        label = agent["name"].strip() or f"Stakeholder {index + 1}"
        status = "Complete" if is_agent_complete(agent) else "Incomplete"
        agent_number = index + 1
        with st.container(border=True, key=f"agent_design_card_{index}"):
            st.markdown(
                f'<p class="agent-design-title agent-design-title--{agent_number}">'
                f"Agent {agent_number}{LABEL_DASH} {html.escape(label)} "
                f'<span class="gold-badge">{status}</span></p>',
                unsafe_allow_html=True,
            )
            render_field_label("Agent name")
            st.session_state.agent_draft[index]["name"] = st.text_input(
                "Agent name",
                value=agent["name"],
                placeholder="e.g. Faculty member concerned about academic integrity",
                key=f"agent_name_{index}",
                label_visibility="collapsed",
            )
            render_field_label(POSITION_LABEL, trailing_dash=False)
            st.session_state.agent_draft[index]["position"] = st.text_area(
                POSITION_LABEL,
                value=agent["position"],
                placeholder=(
                    "Briefly explain what this agent believes, argues, and pushes back against."
                ),
                key=f"agent_position_{index}",
                height=100,
                label_visibility="collapsed",
            )

            st.markdown("**Agent Settings**")
            st.caption("These settings control how this agent speaks during the debate.")

            normalized = normalize_agent(st.session_state.agent_draft[index])
            render_field_label("Temperature")
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=float(normalized["temperature"]),
                help=(
                    "Lower values make the agent more focused and predictable. "
                    "Higher values make the agent more varied and creative."
                ),
                key=f"agent_temperature_{index}",
                label_visibility="collapsed",
            )
            st.session_state.agent_draft[index]["temperature"] = temperature
            st.caption(
                "0.00–0.30: Focused · 0.35–0.65: Balanced · 0.70–1.00: Creative · "
                f"**{temperature_band(temperature)}**"
            )

            current_style = st.session_state.agent_draft[index].get("reasoning_style", "")
            style_index = (
                REASONING_STYLES.index(current_style)
                if current_style in REASONING_STYLES
                else None
            )
            render_field_label("Reasoning Style")
            reasoning_style = st.selectbox(
                "Reasoning Style",
                options=REASONING_STYLES,
                index=style_index,
                format_func=reasoning_style_dropdown_label,
                placeholder=(
                    "Select from the dropdown to select the style of this agent while debating"
                ),
                key=f"agent_reasoning_style_{index}",
                label_visibility="collapsed",
            )
            if reasoning_style:
                st.session_state.agent_draft[index]["reasoning_style"] = reasoning_style

            render_field_label("Tools")
            with st.container(border=True):
                st.markdown("**web_search**")
                st.caption(
                    f"One search tool for this agent. Register up to {MAX_TOOL_URLS_PER_AGENT} "
                    "article URLs it can fetch from during the debate. The agent calls "
                    "web_search only when it needs evidence — nothing is loaded upfront."
                )

                url_slots = split_tool_urls(
                    agent.get("tool_urls") or agent.get("supporting_urls", "")
                )
                updated_slots: list[str] = []
                for slot_index in range(MAX_TOOL_URLS_PER_AGENT):
                    updated_slots.append(
                        st.text_input(
                            f"Registered URL {slot_index + 1}",
                            value=url_slots[slot_index],
                            placeholder="https://example.com/article",
                            key=f"agent_tool_url_{index}_{slot_index}",
                            label_visibility="collapsed",
                        )
                    )

                st.session_state.agent_draft[index]["tool_urls"] = join_tool_urls(updated_slots)

            render_tool_urls_summary(
                st.session_state.agent_draft[index].get("tool_urls", "")
            )

    complete_agents = get_complete_agents(st.session_state.agent_draft)
    agents_completed = len(complete_agents)
    forms_open = len(st.session_state.agent_draft)
    current_stakeholder_complete = (
        not st.session_state.agent_draft
        or is_agent_complete(st.session_state.agent_draft[-1])
    )
    can_add_agent = forms_open < MAX_AGENTS and current_stakeholder_complete
    can_save_agents = agents_completed >= MIN_AGENTS_TO_SAVE

    st.markdown(
        f"**Agents completed{LABEL_DASH}** {agents_completed}/{MAX_AGENTS} "
        f"({MIN_AGENTS_TO_SAVE} required to save)"
    )

    button_cols = st.columns(2)
    next_stakeholder_number = len(st.session_state.agent_draft) + 1
    if forms_open >= MAX_AGENTS:
        add_button_label = "Maximum stakeholders reached"
        add_button_key = "add_stakeholder_max_button"
    elif not current_stakeholder_complete:
        add_button_label = "Complete the above agent design before adding the next one"
        add_button_key = f"add_stakeholder_{next_stakeholder_number}_button"
    else:
        add_button_label = f"Add stakeholder {next_stakeholder_number}"
        add_button_key = f"add_stakeholder_{next_stakeholder_number}_button"
    with button_cols[0]:
        if st.button(
            add_button_label,
            use_container_width=True,
            disabled=not can_add_agent,
            key=add_button_key,
            help=(
                f"Open the form for stakeholder {next_stakeholder_number}"
                if can_add_agent
                else (
                    f"Maximum of {MAX_AGENTS} stakeholders reached"
                    if forms_open >= MAX_AGENTS
                    else "Complete the above agent design before adding the next one"
                )
            ),
        ):
            st.session_state.agent_draft.append(empty_agent())
            st.rerun()
        if forms_open >= MAX_AGENTS:
            st.caption(f"Maximum of {MAX_AGENTS} stakeholders reached.")
    with button_cols[1]:
        if st.button(
            "Save agents",
            type="primary",
            use_container_width=True,
            disabled=not can_save_agents,
            key="save_agents_button",
            help=f"Available after at least {MIN_AGENTS_TO_SAVE} completed stakeholders",
        ):
            validation_error = validate_agents(complete_agents)
            if validation_error:
                st.warning(validation_error)
                return

            st.session_state.custom_agents = complete_agents
            mark_complete("agents")
            st.session_state.active_step = "analysis"
            st.success("Agents saved. Continue to Scenario Analysis.")
            st.rerun()
        if not can_save_agents:
            st.caption(
                f"Complete at least {MIN_AGENTS_TO_SAVE} agents to enable save "
                f"({agents_completed}/{MIN_AGENTS_TO_SAVE})."
            )


def render_analysis_step(step_number: int):
    if not require_scenario() or not require_agents():
        return

    render_scenario_card(st.session_state.scenario)
    render_block(STEP_LABELS["analysis"], STEP_DESCRIPTIONS["analysis"], step_number)

    agent_names = [agent["name"] for agent in st.session_state.custom_agents]
    default_selected = st.session_state.selected_debate_agents or agent_names
    default_selected = [name for name in default_selected if name in agent_names]

    if st.session_state.scenario_analysis_output:
        participating_agents = [
            agent
            for agent in st.session_state.custom_agents
            if agent["name"] in st.session_state.selected_debate_agents
        ]
        render_debate_agents_table(participating_agents)

        rounds_run = st.session_state.debate_round_count
        st.markdown(f"**Debate rounds selected{LABEL_DASH}** {rounds_run}/{MAX_DEBATE_ROUNDS}")
        if st.session_state.scenario_analysis_rounds:
            render_debate_rounds_display(
                st.session_state.scenario_analysis_rounds,
                participating_agents,
            )
        else:
            st.markdown(st.session_state.scenario_analysis_output)

        if st.button("Re-run Scenario Analysis", type="secondary", use_container_width=True):
            clear_scenario_analysis()
            st.rerun()

        if "analysis" not in st.session_state.completed_steps:
            mark_complete("analysis")

        st.info("Proceed to **Final Report** to synthesize the debate.")
        return

    render_field_label("Select agents for the debate")
    selected_names = st.multiselect(
        "Select agents for the debate",
        options=agent_names,
        default=default_selected or agent_names,
        label_visibility="collapsed",
    )
    selected_agents = [
        agent for agent in st.session_state.custom_agents if agent["name"] in selected_names
    ]
    if selected_agents:
        render_debate_agents_table(selected_agents)

    render_field_label("Number of debate rounds")
    round_options = list(range(MIN_DEBATE_ROUNDS, MAX_DEBATE_ROUNDS + 1))
    saved_round = st.session_state.get("debate_round_count")
    default_index = (
        round_options.index(int(saved_round))
        if saved_round in round_options
        else None
    )
    round_count = st.selectbox(
        "Number of debate rounds",
        options=round_options,
        index=default_index,
        placeholder="Select rounds (1 to 4)",
        format_func=lambda value: f"{value} round{'s' if value > 1 else ''}",
        label_visibility="collapsed",
        key="debate_round_select",
    )
    if round_count:
        st.markdown(
            f"**Debate rounds selected{LABEL_DASH}** {round_count}/{MAX_DEBATE_ROUNDS}"
        )

    render_field_label("General Rules / Hard Rules of Debate")
    st.session_state.general_debate_rules = st.text_area(
        "General Rules / Hard Rules of Debate",
        value=st.session_state.general_debate_rules,
        placeholder=(
            "Example: Agents must challenge weak assumptions, avoid agreeing too quickly, "
            "stay respectful, and avoid giving a final decision."
        ),
        height=120,
        help="These rules apply to all agents during the debate.",
        label_visibility="collapsed",
        key="general_debate_rules_input",
    )

    if st.button("Run Scenario Analysis", type="primary", use_container_width=True):
        if len(selected_names) < 2:
            st.warning("Select at least 2 agents for the debate.")
            return
        if not round_count:
            st.warning("Select the number of debate rounds (1 to 4).")
            return
        if round_count < MIN_DEBATE_ROUNDS or round_count > MAX_DEBATE_ROUNDS:
            st.warning(f"Number of rounds must be between {MIN_DEBATE_ROUNDS} and {MAX_DEBATE_ROUNDS}.")
            return

        st.markdown("### Scenario Analysis")
        render_debate_agents_table(selected_agents)
        debate_container = st.container(border=True)
        with debate_container:
            output_area = st.empty()
            status_area = st.empty()

            output, rounds_data = stream_scenario_analysis(
                scenario=st.session_state.scenario,
                selected_agents=selected_agents,
                round_count=int(round_count),
                output_area=output_area,
                status_area=status_area,
                general_debate_rules=st.session_state.general_debate_rules,
            )

        st.session_state.scenario_analysis_output = output
        st.session_state.scenario_analysis_rounds = rounds_data
        st.session_state.debate_round_count = int(round_count)
        st.session_state.selected_debate_agents = selected_names
        st.session_state.moderator_swot = None
        st.session_state.completed_steps.discard("swot")
        mark_complete("analysis")
        st.rerun()


def get_debate_participating_agents() -> list[dict]:
    selected_names = st.session_state.get("selected_debate_agents") or []
    if not selected_names:
        return st.session_state.custom_agents
    return [
        agent
        for agent in st.session_state.custom_agents
        if agent["name"] in selected_names
    ]


def render_swot_step(step_number: int):
    if not require_scenario() or not require_agents():
        return
    if not st.session_state.scenario_analysis_output:
        st.warning("Run **Scenario Analysis** first.")
        return

    render_block(STEP_LABELS["swot"], STEP_DESCRIPTIONS["swot"], step_number)
    participating_agents = get_debate_participating_agents()

    if st.session_state.moderator_swot:
        render_final_report_display(st.session_state.moderator_swot, participating_agents)

        export_text = build_export_markdown(
            st.session_state.scenario,
            st.session_state.custom_agents,
            st.session_state.scenario_analysis_output,
            st.session_state.moderator_swot,
            st.session_state.general_debate_rules,
        )
        st.download_button(
            "Download full export (Markdown)",
            data=export_text,
            file_name="scenario_swarm_export.md",
            mime="text/markdown",
            use_container_width=True,
        )

        if st.button("Re-generate Final Report", type="secondary"):
            st.session_state.moderator_swot = None
            st.rerun()
        return

    if st.button("Generate Final Report", type="primary", use_container_width=True):
        with st.spinner("Moderator is preparing the final report..."):
            st.session_state.moderator_swot = create_final_report(
                st.session_state.scenario,
                participating_agents,
                st.session_state.scenario_analysis_output,
                st.session_state.general_debate_rules,
            )
        mark_complete("swot")
        st.rerun()


def render_results_tabs():
    if st.session_state.active_step in ("setup", "analysis", "swot"):
        return

    has_saved_work = (
        "agents" in st.session_state.completed_steps
        or st.session_state.scenario_analysis_output
        or st.session_state.moderator_swot
    )
    if not has_saved_work:
        return

    st.divider()
    tabs = st.tabs(["Agents", "Scenario Analysis", "Final Report"])

    with tabs[0]:
        if st.session_state.custom_agents:
            render_agents_table(st.session_state.custom_agents)
            render_custom_agents_summary(st.session_state.custom_agents)
        else:
            st.caption("No agents saved yet.")

    with tabs[1]:
        if st.session_state.scenario_analysis_rounds:
            render_debate_agents_table(get_debate_participating_agents())
            render_debate_rounds_display(
                st.session_state.scenario_analysis_rounds,
                get_debate_participating_agents(),
            )
        elif st.session_state.scenario_analysis_output:
            st.markdown(st.session_state.scenario_analysis_output)
        else:
            st.caption("No scenario analysis yet.")

    with tabs[2]:
        if st.session_state.moderator_swot:
            render_final_report_display(
                st.session_state.moderator_swot,
                get_debate_participating_agents(),
            )
        else:
            st.caption("No final report yet.")


def main():
    st.set_page_config(
        page_title="Scenario Swarm | BITSoM",
        page_icon="MS",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()
    render_header()
    render_sidebar()

    step = st.session_state.active_step
    step_renderers = {
        "setup": render_setup_step,
        "agents": render_agent_design_step,
        "analysis": render_analysis_step,
        "swot": render_swot_step,
    }
    if step not in step_renderers:
        st.session_state.active_step = "setup"
        step = "setup"

    step_number = STEP_ORDER.index(step) + 1
    step_renderers[step](step_number)
    render_results_tabs()


if __name__ == "__main__":
    main()
