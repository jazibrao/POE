# Portfolio Optimization Engine - Enterprise Edition
# Version: 3.0.0
# Copyright (c) 2026 Servis Retail. All rights reserved.
#
# A premium, enterprise-grade SKU optimization solution built to
# Microsoft/Google quality standards for retail merchandising teams.

# Main application module.
#
# This package supports both:
#   - Streamlit UI mode (app.py)
#   - CLI mode (cli_engine.py)
#
# For CLI runs, Streamlit may not be installed (or intentionally omitted).
# We therefore import Streamlit defensively and provide a tiny stub for the
# few decorators used at import-time (e.g., @st.cache_data).

try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    class _StreamlitStub:  # minimal shim for CLI imports
        session_state = {}

        @staticmethod
        def cache_data(*args, **kwargs):
            def _decorator(fn):
                return fn

            return _decorator

    st = _StreamlitStub()  # type: ignore
import pandas as pd
import numpy as np
import io
import os
from html import escape as html_escape
from pathlib import Path
import time
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('PortfolioOptimization')

# Add a rotating file logger for production diagnostics
try:
    from logging.handlers import RotatingFileHandler

    def _setup_file_logging() -> None:
        # Cross-platform log directory: APPDATA (Windows), XDG_DATA_HOME (Linux), or ~/.local/share
        base_env = os.environ.get('APPDATA') or os.environ.get('XDG_DATA_HOME') or os.path.join(str(Path.home()), '.local', 'share')
        base = Path(base_env) / 'ServisRetail' / 'PortfolioOptimizationEngine' / 'logs'
        base.mkdir(parents=True, exist_ok=True)
        log_path = base / 'app.log'
        fh = RotatingFileHandler(str(log_path), maxBytes=2_000_000, backupCount=5, encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
        logging.getLogger().addHandler(fh)
    _setup_file_logging()
except Exception as e:
    logging.getLogger('PortfolioOptimization').debug(f"File logging setup failed (non-critical): {e}")


# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

class AppConfig:
    """Application configuration constants."""
    APP_NAME = "Portfolio Optimization Engine"
    APP_VERSION = "3.0.0"
    COMPANY_NAME = "Servis Retail"
    COPYRIGHT_YEAR = "2026"
    
    # Performance settings
    MAX_FILE_SIZE_MB = 100
    CHUNK_SIZE = 10000
    CACHE_TTL = 3600
    
    # UI settings
    SIDEBAR_WIDTH = 300
    MAX_PREVIEW_ROWS = 100
    MAX_DISPLAY_ROWS = 1000
    
    # Business logic defaults
    DEFAULT_EFFICIENT_STORE_PCT = 30
    DEFAULT_GMROI_WEIGHT = 0.60
    DEFAULT_ROS_WEIGHT = 0.40
    DEFAULT_ALPHA_MIN = 0.00
    DEFAULT_ALPHA_MAX = 1.00
    DEFAULT_PEER_POOL_SIZE = 500
    DEFAULT_NETWORK_STOCK_THRESHOLD = 50

    # Gate weighting (Excel-like)
    DEFAULT_GATE_SPF_WEIGHT = 0.70
    DEFAULT_GATE_Q_WEIGHT = 0.30
    DEFAULT_ROBUST_P_LOW = 5.0
    DEFAULT_ROBUST_P_HIGH = 95.0
    DEFAULT_AUTO_SELECT_GATE_WEIGHT = False
    
    # Required sheets (flexible matching)
    REQUIRED_SHEETS = {
        'stores': ['StoresKPIs', 'Stores', 'Store', 'StoreKPIs', 'stores'],
        'tiers': ['TierKPIs', 'Tiers', 'Tier', 'TierTargets', 'tiers'],
        'articles': ['Articles', 'Article', 'SKU', 'SKUs', 'Products', 'articles']
    }


class WorkflowStep(Enum):
    """Workflow step enumeration."""
    UPLOAD = 1
    QUALITY_CHECK = 2
    CONFIGURATION = 3
    OPTIMIZATION = 4
    RESULTS = 5


@dataclass
class OptimizationConfig:
    """Configuration for the optimization engine."""
    efficient_store_pct: float = AppConfig.DEFAULT_EFFICIENT_STORE_PCT
    gmroi_weight: float = AppConfig.DEFAULT_GMROI_WEIGHT
    ros_weight: float = AppConfig.DEFAULT_ROS_WEIGHT
    # Optional: price power weight from DiscountPct (lower discount => higher score)
    # If you do not provide DiscountPct, keep this at 0.
    price_power_weight: float = 0.10
    alpha_min: float = AppConfig.DEFAULT_ALPHA_MIN
    alpha_max: float = AppConfig.DEFAULT_ALPHA_MAX
    gate_weight_spf: float = AppConfig.DEFAULT_GATE_SPF_WEIGHT
    gate_weight_q: float = AppConfig.DEFAULT_GATE_Q_WEIGHT
    robust_p_low: float = AppConfig.DEFAULT_ROBUST_P_LOW
    robust_p_high: float = AppConfig.DEFAULT_ROBUST_P_HIGH
    auto_select_gate_weight: bool = AppConfig.DEFAULT_AUTO_SELECT_GATE_WEIGHT
    peer_pool_size: int = AppConfig.DEFAULT_PEER_POOL_SIZE
    network_stock_threshold: int = AppConfig.DEFAULT_NETWORK_STOCK_THRESHOLD
    period_weeks: int = 52

    # -----------------------------
    # Merchant governance controls
    # -----------------------------
    # Maximum allowed TOTAL SKU change per store per run (percent).
    # Example: 10 means +/-10% vs current total.
    # Set to 0 or 100 to disable capping.
    max_store_change_pct: float = 0.0

    # Optional list of SKUs that must never appear in the Remove list.
    # (Hero / Strategic / Protected SKUs)
    do_not_remove: List[str] = field(default_factory=list)

    # -----------------------------
    # Demand-weighted tier logic
    # -----------------------------
    enable_demand_weighted_benchmark: bool = True
    tvi_clamp_min: float = 0.70
    tvi_clamp_max: float = 1.30
    min_tier_sales_qty_for_tvi: int = 5
    asp_pressure_threshold: float = 1.00
    empty_tier_requires_matrix: bool = True


@dataclass
class QualityCheckResult:
    """Result of a quality check."""
    check_name: str
    status: str  # 'pass', 'warning', 'fail'
    message: str
    details: Optional[List[str]] = None
    metric_value: Optional[Any] = None


@dataclass
class DataSummary:
    """Summary of loaded data."""
    total_stores: int = 0
    total_skus: int = 0
    total_tiers: int = 0
    active_skus: int = 0
    date_range: str = ""
    file_size_mb: float = 0.0
    load_time_seconds: float = 0.0


# =============================================================================
# PREMIUM CSS STYLING
# =============================================================================

def get_premium_css() -> str:
    """Generate premium enterprise-grade CSS styling — Modern Dark-Teal Theme."""
    return """
    <style>
    /* =========================================
       SERVIS RETAIL — PREMIUM ENTERPRISE THEME
       v3.0 | Modern Dark-Teal / Emerald Palette
       ========================================= */

    /* CSS Variables */
    :root {
        --primary: #0F766E;
        --primary-dark: #0D5F59;
        --primary-light: #14B8A6;
        --primary-glow: rgba(20, 184, 166, 0.15);
        --accent: #F97316;
        --accent-light: #FB923C;
        --bg-base: #F8FAFB;
        --bg-card: #FFFFFF;
        --bg-subtle: #F0F4F5;
        --bg-muted: #E8EDEF;
        --text-primary: #111827;
        --text-secondary: #4B5563;
        --text-muted: #9CA3AF;
        --border: #D1D9E0;
        --border-light: #E5EBF0;
        --success: #059669;
        --success-bg: #ECFDF5;
        --warning: #D97706;
        --warning-bg: #FFFBEB;
        --error: #DC2626;
        --error-bg: #FEF2F2;
        --info: #0284C7;
        --info-bg: #F0F9FF;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
        --shadow-lg: 0 12px 32px rgba(0,0,0,0.10);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
    }

    /* ── Global ── */
    .stApp {
        background: var(--bg-base);
    }
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Premium Header (gradient banner) ── */
    .premium-header {
        background: linear-gradient(135deg, #0F766E 0%, #065F46 40%, #1E3A5F 100%);
        padding: 28px 36px;
        border-radius: var(--radius-xl);
        margin-bottom: 28px;
        box-shadow: var(--shadow-lg), inset 0 1px 0 rgba(255,255,255,0.1);
        position: relative;
        overflow: hidden;
    }
    .premium-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .premium-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    }
    .premium-header-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: relative;
        z-index: 1;
    }
    .premium-header-title {
        color: #FFFFFF;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .premium-header-subtitle {
        color: rgba(255,255,255,0.75);
        font-size: 13px;
        font-weight: 400;
        margin-top: 5px;
    }
    .premium-header-version {
        background: rgba(255,255,255,0.12);
        color: #FFFFFF;
        padding: 7px 16px;
        border-radius: 24px;
        font-size: 12px;
        font-weight: 600;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.15);
    }

    /* ── Premium Cards ── */
    .premium-card {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: 28px;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-light);
        margin-bottom: 24px;
        transition: box-shadow 0.2s ease;
    }
    .premium-card:hover {
        box-shadow: var(--shadow-md);
    }
    .premium-card-header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
        padding-bottom: 18px;
        border-bottom: 1px solid var(--border-light);
    }
    .premium-card-icon {
        width: 46px;
        height: 46px;
        border-radius: var(--radius-md);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        margin-right: 16px;
        flex-shrink: 0;
    }
    .premium-card-icon.primary {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(15,118,110,0.3);
    }
    .premium-card-icon.success {
        background: linear-gradient(135deg, var(--success) 0%, #047857 100%);
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(5,150,105,0.3);
    }
    .premium-card-icon.warning {
        background: linear-gradient(135deg, var(--warning) 0%, #B45309 100%);
        color: #FFFFFF;
    }
    .premium-card-icon.info {
        background: linear-gradient(135deg, var(--info) 0%, #0369A1 100%);
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(2,132,199,0.3);
    }
    .premium-card-title {
        font-size: 18px;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    .premium-card-subtitle {
        font-size: 13px;
        color: var(--text-secondary);
        margin-top: 3px;
    }

    /* ── Metric Cards (KPI tiles) ── */
    .metric-card {
        background: var(--bg-card);
        border-radius: var(--radius-md);
        padding: 20px 22px;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-light);
        position: relative;
        overflow: hidden;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 4px; height: 100%;
        border-radius: 4px 0 0 4px;
        background: var(--primary);
    }
    .metric-card.success::before { background: var(--success); }
    .metric-card.warning::before { background: var(--warning); }
    .metric-card.error::before   { background: var(--error); }
    .metric-card.info::before    { background: var(--info); }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.15;
    }
    .metric-change {
        font-size: 12px;
        font-weight: 600;
        margin-top: 6px;
    }
    .metric-change.positive { color: var(--success); }
    .metric-change.negative { color: var(--error); }
    .metric-change.neutral  { color: var(--text-muted); }

    /* ── Alert Boxes ── */
    .alert-box {
        padding: 16px 20px;
        border-radius: var(--radius-md);
        margin-bottom: 16px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    .alert-box.success {
        background: var(--success-bg);
        border: 1px solid rgba(5,150,105,0.25);
        color: #065F46;
    }
    .alert-box.warning {
        background: var(--warning-bg);
        border: 1px solid rgba(217,119,6,0.25);
        color: #92400E;
    }
    .alert-box.error {
        background: var(--error-bg);
        border: 1px solid rgba(220,38,38,0.25);
        color: #991B1B;
    }
    .alert-box.info {
        background: var(--info-bg);
        border: 1px solid rgba(2,132,199,0.25);
        color: #0C4A6E;
    }
    .alert-icon {
        font-size: 18px;
        flex-shrink: 0;
        margin-top: 1px;
    }
    .alert-content { flex: 1; }
    .alert-title {
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 4px;
    }
    .alert-message {
        font-size: 13px;
        line-height: 1.6;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: #FFFFFF;
        border: none;
        border-radius: var(--radius-sm);
        padding: 12px 28px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.2px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(15,118,110,0.25);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(15,118,110,0.35);
        background: linear-gradient(135deg, var(--primary-dark) 0%, #0A4F4A 100%);
    }
    .stButton > button:active {
        transform: translateY(0);
    }
    .success-btn > button {
        background: linear-gradient(135deg, var(--success) 0%, #047857 100%) !important;
        box-shadow: 0 2px 8px rgba(5,150,105,0.25) !important;
    }
    .success-btn > button:hover {
        background: linear-gradient(135deg, #047857 0%, #065F46 100%) !important;
    }

    /* ── File Uploader ── */
    .stFileUploader {
        border-radius: var(--radius-lg);
    }
    .stFileUploader > div {
        border-radius: var(--radius-lg);
    }

    /* ── Sliders ── */
    .stSlider > div > div > div > div {
        background: var(--primary);
    }

    /* ── Select boxes ── */
    .stSelectbox > div > div {
        border-radius: var(--radius-sm);
    }

    /* ── Data Tables ── */
    .stDataFrame {
        border-radius: var(--radius-md);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-subtle);
        border-radius: var(--radius-sm);
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        font-weight: 500;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: var(--bg-card);
        box-shadow: var(--shadow-sm);
    }

    /* ── Progress ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-light) 100%);
        border-radius: 4px;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: var(--bg-subtle);
        border-radius: var(--radius-sm);
        font-weight: 600;
    }

    /* ── QC Badges ── */
    .qc-badge {
        display: inline-flex;
        align-items: center;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .qc-badge.pass {
        background: var(--success-bg);
        color: #065F46;
        border: 1px solid rgba(5,150,105,0.25);
    }
    .qc-badge.warning {
        background: var(--warning-bg);
        color: #92400E;
        border: 1px solid rgba(217,119,6,0.25);
    }
    .qc-badge.fail {
        background: var(--error-bg);
        color: #991B1B;
        border: 1px solid rgba(220,38,38,0.25);
    }

    /* ── Config Sections ── */
    .config-section {
        background: var(--bg-card);
        border-radius: var(--radius-md);
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid var(--border-light);
    }
    .config-section-title {
        font-size: 16px;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border-light);
    }

    /* ── Footer ── */
    .premium-footer {
        text-align: center;
        padding: 28px;
        color: var(--text-muted);
        font-size: 12px;
        border-top: 1px solid var(--border-light);
        margin-top: 48px;
        font-weight: 500;
    }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .premium-header { padding: 18px 22px; }
        .premium-header-title { font-size: 20px; }
        .metric-value { font-size: 22px; }
    }

    /* ── Print ── */
    @media print {
        .premium-header, .stButton { display: none !important; }
    }
    </style>
    """


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_header():
    """Render the premium application header."""
    st.markdown(f"""
    <div class="premium-header">
        <div class="premium-header-content">
            <div>
                <h1 class="premium-header-title">{AppConfig.COMPANY_NAME} | {AppConfig.APP_NAME}</h1>
                <p class="premium-header-subtitle">Enterprise-Grade SKU Allocation & Optimization Platform</p>
            </div>
            <div class="premium-header-version">v{AppConfig.APP_VERSION}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_workflow_stepper(current_step: WorkflowStep):
    """Render the workflow progress stepper using Streamlit columns."""
    steps = [
        ("1", "Upload Data", WorkflowStep.UPLOAD),
        ("2", "Quality Check", WorkflowStep.QUALITY_CHECK),
        ("3", "Configuration", WorkflowStep.CONFIGURATION),
        ("4", "Optimization", WorkflowStep.OPTIMIZATION),
        ("5", "Results", WorkflowStep.RESULTS),
    ]
    
    # Use Streamlit columns for reliable rendering
    cols = st.columns(5)
    
    for idx, (step_num, step_label, step_enum) in enumerate(steps):
        with cols[idx]:
            if step_enum.value < current_step.value:
                # Completed step
                st.markdown(f"""
                <div style="text-align: center; padding: 10px;">
                    <div style="width: 40px; height: 40px; border-radius: 50%; background: #059669; color: white; 
                                display: inline-flex; align-items: center; justify-content: center; font-weight: 700; 
                                font-size: 16px; margin-bottom: 8px; box-shadow: 0 2px 8px rgba(5,150,105,0.3);">✓</div>
                    <div style="font-size: 12px; font-weight: 600; color: #059669;">{step_label}</div>
                </div>
                """, unsafe_allow_html=True)
            elif step_enum.value == current_step.value:
                # Active step
                st.markdown(f"""
                <div style="text-align: center; padding: 10px;">
                    <div style="width: 40px; height: 40px; border-radius: 50%; background: #0F766E; color: white; 
                                display: inline-flex; align-items: center; justify-content: center; font-weight: 700; 
                                font-size: 16px; margin-bottom: 8px; box-shadow: 0 0 0 4px rgba(15,118,110,0.2);">{step_num}</div>
                    <div style="font-size: 12px; font-weight: 700; color: #0F766E;">{step_label}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Pending step
                st.markdown(f"""
                <div style="text-align: center; padding: 10px;">
                    <div style="width: 40px; height: 40px; border-radius: 50%; background: #F0F4F5; color: #9CA3AF; 
                                display: inline-flex; align-items: center; justify-content: center; font-weight: 600; 
                                font-size: 16px; margin-bottom: 8px; border: 2px solid #D1D9E0;">{step_num}</div>
                    <div style="font-size: 12px; font-weight: 500; color: #9CA3AF;">{step_label}</div>
                </div>
                """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, status: str = "primary", change: str = None):
    """Render a metric card."""
    change_html = ""
    if change:
        change_class = "positive" if change.startswith("+") else ("negative" if change.startswith("-") else "neutral")
        change_html = f'<div class="metric-change {html_escape(change_class)}">{html_escape(change)}</div>'

    st.markdown(f"""
    <div class="metric-card {html_escape(status)}">
        <div class="metric-label">{html_escape(label)}</div>
        <div class="metric-value">{html_escape(str(value))}</div>
        {change_html}
    </div>
    """, unsafe_allow_html=True)


def render_alert(alert_type: str, title: str, message: str, allow_html: bool = False):
    """Render an alert box.

    Parameters
    ----------
    allow_html : bool
        When *True* the *message* is rendered as-is (caller is responsible for
        escaping any user-controlled content).  Default ``False`` escapes the
        full message string.
    """
    icons = {
        "success": "✓",
        "warning": "⚠",
        "error": "✕",
        "info": "ℹ"
    }
    safe_type = html_escape(alert_type)
    safe_message = message if allow_html else html_escape(message)
    st.markdown(f"""
    <div class="alert-box {safe_type}">
        <span class="alert-icon">{icons.get(alert_type, 'ℹ')}</span>
        <div class="alert-content">
            <div class="alert-title">{html_escape(title)}</div>
            <div class="alert-message">{safe_message}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_qc_badge(status: str, text: str):
    """Render a QC status badge."""
    return f'<span class="qc-badge {html_escape(status)}">{html_escape(text)}</span>'


def render_card_header(icon: str, title: str, subtitle: str = "", icon_type: str = "primary"):
    """Render a card header with icon."""
    subtitle_html = f'<div class="premium-card-subtitle">{html_escape(subtitle)}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="premium-card-header">
        <div class="premium-card-icon {html_escape(icon_type)}">{html_escape(icon)}</div>
        <div>
            <h3 class="premium-card-title">{html_escape(title)}</h3>
            {subtitle_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def init_session_state():
    """Initialize session state with default values."""
    defaults = {
        'current_step': WorkflowStep.UPLOAD,
        'uploaded_file': None,
        'raw_data': None,
        'stores_df': None,
        'tiers_df': None,
        'articles_df': None,
        'qc_results': None,
        'qc_passed': False,
        'config': OptimizationConfig(),
        'data_summary': DataSummary(),
        'optimization_results': None,
        'optimization_complete': False,
        'error_log': [],
        'processing': False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session_state():
    """Reset session state to initial values."""
    keys_to_reset = [
        'uploaded_file', 'raw_data', 'stores_df', 'tiers_df', 'articles_df',
        'qc_results', 'qc_passed', 'data_summary', 'optimization_results',
        'optimization_complete', 'error_log',
        # Disk-backed upload keys
        '_upload_disk_path', '_upload_file_name', '_upload_file_hash',
        'file_hash',
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.current_step = WorkflowStep.UPLOAD
    st.session_state.config = OptimizationConfig()


def navigate_to(step: WorkflowStep):
    """Navigate to a specific workflow step."""
    st.session_state.current_step = step
    # Handle both old and new Streamlit versions
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except Exception:
            st.warning("Navigation failed. Please refresh the page.")
            return


# =============================================================================
# DATA LOADING & VALIDATION
# =============================================================================

def find_matching_sheet(excel_file: pd.ExcelFile, sheet_type: str) -> Optional[str]:
    """Find a matching sheet name from the Excel file."""
    available_sheets = excel_file.sheet_names
    possible_names = AppConfig.REQUIRED_SHEETS.get(sheet_type, [])
    
    for name in possible_names:
        for sheet in available_sheets:
            if name.lower() == sheet.lower():
                return sheet
    
    return None


def load_excel_data_from_path(file_path: str, file_name: str) -> Tuple[Optional[Dict[str, pd.DataFrame]], Optional[str]]:
    """Load and parse Excel file from a DISK PATH (not bytes).

    This function is intentionally NOT decorated with @st.cache_data.
    Reason: cache_data pickles+hashes ALL function arguments on EVERY rerun.
    For large files (>5 MB), the hashing alone can exceed Streamlit's WebSocket
    heartbeat timeout, causing silent disconnects ("blink and nothing happens").

    Instead, we implement manual caching via session_state keyed on file hash.
    The caller checks the hash BEFORE calling this function.
    """
    try:
        start_time = time.time()

        from engine.io import load_workbook
        stores_df, tiers_df, articles_df = load_workbook(file_path)

        # Resolve sheet names for UI display
        # Skip the expensive pd.ExcelFile re-parse — just use column names
        # from the already-loaded DataFrames to infer which sheets were matched.
        stores_sheet = tiers_sheet = articles_sheet = 'Auto'

        data = {
            'stores': stores_df,
            'tiers': tiers_df,
            'articles': articles_df,
            'sheet_names': {
                'stores': stores_sheet,
                'tiers': tiers_sheet,
                'articles': articles_sheet
            }
        }

        load_time = time.time() - start_time
        data['load_time'] = load_time

        logger.info(f"Loaded Excel file '{file_name}' in {load_time:.2f}s")

        return data, None

    except Exception as e:
        logger.error(f"Error loading Excel file: {str(e)}\n{traceback.format_exc()}")
        return None, f"Error loading file: {str(e)}"


# =============================================================================
# QUALITY CHECKS
# =============================================================================

def run_quality_checks(stores_df: pd.DataFrame, tiers_df: pd.DataFrame,
                       articles_df: pd.DataFrame, config: OptimizationConfig) -> List[QualityCheckResult]:
    """Run business-critical QC checks aligned with the Excel templates and engine logic."""
    results: List[QualityCheckResult] = []

    # --- Schema checks (engine compatible) ---
    # IMPORTANT: engine/io.py normalizes column names during load_workbook()
    # (e.g., "Row Labels" → "StoreID"). So we must check the POST-normalization names.
    req_stores = ["StoreID", "Average of SKU_DC", "RFT", "SPF", "Q_SCORE", "GMROI"]
    req_tiers = ["StoreID", "PriceTier", "Average of SKU_DC", "GMROI"]
    req_articles = [
        "StoreID", "ItemColorName", "PriceTier",
        "Sales_Value", "Sales_Qty", "Gross_Margin", "ClosingStockQty",
    ]

    miss_stores = [c for c in req_stores if c not in stores_df.columns]
    miss_tiers = [c for c in req_tiers if c not in tiers_df.columns]
    miss_articles = [c for c in req_articles if c not in articles_df.columns]

    if miss_stores:
        results.append(QualityCheckResult(
            check_name="StoresKPIs Schema",
            status="fail",
            message=f"Missing required columns in StoresKPIs: {', '.join(miss_stores)}",
            details=miss_stores
        ))
    else:
        results.append(QualityCheckResult(
            check_name="StoresKPIs Schema",
            status="pass",
            message="StoresKPIs required columns OK",
            metric_value=len(stores_df)
        ))

    if miss_tiers:
        results.append(QualityCheckResult(
            check_name="TierKPIs Schema",
            status="fail",
            message=f"Missing required columns in TierKPIs: {', '.join(miss_tiers)}",
            details=miss_tiers
        ))
    else:
        results.append(QualityCheckResult(
            check_name="TierKPIs Schema",
            status="pass",
            message="TierKPIs required columns OK",
            metric_value=len(tiers_df)
        ))

    if miss_articles:
        results.append(QualityCheckResult(
            check_name="Articles Schema",
            status="fail",
            message=f"Missing required columns in Articles: {', '.join(miss_articles)}",
            details=miss_articles
        ))
    else:
        results.append(QualityCheckResult(
            check_name="Articles Schema",
            status="pass",
            message="Articles required columns OK",
            metric_value=len(articles_df)
        ))

    # --- Avg Inventory Cost (strict business rule) ---
    avg_inv = _infer_avg_inv_cost_column(articles_df)
    has_precomputed_gmroi = "GMROI" in articles_df.columns
    if avg_inv is None and not has_precomputed_gmroi:
        results.append(QualityCheckResult(
            check_name="GMROI Denominator",
            status="fail",
            message="Avg Inventory Cost column missing in Articles (required for GMROI = Gross Margin / Avg Inventory Cost)",
            details=["Add column: Avg Inventory Cost (or Avg_Inventory_Cost)"]
        ))
    elif avg_inv is None and has_precomputed_gmroi:
        results.append(QualityCheckResult(
            check_name="GMROI Denominator",
            status="warning",
            message="Avg Inventory Cost missing but pre-computed GMROI found. Engine will use proxy calculation.",
            metric_value="GMROI (pre-computed)"
        ))
    else:
        n0 = int((pd.to_numeric(articles_df[avg_inv], errors='coerce').fillna(0) <= 0).sum())
        status = "warning" if n0 > 0 else "pass"
        msg = "Avg Inventory Cost available" if n0 == 0 else f"Avg Inventory Cost has {n0:,} non-positive values (GMROI may be distorted)"
        results.append(QualityCheckResult(
            check_name="GMROI Denominator",
            status=status,
            message=msg,
            metric_value=avg_inv
        ))

    # --- Time granularity for ROS ---
    has_weeks = any(c in articles_df.columns for c in ["Weeks_Active", "Weeks", "NumWeeks"]) 
    has_monthyear = any(c in articles_df.columns for c in ["MonthYear", "Month_Year", "YearMonth", "YM", "CountofMonthYear"]) 
    if has_weeks:
        results.append(QualityCheckResult(
            check_name="ROS Granularity",
            status="pass",
            message="ROS will be computed as Avg Weekly ROS (Sales_Qty / Weeks_Active)",
            metric_value="Weekly"
        ))
    elif has_monthyear:
        results.append(QualityCheckResult(
            check_name="ROS Granularity",
            status="pass",
            message="ROS will be computed as Avg Monthly ROS (Sales_Qty / unique MonthYear)",
            metric_value="Monthly"
        ))
    else:
        results.append(QualityCheckResult(
            check_name="ROS Granularity",
            status="warning",
            message=f"No week/month indicators found. ROS will fallback to Sales_Qty / {config.period_weeks} weeks.",
            metric_value="Fallback"
        ))

    # --- Null rate analysis (articles) ---
    null_rates = {col: float(articles_df[col].isna().mean()) for col in articles_df.columns}
    worst_col, worst_rate = max(null_rates.items(), key=lambda x: x[1]) if null_rates else (None, 0.0)
    if worst_rate > 0.20:
        results.append(QualityCheckResult(
            check_name="Data Completeness",
            status="fail",
            message=f"High null rate detected in Articles: {worst_col} ({worst_rate*100:.1f}%)",
            metric_value=f"{worst_rate*100:.1f}%"
        ))
    elif worst_rate > 0.05:
        results.append(QualityCheckResult(
            check_name="Data Completeness",
            status="warning",
            message=f"Moderate null rate in Articles: {worst_col} ({worst_rate*100:.1f}%)",
            metric_value=f"{worst_rate*100:.1f}%"
        ))
    else:
        results.append(QualityCheckResult(
            check_name="Data Completeness",
            status="pass",
            message=f"Data completeness looks good (max null: {worst_rate*100:.1f}%)",
            metric_value=f"{(1-worst_rate)*100:.1f}%"
        ))

    # --- Duplicates ---
    # Use StoreID (post-normalization) for duplicate detection
    store_id_col = "StoreID" if "StoreID" in stores_df.columns else ("Row Labels" if "Row Labels" in stores_df.columns else None)
    if store_id_col:
        dup = int(stores_df[store_id_col].duplicated().sum())
        results.append(QualityCheckResult(
            check_name="Duplicate Stores",
            status="warning" if dup > 0 else "pass",
            message=f"Duplicate store IDs: {dup:,}" if dup > 0 else "No duplicate stores",
            metric_value=dup
        ))

    if all(c in tiers_df.columns for c in ["StoreID", "PriceTier"]):
        dup = int(tiers_df.duplicated(["StoreID", "PriceTier"]).sum())
        results.append(QualityCheckResult(
            check_name="Duplicate Store×Tier",
            status="warning" if dup > 0 else "pass",
            message=f"Duplicate StoreID×PriceTier rows: {dup:,}" if dup > 0 else "No duplicate Store×Tier rows",
            metric_value=dup
        ))

    if all(c in articles_df.columns for c in ["StoreID", "ItemColorName"]):
        dup = int(articles_df.duplicated(["StoreID", "ItemColorName"]).sum())
        results.append(QualityCheckResult(
            check_name="Duplicate Store×SKU",
            status="warning" if dup > 0 else "pass",
            message=f"Duplicate StoreID×SKU rows: {dup:,} (OK if time rows exist)" if dup > 0 else "No duplicate Store×SKU rows",
            metric_value=dup
        ))

    # --- Active SKU count (NetworkStock) ---
    if "ClosingStockQty" in articles_df.columns and "ItemColorName" in articles_df.columns:
        stock = articles_df.groupby("ItemColorName")["ClosingStockQty"].sum()
        active = int((stock > float(config.network_stock_threshold)).sum())
        total = int(len(stock))
        pct = (active / total * 100) if total else 0
        status = "pass" if pct >= 50 else "warning"
        results.append(QualityCheckResult(
            check_name="Active SKU Universe",
            status=status,
            message=f"{pct:.1f}% SKUs active (NetworkStock > {config.network_stock_threshold})",
            details=[f"Active: {active:,}", f"Total: {total:,}"],
            metric_value=active
        ))

    # --- Data volume ---
    total_rows = len(articles_df)
    results.append(QualityCheckResult(
        check_name="Data Volume",
        status="warning" if total_rows > 1_000_000 else "pass",
        message=f"Articles rows: {total_rows:,}",
        metric_value=f"{total_rows:,}"
    ))

    return results


# =============================================================================
# DISK-BASED STATE PERSISTENCE (survives Streamlit session resets)
# =============================================================================

try:
    _STATE_DIR = Path(os.environ.get('TEMP', os.environ.get('TMPDIR', '/tmp'))) / 'poe_state'
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    # Fallback: use CWD if temp dir is not writable
    _STATE_DIR = Path.cwd() / '.poe_state'
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        _STATE_DIR = None

_LAST_PATH_FILE = (_STATE_DIR / 'last_file_path.txt') if _STATE_DIR else None


def _save_last_path(file_path: str):
    """Save the last loaded file path to disk (survives session resets)."""
    if _LAST_PATH_FILE is None:
        return
    try:
        # Validate the path is a real Excel file, not an arbitrary system path
        fp = Path(file_path).resolve()
        if fp.suffix.lower() not in ('.xlsx', '.xls'):
            logger.warning(f"Rejected non-Excel path for state persistence: {file_path}")
            return
        _LAST_PATH_FILE.write_text(str(fp), encoding='utf-8')
    except Exception as e:
        logger.debug(f"Failed to save last path: {e}")


def _load_last_path() -> Optional[str]:
    """Load the last file path from disk."""
    if _LAST_PATH_FILE is None:
        return None
    try:
        if _LAST_PATH_FILE.exists():
            p = _LAST_PATH_FILE.read_text(encoding='utf-8').strip()
            fp = Path(p).resolve()
            if fp.exists() and fp.suffix.lower() in ('.xlsx', '.xls'):
                return str(fp)
    except Exception as e:
        logger.debug(f"Failed to load last path: {e}")
    return None


# =============================================================================
# MAIN APPLICATION PAGES
# =============================================================================

def render_upload_page():
    """Render the data upload page — crash-resistant architecture.

    Design goals:
    1. Survive Streamlit session resets (state persisted to disk)
    2. Minimal reruns (st.form batches submit)
    3. Auto-recovery (checks disk state on every render)
    4. Visible diagnostics (errors shown outside CSS containers)
    """
    # ── AUTO-RECOVERY: If session was reset but we have data on disk ──
    if st.session_state.get('stores_df') is None:
        last_path = _load_last_path()
        if last_path:
            _auto_recover(last_path)
            # _auto_recover calls st.rerun() on success.
            # If it did NOT rerun (i.e. recovery failed), fall through
            # to render the normal upload form instead of a blank page.
            if st.session_state.get('stores_df') is not None:
                return  # Recovery succeeded; rerun is pending

    # ── FAST PATH: data already loaded ──
    if (st.session_state.get('stores_df') is not None
            and st.session_state.get('tiers_df') is not None
            and st.session_state.get('articles_df') is not None):

        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        render_card_header("📁", "Upload Your Data",
                           "Upload an Excel file containing your SKU and store data", "primary")

        _loaded_name = st.session_state.get('uploaded_file') or 'Uploaded Excel'
        stores_df = st.session_state.stores_df
        articles_df = st.session_state.articles_df
        tiers_df = st.session_state.tiers_df
        sku_count = articles_df['ItemColorName'].nunique() if 'ItemColorName' in articles_df.columns else '?'

        render_alert("success", "Data Loaded Successfully",
                     f"File: <strong>{html_escape(str(_loaded_name))}</strong> — "
                     f"{len(stores_df):,} stores, {sku_count:,} SKUs, "
                     f"{len(tiers_df):,} tier rows",
                     allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✓ Validate Data & Continue", use_container_width=True,
                         type="primary", key="continue_loaded"):
                navigate_to(WorkflowStep.QUALITY_CHECK)

        st.markdown("---")
        with st.expander("Load a different file", expanded=False):
            _render_path_form()

        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── NORMAL PATH: no data loaded yet ──
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    render_card_header("📁", "Upload Your Data",
                       "Upload an Excel file containing your SKU and store data", "primary")

    upload_tab, path_tab = st.tabs(["📤 Upload File (Cloud / Local)", "📂 Paste File Path (Desktop)"])

    with upload_tab:
        _render_file_uploader()

    with path_tab:
        _render_path_form()

    _show_upload_instructions()

    st.markdown('</div>', unsafe_allow_html=True)


def _auto_recover(file_path_str: str):
    """Auto-recover from session reset by reloading from disk-persisted path."""
    try:
        # Guard against infinite recovery loops
        attempts = st.session_state.get('_recovery_attempts', 0)
        if attempts >= 2:
            st.warning("⚠️ Session keeps resetting. Please paste the file path below and click Load File.")
            if _LAST_PATH_FILE:
                _LAST_PATH_FILE.unlink(missing_ok=True)  # Clear to stop retrying
            return
        st.session_state['_recovery_attempts'] = attempts + 1

        fp = Path(file_path_str)
        if not fp.exists():
            return
        st.info(f"♻️ Recovering session... Reloading: {fp.name}")
        data, error = load_excel_data_from_path(str(fp), fp.name)
        if error or data is None:
            st.warning(f"Could not auto-recover: {error}")
            return
        st.session_state.stores_df = data['stores']
        st.session_state.tiers_df = data['tiers']
        st.session_state.articles_df = data['articles']
        st.session_state.uploaded_file = fp.name
        st.session_state.file_hash = 'recovered'
        st.session_state['_recovery_attempts'] = 0  # Reset counter on success
        file_size_mb = fp.stat().st_size / (1024 * 1024)
        st.session_state.data_summary = DataSummary(
            total_stores=len(data['stores']),
            total_skus=data['articles']['ItemColorName'].nunique() if 'ItemColorName' in data['articles'].columns else 0,
            total_tiers=len(data['tiers']),
            file_size_mb=file_size_mb,
            load_time_seconds=data.get('load_time', 0)
        )
        try:
            st.rerun()
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Auto-recovery failed: {e}")


def _render_file_uploader():
    """Render drag-and-drop file uploader for cloud deployments."""
    uploaded = st.file_uploader(
        "Choose an Excel file",
        type=["xlsx", "xls"],
        help="Drag and drop your .xlsx file here, or click Browse.",
        key="cloud_file_uploader",
    )
    if uploaded is not None:
        file_size_mb = uploaded.size / (1024 * 1024)
        if file_size_mb > AppConfig.MAX_FILE_SIZE_MB:
            st.error(f"❌ File too large: {file_size_mb:.1f} MB (max {AppConfig.MAX_FILE_SIZE_MB} MB)")
            return
        if file_size_mb < 0.001:
            st.error("❌ File appears to be empty.")
            return

        # Check if we already loaded this exact file (avoid re-processing on every rerun)
        # Read bytes ONCE and reuse for both hashing and saving to disk
        raw_bytes = uploaded.getvalue()
        file_hash = hashlib.md5(raw_bytes).hexdigest()
        if st.session_state.get('file_hash') == file_hash and st.session_state.get('stores_df') is not None:
            del raw_bytes  # Free memory
            return  # Already loaded

        with st.spinner("Loading and validating data... (this may take 15–30 seconds for large files)"):
            try:
                from engine.io import load_workbook
                import gc

                # Strategy: try temp-file-on-disk first (lower memory),
                # fall back to BytesIO if filesystem is restricted.
                stores_df = tiers_df = articles_df = None
                try:
                    import tempfile as _tf
                    tmp_fd, tmp_path = _tf.mkstemp(suffix=".xlsx")
                    try:
                        os.write(tmp_fd, raw_bytes)
                        os.close(tmp_fd)
                        del raw_bytes  # Free ~11 MB before openpyxl loads
                        gc.collect()
                        stores_df, tiers_df, articles_df = load_workbook(tmp_path)
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                except (OSError, PermissionError) as disk_err:
                    # Temp file failed (read-only filesystem, etc.) — use BytesIO
                    logger.warning(f"Temp file fallback to BytesIO: {disk_err}")
                    bytes_io = io.BytesIO(raw_bytes)
                    del raw_bytes
                    gc.collect()
                    stores_df, tiers_df, articles_df = load_workbook(bytes_io)
                    del bytes_io

                gc.collect()

                st.session_state.stores_df = stores_df
                st.session_state.tiers_df = tiers_df
                st.session_state.articles_df = articles_df
                st.session_state.uploaded_file = uploaded.name
                st.session_state.file_hash = file_hash

                sku_count = articles_df['ItemColorName'].nunique() if 'ItemColorName' in articles_df.columns else 0
                st.session_state.data_summary = DataSummary(
                    total_stores=len(stores_df),
                    total_skus=sku_count,
                    total_tiers=len(tiers_df),
                    file_size_mb=file_size_mb,
                    load_time_seconds=0,
                )

                render_alert("success", "Data Loaded Successfully",
                             f"File: <strong>{html_escape(uploaded.name)}</strong> — "
                             f"{len(stores_df):,} stores, {sku_count:,} SKUs, "
                             f"{len(tiers_df):,} tier rows",
                             allow_html=True)

                try:
                    st.rerun()
                except Exception:
                    pass

            except MemoryError:
                logger.error("MemoryError during file upload")
                st.error("❌ Out of memory. Your file is too large for Streamlit Cloud's free tier (1 GB RAM). "
                         "Try reducing the number of rows or use the Desktop edition instead.")
            except Exception as e:
                logger.error(f"Upload error: {e}\n{traceback.format_exc()}")
                st.error(f"❌ Error loading file: {e}\n\nPlease check that your file has the required sheets "
                         f"(StoresKPIs, TierKPIs, Articles) and try again.")


def _render_path_form():
    """Render file path form — uses st.form for single-rerun submit."""

    st.markdown("""
    <div class="alert-box info">
        <span class="alert-icon">📂</span>
        <div class="alert-content">
            <div class="alert-title">Enter File Path</div>
            <div class="alert-message">
                Paste the full path to your Excel file below, then click <strong>Load File</strong>.<br>
                <strong>Tip:</strong> In Windows Explorer, right-click your file →
                <em>Copy as path</em>, then paste here.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # st.form batches BOTH the text input and submit button into ONE rerun.
    # This halves the number of reruns compared to separate text_input + button.
    with st.form("file_path_form", clear_on_submit=False):
        file_path_input = st.text_input(
            "Full file path",
            placeholder=r"C:\Users\YourName\Documents\Portfolio_Optimization_Input_Template.xlsx",
            help="Paste the full Windows path to your .xlsx file",
        )
        submitted = st.form_submit_button("📥 Load File", use_container_width=True, type="primary")

    if submitted and file_path_input and file_path_input.strip():
        cleaned = file_path_input.strip().strip('"').strip("'")
        _handle_path_input(Path(cleaned))


def _handle_path_input(file_path: Path):
    """Validate and load an Excel file from a disk path."""
    try:
        if not file_path.exists():
            st.error(f"❌ File not found: `{file_path}`\n\nPlease check the path and try again.")
            return
        if file_path.suffix.lower() not in ('.xlsx', '.xls'):
            st.error("❌ Please select an Excel file (.xlsx or .xls)")
            return

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > AppConfig.MAX_FILE_SIZE_MB:
            st.error(f"❌ File too large: {file_size_mb:.1f} MB (max {AppConfig.MAX_FILE_SIZE_MB} MB)")
            return
        if file_size_mb < 0.001:
            st.error("❌ File appears to be empty (0 bytes).")
            return

        # Persist the path to disk (survives session resets)
        _save_last_path(str(file_path))

        _load_from_disk(file_path, file_path.name)

    except PermissionError:
        st.error("❌ Permission denied.  Make sure the file is not open in Excel and try again.")
    except Exception as e:
        logger.error(f"Path input error: {e}\n{traceback.format_exc()}")
        st.error(f"❌ Error reading file: {e}")


def _load_from_disk(disk_path: Path, file_name: str):
    """Load Excel data from a disk file and store DataFrames in session_state."""
    try:
        file_size_mb = disk_path.stat().st_size / (1024 * 1024)

        # Show file info
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            _display_name = file_name[:30] + "..." if len(file_name) > 30 else file_name
            render_metric_card("File Name", _display_name, "info")
        with col2:
            render_metric_card("File Size", f"{file_size_mb:.2f} MB", "info")
        with col3:
            render_metric_card("Format", "Excel (.xlsx)", "info")

        # Load from disk path
        with st.spinner("Loading and validating data... (this may take 15–30 seconds for large files)"):
            data, error = load_excel_data_from_path(str(disk_path), file_name)

        if error:
            st.error(f"**Error Loading File:** {error}")
            return

        if data is None:
            st.error("The file was read but no data could be extracted.")
            return

        # Store DataFrames in session_state
        st.session_state.stores_df = data['stores']
        st.session_state.tiers_df = data['tiers']
        st.session_state.articles_df = data['articles']
        st.session_state.uploaded_file = file_name
        st.session_state.file_hash = 'loaded'

        # Persist path to disk for recovery
        _save_last_path(str(disk_path))

        stores_df = data['stores']
        tiers_df = data['tiers']
        articles_df = data['articles']

        summary = DataSummary(
            total_stores=len(stores_df),
            total_skus=articles_df['ItemColorName'].nunique() if 'ItemColorName' in articles_df.columns else 0,
            total_tiers=len(tiers_df),
            file_size_mb=file_size_mb,
            load_time_seconds=data.get('load_time', 0)
        )
        st.session_state.data_summary = summary

        render_alert("success", "Data Loaded Successfully",
                     f"Found {summary.total_stores:,} stores, {summary.total_skus:,} unique SKUs, "
                     f"and {summary.total_tiers} tier rows.")

        # Data preview
        st.markdown("### Data Preview")
        preview_tab1, preview_tab2, preview_tab3 = st.tabs(["📊 Stores", "🏷️ Tiers", "📦 Articles"])

        with preview_tab1:
            st.dataframe(stores_df.head(10), use_container_width=True)
            st.caption(f"Showing 10 of {len(stores_df):,} rows")

        with preview_tab2:
            st.dataframe(tiers_df.head(10), use_container_width=True)
            st.caption(f"Showing 10 of {len(tiers_df):,} rows")

        with preview_tab3:
            st.dataframe(articles_df.head(10), use_container_width=True)
            st.caption(f"Showing 10 of {len(articles_df):,} rows")

        # Navigation
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✓ Validate Data & Continue", use_container_width=True, type="primary"):
                navigate_to(WorkflowStep.QUALITY_CHECK)

    except Exception as e:
        logger.error(f"_load_from_disk error: {e}\n{traceback.format_exc()}")
        st.error(f"Error loading data from file: {e}")


def _show_upload_instructions():
    """Show upload instructions when no file is loaded."""
    st.markdown("""
    <div class="alert-box info">
        <span class="alert-icon">ℹ</span>
        <div class="alert-content">
            <div class="alert-title">Required File Format</div>
            <div class="alert-message">
                Your Excel file should contain the following sheets:
                <ul style="margin-top: 8px; margin-bottom: 0;">
                    <li><strong>StoresKPIs</strong> — Store information with StoreID, StoreName</li>
                    <li><strong>TierKPIs</strong> — Price tier definitions and targets</li>
                    <li><strong>Articles</strong> — SKU data with ItemColorName, StoreID, StoreStock, Sales_Qty, etc.</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_quality_check_page():
    """Render the quality check page."""
    if st.session_state.stores_df is None:
        render_alert("warning", "No Data Loaded", "Please upload a data file first.")
        if st.button("← Back to Upload"):
            navigate_to(WorkflowStep.UPLOAD)
        return
    
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    render_card_header("🔍", "Data Quality Dashboard", "Comprehensive validation of your uploaded data", "info")
    
    # Run QC checks
    with st.spinner("Running quality checks..."):
        qc_results = run_quality_checks(
            st.session_state.stores_df,
            st.session_state.tiers_df,
            st.session_state.articles_df,
            st.session_state.config
        )
        st.session_state.qc_results = qc_results

        # Also run the engine's structured QC for deeper diagnostics (non-blocking)
        try:
            from engine.qc import run_qc_checks
            st.session_state.engine_qc = run_qc_checks(
                st.session_state.stores_df,
                st.session_state.tiers_df,
                st.session_state.articles_df,
            )
        except Exception as e:
            logger.warning(f"Engine QC could not run: {e}")
            st.session_state.engine_qc = None
    
    # Summary metrics
    pass_count = sum(1 for r in qc_results if r.status == "pass")
    warning_count = sum(1 for r in qc_results if r.status == "warning")
    fail_count = sum(1 for r in qc_results if r.status == "fail")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Checks", str(len(qc_results)), "info")
    with col2:
        render_metric_card("Passed", str(pass_count), "success")
    with col3:
        render_metric_card("Warnings", str(warning_count), "warning")
    with col4:
        render_metric_card("Failed", str(fail_count), "error")

    # Engine QC tables (advanced)
    if getattr(st.session_state, 'engine_qc', None) is not None:
        with st.expander("Advanced Engine QC Diagnostics", expanded=False):
            qc = st.session_state.engine_qc
            st.markdown("#### Errors")
            st.dataframe(qc.errors, use_container_width=True)
            st.markdown("#### Warnings")
            st.dataframe(qc.warnings, use_container_width=True)
            st.markdown("#### Info")
            st.dataframe(qc.info, use_container_width=True)
            if qc.suggestions is not None and not qc.suggestions.empty:
                st.markdown("#### Suggestions")
                st.dataframe(qc.suggestions, use_container_width=True)
    
    # Detailed results
    st.markdown("### Detailed Results")

    for result in qc_results:
        status_icon = {"pass": "✓", "warning": "⚠", "fail": "✕"}.get(result.status, "?")
        status_color = {"pass": "success", "warning": "warning", "fail": "error"}.get(result.status, "info")
        
        with st.expander(f"{status_icon} {result.check_name} — {result.message}", expanded=(result.status == "fail")):
            st.markdown(f"""
            <div class="alert-box {status_color}">
                <div class="alert-content">
                    <div class="alert-message">{result.message}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if result.details:
                st.markdown("**Details:**")
                for detail in result.details:
                    st.markdown(f"- {detail}")
            
            if result.metric_value is not None:
                st.metric("Value", result.metric_value)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Determine if QC passed
    qc_passed = fail_count == 0
    st.session_state.qc_passed = qc_passed
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← Back to Upload", use_container_width=True):
            navigate_to(WorkflowStep.UPLOAD)
    
    with col3:
        if qc_passed:
            if st.button("Continue to Configuration →", use_container_width=True, type="primary"):
                navigate_to(WorkflowStep.CONFIGURATION)
        else:
            st.button("Fix Issues Before Continuing", use_container_width=True, disabled=True)
            st.caption("Please fix the failed checks before proceeding.")


def render_configuration_page():
    """Render the configuration page."""
    if not st.session_state.qc_passed:
        render_alert("warning", "Quality Check Required", "Please complete the quality check first.")
        if st.button("← Back to Quality Check"):
            navigate_to(WorkflowStep.QUALITY_CHECK)
        return
    
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    render_card_header("⚙️", "Optimization Configuration", "Configure parameters for the SKU optimization engine", "primary")
    
    config = st.session_state.config
    
    # Performance Thresholds Section
    st.markdown("#### Performance Thresholds")
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        efficient_store_pct = st.slider(
            "Efficient Store Percentage",
            min_value=5,
            max_value=95,
            value=int(config.efficient_store_pct),
            step=5,
            help="Top percentage of stores by GMROI to learn optimal SKU mix from"
        )
        st.caption("Higher values include more stores in the learning set")
    
    with col2:
        gmroi_weight = st.slider(
            "GMROI Weight in Hybrid Score",
            min_value=0,
            max_value=100,
            value=int(config.gmroi_weight * 100),
            step=5,
            format="%d%%",
            help="Weight given to GMROI in the hybrid scoring formula"
        )
        gmroi_weight_frac = gmroi_weight / 100.0
        ros_weight_frac = 1.0 - gmroi_weight_frac
        st.info(f"ROS Weight: {ros_weight_frac:.0%} (auto-calculated)")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Optimization Range Section
    st.markdown("#### Optimization Range")
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        alpha_min = st.slider(
            "Minimum Change Factor (Alpha Min)",
            min_value=0.0,
            max_value=0.5,
            value=config.alpha_min,
            step=0.05,
            help="Minimum allowed reduction in SKU count per store"
        )
    
    with col2:
        alpha_max = st.slider(
            "Maximum Change Factor (Alpha Max)",
            min_value=0.5,
            max_value=1.0,
            value=config.alpha_max,
            step=0.05,
            help="Maximum allowed increase in SKU count per store"
        )
    
    peer_pool_size = st.slider(
        "Peer Pool Size",
        min_value=50,
        max_value=2000,
        value=config.peer_pool_size,
        step=50,
        help="Number of top-performing SKUs to consider for additions"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Active SKU Filter Section
    st.markdown("#### Active SKU Filter")
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    
    network_stock_threshold = st.number_input(
        "Minimum Network Stock (Units)",
        min_value=0,
        max_value=10000,
        value=config.network_stock_threshold,
        step=1,
        help="Only include SKUs with total network inventory greater than this value. This filters out low-stock or discontinued items."
    )
    
    st.markdown("""
    <div class="alert-box info">
        <span class="alert-icon">ℹ</span>
        <div class="alert-content">
            <div class="alert-title">Understanding Network Stock Threshold</div>
            <div class="alert-message">
                This filter excludes SKUs with very low inventory across all stores. For example, setting this to 5 means 
                only SKUs with more than 5 total units across the entire network will be included in the optimization.
                This helps focus on SKUs that are actively stocked and available for allocation.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Update configuration - preserve ALL fields, not just the ones shown on this page
    st.session_state.config = OptimizationConfig(
        efficient_store_pct=efficient_store_pct,
        gmroi_weight=gmroi_weight_frac,
        ros_weight=ros_weight_frac,
        price_power_weight=config.price_power_weight,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        peer_pool_size=peer_pool_size,
        network_stock_threshold=network_stock_threshold,
        gate_weight_spf=config.gate_weight_spf,
        gate_weight_q=config.gate_weight_q,
        robust_p_low=config.robust_p_low,
        robust_p_high=config.robust_p_high,
        auto_select_gate_weight=config.auto_select_gate_weight,
        period_weeks=config.period_weeks,
        max_store_change_pct=config.max_store_change_pct,
        do_not_remove=config.do_not_remove,
        enable_demand_weighted_benchmark=config.enable_demand_weighted_benchmark,
        tvi_clamp_min=config.tvi_clamp_min,
        tvi_clamp_max=config.tvi_clamp_max,
        min_tier_sales_qty_for_tvi=config.min_tier_sales_qty_for_tvi,
        asp_pressure_threshold=config.asp_pressure_threshold,
        empty_tier_requires_matrix=config.empty_tier_requires_matrix,
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Configuration Summary
    st.markdown("### Configuration Summary")
    
    summary_cols = st.columns(4)
    with summary_cols[0]:
        render_metric_card("Efficient Stores", f"{efficient_store_pct}%", "primary")
    with summary_cols[1]:
        render_metric_card("GMROI / ROS Split", f"{gmroi_weight_frac:.0%} / {ros_weight_frac:.0%}", "primary")
    with summary_cols[2]:
        render_metric_card("Alpha Range", f"{alpha_min:.2f} - {alpha_max:.2f}", "primary")
    with summary_cols[3]:
        render_metric_card("Min Network Stock", f"{network_stock_threshold} units", "primary")
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← Back to Quality Check", use_container_width=True):
            navigate_to(WorkflowStep.QUALITY_CHECK)
    
    with col3:
        if st.button("Run Optimization →", use_container_width=True, type="primary"):
            navigate_to(WorkflowStep.OPTIMIZATION)


def execute_optimization_pipeline(
    config: OptimizationConfig,
    stores_df: pd.DataFrame,
    tiers_df: pd.DataFrame,
    articles_df: pd.DataFrame,
) -> dict:
    """Compatibility wrapper for earlier builds.

    The production pipeline lives in :func:`execute_portfolio_optimization`.
    This wrapper keeps older UI code paths working and ensures the module
    remains import-safe.
    """
    return execute_portfolio_optimization(
        stores_df=stores_df,
        tiers_df=tiers_df,
        articles_df=articles_df,
        ui_cfg=config,
        progress_cb=None,
    )





# =============================================================================
# OPTIMIZATION ENGINE INTEGRATION (PRODUCTION)
# =============================================================================

def _infer_avg_inv_cost_column(articles_df: pd.DataFrame) -> str | None:
    """Find Avg Inventory Cost column (fuzzy) as per business rule."""
    try:
        from engine.articles import _find_col_fuzzy  # type: ignore
    except Exception:
        return None
    return _find_col_fuzzy(
        articles_df,
        [
            "Avg Inventory Cost",
            "Avg_Inventory_Cost",
            "AvgInventoryCost",
            "AvgInvCost",
            "AverageInventoryCost",
        ],
    )


def execute_portfolio_optimization(
    stores_df: pd.DataFrame,
    tiers_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    ui_cfg: OptimizationConfig,
    progress_cb=None,
) -> Dict[str, pd.DataFrame]:
    """Run the complete optimization pipeline and return all outputs."""

    # Import engine modules (relative imports, no hardcoded paths)
    from engine.config import EngineConfig
    from engine.store_model import build_total_target_model
    from engine.tier_split import split_targets_across_tiers
    from engine.articles import aggregate_articles_assign_tier
    from engine.decisions import (
        build_network_stock,
        attach_scores,
        create_remove_list,
        create_peer_pool_adds,
        apply_active_filter,
        create_transfer_recommendations,
    )

    # -------- strict business checks --------
    avg_inv_col = _infer_avg_inv_cost_column(articles_df)
    has_precomputed_gmroi = "GMROI" in articles_df.columns
    if avg_inv_col is None and not has_precomputed_gmroi:
        raise ValueError(
            "Avg Inventory Cost column is missing in Articles data. "
            "As per business rule, GMROI must be calculated as Gross Margin / Avg Inventory Cost. "
            "Please add 'Avg Inventory Cost' (or Avg_Inventory_Cost) column to Articles sheet."
        )

    # Map UI config (percent -> fraction)
    cfg = EngineConfig(
        efficient_top_pct=float(ui_cfg.efficient_store_pct) / 100.0,
        robust_p_low=float(getattr(ui_cfg, "robust_p_low", 5.0)),
        robust_p_high=float(getattr(ui_cfg, "robust_p_high", 95.0)),
        auto_select_gate_weight=bool(getattr(ui_cfg, "auto_select_gate_weight", False)),
        fixed_gate_weight_spf=float(getattr(ui_cfg, "gate_weight_spf", 0.70)),
        fixed_gate_weight_q=float(getattr(ui_cfg, "gate_weight_q", 0.30)),
        alpha_min=float(ui_cfg.alpha_min),
        alpha_max=float(ui_cfg.alpha_max),
        active_network_stock_threshold=float(ui_cfg.network_stock_threshold),
        peer_pool_top_n_per_tier=int(ui_cfg.peer_pool_size),
        score_weight_gmroi=float(ui_cfg.gmroi_weight),
        score_weight_ros=float(ui_cfg.ros_weight),
        score_weight_price_power=float(getattr(ui_cfg, "price_power_weight", 0.0)),
        period_weeks=int(ui_cfg.period_weeks or 52),
        eligibility_matrix=getattr(ui_cfg, "eligibility_matrix", None),
        do_not_remove=list(getattr(ui_cfg, "do_not_remove", []) or []),
        max_store_change_pct=(
            None
            if float(getattr(ui_cfg, "max_store_change_pct", 0.0) or 0.0) <= 0.0
            or float(getattr(ui_cfg, "max_store_change_pct", 0.0) or 0.0) >= 100.0
            else float(getattr(ui_cfg, "max_store_change_pct", 0.0)) / 100.0
        ),

        # Demand-weighted tier logic
        enable_demand_weighted_benchmark=bool(getattr(ui_cfg, "enable_demand_weighted_benchmark", True)),
        tvi_clamp_min=getattr(ui_cfg, "tvi_clamp_min", None),
        tvi_clamp_max=getattr(ui_cfg, "tvi_clamp_max", None),
        min_tier_sales_qty_for_tvi=int(getattr(ui_cfg, "min_tier_sales_qty_for_tvi", 5)),
        asp_pressure_threshold=float(getattr(ui_cfg, "asp_pressure_threshold", 1.00)),
        empty_tier_requires_matrix=bool(getattr(ui_cfg, "empty_tier_requires_matrix", True)),
    )

    def _cb(msg: str, pct: int):
        if progress_cb:
            progress_cb(msg, pct)

    # 1) Store model
    _cb("Building store target model...", 15)

    # Compute Overall_ROS per store (used for efficient-store hybrid selection)
    # Overall_ROS = Total Sales Qty / Current SKUs / Months Active
    try:
        if "Sales_Qty" in articles_df.columns and "Average of SKU_DC" in stores_df.columns:
            _aq = articles_df.copy()
            _aq["StoreID"] = _aq["StoreID"].astype(str)
            # Months active proxy: median of CountofMonthYear (if present) else 1
            if "CountofMonthYear" in _aq.columns:
                m_active = _aq.groupby("StoreID")["CountofMonthYear"].median().clip(lower=1).to_dict()
            else:
                m_active = {}
            sales_qty = _aq.groupby("StoreID")["Sales_Qty"].sum().to_dict()
            skus = stores_df.set_index("StoreID")["Average of SKU_DC"].astype(float).to_dict()
            overall_ros = {}
            for sid in stores_df["StoreID"].astype(str).tolist():
                q = float(sales_qty.get(sid, 0.0))
                s = float(skus.get(sid, 0.0))
                mo = float(m_active.get(sid, 1.0))
                denom = max(s, 1.0) * max(mo, 1.0)
                overall_ros[sid] = q / denom
            stores_df = stores_df.copy()
            stores_df["Overall_ROS"] = stores_df["StoreID"].astype(str).map(overall_ros).fillna(0.0)
    except Exception:
        # Never break the run if ROS cannot be computed
        pass

    sm = build_total_target_model(stores_df, cfg)
    stores_enriched = sm["stores_enriched"]

    # 2) Tier split
    _cb("Splitting targets across tiers...", 35)
    ts = split_targets_across_tiers(tiers_df, stores_enriched, cfg, articles_raw=articles_df)
    tier_targets = ts["tier_targets"]


    # Store-level overall SKU targets (sum across tiers)
    overall_targets = None
    try:
        if isinstance(tier_targets, pd.DataFrame) and ("StoreID" in tier_targets.columns):
            if ("Current_SKUs_Tier" in tier_targets.columns) and ("Target_SKUs_Tier" in tier_targets.columns):
                overall_targets = (
                    tier_targets.groupby("StoreID", as_index=False)
                    .agg(
                        Current_SKUs=("Current_SKUs_Tier", "sum"),
                        Target_SKUs=("Target_SKUs_Tier", "sum"),
                    )
                )
                overall_targets["Gap"] = overall_targets["Target_SKUs"] - overall_targets["Current_SKUs"]
                overall_targets["Gap_pct"] = np.where(
                    overall_targets["Current_SKUs"].astype(float) > 0,
                    overall_targets["Gap"].astype(float) / overall_targets["Current_SKUs"].astype(float),
                    np.nan,
                )
    except Exception:
        overall_targets = None

    # 3) Aggregate articles to Store×SKU + assign tier
    _cb("Aggregating articles and assigning tiers...", 50)
    am = aggregate_articles_assign_tier(articles_df)
    sku_master = am["sku_master"]

    # 4) Network stock
    _cb("Computing network stock and active SKU universe...", 60)
    network_stock = build_network_stock(sku_master)

    # 5) Score SKUs
    # IMPORTANT:
    # - Add-tier (RetailPrice) must be computed on ACTIVE-only universe when possible.
    # - Peer clustering/store-fit need article-level signal; therefore we pass raw Articles.
    # We merge NetworkStock into sku_master BEFORE scoring.
    _cb("Scoring SKUs (GMROI + ROS)...", 70)
    sku_master_scoring = sku_master.merge(network_stock, on="ItemColorName", how="left").fillna({"NetworkStock": 0})
    sku_scored = attach_scores(sku_master_scoring, cfg)

    # 6) Remove list
    _cb("Generating Remove List...", 80)
    remove_list = create_remove_list(sku_scored, tier_targets, cfg)
    remove_list = apply_active_filter(remove_list, network_stock, cfg)

    # 7) Add list from peer pool
    _cb("Generating Add List from Peer Pool...", 90)
    add_list = create_peer_pool_adds(
        sku_scored,
        stores_enriched,
        tier_targets,
        cfg,
        articles_raw=articles_df,
    )
    # add_list uses ItemColorName column as well
    add_list = apply_active_filter(add_list, network_stock, cfg, sku_col="ItemColorName")


    # 7a-post) Circular recommendation detection
    try:
        from engine.decisions import flag_circular_recommendations
        remove_list, add_list = flag_circular_recommendations(remove_list, add_list)
    except Exception as e:
        logger.warning(f"Circular recommendation check skipped: {e}")

    # 7b) Inter-store transfer recommendations (Phase 2.3 - New Feature)
    _cb("Generating Inter-Store Transfer Recommendations...", 92)
    transfer_recommendations = pd.DataFrame()
    try:
        transfer_recommendations = create_transfer_recommendations(sku_scored, tier_targets, cfg, articles_raw=articles_df)
    except Exception as e:
        logger.warning(f"Transfer recommendations could not be generated: {e}")
        transfer_recommendations = pd.DataFrame()

    # 7c) Add shortfall report (needed vs provided)
    # Merchandising-facing diagnostic: if the engine can't fill all required adds
    # for a Store×Tier (e.g., not enough eligible candidates), we report it.
    add_shortfall_report = pd.DataFrame()
    try:
        if isinstance(tier_targets, pd.DataFrame) and isinstance(add_list, pd.DataFrame) and not tier_targets.empty:
            needed = tier_targets.loc[tier_targets.get('Gap', 0) > 0, ['StoreID', 'PriceTier', 'Gap']].copy()
            needed = needed.rename(columns={'Gap': 'Needed_Adds'})
            provided = (
                add_list.groupby(['StoreID_Target', 'PriceTier'])
                .size()
                .rename('Provided_Adds')
                .reset_index()
                .rename(columns={'StoreID_Target': 'StoreID'})
            )
            add_shortfall_report = needed.merge(provided, on=['StoreID', 'PriceTier'], how='left')
            add_shortfall_report['Provided_Adds'] = add_shortfall_report['Provided_Adds'].fillna(0).astype(int)
            add_shortfall_report['Shortfall'] = (add_shortfall_report['Needed_Adds'].astype(int) - add_shortfall_report['Provided_Adds']).astype(int)
            add_shortfall_report['FillRate'] = (add_shortfall_report['Provided_Adds'] / add_shortfall_report['Needed_Adds'].replace(0, pd.NA)).astype(float)

            add_shortfall_report = add_shortfall_report.sort_values(['Shortfall', 'Needed_Adds'], ascending=[False, False])
            # Keep only meaningful shortfalls
            add_shortfall_report = add_shortfall_report[add_shortfall_report['Shortfall'] > 0].copy()


            # Recommended action (Audit Risk 2)
            # - If no adds were provided for this tier: expand peer pool
            # - If fill rate is very low: relax lifecycle restriction
            # - Otherwise: accept tier mix adjustment
            def _rec_action(r):
                try:
                    prov = int(r.get('Provided_Adds', 0) or 0)
                    fr = r.get('FillRate', None)
                    fr = float(fr) if fr is not None and fr == fr else None
                    if prov == 0:
                        return 'expand peer pool'
                    if fr is not None and fr < 0.50:
                        return 'relax lifecycle restriction'
                    return 'accept tier mix adjustment'
                except Exception:
                    return 'expand peer pool'

            if not add_shortfall_report.empty:
                add_shortfall_report['Recommended_Action'] = add_shortfall_report.apply(_rec_action, axis=1)
    except Exception:
        add_shortfall_report = pd.DataFrame()

    # 8) Peer pool top 500 for reporting
    _cb("Preparing Peer Pool Top list...", 95)
    eff_col = "StoreID" if "StoreID" in stores_enriched.columns else ("Row Labels" if "Row Labels" in stores_enriched.columns else stores_enriched.columns[0])
    eff_ids = set(stores_enriched.loc[stores_enriched["Is_Efficient"] == 1, eff_col].astype(str))
    peer_pool = sku_scored[sku_scored["StoreID"].astype(str).isin(eff_ids)].copy()
    peer_pool = peer_pool.sort_values(["Assigned_Tier", "Score"], ascending=[True, False])

    # 9) Store SKU Grid (final) = CurrentActive - Remove + Add
    _cb("Building Store SKU Grid...", 99)
    store_grid = build_store_sku_grid(sku_master, add_list, remove_list, network_stock, cfg)

    # 10) SKU Status Master (Repeat / Watch / Drop)
    # Merchandising-facing decision categorization table.
    # (Hotfix) Ensure the variable is always defined before returning outputs.
    sku_status_master = pd.DataFrame()
    try:
        sku_status_master = build_sku_status_master(sku_scored, add_list, remove_list, cfg)
    except Exception as e:
        logger.warning(f"SKU_Status_Master could not be generated: {e}")
        sku_status_master = pd.DataFrame()

    _cb("Done", 100)

    # Model summary - ENHANCED (Audit Fix C3 - Transparency)
    # Calculate key statistics for executive summary
    total_stores = len(stores_enriched) if stores_enriched is not None else 0
    total_skus = sku_master["ItemColorName"].nunique() if sku_master is not None else 0
    total_adds = len(add_list) if add_list is not None and not add_list.empty else 0
    total_removes = len(remove_list) if remove_list is not None and not remove_list.empty else 0
    total_transfers = len(transfer_recommendations) if transfer_recommendations is not None and not transfer_recommendations.empty else 0
    
    # Calculate fill rates
    total_positive_gaps = int(tier_targets[tier_targets["Gap"] > 0]["Gap"].sum()) if tier_targets is not None else 0
    total_negative_gaps = int(abs(tier_targets[tier_targets["Gap"] < 0]["Gap"].sum())) if tier_targets is not None else 0
    add_fill_rate = f"{(total_adds / total_positive_gaps * 100):.1f}%" if total_positive_gaps > 0 else "N/A"
    remove_fill_rate = f"{(total_removes / total_negative_gaps * 100):.1f}%" if total_negative_gaps > 0 else "N/A"
    
    # Get alpha mode from tier_targets
    alpha_mode = "Simplified_3Tier" if cfg.use_simplified_alpha else "Legacy_Continuous"
    
    # Get deduplication stats if available
    dedup_stats = getattr(add_list, 'attrs', {}).get('dedup_stats', {}) if add_list is not None else {}
    duplicates_removed = dedup_stats.get('duplicates_removed', 0)
    
    model_summary = pd.DataFrame(
        {
            "Metric": [
                # Executive Summary
                "Total Stores Processed",
                "Total SKUs in Universe",
                "Total Add Recommendations",
                "Total Remove Recommendations",
                "Total Transfer Recommendations",
                "Add Fill Rate (Adds/Positive Gaps)",
                "Remove Fill Rate (Removes/Negative Gaps)",
                "Duplicates Removed from Add List",
                # Model Configuration
                "Best Gate Weights (SPF, Q_SCORE)",
                "Best Model",
                "CV R^2",
                "Train R^2",
                "Intercept",
                "Coefficients",
                "Efficient GMROI Threshold",
                "Efficient Store %",
                # Scoring Configuration
                "GMROI Weight",
                "ROS Weight",
                "Discount Penalty Enabled",
                "Discount Penalty Threshold",
                # Alpha Configuration
                "Alpha Mode",
                "Alpha Min",
                "Alpha Max",
                # Pool Configuration
                "Peer Pool Top-N per Tier",
                "Active Network Stock Threshold",
                "Max Store SKU Change % (Run Cap)",
                "Protected SKUs (Do Not Remove)",
                "Transfers Enabled",
            ],
            "Value": [
                # Executive Summary
                total_stores,
                total_skus,
                total_adds,
                total_removes,
                total_transfers,
                add_fill_rate,
                remove_fill_rate,
                duplicates_removed,
                # Model Configuration
                f"SPF={sm['model'].get('best_weight_spf'):.2f}, Q={sm['model'].get('best_weight_q'):.2f}",
                sm["model"].get("best_model"),
                sm["model"].get("cv_r2"),
                sm["model"].get("train_r2"),
                sm["model"].get("intercept"),
                str(sm["model"].get("coef")),
                sm["model"].get("efficient_gmroi_threshold"),
                f"{ui_cfg.efficient_store_pct}%",
                # Scoring Configuration
                ui_cfg.gmroi_weight,
                ui_cfg.ros_weight,
                "Yes" if cfg.discount_penalty_factor > 0 else "No",
                f"{cfg.discount_penalty_threshold * 100:.0f}%",
                # Alpha Configuration
                alpha_mode,
                ui_cfg.alpha_min,
                ui_cfg.alpha_max,
                # Pool Configuration
                ui_cfg.peer_pool_size,
                ui_cfg.network_stock_threshold,
                ui_cfg.max_store_change_pct,
                len(getattr(ui_cfg, "do_not_remove", []) or []),
                "Yes" if cfg.enable_transfers else "No",
            ],
        }
    )

    # k_value_by_tier + quartiles (Audit 2.6)
    try:
        k_sum = ts.get('k_summary') if isinstance(ts, dict) else None
        if k_sum is not None and not k_sum.empty:
            # Flatten into merchant-friendly strings
            rows = []
            for _, rr in k_sum.iterrows():
                tier = str(rr.get('Tier'))
                q25 = rr.get('Q25')
                q50 = rr.get('Q50_Median')
                q75 = rr.get('Q75')
                rows.append({'Metric': f'k_value_by_tier | {tier}', 'Value': f"Q25={q25:.0f}, Q50={q50:.0f}, Q75={q75:.0f}"})
            model_summary = pd.concat([model_summary, pd.DataFrame(rows)], ignore_index=True)
    except Exception:
        pass


    # --------------------------------------------------------------
    # Output formatting (merchant-friendly)
    # - Keep StoreID as first column
    # - Keep SKU & stock quantities as integers everywhere
    # --------------------------------------------------------------
    try:
        from engine.output_format import format_output
        tier_targets = format_output(tier_targets)
        add_list = format_output(add_list)
        remove_list = format_output(remove_list)
        peer_pool = format_output(peer_pool)
        store_grid = format_output(store_grid)
        model_summary = format_output(model_summary)
        sku_status_master = format_output(sku_status_master)
    except Exception:
        pass

    # Return outputs
    return {
        "Model_Summary": model_summary,
        "Stores_Enriched": stores_enriched,
        "Final_Targets": tier_targets,
        "Overall_Targets": overall_targets,
        "Add_List": add_list,
        "Remove_List": remove_list,
        "Peer_Pool_Top": peer_pool.head(int(ui_cfg.peer_pool_size)),
        "Store_SKU_Grid": store_grid,
        "Add_Shortfall_Report": add_shortfall_report,
        "Transfer_Recommendations": transfer_recommendations,
        "SKU_Status_Master": sku_status_master,
    }


def build_store_sku_grid(
    sku_master: pd.DataFrame,
    add_list: pd.DataFrame,
    remove_list: pd.DataFrame,
    network_stock: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """Create per-store final active SKU grid (CEO-ready).

    Final set = (CurrentActive - Remove) U Add
    CurrentActive means SKUs available in network (NetworkStock > threshold).
    """
    if sku_master is None or sku_master.empty:
        return pd.DataFrame()

    # Active network universe
    net = network_stock.set_index("ItemColorName")["NetworkStock"].to_dict() if network_stock is not None else {}

    cur = sku_master.copy()
    cur["NetworkStock"] = cur["ItemColorName"].map(lambda x: float(net.get(x, 0)))
    cur = cur[cur["NetworkStock"] > float(getattr(cfg, "active_network_stock_threshold", 5))].copy()

    # Current active SKU set per store
    cur_keep = cur[["StoreID", "ItemColorName", "Assigned_Tier"]].copy()
    cur_keep["Action"] = "CURRENT"

    # Remove SKUs
    if remove_list is not None and not remove_list.empty:
        rm = remove_list[["StoreID", "ItemColorName", "Assigned_Tier"]].copy()
        rm["Action"] = "REMOVE"
        # drop removed from current
        cur_keep = cur_keep.merge(
            rm[["StoreID", "ItemColorName"]],
            on=["StoreID", "ItemColorName"],
            how="left",
            indicator=True,
        )
        cur_keep = cur_keep[cur_keep["_merge"] == "left_only"].drop(columns=["_merge"])

    # Add SKUs (store id in add_list is StoreID_Target)
    adds = pd.DataFrame()
    if add_list is not None and not add_list.empty:
        cols = [c for c in ["StoreID_Target", "ItemColorName", "Assigned_Tier", "Score", "Reason"] if c in add_list.columns]
        adds = add_list[cols].copy()
        adds = adds.rename(columns={"StoreID_Target": "StoreID"})
        adds["Action"] = "ADD"

    # Combine
    grid = pd.concat([cur_keep, adds], ignore_index=True)

    # De-duplicate (prefer CURRENT/ADD)
    # If same SKU appears multiple times, keep highest priority: ADD > CURRENT
    priority = {"ADD": 2, "CURRENT": 1}
    grid["_prio"] = grid["Action"].map(lambda x: priority.get(x, 0))
    grid = grid.sort_values(["StoreID", "ItemColorName", "_prio"], ascending=[True, True, False])
    grid = grid.drop_duplicates(["StoreID", "ItemColorName"], keep="first").drop(columns=["_prio"])

    # Order nicely
    base_cols = ["StoreID", "Assigned_Tier", "ItemColorName", "Action"]
    extra_cols = [c for c in ["Score", "Reason", "NetworkStock"] if c in grid.columns]
    grid = grid[base_cols + extra_cols]
    grid = grid.sort_values(["StoreID", "Assigned_Tier", "Action"], ascending=[True, True, False])

    return grid



def build_sku_status_master(
    sku_scored: pd.DataFrame,
    add_list: pd.DataFrame,
    remove_list: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """Create a master SKU status table for merchandising workflows.

    Status categories:
    - REPEAT: Score >= repeat_score_threshold
    - WATCH:  watch_score_threshold <= Score < repeat_score_threshold
    - DROP:   Score < watch_score_threshold OR DiscountPenalty >= drop_discount_penalty_threshold

    Also attaches Action label (ADD / REMOVE / CURRENT) where applicable.
    """
    if sku_scored is None or sku_scored.empty:
        return pd.DataFrame()

    df = sku_scored.copy()
    df["StoreID"] = df["StoreID"].astype(str)
    df["ItemColorName"] = df["ItemColorName"].astype(str)

    rep_thr = float(getattr(cfg, "repeat_score_threshold", 0.70))
    wat_thr = float(getattr(cfg, "watch_score_threshold", 0.40))
    pen_thr = float(getattr(cfg, "drop_discount_penalty_threshold", 0.05))

    # Action map
    add_set = set()
    if add_list is not None and not add_list.empty:
        sid_col = "StoreID_Target" if "StoreID_Target" in add_list.columns else "StoreID"
        add_set = set(zip(add_list[sid_col].astype(str), add_list["ItemColorName"].astype(str)))

    rem_set = set()
    if remove_list is not None and not remove_list.empty:
        sid_col = "StoreID" if "StoreID" in remove_list.columns else "StoreID_Target"
        if "StoreID" in remove_list.columns and "ItemColorName" in remove_list.columns:
            rem_set = set(zip(remove_list["StoreID"].astype(str), remove_list["ItemColorName"].astype(str)))

    def _action(row):
        key = (row["StoreID"], row["ItemColorName"])
        if key in add_set:
            return "ADD"
        if key in rem_set:
            return "REMOVE"
        return "CURRENT"

    df["Action"] = df.apply(_action, axis=1)

    score = pd.to_numeric(df.get("Score", 0), errors="coerce").fillna(0.0)
    penalty = pd.to_numeric(df.get("DiscountPenalty", 0), errors="coerce").fillna(0.0)

    status = np.where(score >= rep_thr, "REPEAT",
             np.where(score >= wat_thr, "WATCH", "DROP"))
    status = np.where(penalty >= pen_thr, "DROP", status)

    df["SKU_Status"] = status

    keep_cols = [c for c in [
        "StoreID","Assigned_Tier","PriceTier","Dept","Category","ItemColorName",
        "Score","GMROI","ROS","DiscountPct","DiscountPenalty","NetworkStock",
        "SKU_Status","Action"
    ] if c in df.columns]
    return df[keep_cols].sort_values(["StoreID","SKU_Status","Score"], ascending=[True, True, False])
def render_optimization_page():
    """Render the optimization execution page."""
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    render_card_header("🚀", "Running Optimization", "Processing your data through the optimization engine", "success")

    if st.session_state.optimization_complete and st.session_state.optimization_results is not None:
        render_alert("success", "Optimization Complete", "The optimization has finished successfully. View your results below.")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("View Results →", use_container_width=True, type="primary"):
                navigate_to(WorkflowStep.RESULTS)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_cb(msg: str, pct: int):
        status_text.markdown(f"**{msg}**")
        progress_bar.progress(int(max(0, min(100, pct))))

    try:
        progress_cb("Initializing...", 5)

        stores_df = st.session_state.stores_df
        tiers_df = st.session_state.tiers_df
        articles_df = st.session_state.articles_df
        config = st.session_state.config

        if stores_df is None or tiers_df is None or articles_df is None:
            raise ValueError("Input data not loaded. Please go back and upload a valid dataset.")

        results = execute_portfolio_optimization(
            stores_df=stores_df,
            tiers_df=tiers_df,
            articles_df=articles_df,
            ui_cfg=config,
            progress_cb=progress_cb,
        )

        st.session_state.optimization_results = results
        st.session_state.optimization_complete = True
        progress_cb("Optimization completed ✅", 100)

        time.sleep(0.3)
        try:
            st.rerun()
        except AttributeError:
            try:
                st.experimental_rerun()
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Optimization error: {str(e)}\n{traceback.format_exc()}")
        render_alert("error", "Optimization Failed", str(e))
        if st.button("← Back to Configuration"):
            navigate_to(WorkflowStep.CONFIGURATION)

    st.markdown('</div>', unsafe_allow_html=True)


def render_results_page():
    """Render the results page."""
    if not st.session_state.optimization_complete:
        render_alert("warning", "No Results Available", "Please run the optimization first.")
        if st.button("← Back to Configuration"):
            navigate_to(WorkflowStep.CONFIGURATION)
        return
    
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    render_card_header("📊", "Optimization Results", "Review and download your optimization outputs", "success")
    
    results = st.session_state.optimization_results
    config = st.session_state.config
    
    # Summary metrics
    summary = st.session_state.data_summary
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Stores Optimized", f"{summary.total_stores:,}", "success")
    with col2:
        render_metric_card("Active SKUs", f"{summary.total_skus:,}", "success")
    with col3:
        render_metric_card("Price Tiers", str(summary.total_tiers), "info")
    with col4:
        render_metric_card("Network Stock Filter", f"> {config.network_stock_threshold} units", "primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Results tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Model Summary",
        "🎯 Final Targets",
        "➕ Add List",
        "➖ Remove List",
        "🏆 Peer Pool",
        "📊 CEO Grid"
    ])

    with tab1:
        st.markdown("### Model Summary")
        st.dataframe(results.get("Model_Summary", pd.DataFrame()), use_container_width=True, hide_index=True)

        st.markdown("### Stores Enriched (Diagnostics)")
        st.dataframe(results.get("Stores_Enriched", pd.DataFrame()).head(50), use_container_width=True)
        if isinstance(results.get("Stores_Enriched"), pd.DataFrame):
            st.caption(f"Showing 50 of {len(results['Stores_Enriched']):,} stores")

    with tab2:
        st.markdown("### Final Targets (Store × PriceTier)")
        df = results.get("Final_Targets", pd.DataFrame())
        st.dataframe(df.head(200), use_container_width=True)
        if not df.empty:
            st.caption(f"Showing 200 of {len(df):,} rows")

    with tab3:
        st.markdown("### Add List (Active SKUs only)")
        df = results.get("Add_List", pd.DataFrame())
        st.dataframe(df.head(200), use_container_width=True)
        if not df.empty:
            st.caption(f"Showing 200 of {len(df):,} rows")

    with tab4:
        st.markdown("### Remove List (Active SKUs only)")
        df = results.get("Remove_List", pd.DataFrame())
        st.dataframe(df.head(200), use_container_width=True)
        if not df.empty:
            st.caption(f"Showing 200 of {len(df):,} rows")

    with tab5:
        st.markdown("### Peer Pool Top")
        df = results.get("Peer_Pool_Top", pd.DataFrame())
        st.dataframe(df.head(300), use_container_width=True)
        if not df.empty:
            st.caption(f"Showing 300 of {len(df):,} rows")

    with tab6:
        st.markdown("### CEO Store-SKU Grid (Final)")
        grid = results.get("Store_SKU_Grid", pd.DataFrame())
        if grid is None or grid.empty:
            st.warning("No grid data available")
        else:
            stores = sorted(grid["StoreID"].astype(str).unique())
            selected_store = st.selectbox("Select Store", options=stores)
            st.markdown(f"**Final Active SKU Grid for Store: {selected_store}**")
            st.caption(f"Includes CURRENT active SKUs minus REMOVES plus ADDS. Active universe defined by NetworkStock > {config.network_stock_threshold}.")
            view = grid[grid["StoreID"].astype(str) == str(selected_store)].copy()
            st.dataframe(view, use_container_width=True)
            st.caption(f"{len(view):,} SKUs in final grid")

    # Download section
    st.markdown("---")
    st.markdown("### Download Results")

    from engine.exports import export_excel_bytes
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Portfolio_Optimization_Outputs_{ts}.xlsx"

    excel_buf = export_excel_bytes({
        "Model_Summary": results.get("Model_Summary"),
        "Stores_Enriched": results.get("Stores_Enriched"),
        "Final_Targets": results.get("Final_Targets"),
        "Add_List": results.get("Add_List"),
        "Remove_List": results.get("Remove_List"),
        "Peer_Pool_Top": results.get("Peer_Pool_Top"),
        "Store_SKU_Grid": results.get("Store_SKU_Grid"),
    })

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 Download Complete Excel Report",
            data=excel_buf,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary"
        )
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← Back to Configuration", use_container_width=True):
            navigate_to(WorkflowStep.CONFIGURATION)
    
    with col3:
        if st.button("🔄 Start New Analysis", use_container_width=True):
            reset_session_state()
            try:
                st.rerun()
            except AttributeError:
                try:
                    st.experimental_rerun()
                except Exception:
                    pass


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title=f"{AppConfig.APP_NAME} | {AppConfig.COMPANY_NAME}",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply premium CSS
    st.markdown(get_premium_css(), unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Render header
    render_header()
    
    # Render workflow stepper
    render_workflow_stepper(st.session_state.current_step)
    
    # Render current page
    current_step = st.session_state.current_step
    
    if current_step == WorkflowStep.UPLOAD:
        render_upload_page()
    elif current_step == WorkflowStep.QUALITY_CHECK:
        render_quality_check_page()
    elif current_step == WorkflowStep.CONFIGURATION:
        render_configuration_page()
    elif current_step == WorkflowStep.OPTIMIZATION:
        render_optimization_page()
    elif current_step == WorkflowStep.RESULTS:
        render_results_page()
    
    # Footer
    st.markdown(f"""
    <div class="premium-footer">
        © {AppConfig.COPYRIGHT_YEAR} {AppConfig.COMPANY_NAME}. All rights reserved. | 
        {AppConfig.APP_NAME} v{AppConfig.APP_VERSION} | 
        Enterprise Edition
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()