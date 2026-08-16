"""
FactoryPulse AI - Full Industrial SCADA + AI SaaS Platform
Single-file Flask application. Backend + embedded HTML/CSS/JS frontend.
Multi-user auth (JWT + bcrypt), PostgreSQL persistence (SQLAlchemy), a real-time
SCADA "Live Monitor" (WebSocket-pushed machine telemetry), Gemini-ready AI anomaly
detection, and 6 data modes: SIMULATION / USB (serial) / PLC (Siemens S7) /
MODBUS (Modbus TCP) / OPCUA (OPC UA) / MQTT.

Every real-hardware mode automatically and transparently falls back to SIMULATION
if the device/broker/server can't be reached, and each live reading is tagged
source="auto" (genuinely pulled from SCADA/PLC/protocol) or source="manual_baseline"
(no live feed - drifting around the manually entered values) so the dashboard can
show operators exactly where each number came from.

Run:
    pip install flask google-generativeai sqlalchemy psycopg2-binary bcrypt pyjwt python-dotenv reportlab

    Optional, for the full SCADA feature set (the app runs fine without them,
    it just falls back to simulation / HTTP polling):
        pip install flask-socketio   # real-time WebSocket push (else: polling)
        pip install pyserial         # USB/serial sensor mode (Arduino/ESP32)
        pip install python-snap7     # Siemens S7 PLC mode
        pip install pymodbus         # Modbus TCP mode (PLCs, meters, sensor gateways)
        pip install opcua            # OPC UA mode (most modern industrial servers)
        pip install paho-mqtt        # MQTT mode (IoT gateways, brokers)

    Note: reportlab powers the "Download Report" PDF export (Reports page). Without
    it, /api/report/pdf returns a clear 503 instead of crashing the app.

    Create a file named ".env" next to app.py (never commit it to git) with:
        GEMINI_API_KEY=your_key
        DATABASE_URL=postgresql://user:password@localhost/factorypulse
        JWT_SECRET=your_random_secret
        DATA_MODE=SIMULATION           # SIMULATION | USB | PLC | MODBUS | OPCUA | MQTT
        SERIAL_PORT=COM3               # only used in USB mode
        PLC_IP=192.168.0.1              # only used in PLC mode
        MODBUS_HOST=192.168.0.10        # only used in MODBUS mode
        MODBUS_PORT=502
        MODBUS_UNIT=1
        OPCUA_ENDPOINT=opc.tcp://192.168.0.20:4840   # only used in OPCUA mode
        MQTT_BROKER=localhost           # only used in MQTT mode
        MQTT_PORT=1883
        MQTT_TOPIC_PREFIX=factorypulse  # expects JSON on <prefix>/<machineCode>/telemetry

        # Optional: email notifications for critical alerts (silently skipped if unset)
        SMTP_HOST=smtp.gmail.com
        SMTP_PORT=587
        SMTP_USER=your_email@gmail.com
        SMTP_PASSWORD=your_app_password
        ALERT_EMAIL_TO=engineer@yourfactory.com

    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
    python app.py

Note: if PostgreSQL is not reachable, the app automatically falls back to a local
SQLite file (factorypulse.db) so it always runs out of the box for development/demo.
"""

import os
import re
import json
import random
import datetime
import functools
import time
import threading
import secrets
from flask import Flask, request, jsonify, g, redirect, send_file, make_response

REPORTLAB_AVAILABLE = False
PDF_FONT = "Helvetica"
PDF_FONT_BOLD = "Helvetica-Bold"
PDF_CJK_FONT = None
try:
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_AVAILABLE = True

    # The built-in Helvetica has no Cyrillic glyphs, so Kazakh/Russian factory and
    # machine names would render as black boxes. Register a Unicode TTF instead.
    # Vera ships inside reportlab itself, so this works on any host; system fonts
    # are preferred when present because they cover more scripts.
    def _register_pdf_font():
        import os as _os
        candidates = [
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans"),
            ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "LiberationSans"),
        ]
        import reportlab as _rl
        bundled = _os.path.join(_os.path.dirname(_rl.__file__), "fonts")
        candidates.append((_os.path.join(bundled, "Vera.ttf"),
                           _os.path.join(bundled, "VeraBd.ttf"), "Vera"))

        for regular, bold, name in candidates:
            try:
                if _os.path.exists(regular):
                    pdfmetrics.registerFont(TTFont(name, regular))
                    bold_name = name
                    if _os.path.exists(bold):
                        bold_name = name + "-Bold"
                        pdfmetrics.registerFont(TTFont(bold_name, bold))
                    return name, bold_name
            except Exception:
                continue
        return "Helvetica", "Helvetica-Bold"

    PDF_FONT, PDF_FONT_BOLD = _register_pdf_font()

    # Latin/Cyrillic fonts contain no CJK glyphs, so Chinese/Japanese/Korean text
    # would render as blank space. Register a CJK font and use it for those
    # languages only; everything else keeps the (smaller) default font.
    PDF_CJK_FONT = None

    def _register_cjk_font():
        import os as _os
        candidates = [
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "NotoSansCJK", 0),
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-DemiLight.ttc", "NotoSansCJK", 0),
            ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "WenQuanYiZenHei", 0),
            ("/etc/alternatives/fonts-japanese-gothic.ttf", "JapaneseGothic", None),
        ]
        for path, name, index in candidates:
            if not _os.path.exists(path):
                continue
            try:
                if index is None:
                    pdfmetrics.registerFont(TTFont(name, path))
                else:
                    pdfmetrics.registerFont(TTFont(name, path, subfontIndex=index))
                return name
            except Exception:
                continue
        return None

    PDF_CJK_FONT = _register_cjk_font()
except ImportError:
    print("reportlab not installed - PDF report export disabled (pip install reportlab to enable).")

CJK_LANGS = {"zh", "ja", "ko"}

# One representative character per script. Used to check, before building a
# report, whether the chosen font can actually draw that language - otherwise
# the PDF comes out with blank gaps where the text should be.
LANG_SAMPLE_CHAR = {
    "zh": "\u4e2d", "ja": "\u65e5", "ko": "\ud55c",
    "hi": "\u092a",   # Devanagari
    "ar": "\u0627",   # Arabic
    "ru": "\u0416", "kk": "\u049a", "uk": "\u0407", "ky": "\u04ae", "uz": "\u0413",
}

# Extra font candidates for scripts that Latin/Cyrillic fonts do not cover.
INDIC_FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf", "LohitDevanagari"),
    ("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf", "NotoSansDevanagari"),
    ("/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.otf", "NotoSansDevanagari"),
    ("/usr/share/fonts/truetype/fonts-deva-extra/samanata.ttf", "Samanata"),
    ("/usr/share/fonts/truetype/Sarai/Sarai.ttf", "Sarai"),
    # Bundled with the app if you ship one yourself
    ("fonts/NotoSansDevanagari-Regular.ttf", "NotoSansDevanagari"),
]
PDF_INDIC_FONT = None


def _try_register_indic():
    global PDF_INDIC_FONT
    if not REPORTLAB_AVAILABLE or PDF_INDIC_FONT is not None:
        return PDF_INDIC_FONT
    import os as _os
    for path, name in INDIC_FONT_CANDIDATES:
        if _os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                PDF_INDIC_FONT = name
                return PDF_INDIC_FONT
            except Exception:
                continue
    return None


def _font_supports(font_name, char):
    """True if the registered font has a glyph for this character."""
    if not char:
        return True
    try:
        face = pdfmetrics.getFont(font_name).face
        # TTFont exposes the parsed cmap through its face object.
        cmap = getattr(face, "charToGlyph", None)
        if cmap is None:
            return True   # built-in Type1 fonts: assume Latin-only callers
        return ord(char) in cmap
    except Exception:
        return True


def pdf_fonts_for(lang):
    """Returns (regular, bold) font names appropriate for the report language."""
    if lang in CJK_LANGS and REPORTLAB_AVAILABLE and PDF_CJK_FONT:
        return PDF_CJK_FONT, PDF_CJK_FONT   # CJK families rarely ship a separate bold
    if lang == "hi" and REPORTLAB_AVAILABLE:
        indic = _try_register_indic()
        if indic:
            return indic, indic
    return PDF_FONT, PDF_FONT_BOLD


def pdf_lang_or_fallback(lang):
    """Guarantees a readable report.

    If the host has no font able to draw this language's script, an English
    report is far more useful than a correctly-translated one full of blank
    boxes - so fall back rather than ship something unreadable."""
    if lang not in PDF_TEXT:
        return "en"
    if not REPORTLAB_AVAILABLE:
        return lang
    sample = LANG_SAMPLE_CHAR.get(lang)
    if not sample:
        return lang   # Latin-script languages are always covered
    font, _ = pdf_fonts_for(lang)
    return lang if _font_supports(font, sample) else "en"

# ------------------------------------------------------------------
# LOAD SECRETS FROM .env (keeps API keys out of the source file)
# ------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed - install it with 'pip install python-dotenv' "
          "to load secrets from a .env file. Falling back to system environment variables only.")

# ------------------------------------------------------------------
# GEMINI AI SETUP
# ------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
GEMINI_ENABLED = False
model = None

try:
    import google.generativeai as genai
    if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-3.5-flash")
        GEMINI_ENABLED = True
except Exception as e:
    print("Gemini not available, using deterministic fallback engine:", e)
    GEMINI_ENABLED = False


def _extract_json(text):
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


# ------------------------------------------------------------------
# DATABASE (PostgreSQL via SQLAlchemy, with local SQLite fallback)
# ------------------------------------------------------------------
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey, inspect, text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, scoped_session

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:password@localhost/factorypulse")

# Connection pooling. Postgres has a hard connection limit (Render's free tier is
# ~97), and every gunicorn worker keeps its own pool, so:
#     total connections = workers x (DB_POOL_SIZE + DB_MAX_OVERFLOW)
# Keep that product comfortably under your database's limit.
DB_POOL_SIZE = int(os.environ.get("DB_POOL_SIZE", "5"))
DB_MAX_OVERFLOW = int(os.environ.get("DB_MAX_OVERFLOW", "5"))
DB_POOL_RECYCLE = int(os.environ.get("DB_POOL_RECYCLE", "280"))  # under typical 300s idle timeouts

try:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,        # silently drops dead connections instead of erroring
        pool_size=DB_POOL_SIZE,
        max_overflow=DB_MAX_OVERFLOW,
        pool_recycle=DB_POOL_RECYCLE,
        pool_timeout=30,
    )
    with engine.connect():
        pass
    print(f"Connected to database: {DATABASE_URL.split('@')[-1]} "
          f"(pool={DB_POOL_SIZE}+{DB_MAX_OVERFLOW})")
except Exception as e:
    print(f"PostgreSQL not reachable ({e}). Falling back to local SQLite (factorypulse.db).")
    DATABASE_URL = "sqlite:///factorypulse.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    full_name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="engineer")  # engineer | manager | admin
    # Interface language, remembered so emails sent from background jobs (where
    # there is no request context) are written in the language the user chose.
    preferred_lang = Column(String(5), default="en")
    reset_token = Column(String(255), nullable=True)
    reset_token_expires = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    factories = relationship("Factory", back_populates="owner", cascade="all, delete-orphan")
    scada_machines = relationship("Machine", back_populates="owner", cascade="all, delete-orphan")


class Factory(Base):
    __tablename__ = "factories"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    factory_name = Column(String(200), nullable=False)
    machines = Column(Integer, default=6)
    machine_type = Column(String(80), default="CNC")
    energy_cost = Column(Float, default=0.12)
    temperature = Column(Float, default=65)
    vibration = Column(Float, default=3.5)
    load = Column(Float, default=60)
    ai_insights = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    owner = relationship("User", back_populates="factories")


class Machine(Base):
    """Full SCADA machine record: identity, live sensors, status, AI prediction, notes."""
    __tablename__ = "machines"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # SECTION 1: Machine Info
    machine_code = Column(String(50), nullable=False)   # e.g. "M-01"
    machine_name = Column(String(150), nullable=False)
    factory_section = Column(String(120), default="")
    operator_name = Column(String(120), default="")

    # SECTION 2: Sensor Data
    temperature = Column(Float, default=0)
    vibration = Column(Float, default=0)
    load = Column(Float, default=0)
    pressure = Column(Float, default=0)
    # Design/operating pressure for THIS machine. Pressure limits are meaningless in
    # the absolute (a pump runs at 8 bar, a compressor at 60), so the AI judges
    # deviation from this nominal rather than a fixed threshold.
    nominal_pressure = Column(Float, default=0)
    voltage = Column(Float, default=0)
    current = Column(Float, default=0)

    # SECTION 3: Status
    status = Column(String(20), default="running")  # running | stopped | maintenance
    error_code = Column(String(50), default="")
    priority_level = Column(String(20), default="normal")  # low | normal | high | critical

    # SECTION 4: AI Prediction
    failure_risk = Column(Float, default=0)
    estimated_failure_time = Column(String(120), default="")

    # SECTION 5: Notes
    notes = Column(Text, default="")

    # ENERGY INTELLIGENCE: expected/actual daily production count, used to compute
    # Specific Energy Consumption (kWh per unit of output)
    daily_output_units = Column(Float, default=0)

    data_mode = Column(String(20), default="SIMULATION")  # SIMULATION | USB | PLC
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    owner = relationship("User", back_populates="scada_machines")


class Alert(Base):
    """Persisted log of critical/warning events, used for the real-time Alerts page
    and for the pilot summary PDF report ('incidents caught before failure').
    Raw sensor values are stored (not a pre-rendered sentence) so the frontend can
    display the alert in whichever language the viewer currently has selected."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    machine_id = Column(Integer, ForeignKey("machines.id"), nullable=True, index=True)
    machine_code = Column(String(50), default="")
    machine_name = Column(String(150), default="")
    severity = Column(String(20), default="warning")  # warning | critical
    alert_type = Column(String(20), default="critical")  # critical | idle_waste
    message = Column(Text, default="")  # English fallback text (used in email/PDF)
    alert_temperature = Column(Float, default=0)
    alert_vibration = Column(Float, default=0)
    alert_status = Column(String(20), default="")
    alert_value = Column(Float, default=0)  # generic numeric payload (e.g. idle kW wasted)
    suggested_actions = Column(String(255), default="")  # comma-separated action codes
    acknowledged = Column(Integer, default=0)  # 0/1 (SQLite-friendly boolean)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ProductionShift(Base):
    """A production shift, which is what OEE is actually calculated over.

    OEE (Availability x Performance x Quality, ISO 22400) is THE standard KPI in
    manufacturing - it is the first number any plant manager asks for, and it is
    what makes this comparable to Siemens/GE-class systems."""
    __tablename__ = "production_shifts"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    machine_id = Column(Integer, ForeignKey("machines.id"), nullable=True, index=True)

    shift_name = Column(String(80), default="Shift A")
    shift_date = Column(DateTime, default=datetime.datetime.utcnow)

    planned_minutes = Column(Float, default=480)      # e.g. 8-hour shift
    downtime_minutes = Column(Float, default=0)
    downtime_reason = Column(String(80), default="")  # breakdown | changeover | no_material | ...

    ideal_cycle_seconds = Column(Float, default=30)   # design speed per unit
    total_units = Column(Integer, default=0)
    good_units = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class WorkOrder(Base):
    """A maintenance task. This closes the loop on alerts: the AI says a bearing
    is failing, someone gets assigned, and completion is tracked. Without this an
    alert is just a notification nobody owns."""
    __tablename__ = "work_orders"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    machine_id = Column(Integer, ForeignKey("machines.id"), nullable=True, index=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=True)

    title = Column(String(200), nullable=False)
    description = Column(Text, default="")
    priority = Column(String(20), default="medium")     # low | medium | high | critical
    status = Column(String(20), default="open")         # open | in_progress | done | cancelled
    assigned_to = Column(String(120), default="")

    root_cause = Column(String(80), default="")
    actions = Column(String(255), default="")           # comma-separated action codes

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    due_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    completion_note = Column(Text, default="")


class AuditLog(Base):
    """Who changed what, and when. Required for ISO-style audits in real plants."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    user_email = Column(String(255), default="")
    action = Column(String(80), default="")             # e.g. machine.create, workorder.complete
    entity_type = Column(String(50), default="")
    entity_id = Column(Integer, nullable=True)
    detail = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)


class TelemetryHistory(Base):
    """Persisted sensor history.

    Signal history used to live only in memory (last 40 samples), so nothing
    survived a restart and no long-range trend was possible. A pilot needs to
    prove 'here is what changed over 10 days', which requires stored history."""
    __tablename__ = "telemetry_history"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    machine_id = Column(Integer, ForeignKey("machines.id"), nullable=False, index=True)
    machine_code = Column(String(50), default="")

    temperature = Column(Float, default=0)
    vibration = Column(Float, default=0)
    load = Column(Float, default=0)
    power_kw = Column(Float, default=0)
    risk = Column(Float, default=0)
    status = Column(String(20), default="")

    recorded_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)


class DashboardState(Base):
    """Per-user state for the overview dashboard's factory-input form.

    Previously this lived in a single module-level dict, which meant it was lost on
    restart AND shared across every logged-in user. It is now persisted per account."""
    __tablename__ = "dashboard_states"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)
    factory_name = Column(String(200), default="Demo Factory")
    machine_count = Column(Integer, default=6)
    machine_type = Column(String(80), default="CNC")
    energy_cost = Column(Float, default=0.12)
    temperature = Column(Float, default=65)
    vibration = Column(Float, default=3.5)
    load = Column(Float, default=60)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class PasswordReset(Base):
    """Short-lived 6-digit email verification codes for the 'Forgot password' flow."""
    __tablename__ = "password_resets"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    email = Column(String(255), nullable=False, index=True)
    code = Column(String(10), nullable=False)
    used = Column(Integer, default=0)  # 0/1
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


Base.metadata.create_all(engine)


def _ensure_schema_migrations():
    """Lightweight, dependency-free auto-migration: if this database file was created
    by an earlier version of this app (before some columns existed), add the missing
    columns instead of crashing with 'no such column' errors. Existing rows/accounts
    are never touched or deleted - this only ever ADDS columns."""
    try:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        migrations = {
            "users": [
                ("role", "VARCHAR(20) DEFAULT 'operator'"),
                ("preferred_lang", "VARCHAR(5) DEFAULT 'en'"),
                ("reset_token", "VARCHAR(255)"),
                ("reset_token_expires", "DATETIME"),
            ],
            "factories": [
                ("ai_insights", "TEXT"),
            ],
            "machines": [
                ("data_mode", "VARCHAR(20) DEFAULT 'SIMULATION'"),
                ("failure_risk", "FLOAT DEFAULT 0"),
                ("estimated_failure_time", "VARCHAR(120) DEFAULT ''"),
                ("daily_output_units", "FLOAT DEFAULT 0"),
                ("nominal_pressure", "FLOAT DEFAULT 0"),
            ],
            "alerts": [
                ("alert_temperature", "FLOAT DEFAULT 0"),
                ("alert_vibration", "FLOAT DEFAULT 0"),
                ("alert_status", "VARCHAR(20) DEFAULT ''"),
                ("alert_type", "VARCHAR(20) DEFAULT 'critical'"),
                ("alert_value", "FLOAT DEFAULT 0"),
                ("suggested_actions", "VARCHAR(255) DEFAULT ''"),
            ],
        }
        with engine.connect() as conn:
            for table, columns in migrations.items():
                if table not in table_names:
                    continue
                existing_cols = {c["name"] for c in inspector.get_columns(table)}
                for col_name, col_def in columns:
                    if col_name not in existing_cols:
                        try:
                            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}"))
                            conn.commit()
                            print(f"Database migration: added missing column '{col_name}' to '{table}'.")
                        except Exception as e:
                            print(f"Migration warning ({table}.{col_name}): {e}")
    except Exception as e:
        print("Schema migration check skipped due to error:", e)


_ensure_schema_migrations()


# ------------------------------------------------------------------
# PASSWORD HASHING (bcrypt)
# ------------------------------------------------------------------
import bcrypt


def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password, password_hash):
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


# ------------------------------------------------------------------
# JWT AUTH
# ------------------------------------------------------------------
import jwt as pyjwt

JWT_SECRET = os.environ.get("JWT_SECRET", "factorypulse-dev-secret-change-me-in-production")
JWT_ALGO = "HS256"


def create_jwt(user_id, remember=False):
    exp_hours = 24 * 30 if remember else 24
    now = datetime.datetime.utcnow()
    payload = {"user_id": user_id, "iat": now, "exp": now + datetime.timedelta(hours=exp_hours)}
    token = pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
    return token if isinstance(token, str) else token.decode("utf-8")


def decode_jwt(token):
    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload.get("user_id")
    except Exception:
        return None


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_strong_password(password):
    return bool(password) and len(password) >= 8 and re.search(r"[A-Za-z]", password) and re.search(r"[0-9]", password)


def require_auth(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.replace("Bearer ", "").strip()
        user_id = decode_jwt(token) if token else None
        if not user_id:
            return jsonify({"error": "unauthorized"}), 401
        db = SessionLocal()
        user = db.get(User, user_id)
        if not user:
            return jsonify({"error": "unauthorized"}), 401
        g.user = user
        g.db = db
        return f(*args, **kwargs)
    return wrapper


ROLE_CAPABILITIES = {
    "engineer": {"technical": True, "business": False, "admin": False},
    "manager": {"technical": False, "business": True, "admin": False},
    "admin": {"technical": True, "business": True, "admin": True},
}


def serialize_user(user):
    role = user.role or "engineer"
    if role == "operator":  # legacy value from earlier versions
        role = "engineer"
    return {
        "id": user.id, "full_name": user.full_name, "email": user.email,
        "role": role,
        "capabilities": ROLE_CAPABILITIES.get(role, ROLE_CAPABILITIES["engineer"]),
    }


def serialize_factory(factory):
    insights = None
    if factory.ai_insights:
        try:
            insights = json.loads(factory.ai_insights)
        except Exception:
            insights = None
    return {
        "id": factory.id,
        "factory_name": factory.factory_name,
        "machines": factory.machines,
        "machine_type": factory.machine_type,
        "energy_cost": factory.energy_cost,
        "temperature": factory.temperature,
        "vibration": factory.vibration,
        "load": factory.load,
        "ai_insights": insights,
        "created_at": factory.created_at.isoformat() if factory.created_at else None,
    }


def factory_to_state(factory):
    return {
        "factory_name": factory.factory_name,
        "machine_count": factory.machines,
        "energy_cost": factory.energy_cost,
        "machine_type": factory.machine_type,
        "temperature": factory.temperature,
        "vibration": factory.vibration,
        "load": factory.load,
    }


def serialize_machine(m):
    return {
        "id": m.id,
        "machine_code": m.machine_code,
        "machine_name": m.machine_name,
        "factory_section": m.factory_section,
        "operator_name": m.operator_name,
        "temperature": m.temperature,
        "vibration": m.vibration,
        "load": m.load,
        "pressure": m.pressure,
        "voltage": m.voltage,
        "current": m.current,
        "status": m.status,
        "error_code": m.error_code,
        "priority_level": m.priority_level,
        "failure_risk": m.failure_risk,
        "estimated_failure_time": m.estimated_failure_time,
        "notes": m.notes,
        "daily_output_units": m.daily_output_units,
        "data_mode": m.data_mode,
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "updated_at": m.updated_at.isoformat() if m.updated_at else None,
    }


ALERT_PRIORITY_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}


def serialize_alert(a):
    actions = [x for x in (a.suggested_actions or "").split(",") if x]
    return {
        "id": a.id,
        "machine_id": a.machine_id,
        "machine_code": a.machine_code,
        "machine_name": a.machine_name,
        "severity": a.severity,
        "priority_rank": ALERT_PRIORITY_RANK.get(a.severity, 0),
        "alert_type": a.alert_type,
        "suggested_actions": actions,
        "message": a.message,
        "temperature": a.alert_temperature,
        "vibration": a.alert_vibration,
        "status": a.alert_status,
        "value": a.alert_value,
        "acknowledged": bool(a.acknowledged),
        "created_at": a.created_at.isoformat() if a.created_at else None,
    }


def standard_reading(machine_id, temperature, vibration, load, pressure, voltage, current, status):
    """The canonical FactoryPulse sensor payload shape used across SIMULATION / USB / PLC modes."""
    return {
        "machineId": machine_id,
        "temperature": round(temperature, 1),
        "vibration": round(vibration, 2),
        "load": round(load, 1),
        "pressure": round(pressure, 2),
        "voltage": round(voltage, 1),
        "current": round(current, 2),
        "status": status,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


# ------------------------------------------------------------------
MACHINE_TYPES = ["CNC", "Compressor", "Conveyor", "Motor", "Pump", "Press", "Robot Arm"]

# Default values used when a user opens the dashboard for the very first time.
# Real per-user values live in the DashboardState table (see get_dashboard_state).
DEFAULT_FACTORY_STATE = {
    "factory_name": "Demo Factory",
    "machine_count": 6,
    "energy_cost": 0.12,
    "machine_type": "CNC",
    "temperature": 65,
    "vibration": 3.5,
    "load": 60,
}


def get_dashboard_state(db, user_id):
    """Loads this user's saved dashboard input, creating it on first use.
    Returns a plain dict in the shape the KPI/analysis helpers expect."""
    row = db.query(DashboardState).filter_by(user_id=user_id).first()
    if row is None:
        row = DashboardState(user_id=user_id, **{
            "factory_name": DEFAULT_FACTORY_STATE["factory_name"],
            "machine_count": DEFAULT_FACTORY_STATE["machine_count"],
            "machine_type": DEFAULT_FACTORY_STATE["machine_type"],
            "energy_cost": DEFAULT_FACTORY_STATE["energy_cost"],
            "temperature": DEFAULT_FACTORY_STATE["temperature"],
            "vibration": DEFAULT_FACTORY_STATE["vibration"],
            "load": DEFAULT_FACTORY_STATE["load"],
        })
        db.add(row)
        db.commit()
    return row


def dashboard_state_to_dict(row):
    return {
        "factory_name": row.factory_name,
        "machine_count": row.machine_count,
        "energy_cost": row.energy_cost,
        "machine_type": row.machine_type,
        "temperature": row.temperature,
        "vibration": row.vibration,
        "load": row.load,
    }


def _machine_status(temperature, vibration, load):
    if temperature > 85 or vibration > 7 or load > 95:
        return "critical"
    if temperature > 70 or vibration > 4 or load > 80:
        return "warning"
    return "running"


def generate_machines(state):
    machines = []
    for i in range(int(state["machine_count"])):
        jitter_t = random.uniform(-8, 8)
        jitter_v = random.uniform(-1.5, 1.5)
        jitter_l = random.uniform(-10, 10)
        temperature = round(max(20, state["temperature"] + jitter_t), 1)
        vibration = round(max(0, state["vibration"] + jitter_v), 2)
        load = round(max(0, min(100, state["load"] + jitter_l)), 1)
        status = _machine_status(temperature, vibration, load)
        machines.append({
            "id": i + 1,
            "name": f"{state['machine_type']}-{i + 1:02d}",
            "temperature": temperature,
            "vibration": vibration,
            "load": load,
            "status": status,
        })
    return machines


def compute_kpis(machines, state):
    total = len(machines) or 1
    active = len([m for m in machines if m["status"] == "running"])
    alerts = len([m for m in machines if m["status"] in ("warning", "critical")])
    avg_load = sum(m["load"] for m in machines) / total
    avg_temp = sum(m["temperature"] for m in machines) / total
    energy_usage = round(state["machine_count"] * avg_load * state["energy_cost"] * random.uniform(0.9, 1.1), 1)
    efficiency = round(max(40, min(99, 100 - (avg_load - 50) * 0.3 - max(0, avg_temp - 60) * 0.4)), 1)
    return {
        "energy_usage": energy_usage,
        "efficiency": efficiency,
        "active_machines": active,
        "total_machines": len(machines),
        "alerts": alerts,
    }


RULE_TEMPLATES = {
'en': {'risk_issue':"{critical} machine(s) in CRITICAL state and {warning} in WARNING state. Average load and temperature trends suggest elevated mechanical stress on {type} units.",'risk_ok':"No critical anomalies detected across {count} {type} machines. Overall risk level is low.",'efficiency':"Current plant efficiency is {eff}% with energy usage at {energy} kWh. {active} of {total} machines are running optimally.",'optimization':"Schedule preventive maintenance for flagged machines within 48 hours, recalibrate load balancing across the line, and consider reducing peak-hour load by 10-15% to lower energy cost without impacting throughput.",'pred_critical':"Motor failure likely in {days} day(s)",'pred_warning':"Elevated wear detected, potential failure within {days} days",'pred_ok':"No imminent failure expected",'rec_critical':"Stop the machine, reduce load immediately, and inspect bearings and cooling system.",'rec_warning':"Schedule inspection within 24 hours and reduce load by 10-15%.",'rec_ok':"Continue routine monitoring; no action needed."},
'ru': {'risk_issue':"{critical} станок(ов) в состоянии КРИТИЧНО и {warning} в состоянии ВНИМАНИЕ. Тенденции нагрузки и температуры указывают на повышенную механическую нагрузку на оборудование типа {type}.",'risk_ok':"Критических аномалий не обнаружено среди {count} станков типа {type}. Общий уровень риска низкий.",'efficiency':"Текущая эффективность завода составляет {eff}% при потреблении энергии {energy} кВт·ч. {active} из {total} станков работают оптимально.",'optimization':"Запланируйте профилактическое обслуживание отмеченных станков в течение 48 часов, перекалибруйте балансировку нагрузки по линии и рассмотрите снижение пиковой нагрузки на 10-15% для экономии энергии без потери производительности.",'pred_critical':"Вероятен отказ двигателя через {days} дн.",'pred_warning':"Обнаружен повышенный износ, возможен отказ в течение {days} дней",'pred_ok':"Неминуемый отказ не ожидается",'rec_critical':"Немедленно остановите станок, снизьте нагрузку и проверьте подшипники и систему охлаждения.",'rec_warning':"Запланируйте осмотр в течение 24 часов и снизьте нагрузку на 10-15%.",'rec_ok':"Продолжайте плановый мониторинг; действий не требуется."},
'kk': {'risk_issue':"{critical} станок СЫНИ күйде, {warning} станок ЕСКЕРТУ күйінде. Жүктеме мен температура үрдісі {type} жабдығына түсетін механикалық қысымның артқанын көрсетеді.",'risk_ok':"{count} {type} станогының арасында сыни ақаулар табылмады. Жалпы тәуекел деңгейі төмен.",'efficiency':"Зауыттың қазіргі тиімділігі {eff}%, энергия тұтыну {energy} кВт·сағ. {total} станоктың {active}-і оңтайлы жұмыс істеп тұр.",'optimization':"Белгіленген станоктарға 48 сағат ішінде алдын алу техникалық қызметін жоспарлаңыз, желі бойынша жүктеме теңгерімін қайта реттеңіз және өнімділікке нұқсан келтірмей энергия шығынын азайту үшін шыңдық сағаттардағы жүктемені 10-15%-ға төмендетуді қарастырыңыз.",'pred_critical':"{days} күннен кейін қозғалтқыштың бұзылу ықтималдығы жоғары",'pred_warning':"Тозудың артқаны байқалды, {days} күн ішінде ақау болуы мүмкін",'pred_ok':"Жақын арада ақау күтілмейді",'rec_critical':"Станокты дереу тоқтатыңыз, жүктемені азайтыңыз және подшипниктер мен салқындату жүйесін тексеріңіз.",'rec_warning':"24 сағат ішінде тексеру жоспарлаңыз және жүктемені 10-15%-ға азайтыңыз.",'rec_ok':"Жоспарлы бақылауды жалғастырыңыз; әрекет қажет емес."},
'de': {'risk_issue':"{critical} Maschine(n) im KRITISCHEN Zustand und {warning} im WARNZUSTAND. Last- und Temperaturtrends deuten auf erhöhte mechanische Belastung der {type}-Einheiten hin.",'risk_ok':"Keine kritischen Anomalien bei {count} {type}-Maschinen festgestellt. Das Gesamtrisiko ist niedrig.",'efficiency':"Die aktuelle Anlageneffizienz beträgt {eff}% bei einem Energieverbrauch von {energy} kWh. {active} von {total} Maschinen laufen optimal.",'optimization':"Planen Sie innerhalb von 48 Stunden eine vorbeugende Wartung der markierten Maschinen, kalibrieren Sie die Lastverteilung neu und erwägen Sie eine Reduzierung der Spitzenlast um 10-15%, um Energiekosten zu senken.",'pred_critical':"Motorausfall wahrscheinlich in {days} Tag(en)",'pred_warning':"Erhöhter Verschleiß festgestellt, möglicher Ausfall innerhalb von {days} Tagen",'pred_ok':"Kein unmittelbarer Ausfall erwartet",'rec_critical':"Maschine sofort stoppen, Last reduzieren und Lager sowie Kühlsystem prüfen.",'rec_warning':"Inspektion innerhalb von 24 Stunden planen und Last um 10-15% reduzieren.",'rec_ok':"Routineüberwachung fortsetzen; keine Maßnahme erforderlich."},
'fr': {'risk_issue':"{critical} machine(s) en état CRITIQUE et {warning} en état d'ALERTE. Les tendances de charge et de température indiquent un stress mécanique accru sur les unités {type}.",'risk_ok':"Aucune anomalie critique détectée parmi {count} machines {type}. Le niveau de risque global est faible.",'efficiency':"L'efficacité actuelle de l'usine est de {eff}% avec une consommation d'énergie de {energy} kWh. {active} machines sur {total} fonctionnent de manière optimale.",'optimization':"Planifiez une maintenance préventive pour les machines signalées dans les 48 heures, recalibrez l'équilibrage de charge et envisagez de réduire la charge aux heures de pointe de 10-15%.",'pred_critical':"Panne moteur probable dans {days} jour(s)",'pred_warning':"Usure élevée détectée, panne possible dans {days} jours",'pred_ok':"Aucune panne imminente prévue",'rec_critical':"Arrêtez la machine immédiatement, réduisez la charge et inspectez les roulements et le système de refroidissement.",'rec_warning':"Planifiez une inspection dans les 24 heures et réduisez la charge de 10-15%.",'rec_ok':"Poursuivez la surveillance de routine ; aucune action requise."},
'es': {'risk_issue':"{critical} máquina(s) en estado CRÍTICO y {warning} en estado de ADVERTENCIA. Las tendencias de carga y temperatura sugieren un mayor estrés mecánico en las unidades {type}.",'risk_ok':"No se detectaron anomalías críticas en {count} máquinas {type}. El nivel de riesgo general es bajo.",'efficiency':"La eficiencia actual de la planta es del {eff}% con un consumo de energía de {energy} kWh. {active} de {total} máquinas funcionan de manera óptima.",'optimization':"Programe mantenimiento preventivo para las máquinas marcadas en 48 horas, recalibre el equilibrio de carga y considere reducir la carga en horas pico un 10-15%.",'pred_critical':"Probable fallo del motor en {days} día(s)",'pred_warning':"Desgaste elevado detectado, posible fallo en {days} días",'pred_ok':"No se espera un fallo inminente",'rec_critical':"Detenga la máquina de inmediato, reduzca la carga e inspeccione los rodamientos y el sistema de refrigeración.",'rec_warning':"Programe una inspección en 24 horas y reduzca la carga en un 10-15%.",'rec_ok':"Continúe el monitoreo rutinario; no se requiere acción."},
'zh': {'risk_issue':"{critical} 台设备处于严重状态，{warning} 台处于警告状态。负载和温度趋势表明 {type} 设备的机械压力正在增加。",'risk_ok':"在 {count} 台 {type} 设备中未检测到严重异常。总体风险水平较低。",'efficiency':"当前工厂效率为 {eff}%，能耗为 {energy} kWh。{total} 台设备中有 {active} 台运行正常。",'optimization':"请在48小时内安排对标记设备的预防性维护，重新校准生产线负载平衡，并考虑将高峰时段负载降低10-15%以降低能源成本。",'pred_critical':"{days} 天内可能发生电机故障",'pred_warning':"检测到磨损加剧，{days} 天内可能发生故障",'pred_ok':"预计不会发生紧急故障",'rec_critical':"立即停机，降低负载，检查轴承和冷却系统。",'rec_warning':"请在24小时内安排检查，并将负载降低10-15%。",'rec_ok':"继续常规监测；无需采取行动。"},
'ar': {'risk_issue':"{critical} آلة في حالة حرجة و {warning} في حالة تحذير. تشير اتجاهات الحمل ودرجة الحرارة إلى زيادة الإجهاد الميكانيكي على وحدات {type}.",'risk_ok':"لم يتم اكتشاف أي حالات شاذة حرجة بين {count} آلة من نوع {type}. مستوى الخطر العام منخفض.",'efficiency':"كفاءة المصنع الحالية هي {eff}% باستهلاك طاقة {energy} كيلوواط. {active} من أصل {total} آلة تعمل بشكل مثالي.",'optimization':"جدولة الصيانة الوقائية للآلات المحددة خلال 48 ساعة، وإعادة معايرة توازن الحمل عبر الخط، والنظر في تقليل حمل ساعات الذروة بنسبة 10-15%.",'pred_critical':"عطل المحرك محتمل خلال {days} يوم",'pred_warning':"تم اكتشاف تآكل مرتفع، احتمال حدوث عطل خلال {days} يوم",'pred_ok':"لا يُتوقع حدوث عطل وشيك",'rec_critical':"أوقف الآلة فورًا، قلل الحمل، وافحص المحامل ونظام التبريد.",'rec_warning':"جدولة فحص خلال 24 ساعة وتقليل الحمل بنسبة 10-15%.",'rec_ok':"واصل المراقبة الروتينية؛ لا حاجة لأي إجراء."},
'tr': {'risk_issue':"{critical} makine KRİTİK durumda ve {warning} makine UYARI durumunda. Yük ve sıcaklık eğilimleri {type} ünitelerinde artan mekanik strese işaret ediyor.",'risk_ok':"{count} adet {type} makine arasında kritik anormallik tespit edilmedi. Genel risk seviyesi düşük.",'efficiency':"Mevcut tesis verimliliği {eff}%, enerji kullanımı {energy} kWh. {total} makineden {active} tanesi optimum çalışıyor.",'optimization':"İşaretli makineler için 48 saat içinde önleyici bakım planlayın, hat boyunca yük dengelemesini yeniden kalibre edin ve enerji maliyetini düşürmek için yoğun saat yükünü %10-15 azaltmayı düşünün.",'pred_critical':"{days} gün içinde motor arızası olası",'pred_warning':"Artan aşınma tespit edildi, {days} gün içinde arıza olasılığı",'pred_ok':"Yakın zamanda arıza beklenmiyor",'rec_critical':"Makineyi hemen durdurun, yükü azaltın ve rulmanları ile soğutma sistemini kontrol edin.",'rec_warning':"24 saat içinde muayene planlayın ve yükü %10-15 azaltın.",'rec_ok':"Rutin izlemeye devam edin; işlem gerekmiyor."},
'it': {'risk_issue':"{critical} macchina/e in stato CRITICO e {warning} in stato di AVVISO. Le tendenze di carico e temperatura suggeriscono uno stress meccanico elevato sulle unità {type}.",'risk_ok':"Nessuna anomalia critica rilevata tra {count} macchine {type}. Il livello di rischio complessivo è basso.",'efficiency':"L'efficienza attuale dello stabilimento è del {eff}% con un consumo energetico di {energy} kWh. {active} macchine su {total} funzionano in modo ottimale.",'optimization':"Pianifica la manutenzione preventiva per le macchine segnalate entro 48 ore, ricalibra il bilanciamento del carico e considera di ridurre il carico nelle ore di punta del 10-15%.",'pred_critical':"Probabile guasto del motore entro {days} giorno/i",'pred_warning':"Rilevata usura elevata, possibile guasto entro {days} giorni",'pred_ok':"Nessun guasto imminente previsto",'rec_critical':"Ferma immediatamente la macchina, riduci il carico e ispeziona i cuscinetti e il sistema di raffreddamento.",'rec_warning':"Pianifica un'ispezione entro 24 ore e riduci il carico del 10-15%.",'rec_ok':"Continua il monitoraggio di routine; nessuna azione richiesta."},
'pt': {'risk_issue':"{critical} máquina(s) em estado CRÍTICO e {warning} em estado de ALERTA. As tendências de carga e temperatura sugerem estresse mecânico elevado nas unidades {type}.",'risk_ok':"Nenhuma anomalia crítica detectada entre {count} máquinas {type}. O nível de risco geral é baixo.",'efficiency':"A eficiência atual da fábrica é de {eff}% com consumo de energia de {energy} kWh. {active} de {total} máquinas estão funcionando de forma otimizada.",'optimization':"Agende manutenção preventiva para as máquinas sinalizadas em 48 horas, recalibre o balanceamento de carga e considere reduzir a carga no horário de pico em 10-15%.",'pred_critical':"Falha do motor provável em {days} dia(s)",'pred_warning':"Desgaste elevado detectado, possível falha em {days} dias",'pred_ok':"Nenhuma falha iminente esperada",'rec_critical':"Pare a máquina imediatamente, reduza a carga e inspecione os rolamentos e o sistema de resfriamento.",'rec_warning':"Agende uma inspeção em 24 horas e reduza a carga em 10-15%.",'rec_ok':"Continue o monitoramento de rotina; nenhuma ação necessária."},
'ja': {'risk_issue':"{critical} 台の機械が重大状態、{warning} 台が警告状態です。負荷と温度の傾向から、{type} ユニットへの機械的ストレスの増加が示唆されます。",'risk_ok':"{count} 台の {type} 機械の中に重大な異常は検出されませんでした。全体的なリスクレベルは低いです。",'efficiency':"現在のプラント効率は {eff}%、エネルギー使用量は {energy} kWhです。{total} 台中 {active} 台が最適に稼働しています。",'optimization':"フラグの立った機械について48時間以内に予防保守を計画し、ライン全体の負荷バランスを再調整し、ピーク時の負荷を10〜15%削減することを検討してください。",'pred_critical':"{days} 日以内にモーター故障の可能性が高いです",'pred_warning':"摩耗の増加を検出、{days} 日以内に故障の可能性があります",'pred_ok':"差し迫った故障は予想されません",'rec_critical':"直ちに機械を停止し、負荷を減らし、軸受と冷却システムを点検してください。",'rec_warning':"24時間以内に点検を計画し、負荷を10〜15%削減してください。",'rec_ok':"通常の監視を継続してください。対応は不要です。"},
'ko': {'risk_issue':"{critical} 대의 기계가 심각 상태, {warning} 대가 경고 상태입니다. 부하 및 온도 추세는 {type} 장비의 기계적 스트레스 증가를 시사합니다.",'risk_ok':"{count} 대의 {type} 기계 중 심각한 이상이 발견되지 않았습니다. 전체 위험 수준은 낮습니다.",'efficiency':"현재 공장 효율성은 {eff}%이며 에너지 사용량은 {energy} kWh입니다. {total} 대 중 {active} 대가 최적으로 작동 중입니다.",'optimization':"표시된 기계에 대해 48시간 이내에 예방 정비를 예약하고, 라인 전체의 부하 균형을 재조정하며, 처리량에 영향을 주지 않으면서 에너지 비용을 낮추기 위해 피크 시간대 부하를 10-15% 줄이는 것을 고려하세요.",'pred_critical':"{days} 일 내 모터 고장 가능성이 높습니다",'pred_warning':"마모 증가가 감지되었으며, {days} 일 내 고장 가능성이 있습니다",'pred_ok':"임박한 고장은 예상되지 않습니다",'rec_critical':"즉시 기계를 정지하고 부하를 줄이며 베어링과 냉각 시스템을 점검하세요.",'rec_warning':"24시간 이내에 점검을 예약하고 부하를 10-15% 줄이세요.",'rec_ok':"정기 모니터링을 계속하세요. 조치가 필요하지 않습니다."},
'hi': {'risk_issue':"{critical} मशीन(ें) गंभीर स्थिति में और {warning} चेतावनी स्थिति में हैं। लोड और तापमान की प्रवृत्तियाँ {type} इकाइयों पर बढ़े हुए यांत्रिक तनाव का संकेत देती हैं।",'risk_ok':"{count} {type} मशीनों में कोई गंभीर विसंगति नहीं पाई गई। समग्र जोखिम स्तर कम है।",'efficiency':"वर्तमान संयंत्र दक्षता {eff}% है, ऊर्जा उपयोग {energy} kWh है। {total} में से {active} मशीनें इष्टतम रूप से चल रही हैं।",'optimization':"चिह्नित मशीनों के लिए 48 घंटों के भीतर निवारक रखरखाव शेड्यूल करें, लाइन भर में लोड संतुलन को पुनः कैलिब्रेट करें, और ऊर्जा लागत कम करने के लिए पीक-ऑवर लोड को 10-15% तक कम करने पर विचार करें।",'pred_critical':"{days} दिन में मोटर विफलता की संभावना है",'pred_warning':"बढ़ी हुई घिसावट का पता चला, {days} दिनों के भीतर विफलता संभव",'pred_ok':"तत्काल विफलता की उम्मीद नहीं है",'rec_critical':"मशीन को तुरंत रोकें, लोड कम करें, और बियरिंग तथा कूलिंग सिस्टम की जांच करें।",'rec_warning':"24 घंटों के भीतर निरीक्षण शेड्यूल करें और लोड 10-15% कम करें।",'rec_ok':"नियमित निगरानी जारी रखें; किसी कार्रवाई की आवश्यकता नहीं है।"},
'uz': {'risk_issue':"{critical} ta stanok TANQIDIY holatda va {warning} ta OGOHLANTIRISH holatida. Yuklama va harorat tendentsiyalari {type} uskunalariga mexanik zo'riqish oshganini ko'rsatadi.",'risk_ok':"{count} ta {type} stanok orasida jiddiy anomaliyalar aniqlanmadi. Umumiy xavf darajasi past.",'efficiency':"Joriy zavod samaradorligi {eff}%, energiya sarfi {energy} kVt·soat. {total} tadan {active} tasi optimal ishlamoqda.",'optimization':"Belgilangan stanoklar uchun 48 soat ichida profilaktik texnik xizmatni rejalashtiring, liniya bo'ylab yuklama balansini qayta sozlang va energiya xarajatlarini kamaytirish uchun cho'qqi soatlardagi yuklamani 10-15% ga kamaytirishni ko'rib chiqing.",'pred_critical':"{days} kundan keyin dvigatel buzilishi ehtimoli yuqori",'pred_warning':"Ortgan eskirish aniqlandi, {days} kun ichida nosozlik ehtimoli bor",'pred_ok':"Yaqin orada nosozlik kutilmaydi",'rec_critical':"Stanokni darhol to'xtating, yuklamani kamaytiring va podshipniklar bilan sovutish tizimini tekshiring.",'rec_warning':"24 soat ichida tekshiruv rejalashtiring va yuklamani 10-15% ga kamaytiring.",'rec_ok':"Rejali monitoringni davom ettiring; harakat talab qilinmaydi."},
'ky': {'risk_issue':"{critical} станок КРИТИКАЛЫК абалда, {warning} ЭСКЕРТҮҮ абалында. Жүктөм жана температура тенденциялары {type} жабдыктарына механикалык стресстин көбөйгөнүн көрсөтөт.",'risk_ok':"{count} {type} станок арасында олуттуу аномалиялар табылган жок. Жалпы тобокелдик деңгээли төмөн.",'efficiency':"Учурдагы заводдун эффективдүүлүгү {eff}%, энергия сарптоо {energy} кВт·саат. {total} станоктон {active} нормалдуу иштеп жатат.",'optimization':"Белгиленген станоктор үчүн 48 саат ичинде алдын алуучу тейлөөнү пландаштырыңыз, линия боюнча жүктөм тең салмактуулугун кайра тууралаңыз жана энергия чыгымдарын азайтуу үчүн чоку сааттардагы жүктөмдү 10-15% га азайтууну карап көрүңүз.",'pred_critical':"{days} күндөн кийин мотордун бузулушу ыктымал",'pred_warning':"Тозуунун көбөйгөнү аныкталды, {days} күн ичинде бузулуу мүмкүн",'pred_ok':"Жакынкы мезгилде бузулуу күтүлбөйт",'rec_critical':"Станокту дароо токтотуңуз, жүктөмдү азайтыңыз жана подшипниктерди, муздатуу тутумун текшериңиз.",'rec_warning':"24 саат ичинде текшерүү пландаштырыңыз жана жүктөмдү 10-15% азайтыңыз.",'rec_ok':"Пландуу байкоону улантыңыз; аракет талап кылынбайт."},
'uk': {'risk_issue':"{critical} верстат(и) у КРИТИЧНОМУ стані та {warning} у стані ПОПЕРЕДЖЕННЯ. Тенденції навантаження та температури вказують на підвищене механічне навантаження на обладнання {type}.",'risk_ok':"Критичних аномалій серед {count} верстатів {type} не виявлено. Загальний рівень ризику низький.",'efficiency':"Поточна ефективність заводу становить {eff}% при споживанні енергії {energy} кВт·год. {active} з {total} верстатів працюють оптимально.",'optimization':"Заплануйте профілактичне обслуговування позначених верстатів протягом 48 годин, перекалібруйте баланс навантаження по лінії та розгляньте зниження пікового навантаження на 10-15%.",'pred_critical':"Ймовірна відмова двигуна через {days} дн.",'pred_warning':"Виявлено підвищений знос, можлива відмова протягом {days} днів",'pred_ok':"Неминучої відмови не очікується",'rec_critical':"Негайно зупиніть верстат, зменшіть навантаження та перевірте підшипники й систему охолодження.",'rec_warning':"Заплануйте огляд протягом 24 годин і зменшіть навантаження на 10-15%.",'rec_ok':"Продовжуйте планове спостереження; дії не потрібні."},
'pl': {'risk_issue':"{critical} maszyn(y) w stanie KRYTYCZNYM i {warning} w stanie OSTRZEŻENIA. Trendy obciążenia i temperatury wskazują na zwiększone naprężenia mechaniczne jednostek {type}.",'risk_ok':"Nie wykryto krytycznych anomalii wśród {count} maszyn {type}. Ogólny poziom ryzyka jest niski.",'efficiency':"Obecna wydajność zakładu wynosi {eff}% przy zużyciu energii {energy} kWh. {active} z {total} maszyn działa optymalnie.",'optimization':"Zaplanuj konserwację zapobiegawczą oznaczonych maszyn w ciągu 48 godzin, przekalibruj równoważenie obciążenia na linii i rozważ zmniejszenie obciążenia szczytowego o 10-15%.",'pred_critical':"Prawdopodobna awaria silnika w ciągu {days} dni",'pred_warning':"Wykryto zwiększone zużycie, możliwa awaria w ciągu {days} dni",'pred_ok':"Nie przewiduje się rychłej awarii",'rec_critical':"Natychmiast zatrzymaj maszynę, zmniejsz obciążenie i sprawdź łożyska oraz układ chłodzenia.",'rec_warning':"Zaplanuj przegląd w ciągu 24 godzin i zmniejsz obciążenie o 10-15%.",'rec_ok':"Kontynuuj rutynowe monitorowanie; żadne działanie nie jest wymagane."},
'nl': {'risk_issue':"{critical} machine(s) in KRITIEKE staat en {warning} in WAARSCHUWINGSstaat. Belasting- en temperatuurtrends wijzen op verhoogde mechanische stress op {type}-eenheden.",'risk_ok':"Geen kritieke afwijkingen gedetecteerd bij {count} {type}-machines. Algeheel risiconiveau is laag.",'efficiency':"De huidige fabrieksefficiëntie is {eff}% met een energieverbruik van {energy} kWh. {active} van de {total} machines draaien optimaal.",'optimization':"Plan binnen 48 uur preventief onderhoud voor gemarkeerde machines, herijk de belastingsverdeling en overweeg de piekbelasting met 10-15% te verlagen.",'pred_critical':"Motorstoring waarschijnlijk binnen {days} dag(en)",'pred_warning':"Verhoogde slijtage gedetecteerd, mogelijke storing binnen {days} dagen",'pred_ok':"Geen dreigende storing verwacht",'rec_critical':"Stop de machine onmiddellijk, verminder de belasting en inspecteer lagers en koelsysteem.",'rec_warning':"Plan een inspectie binnen 24 uur en verminder de belasting met 10-15%.",'rec_ok':"Ga door met routinebewaking; geen actie vereist."},
'sv': {'risk_issue':"{critical} maskin(er) i KRITISKT tillstånd och {warning} i VARNINGSläge. Belastnings- och temperaturtrender tyder på ökad mekanisk påfrestning på {type}-enheter.",'risk_ok':"Inga kritiska avvikelser upptäcktes bland {count} {type}-maskiner. Den totala risknivån är låg.",'efficiency':"Nuvarande anläggningseffektivitet är {eff}% med en energiförbrukning på {energy} kWh. {active} av {total} maskiner körs optimalt.",'optimization':"Schemalägg förebyggande underhåll för flaggade maskiner inom 48 timmar, kalibrera om lastbalanseringen och överväg att minska belastningen under högtrafik med 10-15%.",'pred_critical':"Motorhaveri troligt inom {days} dag(ar)",'pred_warning':"Ökat slitage upptäckt, möjligt haveri inom {days} dagar",'pred_ok':"Inget omedelbart haveri förväntas",'rec_critical':"Stoppa maskinen omedelbart, minska belastningen och inspektera lager och kylsystem.",'rec_warning':"Schemalägg en inspektion inom 24 timmar och minska belastningen med 10-15%.",'rec_ok':"Fortsätt rutinmässig övervakning; ingen åtgärd krävs."},
}

LANG_NAMES = {
'en':'English','ru':'Russian','kk':'Kazakh','de':'German','fr':'French','es':'Spanish','zh':'Chinese',
'ar':'Arabic','tr':'Turkish','it':'Italian','pt':'Portuguese','ja':'Japanese','ko':'Korean','hi':'Hindi',
'uz':'Uzbek','ky':'Kyrgyz','uk':'Ukrainian','pl':'Polish','nl':'Dutch','sv':'Swedish',
}

RESET_EMAIL_TEXT = {
'en': {"subject": "FactoryPulse AI - Your password reset code", "body": "Your verification code is: {code}\n\nThis code expires in 15 minutes. If you did not request this, you can safely ignore this email."},
'ru': {"subject": "FactoryPulse AI - Код для восстановления пароля", "body": "Ваш код подтверждения: {code}\n\nКод действителен 15 минут. Если вы не запрашивали это, просто проигнорируйте письмо."},
'kk': {"subject": "FactoryPulse AI - Құпия сөзді қалпына келтіру коды", "body": "Сіздің растау кодыңыз: {code}\n\nКод 15 минут жарамды. Егер сіз бұны сұрамаған болсаңыз, хатты елемей қалдырыңыз."},
'de': {"subject": "FactoryPulse AI - Ihr Passwort-Reset-Code", "body": "Ihr Bestätigungscode lautet: {code}\n\nDieser Code läuft in 15 Minuten ab. Falls Sie dies nicht angefordert haben, ignorieren Sie diese E-Mail einfach."},
'fr': {"subject": "FactoryPulse AI - Votre code de réinitialisation", "body": "Votre code de vérification est : {code}\n\nCe code expire dans 15 minutes. Si vous n'avez pas demandé cela, ignorez simplement cet e-mail."},
'es': {"subject": "FactoryPulse AI - Su código de restablecimiento", "body": "Su código de verificación es: {code}\n\nEste código caduca en 15 minutos. Si no solicitó esto, simplemente ignore este correo."},
'zh': {"subject": "FactoryPulse AI - 密码重置验证码", "body": "您的验证码是：{code}\n\n此验证码15分钟内有效。如果您没有请求此操作，请忽略此邮件。"},
'ar': {"subject": "FactoryPulse AI - رمز إعادة تعيين كلمة المرور", "body": "رمز التحقق الخاص بك هو: {code}\n\nتنتهي صلاحية هذا الرمز خلال 15 دقيقة. إذا لم تطلب هذا، يمكنك تجاهل هذه الرسالة بأمان."},
'tr': {"subject": "FactoryPulse AI - Şifre sıfırlama kodunuz", "body": "Doğrulama kodunuz: {code}\n\nBu kod 15 dakika içinde sona erer. Bunu siz talep etmediyseniz, bu e-postayı yok sayabilirsiniz."},
'it': {"subject": "FactoryPulse AI - Il tuo codice di reimpostazione", "body": "Il tuo codice di verifica è: {code}\n\nQuesto codice scade tra 15 minuti. Se non hai richiesto questo, ignora pure questa email."},
'pt': {"subject": "FactoryPulse AI - Seu código de redefinição", "body": "Seu código de verificação é: {code}\n\nEste código expira em 15 minutos. Se você não solicitou isso, ignore este e-mail."},
'ja': {"subject": "FactoryPulse AI - パスワードリセットコード", "body": "認証コード: {code}\n\nこのコードは15分で失効します。これをリクエストしていない場合は、このメールを無視してください。"},
'ko': {"subject": "FactoryPulse AI - 비밀번호 재설정 코드", "body": "인증 코드: {code}\n\n이 코드는 15분 후 만료됩니다. 요청하지 않으셨다면 이 이메일을 무시하셔도 됩니다."},
'hi': {"subject": "FactoryPulse AI - पासवर्ड रीसेट कोड", "body": "आपका सत्यापन कोड है: {code}\n\nयह कोड 15 मिनट में समाप्त हो जाएगा। यदि आपने इसका अनुरोध नहीं किया, तो इस ईमेल को अनदेखा करें।"},
'uz': {"subject": "FactoryPulse AI - Parolni tiklash kodi", "body": "Tasdiqlash kodingiz: {code}\n\nBu kod 15 daqiqada tugaydi. Agar buni siz so'ramagan bo'lsangiz, bu xabarni e'tiborsiz qoldiring."},
'ky': {"subject": "FactoryPulse AI - Сырсөздү калыбына келтирүү коду", "body": "Сиздин ырастоо кодуңуз: {code}\n\nБул код 15 мүнөттөн кийин жарамсыз болот. Эгер сиз муну сурабаган болсоңуз, бул катты этибарга албаңыз."},
'uk': {"subject": "FactoryPulse AI - Код відновлення пароля", "body": "Ваш код підтвердження: {code}\n\nКод дійсний 15 хвилин. Якщо ви не запитували це, просто проігноруйте лист."},
'pl': {"subject": "FactoryPulse AI - Twój kod resetowania hasła", "body": "Twój kod weryfikacyjny to: {code}\n\nTen kod wygasa za 15 minut. Jeśli o to nie prosiłeś, po prostu zignoruj tę wiadomość."},
'nl': {"subject": "FactoryPulse AI - Uw wachtwoord-resetcode", "body": "Uw verificatiecode is: {code}\n\nDeze code verloopt over 15 minuten. Als u dit niet heeft aangevraagd, kunt u deze e-mail negeren."},
'sv': {"subject": "FactoryPulse AI - Din kod för lösenordsåterställning", "body": "Din verifieringskod är: {code}\n\nDenna kod upphör om 15 minuter. Om du inte begärde detta kan du ignorera detta e-postmeddelande."},
}


def _rule_based_analysis(state, kpis, machines, lang="en"):
    tpl = RULE_TEMPLATES.get(lang, RULE_TEMPLATES["en"])
    critical = [m for m in machines if m["status"] == "critical"]
    warning = [m for m in machines if m["status"] == "warning"]
    if critical or warning:
        risks = tpl["risk_issue"].format(critical=len(critical), warning=len(warning), type=state["machine_type"])
    else:
        risks = tpl["risk_ok"].format(count=state["machine_count"], type=state["machine_type"])
    efficiency_insights = tpl["efficiency"].format(
        eff=kpis["efficiency"], energy=kpis["energy_usage"],
        active=kpis["active_machines"], total=kpis["total_machines"],
    )
    optimizations = tpl["optimization"]
    return {"risks": risks, "efficiency_insights": efficiency_insights, "optimizations": optimizations}


def analyze_factory(state, kpis, machines, lang="en"):
    if GEMINI_ENABLED:
        try:
            language_name = LANG_NAMES.get(lang, "English")
            prompt = f"""Analyze this factory data. Provide risks, efficiency insights, and optimization suggestions.
Respond entirely in {language_name}.

Factory: {state['factory_name']}
Machine type: {state['machine_type']}
Machine count: {state['machine_count']}
Energy cost: ${state['energy_cost']}/kWh
Average temperature: {state['temperature']} C
Average vibration: {state['vibration']} mm/s
Average load: {state['load']} %

Computed KPIs:
Energy usage: {kpis['energy_usage']} kWh
Efficiency: {kpis['efficiency']}%
Active machines: {kpis['active_machines']}/{kpis['total_machines']}
Alerts: {kpis['alerts']}

Respond ONLY with strict JSON, no markdown, no commentary, all string values written in {language_name}:
{{"risks": "<string>", "efficiency_insights": "<string>", "optimizations": "<string>"}}
"""
            response = model.generate_content(prompt)
            parsed = _extract_json(response.text)
            if parsed and "risks" in parsed:
                return parsed
        except Exception as e:
            print("Gemini analysis error, falling back:", e)

    return _rule_based_analysis(state, kpis, machines, lang)


# ------------------------------------------------------------------
# AI ANOMALY DETECTION (matches the /api/ai/analyze contract exactly)
# ------------------------------------------------------------------
def pressure_deviation(pressure, nominal):
    """Pressure only means something relative to what THIS machine should run at.
    A pump at 8 bar and a compressor at 60 bar are both healthy; both at +30% are not."""
    if not nominal or nominal <= 0 or not pressure:
        return 0.0, "unknown"
    dev = (pressure - nominal) / nominal * 100.0
    if dev > 15:
        kind = "overpressure"
    elif dev < -15:
        kind = "underpressure"
    else:
        kind = "normal"
    return round(dev, 1), kind


def pressure_risk_points(pressure, nominal):
    """Extra risk contributed by pressure deviation. Overpressure is weighted higher
    because it carries rupture/leak risk, which matters most in oil, gas and chemical."""
    dev, kind = pressure_deviation(pressure, nominal)
    if kind == "unknown":
        return 0.0
    magnitude = abs(dev)
    if magnitude <= 10:
        return 0.0
    weight = 1.6 if dev > 0 else 1.0
    return min(35.0, (magnitude - 10) * 0.9 * weight)


def _rule_based_machine_analysis(reading, lang="en"):
    tpl = RULE_TEMPLATES.get(lang, RULE_TEMPLATES["en"])
    temperature = reading.get("temperature", 0)
    vibration = reading.get("vibration", 0)
    load = reading.get("load", 0)
    pressure = reading.get("pressure", 0)
    nominal_pressure = reading.get("nominal_pressure", 0)

    score = (
        max(0, temperature - 60) * 1.3
        + max(0, vibration - 4) * 8
        + max(0, load - 75) * 1.1
        + pressure_risk_points(pressure, nominal_pressure)
    )
    risk = int(max(0, min(99, score)))

    _, pressure_kind = pressure_deviation(pressure, nominal_pressure)
    anomaly = (
        risk > 55 or temperature > 85 or vibration > 10
        or pressure_kind in ("overpressure", "underpressure")
    )

    if risk > 75:
        days = max(1, round((100 - risk) / 8))
        prediction = tpl["pred_critical"].format(days=days)
        recommendation = tpl["rec_critical"]
    elif risk > 40:
        days = max(3, round((100 - risk) / 2.5))
        prediction = tpl["pred_warning"].format(days=days)
        recommendation = tpl["rec_warning"]
    else:
        prediction = tpl["pred_ok"]
        recommendation = tpl["rec_ok"]

    return {"anomaly": bool(anomaly), "risk": risk, "prediction": prediction, "recommendation": recommendation}


def analyze_machine_reading(reading, lang="en"):
    """Gemini-ready anomaly detection for a single machine reading, per the standard data format."""
    if GEMINI_ENABLED:
        try:
            language_name = LANG_NAMES.get(lang, "English")
            prompt = f"""You are an industrial SCADA AI anomaly-detection system.
Respond entirely in {language_name}.

Machine reading (standard FactoryPulse format):
machineId: {reading.get('machineId')}
temperature: {reading.get('temperature')} C
vibration: {reading.get('vibration')} mm/s
load: {reading.get('load')} %
pressure: {reading.get('pressure')} bar
voltage: {reading.get('voltage')} V
current: {reading.get('current')} A
status: {reading.get('status')}

Detect anomalies and predict failure risk.

Respond ONLY with strict JSON, no markdown, no commentary, all string values written in {language_name}:
{{"anomaly": <true|false>, "risk": <int 0-100>, "prediction": "<string>", "recommendation": "<string>"}}
"""
            response = model.generate_content(prompt)
            parsed = _extract_json(response.text)
            if parsed and "risk" in parsed:
                parsed["anomaly"] = bool(parsed.get("anomaly"))
                parsed["risk"] = int(parsed["risk"])
                return parsed
        except Exception as e:
            print("Gemini machine-analysis error, falling back:", e)

    return _rule_based_machine_analysis(reading, lang)


# ------------------------------------------------------------------
# DATA MODE ENGINE: SIMULATION / USB / PLC / MODBUS / OPCUA / MQTT
# (every real-hardware mode falls back to SIMULATION automatically on any failure)
# ------------------------------------------------------------------
DATA_MODE = os.environ.get("DATA_MODE", "SIMULATION").upper()  # SIMULATION|USB|PLC|MODBUS|OPCUA|MQTT
SERIAL_PORT = os.environ.get("SERIAL_PORT", "COM3")
SERIAL_BAUDRATE = int(os.environ.get("SERIAL_BAUDRATE", "9600"))
PLC_IP = os.environ.get("PLC_IP", "192.168.0.1")

MODBUS_HOST = os.environ.get("MODBUS_HOST", "192.168.0.10")
MODBUS_PORT = int(os.environ.get("MODBUS_PORT", "502"))
MODBUS_UNIT = int(os.environ.get("MODBUS_UNIT", "1"))

OPCUA_ENDPOINT = os.environ.get("OPCUA_ENDPOINT", "opc.tcp://192.168.0.20:4840")
OPCUA_NODES = {
    "temperature": os.environ.get("OPCUA_NODE_TEMP", "ns=2;s=Temperature"),
    "vibration": os.environ.get("OPCUA_NODE_VIB", "ns=2;s=Vibration"),
    "load": os.environ.get("OPCUA_NODE_LOAD", "ns=2;s=Load"),
    "pressure": os.environ.get("OPCUA_NODE_PRESSURE", "ns=2;s=Pressure"),
    "voltage": os.environ.get("OPCUA_NODE_VOLTAGE", "ns=2;s=Voltage"),
    "current": os.environ.get("OPCUA_NODE_CURRENT", "ns=2;s=Current"),
}

MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_TOPIC_PREFIX = os.environ.get("MQTT_TOPIC_PREFIX", "factorypulse")
MQTT_STALE_SECONDS = 30

USB_AVAILABLE = False
PLC_AVAILABLE = False
MODBUS_AVAILABLE = False
OPCUA_AVAILABLE = False
MQTT_AVAILABLE = False

try:
    import serial  # pyserial
    USB_AVAILABLE = True
except ImportError:
    print("pyserial not installed - USB mode unavailable (pip install pyserial to enable).")

try:
    import snap7  # python-snap7, talks to Siemens S7 PLCs
    PLC_AVAILABLE = True
except ImportError:
    print("python-snap7 not installed - PLC mode unavailable (pip install python-snap7 to enable).")

try:
    from pymodbus.client import ModbusTcpClient
    MODBUS_AVAILABLE = True
except ImportError:
    print("pymodbus not installed - MODBUS mode unavailable (pip install pymodbus to enable).")

try:
    from opcua import Client as OPCUAClient
    OPCUA_AVAILABLE = True
except ImportError:
    print("opcua not installed - OPCUA mode unavailable (pip install opcua to enable).")

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    print("paho-mqtt not installed - MQTT mode unavailable (pip install paho-mqtt to enable).")


def read_from_usb(machine_code):
    """Read one JSON line from an Arduino/ESP32 over serial. Falls back to None on any failure."""
    if not USB_AVAILABLE:
        return None
    try:
        with serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1) as ser:
            line = ser.readline().decode("utf-8").strip()
            data = json.loads(line)
            return {
                "temperature": float(data.get("temperature", 0)),
                "vibration": float(data.get("vibration", 0)),
                "load": float(data.get("load", 0)),
                "pressure": float(data.get("pressure", 0)),
                "voltage": float(data.get("voltage", 0)),
                "current": float(data.get("current", 0)),
                "error_code": str(data.get("error_code", data.get("error", ""))).strip(),
            }
    except Exception as e:
        print(f"USB read failed for {machine_code}, falling back to simulation:", e)
        return None


def read_from_plc(machine_code):
    """Read temperature/vibration from a Siemens S7 PLC data block. Falls back to None on any failure."""
    if not PLC_AVAILABLE:
        return None
    try:
        client = snap7.client.Client()
        client.connect(PLC_IP, 0, 1)
        temp_bytes = client.db_read(1, 0, 4)
        vib_bytes = client.db_read(1, 4, 4)
        temperature = snap7.util.get_real(temp_bytes, 0)
        vibration = snap7.util.get_real(vib_bytes, 0)
        client.disconnect()
        return {
            "temperature": float(temperature),
            "vibration": float(vibration),
            "load": 0.0, "pressure": 0.0, "voltage": 0.0, "current": 0.0,
        }
    except Exception as e:
        print(f"PLC read failed for {machine_code}, falling back to simulation:", e)
        return None


def read_from_modbus(machine_code):
    """Read 6 scaled holding registers (temp, vibration, load, pressure, voltage, current)
    from a Modbus TCP device (PLC, sensor gateway, energy meter, etc.)."""
    if not MODBUS_AVAILABLE:
        return None
    try:
        client = ModbusTcpClient(MODBUS_HOST, port=MODBUS_PORT)
        if not client.connect():
            return None
        try:
            result = client.read_holding_registers(0, 6, slave=MODBUS_UNIT)
        except TypeError:
            # older pymodbus versions use "unit=" instead of "slave="
            result = client.read_holding_registers(0, 6, unit=MODBUS_UNIT)
        client.close()
        if result.isError():
            return None
        regs = result.registers
        return {
            "temperature": regs[0] / 10.0,
            "vibration": regs[1] / 10.0,
            "load": regs[2] / 10.0,
            "pressure": regs[3] / 10.0,
            "voltage": regs[4] / 10.0,
            "current": regs[5] / 10.0,
        }
    except Exception as e:
        print(f"Modbus read failed for {machine_code}, falling back to simulation:", e)
        return None


def read_from_opcua(machine_code):
    """Read temperature/vibration/load/pressure/voltage/current nodes from an OPC UA server."""
    if not OPCUA_AVAILABLE:
        return None
    try:
        client = OPCUAClient(OPCUA_ENDPOINT)
        client.connect()
        try:
            values = {}
            for key, node_id in OPCUA_NODES.items():
                node = client.get_node(node_id)
                values[key] = float(node.get_value())
            return values
        finally:
            client.disconnect()
    except Exception as e:
        print(f"OPC UA read failed for {machine_code}, falling back to simulation:", e)
        return None


_mqtt_cache = {}  # machine_code -> {"data": {...}, "ts": <epoch seconds>}
_mqtt_client = None


def _mqtt_on_message(client, userdata, msg):
    try:
        parts = msg.topic.split("/")
        machine_code = parts[1] if len(parts) > 1 else "unknown"
        payload = json.loads(msg.payload.decode("utf-8"))
        _mqtt_cache[machine_code] = {"data": payload, "ts": time.time()}
    except Exception as e:
        print("MQTT message parse error:", e)


def start_mqtt_listener():
    """Subscribes to {MQTT_TOPIC_PREFIX}/<machineCode>/telemetry and caches the latest
    reading per machine. Safe to call even if the broker is unreachable - just logs and
    every machine keeps falling back to SIMULATION until the broker comes online."""
    global _mqtt_client
    if not MQTT_AVAILABLE:
        return
    try:
        _mqtt_client = mqtt.Client()
        _mqtt_client.on_message = _mqtt_on_message
        _mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=30)
        _mqtt_client.subscribe(f"{MQTT_TOPIC_PREFIX}/+/telemetry")
        _mqtt_client.loop_start()
        print(f"MQTT listener connected to {MQTT_BROKER}:{MQTT_PORT}, "
              f"subscribed to {MQTT_TOPIC_PREFIX}/+/telemetry")
    except Exception as e:
        print("MQTT connection failed, MQTT mode will fall back to simulation:", e)


def read_from_mqtt(machine_code):
    """Looks up the latest cached MQTT payload for this machine. Returns None (falls back
    to simulation) if nothing has arrived yet or the last message is older than 30s."""
    if not MQTT_AVAILABLE:
        return None
    entry = _mqtt_cache.get(machine_code)
    if not entry or (time.time() - entry["ts"]) > MQTT_STALE_SECONDS:
        return None
    data = entry["data"]
    try:
        return {
            "temperature": float(data.get("temperature", 0)),
            "vibration": float(data.get("vibration", 0)),
            "load": float(data.get("load", 0)),
            "pressure": float(data.get("pressure", 0)),
            "voltage": float(data.get("voltage", 0)),
            "current": float(data.get("current", 0)),
        }
    except Exception as e:
        print(f"MQTT cached payload malformed for {machine_code}:", e)
        return None


if DATA_MODE == "MQTT" and MQTT_AVAILABLE:
    threading.Thread(target=start_mqtt_listener, daemon=True).start()


def read_simulated(m):
    jitter_t = random.uniform(-6, 6)
    jitter_v = random.uniform(-1.0, 1.0)
    jitter_l = random.uniform(-8, 8)
    return {
        "temperature": max(15, m.temperature + jitter_t),
        "vibration": max(0, m.vibration + jitter_v),
        "load": max(0, min(100, m.load + jitter_l)),
        "pressure": max(0, m.pressure + random.uniform(-0.1, 0.1)),
        "voltage": max(0, m.voltage + random.uniform(-3, 3)),
        "current": max(0, m.current + random.uniform(-0.4, 0.4)),
    }


DATA_MODE_READERS = {
    "USB": read_from_usb,
    "PLC": read_from_plc,
    "MODBUS": read_from_modbus,
    "OPCUA": read_from_opcua,
    "MQTT": read_from_mqtt,
}


# ------------------------------------------------------------------
# ENERGY INTELLIGENCE MODULE
# 1) Idle Power Detection      2) Predictive (friction) Energy Loss
# 3) Specific Energy Consumption (kWh/unit)   4) Optimal Load Zone
# ------------------------------------------------------------------
IDLE_LOAD_THRESHOLD = 15  # % load below which a "running" machine is considered idle


def estimate_power_kw(voltage, current):
    """Rough apparent-power estimate for a 3-phase industrial motor (pf ~ 0.9)."""
    return round((voltage * current * 0.9) / 1000, 3)


def optimal_load_for_machine(machine_code):
    """Deterministic per-machine 'sweet spot' load %, derived from a U-shaped
    energy-per-unit curve (SEC(L) = fixed_losses/L + variable_losses*L, minimized
    around 68-77% for a typical industrial motor)."""
    seed = sum(ord(c) for c in machine_code) % 10
    return 68 + seed


def compute_energy_analytics(machine_code, temperature, vibration, load, voltage, current, status, daily_output_units):
    power_kw = estimate_power_kw(voltage, current)

    # 1) Idle Power Detection: powered on, producing nothing
    is_idle = status == "running" and load < IDLE_LOAD_THRESHOLD
    idle_waste_kw = power_kw if is_idle else 0

    # 2) Predictive Energy Loss: rising vibration/temperature -> more mechanical
    # friction -> the motor draws extra active+reactive power for the same output
    friction_overhead_pct = max(0.0, (vibration - 4) * 6 + max(0.0, temperature - 60) * 0.8)
    friction_overhead_pct = min(60.0, friction_overhead_pct)
    extra_power_kw = round(power_kw * friction_overhead_pct / 100, 3)

    # 3) Specific Energy Consumption: kWh spent per unit of output (24h projection)
    daily_energy_kwh = round(power_kw * 24, 2)
    sec = round(daily_energy_kwh / daily_output_units, 3) if daily_output_units and daily_output_units > 0 else None

    # 4) Optimal Load Zone: how far the current load is from this machine's sweet spot
    optimal_load = optimal_load_for_machine(machine_code)
    at_optimal_load = abs(load - optimal_load) <= 5

    return {
        "power_kw": power_kw,
        "is_idle": is_idle,
        "idle_waste_kw": idle_waste_kw,
        "friction_overhead_pct": round(friction_overhead_pct, 1),
        "extra_power_kw": extra_power_kw,
        "daily_energy_kwh": daily_energy_kwh,
        "specific_energy_consumption": sec,
        "optimal_load_pct": optimal_load,
        "current_load_pct": round(load, 1),
        "at_optimal_load": at_optimal_load,
    }


# ==================================================================
# ENTERPRISE INTELLIGENCE ENGINE
#   1) Signal history + time-series trend detection
#   2) Remaining Useful Life (wear model refined by trend)
#   3) Root cause analysis
#   4) Business impact (downtime cost, savings, ROI)
#   5) System-wide risk + anomaly propagation across a factory section
# ==================================================================

SIGNAL_HISTORY_LEN = 40
_signal_history = {}  # machine_code -> {"temperature": [...], "vibration": [...], "load": [...]}

# Failure thresholds used across the intelligence layer
TEMP_LIMIT = 95.0
VIB_LIMIT = 14.0

# Business defaults (overridable per factory via env)
DEFAULT_DOWNTIME_COST_PER_HOUR = float(os.environ.get("DOWNTIME_COST_PER_HOUR", "1200"))
DEFAULT_REPAIR_HOURS = float(os.environ.get("AVG_REPAIR_HOURS", "6"))
DEFAULT_ENERGY_PRICE = float(os.environ.get("ENERGY_PRICE_PER_KWH", "0.12"))


def record_signal(machine_code, temperature, vibration, load):
    """Keeps a short rolling window of each machine's signals for trend analysis."""
    hist = _signal_history.setdefault(machine_code, {"temperature": [], "vibration": [], "load": []})
    for key, value in (("temperature", temperature), ("vibration", vibration), ("load", load)):
        series = hist[key]
        series.append(value)
        if len(series) > SIGNAL_HISTORY_LEN:
            del series[:-SIGNAL_HISTORY_LEN]
    return hist


def linear_trend(values):
    """Least-squares slope per sample plus r^2 (how clean/real the trend is)."""
    n = len(values)
    if n < 5:
        return 0.0, 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((xs[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0, 0.0
    slope = num / den
    ss_tot = sum((v - mean_y) ** 2 for v in values)
    if ss_tot == 0:
        return slope, 1.0
    predicted = [mean_y + slope * (x - mean_x) for x in xs]
    ss_res = sum((values[i] - predicted[i]) ** 2 for i in range(n))
    return slope, max(0.0, 1 - ss_res / ss_tot)


def compute_rul(temperature, vibration, load, hist):
    """Remaining Useful Life in hours.

    Base estimate comes from how far the machine is outside its healthy operating
    envelope (a wear model), then it is tightened when the live signals show a
    genuinely clean rising trend (high r^2) rather than random jitter."""
    t_stress = max(0.0, temperature - 65) / 30.0
    v_stress = max(0.0, vibration - 4) / 10.0
    l_stress = max(0.0, load - 75) / 25.0
    severity = min(1.0, 0.45 * t_stress + 0.45 * v_stress + 0.10 * l_stress)

    if severity <= 0.02:
        return {"rul_hours": None, "confidence": 0.0, "driver": None, "severity": round(severity, 3),
                "temp_slope": 0.0, "vib_slope": 0.0}

    BASE_HOURS = 720.0  # ~30 days for a barely-stressed machine
    rul = max(0.5, BASE_HOURS * (1.0 - severity) ** 3)

    t_slope, t_r2 = linear_trend(hist.get("temperature", []))
    v_slope, v_r2 = linear_trend(hist.get("vibration", []))

    driver, confidence = None, 0.55
    if t_r2 > 0.6 and t_slope > 0:
        driver, confidence = "temperature", min(0.95, 0.6 + t_r2 * 0.35)
        rul *= 0.7
    if v_r2 > 0.6 and v_slope > 0 and (driver is None or v_r2 > t_r2):
        driver, confidence = "vibration", min(0.95, 0.6 + v_r2 * 0.35)
        rul *= 0.7
    if driver is None:
        driver = "temperature" if t_stress >= v_stress else "vibration"

    return {
        "rul_hours": round(rul, 1),
        "confidence": round(confidence, 2),
        "driver": driver,
        "severity": round(severity, 3),
        "temp_slope": round(t_slope, 4),
        "vib_slope": round(v_slope, 4),
    }


def detect_root_causes(temperature, vibration, load, power_kw, rul_info,
                       pressure=0, nominal_pressure=0):
    """Explains WHY, not just THAT - ranked candidate causes with confidence."""
    t_slope = rul_info.get("temp_slope", 0.0)
    v_slope = rul_info.get("vib_slope", 0.0)
    causes = []

    if vibration > 7:
        conf = 0.85 if (vibration > 10 and v_slope > 0) else (0.7 if vibration > 10 else 0.55)
        causes.append({"code": "bearing_wear", "confidence": conf,
                       "component": "bearing", "evidence": f"vibration {vibration} mm/s"})
    if temperature > 80 and load > 75:
        causes.append({"code": "overload_thermal", "confidence": 0.8,
                       "component": "motor", "evidence": f"{temperature}°C at {load}% load"})
    if temperature > 80 and load < 50:
        causes.append({"code": "cooling_failure", "confidence": 0.75,
                       "component": "cooling_system", "evidence": f"{temperature}°C at only {load}% load"})
    if vibration > 6 and power_kw > 2.5:
        causes.append({"code": "misalignment", "confidence": 0.55,
                       "component": "shaft_coupling", "evidence": f"{vibration} mm/s with {power_kw} kW draw"})
    if temperature > 78 and t_slope > 0.05 and vibration < 6:
        causes.append({"code": "lubrication_loss", "confidence": 0.6,
                       "component": "lubrication", "evidence": "rising temperature without vibration rise"})

    # Pressure-driven failure modes. These dominate in oil & gas, chemical and any
    # closed-loop process, and are invisible to a temperature/vibration-only model.
    dev, kind = pressure_deviation(pressure, nominal_pressure)
    if kind == "overpressure":
        conf = 0.9 if dev > 30 else 0.72
        causes.append({"code": "overpressure", "confidence": conf,
                       "component": "relief_valve", "evidence": f"pressure {dev:+.0f}% vs nominal"})
        if temperature > 78:
            causes.append({"code": "blocked_discharge", "confidence": 0.7,
                           "component": "discharge_line", "evidence": "high pressure with rising temperature"})
    elif kind == "underpressure":
        conf = 0.85 if dev < -30 else 0.65
        causes.append({"code": "seal_leak", "confidence": conf,
                       "component": "seal", "evidence": f"pressure {dev:+.0f}% vs nominal"})
        if vibration > 6:
            causes.append({"code": "cavitation", "confidence": 0.75,
                           "component": "impeller", "evidence": "low pressure with elevated vibration"})

    if not causes:
        causes.append({"code": "normal_operation", "confidence": 0.9,
                       "component": None, "evidence": "all signals inside healthy envelope"})

    causes.sort(key=lambda c: -c["confidence"])
    return causes[:3]


def compute_business_impact(risk, rul_hours, energy, energy_price,
                            downtime_cost_per_hour=None, repair_hours=None):
    """Turns every technical finding into money - the layer executives care about."""
    downtime_cost_per_hour = downtime_cost_per_hour or DEFAULT_DOWNTIME_COST_PER_HOUR
    repair_hours = repair_hours or DEFAULT_REPAIR_HOURS

    potential_loss = (risk / 100.0) * downtime_cost_per_hour * repair_hours
    # Catching a fault early converts most of an unplanned outage into planned work.
    saved = potential_loss * 0.75 if risk > 40 else 0.0

    daily_kwh = energy.get("daily_energy_kwh", 0) or 0
    friction_pct = energy.get("friction_overhead_pct", 0) or 0
    idle_kw = energy.get("idle_waste_kw", 0) or 0

    friction_cost_day = daily_kwh * (friction_pct / 100.0) * energy_price
    idle_cost_day = idle_kw * 24 * energy_price
    wasted_energy_cost_day = friction_cost_day + idle_cost_day

    return {
        "potential_loss": round(potential_loss, 2),
        "saved": round(saved, 2),
        "wasted_energy_cost_day": round(wasted_energy_cost_day, 2),
        "wasted_energy_cost_month": round(wasted_energy_cost_day * 30, 2),
        "downtime_cost_per_hour": downtime_cost_per_hour,
        "repair_hours": repair_hours,
    }


# ------------------------------------------------------------------
# OEE ENGINE (ISO 22400)
#   OEE = Availability x Performance x Quality
#   World-class is ~85%; a typical factory sits near 60%.
# ------------------------------------------------------------------
DOWNTIME_REASONS = [
    "breakdown", "changeover", "no_material", "no_operator",
    "planned_maintenance", "quality_issue", "setup", "other",
]


def compute_oee(planned_minutes, downtime_minutes, ideal_cycle_seconds, total_units, good_units):
    """Standard three-factor OEE. Each factor is reported separately because the
    fix is different depending on which one is dragging the score down."""
    planned_minutes = max(0.0, float(planned_minutes or 0))
    downtime_minutes = max(0.0, min(float(downtime_minutes or 0), planned_minutes))
    run_time = planned_minutes - downtime_minutes

    availability = (run_time / planned_minutes) if planned_minutes > 0 else 0.0

    ideal_run_minutes = (float(ideal_cycle_seconds or 0) * int(total_units or 0)) / 60.0
    performance = (ideal_run_minutes / run_time) if run_time > 0 else 0.0
    performance = min(1.0, performance)  # a machine cannot beat its design speed

    quality = (int(good_units or 0) / int(total_units)) if total_units else 0.0
    quality = min(1.0, quality)

    oee = availability * performance * quality

    # Name the weakest factor so the user knows where to act.
    factors = {"availability": availability, "performance": performance, "quality": quality}
    weakest = min(factors, key=factors.get) if total_units or downtime_minutes else None

    if oee >= 0.85:
        grade = "world_class"
    elif oee >= 0.60:
        grade = "typical"
    elif oee >= 0.40:
        grade = "low"
    else:
        grade = "critical"

    return {
        "availability": round(availability * 100, 1),
        "performance": round(performance * 100, 1),
        "quality": round(quality * 100, 1),
        "oee": round(oee * 100, 1),
        "run_time_minutes": round(run_time, 1),
        "downtime_minutes": round(downtime_minutes, 1),
        "total_units": int(total_units or 0),
        "good_units": int(good_units or 0),
        "scrap_units": int(total_units or 0) - int(good_units or 0),
        "weakest_factor": weakest,
        "grade": grade,
    }


def aggregate_oee(shifts):
    """Rolls several shifts into one OEE figure by summing the underlying
    quantities - averaging percentages would distort the result."""
    if not shifts:
        return compute_oee(0, 0, 0, 0, 0)
    return compute_oee(
        planned_minutes=sum(s.planned_minutes or 0 for s in shifts),
        downtime_minutes=sum(s.downtime_minutes or 0 for s in shifts),
        # Weighted average cycle time across shifts
        ideal_cycle_seconds=(
            sum((s.ideal_cycle_seconds or 0) * (s.total_units or 0) for s in shifts)
            / max(1, sum(s.total_units or 0 for s in shifts))
        ),
        total_units=sum(s.total_units or 0 for s in shifts),
        good_units=sum(s.good_units or 0 for s in shifts),
    )


def compute_system_intelligence(machine_states):
    """Factory-wide view: clustering, anomaly propagation and one system risk score.

    machine_states: list of dicts with keys id, code, name, section, risk, status."""
    if not machine_states:
        return {"system_risk": 0, "healthy": 0, "at_risk": 0, "critical": 0,
                "clusters": [], "propagation": []}

    risks = [m["risk"] for m in machine_states]
    worst = max(risks)
    average = sum(risks) / len(risks)
    # A factory is only as healthy as its weakest critical asset, so the worst
    # machine is weighted heavily alongside the fleet average.
    system_risk = round(0.6 * worst + 0.4 * average, 1)

    healthy = len([m for m in machine_states if m["risk"] < 40])
    at_risk = len([m for m in machine_states if 40 <= m["risk"] <= 70])
    critical = len([m for m in machine_states if m["risk"] > 70])

    # Cluster by factory section (machines that share an environment/load path)
    clusters = {}
    for m in machine_states:
        key = m.get("section") or "unassigned"
        c = clusters.setdefault(key, {"section": key, "machines": [], "avg_risk": 0, "max_risk": 0})
        c["machines"].append(m["code"])
        c["max_risk"] = max(c["max_risk"], m["risk"])
    for c in clusters.values():
        members = [m["risk"] for m in machine_states if (m.get("section") or "unassigned") == c["section"]]
        c["avg_risk"] = round(sum(members) / len(members), 1)
        c["count"] = len(members)

    # Anomaly propagation: a critical machine raises the effective risk of peers
    # sharing its section (shared power, thermal load, mechanical line).
    propagation = []
    for source in machine_states:
        if source["risk"] <= 70:
            continue
        for target in machine_states:
            if target["code"] == source["code"]:
                continue
            if (target.get("section") or "unassigned") != (source.get("section") or "unassigned"):
                continue
            added = round((source["risk"] - 70) * 0.4, 1)
            if added > 0:
                propagation.append({
                    "from": source["code"], "to": target["code"],
                    "added_risk": added,
                    "effective_risk": min(99, round(target["risk"] + added, 1)),
                })

    return {
        "system_risk": system_risk,
        "healthy": healthy,
        "at_risk": at_risk,
        "critical": critical,
        "total": len(machine_states),
        "clusters": sorted(clusters.values(), key=lambda c: -c["max_risk"]),
        "propagation": propagation,
    }


def simulate_what_if(machine_code, temperature, vibration, load, voltage, current,
                     temp_delta_pct=0, vib_delta_pct=0, load_delta_pct=0,
                     pressure=0, nominal_pressure=0, pressure_delta_pct=0):
    """Digital Twin simulation: 'what happens if temperature rises 10%?'
    Runs the same intelligence stack against hypothetical values."""
    sim_temp = round(temperature * (1 + temp_delta_pct / 100.0), 1)
    sim_vib = round(vibration * (1 + vib_delta_pct / 100.0), 2)
    sim_load = round(min(100, max(0, load * (1 + load_delta_pct / 100.0))), 1)
    sim_pressure = round(max(0.0, pressure * (1 + pressure_delta_pct / 100.0)), 2)

    p_dev, p_kind = pressure_deviation(sim_pressure, nominal_pressure)

    status = "running"
    if sim_temp > 90 or sim_vib > 12 or abs(p_dev) > 35:
        status = "stopped"
    elif sim_temp > 78 or sim_vib > 7 or p_kind in ("overpressure", "underpressure"):
        status = "maintenance"

    reading = standard_reading(machine_code, sim_temp, sim_vib, sim_load, sim_pressure,
                               voltage, current, status)
    reading["nominal_pressure"] = nominal_pressure
    analysis = _rule_based_machine_analysis(reading, "en")

    hist = _signal_history.get(machine_code, {"temperature": [], "vibration": [], "load": []})
    rul_info = compute_rul(sim_temp, sim_vib, sim_load, hist)
    power_kw = estimate_power_kw(voltage, current)
    causes = detect_root_causes(sim_temp, sim_vib, sim_load, power_kw, rul_info,
                                pressure=pressure, nominal_pressure=nominal_pressure)

    return {
        "input": {"temperature": sim_temp, "vibration": sim_vib, "load": sim_load,
                  "pressure": sim_pressure},
        "pressure_deviation_pct": p_dev,
        "pressure_status": p_kind,
        "deltas": {"temperature_pct": temp_delta_pct, "vibration_pct": vib_delta_pct,
                   "load_pct": load_delta_pct, "pressure_pct": pressure_delta_pct},
        "status": status,
        "failure_probability": analysis["risk"],
        "stress_level": round(rul_info["severity"] * 100, 1),
        "rul_hours": rul_info["rul_hours"],
        "confidence": rul_info["confidence"],
        "root_causes": causes,
    }


def get_live_reading(m):
    """Resolves a live sensor reading for a Machine row using the active DATA_MODE, with
    automatic, transparent fallback to SIMULATION if real hardware/protocol isn't reachable.
    Every reading is tagged with source="auto" (genuinely pulled from SCADA/PLC/protocol) or
    source="manual_baseline" (no live feed available - drifting around the manually entered values)."""
    reader = DATA_MODE_READERS.get(DATA_MODE)
    live = reader(m.machine_code) if reader else None

    if live is None:
        live = read_simulated(m)
        mode_used = "SIMULATION"
        source = "manual_baseline"
    else:
        mode_used = DATA_MODE
        source = "auto"

    status = "running"
    if live["temperature"] > 90 or live["vibration"] > 12:
        status = "stopped"
    elif live["temperature"] > 78 or live["vibration"] > 7:
        status = "maintenance"

    reading = standard_reading(
        m.machine_code, live["temperature"], live["vibration"], live["load"],
        live["pressure"], live["voltage"], live["current"], status,
    )
    reading["mode"] = mode_used
    reading["source"] = source
    reading["error_code"] = live.get("error_code", "")
    if source == "auto" and live.get("error_code") and live["error_code"] != m.error_code:
        m.error_code = live["error_code"]
    reading["energy"] = compute_energy_analytics(
        m.machine_code, live["temperature"], live["vibration"], live["load"],
        live["voltage"], live["current"], status, m.daily_output_units,
    )

    # --- Enterprise intelligence layer ---
    # Pressure is judged against this machine's own nominal, so it must travel with
    # the reading for the risk score, root cause and alert classifier to use it.
    reading["nominal_pressure"] = m.nominal_pressure or 0
    p_dev, p_kind = pressure_deviation(reading.get("pressure", 0), reading["nominal_pressure"])
    reading["pressure_deviation_pct"] = p_dev
    reading["pressure_status"] = p_kind

    hist = record_signal(m.machine_code, live["temperature"], live["vibration"], live["load"])
    rul_info = compute_rul(live["temperature"], live["vibration"], live["load"], hist)
    power_kw = reading["energy"]["power_kw"]
    reading["rul"] = rul_info
    reading["root_causes"] = detect_root_causes(
        live["temperature"], live["vibration"], live["load"], power_kw, rul_info,
        pressure=reading.get("pressure", 0), nominal_pressure=reading["nominal_pressure"],
    )
    quick = _rule_based_machine_analysis(reading, "en")
    reading["risk"] = quick["risk"]
    reading["business"] = compute_business_impact(
        quick["risk"], rul_info["rul_hours"], reading["energy"], DEFAULT_ENERGY_PRICE,
    )
    return reading


# ------------------------------------------------------------------
# FLASK APP
# ------------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------------
# PRODUCTION SCALING CONFIG
# ------------------------------------------------------------------
# How often live telemetry is pushed. Raising this is the single cheapest way to
# cut server load when many users are connected (2s is plenty for a dashboard).
BROADCAST_INTERVAL_SECONDS = float(os.environ.get("BROADCAST_INTERVAL_SECONDS", "2"))

# One stored history sample per N broadcast ticks. At the 2s default that is a
# sample every 60s, which is ~1440 rows/machine/day - fine for trend charts and
# small enough not to bloat a free-tier database.
HISTORY_EVERY_N_TICKS = int(os.environ.get("HISTORY_EVERY_N_TICKS", "30"))

# With multiple gunicorn workers, exactly ONE must run the broadcast loop.
# Set BROADCAST_LEADER=false on every worker except one (or leave default for
# single-worker deployments, where this worker is naturally the leader).
BROADCAST_LEADER = os.environ.get("BROADCAST_LEADER", "true").lower() in ("1", "true", "yes")

# Redis lets multiple workers share socket rooms and state. Without it the app
# still runs correctly on a single worker.
REDIS_URL = os.environ.get("REDIS_URL", "")

# Users with an open socket right now. The broadcast loop only does work for these.
_active_users = set()

SOCKETIO_ENABLED = False
socketio = None
try:
    from flask_socketio import SocketIO, join_room
    socketio_kwargs = {"cors_allowed_origins": "*", "async_mode": "threading"}
    if REDIS_URL:
        # message_queue makes socket rooms work across multiple workers/processes.
        socketio_kwargs["message_queue"] = REDIS_URL
        print("Socket.IO using Redis message queue - multi-worker broadcasting enabled.")
    socketio = SocketIO(app, **socketio_kwargs)
    SOCKETIO_ENABLED = True
except ImportError:
    print("flask-socketio not installed - real-time push disabled, the dashboard will "
          "automatically fall back to HTTP polling (pip install flask-socketio to enable WebSocket).")


# ------------------------------------------------------------------
# ALERTS: critical events -> DB log + instant WebSocket push (if available) + email
# Works in both WebSocket mode and HTTP-polling mode (see /api/machines/<id>/live).
# ------------------------------------------------------------------
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", SMTP_USER)
ALERT_EMAIL_TO = os.environ.get("ALERT_EMAIL_TO", "")
ALERT_COOLDOWN_SECONDS = 300  # at most one alert per machine every 5 minutes

_last_alert_at = {}  # machine_id -> epoch seconds


def smtp_config_status():
    """Reports whether outgoing email can work, and what is missing if not.
    Used at startup and by the /api/smtp/status diagnostic endpoint."""
    missing = []
    if not SMTP_HOST:
        missing.append("SMTP_HOST")
    if not SMTP_USER:
        missing.append("SMTP_USER")
    if not SMTP_PASSWORD:
        missing.append("SMTP_PASSWORD")
    return {"configured": not missing, "missing": missing,
            "host": SMTP_HOST or None, "port": SMTP_PORT, "from": SMTP_FROM or None}


def send_email(to_email, subject, body):
    """Sends one email. Returns (ok, error_detail) so callers can tell the user
    precisely why delivery failed instead of silently swallowing it."""
    status = smtp_config_status()
    if not status["configured"]:
        detail = "SMTP not configured, missing: " + ", ".join(status["missing"])
        print(f"EMAIL NOT SENT to {to_email} - {detail}")
        return False, detail

    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.utils import formataddr
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = formataddr(("FactoryPulse AI", SMTP_FROM))
        msg["To"] = to_email

        # Port 465 requires implicit TLS; 587 and others use STARTTLS.
        if int(SMTP_PORT) == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20) as server:
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_FROM, [to_email], msg.as_string())
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_FROM, [to_email], msg.as_string())

        print(f"Email sent to {to_email}: {subject}")
        return True, None

    except Exception as e:
        # Translate the common SMTP failures into something actionable.
        detail = f"{type(e).__name__}: {e}"
        hint = ""
        text = str(e).lower()
        if "authentication" in text or "username and password" in text or "5.7.8" in text:
            hint = " (check SMTP_USER / SMTP_PASSWORD - Gmail needs a 16-character App Password, not your normal password)"
        elif "timed out" in text or "connection refused" in text:
            hint = " (host or port unreachable - check SMTP_HOST / SMTP_PORT and that outbound mail is allowed)"
        elif "certificate" in text:
            hint = " (TLS problem - try SMTP_PORT 465 or 587)"
        print(f"EMAIL FAILED to {to_email} - {detail}{hint}")
        return False, detail + hint


def send_alert_email(subject, body):
    """Best-effort SMTP send. Silently skipped if SMTP_HOST/ALERT_EMAIL_TO aren't set."""
    if not (SMTP_HOST and ALERT_EMAIL_TO):
        return
    send_email(ALERT_EMAIL_TO, subject, body)  # returns (ok, detail); logged inside


# ------------------------------------------------------------------
# SMART ALERT ENGINE
#   - Priority levels instead of one flat "critical"
#   - Suppresses repeats, but always lets an ESCALATION through
#   - Every alert carries concrete suggested actions
# ------------------------------------------------------------------
ALERT_PRIORITY_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}  # defined above; kept for locality

SUGGESTED_ACTIONS = {
    "immediate_failure_risk": ["stop_machine", "inspect_bearings", "check_cooling"],
    "failure_imminent": ["schedule_shutdown_24h", "order_spare_parts", "reduce_load"],
    "degradation_accelerating": ["schedule_inspection_72h", "reduce_load"],
    "outside_normal_envelope": ["monitor_closely", "verify_sensor"],
    "pressure_anomaly": ["check_relief_valve", "inspect_seals", "verify_sensor"],
    "idle_waste": ["power_down_idle", "review_shift_schedule"],
    "informational": ["no_action"],
}

# Only these priorities are worth interrupting a human for.
ALERT_MIN_PRIORITY = os.environ.get("ALERT_MIN_PRIORITY", "medium")
_alert_state = {}  # machine_id -> {"priority": str, "at": float, "repeats": int}


def classify_alert(reading):
    """Decides how urgent a reading is, and why."""
    temperature = reading.get("temperature", 0)
    vibration = reading.get("vibration", 0)
    load = reading.get("load", 0)
    status = reading.get("status", "running")
    risk = reading.get("risk", 0)
    rul_hours = (reading.get("rul") or {}).get("rul_hours")

    pressure = reading.get("pressure", 0)
    nominal_pressure = reading.get("nominal_pressure", 0)
    p_dev, p_kind = pressure_deviation(pressure, nominal_pressure)

    if status == "stopped" or temperature > 92 or vibration > 12 or abs(p_dev) > 35:
        return "critical", "immediate_failure_risk"
    if risk > 75 or (rul_hours is not None and rul_hours < 8):
        return "critical", "failure_imminent"
    if risk > 55 or (rul_hours is not None and rul_hours < 48):
        return "high", "degradation_accelerating"
    if p_kind in ("overpressure", "underpressure"):
        return "high", "pressure_anomaly"
    if temperature > 74 or vibration > 5 or load > 88 or risk > 25:
        return "medium", "outside_normal_envelope"
    return "low", "informational"


def should_send_alert(machine_id, priority):
    """Spam control that still respects escalation.

    Returns (send: bool, reason: str). A repeat of the same severity inside the
    cooldown is suppressed, but a worsening situation always breaks through."""
    now = time.time()
    prev = _alert_state.get(machine_id)
    rank = ALERT_PRIORITY_RANK[priority]

    if prev is None:
        _alert_state[machine_id] = {"priority": priority, "at": now, "repeats": 1}
        return True, "first_occurrence"

    if rank > ALERT_PRIORITY_RANK[prev["priority"]]:
        _alert_state[machine_id] = {"priority": priority, "at": now, "repeats": 1}
        return True, "escalated"

    if now - prev["at"] < ALERT_COOLDOWN_SECONDS:
        prev["repeats"] += 1
        return False, f"suppressed_duplicate_x{prev['repeats']}"

    _alert_state[machine_id] = {"priority": priority, "at": now, "repeats": 1}
    return True, "cooldown_expired"


# Critical-alert emails, written in the language the recipient chose in the UI.
ALERT_EMAIL_TEXT = {
    "en": {"subject": "FactoryPulse AI - CRITICAL: {machine}", "body": "Machine: {machine} ({code})\nTemperature: {temp}°C\nVibration: {vib} mm/s\nStatus: {status}\n\nRecommended actions:\n{actions}\n\nOpen FactoryPulse AI to review and assign a work order."},
    "ru": {"subject": "FactoryPulse AI - КРИТИЧНО: {machine}", "body": "Станок: {machine} ({code})\nТемпература: {temp}°C\nВибрация: {vib} мм/с\nСтатус: {status}\n\nРекомендуемые действия:\n{actions}\n\nОткройте FactoryPulse AI, чтобы проверить и создать наряд."},
    "kk": {"subject": "FactoryPulse AI - СЫНИ: {machine}", "body": "Станок: {machine} ({code})\nТемпература: {temp}°C\nДіріл: {vib} мм/с\nКүйі: {status}\n\nҰсынылатын әрекеттер:\n{actions}\n\nТексеріп, тапсырыс жасау үшін FactoryPulse AI ашыңыз."},
    "de": {"subject": "FactoryPulse AI - KRITISCH: {machine}", "body": "Maschine: {machine} ({code})\nTemperatur: {temp}°C\nVibration: {vib} mm/s\nStatus: {status}\n\nEmpfohlene Maßnahmen:\n{actions}\n\nÖffnen Sie FactoryPulse AI, um zu prüfen und einen Auftrag zuzuweisen."},
    "fr": {"subject": "FactoryPulse AI - CRITIQUE : {machine}", "body": "Machine : {machine} ({code})\nTempérature : {temp}°C\nVibration : {vib} mm/s\nStatut : {status}\n\nActions recommandées :\n{actions}\n\nOuvrez FactoryPulse AI pour vérifier et assigner un ordre de travail."},
    "es": {"subject": "FactoryPulse AI - CRÍTICO: {machine}", "body": "Máquina: {machine} ({code})\nTemperatura: {temp}°C\nVibración: {vib} mm/s\nEstado: {status}\n\nAcciones recomendadas:\n{actions}\n\nAbra FactoryPulse AI para revisar y asignar una orden de trabajo."},
    "zh": {"subject": "FactoryPulse AI - 严重: {machine}", "body": "设备：{machine}（{code}）\n温度：{temp}°C\n振动：{vib} mm/s\n状态：{status}\n\n建议措施：\n{actions}\n\n请打开 FactoryPulse AI 查看并派发工单。"},
    "ar": {"subject": "FactoryPulse AI - حرج: {machine}", "body": "الآلة: {machine} ({code})\nالحرارة: {temp}°C\nالاهتزاز: {vib} mm/s\nالحالة: {status}\n\nالإجراءات الموصى بها:\n{actions}\n\nافتح FactoryPulse AI للمراجعة وإنشاء أمر عمل."},
    "tr": {"subject": "FactoryPulse AI - KRİTİK: {machine}", "body": "Makine: {machine} ({code})\nSıcaklık: {temp}°C\nTitreşim: {vib} mm/s\nDurum: {status}\n\nÖnerilen işlemler:\n{actions}\n\nİncelemek ve iş emri atamak için FactoryPulse AI'yı açın."},
    "it": {"subject": "FactoryPulse AI - CRITICO: {machine}", "body": "Macchina: {machine} ({code})\nTemperatura: {temp}°C\nVibrazione: {vib} mm/s\nStato: {status}\n\nAzioni consigliate:\n{actions}\n\nApri FactoryPulse AI per verificare e assegnare un ordine di lavoro."},
    "pt": {"subject": "FactoryPulse AI - CRÍTICO: {machine}", "body": "Máquina: {machine} ({code})\nTemperatura: {temp}°C\nVibração: {vib} mm/s\nStatus: {status}\n\nAções recomendadas:\n{actions}\n\nAbra o FactoryPulse AI para revisar e atribuir uma ordem de serviço."},
    "ja": {"subject": "FactoryPulse AI - 重大: {machine}", "body": "機械：{machine}（{code}）\n温度：{temp}°C\n振動：{vib} mm/s\nステータス：{status}\n\n推奨対応：\n{actions}\n\nFactoryPulse AI を開いて確認し、作業指示を割り当ててください。"},
    "ko": {"subject": "FactoryPulse AI - 심각: {machine}", "body": "기계: {machine} ({code})\n온도: {temp}°C\n진동: {vib} mm/s\n상태: {status}\n\n권장 조치:\n{actions}\n\nFactoryPulse AI를 열어 확인하고 작업 지시를 할당하세요."},
    "hi": {"subject": "FactoryPulse AI - गंभीर: {machine}", "body": "मशीन: {machine} ({code})\nतापमान: {temp}°C\nकंपन: {vib} mm/s\nस्थिति: {status}\n\nअनुशंसित कार्रवाई:\n{actions}\n\nसमीक्षा करने और कार्य आदेश सौंपने के लिए FactoryPulse AI खोलें।"},
    "uz": {"subject": "FactoryPulse AI - TANQIDIY: {machine}", "body": "Stanok: {machine} ({code})\nHarorat: {temp}°C\nTebranish: {vib} mm/s\nHolati: {status}\n\nTavsiya etilgan harakatlar:\n{actions}\n\nKo'rib chiqish va ish buyrug'i tayinlash uchun FactoryPulse AI'ni oching."},
    "ky": {"subject": "FactoryPulse AI - КРИТИКАЛЫК: {machine}", "body": "Станок: {machine} ({code})\nТемпература: {temp}°C\nДирилдөө: {vib} mm/s\nАбалы: {status}\n\nСунушталган аракеттер:\n{actions}\n\nТекшерүү жана иш буйругун дайындоо үчүн FactoryPulse AI ачыңыз."},
    "uk": {"subject": "FactoryPulse AI - КРИТИЧНО: {machine}", "body": "Верстат: {machine} ({code})\nТемпература: {temp}°C\nВібрація: {vib} мм/с\nСтатус: {status}\n\nРекомендовані дії:\n{actions}\n\nВідкрийте FactoryPulse AI, щоб перевірити та створити наряд."},
    "pl": {"subject": "FactoryPulse AI - KRYTYCZNE: {machine}", "body": "Maszyna: {machine} ({code})\nTemperatura: {temp}°C\nWibracje: {vib} mm/s\nStatus: {status}\n\nZalecane działania:\n{actions}\n\nOtwórz FactoryPulse AI, aby sprawdzić i przypisać zlecenie."},
    "nl": {"subject": "FactoryPulse AI - KRITIEK: {machine}", "body": "Machine: {machine} ({code})\nTemperatuur: {temp}°C\nTrilling: {vib} mm/s\nStatus: {status}\n\nAanbevolen acties:\n{actions}\n\nOpen FactoryPulse AI om te beoordelen en een werkorder toe te wijzen."},
    "sv": {"subject": "FactoryPulse AI - KRITISK: {machine}", "body": "Maskin: {machine} ({code})\nTemperatur: {temp}°C\nVibration: {vib} mm/s\nStatus: {status}\n\nRekommenderade åtgärder:\n{actions}\n\nÖppna FactoryPulse AI för att granska och tilldela en arbetsorder."},
}

def maybe_create_alert(db, m, reading):
    """Smart alerting: classifies severity, filters noise, attaches suggested actions,
    then persists + pushes over WebSocket + emails (critical only)."""
    priority, reason_code = classify_alert(reading)

    if ALERT_PRIORITY_RANK[priority] < ALERT_PRIORITY_RANK.get(ALERT_MIN_PRIORITY, 1):
        return None

    send, _why = should_send_alert(m.id, priority)
    if not send:
        return None

    temperature = reading.get("temperature", 0)
    vibration = reading.get("vibration", 0)
    status = reading.get("status", "running")
    actions = SUGGESTED_ACTIONS.get(reason_code, ["no_action"])

    message = (
        f"{m.machine_name} ({m.machine_code}): temperature {temperature}°C, "
        f"vibration {vibration} mm/s, status {status}."
    )
    alert = Alert(
        user_id=m.user_id, machine_id=m.id, machine_code=m.machine_code,
        machine_name=m.machine_name,
        severity=priority, alert_type=reason_code, message=message,
        alert_temperature=temperature, alert_vibration=vibration, alert_status=status,
        suggested_actions=",".join(actions),
    )
    db.add(alert)
    db.commit()

    if SOCKETIO_ENABLED:
        socketio.emit("critical_alert", serialize_alert(alert), room=f"user_{m.user_id}")

    # Only genuinely critical events are worth an email at 3am, and it is written
    # in whatever language the account owner set the interface to.
    if priority == "critical":
        owner = db.query(User).filter_by(id=m.user_id).first()
        lang = (getattr(owner, "preferred_lang", None) or "en")
        if lang not in ALERT_EMAIL_TEXT:
            lang = "en"
        tpl = ALERT_EMAIL_TEXT[lang]
        _t = pdf_t(lang)   # shared server-side strings, same wording as the UI
        action_lines = "\n".join(f"- {_t('action_' + a)}" for a in actions)
        send_alert_email(
            tpl["subject"].format(machine=m.machine_name),
            tpl["body"].format(
                machine=m.machine_name, code=m.machine_code,
                temp=temperature, vib=vibration,
                status=_t("status_" + status),
                actions=action_lines,
            ),
        )
    return alert


IDLE_ALERT_COOLDOWN_SECONDS = 900  # at most one idle-waste alert per machine every 15 minutes
_last_idle_alert_at = {}


def maybe_create_idle_alert(db, m, reading):
    """Idle Power Detection: if the machine is powered on and 'running' but producing
    nothing (load below IDLE_LOAD_THRESHOLD), log how much power is being wasted and
    surface it as a (non-critical) alert - e.g. lunch breaks, changeover waits, queue gaps."""
    energy = reading.get("energy") or {}
    if not energy.get("is_idle"):
        return None

    now = time.time()
    if now - _last_idle_alert_at.get(m.id, 0) < IDLE_ALERT_COOLDOWN_SECONDS:
        return None
    _last_idle_alert_at[m.id] = now

    idle_kw = energy.get("idle_waste_kw", 0)
    message = (
        f"{m.machine_name} ({m.machine_code}) is powered on but idle (load "
        f"{reading.get('load', 0)}%), wasting an estimated {idle_kw} kW."
    )
    alert = Alert(
        user_id=m.user_id, machine_id=m.id, machine_code=m.machine_code,
        machine_name=m.machine_name, severity="warning", alert_type="idle_waste", message=message,
        alert_temperature=reading.get("temperature", 0), alert_vibration=reading.get("vibration", 0),
        alert_status=reading.get("status", ""), alert_value=idle_kw,
    )
    db.add(alert)
    db.commit()

    if SOCKETIO_ENABLED:
        socketio.emit("critical_alert", serialize_alert(alert), room=f"user_{m.user_id}")
    return alert


if SOCKETIO_ENABLED:
    @socketio.on("authenticate")
    def handle_authenticate(data):
        token = (data or {}).get("token")
        user_id = decode_jwt(token) if token else None
        if user_id:
            join_room(f"user_{user_id}")
            _active_users.add(user_id)

    @socketio.on("disconnect")
    def handle_disconnect():
        # We can't reliably map a socket back to its user here, so active users are
        # re-populated on every authenticate. The set is pruned periodically below.
        pass

    def _broadcast_loop():
        """Pushes live telemetry to connected users only.

        Production notes:
        - Only ONE worker runs this loop (BROADCAST_LEADER), otherwise every gunicorn
          worker would duplicate the same DB scan and emit N copies of each reading.
        - Machines are only polled for users that currently have a socket open, so an
          idle account costs nothing.
        - DB writes are batched into a single commit per tick instead of per machine.
        """
        tick = 0
        while True:
            time.sleep(BROADCAST_INTERVAL_SECONDS)
            tick += 1
            if not _active_users:
                continue  # nobody is watching - do no work at all

            db = SessionLocal()
            try:
                watching = list(_active_users)
                machines = db.query(Machine).filter(Machine.user_id.in_(watching)).all()
                dirty = False

                for m in machines:
                    reading = get_live_reading(m)
                    socketio.emit("machine_reading", reading, room=f"user_{m.user_id}")
                    if maybe_create_alert(db, m, reading) or maybe_create_idle_alert(db, m, reading):
                        dirty = True

                    # Persist history on a slower cadence than the live push.
                    # Storing every tick would bloat the DB for no extra insight;
                    # one sample per HISTORY_EVERY_N_TICKS is plenty for trends.
                    if tick % HISTORY_EVERY_N_TICKS == 0:
                        db.add(TelemetryHistory(
                            user_id=m.user_id, machine_id=m.id, machine_code=m.machine_code,
                            temperature=reading.get("temperature", 0),
                            vibration=reading.get("vibration", 0),
                            load=reading.get("load", 0),
                            power_kw=(reading.get("energy") or {}).get("power_kw", 0),
                            risk=reading.get("risk", 0),
                            status=reading.get("status", ""),
                        ))
                        dirty = True

                    # Every 10 ticks, re-run AI for machines on a genuine live feed.
                    if tick % 10 == 0 and reading.get("source") == "auto":
                        try:
                            ai_result = analyze_machine_reading(reading, "en")
                            m.failure_risk = ai_result["risk"]
                            m.estimated_failure_time = ai_result["prediction"]
                            dirty = True
                            socketio.emit("machine_ai_update", {
                                "machine_id": m.id, "machine_code": m.machine_code, **ai_result,
                            }, room=f"user_{m.user_id}")
                        except Exception as e:
                            print("auto AI re-analysis error:", e)

                if dirty:
                    db.commit()   # one commit per tick, not one per machine
            except Exception as e:
                print("broadcast loop error:", e)
                try:
                    db.rollback()
                except Exception:
                    pass
            finally:
                SessionLocal.remove()

    if BROADCAST_LEADER:
        threading.Thread(target=_broadcast_loop, daemon=True).start()
        print(f"Broadcast loop started (interval {BROADCAST_INTERVAL_SECONDS}s).")
    else:
        print("Broadcast loop disabled on this worker (another worker is the leader).")


# ------------------------------------------------------------------
# KEEP-ALIVE (prevents free-tier cold starts)
#
# Free hosting tiers spin a service down after ~15 minutes without inbound
# traffic; the next visitor then waits 30-60s on a "service waking up" screen.
# Pinging our own public URL on a timer counts as inbound traffic and keeps the
# instance warm.
#
# Set KEEPALIVE_URL to your public address to enable, e.g.
#     KEEPALIVE_URL=https://factorypulse-ai-ho3j.onrender.com
#
# Trade-off worth knowing: staying awake 24/7 uses ~720 of the 750 free
# instance-hours per month, so it only fits ONE always-on free service.
# ------------------------------------------------------------------
KEEPALIVE_URL = os.environ.get("KEEPALIVE_URL", "").strip().rstrip("/")
KEEPALIVE_INTERVAL_SECONDS = int(os.environ.get("KEEPALIVE_INTERVAL_SECONDS", "600"))  # 10 min


def _keepalive_loop():
    import urllib.request
    target = KEEPALIVE_URL + "/healthz"
    # Wait before the first ping so the app finishes booting.
    time.sleep(60)
    while True:
        try:
            req = urllib.request.Request(target, method="GET")
            req.add_header("User-Agent", "FactoryPulseAI-KeepAlive")
            with urllib.request.urlopen(req, timeout=20) as resp:
                resp.read(64)
        except Exception as e:
            print("keep-alive ping failed:", e)
        time.sleep(KEEPALIVE_INTERVAL_SECONDS)


if KEEPALIVE_URL and BROADCAST_LEADER:
    threading.Thread(target=_keepalive_loop, daemon=True).start()
    print(f"Keep-alive enabled -> {KEEPALIVE_URL}/healthz every {KEEPALIVE_INTERVAL_SECONDS}s "
          f"(prevents cold starts).")
elif not KEEPALIVE_URL:
    print("Keep-alive disabled. Set KEEPALIVE_URL to your public address to avoid "
          "free-tier cold starts.")


@app.teardown_appcontext
def remove_db_session(exception=None):
    SessionLocal.remove()


# ------------------------------------------------------------------
# AUTH ROUTES (public)
# ------------------------------------------------------------------
@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(force=True, silent=True) or {}
    full_name = str(data.get("full_name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    confirm_password = str(data.get("confirm_password", ""))

    if not full_name or not email or not password or not confirm_password:
        return jsonify({"error": "missing_fields"}), 400
    if not EMAIL_RE.match(email):
        return jsonify({"error": "invalid_email"}), 400
    if not is_strong_password(password):
        return jsonify({"error": "weak_password"}), 400
    if password != confirm_password:
        return jsonify({"error": "password_mismatch"}), 400

    db = SessionLocal()
    if db.query(User).filter_by(email=email).first():
        return jsonify({"error": "email_taken"}), 409

    role = str(data.get("role", "engineer")).strip().lower()
    if role not in ROLE_CAPABILITIES:
        role = "engineer"
    lang = str(data.get("lang", "en"))
    if lang not in ALERT_EMAIL_TEXT:
        lang = "en"
    user = User(full_name=full_name, email=email, password_hash=hash_password(password),
                role=role, preferred_lang=lang)
    db.add(user)
    db.commit()

    token = create_jwt(user.id, remember=True)
    return jsonify({"success": True, "token": token, "user": serialize_user(user)})


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True, silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    remember = bool(data.get("remember", False))

    db = SessionLocal()
    user = db.query(User).filter_by(email=email).first()
    if not user or not verify_password(password, user.password_hash):
        return jsonify({"error": "invalid_credentials"}), 401

    token = create_jwt(user.id, remember=remember)
    return jsonify({"success": True, "token": token, "user": serialize_user(user)})


@app.route("/api/forgot-password", methods=["POST"])
def api_forgot_password():
    data = request.get_json(force=True, silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    lang = str(data.get("lang", "en"))

    db = SessionLocal()
    user = db.query(User).filter_by(email=email).first()

    # Always return success either way - never reveal whether an email is registered.
    response = {"success": True}
    if not user:
        return jsonify(response)

    code = f"{secrets.randbelow(1000000):06d}"
    reset = PasswordReset(
        user_id=user.id, email=user.email, code=code,
        expires_at=datetime.datetime.utcnow() + datetime.timedelta(minutes=15),
    )
    db.add(reset)
    db.commit()

    tpl = RESET_EMAIL_TEXT.get(lang, RESET_EMAIL_TEXT["en"])
    sent, error_detail = send_email(user.email, tpl["subject"], tpl["body"].format(code=code))

    if not sent:
        if smtp_config_status()["configured"]:
            # SMTP is set up but delivery failed. Never leak the code to the
            # browser here - anyone could request a reset for someone else's
            # address and read it off the screen. Fail loudly instead.
            print(f"Password reset email FAILED for {user.email}: {error_detail}")
            return jsonify({"error": "email_send_failed", "detail": error_detail}), 502
        # No SMTP configured at all: this is a local/demo install, so surface the
        # code directly to keep the flow testable.
        response["dev_code"] = code
        response["smtp_missing"] = smtp_config_status()["missing"]

    return jsonify(response)


@app.route("/api/smtp/status", methods=["GET"])
@require_auth
def api_smtp_status():
    """Tells you whether email delivery is actually set up, without exposing secrets."""
    return jsonify(smtp_config_status())


@app.route("/api/smtp/test", methods=["POST"])
@require_auth
def api_smtp_test():
    """Sends a real test email to the signed-in user so SMTP can be verified
    before relying on it during a pilot."""
    status = smtp_config_status()
    if not status["configured"]:
        return jsonify({"error": "smtp_not_configured", "missing": status["missing"]}), 400

    ok, detail = send_email(
        g.user.email,
        "FactoryPulse AI - SMTP test",
        "This is a test message from FactoryPulse AI.\n\n"
        "If you are reading it, outgoing email is configured correctly and "
        "password-reset codes and critical alerts will reach this address.",
    )
    if not ok:
        return jsonify({"error": "send_failed", "detail": detail}), 502
    return jsonify({"success": True, "sent_to": g.user.email})


@app.route("/api/verify-reset-code", methods=["POST"])
def api_verify_reset_code():
    data = request.get_json(force=True, silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    code = str(data.get("code", "")).strip()

    db = SessionLocal()
    reset = (
        db.query(PasswordReset)
        .filter_by(email=email, code=code, used=0)
        .order_by(PasswordReset.created_at.desc())
        .first()
    )
    if not reset or reset.expires_at < datetime.datetime.utcnow():
        return jsonify({"error": "invalid_or_expired_code"}), 400

    return jsonify({"success": True})


@app.route("/api/reset-password", methods=["POST"])
def api_reset_password():
    data = request.get_json(force=True, silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    code = str(data.get("code", "")).strip()
    new_password = str(data.get("password", ""))

    if not is_strong_password(new_password):
        return jsonify({"error": "weak_password"}), 400

    db = SessionLocal()
    reset = (
        db.query(PasswordReset)
        .filter_by(email=email, code=code, used=0)
        .order_by(PasswordReset.created_at.desc())
        .first()
    )
    if not reset or reset.expires_at < datetime.datetime.utcnow():
        return jsonify({"error": "invalid_or_expired_code"}), 400

    user = db.query(User).filter_by(id=reset.user_id).first()
    if not user:
        return jsonify({"error": "invalid_or_expired_code"}), 400

    user.password_hash = hash_password(new_password)
    reset.used = 1
    db.commit()

    return jsonify({"success": True})


@app.route("/api/me/language", methods=["POST"])
@require_auth
def api_set_language():
    """Remembers the interface language so background emails match what the user sees."""
    data = request.get_json(force=True, silent=True) or {}
    lang = str(data.get("lang", "en"))
    if lang not in ALERT_EMAIL_TEXT:
        return jsonify({"error": "unsupported_language"}), 400
    g.user.preferred_lang = lang
    g.db.commit()
    return jsonify({"success": True, "lang": lang})


@app.route("/api/me", methods=["GET"])
@require_auth
def api_me():
    return jsonify({"user": serialize_user(g.user)})


# ------------------------------------------------------------------
# FACTORY CRUD ROUTES (protected, per-user)
# ------------------------------------------------------------------
@app.route("/api/factories", methods=["GET"])
@require_auth
def api_get_factories():
    factories = (
        g.db.query(Factory)
        .filter_by(user_id=g.user.id)
        .order_by(Factory.created_at.desc())
        .all()
    )
    return jsonify({"factories": [serialize_factory(f) for f in factories]})


@app.route("/api/factories", methods=["POST"])
@require_auth
def api_create_factory():
    data = request.get_json(force=True, silent=True) or {}
    try:
        factory = Factory(
            user_id=g.user.id,
            factory_name=str(data.get("factory_name", "")).strip() or "Untitled Factory",
            machines=max(1, min(30, int(data.get("machine_count", data.get("machines", 6))))),
            machine_type=str(data.get("machine_type", "CNC")).strip() or "CNC",
            energy_cost=max(0.01, float(data.get("energy_cost", 0.12))),
            temperature=float(data.get("temperature", 65)),
            vibration=float(data.get("vibration", 3.5)),
            load=float(data.get("load", 60)),
        )
    except (TypeError, ValueError):
        return jsonify({"error": "invalid_input"}), 400

    g.db.add(factory)
    g.db.commit()

    # Run the Gemini / fallback AI analysis right away and persist it
    lang = str(data.get("lang", "en"))
    state = factory_to_state(factory)
    machines_sim = generate_machines(state)
    kpis = compute_kpis(machines_sim, state)
    analysis = analyze_factory(state, kpis, machines_sim, lang)
    factory.ai_insights = json.dumps(analysis)
    g.db.commit()

    return jsonify({
        "success": True,
        "factory": serialize_factory(factory),
        "kpis": kpis,
        "machines": machines_sim,
        "analysis": analysis,
    })


@app.route("/api/factories/<int:factory_id>", methods=["PUT"])
@require_auth
def api_update_factory(factory_id):
    factory = g.db.query(Factory).filter_by(id=factory_id, user_id=g.user.id).first()
    if not factory:
        return jsonify({"error": "not_found"}), 404

    data = request.get_json(force=True, silent=True) or {}
    try:
        if "factory_name" in data:
            factory.factory_name = str(data["factory_name"]).strip() or factory.factory_name
        if "machine_count" in data or "machines" in data:
            factory.machines = max(1, min(30, int(data.get("machine_count", data.get("machines")))))
        if "machine_type" in data:
            factory.machine_type = str(data["machine_type"]).strip() or factory.machine_type
        if "energy_cost" in data:
            factory.energy_cost = max(0.01, float(data["energy_cost"]))
        if "temperature" in data:
            factory.temperature = float(data["temperature"])
        if "vibration" in data:
            factory.vibration = float(data["vibration"])
        if "load" in data:
            factory.load = float(data["load"])
    except (TypeError, ValueError):
        return jsonify({"error": "invalid_input"}), 400

    lang = str(data.get("lang", "en"))
    state = factory_to_state(factory)
    machines_sim = generate_machines(state)
    kpis = compute_kpis(machines_sim, state)
    analysis = analyze_factory(state, kpis, machines_sim, lang)
    factory.ai_insights = json.dumps(analysis)
    g.db.commit()

    return jsonify({
        "success": True,
        "factory": serialize_factory(factory),
        "kpis": kpis,
        "machines": machines_sim,
        "analysis": analysis,
    })


@app.route("/api/factories/<int:factory_id>", methods=["DELETE"])
@require_auth
def api_delete_factory(factory_id):
    factory = g.db.query(Factory).filter_by(id=factory_id, user_id=g.user.id).first()
    if not factory:
        return jsonify({"error": "not_found"}), 404
    g.db.delete(factory)
    g.db.commit()
    return jsonify({"success": True})


@app.route("/api/factories/<int:factory_id>/live", methods=["GET"])
@require_auth
def api_factory_live(factory_id):
    factory = g.db.query(Factory).filter_by(id=factory_id, user_id=g.user.id).first()
    if not factory:
        return jsonify({"error": "not_found"}), 404
    state = factory_to_state(factory)
    machines_sim = generate_machines(state)
    kpis = compute_kpis(machines_sim, state)
    return jsonify({"kpis": kpis, "machines": machines_sim, "timestamp": datetime.datetime.utcnow().isoformat()})


# ------------------------------------------------------------------
# SCADA MACHINE ROUTES (full input panel, protected, per-user)
# ------------------------------------------------------------------
@app.route("/api/mode", methods=["GET"])
@require_auth
def api_mode():
    return jsonify({
        "active_mode": DATA_MODE,
        "usb_available": USB_AVAILABLE,
        "plc_available": PLC_AVAILABLE,
        "modbus_available": MODBUS_AVAILABLE,
        "opcua_available": OPCUA_AVAILABLE,
        "mqtt_available": MQTT_AVAILABLE,
        "serial_port": SERIAL_PORT,
        "plc_ip": PLC_IP,
        "modbus_host": f"{MODBUS_HOST}:{MODBUS_PORT}",
        "opcua_endpoint": OPCUA_ENDPOINT,
        "mqtt_broker": f"{MQTT_BROKER}:{MQTT_PORT}",
    })


@app.route("/api/machine", methods=["POST"])
@require_auth
def api_create_machine():
    data = request.get_json(force=True, silent=True) or {}
    try:
        machine = Machine(
            user_id=g.user.id,
            machine_code=str(data.get("machine_code", "")).strip() or f"M-{random.randint(100,999)}",
            machine_name=str(data.get("machine_name", "")).strip() or "Unnamed Machine",
            factory_section=str(data.get("factory_section", "")).strip(),
            operator_name=str(data.get("operator_name", "")).strip(),
            temperature=float(data.get("temperature", 0)),
            vibration=float(data.get("vibration", 0)),
            load=float(data.get("load", 0)),
            pressure=float(data.get("pressure", 0)),
            voltage=float(data.get("voltage", 0)),
            current=float(data.get("current", 0)),
            status=str(data.get("status", "running")),
            error_code=str(data.get("error_code", "")).strip(),
            priority_level=str(data.get("priority_level", "normal")),
            notes=str(data.get("notes", "")),
            daily_output_units=float(data.get("daily_output_units", 0) or 0),
            data_mode=DATA_MODE,
        )
    except (TypeError, ValueError):
        return jsonify({"error": "invalid_input"}), 400

    reading = standard_reading(
        machine.machine_code, machine.temperature, machine.vibration, machine.load,
        machine.pressure, machine.voltage, machine.current, machine.status,
    )
    ai_result = analyze_machine_reading(reading, str(data.get("lang", "en")))
    machine.failure_risk = ai_result["risk"]
    machine.estimated_failure_time = ai_result["prediction"]

    g.db.add(machine)
    g.db.commit()

    return jsonify({"success": True, "machine": serialize_machine(machine), "ai": ai_result})


@app.route("/api/machines", methods=["GET"])
@require_auth
def api_list_machines():
    query = g.db.query(Machine)
    if g.user.role == "admin":
        pass  # admins see every machine on the platform
    else:
        query = query.filter_by(user_id=g.user.id)
    machines = query.order_by(Machine.created_at.desc()).all()
    return jsonify({"machines": [serialize_machine(m) for m in machines]})


@app.route("/api/machines/<int:machine_id>", methods=["DELETE"])
@require_auth
def api_delete_machine(machine_id):
    q = g.db.query(Machine).filter_by(id=machine_id)
    if g.user.role != "admin":
        q = q.filter_by(user_id=g.user.id)
    machine = q.first()
    if not machine:
        return jsonify({"error": "not_found"}), 404
    g.db.delete(machine)
    g.db.commit()
    return jsonify({"success": True})


@app.route("/api/machines/<int:machine_id>/live", methods=["GET"])
@require_auth
def api_machine_live(machine_id):
    machine = g.db.query(Machine).filter_by(id=machine_id, user_id=g.user.id).first()
    if not machine:
        return jsonify({"error": "not_found"}), 404
    reading = get_live_reading(machine)
    maybe_create_alert(g.db, machine, reading)
    maybe_create_idle_alert(g.db, machine, reading)
    g.db.commit()
    return jsonify(reading)


# ------------------------------------------------------------------
# ENTERPRISE ROUTES: system intelligence, digital twin, business ROI
# ------------------------------------------------------------------
@app.route("/api/system/intelligence", methods=["GET"])
@require_auth
def api_system_intelligence():
    """Factory-wide view: clustering, anomaly propagation, one system risk score."""
    machines = g.db.query(Machine).filter_by(user_id=g.user.id).all()
    states, readings = [], []
    for m in machines:
        reading = get_live_reading(m)
        readings.append((m, reading))
        states.append({
            "id": m.id, "code": m.machine_code, "name": m.machine_name,
            "section": m.factory_section or "unassigned",
            "risk": reading.get("risk", 0), "status": reading.get("status", "running"),
        })

    intel = compute_system_intelligence(states)

    # Aggregate the money view across the whole fleet
    total_potential = sum(r.get("business", {}).get("potential_loss", 0) for _, r in readings)
    total_saved = sum(r.get("business", {}).get("saved", 0) for _, r in readings)
    total_wasted_day = sum(r.get("business", {}).get("wasted_energy_cost_day", 0) for _, r in readings)
    efficiency_gain = round(min(35.0, total_saved / max(1.0, total_potential) * 25), 1)

    intel["business"] = {
        "potential_loss": round(total_potential, 2),
        "saved": round(total_saved, 2),
        "wasted_energy_cost_day": round(total_wasted_day, 2),
        "wasted_energy_cost_month": round(total_wasted_day * 30, 2),
        "efficiency_gain_pct": efficiency_gain,
    }
    intel["machines"] = states
    return jsonify(intel)


@app.route("/api/machines/<int:machine_id>/simulate", methods=["POST"])
@require_auth
def api_simulate_machine(machine_id):
    """Digital Twin: 'what happens if temperature rises 10%?'"""
    machine = g.db.query(Machine).filter_by(id=machine_id, user_id=g.user.id).first()
    if not machine:
        return jsonify({"error": "not_found"}), 404

    data = request.get_json(force=True, silent=True) or {}
    try:
        temp_delta = float(data.get("temp_delta_pct", 0))
        vib_delta = float(data.get("vib_delta_pct", 0))
        load_delta = float(data.get("load_delta_pct", 0))
        pressure_delta = float(data.get("pressure_delta_pct", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid_input"}), 400

    baseline = get_live_reading(machine)
    result = simulate_what_if(
        machine.machine_code,
        baseline["temperature"], baseline["vibration"], baseline["load"],
        baseline["voltage"], baseline["current"],
        temp_delta, vib_delta, load_delta,
        pressure=baseline.get("pressure", 0),
        nominal_pressure=machine.nominal_pressure or 0,
        pressure_delta_pct=pressure_delta,
    )
    result["baseline"] = {
        "temperature": baseline["temperature"],
        "vibration": baseline["vibration"],
        "load": baseline["load"],
        "pressure": baseline.get("pressure", 0),
        "pressure_deviation_pct": baseline.get("pressure_deviation_pct", 0),
        "failure_probability": baseline.get("risk", 0),
        "stress_level": round(baseline.get("rul", {}).get("severity", 0) * 100, 1),
        "status": baseline["status"],
    }
    return jsonify(result)


@app.route("/api/business/roi", methods=["GET"])
@require_auth
def api_business_roi():
    """Executive view - every technical finding expressed in money."""
    machines = g.db.query(Machine).filter_by(user_id=g.user.id).all()
    rows, totals = [], {"potential_loss": 0.0, "saved": 0.0, "wasted_day": 0.0}

    for m in machines:
        reading = get_live_reading(m)
        biz = reading.get("business", {})
        rows.append({
            "machine_id": m.id, "machine_code": m.machine_code, "machine_name": m.machine_name,
            "risk": reading.get("risk", 0),
            "rul_hours": reading.get("rul", {}).get("rul_hours"),
            "potential_loss": biz.get("potential_loss", 0),
            "saved": biz.get("saved", 0),
            "wasted_energy_cost_day": biz.get("wasted_energy_cost_day", 0),
            "top_cause": (reading.get("root_causes") or [{}])[0].get("code"),
        })
        totals["potential_loss"] += biz.get("potential_loss", 0)
        totals["saved"] += biz.get("saved", 0)
        totals["wasted_day"] += biz.get("wasted_energy_cost_day", 0)

    rows.sort(key=lambda r: -r["potential_loss"])
    return jsonify({
        "machines": rows,
        "totals": {
            "potential_loss": round(totals["potential_loss"], 2),
            "saved": round(totals["saved"], 2),
            "wasted_energy_cost_day": round(totals["wasted_day"], 2),
            "wasted_energy_cost_month": round(totals["wasted_day"] * 30, 2),
            "efficiency_gain_pct": round(min(35.0, totals["saved"] / max(1.0, totals["potential_loss"]) * 25), 1),
        },
        "assumptions": {
            "downtime_cost_per_hour": DEFAULT_DOWNTIME_COST_PER_HOUR,
            "repair_hours": DEFAULT_REPAIR_HOURS,
            "energy_price_per_kwh": DEFAULT_ENERGY_PRICE,
        },
    })


# ------------------------------------------------------------------
# STORY MODE (investor / demo mode)
#   Injects a scripted degradation into a machine and returns the full
#   narrative: healthy -> early warning -> AI catches it -> money saved.
# ------------------------------------------------------------------
STORY_STAGES = [
    {"key": "healthy",        "temp": 68, "vib": 3.2,  "load": 72, "hours_in": 0},
    {"key": "early_drift",    "temp": 76, "vib": 5.4,  "load": 78, "hours_in": 6},
    {"key": "ai_detects",     "temp": 83, "vib": 7.8,  "load": 84, "hours_in": 14},
    {"key": "critical",       "temp": 91, "vib": 11.6, "load": 90, "hours_in": 22},
]


def _story_stage_snapshot(machine_code, stage, voltage, current, daily_output):
    temp, vib, load = stage["temp"], stage["vib"], stage["load"]

    status = "running"
    if temp > 90 or vib > 12:
        status = "stopped"
    elif temp > 78 or vib > 7:
        status = "maintenance"

    reading = standard_reading(machine_code, temp, vib, load, 1.2, voltage, current, status)
    analysis = _rule_based_machine_analysis(reading, "en")
    reading["risk"] = analysis["risk"]

    energy = compute_energy_analytics(machine_code, temp, vib, load, voltage, current, status, daily_output)
    # Story stages are a scripted curve, so derive RUL from the stage itself
    # rather than from live jitter history.
    rul_info = compute_rul(temp, vib, load, {"temperature": [], "vibration": []})
    reading["rul"] = rul_info
    causes = detect_root_causes(temp, vib, load, energy["power_kw"], rul_info)
    business = compute_business_impact(analysis["risk"], rul_info["rul_hours"], energy, DEFAULT_ENERGY_PRICE)
    priority, reason_code = classify_alert(reading)

    return {
        "stage": stage["key"],
        "hours_in": stage["hours_in"],
        "temperature": temp, "vibration": vib, "load": load,
        "status": status,
        "risk": analysis["risk"],
        "rul_hours": rul_info["rul_hours"],
        "confidence": rul_info["confidence"],
        "root_causes": causes,
        "priority": priority,
        "reason_code": reason_code,
        "suggested_actions": SUGGESTED_ACTIONS.get(reason_code, ["no_action"]),
        "business": business,
        "energy": {
            "power_kw": energy["power_kw"],
            "friction_overhead_pct": energy["friction_overhead_pct"],
            "extra_power_kw": energy["extra_power_kw"],
        },
    }


# ------------------------------------------------------------------
# OEE + PRODUCTION SHIFTS + HISTORY
#   The layer that makes this comparable to a real MES: a standard KPI,
#   categorised downtime, and stored history to prove change over time.
# ------------------------------------------------------------------
@app.route("/api/shifts", methods=["POST"])
@require_auth
def api_create_shift():
    """Logs a completed production shift, which is what OEE is measured over."""
    data = request.get_json(force=True, silent=True) or {}
    machine_id = data.get("machine_id")

    machine = None
    if machine_id:
        machine = g.db.query(Machine).filter_by(id=machine_id, user_id=g.user.id).first()
        if machine is None:
            return jsonify({"error": "not_found"}), 404

    try:
        total_units = max(0, int(data.get("total_units", 0)))
        good_units = max(0, min(total_units, int(data.get("good_units", 0))))
        shift = ProductionShift(
            user_id=g.user.id,
            machine_id=machine.id if machine else None,
            shift_name=str(data.get("shift_name", "Shift A")).strip() or "Shift A",
            planned_minutes=max(1.0, float(data.get("planned_minutes", 480))),
            downtime_minutes=max(0.0, float(data.get("downtime_minutes", 0))),
            downtime_reason=str(data.get("downtime_reason", "")).strip(),
            ideal_cycle_seconds=max(0.1, float(data.get("ideal_cycle_seconds", 30))),
            total_units=total_units,
            good_units=good_units,
        )
    except (TypeError, ValueError):
        return jsonify({"error": "invalid_input"}), 400

    g.db.add(shift)
    g.db.commit()

    return jsonify({
        "success": True,
        "shift_id": shift.id,
        "oee": compute_oee(shift.planned_minutes, shift.downtime_minutes,
                           shift.ideal_cycle_seconds, shift.total_units, shift.good_units),
    })


@app.route("/api/oee", methods=["GET"])
@require_auth
def api_oee():
    """Current OEE across a time window, plus a per-shift breakdown and the
    downtime reasons that are actually costing the most time."""
    days = max(1, min(365, int(request.args.get("days", 7))))
    since = datetime.datetime.utcnow() - datetime.timedelta(days=days)

    shifts = (
        g.db.query(ProductionShift)
        .filter(ProductionShift.user_id == g.user.id, ProductionShift.shift_date >= since)
        .order_by(ProductionShift.shift_date.desc())
        .all()
    )

    overall = aggregate_oee(shifts)

    breakdown = [{
        "id": s.id,
        "shift_name": s.shift_name,
        "date": s.shift_date.isoformat() if s.shift_date else None,
        "machine_id": s.machine_id,
        "downtime_reason": s.downtime_reason,
        **compute_oee(s.planned_minutes, s.downtime_minutes, s.ideal_cycle_seconds,
                      s.total_units, s.good_units),
    } for s in shifts[:60]]

    # Which downtime reasons dominate - this is where a plant actually saves time.
    reasons = {}
    for s in shifts:
        key = s.downtime_reason or "unspecified"
        reasons[key] = reasons.get(key, 0) + (s.downtime_minutes or 0)
    top_reasons = sorted(
        ({"reason": k, "minutes": round(v, 1)} for k, v in reasons.items()),
        key=lambda r: -r["minutes"],
    )[:8]

    total_downtime = sum(s.downtime_minutes or 0 for s in shifts)
    downtime_cost = round(total_downtime / 60.0 * DEFAULT_DOWNTIME_COST_PER_HOUR, 2)

    return jsonify({
        "window_days": days,
        "shift_count": len(shifts),
        "overall": overall,
        "shifts": breakdown,
        "downtime_by_reason": top_reasons,
        "downtime_cost": downtime_cost,
        "benchmark": {"world_class": 85, "typical": 60},
    })


@app.route("/api/history/<int:machine_id>", methods=["GET"])
@require_auth
def api_machine_history(machine_id):
    """Stored sensor history for one machine, so trends over days are provable."""
    machine = g.db.query(Machine).filter_by(id=machine_id, user_id=g.user.id).first()
    if not machine:
        return jsonify({"error": "not_found"}), 404

    hours = max(1, min(24 * 90, int(request.args.get("hours", 24))))
    since = datetime.datetime.utcnow() - datetime.timedelta(hours=hours)

    rows = (
        g.db.query(TelemetryHistory)
        .filter(TelemetryHistory.machine_id == machine.id,
                TelemetryHistory.recorded_at >= since)
        .order_by(TelemetryHistory.recorded_at.asc())
        .all()
    )

    # Down-sample so a long window still returns a chart-sized payload.
    MAX_POINTS = 300
    step = max(1, len(rows) // MAX_POINTS)
    sampled = rows[::step]

    points = [{
        "t": r.recorded_at.isoformat() if r.recorded_at else None,
        "temperature": r.temperature, "vibration": r.vibration,
        "load": r.load, "power_kw": r.power_kw, "risk": r.risk, "status": r.status,
    } for r in sampled]

    def trend_of(key):
        vals = [p[key] for p in points if p[key] is not None]
        slope, r2 = linear_trend(vals)
        return {"slope_per_sample": round(slope, 4), "r_squared": round(r2, 2),
                "direction": "rising" if slope > 0.001 else ("falling" if slope < -0.001 else "flat")}

    return jsonify({
        "machine": {"id": machine.id, "code": machine.machine_code, "name": machine.machine_name},
        "window_hours": hours,
        "sample_count": len(points),
        "points": points,
        "trends": {k: trend_of(k) for k in ("temperature", "vibration", "load", "risk")} if points else {},
    })


# ------------------------------------------------------------------
# WORK ORDERS (CMMS) + AUDIT LOG
# ------------------------------------------------------------------
def audit(db, user, action, entity_type="", entity_id=None, detail=""):
    """Records an auditable action. Never raises - auditing must not break a request."""
    try:
        db.add(AuditLog(
            user_id=getattr(user, "id", None),
            user_email=getattr(user, "email", ""),
            action=action, entity_type=entity_type, entity_id=entity_id,
            detail=str(detail)[:2000],
        ))
    except Exception as e:
        print("audit log failed:", e)


def serialize_work_order(w):
    return {
        "id": w.id,
        "machine_id": w.machine_id,
        "alert_id": w.alert_id,
        "title": w.title,
        "description": w.description,
        "priority": w.priority,
        "priority_rank": ALERT_PRIORITY_RANK.get(w.priority, 0),
        "status": w.status,
        "assigned_to": w.assigned_to,
        "root_cause": w.root_cause,
        "actions": [a for a in (w.actions or "").split(",") if a],
        "created_at": w.created_at.isoformat() if w.created_at else None,
        "due_at": w.due_at.isoformat() if w.due_at else None,
        "completed_at": w.completed_at.isoformat() if w.completed_at else None,
        "completion_note": w.completion_note,
    }


@app.route("/api/work-orders", methods=["GET"])
@require_auth
def api_list_work_orders():
    status = request.args.get("status")
    q = g.db.query(WorkOrder).filter_by(user_id=g.user.id)
    if status:
        q = q.filter_by(status=status)
    orders = q.order_by(WorkOrder.created_at.desc()).limit(200).all()

    counts = {}
    for s in ("open", "in_progress", "done", "cancelled"):
        counts[s] = g.db.query(WorkOrder).filter_by(user_id=g.user.id, status=s).count()

    # Overdue work is the number a maintenance manager actually chases.
    now = datetime.datetime.utcnow()
    overdue = g.db.query(WorkOrder).filter(
        WorkOrder.user_id == g.user.id,
        WorkOrder.status.in_(("open", "in_progress")),
        WorkOrder.due_at.isnot(None),
        WorkOrder.due_at < now,
    ).count()

    return jsonify({
        "work_orders": [serialize_work_order(w) for w in orders],
        "counts": counts,
        "overdue": overdue,
    })


@app.route("/api/work-orders", methods=["POST"])
@require_auth
def api_create_work_order():
    data = request.get_json(force=True, silent=True) or {}

    machine = None
    if data.get("machine_id"):
        machine = g.db.query(Machine).filter_by(id=data["machine_id"], user_id=g.user.id).first()

    # When raised from an alert, inherit its priority, cause and suggested actions
    # so the technician sees exactly what the AI saw.
    alert = None
    if data.get("alert_id"):
        alert = g.db.query(Alert).filter_by(id=data["alert_id"], user_id=g.user.id).first()

    priority = str(data.get("priority") or (alert.severity if alert else "medium")).lower()
    if priority not in ALERT_PRIORITY_RANK:
        priority = "medium"

    actions = data.get("actions")
    if actions is None and alert is not None:
        actions = [a for a in (alert.suggested_actions or "").split(",") if a]
    actions = actions or []

    due_hours = data.get("due_hours")
    due_at = None
    if due_hours is not None:
        try:
            due_at = datetime.datetime.utcnow() + datetime.timedelta(hours=float(due_hours))
        except (TypeError, ValueError):
            due_at = None

    title = str(data.get("title", "")).strip()
    if not title:
        title = f"Maintenance: {machine.machine_name}" if machine else "Maintenance task"

    order = WorkOrder(
        user_id=g.user.id,
        machine_id=machine.id if machine else None,
        alert_id=alert.id if alert else None,
        title=title[:200],
        description=str(data.get("description", "")),
        priority=priority,
        status="open",
        assigned_to=str(data.get("assigned_to", "")).strip()[:120],
        root_cause=str(data.get("root_cause") or (alert.alert_type if alert else ""))[:80],
        actions=",".join(actions)[:255],
        due_at=due_at,
    )
    g.db.add(order)
    g.db.commit()

    audit(g.db, g.user, "workorder.create", "work_order", order.id, title)
    g.db.commit()

    return jsonify({"success": True, "work_order": serialize_work_order(order)})


@app.route("/api/work-orders/<int:order_id>", methods=["PUT"])
@require_auth
def api_update_work_order(order_id):
    order = g.db.query(WorkOrder).filter_by(id=order_id, user_id=g.user.id).first()
    if not order:
        return jsonify({"error": "not_found"}), 404

    data = request.get_json(force=True, silent=True) or {}
    previous_status = order.status

    if "status" in data:
        status = str(data["status"]).lower()
        if status in ("open", "in_progress", "done", "cancelled"):
            order.status = status
            order.completed_at = datetime.datetime.utcnow() if status == "done" else None
    if "assigned_to" in data:
        order.assigned_to = str(data["assigned_to"]).strip()[:120]
    if "priority" in data and str(data["priority"]).lower() in ALERT_PRIORITY_RANK:
        order.priority = str(data["priority"]).lower()
    if "completion_note" in data:
        order.completion_note = str(data["completion_note"])
    if "description" in data:
        order.description = str(data["description"])

    g.db.commit()
    audit(g.db, g.user, "workorder.update", "work_order", order.id,
          f"{previous_status} -> {order.status}")
    g.db.commit()

    return jsonify({"success": True, "work_order": serialize_work_order(order)})


@app.route("/api/work-orders/<int:order_id>", methods=["DELETE"])
@require_auth
def api_delete_work_order(order_id):
    order = g.db.query(WorkOrder).filter_by(id=order_id, user_id=g.user.id).first()
    if not order:
        return jsonify({"error": "not_found"}), 404
    g.db.delete(order)
    g.db.commit()
    audit(g.db, g.user, "workorder.delete", "work_order", order_id, "")
    g.db.commit()
    return jsonify({"success": True})


@app.route("/api/audit-log", methods=["GET"])
@require_auth
def api_audit_log():
    """Admins see every action in the account; other roles see their own."""
    q = g.db.query(AuditLog)
    if g.user.role != "admin":
        q = q.filter_by(user_id=g.user.id)
    entries = q.order_by(AuditLog.created_at.desc()).limit(200).all()
    return jsonify({"entries": [{
        "id": e.id, "user_email": e.user_email, "action": e.action,
        "entity_type": e.entity_type, "entity_id": e.entity_id,
        "detail": e.detail,
        "created_at": e.created_at.isoformat() if e.created_at else None,
    } for e in entries]})


@app.route("/api/meta/downtime-reasons", methods=["GET"])
@require_auth
def api_downtime_reasons():
    return jsonify({"reasons": DOWNTIME_REASONS})


@app.route("/api/story/simulate", methods=["POST"])
@require_auth
def api_story_simulate():
    """Runs the investor narrative on a real machine from the user's account.

    Returns every stage of a bearing failure developing over ~22 hours, what the
    AI saw at each point, and what catching it early is worth in money."""
    data = request.get_json(force=True, silent=True) or {}
    machine_id = data.get("machine_id")

    machine = None
    if machine_id:
        machine = g.db.query(Machine).filter_by(id=machine_id, user_id=g.user.id).first()
    if machine is None:
        machine = g.db.query(Machine).filter_by(user_id=g.user.id).first()

    if machine is None:
        return jsonify({"error": "no_machines"}), 400

    voltage = machine.voltage or 220
    current = machine.current or 8
    daily_output = machine.daily_output_units or 500

    timeline = [
        _story_stage_snapshot(machine.machine_code, stage, voltage, current, daily_output)
        for stage in STORY_STAGES
    ]

    detected = next((s for s in timeline if s["priority"] in ("high", "critical")), timeline[-1])
    worst = timeline[-1]

    # What the factory avoids by acting at detection instead of at breakdown.
    avoided_downtime = worst["business"]["potential_loss"] - detected["business"]["potential_loss"] * 0.25
    hours_of_warning = worst["hours_in"] - detected["hours_in"]

    return jsonify({
        "machine": {"id": machine.id, "code": machine.machine_code, "name": machine.machine_name},
        "timeline": timeline,
        "detection": {
            "stage": detected["stage"],
            "hours_before_failure": hours_of_warning,
            "risk_at_detection": detected["risk"],
            "confidence": detected["confidence"],
            "root_cause": (detected["root_causes"] or [{}])[0].get("code"),
            "actions": detected["suggested_actions"],
        },
        "outcome": {
            "loss_if_ignored": round(worst["business"]["potential_loss"], 2),
            "loss_if_acted": round(detected["business"]["potential_loss"] * 0.25, 2),
            "money_saved": round(max(0.0, avoided_downtime), 2),
            "hours_of_warning": hours_of_warning,
            "extra_power_at_failure_kw": worst["energy"]["extra_power_kw"],
        },
    })


@app.route("/api/alerts", methods=["GET"])
@require_auth
def api_list_alerts():
    alerts = (
        g.db.query(Alert)
        .filter_by(user_id=g.user.id)
        .order_by(Alert.created_at.desc())
        .limit(100)
        .all()
    )
    unacknowledged = g.db.query(Alert).filter_by(user_id=g.user.id, acknowledged=0).count()
    return jsonify({"alerts": [serialize_alert(a) for a in alerts], "unacknowledged": unacknowledged})


@app.route("/api/alerts/<int:alert_id>/acknowledge", methods=["POST"])
@require_auth
def api_acknowledge_alert(alert_id):
    alert = g.db.query(Alert).filter_by(id=alert_id, user_id=g.user.id).first()
    if not alert:
        return jsonify({"error": "not_found"}), 404
    alert.acknowledged = 1
    g.db.commit()
    return jsonify({"success": True})


@app.route("/api/alerts/acknowledge_all", methods=["POST"])
@require_auth
def api_acknowledge_all_alerts():
    g.db.query(Alert).filter_by(user_id=g.user.id, acknowledged=0).update({"acknowledged": 1})
    g.db.commit()
    return jsonify({"success": True})


# ------------------------------------------------------------------
# PDF PILOT SUMMARY REPORT
# ------------------------------------------------------------------
PDF_TEXT = {
'en': {"title":"FactoryPulse AI","subtitle":"Pilot Program Summary Report","prepared_for":"Prepared for","reporting_period":"Reporting period: last {days} days","generated":"Generated","overview":"Overview","factories_monitored":"Factories monitored","machines_monitored":"Machines monitored","total_energy":"Total energy usage","avg_load":"Average load","avg_temp":"Average temperature","critical_incidents":"Critical alerts (incidents caught)","energy_savings":"Estimated energy savings identified","factories":"Factories","col_factory":"Factory","col_machines":"Machines","col_type":"Type","col_energy_cost":"Energy Cost","col_avg_load":"Avg Load","col_avg_temp":"Avg Temp","no_factories":"No factories recorded for this account yet.","oee_title":"Overall Equipment Effectiveness (OEE)","oee_formula":"OEE = Availability x Performance x Quality (ISO 22400).","oee_benchmark":"World-class is 85%; a typical factory sits near 60%.","grade":"Grade","availability":"Availability","performance":"Performance","quality":"Quality","run_time":"Run time","downtime":"Downtime","scrap":"Scrap","downtime_by_reason":"Downtime by reason","col_reason":"Reason","col_minutes":"Minutes","downtime_cost":"Downtime cost over the period","financial_title":"Financial Impact","potential_loss":"Potential loss if unaddressed","loss_avoided":"Loss avoided by early detection","wasted_energy_month":"Wasted energy (per month)","efficiency_gain":"Efficiency gain identified","assumptions":"Assumptions: downtime {downtime}/h, {hours}h average repair, {price}/kWh energy.","predictive_title":"Predictive Maintenance Findings","col_machine":"Machine","col_risk":"Risk","col_remaining_life":"Remaining Life","col_root_cause":"Likely Root Cause","healthy":"healthy","workorders_title":"Maintenance Work Orders","col_task":"Task","col_priority":"Priority","col_status":"Status","col_assigned":"Assigned","none":"None","alerts_title":"Critical Alerts Log","col_date":"Date","col_message":"Message","no_alerts":"No critical incidents were recorded during this period.","footer":"Generated automatically by FactoryPulse AI. Figures are derived from monitored sensor data over the selected period.","units":"units","normal":"Normal","oee_title_short":"OEE","min_short":"min","grade_world_class":"World-class","grade_typical":"Typical","grade_low":"Low","grade_critical":"Critical","reason_breakdown":"Breakdown","reason_changeover":"Changeover","reason_no_material":"No material","reason_no_operator":"No operator","reason_planned_maintenance":"Planned maintenance","reason_quality_issue":"Quality issue","reason_setup":"Setup","reason_other":"Other","reason_unspecified":"Unspecified","cause_bearing_wear":"Bearing wear","cause_overload_thermal":"Thermal overload","cause_cooling_failure":"Cooling failure","cause_misalignment":"Shaft misalignment","cause_lubrication_loss":"Lubrication loss","cause_normal_operation":"Normal operation","wo_open":"Open","wo_in_progress":"In Progress","wo_done":"Done","wo_cancelled":"Cancelled","prio_low":"Low","prio_medium":"Medium","prio_high":"High","prio_critical":"Critical","h_short":"h","d_short":"d","status_running":"Running","status_stopped":"Stopped","status_maintenance":"Maintenance","status_warning":"Warning","status_critical":"Critical","action_stop_machine":"Stop machine","action_inspect_bearings":"Inspect bearings","action_check_cooling":"Check cooling","action_schedule_shutdown_24h":"Schedule shutdown (24h)","action_order_spare_parts":"Order spare parts","action_reduce_load":"Reduce load","action_schedule_inspection_72h":"Schedule inspection (72h)","action_monitor_closely":"Monitor closely","action_verify_sensor":"Verify sensor","action_power_down_idle":"Power down idle machine","action_review_shift_schedule":"Review shift schedule","action_no_action":"No action needed"},
'ru': {"title":"FactoryPulse AI","subtitle":"Итоговый отчёт пилотной программы","prepared_for":"Подготовлено для","reporting_period":"Отчётный период: последние {days} дней","generated":"Сформировано","overview":"Обзор","factories_monitored":"Заводов под наблюдением","machines_monitored":"Станков под наблюдением","total_energy":"Общее энергопотребление","avg_load":"Средняя нагрузка","avg_temp":"Средняя температура","critical_incidents":"Критические оповещения (выявленные инциденты)","energy_savings":"Оценка выявленной экономии энергии","factories":"Заводы","col_factory":"Завод","col_machines":"Станки","col_type":"Тип","col_energy_cost":"Стоимость энергии","col_avg_load":"Ср. нагрузка","col_avg_temp":"Ср. темп.","no_factories":"Для этого аккаунта заводы ещё не добавлены.","oee_title":"Общая эффективность оборудования (OEE)","oee_formula":"OEE = Доступность x Производительность x Качество (ISO 22400).","oee_benchmark":"Мировой уровень — 85%, типичный завод — около 60%.","grade":"Оценка","availability":"Доступность","performance":"Производительность","quality":"Качество","run_time":"Время работы","downtime":"Простой","scrap":"Брак","downtime_by_reason":"Простои по причинам","col_reason":"Причина","col_minutes":"Минуты","downtime_cost":"Стоимость простоя за период","financial_title":"Финансовый эффект","potential_loss":"Потенциальные потери без реакции","loss_avoided":"Потери, предотвращённые ранним обнаружением","wasted_energy_month":"Потери энергии (в месяц)","efficiency_gain":"Выявленный прирост эффективности","assumptions":"Допущения: простой {downtime}/ч, ремонт {hours} ч, энергия {price}/кВт·ч.","predictive_title":"Результаты прогнозной диагностики","col_machine":"Станок","col_risk":"Риск","col_remaining_life":"Остаточный ресурс","col_root_cause":"Вероятная первопричина","healthy":"исправен","workorders_title":"Наряды на обслуживание","col_task":"Задача","col_priority":"Приоритет","col_status":"Статус","col_assigned":"Назначен","none":"Нет","alerts_title":"Журнал критических оповещений","col_date":"Дата","col_message":"Сообщение","no_alerts":"За этот период критических инцидентов не зафиксировано.","footer":"Сформировано автоматически системой FactoryPulse AI. Показатели рассчитаны по данным датчиков за выбранный период.","units":"ед.","normal":"Обычный","oee_title_short":"OEE","min_short":"мин","grade_world_class":"Мировой уровень","grade_typical":"Типичный","grade_low":"Низкий","grade_critical":"Критический","reason_breakdown":"Поломка","reason_changeover":"Переналадка","reason_no_material":"Нет материала","reason_no_operator":"Нет оператора","reason_planned_maintenance":"Плановое ТО","reason_quality_issue":"Проблема качества","reason_setup":"Настройка","reason_other":"Другое","reason_unspecified":"Не указано","cause_bearing_wear":"Износ подшипника","cause_overload_thermal":"Тепловая перегрузка","cause_cooling_failure":"Отказ охлаждения","cause_misalignment":"Расцентровка вала","cause_lubrication_loss":"Потеря смазки","cause_normal_operation":"Нормальная работа","wo_open":"Открыт","wo_in_progress":"В работе","wo_done":"Выполнен","wo_cancelled":"Отменён","prio_low":"Низкий","prio_medium":"Средний","prio_high":"Высокий","prio_critical":"Критический","h_short":"ч","d_short":"дн","status_running":"Работает","status_stopped":"Остановлен","status_maintenance":"Обслуживание","status_warning":"Внимание","status_critical":"Критично","action_stop_machine":"Остановить станок","action_inspect_bearings":"Проверить подшипники","action_check_cooling":"Проверить охлаждение","action_schedule_shutdown_24h":"Запланировать остановку (24ч)","action_order_spare_parts":"Заказать запчасти","action_reduce_load":"Снизить нагрузку","action_schedule_inspection_72h":"Запланировать осмотр (72ч)","action_monitor_closely":"Внимательно наблюдать","action_verify_sensor":"Проверить датчик","action_power_down_idle":"Отключить простаивающий станок","action_review_shift_schedule":"Пересмотреть график смен","action_no_action":"Действий не требуется"},
'kk': {"title":"FactoryPulse AI","subtitle":"Пилоттық бағдарламаның қорытынды есебі","prepared_for":"Дайындалды","reporting_period":"Есеп кезеңі: соңғы {days} күн","generated":"Жасалды","overview":"Шолу","factories_monitored":"Бақыланатын зауыттар","machines_monitored":"Бақыланатын станоктар","total_energy":"Жалпы энергия тұтыну","avg_load":"Орташа жүктеме","avg_temp":"Орташа температура","critical_incidents":"Сыни дабылдар (анықталған оқиғалар)","energy_savings":"Анықталған энергия үнемі (бағалау)","factories":"Зауыттар","col_factory":"Зауыт","col_machines":"Станоктар","col_type":"Түрі","col_energy_cost":"Энергия құны","col_avg_load":"Орт. жүктеме","col_avg_temp":"Орт. темп.","no_factories":"Бұл аккаунтқа зауыттар әлі қосылмаған.","oee_title":"Жабдықтың жалпы тиімділігі (OEE)","oee_formula":"OEE = Қолжетімділік x Өнімділік x Сапа (ISO 22400).","oee_benchmark":"Әлемдік деңгей — 85%, әдеттегі зауыт — 60% шамасында.","grade":"Бағасы","availability":"Қолжетімділік","performance":"Өнімділік","quality":"Сапа","run_time":"Жұмыс уақыты","downtime":"Тоқтап қалу","scrap":"Брак","downtime_by_reason":"Себебі бойынша тоқтап қалу","col_reason":"Себебі","col_minutes":"Минут","downtime_cost":"Кезеңдегі тоқтап қалу құны","financial_title":"Қаржылық әсер","potential_loss":"Әрекетсіз қалғандағы ықтимал шығын","loss_avoided":"Ерте анықтау арқылы болдырмаған шығын","wasted_energy_month":"Энергия шығыны (айына)","efficiency_gain":"Анықталған тиімділік өсімі","assumptions":"Болжамдар: тоқтап қалу {downtime}/сағ, жөндеу {hours} сағ, энергия {price}/кВт·сағ.","predictive_title":"Болжамды диагностика нәтижелері","col_machine":"Станок","col_risk":"Тәуекел","col_remaining_life":"Қалдық ресурс","col_root_cause":"Ықтимал түпкі себеп","healthy":"сау","workorders_title":"Техникалық қызмет тапсырыстары","col_task":"Тапсырма","col_priority":"Приоритет","col_status":"Күйі","col_assigned":"Жауапты","none":"Жоқ","alerts_title":"Сыни дабылдар журналы","col_date":"Күні","col_message":"Хабарлама","no_alerts":"Бұл кезеңде сыни оқиғалар тіркелмеген.","footer":"FactoryPulse AI жүйесімен автоматты жасалды. Көрсеткіштер таңдалған кезеңдегі сенсор деректері бойынша есептелген.","units":"бірлік","normal":"Қалыпты","oee_title_short":"OEE","min_short":"мин","grade_world_class":"Әлемдік деңгей","grade_typical":"Әдеттегі","grade_low":"Төмен","grade_critical":"Сыни","reason_breakdown":"Бұзылу","reason_changeover":"Қайта баптау","reason_no_material":"Материал жоқ","reason_no_operator":"Оператор жоқ","reason_planned_maintenance":"Жоспарлы ТҚ","reason_quality_issue":"Сапа мәселесі","reason_setup":"Орнату","reason_other":"Басқа","reason_unspecified":"Көрсетілмеген","cause_bearing_wear":"Подшипник тозуы","cause_overload_thermal":"Жылулық шамадан тыс жүктеме","cause_cooling_failure":"Салқындату ақауы","cause_misalignment":"Білік центрден ауытқуы","cause_lubrication_loss":"Майлау жоғалуы","cause_normal_operation":"Қалыпты жұмыс","wo_open":"Ашық","wo_in_progress":"Орындалуда","wo_done":"Орындалды","wo_cancelled":"Бас тартылды","prio_low":"Төмен","prio_medium":"Орташа","prio_high":"Жоғары","prio_critical":"Сыни","h_short":"сағ","d_short":"күн","status_running":"Жұмыс істеп тұр","status_stopped":"Тоқтатылды","status_maintenance":"Техникалық қызмет","status_warning":"Ескерту","status_critical":"Сыни","action_stop_machine":"Станокты тоқтату","action_inspect_bearings":"Подшипниктерді тексеру","action_check_cooling":"Салқындатуды тексеру","action_schedule_shutdown_24h":"Тоқтатуды жоспарлау (24сағ)","action_order_spare_parts":"Қосалқы бөлшек тапсырыс беру","action_reduce_load":"Жүктемені азайту","action_schedule_inspection_72h":"Тексеруді жоспарлау (72сағ)","action_monitor_closely":"Мұқият бақылау","action_verify_sensor":"Сенсорды тексеру","action_power_down_idle":"Бос тұрған станокты өшіру","action_review_shift_schedule":"Ауысым кестесін қайта қарау","action_no_action":"Әрекет қажет емес"},
'de': {"title":"FactoryPulse AI","subtitle":"Abschlussbericht des Pilotprogramms","prepared_for":"Erstellt für","reporting_period":"Berichtszeitraum: letzte {days} Tage","generated":"Erstellt am","overview":"Überblick","factories_monitored":"Überwachte Fabriken","machines_monitored":"Überwachte Maschinen","total_energy":"Gesamtenergieverbrauch","avg_load":"Durchschnittliche Last","avg_temp":"Durchschnittstemperatur","critical_incidents":"Kritische Warnungen (erfasste Vorfälle)","energy_savings":"Ermittelte geschätzte Energieeinsparung","factories":"Fabriken","col_factory":"Fabrik","col_machines":"Maschinen","col_type":"Typ","col_energy_cost":"Energiekosten","col_avg_load":"Ø Last","col_avg_temp":"Ø Temp.","no_factories":"Für dieses Konto wurden noch keine Fabriken erfasst.","oee_title":"Gesamtanlageneffektivität (OEE)","oee_formula":"OEE = Verfügbarkeit x Leistung x Qualität (ISO 22400).","oee_benchmark":"Weltklasse ist 85%, eine typische Fabrik liegt bei etwa 60%.","grade":"Bewertung","availability":"Verfügbarkeit","performance":"Leistung","quality":"Qualität","run_time":"Laufzeit","downtime":"Ausfallzeit","scrap":"Ausschuss","downtime_by_reason":"Ausfallzeit nach Grund","col_reason":"Grund","col_minutes":"Minuten","downtime_cost":"Ausfallkosten im Zeitraum","financial_title":"Finanzielle Auswirkung","potential_loss":"Potenzieller Verlust ohne Maßnahmen","loss_avoided":"Durch Früherkennung vermiedener Verlust","wasted_energy_month":"Energieverlust (pro Monat)","efficiency_gain":"Ermittelter Effizienzgewinn","assumptions":"Annahmen: Ausfall {downtime}/h, {hours}h Reparatur, {price}/kWh Energie.","predictive_title":"Ergebnisse der vorausschauenden Wartung","col_machine":"Maschine","col_risk":"Risiko","col_remaining_life":"Restnutzungsdauer","col_root_cause":"Wahrscheinliche Grundursache","healthy":"gesund","workorders_title":"Wartungsaufträge","col_task":"Aufgabe","col_priority":"Priorität","col_status":"Status","col_assigned":"Zugewiesen","none":"Keine","alerts_title":"Protokoll kritischer Warnungen","col_date":"Datum","col_message":"Meldung","no_alerts":"In diesem Zeitraum wurden keine kritischen Vorfälle erfasst.","footer":"Automatisch erstellt von FactoryPulse AI. Die Kennzahlen basieren auf Sensordaten des gewählten Zeitraums.","units":"Einh.","normal":"Normal","oee_title_short":"OEE","min_short":"Min","grade_world_class":"Weltklasse","grade_typical":"Typisch","grade_low":"Niedrig","grade_critical":"Kritisch","reason_breakdown":"Störung","reason_changeover":"Rüsten","reason_no_material":"Kein Material","reason_no_operator":"Kein Bediener","reason_planned_maintenance":"Geplante Wartung","reason_quality_issue":"Qualitätsproblem","reason_setup":"Einrichtung","reason_other":"Sonstiges","reason_unspecified":"Nicht angegeben","cause_bearing_wear":"Lagerverschleiß","cause_overload_thermal":"Thermische Überlastung","cause_cooling_failure":"Kühlungsausfall","cause_misalignment":"Wellenversatz","cause_lubrication_loss":"Schmierverlust","cause_normal_operation":"Normalbetrieb","wo_open":"Offen","wo_in_progress":"In Arbeit","wo_done":"Erledigt","wo_cancelled":"Storniert","prio_low":"Niedrig","prio_medium":"Mittel","prio_high":"Hoch","prio_critical":"Kritisch","h_short":"h","d_short":"T","status_running":"Läuft","status_stopped":"Gestoppt","status_maintenance":"Wartung","status_warning":"Warnung","status_critical":"Kritisch","action_stop_machine":"Maschine stoppen","action_inspect_bearings":"Lager prüfen","action_check_cooling":"Kühlung prüfen","action_schedule_shutdown_24h":"Abschaltung planen (24h)","action_order_spare_parts":"Ersatzteile bestellen","action_reduce_load":"Last reduzieren","action_schedule_inspection_72h":"Inspektion planen (72h)","action_monitor_closely":"Genau beobachten","action_verify_sensor":"Sensor prüfen","action_power_down_idle":"Leerlaufmaschine abschalten","action_review_shift_schedule":"Schichtplan überprüfen","action_no_action":"Keine Maßnahme nötig"},
'fr': {"title":"FactoryPulse AI","subtitle":"Rapport de Synthèse du Programme Pilote","prepared_for":"Préparé pour","reporting_period":"Période : {days} derniers jours","generated":"Généré le","overview":"Vue d'ensemble","factories_monitored":"Usines surveillées","machines_monitored":"Machines surveillées","total_energy":"Consommation totale d'énergie","avg_load":"Charge moyenne","avg_temp":"Température moyenne","critical_incidents":"Alertes critiques (incidents détectés)","energy_savings":"Économies d'énergie identifiées (estimation)","factories":"Usines","col_factory":"Usine","col_machines":"Machines","col_type":"Type","col_energy_cost":"Coût Énergie","col_avg_load":"Charge moy.","col_avg_temp":"Temp. moy.","no_factories":"Aucune usine enregistrée pour ce compte.","oee_title":"Taux de Rendement Synthétique (TRS)","oee_formula":"TRS = Disponibilité x Performance x Qualité (ISO 22400).","oee_benchmark":"Le niveau mondial est 85%, une usine typique avoisine 60%.","grade":"Note","availability":"Disponibilité","performance":"Performance","quality":"Qualité","run_time":"Temps de fonctionnement","downtime":"Arrêt","scrap":"Rebut","downtime_by_reason":"Arrêts par cause","col_reason":"Cause","col_minutes":"Minutes","downtime_cost":"Coût des arrêts sur la période","financial_title":"Impact Financier","potential_loss":"Perte potentielle sans intervention","loss_avoided":"Perte évitée par détection précoce","wasted_energy_month":"Énergie gaspillée (par mois)","efficiency_gain":"Gain d'efficacité identifié","assumptions":"Hypothèses : arrêt {downtime}/h, réparation {hours}h, énergie {price}/kWh.","predictive_title":"Résultats de la Maintenance Prédictive","col_machine":"Machine","col_risk":"Risque","col_remaining_life":"Durée de vie restante","col_root_cause":"Cause racine probable","healthy":"sain","workorders_title":"Ordres de Travail de Maintenance","col_task":"Tâche","col_priority":"Priorité","col_status":"Statut","col_assigned":"Assigné","none":"Aucun","alerts_title":"Journal des Alertes Critiques","col_date":"Date","col_message":"Message","no_alerts":"Aucun incident critique n'a été enregistré sur cette période.","footer":"Généré automatiquement par FactoryPulse AI. Les chiffres proviennent des données capteurs de la période sélectionnée.","units":"unités","normal":"Normal","oee_title_short":"TRS","min_short":"min","grade_world_class":"Niveau mondial","grade_typical":"Typique","grade_low":"Faible","grade_critical":"Critique","reason_breakdown":"Panne","reason_changeover":"Changement de série","reason_no_material":"Pas de matière","reason_no_operator":"Pas d'opérateur","reason_planned_maintenance":"Maintenance planifiée","reason_quality_issue":"Problème qualité","reason_setup":"Réglage","reason_other":"Autre","reason_unspecified":"Non spécifié","cause_bearing_wear":"Usure de roulement","cause_overload_thermal":"Surcharge thermique","cause_cooling_failure":"Panne de refroidissement","cause_misalignment":"Désalignement d'arbre","cause_lubrication_loss":"Perte de lubrification","cause_normal_operation":"Fonctionnement normal","wo_open":"Ouvert","wo_in_progress":"En cours","wo_done":"Terminé","wo_cancelled":"Annulé","prio_low":"Faible","prio_medium":"Moyen","prio_high":"Élevé","prio_critical":"Critique","h_short":"h","d_short":"j","status_running":"En marche","status_stopped":"Arrêtée","status_maintenance":"Maintenance","status_warning":"Avertissement","status_critical":"Critique","action_stop_machine":"Arrêter la machine","action_inspect_bearings":"Inspecter les roulements","action_check_cooling":"Vérifier le refroidissement","action_schedule_shutdown_24h":"Planifier l'arrêt (24h)","action_order_spare_parts":"Commander des pièces","action_reduce_load":"Réduire la charge","action_schedule_inspection_72h":"Planifier l'inspection (72h)","action_monitor_closely":"Surveiller de près","action_verify_sensor":"Vérifier le capteur","action_power_down_idle":"Éteindre la machine au ralenti","action_review_shift_schedule":"Revoir le planning","action_no_action":"Aucune action requise"},
'es': {"title":"FactoryPulse AI","subtitle":"Informe Resumen del Programa Piloto","prepared_for":"Preparado para","reporting_period":"Período: últimos {days} días","generated":"Generado el","overview":"Resumen","factories_monitored":"Fábricas monitorizadas","machines_monitored":"Máquinas monitorizadas","total_energy":"Consumo total de energía","avg_load":"Carga media","avg_temp":"Temperatura media","critical_incidents":"Alertas críticas (incidentes detectados)","energy_savings":"Ahorro energético identificado (estimación)","factories":"Fábricas","col_factory":"Fábrica","col_machines":"Máquinas","col_type":"Tipo","col_energy_cost":"Coste Energía","col_avg_load":"Carga med.","col_avg_temp":"Temp. med.","no_factories":"Aún no hay fábricas registradas en esta cuenta.","oee_title":"Eficiencia General de los Equipos (OEE)","oee_formula":"OEE = Disponibilidad x Rendimiento x Calidad (ISO 22400).","oee_benchmark":"El nivel mundial es 85%; una fábrica típica ronda el 60%.","grade":"Calificación","availability":"Disponibilidad","performance":"Rendimiento","quality":"Calidad","run_time":"Tiempo de funcionamiento","downtime":"Parada","scrap":"Desecho","downtime_by_reason":"Paradas por causa","col_reason":"Causa","col_minutes":"Minutos","downtime_cost":"Coste de paradas del período","financial_title":"Impacto Financiero","potential_loss":"Pérdida potencial sin actuar","loss_avoided":"Pérdida evitada por detección temprana","wasted_energy_month":"Energía desperdiciada (al mes)","efficiency_gain":"Ganancia de eficiencia identificada","assumptions":"Supuestos: parada {downtime}/h, reparación {hours}h, energía {price}/kWh.","predictive_title":"Resultados de Mantenimiento Predictivo","col_machine":"Máquina","col_risk":"Riesgo","col_remaining_life":"Vida útil restante","col_root_cause":"Causa raíz probable","healthy":"saludable","workorders_title":"Órdenes de Trabajo de Mantenimiento","col_task":"Tarea","col_priority":"Prioridad","col_status":"Estado","col_assigned":"Asignado","none":"Ninguna","alerts_title":"Registro de Alertas Críticas","col_date":"Fecha","col_message":"Mensaje","no_alerts":"No se registraron incidentes críticos en este período.","footer":"Generado automáticamente por FactoryPulse AI. Las cifras provienen de los datos de sensores del período seleccionado.","units":"uds.","normal":"Normal","oee_title_short":"OEE","min_short":"min","grade_world_class":"Nivel mundial","grade_typical":"Típico","grade_low":"Bajo","grade_critical":"Crítico","reason_breakdown":"Avería","reason_changeover":"Cambio de formato","reason_no_material":"Sin material","reason_no_operator":"Sin operario","reason_planned_maintenance":"Mantenimiento planificado","reason_quality_issue":"Problema de calidad","reason_setup":"Preparación","reason_other":"Otro","reason_unspecified":"Sin especificar","cause_bearing_wear":"Desgaste de rodamiento","cause_overload_thermal":"Sobrecarga térmica","cause_cooling_failure":"Fallo de refrigeración","cause_misalignment":"Desalineación del eje","cause_lubrication_loss":"Pérdida de lubricación","cause_normal_operation":"Operación normal","wo_open":"Abierta","wo_in_progress":"En curso","wo_done":"Completada","wo_cancelled":"Cancelada","prio_low":"Bajo","prio_medium":"Medio","prio_high":"Alto","prio_critical":"Crítico","h_short":"h","d_short":"d","status_running":"Funcionando","status_stopped":"Detenida","status_maintenance":"Mantenimiento","status_warning":"Advertencia","status_critical":"Crítico","action_stop_machine":"Detener máquina","action_inspect_bearings":"Inspeccionar rodamientos","action_check_cooling":"Revisar refrigeración","action_schedule_shutdown_24h":"Programar parada (24h)","action_order_spare_parts":"Pedir repuestos","action_reduce_load":"Reducir carga","action_schedule_inspection_72h":"Programar inspección (72h)","action_monitor_closely":"Vigilar de cerca","action_verify_sensor":"Verificar sensor","action_power_down_idle":"Apagar máquina inactiva","action_review_shift_schedule":"Revisar turnos","action_no_action":"No se requiere acción"},
'zh': {"title":"FactoryPulse AI","subtitle":"试点项目总结报告","prepared_for":"编制给","reporting_period":"报告期：最近 {days} 天","generated":"生成时间","overview":"概览","factories_monitored":"监控工厂数","machines_monitored":"监控设备数","total_energy":"总能耗","avg_load":"平均负载","avg_temp":"平均温度","critical_incidents":"严重告警（已捕获事件）","energy_savings":"识别出的节能估算","factories":"工厂","col_factory":"工厂","col_machines":"设备","col_type":"类型","col_energy_cost":"能源成本","col_avg_load":"平均负载","col_avg_temp":"平均温度","no_factories":"此账户尚未添加工厂。","oee_title":"设备综合效率 (OEE)","oee_formula":"OEE = 可用率 x 表现性 x 质量 (ISO 22400)。","oee_benchmark":"世界级为85%，典型工厂约60%。","grade":"评级","availability":"可用率","performance":"表现性","quality":"质量","run_time":"运行时间","downtime":"停机","scrap":"废品","downtime_by_reason":"停机原因分析","col_reason":"原因","col_minutes":"分钟","downtime_cost":"期间停机成本","financial_title":"财务影响","potential_loss":"不作为的潜在损失","loss_avoided":"早期发现避免的损失","wasted_energy_month":"浪费的能源（每月）","efficiency_gain":"识别出的效率提升","assumptions":"假设：停机 {downtime}/小时，维修 {hours} 小时，能源 {price}/kWh。","predictive_title":"预测性维护结果","col_machine":"设备","col_risk":"风险","col_remaining_life":"剩余寿命","col_root_cause":"可能的根本原因","healthy":"健康","workorders_title":"维护工单","col_task":"任务","col_priority":"优先级","col_status":"状态","col_assigned":"负责人","none":"无","alerts_title":"严重告警日志","col_date":"日期","col_message":"消息","no_alerts":"本期间未记录严重事件。","footer":"由 FactoryPulse AI 自动生成。数据来源于所选期间的传感器监测数据。","units":"件","normal":"正常","oee_title_short":"OEE","min_short":"分钟","grade_world_class":"世界级","grade_typical":"典型","grade_low":"低","grade_critical":"严重","reason_breakdown":"故障","reason_changeover":"换型","reason_no_material":"缺料","reason_no_operator":"缺人","reason_planned_maintenance":"计划维护","reason_quality_issue":"质量问题","reason_setup":"调试","reason_other":"其他","reason_unspecified":"未指定","cause_bearing_wear":"轴承磨损","cause_overload_thermal":"热过载","cause_cooling_failure":"冷却故障","cause_misalignment":"轴不对中","cause_lubrication_loss":"润滑损失","cause_normal_operation":"正常运行","wo_open":"待处理","wo_in_progress":"进行中","wo_done":"已完成","wo_cancelled":"已取消","prio_low":"低","prio_medium":"中","prio_high":"高","prio_critical":"严重","h_short":"小时","d_short":"天","status_running":"运行中","status_stopped":"已停止","status_maintenance":"维护中","status_warning":"警告","status_critical":"严重","action_stop_machine":"停止设备","action_inspect_bearings":"检查轴承","action_check_cooling":"检查冷却","action_schedule_shutdown_24h":"安排停机（24小时）","action_order_spare_parts":"订购备件","action_reduce_load":"降低负载","action_schedule_inspection_72h":"安排检查（72小时）","action_monitor_closely":"密切监控","action_verify_sensor":"验证传感器","action_power_down_idle":"关闭空转设备","action_review_shift_schedule":"检查班次安排","action_no_action":"无需操作"},
'ar': {"title":"FactoryPulse AI","subtitle":"تقرير ملخص البرنامج التجريبي","prepared_for":"أُعد لـ","reporting_period":"فترة التقرير: آخر {days} يوم","generated":"تاريخ الإنشاء","overview":"نظرة عامة","factories_monitored":"المصانع المراقبة","machines_monitored":"الآلات المراقبة","total_energy":"إجمالي استهلاك الطاقة","avg_load":"متوسط الحمل","avg_temp":"متوسط الحرارة","critical_incidents":"التنبيهات الحرجة (الحوادث المكتشفة)","energy_savings":"توفير الطاقة المُقدَّر","factories":"المصانع","col_factory":"المصنع","col_machines":"الآلات","col_type":"النوع","col_energy_cost":"تكلفة الطاقة","col_avg_load":"متوسط الحمل","col_avg_temp":"متوسط الحرارة","no_factories":"لم تُسجَّل أي مصانع لهذا الحساب بعد.","oee_title":"الفعالية الإجمالية للمعدات (OEE)","oee_formula":"OEE = الجاهزية x الأداء x الجودة (ISO 22400).","oee_benchmark":"المستوى العالمي 85%، والمصنع النموذجي حوالي 60%.","grade":"التقييم","availability":"الجاهزية","performance":"الأداء","quality":"الجودة","run_time":"وقت التشغيل","downtime":"التوقف","scrap":"الهدر","downtime_by_reason":"التوقف حسب السبب","col_reason":"السبب","col_minutes":"دقائق","downtime_cost":"تكلفة التوقف خلال الفترة","financial_title":"الأثر المالي","potential_loss":"الخسارة المحتملة دون تدخل","loss_avoided":"الخسارة المتجنبة بالاكتشاف المبكر","wasted_energy_month":"الطاقة المهدورة (شهريًا)","efficiency_gain":"مكسب الكفاءة المكتشف","assumptions":"الافتراضات: توقف {downtime}/ساعة، إصلاح {hours} ساعة، طاقة {price}/kWh.","predictive_title":"نتائج الصيانة التنبؤية","col_machine":"الآلة","col_risk":"المخاطر","col_remaining_life":"العمر المتبقي","col_root_cause":"السبب الجذري المحتمل","healthy":"سليمة","workorders_title":"أوامر عمل الصيانة","col_task":"المهمة","col_priority":"الأولوية","col_status":"الحالة","col_assigned":"المسؤول","none":"لا يوجد","alerts_title":"سجل التنبيهات الحرجة","col_date":"التاريخ","col_message":"الرسالة","no_alerts":"لم تُسجَّل حوادث حرجة خلال هذه الفترة.","footer":"أُنشئ تلقائيًا بواسطة FactoryPulse AI. الأرقام مستمدة من بيانات المستشعرات خلال الفترة المحددة.","units":"وحدة","normal":"عادي","oee_title_short":"OEE","min_short":"دقيقة","grade_world_class":"المستوى العالمي","grade_typical":"نموذجي","grade_low":"منخفض","grade_critical":"حرج","reason_breakdown":"عطل","reason_changeover":"تغيير الإنتاج","reason_no_material":"لا مواد","reason_no_operator":"لا مشغل","reason_planned_maintenance":"صيانة مخططة","reason_quality_issue":"مشكلة جودة","reason_setup":"إعداد","reason_other":"أخرى","reason_unspecified":"غير محدد","cause_bearing_wear":"تآكل المحمل","cause_overload_thermal":"حمل حراري زائد","cause_cooling_failure":"عطل التبريد","cause_misalignment":"انحراف العمود","cause_lubrication_loss":"فقدان التزييت","cause_normal_operation":"تشغيل طبيعي","wo_open":"مفتوح","wo_in_progress":"قيد التنفيذ","wo_done":"منجز","wo_cancelled":"ملغى","prio_low":"منخفض","prio_medium":"متوسط","prio_high":"عالي","prio_critical":"حرج","h_short":"س","d_short":"ي","status_running":"تعمل","status_stopped":"متوقفة","status_maintenance":"صيانة","status_warning":"تحذير","status_critical":"حرج","action_stop_machine":"إيقاف الآلة","action_inspect_bearings":"فحص المحامل","action_check_cooling":"فحص التبريد","action_schedule_shutdown_24h":"جدولة الإيقاف (24 ساعة)","action_order_spare_parts":"طلب قطع الغيار","action_reduce_load":"تقليل الحمل","action_schedule_inspection_72h":"جدولة الفحص (72 ساعة)","action_monitor_closely":"مراقبة عن كثب","action_verify_sensor":"التحقق من المستشعر","action_power_down_idle":"إطفاء الآلة الخاملة","action_review_shift_schedule":"مراجعة جدول المناوبات","action_no_action":"لا حاجة لأي إجراء"},
'tr': {"title":"FactoryPulse AI","subtitle":"Pilot Program Özet Raporu","prepared_for":"Hazırlanan","reporting_period":"Rapor dönemi: son {days} gün","generated":"Oluşturulma","overview":"Genel Bakış","factories_monitored":"İzlenen fabrikalar","machines_monitored":"İzlenen makineler","total_energy":"Toplam enerji tüketimi","avg_load":"Ortalama yük","avg_temp":"Ortalama sıcaklık","critical_incidents":"Kritik uyarılar (yakalanan olaylar)","energy_savings":"Belirlenen tahmini enerji tasarrufu","factories":"Fabrikalar","col_factory":"Fabrika","col_machines":"Makineler","col_type":"Tip","col_energy_cost":"Enerji Maliyeti","col_avg_load":"Ort. Yük","col_avg_temp":"Ort. Sıc.","no_factories":"Bu hesap için henüz fabrika kaydedilmedi.","oee_title":"Toplam Ekipman Etkinliği (OEE)","oee_formula":"OEE = Kullanılabilirlik x Performans x Kalite (ISO 22400).","oee_benchmark":"Dünya standardı %85, tipik bir fabrika %60 civarındadır.","grade":"Derece","availability":"Kullanılabilirlik","performance":"Performans","quality":"Kalite","run_time":"Çalışma süresi","downtime":"Duruş","scrap":"Fire","downtime_by_reason":"Nedene göre duruş","col_reason":"Neden","col_minutes":"Dakika","downtime_cost":"Dönem duruş maliyeti","financial_title":"Finansal Etki","potential_loss":"Müdahale edilmezse potansiyel kayıp","loss_avoided":"Erken tespitle önlenen kayıp","wasted_energy_month":"İsraf edilen enerji (aylık)","efficiency_gain":"Belirlenen verimlilik artışı","assumptions":"Varsayımlar: duruş {downtime}/sa, onarım {hours} sa, enerji {price}/kWh.","predictive_title":"Kestirimci Bakım Bulguları","col_machine":"Makine","col_risk":"Risk","col_remaining_life":"Kalan Ömür","col_root_cause":"Olası Kök Neden","healthy":"sağlıklı","workorders_title":"Bakım İş Emirleri","col_task":"Görev","col_priority":"Öncelik","col_status":"Durum","col_assigned":"Atanan","none":"Yok","alerts_title":"Kritik Uyarı Günlüğü","col_date":"Tarih","col_message":"Mesaj","no_alerts":"Bu dönemde kritik olay kaydedilmedi.","footer":"FactoryPulse AI tarafından otomatik oluşturuldu. Rakamlar seçilen dönemdeki sensör verilerinden türetilmiştir.","units":"adet","normal":"Normal","oee_title_short":"OEE","min_short":"dk","grade_world_class":"Dünya standardı","grade_typical":"Tipik","grade_low":"Düşük","grade_critical":"Kritik","reason_breakdown":"Arıza","reason_changeover":"Tip değişimi","reason_no_material":"Malzeme yok","reason_no_operator":"Operatör yok","reason_planned_maintenance":"Planlı bakım","reason_quality_issue":"Kalite sorunu","reason_setup":"Kurulum","reason_other":"Diğer","reason_unspecified":"Belirtilmemiş","cause_bearing_wear":"Rulman aşınması","cause_overload_thermal":"Termal aşırı yük","cause_cooling_failure":"Soğutma arızası","cause_misalignment":"Mil kaçıklığı","cause_lubrication_loss":"Yağlama kaybı","cause_normal_operation":"Normal çalışma","wo_open":"Açık","wo_in_progress":"Devam Ediyor","wo_done":"Tamamlandı","wo_cancelled":"İptal","prio_low":"Düşük","prio_medium":"Orta","prio_high":"Yüksek","prio_critical":"Kritik","h_short":"sa","d_short":"g","status_running":"Çalışıyor","status_stopped":"Durduruldu","status_maintenance":"Bakımda","status_warning":"Uyarı","status_critical":"Kritik","action_stop_machine":"Makineyi durdur","action_inspect_bearings":"Rulmanları incele","action_check_cooling":"Soğutmayı kontrol et","action_schedule_shutdown_24h":"Duruş planla (24s)","action_order_spare_parts":"Yedek parça sipariş et","action_reduce_load":"Yükü azalt","action_schedule_inspection_72h":"Muayene planla (72s)","action_monitor_closely":"Yakından izle","action_verify_sensor":"Sensörü doğrula","action_power_down_idle":"Boştaki makineyi kapat","action_review_shift_schedule":"Vardiya planını gözden geçir","action_no_action":"İşlem gerekmiyor"},
'it': {"title":"FactoryPulse AI","subtitle":"Rapporto di Sintesi del Programma Pilota","prepared_for":"Preparato per","reporting_period":"Periodo: ultimi {days} giorni","generated":"Generato il","overview":"Panoramica","factories_monitored":"Fabbriche monitorate","machines_monitored":"Macchine monitorate","total_energy":"Consumo energetico totale","avg_load":"Carico medio","avg_temp":"Temperatura media","critical_incidents":"Allarmi critici (incidenti rilevati)","energy_savings":"Risparmio energetico individuato (stima)","factories":"Fabbriche","col_factory":"Fabbrica","col_machines":"Macchine","col_type":"Tipo","col_energy_cost":"Costo Energia","col_avg_load":"Carico med.","col_avg_temp":"Temp. med.","no_factories":"Nessuna fabbrica registrata per questo account.","oee_title":"Efficienza Generale degli Impianti (OEE)","oee_formula":"OEE = Disponibilità x Prestazioni x Qualità (ISO 22400).","oee_benchmark":"Il livello mondiale è 85%, una fabbrica tipica si attesta sul 60%.","grade":"Valutazione","availability":"Disponibilità","performance":"Prestazioni","quality":"Qualità","run_time":"Tempo di funzionamento","downtime":"Fermo","scrap":"Scarto","downtime_by_reason":"Fermi per causa","col_reason":"Causa","col_minutes":"Minuti","downtime_cost":"Costo dei fermi nel periodo","financial_title":"Impatto Finanziario","potential_loss":"Perdita potenziale senza intervento","loss_avoided":"Perdita evitata con rilevamento precoce","wasted_energy_month":"Energia sprecata (al mese)","efficiency_gain":"Guadagno di efficienza individuato","assumptions":"Ipotesi: fermo {downtime}/h, riparazione {hours}h, energia {price}/kWh.","predictive_title":"Risultati della Manutenzione Predittiva","col_machine":"Macchina","col_risk":"Rischio","col_remaining_life":"Vita Utile Residua","col_root_cause":"Probabile Causa Radice","healthy":"sana","workorders_title":"Ordini di Lavoro di Manutenzione","col_task":"Attività","col_priority":"Priorità","col_status":"Stato","col_assigned":"Assegnato","none":"Nessuno","alerts_title":"Registro Allarmi Critici","col_date":"Data","col_message":"Messaggio","no_alerts":"Nessun incidente critico registrato in questo periodo.","footer":"Generato automaticamente da FactoryPulse AI. I dati derivano dai sensori monitorati nel periodo selezionato.","units":"unità","normal":"Normale","oee_title_short":"OEE","min_short":"min","grade_world_class":"Livello mondiale","grade_typical":"Tipico","grade_low":"Basso","grade_critical":"Critico","reason_breakdown":"Guasto","reason_changeover":"Cambio produzione","reason_no_material":"Materiale mancante","reason_no_operator":"Operatore mancante","reason_planned_maintenance":"Manutenzione pianificata","reason_quality_issue":"Problema qualità","reason_setup":"Setup","reason_other":"Altro","reason_unspecified":"Non specificato","cause_bearing_wear":"Usura del cuscinetto","cause_overload_thermal":"Sovraccarico termico","cause_cooling_failure":"Guasto raffreddamento","cause_misalignment":"Disallineamento albero","cause_lubrication_loss":"Perdita di lubrificazione","cause_normal_operation":"Funzionamento normale","wo_open":"Aperto","wo_in_progress":"In corso","wo_done":"Completato","wo_cancelled":"Annullato","prio_low":"Basso","prio_medium":"Medio","prio_high":"Alto","prio_critical":"Critico","h_short":"h","d_short":"g","status_running":"In funzione","status_stopped":"Ferma","status_maintenance":"Manutenzione","status_warning":"Avviso","status_critical":"Critico","action_stop_machine":"Ferma macchina","action_inspect_bearings":"Ispeziona cuscinetti","action_check_cooling":"Controlla raffreddamento","action_schedule_shutdown_24h":"Pianifica fermo (24h)","action_order_spare_parts":"Ordina ricambi","action_reduce_load":"Riduci carico","action_schedule_inspection_72h":"Pianifica ispezione (72h)","action_monitor_closely":"Monitora da vicino","action_verify_sensor":"Verifica sensore","action_power_down_idle":"Spegni macchina inattiva","action_review_shift_schedule":"Rivedi turni","action_no_action":"Nessuna azione necessaria"},
'pt': {"title":"FactoryPulse AI","subtitle":"Relatório de Síntese do Programa Piloto","prepared_for":"Preparado para","reporting_period":"Período: últimos {days} dias","generated":"Gerado em","overview":"Visão Geral","factories_monitored":"Fábricas monitoradas","machines_monitored":"Máquinas monitoradas","total_energy":"Consumo total de energia","avg_load":"Carga média","avg_temp":"Temperatura média","critical_incidents":"Alertas críticos (incidentes detectados)","energy_savings":"Economia de energia identificada (estimativa)","factories":"Fábricas","col_factory":"Fábrica","col_machines":"Máquinas","col_type":"Tipo","col_energy_cost":"Custo Energia","col_avg_load":"Carga méd.","col_avg_temp":"Temp. méd.","no_factories":"Nenhuma fábrica registrada nesta conta ainda.","oee_title":"Eficiência Global do Equipamento (OEE)","oee_formula":"OEE = Disponibilidade x Desempenho x Qualidade (ISO 22400).","oee_benchmark":"O nível mundial é 85%; uma fábrica típica fica perto de 60%.","grade":"Classificação","availability":"Disponibilidade","performance":"Desempenho","quality":"Qualidade","run_time":"Tempo de funcionamento","downtime":"Parada","scrap":"Refugo","downtime_by_reason":"Paradas por motivo","col_reason":"Motivo","col_minutes":"Minutos","downtime_cost":"Custo de paradas no período","financial_title":"Impacto Financeiro","potential_loss":"Perda potencial sem ação","loss_avoided":"Perda evitada por detecção precoce","wasted_energy_month":"Energia desperdiçada (por mês)","efficiency_gain":"Ganho de eficiência identificado","assumptions":"Premissas: parada {downtime}/h, reparo {hours}h, energia {price}/kWh.","predictive_title":"Resultados da Manutenção Preditiva","col_machine":"Máquina","col_risk":"Risco","col_remaining_life":"Vida Útil Restante","col_root_cause":"Provável Causa Raiz","healthy":"saudável","workorders_title":"Ordens de Serviço de Manutenção","col_task":"Tarefa","col_priority":"Prioridade","col_status":"Status","col_assigned":"Atribuído","none":"Nenhuma","alerts_title":"Registro de Alertas Críticos","col_date":"Data","col_message":"Mensagem","no_alerts":"Nenhum incidente crítico registrado neste período.","footer":"Gerado automaticamente pelo FactoryPulse AI. Os números derivam dos dados de sensores no período selecionado.","units":"un.","normal":"Normal","oee_title_short":"OEE","min_short":"min","grade_world_class":"Nível mundial","grade_typical":"Típico","grade_low":"Baixo","grade_critical":"Crítico","reason_breakdown":"Quebra","reason_changeover":"Troca de produto","reason_no_material":"Sem material","reason_no_operator":"Sem operador","reason_planned_maintenance":"Manutenção planejada","reason_quality_issue":"Problema de qualidade","reason_setup":"Preparação","reason_other":"Outro","reason_unspecified":"Não especificado","cause_bearing_wear":"Desgaste de rolamento","cause_overload_thermal":"Sobrecarga térmica","cause_cooling_failure":"Falha de refrigeração","cause_misalignment":"Desalinhamento do eixo","cause_lubrication_loss":"Perda de lubrificação","cause_normal_operation":"Operação normal","wo_open":"Aberta","wo_in_progress":"Em andamento","wo_done":"Concluída","wo_cancelled":"Cancelada","prio_low":"Baixo","prio_medium":"Médio","prio_high":"Alto","prio_critical":"Crítico","h_short":"h","d_short":"d","status_running":"Em funcionamento","status_stopped":"Parada","status_maintenance":"Manutenção","status_warning":"Aviso","status_critical":"Crítico","action_stop_machine":"Parar máquina","action_inspect_bearings":"Inspecionar rolamentos","action_check_cooling":"Verificar refrigeração","action_schedule_shutdown_24h":"Agendar parada (24h)","action_order_spare_parts":"Pedir peças","action_reduce_load":"Reduzir carga","action_schedule_inspection_72h":"Agendar inspeção (72h)","action_monitor_closely":"Monitorar de perto","action_verify_sensor":"Verificar sensor","action_power_down_idle":"Desligar máquina ociosa","action_review_shift_schedule":"Revisar turnos","action_no_action":"Nenhuma ação necessária"},
'ja': {"title":"FactoryPulse AI","subtitle":"パイロットプログラム総括レポート","prepared_for":"作成先","reporting_period":"対象期間：直近 {days} 日間","generated":"作成日時","overview":"概要","factories_monitored":"監視対象工場数","machines_monitored":"監視対象機械数","total_energy":"総エネルギー使用量","avg_load":"平均負荷","avg_temp":"平均温度","critical_incidents":"重大アラート（検出された事象）","energy_savings":"特定された省エネ効果（推定）","factories":"工場","col_factory":"工場","col_machines":"機械","col_type":"種類","col_energy_cost":"エネルギーコスト","col_avg_load":"平均負荷","col_avg_temp":"平均温度","no_factories":"このアカウントにはまだ工場が登録されていません。","oee_title":"設備総合効率 (OEE)","oee_formula":"OEE = 可用率 x 性能 x 品質 (ISO 22400)。","oee_benchmark":"ワールドクラスは85%、一般的な工場は約60%です。","grade":"評価","availability":"可用率","performance":"性能","quality":"品質","run_time":"稼働時間","downtime":"停止時間","scrap":"不良","downtime_by_reason":"理由別停止時間","col_reason":"理由","col_minutes":"分","downtime_cost":"期間中の停止コスト","financial_title":"財務インパクト","potential_loss":"対処しない場合の潜在損失","loss_avoided":"早期検知により回避した損失","wasted_energy_month":"無駄なエネルギー（月間）","efficiency_gain":"特定された効率向上","assumptions":"前提：停止 {downtime}/時、修理 {hours}時間、電力 {price}/kWh。","predictive_title":"予知保全の結果","col_machine":"機械","col_risk":"リスク","col_remaining_life":"残存耐用時間","col_root_cause":"推定根本原因","healthy":"正常","workorders_title":"保全作業指示","col_task":"タスク","col_priority":"優先度","col_status":"ステータス","col_assigned":"担当者","none":"なし","alerts_title":"重大アラート履歴","col_date":"日時","col_message":"メッセージ","no_alerts":"この期間に重大な事象は記録されていません。","footer":"FactoryPulse AI により自動生成されました。数値は選択期間の센서データに基づいています。","units":"個","normal":"標準","oee_title_short":"設備総合効率","min_short":"分","grade_world_class":"ワールドクラス","grade_typical":"標準的","grade_low":"低い","grade_critical":"重大","reason_breakdown":"故障","reason_changeover":"段取替え","reason_no_material":"材料切れ","reason_no_operator":"作業者不在","reason_planned_maintenance":"計画保全","reason_quality_issue":"品質問題","reason_setup":"セットアップ","reason_other":"その他","reason_unspecified":"未指定","cause_bearing_wear":"軸受摩耗","cause_overload_thermal":"熱過負荷","cause_cooling_failure":"冷却故障","cause_misalignment":"軸芯ずれ","cause_lubrication_loss":"潤滑不足","cause_normal_operation":"正常運転","wo_open":"未着手","wo_in_progress":"進行中","wo_done":"完了","wo_cancelled":"中止","prio_low":"低","prio_medium":"中","prio_high":"高","prio_critical":"重大","h_short":"時間","d_short":"日","status_running":"稼働中","status_stopped":"停止中","status_maintenance":"メンテナンス中","status_warning":"警告","status_critical":"重大","action_stop_machine":"機械を停止","action_inspect_bearings":"軸受を点検","action_check_cooling":"冷却を確認","action_schedule_shutdown_24h":"停止を計画（24時間）","action_order_spare_parts":"予備部品を発注","action_reduce_load":"負荷を下げる","action_schedule_inspection_72h":"点検を計画（72時間）","action_monitor_closely":"注意深く監視","action_verify_sensor":"センサーを確認","action_power_down_idle":"アイドル機械を停止","action_review_shift_schedule":"シフト計画を見直す","action_no_action":"対応不要"},
'ko': {"title":"FactoryPulse AI","subtitle":"파일럿 프로그램 요약 보고서","prepared_for":"수신자","reporting_period":"보고 기간: 최근 {days}일","generated":"생성 일시","overview":"개요","factories_monitored":"모니터링 공장 수","machines_monitored":"모니터링 기계 수","total_energy":"총 에너지 사용량","avg_load":"평균 부하","avg_temp":"평균 온도","critical_incidents":"심각 경고(감지된 사건)","energy_savings":"확인된 에너지 절감(추정)","factories":"공장","col_factory":"공장","col_machines":"기계","col_type":"유형","col_energy_cost":"에너지 비용","col_avg_load":"평균 부하","col_avg_temp":"평균 온도","no_factories":"이 계정에 등록된 공장이 아직 없습니다.","oee_title":"설비종합효율 (OEE)","oee_formula":"OEE = 가동률 x 성능 x 품질 (ISO 22400).","oee_benchmark":"세계 수준은 85%, 일반 공장은 약 60%입니다.","grade":"등급","availability":"가동률","performance":"성능","quality":"품질","run_time":"가동 시간","downtime":"정지 시간","scrap":"불량","downtime_by_reason":"사유별 정지시간","col_reason":"사유","col_minutes":"분","downtime_cost":"기간 정지 비용","financial_title":"재무 영향","potential_loss":"조치하지 않을 경우 잠재 손실","loss_avoided":"조기 감지로 회피한 손실","wasted_energy_month":"낭비된 에너지(월간)","efficiency_gain":"확인된 효율 향상","assumptions":"가정: 다운타임 {downtime}/시간, 수리 {hours}시간, 전력 {price}/kWh.","predictive_title":"예지보전 결과","col_machine":"기계","col_risk":"위험도","col_remaining_life":"잔여 수명","col_root_cause":"추정 근본 원인","healthy":"정상","workorders_title":"정비 작업 지시","col_task":"작업","col_priority":"우선순위","col_status":"상태","col_assigned":"담당자","none":"없음","alerts_title":"심각 경고 이력","col_date":"일시","col_message":"메시지","no_alerts":"이 기간에 심각한 사건이 기록되지 않았습니다.","footer":"FactoryPulse AI가 자동 생성했습니다. 수치는 선택 기간의 센서 데이터를 기반으로 합니다.","units":"개","normal":"보통","oee_title_short":"설비종합효율","min_short":"분","grade_world_class":"세계 수준","grade_typical":"일반","grade_low":"낮음","grade_critical":"심각","reason_breakdown":"고장","reason_changeover":"교체작업","reason_no_material":"자재 부족","reason_no_operator":"작업자 부재","reason_planned_maintenance":"계획 정비","reason_quality_issue":"품질 문제","reason_setup":"셋업","reason_other":"기타","reason_unspecified":"미지정","cause_bearing_wear":"베어링 마모","cause_overload_thermal":"열 과부하","cause_cooling_failure":"냉각 고장","cause_misalignment":"축 정렬 불량","cause_lubrication_loss":"윤활 손실","cause_normal_operation":"정상 운전","wo_open":"대기","wo_in_progress":"진행 중","wo_done":"완료","wo_cancelled":"취소","prio_low":"낮음","prio_medium":"보통","prio_high":"높음","prio_critical":"심각","h_short":"시간","d_short":"일","status_running":"가동 중","status_stopped":"정지됨","status_maintenance":"유지보수 중","status_warning":"경고","status_critical":"심각","action_stop_machine":"기계 정지","action_inspect_bearings":"베어링 점검","action_check_cooling":"냉각 확인","action_schedule_shutdown_24h":"가동 중단 예약 (24시간)","action_order_spare_parts":"예비 부품 주문","action_reduce_load":"부하 감소","action_schedule_inspection_72h":"점검 예약 (72시간)","action_monitor_closely":"면밀히 모니터링","action_verify_sensor":"센서 확인","action_power_down_idle":"유휴 기계 전원 차단","action_review_shift_schedule":"교대 일정 검토","action_no_action":"조치 불필요"},
'hi': {"title":"FactoryPulse AI","subtitle":"पायलट कार्यक्रम सारांश रिपोर्ट","prepared_for":"के लिए तैयार","reporting_period":"रिपोर्ट अवधि: पिछले {days} दिन","generated":"निर्मित","overview":"अवलोकन","factories_monitored":"निगरानी में कारखाने","machines_monitored":"निगरानी में मशीनें","total_energy":"कुल ऊर्जा खपत","avg_load":"औसत लोड","avg_temp":"औसत तापमान","critical_incidents":"गंभीर अलर्ट (पकड़ी गई घटनाएँ)","energy_savings":"पहचानी गई ऊर्जा बचत (अनुमान)","factories":"कारखाने","col_factory":"कारखाना","col_machines":"मशीनें","col_type":"प्रकार","col_energy_cost":"ऊर्जा लागत","col_avg_load":"औसत लोड","col_avg_temp":"औसत तापमान","no_factories":"इस खाते में अभी तक कोई कारखाना दर्ज नहीं है।","oee_title":"समग्र उपकरण प्रभावशीलता (OEE)","oee_formula":"OEE = उपलब्धता x प्रदर्शन x गुणवत्ता (ISO 22400)।","oee_benchmark":"विश्व स्तर 85% है; सामान्य कारखाना लगभग 60% पर रहता है।","grade":"श्रेणी","availability":"उपलब्धता","performance":"प्रदर्शन","quality":"गुणवत्ता","run_time":"चालू समय","downtime":"डाउनटाइम","scrap":"स्क्रैप","downtime_by_reason":"कारण अनुसार डाउनटाइम","col_reason":"कारण","col_minutes":"मिनट","downtime_cost":"अवधि में डाउनटाइम लागत","financial_title":"वित्तीय प्रभाव","potential_loss":"कार्रवाई न करने पर संभावित हानि","loss_avoided":"शीघ्र पहचान से बची हानि","wasted_energy_month":"बर्बाद ऊर्जा (मासिक)","efficiency_gain":"पहचानी गई दक्षता वृद्धि","assumptions":"अनुमान: डाउनटाइम {downtime}/घं, मरम्मत {hours} घं, ऊर्जा {price}/kWh।","predictive_title":"पूर्वानुमानित रखरखाव निष्कर्ष","col_machine":"मशीन","col_risk":"जोखिम","col_remaining_life":"शेष उपयोगी जीवन","col_root_cause":"संभावित मूल कारण","healthy":"स्वस्थ","workorders_title":"रखरखाव कार्य आदेश","col_task":"कार्य","col_priority":"प्राथमिकता","col_status":"स्थिति","col_assigned":"सौंपा गया","none":"कोई नहीं","alerts_title":"गंभीर अलर्ट लॉग","col_date":"दिनांक","col_message":"संदेश","no_alerts":"इस अवधि में कोई गंभीर घटना दर्ज नहीं हुई।","footer":"FactoryPulse AI द्वारा स्वतः निर्मित। आंकड़े चयनित अवधि के सेंसर डेटा से लिए गए हैं।","units":"इकाई","normal":"सामान्य","oee_title_short":"OEE","min_short":"मिनट","grade_world_class":"विश्व स्तर","grade_typical":"सामान्य","grade_low":"कम","grade_critical":"गंभीर","reason_breakdown":"खराबी","reason_changeover":"चेंजओवर","reason_no_material":"सामग्री नहीं","reason_no_operator":"ऑपरेटर नहीं","reason_planned_maintenance":"नियोजित रखरखाव","reason_quality_issue":"गुणवत्ता समस्या","reason_setup":"सेटअप","reason_other":"अन्य","reason_unspecified":"अनिर्दिष्ट","cause_bearing_wear":"बियरिंग घिसाव","cause_overload_thermal":"तापीय अधिभार","cause_cooling_failure":"शीतलन विफलता","cause_misalignment":"शाफ़्ट असंरेखण","cause_lubrication_loss":"स्नेहन हानि","cause_normal_operation":"सामान्य संचालन","wo_open":"खुला","wo_in_progress":"प्रगति में","wo_done":"पूर्ण","wo_cancelled":"रद्द","prio_low":"कम","prio_medium":"मध्यम","prio_high":"उच्च","prio_critical":"गंभीर","h_short":"घं","d_short":"दिन","status_running":"चल रहा है","status_stopped":"रुकी हुई","status_maintenance":"रखरखाव","status_warning":"चेतावनी","status_critical":"गंभीर","action_stop_machine":"मशीन रोकें","action_inspect_bearings":"बियरिंग जांचें","action_check_cooling":"शीतलन जांचें","action_schedule_shutdown_24h":"शटडाउन शेड्यूल करें (24घं)","action_order_spare_parts":"स्पेयर पार्ट्स ऑर्डर करें","action_reduce_load":"लोड कम करें","action_schedule_inspection_72h":"निरीक्षण शेड्यूल करें (72घं)","action_monitor_closely":"बारीकी से निगरानी करें","action_verify_sensor":"सेंसर सत्यापित करें","action_power_down_idle":"निष्क्रिय मशीन बंद करें","action_review_shift_schedule":"शिफ्ट शेड्यूल की समीक्षा करें","action_no_action":"किसी कार्रवाई की आवश्यकता नहीं"},
'uz': {"title":"FactoryPulse AI","subtitle":"Pilot dastur yakuniy hisoboti","prepared_for":"Kimga tayyorlandi","reporting_period":"Hisobot davri: oxirgi {days} kun","generated":"Yaratildi","overview":"Umumiy ko'rinish","factories_monitored":"Kuzatilayotgan zavodlar","machines_monitored":"Kuzatilayotgan stanoklar","total_energy":"Umumiy energiya sarfi","avg_load":"O'rtacha yuklama","avg_temp":"O'rtacha harorat","critical_incidents":"Tanqidiy ogohlantirishlar (aniqlangan hodisalar)","energy_savings":"Aniqlangan energiya tejami (taxminiy)","factories":"Zavodlar","col_factory":"Zavod","col_machines":"Stanoklar","col_type":"Turi","col_energy_cost":"Energiya narxi","col_avg_load":"O'rt. yuklama","col_avg_temp":"O'rt. harorat","no_factories":"Bu hisobga hali zavodlar qo'shilmagan.","oee_title":"Uskunaning umumiy samaradorligi (OEE)","oee_formula":"OEE = Mavjudlik x Unumdorlik x Sifat (ISO 22400).","oee_benchmark":"Jahon darajasi 85%, odatiy zavod esa 60% atrofida.","grade":"Baho","availability":"Mavjudlik","performance":"Unumdorlik","quality":"Sifat","run_time":"Ish vaqti","downtime":"To'xtash","scrap":"Brak","downtime_by_reason":"Sabab bo'yicha to'xtash","col_reason":"Sabab","col_minutes":"Daqiqa","downtime_cost":"Davr uchun to'xtash narxi","financial_title":"Moliyaviy ta'sir","potential_loss":"Harakat qilinmasa potentsial yo'qotish","loss_avoided":"Erta aniqlash bilan oldi olingan yo'qotish","wasted_energy_month":"Isrof energiya (oyiga)","efficiency_gain":"Aniqlangan samaradorlik o'sishi","assumptions":"Taxminlar: to'xtash {downtime}/soat, ta'mir {hours} soat, energiya {price}/kWh.","predictive_title":"Bashoratli texnik xizmat natijalari","col_machine":"Stanok","col_risk":"Xavf","col_remaining_life":"Qolgan foydali muddat","col_root_cause":"Ehtimoliy asosiy sabab","healthy":"sog'lom","workorders_title":"Texnik xizmat ish buyruqlari","col_task":"Vazifa","col_priority":"Muhimlik","col_status":"Holati","col_assigned":"Mas'ul","none":"Yo'q","alerts_title":"Tanqidiy ogohlantirishlar jurnali","col_date":"Sana","col_message":"Xabar","no_alerts":"Bu davrda tanqidiy hodisalar qayd etilmagan.","footer":"FactoryPulse AI tomonidan avtomatik yaratildi. Ko'rsatkichlar tanlangan davrdagi sensor ma'lumotlariga asoslangan.","units":"birlik","normal":"Oddiy","oee_title_short":"OEE","min_short":"daq","grade_world_class":"Jahon darajasi","grade_typical":"Odatiy","grade_low":"Past","grade_critical":"Tanqidiy","reason_breakdown":"Buzilish","reason_changeover":"Qayta sozlash","reason_no_material":"Material yo'q","reason_no_operator":"Operator yo'q","reason_planned_maintenance":"Rejali texnik xizmat","reason_quality_issue":"Sifat muammosi","reason_setup":"Sozlash","reason_other":"Boshqa","reason_unspecified":"Ko'rsatilmagan","cause_bearing_wear":"Podshipnik eskirishi","cause_overload_thermal":"Termik ortiqcha yuk","cause_cooling_failure":"Sovutish nosozligi","cause_misalignment":"Val nomuvofiqligi","cause_lubrication_loss":"Moylash yo'qolishi","cause_normal_operation":"Normal ish","wo_open":"Ochiq","wo_in_progress":"Bajarilmoqda","wo_done":"Bajarildi","wo_cancelled":"Bekor qilindi","prio_low":"Past","prio_medium":"O'rta","prio_high":"Yuqori","prio_critical":"Tanqidiy","h_short":"soat","d_short":"kun","status_running":"Ishlamoqda","status_stopped":"To'xtatilgan","status_maintenance":"Texnik xizmat","status_warning":"Ogohlantirish","status_critical":"Muhim","action_stop_machine":"Stanokni to'xtatish","action_inspect_bearings":"Podshipniklarni tekshirish","action_check_cooling":"Sovutishni tekshirish","action_schedule_shutdown_24h":"To'xtatishni rejalashtirish (24s)","action_order_spare_parts":"Ehtiyot qismlar buyurtma qilish","action_reduce_load":"Yuklamani kamaytirish","action_schedule_inspection_72h":"Tekshiruvni rejalashtirish (72s)","action_monitor_closely":"Diqqat bilan kuzatish","action_verify_sensor":"Sensorni tekshirish","action_power_down_idle":"Bo'sh stanokni o'chirish","action_review_shift_schedule":"Smena jadvalini ko'rib chiqish","action_no_action":"Harakat kerak emas"},
'ky': {"title":"FactoryPulse AI","subtitle":"Пилоттук программанын жыйынтык отчету","prepared_for":"Кимге даярдалды","reporting_period":"Отчет мезгили: акыркы {days} күн","generated":"Түзүлдү","overview":"Жалпы көрүнүш","factories_monitored":"Көзөмөлдөнүүчү заводдор","machines_monitored":"Көзөмөлдөнүүчү станоктор","total_energy":"Жалпы энергия керектөө","avg_load":"Орточо жүктөм","avg_temp":"Орточо температура","critical_incidents":"Критикалык эскертүүлөр (аныкталган окуялар)","energy_savings":"Аныкталган энергия үнөмү (болжолдуу)","factories":"Заводдор","col_factory":"Завод","col_machines":"Станоктор","col_type":"Түрү","col_energy_cost":"Энергия наркы","col_avg_load":"Орт. жүктөм","col_avg_temp":"Орт. темп.","no_factories":"Бул каттоо эсебине азырынча заводдор кошулган эмес.","oee_title":"Жабдуунун жалпы натыйжалуулугу (OEE)","oee_formula":"OEE = Жеткиликтүүлүк x Өндүрүмдүүлүк x Сапат (ISO 22400).","oee_benchmark":"Дүйнөлүк деңгээл 85%, кадимки завод 60% чамасында.","grade":"Баасы","availability":"Жеткиликтүүлүк","performance":"Өндүрүмдүүлүк","quality":"Сапат","run_time":"Иштөө убактысы","downtime":"Токтоп калуу","scrap":"Брак","downtime_by_reason":"Себеби боюнча токтоп калуу","col_reason":"Себеби","col_minutes":"Мүнөт","downtime_cost":"Мезгилдеги токтоп калуу наркы","financial_title":"Каржылык таасир","potential_loss":"Аракет кылынбаса потенциалдуу жоготуу","loss_avoided":"Эрте аныктоо менен алдын алынган жоготуу","wasted_energy_month":"Ысырап энергия (айына)","efficiency_gain":"Аныкталган эффективдүүлүк өсүшү","assumptions":"Болжолдор: токтоп калуу {downtime}/саат, оңдоо {hours} саат, энергия {price}/kWh.","predictive_title":"Болжолдуу тейлөө жыйынтыктары","col_machine":"Станок","col_risk":"Тобокелдик","col_remaining_life":"Калган ресурс","col_root_cause":"Ыктымалдуу негизги себеп","healthy":"сак","workorders_title":"Тейлөө иш буйруктары","col_task":"Тапшырма","col_priority":"Маанилүүлүк","col_status":"Абалы","col_assigned":"Жооптуу","none":"Жок","alerts_title":"Критикалык эскертүүлөр журналы","col_date":"Күнү","col_message":"Кабар","no_alerts":"Бул мезгилде критикалык окуялар катталган эмес.","footer":"FactoryPulse AI тарабынан автоматтык түзүлдү. Көрсөткүчтөр тандалган мезгилдеги сенсор маалыматтарына негизделген.","units":"бирдик","normal":"Кадимки","oee_title_short":"OEE","min_short":"мүн","grade_world_class":"Дүйнөлүк деңгээл","grade_typical":"Кадимки","grade_low":"Төмөн","grade_critical":"Критикалык","reason_breakdown":"Бузулуу","reason_changeover":"Кайра жөндөө","reason_no_material":"Материал жок","reason_no_operator":"Оператор жок","reason_planned_maintenance":"Пландуу тейлөө","reason_quality_issue":"Сапат маселеси","reason_setup":"Орнотуу","reason_other":"Башка","reason_unspecified":"Көрсөтүлгөн эмес","cause_bearing_wear":"Подшипниктин эскириши","cause_overload_thermal":"Жылуулук ашыкча жүктөө","cause_cooling_failure":"Муздатуу бузулушу","cause_misalignment":"Валдын борборунан жылышы","cause_lubrication_loss":"Майлоонун жоголушу","cause_normal_operation":"Кадимки иштөө","wo_open":"Ачык","wo_in_progress":"Аткарылууда","wo_done":"Аткарылды","wo_cancelled":"Жокко чыгарылды","prio_low":"Төмөн","prio_medium":"Орточо","prio_high":"Жогору","prio_critical":"Критикалык","h_short":"саат","d_short":"күн","status_running":"Иштеп жатат","status_stopped":"Токтотулган","status_maintenance":"Тейлөө","status_warning":"Эскертүү","status_critical":"Олуттуу","action_stop_machine":"Станокту токтотуу","action_inspect_bearings":"Подшипниктерди текшерүү","action_check_cooling":"Муздатууну текшерүү","action_schedule_shutdown_24h":"Токтотууну пландаштыруу (24с)","action_order_spare_parts":"Камдык бөлүктөрдү заказ кылуу","action_reduce_load":"Жүктөмдү азайтуу","action_schedule_inspection_72h":"Текшерүүнү пландаштыруу (72с)","action_monitor_closely":"Кылдат байкоо","action_verify_sensor":"Сенсорду текшерүү","action_power_down_idle":"Бош станокту өчүрүү","action_review_shift_schedule":"Смена графигин кароо","action_no_action":"Аракет талап кылынбайт"},
'uk': {"title":"FactoryPulse AI","subtitle":"Підсумковий звіт пілотної програми","prepared_for":"Підготовлено для","reporting_period":"Звітний період: останні {days} днів","generated":"Сформовано","overview":"Огляд","factories_monitored":"Заводів під наглядом","machines_monitored":"Верстатів під наглядом","total_energy":"Загальне енергоспоживання","avg_load":"Середнє навантаження","avg_temp":"Середня температура","critical_incidents":"Критичні сповіщення (виявлені інциденти)","energy_savings":"Оцінка виявленої економії енергії","factories":"Заводи","col_factory":"Завод","col_machines":"Верстати","col_type":"Тип","col_energy_cost":"Вартість енергії","col_avg_load":"Сер. навант.","col_avg_temp":"Сер. темп.","no_factories":"Для цього облікового запису заводи ще не додано.","oee_title":"Загальна ефективність обладнання (OEE)","oee_formula":"OEE = Доступність x Продуктивність x Якість (ISO 22400).","oee_benchmark":"Світовий рівень — 85%, типовий завод — близько 60%.","grade":"Оцінка","availability":"Доступність","performance":"Продуктивність","quality":"Якість","run_time":"Час роботи","downtime":"Простій","scrap":"Брак","downtime_by_reason":"Простої за причинами","col_reason":"Причина","col_minutes":"Хвилини","downtime_cost":"Вартість простоїв за період","financial_title":"Фінансовий вплив","potential_loss":"Потенційні втрати без реакції","loss_avoided":"Втрати, відвернені раннім виявленням","wasted_energy_month":"Втрати енергії (на місяць)","efficiency_gain":"Виявлений приріст ефективності","assumptions":"Припущення: простій {downtime}/год, ремонт {hours} год, енергія {price}/кВт·год.","predictive_title":"Результати прогнозної діагностики","col_machine":"Верстат","col_risk":"Ризик","col_remaining_life":"Залишковий ресурс","col_root_cause":"Ймовірна першопричина","healthy":"справний","workorders_title":"Наряди на обслуговування","col_task":"Завдання","col_priority":"Пріоритет","col_status":"Статус","col_assigned":"Призначено","none":"Немає","alerts_title":"Журнал критичних сповіщень","col_date":"Дата","col_message":"Повідомлення","no_alerts":"За цей період критичних інцидентів не зафіксовано.","footer":"Сформовано автоматично системою FactoryPulse AI. Показники розраховані за даними датчиків за обраний період.","units":"од.","normal":"Звичайний","oee_title_short":"OEE","min_short":"хв","grade_world_class":"Світовий рівень","grade_typical":"Типовий","grade_low":"Низький","grade_critical":"Критичний","reason_breakdown":"Поломка","reason_changeover":"Переналагодження","reason_no_material":"Немає матеріалу","reason_no_operator":"Немає оператора","reason_planned_maintenance":"Планове ТО","reason_quality_issue":"Проблема якості","reason_setup":"Налаштування","reason_other":"Інше","reason_unspecified":"Не вказано","cause_bearing_wear":"Знос підшипника","cause_overload_thermal":"Теплове перевантаження","cause_cooling_failure":"Відмова охолодження","cause_misalignment":"Розцентрування вала","cause_lubrication_loss":"Втрата мастила","cause_normal_operation":"Нормальна робота","wo_open":"Відкритий","wo_in_progress":"В роботі","wo_done":"Виконаний","wo_cancelled":"Скасований","prio_low":"Низький","prio_medium":"Середній","prio_high":"Високий","prio_critical":"Критичний","h_short":"год","d_short":"дн","status_running":"Працює","status_stopped":"Зупинено","status_maintenance":"Обслуговування","status_warning":"Попередження","status_critical":"Критично","action_stop_machine":"Зупинити верстат","action_inspect_bearings":"Перевірити підшипники","action_check_cooling":"Перевірити охолодження","action_schedule_shutdown_24h":"Запланувати зупинку (24год)","action_order_spare_parts":"Замовити запчастини","action_reduce_load":"Знизити навантаження","action_schedule_inspection_72h":"Запланувати огляд (72год)","action_monitor_closely":"Уважно спостерігати","action_verify_sensor":"Перевірити датчик","action_power_down_idle":"Вимкнути простійний верстат","action_review_shift_schedule":"Переглянути графік змін","action_no_action":"Дій не потрібно"},
'pl': {"title":"FactoryPulse AI","subtitle":"Raport Podsumowujący Program Pilotażowy","prepared_for":"Przygotowano dla","reporting_period":"Okres raportu: ostatnie {days} dni","generated":"Wygenerowano","overview":"Przegląd","factories_monitored":"Monitorowane fabryki","machines_monitored":"Monitorowane maszyny","total_energy":"Całkowite zużycie energii","avg_load":"Średnie obciążenie","avg_temp":"Średnia temperatura","critical_incidents":"Krytyczne alerty (wykryte zdarzenia)","energy_savings":"Zidentyfikowane oszczędności energii (szacunek)","factories":"Fabryki","col_factory":"Fabryka","col_machines":"Maszyny","col_type":"Typ","col_energy_cost":"Koszt Energii","col_avg_load":"Śr. obciąż.","col_avg_temp":"Śr. temp.","no_factories":"Brak zarejestrowanych fabryk na tym koncie.","oee_title":"Całkowita Efektywność Wyposażenia (OEE)","oee_formula":"OEE = Dostępność x Wydajność x Jakość (ISO 22400).","oee_benchmark":"Poziom światowy to 85%, typowa fabryka około 60%.","grade":"Ocena","availability":"Dostępność","performance":"Wydajność","quality":"Jakość","run_time":"Czas pracy","downtime":"Przestój","scrap":"Braki","downtime_by_reason":"Przestoje wg przyczyny","col_reason":"Przyczyna","col_minutes":"Minuty","downtime_cost":"Koszt przestojów w okresie","financial_title":"Wpływ Finansowy","potential_loss":"Potencjalna strata bez działania","loss_avoided":"Strata uniknięta dzięki wczesnemu wykryciu","wasted_energy_month":"Zmarnowana energia (miesięcznie)","efficiency_gain":"Zidentyfikowany wzrost wydajności","assumptions":"Założenia: przestój {downtime}/h, naprawa {hours}h, energia {price}/kWh.","predictive_title":"Wyniki Konserwacji Predykcyjnej","col_machine":"Maszyna","col_risk":"Ryzyko","col_remaining_life":"Pozostała Żywotność","col_root_cause":"Prawdopodobna Przyczyna Źródłowa","healthy":"sprawna","workorders_title":"Zlecenia Konserwacji","col_task":"Zadanie","col_priority":"Priorytet","col_status":"Status","col_assigned":"Przypisane","none":"Brak","alerts_title":"Dziennik Alertów Krytycznych","col_date":"Data","col_message":"Wiadomość","no_alerts":"W tym okresie nie odnotowano krytycznych zdarzeń.","footer":"Wygenerowano automatycznie przez FactoryPulse AI. Dane pochodzą z czujników w wybranym okresie.","units":"szt.","normal":"Normalny","oee_title_short":"OEE","min_short":"min","grade_world_class":"Poziom światowy","grade_typical":"Typowy","grade_low":"Niski","grade_critical":"Krytyczny","reason_breakdown":"Awaria","reason_changeover":"Przezbrojenie","reason_no_material":"Brak materiału","reason_no_operator":"Brak operatora","reason_planned_maintenance":"Konserwacja planowa","reason_quality_issue":"Problem jakości","reason_setup":"Ustawianie","reason_other":"Inne","reason_unspecified":"Nieokreślone","cause_bearing_wear":"Zużycie łożyska","cause_overload_thermal":"Przeciążenie termiczne","cause_cooling_failure":"Awaria chłodzenia","cause_misalignment":"Niewspółosiowość wału","cause_lubrication_loss":"Utrata smarowania","cause_normal_operation":"Praca normalna","wo_open":"Otwarte","wo_in_progress":"W trakcie","wo_done":"Zakończone","wo_cancelled":"Anulowane","prio_low":"Niski","prio_medium":"Średni","prio_high":"Wysoki","prio_critical":"Krytyczny","h_short":"h","d_short":"d","status_running":"Działa","status_stopped":"Zatrzymana","status_maintenance":"Konserwacja","status_warning":"Ostrzeżenie","status_critical":"Krytyczne","action_stop_machine":"Zatrzymaj maszynę","action_inspect_bearings":"Sprawdź łożyska","action_check_cooling":"Sprawdź chłodzenie","action_schedule_shutdown_24h":"Zaplanuj postój (24h)","action_order_spare_parts":"Zamów części","action_reduce_load":"Zmniejsz obciążenie","action_schedule_inspection_72h":"Zaplanuj przegląd (72h)","action_monitor_closely":"Uważnie monitoruj","action_verify_sensor":"Zweryfikuj czujnik","action_power_down_idle":"Wyłącz bezczynną maszynę","action_review_shift_schedule":"Przejrzyj grafik zmian","action_no_action":"Nie wymaga działania"},
'nl': {"title":"FactoryPulse AI","subtitle":"Samenvattend Rapport Pilotprogramma","prepared_for":"Opgesteld voor","reporting_period":"Rapportageperiode: laatste {days} dagen","generated":"Gegenereerd op","overview":"Overzicht","factories_monitored":"Bewaakte fabrieken","machines_monitored":"Bewaakte machines","total_energy":"Totaal energieverbruik","avg_load":"Gemiddelde belasting","avg_temp":"Gemiddelde temperatuur","critical_incidents":"Kritieke waarschuwingen (gedetecteerde incidenten)","energy_savings":"Geïdentificeerde energiebesparing (schatting)","factories":"Fabrieken","col_factory":"Fabriek","col_machines":"Machines","col_type":"Type","col_energy_cost":"Energiekosten","col_avg_load":"Gem. belasting","col_avg_temp":"Gem. temp.","no_factories":"Nog geen fabrieken geregistreerd voor dit account.","oee_title":"Totale Installatie-effectiviteit (OEE)","oee_formula":"OEE = Beschikbaarheid x Prestatie x Kwaliteit (ISO 22400).","oee_benchmark":"Wereldklasse is 85%; een typische fabriek zit rond 60%.","grade":"Beoordeling","availability":"Beschikbaarheid","performance":"Prestatie","quality":"Kwaliteit","run_time":"Draaitijd","downtime":"Stilstand","scrap":"Uitval","downtime_by_reason":"Stilstand per oorzaak","col_reason":"Oorzaak","col_minutes":"Minuten","downtime_cost":"Stilstandkosten in de periode","financial_title":"Financiële Impact","potential_loss":"Potentieel verlies zonder actie","loss_avoided":"Verlies vermeden door vroege detectie","wasted_energy_month":"Verspilde energie (per maand)","efficiency_gain":"Geïdentificeerde efficiëntiewinst","assumptions":"Aannames: stilstand {downtime}/u, reparatie {hours}u, energie {price}/kWh.","predictive_title":"Bevindingen Voorspellend Onderhoud","col_machine":"Machine","col_risk":"Risico","col_remaining_life":"Resterende Levensduur","col_root_cause":"Waarschijnlijke Hoofdoorzaak","healthy":"gezond","workorders_title":"Onderhoudswerkorders","col_task":"Taak","col_priority":"Prioriteit","col_status":"Status","col_assigned":"Toegewezen","none":"Geen","alerts_title":"Logboek Kritieke Waarschuwingen","col_date":"Datum","col_message":"Bericht","no_alerts":"In deze periode zijn geen kritieke incidenten geregistreerd.","footer":"Automatisch gegenereerd door FactoryPulse AI. Cijfers zijn afgeleid van sensordata in de geselecteerde periode.","units":"st.","normal":"Normaal","oee_title_short":"OEE","min_short":"min","grade_world_class":"Wereldklasse","grade_typical":"Typisch","grade_low":"Laag","grade_critical":"Kritiek","reason_breakdown":"Storing","reason_changeover":"Omstelling","reason_no_material":"Geen materiaal","reason_no_operator":"Geen operator","reason_planned_maintenance":"Gepland onderhoud","reason_quality_issue":"Kwaliteitsprobleem","reason_setup":"Instellen","reason_other":"Overig","reason_unspecified":"Niet gespecificeerd","cause_bearing_wear":"Lagerslijtage","cause_overload_thermal":"Thermische overbelasting","cause_cooling_failure":"Koelstoring","cause_misalignment":"Asuitlijnfout","cause_lubrication_loss":"Smeringverlies","cause_normal_operation":"Normale werking","wo_open":"Open","wo_in_progress":"In uitvoering","wo_done":"Afgerond","wo_cancelled":"Geannuleerd","prio_low":"Laag","prio_medium":"Gemiddeld","prio_high":"Hoog","prio_critical":"Kritiek","h_short":"u","d_short":"d","status_running":"Actief","status_stopped":"Gestopt","status_maintenance":"Onderhoud","status_warning":"Waarschuwing","status_critical":"Kritiek","action_stop_machine":"Machine stoppen","action_inspect_bearings":"Lagers inspecteren","action_check_cooling":"Koeling controleren","action_schedule_shutdown_24h":"Stilstand plannen (24u)","action_order_spare_parts":"Reserveonderdelen bestellen","action_reduce_load":"Belasting verlagen","action_schedule_inspection_72h":"Inspectie plannen (72u)","action_monitor_closely":"Nauwlettend volgen","action_verify_sensor":"Sensor verifiëren","action_power_down_idle":"Inactieve machine uitschakelen","action_review_shift_schedule":"Ploegrooster herzien","action_no_action":"Geen actie nodig"},
'sv': {"title":"FactoryPulse AI","subtitle":"Sammanfattande Rapport för Pilotprogram","prepared_for":"Framtagen för","reporting_period":"Rapportperiod: senaste {days} dagarna","generated":"Genererad","overview":"Översikt","factories_monitored":"Övervakade fabriker","machines_monitored":"Övervakade maskiner","total_energy":"Total energianvändning","avg_load":"Genomsnittlig belastning","avg_temp":"Genomsnittlig temperatur","critical_incidents":"Kritiska varningar (upptäckta händelser)","energy_savings":"Identifierad energibesparing (uppskattning)","factories":"Fabriker","col_factory":"Fabrik","col_machines":"Maskiner","col_type":"Typ","col_energy_cost":"Energikostnad","col_avg_load":"Snittbelastning","col_avg_temp":"Snitttemp.","no_factories":"Inga fabriker registrerade för detta konto ännu.","oee_title":"Total Utrustningseffektivitet (OEE)","oee_formula":"OEE = Tillgänglighet x Prestanda x Kvalitet (ISO 22400).","oee_benchmark":"Världsklass är 85%; en typisk fabrik ligger nära 60%.","grade":"Betyg","availability":"Tillgänglighet","performance":"Prestanda","quality":"Kvalitet","run_time":"Drifttid","downtime":"Stillestånd","scrap":"Kassation","downtime_by_reason":"Stillestånd per orsak","col_reason":"Orsak","col_minutes":"Minuter","downtime_cost":"Stilleståndskostnad under perioden","financial_title":"Finansiell Påverkan","potential_loss":"Potentiell förlust utan åtgärd","loss_avoided":"Förlust undviken genom tidig upptäckt","wasted_energy_month":"Bortslösad energi (per månad)","efficiency_gain":"Identifierad effektivitetsvinst","assumptions":"Antaganden: stillestånd {downtime}/h, reparation {hours}h, energi {price}/kWh.","predictive_title":"Resultat från Prediktivt Underhåll","col_machine":"Maskin","col_risk":"Risk","col_remaining_life":"Återstående Livslängd","col_root_cause":"Trolig Grundorsak","healthy":"frisk","workorders_title":"Underhållsarbetsordrar","col_task":"Uppgift","col_priority":"Prioritet","col_status":"Status","col_assigned":"Tilldelad","none":"Inga","alerts_title":"Logg för Kritiska Varningar","col_date":"Datum","col_message":"Meddelande","no_alerts":"Inga kritiska händelser registrerades under denna period.","footer":"Genererad automatiskt av FactoryPulse AI. Siffrorna härrör från sensordata under vald period.","units":"st.","normal":"Normal","oee_title_short":"OEE","min_short":"min","grade_world_class":"Världsklass","grade_typical":"Typisk","grade_low":"Låg","grade_critical":"Kritisk","reason_breakdown":"Haveri","reason_changeover":"Omställning","reason_no_material":"Inget material","reason_no_operator":"Ingen operatör","reason_planned_maintenance":"Planerat underhåll","reason_quality_issue":"Kvalitetsproblem","reason_setup":"Inställning","reason_other":"Övrigt","reason_unspecified":"Ej angivet","cause_bearing_wear":"Lagerslitage","cause_overload_thermal":"Termisk överbelastning","cause_cooling_failure":"Kylfel","cause_misalignment":"Axelfelinriktning","cause_lubrication_loss":"Smörjförlust","cause_normal_operation":"Normal drift","wo_open":"Öppen","wo_in_progress":"Pågår","wo_done":"Klar","wo_cancelled":"Avbruten","prio_low":"Låg","prio_medium":"Medel","prio_high":"Hög","prio_critical":"Kritisk","h_short":"h","d_short":"d","status_running":"Igång","status_stopped":"Stoppad","status_maintenance":"Underhåll","status_warning":"Varning","status_critical":"Kritisk","action_stop_machine":"Stoppa maskinen","action_inspect_bearings":"Inspektera lager","action_check_cooling":"Kontrollera kylning","action_schedule_shutdown_24h":"Schemalägg stopp (24h)","action_order_spare_parts":"Beställ reservdelar","action_reduce_load":"Minska belastning","action_schedule_inspection_72h":"Schemalägg inspektion (72h)","action_monitor_closely":"Övervaka noga","action_verify_sensor":"Verifiera sensor","action_power_down_idle":"Stäng av tomgångsmaskin","action_review_shift_schedule":"Se över skiftschema","action_no_action":"Ingen åtgärd behövs"},
}


def pdf_t(lang):
    """Returns a lookup function for PDF strings in the requested language,
    falling back to English for any key a translation is missing."""
    table = PDF_TEXT.get(lang, PDF_TEXT["en"])
    fallback = PDF_TEXT["en"]
    def _(key, **kwargs):
        text = table.get(key, fallback.get(key, key))
        return text.format(**kwargs) if kwargs else text
    return _

def _build_pdf_report(user, factories, machines, alerts, days,
                      oee=None, shifts=None, roi=None, work_orders=None, predictions=None,
                      lang="en"):
    _ = pdf_t(lang)   # every user-facing string below is localised
    # Pick a font that can actually render this language (CJK needs its own).
    font, font_bold = pdf_fonts_for(lang)
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=18 * mm, bottomMargin=18 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    # font is a registered Unicode TTF so Cyrillic/Kazakh names render properly.
    title_style = ParagraphStyle("FPTitle", parent=styles["Title"], textColor=colors.HexColor("#0f172a"), fontSize=22, fontName=font_bold)
    h2_style = ParagraphStyle("FPH2", parent=styles["Heading2"], textColor=colors.HexColor("#0891b2"), spaceBefore=14, spaceAfter=6, fontName=font_bold)
    h3_style = ParagraphStyle("FPH3", parent=styles["Heading3"], fontName=font_bold)
    body_style = ParagraphStyle("FPBody", parent=styles["Normal"], fontSize=10, leading=14, fontName=font)
    small_style = ParagraphStyle("FPSmall", parent=styles["Normal"], fontSize=8, textColor=colors.grey, fontName=font)

    total_machines = len(machines)
    total_energy = round(sum(f.machines * f.load * f.energy_cost for f in factories), 1) if factories else 0
    avg_load = round(sum(m.load for m in machines) / total_machines, 1) if total_machines else 0
    avg_temp = round(sum(m.temperature for m in machines) / total_machines, 1) if total_machines else 0
    critical_alerts = [a for a in alerts if a.severity == "critical"]
    incidents_caught = len(critical_alerts)
    # Heuristic pilot-savings estimate: flagged/critical machines represent avoided unplanned
    # downtime; energy savings are estimated from the standard 10-15% optimization band the
    # AI engine recommends for elevated-load plants.
    estimated_energy_savings_kwh = round(total_energy * 0.12, 1)

    elements = []
    elements.append(Paragraph(_("title"), title_style))
    elements.append(Paragraph(_("subtitle"), h3_style))
    elements.append(Spacer(1, 4 * mm))
    elements.append(Paragraph(
        f'{_("prepared_for")}: {user.full_name} ({user.email})<br/>'
        f'{_("reporting_period", days=days)}<br/>'
        f'{_("generated")}: {datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}',
        body_style,
    ))
    elements.append(Spacer(1, 6 * mm))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#e2e8f0")))

    elements.append(Paragraph(_("overview"), h2_style))
    overview_data = [
        [_("factories_monitored"), str(len(factories))],
        [_("machines_monitored"), str(total_machines)],
        [_("total_energy"), f"{total_energy} kWh"],
        [_("avg_load"), f"{avg_load}%"],
        [_("avg_temp"), f"{avg_temp}°C"],
        [_("critical_incidents"), str(incidents_caught)],
        [_("energy_savings"), f"{estimated_energy_savings_kwh} kWh"],
    ]
    overview_table = Table(overview_data, colWidths=[95 * mm, 70 * mm])
    overview_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#475569")),
        ("FONTNAME", (1, 0), (1, -1), font_bold),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    elements.append(overview_table)

    elements.append(Paragraph(_("factories"), h2_style))
    if factories:
        factory_rows = [[_("col_factory"), _("col_machines"), _("col_type"), _("col_energy_cost"), _("col_avg_load"), _("col_avg_temp")]]
        for f in factories:
            factory_rows.append([
                f.factory_name, str(f.machines), f.machine_type,
                f"${f.energy_cost}/kWh", f"{f.load}%", f"{f.temperature}°C",
            ])
        factory_table = Table(factory_rows, colWidths=[45 * mm, 22 * mm, 25 * mm, 28 * mm, 22 * mm, 22 * mm])
        factory_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), font),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (-1, 0), font_bold),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ]))
        elements.append(factory_table)
    else:
        elements.append(Paragraph(_("no_factories"), body_style))

    # ---------------- OEE (the standard manufacturing KPI) ----------------
    if oee and oee.get("shift_count"):
        o = oee["overall"]
        elements.append(Paragraph(_("oee_title"), h2_style))
        elements.append(Paragraph(
_("oee_formula") + " " + _("oee_benchmark"), small_style))
        elements.append(Spacer(1, 2 * mm))
        oee_rows = [
            [_("oee_title_short"), f"{o['oee']}%", _("grade"), _("grade_" + o["grade"])],
            [_("availability"), f"{o['availability']}%", _("run_time"), f"{o['run_time_minutes']} {_('min_short')}"],
            [_("performance"), f"{o['performance']}%", _("downtime"), f"{o['downtime_minutes']} {_('min_short')}"],
            [_("quality"), f"{o['quality']}%", _("scrap"), f"{o['scrap_units']} {_('units')}"],
        ]
        oee_table = Table(oee_rows, colWidths=[38 * mm, 30 * mm, 45 * mm, 52 * mm])
        oee_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#475569")),
            ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#475569")),
            ("FONTNAME", (1, 0), (1, -1), font_bold),
            ("FONTNAME", (3, 0), (3, -1), font_bold),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#e2e8f0")),
        ]))
        elements.append(oee_table)

        if oee.get("downtime_by_reason"):
            elements.append(Spacer(1, 4 * mm))
            elements.append(Paragraph(_("downtime_by_reason"), body_style))
            dt_rows = [[_("col_reason"), _("col_minutes")]]
            for r in oee["downtime_by_reason"][:8]:
                dt_rows.append([_("reason_" + r["reason"]), str(r["minutes"])])
            dt_table = Table(dt_rows, colWidths=[110 * mm, 55 * mm])
            dt_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), font),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 0), (-1, 0), font_bold),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
            ]))
            elements.append(dt_table)
            elements.append(Paragraph(
                f'{_("downtime_cost")}: ${oee.get("downtime_cost", 0):,.0f}', body_style))

    # ---------------- Financial impact ----------------
    if roi and roi.get("totals"):
        tt = roi["totals"]
        elements.append(Paragraph(_("financial_title"), h2_style))
        roi_rows = [
            [_("potential_loss"), f"${tt.get('potential_loss', 0):,.0f}"],
            [_("loss_avoided"), f"${tt.get('saved', 0):,.0f}"],
            [_("wasted_energy_month"), f"${tt.get('wasted_energy_cost_month', 0):,.0f}"],
            [_("efficiency_gain"), f"+{tt.get('efficiency_gain_pct', 0)}%"],
        ]
        roi_table = Table(roi_rows, colWidths=[95 * mm, 70 * mm])
        roi_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#475569")),
            ("FONTNAME", (1, 0), (1, -1), font_bold),
            ("TEXTCOLOR", (1, 1), (1, 1), colors.HexColor("#047857")),
            ("TEXTCOLOR", (1, 0), (1, 0), colors.HexColor("#b91c1c")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#e2e8f0")),
        ]))
        elements.append(roi_table)
        a = roi.get("assumptions", {})
        if a:
            elements.append(Paragraph(
                _("assumptions",
                  downtime=f"${a.get('downtime_cost_per_hour', 0):,.0f}",
                  hours=a.get('repair_hours', 0),
                  price=f"${a.get('energy_price_per_kwh', 0)}"), small_style))

    # ---------------- Predictive maintenance findings ----------------
    if predictions:
        elements.append(Paragraph(_("predictive_title"), h2_style))
        pred_rows = [[_("col_machine"), _("col_risk"), _("col_remaining_life"), _("col_root_cause")]]
        for p in predictions[:15]:
            rul = p["rul_hours"]
            rul_txt = _("healthy") if rul is None else (f"{rul}{_('h_short')}" if rul < 48 else f"{round(rul / 24, 1)}{_('d_short')}")
            pred_rows.append([
                p["name"], f"{p['risk']}%", rul_txt,
                _("cause_" + p["cause"]) if p["cause"] else "-",
            ])
        pred_table = Table(pred_rows, colWidths=[45 * mm, 22 * mm, 33 * mm, 65 * mm])
        pred_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), font),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (-1, 0), font_bold),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f3ff")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ]))
        elements.append(pred_table)

    # ---------------- Maintenance work orders ----------------
    if work_orders:
        elements.append(Paragraph(_("workorders_title"), h2_style))
        counts = {}
        for w in work_orders:
            counts[w.status] = counts.get(w.status, 0) + 1
        summary = ", ".join(f'{_("wo_" + k)}: {v}' for k, v in sorted(counts.items()))
        elements.append(Paragraph(summary or _("none"), body_style))
        elements.append(Spacer(1, 2 * mm))

        wo_rows = [[_("col_task"), _("col_priority"), _("col_status"), _("col_assigned")]]
        for w in work_orders[:15]:
            wo_rows.append([
                Paragraph(w.title or "-", small_style),
                _("prio_" + w.priority) if w.priority else "-",
                _("wo_" + w.status) if w.status else "-",
                w.assigned_to or "-",
            ])
        wo_table = Table(wo_rows, colWidths=[70 * mm, 25 * mm, 30 * mm, 40 * mm])
        wo_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0891b2")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), font),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (-1, 0), font_bold),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecfeff")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ]))
        elements.append(wo_table)

    elements.append(Paragraph(_("alerts_title"), h2_style))
    if critical_alerts:
        alert_rows = [[_("col_machine"), _("col_message"), _("col_date")]]
        for a in critical_alerts[:20]:
            alert_rows.append([
                a.machine_name or a.machine_code,
                Paragraph(a.message, small_style),
                a.created_at.strftime("%Y-%m-%d %H:%M") if a.created_at else "",
            ])
        alert_table = Table(alert_rows, colWidths=[30 * mm, 100 * mm, 30 * mm])
        alert_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dc2626")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), font),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (-1, 0), font_bold),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fef2f2")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ]))
        elements.append(alert_table)
    else:
        elements.append(Paragraph(_("no_alerts"), body_style))

    elements.append(Spacer(1, 10 * mm))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#e2e8f0")))
    elements.append(Spacer(1, 3 * mm))
    elements.append(Paragraph(
_("footer"), small_style,
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer


@app.route("/api/report/pdf", methods=["GET"])
@require_auth
def api_report_pdf():
    if not REPORTLAB_AVAILABLE:
        return jsonify({"error": "pdf_unavailable", "message": "reportlab is not installed on the server"}), 503

    days = max(1, min(365, int(request.args.get("days", 30))))
    since = datetime.datetime.utcnow() - datetime.timedelta(days=days)

    factories = g.db.query(Factory).filter_by(user_id=g.user.id).all()
    machines = g.db.query(Machine).filter_by(user_id=g.user.id).all()
    alerts = (
        g.db.query(Alert)
        .filter(Alert.user_id == g.user.id, Alert.created_at >= since)
        .order_by(Alert.created_at.desc())
        .all()
    )

    # --- OEE over the same window ---
    shifts = (
        g.db.query(ProductionShift)
        .filter(ProductionShift.user_id == g.user.id, ProductionShift.shift_date >= since)
        .order_by(ProductionShift.shift_date.desc())
        .all()
    )
    reasons = {}
    for s in shifts:
        key = s.downtime_reason or "unspecified"
        reasons[key] = reasons.get(key, 0) + (s.downtime_minutes or 0)
    total_downtime = sum(s.downtime_minutes or 0 for s in shifts)
    oee_payload = {
        "shift_count": len(shifts),
        "overall": aggregate_oee(shifts),
        "downtime_by_reason": sorted(
            ({"reason": k, "minutes": round(v, 1)} for k, v in reasons.items()),
            key=lambda r: -r["minutes"],
        ),
        "downtime_cost": round(total_downtime / 60.0 * DEFAULT_DOWNTIME_COST_PER_HOUR, 2),
    }

    # --- Money + AI findings, taken from live readings ---
    roi_totals = {"potential_loss": 0.0, "saved": 0.0, "wasted_day": 0.0}
    predictions = []
    for m in machines:
        try:
            reading = get_live_reading(m)
        except Exception:
            continue
        biz = reading.get("business", {})
        roi_totals["potential_loss"] += biz.get("potential_loss", 0)
        roi_totals["saved"] += biz.get("saved", 0)
        roi_totals["wasted_day"] += biz.get("wasted_energy_cost_day", 0)
        predictions.append({
            "name": f"{m.machine_name} ({m.machine_code})",
            "risk": reading.get("risk", 0),
            "rul_hours": (reading.get("rul") or {}).get("rul_hours"),
            "cause": (reading.get("root_causes") or [{}])[0].get("code"),
        })
    predictions.sort(key=lambda p: -p["risk"])

    roi_payload = {
        "totals": {
            "potential_loss": round(roi_totals["potential_loss"], 2),
            "saved": round(roi_totals["saved"], 2),
            "wasted_energy_cost_month": round(roi_totals["wasted_day"] * 30, 2),
            "efficiency_gain_pct": round(
                min(35.0, roi_totals["saved"] / max(1.0, roi_totals["potential_loss"]) * 25), 1),
        },
        "assumptions": {
            "downtime_cost_per_hour": DEFAULT_DOWNTIME_COST_PER_HOUR,
            "repair_hours": DEFAULT_REPAIR_HOURS,
            "energy_price_per_kwh": DEFAULT_ENERGY_PRICE,
        },
    }

    work_orders = (
        g.db.query(WorkOrder)
        .filter(WorkOrder.user_id == g.user.id, WorkOrder.created_at >= since)
        .order_by(WorkOrder.created_at.desc())
        .all()
    )

    # Fall back to English if this host has no font that can draw the requested
    # script - a readable report beats a correctly-translated blank one.
    lang = pdf_lang_or_fallback(str(request.args.get("lang", "en")))

    buffer = _build_pdf_report(
        g.user, factories, machines, alerts, days,
        oee=oee_payload, shifts=shifts, roi=roi_payload,
        work_orders=work_orders, predictions=predictions,
        lang=lang,
    )
    filename = f"FactoryPulseAI_Report_{lang}_{datetime.datetime.utcnow().strftime('%Y%m%d')}.pdf"
    # Log the language actually used, so a mismatch between what the user sees
    # on screen and what lands in the PDF is diagnosable from the server output.
    print(f"PDF report generated for {g.user.email}: requested lang="
          f"{request.args.get('lang', 'en')}, used lang={lang}")

    response = send_file(buffer, mimetype="application/pdf",
                         as_attachment=True, download_name=filename)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Report-Language"] = lang
    return response


@app.route("/api/ai/analyze", methods=["POST"])
@require_auth
def api_ai_analyze():
    data = request.get_json(force=True, silent=True) or {}
    reading = standard_reading(
        data.get("machineId", "M-00"),
        float(data.get("temperature", 0)), float(data.get("vibration", 0)), float(data.get("load", 0)),
        float(data.get("pressure", 0)), float(data.get("voltage", 0)), float(data.get("current", 0)),
        data.get("status", "running"),
    )
    result = analyze_machine_reading(reading, str(data.get("lang", "en")))
    return jsonify(result)


# ------------------------------------------------------------------
# LEGACY PUBLIC DEMO ROUTES (unchanged - kept for backward compatibility)
# ------------------------------------------------------------------
@app.route("/api/data", methods=["GET"])
@require_auth
def api_data():
    state = dashboard_state_to_dict(get_dashboard_state(g.db, g.user.id))
    machines = generate_machines(state)
    kpis = compute_kpis(machines, state)
    return jsonify({
        "factory_name": state["factory_name"],
        "state": state,
        "kpis": kpis,
        "machines": machines,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "gemini_enabled": GEMINI_ENABLED,
    })


@app.route("/api/factory", methods=["POST"])
@require_auth
def api_factory():
    """Saves this user's dashboard factory input so it survives logout and restarts."""
    data = request.get_json(force=True, silent=True) or {}
    row = get_dashboard_state(g.db, g.user.id)
    try:
        row.factory_name = str(data.get("factory_name", "")).strip() or "Demo Factory"
        row.machine_count = max(1, min(30, int(data.get("machine_count", 6))))
        row.energy_cost = max(0.01, float(data.get("energy_cost", 0.12)))
        row.machine_type = str(data.get("machine_type", "CNC")).strip() or "CNC"
        row.temperature = float(data.get("temperature", 65))
        row.vibration = float(data.get("vibration", 3.5))
        row.load = float(data.get("load", 60))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid input"}), 400

    g.db.commit()

    state = dashboard_state_to_dict(row)
    machines = generate_machines(state)
    kpis = compute_kpis(machines, state)
    return jsonify({
        "success": True,
        "factory_name": state["factory_name"],
        "state": state,
        "kpis": kpis,
        "machines": machines,
    })


@app.route("/api/analyze", methods=["POST"])
@require_auth
def api_analyze():
    data = request.get_json(force=True, silent=True) or {}
    lang = str(data.get("lang", "en"))
    state = dashboard_state_to_dict(get_dashboard_state(g.db, g.user.id))
    machines = generate_machines(state)
    kpis = compute_kpis(machines, state)
    analysis = analyze_factory(state, kpis, machines, lang)
    return jsonify({"success": True, "analysis": analysis, "kpis": kpis})


@app.route("/api/meta", methods=["GET"])
def api_meta():
    return jsonify({"machine_types": MACHINE_TYPES})





INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Dashboard - FactoryPulse AI | Industrial AI SCADA Platform</title>
<meta name="robots" content="noindex, nofollow" />
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  html, body { height: 100%; }
  body {
    min-height: 100vh;
    background: radial-gradient(circle at 15% 0%, #1a1f3a 0%, #0a0e1f 45%, #030409 100%);
    font-family: 'Segoe UI', system-ui, sans-serif;
    color: #e6edf5;
    position: relative;
    overflow-x: hidden;
  }
  .blob { position: fixed; border-radius: 9999px; filter: blur(100px); opacity: .25; pointer-events: none; animation: floatBlob 16s ease-in-out infinite; z-index: 0; }
  @keyframes floatBlob { 0%,100% { transform: translate(0,0); } 50% { transform: translate(40px,-30px); } }
  .glass { background: rgba(255,255,255,0.045); border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); position: relative; z-index: 1; }
  .glass-strong { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.14); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); }
  .fade-in { animation: fadeIn .5s ease both; }
  @keyframes fadeIn { from { opacity:0; transform: translateY(10px);} to { opacity:1; transform:none; } }
  .glow-btn { background: linear-gradient(135deg, #06b6d4, #7c3aed, #3b82f6); background-size: 200% 200%; transition: box-shadow .25s ease, transform .15s ease; animation: gradientShift 6s ease infinite; }
  @keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
  .glow-btn:hover { box-shadow: 0 0 32px rgba(34,211,238,.5); transform: translateY(-1px); }
  .glow-btn:active { transform: translateY(0) scale(.98); }
  .neon-text { text-shadow: 0 0 18px rgba(34,211,238,.5); }
  .input-field { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12); transition: border-color .2s ease, box-shadow .2s ease, background .2s ease; }
  .input-field:focus { outline: none; border-color: #22d3ee; background: rgba(255,255,255,0.08); box-shadow: 0 0 0 4px rgba(34,211,238,.15), 0 0 24px rgba(34,211,238,.2); }
  .status-running { color: #34d399; }
  .status-warning { color: #fbbf24; }
  .status-critical { color: #f87171; }
  .bg-status-running { background: #34d399; }
  .bg-status-warning { background: #fbbf24; }
  .bg-status-critical { background: #f87171; }
  .pulse-dot { position: relative; display: inline-flex; }
  .pulse-dot::before { content: ""; position: absolute; inset: -5px; border-radius: 9999px; border: 1px solid currentColor; opacity: .5; animation: pulseRing 2s ease-out infinite; }
  @keyframes pulseRing { 0%{transform:scale(.7); opacity:.6} 100%{transform:scale(2); opacity:0} }
  .spinner { width: 18px; height: 18px; border-radius: 50%; border: 2.5px solid rgba(255,255,255,0.25); border-top-color: #22d3ee; animation: spin .7s linear infinite; display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .machine-card { transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease; }
  .machine-card:hover { transform: translateY(-2px); border-color: rgba(34,211,238,.35); box-shadow: 0 12px 30px rgba(0,0,0,.4); }
  ::-webkit-scrollbar { width: 8px; }
  ::-webkit-scrollbar-thumb { background: rgba(148,163,184,.4); border-radius: 8px; }
  select { color: #e2e8f0; color-scheme: dark; }
  select option { color: #0f172a; background: #ffffff; }
  .gauge-value { font-family: 'Consolas', monospace; }
  .fl-label { transition: all .18s ease; }
  .toast { animation: toastIn .25s ease both; }
  @keyframes toastIn { from { opacity:0; transform: translateY(-8px);} to { opacity:1; transform:none; } }
  .ai-section-title { letter-spacing: .04em; }
  .nav-btn.active, .nav-btn-m.active { background: rgba(34,211,238,0.16); color: #22d3ee; }
  .factory-card { transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease; }
  .factory-card:hover { transform: translateY(-2px); border-color: rgba(34,211,238,.35); box-shadow: 0 12px 30px rgba(0,0,0,.4); }
  .anim-error { animation: shakeIn .35s ease both; }
  @keyframes shakeIn { 0%{opacity:0; transform:translateX(-6px);} 60%{transform:translateX(3px);} 100%{opacity:1; transform:none;} }
</style>
</head>
<body>

<div class="blob" style="width:420px;height:420px;background:#0891b2;top:-10%;left:5%"></div>
<div class="blob" style="width:380px;height:380px;background:#7c3aed;bottom:-14%;right:0%"></div>
<div class="blob" style="width:320px;height:320px;background:#3b82f6;top:40%;left:60%"></div>

<div id="toast-container" class="fixed top-4 right-4 z-50 flex flex-col gap-2"></div>

<div class="relative z-10 max-w-7xl mx-auto px-4 md:px-8 py-6 flex gap-4">

  <!-- SIDEBAR -->
  <aside class="w-56 hidden md:flex flex-col glass rounded-2xl p-4 h-fit sticky top-6 fade-in shrink-0">
    <nav class="flex flex-col gap-1">
      <button class="nav-btn active text-left px-3 py-2.5 rounded-xl hover:bg-white/10 transition flex items-center gap-2.5 text-sm" data-page="dashboard">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/></svg>
        <span data-t="nav_dashboard">Dashboard</span>
      </button>
      <button class="nav-btn text-left px-3 py-2.5 rounded-xl hover:bg-white/10 transition flex items-center gap-2.5 text-sm" data-page="factories">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 20h20M4 20V10l6 4v-4l6 4V6l4 3v11"/></svg>
        <span data-t="nav_factories">Factories</span>
      </button>
      <button class="nav-btn text-left px-3 py-2.5 rounded-xl hover:bg-white/10 transition flex items-center gap-2.5 text-sm" data-page="live">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
        <span data-t="nav_live_monitor">Live Monitor</span>
      </button>
      <button class="nav-btn text-left px-3 py-2.5 rounded-xl hover:bg-white/10 transition flex items-center gap-2.5 text-sm" data-page="twin">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.27 6.96 8.73 5.05 8.73-5.05M12 22.08V12"/></svg>
        <span data-t="nav_digital_twin">Digital Twin</span>
      </button>
      <button class="nav-btn text-left px-3 py-2.5 rounded-xl hover:bg-white/10 transition flex items-center gap-2.5 text-sm" data-page="system" data-role="technical">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><circle cx="5" cy="5" r="2"/><circle cx="19" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/><path d="m6.5 6.5 3.2 3.2M17.5 6.5l-3.2 3.2M6.5 17.5l3.2-3.2M17.5 17.5l-3.2-3.2"/></svg>
        <span data-t="nav_system_intel">System Intelligence</span>
      </button>
      <button class="nav-btn text-left px-3 py-2.5 rounded-xl hover:bg-white/10 transition flex items-center gap-2.5 text-sm" data-page="roi" data-role="business">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
        <span data-t="nav_roi">ROI Dashboard</span>
      </button>
      <button class="nav-btn text-left px-3 py-2.5 rounded-xl hover:bg-white/10 transition flex items-center gap-2.5 text-sm" data-page="history">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/><path d="M12 7v5l4 2"/></svg>
        <span data-t="nav_history">History</span>
      </button>
      <button class="nav-btn text-left px-3 py-2.5 rounded-xl hover:bg-white/10 transition flex items-center gap-2.5 text-sm" data-page="oee">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 16v-5M12 16V8M17 16v-9"/></svg>
        <span data-t="nav_oee">OEE</span>
      </button>
      <button class="nav-btn text-left px-3 py-2.5 rounded-xl hover:bg-white/10 transition flex items-center gap-2.5 text-sm relative" data-page="workorders">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76Z"/></svg>
        <span data-t="nav_workorders">Work Orders</span>
        <span id="wo-badge" class="hidden ml-auto text-xs font-bold bg-amber-500 text-white rounded-full px-1.5 py-0.5 min-w-[18px] text-center">0</span>
      </button>
      <button class="nav-btn text-left px-3 py-2.5 rounded-xl hover:bg-white/10 transition flex items-center gap-2.5 text-sm" data-page="story">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3 1.9 4.6L18 9l-4.1 1.4L12 15l-1.9-4.6L6 9l4.1-1.4Z"/><path d="M5 19l.8-2 2-.8-2-.8L5 13l-.8 2-2 .8 2 .8Z"/></svg>
        <span data-t="nav_story">Story Mode</span>
      </button>
      <button class="nav-btn text-left px-3 py-2.5 rounded-xl hover:bg-white/10 transition flex items-center gap-2.5 text-sm relative" data-page="alerts">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4M12 17h.01M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"/></svg>
        <span data-t="nav_alerts">Alerts</span>
        <span id="alerts-badge" class="hidden ml-auto text-xs font-bold bg-red-500 text-white rounded-full px-1.5 py-0.5 min-w-[18px] text-center">0</span>
      </button>
      <button class="nav-btn text-left px-3 py-2.5 rounded-xl hover:bg-white/10 transition flex items-center gap-2.5 text-sm" data-page="ai">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3 1.9 4.6L18 9l-4.1 1.4L12 15l-1.9-4.6L6 9l4.1-1.4Z"/><path d="M5 19l.8-2 2-.8-2-.8L5 13l-.8 2-2 .8 2 .8Z"/></svg>
        <span data-t="nav_ai_insights">AI Insights</span>
      </button>
    </nav>
  </aside>

  <!-- MAIN -->
  <div class="flex-1 min-w-0">

  <!-- HEADER -->
  <header class="flex items-center justify-between mb-8 flex-wrap gap-4 fade-in">
    <div class="flex items-center gap-3">
      <div class="w-11 h-11 rounded-2xl flex items-center justify-center glow-btn shrink-0">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2-7 4 14 2-7h6"/></svg>
      </div>
      <div>
        <div class="font-bold text-xl tracking-tight neon-text">FactoryPulse<span class="text-cyan-400">AI</span> <span class="text-xs align-top px-1.5 py-0.5 rounded-full" style="background:#f59e0b; color:#111827;">v3-SCADA</span></div>
        <div class="text-xs text-slate-400" data-t="tagline">Global Industrial Intelligence Platform</div>
      </div>
    </div>
    <div class="flex items-center gap-3">
      <div class="flex items-center gap-2 text-xs text-slate-400 glass rounded-full px-3 py-1.5">
        <span class="w-2 h-2 rounded-full bg-emerald-400 pulse-dot"></span>
        <span data-t="live_label">Live</span>
        <span id="last-updated" class="gauge-value"></span>
      </div>
      <select id="lang-select" class="input-field rounded-xl text-sm px-3 py-2 outline-none"></select>
      <button id="btn-download-report" class="input-field rounded-xl px-3 py-2 text-xs text-slate-300 hover:text-white flex items-center gap-1.5">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12m0 0-4-4m4 4 4-4M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>
        <span data-t="download_report_btn">Report</span>
      </button>
      <div class="hidden sm:flex items-center gap-2 glass rounded-full pl-1 pr-3 py-1">
        <div class="w-7 h-7 rounded-full glow-btn flex items-center justify-center text-xs font-bold" id="user-avatar">?</div>
        <span id="user-name" class="text-sm font-medium">—</span>
        <span id="user-role-badge" class="text-xs px-1.5 py-0.5 rounded-full" style="background:rgba(34,211,238,.15); color:#22d3ee;"></span>
      </div>
      <button id="btn-logout" class="input-field rounded-xl px-3 py-2 text-xs text-slate-300 hover:text-white" data-t="logout_btn">Log Out</button>
    </div>
  </header>

  <!-- MOBILE NAV -->
  <nav class="md:hidden flex gap-2 mb-6 overflow-x-auto">
    <button class="nav-btn-m active glass rounded-xl px-3 py-2 text-xs whitespace-nowrap" data-page="dashboard" data-t="nav_dashboard">Dashboard</button>
    <button class="nav-btn-m glass rounded-xl px-3 py-2 text-xs whitespace-nowrap" data-page="factories" data-t="nav_factories">Factories</button>
    <button class="nav-btn-m glass rounded-xl px-3 py-2 text-xs whitespace-nowrap" data-page="live" data-t="nav_live_monitor">Live Monitor</button>
    <button class="nav-btn-m glass rounded-xl px-3 py-2 text-xs whitespace-nowrap" data-page="twin" data-t="nav_digital_twin">Digital Twin</button>
    <button class="nav-btn-m glass rounded-xl px-3 py-2 text-xs whitespace-nowrap" data-page="system" data-role="technical" data-t="nav_system_intel">System Intelligence</button>
    <button class="nav-btn-m glass rounded-xl px-3 py-2 text-xs whitespace-nowrap" data-page="roi" data-role="business" data-t="nav_roi">ROI Dashboard</button>
    <button class="nav-btn-m glass rounded-xl px-3 py-2 text-xs whitespace-nowrap" data-page="history" data-t="nav_history">History</button>
    <button class="nav-btn-m glass rounded-xl px-3 py-2 text-xs whitespace-nowrap" data-page="oee" data-t="nav_oee">OEE</button>
    <button class="nav-btn-m glass rounded-xl px-3 py-2 text-xs whitespace-nowrap" data-page="workorders" data-t="nav_workorders">Work Orders</button>
    <button class="nav-btn-m glass rounded-xl px-3 py-2 text-xs whitespace-nowrap" data-page="story" data-t="nav_story">Story Mode</button>
    <button class="nav-btn-m glass rounded-xl px-3 py-2 text-xs whitespace-nowrap relative" data-page="alerts" data-t="nav_alerts">Alerts</button>
    <button class="nav-btn-m glass rounded-xl px-3 py-2 text-xs whitespace-nowrap" data-page="ai" data-t="nav_ai_insights">AI Insights</button>
  </nav>

  <!-- ==================== PAGE: DASHBOARD ==================== -->
  <section id="page-dashboard" class="page">
  <!-- KPI CARDS -->
  <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6 fade-in">
    <div class="glass rounded-2xl p-5">
      <div class="flex items-center justify-between mb-2">
        <div class="text-xs text-slate-400" data-t="kpi_energy">Energy Usage</div>
        <div class="w-8 h-8 rounded-lg flex items-center justify-center" style="background:rgba(34,211,238,.15); color:#22d3ee;">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2 3 14h7l-1 8 10-12h-7l1-8Z"/></svg>
        </div>
      </div>
      <div id="kpi-energy" class="text-3xl font-bold gauge-value">0 <span class="text-sm font-normal" data-t="kwh_unit">kWh</span></div>
    </div>
    <div class="glass rounded-2xl p-5">
      <div class="flex items-center justify-between mb-2">
        <div class="text-xs text-slate-400" data-t="kpi_efficiency">Efficiency</div>
        <div class="w-8 h-8 rounded-lg flex items-center justify-center" style="background:rgba(167,139,250,.15); color:#a78bfa;">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18M18.7 8 12 14.7 8.5 11.2 3.7 16"/></svg>
        </div>
      </div>
      <div id="kpi-efficiency" class="text-3xl font-bold gauge-value">0%</div>
    </div>
    <div class="glass rounded-2xl p-5">
      <div class="flex items-center justify-between mb-2">
        <div class="text-xs text-slate-400" data-t="kpi_active">Active Machines</div>
        <div class="w-8 h-8 rounded-lg flex items-center justify-center" style="background:rgba(52,211,153,.15); color:#34d399;">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="6" width="12" height="12" rx="1"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/></svg>
        </div>
      </div>
      <div id="kpi-active" class="text-3xl font-bold gauge-value">0/0</div>
    </div>
    <div class="glass rounded-2xl p-5">
      <div class="flex items-center justify-between mb-2">
        <div class="text-xs text-slate-400" data-t="kpi_alerts">Alerts</div>
        <div class="w-8 h-8 rounded-lg flex items-center justify-center" style="background:rgba(248,113,113,.15); color:#f87171;">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4M12 17h.01M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"/></svg>
        </div>
      </div>
      <div id="kpi-alerts" class="text-3xl font-bold status-critical gauge-value">0</div>
    </div>
  </div>

  <div class="grid lg:grid-cols-3 gap-4 mb-6">

    <!-- LEFT: CHART + MACHINE STATUS -->
    <div class="lg:col-span-2 flex flex-col gap-4">
      <div class="glass rounded-2xl p-5 fade-in">
        <h3 class="font-semibold mb-3" data-t="chart_title">Real-Time Performance</h3>
        <canvas id="liveChart" height="110"></canvas>
      </div>
      <div class="glass rounded-2xl p-5 fade-in">
        <h3 class="font-semibold mb-3" data-t="machine_status_title">Machine Status</h3>
        <div id="machine-list" class="grid sm:grid-cols-2 gap-3"></div>
      </div>
    </div>

    <!-- RIGHT: FACTORY FORM + AI PANEL -->
    <div class="flex flex-col gap-4">
      <div class="glass rounded-2xl p-5 fade-in">
        <h3 class="font-semibold mb-4" data-t="form_title">Factory Data Input</h3>
        <form id="factory-form" class="flex flex-col gap-4">
          <div class="relative">
            <input id="f-name" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2" />
            <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="factory_name_label">Factory Name</label>
          </div>
          <div class="relative">
            <input id="f-count" type="number" min="1" max="30" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2" />
            <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="machine_count_label">Number of Machines</label>
          </div>
          <div class="relative">
            <input id="f-cost" type="number" step="0.01" min="0" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2" />
            <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="energy_cost_label">Energy Cost ($/kWh)</label>
          </div>
          <div>
            <label class="text-xs text-slate-400 mb-1.5 block" data-t="machine_type_label">Machine Type</label>
            <select id="f-type" class="input-field w-full rounded-xl text-sm px-3 py-2.5"></select>
          </div>
          <div class="relative">
            <input id="f-temp" type="number" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2" />
            <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="temperature_label">Temperature (°C)</label>
          </div>
          <div class="relative">
            <input id="f-vibration" type="number" step="0.1" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2" />
            <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="vibration_label">Vibration (mm/s)</label>
          </div>
          <div class="relative">
            <input id="f-load" type="number" min="0" max="100" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2" />
            <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="load_label">Load (%)</label>
          </div>
          <button type="submit" id="submit-btn" class="glow-btn rounded-xl py-3 text-sm font-semibold flex items-center justify-center gap-2 mt-1">
            <span id="submit-spinner" class="spinner hidden"></span>
            <span id="submit-label" data-t="submit_btn">Analyze Factory</span>
          </button>
        </form>
      </div>

      <div class="glass rounded-2xl p-5 fade-in flex-1">
        <h3 class="font-semibold mb-4 flex items-center gap-2">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3 1.9 4.6L18 9l-4.1 1.4L12 15l-1.9-4.6L6 9l4.1-1.4Z"/><path d="M5 19l.8-2 2-.8-2-.8L5 13l-.8 2-2 .8 2 .8Z"/></svg>
          <span data-t="ai_panel_title">AI Insights</span>
        </h3>
        <div id="ai-panel-content">
          <div class="text-sm text-slate-400" data-t="ai_placeholder">Submit factory data to generate an AI analysis.</div>
        </div>
      </div>
    </div>
  </div>
  </section>

  <!-- ==================== PAGE: FACTORIES ==================== -->
  <section id="page-factories" class="page hidden">
    <div class="flex items-center justify-between mb-5 flex-wrap gap-3">
      <h2 class="text-xl font-bold" data-t="my_factories_title">My Factories</h2>
      <button id="open-factory-modal" class="glow-btn rounded-xl px-4 py-2.5 text-sm font-semibold" data-t="add_factory_btn">+ Add Factory</button>
    </div>
    <div id="factories-grid" class="grid md:grid-cols-2 lg:grid-cols-3 gap-4"></div>
    <div id="factories-empty" class="hidden glass rounded-2xl p-10 text-center text-slate-400" data-t="no_factories_yet">You haven't added any factories yet.</div>
  </section>

  <!-- ==================== PAGE: LIVE MONITOR (SCADA) ==================== -->
  <section id="page-live" class="page hidden">
    <div class="flex items-center justify-between mb-4 flex-wrap gap-3">
      <h2 class="text-xl font-bold" data-t="nav_live_monitor">Live Monitor</h2>
      <button id="open-machine-modal" class="glow-btn rounded-xl px-4 py-2.5 text-sm font-semibold" data-t="add_machine_scada_btn">+ Add Machine</button>
    </div>

    <div class="glass rounded-2xl p-4 mb-5 flex items-center gap-3 flex-wrap text-xs">
      <span class="px-2.5 py-1 rounded-full font-semibold" id="mode-badge" style="background:rgba(34,211,238,.15); color:#22d3ee;">SIMULATION</span>
      <span class="text-slate-400" data-t="usb_status">USB:</span> <span id="usb-status" class="text-slate-300">—</span>
      <span class="text-slate-400" data-t="plc_status">PLC:</span> <span id="plc-status" class="text-slate-300">—</span>
      <span class="text-slate-400">Modbus:</span> <span id="modbus-status" class="text-slate-300">—</span>
      <span class="text-slate-400">OPC UA:</span> <span id="opcua-status" class="text-slate-300">—</span>
      <span class="text-slate-400">MQTT:</span> <span id="mqtt-status" class="text-slate-300">—</span>
      <span class="ml-auto flex items-center gap-1.5 text-slate-400">
        <span id="ws-dot" class="w-2 h-2 rounded-full bg-slate-500"></span>
        <span id="ws-label" data-t="polling_mode">Polling</span>
      </span>
    </div>

    <div class="glass rounded-2xl p-5 mb-5">
      <h3 class="font-semibold mb-3" data-t="live_chart_title">Live Sensor Chart</h3>
      <canvas id="scadaChart" height="90"></canvas>
    </div>

    <div class="glass rounded-2xl p-5">
      <h3 class="font-semibold mb-3" data-t="machines_table_title">Machines</h3>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="text-slate-400 text-left">
            <tr class="border-b border-white/10">
              <th class="pb-2 pr-3" data-t="machine_code_col">Code</th>
              <th class="pb-2 pr-3" data-t="machine_name_col">Name</th>
              <th class="pb-2 pr-3" data-t="status_col">Status</th>
              <th class="pb-2 pr-3">°C</th>
              <th class="pb-2 pr-3">mm/s</th>
              <th class="pb-2 pr-3">%</th>
              <th class="pb-2 pr-3" data-t="risk_col">Risk</th>
              <th class="pb-2 pr-3" data-t="source_col">Source</th>
              <th class="pb-2"></th>
            </tr>
          </thead>
          <tbody id="scada-table-body"></tbody>
        </table>
      </div>
      <div id="scada-empty" class="hidden text-center text-slate-400 py-8 text-sm" data-t="no_machines_yet">No machines yet. Click "+ Add Machine".</div>
    </div>
  </section>

  <!-- ==================== PAGE: 3D DIGITAL TWIN ==================== -->
  <section id="page-twin" class="page hidden">
    <div class="flex items-center justify-between mb-4 flex-wrap gap-3">
      <h2 class="text-xl font-bold" data-t="nav_digital_twin">Digital Twin</h2>
      <div class="flex items-center gap-3 text-xs text-slate-400 flex-wrap">
        <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full" style="background:#34d399;"></span><span data-t="status_running">Running</span></span>
        <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full" style="background:#fbbf24;"></span><span data-t="status_maintenance">Maintenance</span></span>
        <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full" style="background:#f87171;"></span><span data-t="status_stopped">Stopped</span></span>
      </div>
    </div>
    <p class="text-xs text-slate-500 mb-4" data-t="twin_hint">Drag to rotate, scroll to zoom, click a machine to see its live details.</p>
    <div class="glass rounded-2xl p-2 relative" style="height:520px;">
      <div id="twin-canvas-container" style="width:100%; height:100%; border-radius:14px; overflow:hidden;"></div>
      <div id="twin-empty" class="hidden absolute inset-0 flex items-center justify-center text-slate-400 text-sm" data-t="no_machines_yet">No machines yet. Click "+ Add Machine".</div>
      <div id="twin-tooltip" class="hidden absolute glass rounded-xl p-3 text-xs pointer-events-none" style="min-width:160px; z-index:10;"></div>
    </div>

    <!-- WHAT-IF SIMULATION -->
    <div class="glass rounded-2xl p-5 mt-4">
      <h3 class="font-semibold mb-1" data-t="simulation_title">What-If Simulation</h3>
      <p class="text-xs text-slate-500 mb-4" data-t="simulation_hint">Pick a machine, shift its sensor values, and see how failure probability reacts.</p>

      <div class="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
        <div>
          <label class="text-xs text-slate-400 mb-1.5 block" data-t="machines_table_title">Machine</label>
          <select id="sim-machine" class="input-field w-full rounded-xl text-sm px-3 py-2.5"></select>
        </div>
        <div>
          <label class="text-xs text-slate-400 mb-1.5 block"><span data-t="temperature_label">Temperature</span> <span id="sim-temp-val" class="text-cyan-400">+0%</span></label>
          <input id="sim-temp" type="range" min="-30" max="50" value="0" step="5" class="w-full accent-cyan-400" />
        </div>
        <div>
          <label class="text-xs text-slate-400 mb-1.5 block"><span data-t="vibration_label">Vibration</span> <span id="sim-vib-val" class="text-cyan-400">+0%</span></label>
          <input id="sim-vib" type="range" min="-30" max="100" value="0" step="5" class="w-full accent-cyan-400" />
        </div>
        <div>
          <label class="text-xs text-slate-400 mb-1.5 block"><span data-t="load_label">Load</span> <span id="sim-load-val" class="text-cyan-400">+0%</span></label>
          <input id="sim-load" type="range" min="-50" max="40" value="0" step="5" class="w-full accent-cyan-400" />
        </div>
      </div>

      <button id="btn-run-simulation" class="glow-btn rounded-xl px-5 py-2.5 text-sm font-semibold flex items-center gap-2">
        <span id="sim-spinner" class="spinner hidden"></span>
        <span data-t="run_simulation_btn">Run Simulation</span>
      </button>

      <div id="sim-results" class="hidden mt-5 grid sm:grid-cols-2 gap-4"></div>
    </div>
  </section>

  <!-- ==================== PAGE: SYSTEM INTELLIGENCE ==================== -->
  <section id="page-system" class="page hidden">
    <div class="flex items-center justify-between mb-5 flex-wrap gap-3">
      <h2 class="text-xl font-bold" data-t="nav_system_intel">System Intelligence</h2>
      <button id="btn-refresh-system" class="input-field rounded-xl px-4 py-2 text-sm" data-t="refresh_btn">Refresh</button>
    </div>

    <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <div class="glass rounded-2xl p-5">
        <div class="text-xs text-slate-400 mb-2" data-t="system_risk_label">System Risk</div>
        <div id="sys-risk" class="text-3xl font-bold gauge-value">—</div>
      </div>
      <div class="glass rounded-2xl p-5">
        <div class="text-xs text-slate-400 mb-2" data-t="healthy_label">Healthy</div>
        <div id="sys-healthy" class="text-3xl font-bold gauge-value" style="color:#34d399">—</div>
      </div>
      <div class="glass rounded-2xl p-5">
        <div class="text-xs text-slate-400 mb-2" data-t="at_risk_label">At Risk</div>
        <div id="sys-atrisk" class="text-3xl font-bold gauge-value" style="color:#fbbf24">—</div>
      </div>
      <div class="glass rounded-2xl p-5">
        <div class="text-xs text-slate-400 mb-2" data-t="status_critical">Critical</div>
        <div id="sys-critical" class="text-3xl font-bold gauge-value" style="color:#f87171">—</div>
      </div>
    </div>

    <div class="grid lg:grid-cols-2 gap-4">
      <div class="glass rounded-2xl p-5">
        <h3 class="font-semibold mb-3" data-t="clusters_title">Machine Clusters</h3>
        <div id="sys-clusters" class="flex flex-col gap-3"></div>
      </div>
      <div class="glass rounded-2xl p-5">
        <h3 class="font-semibold mb-3" data-t="propagation_title">Anomaly Propagation</h3>
        <p class="text-xs text-slate-500 mb-3" data-t="propagation_hint">How a failing machine raises the effective risk of its neighbours.</p>
        <div id="sys-propagation" class="flex flex-col gap-3"></div>
        <div id="sys-propagation-empty" class="hidden text-sm text-slate-400" data-t="no_propagation">No anomaly propagation detected.</div>
      </div>
    </div>
  </section>

  <!-- ==================== PAGE: ROI DASHBOARD ==================== -->
  <section id="page-roi" class="page hidden">
    <div class="flex items-center justify-between mb-5 flex-wrap gap-3">
      <h2 class="text-xl font-bold" data-t="nav_roi">ROI Dashboard</h2>
      <button id="btn-refresh-roi" class="input-field rounded-xl px-4 py-2 text-sm" data-t="refresh_btn">Refresh</button>
    </div>

    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <div class="glass rounded-2xl p-5" style="border-left:3px solid #f87171">
        <div class="text-xs text-slate-400 mb-2" data-t="potential_loss_label">Potential Loss</div>
        <div id="roi-loss" class="text-2xl font-bold gauge-value" style="color:#f87171">—</div>
      </div>
      <div class="glass rounded-2xl p-5" style="border-left:3px solid #34d399">
        <div class="text-xs text-slate-400 mb-2" data-t="saved_label">Saved by AI</div>
        <div id="roi-saved" class="text-2xl font-bold gauge-value" style="color:#34d399">—</div>
      </div>
      <div class="glass rounded-2xl p-5" style="border-left:3px solid #fbbf24">
        <div class="text-xs text-slate-400 mb-2" data-t="wasted_energy_label">Wasted Energy / month</div>
        <div id="roi-energy" class="text-2xl font-bold gauge-value" style="color:#fbbf24">—</div>
      </div>
      <div class="glass rounded-2xl p-5" style="border-left:3px solid #22d3ee">
        <div class="text-xs text-slate-400 mb-2" data-t="efficiency_gain_label">Efficiency Gain</div>
        <div id="roi-efficiency" class="text-2xl font-bold gauge-value" style="color:#22d3ee">—</div>
      </div>
    </div>

    <div class="glass rounded-2xl p-5">
      <h3 class="font-semibold mb-3" data-t="cost_by_machine_title">Cost Exposure by Machine</h3>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="text-slate-400 text-left">
            <tr class="border-b border-white/10">
              <th class="pb-2 pr-3" data-t="machine_name_col">Name</th>
              <th class="pb-2 pr-3" data-t="risk_col">Risk</th>
              <th class="pb-2 pr-3" data-t="rul_col">RUL</th>
              <th class="pb-2 pr-3" data-t="potential_loss_label">Potential Loss</th>
              <th class="pb-2 pr-3" data-t="saved_label">Saved</th>
              <th class="pb-2" data-t="top_cause_col">Top Cause</th>
            </tr>
          </thead>
          <tbody id="roi-table-body"></tbody>
        </table>
      </div>
      <div id="roi-empty" class="hidden text-center text-slate-400 py-8 text-sm" data-t="no_machines_yet">No machines yet.</div>
      <p id="roi-assumptions" class="text-xs text-slate-500 mt-4"></p>
    </div>
  </section>

  <!-- ==================== PAGE: HISTORY / TRENDS ==================== -->
  <section id="page-history" class="page hidden">
    <div class="flex items-center justify-between mb-2 flex-wrap gap-3">
      <h2 class="text-xl font-bold" data-t="nav_history">History</h2>
      <div class="flex gap-2 flex-wrap">
        <select id="hist-machine" class="input-field rounded-xl text-sm px-3 py-2"></select>
        <select id="hist-hours" class="input-field rounded-xl text-sm px-3 py-2">
          <option value="24" data-t="range_24h">24 hours</option>
          <option value="72" data-t="range_3d">3 days</option>
          <option value="240" selected data-t="range_10d">10 days</option>
          <option value="720" data-t="range_30d">30 days</option>
        </select>
      </div>
    </div>
    <p class="text-xs text-slate-500 mb-5" data-t="history_hint">Stored sensor history - this is how you prove what actually changed over the pilot.</p>

    <!-- BEFORE / AFTER SUMMARY -->
    <div id="hist-summary" class="hidden grid grid-cols-2 lg:grid-cols-4 gap-4 mb-5"></div>

    <div class="glass rounded-2xl p-5 mb-5">
      <h3 class="font-semibold mb-3" data-t="sensor_trend_title">Sensor Trend</h3>
      <div style="height:280px"><canvas id="hist-chart"></canvas></div>
    </div>

    <div class="glass rounded-2xl p-5">
      <h3 class="font-semibold mb-3" data-t="risk_trend_title">Risk Trend</h3>
      <div style="height:200px"><canvas id="hist-risk-chart"></canvas></div>
    </div>

    <div id="hist-empty" class="hidden glass rounded-2xl p-10 text-center text-slate-400 text-sm" data-t="no_history_yet">No history recorded yet. Keep the Live Monitor open for a few minutes and data will start accumulating.</div>
  </section>

  <!-- ==================== PAGE: OEE ==================== -->
  <section id="page-oee" class="page hidden">
    <div class="flex items-center justify-between mb-2 flex-wrap gap-3">
      <h2 class="text-xl font-bold" data-t="nav_oee">OEE</h2>
      <div class="flex gap-2">
        <select id="oee-days" class="input-field rounded-xl text-sm px-3 py-2">
          <option value="1" data-t="range_1d">Today</option>
          <option value="7" selected data-t="range_7d">7 days</option>
          <option value="30" data-t="range_30d">30 days</option>
        </select>
        <button id="btn-log-shift" class="glow-btn rounded-xl px-4 py-2 text-sm font-semibold" data-t="log_shift_btn">Log Shift</button>
      </div>
    </div>
    <p class="text-xs text-slate-500 mb-5" data-t="oee_hint">Availability x Performance x Quality (ISO 22400). World-class is 85%; a typical factory sits near 60%.</p>

    <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-5">
      <div class="glass rounded-2xl p-5" style="border-left:3px solid #22d3ee">
        <div class="text-xs text-slate-400 mb-2" data-t="nav_oee">OEE</div>
        <div id="oee-total" class="text-3xl font-bold gauge-value">-</div>
        <div id="oee-grade" class="text-xs mt-1"></div>
      </div>
      <div class="glass rounded-2xl p-5">
        <div class="text-xs text-slate-400 mb-2" data-t="availability_label">Availability</div>
        <div id="oee-avail" class="text-2xl font-bold gauge-value">-</div>
      </div>
      <div class="glass rounded-2xl p-5">
        <div class="text-xs text-slate-400 mb-2" data-t="performance_label">Performance</div>
        <div id="oee-perf" class="text-2xl font-bold gauge-value">-</div>
      </div>
      <div class="glass rounded-2xl p-5">
        <div class="text-xs text-slate-400 mb-2" data-t="quality_label">Quality</div>
        <div id="oee-qual" class="text-2xl font-bold gauge-value">-</div>
      </div>
    </div>

    <div class="grid lg:grid-cols-2 gap-4 mb-5">
      <div class="glass rounded-2xl p-5">
        <h3 class="font-semibold mb-3" data-t="downtime_by_reason_title">Downtime by Reason</h3>
        <div id="oee-reasons" class="flex flex-col gap-2"></div>
        <div id="oee-downtime-cost" class="text-sm mt-3"></div>
      </div>
      <div class="glass rounded-2xl p-5">
        <h3 class="font-semibold mb-3" data-t="oee_trend_title">OEE Trend</h3>
        <div style="height:200px"><canvas id="oee-chart"></canvas></div>
      </div>
    </div>

    <div class="glass rounded-2xl p-5">
      <h3 class="font-semibold mb-3" data-t="shifts_title">Shifts</h3>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="text-slate-400 text-left">
            <tr class="border-b border-white/10">
              <th class="pb-2 pr-3" data-t="shift_col">Shift</th>
              <th class="pb-2 pr-3" data-t="nav_oee">OEE</th>
              <th class="pb-2 pr-3" data-t="availability_label">A</th>
              <th class="pb-2 pr-3" data-t="performance_label">P</th>
              <th class="pb-2 pr-3" data-t="quality_label">Q</th>
              <th class="pb-2 pr-3" data-t="downtime_col">Downtime</th>
              <th class="pb-2" data-t="weakest_factor_label">Weakest</th>
            </tr>
          </thead>
          <tbody id="oee-table-body"></tbody>
        </table>
      </div>
      <div id="oee-empty" class="hidden text-center text-slate-400 py-8 text-sm" data-t="no_shifts_yet">No shifts logged yet. Click "Log Shift" to add one.</div>
    </div>
  </section>

  <!-- ==================== PAGE: WORK ORDERS ==================== -->
  <section id="page-workorders" class="page hidden">
    <div class="flex items-center justify-between mb-2 flex-wrap gap-3">
      <h2 class="text-xl font-bold" data-t="nav_workorders">Work Orders</h2>
      <button id="btn-new-wo" class="glow-btn rounded-xl px-4 py-2 text-sm font-semibold" data-t="new_work_order_btn">+ New Work Order</button>
    </div>
    <p class="text-xs text-slate-500 mb-5" data-t="workorders_hint">Turns an AI prediction into tracked, assignable work - so a warning actually ends in a repair.</p>

    <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-5">
      <div class="glass rounded-2xl p-4">
        <div class="text-xs text-slate-400 mb-1" data-t="wo_status_open">Open</div>
        <div id="wo-open" class="text-2xl font-bold gauge-value" style="color:#fbbf24">-</div>
      </div>
      <div class="glass rounded-2xl p-4">
        <div class="text-xs text-slate-400 mb-1" data-t="wo_status_in_progress">In Progress</div>
        <div id="wo-progress" class="text-2xl font-bold gauge-value" style="color:#22d3ee">-</div>
      </div>
      <div class="glass rounded-2xl p-4">
        <div class="text-xs text-slate-400 mb-1" data-t="wo_status_done">Done</div>
        <div id="wo-done" class="text-2xl font-bold gauge-value" style="color:#34d399">-</div>
      </div>
      <div class="glass rounded-2xl p-4">
        <div class="text-xs text-slate-400 mb-1" data-t="overdue_label">Overdue</div>
        <div id="wo-avg" class="text-2xl font-bold gauge-value">-</div>
      </div>
    </div>

    <div class="glass rounded-2xl p-5">
      <div id="wo-list" class="flex flex-col gap-3"></div>
      <div id="wo-empty" class="hidden text-center text-slate-400 py-8 text-sm" data-t="no_work_orders">No work orders yet.</div>
    </div>
  </section>

  <!-- LOG SHIFT MODAL -->
  <div id="modal-shift" class="fixed inset-0 bg-black/70 hidden items-center justify-center z-40 p-4">
    <div class="glass-strong rounded-2xl p-6 w-full max-w-md fade-in" style="max-height:90vh; overflow-y:auto;">
      <div class="flex items-center justify-between mb-4">
        <h3 class="font-bold text-lg" data-t="log_shift_btn">Log Shift</h3>
        <button class="close-shift-modal text-slate-400 hover:text-slate-200">&times;</button>
      </div>
      <div class="flex flex-col gap-3">
        <input id="sh-name" placeholder="Shift A" data-t-placeholder="ph_shift_name" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        <select id="sh-machine" class="input-field w-full rounded-xl text-sm px-3 py-2.5"></select>
        <div class="grid grid-cols-2 gap-3">
          <input id="sh-planned" type="number" value="480" placeholder="480" data-t-placeholder="ph_planned_minutes" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
          <input id="sh-downtime" type="number" value="0" placeholder="0" data-t-placeholder="ph_downtime_minutes" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        </div>
        <select id="sh-reason" class="input-field w-full rounded-xl text-sm px-3 py-2.5"></select>
        <div class="grid grid-cols-2 gap-3">
          <input id="sh-total" type="number" value="0" placeholder="0" data-t-placeholder="ph_total_units" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
          <input id="sh-good" type="number" value="0" placeholder="0" data-t-placeholder="ph_good_units" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        </div>
        <input id="sh-cycle" type="number" step="0.1" value="30" placeholder="30" data-t-placeholder="ph_cycle_seconds" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        <button id="btn-save-shift" class="glow-btn rounded-xl py-3 text-sm font-semibold mt-1" data-t="save_btn">Save</button>
      </div>
    </div>
  </div>

  <!-- NEW WORK ORDER MODAL -->
  <div id="modal-wo" class="fixed inset-0 bg-black/70 hidden items-center justify-center z-40 p-4">
    <div class="glass-strong rounded-2xl p-6 w-full max-w-md fade-in" style="max-height:90vh; overflow-y:auto;">
      <div class="flex items-center justify-between mb-4">
        <h3 class="font-bold text-lg" data-t="new_work_order_btn">New Work Order</h3>
        <button class="close-wo-modal text-slate-400 hover:text-slate-200">&times;</button>
      </div>
      <div class="flex flex-col gap-3">
        <input id="wo-title" placeholder="Title" data-t-placeholder="ph_wo_title" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        <select id="wo-machine" class="input-field w-full rounded-xl text-sm px-3 py-2.5"></select>
        <select id="wo-priority" class="input-field w-full rounded-xl text-sm px-3 py-2.5">
          <option value="low" data-t="priority_low">Low</option>
          <option value="medium" selected data-t="priority_medium">Medium</option>
          <option value="high" data-t="priority_high">High</option>
          <option value="critical" data-t="priority_critical">Critical</option>
        </select>
        <input id="wo-assigned" placeholder="Assigned to" data-t-placeholder="ph_assigned_to" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        <textarea id="wo-desc" rows="3" placeholder="Description" data-t-placeholder="ph_wo_description" class="input-field w-full rounded-xl text-sm px-3 py-2.5"></textarea>
        <button id="btn-save-wo" class="glow-btn rounded-xl py-3 text-sm font-semibold mt-1" data-t="save_btn">Save</button>
      </div>
    </div>
  </div>

  <!-- ==================== PAGE: STORY MODE (INVESTOR DEMO) ==================== -->
  <section id="page-story" class="page hidden">
    <div class="flex items-center justify-between mb-2 flex-wrap gap-3">
      <h2 class="text-xl font-bold" data-t="nav_story">Story Mode</h2>
      <select id="story-machine" class="input-field rounded-xl text-sm px-3 py-2"></select>
    </div>
    <p class="text-xs text-slate-500 mb-5" data-t="story_hint">Replays a bearing failure developing over 22 hours and shows exactly when the AI caught it — and what that was worth.</p>

    <button id="btn-run-story" class="glow-btn rounded-xl px-6 py-3 text-sm font-semibold flex items-center gap-2 mb-6">
      <span id="story-spinner" class="spinner hidden"></span>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4M12 17h.01M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"/></svg>
      <span data-t="simulate_failure_btn">Simulate Failure</span>
    </button>

    <div id="story-results" class="hidden">
      <!-- OUTCOME HEADLINE -->
      <div class="glass rounded-2xl p-6 mb-5" style="border:1px solid rgba(52,211,153,.35)">
        <div class="text-xs uppercase text-emerald-400 ai-section-title mb-3" data-t="outcome_title">Outcome</div>
        <div class="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <div class="text-xs text-slate-400 mb-1" data-t="warning_time_label">Early Warning</div>
            <div id="story-warning" class="text-2xl font-bold gauge-value" style="color:#22d3ee">—</div>
          </div>
          <div>
            <div class="text-xs text-slate-400 mb-1" data-t="loss_ignored_label">Loss if Ignored</div>
            <div id="story-loss-ignored" class="text-2xl font-bold gauge-value" style="color:#f87171">—</div>
          </div>
          <div>
            <div class="text-xs text-slate-400 mb-1" data-t="loss_acted_label">Loss if Acted On</div>
            <div id="story-loss-acted" class="text-2xl font-bold gauge-value" style="color:#fbbf24">—</div>
          </div>
          <div>
            <div class="text-xs text-slate-400 mb-1" data-t="money_saved_label">Money Saved</div>
            <div id="story-saved" class="text-2xl font-bold gauge-value" style="color:#34d399">—</div>
          </div>
        </div>
        <div id="story-detection" class="text-sm text-slate-300 mt-4"></div>
      </div>

      <!-- TIMELINE -->
      <div class="glass rounded-2xl p-5">
        <h3 class="font-semibold mb-4" data-t="timeline_title">Failure Timeline</h3>
        <div id="story-timeline" class="flex flex-col gap-3"></div>
      </div>
    </div>

    <div id="story-empty" class="hidden glass rounded-2xl p-10 text-center text-slate-400" data-t="no_machines_yet">No machines yet.</div>
  </section>

  <!-- ==================== PAGE: ALERTS ==================== -->
  <section id="page-alerts" class="page hidden">
    <div class="flex items-center justify-between mb-5 flex-wrap gap-3">
      <h2 class="text-xl font-bold" data-t="nav_alerts">Alerts</h2>
      <button id="btn-ack-all" class="input-field rounded-xl px-4 py-2 text-sm" data-t="acknowledge_all_btn">Acknowledge All</button>
    </div>
    <div id="alerts-list" class="flex flex-col gap-3"></div>
    <div id="alerts-empty" class="hidden glass rounded-2xl p-10 text-center text-slate-400" data-t="no_alerts_yet">No alerts. Everything is running smoothly.</div>
  </section>

  <!-- ==================== PAGE: AI INSIGHTS ==================== -->
  <section id="page-ai" class="page hidden">
    <h2 class="text-xl font-bold mb-5" data-t="ai_insights_feed_title">AI Insights Feed</h2>
    <div id="ai-feed" class="flex flex-col gap-4"></div>
    <div id="ai-feed-empty" class="hidden glass rounded-2xl p-10 text-center text-slate-400" data-t="no_ai_insights_yet">No AI insights yet. Add a factory to get started.</div>
  </section>

  </div>
</div>

<!-- FACTORY MODAL (create / edit) -->
<div id="modal-factory" class="fixed inset-0 bg-black/70 hidden items-center justify-center z-40 p-4">
  <div class="glass-strong rounded-2xl p-6 w-full max-w-md fade-in" style="max-height:90vh; overflow-y:auto;">
    <div class="flex items-center justify-between mb-4">
      <h3 id="factory-modal-title" class="font-bold text-lg" data-t="add_factory_btn">+ Add Factory</h3>
      <button class="close-factory-modal text-slate-400 hover:text-slate-200"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18M6 6l12 12"/></svg></button>
    </div>
    <form id="modal-factory-form" class="flex flex-col gap-4">
      <div class="relative">
        <input id="m-name" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2" />
        <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="factory_name_label">Factory Name</label>
      </div>
      <div class="relative">
        <input id="m-count" type="number" min="1" max="30" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2" />
        <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="machine_count_label">Number of Machines</label>
      </div>
      <div class="relative">
        <input id="m-cost" type="number" step="0.01" min="0" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2" />
        <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="energy_cost_label">Energy Cost ($/kWh)</label>
      </div>
      <div>
        <label class="text-xs text-slate-400 mb-1.5 block" data-t="machine_type_label">Machine Type</label>
        <select id="m-type" class="input-field w-full rounded-xl text-sm px-3 py-2.5"></select>
      </div>
      <div class="relative">
        <input id="m-temp" type="number" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2" />
        <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="temperature_label">Temperature (°C)</label>
      </div>
      <div class="relative">
        <input id="m-vibration" type="number" step="0.1" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2" />
        <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="vibration_label">Vibration (mm/s)</label>
      </div>
      <div class="relative">
        <input id="m-load" type="number" min="0" max="100" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2" />
        <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="load_label">Load (%)</label>
      </div>
      <div class="flex justify-end gap-2 mt-1">
        <button type="button" class="close-factory-modal px-4 py-2 rounded-xl bg-white/10 hover:bg-white/20 text-sm" data-t="cancel_btn">Cancel</button>
        <button type="submit" id="modal-factory-submit" class="glow-btn rounded-xl px-4 py-2 text-sm font-semibold flex items-center gap-2">
          <span id="modal-factory-spinner" class="spinner hidden"></span>
          <span id="modal-factory-submit-label" data-t="save_btn">Save Changes</span>
        </button>
      </div>
    </form>
  </div>
</div>

<!-- SCADA MACHINE MODAL (5-section input panel) -->
<div id="modal-machine-scada" class="fixed inset-0 bg-black/70 hidden items-center justify-center z-40 p-4">
  <div class="glass-strong rounded-2xl p-6 w-full max-w-lg fade-in" style="max-height:92vh; overflow-y:auto;">
    <div class="flex items-center justify-between mb-4">
      <h3 class="font-bold text-lg" data-t="add_machine_scada_btn">+ Add Machine</h3>
      <button class="close-machine-modal text-slate-400 hover:text-slate-200"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18M6 6l12 12"/></svg></button>
    </div>
    <form id="scada-machine-form" class="flex flex-col gap-4">

      <div class="text-xs uppercase tracking-wide text-cyan-400 font-semibold mt-1" data-t="section_machine_info">Machine Info</div>
      <input id="sm-code" placeholder="Machine ID (e.g. M-01)" data-t-placeholder="ph_machine_id" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
      <input id="sm-name" placeholder="Machine Name" data-t-placeholder="ph_machine_name" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
      <input id="sm-section" placeholder="Factory Section" data-t-placeholder="ph_factory_section" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
      <input id="sm-operator" placeholder="Operator Name" data-t-placeholder="ph_operator_name" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />

      <div class="text-xs uppercase tracking-wide text-cyan-400 font-semibold mt-2" data-t="section_sensor_data">Sensor Data</div>
      <div class="grid grid-cols-2 gap-3">
        <input id="sm-temp" type="number" placeholder="Temperature (°C)" data-t-placeholder="temperature_label" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        <input id="sm-vibration" type="number" step="0.1" placeholder="Vibration (mm/s)" data-t-placeholder="vibration_label" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        <input id="sm-load" type="number" placeholder="Load (%)" data-t-placeholder="load_label" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        <input id="sm-pressure" type="number" step="0.1" placeholder="Pressure (bar)" data-t-placeholder="ph_pressure" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        <input id="sm-voltage" type="number" placeholder="Voltage (V)" data-t-placeholder="ph_voltage" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        <input id="sm-current" type="number" step="0.1" placeholder="Current (A)" data-t-placeholder="ph_current" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
      </div>

      <div class="text-xs uppercase tracking-wide text-cyan-400 font-semibold mt-2" data-t="section_status">Status</div>
      <select id="sm-status" class="input-field w-full rounded-xl text-sm px-3 py-2.5">
        <option value="running" data-t="status_running">Running</option>
        <option value="stopped" data-t="status_stopped">Stopped</option>
        <option value="maintenance" data-t="status_maintenance">Maintenance</option>
      </select>
      <div class="grid grid-cols-2 gap-3">
        <input id="sm-error" placeholder="Error Code" data-t-placeholder="ph_error_code" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        <select id="sm-priority" class="input-field w-full rounded-xl text-sm px-3 py-2.5">
          <option value="low" data-t="priority_low">Low</option>
          <option value="normal" selected data-t="priority_normal">Normal</option>
          <option value="high" data-t="priority_high">High</option>
          <option value="critical" data-t="priority_critical">Critical</option>
        </select>
      </div>

      <div class="text-xs uppercase tracking-wide text-cyan-400 font-semibold mt-2" data-t="section_energy_intel">Energy Intelligence</div>
      <input id="sm-output" type="number" step="1" placeholder="Daily Output (units)" data-t-placeholder="ph_daily_output" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
      <p class="text-xs text-slate-500" data-t="daily_output_hint">Used to calculate Specific Energy Consumption (kWh per unit).</p>

      <div class="text-xs uppercase tracking-wide text-cyan-400 font-semibold mt-2" data-t="section_notes">Notes</div>
      <textarea id="sm-notes" rows="3" class="input-field w-full rounded-xl text-sm px-3 py-2.5" placeholder="Notes..." data-t-placeholder="ph_notes"></textarea>

      <div class="flex justify-end gap-2 mt-1">
        <button type="button" class="close-machine-modal px-4 py-2 rounded-xl bg-white/10 hover:bg-white/20 text-sm" data-t="cancel_btn">Cancel</button>
        <button type="submit" id="scada-machine-submit" class="glow-btn rounded-xl px-4 py-2 text-sm font-semibold flex items-center gap-2">
          <span id="scada-machine-spinner" class="spinner hidden"></span>
          <span data-t="save_and_analyze_btn">Save &amp; Analyze</span>
        </button>
      </div>
    </form>
  </div>
</div>

<!-- ENERGY INSIGHTS MODAL -->
<div id="modal-energy-insights" class="fixed inset-0 bg-black/70 hidden items-center justify-center z-40 p-4">
  <div class="glass-strong rounded-2xl p-6 w-full max-w-md fade-in" style="max-height:90vh; overflow-y:auto;">
    <div class="flex items-center justify-between mb-4">
      <h3 class="font-bold text-lg flex items-center gap-2">⚡ <span id="energy-insights-machine-name" data-t="energy_insights_title">Energy Intelligence</span></h3>
      <button class="close-energy-modal text-slate-400 hover:text-slate-200"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18M6 6l12 12"/></svg></button>
    </div>
    <div id="energy-insights-body" class="flex flex-col gap-4"></div>
  </div>
</div>


<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const translations = {
  en: {tagline:"Global Industrial Intelligence Platform",live_label:"Live",kpi_energy:"Energy Usage",kpi_efficiency:"Efficiency",kpi_active:"Active Machines",kpi_alerts:"Alerts",kwh_unit:"kWh",chart_title:"Real-Time Performance",machine_status_title:"Machine Status",status_running:"Running",status_warning:"Warning",status_critical:"Critical",form_title:"Factory Data Input",factory_name_label:"Factory Name",machine_count_label:"Number of Machines",energy_cost_label:"Energy Cost ($/kWh)",machine_type_label:"Machine Type",temperature_label:"Temperature (°C)",vibration_label:"Vibration (mm/s)",load_label:"Load (%)",submit_btn:"Analyze Factory",submitting:"Updating...",ai_panel_title:"AI Insights",ai_placeholder:"Submit factory data to generate an AI analysis.",ai_analyzing:"Analyzing...",ai_risks:"Risks",ai_efficiency_insights:"Efficiency Insights",ai_optimizations:"Optimization Suggestions",toast_updated:"Factory data updated",toast_analysis_done:"AI analysis complete",toast_error:"Something went wrong",nav_dashboard:"Dashboard",nav_factories:"Factories",nav_ai_insights:"AI Insights",logout_btn:"Log Out",login_title:"Welcome back",login_subtitle:"Sign in to your FactoryPulse AI account",ph_email:"Email",ph_password:"Password",remember_me:"Remember me",login_btn:"Log In",login_link_register:"Don't have an account? Create one",register_title:"Create your account",register_subtitle:"Start monitoring your factories with AI",ph_full_name:"Full Name",ph_confirm_password:"Confirm Password",register_btn:"Create Account",register_link_login:"Already have an account? Sign in",err_missing_fields:"Please fill in all fields",err_invalid_email:"Please enter a valid email address",err_weak_password:"Password must be at least 8 characters with a letter and a number",err_password_mismatch:"Passwords do not match",err_invalid_credentials:"Invalid email or password",err_email_taken:"This email is already registered",err_generic:"Something went wrong. Please try again",my_factories_title:"My Factories",add_factory_btn:"+ Add Factory",edit_factory_btn:"Edit",delete_factory_btn:"Delete",confirm_delete_factory:"Delete this factory? This cannot be undone.",no_factories_yet:"You haven't added any factories yet.",factory_created_toast:"Factory created and analyzed",factory_updated_toast:"Factory updated",factory_deleted_toast:"Factory deleted",ai_insights_feed_title:"AI Insights Feed",no_ai_insights_yet:"No AI insights yet. Add a factory to get started.",reanalyze_btn:"Re-analyze",view_insights_btn:"View Insights",created_label:"Created",cancel_btn:"Cancel",save_btn:"Save Changes",nav_live_monitor:"Live Monitor",add_machine_scada_btn:"+ Add Machine",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Polling",live_chart_title:"Live Sensor Chart",machines_table_title:"Machines",machine_code_col:"Code",machine_name_col:"Name",status_col:"Status",risk_col:"Risk",no_machines_yet:"No machines yet. Click \"+ Add Machine\".",section_machine_info:"Machine Info",section_sensor_data:"Sensor Data",section_status:"Status",section_notes:"Notes",status_stopped:"Stopped",status_maintenance:"Maintenance",priority_low:"Low",priority_normal:"Normal",priority_high:"High",priority_critical:"Critical",save_and_analyze_btn:"Save & Analyze",source_col:"Source",source_auto:"Auto (SCADA)",source_manual:"Manual",nav_alerts:"Alerts",acknowledge_btn:"Acknowledge",acknowledged_label:"Acknowledged",acknowledge_all_btn:"Acknowledge All",no_alerts_yet:"No alerts. Everything is running smoothly.",download_report_btn:"Report",alert_details_template:"Temperature {temp}°C, vibration {vib} mm/s, status: {status}",section_energy_intel:"Energy Intelligence",daily_output_hint:"Used to calculate Specific Energy Consumption (kWh per unit).",energy_insights_title:"Energy Intelligence",idle_power_title:"Idle Power Detection",idle_active_msg:"Machine is idle - wasting approx. {kw} kW right now.",idle_none_msg:"No idle power waste detected.",friction_loss_title:"Predictive Energy Loss",friction_active_msg:"Elevated friction detected: +{pct}% power overhead (~{kw} kW extra). Schedule maintenance to prevent losses.",friction_none_msg:"No abnormal friction losses detected.",sec_title:"Specific Energy Consumption",sec_label:"kWh per unit",sec_unit:"kWh/unit",sec_no_data_msg:"Enter Daily Output when adding this machine to see this metric.",optimal_load_title:"Optimal Load Zone",optimal_load_label:"Optimal load",current_load_label:"Current load",at_optimal_msg:"Running in the optimal load zone.",adjust_to_optimal_msg:"Adjust load toward {pct}% to minimize energy per unit.",nav_digital_twin:"Digital Twin",twin_hint:"Drag to rotate, scroll to zoom, click a machine to see its live details.",twin_unavailable_msg:"3D view could not load (check your internet connection for the Three.js library).",failure_prediction_title:"Failure Prediction",report_lib_missing_msg:"PDF export needs the reportlab library. Run: pip install reportlab, then restart the server.",ph_machine_id:"Machine ID (e.g. M-01)",ph_machine_name:"Machine Name",ph_factory_section:"Factory Section",ph_operator_name:"Operator Name",ph_pressure:"Pressure (bar)",ph_voltage:"Voltage (V)",ph_current:"Current (A)",ph_error_code:"Error Code",ph_daily_output:"Daily Output (units)",ph_notes:"Notes...",nav_system_intel:"System Intelligence",nav_roi:"ROI Dashboard",refresh_btn:"Refresh",system_risk_label:"System Risk",healthy_label:"Healthy",at_risk_label:"At Risk",clusters_title:"Machine Clusters",propagation_title:"Anomaly Propagation",propagation_hint:"How a failing machine raises the effective risk of its neighbours.",no_propagation:"No anomaly propagation detected.",avg_risk_label:"Avg risk",added_risk_label:"Added risk",effective_risk_label:"Effective risk",simulation_title:"What-If Simulation",simulation_hint:"Pick a machine, shift its sensor values, and see how failure probability reacts.",run_simulation_btn:"Run Simulation",failure_probability_label:"Failure Probability",stress_level_label:"Stress Level",predicted_status_label:"Predicted Status",confidence_label:"Confidence",root_cause_title:"Root Cause",rul_col:"RUL",rul_healthy:"Healthy",potential_loss_label:"Potential Loss",saved_label:"Saved by AI",wasted_energy_label:"Wasted Energy / month",efficiency_gain_label:"Efficiency Gain",cost_by_machine_title:"Cost Exposure by Machine",top_cause_col:"Top Cause",roi_assumptions_msg:"Assumptions: downtime {downtime}/h, {hours}h average repair, {price}/kWh energy.",role_label:"Your Role",role_engineer:"Engineer",role_manager:"Manager",role_admin:"Admin",cause_bearing_wear:"Bearing wear",cause_overload_thermal:"Thermal overload",cause_cooling_failure:"Cooling failure",cause_misalignment:"Shaft misalignment",cause_lubrication_loss:"Lubrication loss",cause_normal_operation:"Normal operation",nav_story:"Story Mode",story_hint:"Replays a bearing failure developing over 22 hours and shows exactly when the AI caught it — and what that was worth.",simulate_failure_btn:"Simulate Failure",outcome_title:"Outcome",warning_time_label:"Early Warning",loss_ignored_label:"Loss if Ignored",loss_acted_label:"Loss if Acted On",money_saved_label:"Money Saved",timeline_title:"Failure Timeline",story_detection_msg:"AI flagged this {hours} hours before breakdown, at {risk}% risk — root cause: {cause} ({confidence}% confidence).",story_stage_healthy:"Healthy",story_stage_early_drift:"Early Drift",story_stage_ai_detects:"AI Detects",story_stage_critical:"Critical",priority_low:"Low",priority_medium:"Medium",priority_high:"High",priority_critical:"Critical",action_stop_machine:"Stop machine",action_inspect_bearings:"Inspect bearings",action_check_cooling:"Check cooling",action_schedule_shutdown_24h:"Schedule shutdown (24h)",action_order_spare_parts:"Order spare parts",action_reduce_load:"Reduce load",action_schedule_inspection_72h:"Schedule inspection (72h)",action_monitor_closely:"Monitor closely",action_verify_sensor:"Verify sensor",action_power_down_idle:"Power down idle machine",action_review_shift_schedule:"Review shift schedule",action_no_action:"No action needed",nav_oee:"OEE",nav_workorders:"Work Orders",oee_hint:"Availability x Performance x Quality (ISO 22400). World-class is 85%; a typical factory sits near 60%.",availability_label:"Availability",performance_label:"Performance",quality_label:"Quality",weakest_factor_label:"Weakest",downtime_by_reason_title:"Downtime by Reason",downtime_cost_label:"Downtime cost",oee_trend_title:"OEE Trend",shifts_title:"Shifts",shift_col:"Shift",downtime_col:"Downtime",log_shift_btn:"Log Shift",no_shifts_yet:"No shifts logged yet.",minutes_short:"min",range_1d:"Today",range_7d:"7 days",range_30d:"30 days",all_machines_option:"All machines",reason_unspecified:"Unspecified",err_good_exceeds_total:"Good units cannot exceed total units",oee_grade_world_class:"World-class",oee_grade_typical:"Typical",oee_grade_low:"Low",oee_grade_critical:"Critical",reason_breakdown:"Breakdown",reason_changeover:"Changeover",reason_no_material:"No material",reason_no_operator:"No operator",reason_planned_maintenance:"Planned maintenance",reason_quality_issue:"Quality issue",reason_setup:"Setup",reason_other:"Other",workorders_hint:"Turns an AI prediction into tracked, assignable work — so a warning actually ends in a repair.",new_work_order_btn:"+ New Work Order",no_work_orders:"No work orders yet.",avg_completion_label:"Avg. Completion",assigned_label:"Assigned",source_ai:"AI",wo_status_open:"Open",wo_status_in_progress:"In Progress",wo_status_done:"Done",wo_status_cancelled:"Cancelled",wo_advance_to_in_progress:"Start work",wo_advance_to_done:"Mark done",ph_shift_name:"Shift name",ph_planned_minutes:"Planned minutes",ph_downtime_minutes:"Downtime minutes",ph_total_units:"Total units",ph_good_units:"Good units",ph_cycle_seconds:"Ideal cycle (seconds)",ph_wo_title:"Title",ph_assigned_to:"Assigned to",ph_wo_description:"Description",overdue_label:"Overdue",nav_history:"History",history_hint:"Stored sensor history — this is how you prove what actually changed over the pilot.",no_history_yet:"No history recorded yet. Keep the Live Monitor open for a few minutes and data will start accumulating.",range_24h:"24 hours",range_3d:"3 days",range_10d:"10 days",sensor_trend_title:"Sensor Trend",risk_trend_title:"Risk Trend",trend_flat:"no change",trend_rising:"rising",trend_falling:"falling",create_work_order_btn:"Create Work Order",work_order_created_msg:"Work order created",alertreason_immediate_failure_risk:"Immediate failure risk",alertreason_failure_imminent:"Failure imminent",alertreason_degradation_accelerating:"Degradation accelerating",alertreason_outside_normal_envelope:"Outside normal envelope",alertreason_pressure_anomaly:"Pressure anomaly",alertreason_idle_waste:"Idle energy waste",alertreason_informational:"Informational",alert_reason_label:"Reason"},
  ru: {tagline:"Глобальная платформа промышленного интеллекта",live_label:"Live",kpi_energy:"Потребление энергии",kpi_efficiency:"Эффективность",kpi_active:"Активные станки",kpi_alerts:"Оповещения",kwh_unit:"кВт·ч",chart_title:"Показатели в реальном времени",machine_status_title:"Статус станков",status_running:"Работает",status_warning:"Внимание",status_critical:"Критично",form_title:"Ввод данных завода",factory_name_label:"Название завода",machine_count_label:"Количество станков",energy_cost_label:"Стоимость энергии ($/кВт·ч)",machine_type_label:"Тип станка",temperature_label:"Температура (°C)",vibration_label:"Вибрация (мм/с)",load_label:"Нагрузка (%)",submit_btn:"Анализировать завод",submitting:"Обновление...",ai_panel_title:"AI-аналитика",ai_placeholder:"Отправьте данные завода, чтобы получить AI-анализ.",ai_analyzing:"Анализ...",ai_risks:"Риски",ai_efficiency_insights:"Анализ эффективности",ai_optimizations:"Рекомендации по оптимизации",toast_updated:"Данные завода обновлены",toast_analysis_done:"AI-анализ завершён",toast_error:"Произошла ошибка",nav_dashboard:"Панель",nav_factories:"Заводы",nav_ai_insights:"AI-аналитика",logout_btn:"Выход",login_title:"С возвращением",login_subtitle:"Войдите в аккаунт FactoryPulse AI",ph_email:"Email",ph_password:"Пароль",remember_me:"Запомнить меня",login_btn:"Войти",login_link_register:"Нет аккаунта? Создать",register_title:"Создать аккаунт",register_subtitle:"Начните мониторинг заводов с помощью AI",ph_full_name:"Полное имя",ph_confirm_password:"Подтвердите пароль",register_btn:"Создать аккаунт",register_link_login:"Уже есть аккаунт? Войти",err_missing_fields:"Заполните все поля",err_invalid_email:"Введите корректный email",err_weak_password:"Пароль должен быть от 8 символов, с буквой и цифрой",err_password_mismatch:"Пароли не совпадают",err_invalid_credentials:"Неверный email или пароль",err_email_taken:"Этот email уже зарегистрирован",err_generic:"Что-то пошло не так. Попробуйте снова",my_factories_title:"Мои заводы",add_factory_btn:"+ Добавить завод",edit_factory_btn:"Изменить",delete_factory_btn:"Удалить",confirm_delete_factory:"Удалить этот завод? Это действие нельзя отменить.",no_factories_yet:"Вы ещё не добавили ни одного завода.",factory_created_toast:"Завод создан и проанализирован",factory_updated_toast:"Завод обновлён",factory_deleted_toast:"Завод удалён",ai_insights_feed_title:"Лента AI-аналитики",no_ai_insights_yet:"Пока нет AI-аналитики. Добавьте завод, чтобы начать.",reanalyze_btn:"Проанализировать снова",view_insights_btn:"Смотреть аналитику",created_label:"Создано",cancel_btn:"Отмена",save_btn:"Сохранить изменения",nav_live_monitor:"Мониторинг",add_machine_scada_btn:"+ Добавить станок",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Опрос",live_chart_title:"График датчиков в реальном времени",machines_table_title:"Станки",machine_code_col:"Код",machine_name_col:"Название",status_col:"Статус",risk_col:"Риск",no_machines_yet:"Станков пока нет. Нажмите «+ Добавить станок».",section_machine_info:"Информация о станке",section_sensor_data:"Данные датчиков",section_status:"Статус",section_notes:"Заметки",status_stopped:"Остановлен",status_maintenance:"Обслуживание",priority_low:"Низкий",priority_normal:"Обычный",priority_high:"Высокий",priority_critical:"Критический",save_and_analyze_btn:"Сохранить и анализировать",source_col:"Источник",source_auto:"Авто (SCADA)",source_manual:"Вручную",nav_alerts:"Оповещения",acknowledge_btn:"Подтвердить",acknowledged_label:"Подтверждено",acknowledge_all_btn:"Подтвердить все",no_alerts_yet:"Оповещений нет. Всё работает штатно.",download_report_btn:"Отчёт",alert_details_template:"Температура {temp}°C, вибрация {vib} мм/с, статус: {status}",section_energy_intel:"Энергетический интеллект",daily_output_hint:"Используется для расчёта удельного энергопотребления (кВт·ч на единицу).",energy_insights_title:"Энергетический интеллект",idle_power_title:"Обнаружение холостого хода",idle_active_msg:"Станок простаивает — расходуется примерно {kw} кВт впустую.",idle_none_msg:"Потерь энергии на холостом ходу не обнаружено.",friction_loss_title:"Прогноз потерь энергии",friction_active_msg:"Обнаружено повышенное трение: +{pct}% лишней мощности (~{kw} кВт). Запланируйте обслуживание, чтобы избежать потерь.",friction_none_msg:"Аномального трения не обнаружено.",sec_title:"Удельное энергопотребление",sec_label:"кВт·ч на единицу",sec_unit:"кВт·ч/ед.",sec_no_data_msg:"Укажите суточный объём выпуска при добавлении станка, чтобы увидеть этот показатель.",optimal_load_title:"Оптимальная зона нагрузки",optimal_load_label:"Оптимальная нагрузка",current_load_label:"Текущая нагрузка",at_optimal_msg:"Работает в оптимальной зоне нагрузки.",adjust_to_optimal_msg:"Приблизьте нагрузку к {pct}%, чтобы минимизировать расход энергии на единицу.",nav_digital_twin:"Цифровой двойник",twin_hint:"Перетаскивайте для поворота, прокручивайте для масштаба, нажмите на станок для подробностей.",twin_unavailable_msg:"Не удалось загрузить 3D-вид (проверьте подключение к интернету для библиотеки Three.js).",failure_prediction_title:"Прогноз отказа",report_lib_missing_msg:"Для экспорта в PDF нужна библиотека reportlab. Выполните: pip install reportlab, затем перезапустите сервер.",ph_machine_id:"ID станка (напр. M-01)",ph_machine_name:"Название станка",ph_factory_section:"Участок завода",ph_operator_name:"Имя оператора",ph_pressure:"Давление (бар)",ph_voltage:"Напряжение (В)",ph_current:"Ток (А)",ph_error_code:"Код ошибки",ph_daily_output:"Суточный выпуск (ед.)",ph_notes:"Заметки...",nav_system_intel:"Системная аналитика",nav_roi:"ROI-панель",refresh_btn:"Обновить",system_risk_label:"Системный риск",healthy_label:"Исправны",at_risk_label:"В зоне риска",clusters_title:"Кластеры станков",propagation_title:"Распространение аномалий",propagation_hint:"Как отказ одного станка повышает риск соседних.",no_propagation:"Распространение аномалий не обнаружено.",avg_risk_label:"Средний риск",added_risk_label:"Добавленный риск",effective_risk_label:"Итоговый риск",simulation_title:"Что-если симуляция",simulation_hint:"Выберите станок, измените показания датчиков и посмотрите на изменение риска.",run_simulation_btn:"Запустить симуляцию",failure_probability_label:"Вероятность отказа",stress_level_label:"Уровень нагрузки",predicted_status_label:"Прогноз статуса",confidence_label:"Достоверность",root_cause_title:"Первопричина",rul_col:"Остаточный ресурс",rul_healthy:"Исправен",potential_loss_label:"Потенциальные потери",saved_label:"Сэкономлено AI",wasted_energy_label:"Потери энергии / месяц",efficiency_gain_label:"Прирост эффективности",cost_by_machine_title:"Финансовый риск по станкам",top_cause_col:"Основная причина",roi_assumptions_msg:"Допущения: простой {downtime}/ч, ремонт {hours} ч, энергия {price}/кВт·ч.",role_label:"Ваша роль",role_engineer:"Инженер",role_manager:"Менеджер",role_admin:"Администратор",cause_bearing_wear:"Износ подшипника",cause_overload_thermal:"Тепловая перегрузка",cause_cooling_failure:"Отказ охлаждения",cause_misalignment:"Расцентровка вала",cause_lubrication_loss:"Потеря смазки",cause_normal_operation:"Нормальная работа",nav_story:"Режим истории",story_hint:"Воспроизводит развитие отказа подшипника за 22 часа и показывает, когда именно AI его обнаружил — и сколько это стоило.",simulate_failure_btn:"Смоделировать отказ",outcome_title:"Итог",warning_time_label:"Раннее предупреждение",loss_ignored_label:"Потери без реакции",loss_acted_label:"Потери при реакции",money_saved_label:"Сэкономлено",timeline_title:"Хронология отказа",story_detection_msg:"AI обнаружил это за {hours} ч до поломки, при риске {risk}% — первопричина: {cause} (достоверность {confidence}%).",story_stage_healthy:"Исправен",story_stage_early_drift:"Начало отклонения",story_stage_ai_detects:"AI обнаружил",story_stage_critical:"Критично",priority_low:"Низкий",priority_medium:"Средний",priority_high:"Высокий",priority_critical:"Критический",action_stop_machine:"Остановить станок",action_inspect_bearings:"Проверить подшипники",action_check_cooling:"Проверить охлаждение",action_schedule_shutdown_24h:"Запланировать остановку (24ч)",action_order_spare_parts:"Заказать запчасти",action_reduce_load:"Снизить нагрузку",action_schedule_inspection_72h:"Запланировать осмотр (72ч)",action_monitor_closely:"Внимательно наблюдать",action_verify_sensor:"Проверить датчик",action_power_down_idle:"Отключить простаивающий станок",action_review_shift_schedule:"Пересмотреть график смен",action_no_action:"Действий не требуется",nav_oee:"OEE",nav_workorders:"Наряды",oee_hint:"Доступность x Производительность x Качество (ISO 22400). Мировой уровень — 85%, типичный завод — около 60%.",availability_label:"Доступность",performance_label:"Производительность",quality_label:"Качество",weakest_factor_label:"Слабое звено",downtime_by_reason_title:"Простои по причинам",downtime_cost_label:"Стоимость простоя",oee_trend_title:"Динамика OEE",shifts_title:"Смены",shift_col:"Смена",downtime_col:"Простой",log_shift_btn:"Внести смену",no_shifts_yet:"Смены ещё не внесены.",minutes_short:"мин",range_1d:"Сегодня",range_7d:"7 дней",range_30d:"30 дней",all_machines_option:"Все станки",reason_unspecified:"Не указано",err_good_exceeds_total:"Годных не может быть больше общего количества",oee_grade_world_class:"Мировой уровень",oee_grade_typical:"Типичный",oee_grade_low:"Низкий",oee_grade_critical:"Критический",reason_breakdown:"Поломка",reason_changeover:"Переналадка",reason_no_material:"Нет материала",reason_no_operator:"Нет оператора",reason_planned_maintenance:"Плановое ТО",reason_quality_issue:"Проблема качества",reason_setup:"Настройка",reason_other:"Другое",workorders_hint:"Превращает прогноз AI в отслеживаемую задачу — чтобы предупреждение закончилось ремонтом.",new_work_order_btn:"+ Новый наряд",no_work_orders:"Нарядов пока нет.",avg_completion_label:"Ср. выполнение",assigned_label:"Назначен",source_ai:"AI",wo_status_open:"Открыт",wo_status_in_progress:"В работе",wo_status_done:"Выполнен",wo_status_cancelled:"Отменён",wo_advance_to_in_progress:"Начать работу",wo_advance_to_done:"Отметить выполненным",ph_shift_name:"Название смены",ph_planned_minutes:"Плановые минуты",ph_downtime_minutes:"Минуты простоя",ph_total_units:"Всего единиц",ph_good_units:"Годных единиц",ph_cycle_seconds:"Идеальный цикл (сек)",ph_wo_title:"Название",ph_assigned_to:"Ответственный",ph_wo_description:"Описание",overdue_label:"Просрочено",nav_history:"История",history_hint:"Сохранённая история датчиков — так вы докажете, что реально изменилось за пилот.",no_history_yet:"История ещё не записана. Подержите Мониторинг открытым несколько минут, и данные начнут накапливаться.",range_24h:"24 часа",range_3d:"3 дня",range_10d:"10 дней",sensor_trend_title:"Динамика датчиков",risk_trend_title:"Динамика риска",trend_flat:"без изменений",trend_rising:"растёт",trend_falling:"снижается",create_work_order_btn:"Создать наряд",work_order_created_msg:"Наряд создан",alertreason_immediate_failure_risk:"Риск немедленного отказа",alertreason_failure_imminent:"Отказ неизбежен",alertreason_degradation_accelerating:"Ускоряющийся износ",alertreason_outside_normal_envelope:"Выход за штатные пределы",alertreason_pressure_anomaly:"Аномалия давления",alertreason_idle_waste:"Потери на холостом ходу",alertreason_informational:"Информационное",alert_reason_label:"Причина"},
  kk: {tagline:"Жаһандық өнеркәсіптік интеллект платформасы",live_label:"Тікелей эфир",kpi_energy:"Энергия тұтыну",kpi_efficiency:"Тиімділік",kpi_active:"Белсенді станоктар",kpi_alerts:"Дабылдар",kwh_unit:"кВт·сағ",chart_title:"Нақты уақыттағы көрсеткіштер",machine_status_title:"Станоктар күйі",status_running:"Жұмыс істеп тұр",status_warning:"Ескерту",status_critical:"Сыни",form_title:"Зауыт деректерін енгізу",factory_name_label:"Зауыт атауы",machine_count_label:"Станоктар саны",energy_cost_label:"Энергия құны ($/кВт·сағ)",machine_type_label:"Станок түрі",temperature_label:"Температура (°C)",vibration_label:"Діріл (мм/с)",load_label:"Жүктеме (%)",submit_btn:"Зауытты талдау",submitting:"Жаңартылуда...",ai_panel_title:"AI-талдау",ai_placeholder:"AI-талдау алу үшін зауыт деректерін жіберіңіз.",ai_analyzing:"Талдануда...",ai_risks:"Тәуекелдер",ai_efficiency_insights:"Тиімділік талдауы",ai_optimizations:"Оңтайландыру ұсыныстары",toast_updated:"Зауыт деректері жаңартылды",toast_analysis_done:"AI-талдау аяқталды",toast_error:"Қате орын алды",nav_dashboard:"Басқару тақтасы",nav_factories:"Зауыттар",nav_ai_insights:"AI-талдау",logout_btn:"Шығу",login_title:"Қайта қош келдіңіз",login_subtitle:"FactoryPulse AI аккаунтыңызға кіріңіз",ph_email:"Email",ph_password:"Құпия сөз",remember_me:"Мені есте сақтау",login_btn:"Кіру",login_link_register:"Аккаунтыңыз жоқ па? Тіркелу",register_title:"Аккаунт құру",register_subtitle:"Зауыттарды AI арқылы бақылауды бастаңыз",ph_full_name:"Толық аты-жөні",ph_confirm_password:"Құпия сөзді қайталаңыз",register_btn:"Аккаунт құру",register_link_login:"Аккаунтыңыз бар ма? Кіру",err_missing_fields:"Барлық өрістерді толтырыңыз",err_invalid_email:"Дұрыс email мекенжайын енгізіңіз",err_weak_password:"Құпия сөз кемінде 8 таңба, әріп пен сан болуы керек",err_password_mismatch:"Құпия сөздер сәйкес келмейді",err_invalid_credentials:"Қате email немесе құпия сөз",err_email_taken:"Бұл email тіркелген",err_generic:"Қате орын алды. Қайталап көріңіз",my_factories_title:"Менің зауыттарым",add_factory_btn:"+ Зауыт қосу",edit_factory_btn:"Өзгерту",delete_factory_btn:"Жою",confirm_delete_factory:"Бұл зауытты жоясыз ба? Бұл әрекетті кері қайтару мүмкін емес.",no_factories_yet:"Сіз әлі ешбір зауыт қосқан жоқсыз.",factory_created_toast:"Зауыт құрылды және талданды",factory_updated_toast:"Зауыт жаңартылды",factory_deleted_toast:"Зауыт жойылды",ai_insights_feed_title:"AI-талдау таспасы",no_ai_insights_yet:"AI-талдау әлі жоқ. Бастау үшін зауыт қосыңыз.",reanalyze_btn:"Қайта талдау",view_insights_btn:"Талдауды көру",created_label:"Құрылған күні",cancel_btn:"Бас тарту",save_btn:"Өзгерістерді сақтау",nav_live_monitor:"Тікелей мониторинг",add_machine_scada_btn:"+ Станок қосу",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Сұрау",live_chart_title:"Нақты уақыттағы сенсор графигі",machines_table_title:"Станоктар",machine_code_col:"Код",machine_name_col:"Атауы",status_col:"Күй",risk_col:"Тәуекел",no_machines_yet:"Станоктар әлі жоқ. «+ Станок қосу» басыңыз.",section_machine_info:"Станок туралы ақпарат",section_sensor_data:"Сенсор деректері",section_status:"Күй",section_notes:"Ескертпелер",status_stopped:"Тоқтатылды",status_maintenance:"Техникалық қызмет",priority_low:"Төмен",priority_normal:"Қалыпты",priority_high:"Жоғары",priority_critical:"Сыни",save_and_analyze_btn:"Сақтау және талдау",source_col:"Дереккөз",source_auto:"Авто (SCADA)",source_manual:"Қолмен",nav_alerts:"Дабылдар",acknowledge_btn:"Растау",acknowledged_label:"Расталды",acknowledge_all_btn:"Барлығын растау",no_alerts_yet:"Дабылдар жоқ. Бәрі қалыпты жұмыс істеп тұр.",download_report_btn:"Есеп",alert_details_template:"Температура {temp}°C, діріл {vib} мм/с, күйі: {status}",section_energy_intel:"Энергетикалық интеллект",daily_output_hint:"Бірлік өнімге кететін энергияны есептеу үшін қолданылады (кВт·сағ/бірлік).",energy_insights_title:"Энергетикалық интеллект",idle_power_title:"Бос жүрісті анықтау",idle_active_msg:"Станок бос тұр — шамамен {kw} кВт босқа кетіп жатыр.",idle_none_msg:"Бос жүріс шығыны табылмады.",friction_loss_title:"Энергия шығынын болжау",friction_active_msg:"Артық үйкеліс табылды: +{pct}% қосымша қуат (~{kw} кВт). Шығынды болдырмау үшін техникалық қызмет жоспарлаңыз.",friction_none_msg:"Ауытқыған үйкеліс табылмады.",sec_title:"Бірлік өнімге кететін қуат",sec_label:"кВт·сағ/бірлік",sec_unit:"кВт·сағ/бірлік",sec_no_data_msg:"Бұл көрсеткішті көру үшін станок қосқанда күнделікті өнім санын енгізіңіз.",optimal_load_title:"Оңтайлы жұмыс режимі",optimal_load_label:"Оңтайлы жүктеме",current_load_label:"Ағымдағы жүктеме",at_optimal_msg:"Оңтайлы жүктеме аймағында жұмыс істеп тұр.",adjust_to_optimal_msg:"Бірлікке кететін энергияны азайту үшін жүктемені {pct}%-ға жақындатыңыз.",nav_digital_twin:"Цифрлық егіз",twin_hint:"Айналдыру үшін сүйреңіз, масштабтау үшін айналдырыңыз, толық мәлімет үшін станокты басыңыз.",twin_unavailable_msg:"3D көрініс жүктелмеді (Three.js кітапханасы үшін интернет байланысын тексеріңіз).",failure_prediction_title:"Ақаудың болжамы",report_lib_missing_msg:"PDF экспорты үшін reportlab кітапханасы керек. Орындаңыз: pip install reportlab, содан кейін серверді қайта қосыңыз.",ph_machine_id:"Станок ID (мыс. M-01)",ph_machine_name:"Станок атауы",ph_factory_section:"Зауыт учаскесі",ph_operator_name:"Оператор аты",ph_pressure:"Қысым (бар)",ph_voltage:"Кернеу (В)",ph_current:"Ток (А)",ph_error_code:"Қате коды",ph_daily_output:"Күнделікті өнім (бірлік)",ph_notes:"Ескертпелер...",nav_system_intel:"Жүйелік талдау",nav_roi:"ROI тақтасы",refresh_btn:"Жаңарту",system_risk_label:"Жүйелік тәуекел",healthy_label:"Сау",at_risk_label:"Тәуекелде",clusters_title:"Станок кластерлері",propagation_title:"Ақаудың таралуы",propagation_hint:"Бір станоктың ақауы көршілеріне қалай әсер етеді.",no_propagation:"Ақаудың таралуы анықталмады.",avg_risk_label:"Орташа тәуекел",added_risk_label:"Қосылған тәуекел",effective_risk_label:"Нақты тәуекел",simulation_title:"Не-болса симуляциясы",simulation_hint:"Станокты таңдап, сенсор мәндерін өзгертіп, тәуекелдің қалай өзгеретінін көріңіз.",run_simulation_btn:"Симуляцияны іске қосу",failure_probability_label:"Ақау ықтималдығы",stress_level_label:"Кернеу деңгейі",predicted_status_label:"Күй болжамы",confidence_label:"Сенімділік",root_cause_title:"Түпкі себеп",rul_col:"Қалдық ресурс",rul_healthy:"Сау",potential_loss_label:"Ықтимал шығын",saved_label:"AI үнемдеді",wasted_energy_label:"Энергия шығыны / ай",efficiency_gain_label:"Тиімділік өсімі",cost_by_machine_title:"Станоктар бойынша қаржы тәуекелі",top_cause_col:"Негізгі себеп",roi_assumptions_msg:"Болжамдар: тоқтап қалу {downtime}/сағ, жөндеу {hours} сағ, энергия {price}/кВт·сағ.",role_label:"Сіздің рөліңіз",role_engineer:"Инженер",role_manager:"Менеджер",role_admin:"Әкімші",cause_bearing_wear:"Подшипник тозуы",cause_overload_thermal:"Жылулық шамадан тыс жүктеме",cause_cooling_failure:"Салқындату ақауы",cause_misalignment:"Білік центрден ауытқуы",cause_lubrication_loss:"Майлау жоғалуы",cause_normal_operation:"Қалыпты жұмыс",nav_story:"Оқиға режимі",story_hint:"Подшипник ақауының 22 сағат ішінде дамуын ойнатып, AI оны дәл қашан анықтағанын және бұл қанша тұратынын көрсетеді.",simulate_failure_btn:"Ақауды модельдеу",outcome_title:"Нәтиже",warning_time_label:"Ерте ескерту",loss_ignored_label:"Әрекетсіз шығын",loss_acted_label:"Әрекет еткендегі шығын",money_saved_label:"Үнемделді",timeline_title:"Ақау хронологиясы",story_detection_msg:"AI мұны бұзылудан {hours} сағат бұрын, {risk}% тәуекелде анықтады — түпкі себеп: {cause} (сенімділік {confidence}%).",story_stage_healthy:"Сау",story_stage_early_drift:"Ауытқу басы",story_stage_ai_detects:"AI анықтады",story_stage_critical:"Сыни",priority_low:"Төмен",priority_medium:"Орташа",priority_high:"Жоғары",priority_critical:"Сыни",action_stop_machine:"Станокты тоқтату",action_inspect_bearings:"Подшипниктерді тексеру",action_check_cooling:"Салқындатуды тексеру",action_schedule_shutdown_24h:"Тоқтатуды жоспарлау (24сағ)",action_order_spare_parts:"Қосалқы бөлшек тапсырыс беру",action_reduce_load:"Жүктемені азайту",action_schedule_inspection_72h:"Тексеруді жоспарлау (72сағ)",action_monitor_closely:"Мұқият бақылау",action_verify_sensor:"Сенсорды тексеру",action_power_down_idle:"Бос тұрған станокты өшіру",action_review_shift_schedule:"Ауысым кестесін қайта қарау",action_no_action:"Әрекет қажет емес",nav_oee:"OEE",nav_workorders:"Тапсырыстар",oee_hint:"Қолжетімділік x Өнімділік x Сапа (ISO 22400). Әлемдік деңгей — 85%, әдеттегі зауыт — 60% шамасында.",availability_label:"Қолжетімділік",performance_label:"Өнімділік",quality_label:"Сапа",weakest_factor_label:"Әлсіз буын",downtime_by_reason_title:"Себебі бойынша тоқтап қалу",downtime_cost_label:"Тоқтап қалу құны",oee_trend_title:"OEE динамикасы",shifts_title:"Ауысымдар",shift_col:"Ауысым",downtime_col:"Тоқтап қалу",log_shift_btn:"Ауысым енгізу",no_shifts_yet:"Ауысымдар әлі енгізілмеген.",minutes_short:"мин",range_1d:"Бүгін",range_7d:"7 күн",range_30d:"30 күн",all_machines_option:"Барлық станоктар",reason_unspecified:"Көрсетілмеген",err_good_exceeds_total:"Жарамды саны жалпы санынан көп бола алмайды",oee_grade_world_class:"Әлемдік деңгей",oee_grade_typical:"Әдеттегі",oee_grade_low:"Төмен",oee_grade_critical:"Сыни",reason_breakdown:"Бұзылу",reason_changeover:"Қайта баптау",reason_no_material:"Материал жоқ",reason_no_operator:"Оператор жоқ",reason_planned_maintenance:"Жоспарлы ТҚ",reason_quality_issue:"Сапа мәселесі",reason_setup:"Орнату",reason_other:"Басқа",workorders_hint:"AI болжамын бақыланатын тапсырысқа айналдырады — ескерту нақты жөндеумен аяқталсын.",new_work_order_btn:"+ Жаңа тапсырыс",no_work_orders:"Тапсырыстар әлі жоқ.",avg_completion_label:"Орт. орындалу",assigned_label:"Тағайындалды",source_ai:"AI",wo_status_open:"Ашық",wo_status_in_progress:"Орындалуда",wo_status_done:"Орындалды",wo_status_cancelled:"Бас тартылды",wo_advance_to_in_progress:"Жұмысты бастау",wo_advance_to_done:"Орындалды деп белгілеу",ph_shift_name:"Ауысым атауы",ph_planned_minutes:"Жоспарлы минут",ph_downtime_minutes:"Тоқтап қалған минут",ph_total_units:"Жалпы саны",ph_good_units:"Жарамды саны",ph_cycle_seconds:"Оңтайлы цикл (сек)",ph_wo_title:"Атауы",ph_assigned_to:"Жауапты",ph_wo_description:"Сипаттама",overdue_label:"Мерзімі өткен",nav_history:"Тарих",history_hint:"Сақталған сенсор тарихы — пилот кезінде не өзгергенін осылай дәлелдейсіз.",no_history_yet:"Тарих әлі жазылмаған. Тікелей мониторингті бірнеше минут ашық қалдырсаңыз, дерек жинала бастайды.",range_24h:"24 сағат",range_3d:"3 күн",range_10d:"10 күн",sensor_trend_title:"Сенсор динамикасы",risk_trend_title:"Тәуекел динамикасы",trend_flat:"өзгеріссіз",trend_rising:"өсуде",trend_falling:"төмендеуде",create_work_order_btn:"Тапсырыс жасау",work_order_created_msg:"Тапсырыс жасалды",alertreason_immediate_failure_risk:"Дереу бұзылу қаупі",alertreason_failure_imminent:"Ақау таяу",alertreason_degradation_accelerating:"Тозу үдеуде",alertreason_outside_normal_envelope:"Қалыпты шектен тыс",alertreason_pressure_anomaly:"Қысым ауытқуы",alertreason_idle_waste:"Бос жүріс шығыны",alertreason_informational:"Ақпараттық",alert_reason_label:"Себебі"},
  de: {tagline:"Globale Industrielle Intelligenzplattform",live_label:"Live",kpi_energy:"Energieverbrauch",kpi_efficiency:"Effizienz",kpi_active:"Aktive Maschinen",kpi_alerts:"Warnungen",kwh_unit:"kWh",chart_title:"Echtzeit-Leistung",machine_status_title:"Maschinenstatus",status_running:"Läuft",status_warning:"Warnung",status_critical:"Kritisch",form_title:"Fabrikdateneingabe",factory_name_label:"Fabrikname",machine_count_label:"Anzahl der Maschinen",energy_cost_label:"Energiekosten ($/kWh)",machine_type_label:"Maschinentyp",temperature_label:"Temperatur (°C)",vibration_label:"Vibration (mm/s)",load_label:"Last (%)",submit_btn:"Fabrik Analysieren",submitting:"Aktualisieren...",ai_panel_title:"KI-Einblicke",ai_placeholder:"Senden Sie Fabrikdaten, um eine KI-Analyse zu erstellen.",ai_analyzing:"Analysiere...",ai_risks:"Risiken",ai_efficiency_insights:"Effizienzanalyse",ai_optimizations:"Optimierungsvorschläge",toast_updated:"Fabrikdaten aktualisiert",toast_analysis_done:"KI-Analyse abgeschlossen",toast_error:"Etwas ist schiefgelaufen",nav_dashboard:"Übersicht",nav_factories:"Fabriken",nav_ai_insights:"KI-Einblicke",logout_btn:"Abmelden",login_title:"Willkommen zurück",login_subtitle:"Melden Sie sich bei Ihrem FactoryPulse AI-Konto an",ph_email:"E-Mail",ph_password:"Passwort",remember_me:"Angemeldet bleiben",login_btn:"Einloggen",login_link_register:"Kein Konto? Jetzt erstellen",register_title:"Konto erstellen",register_subtitle:"Beginnen Sie mit der KI-Überwachung Ihrer Fabriken",ph_full_name:"Vollständiger Name",ph_confirm_password:"Passwort bestätigen",register_btn:"Konto erstellen",register_link_login:"Bereits ein Konto? Anmelden",err_missing_fields:"Bitte füllen Sie alle Felder aus",err_invalid_email:"Bitte geben Sie eine gültige E-Mail-Adresse ein",err_weak_password:"Passwort muss mind. 8 Zeichen, einen Buchstaben und eine Zahl enthalten",err_password_mismatch:"Passwörter stimmen nicht überein",err_invalid_credentials:"Ungültige E-Mail oder Passwort",err_email_taken:"Diese E-Mail ist bereits registriert",err_generic:"Etwas ist schiefgelaufen. Bitte erneut versuchen",my_factories_title:"Meine Fabriken",add_factory_btn:"+ Fabrik Hinzufügen",edit_factory_btn:"Bearbeiten",delete_factory_btn:"Löschen",confirm_delete_factory:"Diese Fabrik löschen? Dies kann nicht rückgängig gemacht werden.",no_factories_yet:"Sie haben noch keine Fabriken hinzugefügt.",factory_created_toast:"Fabrik erstellt und analysiert",factory_updated_toast:"Fabrik aktualisiert",factory_deleted_toast:"Fabrik gelöscht",ai_insights_feed_title:"KI-Einblicke Feed",no_ai_insights_yet:"Noch keine KI-Einblicke. Fügen Sie eine Fabrik hinzu.",reanalyze_btn:"Erneut analysieren",view_insights_btn:"Einblicke Anzeigen",created_label:"Erstellt",cancel_btn:"Abbrechen",save_btn:"Änderungen Speichern",nav_live_monitor:"Live-Überwachung",add_machine_scada_btn:"+ Maschine Hinzufügen",usb_status:"USB:",plc_status:"SPS:",polling_mode:"Abfrage",live_chart_title:"Live-Sensordiagramm",machines_table_title:"Maschinen",machine_code_col:"Code",machine_name_col:"Name",status_col:"Status",risk_col:"Risiko",no_machines_yet:"Noch keine Maschinen. Klicken Sie auf „+ Maschine Hinzufügen“.",section_machine_info:"Maschineninformationen",section_sensor_data:"Sensordaten",section_status:"Status",section_notes:"Notizen",status_stopped:"Gestoppt",status_maintenance:"Wartung",priority_low:"Niedrig",priority_normal:"Normal",priority_high:"Hoch",priority_critical:"Kritisch",save_and_analyze_btn:"Speichern & Analysieren",source_col:"Quelle",source_auto:"Auto (SCADA)",source_manual:"Manuell",nav_alerts:"Warnungen",acknowledge_btn:"Bestätigen",acknowledged_label:"Bestätigt",acknowledge_all_btn:"Alle Bestätigen",no_alerts_yet:"Keine Warnungen. Alles läuft reibungslos.",download_report_btn:"Bericht",alert_details_template:"Temperatur {temp}°C, Vibration {vib} mm/s, Status: {status}",section_energy_intel:"Energieintelligenz",daily_output_hint:"Wird zur Berechnung des spezifischen Energieverbrauchs verwendet (kWh pro Einheit).",energy_insights_title:"Energieintelligenz",idle_power_title:"Leerlaufenergie-Erkennung",idle_active_msg:"Maschine im Leerlauf – aktuell werden etwa {kw} kW verschwendet.",idle_none_msg:"Kein Leerlaufenergieverlust festgestellt.",friction_loss_title:"Vorausschauender Energieverlust",friction_active_msg:"Erhöhte Reibung festgestellt: +{pct}% Mehrleistung (~{kw} kW extra). Wartung einplanen, um Verluste zu vermeiden.",friction_none_msg:"Keine abnormale Reibung festgestellt.",sec_title:"Spezifischer Energieverbrauch",sec_label:"kWh pro Einheit",sec_unit:"kWh/Einheit",sec_no_data_msg:"Geben Sie die Tagesproduktion beim Hinzufügen der Maschine an, um diese Kennzahl zu sehen.",optimal_load_title:"Optimale Lastzone",optimal_load_label:"Optimale Last",current_load_label:"Aktuelle Last",at_optimal_msg:"Läuft in der optimalen Lastzone.",adjust_to_optimal_msg:"Last auf {pct}% anpassen, um den Energieverbrauch pro Einheit zu minimieren.",nav_digital_twin:"Digitaler Zwilling",twin_hint:"Ziehen zum Drehen, Scrollen zum Zoomen, Maschine anklicken für Live-Details.",twin_unavailable_msg:"3D-Ansicht konnte nicht geladen werden (Internetverbindung für Three.js prüfen).",failure_prediction_title:"Ausfallvorhersage",report_lib_missing_msg:"Für den PDF-Export wird die reportlab-Bibliothek benötigt. Führen Sie aus: pip install reportlab und starten Sie den Server neu.",ph_machine_id:"Maschinen-ID (z.B. M-01)",ph_machine_name:"Maschinenname",ph_factory_section:"Fabrikbereich",ph_operator_name:"Bedienername",ph_pressure:"Druck (bar)",ph_voltage:"Spannung (V)",ph_current:"Strom (A)",ph_error_code:"Fehlercode",ph_daily_output:"Tagesproduktion (Einheiten)",ph_notes:"Notizen...",nav_system_intel:"Systemintelligenz",nav_roi:"ROI-Dashboard",refresh_btn:"Aktualisieren",system_risk_label:"Systemrisiko",healthy_label:"Gesund",at_risk_label:"Gefährdet",clusters_title:"Maschinencluster",propagation_title:"Anomalie-Ausbreitung",propagation_hint:"Wie eine ausfallende Maschine das Risiko ihrer Nachbarn erhöht.",no_propagation:"Keine Anomalie-Ausbreitung erkannt.",avg_risk_label:"Durchschn. Risiko",added_risk_label:"Zusätzliches Risiko",effective_risk_label:"Effektives Risiko",simulation_title:"Was-wäre-wenn-Simulation",simulation_hint:"Maschine wählen, Sensorwerte verschieben und die Ausfallwahrscheinlichkeit beobachten.",run_simulation_btn:"Simulation starten",failure_probability_label:"Ausfallwahrscheinlichkeit",stress_level_label:"Belastungsgrad",predicted_status_label:"Prognostizierter Status",confidence_label:"Konfidenz",root_cause_title:"Grundursache",rul_col:"Restnutzungsdauer",rul_healthy:"Gesund",potential_loss_label:"Potenzieller Verlust",saved_label:"Durch KI gespart",wasted_energy_label:"Energieverlust / Monat",efficiency_gain_label:"Effizienzgewinn",cost_by_machine_title:"Kostenrisiko pro Maschine",top_cause_col:"Hauptursache",roi_assumptions_msg:"Annahmen: Ausfall {downtime}/h, {hours}h Reparatur, {price}/kWh Energie.",role_label:"Ihre Rolle",role_engineer:"Ingenieur",role_manager:"Manager",role_admin:"Administrator",cause_bearing_wear:"Lagerverschleiß",cause_overload_thermal:"Thermische Überlastung",cause_cooling_failure:"Kühlungsausfall",cause_misalignment:"Wellenversatz",cause_lubrication_loss:"Schmierverlust",cause_normal_operation:"Normalbetrieb",nav_story:"Story-Modus",story_hint:"Spielt einen über 22 Stunden entstehenden Lagerschaden nach und zeigt genau, wann die KI ihn erkannte — und was das wert war.",simulate_failure_btn:"Ausfall Simulieren",outcome_title:"Ergebnis",warning_time_label:"Frühwarnung",loss_ignored_label:"Verlust ohne Reaktion",loss_acted_label:"Verlust bei Reaktion",money_saved_label:"Gespart",timeline_title:"Ausfall-Zeitleiste",story_detection_msg:"Die KI meldete dies {hours} Stunden vor dem Ausfall bei {risk}% Risiko — Grundursache: {cause} ({confidence}% Konfidenz).",story_stage_healthy:"Gesund",story_stage_early_drift:"Erste Abweichung",story_stage_ai_detects:"KI Erkennt",story_stage_critical:"Kritisch",priority_low:"Niedrig",priority_medium:"Mittel",priority_high:"Hoch",priority_critical:"Kritisch",action_stop_machine:"Maschine stoppen",action_inspect_bearings:"Lager prüfen",action_check_cooling:"Kühlung prüfen",action_schedule_shutdown_24h:"Abschaltung planen (24h)",action_order_spare_parts:"Ersatzteile bestellen",action_reduce_load:"Last reduzieren",action_schedule_inspection_72h:"Inspektion planen (72h)",action_monitor_closely:"Genau beobachten",action_verify_sensor:"Sensor prüfen",action_power_down_idle:"Leerlaufmaschine abschalten",action_review_shift_schedule:"Schichtplan überprüfen",action_no_action:"Keine Maßnahme nötig",nav_oee:"OEE",nav_workorders:"Aufträge",oee_hint:"Verfügbarkeit x Leistung x Qualität (ISO 22400). Weltklasse ist 85%, eine typische Fabrik liegt bei etwa 60%.",availability_label:"Verfügbarkeit",performance_label:"Leistung",quality_label:"Qualität",weakest_factor_label:"Schwächster",downtime_by_reason_title:"Ausfallzeit nach Grund",downtime_cost_label:"Ausfallkosten",oee_trend_title:"OEE-Trend",shifts_title:"Schichten",shift_col:"Schicht",downtime_col:"Ausfallzeit",log_shift_btn:"Schicht erfassen",no_shifts_yet:"Noch keine Schichten erfasst.",minutes_short:"Min",range_1d:"Heute",range_7d:"7 Tage",range_30d:"30 Tage",all_machines_option:"Alle Maschinen",reason_unspecified:"Nicht angegeben",err_good_exceeds_total:"Gutteile können die Gesamtmenge nicht überschreiten",oee_grade_world_class:"Weltklasse",oee_grade_typical:"Typisch",oee_grade_low:"Niedrig",oee_grade_critical:"Kritisch",reason_breakdown:"Störung",reason_changeover:"Rüsten",reason_no_material:"Kein Material",reason_no_operator:"Kein Bediener",reason_planned_maintenance:"Geplante Wartung",reason_quality_issue:"Qualitätsproblem",reason_setup:"Einrichtung",reason_other:"Sonstiges",workorders_hint:"Macht aus einer KI-Prognose eine nachverfolgbare Aufgabe — damit eine Warnung in einer Reparatur endet.",new_work_order_btn:"+ Neuer Auftrag",no_work_orders:"Noch keine Aufträge.",avg_completion_label:"Ø Bearbeitung",assigned_label:"Zugewiesen",source_ai:"KI",wo_status_open:"Offen",wo_status_in_progress:"In Arbeit",wo_status_done:"Erledigt",wo_status_cancelled:"Storniert",wo_advance_to_in_progress:"Arbeit beginnen",wo_advance_to_done:"Als erledigt markieren",ph_shift_name:"Schichtname",ph_planned_minutes:"Geplante Minuten",ph_downtime_minutes:"Ausfallminuten",ph_total_units:"Gesamtstückzahl",ph_good_units:"Gutteile",ph_cycle_seconds:"Idealzyklus (Sek.)",ph_wo_title:"Titel",ph_assigned_to:"Zugewiesen an",ph_wo_description:"Beschreibung",overdue_label:"Überfällig",nav_history:"Verlauf",history_hint:"Gespeicherte Sensorhistorie — damit belegen Sie, was sich im Pilot wirklich verändert hat.",no_history_yet:"Noch kein Verlauf aufgezeichnet. Lassen Sie das Live-Monitoring einige Minuten offen, dann sammeln sich Daten.",range_24h:"24 Stunden",range_3d:"3 Tage",range_10d:"10 Tage",sensor_trend_title:"Sensorverlauf",risk_trend_title:"Risikoverlauf",trend_flat:"unverändert",trend_rising:"steigend",trend_falling:"fallend",create_work_order_btn:"Auftrag erstellen",work_order_created_msg:"Auftrag erstellt",alertreason_immediate_failure_risk:"Unmittelbares Ausfallrisiko",alertreason_failure_imminent:"Ausfall unmittelbar bevorstehend",alertreason_degradation_accelerating:"Verschleiß beschleunigt sich",alertreason_outside_normal_envelope:"Außerhalb des Normalbereichs",alertreason_pressure_anomaly:"Druckanomalie",alertreason_idle_waste:"Leerlauf-Energieverlust",alertreason_informational:"Informativ",alert_reason_label:"Grund"},
  fr: {tagline:"Plateforme mondiale d'intelligence industrielle",live_label:"En direct",kpi_energy:"Consommation d'Énergie",kpi_efficiency:"Efficacité",kpi_active:"Machines Actives",kpi_alerts:"Alertes",kwh_unit:"kWh",chart_title:"Performance en Temps Réel",machine_status_title:"État des Machines",status_running:"En marche",status_warning:"Avertissement",status_critical:"Critique",form_title:"Saisie des Données d'Usine",factory_name_label:"Nom de l'Usine",machine_count_label:"Nombre de Machines",energy_cost_label:"Coût de l'Énergie ($/kWh)",machine_type_label:"Type de Machine",temperature_label:"Température (°C)",vibration_label:"Vibration (mm/s)",load_label:"Charge (%)",submit_btn:"Analyser l'Usine",submitting:"Mise à jour...",ai_panel_title:"Analyses IA",ai_placeholder:"Envoyez les données de l'usine pour générer une analyse IA.",ai_analyzing:"Analyse en cours...",ai_risks:"Risques",ai_efficiency_insights:"Analyse d'Efficacité",ai_optimizations:"Suggestions d'Optimisation",toast_updated:"Données d'usine mises à jour",toast_analysis_done:"Analyse IA terminée",toast_error:"Une erreur est survenue",nav_dashboard:"Tableau de Bord",nav_factories:"Usines",nav_ai_insights:"Analyses IA",logout_btn:"Déconnexion",login_title:"Content de vous revoir",login_subtitle:"Connectez-vous à votre compte FactoryPulse AI",ph_email:"E-mail",ph_password:"Mot de passe",remember_me:"Se souvenir de moi",login_btn:"Se connecter",login_link_register:"Pas de compte ? Créez-en un",register_title:"Créer votre compte",register_subtitle:"Commencez à surveiller vos usines avec l'IA",ph_full_name:"Nom Complet",ph_confirm_password:"Confirmer le Mot de Passe",register_btn:"Créer un Compte",register_link_login:"Déjà un compte ? Se connecter",err_missing_fields:"Veuillez remplir tous les champs",err_invalid_email:"Veuillez entrer une adresse e-mail valide",err_weak_password:"Le mot de passe doit contenir 8 caractères min., une lettre et un chiffre",err_password_mismatch:"Les mots de passe ne correspondent pas",err_invalid_credentials:"E-mail ou mot de passe incorrect",err_email_taken:"Cet e-mail est déjà enregistré",err_generic:"Une erreur est survenue. Veuillez réessayer",my_factories_title:"Mes Usines",add_factory_btn:"+ Ajouter une Usine",edit_factory_btn:"Modifier",delete_factory_btn:"Supprimer",confirm_delete_factory:"Supprimer cette usine ? Cette action est irréversible.",no_factories_yet:"Vous n'avez pas encore ajouté d'usine.",factory_created_toast:"Usine créée et analysée",factory_updated_toast:"Usine mise à jour",factory_deleted_toast:"Usine supprimée",ai_insights_feed_title:"Flux d'Analyses IA",no_ai_insights_yet:"Aucune analyse IA pour l'instant. Ajoutez une usine.",reanalyze_btn:"Réanalyser",view_insights_btn:"Voir les Analyses",created_label:"Créée le",cancel_btn:"Annuler",save_btn:"Enregistrer les Modifications",nav_live_monitor:"Surveillance en Direct",add_machine_scada_btn:"+ Ajouter une Machine",usb_status:"USB :",plc_status:"API :",polling_mode:"Interrogation",live_chart_title:"Graphique des Capteurs en Direct",machines_table_title:"Machines",machine_code_col:"Code",machine_name_col:"Nom",status_col:"Statut",risk_col:"Risque",no_machines_yet:"Aucune machine pour l'instant. Cliquez sur « + Ajouter une Machine ».",section_machine_info:"Informations sur la Machine",section_sensor_data:"Données des Capteurs",section_status:"Statut",section_notes:"Notes",status_stopped:"Arrêtée",status_maintenance:"Maintenance",priority_low:"Faible",priority_normal:"Normale",priority_high:"Élevée",priority_critical:"Critique",save_and_analyze_btn:"Enregistrer et Analyser",source_col:"Source",source_auto:"Auto (SCADA)",source_manual:"Manuel",nav_alerts:"Alertes",acknowledge_btn:"Confirmer",acknowledged_label:"Confirmé",acknowledge_all_btn:"Tout Confirmer",no_alerts_yet:"Aucune alerte. Tout fonctionne normalement.",download_report_btn:"Rapport",alert_details_template:"Température {temp}°C, vibration {vib} mm/s, statut : {status}",section_energy_intel:"Intelligence Énergétique",daily_output_hint:"Utilisé pour calculer la consommation énergétique spécifique (kWh par unité).",energy_insights_title:"Intelligence Énergétique",idle_power_title:"Détection de Puissance au Ralenti",idle_active_msg:"Machine au ralenti - environ {kw} kW gaspillés actuellement.",idle_none_msg:"Aucun gaspillage d'énergie au ralenti détecté.",friction_loss_title:"Perte d'Énergie Prédictive",friction_active_msg:"Friction élevée détectée : +{pct}% de surcharge (~{kw} kW en plus). Planifiez une maintenance pour éviter les pertes.",friction_none_msg:"Aucune friction anormale détectée.",sec_title:"Consommation Énergétique Spécifique",sec_label:"kWh par unité",sec_unit:"kWh/unité",sec_no_data_msg:"Indiquez la production quotidienne lors de l'ajout de la machine pour voir cette mesure.",optimal_load_title:"Zone de Charge Optimale",optimal_load_label:"Charge optimale",current_load_label:"Charge actuelle",at_optimal_msg:"Fonctionne dans la zone de charge optimale.",adjust_to_optimal_msg:"Ajustez la charge vers {pct}% pour minimiser l'énergie par unité.",nav_digital_twin:"Jumeau Numérique",twin_hint:"Faites glisser pour pivoter, défilez pour zoomer, cliquez sur une machine pour ses détails en direct.",twin_unavailable_msg:"Impossible de charger la vue 3D (vérifiez votre connexion internet pour Three.js).",failure_prediction_title:"Prédiction de Panne",report_lib_missing_msg:"L'export PDF nécessite la bibliothèque reportlab. Exécutez : pip install reportlab, puis redémarrez le serveur.",ph_machine_id:"ID Machine (ex. M-01)",ph_machine_name:"Nom de la Machine",ph_factory_section:"Section de l'Usine",ph_operator_name:"Nom de l'Opérateur",ph_pressure:"Pression (bar)",ph_voltage:"Tension (V)",ph_current:"Courant (A)",ph_error_code:"Code d'Erreur",ph_daily_output:"Production Quotidienne (unités)",ph_notes:"Notes...",nav_system_intel:"Intelligence Système",nav_roi:"Tableau ROI",refresh_btn:"Actualiser",system_risk_label:"Risque Système",healthy_label:"Sains",at_risk_label:"À Risque",clusters_title:"Clusters de Machines",propagation_title:"Propagation d'Anomalies",propagation_hint:"Comment une machine défaillante augmente le risque de ses voisines.",no_propagation:"Aucune propagation d'anomalie détectée.",avg_risk_label:"Risque moyen",added_risk_label:"Risque ajouté",effective_risk_label:"Risque effectif",simulation_title:"Simulation Hypothétique",simulation_hint:"Choisissez une machine, modifiez ses capteurs et observez la probabilité de panne.",run_simulation_btn:"Lancer la Simulation",failure_probability_label:"Probabilité de Panne",stress_level_label:"Niveau de Contrainte",predicted_status_label:"Statut Prévu",confidence_label:"Confiance",root_cause_title:"Cause Racine",rul_col:"Durée de Vie Restante",rul_healthy:"Sain",potential_loss_label:"Perte Potentielle",saved_label:"Économisé par l'IA",wasted_energy_label:"Énergie Gaspillée / mois",efficiency_gain_label:"Gain d'Efficacité",cost_by_machine_title:"Exposition Financière par Machine",top_cause_col:"Cause Principale",roi_assumptions_msg:"Hypothèses : arrêt {downtime}/h, réparation {hours}h, énergie {price}/kWh.",role_label:"Votre Rôle",role_engineer:"Ingénieur",role_manager:"Manager",role_admin:"Administrateur",cause_bearing_wear:"Usure de roulement",cause_overload_thermal:"Surcharge thermique",cause_cooling_failure:"Panne de refroidissement",cause_misalignment:"Désalignement d'arbre",cause_lubrication_loss:"Perte de lubrification",cause_normal_operation:"Fonctionnement normal",nav_story:"Mode Récit",story_hint:"Rejoue une panne de roulement se développant sur 22 heures et montre exactement quand l'IA l'a détectée — et ce que cela valait.",simulate_failure_btn:"Simuler une Panne",outcome_title:"Résultat",warning_time_label:"Alerte Précoce",loss_ignored_label:"Perte sans Réaction",loss_acted_label:"Perte avec Réaction",money_saved_label:"Économisé",timeline_title:"Chronologie de la Panne",story_detection_msg:"L'IA l'a signalé {hours} heures avant la panne, à {risk}% de risque — cause racine : {cause} (confiance {confidence}%).",story_stage_healthy:"Sain",story_stage_early_drift:"Première Dérive",story_stage_ai_detects:"L'IA Détecte",story_stage_critical:"Critique",priority_low:"Faible",priority_medium:"Moyen",priority_high:"Élevé",priority_critical:"Critique",action_stop_machine:"Arrêter la machine",action_inspect_bearings:"Inspecter les roulements",action_check_cooling:"Vérifier le refroidissement",action_schedule_shutdown_24h:"Planifier l'arrêt (24h)",action_order_spare_parts:"Commander des pièces",action_reduce_load:"Réduire la charge",action_schedule_inspection_72h:"Planifier l'inspection (72h)",action_monitor_closely:"Surveiller de près",action_verify_sensor:"Vérifier le capteur",action_power_down_idle:"Éteindre la machine au ralenti",action_review_shift_schedule:"Revoir le planning",action_no_action:"Aucune action requise",nav_oee:"TRS",nav_workorders:"Ordres de Travail",oee_hint:"Disponibilité x Performance x Qualité (ISO 22400). Le niveau mondial est 85%, une usine typique avoisine 60%.",availability_label:"Disponibilité",performance_label:"Performance",quality_label:"Qualité",weakest_factor_label:"Le plus faible",downtime_by_reason_title:"Arrêts par Cause",downtime_cost_label:"Coût d'arrêt",oee_trend_title:"Tendance TRS",shifts_title:"Équipes",shift_col:"Équipe",downtime_col:"Arrêt",log_shift_btn:"Saisir une Équipe",no_shifts_yet:"Aucune équipe saisie.",minutes_short:"min",range_1d:"Aujourd'hui",range_7d:"7 jours",range_30d:"30 jours",all_machines_option:"Toutes les machines",reason_unspecified:"Non spécifié",err_good_exceeds_total:"Les pièces bonnes ne peuvent dépasser le total",oee_grade_world_class:"Niveau mondial",oee_grade_typical:"Typique",oee_grade_low:"Faible",oee_grade_critical:"Critique",reason_breakdown:"Panne",reason_changeover:"Changement de série",reason_no_material:"Pas de matière",reason_no_operator:"Pas d'opérateur",reason_planned_maintenance:"Maintenance planifiée",reason_quality_issue:"Problème qualité",reason_setup:"Réglage",reason_other:"Autre",workorders_hint:"Transforme une prédiction IA en tâche suivie — pour qu'une alerte se termine par une réparation.",new_work_order_btn:"+ Nouvel Ordre",no_work_orders:"Aucun ordre de travail.",avg_completion_label:"Achèvement moy.",assigned_label:"Assigné",source_ai:"IA",wo_status_open:"Ouvert",wo_status_in_progress:"En cours",wo_status_done:"Terminé",wo_status_cancelled:"Annulé",wo_advance_to_in_progress:"Commencer",wo_advance_to_done:"Marquer terminé",ph_shift_name:"Nom de l'équipe",ph_planned_minutes:"Minutes planifiées",ph_downtime_minutes:"Minutes d'arrêt",ph_total_units:"Unités totales",ph_good_units:"Unités bonnes",ph_cycle_seconds:"Cycle idéal (sec)",ph_wo_title:"Titre",ph_assigned_to:"Assigné à",ph_wo_description:"Description",overdue_label:"En retard",nav_history:"Historique",history_hint:"Historique des capteurs enregistré — c'est ainsi que vous prouvez ce qui a réellement changé pendant le pilote.",no_history_yet:"Aucun historique enregistré. Laissez le Monitoring ouvert quelques minutes et les données s'accumuleront.",range_24h:"24 heures",range_3d:"3 jours",range_10d:"10 jours",sensor_trend_title:"Tendance des Capteurs",risk_trend_title:"Tendance du Risque",trend_flat:"inchangé",trend_rising:"en hausse",trend_falling:"en baisse",create_work_order_btn:"Créer un Ordre",work_order_created_msg:"Ordre de travail créé",alertreason_immediate_failure_risk:"Risque de panne immédiat",alertreason_failure_imminent:"Panne imminente",alertreason_degradation_accelerating:"Dégradation qui s'accélère",alertreason_outside_normal_envelope:"Hors plage normale",alertreason_pressure_anomaly:"Anomalie de pression",alertreason_idle_waste:"Gaspillage au ralenti",alertreason_informational:"Informatif",alert_reason_label:"Motif"},
  es: {tagline:"Plataforma Global de Inteligencia Industrial",live_label:"En vivo",kpi_energy:"Uso de Energía",kpi_efficiency:"Eficiencia",kpi_active:"Máquinas Activas",kpi_alerts:"Alertas",kwh_unit:"kWh",chart_title:"Rendimiento en Tiempo Real",machine_status_title:"Estado de Máquinas",status_running:"Funcionando",status_warning:"Advertencia",status_critical:"Crítico",form_title:"Entrada de Datos de Fábrica",factory_name_label:"Nombre de Fábrica",machine_count_label:"Número de Máquinas",energy_cost_label:"Costo de Energía ($/kWh)",machine_type_label:"Tipo de Máquina",temperature_label:"Temperatura (°C)",vibration_label:"Vibración (mm/s)",load_label:"Carga (%)",submit_btn:"Analizar Fábrica",submitting:"Actualizando...",ai_panel_title:"Perspectivas IA",ai_placeholder:"Envíe datos de fábrica para generar un análisis IA.",ai_analyzing:"Analizando...",ai_risks:"Riesgos",ai_efficiency_insights:"Análisis de Eficiencia",ai_optimizations:"Sugerencias de Optimización",toast_updated:"Datos de fábrica actualizados",toast_analysis_done:"Análisis IA completo",toast_error:"Algo salió mal",nav_dashboard:"Panel",nav_factories:"Fábricas",nav_ai_insights:"Perspectivas IA",logout_btn:"Cerrar Sesión",login_title:"Bienvenido de nuevo",login_subtitle:"Inicia sesión en tu cuenta de FactoryPulse AI",ph_email:"Correo electrónico",ph_password:"Contraseña",remember_me:"Recuérdame",login_btn:"Iniciar Sesión",login_link_register:"¿No tienes cuenta? Crea una",register_title:"Crea tu cuenta",register_subtitle:"Empieza a monitorear tus fábricas con IA",ph_full_name:"Nombre Completo",ph_confirm_password:"Confirmar Contraseña",register_btn:"Crear Cuenta",register_link_login:"¿Ya tienes cuenta? Inicia sesión",err_missing_fields:"Por favor complete todos los campos",err_invalid_email:"Por favor ingrese un correo válido",err_weak_password:"La contraseña debe tener mín. 8 caracteres, una letra y un número",err_password_mismatch:"Las contraseñas no coinciden",err_invalid_credentials:"Correo o contraseña incorrectos",err_email_taken:"Este correo ya está registrado",err_generic:"Algo salió mal. Inténtalo de nuevo",my_factories_title:"Mis Fábricas",add_factory_btn:"+ Añadir Fábrica",edit_factory_btn:"Editar",delete_factory_btn:"Eliminar",confirm_delete_factory:"¿Eliminar esta fábrica? Esta acción no se puede deshacer.",no_factories_yet:"Aún no has añadido ninguna fábrica.",factory_created_toast:"Fábrica creada y analizada",factory_updated_toast:"Fábrica actualizada",factory_deleted_toast:"Fábrica eliminada",ai_insights_feed_title:"Feed de Perspectivas IA",no_ai_insights_yet:"Aún no hay perspectivas IA. Añade una fábrica.",reanalyze_btn:"Reanalizar",view_insights_btn:"Ver Perspectivas",created_label:"Creada",cancel_btn:"Cancelar",save_btn:"Guardar Cambios",nav_live_monitor:"Monitor en Vivo",add_machine_scada_btn:"+ Añadir Máquina",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Sondeo",live_chart_title:"Gráfico de Sensores en Vivo",machines_table_title:"Máquinas",machine_code_col:"Código",machine_name_col:"Nombre",status_col:"Estado",risk_col:"Riesgo",no_machines_yet:"Aún no hay máquinas. Haga clic en «+ Añadir Máquina».",section_machine_info:"Información de la Máquina",section_sensor_data:"Datos de Sensores",section_status:"Estado",section_notes:"Notas",status_stopped:"Detenida",status_maintenance:"Mantenimiento",priority_low:"Baja",priority_normal:"Normal",priority_high:"Alta",priority_critical:"Crítica",save_and_analyze_btn:"Guardar y Analizar",source_col:"Fuente",source_auto:"Auto (SCADA)",source_manual:"Manual",nav_alerts:"Alertas",acknowledge_btn:"Reconocer",acknowledged_label:"Reconocido",acknowledge_all_btn:"Reconocer Todo",no_alerts_yet:"Sin alertas. Todo funciona correctamente.",download_report_btn:"Informe",alert_details_template:"Temperatura {temp}°C, vibración {vib} mm/s, estado: {status}",section_energy_intel:"Inteligencia Energética",daily_output_hint:"Se usa para calcular el consumo energético específico (kWh por unidad).",energy_insights_title:"Inteligencia Energética",idle_power_title:"Detección de Potencia en Inactividad",idle_active_msg:"Máquina inactiva - se desperdician aprox. {kw} kW ahora mismo.",idle_none_msg:"No se detectó desperdicio de energía en inactividad.",friction_loss_title:"Pérdida de Energía Predictiva",friction_active_msg:"Fricción elevada detectada: +{pct}% de sobrecarga (~{kw} kW extra). Programe mantenimiento para evitar pérdidas.",friction_none_msg:"No se detectó fricción anormal.",sec_title:"Consumo Energético Específico",sec_label:"kWh por unidad",sec_unit:"kWh/unidad",sec_no_data_msg:"Ingrese la producción diaria al añadir esta máquina para ver esta métrica.",optimal_load_title:"Zona de Carga Óptima",optimal_load_label:"Carga óptima",current_load_label:"Carga actual",at_optimal_msg:"Funcionando en la zona de carga óptima.",adjust_to_optimal_msg:"Ajuste la carga hacia {pct}% para minimizar la energía por unidad.",nav_digital_twin:"Gemelo Digital",twin_hint:"Arrastre para rotar, desplácese para acercar, haga clic en una máquina para ver sus detalles en vivo.",twin_unavailable_msg:"No se pudo cargar la vista 3D (verifique su conexión a internet para Three.js).",failure_prediction_title:"Predicción de Fallo",report_lib_missing_msg:"La exportación a PDF necesita la librería reportlab. Ejecute: pip install reportlab, luego reinicie el servidor.",ph_machine_id:"ID de Máquina (ej. M-01)",ph_machine_name:"Nombre de la Máquina",ph_factory_section:"Sección de la Fábrica",ph_operator_name:"Nombre del Operador",ph_pressure:"Presión (bar)",ph_voltage:"Voltaje (V)",ph_current:"Corriente (A)",ph_error_code:"Código de Error",ph_daily_output:"Producción Diaria (unidades)",ph_notes:"Notas...",nav_system_intel:"Inteligencia del Sistema",nav_roi:"Panel ROI",refresh_btn:"Actualizar",system_risk_label:"Riesgo del Sistema",healthy_label:"Saludables",at_risk_label:"En Riesgo",clusters_title:"Clústeres de Máquinas",propagation_title:"Propagación de Anomalías",propagation_hint:"Cómo una máquina en fallo eleva el riesgo de sus vecinas.",no_propagation:"No se detectó propagación de anomalías.",avg_risk_label:"Riesgo medio",added_risk_label:"Riesgo añadido",effective_risk_label:"Riesgo efectivo",simulation_title:"Simulación Hipotética",simulation_hint:"Elija una máquina, ajuste sus sensores y vea cómo cambia la probabilidad de fallo.",run_simulation_btn:"Ejecutar Simulación",failure_probability_label:"Probabilidad de Fallo",stress_level_label:"Nivel de Estrés",predicted_status_label:"Estado Previsto",confidence_label:"Confianza",root_cause_title:"Causa Raíz",rul_col:"Vida Útil Restante",rul_healthy:"Saludable",potential_loss_label:"Pérdida Potencial",saved_label:"Ahorrado por IA",wasted_energy_label:"Energía Desperdiciada / mes",efficiency_gain_label:"Ganancia de Eficiencia",cost_by_machine_title:"Exposición de Costes por Máquina",top_cause_col:"Causa Principal",roi_assumptions_msg:"Supuestos: parada {downtime}/h, reparación {hours}h, energía {price}/kWh.",role_label:"Su Rol",role_engineer:"Ingeniero",role_manager:"Gerente",role_admin:"Administrador",cause_bearing_wear:"Desgaste de rodamiento",cause_overload_thermal:"Sobrecarga térmica",cause_cooling_failure:"Fallo de refrigeración",cause_misalignment:"Desalineación del eje",cause_lubrication_loss:"Pérdida de lubricación",cause_normal_operation:"Operación normal",nav_story:"Modo Historia",story_hint:"Reproduce un fallo de rodamiento desarrollándose en 22 horas y muestra exactamente cuándo lo detectó la IA — y cuánto valía.",simulate_failure_btn:"Simular Fallo",outcome_title:"Resultado",warning_time_label:"Aviso Temprano",loss_ignored_label:"Pérdida sin Reacción",loss_acted_label:"Pérdida con Reacción",money_saved_label:"Ahorrado",timeline_title:"Cronología del Fallo",story_detection_msg:"La IA lo detectó {hours} horas antes de la avería, con {risk}% de riesgo — causa raíz: {cause} ({confidence}% de confianza).",story_stage_healthy:"Saludable",story_stage_early_drift:"Primera Desviación",story_stage_ai_detects:"La IA Detecta",story_stage_critical:"Crítico",priority_low:"Bajo",priority_medium:"Medio",priority_high:"Alto",priority_critical:"Crítico",action_stop_machine:"Detener máquina",action_inspect_bearings:"Inspeccionar rodamientos",action_check_cooling:"Revisar refrigeración",action_schedule_shutdown_24h:"Programar parada (24h)",action_order_spare_parts:"Pedir repuestos",action_reduce_load:"Reducir carga",action_schedule_inspection_72h:"Programar inspección (72h)",action_monitor_closely:"Vigilar de cerca",action_verify_sensor:"Verificar sensor",action_power_down_idle:"Apagar máquina inactiva",action_review_shift_schedule:"Revisar turnos",action_no_action:"No se requiere acción",nav_oee:"OEE",nav_workorders:"Órdenes de Trabajo",oee_hint:"Disponibilidad x Rendimiento x Calidad (ISO 22400). El nivel mundial es 85%; una fábrica típica ronda el 60%.",availability_label:"Disponibilidad",performance_label:"Rendimiento",quality_label:"Calidad",weakest_factor_label:"Más débil",downtime_by_reason_title:"Paradas por Causa",downtime_cost_label:"Costo de parada",oee_trend_title:"Tendencia OEE",shifts_title:"Turnos",shift_col:"Turno",downtime_col:"Parada",log_shift_btn:"Registrar Turno",no_shifts_yet:"Aún no hay turnos registrados.",minutes_short:"min",range_1d:"Hoy",range_7d:"7 días",range_30d:"30 días",all_machines_option:"Todas las máquinas",reason_unspecified:"Sin especificar",err_good_exceeds_total:"Las piezas buenas no pueden superar el total",oee_grade_world_class:"Nivel mundial",oee_grade_typical:"Típico",oee_grade_low:"Bajo",oee_grade_critical:"Crítico",reason_breakdown:"Avería",reason_changeover:"Cambio de formato",reason_no_material:"Sin material",reason_no_operator:"Sin operario",reason_planned_maintenance:"Mantenimiento planificado",reason_quality_issue:"Problema de calidad",reason_setup:"Preparación",reason_other:"Otro",workorders_hint:"Convierte una predicción de IA en tarea rastreable — para que una alerta acabe en reparación.",new_work_order_btn:"+ Nueva Orden",no_work_orders:"Aún no hay órdenes.",avg_completion_label:"Finalización prom.",assigned_label:"Asignado",source_ai:"IA",wo_status_open:"Abierta",wo_status_in_progress:"En curso",wo_status_done:"Completada",wo_status_cancelled:"Cancelada",wo_advance_to_in_progress:"Iniciar trabajo",wo_advance_to_done:"Marcar completada",ph_shift_name:"Nombre del turno",ph_planned_minutes:"Minutos planificados",ph_downtime_minutes:"Minutos de parada",ph_total_units:"Unidades totales",ph_good_units:"Unidades buenas",ph_cycle_seconds:"Ciclo ideal (seg)",ph_wo_title:"Título",ph_assigned_to:"Asignado a",ph_wo_description:"Descripción",overdue_label:"Vencidas",nav_history:"Historial",history_hint:"Historial de sensores guardado: así demuestra qué cambió realmente durante el piloto.",no_history_yet:"Aún no hay historial. Mantenga el Monitoreo abierto unos minutos y los datos empezarán a acumularse.",range_24h:"24 horas",range_3d:"3 días",range_10d:"10 días",sensor_trend_title:"Tendencia de Sensores",risk_trend_title:"Tendencia de Riesgo",trend_flat:"sin cambios",trend_rising:"subiendo",trend_falling:"bajando",create_work_order_btn:"Crear Orden",work_order_created_msg:"Orden de trabajo creada",alertreason_immediate_failure_risk:"Riesgo de fallo inmediato",alertreason_failure_imminent:"Fallo inminente",alertreason_degradation_accelerating:"Degradación acelerándose",alertreason_outside_normal_envelope:"Fuera del rango normal",alertreason_pressure_anomaly:"Anomalía de presión",alertreason_idle_waste:"Desperdicio en inactividad",alertreason_informational:"Informativo",alert_reason_label:"Motivo"},
  zh: {tagline:"全球工业智能平台",live_label:"实时",kpi_energy:"能源使用量",kpi_efficiency:"效率",kpi_active:"运行中设备",kpi_alerts:"警报",kwh_unit:"kWh",chart_title:"实时性能",machine_status_title:"设备状态",status_running:"运行中",status_warning:"警告",status_critical:"严重",form_title:"工厂数据输入",factory_name_label:"工厂名称",machine_count_label:"设备数量",energy_cost_label:"能源成本 ($/kWh)",machine_type_label:"设备类型",temperature_label:"温度 (°C)",vibration_label:"振动 (mm/s)",load_label:"负载 (%)",submit_btn:"分析工厂",submitting:"更新中...",ai_panel_title:"AI 洞察",ai_placeholder:"提交工厂数据以生成AI分析。",ai_analyzing:"分析中...",ai_risks:"风险",ai_efficiency_insights:"效率分析",ai_optimizations:"优化建议",toast_updated:"工厂数据已更新",toast_analysis_done:"AI分析已完成",toast_error:"出现错误",nav_dashboard:"仪表盘",nav_factories:"工厂",nav_ai_insights:"AI洞察",logout_btn:"退出",login_title:"欢迎回来",login_subtitle:"登录您的 FactoryPulse AI 账户",ph_email:"电子邮件",ph_password:"密码",remember_me:"记住我",login_btn:"登录",login_link_register:"没有账户？创建一个",register_title:"创建账户",register_subtitle:"开始使用AI监控您的工厂",ph_full_name:"全名",ph_confirm_password:"确认密码",register_btn:"创建账户",register_link_login:"已有账户？登录",err_missing_fields:"请填写所有字段",err_invalid_email:"请输入有效的电子邮件地址",err_weak_password:"密码至少8位，需包含字母和数字",err_password_mismatch:"两次密码不一致",err_invalid_credentials:"电子邮件或密码错误",err_email_taken:"该电子邮件已被注册",err_generic:"出现错误，请重试",my_factories_title:"我的工厂",add_factory_btn:"+ 添加工厂",edit_factory_btn:"编辑",delete_factory_btn:"删除",confirm_delete_factory:"删除此工厂？此操作无法撤销。",no_factories_yet:"您还没有添加任何工厂。",factory_created_toast:"工厂已创建并分析",factory_updated_toast:"工厂已更新",factory_deleted_toast:"工厂已删除",ai_insights_feed_title:"AI洞察动态",no_ai_insights_yet:"暂无AI洞察。请添加工厂开始。",reanalyze_btn:"重新分析",view_insights_btn:"查看洞察",created_label:"创建于",cancel_btn:"取消",save_btn:"保存更改",nav_live_monitor:"实时监控",add_machine_scada_btn:"+ 添加设备",usb_status:"USB:",plc_status:"PLC:",polling_mode:"轮询",live_chart_title:"实时传感器图表",machines_table_title:"设备",machine_code_col:"编号",machine_name_col:"名称",status_col:"状态",risk_col:"风险",no_machines_yet:"暂无设备。点击“+ 添加设备”。",section_machine_info:"设备信息",section_sensor_data:"传感器数据",section_status:"状态",section_notes:"备注",status_stopped:"已停止",status_maintenance:"维护中",priority_low:"低",priority_normal:"正常",priority_high:"高",priority_critical:"严重",save_and_analyze_btn:"保存并分析",source_col:"数据来源",source_auto:"自动 (SCADA)",source_manual:"手动",nav_alerts:"警报",acknowledge_btn:"确认",acknowledged_label:"已确认",acknowledge_all_btn:"全部确认",no_alerts_yet:"暂无警报，一切运行正常。",download_report_btn:"报告",alert_details_template:"温度 {temp}°C，振动 {vib} mm/s，状态：{status}",section_energy_intel:"能源智能",daily_output_hint:"用于计算单位能耗（kWh/单位）。",energy_insights_title:"能源智能",idle_power_title:"空转功率检测",idle_active_msg:"设备处于空转状态 — 目前大约浪费 {kw} kW。",idle_none_msg:"未检测到空转能耗浪费。",friction_loss_title:"预测性能量损耗",friction_active_msg:"检测到摩擦增加：额外功率 +{pct}%（约 {kw} kW）。请安排维护以避免损耗。",friction_none_msg:"未检测到异常摩擦。",sec_title:"单位能耗",sec_label:"每单位kWh",sec_unit:"kWh/单位",sec_no_data_msg:"添加设备时请输入日产量以查看此指标。",optimal_load_title:"最佳负载区间",optimal_load_label:"最佳负载",current_load_label:"当前负载",at_optimal_msg:"正在最佳负载区间运行。",adjust_to_optimal_msg:"将负载调整至 {pct}% 以最小化单位能耗。",nav_digital_twin:"数字孪生",twin_hint:"拖动旋转，滚动缩放，点击设备查看实时详情。",twin_unavailable_msg:"无法加载3D视图（请检查Three.js库的网络连接）。",failure_prediction_title:"故障预测",report_lib_missing_msg:"PDF导出需要reportlab库。请运行：pip install reportlab，然后重启服务器。",ph_machine_id:"设备编号（如 M-01）",ph_machine_name:"设备名称",ph_factory_section:"工厂车间",ph_operator_name:"操作员姓名",ph_pressure:"压力（bar）",ph_voltage:"电压（V）",ph_current:"电流（A）",ph_error_code:"错误代码",ph_daily_output:"日产量（单位）",ph_notes:"备注...",nav_system_intel:"系统智能",nav_roi:"ROI 仪表板",refresh_btn:"刷新",system_risk_label:"系统风险",healthy_label:"健康",at_risk_label:"有风险",clusters_title:"设备集群",propagation_title:"异常传播",propagation_hint:"一台故障设备如何提高相邻设备的风险。",no_propagation:"未检测到异常传播。",avg_risk_label:"平均风险",added_risk_label:"附加风险",effective_risk_label:"实际风险",simulation_title:"假设模拟",simulation_hint:"选择设备，调整传感器数值，观察故障概率的变化。",run_simulation_btn:"运行模拟",failure_probability_label:"故障概率",stress_level_label:"应力水平",predicted_status_label:"预测状态",confidence_label:"置信度",root_cause_title:"根本原因",rul_col:"剩余寿命",rul_healthy:"健康",potential_loss_label:"潜在损失",saved_label:"AI 节省",wasted_energy_label:"浪费能源 / 月",efficiency_gain_label:"效率提升",cost_by_machine_title:"各设备成本风险",top_cause_col:"主要原因",roi_assumptions_msg:"假设：停机 {downtime}/小时，维修 {hours} 小时，能源 {price}/kWh。",role_label:"您的角色",role_engineer:"工程师",role_manager:"经理",role_admin:"管理员",cause_bearing_wear:"轴承磨损",cause_overload_thermal:"热过载",cause_cooling_failure:"冷却故障",cause_misalignment:"轴不对中",cause_lubrication_loss:"润滑损失",cause_normal_operation:"正常运行",nav_story:"故事模式",story_hint:"重现22小时内轴承故障的发展过程，精确展示AI何时发现它——以及这价值多少。",simulate_failure_btn:"模拟故障",outcome_title:"结果",warning_time_label:"提前预警",loss_ignored_label:"不作为的损失",loss_acted_label:"采取行动的损失",money_saved_label:"节省金额",timeline_title:"故障时间线",story_detection_msg:"AI在故障前 {hours} 小时发现，风险 {risk}%——根本原因：{cause}（置信度 {confidence}%）。",story_stage_healthy:"健康",story_stage_early_drift:"初期偏移",story_stage_ai_detects:"AI检测到",story_stage_critical:"严重",priority_low:"低",priority_medium:"中",priority_high:"高",priority_critical:"严重",action_stop_machine:"停止设备",action_inspect_bearings:"检查轴承",action_check_cooling:"检查冷却",action_schedule_shutdown_24h:"安排停机（24小时）",action_order_spare_parts:"订购备件",action_reduce_load:"降低负载",action_schedule_inspection_72h:"安排检查（72小时）",action_monitor_closely:"密切监控",action_verify_sensor:"验证传感器",action_power_down_idle:"关闭空转设备",action_review_shift_schedule:"检查班次安排",action_no_action:"无需操作",nav_oee:"OEE",nav_workorders:"工单",oee_hint:"可用率 x 表现性 x 质量 (ISO 22400)。世界级为85%，典型工厂约60%。",availability_label:"可用率",performance_label:"表现性",quality_label:"质量",weakest_factor_label:"最弱项",downtime_by_reason_title:"停机原因分析",downtime_cost_label:"停机成本",oee_trend_title:"OEE趋势",shifts_title:"班次",shift_col:"班次",downtime_col:"停机",log_shift_btn:"记录班次",no_shifts_yet:"尚未记录班次。",minutes_short:"分钟",range_1d:"今天",range_7d:"7天",range_30d:"30天",all_machines_option:"所有设备",reason_unspecified:"未指定",err_good_exceeds_total:"合格品不能超过总数",oee_grade_world_class:"世界级",oee_grade_typical:"典型",oee_grade_low:"低",oee_grade_critical:"严重",reason_breakdown:"故障",reason_changeover:"换型",reason_no_material:"缺料",reason_no_operator:"缺人",reason_planned_maintenance:"计划维护",reason_quality_issue:"质量问题",reason_setup:"调试",reason_other:"其他",workorders_hint:"将AI预测转化为可跟踪的任务——让预警真正以维修告终。",new_work_order_btn:"+ 新建工单",no_work_orders:"暂无工单。",avg_completion_label:"平均完成",assigned_label:"负责人",source_ai:"AI",wo_status_open:"待处理",wo_status_in_progress:"进行中",wo_status_done:"已完成",wo_status_cancelled:"已取消",wo_advance_to_in_progress:"开始工作",wo_advance_to_done:"标记完成",ph_shift_name:"班次名称",ph_planned_minutes:"计划分钟",ph_downtime_minutes:"停机分钟",ph_total_units:"总数量",ph_good_units:"合格数量",ph_cycle_seconds:"理想节拍（秒）",ph_wo_title:"标题",ph_assigned_to:"负责人",ph_wo_description:"描述",overdue_label:"逾期",nav_history:"历史",history_hint:"已存储的传感器历史——用它证明试点期间到底改变了什么。",no_history_yet:"尚未记录历史。保持实时监控页面打开几分钟，数据就会开始累积。",range_24h:"24小时",range_3d:"3天",range_10d:"10天",sensor_trend_title:"传感器趋势",risk_trend_title:"风险趋势",trend_flat:"无变化",trend_rising:"上升",trend_falling:"下降",create_work_order_btn:"创建工单",work_order_created_msg:"工单已创建",alertreason_immediate_failure_risk:"立即故障风险",alertreason_failure_imminent:"故障迫在眉睫",alertreason_degradation_accelerating:"劣化加速",alertreason_outside_normal_envelope:"超出正常范围",alertreason_pressure_anomaly:"压力异常",alertreason_idle_waste:"空转能耗浪费",alertreason_informational:"提示信息",alert_reason_label:"原因"},
  ar: {tagline:"منصة الذكاء الصناعي العالمية",live_label:"مباشر",kpi_energy:"استهلاك الطاقة",kpi_efficiency:"الكفاءة",kpi_active:"الآلات النشطة",kpi_alerts:"التنبيهات",kwh_unit:"kWh",chart_title:"الأداء في الوقت الفعلي",machine_status_title:"حالة الآلات",status_running:"تعمل",status_warning:"تحذير",status_critical:"حرج",form_title:"إدخال بيانات المصنع",factory_name_label:"اسم المصنع",machine_count_label:"عدد الآلات",energy_cost_label:"تكلفة الطاقة ($/kWh)",machine_type_label:"نوع الآلة",temperature_label:"درجة الحرارة (°C)",vibration_label:"الاهتزاز (مم/ث)",load_label:"الحمل (%)",submit_btn:"تحليل المصنع",submitting:"جارٍ التحديث...",ai_panel_title:"رؤى الذكاء الاصطناعي",ai_placeholder:"أرسل بيانات المصنع لإنشاء تحليل بالذكاء الاصطناعي.",ai_analyzing:"جارٍ التحليل...",ai_risks:"المخاطر",ai_efficiency_insights:"تحليل الكفاءة",ai_optimizations:"اقتراحات التحسين",toast_updated:"تم تحديث بيانات المصنع",toast_analysis_done:"اكتمل تحليل الذكاء الاصطناعي",toast_error:"حدث خطأ ما",nav_dashboard:"لوحة التحكم",nav_factories:"المصانع",nav_ai_insights:"رؤى الذكاء الاصطناعي",logout_btn:"تسجيل الخروج",login_title:"مرحباً بعودتك",login_subtitle:"سجل الدخول إلى حساب FactoryPulse AI الخاص بك",ph_email:"البريد الإلكتروني",ph_password:"كلمة المرور",remember_me:"تذكرني",login_btn:"تسجيل الدخول",login_link_register:"ليس لديك حساب؟ أنشئ واحداً",register_title:"إنشاء حسابك",register_subtitle:"ابدأ بمراقبة مصانعك بالذكاء الاصطناعي",ph_full_name:"الاسم الكامل",ph_confirm_password:"تأكيد كلمة المرور",register_btn:"إنشاء حساب",register_link_login:"لديك حساب بالفعل؟ سجل الدخول",err_missing_fields:"يرجى ملء جميع الحقول",err_invalid_email:"يرجى إدخال بريد إلكتروني صالح",err_weak_password:"يجب أن تكون كلمة المرور 8 أحرف على الأقل وتحتوي على حرف ورقم",err_password_mismatch:"كلمتا المرور غير متطابقتين",err_invalid_credentials:"البريد الإلكتروني أو كلمة المرور غير صحيحة",err_email_taken:"هذا البريد الإلكتروني مسجل بالفعل",err_generic:"حدث خطأ ما. يرجى المحاولة مرة أخرى",my_factories_title:"مصانعي",add_factory_btn:"+ إضافة مصنع",edit_factory_btn:"تعديل",delete_factory_btn:"حذف",confirm_delete_factory:"هل تريد حذف هذا المصنع؟ لا يمكن التراجع عن هذا.",no_factories_yet:"لم تقم بإضافة أي مصنع بعد.",factory_created_toast:"تم إنشاء المصنع وتحليله",factory_updated_toast:"تم تحديث المصنع",factory_deleted_toast:"تم حذف المصنع",ai_insights_feed_title:"موجز رؤى الذكاء الاصطناعي",no_ai_insights_yet:"لا توجد رؤى بعد. أضف مصنعاً للبدء.",reanalyze_btn:"إعادة التحليل",view_insights_btn:"عرض الرؤى",created_label:"تاريخ الإنشاء",cancel_btn:"إلغاء",save_btn:"حفظ التغييرات",nav_live_monitor:"المراقبة المباشرة",add_machine_scada_btn:"+ إضافة آلة",usb_status:"USB:",plc_status:"PLC:",polling_mode:"استطلاع",live_chart_title:"مخطط المستشعرات المباشر",machines_table_title:"الآلات",machine_code_col:"الرمز",machine_name_col:"الاسم",status_col:"الحالة",risk_col:"الخطر",no_machines_yet:"لا توجد آلات بعد. انقر على «+ إضافة آلة».",section_machine_info:"معلومات الآلة",section_sensor_data:"بيانات المستشعر",section_status:"الحالة",section_notes:"ملاحظات",status_stopped:"متوقفة",status_maintenance:"صيانة",priority_low:"منخفضة",priority_normal:"عادية",priority_high:"عالية",priority_critical:"حرجة",save_and_analyze_btn:"حفظ وتحليل",source_col:"المصدر",source_auto:"تلقائي (SCADA)",source_manual:"يدوي",nav_alerts:"التنبيهات",acknowledge_btn:"إقرار",acknowledged_label:"تم الإقرار",acknowledge_all_btn:"إقرار الكل",no_alerts_yet:"لا توجد تنبيهات. كل شيء يعمل بسلاسة.",download_report_btn:"تقرير",alert_details_template:"درجة الحرارة {temp}°C، الاهتزاز {vib} مم/ث، الحالة: {status}",section_energy_intel:"ذكاء الطاقة",daily_output_hint:"يُستخدم لحساب استهلاك الطاقة النوعي (kWh لكل وحدة).",energy_insights_title:"ذكاء الطاقة",idle_power_title:"كشف طاقة الخمول",idle_active_msg:"الآلة خاملة - يُهدر حاليًا حوالي {kw} كيلوواط.",idle_none_msg:"لم يتم اكتشاف هدر طاقة أثناء الخمول.",friction_loss_title:"فقدان الطاقة التنبؤي",friction_active_msg:"تم اكتشاف احتكاك مرتفع: +{pct}% زيادة في الطاقة (~{kw} كيلوواط إضافية). جدولة الصيانة لتجنب الخسائر.",friction_none_msg:"لم يتم اكتشاف احتكاك غير طبيعي.",sec_title:"استهلاك الطاقة النوعي",sec_label:"kWh لكل وحدة",sec_unit:"kWh/وحدة",sec_no_data_msg:"أدخل الإنتاج اليومي عند إضافة هذه الآلة لرؤية هذا المقياس.",optimal_load_title:"منطقة الحمل الأمثل",optimal_load_label:"الحمل الأمثل",current_load_label:"الحمل الحالي",at_optimal_msg:"يعمل في منطقة الحمل الأمثل.",adjust_to_optimal_msg:"اضبط الحمل نحو {pct}% لتقليل الطاقة لكل وحدة.",nav_digital_twin:"التوأم الرقمي",twin_hint:"اسحب للتدوير، مرر للتكبير، انقر على آلة لرؤية تفاصيلها المباشرة.",twin_unavailable_msg:"تعذر تحميل العرض ثلاثي الأبعاد (تحقق من اتصال الإنترنت لمكتبة Three.js).",failure_prediction_title:"توقع العطل",report_lib_missing_msg:"يتطلب تصدير PDF مكتبة reportlab. نفّذ: pip install reportlab، ثم أعد تشغيل الخادم.",ph_machine_id:"معرف الآلة (مثل M-01)",ph_machine_name:"اسم الآلة",ph_factory_section:"قسم المصنع",ph_operator_name:"اسم المشغل",ph_pressure:"الضغط (بار)",ph_voltage:"الجهد (فولت)",ph_current:"التيار (أمبير)",ph_error_code:"رمز الخطأ",ph_daily_output:"الإنتاج اليومي (وحدة)",ph_notes:"ملاحظات...",nav_system_intel:"ذكاء النظام",nav_roi:"لوحة العائد",refresh_btn:"تحديث",system_risk_label:"مخاطر النظام",healthy_label:"سليمة",at_risk_label:"في خطر",clusters_title:"مجموعات الآلات",propagation_title:"انتشار الشذوذ",propagation_hint:"كيف ترفع آلة معطلة مخاطر الآلات المجاورة.",no_propagation:"لم يتم اكتشاف انتشار للشذوذ.",avg_risk_label:"متوسط المخاطر",added_risk_label:"مخاطر مضافة",effective_risk_label:"المخاطر الفعلية",simulation_title:"محاكاة ماذا-لو",simulation_hint:"اختر آلة، غيّر قيم المستشعرات، وشاهد كيف يتغير احتمال العطل.",run_simulation_btn:"تشغيل المحاكاة",failure_probability_label:"احتمال العطل",stress_level_label:"مستوى الإجهاد",predicted_status_label:"الحالة المتوقعة",confidence_label:"الثقة",root_cause_title:"السبب الجذري",rul_col:"العمر المتبقي",rul_healthy:"سليمة",potential_loss_label:"الخسارة المحتملة",saved_label:"وفّره الذكاء الاصطناعي",wasted_energy_label:"الطاقة المهدورة / شهر",efficiency_gain_label:"مكسب الكفاءة",cost_by_machine_title:"التعرض للتكلفة حسب الآلة",top_cause_col:"السبب الرئيسي",roi_assumptions_msg:"الافتراضات: توقف {downtime}/ساعة، إصلاح {hours} ساعة، طاقة {price}/kWh.",role_label:"دورك",role_engineer:"مهندس",role_manager:"مدير",role_admin:"مسؤول",cause_bearing_wear:"تآكل المحمل",cause_overload_thermal:"حمل حراري زائد",cause_cooling_failure:"عطل التبريد",cause_misalignment:"انحراف العمود",cause_lubrication_loss:"فقدان التزييت",cause_normal_operation:"تشغيل طبيعي",nav_story:"وضع القصة",story_hint:"يعيد تشغيل عطل محمل يتطور خلال 22 ساعة ويوضح بالضبط متى اكتشفه الذكاء الاصطناعي — وكم كانت قيمة ذلك.",simulate_failure_btn:"محاكاة العطل",outcome_title:"النتيجة",warning_time_label:"إنذار مبكر",loss_ignored_label:"الخسارة دون تدخل",loss_acted_label:"الخسارة مع التدخل",money_saved_label:"المبلغ الموفر",timeline_title:"الجدول الزمني للعطل",story_detection_msg:"اكتشف الذكاء الاصطناعي ذلك قبل {hours} ساعة من العطل، بمخاطر {risk}% — السبب الجذري: {cause} (ثقة {confidence}%).",story_stage_healthy:"سليمة",story_stage_early_drift:"انحراف مبكر",story_stage_ai_detects:"الذكاء الاصطناعي يكتشف",story_stage_critical:"حرج",priority_low:"منخفض",priority_medium:"متوسط",priority_high:"عالي",priority_critical:"حرج",action_stop_machine:"إيقاف الآلة",action_inspect_bearings:"فحص المحامل",action_check_cooling:"فحص التبريد",action_schedule_shutdown_24h:"جدولة الإيقاف (24 ساعة)",action_order_spare_parts:"طلب قطع الغيار",action_reduce_load:"تقليل الحمل",action_schedule_inspection_72h:"جدولة الفحص (72 ساعة)",action_monitor_closely:"مراقبة عن كثب",action_verify_sensor:"التحقق من المستشعر",action_power_down_idle:"إطفاء الآلة الخاملة",action_review_shift_schedule:"مراجعة جدول المناوبات",action_no_action:"لا حاجة لأي إجراء",nav_oee:"OEE",nav_workorders:"أوامر العمل",oee_hint:"الجاهزية x الأداء x الجودة (ISO 22400). المستوى العالمي 85%، والمصنع النموذجي حوالي 60%.",availability_label:"الجاهزية",performance_label:"الأداء",quality_label:"الجودة",weakest_factor_label:"الأضعف",downtime_by_reason_title:"التوقف حسب السبب",downtime_cost_label:"تكلفة التوقف",oee_trend_title:"اتجاه OEE",shifts_title:"الورديات",shift_col:"وردية",downtime_col:"توقف",log_shift_btn:"تسجيل وردية",no_shifts_yet:"لم يتم تسجيل ورديات بعد.",minutes_short:"دقيقة",range_1d:"اليوم",range_7d:"7 أيام",range_30d:"30 يوم",all_machines_option:"كل الآلات",reason_unspecified:"غير محدد",err_good_exceeds_total:"لا يمكن أن تتجاوز القطع السليمة الإجمالي",oee_grade_world_class:"المستوى العالمي",oee_grade_typical:"نموذجي",oee_grade_low:"منخفض",oee_grade_critical:"حرج",reason_breakdown:"عطل",reason_changeover:"تغيير الإنتاج",reason_no_material:"لا مواد",reason_no_operator:"لا مشغل",reason_planned_maintenance:"صيانة مخططة",reason_quality_issue:"مشكلة جودة",reason_setup:"إعداد",reason_other:"أخرى",workorders_hint:"يحوّل تنبؤ الذكاء الاصطناعي إلى مهمة متتبعة — لينتهي التحذير بإصلاح فعلي.",new_work_order_btn:"+ أمر عمل جديد",no_work_orders:"لا توجد أوامر عمل.",avg_completion_label:"متوسط الإنجاز",assigned_label:"مُسند إلى",source_ai:"ذكاء اصطناعي",wo_status_open:"مفتوح",wo_status_in_progress:"قيد التنفيذ",wo_status_done:"منجز",wo_status_cancelled:"ملغى",wo_advance_to_in_progress:"بدء العمل",wo_advance_to_done:"وضع علامة منجز",ph_shift_name:"اسم الوردية",ph_planned_minutes:"الدقائق المخططة",ph_downtime_minutes:"دقائق التوقف",ph_total_units:"إجمالي الوحدات",ph_good_units:"الوحدات السليمة",ph_cycle_seconds:"الدورة المثالية (ثانية)",ph_wo_title:"العنوان",ph_assigned_to:"مُسند إلى",ph_wo_description:"الوصف",overdue_label:"متأخر",nav_history:"السجل",history_hint:"سجل المستشعرات المحفوظ — هكذا تثبت ما تغيّر فعليًا خلال التجربة.",no_history_yet:"لم يتم تسجيل سجل بعد. اترك المراقبة المباشرة مفتوحة لبضع دقائق وستبدأ البيانات بالتراكم.",range_24h:"24 ساعة",range_3d:"3 أيام",range_10d:"10 أيام",sensor_trend_title:"اتجاه المستشعرات",risk_trend_title:"اتجاه المخاطر",trend_flat:"بدون تغيير",trend_rising:"في ارتفاع",trend_falling:"في انخفاض",create_work_order_btn:"إنشاء أمر عمل",work_order_created_msg:"تم إنشاء أمر العمل",alertreason_immediate_failure_risk:"خطر عطل فوري",alertreason_failure_imminent:"عطل وشيك",alertreason_degradation_accelerating:"تدهور متسارع",alertreason_outside_normal_envelope:"خارج النطاق الطبيعي",alertreason_pressure_anomaly:"شذوذ في الضغط",alertreason_idle_waste:"هدر طاقة الخمول",alertreason_informational:"معلوماتي",alert_reason_label:"السبب"},
  tr: {tagline:"Küresel Endüstriyel Zeka Platformu",live_label:"Canlı",kpi_energy:"Enerji Kullanımı",kpi_efficiency:"Verimlilik",kpi_active:"Aktif Makineler",kpi_alerts:"Uyarılar",kwh_unit:"kWh",chart_title:"Gerçek Zamanlı Performans",machine_status_title:"Makine Durumu",status_running:"Çalışıyor",status_warning:"Uyarı",status_critical:"Kritik",form_title:"Fabrika Veri Girişi",factory_name_label:"Fabrika Adı",machine_count_label:"Makine Sayısı",energy_cost_label:"Enerji Maliyeti ($/kWh)",machine_type_label:"Makine Türü",temperature_label:"Sıcaklık (°C)",vibration_label:"Titreşim (mm/s)",load_label:"Yük (%)",submit_btn:"Fabrikayı Analiz Et",submitting:"Güncelleniyor...",ai_panel_title:"AI Analizleri",ai_placeholder:"AI analizi oluşturmak için fabrika verilerini gönderin.",ai_analyzing:"Analiz ediliyor...",ai_risks:"Riskler",ai_efficiency_insights:"Verimlilik Analizi",ai_optimizations:"Optimizasyon Önerileri",toast_updated:"Fabrika verileri güncellendi",toast_analysis_done:"AI analizi tamamlandı",toast_error:"Bir şeyler ters gitti",nav_dashboard:"Panel",nav_factories:"Fabrikalar",nav_ai_insights:"AI Analizleri",logout_btn:"Çıkış Yap",login_title:"Tekrar hoş geldiniz",login_subtitle:"FactoryPulse AI hesabınıza giriş yapın",ph_email:"E-posta",ph_password:"Şifre",remember_me:"Beni hatırla",login_btn:"Giriş Yap",login_link_register:"Hesabınız yok mu? Oluşturun",register_title:"Hesabınızı oluşturun",register_subtitle:"Fabrikalarınızı AI ile izlemeye başlayın",ph_full_name:"Ad Soyad",ph_confirm_password:"Şifreyi Onayla",register_btn:"Hesap Oluştur",register_link_login:"Zaten hesabınız var mı? Giriş yapın",err_missing_fields:"Lütfen tüm alanları doldurun",err_invalid_email:"Lütfen geçerli bir e-posta adresi girin",err_weak_password:"Şifre en az 8 karakter, bir harf ve bir rakam içermeli",err_password_mismatch:"Şifreler eşleşmiyor",err_invalid_credentials:"E-posta veya şifre hatalı",err_email_taken:"Bu e-posta zaten kayıtlı",err_generic:"Bir şeyler ters gitti. Tekrar deneyin",my_factories_title:"Fabrikalarım",add_factory_btn:"+ Fabrika Ekle",edit_factory_btn:"Düzenle",delete_factory_btn:"Sil",confirm_delete_factory:"Bu fabrika silinsin mi? Bu işlem geri alınamaz.",no_factories_yet:"Henüz fabrika eklemediniz.",factory_created_toast:"Fabrika oluşturuldu ve analiz edildi",factory_updated_toast:"Fabrika güncellendi",factory_deleted_toast:"Fabrika silindi",ai_insights_feed_title:"AI Analiz Akışı",no_ai_insights_yet:"Henüz AI analizi yok. Başlamak için fabrika ekleyin.",reanalyze_btn:"Yeniden Analiz Et",view_insights_btn:"Analizleri Görüntüle",created_label:"Oluşturulma",cancel_btn:"İptal",save_btn:"Değişiklikleri Kaydet",nav_live_monitor:"Canlı İzleme",add_machine_scada_btn:"+ Makine Ekle",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Sorgulama",live_chart_title:"Canlı Sensör Grafiği",machines_table_title:"Makineler",machine_code_col:"Kod",machine_name_col:"Ad",status_col:"Durum",risk_col:"Risk",no_machines_yet:"Henüz makine yok. \"+ Makine Ekle\"ye tıklayın.",section_machine_info:"Makine Bilgisi",section_sensor_data:"Sensör Verileri",section_status:"Durum",section_notes:"Notlar",status_stopped:"Durduruldu",status_maintenance:"Bakımda",priority_low:"Düşük",priority_normal:"Normal",priority_high:"Yüksek",priority_critical:"Kritik",save_and_analyze_btn:"Kaydet ve Analiz Et",source_col:"Kaynak",source_auto:"Otomatik (SCADA)",source_manual:"Manuel",nav_alerts:"Uyarılar",acknowledge_btn:"Onayla",acknowledged_label:"Onaylandı",acknowledge_all_btn:"Tümünü Onayla",no_alerts_yet:"Uyarı yok. Her şey sorunsuz çalışıyor.",download_report_btn:"Rapor",alert_details_template:"Sıcaklık {temp}°C, titreşim {vib} mm/s, durum: {status}",section_energy_intel:"Enerji Zekası",daily_output_hint:"Özgül enerji tüketimini hesaplamak için kullanılır (birim başına kWh).",energy_insights_title:"Enerji Zekası",idle_power_title:"Boşta Güç Tespiti",idle_active_msg:"Makine boşta - şu anda yaklaşık {kw} kW israf ediliyor.",idle_none_msg:"Boşta enerji israfı tespit edilmedi.",friction_loss_title:"Öngörülü Enerji Kaybı",friction_active_msg:"Artan sürtünme tespit edildi: +%{pct} fazla güç (~{kw} kW ekstra). Kayıpları önlemek için bakım planlayın.",friction_none_msg:"Anormal sürtünme tespit edilmedi.",sec_title:"Özgül Enerji Tüketimi",sec_label:"birim başına kWh",sec_unit:"kWh/birim",sec_no_data_msg:"Bu metriği görmek için makineyi eklerken günlük üretimi girin.",optimal_load_title:"Optimal Yük Bölgesi",optimal_load_label:"Optimal yük",current_load_label:"Mevcut yük",at_optimal_msg:"Optimal yük bölgesinde çalışıyor.",adjust_to_optimal_msg:"Birim başına enerjiyi en aza indirmek için yükü %{pct}'e ayarlayın.",nav_digital_twin:"Dijital İkiz",twin_hint:"Döndürmek için sürükleyin, yakınlaştırmak için kaydırın, canlı detaylar için bir makineye tıklayın.",twin_unavailable_msg:"3D görünüm yüklenemedi (Three.js kütüphanesi için internet bağlantınızı kontrol edin).",failure_prediction_title:"Arıza Tahmini",report_lib_missing_msg:"PDF dışa aktarma için reportlab kütüphanesi gerekir. Çalıştırın: pip install reportlab, ardından sunucuyu yeniden başlatın.",ph_machine_id:"Makine ID (örn. M-01)",ph_machine_name:"Makine Adı",ph_factory_section:"Fabrika Bölümü",ph_operator_name:"Operatör Adı",ph_pressure:"Basınç (bar)",ph_voltage:"Voltaj (V)",ph_current:"Akım (A)",ph_error_code:"Hata Kodu",ph_daily_output:"Günlük Üretim (birim)",ph_notes:"Notlar...",nav_system_intel:"Sistem Zekası",nav_roi:"ROI Panosu",refresh_btn:"Yenile",system_risk_label:"Sistem Riski",healthy_label:"Sağlıklı",at_risk_label:"Riskli",clusters_title:"Makine Kümeleri",propagation_title:"Anomali Yayılımı",propagation_hint:"Arızalı bir makinenin komşularının riskini nasıl artırdığı.",no_propagation:"Anomali yayılımı tespit edilmedi.",avg_risk_label:"Ort. risk",added_risk_label:"Eklenen risk",effective_risk_label:"Etkin risk",simulation_title:"Ne-Olur Simülasyonu",simulation_hint:"Bir makine seçin, sensör değerlerini kaydırın ve arıza olasılığını izleyin.",run_simulation_btn:"Simülasyonu Çalıştır",failure_probability_label:"Arıza Olasılığı",stress_level_label:"Zorlanma Seviyesi",predicted_status_label:"Tahmini Durum",confidence_label:"Güven",root_cause_title:"Kök Neden",rul_col:"Kalan Ömür",rul_healthy:"Sağlıklı",potential_loss_label:"Potansiyel Kayıp",saved_label:"AI ile Tasarruf",wasted_energy_label:"İsraf Edilen Enerji / ay",efficiency_gain_label:"Verimlilik Artışı",cost_by_machine_title:"Makine Bazlı Maliyet Riski",top_cause_col:"Ana Neden",roi_assumptions_msg:"Varsayımlar: duruş {downtime}/sa, onarım {hours} sa, enerji {price}/kWh.",role_label:"Rolünüz",role_engineer:"Mühendis",role_manager:"Yönetici",role_admin:"Admin",cause_bearing_wear:"Rulman aşınması",cause_overload_thermal:"Termal aşırı yük",cause_cooling_failure:"Soğutma arızası",cause_misalignment:"Mil kaçıklığı",cause_lubrication_loss:"Yağlama kaybı",cause_normal_operation:"Normal çalışma",nav_story:"Hikaye Modu",story_hint:"22 saat içinde gelişen bir rulman arızasını yeniden oynatır ve yapay zekanın onu tam olarak ne zaman yakaladığını — ve bunun değerini gösterir.",simulate_failure_btn:"Arıza Simüle Et",outcome_title:"Sonuç",warning_time_label:"Erken Uyarı",loss_ignored_label:"Müdahalesiz Kayıp",loss_acted_label:"Müdahaleli Kayıp",money_saved_label:"Tasarruf",timeline_title:"Arıza Zaman Çizelgesi",story_detection_msg:"Yapay zeka bunu arızadan {hours} saat önce, %{risk} riskte tespit etti — kök neden: {cause} (%{confidence} güven).",story_stage_healthy:"Sağlıklı",story_stage_early_drift:"İlk Sapma",story_stage_ai_detects:"Yapay Zeka Tespit Etti",story_stage_critical:"Kritik",priority_low:"Düşük",priority_medium:"Orta",priority_high:"Yüksek",priority_critical:"Kritik",action_stop_machine:"Makineyi durdur",action_inspect_bearings:"Rulmanları incele",action_check_cooling:"Soğutmayı kontrol et",action_schedule_shutdown_24h:"Duruş planla (24s)",action_order_spare_parts:"Yedek parça sipariş et",action_reduce_load:"Yükü azalt",action_schedule_inspection_72h:"Muayene planla (72s)",action_monitor_closely:"Yakından izle",action_verify_sensor:"Sensörü doğrula",action_power_down_idle:"Boştaki makineyi kapat",action_review_shift_schedule:"Vardiya planını gözden geçir",action_no_action:"İşlem gerekmiyor",nav_oee:"OEE",nav_workorders:"İş Emirleri",oee_hint:"Kullanılabilirlik x Performans x Kalite (ISO 22400). Dünya standardı %85, tipik bir fabrika ise %60 civarındadır.",availability_label:"Kullanılabilirlik",performance_label:"Performans",quality_label:"Kalite",weakest_factor_label:"En zayıf",downtime_by_reason_title:"Nedene Göre Duruş",downtime_cost_label:"Duruş maliyeti",oee_trend_title:"OEE Trendi",shifts_title:"Vardiyalar",shift_col:"Vardiya",downtime_col:"Duruş",log_shift_btn:"Vardiya Kaydet",no_shifts_yet:"Henüz vardiya kaydedilmedi.",minutes_short:"dk",range_1d:"Bugün",range_7d:"7 gün",range_30d:"30 gün",all_machines_option:"Tüm makineler",reason_unspecified:"Belirtilmemiş",err_good_exceeds_total:"Sağlam adet toplamı aşamaz",oee_grade_world_class:"Dünya standardı",oee_grade_typical:"Tipik",oee_grade_low:"Düşük",oee_grade_critical:"Kritik",reason_breakdown:"Arıza",reason_changeover:"Tip değişimi",reason_no_material:"Malzeme yok",reason_no_operator:"Operatör yok",reason_planned_maintenance:"Planlı bakım",reason_quality_issue:"Kalite sorunu",reason_setup:"Kurulum",reason_other:"Diğer",workorders_hint:"AI tahminini takip edilebilir bir göreve dönüştürür — böylece uyarı gerçekten onarımla biter.",new_work_order_btn:"+ Yeni İş Emri",no_work_orders:"Henüz iş emri yok.",avg_completion_label:"Ort. Tamamlama",assigned_label:"Atanan",source_ai:"AI",wo_status_open:"Açık",wo_status_in_progress:"Devam Ediyor",wo_status_done:"Tamamlandı",wo_status_cancelled:"İptal",wo_advance_to_in_progress:"İşe başla",wo_advance_to_done:"Tamamlandı işaretle",ph_shift_name:"Vardiya adı",ph_planned_minutes:"Planlı dakika",ph_downtime_minutes:"Duruş dakikası",ph_total_units:"Toplam adet",ph_good_units:"Sağlam adet",ph_cycle_seconds:"İdeal çevrim (sn)",ph_wo_title:"Başlık",ph_assigned_to:"Atanan kişi",ph_wo_description:"Açıklama",overdue_label:"Gecikmiş",nav_history:"Geçmiş",history_hint:"Kaydedilmiş sensör geçmişi — pilot süresince gerçekte neyin değiştiğini böyle kanıtlarsınız.",no_history_yet:"Henüz geçmiş kaydedilmedi. Canlı İzleme'yi birkaç dakika açık tutun, veriler birikmeye başlar.",range_24h:"24 saat",range_3d:"3 gün",range_10d:"10 gün",sensor_trend_title:"Sensör Trendi",risk_trend_title:"Risk Trendi",trend_flat:"değişim yok",trend_rising:"yükseliyor",trend_falling:"düşüyor",create_work_order_btn:"İş Emri Oluştur",work_order_created_msg:"İş emri oluşturuldu",alertreason_immediate_failure_risk:"Ani arıza riski",alertreason_failure_imminent:"Arıza çok yakın",alertreason_degradation_accelerating:"Bozulma hızlanıyor",alertreason_outside_normal_envelope:"Normal aralık dışında",alertreason_pressure_anomaly:"Basınç anomalisi",alertreason_idle_waste:"Boşta enerji israfı",alertreason_informational:"Bilgilendirme",alert_reason_label:"Neden"},
  it: {tagline:"Piattaforma Globale di Intelligenza Industriale",live_label:"In diretta",kpi_energy:"Consumo Energetico",kpi_efficiency:"Efficienza",kpi_active:"Macchine Attive",kpi_alerts:"Avvisi",kwh_unit:"kWh",chart_title:"Prestazioni in Tempo Reale",machine_status_title:"Stato delle Macchine",status_running:"In funzione",status_warning:"Avviso",status_critical:"Critico",form_title:"Inserimento Dati Fabbrica",factory_name_label:"Nome Fabbrica",machine_count_label:"Numero di Macchine",energy_cost_label:"Costo Energia ($/kWh)",machine_type_label:"Tipo di Macchina",temperature_label:"Temperatura (°C)",vibration_label:"Vibrazione (mm/s)",load_label:"Carico (%)",submit_btn:"Analizza Fabbrica",submitting:"Aggiornamento...",ai_panel_title:"Analisi IA",ai_placeholder:"Invia i dati della fabbrica per generare un'analisi IA.",ai_analyzing:"Analisi in corso...",ai_risks:"Rischi",ai_efficiency_insights:"Analisi dell'Efficienza",ai_optimizations:"Suggerimenti di Ottimizzazione",toast_updated:"Dati fabbrica aggiornati",toast_analysis_done:"Analisi IA completata",toast_error:"Qualcosa è andato storto",nav_dashboard:"Dashboard",nav_factories:"Fabbriche",nav_ai_insights:"Analisi IA",logout_btn:"Esci",login_title:"Bentornato",login_subtitle:"Accedi al tuo account FactoryPulse AI",ph_email:"Email",ph_password:"Password",remember_me:"Ricordami",login_btn:"Accedi",login_link_register:"Non hai un account? Creane uno",register_title:"Crea il tuo account",register_subtitle:"Inizia a monitorare le tue fabbriche con l'IA",ph_full_name:"Nome Completo",ph_confirm_password:"Conferma Password",register_btn:"Crea Account",register_link_login:"Hai già un account? Accedi",err_missing_fields:"Si prega di compilare tutti i campi",err_invalid_email:"Inserisci un indirizzo email valido",err_weak_password:"La password deve avere almeno 8 caratteri, una lettera e un numero",err_password_mismatch:"Le password non corrispondono",err_invalid_credentials:"Email o password errati",err_email_taken:"Questa email è già registrata",err_generic:"Qualcosa è andato storto. Riprova",my_factories_title:"Le Mie Fabbriche",add_factory_btn:"+ Aggiungi Fabbrica",edit_factory_btn:"Modifica",delete_factory_btn:"Elimina",confirm_delete_factory:"Eliminare questa fabbrica? Questa azione non può essere annullata.",no_factories_yet:"Non hai ancora aggiunto nessuna fabbrica.",factory_created_toast:"Fabbrica creata e analizzata",factory_updated_toast:"Fabbrica aggiornata",factory_deleted_toast:"Fabbrica eliminata",ai_insights_feed_title:"Feed di Analisi IA",no_ai_insights_yet:"Nessuna analisi IA ancora. Aggiungi una fabbrica.",reanalyze_btn:"Rianalizza",view_insights_btn:"Vedi Analisi",created_label:"Creata il",cancel_btn:"Annulla",save_btn:"Salva Modifiche",nav_live_monitor:"Monitoraggio Live",add_machine_scada_btn:"+ Aggiungi Macchina",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Polling",live_chart_title:"Grafico Sensori in Tempo Reale",machines_table_title:"Macchine",machine_code_col:"Codice",machine_name_col:"Nome",status_col:"Stato",risk_col:"Rischio",no_machines_yet:"Nessuna macchina ancora. Fai clic su «+ Aggiungi Macchina».",section_machine_info:"Informazioni Macchina",section_sensor_data:"Dati dei Sensori",section_status:"Stato",section_notes:"Note",status_stopped:"Ferma",status_maintenance:"Manutenzione",priority_low:"Bassa",priority_normal:"Normale",priority_high:"Alta",priority_critical:"Critica",save_and_analyze_btn:"Salva e Analizza",source_col:"Origine",source_auto:"Auto (SCADA)",source_manual:"Manuale",nav_alerts:"Avvisi",acknowledge_btn:"Conferma",acknowledged_label:"Confermato",acknowledge_all_btn:"Conferma Tutti",no_alerts_yet:"Nessun avviso. Tutto funziona correttamente.",download_report_btn:"Rapporto",alert_details_template:"Temperatura {temp}°C, vibrazione {vib} mm/s, stato: {status}",section_energy_intel:"Intelligenza Energetica",daily_output_hint:"Usato per calcolare il consumo energetico specifico (kWh per unità).",energy_insights_title:"Intelligenza Energetica",idle_power_title:"Rilevamento Potenza in Inattività",idle_active_msg:"Macchina inattiva - circa {kw} kW sprecati ora.",idle_none_msg:"Nessuno spreco di energia in inattività rilevato.",friction_loss_title:"Perdita di Energia Predittiva",friction_active_msg:"Attrito elevato rilevato: +{pct}% di sovraccarico (~{kw} kW extra). Pianifica la manutenzione per evitare perdite.",friction_none_msg:"Nessun attrito anomalo rilevato.",sec_title:"Consumo Energetico Specifico",sec_label:"kWh per unità",sec_unit:"kWh/unità",sec_no_data_msg:"Inserisci la produzione giornaliera aggiungendo questa macchina per vedere questa metrica.",optimal_load_title:"Zona di Carico Ottimale",optimal_load_label:"Carico ottimale",current_load_label:"Carico attuale",at_optimal_msg:"In funzione nella zona di carico ottimale.",adjust_to_optimal_msg:"Regola il carico verso {pct}% per minimizzare l'energia per unità.",nav_digital_twin:"Gemello Digitale",twin_hint:"Trascina per ruotare, scorri per zoomare, clicca su una macchina per i dettagli in tempo reale.",twin_unavailable_msg:"Impossibile caricare la vista 3D (controlla la connessione internet per Three.js).",failure_prediction_title:"Previsione del Guasto",report_lib_missing_msg:"L'esportazione PDF richiede la libreria reportlab. Esegui: pip install reportlab, poi riavvia il server.",ph_machine_id:"ID Macchina (es. M-01)",ph_machine_name:"Nome Macchina",ph_factory_section:"Sezione Fabbrica",ph_operator_name:"Nome Operatore",ph_pressure:"Pressione (bar)",ph_voltage:"Tensione (V)",ph_current:"Corrente (A)",ph_error_code:"Codice Errore",ph_daily_output:"Produzione Giornaliera (unità)",ph_notes:"Note...",nav_system_intel:"Intelligenza di Sistema",nav_roi:"Dashboard ROI",refresh_btn:"Aggiorna",system_risk_label:"Rischio di Sistema",healthy_label:"Sane",at_risk_label:"A Rischio",clusters_title:"Cluster di Macchine",propagation_title:"Propagazione Anomalie",propagation_hint:"Come una macchina in avaria aumenta il rischio delle vicine.",no_propagation:"Nessuna propagazione di anomalie rilevata.",avg_risk_label:"Rischio medio",added_risk_label:"Rischio aggiunto",effective_risk_label:"Rischio effettivo",simulation_title:"Simulazione What-If",simulation_hint:"Scegli una macchina, modifica i sensori e osserva la probabilità di guasto.",run_simulation_btn:"Avvia Simulazione",failure_probability_label:"Probabilità di Guasto",stress_level_label:"Livello di Stress",predicted_status_label:"Stato Previsto",confidence_label:"Confidenza",root_cause_title:"Causa Radice",rul_col:"Vita Utile Residua",rul_healthy:"Sana",potential_loss_label:"Perdita Potenziale",saved_label:"Risparmiato dall'IA",wasted_energy_label:"Energia Sprecata / mese",efficiency_gain_label:"Guadagno di Efficienza",cost_by_machine_title:"Esposizione ai Costi per Macchina",top_cause_col:"Causa Principale",roi_assumptions_msg:"Ipotesi: fermo {downtime}/h, riparazione {hours}h, energia {price}/kWh.",role_label:"Il Tuo Ruolo",role_engineer:"Ingegnere",role_manager:"Manager",role_admin:"Amministratore",cause_bearing_wear:"Usura del cuscinetto",cause_overload_thermal:"Sovraccarico termico",cause_cooling_failure:"Guasto raffreddamento",cause_misalignment:"Disallineamento albero",cause_lubrication_loss:"Perdita di lubrificazione",cause_normal_operation:"Funzionamento normale",nav_story:"Modalità Storia",story_hint:"Riproduce un guasto al cuscinetto che si sviluppa in 22 ore e mostra esattamente quando l'IA lo ha rilevato — e quanto valeva.",simulate_failure_btn:"Simula Guasto",outcome_title:"Risultato",warning_time_label:"Preavviso",loss_ignored_label:"Perdita senza Intervento",loss_acted_label:"Perdita con Intervento",money_saved_label:"Risparmiato",timeline_title:"Cronologia del Guasto",story_detection_msg:"L'IA lo ha segnalato {hours} ore prima del guasto, al {risk}% di rischio — causa radice: {cause} (confidenza {confidence}%).",story_stage_healthy:"Sana",story_stage_early_drift:"Prima Deriva",story_stage_ai_detects:"L'IA Rileva",story_stage_critical:"Critico",priority_low:"Basso",priority_medium:"Medio",priority_high:"Alto",priority_critical:"Critico",action_stop_machine:"Ferma macchina",action_inspect_bearings:"Ispeziona cuscinetti",action_check_cooling:"Controlla raffreddamento",action_schedule_shutdown_24h:"Pianifica fermo (24h)",action_order_spare_parts:"Ordina ricambi",action_reduce_load:"Riduci carico",action_schedule_inspection_72h:"Pianifica ispezione (72h)",action_monitor_closely:"Monitora da vicino",action_verify_sensor:"Verifica sensore",action_power_down_idle:"Spegni macchina inattiva",action_review_shift_schedule:"Rivedi turni",action_no_action:"Nessuna azione necessaria",nav_oee:"OEE",nav_workorders:"Ordini di Lavoro",oee_hint:"Disponibilità x Prestazioni x Qualità (ISO 22400). Il livello mondiale è 85%, una fabbrica tipica si attesta sul 60%.",availability_label:"Disponibilità",performance_label:"Prestazioni",quality_label:"Qualità",weakest_factor_label:"Più debole",downtime_by_reason_title:"Fermi per Causa",downtime_cost_label:"Costo dei fermi",oee_trend_title:"Andamento OEE",shifts_title:"Turni",shift_col:"Turno",downtime_col:"Fermo",log_shift_btn:"Registra Turno",no_shifts_yet:"Nessun turno registrato.",minutes_short:"min",range_1d:"Oggi",range_7d:"7 giorni",range_30d:"30 giorni",all_machines_option:"Tutte le macchine",reason_unspecified:"Non specificato",err_good_exceeds_total:"I pezzi buoni non possono superare il totale",oee_grade_world_class:"Livello mondiale",oee_grade_typical:"Tipico",oee_grade_low:"Basso",oee_grade_critical:"Critico",reason_breakdown:"Guasto",reason_changeover:"Cambio produzione",reason_no_material:"Materiale mancante",reason_no_operator:"Operatore mancante",reason_planned_maintenance:"Manutenzione pianificata",reason_quality_issue:"Problema qualità",reason_setup:"Setup",reason_other:"Altro",workorders_hint:"Trasforma una previsione IA in un compito tracciato — così un allarme finisce in una riparazione.",new_work_order_btn:"+ Nuovo Ordine",no_work_orders:"Nessun ordine di lavoro.",avg_completion_label:"Completamento medio",assigned_label:"Assegnato",source_ai:"IA",wo_status_open:"Aperto",wo_status_in_progress:"In corso",wo_status_done:"Completato",wo_status_cancelled:"Annullato",wo_advance_to_in_progress:"Inizia lavoro",wo_advance_to_done:"Segna completato",ph_shift_name:"Nome turno",ph_planned_minutes:"Minuti pianificati",ph_downtime_minutes:"Minuti di fermo",ph_total_units:"Unità totali",ph_good_units:"Unità buone",ph_cycle_seconds:"Ciclo ideale (sec)",ph_wo_title:"Titolo",ph_assigned_to:"Assegnato a",ph_wo_description:"Descrizione",overdue_label:"In ritardo",nav_history:"Cronologia",history_hint:"Cronologia dei sensori salvata — così dimostri cosa è realmente cambiato durante il pilota.",no_history_yet:"Nessuna cronologia registrata. Tieni il Monitoraggio aperto qualche minuto e i dati inizieranno ad accumularsi.",range_24h:"24 ore",range_3d:"3 giorni",range_10d:"10 giorni",sensor_trend_title:"Andamento Sensori",risk_trend_title:"Andamento Rischio",trend_flat:"nessuna variazione",trend_rising:"in aumento",trend_falling:"in calo",create_work_order_btn:"Crea Ordine",work_order_created_msg:"Ordine di lavoro creato",alertreason_immediate_failure_risk:"Rischio di guasto immediato",alertreason_failure_imminent:"Guasto imminente",alertreason_degradation_accelerating:"Degrado in accelerazione",alertreason_outside_normal_envelope:"Fuori dal range normale",alertreason_pressure_anomaly:"Anomalia di pressione",alertreason_idle_waste:"Spreco in inattività",alertreason_informational:"Informativo",alert_reason_label:"Motivo"},
  pt: {tagline:"Plataforma Global de Inteligência Industrial",live_label:"Ao vivo",kpi_energy:"Uso de Energia",kpi_efficiency:"Eficiência",kpi_active:"Máquinas Ativas",kpi_alerts:"Alertas",kwh_unit:"kWh",chart_title:"Desempenho em Tempo Real",machine_status_title:"Status das Máquinas",status_running:"Em funcionamento",status_warning:"Aviso",status_critical:"Crítico",form_title:"Entrada de Dados da Fábrica",factory_name_label:"Nome da Fábrica",machine_count_label:"Número de Máquinas",energy_cost_label:"Custo de Energia ($/kWh)",machine_type_label:"Tipo de Máquina",temperature_label:"Temperatura (°C)",vibration_label:"Vibração (mm/s)",load_label:"Carga (%)",submit_btn:"Analisar Fábrica",submitting:"Atualizando...",ai_panel_title:"Insights de IA",ai_placeholder:"Envie os dados da fábrica para gerar uma análise de IA.",ai_analyzing:"Analisando...",ai_risks:"Riscos",ai_efficiency_insights:"Análise de Eficiência",ai_optimizations:"Sugestões de Otimização",toast_updated:"Dados da fábrica atualizados",toast_analysis_done:"Análise de IA concluída",toast_error:"Algo deu errado",nav_dashboard:"Painel",nav_factories:"Fábricas",nav_ai_insights:"Insights de IA",logout_btn:"Sair",login_title:"Bem-vindo de volta",login_subtitle:"Entre na sua conta FactoryPulse AI",ph_email:"E-mail",ph_password:"Senha",remember_me:"Lembrar de mim",login_btn:"Entrar",login_link_register:"Não tem conta? Crie uma",register_title:"Crie sua conta",register_subtitle:"Comece a monitorar suas fábricas com IA",ph_full_name:"Nome Completo",ph_confirm_password:"Confirmar Senha",register_btn:"Criar Conta",register_link_login:"Já tem conta? Entrar",err_missing_fields:"Por favor preencha todos os campos",err_invalid_email:"Por favor insira um e-mail válido",err_weak_password:"A senha deve ter no mínimo 8 caracteres, uma letra e um número",err_password_mismatch:"As senhas não coincidem",err_invalid_credentials:"E-mail ou senha incorretos",err_email_taken:"Este e-mail já está registrado",err_generic:"Algo deu errado. Tente novamente",my_factories_title:"Minhas Fábricas",add_factory_btn:"+ Adicionar Fábrica",edit_factory_btn:"Editar",delete_factory_btn:"Excluir",confirm_delete_factory:"Excluir esta fábrica? Esta ação não pode ser desfeita.",no_factories_yet:"Você ainda não adicionou nenhuma fábrica.",factory_created_toast:"Fábrica criada e analisada",factory_updated_toast:"Fábrica atualizada",factory_deleted_toast:"Fábrica excluída",ai_insights_feed_title:"Feed de Insights de IA",no_ai_insights_yet:"Ainda sem insights de IA. Adicione uma fábrica.",reanalyze_btn:"Reanalisar",view_insights_btn:"Ver Insights",created_label:"Criada em",cancel_btn:"Cancelar",save_btn:"Salvar Alterações",nav_live_monitor:"Monitor ao Vivo",add_machine_scada_btn:"+ Adicionar Máquina",usb_status:"USB:",plc_status:"CLP:",polling_mode:"Sondagem",live_chart_title:"Gráfico de Sensores ao Vivo",machines_table_title:"Máquinas",machine_code_col:"Código",machine_name_col:"Nome",status_col:"Status",risk_col:"Risco",no_machines_yet:"Ainda sem máquinas. Clique em «+ Adicionar Máquina».",section_machine_info:"Informações da Máquina",section_sensor_data:"Dados do Sensor",section_status:"Status",section_notes:"Notas",status_stopped:"Parada",status_maintenance:"Manutenção",priority_low:"Baixa",priority_normal:"Normal",priority_high:"Alta",priority_critical:"Crítica",save_and_analyze_btn:"Salvar e Analisar",source_col:"Origem",source_auto:"Auto (SCADA)",source_manual:"Manual",nav_alerts:"Alertas",acknowledge_btn:"Confirmar",acknowledged_label:"Confirmado",acknowledge_all_btn:"Confirmar Todos",no_alerts_yet:"Sem alertas. Tudo funcionando normalmente.",download_report_btn:"Relatório",alert_details_template:"Temperatura {temp}°C, vibração {vib} mm/s, status: {status}",section_energy_intel:"Inteligência Energética",daily_output_hint:"Usado para calcular o consumo energético específico (kWh por unidade).",energy_insights_title:"Inteligência Energética",idle_power_title:"Detecção de Potência Ociosa",idle_active_msg:"Máquina ociosa - cerca de {kw} kW desperdiçados agora.",idle_none_msg:"Nenhum desperdício de energia ociosa detectado.",friction_loss_title:"Perda de Energia Preditiva",friction_active_msg:"Atrito elevado detectado: +{pct}% de sobrecarga (~{kw} kW extra). Agende manutenção para evitar perdas.",friction_none_msg:"Nenhum atrito anormal detectado.",sec_title:"Consumo Energético Específico",sec_label:"kWh por unidade",sec_unit:"kWh/unidade",sec_no_data_msg:"Insira a produção diária ao adicionar esta máquina para ver esta métrica.",optimal_load_title:"Zona de Carga Ideal",optimal_load_label:"Carga ideal",current_load_label:"Carga atual",at_optimal_msg:"Funcionando na zona de carga ideal.",adjust_to_optimal_msg:"Ajuste a carga para {pct}% para minimizar a energia por unidade.",nav_digital_twin:"Gêmeo Digital",twin_hint:"Arraste para girar, role para ampliar, clique em uma máquina para ver detalhes ao vivo.",twin_unavailable_msg:"Não foi possível carregar a visualização 3D (verifique sua conexão com a internet para o Three.js).",failure_prediction_title:"Previsão de Falha",report_lib_missing_msg:"A exportação em PDF precisa da biblioteca reportlab. Execute: pip install reportlab, depois reinicie o servidor.",ph_machine_id:"ID da Máquina (ex. M-01)",ph_machine_name:"Nome da Máquina",ph_factory_section:"Seção da Fábrica",ph_operator_name:"Nome do Operador",ph_pressure:"Pressão (bar)",ph_voltage:"Tensão (V)",ph_current:"Corrente (A)",ph_error_code:"Código de Erro",ph_daily_output:"Produção Diária (unidades)",ph_notes:"Notas...",nav_system_intel:"Inteligência do Sistema",nav_roi:"Painel de ROI",refresh_btn:"Atualizar",system_risk_label:"Risco do Sistema",healthy_label:"Saudáveis",at_risk_label:"Em Risco",clusters_title:"Clusters de Máquinas",propagation_title:"Propagação de Anomalias",propagation_hint:"Como uma máquina com falha eleva o risco das vizinhas.",no_propagation:"Nenhuma propagação de anomalia detectada.",avg_risk_label:"Risco médio",added_risk_label:"Risco adicionado",effective_risk_label:"Risco efetivo",simulation_title:"Simulação E-Se",simulation_hint:"Escolha uma máquina, ajuste os sensores e veja a probabilidade de falha mudar.",run_simulation_btn:"Executar Simulação",failure_probability_label:"Probabilidade de Falha",stress_level_label:"Nível de Estresse",predicted_status_label:"Status Previsto",confidence_label:"Confiança",root_cause_title:"Causa Raiz",rul_col:"Vida Útil Restante",rul_healthy:"Saudável",potential_loss_label:"Perda Potencial",saved_label:"Economizado pela IA",wasted_energy_label:"Energia Desperdiçada / mês",efficiency_gain_label:"Ganho de Eficiência",cost_by_machine_title:"Exposição de Custo por Máquina",top_cause_col:"Causa Principal",roi_assumptions_msg:"Premissas: parada {downtime}/h, reparo {hours}h, energia {price}/kWh.",role_label:"Sua Função",role_engineer:"Engenheiro",role_manager:"Gerente",role_admin:"Administrador",cause_bearing_wear:"Desgaste de rolamento",cause_overload_thermal:"Sobrecarga térmica",cause_cooling_failure:"Falha de refrigeração",cause_misalignment:"Desalinhamento do eixo",cause_lubrication_loss:"Perda de lubrificação",cause_normal_operation:"Operação normal",nav_story:"Modo História",story_hint:"Reproduz uma falha de rolamento se desenvolvendo em 22 horas e mostra exatamente quando a IA a detectou — e quanto isso valia.",simulate_failure_btn:"Simular Falha",outcome_title:"Resultado",warning_time_label:"Aviso Antecipado",loss_ignored_label:"Perda sem Ação",loss_acted_label:"Perda com Ação",money_saved_label:"Economizado",timeline_title:"Linha do Tempo da Falha",story_detection_msg:"A IA sinalizou isso {hours} horas antes da quebra, com {risk}% de risco — causa raiz: {cause} ({confidence}% de confiança).",story_stage_healthy:"Saudável",story_stage_early_drift:"Primeiro Desvio",story_stage_ai_detects:"IA Detecta",story_stage_critical:"Crítico",priority_low:"Baixo",priority_medium:"Médio",priority_high:"Alto",priority_critical:"Crítico",action_stop_machine:"Parar máquina",action_inspect_bearings:"Inspecionar rolamentos",action_check_cooling:"Verificar refrigeração",action_schedule_shutdown_24h:"Agendar parada (24h)",action_order_spare_parts:"Pedir peças",action_reduce_load:"Reduzir carga",action_schedule_inspection_72h:"Agendar inspeção (72h)",action_monitor_closely:"Monitorar de perto",action_verify_sensor:"Verificar sensor",action_power_down_idle:"Desligar máquina ociosa",action_review_shift_schedule:"Revisar turnos",action_no_action:"Nenhuma ação necessária",nav_oee:"OEE",nav_workorders:"Ordens de Serviço",oee_hint:"Disponibilidade x Desempenho x Qualidade (ISO 22400). O nível mundial é 85%; uma fábrica típica fica perto de 60%.",availability_label:"Disponibilidade",performance_label:"Desempenho",quality_label:"Qualidade",weakest_factor_label:"Mais fraco",downtime_by_reason_title:"Paradas por Motivo",downtime_cost_label:"Custo de parada",oee_trend_title:"Tendência OEE",shifts_title:"Turnos",shift_col:"Turno",downtime_col:"Parada",log_shift_btn:"Registrar Turno",no_shifts_yet:"Nenhum turno registrado.",minutes_short:"min",range_1d:"Hoje",range_7d:"7 dias",range_30d:"30 dias",all_machines_option:"Todas as máquinas",reason_unspecified:"Não especificado",err_good_exceeds_total:"Peças boas não podem exceder o total",oee_grade_world_class:"Nível mundial",oee_grade_typical:"Típico",oee_grade_low:"Baixo",oee_grade_critical:"Crítico",reason_breakdown:"Quebra",reason_changeover:"Troca de produto",reason_no_material:"Sem material",reason_no_operator:"Sem operador",reason_planned_maintenance:"Manutenção planejada",reason_quality_issue:"Problema de qualidade",reason_setup:"Preparação",reason_other:"Outro",workorders_hint:"Transforma uma previsão de IA em tarefa rastreável — para que um alerta termine em reparo.",new_work_order_btn:"+ Nova Ordem",no_work_orders:"Nenhuma ordem de serviço.",avg_completion_label:"Conclusão méd.",assigned_label:"Atribuído",source_ai:"IA",wo_status_open:"Aberta",wo_status_in_progress:"Em andamento",wo_status_done:"Concluída",wo_status_cancelled:"Cancelada",wo_advance_to_in_progress:"Iniciar trabalho",wo_advance_to_done:"Marcar concluída",ph_shift_name:"Nome do turno",ph_planned_minutes:"Minutos planejados",ph_downtime_minutes:"Minutos de parada",ph_total_units:"Unidades totais",ph_good_units:"Unidades boas",ph_cycle_seconds:"Ciclo ideal (seg)",ph_wo_title:"Título",ph_assigned_to:"Atribuído a",ph_wo_description:"Descrição",overdue_label:"Atrasadas",nav_history:"Histórico",history_hint:"Histórico de sensores armazenado — é assim que você prova o que realmente mudou durante o piloto.",no_history_yet:"Nenhum histórico registrado. Mantenha o Monitoramento aberto por alguns minutos e os dados começarão a acumular.",range_24h:"24 horas",range_3d:"3 dias",range_10d:"10 dias",sensor_trend_title:"Tendência dos Sensores",risk_trend_title:"Tendência de Risco",trend_flat:"sem mudança",trend_rising:"subindo",trend_falling:"caindo",create_work_order_btn:"Criar Ordem",work_order_created_msg:"Ordem de serviço criada",alertreason_immediate_failure_risk:"Risco de falha imediata",alertreason_failure_imminent:"Falha iminente",alertreason_degradation_accelerating:"Degradação acelerando",alertreason_outside_normal_envelope:"Fora da faixa normal",alertreason_pressure_anomaly:"Anomalia de pressão",alertreason_idle_waste:"Desperdício em ociosidade",alertreason_informational:"Informativo",alert_reason_label:"Motivo"},
  ja: {tagline:"グローバル産業インテリジェンスプラットフォーム",live_label:"ライブ",kpi_energy:"エネルギー使用量",kpi_efficiency:"効率",kpi_active:"稼働中の機械",kpi_alerts:"アラート",kwh_unit:"kWh",chart_title:"リアルタイムパフォーマンス",machine_status_title:"機械の状態",status_running:"稼働中",status_warning:"警告",status_critical:"重大",form_title:"工場データ入力",factory_name_label:"工場名",machine_count_label:"機械の数",energy_cost_label:"エネルギーコスト ($/kWh)",machine_type_label:"機械の種類",temperature_label:"温度 (°C)",vibration_label:"振動 (mm/s)",load_label:"負荷 (%)",submit_btn:"工場を分析",submitting:"更新中...",ai_panel_title:"AIインサイト",ai_placeholder:"工場データを送信してAI分析を生成してください。",ai_analyzing:"分析中...",ai_risks:"リスク",ai_efficiency_insights:"効率分析",ai_optimizations:"最適化提案",toast_updated:"工場データが更新されました",toast_analysis_done:"AI分析が完了しました",toast_error:"問題が発生しました",nav_dashboard:"ダッシュボード",nav_factories:"工場",nav_ai_insights:"AIインサイト",logout_btn:"ログアウト",login_title:"おかえりなさい",login_subtitle:"FactoryPulse AI アカウントにログイン",ph_email:"メールアドレス",ph_password:"パスワード",remember_me:"ログイン状態を保持",login_btn:"ログイン",login_link_register:"アカウントをお持ちでないですか？作成する",register_title:"アカウントを作成",register_subtitle:"AIで工場の監視を始めましょう",ph_full_name:"氏名",ph_confirm_password:"パスワードの確認",register_btn:"アカウント作成",register_link_login:"すでにアカウントをお持ちですか？ログイン",err_missing_fields:"すべての項目を入力してください",err_invalid_email:"有効なメールアドレスを入力してください",err_weak_password:"パスワードは8文字以上で、文字と数字を含める必要があります",err_password_mismatch:"パスワードが一致しません",err_invalid_credentials:"メールアドレスまたはパスワードが正しくありません",err_email_taken:"このメールアドレスは既に登録されています",err_generic:"エラーが発生しました。再試行してください",my_factories_title:"マイ工場",add_factory_btn:"+ 工場を追加",edit_factory_btn:"編集",delete_factory_btn:"削除",confirm_delete_factory:"この工場を削除しますか？元に戻せません。",no_factories_yet:"まだ工場を追加していません。",factory_created_toast:"工場が作成・分析されました",factory_updated_toast:"工場が更新されました",factory_deleted_toast:"工場が削除されました",ai_insights_feed_title:"AIインサイトフィード",no_ai_insights_yet:"AIインサイトはまだありません。工場を追加してください。",reanalyze_btn:"再分析",view_insights_btn:"インサイトを見る",created_label:"作成日",cancel_btn:"キャンセル",save_btn:"変更を保存",nav_live_monitor:"ライブモニター",add_machine_scada_btn:"+ 機械を追加",usb_status:"USB:",plc_status:"PLC:",polling_mode:"ポーリング",live_chart_title:"ライブセンサーチャート",machines_table_title:"機械",machine_code_col:"コード",machine_name_col:"名前",status_col:"ステータス",risk_col:"リスク",no_machines_yet:"まだ機械がありません。「+ 機械を追加」をクリックしてください。",section_machine_info:"機械情報",section_sensor_data:"センサーデータ",section_status:"ステータス",section_notes:"メモ",status_stopped:"停止中",status_maintenance:"メンテナンス中",priority_low:"低",priority_normal:"通常",priority_high:"高",priority_critical:"重大",save_and_analyze_btn:"保存して分析",source_col:"データ元",source_auto:"自動 (SCADA)",source_manual:"手動",nav_alerts:"アラート",acknowledge_btn:"確認",acknowledged_label:"確認済み",acknowledge_all_btn:"すべて確認",no_alerts_yet:"アラートはありません。すべて正常に稼働しています。",download_report_btn:"レポート",alert_details_template:"温度 {temp}°C、振動 {vib} mm/s、状態：{status}",section_energy_intel:"エネルギーインテリジェンス",daily_output_hint:"単位あたりのエネルギー消費量（kWh/単位）を計算するために使用します。",energy_insights_title:"エネルギーインテリジェンス",idle_power_title:"アイドル電力検出",idle_active_msg:"機械がアイドル状態です - 現在約{kw} kWが無駄になっています。",idle_none_msg:"アイドル時のエネルギー浪費は検出されていません。",friction_loss_title:"予測エネルギー損失",friction_active_msg:"摩擦増加を検出：電力オーバーヘッド +{pct}%（約{kw} kW増加）。損失を防ぐためメンテナンスを計画してください。",friction_none_msg:"異常な摩擦は検出されていません。",sec_title:"原単位エネルギー消費量",sec_label:"単位あたりkWh",sec_unit:"kWh/単位",sec_no_data_msg:"この指標を表示するには、機械追加時に1日の生産量を入力してください。",optimal_load_title:"最適負荷ゾーン",optimal_load_label:"最適負荷",current_load_label:"現在の負荷",at_optimal_msg:"最適負荷ゾーンで稼働中です。",adjust_to_optimal_msg:"単位あたりのエネルギーを最小化するには、負荷を{pct}%に調整してください。",nav_digital_twin:"デジタルツイン",twin_hint:"ドラッグで回転、スクロールでズーム、機械をクリックするとライブ詳細が表示されます。",twin_unavailable_msg:"3Dビューを読み込めませんでした（Three.jsライブラリのインターネット接続を確認してください）。",failure_prediction_title:"故障予測",report_lib_missing_msg:"PDFエクスポートにはreportlabライブラリが必要です。pip install reportlab を実行してからサーバーを再起動してください。",ph_machine_id:"機械ID（例：M-01）",ph_machine_name:"機械名",ph_factory_section:"工場セクション",ph_operator_name:"オペレーター名",ph_pressure:"圧力（bar）",ph_voltage:"電圧（V）",ph_current:"電流（A）",ph_error_code:"エラーコード",ph_daily_output:"日産量（単位）",ph_notes:"メモ...",nav_system_intel:"システムインテリジェンス",nav_roi:"ROIダッシュボード",refresh_btn:"更新",system_risk_label:"システムリスク",healthy_label:"正常",at_risk_label:"リスクあり",clusters_title:"機械クラスター",propagation_title:"異常伝播",propagation_hint:"故障した機械が近隣機械のリスクをどう高めるか。",no_propagation:"異常伝播は検出されていません。",avg_risk_label:"平均リスク",added_risk_label:"追加リスク",effective_risk_label:"実効リスク",simulation_title:"What-Ifシミュレーション",simulation_hint:"機械を選び、センサー値を変えて故障確率の変化を確認します。",run_simulation_btn:"シミュレーション実行",failure_probability_label:"故障確率",stress_level_label:"ストレスレベル",predicted_status_label:"予測ステータス",confidence_label:"信頼度",root_cause_title:"根本原因",rul_col:"残存耐用時間",rul_healthy:"正常",potential_loss_label:"潜在的損失",saved_label:"AIによる節約",wasted_energy_label:"無駄なエネルギー / 月",efficiency_gain_label:"効率向上",cost_by_machine_title:"機械別コストリスク",top_cause_col:"主要因",roi_assumptions_msg:"前提：停止 {downtime}/時、修理 {hours}時間、電力 {price}/kWh。",role_label:"あなたの役割",role_engineer:"エンジニア",role_manager:"マネージャー",role_admin:"管理者",cause_bearing_wear:"軸受摩耗",cause_overload_thermal:"熱過負荷",cause_cooling_failure:"冷却故障",cause_misalignment:"軸芯ずれ",cause_lubrication_loss:"潤滑不足",cause_normal_operation:"正常運転",nav_story:"ストーリーモード",story_hint:"22時間かけて進行する軸受故障を再現し、AIがいつ検知したか、その価値がいくらだったかを正確に示します。",simulate_failure_btn:"故障をシミュレート",outcome_title:"結果",warning_time_label:"早期警告",loss_ignored_label:"放置した場合の損失",loss_acted_label:"対応した場合の損失",money_saved_label:"節約額",timeline_title:"故障タイムライン",story_detection_msg:"AIは故障の{hours}時間前、リスク{risk}%の時点で検知しました — 根本原因：{cause}（信頼度{confidence}%）。",story_stage_healthy:"正常",story_stage_early_drift:"初期変動",story_stage_ai_detects:"AI検知",story_stage_critical:"重大",priority_low:"低",priority_medium:"中",priority_high:"高",priority_critical:"重大",action_stop_machine:"機械を停止",action_inspect_bearings:"軸受を点検",action_check_cooling:"冷却を確認",action_schedule_shutdown_24h:"停止を計画（24時間）",action_order_spare_parts:"予備部品を発注",action_reduce_load:"負荷を下げる",action_schedule_inspection_72h:"点検を計画（72時間）",action_monitor_closely:"注意深く監視",action_verify_sensor:"センサーを確認",action_power_down_idle:"アイドル機械を停止",action_review_shift_schedule:"シフト計画を見直す",action_no_action:"対応不要",nav_oee:"設備総合効率",nav_workorders:"作業指示",oee_hint:"可用率 x 性能 x 品質 (ISO 22400)。ワールドクラスは85%、一般的な工場は約60%です。",availability_label:"可用率",performance_label:"性能",quality_label:"品質",weakest_factor_label:"最弱項目",downtime_by_reason_title:"理由別停止時間",downtime_cost_label:"停止コスト",oee_trend_title:"OEE推移",shifts_title:"シフト",shift_col:"シフト",downtime_col:"停止",log_shift_btn:"シフトを記録",no_shifts_yet:"まだシフトが記録されていません。",minutes_short:"分",range_1d:"今日",range_7d:"7日間",range_30d:"30日間",all_machines_option:"すべての機械",reason_unspecified:"未指定",err_good_exceeds_total:"良品数は総数を超えられません",oee_grade_world_class:"ワールドクラス",oee_grade_typical:"標準的",oee_grade_low:"低い",oee_grade_critical:"重大",reason_breakdown:"故障",reason_changeover:"段取替え",reason_no_material:"材料切れ",reason_no_operator:"作業者不在",reason_planned_maintenance:"計画保全",reason_quality_issue:"品質問題",reason_setup:"セットアップ",reason_other:"その他",workorders_hint:"AIの予測を追跡可能なタスクに変換 — 警告が確実に修理につながります。",new_work_order_btn:"+ 新規作業指示",no_work_orders:"作業指示はまだありません。",avg_completion_label:"平均完了時間",assigned_label:"担当者",source_ai:"AI",wo_status_open:"未着手",wo_status_in_progress:"進行中",wo_status_done:"完了",wo_status_cancelled:"中止",wo_advance_to_in_progress:"作業開始",wo_advance_to_done:"完了にする",ph_shift_name:"シフト名",ph_planned_minutes:"計画時間（分）",ph_downtime_minutes:"停止時間（分）",ph_total_units:"総数",ph_good_units:"良品数",ph_cycle_seconds:"理想サイクル（秒）",ph_wo_title:"タイトル",ph_assigned_to:"担当者",ph_wo_description:"説明",overdue_label:"期限超過",nav_history:"履歴",history_hint:"保存されたセンサー履歴 — パイロット期間に何が実際に変わったかを証明します。",no_history_yet:"履歴はまだ記録されていません。ライブ監視を数分開いたままにするとデータが蓄積され始めます。",range_24h:"24時間",range_3d:"3日間",range_10d:"10日間",sensor_trend_title:"センサー推移",risk_trend_title:"リスク推移",trend_flat:"変化なし",trend_rising:"上昇",trend_falling:"下降",create_work_order_btn:"作業指示を作成",work_order_created_msg:"作業指示を作成しました",alertreason_immediate_failure_risk:"即時故障リスク",alertreason_failure_imminent:"故障が差し迫っている",alertreason_degradation_accelerating:"劣化が加速中",alertreason_outside_normal_envelope:"正常範囲外",alertreason_pressure_anomaly:"圧力異常",alertreason_idle_waste:"アイドル時のエネルギー浪費",alertreason_informational:"情報",alert_reason_label:"理由"},
  ko: {tagline:"글로벌 산업 인텔리전스 플랫폼",live_label:"실시간",kpi_energy:"에너지 사용량",kpi_efficiency:"효율성",kpi_active:"가동 중인 기계",kpi_alerts:"경고",kwh_unit:"kWh",chart_title:"실시간 성능",machine_status_title:"기계 상태",status_running:"가동 중",status_warning:"경고",status_critical:"심각",form_title:"공장 데이터 입력",factory_name_label:"공장 이름",machine_count_label:"기계 수",energy_cost_label:"에너지 비용 ($/kWh)",machine_type_label:"기계 유형",temperature_label:"온도 (°C)",vibration_label:"진동 (mm/s)",load_label:"부하 (%)",submit_btn:"공장 분석",submitting:"업데이트 중...",ai_panel_title:"AI 인사이트",ai_placeholder:"AI 분석을 생성하려면 공장 데이터를 제출하세요.",ai_analyzing:"분석 중...",ai_risks:"위험 요소",ai_efficiency_insights:"효율성 분석",ai_optimizations:"최적화 제안",toast_updated:"공장 데이터가 업데이트되었습니다",toast_analysis_done:"AI 분석이 완료되었습니다",toast_error:"문제가 발생했습니다",nav_dashboard:"대시보드",nav_factories:"공장",nav_ai_insights:"AI 인사이트",logout_btn:"로그아웃",login_title:"다시 오신 것을 환영합니다",login_subtitle:"FactoryPulse AI 계정에 로그인하세요",ph_email:"이메일",ph_password:"비밀번호",remember_me:"로그인 상태 유지",login_btn:"로그인",login_link_register:"계정이 없으신가요? 계정 만들기",register_title:"계정 만들기",register_subtitle:"AI로 공장 모니터링을 시작하세요",ph_full_name:"성명",ph_confirm_password:"비밀번호 확인",register_btn:"계정 생성",register_link_login:"이미 계정이 있으신가요? 로그인",err_missing_fields:"모든 항목을 입력해주세요",err_invalid_email:"유효한 이메일 주소를 입력하세요",err_weak_password:"비밀번호는 8자 이상, 문자와 숫자를 포함해야 합니다",err_password_mismatch:"비밀번호가 일치하지 않습니다",err_invalid_credentials:"이메일 또는 비밀번호가 올바르지 않습니다",err_email_taken:"이미 등록된 이메일입니다",err_generic:"문제가 발생했습니다. 다시 시도해주세요",my_factories_title:"내 공장",add_factory_btn:"+ 공장 추가",edit_factory_btn:"수정",delete_factory_btn:"삭제",confirm_delete_factory:"이 공장을 삭제하시겠습니까? 되돌릴 수 없습니다.",no_factories_yet:"아직 추가된 공장이 없습니다.",factory_created_toast:"공장이 생성되고 분석되었습니다",factory_updated_toast:"공장이 업데이트되었습니다",factory_deleted_toast:"공장이 삭제되었습니다",ai_insights_feed_title:"AI 인사이트 피드",no_ai_insights_yet:"아직 AI 인사이트가 없습니다. 공장을 추가하세요.",reanalyze_btn:"다시 분석",view_insights_btn:"인사이트 보기",created_label:"생성일",cancel_btn:"취소",save_btn:"변경사항 저장",nav_live_monitor:"실시간 모니터",add_machine_scada_btn:"+ 기계 추가",usb_status:"USB:",plc_status:"PLC:",polling_mode:"폴링",live_chart_title:"실시간 센서 차트",machines_table_title:"기계",machine_code_col:"코드",machine_name_col:"이름",status_col:"상태",risk_col:"위험",no_machines_yet:"아직 기계가 없습니다. \"+ 기계 추가\"를 클릭하세요.",section_machine_info:"기계 정보",section_sensor_data:"센서 데이터",section_status:"상태",section_notes:"메모",status_stopped:"정지됨",status_maintenance:"유지보수 중",priority_low:"낮음",priority_normal:"보통",priority_high:"높음",priority_critical:"긴급",save_and_analyze_btn:"저장 및 분석",source_col:"소스",source_auto:"자동 (SCADA)",source_manual:"수동",nav_alerts:"경고",acknowledge_btn:"확인",acknowledged_label:"확인됨",acknowledge_all_btn:"모두 확인",no_alerts_yet:"경고가 없습니다. 모든 것이 정상 작동 중입니다.",download_report_btn:"보고서",alert_details_template:"온도 {temp}°C, 진동 {vib} mm/s, 상태: {status}",section_energy_intel:"에너지 인텔리전스",daily_output_hint:"단위당 에너지 소비량(kWh/단위)을 계산하는 데 사용됩니다.",energy_insights_title:"에너지 인텔리전스",idle_power_title:"유휴 전력 감지",idle_active_msg:"기계가 유휴 상태입니다 - 현재 약 {kw} kW가 낭비되고 있습니다.",idle_none_msg:"유휴 에너지 낭비가 감지되지 않았습니다.",friction_loss_title:"예측 에너지 손실",friction_active_msg:"마찰 증가 감지: 전력 오버헤드 +{pct}%(약 {kw} kW 추가). 손실을 방지하려면 정비를 예약하세요.",friction_none_msg:"비정상적인 마찰이 감지되지 않았습니다.",sec_title:"단위당 에너지 소비량",sec_label:"단위당 kWh",sec_unit:"kWh/단위",sec_no_data_msg:"이 지표를 보려면 기계 추가 시 일일 생산량을 입력하세요.",optimal_load_title:"최적 부하 구간",optimal_load_label:"최적 부하",current_load_label:"현재 부하",at_optimal_msg:"최적 부하 구간에서 작동 중입니다.",adjust_to_optimal_msg:"단위당 에너지를 최소화하려면 부하를 {pct}%로 조정하세요.",nav_digital_twin:"디지털 트윈",twin_hint:"드래그하여 회전, 스크롤하여 확대/축소, 기계를 클릭하면 실시간 세부정보를 볼 수 있습니다.",twin_unavailable_msg:"3D 보기를 로드할 수 없습니다 (Three.js 라이브러리의 인터넷 연결을 확인하세요).",failure_prediction_title:"고장 예측",report_lib_missing_msg:"PDF 내보내기에는 reportlab 라이브러리가 필요합니다. pip install reportlab을 실행한 후 서버를 재시작하세요.",ph_machine_id:"기계 ID (예: M-01)",ph_machine_name:"기계 이름",ph_factory_section:"공장 구역",ph_operator_name:"운영자 이름",ph_pressure:"압력 (bar)",ph_voltage:"전압 (V)",ph_current:"전류 (A)",ph_error_code:"오류 코드",ph_daily_output:"일일 생산량 (단위)",ph_notes:"메모...",nav_system_intel:"시스템 인텔리전스",nav_roi:"ROI 대시보드",refresh_btn:"새로고침",system_risk_label:"시스템 위험",healthy_label:"정상",at_risk_label:"위험",clusters_title:"기계 클러스터",propagation_title:"이상 전파",propagation_hint:"고장난 기계가 인접 기계의 위험을 어떻게 높이는지.",no_propagation:"이상 전파가 감지되지 않았습니다.",avg_risk_label:"평균 위험",added_risk_label:"추가 위험",effective_risk_label:"실효 위험",simulation_title:"What-If 시뮬레이션",simulation_hint:"기계를 선택하고 센서 값을 조정해 고장 확률 변화를 확인하세요.",run_simulation_btn:"시뮬레이션 실행",failure_probability_label:"고장 확률",stress_level_label:"응력 수준",predicted_status_label:"예측 상태",confidence_label:"신뢰도",root_cause_title:"근본 원인",rul_col:"잔여 수명",rul_healthy:"정상",potential_loss_label:"잠재 손실",saved_label:"AI 절감액",wasted_energy_label:"낭비 에너지 / 월",efficiency_gain_label:"효율 향상",cost_by_machine_title:"기계별 비용 리스크",top_cause_col:"주요 원인",roi_assumptions_msg:"가정: 다운타임 {downtime}/시간, 수리 {hours}시간, 전력 {price}/kWh.",role_label:"귀하의 역할",role_engineer:"엔지니어",role_manager:"매니저",role_admin:"관리자",cause_bearing_wear:"베어링 마모",cause_overload_thermal:"열 과부하",cause_cooling_failure:"냉각 고장",cause_misalignment:"축 정렬 불량",cause_lubrication_loss:"윤활 손실",cause_normal_operation:"정상 운전",nav_story:"스토리 모드",story_hint:"22시간에 걸쳐 진행되는 베어링 고장을 재현하고, AI가 정확히 언제 발견했는지 — 그리고 그 가치가 얼마인지 보여줍니다.",simulate_failure_btn:"고장 시뮬레이션",outcome_title:"결과",warning_time_label:"조기 경고",loss_ignored_label:"방치 시 손실",loss_acted_label:"조치 시 손실",money_saved_label:"절감액",timeline_title:"고장 타임라인",story_detection_msg:"AI가 고장 {hours}시간 전, 위험도 {risk}%에서 감지했습니다 — 근본 원인: {cause} (신뢰도 {confidence}%).",story_stage_healthy:"정상",story_stage_early_drift:"초기 편차",story_stage_ai_detects:"AI 감지",story_stage_critical:"심각",priority_low:"낮음",priority_medium:"보통",priority_high:"높음",priority_critical:"심각",action_stop_machine:"기계 정지",action_inspect_bearings:"베어링 점검",action_check_cooling:"냉각 확인",action_schedule_shutdown_24h:"가동 중단 예약 (24시간)",action_order_spare_parts:"예비 부품 주문",action_reduce_load:"부하 감소",action_schedule_inspection_72h:"점검 예약 (72시간)",action_monitor_closely:"면밀히 모니터링",action_verify_sensor:"센서 확인",action_power_down_idle:"유휴 기계 전원 차단",action_review_shift_schedule:"교대 일정 검토",action_no_action:"조치 불필요",nav_oee:"설비종합효율",nav_workorders:"작업 지시",oee_hint:"가동률 x 성능 x 품질 (ISO 22400). 세계 수준은 85%, 일반 공장은 약 60%입니다.",availability_label:"가동률",performance_label:"성능",quality_label:"품질",weakest_factor_label:"최약점",downtime_by_reason_title:"사유별 정지시간",downtime_cost_label:"정지 비용",oee_trend_title:"OEE 추이",shifts_title:"교대",shift_col:"교대",downtime_col:"정지",log_shift_btn:"교대 기록",no_shifts_yet:"아직 기록된 교대가 없습니다.",minutes_short:"분",range_1d:"오늘",range_7d:"7일",range_30d:"30일",all_machines_option:"모든 기계",reason_unspecified:"미지정",err_good_exceeds_total:"양품 수는 총수를 초과할 수 없습니다",oee_grade_world_class:"세계 수준",oee_grade_typical:"일반",oee_grade_low:"낮음",oee_grade_critical:"심각",reason_breakdown:"고장",reason_changeover:"교체작업",reason_no_material:"자재 부족",reason_no_operator:"작업자 부재",reason_planned_maintenance:"계획 정비",reason_quality_issue:"품질 문제",reason_setup:"셋업",reason_other:"기타",workorders_hint:"AI 예측을 추적 가능한 작업으로 전환 — 경고가 실제 수리로 이어지도록 합니다.",new_work_order_btn:"+ 새 작업 지시",no_work_orders:"작업 지시가 없습니다.",avg_completion_label:"평균 완료",assigned_label:"담당자",source_ai:"AI",wo_status_open:"대기",wo_status_in_progress:"진행 중",wo_status_done:"완료",wo_status_cancelled:"취소",wo_advance_to_in_progress:"작업 시작",wo_advance_to_done:"완료 처리",ph_shift_name:"교대 이름",ph_planned_minutes:"계획 시간(분)",ph_downtime_minutes:"정지 시간(분)",ph_total_units:"총 수량",ph_good_units:"양품 수량",ph_cycle_seconds:"이상 사이클(초)",ph_wo_title:"제목",ph_assigned_to:"담당자",ph_wo_description:"설명",overdue_label:"기한 초과",nav_history:"이력",history_hint:"저장된 센서 이력 — 파일럿 기간에 실제로 무엇이 바뀌었는지 증명합니다.",no_history_yet:"아직 기록된 이력이 없습니다. 실시간 모니터링을 몇 분간 열어두면 데이터가 쌓이기 시작합니다.",range_24h:"24시간",range_3d:"3일",range_10d:"10일",sensor_trend_title:"센서 추이",risk_trend_title:"위험도 추이",trend_flat:"변화 없음",trend_rising:"상승",trend_falling:"하락",create_work_order_btn:"작업 지시 생성",work_order_created_msg:"작업 지시가 생성되었습니다",alertreason_immediate_failure_risk:"즉각적인 고장 위험",alertreason_failure_imminent:"고장 임박",alertreason_degradation_accelerating:"열화 가속 중",alertreason_outside_normal_envelope:"정상 범위 이탈",alertreason_pressure_anomaly:"압력 이상",alertreason_idle_waste:"유휴 에너지 낭비",alertreason_informational:"정보",alert_reason_label:"사유"},
  hi: {tagline:"वैश्विक औद्योगिक बुद्धिमत्ता मंच",live_label:"लाइव",kpi_energy:"ऊर्जा उपयोग",kpi_efficiency:"दक्षता",kpi_active:"सक्रिय मशीनें",kpi_alerts:"अलर्ट",kwh_unit:"kWh",chart_title:"रीयल-टाइम प्रदर्शन",machine_status_title:"मशीन की स्थिति",status_running:"चल रहा है",status_warning:"चेतावनी",status_critical:"गंभीर",form_title:"फ़ैक्टरी डेटा इनपुट",factory_name_label:"फ़ैक्टरी का नाम",machine_count_label:"मशीनों की संख्या",energy_cost_label:"ऊर्जा लागत ($/kWh)",machine_type_label:"मशीन प्रकार",temperature_label:"तापमान (°C)",vibration_label:"कंपन (mm/s)",load_label:"लोड (%)",submit_btn:"फ़ैक्टरी का विश्लेषण करें",submitting:"अद्यतन हो रहा है...",ai_panel_title:"AI अंतर्दृष्टि",ai_placeholder:"AI विश्लेषण उत्पन्न करने के लिए फ़ैक्टरी डेटा सबमिट करें।",ai_analyzing:"विश्लेषण हो रहा है...",ai_risks:"जोखिम",ai_efficiency_insights:"दक्षता विश्लेषण",ai_optimizations:"अनुकूलन सुझाव",toast_updated:"फ़ैक्टरी डेटा अपडेट किया गया",toast_analysis_done:"AI विश्लेषण पूर्ण हुआ",toast_error:"कुछ गलत हो गया",nav_dashboard:"डैशबोर्ड",nav_factories:"फ़ैक्टरियाँ",nav_ai_insights:"AI अंतर्दृष्टि",logout_btn:"लॉग आउट",login_title:"वापसी पर स्वागत है",login_subtitle:"अपने FactoryPulse AI खाते में लॉग इन करें",ph_email:"ईमेल",ph_password:"पासवर्ड",remember_me:"मुझे याद रखें",login_btn:"लॉग इन करें",login_link_register:"खाता नहीं है? एक बनाएं",register_title:"अपना खाता बनाएं",register_subtitle:"AI के साथ अपनी फ़ैक्टरियों की निगरानी शुरू करें",ph_full_name:"पूरा नाम",ph_confirm_password:"पासवर्ड की पुष्टि करें",register_btn:"खाता बनाएं",register_link_login:"पहले से खाता है? लॉग इन करें",err_missing_fields:"कृपया सभी फ़ील्ड भरें",err_invalid_email:"कृपया एक मान्य ईमेल पता दर्ज करें",err_weak_password:"पासवर्ड कम से कम 8 अक्षर, एक अक्षर और एक अंक होना चाहिए",err_password_mismatch:"पासवर्ड मेल नहीं खाते",err_invalid_credentials:"गलत ईमेल या पासवर्ड",err_email_taken:"यह ईमेल पहले से पंजीकृत है",err_generic:"कुछ गलत हो गया। कृपया पुनः प्रयास करें",my_factories_title:"मेरी फ़ैक्टरियाँ",add_factory_btn:"+ फ़ैक्टरी जोड़ें",edit_factory_btn:"संपादित करें",delete_factory_btn:"हटाएं",confirm_delete_factory:"इस फ़ैक्टरी को हटाएं? इसे पूर्ववत नहीं किया जा सकता।",no_factories_yet:"आपने अभी तक कोई फ़ैक्टरी नहीं जोड़ी है।",factory_created_toast:"फ़ैक्टरी बनाई और विश्लेषित की गई",factory_updated_toast:"फ़ैक्टरी अपडेट की गई",factory_deleted_toast:"फ़ैक्टरी हटाई गई",ai_insights_feed_title:"AI अंतर्दृष्टि फ़ीड",no_ai_insights_yet:"अभी तक कोई AI अंतर्दृष्टि नहीं। शुरू करने के लिए एक फ़ैक्टरी जोड़ें।",reanalyze_btn:"पुनः विश्लेषण करें",view_insights_btn:"अंतर्दृष्टि देखें",created_label:"बनाया गया",cancel_btn:"रद्द करें",save_btn:"परिवर्तन सहेजें",nav_live_monitor:"लाइव मॉनिटर",add_machine_scada_btn:"+ मशीन जोड़ें",usb_status:"USB:",plc_status:"PLC:",polling_mode:"पोलिंग",live_chart_title:"लाइव सेंसर चार्ट",machines_table_title:"मशीनें",machine_code_col:"कोड",machine_name_col:"नाम",status_col:"स्थिति",risk_col:"जोखिम",no_machines_yet:"अभी तक कोई मशीन नहीं। \"+ मशीन जोड़ें\" पर क्लिक करें।",section_machine_info:"मशीन जानकारी",section_sensor_data:"सेंसर डेटा",section_status:"स्थिति",section_notes:"नोट्स",status_stopped:"रुकी हुई",status_maintenance:"रखरखाव",priority_low:"कम",priority_normal:"सामान्य",priority_high:"उच्च",priority_critical:"गंभीर",save_and_analyze_btn:"सहेजें और विश्लेषण करें",source_col:"स्रोत",source_auto:"स्वचालित (SCADA)",source_manual:"मैन्युअल",nav_alerts:"अलर्ट",acknowledge_btn:"स्वीकार करें",acknowledged_label:"स्वीकृत",acknowledge_all_btn:"सभी स्वीकार करें",no_alerts_yet:"कोई अलर्ट नहीं। सब कुछ सुचारू रूप से चल रहा है।",download_report_btn:"रिपोर्ट",alert_details_template:"तापमान {temp}°C, कंपन {vib} mm/s, स्थिति: {status}",section_energy_intel:"ऊर्जा इंटेलिजेंस",daily_output_hint:"विशिष्ट ऊर्जा खपत (प्रति इकाई kWh) की गणना के लिए उपयोग किया जाता है।",energy_insights_title:"ऊर्जा इंटेलिजेंस",idle_power_title:"निष्क्रिय शक्ति का पता लगाना",idle_active_msg:"मशीन निष्क्रिय है - अभी लगभग {kw} kW बर्बाद हो रहा है।",idle_none_msg:"कोई निष्क्रिय ऊर्जा बर्बादी नहीं पाई गई।",friction_loss_title:"पूर्वानुमानित ऊर्जा हानि",friction_active_msg:"बढ़ा हुआ घर्षण पाया गया: +{pct}% अतिरिक्त शक्ति (~{kw} kW अतिरिक्त)। हानि रोकने के लिए रखरखाव शेड्यूल करें।",friction_none_msg:"कोई असामान्य घर्षण नहीं पाया गया।",sec_title:"विशिष्ट ऊर्जा खपत",sec_label:"प्रति इकाई kWh",sec_unit:"kWh/इकाई",sec_no_data_msg:"यह मीट्रिक देखने के लिए मशीन जोड़ते समय दैनिक उत्पादन दर्ज करें।",optimal_load_title:"इष्टतम लोड ज़ोन",optimal_load_label:"इष्टतम लोड",current_load_label:"वर्तमान लोड",at_optimal_msg:"इष्टतम लोड ज़ोन में चल रहा है।",adjust_to_optimal_msg:"प्रति इकाई ऊर्जा कम करने के लिए लोड को {pct}% की ओर समायोजित करें।",nav_digital_twin:"डिजिटल ट्विन",twin_hint:"घुमाने के लिए खींचें, ज़ूम करने के लिए स्क्रॉल करें, लाइव विवरण देखने के लिए मशीन पर क्लिक करें।",twin_unavailable_msg:"3D दृश्य लोड नहीं हो सका (Three.js लाइब्रेरी के लिए अपना इंटरनेट कनेक्शन जांचें)।",failure_prediction_title:"विफलता पूर्वानुमान",report_lib_missing_msg:"PDF निर्यात के लिए reportlab लाइब्रेरी की आवश्यकता है। चलाएं: pip install reportlab, फिर सर्वर को पुनः आरंभ करें।",ph_machine_id:"मशीन ID (जैसे M-01)",ph_machine_name:"मशीन नाम",ph_factory_section:"फ़ैक्टरी सेक्शन",ph_operator_name:"ऑपरेटर नाम",ph_pressure:"दबाव (bar)",ph_voltage:"वोल्टेज (V)",ph_current:"करंट (A)",ph_error_code:"त्रुटि कोड",ph_daily_output:"दैनिक उत्पादन (इकाई)",ph_notes:"नोट्स...",nav_system_intel:"सिस्टम इंटेलिजेंस",nav_roi:"ROI डैशबोर्ड",refresh_btn:"रीफ़्रेश",system_risk_label:"सिस्टम जोखिम",healthy_label:"स्वस्थ",at_risk_label:"जोखिम में",clusters_title:"मशीन क्लस्टर",propagation_title:"विसंगति प्रसार",propagation_hint:"एक खराब मशीन पड़ोसी मशीनों का जोखिम कैसे बढ़ाती है।",no_propagation:"कोई विसंगति प्रसार नहीं मिला।",avg_risk_label:"औसत जोखिम",added_risk_label:"अतिरिक्त जोखिम",effective_risk_label:"प्रभावी जोखिम",simulation_title:"व्हाट-इफ़ सिमुलेशन",simulation_hint:"मशीन चुनें, सेंसर मान बदलें और विफलता संभावना देखें।",run_simulation_btn:"सिमुलेशन चलाएं",failure_probability_label:"विफलता संभावना",stress_level_label:"तनाव स्तर",predicted_status_label:"अनुमानित स्थिति",confidence_label:"विश्वास",root_cause_title:"मूल कारण",rul_col:"शेष उपयोगी जीवन",rul_healthy:"स्वस्थ",potential_loss_label:"संभावित हानि",saved_label:"AI द्वारा बचत",wasted_energy_label:"बर्बाद ऊर्जा / माह",efficiency_gain_label:"दक्षता लाभ",cost_by_machine_title:"मशीन अनुसार लागत जोखिम",top_cause_col:"मुख्य कारण",roi_assumptions_msg:"अनुमान: डाउनटाइम {downtime}/घं, मरम्मत {hours} घं, ऊर्जा {price}/kWh।",role_label:"आपकी भूमिका",role_engineer:"इंजीनियर",role_manager:"प्रबंधक",role_admin:"व्यवस्थापक",cause_bearing_wear:"बियरिंग घिसाव",cause_overload_thermal:"तापीय अधिभार",cause_cooling_failure:"शीतलन विफलता",cause_misalignment:"शाफ़्ट असंरेखण",cause_lubrication_loss:"स्नेहन हानि",cause_normal_operation:"सामान्य संचालन",nav_story:"स्टोरी मोड",story_hint:"22 घंटों में विकसित होने वाली बियरिंग विफलता को दोहराता है और दिखाता है कि AI ने इसे कब पकड़ा — और इसका मूल्य क्या था।",simulate_failure_btn:"विफलता का अनुकरण करें",outcome_title:"परिणाम",warning_time_label:"प्रारंभिक चेतावनी",loss_ignored_label:"बिना कार्रवाई हानि",loss_acted_label:"कार्रवाई के साथ हानि",money_saved_label:"बचत",timeline_title:"विफलता समयरेखा",story_detection_msg:"AI ने इसे खराबी से {hours} घंटे पहले, {risk}% जोखिम पर पकड़ा — मूल कारण: {cause} ({confidence}% विश्वास)।",story_stage_healthy:"स्वस्थ",story_stage_early_drift:"प्रारंभिक विचलन",story_stage_ai_detects:"AI ने पहचाना",story_stage_critical:"गंभीर",priority_low:"कम",priority_medium:"मध्यम",priority_high:"उच्च",priority_critical:"गंभीर",action_stop_machine:"मशीन रोकें",action_inspect_bearings:"बियरिंग जांचें",action_check_cooling:"शीतलन जांचें",action_schedule_shutdown_24h:"शटडाउन शेड्यूल करें (24घं)",action_order_spare_parts:"स्पेयर पार्ट्स ऑर्डर करें",action_reduce_load:"लोड कम करें",action_schedule_inspection_72h:"निरीक्षण शेड्यूल करें (72घं)",action_monitor_closely:"बारीकी से निगरानी करें",action_verify_sensor:"सेंसर सत्यापित करें",action_power_down_idle:"निष्क्रिय मशीन बंद करें",action_review_shift_schedule:"शिफ्ट शेड्यूल की समीक्षा करें",action_no_action:"किसी कार्रवाई की आवश्यकता नहीं",nav_oee:"OEE",nav_workorders:"कार्य आदेश",oee_hint:"उपलब्धता x प्रदर्शन x गुणवत्ता (ISO 22400)। विश्व स्तर 85% है; सामान्य कारखाना लगभग 60% पर रहता है।",availability_label:"उपलब्धता",performance_label:"प्रदर्शन",quality_label:"गुणवत्ता",weakest_factor_label:"सबसे कमजोर",downtime_by_reason_title:"कारण अनुसार डाउनटाइम",downtime_cost_label:"डाउनटाइम लागत",oee_trend_title:"OEE रुझान",shifts_title:"शिफ्ट",shift_col:"शिफ्ट",downtime_col:"डाउनटाइम",log_shift_btn:"शिफ्ट दर्ज करें",no_shifts_yet:"अभी तक कोई शिफ्ट दर्ज नहीं।",minutes_short:"मिनट",range_1d:"आज",range_7d:"7 दिन",range_30d:"30 दिन",all_machines_option:"सभी मशीनें",reason_unspecified:"अनिर्दिष्ट",err_good_exceeds_total:"अच्छी इकाइयाँ कुल से अधिक नहीं हो सकतीं",oee_grade_world_class:"विश्व स्तर",oee_grade_typical:"सामान्य",oee_grade_low:"कम",oee_grade_critical:"गंभीर",reason_breakdown:"खराबी",reason_changeover:"चेंजओवर",reason_no_material:"सामग्री नहीं",reason_no_operator:"ऑपरेटर नहीं",reason_planned_maintenance:"नियोजित रखरखाव",reason_quality_issue:"गुणवत्ता समस्या",reason_setup:"सेटअप",reason_other:"अन्य",workorders_hint:"AI पूर्वानुमान को ट्रैक करने योग्य कार्य में बदलता है — ताकि चेतावनी वास्तव में मरम्मत पर समाप्त हो।",new_work_order_btn:"+ नया कार्य आदेश",no_work_orders:"अभी तक कोई कार्य आदेश नहीं।",avg_completion_label:"औसत पूर्णता",assigned_label:"सौंपा गया",source_ai:"AI",wo_status_open:"खुला",wo_status_in_progress:"प्रगति में",wo_status_done:"पूर्ण",wo_status_cancelled:"रद्द",wo_advance_to_in_progress:"कार्य शुरू करें",wo_advance_to_done:"पूर्ण चिह्नित करें",ph_shift_name:"शिफ्ट नाम",ph_planned_minutes:"नियोजित मिनट",ph_downtime_minutes:"डाउनटाइम मिनट",ph_total_units:"कुल इकाइयाँ",ph_good_units:"अच्छी इकाइयाँ",ph_cycle_seconds:"आदर्श चक्र (सेकंड)",ph_wo_title:"शीर्षक",ph_assigned_to:"सौंपा गया",ph_wo_description:"विवरण",overdue_label:"समय सीमा पार",nav_history:"इतिहास",history_hint:"संग्रहीत सेंसर इतिहास — इसी से आप साबित करते हैं कि पायलट के दौरान वास्तव में क्या बदला।",no_history_yet:"अभी तक कोई इतिहास दर्ज नहीं। लाइव मॉनिटर को कुछ मिनट खुला रखें, डेटा जमा होने लगेगा।",range_24h:"24 घंटे",range_3d:"3 दिन",range_10d:"10 दिन",sensor_trend_title:"सेंसर रुझान",risk_trend_title:"जोखिम रुझान",trend_flat:"कोई बदलाव नहीं",trend_rising:"बढ़ रहा",trend_falling:"घट रहा",create_work_order_btn:"कार्य आदेश बनाएं",work_order_created_msg:"कार्य आदेश बनाया गया",alertreason_immediate_failure_risk:"तत्काल विफलता जोखिम",alertreason_failure_imminent:"विफलता निकट",alertreason_degradation_accelerating:"गिरावट तेज हो रही",alertreason_outside_normal_envelope:"सामान्य सीमा से बाहर",alertreason_pressure_anomaly:"दबाव विसंगति",alertreason_idle_waste:"निष्क्रिय ऊर्जा बर्बादी",alertreason_informational:"सूचनात्मक",alert_reason_label:"कारण"},
  uz: {tagline:"Global sanoat intellekti platformasi",live_label:"Jonli",kpi_energy:"Energiya sarfi",kpi_efficiency:"Samaradorlik",kpi_active:"Faol stanoklar",kpi_alerts:"Ogohlantirishlar",kwh_unit:"kWh",chart_title:"Real vaqtdagi ko'rsatkichlar",machine_status_title:"Stanoklar holati",status_running:"Ishlamoqda",status_warning:"Ogohlantirish",status_critical:"Muhim",form_title:"Zavod ma'lumotlarini kiritish",factory_name_label:"Zavod nomi",machine_count_label:"Stanoklar soni",energy_cost_label:"Energiya narxi ($/kWh)",machine_type_label:"Stanok turi",temperature_label:"Harorat (°C)",vibration_label:"Tebranish (mm/s)",load_label:"Yuklama (%)",submit_btn:"Zavodni tahlil qilish",submitting:"Yangilanmoqda...",ai_panel_title:"AI tahlili",ai_placeholder:"AI tahlilini olish uchun zavod ma'lumotlarini yuboring.",ai_analyzing:"Tahlil qilinmoqda...",ai_risks:"Xavflar",ai_efficiency_insights:"Samaradorlik tahlili",ai_optimizations:"Optimallashtirish tavsiyalari",toast_updated:"Zavod ma'lumotlari yangilandi",toast_analysis_done:"AI tahlili yakunlandi",toast_error:"Xatolik yuz berdi",nav_dashboard:"Boshqaruv paneli",nav_factories:"Zavodlar",nav_ai_insights:"AI tahlili",logout_btn:"Chiqish",login_title:"Xush kelibsiz",login_subtitle:"FactoryPulse AI hisobingizga kiring",ph_email:"Elektron pochta",ph_password:"Parol",remember_me:"Meni eslab qol",login_btn:"Kirish",login_link_register:"Hisobingiz yo'qmi? Yarating",register_title:"Hisob yarating",register_subtitle:"Zavodlaringizni AI bilan kuzatishni boshlang",ph_full_name:"To'liq ism",ph_confirm_password:"Parolni tasdiqlang",register_btn:"Hisob yaratish",register_link_login:"Hisobingiz bormi? Kiring",err_missing_fields:"Barcha maydonlarni to'ldiring",err_invalid_email:"Yaroqli elektron pochta manzilini kiriting",err_weak_password:"Parol kamida 8 belgidan, harf va raqamdan iborat bo'lishi kerak",err_password_mismatch:"Parollar mos kelmaydi",err_invalid_credentials:"Elektron pochta yoki parol noto'g'ri",err_email_taken:"Bu elektron pochta allaqachon ro'yxatdan o'tgan",err_generic:"Xatolik yuz berdi. Qaytadan urinib ko'ring",my_factories_title:"Mening Zavodlarim",add_factory_btn:"+ Zavod qo'shish",edit_factory_btn:"Tahrirlash",delete_factory_btn:"O'chirish",confirm_delete_factory:"Bu zavodni o'chirasizmi? Buni bekor qilib bo'lmaydi.",no_factories_yet:"Siz hali hech qanday zavod qo'shmagansiz.",factory_created_toast:"Zavod yaratildi va tahlil qilindi",factory_updated_toast:"Zavod yangilandi",factory_deleted_toast:"Zavod o'chirildi",ai_insights_feed_title:"AI Tahlili Lentasi",no_ai_insights_yet:"Hali AI tahlili yo'q. Boshlash uchun zavod qo'shing.",reanalyze_btn:"Qayta tahlil qilish",view_insights_btn:"Tahlilni ko'rish",created_label:"Yaratilgan",cancel_btn:"Bekor qilish",save_btn:"O'zgarishlarni saqlash",nav_live_monitor:"Jonli monitoring",add_machine_scada_btn:"+ Stanok qo'shish",usb_status:"USB:",plc_status:"PLC:",polling_mode:"So'rov",live_chart_title:"Jonli sensor grafigi",machines_table_title:"Stanoklar",machine_code_col:"Kod",machine_name_col:"Nomi",status_col:"Holat",risk_col:"Xavf",no_machines_yet:"Hali stanoklar yo'q. \"+ Stanok qo'shish\"ni bosing.",section_machine_info:"Stanok ma'lumoti",section_sensor_data:"Sensor ma'lumotlari",section_status:"Holat",section_notes:"Eslatmalar",status_stopped:"To'xtatilgan",status_maintenance:"Texnik xizmat",priority_low:"Past",priority_normal:"Oddiy",priority_high:"Yuqori",priority_critical:"Muhim",save_and_analyze_btn:"Saqlash va tahlil qilish",source_col:"Manba",source_auto:"Avtomatik (SCADA)",source_manual:"Qo'lda",nav_alerts:"Ogohlantirishlar",acknowledge_btn:"Tasdiqlash",acknowledged_label:"Tasdiqlangan",acknowledge_all_btn:"Barchasini Tasdiqlash",no_alerts_yet:"Ogohlantirishlar yo'q. Hammasi yaxshi ishlamoqda.",download_report_btn:"Hisobot",alert_details_template:"Harorat {temp}°C, tebranish {vib} mm/s, holat: {status}",section_energy_intel:"Energiya intellekti",daily_output_hint:"Solishtirma energiya sarfini (birlik uchun kVt·soat) hisoblash uchun ishlatiladi.",energy_insights_title:"Energiya intellekti",idle_power_title:"Bo'sh yurish quvvatini aniqlash",idle_active_msg:"Stanok bo'sh turibdi - hozir taxminan {kw} kVt behuda sarflanmoqda.",idle_none_msg:"Bo'sh yurish energiya isrofi aniqlanmadi.",friction_loss_title:"Bashoratli energiya yo'qotishi",friction_active_msg:"Oshgan ishqalanish aniqlandi: +{pct}% ortiqcha quvvat (~{kw} kVt qo'shimcha). Isrofni oldini olish uchun texnik xizmatni rejalashtiring.",friction_none_msg:"Anomal ishqalanish aniqlanmadi.",sec_title:"Solishtirma energiya sarfi",sec_label:"birlik uchun kVt·soat",sec_unit:"kVt·soat/birlik",sec_no_data_msg:"Bu ko'rsatkichni ko'rish uchun stanok qo'shishda kunlik ishlab chiqarishni kiriting.",optimal_load_title:"Optimal yuklama zonasi",optimal_load_label:"Optimal yuklama",current_load_label:"Joriy yuklama",at_optimal_msg:"Optimal yuklama zonasida ishlamoqda.",adjust_to_optimal_msg:"Birlik uchun energiyani minimallashtirish uchun yuklamani {pct}% ga moslang.",nav_digital_twin:"Raqamli egizak",twin_hint:"Aylantirish uchun torting, kattalashtirish uchun aylantiring, jonli tafsilotlar uchun stanokni bosing.",twin_unavailable_msg:"3D ko'rinishni yuklab bo'lmadi (Three.js kutubxonasi uchun internet aloqangizni tekshiring).",failure_prediction_title:"Nosozlik bashorati",report_lib_missing_msg:"PDF eksport qilish uchun reportlab kutubxonasi kerak. Bajaring: pip install reportlab, keyin serverni qayta ishga tushiring.",ph_machine_id:"Stanok ID (masalan M-01)",ph_machine_name:"Stanok nomi",ph_factory_section:"Zavod uchastkasi",ph_operator_name:"Operator ismi",ph_pressure:"Bosim (bar)",ph_voltage:"Kuchlanish (V)",ph_current:"Tok (A)",ph_error_code:"Xato kodi",ph_daily_output:"Kunlik ishlab chiqarish (birlik)",ph_notes:"Eslatmalar...",nav_system_intel:"Tizim intellekti",nav_roi:"ROI paneli",refresh_btn:"Yangilash",system_risk_label:"Tizim xavfi",healthy_label:"Sog'lom",at_risk_label:"Xavf ostida",clusters_title:"Stanok klasterlari",propagation_title:"Anomaliya tarqalishi",propagation_hint:"Nosoz stanok qo'shni stanoklar xavfini qanday oshiradi.",no_propagation:"Anomaliya tarqalishi aniqlanmadi.",avg_risk_label:"O'rtacha xavf",added_risk_label:"Qo'shilgan xavf",effective_risk_label:"Haqiqiy xavf",simulation_title:"Nima-bo'lsa simulyatsiyasi",simulation_hint:"Stanokni tanlang, sensor qiymatlarini o'zgartiring va nosozlik ehtimolini kuzating.",run_simulation_btn:"Simulyatsiyani ishga tushirish",failure_probability_label:"Nosozlik ehtimoli",stress_level_label:"Zo'riqish darajasi",predicted_status_label:"Bashorat holati",confidence_label:"Ishonchlilik",root_cause_title:"Asosiy sabab",rul_col:"Qolgan foydali muddat",rul_healthy:"Sog'lom",potential_loss_label:"Potentsial yo'qotish",saved_label:"AI tejadi",wasted_energy_label:"Isrof energiya / oy",efficiency_gain_label:"Samaradorlik o'sishi",cost_by_machine_title:"Stanoklar bo'yicha xarajat xavfi",top_cause_col:"Asosiy sabab",roi_assumptions_msg:"Taxminlar: to'xtash {downtime}/soat, ta'mir {hours} soat, energiya {price}/kWh.",role_label:"Sizning rolingiz",role_engineer:"Muhandis",role_manager:"Menejer",role_admin:"Administrator",cause_bearing_wear:"Podshipnik eskirishi",cause_overload_thermal:"Termik ortiqcha yuk",cause_cooling_failure:"Sovutish nosozligi",cause_misalignment:"Val nomuvofiqligi",cause_lubrication_loss:"Moylash yo'qolishi",cause_normal_operation:"Normal ish",nav_story:"Hikoya rejimi",story_hint:"22 soat davomida rivojlanayotgan podshipnik nosozligini qayta ijro etadi va AI uni qachon aniqlaganini — va bu qancha turishini ko'rsatadi.",simulate_failure_btn:"Nosozlikni modellashtirish",outcome_title:"Natija",warning_time_label:"Erta ogohlantirish",loss_ignored_label:"Harakatsiz yo'qotish",loss_acted_label:"Harakat bilan yo'qotish",money_saved_label:"Tejalgan",timeline_title:"Nosozlik xronologiyasi",story_detection_msg:"AI buni buzilishdan {hours} soat oldin, {risk}% xavfda aniqladi — asosiy sabab: {cause} ({confidence}% ishonch).",story_stage_healthy:"Sog'lom",story_stage_early_drift:"Dastlabki og'ish",story_stage_ai_detects:"AI aniqladi",story_stage_critical:"Tanqidiy",priority_low:"Past",priority_medium:"O'rta",priority_high:"Yuqori",priority_critical:"Tanqidiy",action_stop_machine:"Stanokni to'xtatish",action_inspect_bearings:"Podshipniklarni tekshirish",action_check_cooling:"Sovutishni tekshirish",action_schedule_shutdown_24h:"To'xtatishni rejalashtirish (24s)",action_order_spare_parts:"Ehtiyot qismlar buyurtma qilish",action_reduce_load:"Yuklamani kamaytirish",action_schedule_inspection_72h:"Tekshiruvni rejalashtirish (72s)",action_monitor_closely:"Diqqat bilan kuzatish",action_verify_sensor:"Sensorni tekshirish",action_power_down_idle:"Bo'sh stanokni o'chirish",action_review_shift_schedule:"Smena jadvalini ko'rib chiqish",action_no_action:"Harakat kerak emas",nav_oee:"OEE",nav_workorders:"Ish buyruqlari",oee_hint:"Mavjudlik x Unumdorlik x Sifat (ISO 22400). Jahon darajasi 85%, odatiy zavod esa 60% atrofida.",availability_label:"Mavjudlik",performance_label:"Unumdorlik",quality_label:"Sifat",weakest_factor_label:"Eng zaif",downtime_by_reason_title:"Sabab bo'yicha to'xtash",downtime_cost_label:"To'xtash narxi",oee_trend_title:"OEE dinamikasi",shifts_title:"Smenalar",shift_col:"Smena",downtime_col:"To'xtash",log_shift_btn:"Smena kiritish",no_shifts_yet:"Hali smenalar kiritilmagan.",minutes_short:"daq",range_1d:"Bugun",range_7d:"7 kun",range_30d:"30 kun",all_machines_option:"Barcha stanoklar",reason_unspecified:"Ko'rsatilmagan",err_good_exceeds_total:"Yaroqli soni umumiy sondan ko'p bo'la olmaydi",oee_grade_world_class:"Jahon darajasi",oee_grade_typical:"Odatiy",oee_grade_low:"Past",oee_grade_critical:"Tanqidiy",reason_breakdown:"Buzilish",reason_changeover:"Qayta sozlash",reason_no_material:"Material yo'q",reason_no_operator:"Operator yo'q",reason_planned_maintenance:"Rejali texnik xizmat",reason_quality_issue:"Sifat muammosi",reason_setup:"Sozlash",reason_other:"Boshqa",workorders_hint:"AI bashoratini kuzatiladigan vazifaga aylantiradi — ogohlantirish haqiqiy ta'mir bilan tugasin.",new_work_order_btn:"+ Yangi ish buyrug'i",no_work_orders:"Hali ish buyruqlari yo'q.",avg_completion_label:"O'rt. bajarilish",assigned_label:"Tayinlangan",source_ai:"AI",wo_status_open:"Ochiq",wo_status_in_progress:"Bajarilmoqda",wo_status_done:"Bajarildi",wo_status_cancelled:"Bekor qilindi",wo_advance_to_in_progress:"Ishni boshlash",wo_advance_to_done:"Bajarildi deb belgilash",ph_shift_name:"Smena nomi",ph_planned_minutes:"Rejali daqiqalar",ph_downtime_minutes:"To'xtash daqiqalari",ph_total_units:"Umumiy soni",ph_good_units:"Yaroqli soni",ph_cycle_seconds:"Ideal sikl (son)",ph_wo_title:"Sarlavha",ph_assigned_to:"Mas'ul",ph_wo_description:"Tavsif",overdue_label:"Muddati o'tgan",nav_history:"Tarix",history_hint:"Saqlangan sensor tarixi — pilot davomida nima o'zgarganini shu bilan isbotlaysiz.",no_history_yet:"Tarix hali yozilmagan. Jonli monitoringni bir necha daqiqa ochiq qoldiring, ma'lumot to'plana boshlaydi.",range_24h:"24 soat",range_3d:"3 kun",range_10d:"10 kun",sensor_trend_title:"Sensor dinamikasi",risk_trend_title:"Xavf dinamikasi",trend_flat:"o'zgarishsiz",trend_rising:"o'smoqda",trend_falling:"kamaymoqda",create_work_order_btn:"Ish buyrug'i yaratish",work_order_created_msg:"Ish buyrug'i yaratildi",alertreason_immediate_failure_risk:"Darhol buzilish xavfi",alertreason_failure_imminent:"Nosozlik yaqin",alertreason_degradation_accelerating:"Eskirish tezlashmoqda",alertreason_outside_normal_envelope:"Normal chegaradan tashqarida",alertreason_pressure_anomaly:"Bosim anomaliyasi",alertreason_idle_waste:"Bo'sh yurish isrofi",alertreason_informational:"Ma'lumot",alert_reason_label:"Sabab"},
  ky: {tagline:"Глобалдык өнөр жай интеллект платформасы",live_label:"Түз эфир",kpi_energy:"Энергия сарпталышы",kpi_efficiency:"Эффективдүүлүк",kpi_active:"Активдүү станоктор",kpi_alerts:"Дабылдар",kwh_unit:"кВт·саат",chart_title:"Реалдуу убакыттагы көрсөткүчтөр",machine_status_title:"Станоктордун абалы",status_running:"Иштеп жатат",status_warning:"Эскертүү",status_critical:"Олуттуу",form_title:"Завод маалыматтарын киргизүү",factory_name_label:"Заводдун аты",machine_count_label:"Станоктордун саны",energy_cost_label:"Энергия наркы ($/кВт·саат)",machine_type_label:"Станоктун түрү",temperature_label:"Температура (°C)",vibration_label:"Дирилдөө (мм/с)",load_label:"Жүктөм (%)",submit_btn:"Заводду талдоо",submitting:"Жаңыртылууда...",ai_panel_title:"AI-талдоо",ai_placeholder:"AI-талдоо алуу үчүн завод маалыматтарын жөнөтүңүз.",ai_analyzing:"Талдануда...",ai_risks:"Тобокелдиктер",ai_efficiency_insights:"Эффективдүүлүк талдоосу",ai_optimizations:"Оптималдаштыруу сунуштары",toast_updated:"Завод маалыматтары жаңыртылды",toast_analysis_done:"AI-талдоо аяктады",toast_error:"Ката кетти",nav_dashboard:"Башкаруу панели",nav_factories:"Заводдор",nav_ai_insights:"AI-талдоо",logout_btn:"Чыгуу",login_title:"Кайра кош келиңиз",login_subtitle:"FactoryPulse AI каттоо эсебиңизге кириңиз",ph_email:"Электрондук почта",ph_password:"Сырсөз",remember_me:"Мени эстеп кал",login_btn:"Кирүү",login_link_register:"Каттоо эсебиңиз жокпу? Түзүү",register_title:"Каттоо эсебин түзүү",register_subtitle:"Заводдоруңузду AI менен байкоону баштаңыз",ph_full_name:"Толук аты-жөнү",ph_confirm_password:"Сырсөздү ырастаңыз",register_btn:"Каттоо эсебин түзүү",register_link_login:"Каттоо эсебиңиз барбы? Кирүү",err_missing_fields:"Бардык талааларды толтуруңуз",err_invalid_email:"Жарактуу электрондук почта дарегин киргизиңиз",err_weak_password:"Сырсөз кеминде 8 белги, тамга жана сан камтышы керек",err_password_mismatch:"Сырсөздөр дал келбейт",err_invalid_credentials:"Электрондук почта же сырсөз туура эмес",err_email_taken:"Бул электрондук почта мурунтан катталган",err_generic:"Ката кетти. Кайра аракет кылыңыз",my_factories_title:"Менин Заводдорум",add_factory_btn:"+ Завод кошуу",edit_factory_btn:"Түзөтүү",delete_factory_btn:"Өчүрүү",confirm_delete_factory:"Бул заводду өчүрөсүзбү? Бул аракетти артка кайтарууга болбойт.",no_factories_yet:"Сиз азырынча эч кандай завод кошкон жоксуз.",factory_created_toast:"Завод түзүлдү жана талданды",factory_updated_toast:"Завод жаңыртылды",factory_deleted_toast:"Завод өчүрүлдү",ai_insights_feed_title:"AI-талдоо тизмеси",no_ai_insights_yet:"Азырынча AI-талдоо жок. Баштоо үчүн завод кошуңуз.",reanalyze_btn:"Кайра талдоо",view_insights_btn:"Талдоону көрүү",created_label:"Түзүлгөн күнү",cancel_btn:"Жокко чыгаруу",save_btn:"Өзгөртүүлөрдү сактоо",nav_live_monitor:"Түз мониторинг",add_machine_scada_btn:"+ Станок кошуу",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Сурам",live_chart_title:"Реалдуу убакыттагы сенсор графиги",machines_table_title:"Станоктор",machine_code_col:"Код",machine_name_col:"Аты",status_col:"Абалы",risk_col:"Тобокелдик",no_machines_yet:"Азырынча станоктор жок. \"+ Станок кошуу\"ну басыңыз.",section_machine_info:"Станок маалыматы",section_sensor_data:"Сенсор маалыматтары",section_status:"Абалы",section_notes:"Эскертүүлөр",status_stopped:"Токтотулган",status_maintenance:"Тейлөө",priority_low:"Төмөн",priority_normal:"Кадимки",priority_high:"Жогору",priority_critical:"Олуттуу",save_and_analyze_btn:"Сактоо жана талдоо",source_col:"Булак",source_auto:"Автоматтык (SCADA)",source_manual:"Кол менен",nav_alerts:"Дабылдар",acknowledge_btn:"Ырастоо",acknowledged_label:"Ырасталды",acknowledge_all_btn:"Баарын ырастоо",no_alerts_yet:"Дабылдар жок. Баары жакшы иштеп жатат.",download_report_btn:"Отчет",alert_details_template:"Температура {temp}°C, дирилдөө {vib} мм/с, абалы: {status}",section_energy_intel:"Энергия интеллекти",daily_output_hint:"Бирдикке кеткен энергия сарптоону (бирдик үчүн кВт·саат) эсептөө үчүн колдонулат.",energy_insights_title:"Энергия интеллекти",idle_power_title:"Бош жүрүштү аныктоо",idle_active_msg:"Станок бош турат - учурда болжол менен {kw} кВт бекер коротулуп жатат.",idle_none_msg:"Бош жүрүш чыгымы табылган жок.",friction_loss_title:"Болжолдуу энергия жоготуусу",friction_active_msg:"Жогорулаган үйкөлүш табылды: +{pct}% кошумча кубат (~{kw} кВт кошумча). Чыгымдын алдын алуу үчүн тейлөөнү пландаштырыңыз.",friction_none_msg:"Аномалдуу үйкөлүш табылган жок.",sec_title:"Бирдикке кеткен энергия",sec_label:"бирдик үчүн кВт·саат",sec_unit:"кВт·саат/бирдик",sec_no_data_msg:"Бул көрсөткүчтү көрүү үчүн станок кошууда күндөлүк өндүрүштү киргизиңиз.",optimal_load_title:"Оптималдуу жүктөм аймагы",optimal_load_label:"Оптималдуу жүктөм",current_load_label:"Учурдагы жүктөм",at_optimal_msg:"Оптималдуу жүктөм аймагында иштеп жатат.",adjust_to_optimal_msg:"Бирдикке кеткен энергияны азайтуу үчүн жүктөмдү {pct}%га жакындатыңыз.",nav_digital_twin:"Санарип эгиз",twin_hint:"Айландыруу үчүн сүйрөңүз, чоңойтуу үчүн айландырыңыз, түз маалымат үчүн станокту басыңыз.",twin_unavailable_msg:"3D көрүнүш жүктөлгөн жок (Three.js китепканасы үчүн интернет байланышын текшериңиз).",failure_prediction_title:"Бузулуу болжому",report_lib_missing_msg:"PDF экспорттоо үчүн reportlab китепканасы керек. Аткарыңыз: pip install reportlab, андан кийин серверди кайра иштетиңиз.",ph_machine_id:"Станок ID (мис. M-01)",ph_machine_name:"Станок аты",ph_factory_section:"Завод участогу",ph_operator_name:"Оператордун аты",ph_pressure:"Басым (bar)",ph_voltage:"Чыңалуу (V)",ph_current:"Ток (A)",ph_error_code:"Ката коду",ph_daily_output:"Күндөлүк өндүрүш (бирдик)",ph_notes:"Эскертүүлөр...",nav_system_intel:"Тутум интеллекти",nav_roi:"ROI панели",refresh_btn:"Жаңыртуу",system_risk_label:"Тутум тобокелдиги",healthy_label:"Сак",at_risk_label:"Тобокелдикте",clusters_title:"Станок кластерлери",propagation_title:"Аномалиянын жайылышы",propagation_hint:"Бузулган станок кошуналарынын тобокелдигин кантип жогорулатат.",no_propagation:"Аномалиянын жайылышы аныкталган жок.",avg_risk_label:"Орточо тобокелдик",added_risk_label:"Кошумча тобокелдик",effective_risk_label:"Иш жүзүндөгү тобокелдик",simulation_title:"Эмне-болсо симуляциясы",simulation_hint:"Станокту тандап, сенсор маанилерин өзгөртүп, бузулуу ыктымалдыгын карап көрүңүз.",run_simulation_btn:"Симуляцияны иштетүү",failure_probability_label:"Бузулуу ыктымалдыгы",stress_level_label:"Чыңалуу деңгээли",predicted_status_label:"Болжолдонгон абал",confidence_label:"Ишенимдүүлүк",root_cause_title:"Негизги себеп",rul_col:"Калган ресурс",rul_healthy:"Сак",potential_loss_label:"Потенциалдуу жоготуу",saved_label:"AI үнөмдөдү",wasted_energy_label:"Ысырап энергия / ай",efficiency_gain_label:"Эффективдүүлүк өсүшү",cost_by_machine_title:"Станоктор боюнча чыгым тобокелдиги",top_cause_col:"Негизги себеп",roi_assumptions_msg:"Болжолдор: токтоп калуу {downtime}/саат, оңдоо {hours} саат, энергия {price}/kWh.",role_label:"Сиздин ролуңуз",role_engineer:"Инженер",role_manager:"Менеджер",role_admin:"Администратор",cause_bearing_wear:"Подшипниктин эскириши",cause_overload_thermal:"Жылуулук ашыкча жүктөө",cause_cooling_failure:"Муздатуу бузулушу",cause_misalignment:"Валдын борборунан жылышы",cause_lubrication_loss:"Майлоонун жоголушу",cause_normal_operation:"Кадимки иштөө",nav_story:"Окуя режими",story_hint:"22 саат ичинде өнүгүп жаткан подшипник бузулушун кайра ойнотуп, AI аны так качан аныктаганын — жана бул канча турганын көрсөтөт.",simulate_failure_btn:"Бузулууну моделдөө",outcome_title:"Жыйынтык",warning_time_label:"Эрте эскертүү",loss_ignored_label:"Аракетсиз чыгым",loss_acted_label:"Аракет менен чыгым",money_saved_label:"Үнөмдөлдү",timeline_title:"Бузулуу хронологиясы",story_detection_msg:"AI муну бузулуудан {hours} саат мурун, {risk}% тобокелдикте аныктады — негизги себеп: {cause} ({confidence}% ишеним).",story_stage_healthy:"Сак",story_stage_early_drift:"Алгачкы четтөө",story_stage_ai_detects:"AI аныктады",story_stage_critical:"Критикалык",priority_low:"Төмөн",priority_medium:"Орточо",priority_high:"Жогору",priority_critical:"Критикалык",action_stop_machine:"Станокту токтотуу",action_inspect_bearings:"Подшипниктерди текшерүү",action_check_cooling:"Муздатууну текшерүү",action_schedule_shutdown_24h:"Токтотууну пландаштыруу (24с)",action_order_spare_parts:"Камдык бөлүктөрдү заказ кылуу",action_reduce_load:"Жүктөмдү азайтуу",action_schedule_inspection_72h:"Текшерүүнү пландаштыруу (72с)",action_monitor_closely:"Кылдат байкоо",action_verify_sensor:"Сенсорду текшерүү",action_power_down_idle:"Бош станокту өчүрүү",action_review_shift_schedule:"Смена графигин кароо",action_no_action:"Аракет талап кылынбайт",nav_oee:"OEE",nav_workorders:"Иш буйруктары",oee_hint:"Жеткиликтүүлүк x Өндүрүмдүүлүк x Сапат (ISO 22400). Дүйнөлүк деңгээл 85%, кадимки завод 60% чамасында.",availability_label:"Жеткиликтүүлүк",performance_label:"Өндүрүмдүүлүк",quality_label:"Сапат",weakest_factor_label:"Эң алсыз",downtime_by_reason_title:"Себеби боюнча токтоп калуу",downtime_cost_label:"Токтоп калуу баасы",oee_trend_title:"OEE динамикасы",shifts_title:"Сменалар",shift_col:"Смена",downtime_col:"Токтоп калуу",log_shift_btn:"Смена киргизүү",no_shifts_yet:"Азырынча сменалар киргизилген жок.",minutes_short:"мүн",range_1d:"Бүгүн",range_7d:"7 күн",range_30d:"30 күн",all_machines_option:"Бардык станоктор",reason_unspecified:"Көрсөтүлгөн эмес",err_good_exceeds_total:"Жарамдуу саны жалпы санынан ашпашы керек",oee_grade_world_class:"Дүйнөлүк деңгээл",oee_grade_typical:"Кадимки",oee_grade_low:"Төмөн",oee_grade_critical:"Критикалык",reason_breakdown:"Бузулуу",reason_changeover:"Кайра жөндөө",reason_no_material:"Материал жок",reason_no_operator:"Оператор жок",reason_planned_maintenance:"Пландуу тейлөө",reason_quality_issue:"Сапат маселеси",reason_setup:"Орнотуу",reason_other:"Башка",workorders_hint:"AI болжолун көзөмөлдөнүүчү тапшырмага айландырат — эскертүү чыныгы оңдоо менен аякташы үчүн.",new_work_order_btn:"+ Жаңы иш буйругу",no_work_orders:"Азырынча иш буйруктары жок.",avg_completion_label:"Орт. аткаруу",assigned_label:"Дайындалган",source_ai:"AI",wo_status_open:"Ачык",wo_status_in_progress:"Аткарылууда",wo_status_done:"Аткарылды",wo_status_cancelled:"Жокко чыгарылды",wo_advance_to_in_progress:"Ишти баштоо",wo_advance_to_done:"Аткарылды деп белгилөө",ph_shift_name:"Смена аты",ph_planned_minutes:"Пландуу мүнөттөр",ph_downtime_minutes:"Токтоп калган мүнөттөр",ph_total_units:"Жалпы саны",ph_good_units:"Жарамдуу саны",ph_cycle_seconds:"Идеалдуу цикл (сек)",ph_wo_title:"Аталышы",ph_assigned_to:"Жооптуу",ph_wo_description:"Сүрөттөмө",overdue_label:"Мөөнөтү өткөн",nav_history:"Тарых",history_hint:"Сакталган сенсор тарыхы — пилот учурунда эмне өзгөргөнүн ушуну менен далилдейсиз.",no_history_yet:"Тарых азырынча жазылган жок. Түз мониторингди бир нече мүнөт ачык калтырсаңыз, маалымат чогула баштайт.",range_24h:"24 саат",range_3d:"3 күн",range_10d:"10 күн",sensor_trend_title:"Сенсор динамикасы",risk_trend_title:"Тобокелдик динамикасы",trend_flat:"өзгөрүүсүз",trend_rising:"өсүүдө",trend_falling:"төмөндөөдө",create_work_order_btn:"Иш буйругун түзүү",work_order_created_msg:"Иш буйругу түзүлдү",alertreason_immediate_failure_risk:"Дароо бузулуу коркунучу",alertreason_failure_imminent:"Бузулуу жакын",alertreason_degradation_accelerating:"Тозуу тездеп жатат",alertreason_outside_normal_envelope:"Кадимки чектен тышкары",alertreason_pressure_anomaly:"Басым аномалиясы",alertreason_idle_waste:"Бош жүрүш чыгымы",alertreason_informational:"Маалыматтык",alert_reason_label:"Себеби"},
  uk: {tagline:"Глобальна платформа промислового інтелекту",live_label:"Наживо",kpi_energy:"Споживання енергії",kpi_efficiency:"Ефективність",kpi_active:"Активні верстати",kpi_alerts:"Сповіщення",kwh_unit:"кВт·год",chart_title:"Показники в реальному часі",machine_status_title:"Статус верстатів",status_running:"Працює",status_warning:"Попередження",status_critical:"Критично",form_title:"Введення даних заводу",factory_name_label:"Назва заводу",machine_count_label:"Кількість верстатів",energy_cost_label:"Вартість енергії ($/кВт·год)",machine_type_label:"Тип верстата",temperature_label:"Температура (°C)",vibration_label:"Вібрація (мм/с)",load_label:"Навантаження (%)",submit_btn:"Аналізувати завод",submitting:"Оновлення...",ai_panel_title:"AI-аналітика",ai_placeholder:"Надішліть дані заводу, щоб отримати AI-аналіз.",ai_analyzing:"Аналіз...",ai_risks:"Ризики",ai_efficiency_insights:"Аналіз ефективності",ai_optimizations:"Рекомендації з оптимізації",toast_updated:"Дані заводу оновлено",toast_analysis_done:"AI-аналіз завершено",toast_error:"Сталася помилка",nav_dashboard:"Панель",nav_factories:"Заводи",nav_ai_insights:"AI-аналітика",logout_btn:"Вийти",login_title:"З поверненням",login_subtitle:"Увійдіть у свій обліковий запис FactoryPulse AI",ph_email:"Електронна пошта",ph_password:"Пароль",remember_me:"Запам'ятати мене",login_btn:"Увійти",login_link_register:"Немає акаунту? Створити",register_title:"Створіть акаунт",register_subtitle:"Почніть моніторинг заводів за допомогою AI",ph_full_name:"Повне ім'я",ph_confirm_password:"Підтвердіть пароль",register_btn:"Створити акаунт",register_link_login:"Вже є акаунт? Увійти",err_missing_fields:"Будь ласка, заповніть усі поля",err_invalid_email:"Введіть дійсну електронну адресу",err_weak_password:"Пароль має містити щонайменше 8 символів, літеру та цифру",err_password_mismatch:"Паролі не збігаються",err_invalid_credentials:"Невірна електронна пошта або пароль",err_email_taken:"Ця електронна пошта вже зареєстрована",err_generic:"Сталася помилка. Спробуйте ще раз",my_factories_title:"Мої Заводи",add_factory_btn:"+ Додати завод",edit_factory_btn:"Редагувати",delete_factory_btn:"Видалити",confirm_delete_factory:"Видалити цей завод? Цю дію не можна скасувати.",no_factories_yet:"Ви ще не додали жодного заводу.",factory_created_toast:"Завод створено та проаналізовано",factory_updated_toast:"Завод оновлено",factory_deleted_toast:"Завод видалено",ai_insights_feed_title:"Стрічка AI-аналітики",no_ai_insights_yet:"Ще немає AI-аналітики. Додайте завод.",reanalyze_btn:"Проаналізувати знову",view_insights_btn:"Переглянути аналітику",created_label:"Створено",cancel_btn:"Скасувати",save_btn:"Зберегти зміни",nav_live_monitor:"Моніторинг",add_machine_scada_btn:"+ Додати верстат",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Опитування",live_chart_title:"Графік датчиків у реальному часі",machines_table_title:"Верстати",machine_code_col:"Код",machine_name_col:"Назва",status_col:"Статус",risk_col:"Ризик",no_machines_yet:"Верстатів поки немає. Натисніть «+ Додати верстат».",section_machine_info:"Інформація про верстат",section_sensor_data:"Дані датчиків",section_status:"Статус",section_notes:"Примітки",status_stopped:"Зупинено",status_maintenance:"Обслуговування",priority_low:"Низький",priority_normal:"Звичайний",priority_high:"Високий",priority_critical:"Критичний",save_and_analyze_btn:"Зберегти та проаналізувати",source_col:"Джерело",source_auto:"Авто (SCADA)",source_manual:"Вручну",nav_alerts:"Сповіщення",acknowledge_btn:"Підтвердити",acknowledged_label:"Підтверджено",acknowledge_all_btn:"Підтвердити все",no_alerts_yet:"Сповіщень немає. Все працює нормально.",download_report_btn:"Звіт",alert_details_template:"Температура {temp}°C, вібрація {vib} мм/с, статус: {status}",section_energy_intel:"Енергетичний інтелект",daily_output_hint:"Використовується для розрахунку питомого енергоспоживання (кВт·год на одиницю).",energy_insights_title:"Енергетичний інтелект",idle_power_title:"Виявлення холостого ходу",idle_active_msg:"Верстат простоює — зараз витрачається приблизно {kw} кВт даремно.",idle_none_msg:"Втрат енергії на холостому ходу не виявлено.",friction_loss_title:"Прогноз втрат енергії",friction_active_msg:"Виявлено підвищене тертя: +{pct}% зайвої потужності (~{kw} кВт). Заплануйте обслуговування, щоб уникнути втрат.",friction_none_msg:"Аномального тертя не виявлено.",sec_title:"Питоме енергоспоживання",sec_label:"кВт·год на одиницю",sec_unit:"кВт·год/од.",sec_no_data_msg:"Вкажіть добовий обсяг випуску при додаванні верстата, щоб побачити цей показник.",optimal_load_title:"Оптимальна зона навантаження",optimal_load_label:"Оптимальне навантаження",current_load_label:"Поточне навантаження",at_optimal_msg:"Працює в оптимальній зоні навантаження.",adjust_to_optimal_msg:"Наблизьте навантаження до {pct}%, щоб мінімізувати енергію на одиницю.",nav_digital_twin:"Цифровий двійник",twin_hint:"Перетягуйте для повороту, прокручуйте для масштабування, натисніть на верстат для деталей у реальному часі.",twin_unavailable_msg:"Не вдалося завантажити 3D-вигляд (перевірте підключення до інтернету для Three.js).",failure_prediction_title:"Прогноз відмови",report_lib_missing_msg:"Для експорту в PDF потрібна бібліотека reportlab. Виконайте: pip install reportlab, потім перезапустіть сервер.",ph_machine_id:"ID верстата (напр. M-01)",ph_machine_name:"Назва верстата",ph_factory_section:"Дільниця заводу",ph_operator_name:"Ім'я оператора",ph_pressure:"Тиск (бар)",ph_voltage:"Напруга (В)",ph_current:"Струм (А)",ph_error_code:"Код помилки",ph_daily_output:"Добовий випуск (од.)",ph_notes:"Примітки...",nav_system_intel:"Системна аналітика",nav_roi:"ROI-панель",refresh_btn:"Оновити",system_risk_label:"Системний ризик",healthy_label:"Справні",at_risk_label:"У зоні ризику",clusters_title:"Кластери верстатів",propagation_title:"Поширення аномалій",propagation_hint:"Як відмова одного верстата підвищує ризик сусідніх.",no_propagation:"Поширення аномалій не виявлено.",avg_risk_label:"Середній ризик",added_risk_label:"Доданий ризик",effective_risk_label:"Ефективний ризик",simulation_title:"Що-якщо симуляція",simulation_hint:"Оберіть верстат, змініть покази датчиків і подивіться на зміну ризику.",run_simulation_btn:"Запустити симуляцію",failure_probability_label:"Ймовірність відмови",stress_level_label:"Рівень навантаження",predicted_status_label:"Прогноз статусу",confidence_label:"Достовірність",root_cause_title:"Першопричина",rul_col:"Залишковий ресурс",rul_healthy:"Справний",potential_loss_label:"Потенційні втрати",saved_label:"Заощаджено AI",wasted_energy_label:"Втрати енергії / місяць",efficiency_gain_label:"Приріст ефективності",cost_by_machine_title:"Фінансовий ризик за верстатами",top_cause_col:"Основна причина",roi_assumptions_msg:"Припущення: простій {downtime}/год, ремонт {hours} год, енергія {price}/кВт·год.",role_label:"Ваша роль",role_engineer:"Інженер",role_manager:"Менеджер",role_admin:"Адміністратор",cause_bearing_wear:"Знос підшипника",cause_overload_thermal:"Теплове перевантаження",cause_cooling_failure:"Відмова охолодження",cause_misalignment:"Розцентрування вала",cause_lubrication_loss:"Втрата мастила",cause_normal_operation:"Нормальна робота",nav_story:"Режим історії",story_hint:"Відтворює розвиток відмови підшипника за 22 години та показує, коли саме AI її виявив — і скільки це коштувало.",simulate_failure_btn:"Змоделювати відмову",outcome_title:"Підсумок",warning_time_label:"Раннє попередження",loss_ignored_label:"Втрати без реакції",loss_acted_label:"Втрати з реакцією",money_saved_label:"Заощаджено",timeline_title:"Хронологія відмови",story_detection_msg:"AI виявив це за {hours} год до поломки, при ризику {risk}% — першопричина: {cause} (достовірність {confidence}%).",story_stage_healthy:"Справний",story_stage_early_drift:"Початок відхилення",story_stage_ai_detects:"AI виявив",story_stage_critical:"Критично",priority_low:"Низький",priority_medium:"Середній",priority_high:"Високий",priority_critical:"Критичний",action_stop_machine:"Зупинити верстат",action_inspect_bearings:"Перевірити підшипники",action_check_cooling:"Перевірити охолодження",action_schedule_shutdown_24h:"Запланувати зупинку (24год)",action_order_spare_parts:"Замовити запчастини",action_reduce_load:"Знизити навантаження",action_schedule_inspection_72h:"Запланувати огляд (72год)",action_monitor_closely:"Уважно спостерігати",action_verify_sensor:"Перевірити датчик",action_power_down_idle:"Вимкнути простійний верстат",action_review_shift_schedule:"Переглянути графік змін",action_no_action:"Дій не потрібно",nav_oee:"OEE",nav_workorders:"Наряди",oee_hint:"Доступність x Продуктивність x Якість (ISO 22400). Світовий рівень — 85%, типовий завод — близько 60%.",availability_label:"Доступність",performance_label:"Продуктивність",quality_label:"Якість",weakest_factor_label:"Найслабша ланка",downtime_by_reason_title:"Простої за причинами",downtime_cost_label:"Вартість простою",oee_trend_title:"Динаміка OEE",shifts_title:"Зміни",shift_col:"Зміна",downtime_col:"Простій",log_shift_btn:"Внести зміну",no_shifts_yet:"Зміни ще не внесено.",minutes_short:"хв",range_1d:"Сьогодні",range_7d:"7 днів",range_30d:"30 днів",all_machines_option:"Усі верстати",reason_unspecified:"Не вказано",err_good_exceeds_total:"Придатних не може бути більше загальної кількості",oee_grade_world_class:"Світовий рівень",oee_grade_typical:"Типовий",oee_grade_low:"Низький",oee_grade_critical:"Критичний",reason_breakdown:"Поломка",reason_changeover:"Переналагодження",reason_no_material:"Немає матеріалу",reason_no_operator:"Немає оператора",reason_planned_maintenance:"Планове ТО",reason_quality_issue:"Проблема якості",reason_setup:"Налаштування",reason_other:"Інше",workorders_hint:"Перетворює прогноз AI на відстежуване завдання — щоб попередження закінчилося ремонтом.",new_work_order_btn:"+ Новий наряд",no_work_orders:"Нарядів поки немає.",avg_completion_label:"Сер. виконання",assigned_label:"Призначено",source_ai:"AI",wo_status_open:"Відкритий",wo_status_in_progress:"В роботі",wo_status_done:"Виконаний",wo_status_cancelled:"Скасований",wo_advance_to_in_progress:"Почати роботу",wo_advance_to_done:"Позначити виконаним",ph_shift_name:"Назва зміни",ph_planned_minutes:"Планові хвилини",ph_downtime_minutes:"Хвилини простою",ph_total_units:"Усього одиниць",ph_good_units:"Придатних одиниць",ph_cycle_seconds:"Ідеальний цикл (сек)",ph_wo_title:"Назва",ph_assigned_to:"Відповідальний",ph_wo_description:"Опис",overdue_label:"Прострочено",nav_history:"Історія",history_hint:"Збережена історія датчиків — так ви доведете, що реально змінилося за пілот.",no_history_yet:"Історію ще не записано. Потримайте Моніторинг відкритим кілька хвилин, і дані почнуть накопичуватися.",range_24h:"24 години",range_3d:"3 дні",range_10d:"10 днів",sensor_trend_title:"Динаміка датчиків",risk_trend_title:"Динаміка ризику",trend_flat:"без змін",trend_rising:"зростає",trend_falling:"знижується",create_work_order_btn:"Створити наряд",work_order_created_msg:"Наряд створено",alertreason_immediate_failure_risk:"Ризик негайної відмови",alertreason_failure_imminent:"Відмова неминуча",alertreason_degradation_accelerating:"Знос прискорюється",alertreason_outside_normal_envelope:"Вихід за штатні межі",alertreason_pressure_anomaly:"Аномалія тиску",alertreason_idle_waste:"Втрати на холостому ходу",alertreason_informational:"Інформаційне",alert_reason_label:"Причина"},
  pl: {tagline:"Globalna Platforma Inteligencji Przemysłowej",live_label:"Na żywo",kpi_energy:"Zużycie Energii",kpi_efficiency:"Wydajność",kpi_active:"Aktywne Maszyny",kpi_alerts:"Alerty",kwh_unit:"kWh",chart_title:"Wydajność w Czasie Rzeczywistym",machine_status_title:"Status Maszyn",status_running:"Działa",status_warning:"Ostrzeżenie",status_critical:"Krytyczne",form_title:"Wprowadzanie Danych Fabryki",factory_name_label:"Nazwa Fabryki",machine_count_label:"Liczba Maszyn",energy_cost_label:"Koszt Energii ($/kWh)",machine_type_label:"Typ Maszyny",temperature_label:"Temperatura (°C)",vibration_label:"Wibracje (mm/s)",load_label:"Obciążenie (%)",submit_btn:"Analizuj Fabrykę",submitting:"Aktualizowanie...",ai_panel_title:"Analizy AI",ai_placeholder:"Prześlij dane fabryki, aby wygenerować analizę AI.",ai_analyzing:"Analizowanie...",ai_risks:"Ryzyka",ai_efficiency_insights:"Analiza Wydajności",ai_optimizations:"Sugestie Optymalizacji",toast_updated:"Dane fabryki zaktualizowane",toast_analysis_done:"Analiza AI zakończona",toast_error:"Coś poszło nie tak",nav_dashboard:"Panel",nav_factories:"Fabryki",nav_ai_insights:"Analizy AI",logout_btn:"Wyloguj",login_title:"Witamy z powrotem",login_subtitle:"Zaloguj się do swojego konta FactoryPulse AI",ph_email:"E-mail",ph_password:"Hasło",remember_me:"Zapamiętaj mnie",login_btn:"Zaloguj się",login_link_register:"Nie masz konta? Utwórz je",register_title:"Utwórz konto",register_subtitle:"Zacznij monitorować swoje fabryki z AI",ph_full_name:"Imię i Nazwisko",ph_confirm_password:"Potwierdź Hasło",register_btn:"Utwórz Konto",register_link_login:"Masz już konto? Zaloguj się",err_missing_fields:"Proszę wypełnić wszystkie pola",err_invalid_email:"Proszę podać prawidłowy adres e-mail",err_weak_password:"Hasło musi mieć min. 8 znaków, literę i cyfrę",err_password_mismatch:"Hasła nie pasują do siebie",err_invalid_credentials:"Nieprawidłowy e-mail lub hasło",err_email_taken:"Ten e-mail jest już zarejestrowany",err_generic:"Coś poszło nie tak. Spróbuj ponownie",my_factories_title:"Moje Fabryki",add_factory_btn:"+ Dodaj Fabrykę",edit_factory_btn:"Edytuj",delete_factory_btn:"Usuń",confirm_delete_factory:"Usunąć tę fabrykę? Tej czynności nie można cofnąć.",no_factories_yet:"Nie dodałeś jeszcze żadnej fabryki.",factory_created_toast:"Fabryka utworzona i przeanalizowana",factory_updated_toast:"Fabryka zaktualizowana",factory_deleted_toast:"Fabryka usunięta",ai_insights_feed_title:"Kanał Analiz AI",no_ai_insights_yet:"Brak analiz AI. Dodaj fabrykę, aby zacząć.",reanalyze_btn:"Analizuj Ponownie",view_insights_btn:"Zobacz Analizy",created_label:"Utworzono",cancel_btn:"Anuluj",save_btn:"Zapisz Zmiany",nav_live_monitor:"Monitoring na Żywo",add_machine_scada_btn:"+ Dodaj Maszynę",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Odpytywanie",live_chart_title:"Wykres Czujników na Żywo",machines_table_title:"Maszyny",machine_code_col:"Kod",machine_name_col:"Nazwa",status_col:"Status",risk_col:"Ryzyko",no_machines_yet:"Brak maszyn. Kliknij „+ Dodaj Maszynę”.",section_machine_info:"Informacje o Maszynie",section_sensor_data:"Dane Czujników",section_status:"Status",section_notes:"Notatki",status_stopped:"Zatrzymana",status_maintenance:"Konserwacja",priority_low:"Niski",priority_normal:"Normalny",priority_high:"Wysoki",priority_critical:"Krytyczny",save_and_analyze_btn:"Zapisz i Analizuj",source_col:"Źródło",source_auto:"Auto (SCADA)",source_manual:"Ręcznie",nav_alerts:"Alerty",acknowledge_btn:"Potwierdź",acknowledged_label:"Potwierdzone",acknowledge_all_btn:"Potwierdź Wszystkie",no_alerts_yet:"Brak alertów. Wszystko działa prawidłowo.",download_report_btn:"Raport",alert_details_template:"Temperatura {temp}°C, wibracje {vib} mm/s, status: {status}",section_energy_intel:"Inteligencja Energetyczna",daily_output_hint:"Używane do obliczania jednostkowego zużycia energii (kWh na jednostkę).",energy_insights_title:"Inteligencja Energetyczna",idle_power_title:"Wykrywanie Mocy Jałowej",idle_active_msg:"Maszyna bezczynna - obecnie marnowane jest ok. {kw} kW.",idle_none_msg:"Nie wykryto marnowania energii w trybie bezczynności.",friction_loss_title:"Predykcyjna Utrata Energii",friction_active_msg:"Wykryto zwiększone tarcie: +{pct}% dodatkowej mocy (~{kw} kW więcej). Zaplanuj konserwację, aby zapobiec stratom.",friction_none_msg:"Nie wykryto nieprawidłowego tarcia.",sec_title:"Jednostkowe Zużycie Energii",sec_label:"kWh na jednostkę",sec_unit:"kWh/jednostkę",sec_no_data_msg:"Podaj dzienną produkcję podczas dodawania maszyny, aby zobaczyć ten wskaźnik.",optimal_load_title:"Optymalna Strefa Obciążenia",optimal_load_label:"Optymalne obciążenie",current_load_label:"Obecne obciążenie",at_optimal_msg:"Działa w optymalnej strefie obciążenia.",adjust_to_optimal_msg:"Dostosuj obciążenie do {pct}%, aby zminimalizować energię na jednostkę.",nav_digital_twin:"Cyfrowy Bliźniak",twin_hint:"Przeciągnij, aby obrócić, przewiń, aby powiększyć, kliknij maszynę, aby zobaczyć szczegóły na żywo.",twin_unavailable_msg:"Nie udało się załadować widoku 3D (sprawdź połączenie internetowe dla biblioteki Three.js).",failure_prediction_title:"Prognoza Awarii",report_lib_missing_msg:"Eksport do PDF wymaga biblioteki reportlab. Uruchom: pip install reportlab, a następnie zrestartuj serwer.",ph_machine_id:"ID Maszyny (np. M-01)",ph_machine_name:"Nazwa Maszyny",ph_factory_section:"Sekcja Fabryki",ph_operator_name:"Imię Operatora",ph_pressure:"Ciśnienie (bar)",ph_voltage:"Napięcie (V)",ph_current:"Prąd (A)",ph_error_code:"Kod Błędu",ph_daily_output:"Dzienna Produkcja (jednostki)",ph_notes:"Notatki...",nav_system_intel:"Inteligencja Systemu",nav_roi:"Pulpit ROI",refresh_btn:"Odśwież",system_risk_label:"Ryzyko Systemu",healthy_label:"Sprawne",at_risk_label:"Zagrożone",clusters_title:"Klastry Maszyn",propagation_title:"Propagacja Anomalii",propagation_hint:"Jak awaria jednej maszyny podnosi ryzyko sąsiednich.",no_propagation:"Nie wykryto propagacji anomalii.",avg_risk_label:"Śr. ryzyko",added_risk_label:"Dodane ryzyko",effective_risk_label:"Ryzyko efektywne",simulation_title:"Symulacja Co-Jeśli",simulation_hint:"Wybierz maszynę, zmień wartości czujników i obserwuj prawdopodobieństwo awarii.",run_simulation_btn:"Uruchom Symulację",failure_probability_label:"Prawdopodobieństwo Awarii",stress_level_label:"Poziom Obciążenia",predicted_status_label:"Przewidywany Status",confidence_label:"Pewność",root_cause_title:"Przyczyna Źródłowa",rul_col:"Pozostała Żywotność",rul_healthy:"Sprawna",potential_loss_label:"Potencjalna Strata",saved_label:"Zaoszczędzone przez AI",wasted_energy_label:"Zmarnowana Energia / mies.",efficiency_gain_label:"Wzrost Wydajności",cost_by_machine_title:"Ryzyko Kosztowe wg Maszyn",top_cause_col:"Główna Przyczyna",roi_assumptions_msg:"Założenia: przestój {downtime}/h, naprawa {hours}h, energia {price}/kWh.",role_label:"Twoja Rola",role_engineer:"Inżynier",role_manager:"Menedżer",role_admin:"Administrator",cause_bearing_wear:"Zużycie łożyska",cause_overload_thermal:"Przeciążenie termiczne",cause_cooling_failure:"Awaria chłodzenia",cause_misalignment:"Niewspółosiowość wału",cause_lubrication_loss:"Utrata smarowania",cause_normal_operation:"Praca normalna",nav_story:"Tryb Historii",story_hint:"Odtwarza awarię łożyska rozwijającą się przez 22 godziny i pokazuje dokładnie, kiedy AI ją wykryła — i ile to było warte.",simulate_failure_btn:"Symuluj Awarię",outcome_title:"Wynik",warning_time_label:"Wczesne Ostrzeżenie",loss_ignored_label:"Strata bez Reakcji",loss_acted_label:"Strata z Reakcją",money_saved_label:"Zaoszczędzono",timeline_title:"Oś Czasu Awarii",story_detection_msg:"AI wykryła to {hours} godzin przed awarią, przy {risk}% ryzyka — przyczyna źródłowa: {cause} (pewność {confidence}%).",story_stage_healthy:"Sprawna",story_stage_early_drift:"Pierwsze Odchylenie",story_stage_ai_detects:"AI Wykrywa",story_stage_critical:"Krytyczny",priority_low:"Niski",priority_medium:"Średni",priority_high:"Wysoki",priority_critical:"Krytyczny",action_stop_machine:"Zatrzymaj maszynę",action_inspect_bearings:"Sprawdź łożyska",action_check_cooling:"Sprawdź chłodzenie",action_schedule_shutdown_24h:"Zaplanuj postój (24h)",action_order_spare_parts:"Zamów części",action_reduce_load:"Zmniejsz obciążenie",action_schedule_inspection_72h:"Zaplanuj przegląd (72h)",action_monitor_closely:"Uważnie monitoruj",action_verify_sensor:"Zweryfikuj czujnik",action_power_down_idle:"Wyłącz bezczynną maszynę",action_review_shift_schedule:"Przejrzyj grafik zmian",action_no_action:"Nie wymaga działania",nav_oee:"OEE",nav_workorders:"Zlecenia Pracy",oee_hint:"Dostępność x Wydajność x Jakość (ISO 22400). Poziom światowy to 85%, typowa fabryka około 60%.",availability_label:"Dostępność",performance_label:"Wydajność",quality_label:"Jakość",weakest_factor_label:"Najsłabszy",downtime_by_reason_title:"Przestoje wg Przyczyny",downtime_cost_label:"Koszt przestoju",oee_trend_title:"Trend OEE",shifts_title:"Zmiany",shift_col:"Zmiana",downtime_col:"Przestój",log_shift_btn:"Zapisz Zmianę",no_shifts_yet:"Nie zapisano jeszcze zmian.",minutes_short:"min",range_1d:"Dzisiaj",range_7d:"7 dni",range_30d:"30 dni",all_machines_option:"Wszystkie maszyny",reason_unspecified:"Nieokreślone",err_good_exceeds_total:"Sztuki dobre nie mogą przekroczyć całości",oee_grade_world_class:"Poziom światowy",oee_grade_typical:"Typowy",oee_grade_low:"Niski",oee_grade_critical:"Krytyczny",reason_breakdown:"Awaria",reason_changeover:"Przezbrojenie",reason_no_material:"Brak materiału",reason_no_operator:"Brak operatora",reason_planned_maintenance:"Konserwacja planowa",reason_quality_issue:"Problem jakości",reason_setup:"Ustawianie",reason_other:"Inne",workorders_hint:"Zamienia prognozę AI w śledzone zadanie — aby ostrzeżenie kończyło się naprawą.",new_work_order_btn:"+ Nowe Zlecenie",no_work_orders:"Brak zleceń pracy.",avg_completion_label:"Śr. realizacja",assigned_label:"Przypisane",source_ai:"AI",wo_status_open:"Otwarte",wo_status_in_progress:"W trakcie",wo_status_done:"Zakończone",wo_status_cancelled:"Anulowane",wo_advance_to_in_progress:"Rozpocznij pracę",wo_advance_to_done:"Oznacz jako zakończone",ph_shift_name:"Nazwa zmiany",ph_planned_minutes:"Zaplanowane minuty",ph_downtime_minutes:"Minuty przestoju",ph_total_units:"Łączna liczba",ph_good_units:"Sztuki dobre",ph_cycle_seconds:"Cykl idealny (sek)",ph_wo_title:"Tytuł",ph_assigned_to:"Przypisane do",ph_wo_description:"Opis",overdue_label:"Zaległe",nav_history:"Historia",history_hint:"Zapisana historia czujników — tak udowodnisz, co naprawdę zmieniło się podczas pilotażu.",no_history_yet:"Nie zapisano jeszcze historii. Zostaw Monitoring otwarty na kilka minut, a dane zaczną się gromadzić.",range_24h:"24 godziny",range_3d:"3 dni",range_10d:"10 dni",sensor_trend_title:"Trend Czujników",risk_trend_title:"Trend Ryzyka",trend_flat:"bez zmian",trend_rising:"rośnie",trend_falling:"spada",create_work_order_btn:"Utwórz Zlecenie",work_order_created_msg:"Zlecenie utworzone",alertreason_immediate_failure_risk:"Ryzyko natychmiastowej awarii",alertreason_failure_imminent:"Awaria nieuchronna",alertreason_degradation_accelerating:"Przyspieszające zużycie",alertreason_outside_normal_envelope:"Poza normalnym zakresem",alertreason_pressure_anomaly:"Anomalia ciśnienia",alertreason_idle_waste:"Marnowanie energii na biegu jałowym",alertreason_informational:"Informacyjne",alert_reason_label:"Powód"},
  nl: {tagline:"Wereldwijd Industrieel Intelligentieplatform",live_label:"Live",kpi_energy:"Energieverbruik",kpi_efficiency:"Efficiëntie",kpi_active:"Actieve Machines",kpi_alerts:"Meldingen",kwh_unit:"kWh",chart_title:"Realtime Prestaties",machine_status_title:"Machinestatus",status_running:"Actief",status_warning:"Waarschuwing",status_critical:"Kritiek",form_title:"Fabrieksgegevens Invoeren",factory_name_label:"Fabrieksnaam",machine_count_label:"Aantal Machines",energy_cost_label:"Energiekosten ($/kWh)",machine_type_label:"Machinetype",temperature_label:"Temperatuur (°C)",vibration_label:"Trilling (mm/s)",load_label:"Belasting (%)",submit_btn:"Fabriek Analyseren",submitting:"Bijwerken...",ai_panel_title:"AI-inzichten",ai_placeholder:"Verzend fabrieksgegevens om een AI-analyse te genereren.",ai_analyzing:"Analyseren...",ai_risks:"Risico's",ai_efficiency_insights:"Efficiëntieanalyse",ai_optimizations:"Optimalisatiesuggesties",toast_updated:"Fabrieksgegevens bijgewerkt",toast_analysis_done:"AI-analyse voltooid",toast_error:"Er is iets misgegaan",nav_dashboard:"Dashboard",nav_factories:"Fabrieken",nav_ai_insights:"AI-inzichten",logout_btn:"Uitloggen",login_title:"Welkom terug",login_subtitle:"Log in op uw FactoryPulse AI-account",ph_email:"E-mail",ph_password:"Wachtwoord",remember_me:"Onthoud mij",login_btn:"Inloggen",login_link_register:"Geen account? Maak er een",register_title:"Maak uw account aan",register_subtitle:"Begin met AI-monitoring van uw fabrieken",ph_full_name:"Volledige Naam",ph_confirm_password:"Bevestig Wachtwoord",register_btn:"Account Aanmaken",register_link_login:"Heeft u al een account? Inloggen",err_missing_fields:"Vul alle velden in",err_invalid_email:"Voer een geldig e-mailadres in",err_weak_password:"Wachtwoord moet minimaal 8 tekens, een letter en een cijfer bevatten",err_password_mismatch:"Wachtwoorden komen niet overeen",err_invalid_credentials:"Ongeldige e-mail of wachtwoord",err_email_taken:"Dit e-mailadres is al geregistreerd",err_generic:"Er is iets misgegaan. Probeer het opnieuw",my_factories_title:"Mijn Fabrieken",add_factory_btn:"+ Fabriek Toevoegen",edit_factory_btn:"Bewerken",delete_factory_btn:"Verwijderen",confirm_delete_factory:"Deze fabriek verwijderen? Dit kan niet ongedaan worden gemaakt.",no_factories_yet:"U heeft nog geen fabrieken toegevoegd.",factory_created_toast:"Fabriek aangemaakt en geanalyseerd",factory_updated_toast:"Fabriek bijgewerkt",factory_deleted_toast:"Fabriek verwijderd",ai_insights_feed_title:"AI-inzichten Feed",no_ai_insights_yet:"Nog geen AI-inzichten. Voeg een fabriek toe.",reanalyze_btn:"Opnieuw Analyseren",view_insights_btn:"Bekijk Inzichten",created_label:"Aangemaakt",cancel_btn:"Annuleren",save_btn:"Wijzigingen Opslaan",nav_live_monitor:"Live Monitor",add_machine_scada_btn:"+ Machine Toevoegen",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Polling",live_chart_title:"Live Sensorgrafiek",machines_table_title:"Machines",machine_code_col:"Code",machine_name_col:"Naam",status_col:"Status",risk_col:"Risico",no_machines_yet:"Nog geen machines. Klik op „+ Machine Toevoegen”.",section_machine_info:"Machine-informatie",section_sensor_data:"Sensorgegevens",section_status:"Status",section_notes:"Notities",status_stopped:"Gestopt",status_maintenance:"Onderhoud",priority_low:"Laag",priority_normal:"Normaal",priority_high:"Hoog",priority_critical:"Kritiek",save_and_analyze_btn:"Opslaan & Analyseren",source_col:"Bron",source_auto:"Auto (SCADA)",source_manual:"Handmatig",nav_alerts:"Meldingen",acknowledge_btn:"Bevestigen",acknowledged_label:"Bevestigd",acknowledge_all_btn:"Alles Bevestigen",no_alerts_yet:"Geen meldingen. Alles werkt naar behoren.",download_report_btn:"Rapport",alert_details_template:"Temperatuur {temp}°C, trilling {vib} mm/s, status: {status}",section_energy_intel:"Energie-intelligentie",daily_output_hint:"Wordt gebruikt om het specifieke energieverbruik te berekenen (kWh per eenheid).",energy_insights_title:"Energie-intelligentie",idle_power_title:"Stationair Vermogen Detectie",idle_active_msg:"Machine is inactief - momenteel wordt ongeveer {kw} kW verspild.",idle_none_msg:"Geen stationaire energieverspilling gedetecteerd.",friction_loss_title:"Voorspellend Energieverlies",friction_active_msg:"Verhoogde wrijving gedetecteerd: +{pct}% extra vermogen (~{kw} kW extra). Plan onderhoud om verliezen te voorkomen.",friction_none_msg:"Geen abnormale wrijving gedetecteerd.",sec_title:"Specifiek Energieverbruik",sec_label:"kWh per eenheid",sec_unit:"kWh/eenheid",sec_no_data_msg:"Voer de dagelijkse output in bij het toevoegen van deze machine om deze metriek te zien.",optimal_load_title:"Optimale Belastingzone",optimal_load_label:"Optimale belasting",current_load_label:"Huidige belasting",at_optimal_msg:"Draait in de optimale belastingzone.",adjust_to_optimal_msg:"Pas de belasting aan naar {pct}% om energie per eenheid te minimaliseren.",nav_digital_twin:"Digitale Tweeling",twin_hint:"Sleep om te draaien, scroll om te zoomen, klik op een machine voor live details.",twin_unavailable_msg:"3D-weergave kon niet worden geladen (controleer uw internetverbinding voor Three.js).",failure_prediction_title:"Storingsvoorspelling",report_lib_missing_msg:"PDF-export vereist de reportlab-bibliotheek. Voer uit: pip install reportlab en herstart de server.",ph_machine_id:"Machine-ID (bijv. M-01)",ph_machine_name:"Machinenaam",ph_factory_section:"Fabriekssectie",ph_operator_name:"Naam Operator",ph_pressure:"Druk (bar)",ph_voltage:"Spanning (V)",ph_current:"Stroom (A)",ph_error_code:"Foutcode",ph_daily_output:"Dagelijkse Productie (eenheden)",ph_notes:"Notities...",nav_system_intel:"Systeemintelligentie",nav_roi:"ROI-dashboard",refresh_btn:"Vernieuwen",system_risk_label:"Systeemrisico",healthy_label:"Gezond",at_risk_label:"Risicovol",clusters_title:"Machineclusters",propagation_title:"Anomalieverspreiding",propagation_hint:"Hoe een falende machine het risico van buurmachines verhoogt.",no_propagation:"Geen anomalieverspreiding gedetecteerd.",avg_risk_label:"Gem. risico",added_risk_label:"Toegevoegd risico",effective_risk_label:"Effectief risico",simulation_title:"Wat-Als Simulatie",simulation_hint:"Kies een machine, verschuif de sensorwaarden en bekijk de storingskans.",run_simulation_btn:"Simulatie Uitvoeren",failure_probability_label:"Storingskans",stress_level_label:"Belastingsniveau",predicted_status_label:"Voorspelde Status",confidence_label:"Betrouwbaarheid",root_cause_title:"Hoofdoorzaak",rul_col:"Resterende Levensduur",rul_healthy:"Gezond",potential_loss_label:"Potentieel Verlies",saved_label:"Bespaard door AI",wasted_energy_label:"Verspilde Energie / maand",efficiency_gain_label:"Efficiëntiewinst",cost_by_machine_title:"Kostenrisico per Machine",top_cause_col:"Hoofdoorzaak",roi_assumptions_msg:"Aannames: stilstand {downtime}/u, reparatie {hours}u, energie {price}/kWh.",role_label:"Uw Rol",role_engineer:"Ingenieur",role_manager:"Manager",role_admin:"Beheerder",cause_bearing_wear:"Lagerslijtage",cause_overload_thermal:"Thermische overbelasting",cause_cooling_failure:"Koelstoring",cause_misalignment:"Asuitlijnfout",cause_lubrication_loss:"Smeringverlies",cause_normal_operation:"Normale werking",nav_story:"Verhaalmodus",story_hint:"Speelt een lagerstoring af die zich over 22 uur ontwikkelt en toont precies wanneer de AI het opmerkte — en wat dat waard was.",simulate_failure_btn:"Storing Simuleren",outcome_title:"Uitkomst",warning_time_label:"Vroege Waarschuwing",loss_ignored_label:"Verlies zonder Actie",loss_acted_label:"Verlies met Actie",money_saved_label:"Bespaard",timeline_title:"Storingstijdlijn",story_detection_msg:"De AI signaleerde dit {hours} uur voor de storing, bij {risk}% risico — hoofdoorzaak: {cause} ({confidence}% betrouwbaarheid).",story_stage_healthy:"Gezond",story_stage_early_drift:"Eerste Afwijking",story_stage_ai_detects:"AI Detecteert",story_stage_critical:"Kritiek",priority_low:"Laag",priority_medium:"Gemiddeld",priority_high:"Hoog",priority_critical:"Kritiek",action_stop_machine:"Machine stoppen",action_inspect_bearings:"Lagers inspecteren",action_check_cooling:"Koeling controleren",action_schedule_shutdown_24h:"Stilstand plannen (24u)",action_order_spare_parts:"Reserveonderdelen bestellen",action_reduce_load:"Belasting verlagen",action_schedule_inspection_72h:"Inspectie plannen (72u)",action_monitor_closely:"Nauwlettend volgen",action_verify_sensor:"Sensor verifiëren",action_power_down_idle:"Inactieve machine uitschakelen",action_review_shift_schedule:"Ploegrooster herzien",action_no_action:"Geen actie nodig",nav_oee:"OEE",nav_workorders:"Werkorders",oee_hint:"Beschikbaarheid x Prestatie x Kwaliteit (ISO 22400). Wereldklasse is 85%; een typische fabriek zit rond 60%.",availability_label:"Beschikbaarheid",performance_label:"Prestatie",quality_label:"Kwaliteit",weakest_factor_label:"Zwakste",downtime_by_reason_title:"Stilstand per Oorzaak",downtime_cost_label:"Stilstandkosten",oee_trend_title:"OEE-trend",shifts_title:"Ploegen",shift_col:"Ploeg",downtime_col:"Stilstand",log_shift_btn:"Ploeg Vastleggen",no_shifts_yet:"Nog geen ploegen vastgelegd.",minutes_short:"min",range_1d:"Vandaag",range_7d:"7 dagen",range_30d:"30 dagen",all_machines_option:"Alle machines",reason_unspecified:"Niet gespecificeerd",err_good_exceeds_total:"Goede stuks kunnen het totaal niet overschrijden",oee_grade_world_class:"Wereldklasse",oee_grade_typical:"Typisch",oee_grade_low:"Laag",oee_grade_critical:"Kritiek",reason_breakdown:"Storing",reason_changeover:"Omstelling",reason_no_material:"Geen materiaal",reason_no_operator:"Geen operator",reason_planned_maintenance:"Gepland onderhoud",reason_quality_issue:"Kwaliteitsprobleem",reason_setup:"Instellen",reason_other:"Overig",workorders_hint:"Zet een AI-voorspelling om in een gevolgde taak — zodat een waarschuwing eindigt in een reparatie.",new_work_order_btn:"+ Nieuwe Werkorder",no_work_orders:"Nog geen werkorders.",avg_completion_label:"Gem. afronding",assigned_label:"Toegewezen",source_ai:"AI",wo_status_open:"Open",wo_status_in_progress:"In uitvoering",wo_status_done:"Afgerond",wo_status_cancelled:"Geannuleerd",wo_advance_to_in_progress:"Werk starten",wo_advance_to_done:"Markeer als afgerond",ph_shift_name:"Ploegnaam",ph_planned_minutes:"Geplande minuten",ph_downtime_minutes:"Stilstandminuten",ph_total_units:"Totaal stuks",ph_good_units:"Goede stuks",ph_cycle_seconds:"Ideale cyclus (sec)",ph_wo_title:"Titel",ph_assigned_to:"Toegewezen aan",ph_wo_description:"Omschrijving",overdue_label:"Te laat",nav_history:"Geschiedenis",history_hint:"Opgeslagen sensorhistorie — zo bewijst u wat er tijdens de pilot echt is veranderd.",no_history_yet:"Nog geen historie opgeslagen. Houd de Live Monitor een paar minuten open, dan begint data zich op te bouwen.",range_24h:"24 uur",range_3d:"3 dagen",range_10d:"10 dagen",sensor_trend_title:"Sensortrend",risk_trend_title:"Risicotrend",trend_flat:"geen verandering",trend_rising:"stijgend",trend_falling:"dalend",create_work_order_btn:"Werkorder Maken",work_order_created_msg:"Werkorder aangemaakt",alertreason_immediate_failure_risk:"Onmiddellijk storingsrisico",alertreason_failure_imminent:"Storing ophanden",alertreason_degradation_accelerating:"Degradatie versnelt",alertreason_outside_normal_envelope:"Buiten normaal bereik",alertreason_pressure_anomaly:"Drukafwijking",alertreason_idle_waste:"Energieverspilling bij stilstand",alertreason_informational:"Informatief",alert_reason_label:"Reden"},
  sv: {tagline:"Global Industriell Intelligensplattform",live_label:"Live",kpi_energy:"Energiförbrukning",kpi_efficiency:"Effektivitet",kpi_active:"Aktiva Maskiner",kpi_alerts:"Varningar",kwh_unit:"kWh",chart_title:"Realtidsprestanda",machine_status_title:"Maskinstatus",status_running:"Igång",status_warning:"Varning",status_critical:"Kritisk",form_title:"Fabriksdatainmatning",factory_name_label:"Fabriksnamn",machine_count_label:"Antal Maskiner",energy_cost_label:"Energikostnad ($/kWh)",machine_type_label:"Maskintyp",temperature_label:"Temperatur (°C)",vibration_label:"Vibration (mm/s)",load_label:"Belastning (%)",submit_btn:"Analysera Fabrik",submitting:"Uppdaterar...",ai_panel_title:"AI-insikter",ai_placeholder:"Skicka fabriksdata för att generera en AI-analys.",ai_analyzing:"Analyserar...",ai_risks:"Risker",ai_efficiency_insights:"Effektivitetsanalys",ai_optimizations:"Optimeringsförslag",toast_updated:"Fabriksdata uppdaterad",toast_analysis_done:"AI-analys klar",toast_error:"Något gick fel",nav_dashboard:"Instrumentpanel",nav_factories:"Fabriker",nav_ai_insights:"AI-insikter",logout_btn:"Logga ut",login_title:"Välkommen tillbaka",login_subtitle:"Logga in på ditt FactoryPulse AI-konto",ph_email:"E-post",ph_password:"Lösenord",remember_me:"Kom ihåg mig",login_btn:"Logga in",login_link_register:"Inget konto? Skapa ett",register_title:"Skapa ditt konto",register_subtitle:"Börja övervaka dina fabriker med AI",ph_full_name:"Fullständigt Namn",ph_confirm_password:"Bekräfta Lösenord",register_btn:"Skapa Konto",register_link_login:"Har du redan ett konto? Logga in",err_missing_fields:"Vänligen fyll i alla fält",err_invalid_email:"Ange en giltig e-postadress",err_weak_password:"Lösenordet måste vara minst 8 tecken med en bokstav och en siffra",err_password_mismatch:"Lösenorden matchar inte",err_invalid_credentials:"Felaktig e-post eller lösenord",err_email_taken:"Denna e-post är redan registrerad",err_generic:"Något gick fel. Försök igen",my_factories_title:"Mina Fabriker",add_factory_btn:"+ Lägg till Fabrik",edit_factory_btn:"Redigera",delete_factory_btn:"Ta bort",confirm_delete_factory:"Ta bort denna fabrik? Detta kan inte ångras.",no_factories_yet:"Du har inte lagt till några fabriker än.",factory_created_toast:"Fabrik skapad och analyserad",factory_updated_toast:"Fabrik uppdaterad",factory_deleted_toast:"Fabrik borttagen",ai_insights_feed_title:"AI-insikter Flöde",no_ai_insights_yet:"Inga AI-insikter än. Lägg till en fabrik.",reanalyze_btn:"Analysera Igen",view_insights_btn:"Visa Insikter",created_label:"Skapad",cancel_btn:"Avbryt",save_btn:"Spara Ändringar",nav_live_monitor:"Livemonitor",add_machine_scada_btn:"+ Lägg till Maskin",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Polling",live_chart_title:"Live Sensordiagram",machines_table_title:"Maskiner",machine_code_col:"Kod",machine_name_col:"Namn",status_col:"Status",risk_col:"Risk",no_machines_yet:"Inga maskiner än. Klicka på \"+ Lägg till Maskin\".",section_machine_info:"Maskininformation",section_sensor_data:"Sensordata",section_status:"Status",section_notes:"Anteckningar",status_stopped:"Stoppad",status_maintenance:"Underhåll",priority_low:"Låg",priority_normal:"Normal",priority_high:"Hög",priority_critical:"Kritisk",save_and_analyze_btn:"Spara & Analysera",source_col:"Källa",source_auto:"Auto (SCADA)",source_manual:"Manuell",nav_alerts:"Varningar",acknowledge_btn:"Bekräfta",acknowledged_label:"Bekräftad",acknowledge_all_btn:"Bekräfta Alla",no_alerts_yet:"Inga varningar. Allt fungerar smidigt.",download_report_btn:"Rapport",alert_details_template:"Temperatur {temp}°C, vibration {vib} mm/s, status: {status}",section_energy_intel:"Energiintelligens",daily_output_hint:"Används för att beräkna specifik energiförbrukning (kWh per enhet).",energy_insights_title:"Energiintelligens",idle_power_title:"Detektering av Tomgångseffekt",idle_active_msg:"Maskinen står i tomgång - ungefär {kw} kW slösas just nu.",idle_none_msg:"Ingen tomgångsenergiförlust upptäckt.",friction_loss_title:"Prediktiv Energiförlust",friction_active_msg:"Ökad friktion upptäckt: +{pct}% extra effekt (~{kw} kW extra). Schemalägg underhåll för att förhindra förluster.",friction_none_msg:"Ingen onormal friktion upptäckt.",sec_title:"Specifik Energiförbrukning",sec_label:"kWh per enhet",sec_unit:"kWh/enhet",sec_no_data_msg:"Ange daglig produktion när du lägger till denna maskin för att se detta mått.",optimal_load_title:"Optimal Belastningszon",optimal_load_label:"Optimal belastning",current_load_label:"Aktuell belastning",at_optimal_msg:"Körs i den optimala belastningszonen.",adjust_to_optimal_msg:"Justera belastningen mot {pct}% för att minimera energi per enhet.",nav_digital_twin:"Digital Tvilling",twin_hint:"Dra för att rotera, scrolla för att zooma, klicka på en maskin för live-detaljer.",twin_unavailable_msg:"3D-vyn kunde inte laddas (kontrollera din internetanslutning för Three.js).",failure_prediction_title:"Felprognos",report_lib_missing_msg:"PDF-export kräver reportlab-biblioteket. Kör: pip install reportlab, starta sedan om servern.",ph_machine_id:"Maskin-ID (t.ex. M-01)",ph_machine_name:"Maskinnamn",ph_factory_section:"Fabrikssektion",ph_operator_name:"Operatörsnamn",ph_pressure:"Tryck (bar)",ph_voltage:"Spänning (V)",ph_current:"Ström (A)",ph_error_code:"Felkod",ph_daily_output:"Daglig Produktion (enheter)",ph_notes:"Anteckningar...",nav_system_intel:"Systemintelligens",nav_roi:"ROI-panel",refresh_btn:"Uppdatera",system_risk_label:"Systemrisk",healthy_label:"Friska",at_risk_label:"I riskzonen",clusters_title:"Maskinkluster",propagation_title:"Anomalispridning",propagation_hint:"Hur en havererande maskin höjer risken för sina grannar.",no_propagation:"Ingen anomalispridning upptäckt.",avg_risk_label:"Snittrisk",added_risk_label:"Tillagd risk",effective_risk_label:"Effektiv risk",simulation_title:"Tänk-Om-Simulering",simulation_hint:"Välj en maskin, ändra sensorvärden och se hur felsannolikheten reagerar.",run_simulation_btn:"Kör Simulering",failure_probability_label:"Felsannolikhet",stress_level_label:"Belastningsnivå",predicted_status_label:"Förutspådd Status",confidence_label:"Konfidens",root_cause_title:"Grundorsak",rul_col:"Återstående Livslängd",rul_healthy:"Frisk",potential_loss_label:"Potentiell Förlust",saved_label:"Sparat av AI",wasted_energy_label:"Bortslösad Energi / månad",efficiency_gain_label:"Effektivitetsvinst",cost_by_machine_title:"Kostnadsexponering per Maskin",top_cause_col:"Huvudorsak",roi_assumptions_msg:"Antaganden: stillestånd {downtime}/h, reparation {hours}h, energi {price}/kWh.",role_label:"Din Roll",role_engineer:"Ingenjör",role_manager:"Chef",role_admin:"Administratör",cause_bearing_wear:"Lagerslitage",cause_overload_thermal:"Termisk överbelastning",cause_cooling_failure:"Kylfel",cause_misalignment:"Axelfelinriktning",cause_lubrication_loss:"Smörjförlust",cause_normal_operation:"Normal drift",nav_story:"Berättelseläge",story_hint:"Spelar upp ett lagerhaveri som utvecklas över 22 timmar och visar exakt när AI:n upptäckte det — och vad det var värt.",simulate_failure_btn:"Simulera Haveri",outcome_title:"Utfall",warning_time_label:"Tidig Varning",loss_ignored_label:"Förlust utan Åtgärd",loss_acted_label:"Förlust med Åtgärd",money_saved_label:"Sparat",timeline_title:"Haveritidslinje",story_detection_msg:"AI:n flaggade detta {hours} timmar före haveriet, vid {risk}% risk — grundorsak: {cause} ({confidence}% konfidens).",story_stage_healthy:"Frisk",story_stage_early_drift:"Första Avvikelsen",story_stage_ai_detects:"AI Upptäcker",story_stage_critical:"Kritisk",priority_low:"Låg",priority_medium:"Medel",priority_high:"Hög",priority_critical:"Kritisk",action_stop_machine:"Stoppa maskinen",action_inspect_bearings:"Inspektera lager",action_check_cooling:"Kontrollera kylning",action_schedule_shutdown_24h:"Schemalägg stopp (24h)",action_order_spare_parts:"Beställ reservdelar",action_reduce_load:"Minska belastning",action_schedule_inspection_72h:"Schemalägg inspektion (72h)",action_monitor_closely:"Övervaka noga",action_verify_sensor:"Verifiera sensor",action_power_down_idle:"Stäng av tomgångsmaskin",action_review_shift_schedule:"Se över skiftschema",action_no_action:"Ingen åtgärd behövs",nav_oee:"OEE",nav_workorders:"Arbetsordrar",oee_hint:"Tillgänglighet x Prestanda x Kvalitet (ISO 22400). Världsklass är 85%; en typisk fabrik ligger nära 60%.",availability_label:"Tillgänglighet",performance_label:"Prestanda",quality_label:"Kvalitet",weakest_factor_label:"Svagast",downtime_by_reason_title:"Stillestånd per Orsak",downtime_cost_label:"Stilleståndskostnad",oee_trend_title:"OEE-trend",shifts_title:"Skift",shift_col:"Skift",downtime_col:"Stillestånd",log_shift_btn:"Registrera Skift",no_shifts_yet:"Inga skift registrerade ännu.",minutes_short:"min",range_1d:"Idag",range_7d:"7 dagar",range_30d:"30 dagar",all_machines_option:"Alla maskiner",reason_unspecified:"Ej angivet",err_good_exceeds_total:"Godkända enheter kan inte överstiga totalen",oee_grade_world_class:"Världsklass",oee_grade_typical:"Typisk",oee_grade_low:"Låg",oee_grade_critical:"Kritisk",reason_breakdown:"Haveri",reason_changeover:"Omställning",reason_no_material:"Inget material",reason_no_operator:"Ingen operatör",reason_planned_maintenance:"Planerat underhåll",reason_quality_issue:"Kvalitetsproblem",reason_setup:"Inställning",reason_other:"Övrigt",workorders_hint:"Förvandlar en AI-prognos till en spårad uppgift — så att en varning faktiskt slutar i en reparation.",new_work_order_btn:"+ Ny Arbetsorder",no_work_orders:"Inga arbetsordrar ännu.",avg_completion_label:"Snitt slutförande",assigned_label:"Tilldelad",source_ai:"AI",wo_status_open:"Öppen",wo_status_in_progress:"Pågår",wo_status_done:"Klar",wo_status_cancelled:"Avbruten",wo_advance_to_in_progress:"Starta arbete",wo_advance_to_done:"Markera klar",ph_shift_name:"Skiftnamn",ph_planned_minutes:"Planerade minuter",ph_downtime_minutes:"Stilleståndsminuter",ph_total_units:"Totalt antal",ph_good_units:"Godkända enheter",ph_cycle_seconds:"Idealcykel (sek)",ph_wo_title:"Titel",ph_assigned_to:"Tilldelad till",ph_wo_description:"Beskrivning",overdue_label:"Försenad",nav_history:"Historik",history_hint:"Sparad sensorhistorik — så bevisar du vad som faktiskt förändrades under pilotprojektet.",no_history_yet:"Ingen historik registrerad ännu. Håll Live-övervakningen öppen några minuter så börjar data samlas.",range_24h:"24 timmar",range_3d:"3 dagar",range_10d:"10 dagar",sensor_trend_title:"Sensortrend",risk_trend_title:"Risktrend",trend_flat:"oförändrad",trend_rising:"stigande",trend_falling:"fallande",create_work_order_btn:"Skapa Arbetsorder",work_order_created_msg:"Arbetsorder skapad",alertreason_immediate_failure_risk:"Omedelbar haverirrisk",alertreason_failure_imminent:"Haveri nära förestående",alertreason_degradation_accelerating:"Accelererande slitage",alertreason_outside_normal_envelope:"Utanför normalt intervall",alertreason_pressure_anomaly:"Tryckavvikelse",alertreason_idle_waste:"Energispill vid tomgång",alertreason_informational:"Informativt",alert_reason_label:"Orsak"},
};

let currentLang = localStorage.getItem("fp_lang") || "en";
if (!translations[currentLang]) currentLang = "en";
const RTL_LANGS = ["ar"];

function t(key) {
  return (translations[currentLang] && translations[currentLang][key]) || translations.en[key] || key;
}

function applyTranslations() {
  document.documentElement.lang = currentLang;
  document.documentElement.dir = RTL_LANGS.includes(currentLang) ? "rtl" : "ltr";
  document.querySelectorAll("[data-t]").forEach(el => { el.textContent = t(el.getAttribute("data-t")); });
  document.querySelectorAll("[data-t-placeholder]").forEach(el => { el.placeholder = t(el.getAttribute("data-t-placeholder")); });
  buildMachineTypeSelect();
  updateChartLabels();
  if (lastMachines.length) renderMachineList(lastMachines);
  if (lastKpis) renderKpis(lastKpis);
  if (scadaMachines && scadaMachines.length) renderScadaTable();
}

function buildLangSelector() {
  const langNames = {
    en:"English", ru:"Русский", kk:"Қазақша", de:"Deutsch", fr:"Français", es:"Español",
    zh:"中文", ar:"العربية", tr:"Türkçe", it:"Italiano", pt:"Português", ja:"日本語",
    ko:"한국어", hi:"हिन्दी", uz:"Oʻzbekcha", ky:"Кыргызча", uk:"Українська", pl:"Polski",
    nl:"Nederlands", sv:"Svenska"
  };
  const sel = document.getElementById("lang-select");
  sel.innerHTML = "";
  Object.keys(translations).forEach(code => {
    const opt = document.createElement("option");
    opt.value = code;
    opt.textContent = langNames[code] || code;
    if (code === currentLang) opt.selected = true;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", e => {
    currentLang = e.target.value;
    localStorage.setItem("fp_lang", currentLang);
    applyTranslations();
    // Persist server-side so alert emails are sent in this language too.
    if (typeof authToken !== "undefined" && authToken) {
      authApi("/api/me/language", { method: "POST", body: JSON.stringify({ lang: currentLang }) }).catch(() => {});
    }
    reanalyzeOnLanguageChange();
  });
}

let hasAiAnalysis = false;

async function reanalyzeOnLanguageChange() {
  // Legacy dashboard AI panel: only re-run if an analysis is already showing
  if (hasAiAnalysis) {
    const payload = {
      factory_name: document.getElementById("f-name").value.trim(),
      machine_count: parseInt(document.getElementById("f-count").value || 6),
      energy_cost: parseFloat(document.getElementById("f-cost").value || 0.12),
      machine_type: document.getElementById("f-type").value,
      temperature: parseFloat(document.getElementById("f-temp").value || 65),
      vibration: parseFloat(document.getElementById("f-vibration").value || 3.5),
      load: parseFloat(document.getElementById("f-load").value || 60),
      lang: currentLang,
    };
    try {
      const result = await authApi("/api/analyze", { method: "POST", body: JSON.stringify(payload) });
      renderAiAnalysis(result.analysis);
    } catch (e) {}
  }

  // AI Insights feed: if it's the visible page, re-analyze every listed factory
  // in the new language so nothing on screen is left in the old language.
  const aiPage = document.getElementById("page-ai");
  if (aiPage && !aiPage.classList.contains("hidden") && userFactories && userFactories.length) {
    for (const factory of userFactories) {
      try {
        await authApi(`/api/factories/${factory.id}`, {
          method: "PUT",
          body: JSON.stringify({
            factory_name: factory.factory_name, machine_count: factory.machines, energy_cost: factory.energy_cost,
            machine_type: factory.machine_type, temperature: factory.temperature, vibration: factory.vibration, load: factory.load,
            lang: currentLang,
          }),
        });
      } catch (e) {}
    }
    await loadAiFeed();
  }
}

let MACHINE_TYPES = ["CNC", "Compressor", "Conveyor", "Motor", "Pump", "Press", "Robot Arm"];
const MACHINE_TYPE_LABELS = {
  en: {CNC:"CNC", Compressor:"Compressor", Conveyor:"Conveyor", Motor:"Motor", Pump:"Pump", Press:"Press", "Robot Arm":"Robot Arm"},
  ru: {CNC:"ЧПУ станок", Compressor:"Компрессор", Conveyor:"Конвейер", Motor:"Двигатель", Pump:"Насос", Press:"Пресс", "Robot Arm":"Робот-манипулятор"},
  kk: {CNC:"ЧПУ станогы", Compressor:"Компрессор", Conveyor:"Конвейер", Motor:"Қозғалтқыш", Pump:"Сорғы", Press:"Пресс", "Robot Arm":"Робот-манипулятор"},
  de: {CNC:"CNC-Maschine", Compressor:"Kompressor", Conveyor:"Förderband", Motor:"Motor", Pump:"Pumpe", Press:"Presse", "Robot Arm":"Roboterarm"},
  fr: {CNC:"Machine CNC", Compressor:"Compresseur", Conveyor:"Convoyeur", Motor:"Moteur", Pump:"Pompe", Press:"Presse", "Robot Arm":"Bras Robotisé"},
  es: {CNC:"Máquina CNC", Compressor:"Compresor", Conveyor:"Transportador", Motor:"Motor", Pump:"Bomba", Press:"Prensa", "Robot Arm":"Brazo Robótico"},
  zh: {CNC:"数控机床", Compressor:"压缩机", Conveyor:"传送带", Motor:"电机", Pump:"泵", Press:"压力机", "Robot Arm":"机械臂"},
  ar: {CNC:"آلة CNC", Compressor:"ضاغط", Conveyor:"ناقل", Motor:"محرك", Pump:"مضخة", Press:"مكبس", "Robot Arm":"ذراع روبوتية"},
  tr: {CNC:"CNC Makinesi", Compressor:"Kompresör", Conveyor:"Konveyör", Motor:"Motor", Pump:"Pompa", Press:"Pres", "Robot Arm":"Robot Kol"},
  it: {CNC:"Macchina CNC", Compressor:"Compressore", Conveyor:"Nastro Trasportatore", Motor:"Motore", Pump:"Pompa", Press:"Pressa", "Robot Arm":"Braccio Robotico"},
  pt: {CNC:"Máquina CNC", Compressor:"Compressor", Conveyor:"Esteira", Motor:"Motor", Pump:"Bomba", Press:"Prensa", "Robot Arm":"Braço Robótico"},
  ja: {CNC:"CNC工作機械", Compressor:"コンプレッサー", Conveyor:"コンベア", Motor:"モーター", Pump:"ポンプ", Press:"プレス", "Robot Arm":"ロボットアーム"},
  ko: {CNC:"CNC 기계", Compressor:"압축기", Conveyor:"컨베이어", Motor:"모터", Pump:"펌프", Press:"프레스", "Robot Arm":"로봇 팔"},
  hi: {CNC:"CNC मशीन", Compressor:"कंप्रेसर", Conveyor:"कन्वेयर", Motor:"मोटर", Pump:"पंप", Press:"प्रेस", "Robot Arm":"रोबोट आर्म"},
  uz: {CNC:"CNC stanogi", Compressor:"Kompressor", Conveyor:"Konveyer", Motor:"Motor", Pump:"Nasos", Press:"Press", "Robot Arm":"Robot qo'l"},
  ky: {CNC:"ЧПУ станогу", Compressor:"Компрессор", Conveyor:"Конвейер", Motor:"Кыймылдаткыч", Pump:"Соргуч", Press:"Пресс", "Robot Arm":"Робот-манипулятор"},
  uk: {CNC:"ЧПУ верстат", Compressor:"Компресор", Conveyor:"Конвеєр", Motor:"Двигун", Pump:"Насос", Press:"Прес", "Robot Arm":"Робот-маніпулятор"},
  pl: {CNC:"Maszyna CNC", Compressor:"Sprężarka", Conveyor:"Przenośnik", Motor:"Silnik", Pump:"Pompa", Press:"Prasa", "Robot Arm":"Ramię Robota"},
  nl: {CNC:"CNC-machine", Compressor:"Compressor", Conveyor:"Transportband", Motor:"Motor", Pump:"Pomp", Press:"Pers", "Robot Arm":"Robotarm"},
  sv: {CNC:"CNC-maskin", Compressor:"Kompressor", Conveyor:"Transportband", Motor:"Motor", Pump:"Pump", Press:"Press", "Robot Arm":"Robotarm"},
};
function machineTypeLabel(code) {
  return (MACHINE_TYPE_LABELS[currentLang] && MACHINE_TYPE_LABELS[currentLang][code]) || code;
}
function buildMachineTypeSelect() {
  ["f-type", "m-type"].forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = "";
    MACHINE_TYPES.forEach(mt => {
      const opt = document.createElement("option");
      opt.value = mt;
      opt.textContent = machineTypeLabel(mt);
      sel.appendChild(opt);
    });
    if (prev) sel.value = prev;
  });
}

function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  const colors = { info: "bg-cyan-600", success: "bg-emerald-500", error: "bg-red-500" };
  const el = document.createElement("div");
  el.className = `toast ${colors[type] || colors.info} text-white text-sm px-4 py-3 rounded-xl shadow-lg max-w-xs`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .3s"; setTimeout(() => el.remove(), 300); }, 3000);
}

/* ============================ CHART ============================ */
let liveChart;
function initChart() {
  const ctx = document.getElementById("liveChart");
  liveChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: t("kpi_energy"), data: [], borderColor: "#22d3ee", backgroundColor: "rgba(34,211,238,.08)", tension: .35, fill: true, pointRadius: 0, borderWidth: 2 },
        { label: t("kpi_efficiency"), data: [], borderColor: "#a78bfa", backgroundColor: "rgba(167,139,250,.08)", tension: .35, fill: true, pointRadius: 0, borderWidth: 2, yAxisID: "y1" },
      ],
    },
    options: {
      responsive: true,
      animation: { duration: 400 },
      plugins: { legend: { labels: { color: "#cbd5e1" } } },
      scales: {
        x: { ticks: { color: "#64748b", maxTicksLimit: 6 }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { ticks: { color: "#64748b" }, grid: { color: "rgba(255,255,255,0.04)" } },
        y1: { position: "right", ticks: { color: "#64748b" }, grid: { display: false }, min: 0, max: 100 },
      },
    },
  });
}
function pushChartPoint(kpis) {
  const label = new Date().toLocaleTimeString();
  liveChart.data.labels.push(label);
  liveChart.data.datasets[0].data.push(kpis.energy_usage);
  liveChart.data.datasets[1].data.push(kpis.efficiency);
  if (liveChart.data.labels.length > 20) {
    liveChart.data.labels.shift();
    liveChart.data.datasets.forEach(ds => ds.data.shift());
  }
  liveChart.update("none");
}

function updateChartLabels() {
  if (liveChart) {
    liveChart.data.datasets[0].label = t("kpi_energy");
    liveChart.data.datasets[1].label = t("kpi_efficiency");
    liveChart.update("none");
  }
  if (typeof scadaChart !== "undefined" && scadaChart) {
    scadaChart.data.datasets[0].label = t("temperature_label") || "Temp °C";
    scadaChart.data.datasets[1].label = t("vibration_label") || "Vibration mm/s";
    scadaChart.data.datasets[2].label = t("load_label") || "Load %";
    scadaChart.update("none");
  }
}

/* ============================ RENDER ============================ */
let lastMachines = [];
let lastKpis = null;

function statusLabel(status) {
  if (status === "critical") return t("status_critical");
  if (status === "warning") return t("status_warning");
  if (status === "stopped") return t("status_stopped");
  if (status === "maintenance") return t("status_maintenance");
  return t("status_running");
}

function renderKpis(kpis) {
  document.getElementById("kpi-energy").innerHTML = kpis.energy_usage + ' <span class="text-sm font-normal">' + t("kwh_unit") + '</span>';
  document.getElementById("kpi-efficiency").textContent = kpis.efficiency + "%";
  document.getElementById("kpi-active").textContent = kpis.active_machines + "/" + kpis.total_machines;
  document.getElementById("kpi-alerts").textContent = kpis.alerts;
}

function renderMachineList(machines) {
  const list = document.getElementById("machine-list");
  list.innerHTML = "";
  machines.forEach(m => {
    const card = document.createElement("div");
    card.className = "machine-card glass rounded-xl p-3 border border-white/10";
    card.innerHTML = `
      <div class="flex items-center justify-between mb-1.5">
        <span class="font-medium text-sm">${m.name}</span>
        <span class="flex items-center gap-1.5 text-xs status-${m.status}">
          <span class="w-2 h-2 rounded-full bg-status-${m.status}"></span>${statusLabel(m.status)}
        </span>
      </div>
      <div class="text-xs text-slate-400 gauge-value">🌡 ${m.temperature}°C · 📳 ${m.vibration} mm/s · ⚙ ${m.load}%</div>
    `;
    list.appendChild(card);
  });
}

/* ============================ AI PANEL ============================ */
function renderAiAnalysis(analysis) {
  const panel = document.getElementById("ai-panel-content");
  panel.innerHTML = `
    <div class="flex flex-col gap-4">
      <div>
        <div class="text-xs uppercase text-red-400 ai-section-title mb-1">${t("ai_risks")}</div>
        <div class="text-sm text-slate-200 leading-relaxed">${analysis.risks}</div>
      </div>
      <div>
        <div class="text-xs uppercase text-cyan-400 ai-section-title mb-1">${t("ai_efficiency_insights")}</div>
        <div class="text-sm text-slate-200 leading-relaxed">${analysis.efficiency_insights}</div>
      </div>
      <div>
        <div class="text-xs uppercase text-emerald-400 ai-section-title mb-1">${t("ai_optimizations")}</div>
        <div class="text-sm text-slate-200 leading-relaxed">${analysis.optimizations}</div>
      </div>
    </div>
  `;
}

/* ============================ API ============================ */
async function api(path, options = {}) {
  const headers = Object.assign({ "Content-Type": "application/json" }, options.headers || {});
  const res = await fetch(path, Object.assign({}, options, { headers }));
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "request failed");
  return data;
}

async function refreshData() {
  try {
    const data = await authApi("/api/data");
    lastMachines = data.machines;
    lastKpis = data.kpis;
    renderKpis(data.kpis);
    renderMachineList(data.machines);
    pushChartPoint(data.kpis);
    document.getElementById("last-updated").textContent = new Date().toLocaleTimeString();
  } catch (e) {
    console.error("refresh error", e);
  }
}

document.getElementById("factory-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = {
    factory_name: document.getElementById("f-name").value.trim(),
    machine_count: parseInt(document.getElementById("f-count").value || 6),
    energy_cost: parseFloat(document.getElementById("f-cost").value || 0.12),
    machine_type: document.getElementById("f-type").value,
    temperature: parseFloat(document.getElementById("f-temp").value || 65),
    vibration: parseFloat(document.getElementById("f-vibration").value || 3.5),
    load: parseFloat(document.getElementById("f-load").value || 60),
    lang: currentLang,
  };
  const btn = document.getElementById("submit-btn");
  const spinner = document.getElementById("submit-spinner");
  const label = document.getElementById("submit-label");
  btn.classList.add("opacity-70");
  spinner.classList.remove("hidden");
  label.textContent = t("submitting");
  try {
    await authApi("/api/factory", { method: "POST", body: JSON.stringify(payload) });
    showToast(t("toast_updated"), "success");
    await refreshData();
    label.textContent = t("ai_analyzing");
    const result = await authApi("/api/analyze", { method: "POST", body: JSON.stringify(payload) });
    renderAiAnalysis(result.analysis);
    hasAiAnalysis = true;
    showToast(t("toast_analysis_done"), "success");
  } catch (err) {
    showToast(t("toast_error"), "error");
  } finally {
    btn.classList.remove("opacity-70");
    spinner.classList.add("hidden");
    label.textContent = t("submit_btn");
  }
});

/* ============================ AUTH ============================ */
let authToken = localStorage.getItem("fp_token") || null;
let currentUser = null;

async function authApi(path, options = {}) {
  const headers = Object.assign({ "Content-Type": "application/json" }, options.headers || {});
  if (authToken) headers["Authorization"] = "Bearer " + authToken;
  const res = await fetch(path, Object.assign({}, options, { headers }));
  let data = {};
  try { data = await res.json(); } catch (e) {}
  if (res.status === 401) {
    localStorage.removeItem("fp_token");
    window.location.href = "/login";
    throw new Error("unauthorized");
  }
  if (!res.ok) {
    const err = new Error(data.error || "request_failed");
    err.code = data.error;
    throw err;
  }
  return data;
}

async function requireAuthOrRedirect() {
  // No token = a visitor who has never signed in. Send them to registration
  // (the register page links to login for people who already have an account).
  if (!authToken) {
    window.location.href = localStorage.getItem("fp_seen") ? "/login" : "/register";
    return false;
  }
  try {
    const data = await authApi("/api/me");
    currentUser = data.user;
    localStorage.setItem("fp_seen", "1");
    document.getElementById("user-name").textContent = currentUser.full_name;
    document.getElementById("user-avatar").textContent = (currentUser.full_name || "?").trim().charAt(0).toUpperCase();
    applyRoleVisibility(currentUser.capabilities);
    const roleBadge = document.getElementById("user-role-badge");
    if (roleBadge && currentUser.role) roleBadge.textContent = t("role_" + currentUser.role);
    return true;
  } catch (e) {
    return false;
  }
}

document.getElementById("btn-logout").addEventListener("click", () => {
  localStorage.removeItem("fp_token");
  window.location.href = "/login";
});

/* ============================ SIDEBAR NAV ============================ */
function showPage(page) {
  ["dashboard", "factories", "live", "twin", "system", "roi", "history", "oee", "workorders", "story", "alerts", "ai"].forEach(p => {
    document.getElementById("page-" + p).classList.toggle("hidden", p !== page);
  });
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.toggle("active", b.dataset.page === page));
  document.querySelectorAll(".nav-btn-m").forEach(b => b.classList.toggle("active", b.dataset.page === page));
  if (page === "factories") loadFactories();
  if (page === "ai") loadAiFeed();
  if (page === "live") initLiveMonitor();
  if (page === "alerts") loadAlerts();
  if (page === "twin") initDigitalTwin();
  if (page === "system") loadSystemIntelligence();
  if (page === "roi") loadRoiDashboard();
  if (page === "story") initStoryMode();
  if (page === "history") initHistory();
  if (page === "oee") loadOee();
  if (page === "workorders") loadWorkOrders();
}
document.querySelectorAll(".nav-btn, .nav-btn-m").forEach(btn => {
  btn.addEventListener("click", () => showPage(btn.dataset.page));
});

/* ============================ FACTORIES CRUD ============================ */
let userFactories = [];
let editingFactoryId = null;

function openFactoryModal(factory) {
  editingFactoryId = factory ? factory.id : null;
  document.getElementById("factory-modal-title").textContent = factory ? t("edit_factory_btn") : t("add_factory_btn");
  document.getElementById("modal-factory-submit-label").textContent = t("save_btn");
  document.getElementById("m-name").value = factory ? factory.factory_name : "";
  document.getElementById("m-count").value = factory ? factory.machines : 6;
  document.getElementById("m-cost").value = factory ? factory.energy_cost : 0.12;
  document.getElementById("m-type").value = factory ? factory.machine_type : MACHINE_TYPES[0];
  document.getElementById("m-temp").value = factory ? factory.temperature : 65;
  document.getElementById("m-vibration").value = factory ? factory.vibration : 3.5;
  document.getElementById("m-load").value = factory ? factory.load : 60;
  const modal = document.getElementById("modal-factory");
  modal.classList.remove("hidden");
  modal.classList.add("flex");
}
function closeFactoryModal() {
  const modal = document.getElementById("modal-factory");
  modal.classList.add("hidden");
  modal.classList.remove("flex");
  editingFactoryId = null;
}
document.getElementById("open-factory-modal").addEventListener("click", () => openFactoryModal(null));
document.querySelectorAll(".close-factory-modal").forEach(btn => btn.addEventListener("click", closeFactoryModal));

document.getElementById("modal-factory-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = {
    factory_name: document.getElementById("m-name").value.trim(),
    machine_count: parseInt(document.getElementById("m-count").value || 6),
    energy_cost: parseFloat(document.getElementById("m-cost").value || 0.12),
    machine_type: document.getElementById("m-type").value,
    temperature: parseFloat(document.getElementById("m-temp").value || 65),
    vibration: parseFloat(document.getElementById("m-vibration").value || 3.5),
    load: parseFloat(document.getElementById("m-load").value || 60),
    lang: currentLang,
  };
  if (!payload.factory_name) { showToast(t("err_missing_fields"), "error"); return; }
  const spinner = document.getElementById("modal-factory-spinner");
  spinner.classList.remove("hidden");
  try {
    if (editingFactoryId) {
      await authApi(`/api/factories/${editingFactoryId}`, { method: "PUT", body: JSON.stringify(payload) });
      showToast(t("factory_updated_toast"), "success");
    } else {
      await authApi("/api/factories", { method: "POST", body: JSON.stringify(payload) });
      showToast(t("factory_created_toast"), "success");
    }
    closeFactoryModal();
    await loadFactories();
  } catch (err) {
    showToast(t("err_generic"), "error");
  } finally {
    spinner.classList.add("hidden");
  }
});

async function loadFactories() {
  try {
    const data = await authApi("/api/factories");
    userFactories = data.factories;
    renderFactoriesGrid();
  } catch (e) {}
}

function renderFactoriesGrid() {
  const grid = document.getElementById("factories-grid");
  const empty = document.getElementById("factories-empty");
  grid.innerHTML = "";
  empty.classList.toggle("hidden", userFactories.length > 0);
  userFactories.forEach(f => {
    const card = document.createElement("div");
    card.className = "factory-card glass rounded-2xl p-5";
    const created = f.created_at ? new Date(f.created_at).toLocaleDateString() : "";
    card.innerHTML = `
      <div class="flex items-start justify-between mb-2">
        <div>
          <div class="font-semibold text-base">${f.factory_name}</div>
          <div class="text-xs text-slate-400">${machineTypeLabel(f.machine_type)} · ${f.machines} ${t("kpi_active").toLowerCase()}</div>
        </div>
      </div>
      <div class="grid grid-cols-3 gap-2 my-3 text-center">
        <div class="glass rounded-lg p-2"><div class="text-xs text-slate-500">°C</div><div class="font-mono text-sm">${f.temperature}</div></div>
        <div class="glass rounded-lg p-2"><div class="text-xs text-slate-500">mm/s</div><div class="font-mono text-sm">${f.vibration}</div></div>
        <div class="glass rounded-lg p-2"><div class="text-xs text-slate-500">%</div><div class="font-mono text-sm">${f.load}</div></div>
      </div>
      <div class="text-xs text-slate-500 mb-3">${t("created_label")}: ${created}</div>
      <div class="flex gap-2">
        <button class="btn-view-insights flex-1 input-field rounded-lg py-2 text-xs font-medium hover:border-cyan-400" data-id="${f.id}">${t("view_insights_btn")}</button>
        <button class="btn-edit-factory input-field rounded-lg px-3 py-2 text-xs" data-id="${f.id}">${t("edit_factory_btn")}</button>
        <button class="btn-delete-factory input-field rounded-lg px-3 py-2 text-xs text-red-400 hover:border-red-400" data-id="${f.id}">${t("delete_factory_btn")}</button>
      </div>
    `;
    grid.appendChild(card);
  });

  grid.querySelectorAll(".btn-edit-factory").forEach(btn => {
    btn.addEventListener("click", () => {
      const factory = userFactories.find(x => x.id == btn.dataset.id);
      if (factory) openFactoryModal(factory);
    });
  });
  grid.querySelectorAll(".btn-delete-factory").forEach(btn => {
    btn.addEventListener("click", async () => {
      if (!confirm(t("confirm_delete_factory"))) return;
      try {
        await authApi(`/api/factories/${btn.dataset.id}`, { method: "DELETE" });
        showToast(t("factory_deleted_toast"), "success");
        await loadFactories();
      } catch (e) { showToast(t("err_generic"), "error"); }
    });
  });
  grid.querySelectorAll(".btn-view-insights").forEach(btn => {
    btn.addEventListener("click", () => { showPage("ai"); });
  });
}

/* ============================ AI INSIGHTS FEED ============================ */
/* ============================ LIVE MONITOR (SCADA) ============================ */
let scadaMachines = [];
let scadaChart = null;
let scadaSocket = null;
let scadaPollInterval = null;
let liveMonitorInitialized = false;

function riskColor(risk) {
  if (risk > 65) return "#f87171";
  if (risk > 35) return "#fbbf24";
  return "#34d399";
}
function statusDotClass(status) {
  if (status === "stopped") return "bg-red-400";
  if (status === "maintenance") return "bg-amber-400";
  return "bg-emerald-400";
}

function initScadaChart() {
  const ctx = document.getElementById("scadaChart");
  if (!ctx || scadaChart) return;
  scadaChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "Temp °C", data: [], borderColor: "#22d3ee", backgroundColor: "rgba(34,211,238,.08)", tension: .35, fill: true, pointRadius: 0, borderWidth: 2 },
        { label: "Vibration mm/s", data: [], borderColor: "#a78bfa", backgroundColor: "rgba(167,139,250,.08)", tension: .35, fill: true, pointRadius: 0, borderWidth: 2 },
        { label: "Load %", data: [], borderColor: "#34d399", backgroundColor: "rgba(52,211,153,.08)", tension: .35, fill: true, pointRadius: 0, borderWidth: 2 },
      ],
    },
    options: {
      responsive: true, animation: { duration: 300 },
      plugins: { legend: { labels: { color: "#cbd5e1" } } },
      scales: {
        x: { ticks: { color: "#64748b", maxTicksLimit: 6 }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { ticks: { color: "#64748b" }, grid: { color: "rgba(255,255,255,0.04)" } },
      },
    },
  });
}

function pushScadaChartPoint(reading) {
  if (!scadaChart) return;
  const label = new Date().toLocaleTimeString();
  scadaChart.data.labels.push(label);
  scadaChart.data.datasets[0].data.push(reading.temperature);
  scadaChart.data.datasets[1].data.push(reading.vibration);
  scadaChart.data.datasets[2].data.push(reading.load);
  if (scadaChart.data.labels.length > 20) {
    scadaChart.data.labels.shift();
    scadaChart.data.datasets.forEach(ds => ds.data.shift());
  }
  scadaChart.update("none");
}

function sourceBadgeHtml(source) {
  if (source === "auto") {
    return `<span class="px-2 py-0.5 rounded-full text-xs font-semibold" style="background:rgba(52,211,153,.15); color:#34d399;">${t("source_auto")}</span>`;
  }
  return `<span class="px-2 py-0.5 rounded-full text-xs font-semibold" style="background:rgba(148,163,184,.15); color:#94a3b8;">${t("source_manual")}</span>`;
}

function applyLiveReadingToRow(reading) {
  if (typeof updateTwinMachineFromReading === "function") updateTwinMachineFromReading(reading);
  const row = document.querySelector(`tr[data-code="${reading.machineId}"]`);
  if (!row) return;
  row.querySelector(".cell-temp").textContent = reading.temperature;
  row.querySelector(".cell-vib").textContent = reading.vibration;
  row.querySelector(".cell-load").textContent = reading.load;
  const dot = row.querySelector(".status-dot");
  if (dot) dot.className = "status-dot w-2.5 h-2.5 rounded-full inline-block mr-1.5 " + statusDotClass(reading.status);
  const statusText = row.querySelector(".cell-status");
  if (statusText) statusText.textContent = statusLabel(reading.status);
  const sourceCell = row.querySelector(".cell-source");
  if (sourceCell && reading.source) sourceCell.innerHTML = sourceBadgeHtml(reading.source);
  const errCell = row.querySelector(".cell-error");
  if (errCell) {
    errCell.innerHTML = reading.error_code
      ? `<span class="ml-1.5 px-1.5 py-0.5 rounded text-xs" style="background:rgba(248,113,113,.15); color:#f87171;" title="${reading.error_code}">⚠ ${reading.error_code}</span>`
      : "";
  }
  pushScadaChartPoint(reading);
}

function applyAiUpdateToRow(update) {
  const row = document.querySelector(`tr[data-id="${update.machine_id}"]`);
  if (!row) return;
  const cell = row.querySelector(".cell-risk");
  if (cell) {
    cell.style.color = riskColor(update.risk);
    const percentEl = cell.querySelector("div:first-child");
    if (percentEl) percentEl.textContent = update.risk + "%";
    const predictionEl = cell.querySelector(".cell-prediction");
    if (predictionEl && update.prediction) {
      predictionEl.textContent = update.prediction;
      predictionEl.title = update.prediction;
    }
  }
}

async function loadMachineMode() {
  try {
    const data = await authApi("/api/mode");
    document.getElementById("mode-badge").textContent = data.active_mode;
    document.getElementById("usb-status").textContent = data.usb_available ? "✓" : "—";
    document.getElementById("plc-status").textContent = data.plc_available ? "✓" : "—";
    document.getElementById("modbus-status").textContent = data.modbus_available ? "✓" : "—";
    document.getElementById("opcua-status").textContent = data.opcua_available ? "✓" : "—";
    document.getElementById("mqtt-status").textContent = data.mqtt_available ? "✓" : "—";
  } catch (e) {}
}

function renderScadaTable() {
  const tbody = document.getElementById("scada-table-body");
  const empty = document.getElementById("scada-empty");
  tbody.innerHTML = "";
  empty.classList.toggle("hidden", scadaMachines.length > 0);
  scadaMachines.forEach(m => {
    const tr = document.createElement("tr");
    tr.className = "border-b border-white/5";
    tr.dataset.code = m.machine_code;
    tr.dataset.id = m.id;
    tr.innerHTML = `
      <td class="py-2 pr-3 font-medium gauge-value">${m.machine_code}${m.error_code ? `<span class="cell-error ml-1.5 px-1.5 py-0.5 rounded text-xs" style="background:rgba(248,113,113,.15); color:#f87171;" title="${m.error_code}">⚠ ${m.error_code}</span>` : '<span class="cell-error"></span>'}</td>
      <td class="py-2 pr-3">${m.machine_name}</td>
      <td class="py-2 pr-3"><span class="status-dot w-2.5 h-2.5 rounded-full inline-block mr-1.5 ${statusDotClass(m.status)}"></span><span class="cell-status">${statusLabel(m.status)}</span></td>
      <td class="py-2 pr-3 gauge-value cell-temp">${m.temperature}</td>
      <td class="py-2 pr-3 gauge-value cell-vib">${m.vibration}</td>
      <td class="py-2 pr-3 gauge-value cell-load">${m.load}</td>
      <td class="py-2 pr-3 gauge-value cell-risk" style="color:${riskColor(m.failure_risk)}">
        <div class="font-semibold">${m.failure_risk}%</div>
        <div class="text-xs font-normal text-slate-400 cell-prediction" title="${m.estimated_failure_time || ''}">${m.estimated_failure_time || ''}</div>
      </td>
      <td class="py-2 pr-3 cell-source">${sourceBadgeHtml("manual_baseline")}</td>
      <td class="py-2 text-right whitespace-nowrap">
        <button class="btn-energy-insights text-slate-400 hover:text-cyan-400 mr-2" data-id="${m.id}" title="${t('energy_insights_title')}">⚡</button>
        <button class="btn-delete-scada-machine text-slate-500 hover:text-red-400" data-id="${m.id}">🗑</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
  tbody.querySelectorAll(".btn-energy-insights").forEach(btn => {
    btn.addEventListener("click", () => openEnergyInsights(btn.dataset.id));
  });
  tbody.querySelectorAll(".btn-delete-scada-machine").forEach(btn => {
    btn.addEventListener("click", async () => {
      try {
        await authApi(`/api/machines/${btn.dataset.id}`, { method: "DELETE" });
        await loadScadaMachines();
      } catch (e) { showToast(t("err_generic"), "error"); }
    });
  });
}

async function loadScadaMachines() {
  try {
    const data = await authApi("/api/machines");
    scadaMachines = data.machines;
    renderScadaTable();
  } catch (e) {}
}

/* ============================ ENERGY INSIGHTS ============================ */
function energyMetricRow(label, value, color) {
  return `
    <div class="glass rounded-xl p-3 flex items-center justify-between">
      <span class="text-xs text-slate-400">${label}</span>
      <span class="font-semibold gauge-value" style="color:${color || '#e6edf5'}">${value}</span>
    </div>
  `;
}

function renderEnergyInsights(m, energy) {
  const body = document.getElementById("energy-insights-body");
  document.getElementById("energy-insights-machine-name").textContent = `${t("energy_insights_title")} — ${m.machine_name}`;

  let html = "";

  // 0) Failure Prediction (AI risk % + estimated days-to-failure)
  const riskC = riskColor(m.failure_risk);
  html += `<div>
    <div class="text-xs uppercase text-red-400 ai-section-title mb-2">${t("failure_prediction_title")}</div>
    <div class="glass rounded-xl p-3 flex items-center justify-between mb-1">
      <span class="text-xs text-slate-400">${t("risk_col")}</span>
      <span class="font-bold gauge-value" style="color:${riskC}">${m.failure_risk}%</span>
    </div>
    ${m.estimated_failure_time ? `<div class="text-sm mt-1" style="color:${riskC}">⏱ ${m.estimated_failure_time}</div>` : ""}
  </div>`;

  // 1) Idle Power Detection
  html += `<div>
    <div class="text-xs uppercase text-amber-400 ai-section-title mb-2">${t("idle_power_title")}</div>
    ${energy.is_idle
      ? `<div class="rounded-xl p-3 mb-1" style="background:rgba(251,191,36,.12); border:1px solid rgba(251,191,36,.35);">
           <div class="font-semibold text-sm" style="color:#fbbf24;">⚠ ${t("idle_active_msg").replace("{kw}", energy.idle_waste_kw)}</div>
         </div>`
      : `<div class="text-sm text-slate-400">${t("idle_none_msg")}</div>`}
  </div>`;

  // 2) Predictive Energy Loss (friction)
  html += `<div>
    <div class="text-xs uppercase text-red-400 ai-section-title mb-2">${t("friction_loss_title")}</div>
    ${energy.friction_overhead_pct > 10
      ? `<div class="rounded-xl p-3 mb-1" style="background:rgba(248,113,113,.12); border:1px solid rgba(248,113,113,.35);">
           <div class="font-semibold text-sm" style="color:#f87171;">⚠ ${t("friction_active_msg").replace("{pct}", energy.friction_overhead_pct).replace("{kw}", energy.extra_power_kw)}</div>
         </div>`
      : `<div class="text-sm text-slate-400">${t("friction_none_msg")}</div>`}
  </div>`;

  // 3) Specific Energy Consumption
  html += `<div>
    <div class="text-xs uppercase text-cyan-400 ai-section-title mb-2">${t("sec_title")}</div>
    ${energy.specific_energy_consumption !== null && energy.specific_energy_consumption !== undefined
      ? energyMetricRow(t("sec_label"), energy.specific_energy_consumption + " " + t("sec_unit"), "#22d3ee")
      : `<div class="text-sm text-slate-400">${t("sec_no_data_msg")}</div>`}
  </div>`;

  // 4) Optimal Load Zone
  html += `<div>
    <div class="text-xs uppercase text-emerald-400 ai-section-title mb-2">${t("optimal_load_title")}</div>
    ${energyMetricRow(t("optimal_load_label"), energy.optimal_load_pct + "%", "#34d399")}
    ${energyMetricRow(t("current_load_label"), energy.current_load_pct + "%", energy.at_optimal_load ? "#34d399" : "#fbbf24")}
    <div class="text-xs mt-2" style="color:${energy.at_optimal_load ? '#34d399' : '#fbbf24'}">
      ${energy.at_optimal_load ? "✓ " + t("at_optimal_msg") : "→ " + t("adjust_to_optimal_msg").replace("{pct}", energy.optimal_load_pct)}
    </div>
  </div>`;

  body.innerHTML = html;
}

async function openEnergyInsights(machineId) {
  const m = scadaMachines.find(x => x.id == machineId);
  if (!m) return;
  const modal = document.getElementById("modal-energy-insights");
  modal.classList.remove("hidden");
  modal.classList.add("flex");
  document.getElementById("energy-insights-body").innerHTML = `<div class="flex justify-center py-6"><span class="spinner"></span></div>`;
  try {
    const reading = await authApi(`/api/machines/${machineId}/live`);
    renderEnergyInsights(m, reading.energy);
  } catch (e) {
    document.getElementById("energy-insights-body").innerHTML = `<div class="text-sm text-red-400">${t("err_generic")}</div>`;
  }
}

document.querySelectorAll(".close-energy-modal").forEach(btn => {
  btn.addEventListener("click", () => {
    const modal = document.getElementById("modal-energy-insights");
    modal.classList.add("hidden");
    modal.classList.remove("flex");
  });
});

function connectScadaSocket() {
  if (typeof io === "undefined") { startScadaPolling(); return; }
  try {
    scadaSocket = io();
    scadaSocket.on("connect", () => {
      scadaSocket.emit("authenticate", { token: authToken });
      document.getElementById("ws-dot").className = "w-2 h-2 rounded-full bg-emerald-400";
      document.getElementById("ws-label").textContent = "WebSocket";
      if (scadaPollInterval) { clearInterval(scadaPollInterval); scadaPollInterval = null; }
    });
    scadaSocket.on("machine_reading", (reading) => applyLiveReadingToRow(reading));
    scadaSocket.on("machine_ai_update", (update) => applyAiUpdateToRow(update));
    scadaSocket.on("critical_alert", (alert) => handleIncomingCriticalAlert(alert));
    scadaSocket.on("connect_error", () => startScadaPolling());
    scadaSocket.on("disconnect", () => startScadaPolling());
  } catch (e) {
    startScadaPolling();
  }
}

function startScadaPolling() {
  document.getElementById("ws-dot").className = "w-2 h-2 rounded-full bg-slate-500";
  document.getElementById("ws-label").textContent = t("polling_mode");
  if (scadaPollInterval) return;
  scadaPollInterval = setInterval(async () => {
    for (const m of scadaMachines) {
      try {
        const reading = await authApi(`/api/machines/${m.id}/live`);
        applyLiveReadingToRow(reading);
      } catch (e) {}
    }
  }, 2000);
}

function initLiveMonitor() {
  initScadaChart();
  loadMachineMode();
  loadScadaMachines();
  if (!liveMonitorInitialized) {
    liveMonitorInitialized = true;
    connectScadaSocket();
  }
}

/* ============================ 3D DIGITAL TWIN ============================ */
let twinInitialized = false;
let twinScene, twinCamera, twinRenderer, twinControls, twinRaycaster, twinMouse;
let twinMachineMeshes = {}; // machine_code -> THREE.Mesh
let twinHoveredMesh = null;
let twinResizeObserver = null;

function twinStatusColor(status) {
  if (status === "stopped" || status === "critical") return 0xf87171;
  if (status === "maintenance" || status === "warning") return 0xfbbf24;
  return 0x34d399;
}

function setupTwinScene() {
  const container = document.getElementById("twin-canvas-container");
  const width = container.clientWidth || 800;
  const height = container.clientHeight || 500;

  twinScene = new THREE.Scene();
  twinScene.background = new THREE.Color(0x0b1120);
  twinScene.fog = new THREE.Fog(0x0b1120, 18, 45);

  twinCamera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  twinCamera.position.set(10, 10, 14);

  twinRenderer = new THREE.WebGLRenderer({ antialias: true });
  twinRenderer.setSize(width, height);
  twinRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.innerHTML = "";
  container.appendChild(twinRenderer.domElement);

  twinControls = new THREE.OrbitControls(twinCamera, twinRenderer.domElement);
  twinControls.enableDamping = true;
  twinControls.dampingFactor = 0.08;
  twinControls.maxPolarAngle = Math.PI / 2.1;
  twinControls.target.set(0, 0, 0);

  const ambient = new THREE.AmbientLight(0x8899aa, 0.7);
  twinScene.add(ambient);
  const dirLight = new THREE.DirectionalLight(0x66d9ef, 0.9);
  dirLight.position.set(8, 14, 6);
  twinScene.add(dirLight);
  const rimLight = new THREE.PointLight(0x7c3aed, 0.6, 40);
  rimLight.position.set(-10, 6, -10);
  twinScene.add(rimLight);

  // Factory floor
  const floorGeo = new THREE.PlaneGeometry(30, 30);
  const floorMat = new THREE.MeshStandardMaterial({ color: 0x111827, metalness: 0.2, roughness: 0.9 });
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  twinScene.add(floor);

  const grid = new THREE.GridHelper(30, 30, 0x22d3ee, 0x1e293b);
  grid.position.y = 0.01;
  twinScene.add(grid);

  twinRaycaster = new THREE.Raycaster();
  twinMouse = new THREE.Vector2();

  twinRenderer.domElement.addEventListener("mousemove", onTwinMouseMove);
  twinRenderer.domElement.addEventListener("click", onTwinClick);
  twinRenderer.domElement.addEventListener("mouseleave", () => {
    document.getElementById("twin-tooltip").classList.add("hidden");
  });

  if (window.ResizeObserver) {
    twinResizeObserver = new ResizeObserver(() => resizeTwinRenderer());
    twinResizeObserver.observe(container);
  }

  animateTwin();
}

function resizeTwinRenderer() {
  if (!twinRenderer) return;
  const container = document.getElementById("twin-canvas-container");
  const width = container.clientWidth || 800;
  const height = container.clientHeight || 500;
  twinCamera.aspect = width / height;
  twinCamera.updateProjectionMatrix();
  twinRenderer.setSize(width, height);
}

function animateTwin() {
  requestAnimationFrame(animateTwin);
  if (!twinRenderer) return;
  twinControls.update();
  const t = Date.now() * 0.002;
  Object.values(twinMachineMeshes).forEach(mesh => {
    if (mesh.userData.status === "stopped" || mesh.userData.status === "critical") {
      mesh.material.emissiveIntensity = 0.5 + Math.sin(t * 3) * 0.35;
    }
  });
  twinRenderer.render(twinScene, twinCamera);
}

function buildTwinMachines(machines) {
  // clear old meshes
  Object.values(twinMachineMeshes).forEach(mesh => twinScene.remove(mesh));
  twinMachineMeshes = {};

  const empty = document.getElementById("twin-empty");
  empty.classList.toggle("hidden", machines.length > 0);
  if (machines.length === 0) return;

  const cols = Math.ceil(Math.sqrt(machines.length));
  const spacing = 4;
  const offset = ((cols - 1) * spacing) / 2;

  machines.forEach((m, i) => {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const height = 1.2 + (m.load / 100) * 1.6;
    const geo = new THREE.BoxGeometry(1.6, height, 1.6);
    const color = twinStatusColor(m.status);
    const mat = new THREE.MeshStandardMaterial({
      color, emissive: color, emissiveIntensity: 0.35, metalness: 0.3, roughness: 0.5,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(col * spacing - offset, height / 2, row * spacing - offset);
    mesh.userData = { id: m.id, machine_code: m.machine_code, machine_name: m.machine_name, status: m.status, temperature: m.temperature, vibration: m.vibration, load: m.load, failure_risk: m.failure_risk };
    twinScene.add(mesh);
    twinMachineMeshes[m.machine_code] = mesh;
  });
}

function onTwinMouseMove(event) {
  const rect = twinRenderer.domElement.getBoundingClientRect();
  twinMouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  twinMouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  twinRaycaster.setFromCamera(twinMouse, twinCamera);
  const meshes = Object.values(twinMachineMeshes);
  const hits = twinRaycaster.intersectObjects(meshes);
  const tooltip = document.getElementById("twin-tooltip");

  if (hits.length > 0) {
    const mesh = hits[0].object;
    const d = mesh.userData;
    tooltip.innerHTML = `
      <div class="font-semibold mb-1">${d.machine_name} (${d.machine_code})</div>
      <div class="text-slate-300">${statusLabel(d.status)}</div>
      <div class="text-slate-400 gauge-value mt-1">🌡 ${d.temperature}°C · 📳 ${d.vibration} mm/s · ⚙ ${d.load}%</div>
      <div class="text-slate-400 gauge-value">${t("risk_col")}: ${d.failure_risk}%</div>
    `;
    tooltip.style.left = (event.clientX - rect.left + 12) + "px";
    tooltip.style.top = (event.clientY - rect.top + 12) + "px";
    tooltip.classList.remove("hidden");
    twinRenderer.domElement.style.cursor = "pointer";
  } else {
    tooltip.classList.add("hidden");
    twinRenderer.domElement.style.cursor = "default";
  }
}

function onTwinClick(event) {
  const rect = twinRenderer.domElement.getBoundingClientRect();
  twinMouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  twinMouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  twinRaycaster.setFromCamera(twinMouse, twinCamera);
  const meshes = Object.values(twinMachineMeshes);
  const hits = twinRaycaster.intersectObjects(meshes);
  if (hits.length > 0) {
    openEnergyInsights(hits[0].object.userData.id);
  }
}

function updateTwinMachineFromReading(reading) {
  const mesh = twinMachineMeshes[reading.machineId];
  if (!mesh) return;
  const color = twinStatusColor(reading.status);
  mesh.material.color.setHex(color);
  mesh.material.emissive.setHex(color);
  mesh.userData.status = reading.status;
  mesh.userData.temperature = reading.temperature;
  mesh.userData.vibration = reading.vibration;
  mesh.userData.load = reading.load;
  const newHeight = 1.2 + (reading.load / 100) * 1.6;
  mesh.scale.y = newHeight / mesh.geometry.parameters.height;
  mesh.position.y = newHeight / 2;
}

async function initDigitalTwin() {
  if (typeof THREE === "undefined") {
    document.getElementById("twin-canvas-container").innerHTML =
      `<div class="flex items-center justify-center h-full text-sm text-slate-400 p-6 text-center">${t("twin_unavailable_msg")}</div>`;
    return;
  }
  if (!twinInitialized) {
    twinInitialized = true;
    setupTwinScene();
  } else {
    resizeTwinRenderer();
  }
  try {
    const data = await authApi("/api/machines");
    scadaMachines = data.machines;
    buildTwinMachines(scadaMachines);
    buildSimMachineSelect();
  } catch (e) {}
}

/* ==================== WHAT-IF SIMULATION (Digital Twin) ==================== */
function buildSimMachineSelect() {
  const sel = document.getElementById("sim-machine");
  if (!sel) return;
  const prev = sel.value;
  sel.innerHTML = "";
  scadaMachines.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.machine_code} — ${m.machine_name}`;
    sel.appendChild(opt);
  });
  if (prev) sel.value = prev;
}

["sim-temp", "sim-vib", "sim-load"].forEach(id => {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener("input", () => {
    const v = parseInt(el.value);
    document.getElementById(id + "-val").textContent = (v >= 0 ? "+" : "") + v + "%";
  });
});

function simMetricCard(label, before, after, unit, higherIsWorse) {
  const changed = before !== after;
  const worse = higherIsWorse ? after > before : after < before;
  const color = !changed ? "#94a3b8" : (worse ? "#f87171" : "#34d399");
  const arrow = !changed ? "" : (after > before ? "↑" : "↓");
  return `
    <div class="glass rounded-xl p-4">
      <div class="text-xs text-slate-400 mb-2">${label}</div>
      <div class="flex items-center gap-2">
        <span class="text-slate-500 gauge-value text-sm">${before}${unit}</span>
        <span class="text-slate-500">→</span>
        <span class="font-bold gauge-value text-lg" style="color:${color}">${after}${unit} ${arrow}</span>
      </div>
    </div>`;
}

document.getElementById("btn-run-simulation").addEventListener("click", async () => {
  const machineId = document.getElementById("sim-machine").value;
  if (!machineId) { showToast(t("no_machines_yet"), "error"); return; }

  const spinner = document.getElementById("sim-spinner");
  spinner.classList.remove("hidden");
  try {
    const res = await authApi(`/api/machines/${machineId}/simulate`, {
      method: "POST",
      body: JSON.stringify({
        temp_delta_pct: parseFloat(document.getElementById("sim-temp").value),
        vib_delta_pct: parseFloat(document.getElementById("sim-vib").value),
        load_delta_pct: parseFloat(document.getElementById("sim-load").value),
      }),
    });
    const b = res.baseline;
    const box = document.getElementById("sim-results");
    let html = "";
    html += simMetricCard(t("failure_probability_label"), b.failure_probability, res.failure_probability, "%", true);
    html += simMetricCard(t("stress_level_label"), b.stress_level, res.stress_level, "%", true);
    html += simMetricCard(t("temperature_label"), b.temperature, res.input.temperature, "°C", true);
    html += simMetricCard(t("vibration_label"), b.vibration, res.input.vibration, " mm/s", true);

    const rulTxt = res.rul_hours === null ? t("rul_healthy") : formatRul(res.rul_hours);
    html += `
      <div class="glass rounded-xl p-4 sm:col-span-2">
        <div class="text-xs text-slate-400 mb-2">${t("predicted_status_label")}</div>
        <div class="flex items-center gap-2 mb-3">
          <span class="w-2.5 h-2.5 rounded-full ${statusDotClass(b.status)}"></span>
          <span class="text-slate-500 text-sm">${statusLabel(b.status)}</span>
          <span class="text-slate-500">→</span>
          <span class="w-2.5 h-2.5 rounded-full ${statusDotClass(res.status)}"></span>
          <span class="font-semibold" style="color:${twinStatusColorHex(res.status)}">${statusLabel(res.status)}</span>
        </div>
        <div class="text-xs text-slate-400 mb-1">${t("rul_col")}: <span class="gauge-value text-slate-200">${rulTxt}</span>
          <span class="text-slate-500">(${t("confidence_label")} ${Math.round((res.confidence || 0) * 100)}%)</span></div>
        <div class="text-xs text-slate-400 mt-2">${t("root_cause_title")}:
          ${(res.root_causes || []).map(c => `<span class="ml-1 px-2 py-0.5 rounded-full" style="background:rgba(248,113,113,.15); color:#f87171">${t("cause_" + c.code)}</span>`).join("")}
        </div>
      </div>`;
    box.innerHTML = html;
    box.classList.remove("hidden");
  } catch (e) {
    showToast(t("err_generic"), "error");
  } finally {
    spinner.classList.add("hidden");
  }
});

function twinStatusColorHex(status) {
  if (status === "stopped" || status === "critical") return "#f87171";
  if (status === "maintenance" || status === "warning") return "#fbbf24";
  return "#34d399";
}

function formatRul(hours) {
  if (hours === null || hours === undefined) return t("rul_healthy");
  if (hours < 48) return hours + "h";
  return Math.round(hours / 24 * 10) / 10 + "d";
}

/* ==================== SYSTEM INTELLIGENCE ==================== */
async function loadSystemIntelligence() {
  try {
    const data = await authApi("/api/system/intelligence");

    const riskEl = document.getElementById("sys-risk");
    riskEl.textContent = data.system_risk + "%";
    riskEl.style.color = riskColor(data.system_risk);
    document.getElementById("sys-healthy").textContent = data.healthy;
    document.getElementById("sys-atrisk").textContent = data.at_risk;
    document.getElementById("sys-critical").textContent = data.critical;

    const clusters = document.getElementById("sys-clusters");
    clusters.innerHTML = (data.clusters || []).map(c => `
      <div class="glass rounded-xl p-3">
        <div class="flex items-center justify-between mb-1">
          <span class="font-medium text-sm">${c.section}</span>
          <span class="gauge-value font-semibold" style="color:${riskColor(c.max_risk)}">${c.max_risk}%</span>
        </div>
        <div class="text-xs text-slate-400">${c.count} ${t("machines_table_title").toLowerCase()} · ${t("avg_risk_label")}: ${c.avg_risk}%</div>
        <div class="text-xs text-slate-500 mt-1 gauge-value">${c.machines.join(", ")}</div>
      </div>`).join("");

    const prop = document.getElementById("sys-propagation");
    const propEmpty = document.getElementById("sys-propagation-empty");
    const items = data.propagation || [];
    propEmpty.classList.toggle("hidden", items.length > 0);
    prop.innerHTML = items.map(p => `
      <div class="glass rounded-xl p-3" style="border-left:3px solid #fbbf24">
        <div class="text-sm"><span class="gauge-value font-semibold" style="color:#f87171">${p.from}</span>
          <span class="text-slate-400 mx-1">→</span>
          <span class="gauge-value font-semibold">${p.to}</span></div>
        <div class="text-xs text-slate-400 mt-1">${t("added_risk_label")}: +${p.added_risk}% → ${t("effective_risk_label")}: <span style="color:${riskColor(p.effective_risk)}">${p.effective_risk}%</span></div>
      </div>`).join("");
  } catch (e) {}
}
document.getElementById("btn-refresh-system").addEventListener("click", loadSystemIntelligence);

/* ==================== ROI DASHBOARD ==================== */
function money(v) {
  return "$" + (v || 0).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

async function loadRoiDashboard() {
  try {
    const data = await authApi("/api/business/roi");
    const tt = data.totals;

    document.getElementById("roi-loss").textContent = money(tt.potential_loss);
    document.getElementById("roi-saved").textContent = money(tt.saved);
    document.getElementById("roi-energy").textContent = money(tt.wasted_energy_cost_month);
    document.getElementById("roi-efficiency").textContent = "+" + tt.efficiency_gain_pct + "%";

    const tbody = document.getElementById("roi-table-body");
    const empty = document.getElementById("roi-empty");
    empty.classList.toggle("hidden", (data.machines || []).length > 0);
    tbody.innerHTML = (data.machines || []).map(m => `
      <tr class="border-b border-white/5">
        <td class="py-2 pr-3">${m.machine_name}<div class="text-xs text-slate-500 gauge-value">${m.machine_code}</div></td>
        <td class="py-2 pr-3 gauge-value font-semibold" style="color:${riskColor(m.risk)}">${m.risk}%</td>
        <td class="py-2 pr-3 gauge-value">${formatRul(m.rul_hours)}</td>
        <td class="py-2 pr-3 gauge-value" style="color:#f87171">${money(m.potential_loss)}</td>
        <td class="py-2 pr-3 gauge-value" style="color:#34d399">${money(m.saved)}</td>
        <td class="py-2 text-xs">${m.top_cause ? t("cause_" + m.top_cause) : "—"}</td>
      </tr>`).join("");

    const a = data.assumptions;
    document.getElementById("roi-assumptions").textContent =
      t("roi_assumptions_msg")
        .replace("{downtime}", money(a.downtime_cost_per_hour))
        .replace("{hours}", a.repair_hours)
        .replace("{price}", a.energy_price_per_kwh);
  } catch (e) {}
}
document.getElementById("btn-refresh-roi").addEventListener("click", loadRoiDashboard);

/* ==================== STORY MODE (INVESTOR DEMO) ==================== */
const STORY_STAGE_COLORS = { low: "#34d399", medium: "#fbbf24", high: "#fb923c", critical: "#f87171" };

async function initStoryMode() {
  try {
    const data = await authApi("/api/machines");
    scadaMachines = data.machines;
    const sel = document.getElementById("story-machine");
    const prev = sel.value;
    sel.innerHTML = "";
    scadaMachines.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = `${m.machine_code} — ${m.machine_name}`;
      sel.appendChild(opt);
    });
    if (prev) sel.value = prev;

    const empty = document.getElementById("story-empty");
    const hasMachines = scadaMachines.length > 0;
    empty.classList.toggle("hidden", hasMachines);
    document.getElementById("btn-run-story").classList.toggle("hidden", !hasMachines);
    sel.classList.toggle("hidden", !hasMachines);
  } catch (e) {}
}

function renderStoryStage(stage, index) {
  const color = STORY_STAGE_COLORS[stage.priority] || "#94a3b8";
  const rul = stage.rul_hours === null ? t("rul_healthy") : formatRul(stage.rul_hours);
  const cause = (stage.root_causes || [{}])[0];
  const actions = (stage.suggested_actions || [])
    .map(a => `<span class="px-2 py-0.5 rounded-full text-xs mr-1" style="background:rgba(255,255,255,.07)">${t("action_" + a)}</span>`)
    .join("");

  const el = document.createElement("div");
  el.className = "glass rounded-xl p-4 fade-in";
  el.style.borderLeft = `3px solid ${color}`;
  el.style.animationDelay = (index * 0.35) + "s";
  el.innerHTML = `
    <div class="flex items-center justify-between mb-2 flex-wrap gap-2">
      <div class="flex items-center gap-2">
        <span class="gauge-value text-sm text-slate-400">+${stage.hours_in}h</span>
        <span class="font-semibold" style="color:${color}">${t("story_stage_" + stage.stage)}</span>
      </div>
      <span class="text-xs px-2 py-0.5 rounded-full font-semibold" style="background:${color}22; color:${color}">${t("priority_" + stage.priority)}</span>
    </div>
    <div class="text-xs text-slate-400 gauge-value mb-2">
      🌡 ${stage.temperature}°C · 📳 ${stage.vibration} mm/s · ⚙ ${stage.load}%
      · ${t("risk_col")}: <span style="color:${riskColor(stage.risk)}">${stage.risk}%</span>
      · ${t("rul_col")}: ${rul}
    </div>
    ${cause.code ? `<div class="text-xs text-slate-400 mb-2">${t("root_cause_title")}: <span class="text-slate-200">${t("cause_" + cause.code)}</span></div>` : ""}
    <div class="mt-1">${actions}</div>
  `;
  return el;
}


/* ==================== HISTORY / TRENDS ==================== */
let histChart = null, histRiskChart = null;

async function initHistory() {
  const sel = document.getElementById("hist-machine");
  const prev = sel.value;
  try {
    const data = await authApi("/api/machines");
    scadaMachines = data.machines;
    sel.innerHTML = "";
    scadaMachines.forEach(m => {
      const o = document.createElement("option");
      o.value = m.id; o.textContent = `${m.machine_code} - ${m.machine_name}`;
      sel.appendChild(o);
    });
    if (prev) sel.value = prev;
  } catch (e) {}
  await loadHistory();
}

function histTrendCard(label, first, last, unit, higherIsWorse, trend) {
  const delta = +(last - first).toFixed(2);
  const changed = Math.abs(delta) > 0.01;
  const worse = higherIsWorse ? delta > 0 : delta < 0;
  const color = !changed ? "#94a3b8" : (worse ? "#f87171" : "#34d399");
  const arrow = !changed ? "" : (delta > 0 ? "&#9650;" : "&#9660;");
  const dirKey = trend && trend.direction ? "trend_" + trend.direction : null;
  return `
    <div class="glass rounded-2xl p-4">
      <div class="text-xs text-slate-400 mb-2">${label}</div>
      <div class="flex items-baseline gap-2 flex-wrap">
        <span class="text-slate-500 gauge-value text-sm">${first}${unit}</span>
        <span class="text-slate-500">&rarr;</span>
        <span class="text-xl font-bold gauge-value" style="color:${color}">${last}${unit}</span>
      </div>
      <div class="text-xs mt-1" style="color:${color}">
        ${arrow} ${changed ? (delta > 0 ? "+" : "") + delta + unit : t("trend_flat")}
        ${dirKey ? `<span class="text-slate-500 ml-1">(${t(dirKey)})</span>` : ""}
      </div>
    </div>`;
}

async function loadHistory() {
  const machineId = document.getElementById("hist-machine").value;
  const hours = document.getElementById("hist-hours").value || 240;
  const empty = document.getElementById("hist-empty");
  const summary = document.getElementById("hist-summary");

  if (!machineId) {
    empty.classList.remove("hidden");
    summary.classList.add("hidden");
    return;
  }

  try {
    const d = await authApi(`/api/history/${machineId}?hours=${hours}`);
    const pts = d.points || [];

    empty.classList.toggle("hidden", pts.length > 0);
    summary.classList.toggle("hidden", pts.length === 0);
    if (!pts.length) {
      if (histChart) { histChart.destroy(); histChart = null; }
      if (histRiskChart) { histRiskChart.destroy(); histRiskChart = null; }
      return;
    }

    // Before/after: the single number a pilot is judged on.
    const first = pts[0], last = pts[pts.length - 1];
    const tr = d.trends || {};
    summary.innerHTML =
      histTrendCard(t("temperature_label"), first.temperature, last.temperature, "&deg;C", true, tr.temperature) +
      histTrendCard(t("vibration_label"), first.vibration, last.vibration, " mm/s", true, tr.vibration) +
      histTrendCard(t("load_label"), first.load, last.load, "%", false, tr.load) +
      histTrendCard(t("risk_col"), first.risk, last.risk, "%", true, tr.risk);

    const labels = pts.map(p => {
      const dt = new Date(p.t);
      return hours > 48
        ? `${dt.getDate()}.${dt.getMonth() + 1}`
        : dt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    });

    const axisOpts = {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: "#cbd5e1", boxWidth: 12 } } },
      scales: {
        y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,.06)" } },
        x: { ticks: { color: "#94a3b8", maxTicksLimit: 10 }, grid: { display: false } },
      },
    };

    if (histChart) histChart.destroy();
    histChart = new Chart(document.getElementById("hist-chart"), {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: t("temperature_label"), data: pts.map(p => p.temperature),
            borderColor: "#22d3ee", backgroundColor: "rgba(34,211,238,.10)", fill: true, tension: .3, pointRadius: 0, borderWidth: 2 },
          { label: t("vibration_label"), data: pts.map(p => p.vibration),
            borderColor: "#a78bfa", backgroundColor: "transparent", tension: .3, pointRadius: 0, borderWidth: 2 },
          { label: t("load_label"), data: pts.map(p => p.load),
            borderColor: "#34d399", backgroundColor: "transparent", tension: .3, pointRadius: 0, borderWidth: 2 },
        ],
      },
      options: axisOpts,
    });

    if (histRiskChart) histRiskChart.destroy();
    histRiskChart = new Chart(document.getElementById("hist-risk-chart"), {
      type: "line",
      data: {
        labels,
        datasets: [{ label: t("risk_col"), data: pts.map(p => p.risk),
          borderColor: "#f87171", backgroundColor: "rgba(248,113,113,.12)", fill: true, tension: .3, pointRadius: 0, borderWidth: 2 }],
      },
      options: { ...axisOpts, scales: { ...axisOpts.scales, y: { ...axisOpts.scales.y, min: 0, max: 100 } } },
    });
  } catch (e) {
    empty.classList.remove("hidden");
    summary.classList.add("hidden");
  }
}

document.getElementById("hist-machine").addEventListener("change", loadHistory);
document.getElementById("hist-hours").addEventListener("change", loadHistory);

/* ==================== OEE ==================== */
let oeeChart = null;

function oeeColor(v) {
  if (v >= 85) return "#34d399";
  if (v >= 60) return "#22d3ee";
  if (v >= 40) return "#fbbf24";
  return "#f87171";
}

async function loadOee() {
  const days = document.getElementById("oee-days").value || 7;
  try {
    const d = await authApi(`/api/oee?days=${days}`);
    const o = d.overall;

    const totalEl = document.getElementById("oee-total");
    totalEl.textContent = o.oee + "%";
    totalEl.style.color = oeeColor(o.oee);
    const gradeEl = document.getElementById("oee-grade");
    gradeEl.textContent = t("oee_grade_" + o.grade);
    gradeEl.style.color = oeeColor(o.oee);

    document.getElementById("oee-avail").textContent = o.availability + "%";
    document.getElementById("oee-perf").textContent = o.performance + "%";
    document.getElementById("oee-qual").textContent = o.quality + "%";

    // Downtime by reason
    const reasons = document.getElementById("oee-reasons");
    const maxMin = Math.max(1, ...(d.downtime_by_reason || []).map(r => r.minutes));
    reasons.innerHTML = (d.downtime_by_reason || []).map(r => `
      <div>
        <div class="flex justify-between text-xs mb-1">
          <span>${t("reason_" + r.reason)}</span>
          <span class="gauge-value text-slate-400">${r.minutes} ${t("minutes_short")}</span>
        </div>
        <div style="height:6px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden">
          <div style="height:100%;width:${(r.minutes / maxMin * 100).toFixed(0)}%;background:#fbbf24"></div>
        </div>
      </div>`).join("") || `<div class="text-sm text-slate-400">${t("no_shifts_yet")}</div>`;

    document.getElementById("oee-downtime-cost").innerHTML =
      `<span class="text-slate-400">${t("downtime_cost_label")}:</span> <span class="gauge-value font-semibold" style="color:#f87171">${money(d.downtime_cost)}</span>`;

    // Trend chart (oldest -> newest)
    const shifts = (d.shifts || []).slice().reverse();
    const ctx = document.getElementById("oee-chart");
    if (oeeChart) oeeChart.destroy();
    if (shifts.length && ctx) {
      oeeChart = new Chart(ctx, {
        type: "line",
        data: {
          labels: shifts.map(s => s.shift_name),
          datasets: [{
            label: t("nav_oee"), data: shifts.map(s => s.oee),
            borderColor: "#22d3ee", backgroundColor: "rgba(34,211,238,.12)",
            fill: true, tension: 0.35, pointRadius: 3,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { min: 0, max: 100, ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,.06)" } },
            x: { ticks: { color: "#94a3b8" }, grid: { display: false } },
          },
        },
      });
    }

    // Shift table
    const tbody = document.getElementById("oee-table-body");
    document.getElementById("oee-empty").classList.toggle("hidden", (d.shifts || []).length > 0);
    tbody.innerHTML = (d.shifts || []).map(s => `
      <tr class="border-b border-white/5">
        <td class="py-2 pr-3">${s.shift_name}<div class="text-xs text-slate-500">${(s.date || "").slice(0, 10)}</div></td>
        <td class="py-2 pr-3 gauge-value font-semibold" style="color:${oeeColor(s.oee)}">${s.oee}%</td>
        <td class="py-2 pr-3 gauge-value">${s.availability}%</td>
        <td class="py-2 pr-3 gauge-value">${s.performance}%</td>
        <td class="py-2 pr-3 gauge-value">${s.quality}%</td>
        <td class="py-2 pr-3 gauge-value">${s.downtime_minutes} ${t("minutes_short")}</td>
        <td class="py-2 text-xs">${s.weakest_factor ? t(s.weakest_factor + "_label") : "-"}</td>
      </tr>`).join("");
  } catch (e) {}
}

document.getElementById("oee-days").addEventListener("change", loadOee);

/* --- Log Shift modal --- */
async function openShiftModal() {
  const m = document.getElementById("modal-shift");
  const sel = document.getElementById("sh-machine");
  sel.innerHTML = `<option value="">${t("all_machines_option")}</option>`;
  try {
    const data = await authApi("/api/machines");
    scadaMachines = data.machines;
    scadaMachines.forEach(x => {
      const o = document.createElement("option");
      o.value = x.id; o.textContent = `${x.machine_code} - ${x.machine_name}`;
      sel.appendChild(o);
    });
  } catch (e) {}
  try {
    const r = await authApi("/api/meta/downtime-reasons");
    const rs = document.getElementById("sh-reason");
    rs.innerHTML = `<option value="">${t("reason_unspecified")}</option>`;
    (r.reasons || []).forEach(code => {
      const o = document.createElement("option");
      o.value = code; o.textContent = t("reason_" + code);
      rs.appendChild(o);
    });
  } catch (e) {}
  m.classList.remove("hidden"); m.classList.add("flex");
}
document.getElementById("btn-log-shift").addEventListener("click", openShiftModal);
document.querySelectorAll(".close-shift-modal").forEach(b => b.addEventListener("click", () => {
  const m = document.getElementById("modal-shift");
  m.classList.add("hidden"); m.classList.remove("flex");
}));

document.getElementById("btn-save-shift").addEventListener("click", async () => {
  const total = parseInt(document.getElementById("sh-total").value || 0);
  const good = parseInt(document.getElementById("sh-good").value || 0);
  if (good > total) { showToast(t("err_good_exceeds_total"), "error"); return; }
  const payload = {
    shift_name: document.getElementById("sh-name").value.trim() || "Shift A",
    machine_id: document.getElementById("sh-machine").value || null,
    planned_minutes: parseFloat(document.getElementById("sh-planned").value || 480),
    downtime_minutes: parseFloat(document.getElementById("sh-downtime").value || 0),
    downtime_reason: document.getElementById("sh-reason").value,
    ideal_cycle_seconds: parseFloat(document.getElementById("sh-cycle").value || 30),
    total_units: total, good_units: good,
  };
  try {
    const r = await authApi("/api/shifts", { method: "POST", body: JSON.stringify(payload) });
    const m = document.getElementById("modal-shift");
    m.classList.add("hidden"); m.classList.remove("flex");
    showToast(`${t("nav_oee")}: ${r.oee.oee}%`, "success");
    await loadOee();
  } catch (e) { showToast(t("err_generic"), "error"); }
});

/* ==================== WORK ORDERS ==================== */
const WO_STATUS_COLORS = { open: "#fbbf24", in_progress: "#22d3ee", done: "#34d399", cancelled: "#94a3b8" };

async function loadWorkOrders() {
  try {
    const d = await authApi("/api/work-orders");
    const c = d.counts || {};
    document.getElementById("wo-open").textContent = c.open || 0;
    document.getElementById("wo-progress").textContent = c.in_progress || 0;
    document.getElementById("wo-done").textContent = c.done || 0;
    document.getElementById("wo-avg").textContent = (d.overdue || 0) + "";

    const openTotal = (c.open || 0) + (c.in_progress || 0);
    const badge = document.getElementById("wo-badge");
    if (badge) {
      badge.textContent = openTotal;
      badge.classList.toggle("hidden", !openTotal);
    }

    const list = document.getElementById("wo-list");
    const orders = d.work_orders || [];
    document.getElementById("wo-empty").classList.toggle("hidden", orders.length > 0);
    list.innerHTML = orders.map(w => {
      const color = WO_STATUS_COLORS[w.status] || "#94a3b8";
      const next = w.status === "open" ? "in_progress" : (w.status === "in_progress" ? "done" : null);
      return `
      <div class="glass rounded-xl p-4" style="border-left:3px solid ${color}">
        <div class="flex items-start justify-between gap-3 flex-wrap mb-2">
          <div>
            <div class="font-semibold text-sm">${w.title}</div>
            <div class="text-xs text-slate-400 mt-0.5">${w.description || ""}</div>
          </div>
          <div class="flex items-center gap-2">
            ${w.alert_id ? `<span class="text-xs px-2 py-0.5 rounded-full" style="background:rgba(124,58,237,.2);color:#a78bfa">${t("source_ai")}</span>` : ""}
            <span class="text-xs px-2 py-0.5 rounded-full font-semibold" style="background:${color}22;color:${color}">${t("wo_status_" + w.status)}</span>
          </div>
        </div>
        <div class="text-xs text-slate-400 flex flex-wrap gap-3">
          <span>${t("priority_" + w.priority)}</span>
          ${w.assigned_to ? `<span>${t("assigned_label")}: ${w.assigned_to}</span>` : ""}
          ${w.root_cause ? `<span>${t("alert_reason_label")}: ${t("alertreason_" + w.root_cause)}</span>` : ""}
          ${(w.actions || []).length ? `<span>${w.actions.map(a => t("action_" + a)).join(", ")}</span>` : ""}
        </div>
        ${next ? `<button class="wo-advance mt-3 input-field rounded-lg px-3 py-1.5 text-xs" data-id="${w.id}" data-next="${next}">${t("wo_advance_to_" + next)}</button>` : ""}
      </div>`;
    }).join("");

    list.querySelectorAll(".wo-advance").forEach(btn => {
      btn.addEventListener("click", async () => {
        try {
          await authApi(`/api/work-orders/${btn.dataset.id}`, {
            method: "PUT", body: JSON.stringify({ status: btn.dataset.next }),
          });
          await loadWorkOrders();
        } catch (e) { showToast(t("err_generic"), "error"); }
      });
    });
  } catch (e) {}
}

async function openWoModal() {
  const m = document.getElementById("modal-wo");
  const sel = document.getElementById("wo-machine");
  sel.innerHTML = `<option value="">${t("all_machines_option")}</option>`;
  try {
    const data = await authApi("/api/machines");
    scadaMachines = data.machines;
    scadaMachines.forEach(x => {
      const o = document.createElement("option");
      o.value = x.id; o.textContent = `${x.machine_code} - ${x.machine_name}`;
      sel.appendChild(o);
    });
  } catch (e) {}
  m.classList.remove("hidden"); m.classList.add("flex");
}
document.getElementById("btn-new-wo").addEventListener("click", openWoModal);
document.querySelectorAll(".close-wo-modal").forEach(b => b.addEventListener("click", () => {
  const m = document.getElementById("modal-wo");
  m.classList.add("hidden"); m.classList.remove("flex");
}));

document.getElementById("btn-save-wo").addEventListener("click", async () => {
  const title = document.getElementById("wo-title").value.trim();
  if (!title) { showToast(t("err_missing_fields"), "error"); return; }
  try {
    await authApi("/api/work-orders", {
      method: "POST",
      body: JSON.stringify({
        title,
        machine_id: document.getElementById("wo-machine").value || null,
        priority: document.getElementById("wo-priority").value,
        assigned_to: document.getElementById("wo-assigned").value.trim(),
        description: document.getElementById("wo-desc").value.trim(),
      }),
    });
    const m = document.getElementById("modal-wo");
    m.classList.add("hidden"); m.classList.remove("flex");
    document.getElementById("wo-title").value = "";
    document.getElementById("wo-desc").value = "";
    await loadWorkOrders();
  } catch (e) { showToast(t("err_generic"), "error"); }
});

document.getElementById("btn-run-story").addEventListener("click", async () => {
  const machineId = document.getElementById("story-machine").value;
  if (!machineId) { showToast(t("no_machines_yet"), "error"); return; }

  const spinner = document.getElementById("story-spinner");
  const results = document.getElementById("story-results");
  spinner.classList.remove("hidden");
  results.classList.add("hidden");

  try {
    const res = await authApi("/api/story/simulate", {
      method: "POST",
      body: JSON.stringify({ machine_id: parseInt(machineId) }),
    });

    const o = res.outcome, d = res.detection;
    document.getElementById("story-warning").textContent = o.hours_of_warning + "h";
    document.getElementById("story-loss-ignored").textContent = money(o.loss_if_ignored);
    document.getElementById("story-loss-acted").textContent = money(o.loss_if_acted);
    document.getElementById("story-saved").textContent = money(o.money_saved);

    document.getElementById("story-detection").innerHTML =
      t("story_detection_msg")
        .replace("{hours}", d.hours_before_failure)
        .replace("{risk}", d.risk_at_detection)
        .replace("{cause}", d.root_cause ? t("cause_" + d.root_cause) : "—")
        .replace("{confidence}", Math.round((d.confidence || 0) * 100));

    // Reveal the timeline stage by stage so the story reads like a story.
    const tl = document.getElementById("story-timeline");
    tl.innerHTML = "";
    (res.timeline || []).forEach((stage, i) => tl.appendChild(renderStoryStage(stage, i)));

    results.classList.remove("hidden");
  } catch (e) {
    showToast(t("err_generic"), "error");
  } finally {
    spinner.classList.add("hidden");
  }
});

/* ==================== ROLE-BASED UI ==================== */
function applyRoleVisibility(capabilities) {
  const caps = capabilities || { technical: true, business: true, admin: false };
  document.querySelectorAll("[data-role]").forEach(el => {
    const need = el.getAttribute("data-role");
    const allowed = need === "technical" ? caps.technical : (need === "business" ? caps.business : true);
    el.classList.toggle("hidden", !allowed);
  });
}

document.getElementById("open-machine-modal").addEventListener("click", () => {
  const modal = document.getElementById("modal-machine-scada");
  modal.classList.remove("hidden"); modal.classList.add("flex");
});
document.querySelectorAll(".close-machine-modal").forEach(btn => {
  btn.addEventListener("click", () => {
    const modal = document.getElementById("modal-machine-scada");
    modal.classList.add("hidden"); modal.classList.remove("flex");
  });
});
document.getElementById("scada-machine-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = {
    machine_code: document.getElementById("sm-code").value.trim(),
    machine_name: document.getElementById("sm-name").value.trim(),
    factory_section: document.getElementById("sm-section").value.trim(),
    operator_name: document.getElementById("sm-operator").value.trim(),
    temperature: parseFloat(document.getElementById("sm-temp").value || 0),
    vibration: parseFloat(document.getElementById("sm-vibration").value || 0),
    load: parseFloat(document.getElementById("sm-load").value || 0),
    pressure: parseFloat(document.getElementById("sm-pressure").value || 0),
    voltage: parseFloat(document.getElementById("sm-voltage").value || 0),
    current: parseFloat(document.getElementById("sm-current").value || 0),
    status: document.getElementById("sm-status").value,
    error_code: document.getElementById("sm-error").value.trim(),
    priority_level: document.getElementById("sm-priority").value,
    daily_output_units: parseFloat(document.getElementById("sm-output").value || 0),
    notes: document.getElementById("sm-notes").value,
    lang: currentLang,
  };
  const spinner = document.getElementById("scada-machine-spinner");
  spinner.classList.remove("hidden");
  try {
    await authApi("/api/machine", { method: "POST", body: JSON.stringify(payload) });
    document.getElementById("modal-machine-scada").classList.add("hidden");
    document.getElementById("modal-machine-scada").classList.remove("flex");
    document.getElementById("scada-machine-form").reset();
    showToast(t("toast_analysis_done"), "success");
    await loadScadaMachines();
  } catch (err) {
    showToast(t("err_generic"), "error");
  } finally {
    spinner.classList.add("hidden");
  }
});

/* ============================ ALERTS ============================ */
let lastAlerts = [];

function severityColor(sev) {
  return sev === "critical" ? "#f87171" : "#fbbf24";
}

function updateAlertsBadge(count) {
  const badges = document.querySelectorAll("#alerts-badge");
  badges.forEach(b => {
    if (count > 0) {
      b.textContent = count > 99 ? "99+" : String(count);
      b.classList.remove("hidden");
    } else {
      b.classList.add("hidden");
    }
  });
}

async function loadAlerts() {
  try {
    const data = await authApi("/api/alerts");
    lastAlerts = data.alerts;
    updateAlertsBadge(data.unacknowledged);
    renderAlertsList();
  } catch (e) {}
}

function localizedAlertMessage(a) {
  if (a.temperature === undefined || a.temperature === null) return a.message;
  const statusText = statusLabel(a.status) || a.status;
  return t("alert_details_template")
    .replace("{temp}", a.temperature)
    .replace("{vib}", a.vibration)
    .replace("{status}", statusText);
}

function renderAlertsList() {
  const list = document.getElementById("alerts-list");
  const empty = document.getElementById("alerts-empty");
  list.innerHTML = "";
  empty.classList.toggle("hidden", lastAlerts.length > 0);
  lastAlerts.forEach(a => {
    const card = document.createElement("div");
    card.className = "glass rounded-xl p-4 flex items-start justify-between gap-3 fade-in";
    card.style.borderLeft = `3px solid ${severityColor(a.severity)}`;
    const time = a.created_at ? new Date(a.created_at).toLocaleString() : "";
    // The actions the AI recommends - showing them turns a warning into instructions.
    const actions = (a.suggested_actions || [])
      .filter(x => x && x !== "no_action")
      .map(x => `<span class="text-xs px-2 py-0.5 rounded-full mr-1 mt-1 inline-block" style="background:rgba(255,255,255,.07);color:#cbd5e1">${t("action_" + x)}</span>`)
      .join("");

    card.innerHTML = `
      <div class="min-w-0 flex-1">
        <div class="font-semibold text-sm" style="color:${severityColor(a.severity)}">${(a.machine_name || a.machine_code)} — ${t("priority_" + a.severity) || a.severity.toUpperCase()}</div>
        <div class="text-sm text-slate-300 mt-1">${localizedAlertMessage(a)}</div>
        ${a.alert_type && a.alert_type !== "critical" ? `<div class="text-xs text-slate-400 mt-1">${t("alert_reason_label")}: ${t("alertreason_" + a.alert_type)}</div>` : ""}
        ${actions ? `<div class="mt-1.5">${actions}</div>` : ""}
        <div class="text-xs text-slate-500 mt-1 gauge-value">${time}</div>
      </div>
      <div class="flex flex-col gap-2 items-end whitespace-nowrap">
        ${a.acknowledged
          ? `<span class="text-xs text-slate-500">&#10003; ${t("acknowledged_label")}</span>`
          : `<button class="btn-ack-alert input-field rounded-lg px-3 py-1.5 text-xs" data-id="${a.id}">${t("acknowledge_btn")}</button>`}
        <button class="btn-alert-to-wo input-field rounded-lg px-3 py-1.5 text-xs" data-id="${a.id}">&#128295; ${t("create_work_order_btn")}</button>
      </div>
    `;
    list.appendChild(card);
  });
  list.querySelectorAll(".btn-ack-alert").forEach(btn => {
    btn.addEventListener("click", async () => {
      try {
        await authApi(`/api/alerts/${btn.dataset.id}/acknowledge`, { method: "POST" });
        await loadAlerts();
      } catch (e) {}
    });
  });

  // One click from "the AI found a problem" to "someone is assigned to fix it".
  list.querySelectorAll(".btn-alert-to-wo").forEach(btn => {
    btn.addEventListener("click", async () => {
      const alert = lastAlerts.find(x => String(x.id) === String(btn.dataset.id));
      if (!alert) return;
      btn.disabled = true;
      try {
        await authApi("/api/work-orders", {
          method: "POST",
          body: JSON.stringify({
            alert_id: alert.id,
            machine_id: alert.machine_id,
            title: `${alert.machine_name || alert.machine_code} — ${t("alertreason_" + alert.alert_type)}`.trim(),
            description: localizedAlertMessage(alert),
            priority: alert.severity,
          }),
        });
        showToast(t("work_order_created_msg"), "success");
        await loadWorkOrders();
      } catch (e) {
        showToast(t("err_generic"), "error");
      } finally {
        btn.disabled = false;
      }
    });
  });
}

document.getElementById("btn-ack-all").addEventListener("click", async () => {
  try {
    await authApi("/api/alerts/acknowledge_all", { method: "POST" });
    await loadAlerts();
  } catch (e) {}
});

function handleIncomingCriticalAlert(alert) {
  const localizedMsg = localizedAlertMessage(alert);
  showToast(`🚨 ${alert.machine_name || alert.machine_code}: ${localizedMsg}`, "error");
  if (window.Notification && Notification.permission === "granted") {
    try {
      new Notification(`FactoryPulse AI — ${t("priority_" + (alert.severity || "critical"))}`, { body: localizedMsg, tag: "fp-alert-" + alert.id });
    } catch (e) {}
  }
  const badge = document.getElementById("alerts-badge");
  const current = parseInt(badge.textContent) || 0;
  updateAlertsBadge(badge.classList.contains("hidden") ? 1 : current + 1);
  if (!document.getElementById("page-alerts").classList.contains("hidden")) {
    lastAlerts.unshift(alert);
    renderAlertsList();
  }
}

if (window.Notification && Notification.permission === "default") {
  Notification.requestPermission();
}

/* ============================ PDF REPORT ============================ */
document.getElementById("btn-download-report").addEventListener("click", async () => {
  const btn = document.getElementById("btn-download-report");
  const originalHtml = btn.innerHTML;
  btn.innerHTML = `<span class="spinner"></span>`;
  try {
    // Read the language straight from the selector rather than trusting a
    // module variable that an older cached script may not have set.
    const sel = document.getElementById("lang-select");
    const lang = (sel && sel.value) || currentLang || localStorage.getItem("fp_lang") || "en";

    // cache-bust so a stale PDF is never re-served by the browser
    const url = `/api/report/pdf?days=30&lang=${encodeURIComponent(lang)}&_=${Date.now()}`;
    const res = await fetch(url, {
      headers: { "Authorization": "Bearer " + authToken },
      cache: "no-store",
    });
    if (!res.ok) {
      let errCode = "";
      try { errCode = (await res.json()).error; } catch (e2) {}
      if (errCode === "pdf_unavailable") {
        showToast(t("report_lib_missing_msg"), "error");
      } else {
        showToast(t("err_generic"), "error");
      }
      return;
    }
    const blob = await res.blob();
    const objUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = objUrl;
    a.download = `FactoryPulseAI_Report_${lang}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(objUrl);
  } catch (e) {
    showToast(t("err_generic"), "error");
  } finally {
    btn.innerHTML = originalHtml;
  }
});

async function loadAiFeed() {
  try {
    const data = await authApi("/api/factories");
    userFactories = data.factories;
    renderAiFeed();
  } catch (e) {}
}

function renderAiFeed() {
  const feed = document.getElementById("ai-feed");
  const empty = document.getElementById("ai-feed-empty");
  const withInsights = userFactories.filter(f => f.ai_insights);
  feed.innerHTML = "";
  empty.classList.toggle("hidden", withInsights.length > 0);
  withInsights.forEach(f => {
    const card = document.createElement("div");
    card.className = "glass rounded-2xl p-5 fade-in";
    const a = f.ai_insights;
    card.innerHTML = `
      <div class="flex items-center justify-between mb-3">
        <div class="font-semibold">${f.factory_name}</div>
        <button class="btn-reanalyze input-field rounded-lg px-3 py-1.5 text-xs" data-id="${f.id}">${t("reanalyze_btn")}</button>
      </div>
      <div class="flex flex-col gap-3">
        <div><div class="text-xs uppercase text-red-400 ai-section-title mb-1">${t("ai_risks")}</div><div class="text-sm text-slate-200 leading-relaxed">${a.risks}</div></div>
        <div><div class="text-xs uppercase text-cyan-400 ai-section-title mb-1">${t("ai_efficiency_insights")}</div><div class="text-sm text-slate-200 leading-relaxed">${a.efficiency_insights}</div></div>
        <div><div class="text-xs uppercase text-emerald-400 ai-section-title mb-1">${t("ai_optimizations")}</div><div class="text-sm text-slate-200 leading-relaxed">${a.optimizations}</div></div>
      </div>
    `;
    feed.appendChild(card);
  });
  feed.querySelectorAll(".btn-reanalyze").forEach(btn => {
    btn.addEventListener("click", async () => {
      const factory = userFactories.find(x => x.id == btn.dataset.id);
      if (!factory) return;
      btn.textContent = t("ai_analyzing");
      try {
        await authApi(`/api/factories/${factory.id}`, {
          method: "PUT",
          body: JSON.stringify({
            factory_name: factory.factory_name, machine_count: factory.machines, energy_cost: factory.energy_cost,
            machine_type: factory.machine_type, temperature: factory.temperature, vibration: factory.vibration, load: factory.load,
            lang: currentLang,
          }),
        });
        showToast(t("toast_analysis_done"), "success");
        await loadAiFeed();
      } catch (e) {
        showToast(t("err_generic"), "error");
        btn.textContent = t("reanalyze_btn");
      }
    });
  });
}

/* ============================ INIT ============================ */
document.addEventListener("contextmenu", (e) => e.preventDefault());

document.addEventListener("DOMContentLoaded", async () => {
  const ok = await requireAuthOrRedirect();
  if (!ok) return;

  buildLangSelector();
  buildMachineTypeSelect();
  applyTranslations();
  initChart();
  try {
    const meta = await api("/api/meta");
    if (meta.machine_types) { MACHINE_TYPES = meta.machine_types; buildMachineTypeSelect(); }
  } catch (e) {}
  // Restore this user's previously saved factory input instead of hardcoded
  // demo values, so their data is still there after logging back in.
  try {
    const saved = await authApi("/api/data");
    const s = saved.state || {};
    document.getElementById("f-name").value = s.factory_name ?? "Demo Factory";
    document.getElementById("f-count").value = s.machine_count ?? 6;
    document.getElementById("f-cost").value = s.energy_cost ?? 0.12;
    document.getElementById("f-temp").value = s.temperature ?? 65;
    document.getElementById("f-vibration").value = s.vibration ?? 3.5;
    document.getElementById("f-load").value = s.load ?? 60;
    if (s.machine_type) document.getElementById("f-type").value = s.machine_type;
  } catch (e) {
    document.getElementById("f-name").value = "Demo Factory";
    document.getElementById("f-count").value = 6;
    document.getElementById("f-cost").value = 0.12;
    document.getElementById("f-temp").value = 65;
    document.getElementById("f-vibration").value = 3.5;
    document.getElementById("f-load").value = 60;
  }
  await refreshData();
  setInterval(refreshData, 2000);

  // Connect the real-time socket app-wide (not just when visiting Live Monitor),
  // so critical alerts arrive no matter which page the user is currently on.
  if (!liveMonitorInitialized) {
    liveMonitorInitialized = true;
    connectScadaSocket();
  }
  loadAlerts();
});
</script>
</body>
</html>
"""


LOGIN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>FactoryPulse AI - Industrial AI & SCADA Platform | Real-Time Factory Monitoring</title>
<meta name="description" content="FactoryPulse AI is a global industrial intelligence platform with real-time SCADA monitoring, AI-powered predictive maintenance, energy analytics, and a 3D digital twin. Connect via USB, PLC, Modbus, OPC UA, or MQTT." />
<meta name="keywords" content="FactoryPulse AI, industrial AI, SCADA platform, predictive maintenance, digital twin, factory monitoring, PLC integration, energy analytics, IoT manufacturing" />
<meta name="robots" content="index, follow" />
<link rel="canonical" href="/login" />
<meta property="og:title" content="FactoryPulse AI - Industrial AI & SCADA Platform" />
<meta property="og:description" content="Real-time factory monitoring, AI predictive maintenance, and a 3D digital twin for modern manufacturing." />
<meta property="og:type" content="website" />
<meta property="og:site_name" content="FactoryPulse AI" />
<meta name="twitter:card" content="summary" />
<meta name="twitter:title" content="FactoryPulse AI - Industrial AI & SCADA Platform" />
<meta name="twitter:description" content="Real-time factory monitoring, AI predictive maintenance, and a 3D digital twin for modern manufacturing." />
<script src="https://cdn.tailwindcss.com"></script>
<style>
  html, body { height: 100%; }
  body {
    min-height: 100vh;
    background: radial-gradient(circle at 15% 0%, #1a1f3a 0%, #0a0e1f 45%, #030409 100%);
    font-family: 'Segoe UI', system-ui, sans-serif;
    color: #e6edf5;
    position: relative;
    overflow-x: hidden;
  }
  .blob { position: fixed; border-radius: 9999px; filter: blur(100px); opacity: .25; pointer-events: none; animation: floatBlob 16s ease-in-out infinite; z-index: 0; }
  @keyframes floatBlob { 0%,100% { transform: translate(0,0); } 50% { transform: translate(40px,-30px); } }
  .glass { background: rgba(255,255,255,0.045); border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); position: relative; z-index: 1; }
  .glass-strong { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.14); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); }
  .fade-in { animation: fadeIn .5s ease both; }
  @keyframes fadeIn { from { opacity:0; transform: translateY(10px);} to { opacity:1; transform:none; } }
  .glow-btn { background: linear-gradient(135deg, #06b6d4, #7c3aed, #3b82f6); background-size: 200% 200%; transition: box-shadow .25s ease, transform .15s ease; animation: gradientShift 6s ease infinite; }
  @keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
  .glow-btn:hover { box-shadow: 0 0 32px rgba(34,211,238,.5); transform: translateY(-1px); }
  .glow-btn:active { transform: translateY(0) scale(.98); }
  .neon-text { text-shadow: 0 0 18px rgba(34,211,238,.5); }
  .input-field { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12); transition: border-color .2s ease, box-shadow .2s ease, background .2s ease; }
  .input-field:focus { outline: none; border-color: #22d3ee; background: rgba(255,255,255,0.08); box-shadow: 0 0 0 4px rgba(34,211,238,.15), 0 0 24px rgba(34,211,238,.2); }
  .status-running { color: #34d399; }
  .status-warning { color: #fbbf24; }
  .status-critical { color: #f87171; }
  .bg-status-running { background: #34d399; }
  .bg-status-warning { background: #fbbf24; }
  .bg-status-critical { background: #f87171; }
  .pulse-dot { position: relative; display: inline-flex; }
  .pulse-dot::before { content: ""; position: absolute; inset: -5px; border-radius: 9999px; border: 1px solid currentColor; opacity: .5; animation: pulseRing 2s ease-out infinite; }
  @keyframes pulseRing { 0%{transform:scale(.7); opacity:.6} 100%{transform:scale(2); opacity:0} }
  .spinner { width: 18px; height: 18px; border-radius: 50%; border: 2.5px solid rgba(255,255,255,0.25); border-top-color: #22d3ee; animation: spin .7s linear infinite; display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .machine-card { transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease; }
  .machine-card:hover { transform: translateY(-2px); border-color: rgba(34,211,238,.35); box-shadow: 0 12px 30px rgba(0,0,0,.4); }
  ::-webkit-scrollbar { width: 8px; }
  ::-webkit-scrollbar-thumb { background: rgba(148,163,184,.4); border-radius: 8px; }
  select { color: #e2e8f0; color-scheme: dark; }
  select option { color: #0f172a; background: #ffffff; }
  .gauge-value { font-family: 'Consolas', monospace; }
  .fl-label { transition: all .18s ease; }
  .toast { animation: toastIn .25s ease both; }
  @keyframes toastIn { from { opacity:0; transform: translateY(-8px);} to { opacity:1; transform:none; } }
  .ai-section-title { letter-spacing: .04em; }
  .nav-btn.active, .nav-btn-m.active { background: rgba(34,211,238,0.16); color: #22d3ee; }
  .factory-card { transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease; }
  .factory-card:hover { transform: translateY(-2px); border-color: rgba(34,211,238,.35); box-shadow: 0 12px 30px rgba(0,0,0,.4); }
  .anim-error { animation: shakeIn .35s ease both; }
  @keyframes shakeIn { 0%{opacity:0; transform:translateX(-6px);} 60%{transform:translateX(3px);} 100%{opacity:1; transform:none;} }
</style>
</head>
<body>

<div class="blob" style="width:420px;height:420px;background:#0891b2;top:-10%;left:5%"></div>
<div class="blob" style="width:380px;height:380px;background:#7c3aed;bottom:-14%;right:0%"></div>

<div id="toast-container" class="fixed top-4 right-4 z-50 flex flex-col gap-2"></div>

<div class="relative z-10 min-h-screen flex items-center justify-center p-4">
  <div class="w-full max-w-md fade-in">
    <div class="flex items-center justify-center gap-3 mb-6">
      <div class="w-12 h-12 rounded-2xl flex items-center justify-center glow-btn shrink-0">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2-7 4 14 2-7h6"/></svg>
      </div>
      <div class="text-left">
        <div class="font-bold text-xl tracking-tight neon-text">FactoryPulse<span class="text-cyan-400">AI</span></div>
        <div class="text-xs text-slate-400" data-t="tagline">Global Industrial Intelligence Platform</div>
      </div>
    </div>

    <div class="glass-strong rounded-3xl p-8">
      <div class="flex justify-center mb-6">
        <select id="lang-select" class="input-field rounded-xl text-xs px-3 py-1.5 outline-none"></select>
      </div>

      <h1 class="text-lg font-semibold mb-1" data-t="login_title">Welcome back</h1>
      <p class="text-sm text-slate-400 mb-6" data-t="login_subtitle">Sign in to your FactoryPulse AI account</p>

      <form id="login-form" class="flex flex-col gap-4">
        <div class="relative">
          <input id="email" type="email" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2" />
          <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="ph_email">Email</label>
        </div>
        <div class="relative">
          <input id="password" type="password" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2 pr-10" />
          <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="ph_password">Password</label>
          <button type="button" class="toggle-eye absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200" data-target="password"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7Z"/><circle cx="12" cy="12" r="3"/></svg></button>
        </div>
        <div class="flex items-center justify-between">
          <label class="flex items-center gap-2 text-xs text-slate-400 select-none cursor-pointer">
            <input id="remember" type="checkbox" class="w-4 h-4 rounded accent-cyan-400" />
            <span data-t="remember_me">Remember me</span>
          </label>
          <button type="button" id="show-forgot-password" class="text-xs text-cyan-400 hover:text-cyan-300 font-medium" data-t="forgot_password_link">Forgot password?</button>
        </div>
        <div id="form-error" class="hidden anim-error text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded-xl px-3 py-2"></div>
        <button type="submit" id="submit-btn" class="glow-btn rounded-xl py-3 text-sm font-semibold flex items-center justify-center gap-2 mt-1">
          <span id="submit-spinner" class="spinner hidden"></span>
          <span id="submit-label" data-t="login_btn">Log In</span>
        </button>
        <p class="text-xs text-slate-400 text-center mt-1">
          <a href="/register" class="text-cyan-400 hover:text-cyan-300 font-medium" data-t="login_link_register">Don't have an account? Create one</a>
        </p>
      </form>

      <form id="forgot-password-form" class="hidden flex-col gap-4">
        <p class="text-xs text-slate-400" data-t="forgot_password_subtitle">Enter your email and we'll send you a verification code.</p>
        <div class="relative">
          <input id="forgot-email" type="email" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2" />
          <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="ph_email">Email</label>
        </div>
        <div id="forgot-error" class="hidden anim-error text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded-xl px-3 py-2"></div>
        <button type="submit" id="forgot-submit-btn" class="glow-btn rounded-xl py-3 text-sm font-semibold flex items-center justify-center gap-2 mt-1">
          <span id="forgot-submit-spinner" class="spinner hidden"></span>
          <span data-t="send_code_btn">Send Code</span>
        </button>
        <p class="text-xs text-slate-400 text-center mt-1">
          <button type="button" id="show-login-form" class="text-cyan-400 hover:text-cyan-300 font-medium" data-t="back_to_login_link">Back to login</button>
        </p>
      </form>

      <form id="verify-code-form" class="hidden flex-col gap-4">
        <p class="text-xs text-slate-400" data-t="enter_code_subtitle">Enter the 6-digit code we sent to your email.</p>
        <div id="dev-code-hint" class="hidden text-xs text-amber-400 bg-amber-500/10 border border-amber-500/30 rounded-xl px-3 py-2"></div>
        <div class="relative">
          <input id="reset-code" type="text" inputmode="numeric" maxlength="6" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2 tracking-[0.3em] text-center font-mono" />
          <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="ph_code">Verification Code</label>
        </div>
        <div id="verify-error" class="hidden anim-error text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded-xl px-3 py-2"></div>
        <button type="submit" id="verify-submit-btn" class="glow-btn rounded-xl py-3 text-sm font-semibold flex items-center justify-center gap-2 mt-1">
          <span id="verify-submit-spinner" class="spinner hidden"></span>
          <span data-t="verify_code_btn">Verify Code</span>
        </button>
        <p class="text-xs text-slate-400 text-center mt-1">
          <button type="button" id="resend-code-btn" class="text-cyan-400 hover:text-cyan-300 font-medium" data-t="resend_code_link">Resend code</button>
        </p>
      </form>

      <form id="new-password-form" class="hidden flex-col gap-4">
        <p class="text-xs text-slate-400" data-t="set_new_password_subtitle">Choose a new password for your account.</p>
        <div class="relative">
          <input id="new-password" type="password" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2 pr-10" />
          <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="ph_new_password">New Password</label>
          <button type="button" class="toggle-eye absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200" data-target="new-password"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7Z"/><circle cx="12" cy="12" r="3"/></svg></button>
        </div>
        <div class="relative">
          <input id="new-password-confirm" type="password" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2 pr-10" />
          <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="ph_confirm_password">Confirm Password</label>
          <button type="button" class="toggle-eye absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200" data-target="new-password-confirm"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7Z"/><circle cx="12" cy="12" r="3"/></svg></button>
        </div>
        <div id="newpass-error" class="hidden anim-error text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded-xl px-3 py-2"></div>
        <button type="submit" id="newpass-submit-btn" class="glow-btn rounded-xl py-3 text-sm font-semibold flex items-center justify-center gap-2 mt-1">
          <span id="newpass-submit-spinner" class="spinner hidden"></span>
          <span data-t="set_password_btn">Set New Password</span>
        </button>
      </form>
    </div>
  </div>
</div>

<script>
const translations = {
  en: {tagline:"Global Industrial Intelligence Platform",live_label:"Live",kpi_energy:"Energy Usage",kpi_efficiency:"Efficiency",kpi_active:"Active Machines",kpi_alerts:"Alerts",kwh_unit:"kWh",chart_title:"Real-Time Performance",machine_status_title:"Machine Status",status_running:"Running",status_warning:"Warning",status_critical:"Critical",form_title:"Factory Data Input",factory_name_label:"Factory Name",machine_count_label:"Number of Machines",energy_cost_label:"Energy Cost ($/kWh)",machine_type_label:"Machine Type",temperature_label:"Temperature (°C)",vibration_label:"Vibration (mm/s)",load_label:"Load (%)",submit_btn:"Analyze Factory",submitting:"Updating...",ai_panel_title:"AI Insights",ai_placeholder:"Submit factory data to generate an AI analysis.",ai_analyzing:"Analyzing...",ai_risks:"Risks",ai_efficiency_insights:"Efficiency Insights",ai_optimizations:"Optimization Suggestions",toast_updated:"Factory data updated",toast_analysis_done:"AI analysis complete",toast_error:"Something went wrong",nav_dashboard:"Dashboard",nav_factories:"Factories",nav_ai_insights:"AI Insights",logout_btn:"Log Out",login_title:"Welcome back",login_subtitle:"Sign in to your FactoryPulse AI account",ph_email:"Email",ph_password:"Password",remember_me:"Remember me",login_btn:"Log In",login_link_register:"Don't have an account? Create one",register_title:"Create your account",register_subtitle:"Start monitoring your factories with AI",ph_full_name:"Full Name",ph_confirm_password:"Confirm Password",register_btn:"Create Account",register_link_login:"Already have an account? Sign in",err_missing_fields:"Please fill in all fields",err_invalid_email:"Please enter a valid email address",err_weak_password:"Password must be at least 8 characters with a letter and a number",err_password_mismatch:"Passwords do not match",err_invalid_credentials:"Invalid email or password",err_email_taken:"This email is already registered",err_generic:"Something went wrong. Please try again",my_factories_title:"My Factories",add_factory_btn:"+ Add Factory",edit_factory_btn:"Edit",delete_factory_btn:"Delete",confirm_delete_factory:"Delete this factory? This cannot be undone.",no_factories_yet:"You haven't added any factories yet.",factory_created_toast:"Factory created and analyzed",factory_updated_toast:"Factory updated",factory_deleted_toast:"Factory deleted",ai_insights_feed_title:"AI Insights Feed",no_ai_insights_yet:"No AI insights yet. Add a factory to get started.",reanalyze_btn:"Re-analyze",view_insights_btn:"View Insights",created_label:"Created",cancel_btn:"Cancel",save_btn:"Save Changes",forgot_password_link:"Forgot password?",forgot_password_subtitle:"Enter your email and we'll send you a verification code.",send_code_btn:"Send Code",back_to_login_link:"Back to login",enter_code_subtitle:"Enter the 6-digit code we sent to your email.",dev_code_hint_msg:"Email not configured on the server - your code is: {code}",ph_code:"Verification Code",verify_code_btn:"Verify Code",resend_code_link:"Resend code",err_invalid_or_expired_code:"Invalid or expired code",set_new_password_subtitle:"Choose a new password for your account.",ph_new_password:"New Password",set_password_btn:"Set New Password",password_reset_success_msg:"Password reset successfully. You can now log in.",err_email_send_failed:"Could not send the email. Please contact your administrator."},
  ru: {tagline:"Глобальная платформа промышленного интеллекта",live_label:"Live",kpi_energy:"Потребление энергии",kpi_efficiency:"Эффективность",kpi_active:"Активные станки",kpi_alerts:"Оповещения",kwh_unit:"кВт·ч",chart_title:"Показатели в реальном времени",machine_status_title:"Статус станков",status_running:"Работает",status_warning:"Внимание",status_critical:"Критично",form_title:"Ввод данных завода",factory_name_label:"Название завода",machine_count_label:"Количество станков",energy_cost_label:"Стоимость энергии ($/кВт·ч)",machine_type_label:"Тип станка",temperature_label:"Температура (°C)",vibration_label:"Вибрация (мм/с)",load_label:"Нагрузка (%)",submit_btn:"Анализировать завод",submitting:"Обновление...",ai_panel_title:"AI-аналитика",ai_placeholder:"Отправьте данные завода, чтобы получить AI-анализ.",ai_analyzing:"Анализ...",ai_risks:"Риски",ai_efficiency_insights:"Анализ эффективности",ai_optimizations:"Рекомендации по оптимизации",toast_updated:"Данные завода обновлены",toast_analysis_done:"AI-анализ завершён",toast_error:"Произошла ошибка",nav_dashboard:"Панель",nav_factories:"Заводы",nav_ai_insights:"AI-аналитика",logout_btn:"Выход",login_title:"С возвращением",login_subtitle:"Войдите в аккаунт FactoryPulse AI",ph_email:"Email",ph_password:"Пароль",remember_me:"Запомнить меня",login_btn:"Войти",login_link_register:"Нет аккаунта? Создать",register_title:"Создать аккаунт",register_subtitle:"Начните мониторинг заводов с помощью AI",ph_full_name:"Полное имя",ph_confirm_password:"Подтвердите пароль",register_btn:"Создать аккаунт",register_link_login:"Уже есть аккаунт? Войти",err_missing_fields:"Заполните все поля",err_invalid_email:"Введите корректный email",err_weak_password:"Пароль должен быть от 8 символов, с буквой и цифрой",err_password_mismatch:"Пароли не совпадают",err_invalid_credentials:"Неверный email или пароль",err_email_taken:"Этот email уже зарегистрирован",err_generic:"Что-то пошло не так. Попробуйте снова",my_factories_title:"Мои заводы",add_factory_btn:"+ Добавить завод",edit_factory_btn:"Изменить",delete_factory_btn:"Удалить",confirm_delete_factory:"Удалить этот завод? Это действие нельзя отменить.",no_factories_yet:"Вы ещё не добавили ни одного завода.",factory_created_toast:"Завод создан и проанализирован",factory_updated_toast:"Завод обновлён",factory_deleted_toast:"Завод удалён",ai_insights_feed_title:"Лента AI-аналитики",no_ai_insights_yet:"Пока нет AI-аналитики. Добавьте завод, чтобы начать.",reanalyze_btn:"Проанализировать снова",view_insights_btn:"Смотреть аналитику",created_label:"Создано",cancel_btn:"Отмена",save_btn:"Сохранить изменения",forgot_password_link:"Забыли пароль?",forgot_password_subtitle:"Введите email, и мы отправим вам код подтверждения.",send_code_btn:"Отправить код",back_to_login_link:"Назад ко входу",enter_code_subtitle:"Введите 6-значный код, отправленный на вашу почту.",dev_code_hint_msg:"Почта не настроена на сервере — ваш код: {code}",ph_code:"Код подтверждения",verify_code_btn:"Подтвердить код",resend_code_link:"Отправить код повторно",err_invalid_or_expired_code:"Неверный или истёкший код",set_new_password_subtitle:"Выберите новый пароль для аккаунта.",ph_new_password:"Новый пароль",set_password_btn:"Установить новый пароль",password_reset_success_msg:"Пароль успешно изменён. Теперь вы можете войти.",err_email_send_failed:"Не удалось отправить письмо. Обратитесь к администратору."},
  kk: {tagline:"Жаһандық өнеркәсіптік интеллект платформасы",live_label:"Тікелей эфир",kpi_energy:"Энергия тұтыну",kpi_efficiency:"Тиімділік",kpi_active:"Белсенді станоктар",kpi_alerts:"Дабылдар",kwh_unit:"кВт·сағ",chart_title:"Нақты уақыттағы көрсеткіштер",machine_status_title:"Станоктар күйі",status_running:"Жұмыс істеп тұр",status_warning:"Ескерту",status_critical:"Сыни",form_title:"Зауыт деректерін енгізу",factory_name_label:"Зауыт атауы",machine_count_label:"Станоктар саны",energy_cost_label:"Энергия құны ($/кВт·сағ)",machine_type_label:"Станок түрі",temperature_label:"Температура (°C)",vibration_label:"Діріл (мм/с)",load_label:"Жүктеме (%)",submit_btn:"Зауытты талдау",submitting:"Жаңартылуда...",ai_panel_title:"AI-талдау",ai_placeholder:"AI-талдау алу үшін зауыт деректерін жіберіңіз.",ai_analyzing:"Талдануда...",ai_risks:"Тәуекелдер",ai_efficiency_insights:"Тиімділік талдауы",ai_optimizations:"Оңтайландыру ұсыныстары",toast_updated:"Зауыт деректері жаңартылды",toast_analysis_done:"AI-талдау аяқталды",toast_error:"Қате орын алды",nav_dashboard:"Басқару тақтасы",nav_factories:"Зауыттар",nav_ai_insights:"AI-талдау",logout_btn:"Шығу",login_title:"Қайта қош келдіңіз",login_subtitle:"FactoryPulse AI аккаунтыңызға кіріңіз",ph_email:"Email",ph_password:"Құпия сөз",remember_me:"Мені есте сақтау",login_btn:"Кіру",login_link_register:"Аккаунтыңыз жоқ па? Тіркелу",register_title:"Аккаунт құру",register_subtitle:"Зауыттарды AI арқылы бақылауды бастаңыз",ph_full_name:"Толық аты-жөні",ph_confirm_password:"Құпия сөзді қайталаңыз",register_btn:"Аккаунт құру",register_link_login:"Аккаунтыңыз бар ма? Кіру",err_missing_fields:"Барлық өрістерді толтырыңыз",err_invalid_email:"Дұрыс email мекенжайын енгізіңіз",err_weak_password:"Құпия сөз кемінде 8 таңба, әріп пен сан болуы керек",err_password_mismatch:"Құпия сөздер сәйкес келмейді",err_invalid_credentials:"Қате email немесе құпия сөз",err_email_taken:"Бұл email тіркелген",err_generic:"Қате орын алды. Қайталап көріңіз",my_factories_title:"Менің зауыттарым",add_factory_btn:"+ Зауыт қосу",edit_factory_btn:"Өзгерту",delete_factory_btn:"Жою",confirm_delete_factory:"Бұл зауытты жоясыз ба? Бұл әрекетті кері қайтару мүмкін емес.",no_factories_yet:"Сіз әлі ешбір зауыт қосқан жоқсыз.",factory_created_toast:"Зауыт құрылды және талданды",factory_updated_toast:"Зауыт жаңартылды",factory_deleted_toast:"Зауыт жойылды",ai_insights_feed_title:"AI-талдау таспасы",no_ai_insights_yet:"AI-талдау әлі жоқ. Бастау үшін зауыт қосыңыз.",reanalyze_btn:"Қайта талдау",view_insights_btn:"Талдауды көру",created_label:"Құрылған күні",cancel_btn:"Бас тарту",save_btn:"Өзгерістерді сақтау",forgot_password_link:"Құпия сөзді ұмыттыңыз ба?",forgot_password_subtitle:"Email енгізіңіз, сізге растау коды жіберіледі.",send_code_btn:"Код жіберу",back_to_login_link:"Кіруге оралу",enter_code_subtitle:"Поштаңызға жіберілген 6 таңбалы кодты енгізіңіз.",dev_code_hint_msg:"Серверде пошта баптаулмаған — сіздің кодыңыз: {code}",ph_code:"Растау коды",verify_code_btn:"Кодты растау",resend_code_link:"Кодты қайта жіберу",err_invalid_or_expired_code:"Код қате немесе мерзімі өтіп кеткен",set_new_password_subtitle:"Аккаунтыңызға жаңа құпия сөз таңдаңыз.",ph_new_password:"Жаңа құпия сөз",set_password_btn:"Жаңа құпия сөзді орнату",password_reset_success_msg:"Құпия сөз сәтті өзгертілді. Енді кіре аласыз.",err_email_send_failed:"Хат жіберілмеді. Әкімшіге хабарласыңыз."},
  de: {tagline:"Globale Industrielle Intelligenzplattform",live_label:"Live",kpi_energy:"Energieverbrauch",kpi_efficiency:"Effizienz",kpi_active:"Aktive Maschinen",kpi_alerts:"Warnungen",kwh_unit:"kWh",chart_title:"Echtzeit-Leistung",machine_status_title:"Maschinenstatus",status_running:"Läuft",status_warning:"Warnung",status_critical:"Kritisch",form_title:"Fabrikdateneingabe",factory_name_label:"Fabrikname",machine_count_label:"Anzahl der Maschinen",energy_cost_label:"Energiekosten ($/kWh)",machine_type_label:"Maschinentyp",temperature_label:"Temperatur (°C)",vibration_label:"Vibration (mm/s)",load_label:"Last (%)",submit_btn:"Fabrik Analysieren",submitting:"Aktualisieren...",ai_panel_title:"KI-Einblicke",ai_placeholder:"Senden Sie Fabrikdaten, um eine KI-Analyse zu erstellen.",ai_analyzing:"Analysiere...",ai_risks:"Risiken",ai_efficiency_insights:"Effizienzanalyse",ai_optimizations:"Optimierungsvorschläge",toast_updated:"Fabrikdaten aktualisiert",toast_analysis_done:"KI-Analyse abgeschlossen",toast_error:"Etwas ist schiefgelaufen",nav_dashboard:"Übersicht",nav_factories:"Fabriken",nav_ai_insights:"KI-Einblicke",logout_btn:"Abmelden",login_title:"Willkommen zurück",login_subtitle:"Melden Sie sich bei Ihrem FactoryPulse AI-Konto an",ph_email:"E-Mail",ph_password:"Passwort",remember_me:"Angemeldet bleiben",login_btn:"Einloggen",login_link_register:"Kein Konto? Jetzt erstellen",register_title:"Konto erstellen",register_subtitle:"Beginnen Sie mit der KI-Überwachung Ihrer Fabriken",ph_full_name:"Vollständiger Name",ph_confirm_password:"Passwort bestätigen",register_btn:"Konto erstellen",register_link_login:"Bereits ein Konto? Anmelden",err_missing_fields:"Bitte füllen Sie alle Felder aus",err_invalid_email:"Bitte geben Sie eine gültige E-Mail-Adresse ein",err_weak_password:"Passwort muss mind. 8 Zeichen, einen Buchstaben und eine Zahl enthalten",err_password_mismatch:"Passwörter stimmen nicht überein",err_invalid_credentials:"Ungültige E-Mail oder Passwort",err_email_taken:"Diese E-Mail ist bereits registriert",err_generic:"Etwas ist schiefgelaufen. Bitte erneut versuchen",my_factories_title:"Meine Fabriken",add_factory_btn:"+ Fabrik Hinzufügen",edit_factory_btn:"Bearbeiten",delete_factory_btn:"Löschen",confirm_delete_factory:"Diese Fabrik löschen? Dies kann nicht rückgängig gemacht werden.",no_factories_yet:"Sie haben noch keine Fabriken hinzugefügt.",factory_created_toast:"Fabrik erstellt und analysiert",factory_updated_toast:"Fabrik aktualisiert",factory_deleted_toast:"Fabrik gelöscht",ai_insights_feed_title:"KI-Einblicke Feed",no_ai_insights_yet:"Noch keine KI-Einblicke. Fügen Sie eine Fabrik hinzu.",reanalyze_btn:"Erneut analysieren",view_insights_btn:"Einblicke Anzeigen",created_label:"Erstellt",cancel_btn:"Abbrechen",save_btn:"Änderungen Speichern",forgot_password_link:"Passwort vergessen?",forgot_password_subtitle:"Geben Sie Ihre E-Mail ein und wir senden Ihnen einen Bestätigungscode.",send_code_btn:"Code senden",back_to_login_link:"Zurück zur Anmeldung",enter_code_subtitle:"Geben Sie den 6-stelligen Code ein, den wir an Ihre E-Mail gesendet haben.",dev_code_hint_msg:"E-Mail auf dem Server nicht konfiguriert - Ihr Code lautet: {code}",ph_code:"Bestätigungscode",verify_code_btn:"Code bestätigen",resend_code_link:"Code erneut senden",err_invalid_or_expired_code:"Ungültiger oder abgelaufener Code",set_new_password_subtitle:"Wählen Sie ein neues Passwort für Ihr Konto.",ph_new_password:"Neues Passwort",set_password_btn:"Neues Passwort Festlegen",password_reset_success_msg:"Passwort erfolgreich zurückgesetzt. Sie können sich jetzt anmelden.",err_email_send_failed:"E-Mail konnte nicht gesendet werden. Bitte wenden Sie sich an Ihren Administrator."},
  fr: {tagline:"Plateforme mondiale d'intelligence industrielle",live_label:"En direct",kpi_energy:"Consommation d'Énergie",kpi_efficiency:"Efficacité",kpi_active:"Machines Actives",kpi_alerts:"Alertes",kwh_unit:"kWh",chart_title:"Performance en Temps Réel",machine_status_title:"État des Machines",status_running:"En marche",status_warning:"Avertissement",status_critical:"Critique",form_title:"Saisie des Données d'Usine",factory_name_label:"Nom de l'Usine",machine_count_label:"Nombre de Machines",energy_cost_label:"Coût de l'Énergie ($/kWh)",machine_type_label:"Type de Machine",temperature_label:"Température (°C)",vibration_label:"Vibration (mm/s)",load_label:"Charge (%)",submit_btn:"Analyser l'Usine",submitting:"Mise à jour...",ai_panel_title:"Analyses IA",ai_placeholder:"Envoyez les données de l'usine pour générer une analyse IA.",ai_analyzing:"Analyse en cours...",ai_risks:"Risques",ai_efficiency_insights:"Analyse d'Efficacité",ai_optimizations:"Suggestions d'Optimisation",toast_updated:"Données d'usine mises à jour",toast_analysis_done:"Analyse IA terminée",toast_error:"Une erreur est survenue",nav_dashboard:"Tableau de Bord",nav_factories:"Usines",nav_ai_insights:"Analyses IA",logout_btn:"Déconnexion",login_title:"Content de vous revoir",login_subtitle:"Connectez-vous à votre compte FactoryPulse AI",ph_email:"E-mail",ph_password:"Mot de passe",remember_me:"Se souvenir de moi",login_btn:"Se connecter",login_link_register:"Pas de compte ? Créez-en un",register_title:"Créer votre compte",register_subtitle:"Commencez à surveiller vos usines avec l'IA",ph_full_name:"Nom Complet",ph_confirm_password:"Confirmer le Mot de Passe",register_btn:"Créer un Compte",register_link_login:"Déjà un compte ? Se connecter",err_missing_fields:"Veuillez remplir tous les champs",err_invalid_email:"Veuillez entrer une adresse e-mail valide",err_weak_password:"Le mot de passe doit contenir 8 caractères min., une lettre et un chiffre",err_password_mismatch:"Les mots de passe ne correspondent pas",err_invalid_credentials:"E-mail ou mot de passe incorrect",err_email_taken:"Cet e-mail est déjà enregistré",err_generic:"Une erreur est survenue. Veuillez réessayer",my_factories_title:"Mes Usines",add_factory_btn:"+ Ajouter une Usine",edit_factory_btn:"Modifier",delete_factory_btn:"Supprimer",confirm_delete_factory:"Supprimer cette usine ? Cette action est irréversible.",no_factories_yet:"Vous n'avez pas encore ajouté d'usine.",factory_created_toast:"Usine créée et analysée",factory_updated_toast:"Usine mise à jour",factory_deleted_toast:"Usine supprimée",ai_insights_feed_title:"Flux d'Analyses IA",no_ai_insights_yet:"Aucune analyse IA pour l'instant. Ajoutez une usine.",reanalyze_btn:"Réanalyser",view_insights_btn:"Voir les Analyses",created_label:"Créée le",cancel_btn:"Annuler",save_btn:"Enregistrer les Modifications",forgot_password_link:"Mot de passe oublié ?",forgot_password_subtitle:"Entrez votre e-mail et nous vous enverrons un code de vérification.",send_code_btn:"Envoyer le Code",back_to_login_link:"Retour à la connexion",enter_code_subtitle:"Entrez le code à 6 chiffres envoyé à votre e-mail.",dev_code_hint_msg:"E-mail non configuré sur le serveur - votre code est : {code}",ph_code:"Code de Vérification",verify_code_btn:"Vérifier le Code",resend_code_link:"Renvoyer le code",err_invalid_or_expired_code:"Code invalide ou expiré",set_new_password_subtitle:"Choisissez un nouveau mot de passe pour votre compte.",ph_new_password:"Nouveau Mot de Passe",set_password_btn:"Définir le Nouveau Mot de Passe",password_reset_success_msg:"Mot de passe réinitialisé avec succès. Vous pouvez maintenant vous connecter.",err_email_send_failed:"Impossible d'envoyer l'e-mail. Veuillez contacter votre administrateur."},
  es: {tagline:"Plataforma Global de Inteligencia Industrial",live_label:"En vivo",kpi_energy:"Uso de Energía",kpi_efficiency:"Eficiencia",kpi_active:"Máquinas Activas",kpi_alerts:"Alertas",kwh_unit:"kWh",chart_title:"Rendimiento en Tiempo Real",machine_status_title:"Estado de Máquinas",status_running:"Funcionando",status_warning:"Advertencia",status_critical:"Crítico",form_title:"Entrada de Datos de Fábrica",factory_name_label:"Nombre de Fábrica",machine_count_label:"Número de Máquinas",energy_cost_label:"Costo de Energía ($/kWh)",machine_type_label:"Tipo de Máquina",temperature_label:"Temperatura (°C)",vibration_label:"Vibración (mm/s)",load_label:"Carga (%)",submit_btn:"Analizar Fábrica",submitting:"Actualizando...",ai_panel_title:"Perspectivas IA",ai_placeholder:"Envíe datos de fábrica para generar un análisis IA.",ai_analyzing:"Analizando...",ai_risks:"Riesgos",ai_efficiency_insights:"Análisis de Eficiencia",ai_optimizations:"Sugerencias de Optimización",toast_updated:"Datos de fábrica actualizados",toast_analysis_done:"Análisis IA completo",toast_error:"Algo salió mal",nav_dashboard:"Panel",nav_factories:"Fábricas",nav_ai_insights:"Perspectivas IA",logout_btn:"Cerrar Sesión",login_title:"Bienvenido de nuevo",login_subtitle:"Inicia sesión en tu cuenta de FactoryPulse AI",ph_email:"Correo electrónico",ph_password:"Contraseña",remember_me:"Recuérdame",login_btn:"Iniciar Sesión",login_link_register:"¿No tienes cuenta? Crea una",register_title:"Crea tu cuenta",register_subtitle:"Empieza a monitorear tus fábricas con IA",ph_full_name:"Nombre Completo",ph_confirm_password:"Confirmar Contraseña",register_btn:"Crear Cuenta",register_link_login:"¿Ya tienes cuenta? Inicia sesión",err_missing_fields:"Por favor complete todos los campos",err_invalid_email:"Por favor ingrese un correo válido",err_weak_password:"La contraseña debe tener mín. 8 caracteres, una letra y un número",err_password_mismatch:"Las contraseñas no coinciden",err_invalid_credentials:"Correo o contraseña incorrectos",err_email_taken:"Este correo ya está registrado",err_generic:"Algo salió mal. Inténtalo de nuevo",my_factories_title:"Mis Fábricas",add_factory_btn:"+ Añadir Fábrica",edit_factory_btn:"Editar",delete_factory_btn:"Eliminar",confirm_delete_factory:"¿Eliminar esta fábrica? Esta acción no se puede deshacer.",no_factories_yet:"Aún no has añadido ninguna fábrica.",factory_created_toast:"Fábrica creada y analizada",factory_updated_toast:"Fábrica actualizada",factory_deleted_toast:"Fábrica eliminada",ai_insights_feed_title:"Feed de Perspectivas IA",no_ai_insights_yet:"Aún no hay perspectivas IA. Añade una fábrica.",reanalyze_btn:"Reanalizar",view_insights_btn:"Ver Perspectivas",created_label:"Creada",cancel_btn:"Cancelar",save_btn:"Guardar Cambios",forgot_password_link:"¿Olvidó su contraseña?",forgot_password_subtitle:"Ingrese su correo y le enviaremos un código de verificación.",send_code_btn:"Enviar Código",back_to_login_link:"Volver al inicio de sesión",enter_code_subtitle:"Ingrese el código de 6 dígitos enviado a su correo.",dev_code_hint_msg:"Correo no configurado en el servidor - su código es: {code}",ph_code:"Código de Verificación",verify_code_btn:"Verificar Código",resend_code_link:"Reenviar código",err_invalid_or_expired_code:"Código inválido o caducado",set_new_password_subtitle:"Elija una nueva contraseña para su cuenta.",ph_new_password:"Nueva Contraseña",set_password_btn:"Establecer Nueva Contraseña",password_reset_success_msg:"Contraseña restablecida con éxito. Ya puede iniciar sesión.",err_email_send_failed:"No se pudo enviar el correo. Contacte a su administrador."},
  zh: {tagline:"全球工业智能平台",live_label:"实时",kpi_energy:"能源使用量",kpi_efficiency:"效率",kpi_active:"运行中设备",kpi_alerts:"警报",kwh_unit:"kWh",chart_title:"实时性能",machine_status_title:"设备状态",status_running:"运行中",status_warning:"警告",status_critical:"严重",form_title:"工厂数据输入",factory_name_label:"工厂名称",machine_count_label:"设备数量",energy_cost_label:"能源成本 ($/kWh)",machine_type_label:"设备类型",temperature_label:"温度 (°C)",vibration_label:"振动 (mm/s)",load_label:"负载 (%)",submit_btn:"分析工厂",submitting:"更新中...",ai_panel_title:"AI 洞察",ai_placeholder:"提交工厂数据以生成AI分析。",ai_analyzing:"分析中...",ai_risks:"风险",ai_efficiency_insights:"效率分析",ai_optimizations:"优化建议",toast_updated:"工厂数据已更新",toast_analysis_done:"AI分析已完成",toast_error:"出现错误",nav_dashboard:"仪表盘",nav_factories:"工厂",nav_ai_insights:"AI洞察",logout_btn:"退出",login_title:"欢迎回来",login_subtitle:"登录您的 FactoryPulse AI 账户",ph_email:"电子邮件",ph_password:"密码",remember_me:"记住我",login_btn:"登录",login_link_register:"没有账户？创建一个",register_title:"创建账户",register_subtitle:"开始使用AI监控您的工厂",ph_full_name:"全名",ph_confirm_password:"确认密码",register_btn:"创建账户",register_link_login:"已有账户？登录",err_missing_fields:"请填写所有字段",err_invalid_email:"请输入有效的电子邮件地址",err_weak_password:"密码至少8位，需包含字母和数字",err_password_mismatch:"两次密码不一致",err_invalid_credentials:"电子邮件或密码错误",err_email_taken:"该电子邮件已被注册",err_generic:"出现错误，请重试",my_factories_title:"我的工厂",add_factory_btn:"+ 添加工厂",edit_factory_btn:"编辑",delete_factory_btn:"删除",confirm_delete_factory:"删除此工厂？此操作无法撤销。",no_factories_yet:"您还没有添加任何工厂。",factory_created_toast:"工厂已创建并分析",factory_updated_toast:"工厂已更新",factory_deleted_toast:"工厂已删除",ai_insights_feed_title:"AI洞察动态",no_ai_insights_yet:"暂无AI洞察。请添加工厂开始。",reanalyze_btn:"重新分析",view_insights_btn:"查看洞察",created_label:"创建于",cancel_btn:"取消",save_btn:"保存更改",forgot_password_link:"忘记密码？",forgot_password_subtitle:"输入您的电子邮件，我们将发送验证码。",send_code_btn:"发送验证码",back_to_login_link:"返回登录",enter_code_subtitle:"输入发送到您邮箱的6位验证码。",dev_code_hint_msg:"服务器未配置邮件 - 您的验证码是：{code}",ph_code:"验证码",verify_code_btn:"验证码",resend_code_link:"重新发送验证码",err_invalid_or_expired_code:"验证码无效或已过期",set_new_password_subtitle:"为您的账户选择新密码。",ph_new_password:"新密码",set_password_btn:"设置新密码",password_reset_success_msg:"密码重置成功。您现在可以登录了。",err_email_send_failed:"无法发送邮件，请联系管理员。"},
  ar: {tagline:"منصة الذكاء الصناعي العالمية",live_label:"مباشر",kpi_energy:"استهلاك الطاقة",kpi_efficiency:"الكفاءة",kpi_active:"الآلات النشطة",kpi_alerts:"التنبيهات",kwh_unit:"kWh",chart_title:"الأداء في الوقت الفعلي",machine_status_title:"حالة الآلات",status_running:"تعمل",status_warning:"تحذير",status_critical:"حرج",form_title:"إدخال بيانات المصنع",factory_name_label:"اسم المصنع",machine_count_label:"عدد الآلات",energy_cost_label:"تكلفة الطاقة ($/kWh)",machine_type_label:"نوع الآلة",temperature_label:"درجة الحرارة (°C)",vibration_label:"الاهتزاز (مم/ث)",load_label:"الحمل (%)",submit_btn:"تحليل المصنع",submitting:"جارٍ التحديث...",ai_panel_title:"رؤى الذكاء الاصطناعي",ai_placeholder:"أرسل بيانات المصنع لإنشاء تحليل بالذكاء الاصطناعي.",ai_analyzing:"جارٍ التحليل...",ai_risks:"المخاطر",ai_efficiency_insights:"تحليل الكفاءة",ai_optimizations:"اقتراحات التحسين",toast_updated:"تم تحديث بيانات المصنع",toast_analysis_done:"اكتمل تحليل الذكاء الاصطناعي",toast_error:"حدث خطأ ما",nav_dashboard:"لوحة التحكم",nav_factories:"المصانع",nav_ai_insights:"رؤى الذكاء الاصطناعي",logout_btn:"تسجيل الخروج",login_title:"مرحباً بعودتك",login_subtitle:"سجل الدخول إلى حساب FactoryPulse AI الخاص بك",ph_email:"البريد الإلكتروني",ph_password:"كلمة المرور",remember_me:"تذكرني",login_btn:"تسجيل الدخول",login_link_register:"ليس لديك حساب؟ أنشئ واحداً",register_title:"إنشاء حسابك",register_subtitle:"ابدأ بمراقبة مصانعك بالذكاء الاصطناعي",ph_full_name:"الاسم الكامل",ph_confirm_password:"تأكيد كلمة المرور",register_btn:"إنشاء حساب",register_link_login:"لديك حساب بالفعل؟ سجل الدخول",err_missing_fields:"يرجى ملء جميع الحقول",err_invalid_email:"يرجى إدخال بريد إلكتروني صالح",err_weak_password:"يجب أن تكون كلمة المرور 8 أحرف على الأقل وتحتوي على حرف ورقم",err_password_mismatch:"كلمتا المرور غير متطابقتين",err_invalid_credentials:"البريد الإلكتروني أو كلمة المرور غير صحيحة",err_email_taken:"هذا البريد الإلكتروني مسجل بالفعل",err_generic:"حدث خطأ ما. يرجى المحاولة مرة أخرى",my_factories_title:"مصانعي",add_factory_btn:"+ إضافة مصنع",edit_factory_btn:"تعديل",delete_factory_btn:"حذف",confirm_delete_factory:"هل تريد حذف هذا المصنع؟ لا يمكن التراجع عن هذا.",no_factories_yet:"لم تقم بإضافة أي مصنع بعد.",factory_created_toast:"تم إنشاء المصنع وتحليله",factory_updated_toast:"تم تحديث المصنع",factory_deleted_toast:"تم حذف المصنع",ai_insights_feed_title:"موجز رؤى الذكاء الاصطناعي",no_ai_insights_yet:"لا توجد رؤى بعد. أضف مصنعاً للبدء.",reanalyze_btn:"إعادة التحليل",view_insights_btn:"عرض الرؤى",created_label:"تاريخ الإنشاء",cancel_btn:"إلغاء",save_btn:"حفظ التغييرات",forgot_password_link:"نسيت كلمة المرور؟",forgot_password_subtitle:"أدخل بريدك الإلكتروني وسنرسل لك رمز التحقق.",send_code_btn:"إرسال الرمز",back_to_login_link:"العودة لتسجيل الدخول",enter_code_subtitle:"أدخل الرمز المكون من 6 أرقام الذي أرسلناه إلى بريدك الإلكتروني.",dev_code_hint_msg:"البريد الإلكتروني غير مُعد على الخادم - رمزك هو: {code}",ph_code:"رمز التحقق",verify_code_btn:"التحقق من الرمز",resend_code_link:"إعادة إرسال الرمز",err_invalid_or_expired_code:"رمز غير صالح أو منتهي الصلاحية",set_new_password_subtitle:"اختر كلمة مرور جديدة لحسابك.",ph_new_password:"كلمة المرور الجديدة",set_password_btn:"تعيين كلمة المرور الجديدة",password_reset_success_msg:"تمت إعادة تعيين كلمة المرور بنجاح. يمكنك الآن تسجيل الدخول.",err_email_send_failed:"تعذر إرسال البريد الإلكتروني. يرجى الاتصال بالمسؤول."},
  tr: {tagline:"Küresel Endüstriyel Zeka Platformu",live_label:"Canlı",kpi_energy:"Enerji Kullanımı",kpi_efficiency:"Verimlilik",kpi_active:"Aktif Makineler",kpi_alerts:"Uyarılar",kwh_unit:"kWh",chart_title:"Gerçek Zamanlı Performans",machine_status_title:"Makine Durumu",status_running:"Çalışıyor",status_warning:"Uyarı",status_critical:"Kritik",form_title:"Fabrika Veri Girişi",factory_name_label:"Fabrika Adı",machine_count_label:"Makine Sayısı",energy_cost_label:"Enerji Maliyeti ($/kWh)",machine_type_label:"Makine Türü",temperature_label:"Sıcaklık (°C)",vibration_label:"Titreşim (mm/s)",load_label:"Yük (%)",submit_btn:"Fabrikayı Analiz Et",submitting:"Güncelleniyor...",ai_panel_title:"AI Analizleri",ai_placeholder:"AI analizi oluşturmak için fabrika verilerini gönderin.",ai_analyzing:"Analiz ediliyor...",ai_risks:"Riskler",ai_efficiency_insights:"Verimlilik Analizi",ai_optimizations:"Optimizasyon Önerileri",toast_updated:"Fabrika verileri güncellendi",toast_analysis_done:"AI analizi tamamlandı",toast_error:"Bir şeyler ters gitti",nav_dashboard:"Panel",nav_factories:"Fabrikalar",nav_ai_insights:"AI Analizleri",logout_btn:"Çıkış Yap",login_title:"Tekrar hoş geldiniz",login_subtitle:"FactoryPulse AI hesabınıza giriş yapın",ph_email:"E-posta",ph_password:"Şifre",remember_me:"Beni hatırla",login_btn:"Giriş Yap",login_link_register:"Hesabınız yok mu? Oluşturun",register_title:"Hesabınızı oluşturun",register_subtitle:"Fabrikalarınızı AI ile izlemeye başlayın",ph_full_name:"Ad Soyad",ph_confirm_password:"Şifreyi Onayla",register_btn:"Hesap Oluştur",register_link_login:"Zaten hesabınız var mı? Giriş yapın",err_missing_fields:"Lütfen tüm alanları doldurun",err_invalid_email:"Lütfen geçerli bir e-posta adresi girin",err_weak_password:"Şifre en az 8 karakter, bir harf ve bir rakam içermeli",err_password_mismatch:"Şifreler eşleşmiyor",err_invalid_credentials:"E-posta veya şifre hatalı",err_email_taken:"Bu e-posta zaten kayıtlı",err_generic:"Bir şeyler ters gitti. Tekrar deneyin",my_factories_title:"Fabrikalarım",add_factory_btn:"+ Fabrika Ekle",edit_factory_btn:"Düzenle",delete_factory_btn:"Sil",confirm_delete_factory:"Bu fabrika silinsin mi? Bu işlem geri alınamaz.",no_factories_yet:"Henüz fabrika eklemediniz.",factory_created_toast:"Fabrika oluşturuldu ve analiz edildi",factory_updated_toast:"Fabrika güncellendi",factory_deleted_toast:"Fabrika silindi",ai_insights_feed_title:"AI Analiz Akışı",no_ai_insights_yet:"Henüz AI analizi yok. Başlamak için fabrika ekleyin.",reanalyze_btn:"Yeniden Analiz Et",view_insights_btn:"Analizleri Görüntüle",created_label:"Oluşturulma",cancel_btn:"İptal",save_btn:"Değişiklikleri Kaydet",forgot_password_link:"Şifrenizi mi unuttunuz?",forgot_password_subtitle:"E-postanızı girin, size bir doğrulama kodu gönderelim.",send_code_btn:"Kod Gönder",back_to_login_link:"Girişe dön",enter_code_subtitle:"E-postanıza gönderilen 6 haneli kodu girin.",dev_code_hint_msg:"Sunucuda e-posta yapılandırılmamış - kodunuz: {code}",ph_code:"Doğrulama Kodu",verify_code_btn:"Kodu Doğrula",resend_code_link:"Kodu yeniden gönder",err_invalid_or_expired_code:"Geçersiz veya süresi dolmuş kod",set_new_password_subtitle:"Hesabınız için yeni bir şifre seçin.",ph_new_password:"Yeni Şifre",set_password_btn:"Yeni Şifreyi Ayarla",password_reset_success_msg:"Şifre başarıyla sıfırlandı. Şimdi giriş yapabilirsiniz.",err_email_send_failed:"E-posta gönderilemedi. Lütfen yöneticinizle iletişime geçin."},
  it: {tagline:"Piattaforma Globale di Intelligenza Industriale",live_label:"In diretta",kpi_energy:"Consumo Energetico",kpi_efficiency:"Efficienza",kpi_active:"Macchine Attive",kpi_alerts:"Avvisi",kwh_unit:"kWh",chart_title:"Prestazioni in Tempo Reale",machine_status_title:"Stato delle Macchine",status_running:"In funzione",status_warning:"Avviso",status_critical:"Critico",form_title:"Inserimento Dati Fabbrica",factory_name_label:"Nome Fabbrica",machine_count_label:"Numero di Macchine",energy_cost_label:"Costo Energia ($/kWh)",machine_type_label:"Tipo di Macchina",temperature_label:"Temperatura (°C)",vibration_label:"Vibrazione (mm/s)",load_label:"Carico (%)",submit_btn:"Analizza Fabbrica",submitting:"Aggiornamento...",ai_panel_title:"Analisi IA",ai_placeholder:"Invia i dati della fabbrica per generare un'analisi IA.",ai_analyzing:"Analisi in corso...",ai_risks:"Rischi",ai_efficiency_insights:"Analisi dell'Efficienza",ai_optimizations:"Suggerimenti di Ottimizzazione",toast_updated:"Dati fabbrica aggiornati",toast_analysis_done:"Analisi IA completata",toast_error:"Qualcosa è andato storto",nav_dashboard:"Dashboard",nav_factories:"Fabbriche",nav_ai_insights:"Analisi IA",logout_btn:"Esci",login_title:"Bentornato",login_subtitle:"Accedi al tuo account FactoryPulse AI",ph_email:"Email",ph_password:"Password",remember_me:"Ricordami",login_btn:"Accedi",login_link_register:"Non hai un account? Creane uno",register_title:"Crea il tuo account",register_subtitle:"Inizia a monitorare le tue fabbriche con l'IA",ph_full_name:"Nome Completo",ph_confirm_password:"Conferma Password",register_btn:"Crea Account",register_link_login:"Hai già un account? Accedi",err_missing_fields:"Si prega di compilare tutti i campi",err_invalid_email:"Inserisci un indirizzo email valido",err_weak_password:"La password deve avere almeno 8 caratteri, una lettera e un numero",err_password_mismatch:"Le password non corrispondono",err_invalid_credentials:"Email o password errati",err_email_taken:"Questa email è già registrata",err_generic:"Qualcosa è andato storto. Riprova",my_factories_title:"Le Mie Fabbriche",add_factory_btn:"+ Aggiungi Fabbrica",edit_factory_btn:"Modifica",delete_factory_btn:"Elimina",confirm_delete_factory:"Eliminare questa fabbrica? Questa azione non può essere annullata.",no_factories_yet:"Non hai ancora aggiunto nessuna fabbrica.",factory_created_toast:"Fabbrica creata e analizzata",factory_updated_toast:"Fabbrica aggiornata",factory_deleted_toast:"Fabbrica eliminata",ai_insights_feed_title:"Feed di Analisi IA",no_ai_insights_yet:"Nessuna analisi IA ancora. Aggiungi una fabbrica.",reanalyze_btn:"Rianalizza",view_insights_btn:"Vedi Analisi",created_label:"Creata il",cancel_btn:"Annulla",save_btn:"Salva Modifiche",forgot_password_link:"Password dimenticata?",forgot_password_subtitle:"Inserisci la tua email e ti invieremo un codice di verifica.",send_code_btn:"Invia Codice",back_to_login_link:"Torna al login",enter_code_subtitle:"Inserisci il codice a 6 cifre inviato alla tua email.",dev_code_hint_msg:"Email non configurata sul server - il tuo codice è: {code}",ph_code:"Codice di Verifica",verify_code_btn:"Verifica Codice",resend_code_link:"Reinvia codice",err_invalid_or_expired_code:"Codice non valido o scaduto",set_new_password_subtitle:"Scegli una nuova password per il tuo account.",ph_new_password:"Nuova Password",set_password_btn:"Imposta Nuova Password",password_reset_success_msg:"Password reimpostata con successo. Ora puoi accedere.",err_email_send_failed:"Impossibile inviare l'email. Contatta l'amministratore."},
  pt: {tagline:"Plataforma Global de Inteligência Industrial",live_label:"Ao vivo",kpi_energy:"Uso de Energia",kpi_efficiency:"Eficiência",kpi_active:"Máquinas Ativas",kpi_alerts:"Alertas",kwh_unit:"kWh",chart_title:"Desempenho em Tempo Real",machine_status_title:"Status das Máquinas",status_running:"Em funcionamento",status_warning:"Aviso",status_critical:"Crítico",form_title:"Entrada de Dados da Fábrica",factory_name_label:"Nome da Fábrica",machine_count_label:"Número de Máquinas",energy_cost_label:"Custo de Energia ($/kWh)",machine_type_label:"Tipo de Máquina",temperature_label:"Temperatura (°C)",vibration_label:"Vibração (mm/s)",load_label:"Carga (%)",submit_btn:"Analisar Fábrica",submitting:"Atualizando...",ai_panel_title:"Insights de IA",ai_placeholder:"Envie os dados da fábrica para gerar uma análise de IA.",ai_analyzing:"Analisando...",ai_risks:"Riscos",ai_efficiency_insights:"Análise de Eficiência",ai_optimizations:"Sugestões de Otimização",toast_updated:"Dados da fábrica atualizados",toast_analysis_done:"Análise de IA concluída",toast_error:"Algo deu errado",nav_dashboard:"Painel",nav_factories:"Fábricas",nav_ai_insights:"Insights de IA",logout_btn:"Sair",login_title:"Bem-vindo de volta",login_subtitle:"Entre na sua conta FactoryPulse AI",ph_email:"E-mail",ph_password:"Senha",remember_me:"Lembrar de mim",login_btn:"Entrar",login_link_register:"Não tem conta? Crie uma",register_title:"Crie sua conta",register_subtitle:"Comece a monitorar suas fábricas com IA",ph_full_name:"Nome Completo",ph_confirm_password:"Confirmar Senha",register_btn:"Criar Conta",register_link_login:"Já tem conta? Entrar",err_missing_fields:"Por favor preencha todos os campos",err_invalid_email:"Por favor insira um e-mail válido",err_weak_password:"A senha deve ter no mínimo 8 caracteres, uma letra e um número",err_password_mismatch:"As senhas não coincidem",err_invalid_credentials:"E-mail ou senha incorretos",err_email_taken:"Este e-mail já está registrado",err_generic:"Algo deu errado. Tente novamente",my_factories_title:"Minhas Fábricas",add_factory_btn:"+ Adicionar Fábrica",edit_factory_btn:"Editar",delete_factory_btn:"Excluir",confirm_delete_factory:"Excluir esta fábrica? Esta ação não pode ser desfeita.",no_factories_yet:"Você ainda não adicionou nenhuma fábrica.",factory_created_toast:"Fábrica criada e analisada",factory_updated_toast:"Fábrica atualizada",factory_deleted_toast:"Fábrica excluída",ai_insights_feed_title:"Feed de Insights de IA",no_ai_insights_yet:"Ainda sem insights de IA. Adicione uma fábrica.",reanalyze_btn:"Reanalisar",view_insights_btn:"Ver Insights",created_label:"Criada em",cancel_btn:"Cancelar",save_btn:"Salvar Alterações",forgot_password_link:"Esqueceu a senha?",forgot_password_subtitle:"Digite seu e-mail e enviaremos um código de verificação.",send_code_btn:"Enviar Código",back_to_login_link:"Voltar ao login",enter_code_subtitle:"Digite o código de 6 dígitos enviado ao seu e-mail.",dev_code_hint_msg:"E-mail não configurado no servidor - seu código é: {code}",ph_code:"Código de Verificação",verify_code_btn:"Verificar Código",resend_code_link:"Reenviar código",err_invalid_or_expired_code:"Código inválido ou expirado",set_new_password_subtitle:"Escolha uma nova senha para sua conta.",ph_new_password:"Nova Senha",set_password_btn:"Definir Nova Senha",password_reset_success_msg:"Senha redefinida com sucesso. Agora você pode fazer login.",err_email_send_failed:"Não foi possível enviar o e-mail. Contate o administrador."},
  ja: {tagline:"グローバル産業インテリジェンスプラットフォーム",live_label:"ライブ",kpi_energy:"エネルギー使用量",kpi_efficiency:"効率",kpi_active:"稼働中の機械",kpi_alerts:"アラート",kwh_unit:"kWh",chart_title:"リアルタイムパフォーマンス",machine_status_title:"機械の状態",status_running:"稼働中",status_warning:"警告",status_critical:"重大",form_title:"工場データ入力",factory_name_label:"工場名",machine_count_label:"機械の数",energy_cost_label:"エネルギーコスト ($/kWh)",machine_type_label:"機械の種類",temperature_label:"温度 (°C)",vibration_label:"振動 (mm/s)",load_label:"負荷 (%)",submit_btn:"工場を分析",submitting:"更新中...",ai_panel_title:"AIインサイト",ai_placeholder:"工場データを送信してAI分析を生成してください。",ai_analyzing:"分析中...",ai_risks:"リスク",ai_efficiency_insights:"効率分析",ai_optimizations:"最適化提案",toast_updated:"工場データが更新されました",toast_analysis_done:"AI分析が完了しました",toast_error:"問題が発生しました",nav_dashboard:"ダッシュボード",nav_factories:"工場",nav_ai_insights:"AIインサイト",logout_btn:"ログアウト",login_title:"おかえりなさい",login_subtitle:"FactoryPulse AI アカウントにログイン",ph_email:"メールアドレス",ph_password:"パスワード",remember_me:"ログイン状態を保持",login_btn:"ログイン",login_link_register:"アカウントをお持ちでないですか？作成する",register_title:"アカウントを作成",register_subtitle:"AIで工場の監視を始めましょう",ph_full_name:"氏名",ph_confirm_password:"パスワードの確認",register_btn:"アカウント作成",register_link_login:"すでにアカウントをお持ちですか？ログイン",err_missing_fields:"すべての項目を入力してください",err_invalid_email:"有効なメールアドレスを入力してください",err_weak_password:"パスワードは8文字以上で、文字と数字を含める必要があります",err_password_mismatch:"パスワードが一致しません",err_invalid_credentials:"メールアドレスまたはパスワードが正しくありません",err_email_taken:"このメールアドレスは既に登録されています",err_generic:"エラーが発生しました。再試行してください",my_factories_title:"マイ工場",add_factory_btn:"+ 工場を追加",edit_factory_btn:"編集",delete_factory_btn:"削除",confirm_delete_factory:"この工場を削除しますか？元に戻せません。",no_factories_yet:"まだ工場を追加していません。",factory_created_toast:"工場が作成・分析されました",factory_updated_toast:"工場が更新されました",factory_deleted_toast:"工場が削除されました",ai_insights_feed_title:"AIインサイトフィード",no_ai_insights_yet:"AIインサイトはまだありません。工場を追加してください。",reanalyze_btn:"再分析",view_insights_btn:"インサイトを見る",created_label:"作成日",cancel_btn:"キャンセル",save_btn:"変更を保存",forgot_password_link:"パスワードをお忘れですか？",forgot_password_subtitle:"メールアドレスを入力すると、確認コードをお送りします。",send_code_btn:"コードを送信",back_to_login_link:"ログインに戻る",enter_code_subtitle:"メールに送信された6桁のコードを入力してください。",dev_code_hint_msg:"サーバーでメールが設定されていません - あなたのコード: {code}",ph_code:"確認コード",verify_code_btn:"コードを確認",resend_code_link:"コードを再送信",err_invalid_or_expired_code:"無効または期限切れのコードです",set_new_password_subtitle:"アカウントの新しいパスワードを選択してください。",ph_new_password:"新しいパスワード",set_password_btn:"新しいパスワードを設定",password_reset_success_msg:"パスワードが正常にリセットされました。ログインできます。",err_email_send_failed:"メールを送信できませんでした。管理者にお問い合わせください。"},
  ko: {tagline:"글로벌 산업 인텔리전스 플랫폼",live_label:"실시간",kpi_energy:"에너지 사용량",kpi_efficiency:"효율성",kpi_active:"가동 중인 기계",kpi_alerts:"경고",kwh_unit:"kWh",chart_title:"실시간 성능",machine_status_title:"기계 상태",status_running:"가동 중",status_warning:"경고",status_critical:"심각",form_title:"공장 데이터 입력",factory_name_label:"공장 이름",machine_count_label:"기계 수",energy_cost_label:"에너지 비용 ($/kWh)",machine_type_label:"기계 유형",temperature_label:"온도 (°C)",vibration_label:"진동 (mm/s)",load_label:"부하 (%)",submit_btn:"공장 분석",submitting:"업데이트 중...",ai_panel_title:"AI 인사이트",ai_placeholder:"AI 분석을 생성하려면 공장 데이터를 제출하세요.",ai_analyzing:"분석 중...",ai_risks:"위험 요소",ai_efficiency_insights:"효율성 분석",ai_optimizations:"최적화 제안",toast_updated:"공장 데이터가 업데이트되었습니다",toast_analysis_done:"AI 분석이 완료되었습니다",toast_error:"문제가 발생했습니다",nav_dashboard:"대시보드",nav_factories:"공장",nav_ai_insights:"AI 인사이트",logout_btn:"로그아웃",login_title:"다시 오신 것을 환영합니다",login_subtitle:"FactoryPulse AI 계정에 로그인하세요",ph_email:"이메일",ph_password:"비밀번호",remember_me:"로그인 상태 유지",login_btn:"로그인",login_link_register:"계정이 없으신가요? 계정 만들기",register_title:"계정 만들기",register_subtitle:"AI로 공장 모니터링을 시작하세요",ph_full_name:"성명",ph_confirm_password:"비밀번호 확인",register_btn:"계정 생성",register_link_login:"이미 계정이 있으신가요? 로그인",err_missing_fields:"모든 항목을 입력해주세요",err_invalid_email:"유효한 이메일 주소를 입력하세요",err_weak_password:"비밀번호는 8자 이상, 문자와 숫자를 포함해야 합니다",err_password_mismatch:"비밀번호가 일치하지 않습니다",err_invalid_credentials:"이메일 또는 비밀번호가 올바르지 않습니다",err_email_taken:"이미 등록된 이메일입니다",err_generic:"문제가 발생했습니다. 다시 시도해주세요",my_factories_title:"내 공장",add_factory_btn:"+ 공장 추가",edit_factory_btn:"수정",delete_factory_btn:"삭제",confirm_delete_factory:"이 공장을 삭제하시겠습니까? 되돌릴 수 없습니다.",no_factories_yet:"아직 추가된 공장이 없습니다.",factory_created_toast:"공장이 생성되고 분석되었습니다",factory_updated_toast:"공장이 업데이트되었습니다",factory_deleted_toast:"공장이 삭제되었습니다",ai_insights_feed_title:"AI 인사이트 피드",no_ai_insights_yet:"아직 AI 인사이트가 없습니다. 공장을 추가하세요.",reanalyze_btn:"다시 분석",view_insights_btn:"인사이트 보기",created_label:"생성일",cancel_btn:"취소",save_btn:"변경사항 저장",forgot_password_link:"비밀번호를 잊으셨나요?",forgot_password_subtitle:"이메일을 입력하시면 인증 코드를 보내드립니다.",send_code_btn:"코드 전송",back_to_login_link:"로그인으로 돌아가기",enter_code_subtitle:"이메일로 전송된 6자리 코드를 입력하세요.",dev_code_hint_msg:"서버에 이메일이 구성되지 않았습니다 - 코드: {code}",ph_code:"인증 코드",verify_code_btn:"코드 확인",resend_code_link:"코드 재전송",err_invalid_or_expired_code:"잘못되었거나 만료된 코드입니다",set_new_password_subtitle:"계정의 새 비밀번호를 선택하세요.",ph_new_password:"새 비밀번호",set_password_btn:"새 비밀번호 설정",password_reset_success_msg:"비밀번호가 성공적으로 재설정되었습니다. 이제 로그인할 수 있습니다.",err_email_send_failed:"이메일을 보낼 수 없습니다. 관리자에게 문의하세요."},
  hi: {tagline:"वैश्विक औद्योगिक बुद्धिमत्ता मंच",live_label:"लाइव",kpi_energy:"ऊर्जा उपयोग",kpi_efficiency:"दक्षता",kpi_active:"सक्रिय मशीनें",kpi_alerts:"अलर्ट",kwh_unit:"kWh",chart_title:"रीयल-टाइम प्रदर्शन",machine_status_title:"मशीन की स्थिति",status_running:"चल रहा है",status_warning:"चेतावनी",status_critical:"गंभीर",form_title:"फ़ैक्टरी डेटा इनपुट",factory_name_label:"फ़ैक्टरी का नाम",machine_count_label:"मशीनों की संख्या",energy_cost_label:"ऊर्जा लागत ($/kWh)",machine_type_label:"मशीन प्रकार",temperature_label:"तापमान (°C)",vibration_label:"कंपन (mm/s)",load_label:"लोड (%)",submit_btn:"फ़ैक्टरी का विश्लेषण करें",submitting:"अद्यतन हो रहा है...",ai_panel_title:"AI अंतर्दृष्टि",ai_placeholder:"AI विश्लेषण उत्पन्न करने के लिए फ़ैक्टरी डेटा सबमिट करें।",ai_analyzing:"विश्लेषण हो रहा है...",ai_risks:"जोखिम",ai_efficiency_insights:"दक्षता विश्लेषण",ai_optimizations:"अनुकूलन सुझाव",toast_updated:"फ़ैक्टरी डेटा अपडेट किया गया",toast_analysis_done:"AI विश्लेषण पूर्ण हुआ",toast_error:"कुछ गलत हो गया",nav_dashboard:"डैशबोर्ड",nav_factories:"फ़ैक्टरियाँ",nav_ai_insights:"AI अंतर्दृष्टि",logout_btn:"लॉग आउट",login_title:"वापसी पर स्वागत है",login_subtitle:"अपने FactoryPulse AI खाते में लॉग इन करें",ph_email:"ईमेल",ph_password:"पासवर्ड",remember_me:"मुझे याद रखें",login_btn:"लॉग इन करें",login_link_register:"खाता नहीं है? एक बनाएं",register_title:"अपना खाता बनाएं",register_subtitle:"AI के साथ अपनी फ़ैक्टरियों की निगरानी शुरू करें",ph_full_name:"पूरा नाम",ph_confirm_password:"पासवर्ड की पुष्टि करें",register_btn:"खाता बनाएं",register_link_login:"पहले से खाता है? लॉग इन करें",err_missing_fields:"कृपया सभी फ़ील्ड भरें",err_invalid_email:"कृपया एक मान्य ईमेल पता दर्ज करें",err_weak_password:"पासवर्ड कम से कम 8 अक्षर, एक अक्षर और एक अंक होना चाहिए",err_password_mismatch:"पासवर्ड मेल नहीं खाते",err_invalid_credentials:"गलत ईमेल या पासवर्ड",err_email_taken:"यह ईमेल पहले से पंजीकृत है",err_generic:"कुछ गलत हो गया। कृपया पुनः प्रयास करें",my_factories_title:"मेरी फ़ैक्टरियाँ",add_factory_btn:"+ फ़ैक्टरी जोड़ें",edit_factory_btn:"संपादित करें",delete_factory_btn:"हटाएं",confirm_delete_factory:"इस फ़ैक्टरी को हटाएं? इसे पूर्ववत नहीं किया जा सकता।",no_factories_yet:"आपने अभी तक कोई फ़ैक्टरी नहीं जोड़ी है।",factory_created_toast:"फ़ैक्टरी बनाई और विश्लेषित की गई",factory_updated_toast:"फ़ैक्टरी अपडेट की गई",factory_deleted_toast:"फ़ैक्टरी हटाई गई",ai_insights_feed_title:"AI अंतर्दृष्टि फ़ीड",no_ai_insights_yet:"अभी तक कोई AI अंतर्दृष्टि नहीं। शुरू करने के लिए एक फ़ैक्टरी जोड़ें।",reanalyze_btn:"पुनः विश्लेषण करें",view_insights_btn:"अंतर्दृष्टि देखें",created_label:"बनाया गया",cancel_btn:"रद्द करें",save_btn:"परिवर्तन सहेजें",forgot_password_link:"पासवर्ड भूल गए?",forgot_password_subtitle:"अपना ईमेल दर्ज करें, हम आपको एक सत्यापन कोड भेजेंगे।",send_code_btn:"कोड भेजें",back_to_login_link:"लॉगिन पर वापस जाएं",enter_code_subtitle:"अपने ईमेल पर भेजा गया 6-अंकीय कोड दर्ज करें।",dev_code_hint_msg:"सर्वर पर ईमेल कॉन्फ़िगर नहीं है - आपका कोड है: {code}",ph_code:"सत्यापन कोड",verify_code_btn:"कोड सत्यापित करें",resend_code_link:"कोड पुनः भेजें",err_invalid_or_expired_code:"अमान्य या समाप्त कोड",set_new_password_subtitle:"अपने खाते के लिए नया पासवर्ड चुनें।",ph_new_password:"नया पासवर्ड",set_password_btn:"नया पासवर्ड सेट करें",password_reset_success_msg:"पासवर्ड सफलतापूर्वक रीसेट हो गया। अब आप लॉगिन कर सकते हैं।",err_email_send_failed:"ईमेल नहीं भेजा जा सका। कृपया व्यवस्थापक से संपर्क करें।"},
  uz: {tagline:"Global sanoat intellekti platformasi",live_label:"Jonli",kpi_energy:"Energiya sarfi",kpi_efficiency:"Samaradorlik",kpi_active:"Faol stanoklar",kpi_alerts:"Ogohlantirishlar",kwh_unit:"kWh",chart_title:"Real vaqtdagi ko'rsatkichlar",machine_status_title:"Stanoklar holati",status_running:"Ishlamoqda",status_warning:"Ogohlantirish",status_critical:"Muhim",form_title:"Zavod ma'lumotlarini kiritish",factory_name_label:"Zavod nomi",machine_count_label:"Stanoklar soni",energy_cost_label:"Energiya narxi ($/kWh)",machine_type_label:"Stanok turi",temperature_label:"Harorat (°C)",vibration_label:"Tebranish (mm/s)",load_label:"Yuklama (%)",submit_btn:"Zavodni tahlil qilish",submitting:"Yangilanmoqda...",ai_panel_title:"AI tahlili",ai_placeholder:"AI tahlilini olish uchun zavod ma'lumotlarini yuboring.",ai_analyzing:"Tahlil qilinmoqda...",ai_risks:"Xavflar",ai_efficiency_insights:"Samaradorlik tahlili",ai_optimizations:"Optimallashtirish tavsiyalari",toast_updated:"Zavod ma'lumotlari yangilandi",toast_analysis_done:"AI tahlili yakunlandi",toast_error:"Xatolik yuz berdi",nav_dashboard:"Boshqaruv paneli",nav_factories:"Zavodlar",nav_ai_insights:"AI tahlili",logout_btn:"Chiqish",login_title:"Xush kelibsiz",login_subtitle:"FactoryPulse AI hisobingizga kiring",ph_email:"Elektron pochta",ph_password:"Parol",remember_me:"Meni eslab qol",login_btn:"Kirish",login_link_register:"Hisobingiz yo'qmi? Yarating",register_title:"Hisob yarating",register_subtitle:"Zavodlaringizni AI bilan kuzatishni boshlang",ph_full_name:"To'liq ism",ph_confirm_password:"Parolni tasdiqlang",register_btn:"Hisob yaratish",register_link_login:"Hisobingiz bormi? Kiring",err_missing_fields:"Barcha maydonlarni to'ldiring",err_invalid_email:"Yaroqli elektron pochta manzilini kiriting",err_weak_password:"Parol kamida 8 belgidan, harf va raqamdan iborat bo'lishi kerak",err_password_mismatch:"Parollar mos kelmaydi",err_invalid_credentials:"Elektron pochta yoki parol noto'g'ri",err_email_taken:"Bu elektron pochta allaqachon ro'yxatdan o'tgan",err_generic:"Xatolik yuz berdi. Qaytadan urinib ko'ring",my_factories_title:"Mening Zavodlarim",add_factory_btn:"+ Zavod qo'shish",edit_factory_btn:"Tahrirlash",delete_factory_btn:"O'chirish",confirm_delete_factory:"Bu zavodni o'chirasizmi? Buni bekor qilib bo'lmaydi.",no_factories_yet:"Siz hali hech qanday zavod qo'shmagansiz.",factory_created_toast:"Zavod yaratildi va tahlil qilindi",factory_updated_toast:"Zavod yangilandi",factory_deleted_toast:"Zavod o'chirildi",ai_insights_feed_title:"AI Tahlili Lentasi",no_ai_insights_yet:"Hali AI tahlili yo'q. Boshlash uchun zavod qo'shing.",reanalyze_btn:"Qayta tahlil qilish",view_insights_btn:"Tahlilni ko'rish",created_label:"Yaratilgan",cancel_btn:"Bekor qilish",save_btn:"O'zgarishlarni saqlash",forgot_password_link:"Parolni unutdingizmi?",forgot_password_subtitle:"Emailingizni kiriting, biz sizga tasdiqlash kodini yuboramiz.",send_code_btn:"Kod yuborish",back_to_login_link:"Kirishga qaytish",enter_code_subtitle:"Emailingizga yuborilgan 6 xonali kodni kiriting.",dev_code_hint_msg:"Serverda email sozlanmagan - kodingiz: {code}",ph_code:"Tasdiqlash kodi",verify_code_btn:"Kodni tasdiqlash",resend_code_link:"Kodni qayta yuborish",err_invalid_or_expired_code:"Kod noto'g'ri yoki muddati o'tgan",set_new_password_subtitle:"Hisobingiz uchun yangi parol tanlang.",ph_new_password:"Yangi parol",set_password_btn:"Yangi parolni o'rnatish",password_reset_success_msg:"Parol muvaffaqiyatli qayta o'rnatildi. Endi tizimga kirishingiz mumkin.",err_email_send_failed:"Xat yuborilmadi. Administrator bilan bog'laning."},
  ky: {tagline:"Глобалдык өнөр жай интеллект платформасы",live_label:"Түз эфир",kpi_energy:"Энергия сарпталышы",kpi_efficiency:"Эффективдүүлүк",kpi_active:"Активдүү станоктор",kpi_alerts:"Дабылдар",kwh_unit:"кВт·саат",chart_title:"Реалдуу убакыттагы көрсөткүчтөр",machine_status_title:"Станоктордун абалы",status_running:"Иштеп жатат",status_warning:"Эскертүү",status_critical:"Олуттуу",form_title:"Завод маалыматтарын киргизүү",factory_name_label:"Заводдун аты",machine_count_label:"Станоктордун саны",energy_cost_label:"Энергия наркы ($/кВт·саат)",machine_type_label:"Станоктун түрү",temperature_label:"Температура (°C)",vibration_label:"Дирилдөө (мм/с)",load_label:"Жүктөм (%)",submit_btn:"Заводду талдоо",submitting:"Жаңыртылууда...",ai_panel_title:"AI-талдоо",ai_placeholder:"AI-талдоо алуу үчүн завод маалыматтарын жөнөтүңүз.",ai_analyzing:"Талдануда...",ai_risks:"Тобокелдиктер",ai_efficiency_insights:"Эффективдүүлүк талдоосу",ai_optimizations:"Оптималдаштыруу сунуштары",toast_updated:"Завод маалыматтары жаңыртылды",toast_analysis_done:"AI-талдоо аяктады",toast_error:"Ката кетти",nav_dashboard:"Башкаруу панели",nav_factories:"Заводдор",nav_ai_insights:"AI-талдоо",logout_btn:"Чыгуу",login_title:"Кайра кош келиңиз",login_subtitle:"FactoryPulse AI каттоо эсебиңизге кириңиз",ph_email:"Электрондук почта",ph_password:"Сырсөз",remember_me:"Мени эстеп кал",login_btn:"Кирүү",login_link_register:"Каттоо эсебиңиз жокпу? Түзүү",register_title:"Каттоо эсебин түзүү",register_subtitle:"Заводдоруңузду AI менен байкоону баштаңыз",ph_full_name:"Толук аты-жөнү",ph_confirm_password:"Сырсөздү ырастаңыз",register_btn:"Каттоо эсебин түзүү",register_link_login:"Каттоо эсебиңиз барбы? Кирүү",err_missing_fields:"Бардык талааларды толтуруңуз",err_invalid_email:"Жарактуу электрондук почта дарегин киргизиңиз",err_weak_password:"Сырсөз кеминде 8 белги, тамга жана сан камтышы керек",err_password_mismatch:"Сырсөздөр дал келбейт",err_invalid_credentials:"Электрондук почта же сырсөз туура эмес",err_email_taken:"Бул электрондук почта мурунтан катталган",err_generic:"Ката кетти. Кайра аракет кылыңыз",my_factories_title:"Менин Заводдорум",add_factory_btn:"+ Завод кошуу",edit_factory_btn:"Түзөтүү",delete_factory_btn:"Өчүрүү",confirm_delete_factory:"Бул заводду өчүрөсүзбү? Бул аракетти артка кайтарууга болбойт.",no_factories_yet:"Сиз азырынча эч кандай завод кошкон жоксуз.",factory_created_toast:"Завод түзүлдү жана талданды",factory_updated_toast:"Завод жаңыртылды",factory_deleted_toast:"Завод өчүрүлдү",ai_insights_feed_title:"AI-талдоо тизмеси",no_ai_insights_yet:"Азырынча AI-талдоо жок. Баштоо үчүн завод кошуңуз.",reanalyze_btn:"Кайра талдоо",view_insights_btn:"Талдоону көрүү",created_label:"Түзүлгөн күнү",cancel_btn:"Жокко чыгаруу",save_btn:"Өзгөртүүлөрдү сактоо",forgot_password_link:"Сырсөздү унуттуңузбу?",forgot_password_subtitle:"Email киргизиңиз, биз сизге ырастоо кодун жөнөтөбүз.",send_code_btn:"Код жөнөтүү",back_to_login_link:"Кирүүгө кайтуу",enter_code_subtitle:"Email дарегиңизге жөнөтүлгөн 6 сандуу кодду киргизиңиз.",dev_code_hint_msg:"Серверде email орнотулган эмес - сиздин кодуңуз: {code}",ph_code:"Ырастоо коду",verify_code_btn:"Кодду ырастоо",resend_code_link:"Кодду кайра жөнөтүү",err_invalid_or_expired_code:"Ката же мөөнөтү өткөн код",set_new_password_subtitle:"Каттоо эсебиңиз үчүн жаңы сырсөз тандаңыз.",ph_new_password:"Жаңы сырсөз",set_password_btn:"Жаңы сырсөздү орнотуу",password_reset_success_msg:"Сырсөз ийгиликтүү өзгөртүлдү. Эми кире аласыз.",err_email_send_failed:"Кат жөнөтүлгөн жок. Администраторго кайрылыңыз."},
  uk: {tagline:"Глобальна платформа промислового інтелекту",live_label:"Наживо",kpi_energy:"Споживання енергії",kpi_efficiency:"Ефективність",kpi_active:"Активні верстати",kpi_alerts:"Сповіщення",kwh_unit:"кВт·год",chart_title:"Показники в реальному часі",machine_status_title:"Статус верстатів",status_running:"Працює",status_warning:"Попередження",status_critical:"Критично",form_title:"Введення даних заводу",factory_name_label:"Назва заводу",machine_count_label:"Кількість верстатів",energy_cost_label:"Вартість енергії ($/кВт·год)",machine_type_label:"Тип верстата",temperature_label:"Температура (°C)",vibration_label:"Вібрація (мм/с)",load_label:"Навантаження (%)",submit_btn:"Аналізувати завод",submitting:"Оновлення...",ai_panel_title:"AI-аналітика",ai_placeholder:"Надішліть дані заводу, щоб отримати AI-аналіз.",ai_analyzing:"Аналіз...",ai_risks:"Ризики",ai_efficiency_insights:"Аналіз ефективності",ai_optimizations:"Рекомендації з оптимізації",toast_updated:"Дані заводу оновлено",toast_analysis_done:"AI-аналіз завершено",toast_error:"Сталася помилка",nav_dashboard:"Панель",nav_factories:"Заводи",nav_ai_insights:"AI-аналітика",logout_btn:"Вийти",login_title:"З поверненням",login_subtitle:"Увійдіть у свій обліковий запис FactoryPulse AI",ph_email:"Електронна пошта",ph_password:"Пароль",remember_me:"Запам'ятати мене",login_btn:"Увійти",login_link_register:"Немає акаунту? Створити",register_title:"Створіть акаунт",register_subtitle:"Почніть моніторинг заводів за допомогою AI",ph_full_name:"Повне ім'я",ph_confirm_password:"Підтвердіть пароль",register_btn:"Створити акаунт",register_link_login:"Вже є акаунт? Увійти",err_missing_fields:"Будь ласка, заповніть усі поля",err_invalid_email:"Введіть дійсну електронну адресу",err_weak_password:"Пароль має містити щонайменше 8 символів, літеру та цифру",err_password_mismatch:"Паролі не збігаються",err_invalid_credentials:"Невірна електронна пошта або пароль",err_email_taken:"Ця електронна пошта вже зареєстрована",err_generic:"Сталася помилка. Спробуйте ще раз",my_factories_title:"Мої Заводи",add_factory_btn:"+ Додати завод",edit_factory_btn:"Редагувати",delete_factory_btn:"Видалити",confirm_delete_factory:"Видалити цей завод? Цю дію не можна скасувати.",no_factories_yet:"Ви ще не додали жодного заводу.",factory_created_toast:"Завод створено та проаналізовано",factory_updated_toast:"Завод оновлено",factory_deleted_toast:"Завод видалено",ai_insights_feed_title:"Стрічка AI-аналітики",no_ai_insights_yet:"Ще немає AI-аналітики. Додайте завод.",reanalyze_btn:"Проаналізувати знову",view_insights_btn:"Переглянути аналітику",created_label:"Створено",cancel_btn:"Скасувати",save_btn:"Зберегти зміни",forgot_password_link:"Забули пароль?",forgot_password_subtitle:"Введіть email, і ми надішлемо вам код підтвердження.",send_code_btn:"Надіслати код",back_to_login_link:"Назад до входу",enter_code_subtitle:"Введіть 6-значний код, надісланий на вашу пошту.",dev_code_hint_msg:"Пошта не налаштована на сервері — ваш код: {code}",ph_code:"Код підтвердження",verify_code_btn:"Підтвердити код",resend_code_link:"Надіслати код повторно",err_invalid_or_expired_code:"Невірний або прострочений код",set_new_password_subtitle:"Виберіть новий пароль для облікового запису.",ph_new_password:"Новий пароль",set_password_btn:"Встановити новий пароль",password_reset_success_msg:"Пароль успішно змінено. Тепер ви можете увійти.",err_email_send_failed:"Не вдалося надіслати лист. Зверніться до адміністратора."},
  pl: {tagline:"Globalna Platforma Inteligencji Przemysłowej",live_label:"Na żywo",kpi_energy:"Zużycie Energii",kpi_efficiency:"Wydajność",kpi_active:"Aktywne Maszyny",kpi_alerts:"Alerty",kwh_unit:"kWh",chart_title:"Wydajność w Czasie Rzeczywistym",machine_status_title:"Status Maszyn",status_running:"Działa",status_warning:"Ostrzeżenie",status_critical:"Krytyczne",form_title:"Wprowadzanie Danych Fabryki",factory_name_label:"Nazwa Fabryki",machine_count_label:"Liczba Maszyn",energy_cost_label:"Koszt Energii ($/kWh)",machine_type_label:"Typ Maszyny",temperature_label:"Temperatura (°C)",vibration_label:"Wibracje (mm/s)",load_label:"Obciążenie (%)",submit_btn:"Analizuj Fabrykę",submitting:"Aktualizowanie...",ai_panel_title:"Analizy AI",ai_placeholder:"Prześlij dane fabryki, aby wygenerować analizę AI.",ai_analyzing:"Analizowanie...",ai_risks:"Ryzyka",ai_efficiency_insights:"Analiza Wydajności",ai_optimizations:"Sugestie Optymalizacji",toast_updated:"Dane fabryki zaktualizowane",toast_analysis_done:"Analiza AI zakończona",toast_error:"Coś poszło nie tak",nav_dashboard:"Panel",nav_factories:"Fabryki",nav_ai_insights:"Analizy AI",logout_btn:"Wyloguj",login_title:"Witamy z powrotem",login_subtitle:"Zaloguj się do swojego konta FactoryPulse AI",ph_email:"E-mail",ph_password:"Hasło",remember_me:"Zapamiętaj mnie",login_btn:"Zaloguj się",login_link_register:"Nie masz konta? Utwórz je",register_title:"Utwórz konto",register_subtitle:"Zacznij monitorować swoje fabryki z AI",ph_full_name:"Imię i Nazwisko",ph_confirm_password:"Potwierdź Hasło",register_btn:"Utwórz Konto",register_link_login:"Masz już konto? Zaloguj się",err_missing_fields:"Proszę wypełnić wszystkie pola",err_invalid_email:"Proszę podać prawidłowy adres e-mail",err_weak_password:"Hasło musi mieć min. 8 znaków, literę i cyfrę",err_password_mismatch:"Hasła nie pasują do siebie",err_invalid_credentials:"Nieprawidłowy e-mail lub hasło",err_email_taken:"Ten e-mail jest już zarejestrowany",err_generic:"Coś poszło nie tak. Spróbuj ponownie",my_factories_title:"Moje Fabryki",add_factory_btn:"+ Dodaj Fabrykę",edit_factory_btn:"Edytuj",delete_factory_btn:"Usuń",confirm_delete_factory:"Usunąć tę fabrykę? Tej czynności nie można cofnąć.",no_factories_yet:"Nie dodałeś jeszcze żadnej fabryki.",factory_created_toast:"Fabryka utworzona i przeanalizowana",factory_updated_toast:"Fabryka zaktualizowana",factory_deleted_toast:"Fabryka usunięta",ai_insights_feed_title:"Kanał Analiz AI",no_ai_insights_yet:"Brak analiz AI. Dodaj fabrykę, aby zacząć.",reanalyze_btn:"Analizuj Ponownie",view_insights_btn:"Zobacz Analizy",created_label:"Utworzono",cancel_btn:"Anuluj",save_btn:"Zapisz Zmiany",forgot_password_link:"Zapomniałeś hasła?",forgot_password_subtitle:"Wpisz swój e-mail, a wyślemy Ci kod weryfikacyjny.",send_code_btn:"Wyślij Kod",back_to_login_link:"Powrót do logowania",enter_code_subtitle:"Wprowadź 6-cyfrowy kod wysłany na Twój e-mail.",dev_code_hint_msg:"E-mail nie skonfigurowany na serwerze - Twój kod to: {code}",ph_code:"Kod Weryfikacyjny",verify_code_btn:"Zweryfikuj Kod",resend_code_link:"Wyślij kod ponownie",err_invalid_or_expired_code:"Nieprawidłowy lub wygasły kod",set_new_password_subtitle:"Wybierz nowe hasło dla swojego konta.",ph_new_password:"Nowe Hasło",set_password_btn:"Ustaw Nowe Hasło",password_reset_success_msg:"Hasło pomyślnie zresetowane. Możesz teraz się zalogować.",err_email_send_failed:"Nie udało się wysłać e-maila. Skontaktuj się z administratorem."},
  nl: {tagline:"Wereldwijd Industrieel Intelligentieplatform",live_label:"Live",kpi_energy:"Energieverbruik",kpi_efficiency:"Efficiëntie",kpi_active:"Actieve Machines",kpi_alerts:"Meldingen",kwh_unit:"kWh",chart_title:"Realtime Prestaties",machine_status_title:"Machinestatus",status_running:"Actief",status_warning:"Waarschuwing",status_critical:"Kritiek",form_title:"Fabrieksgegevens Invoeren",factory_name_label:"Fabrieksnaam",machine_count_label:"Aantal Machines",energy_cost_label:"Energiekosten ($/kWh)",machine_type_label:"Machinetype",temperature_label:"Temperatuur (°C)",vibration_label:"Trilling (mm/s)",load_label:"Belasting (%)",submit_btn:"Fabriek Analyseren",submitting:"Bijwerken...",ai_panel_title:"AI-inzichten",ai_placeholder:"Verzend fabrieksgegevens om een AI-analyse te genereren.",ai_analyzing:"Analyseren...",ai_risks:"Risico's",ai_efficiency_insights:"Efficiëntieanalyse",ai_optimizations:"Optimalisatiesuggesties",toast_updated:"Fabrieksgegevens bijgewerkt",toast_analysis_done:"AI-analyse voltooid",toast_error:"Er is iets misgegaan",nav_dashboard:"Dashboard",nav_factories:"Fabrieken",nav_ai_insights:"AI-inzichten",logout_btn:"Uitloggen",login_title:"Welkom terug",login_subtitle:"Log in op uw FactoryPulse AI-account",ph_email:"E-mail",ph_password:"Wachtwoord",remember_me:"Onthoud mij",login_btn:"Inloggen",login_link_register:"Geen account? Maak er een",register_title:"Maak uw account aan",register_subtitle:"Begin met AI-monitoring van uw fabrieken",ph_full_name:"Volledige Naam",ph_confirm_password:"Bevestig Wachtwoord",register_btn:"Account Aanmaken",register_link_login:"Heeft u al een account? Inloggen",err_missing_fields:"Vul alle velden in",err_invalid_email:"Voer een geldig e-mailadres in",err_weak_password:"Wachtwoord moet minimaal 8 tekens, een letter en een cijfer bevatten",err_password_mismatch:"Wachtwoorden komen niet overeen",err_invalid_credentials:"Ongeldige e-mail of wachtwoord",err_email_taken:"Dit e-mailadres is al geregistreerd",err_generic:"Er is iets misgegaan. Probeer het opnieuw",my_factories_title:"Mijn Fabrieken",add_factory_btn:"+ Fabriek Toevoegen",edit_factory_btn:"Bewerken",delete_factory_btn:"Verwijderen",confirm_delete_factory:"Deze fabriek verwijderen? Dit kan niet ongedaan worden gemaakt.",no_factories_yet:"U heeft nog geen fabrieken toegevoegd.",factory_created_toast:"Fabriek aangemaakt en geanalyseerd",factory_updated_toast:"Fabriek bijgewerkt",factory_deleted_toast:"Fabriek verwijderd",ai_insights_feed_title:"AI-inzichten Feed",no_ai_insights_yet:"Nog geen AI-inzichten. Voeg een fabriek toe.",reanalyze_btn:"Opnieuw Analyseren",view_insights_btn:"Bekijk Inzichten",created_label:"Aangemaakt",cancel_btn:"Annuleren",save_btn:"Wijzigingen Opslaan",forgot_password_link:"Wachtwoord vergeten?",forgot_password_subtitle:"Voer uw e-mail in en we sturen u een verificatiecode.",send_code_btn:"Code Versturen",back_to_login_link:"Terug naar inloggen",enter_code_subtitle:"Voer de 6-cijferige code in die naar uw e-mail is gestuurd.",dev_code_hint_msg:"E-mail niet geconfigureerd op de server - uw code is: {code}",ph_code:"Verificatiecode",verify_code_btn:"Code Verifiëren",resend_code_link:"Code opnieuw versturen",err_invalid_or_expired_code:"Ongeldige of verlopen code",set_new_password_subtitle:"Kies een nieuw wachtwoord voor uw account.",ph_new_password:"Nieuw Wachtwoord",set_password_btn:"Nieuw Wachtwoord Instellen",password_reset_success_msg:"Wachtwoord succesvol gereset. U kunt nu inloggen.",err_email_send_failed:"Kon de e-mail niet verzenden. Neem contact op met uw beheerder."},
  sv: {tagline:"Global Industriell Intelligensplattform",live_label:"Live",kpi_energy:"Energiförbrukning",kpi_efficiency:"Effektivitet",kpi_active:"Aktiva Maskiner",kpi_alerts:"Varningar",kwh_unit:"kWh",chart_title:"Realtidsprestanda",machine_status_title:"Maskinstatus",status_running:"Igång",status_warning:"Varning",status_critical:"Kritisk",form_title:"Fabriksdatainmatning",factory_name_label:"Fabriksnamn",machine_count_label:"Antal Maskiner",energy_cost_label:"Energikostnad ($/kWh)",machine_type_label:"Maskintyp",temperature_label:"Temperatur (°C)",vibration_label:"Vibration (mm/s)",load_label:"Belastning (%)",submit_btn:"Analysera Fabrik",submitting:"Uppdaterar...",ai_panel_title:"AI-insikter",ai_placeholder:"Skicka fabriksdata för att generera en AI-analys.",ai_analyzing:"Analyserar...",ai_risks:"Risker",ai_efficiency_insights:"Effektivitetsanalys",ai_optimizations:"Optimeringsförslag",toast_updated:"Fabriksdata uppdaterad",toast_analysis_done:"AI-analys klar",toast_error:"Något gick fel",nav_dashboard:"Instrumentpanel",nav_factories:"Fabriker",nav_ai_insights:"AI-insikter",logout_btn:"Logga ut",login_title:"Välkommen tillbaka",login_subtitle:"Logga in på ditt FactoryPulse AI-konto",ph_email:"E-post",ph_password:"Lösenord",remember_me:"Kom ihåg mig",login_btn:"Logga in",login_link_register:"Inget konto? Skapa ett",register_title:"Skapa ditt konto",register_subtitle:"Börja övervaka dina fabriker med AI",ph_full_name:"Fullständigt Namn",ph_confirm_password:"Bekräfta Lösenord",register_btn:"Skapa Konto",register_link_login:"Har du redan ett konto? Logga in",err_missing_fields:"Vänligen fyll i alla fält",err_invalid_email:"Ange en giltig e-postadress",err_weak_password:"Lösenordet måste vara minst 8 tecken med en bokstav och en siffra",err_password_mismatch:"Lösenorden matchar inte",err_invalid_credentials:"Felaktig e-post eller lösenord",err_email_taken:"Denna e-post är redan registrerad",err_generic:"Något gick fel. Försök igen",my_factories_title:"Mina Fabriker",add_factory_btn:"+ Lägg till Fabrik",edit_factory_btn:"Redigera",delete_factory_btn:"Ta bort",confirm_delete_factory:"Ta bort denna fabrik? Detta kan inte ångras.",no_factories_yet:"Du har inte lagt till några fabriker än.",factory_created_toast:"Fabrik skapad och analyserad",factory_updated_toast:"Fabrik uppdaterad",factory_deleted_toast:"Fabrik borttagen",ai_insights_feed_title:"AI-insikter Flöde",no_ai_insights_yet:"Inga AI-insikter än. Lägg till en fabrik.",reanalyze_btn:"Analysera Igen",view_insights_btn:"Visa Insikter",created_label:"Skapad",cancel_btn:"Avbryt",save_btn:"Spara Ändringar",forgot_password_link:"Glömt lösenordet?",forgot_password_subtitle:"Ange din e-post så skickar vi en verifieringskod.",send_code_btn:"Skicka Kod",back_to_login_link:"Tillbaka till inloggning",enter_code_subtitle:"Ange den 6-siffriga koden som skickades till din e-post.",dev_code_hint_msg:"E-post inte konfigurerad på servern - din kod är: {code}",ph_code:"Verifieringskod",verify_code_btn:"Verifiera Kod",resend_code_link:"Skicka koden igen",err_invalid_or_expired_code:"Ogiltig eller utgången kod",set_new_password_subtitle:"Välj ett nytt lösenord för ditt konto.",ph_new_password:"Nytt Lösenord",set_password_btn:"Ange Nytt Lösenord",password_reset_success_msg:"Lösenordet har återställts. Du kan nu logga in.",err_email_send_failed:"Kunde inte skicka e-postmeddelandet. Kontakta din administratör."},
};

let currentLang = localStorage.getItem("fp_lang") || "en";
if (!translations[currentLang]) currentLang = "en";
const RTL_LANGS = ["ar"];
function t(key) { return (translations[currentLang] && translations[currentLang][key]) || translations.en[key] || key; }
function applyTranslations() {
  document.documentElement.lang = currentLang;
  document.documentElement.dir = RTL_LANGS.includes(currentLang) ? "rtl" : "ltr";
  document.querySelectorAll("[data-t]").forEach(el => { el.textContent = t(el.getAttribute("data-t")); });
}
function buildLangSelector() {
  const langNames = { en:"English", ru:"\u0420\u0443\u0441\u0441\u043a\u0438\u0439", kk:"\u049a\u0430\u0437\u0430\u049b\u0448\u0430", de:"Deutsch", fr:"Fran\u00e7ais", es:"Espa\u00f1ol", zh:"\u4e2d\u6587", ar:"\u0627\u0644\u0639\u0631\u0628\u064a\u0629", tr:"T\u00fcrk\u00e7e", it:"Italiano", pt:"Portugu\u00eas", ja:"\u65e5\u672c\u8a9e", ko:"\ud55c\uad6d\uc5b4", hi:"\u0939\u093f\u0928\u094d\u0926\u0940", uz:"O\u02bbzbekcha", ky:"\u041a\u044b\u0440\u0433\u044b\u0437\u0447\u0430", uk:"\u0423\u043a\u0440\u0430\u0457\u043d\u0441\u044c\u043a\u0430", pl:"Polski", nl:"Nederlands", sv:"Svenska" };
  const sel = document.getElementById("lang-select");
  sel.innerHTML = "";
  Object.keys(translations).forEach(code => {
    const opt = document.createElement("option");
    opt.value = code; opt.textContent = langNames[code] || code;
    if (code === currentLang) opt.selected = true;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", e => { currentLang = e.target.value; localStorage.setItem("fp_lang", currentLang); applyTranslations(); });
}
function showError(msg) {
  const el = document.getElementById("form-error");
  el.textContent = msg;
  el.classList.remove("hidden");
  el.classList.remove("anim-error"); void el.offsetWidth; el.classList.add("anim-error");
}
function hideError() { document.getElementById("form-error").classList.add("hidden"); }

function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  const colors = { info: "bg-cyan-600", success: "bg-emerald-500", error: "bg-red-500" };
  const el = document.createElement("div");
  el.className = `toast ${colors[type] || colors.info} text-white text-sm px-4 py-3 rounded-xl shadow-lg max-w-xs`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .3s"; setTimeout(() => el.remove(), 300); }, 3500);
}

const EYE_ICON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7Z"/><circle cx="12" cy="12" r="3"/></svg>';
const EYE_OFF_ICON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.9 4.24A10.94 10.94 0 0 1 12 4c7 0 11 7 11 7a13.16 13.16 0 0 1-1.67 2.68M6.61 6.61A13.526 13.526 0 0 0 1 12s4 7 11 7a10.94 10.94 0 0 0 5.11-1.24M14.12 14.12a3 3 0 1 1-4.24-4.24"/><path d="m1 1 22 22"/></svg>';
function initPasswordToggles() {
  document.querySelectorAll(".toggle-eye").forEach(btn => {
    btn.addEventListener("click", () => {
      const input = document.getElementById(btn.dataset.target);
      if (!input) return;
      const isPw = input.type === "password";
      input.type = isPw ? "text" : "password";
      btn.innerHTML = isPw ? EYE_OFF_ICON : EYE_ICON;
    });
  });
}

document.addEventListener("contextmenu", (e) => e.preventDefault());

document.addEventListener("DOMContentLoaded", () => {
  buildLangSelector();
  applyTranslations();
  initPasswordToggles();
  if (localStorage.getItem("fp_token")) { window.location.href = "/dashboard"; return; }

  document.getElementById("login-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError();
    const email = document.getElementById("email").value.trim();
    const password = document.getElementById("password").value;
    const remember = document.getElementById("remember").checked;
    if (!email || !password) { showError(t("err_missing_fields")); return; }

    const btn = document.getElementById("submit-btn");
    const spinner = document.getElementById("submit-spinner");
    btn.classList.add("opacity-70");
    spinner.classList.remove("hidden");
    try {
      const res = await fetch("/api/login", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password, remember }),
      });
      const data = await res.json();
      if (!res.ok) {
        if (data.error === "invalid_credentials") showError(t("err_invalid_credentials"));
        else showError(t("err_generic"));
        return;
      }
      localStorage.setItem("fp_token", data.token);
      window.location.href = "/dashboard";
    } catch (err) {
      showError(t("err_generic"));
    } finally {
      btn.classList.remove("opacity-70");
      spinner.classList.add("hidden");
    }
  });

  /* ============ FORGOT PASSWORD (email -> code -> new password) ============ */
  let resetEmail = "";

  function showAuthForm(id) {
    ["login-form", "forgot-password-form", "verify-code-form", "new-password-form"].forEach(fid => {
      const el = document.getElementById(fid);
      el.classList.toggle("hidden", fid !== id);
      el.classList.toggle("flex", fid === id);
    });
  }

  document.getElementById("show-forgot-password").addEventListener("click", () => {
    hideError();
    showAuthForm("forgot-password-form");
  });
  document.getElementById("show-login-form").addEventListener("click", () => {
    showAuthForm("login-form");
  });

  function showFormError(elId, msg) {
    const el = document.getElementById(elId);
    el.textContent = msg;
    el.classList.remove("hidden");
    el.classList.remove("anim-error"); void el.offsetWidth; el.classList.add("anim-error");
  }
  function hideFormError(elId) { document.getElementById(elId).classList.add("hidden"); }

  async function requestResetCode() {
    hideFormError("forgot-error");
    resetEmail = document.getElementById("forgot-email").value.trim().toLowerCase();
    if (!resetEmail) { showFormError("forgot-error", t("err_missing_fields")); return false; }

    const btn = document.getElementById("forgot-submit-btn");
    const spinner = document.getElementById("forgot-submit-spinner");
    btn.classList.add("opacity-70");
    spinner.classList.remove("hidden");
    try {
      const res = await fetch("/api/forgot-password", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: resetEmail, lang: currentLang }),
      });
      const data = await res.json();
      if (!res.ok) {
        // Mail is configured but delivery failed - tell the user plainly instead
        // of pretending a code is on its way.
        showFormError("forgot-error",
          data.error === "email_send_failed" ? t("err_email_send_failed") : t("err_generic"));
        return false;
      }
      const hint = document.getElementById("dev-code-hint");
      if (data.dev_code) {
        hint.textContent = t("dev_code_hint_msg").replace("{code}", data.dev_code);
        hint.classList.remove("hidden");
      } else {
        hint.classList.add("hidden");
      }
      showAuthForm("verify-code-form");
      document.getElementById("reset-code").value = "";
      return true;
    } catch (err) {
      showFormError("forgot-error", t("err_generic"));
      return false;
    } finally {
      btn.classList.remove("opacity-70");
      spinner.classList.add("hidden");
    }
  }

  document.getElementById("forgot-password-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    await requestResetCode();
  });

  document.getElementById("resend-code-btn").addEventListener("click", async () => {
    await requestResetCode();
  });

  document.getElementById("verify-code-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    hideFormError("verify-error");
    const code = document.getElementById("reset-code").value.trim();
    if (!code) { showFormError("verify-error", t("err_missing_fields")); return; }

    const btn = document.getElementById("verify-submit-btn");
    const spinner = document.getElementById("verify-submit-spinner");
    btn.classList.add("opacity-70");
    spinner.classList.remove("hidden");
    try {
      const res = await fetch("/api/verify-reset-code", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: resetEmail, code }),
      });
      const data = await res.json();
      if (!res.ok) {
        showFormError("verify-error", t("err_invalid_or_expired_code"));
        return;
      }
      showAuthForm("new-password-form");
    } catch (err) {
      showFormError("verify-error", t("err_generic"));
    } finally {
      btn.classList.remove("opacity-70");
      spinner.classList.add("hidden");
    }
  });

  document.getElementById("new-password-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    hideFormError("newpass-error");
    const code = document.getElementById("reset-code").value.trim();
    const newPassword = document.getElementById("new-password").value;
    const confirmPassword = document.getElementById("new-password-confirm").value;

    if (!newPassword || !confirmPassword) { showFormError("newpass-error", t("err_missing_fields")); return; }
    if (!isStrongPasswordCheck(newPassword)) { showFormError("newpass-error", t("err_weak_password")); return; }
    if (newPassword !== confirmPassword) { showFormError("newpass-error", t("err_password_mismatch")); return; }

    const btn = document.getElementById("newpass-submit-btn");
    const spinner = document.getElementById("newpass-submit-spinner");
    btn.classList.add("opacity-70");
    spinner.classList.remove("hidden");
    try {
      const res = await fetch("/api/reset-password", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: resetEmail, code, password: newPassword }),
      });
      const data = await res.json();
      if (!res.ok) {
        if (data.error === "weak_password") showFormError("newpass-error", t("err_weak_password"));
        else showFormError("newpass-error", t("err_invalid_or_expired_code"));
        return;
      }
      showAuthForm("login-form");
      document.getElementById("email").value = resetEmail;
      showToast(t("password_reset_success_msg"), "success");
    } catch (err) {
      showFormError("newpass-error", t("err_generic"));
    } finally {
      btn.classList.remove("opacity-70");
      spinner.classList.add("hidden");
    }
  });

  function isStrongPasswordCheck(pw) {
    return pw.length >= 8 && /[A-Za-z]/.test(pw) && /[0-9]/.test(pw);
  }
});
</script>
</body>
</html>
"""


REGISTER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Sign Up - FactoryPulse AI | Industrial AI & SCADA Platform</title>
<meta name="description" content="Create your FactoryPulse AI account to start monitoring factories with real-time SCADA data, AI predictive maintenance, and energy analytics." />
<meta name="robots" content="index, follow" />
<link rel="canonical" href="/register" />
<script src="https://cdn.tailwindcss.com"></script>
<style>
  html, body { height: 100%; }
  body {
    min-height: 100vh;
    background: radial-gradient(circle at 15% 0%, #1a1f3a 0%, #0a0e1f 45%, #030409 100%);
    font-family: 'Segoe UI', system-ui, sans-serif;
    color: #e6edf5;
    position: relative;
    overflow-x: hidden;
  }
  .blob { position: fixed; border-radius: 9999px; filter: blur(100px); opacity: .25; pointer-events: none; animation: floatBlob 16s ease-in-out infinite; z-index: 0; }
  @keyframes floatBlob { 0%,100% { transform: translate(0,0); } 50% { transform: translate(40px,-30px); } }
  .glass { background: rgba(255,255,255,0.045); border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); position: relative; z-index: 1; }
  .glass-strong { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.14); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); }
  .fade-in { animation: fadeIn .5s ease both; }
  @keyframes fadeIn { from { opacity:0; transform: translateY(10px);} to { opacity:1; transform:none; } }
  .glow-btn { background: linear-gradient(135deg, #06b6d4, #7c3aed, #3b82f6); background-size: 200% 200%; transition: box-shadow .25s ease, transform .15s ease; animation: gradientShift 6s ease infinite; }
  @keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
  .glow-btn:hover { box-shadow: 0 0 32px rgba(34,211,238,.5); transform: translateY(-1px); }
  .glow-btn:active { transform: translateY(0) scale(.98); }
  .neon-text { text-shadow: 0 0 18px rgba(34,211,238,.5); }
  .input-field { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12); transition: border-color .2s ease, box-shadow .2s ease, background .2s ease; }
  .input-field:focus { outline: none; border-color: #22d3ee; background: rgba(255,255,255,0.08); box-shadow: 0 0 0 4px rgba(34,211,238,.15), 0 0 24px rgba(34,211,238,.2); }
  .status-running { color: #34d399; }
  .status-warning { color: #fbbf24; }
  .status-critical { color: #f87171; }
  .bg-status-running { background: #34d399; }
  .bg-status-warning { background: #fbbf24; }
  .bg-status-critical { background: #f87171; }
  .pulse-dot { position: relative; display: inline-flex; }
  .pulse-dot::before { content: ""; position: absolute; inset: -5px; border-radius: 9999px; border: 1px solid currentColor; opacity: .5; animation: pulseRing 2s ease-out infinite; }
  @keyframes pulseRing { 0%{transform:scale(.7); opacity:.6} 100%{transform:scale(2); opacity:0} }
  .spinner { width: 18px; height: 18px; border-radius: 50%; border: 2.5px solid rgba(255,255,255,0.25); border-top-color: #22d3ee; animation: spin .7s linear infinite; display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .machine-card { transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease; }
  .machine-card:hover { transform: translateY(-2px); border-color: rgba(34,211,238,.35); box-shadow: 0 12px 30px rgba(0,0,0,.4); }
  ::-webkit-scrollbar { width: 8px; }
  ::-webkit-scrollbar-thumb { background: rgba(148,163,184,.4); border-radius: 8px; }
  select { color: #e2e8f0; color-scheme: dark; }
  select option { color: #0f172a; background: #ffffff; }
  .gauge-value { font-family: 'Consolas', monospace; }
  .fl-label { transition: all .18s ease; }
  .toast { animation: toastIn .25s ease both; }
  @keyframes toastIn { from { opacity:0; transform: translateY(-8px);} to { opacity:1; transform:none; } }
  .ai-section-title { letter-spacing: .04em; }
  .nav-btn.active, .nav-btn-m.active { background: rgba(34,211,238,0.16); color: #22d3ee; }
  .factory-card { transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease; }
  .factory-card:hover { transform: translateY(-2px); border-color: rgba(34,211,238,.35); box-shadow: 0 12px 30px rgba(0,0,0,.4); }
  .anim-error { animation: shakeIn .35s ease both; }
  @keyframes shakeIn { 0%{opacity:0; transform:translateX(-6px);} 60%{transform:translateX(3px);} 100%{opacity:1; transform:none;} }
</style>
</head>
<body>

<div class="blob" style="width:420px;height:420px;background:#0891b2;top:-10%;left:5%"></div>
<div class="blob" style="width:380px;height:380px;background:#7c3aed;bottom:-14%;right:0%"></div>

<div id="toast-container" class="fixed top-4 right-4 z-50 flex flex-col gap-2"></div>

<div class="relative z-10 min-h-screen flex items-center justify-center p-4 py-10">
  <div class="w-full max-w-md fade-in">
    <div class="flex items-center justify-center gap-3 mb-6">
      <div class="w-12 h-12 rounded-2xl flex items-center justify-center glow-btn shrink-0">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2-7 4 14 2-7h6"/></svg>
      </div>
      <div class="text-left">
        <div class="font-bold text-xl tracking-tight neon-text">FactoryPulse<span class="text-cyan-400">AI</span></div>
        <div class="text-xs text-slate-400" data-t="tagline">Global Industrial Intelligence Platform</div>
      </div>
    </div>

    <div class="glass-strong rounded-3xl p-8">
      <div class="flex justify-center mb-6">
        <select id="lang-select" class="input-field rounded-xl text-xs px-3 py-1.5 outline-none"></select>
      </div>

      <h1 class="text-lg font-semibold mb-1" data-t="register_title">Create your account</h1>
      <p class="text-sm text-slate-400 mb-6" data-t="register_subtitle">Start monitoring your factories with AI</p>

      <form id="register-form" class="flex flex-col gap-4">
        <div class="relative">
          <input id="full-name" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2" />
          <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="ph_full_name">Full Name</label>
        </div>
        <div class="relative">
          <input id="email" type="email" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2" />
          <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="ph_email">Email</label>
        </div>
        <div class="relative">
          <input id="password" type="password" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2 pr-10" />
          <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="ph_password">Password</label>
          <button type="button" class="toggle-eye absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200" data-target="password"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7Z"/><circle cx="12" cy="12" r="3"/></svg></button>
        </div>
        <div class="relative">
          <input id="confirm-password" type="password" placeholder=" " class="peer input-field w-full rounded-xl text-sm px-3 pt-5 pb-2 pr-10" />
          <label class="fl-label absolute left-3 top-2 text-xs text-slate-400 peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-xs peer-focus:text-cyan-400" data-t="ph_confirm_password">Confirm Password</label>
          <button type="button" class="toggle-eye absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200" data-target="confirm-password"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7Z"/><circle cx="12" cy="12" r="3"/></svg></button>
        </div>
        <div>
          <label class="text-xs text-slate-400 mb-1.5 block" data-t="role_label">Your Role</label>
          <select id="reg-role" class="input-field w-full rounded-xl text-sm px-3 py-2.5">
            <option value="engineer" data-t="role_engineer">Engineer</option>
            <option value="manager" data-t="role_manager">Manager</option>
            <option value="admin" data-t="role_admin">Admin</option>
          </select>
        </div>
        <div id="form-error" class="hidden anim-error text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded-xl px-3 py-2"></div>
        <button type="submit" id="submit-btn" class="glow-btn rounded-xl py-3 text-sm font-semibold flex items-center justify-center gap-2 mt-1">
          <span id="submit-spinner" class="spinner hidden"></span>
          <span id="submit-label" data-t="register_btn">Create Account</span>
        </button>
        <p class="text-xs text-slate-400 text-center mt-1">
          <a href="/login" class="text-cyan-400 hover:text-cyan-300 font-medium" data-t="register_link_login">Already have an account? Sign in</a>
        </p>
      </form>
    </div>
  </div>
</div>

<script>
const translations = {
  en: {tagline:"Global Industrial Intelligence Platform",live_label:"Live",kpi_energy:"Energy Usage",kpi_efficiency:"Efficiency",kpi_active:"Active Machines",kpi_alerts:"Alerts",kwh_unit:"kWh",chart_title:"Real-Time Performance",machine_status_title:"Machine Status",status_running:"Running",status_warning:"Warning",status_critical:"Critical",form_title:"Factory Data Input",factory_name_label:"Factory Name",machine_count_label:"Number of Machines",energy_cost_label:"Energy Cost ($/kWh)",machine_type_label:"Machine Type",temperature_label:"Temperature (°C)",vibration_label:"Vibration (mm/s)",load_label:"Load (%)",submit_btn:"Analyze Factory",submitting:"Updating...",ai_panel_title:"AI Insights",ai_placeholder:"Submit factory data to generate an AI analysis.",ai_analyzing:"Analyzing...",ai_risks:"Risks",ai_efficiency_insights:"Efficiency Insights",ai_optimizations:"Optimization Suggestions",toast_updated:"Factory data updated",toast_analysis_done:"AI analysis complete",toast_error:"Something went wrong",nav_dashboard:"Dashboard",nav_factories:"Factories",nav_ai_insights:"AI Insights",logout_btn:"Log Out",login_title:"Welcome back",login_subtitle:"Sign in to your FactoryPulse AI account",ph_email:"Email",ph_password:"Password",remember_me:"Remember me",login_btn:"Log In",login_link_register:"Don't have an account? Create one",register_title:"Create your account",register_subtitle:"Start monitoring your factories with AI",ph_full_name:"Full Name",ph_confirm_password:"Confirm Password",register_btn:"Create Account",register_link_login:"Already have an account? Sign in",err_missing_fields:"Please fill in all fields",err_invalid_email:"Please enter a valid email address",err_weak_password:"Password must be at least 8 characters with a letter and a number",err_password_mismatch:"Passwords do not match",err_invalid_credentials:"Invalid email or password",err_email_taken:"This email is already registered",err_generic:"Something went wrong. Please try again",my_factories_title:"My Factories",add_factory_btn:"+ Add Factory",edit_factory_btn:"Edit",delete_factory_btn:"Delete",confirm_delete_factory:"Delete this factory? This cannot be undone.",no_factories_yet:"You haven't added any factories yet.",factory_created_toast:"Factory created and analyzed",factory_updated_toast:"Factory updated",factory_deleted_toast:"Factory deleted",ai_insights_feed_title:"AI Insights Feed",no_ai_insights_yet:"No AI insights yet. Add a factory to get started.",reanalyze_btn:"Re-analyze",view_insights_btn:"View Insights",created_label:"Created",cancel_btn:"Cancel",save_btn:"Save Changes",role_label:"Your Role",role_engineer:"Engineer",role_manager:"Manager",role_admin:"Admin"},
  ru: {tagline:"Глобальная платформа промышленного интеллекта",live_label:"Live",kpi_energy:"Потребление энергии",kpi_efficiency:"Эффективность",kpi_active:"Активные станки",kpi_alerts:"Оповещения",kwh_unit:"кВт·ч",chart_title:"Показатели в реальном времени",machine_status_title:"Статус станков",status_running:"Работает",status_warning:"Внимание",status_critical:"Критично",form_title:"Ввод данных завода",factory_name_label:"Название завода",machine_count_label:"Количество станков",energy_cost_label:"Стоимость энергии ($/кВт·ч)",machine_type_label:"Тип станка",temperature_label:"Температура (°C)",vibration_label:"Вибрация (мм/с)",load_label:"Нагрузка (%)",submit_btn:"Анализировать завод",submitting:"Обновление...",ai_panel_title:"AI-аналитика",ai_placeholder:"Отправьте данные завода, чтобы получить AI-анализ.",ai_analyzing:"Анализ...",ai_risks:"Риски",ai_efficiency_insights:"Анализ эффективности",ai_optimizations:"Рекомендации по оптимизации",toast_updated:"Данные завода обновлены",toast_analysis_done:"AI-анализ завершён",toast_error:"Произошла ошибка",nav_dashboard:"Панель",nav_factories:"Заводы",nav_ai_insights:"AI-аналитика",logout_btn:"Выход",login_title:"С возвращением",login_subtitle:"Войдите в аккаунт FactoryPulse AI",ph_email:"Email",ph_password:"Пароль",remember_me:"Запомнить меня",login_btn:"Войти",login_link_register:"Нет аккаунта? Создать",register_title:"Создать аккаунт",register_subtitle:"Начните мониторинг заводов с помощью AI",ph_full_name:"Полное имя",ph_confirm_password:"Подтвердите пароль",register_btn:"Создать аккаунт",register_link_login:"Уже есть аккаунт? Войти",err_missing_fields:"Заполните все поля",err_invalid_email:"Введите корректный email",err_weak_password:"Пароль должен быть от 8 символов, с буквой и цифрой",err_password_mismatch:"Пароли не совпадают",err_invalid_credentials:"Неверный email или пароль",err_email_taken:"Этот email уже зарегистрирован",err_generic:"Что-то пошло не так. Попробуйте снова",my_factories_title:"Мои заводы",add_factory_btn:"+ Добавить завод",edit_factory_btn:"Изменить",delete_factory_btn:"Удалить",confirm_delete_factory:"Удалить этот завод? Это действие нельзя отменить.",no_factories_yet:"Вы ещё не добавили ни одного завода.",factory_created_toast:"Завод создан и проанализирован",factory_updated_toast:"Завод обновлён",factory_deleted_toast:"Завод удалён",ai_insights_feed_title:"Лента AI-аналитики",no_ai_insights_yet:"Пока нет AI-аналитики. Добавьте завод, чтобы начать.",reanalyze_btn:"Проанализировать снова",view_insights_btn:"Смотреть аналитику",created_label:"Создано",cancel_btn:"Отмена",save_btn:"Сохранить изменения",role_label:"Ваша роль",role_engineer:"Инженер",role_manager:"Менеджер",role_admin:"Администратор"},
  kk: {tagline:"Жаһандық өнеркәсіптік интеллект платформасы",live_label:"Тікелей эфир",kpi_energy:"Энергия тұтыну",kpi_efficiency:"Тиімділік",kpi_active:"Белсенді станоктар",kpi_alerts:"Дабылдар",kwh_unit:"кВт·сағ",chart_title:"Нақты уақыттағы көрсеткіштер",machine_status_title:"Станоктар күйі",status_running:"Жұмыс істеп тұр",status_warning:"Ескерту",status_critical:"Сыни",form_title:"Зауыт деректерін енгізу",factory_name_label:"Зауыт атауы",machine_count_label:"Станоктар саны",energy_cost_label:"Энергия құны ($/кВт·сағ)",machine_type_label:"Станок түрі",temperature_label:"Температура (°C)",vibration_label:"Діріл (мм/с)",load_label:"Жүктеме (%)",submit_btn:"Зауытты талдау",submitting:"Жаңартылуда...",ai_panel_title:"AI-талдау",ai_placeholder:"AI-талдау алу үшін зауыт деректерін жіберіңіз.",ai_analyzing:"Талдануда...",ai_risks:"Тәуекелдер",ai_efficiency_insights:"Тиімділік талдауы",ai_optimizations:"Оңтайландыру ұсыныстары",toast_updated:"Зауыт деректері жаңартылды",toast_analysis_done:"AI-талдау аяқталды",toast_error:"Қате орын алды",nav_dashboard:"Басқару тақтасы",nav_factories:"Зауыттар",nav_ai_insights:"AI-талдау",logout_btn:"Шығу",login_title:"Қайта қош келдіңіз",login_subtitle:"FactoryPulse AI аккаунтыңызға кіріңіз",ph_email:"Email",ph_password:"Құпия сөз",remember_me:"Мені есте сақтау",login_btn:"Кіру",login_link_register:"Аккаунтыңыз жоқ па? Тіркелу",register_title:"Аккаунт құру",register_subtitle:"Зауыттарды AI арқылы бақылауды бастаңыз",ph_full_name:"Толық аты-жөні",ph_confirm_password:"Құпия сөзді қайталаңыз",register_btn:"Аккаунт құру",register_link_login:"Аккаунтыңыз бар ма? Кіру",err_missing_fields:"Барлық өрістерді толтырыңыз",err_invalid_email:"Дұрыс email мекенжайын енгізіңіз",err_weak_password:"Құпия сөз кемінде 8 таңба, әріп пен сан болуы керек",err_password_mismatch:"Құпия сөздер сәйкес келмейді",err_invalid_credentials:"Қате email немесе құпия сөз",err_email_taken:"Бұл email тіркелген",err_generic:"Қате орын алды. Қайталап көріңіз",my_factories_title:"Менің зауыттарым",add_factory_btn:"+ Зауыт қосу",edit_factory_btn:"Өзгерту",delete_factory_btn:"Жою",confirm_delete_factory:"Бұл зауытты жоясыз ба? Бұл әрекетті кері қайтару мүмкін емес.",no_factories_yet:"Сіз әлі ешбір зауыт қосқан жоқсыз.",factory_created_toast:"Зауыт құрылды және талданды",factory_updated_toast:"Зауыт жаңартылды",factory_deleted_toast:"Зауыт жойылды",ai_insights_feed_title:"AI-талдау таспасы",no_ai_insights_yet:"AI-талдау әлі жоқ. Бастау үшін зауыт қосыңыз.",reanalyze_btn:"Қайта талдау",view_insights_btn:"Талдауды көру",created_label:"Құрылған күні",cancel_btn:"Бас тарту",save_btn:"Өзгерістерді сақтау",role_label:"Сіздің рөліңіз",role_engineer:"Инженер",role_manager:"Менеджер",role_admin:"Әкімші"},
  de: {tagline:"Globale Industrielle Intelligenzplattform",live_label:"Live",kpi_energy:"Energieverbrauch",kpi_efficiency:"Effizienz",kpi_active:"Aktive Maschinen",kpi_alerts:"Warnungen",kwh_unit:"kWh",chart_title:"Echtzeit-Leistung",machine_status_title:"Maschinenstatus",status_running:"Läuft",status_warning:"Warnung",status_critical:"Kritisch",form_title:"Fabrikdateneingabe",factory_name_label:"Fabrikname",machine_count_label:"Anzahl der Maschinen",energy_cost_label:"Energiekosten ($/kWh)",machine_type_label:"Maschinentyp",temperature_label:"Temperatur (°C)",vibration_label:"Vibration (mm/s)",load_label:"Last (%)",submit_btn:"Fabrik Analysieren",submitting:"Aktualisieren...",ai_panel_title:"KI-Einblicke",ai_placeholder:"Senden Sie Fabrikdaten, um eine KI-Analyse zu erstellen.",ai_analyzing:"Analysiere...",ai_risks:"Risiken",ai_efficiency_insights:"Effizienzanalyse",ai_optimizations:"Optimierungsvorschläge",toast_updated:"Fabrikdaten aktualisiert",toast_analysis_done:"KI-Analyse abgeschlossen",toast_error:"Etwas ist schiefgelaufen",nav_dashboard:"Übersicht",nav_factories:"Fabriken",nav_ai_insights:"KI-Einblicke",logout_btn:"Abmelden",login_title:"Willkommen zurück",login_subtitle:"Melden Sie sich bei Ihrem FactoryPulse AI-Konto an",ph_email:"E-Mail",ph_password:"Passwort",remember_me:"Angemeldet bleiben",login_btn:"Einloggen",login_link_register:"Kein Konto? Jetzt erstellen",register_title:"Konto erstellen",register_subtitle:"Beginnen Sie mit der KI-Überwachung Ihrer Fabriken",ph_full_name:"Vollständiger Name",ph_confirm_password:"Passwort bestätigen",register_btn:"Konto erstellen",register_link_login:"Bereits ein Konto? Anmelden",err_missing_fields:"Bitte füllen Sie alle Felder aus",err_invalid_email:"Bitte geben Sie eine gültige E-Mail-Adresse ein",err_weak_password:"Passwort muss mind. 8 Zeichen, einen Buchstaben und eine Zahl enthalten",err_password_mismatch:"Passwörter stimmen nicht überein",err_invalid_credentials:"Ungültige E-Mail oder Passwort",err_email_taken:"Diese E-Mail ist bereits registriert",err_generic:"Etwas ist schiefgelaufen. Bitte erneut versuchen",my_factories_title:"Meine Fabriken",add_factory_btn:"+ Fabrik Hinzufügen",edit_factory_btn:"Bearbeiten",delete_factory_btn:"Löschen",confirm_delete_factory:"Diese Fabrik löschen? Dies kann nicht rückgängig gemacht werden.",no_factories_yet:"Sie haben noch keine Fabriken hinzugefügt.",factory_created_toast:"Fabrik erstellt und analysiert",factory_updated_toast:"Fabrik aktualisiert",factory_deleted_toast:"Fabrik gelöscht",ai_insights_feed_title:"KI-Einblicke Feed",no_ai_insights_yet:"Noch keine KI-Einblicke. Fügen Sie eine Fabrik hinzu.",reanalyze_btn:"Erneut analysieren",view_insights_btn:"Einblicke Anzeigen",created_label:"Erstellt",cancel_btn:"Abbrechen",save_btn:"Änderungen Speichern",role_label:"Ihre Rolle",role_engineer:"Ingenieur",role_manager:"Manager",role_admin:"Administrator"},
  fr: {tagline:"Plateforme mondiale d'intelligence industrielle",live_label:"En direct",kpi_energy:"Consommation d'Énergie",kpi_efficiency:"Efficacité",kpi_active:"Machines Actives",kpi_alerts:"Alertes",kwh_unit:"kWh",chart_title:"Performance en Temps Réel",machine_status_title:"État des Machines",status_running:"En marche",status_warning:"Avertissement",status_critical:"Critique",form_title:"Saisie des Données d'Usine",factory_name_label:"Nom de l'Usine",machine_count_label:"Nombre de Machines",energy_cost_label:"Coût de l'Énergie ($/kWh)",machine_type_label:"Type de Machine",temperature_label:"Température (°C)",vibration_label:"Vibration (mm/s)",load_label:"Charge (%)",submit_btn:"Analyser l'Usine",submitting:"Mise à jour...",ai_panel_title:"Analyses IA",ai_placeholder:"Envoyez les données de l'usine pour générer une analyse IA.",ai_analyzing:"Analyse en cours...",ai_risks:"Risques",ai_efficiency_insights:"Analyse d'Efficacité",ai_optimizations:"Suggestions d'Optimisation",toast_updated:"Données d'usine mises à jour",toast_analysis_done:"Analyse IA terminée",toast_error:"Une erreur est survenue",nav_dashboard:"Tableau de Bord",nav_factories:"Usines",nav_ai_insights:"Analyses IA",logout_btn:"Déconnexion",login_title:"Content de vous revoir",login_subtitle:"Connectez-vous à votre compte FactoryPulse AI",ph_email:"E-mail",ph_password:"Mot de passe",remember_me:"Se souvenir de moi",login_btn:"Se connecter",login_link_register:"Pas de compte ? Créez-en un",register_title:"Créer votre compte",register_subtitle:"Commencez à surveiller vos usines avec l'IA",ph_full_name:"Nom Complet",ph_confirm_password:"Confirmer le Mot de Passe",register_btn:"Créer un Compte",register_link_login:"Déjà un compte ? Se connecter",err_missing_fields:"Veuillez remplir tous les champs",err_invalid_email:"Veuillez entrer une adresse e-mail valide",err_weak_password:"Le mot de passe doit contenir 8 caractères min., une lettre et un chiffre",err_password_mismatch:"Les mots de passe ne correspondent pas",err_invalid_credentials:"E-mail ou mot de passe incorrect",err_email_taken:"Cet e-mail est déjà enregistré",err_generic:"Une erreur est survenue. Veuillez réessayer",my_factories_title:"Mes Usines",add_factory_btn:"+ Ajouter une Usine",edit_factory_btn:"Modifier",delete_factory_btn:"Supprimer",confirm_delete_factory:"Supprimer cette usine ? Cette action est irréversible.",no_factories_yet:"Vous n'avez pas encore ajouté d'usine.",factory_created_toast:"Usine créée et analysée",factory_updated_toast:"Usine mise à jour",factory_deleted_toast:"Usine supprimée",ai_insights_feed_title:"Flux d'Analyses IA",no_ai_insights_yet:"Aucune analyse IA pour l'instant. Ajoutez une usine.",reanalyze_btn:"Réanalyser",view_insights_btn:"Voir les Analyses",created_label:"Créée le",cancel_btn:"Annuler",save_btn:"Enregistrer les Modifications",role_label:"Votre Rôle",role_engineer:"Ingénieur",role_manager:"Manager",role_admin:"Administrateur"},
  es: {tagline:"Plataforma Global de Inteligencia Industrial",live_label:"En vivo",kpi_energy:"Uso de Energía",kpi_efficiency:"Eficiencia",kpi_active:"Máquinas Activas",kpi_alerts:"Alertas",kwh_unit:"kWh",chart_title:"Rendimiento en Tiempo Real",machine_status_title:"Estado de Máquinas",status_running:"Funcionando",status_warning:"Advertencia",status_critical:"Crítico",form_title:"Entrada de Datos de Fábrica",factory_name_label:"Nombre de Fábrica",machine_count_label:"Número de Máquinas",energy_cost_label:"Costo de Energía ($/kWh)",machine_type_label:"Tipo de Máquina",temperature_label:"Temperatura (°C)",vibration_label:"Vibración (mm/s)",load_label:"Carga (%)",submit_btn:"Analizar Fábrica",submitting:"Actualizando...",ai_panel_title:"Perspectivas IA",ai_placeholder:"Envíe datos de fábrica para generar un análisis IA.",ai_analyzing:"Analizando...",ai_risks:"Riesgos",ai_efficiency_insights:"Análisis de Eficiencia",ai_optimizations:"Sugerencias de Optimización",toast_updated:"Datos de fábrica actualizados",toast_analysis_done:"Análisis IA completo",toast_error:"Algo salió mal",nav_dashboard:"Panel",nav_factories:"Fábricas",nav_ai_insights:"Perspectivas IA",logout_btn:"Cerrar Sesión",login_title:"Bienvenido de nuevo",login_subtitle:"Inicia sesión en tu cuenta de FactoryPulse AI",ph_email:"Correo electrónico",ph_password:"Contraseña",remember_me:"Recuérdame",login_btn:"Iniciar Sesión",login_link_register:"¿No tienes cuenta? Crea una",register_title:"Crea tu cuenta",register_subtitle:"Empieza a monitorear tus fábricas con IA",ph_full_name:"Nombre Completo",ph_confirm_password:"Confirmar Contraseña",register_btn:"Crear Cuenta",register_link_login:"¿Ya tienes cuenta? Inicia sesión",err_missing_fields:"Por favor complete todos los campos",err_invalid_email:"Por favor ingrese un correo válido",err_weak_password:"La contraseña debe tener mín. 8 caracteres, una letra y un número",err_password_mismatch:"Las contraseñas no coinciden",err_invalid_credentials:"Correo o contraseña incorrectos",err_email_taken:"Este correo ya está registrado",err_generic:"Algo salió mal. Inténtalo de nuevo",my_factories_title:"Mis Fábricas",add_factory_btn:"+ Añadir Fábrica",edit_factory_btn:"Editar",delete_factory_btn:"Eliminar",confirm_delete_factory:"¿Eliminar esta fábrica? Esta acción no se puede deshacer.",no_factories_yet:"Aún no has añadido ninguna fábrica.",factory_created_toast:"Fábrica creada y analizada",factory_updated_toast:"Fábrica actualizada",factory_deleted_toast:"Fábrica eliminada",ai_insights_feed_title:"Feed de Perspectivas IA",no_ai_insights_yet:"Aún no hay perspectivas IA. Añade una fábrica.",reanalyze_btn:"Reanalizar",view_insights_btn:"Ver Perspectivas",created_label:"Creada",cancel_btn:"Cancelar",save_btn:"Guardar Cambios",role_label:"Su Rol",role_engineer:"Ingeniero",role_manager:"Gerente",role_admin:"Administrador"},
  zh: {tagline:"全球工业智能平台",live_label:"实时",kpi_energy:"能源使用量",kpi_efficiency:"效率",kpi_active:"运行中设备",kpi_alerts:"警报",kwh_unit:"kWh",chart_title:"实时性能",machine_status_title:"设备状态",status_running:"运行中",status_warning:"警告",status_critical:"严重",form_title:"工厂数据输入",factory_name_label:"工厂名称",machine_count_label:"设备数量",energy_cost_label:"能源成本 ($/kWh)",machine_type_label:"设备类型",temperature_label:"温度 (°C)",vibration_label:"振动 (mm/s)",load_label:"负载 (%)",submit_btn:"分析工厂",submitting:"更新中...",ai_panel_title:"AI 洞察",ai_placeholder:"提交工厂数据以生成AI分析。",ai_analyzing:"分析中...",ai_risks:"风险",ai_efficiency_insights:"效率分析",ai_optimizations:"优化建议",toast_updated:"工厂数据已更新",toast_analysis_done:"AI分析已完成",toast_error:"出现错误",nav_dashboard:"仪表盘",nav_factories:"工厂",nav_ai_insights:"AI洞察",logout_btn:"退出",login_title:"欢迎回来",login_subtitle:"登录您的 FactoryPulse AI 账户",ph_email:"电子邮件",ph_password:"密码",remember_me:"记住我",login_btn:"登录",login_link_register:"没有账户？创建一个",register_title:"创建账户",register_subtitle:"开始使用AI监控您的工厂",ph_full_name:"全名",ph_confirm_password:"确认密码",register_btn:"创建账户",register_link_login:"已有账户？登录",err_missing_fields:"请填写所有字段",err_invalid_email:"请输入有效的电子邮件地址",err_weak_password:"密码至少8位，需包含字母和数字",err_password_mismatch:"两次密码不一致",err_invalid_credentials:"电子邮件或密码错误",err_email_taken:"该电子邮件已被注册",err_generic:"出现错误，请重试",my_factories_title:"我的工厂",add_factory_btn:"+ 添加工厂",edit_factory_btn:"编辑",delete_factory_btn:"删除",confirm_delete_factory:"删除此工厂？此操作无法撤销。",no_factories_yet:"您还没有添加任何工厂。",factory_created_toast:"工厂已创建并分析",factory_updated_toast:"工厂已更新",factory_deleted_toast:"工厂已删除",ai_insights_feed_title:"AI洞察动态",no_ai_insights_yet:"暂无AI洞察。请添加工厂开始。",reanalyze_btn:"重新分析",view_insights_btn:"查看洞察",created_label:"创建于",cancel_btn:"取消",save_btn:"保存更改",role_label:"您的角色",role_engineer:"工程师",role_manager:"经理",role_admin:"管理员"},
  ar: {tagline:"منصة الذكاء الصناعي العالمية",live_label:"مباشر",kpi_energy:"استهلاك الطاقة",kpi_efficiency:"الكفاءة",kpi_active:"الآلات النشطة",kpi_alerts:"التنبيهات",kwh_unit:"kWh",chart_title:"الأداء في الوقت الفعلي",machine_status_title:"حالة الآلات",status_running:"تعمل",status_warning:"تحذير",status_critical:"حرج",form_title:"إدخال بيانات المصنع",factory_name_label:"اسم المصنع",machine_count_label:"عدد الآلات",energy_cost_label:"تكلفة الطاقة ($/kWh)",machine_type_label:"نوع الآلة",temperature_label:"درجة الحرارة (°C)",vibration_label:"الاهتزاز (مم/ث)",load_label:"الحمل (%)",submit_btn:"تحليل المصنع",submitting:"جارٍ التحديث...",ai_panel_title:"رؤى الذكاء الاصطناعي",ai_placeholder:"أرسل بيانات المصنع لإنشاء تحليل بالذكاء الاصطناعي.",ai_analyzing:"جارٍ التحليل...",ai_risks:"المخاطر",ai_efficiency_insights:"تحليل الكفاءة",ai_optimizations:"اقتراحات التحسين",toast_updated:"تم تحديث بيانات المصنع",toast_analysis_done:"اكتمل تحليل الذكاء الاصطناعي",toast_error:"حدث خطأ ما",nav_dashboard:"لوحة التحكم",nav_factories:"المصانع",nav_ai_insights:"رؤى الذكاء الاصطناعي",logout_btn:"تسجيل الخروج",login_title:"مرحباً بعودتك",login_subtitle:"سجل الدخول إلى حساب FactoryPulse AI الخاص بك",ph_email:"البريد الإلكتروني",ph_password:"كلمة المرور",remember_me:"تذكرني",login_btn:"تسجيل الدخول",login_link_register:"ليس لديك حساب؟ أنشئ واحداً",register_title:"إنشاء حسابك",register_subtitle:"ابدأ بمراقبة مصانعك بالذكاء الاصطناعي",ph_full_name:"الاسم الكامل",ph_confirm_password:"تأكيد كلمة المرور",register_btn:"إنشاء حساب",register_link_login:"لديك حساب بالفعل؟ سجل الدخول",err_missing_fields:"يرجى ملء جميع الحقول",err_invalid_email:"يرجى إدخال بريد إلكتروني صالح",err_weak_password:"يجب أن تكون كلمة المرور 8 أحرف على الأقل وتحتوي على حرف ورقم",err_password_mismatch:"كلمتا المرور غير متطابقتين",err_invalid_credentials:"البريد الإلكتروني أو كلمة المرور غير صحيحة",err_email_taken:"هذا البريد الإلكتروني مسجل بالفعل",err_generic:"حدث خطأ ما. يرجى المحاولة مرة أخرى",my_factories_title:"مصانعي",add_factory_btn:"+ إضافة مصنع",edit_factory_btn:"تعديل",delete_factory_btn:"حذف",confirm_delete_factory:"هل تريد حذف هذا المصنع؟ لا يمكن التراجع عن هذا.",no_factories_yet:"لم تقم بإضافة أي مصنع بعد.",factory_created_toast:"تم إنشاء المصنع وتحليله",factory_updated_toast:"تم تحديث المصنع",factory_deleted_toast:"تم حذف المصنع",ai_insights_feed_title:"موجز رؤى الذكاء الاصطناعي",no_ai_insights_yet:"لا توجد رؤى بعد. أضف مصنعاً للبدء.",reanalyze_btn:"إعادة التحليل",view_insights_btn:"عرض الرؤى",created_label:"تاريخ الإنشاء",cancel_btn:"إلغاء",save_btn:"حفظ التغييرات",role_label:"دورك",role_engineer:"مهندس",role_manager:"مدير",role_admin:"مسؤول"},
  tr: {tagline:"Küresel Endüstriyel Zeka Platformu",live_label:"Canlı",kpi_energy:"Enerji Kullanımı",kpi_efficiency:"Verimlilik",kpi_active:"Aktif Makineler",kpi_alerts:"Uyarılar",kwh_unit:"kWh",chart_title:"Gerçek Zamanlı Performans",machine_status_title:"Makine Durumu",status_running:"Çalışıyor",status_warning:"Uyarı",status_critical:"Kritik",form_title:"Fabrika Veri Girişi",factory_name_label:"Fabrika Adı",machine_count_label:"Makine Sayısı",energy_cost_label:"Enerji Maliyeti ($/kWh)",machine_type_label:"Makine Türü",temperature_label:"Sıcaklık (°C)",vibration_label:"Titreşim (mm/s)",load_label:"Yük (%)",submit_btn:"Fabrikayı Analiz Et",submitting:"Güncelleniyor...",ai_panel_title:"AI Analizleri",ai_placeholder:"AI analizi oluşturmak için fabrika verilerini gönderin.",ai_analyzing:"Analiz ediliyor...",ai_risks:"Riskler",ai_efficiency_insights:"Verimlilik Analizi",ai_optimizations:"Optimizasyon Önerileri",toast_updated:"Fabrika verileri güncellendi",toast_analysis_done:"AI analizi tamamlandı",toast_error:"Bir şeyler ters gitti",nav_dashboard:"Panel",nav_factories:"Fabrikalar",nav_ai_insights:"AI Analizleri",logout_btn:"Çıkış Yap",login_title:"Tekrar hoş geldiniz",login_subtitle:"FactoryPulse AI hesabınıza giriş yapın",ph_email:"E-posta",ph_password:"Şifre",remember_me:"Beni hatırla",login_btn:"Giriş Yap",login_link_register:"Hesabınız yok mu? Oluşturun",register_title:"Hesabınızı oluşturun",register_subtitle:"Fabrikalarınızı AI ile izlemeye başlayın",ph_full_name:"Ad Soyad",ph_confirm_password:"Şifreyi Onayla",register_btn:"Hesap Oluştur",register_link_login:"Zaten hesabınız var mı? Giriş yapın",err_missing_fields:"Lütfen tüm alanları doldurun",err_invalid_email:"Lütfen geçerli bir e-posta adresi girin",err_weak_password:"Şifre en az 8 karakter, bir harf ve bir rakam içermeli",err_password_mismatch:"Şifreler eşleşmiyor",err_invalid_credentials:"E-posta veya şifre hatalı",err_email_taken:"Bu e-posta zaten kayıtlı",err_generic:"Bir şeyler ters gitti. Tekrar deneyin",my_factories_title:"Fabrikalarım",add_factory_btn:"+ Fabrika Ekle",edit_factory_btn:"Düzenle",delete_factory_btn:"Sil",confirm_delete_factory:"Bu fabrika silinsin mi? Bu işlem geri alınamaz.",no_factories_yet:"Henüz fabrika eklemediniz.",factory_created_toast:"Fabrika oluşturuldu ve analiz edildi",factory_updated_toast:"Fabrika güncellendi",factory_deleted_toast:"Fabrika silindi",ai_insights_feed_title:"AI Analiz Akışı",no_ai_insights_yet:"Henüz AI analizi yok. Başlamak için fabrika ekleyin.",reanalyze_btn:"Yeniden Analiz Et",view_insights_btn:"Analizleri Görüntüle",created_label:"Oluşturulma",cancel_btn:"İptal",save_btn:"Değişiklikleri Kaydet",role_label:"Rolünüz",role_engineer:"Mühendis",role_manager:"Yönetici",role_admin:"Admin"},
  it: {tagline:"Piattaforma Globale di Intelligenza Industriale",live_label:"In diretta",kpi_energy:"Consumo Energetico",kpi_efficiency:"Efficienza",kpi_active:"Macchine Attive",kpi_alerts:"Avvisi",kwh_unit:"kWh",chart_title:"Prestazioni in Tempo Reale",machine_status_title:"Stato delle Macchine",status_running:"In funzione",status_warning:"Avviso",status_critical:"Critico",form_title:"Inserimento Dati Fabbrica",factory_name_label:"Nome Fabbrica",machine_count_label:"Numero di Macchine",energy_cost_label:"Costo Energia ($/kWh)",machine_type_label:"Tipo di Macchina",temperature_label:"Temperatura (°C)",vibration_label:"Vibrazione (mm/s)",load_label:"Carico (%)",submit_btn:"Analizza Fabbrica",submitting:"Aggiornamento...",ai_panel_title:"Analisi IA",ai_placeholder:"Invia i dati della fabbrica per generare un'analisi IA.",ai_analyzing:"Analisi in corso...",ai_risks:"Rischi",ai_efficiency_insights:"Analisi dell'Efficienza",ai_optimizations:"Suggerimenti di Ottimizzazione",toast_updated:"Dati fabbrica aggiornati",toast_analysis_done:"Analisi IA completata",toast_error:"Qualcosa è andato storto",nav_dashboard:"Dashboard",nav_factories:"Fabbriche",nav_ai_insights:"Analisi IA",logout_btn:"Esci",login_title:"Bentornato",login_subtitle:"Accedi al tuo account FactoryPulse AI",ph_email:"Email",ph_password:"Password",remember_me:"Ricordami",login_btn:"Accedi",login_link_register:"Non hai un account? Creane uno",register_title:"Crea il tuo account",register_subtitle:"Inizia a monitorare le tue fabbriche con l'IA",ph_full_name:"Nome Completo",ph_confirm_password:"Conferma Password",register_btn:"Crea Account",register_link_login:"Hai già un account? Accedi",err_missing_fields:"Si prega di compilare tutti i campi",err_invalid_email:"Inserisci un indirizzo email valido",err_weak_password:"La password deve avere almeno 8 caratteri, una lettera e un numero",err_password_mismatch:"Le password non corrispondono",err_invalid_credentials:"Email o password errati",err_email_taken:"Questa email è già registrata",err_generic:"Qualcosa è andato storto. Riprova",my_factories_title:"Le Mie Fabbriche",add_factory_btn:"+ Aggiungi Fabbrica",edit_factory_btn:"Modifica",delete_factory_btn:"Elimina",confirm_delete_factory:"Eliminare questa fabbrica? Questa azione non può essere annullata.",no_factories_yet:"Non hai ancora aggiunto nessuna fabbrica.",factory_created_toast:"Fabbrica creata e analizzata",factory_updated_toast:"Fabbrica aggiornata",factory_deleted_toast:"Fabbrica eliminata",ai_insights_feed_title:"Feed di Analisi IA",no_ai_insights_yet:"Nessuna analisi IA ancora. Aggiungi una fabbrica.",reanalyze_btn:"Rianalizza",view_insights_btn:"Vedi Analisi",created_label:"Creata il",cancel_btn:"Annulla",save_btn:"Salva Modifiche",role_label:"Il Tuo Ruolo",role_engineer:"Ingegnere",role_manager:"Manager",role_admin:"Amministratore"},
  pt: {tagline:"Plataforma Global de Inteligência Industrial",live_label:"Ao vivo",kpi_energy:"Uso de Energia",kpi_efficiency:"Eficiência",kpi_active:"Máquinas Ativas",kpi_alerts:"Alertas",kwh_unit:"kWh",chart_title:"Desempenho em Tempo Real",machine_status_title:"Status das Máquinas",status_running:"Em funcionamento",status_warning:"Aviso",status_critical:"Crítico",form_title:"Entrada de Dados da Fábrica",factory_name_label:"Nome da Fábrica",machine_count_label:"Número de Máquinas",energy_cost_label:"Custo de Energia ($/kWh)",machine_type_label:"Tipo de Máquina",temperature_label:"Temperatura (°C)",vibration_label:"Vibração (mm/s)",load_label:"Carga (%)",submit_btn:"Analisar Fábrica",submitting:"Atualizando...",ai_panel_title:"Insights de IA",ai_placeholder:"Envie os dados da fábrica para gerar uma análise de IA.",ai_analyzing:"Analisando...",ai_risks:"Riscos",ai_efficiency_insights:"Análise de Eficiência",ai_optimizations:"Sugestões de Otimização",toast_updated:"Dados da fábrica atualizados",toast_analysis_done:"Análise de IA concluída",toast_error:"Algo deu errado",nav_dashboard:"Painel",nav_factories:"Fábricas",nav_ai_insights:"Insights de IA",logout_btn:"Sair",login_title:"Bem-vindo de volta",login_subtitle:"Entre na sua conta FactoryPulse AI",ph_email:"E-mail",ph_password:"Senha",remember_me:"Lembrar de mim",login_btn:"Entrar",login_link_register:"Não tem conta? Crie uma",register_title:"Crie sua conta",register_subtitle:"Comece a monitorar suas fábricas com IA",ph_full_name:"Nome Completo",ph_confirm_password:"Confirmar Senha",register_btn:"Criar Conta",register_link_login:"Já tem conta? Entrar",err_missing_fields:"Por favor preencha todos os campos",err_invalid_email:"Por favor insira um e-mail válido",err_weak_password:"A senha deve ter no mínimo 8 caracteres, uma letra e um número",err_password_mismatch:"As senhas não coincidem",err_invalid_credentials:"E-mail ou senha incorretos",err_email_taken:"Este e-mail já está registrado",err_generic:"Algo deu errado. Tente novamente",my_factories_title:"Minhas Fábricas",add_factory_btn:"+ Adicionar Fábrica",edit_factory_btn:"Editar",delete_factory_btn:"Excluir",confirm_delete_factory:"Excluir esta fábrica? Esta ação não pode ser desfeita.",no_factories_yet:"Você ainda não adicionou nenhuma fábrica.",factory_created_toast:"Fábrica criada e analisada",factory_updated_toast:"Fábrica atualizada",factory_deleted_toast:"Fábrica excluída",ai_insights_feed_title:"Feed de Insights de IA",no_ai_insights_yet:"Ainda sem insights de IA. Adicione uma fábrica.",reanalyze_btn:"Reanalisar",view_insights_btn:"Ver Insights",created_label:"Criada em",cancel_btn:"Cancelar",save_btn:"Salvar Alterações",role_label:"Sua Função",role_engineer:"Engenheiro",role_manager:"Gerente",role_admin:"Administrador"},
  ja: {tagline:"グローバル産業インテリジェンスプラットフォーム",live_label:"ライブ",kpi_energy:"エネルギー使用量",kpi_efficiency:"効率",kpi_active:"稼働中の機械",kpi_alerts:"アラート",kwh_unit:"kWh",chart_title:"リアルタイムパフォーマンス",machine_status_title:"機械の状態",status_running:"稼働中",status_warning:"警告",status_critical:"重大",form_title:"工場データ入力",factory_name_label:"工場名",machine_count_label:"機械の数",energy_cost_label:"エネルギーコスト ($/kWh)",machine_type_label:"機械の種類",temperature_label:"温度 (°C)",vibration_label:"振動 (mm/s)",load_label:"負荷 (%)",submit_btn:"工場を分析",submitting:"更新中...",ai_panel_title:"AIインサイト",ai_placeholder:"工場データを送信してAI分析を生成してください。",ai_analyzing:"分析中...",ai_risks:"リスク",ai_efficiency_insights:"効率分析",ai_optimizations:"最適化提案",toast_updated:"工場データが更新されました",toast_analysis_done:"AI分析が完了しました",toast_error:"問題が発生しました",nav_dashboard:"ダッシュボード",nav_factories:"工場",nav_ai_insights:"AIインサイト",logout_btn:"ログアウト",login_title:"おかえりなさい",login_subtitle:"FactoryPulse AI アカウントにログイン",ph_email:"メールアドレス",ph_password:"パスワード",remember_me:"ログイン状態を保持",login_btn:"ログイン",login_link_register:"アカウントをお持ちでないですか？作成する",register_title:"アカウントを作成",register_subtitle:"AIで工場の監視を始めましょう",ph_full_name:"氏名",ph_confirm_password:"パスワードの確認",register_btn:"アカウント作成",register_link_login:"すでにアカウントをお持ちですか？ログイン",err_missing_fields:"すべての項目を入力してください",err_invalid_email:"有効なメールアドレスを入力してください",err_weak_password:"パスワードは8文字以上で、文字と数字を含める必要があります",err_password_mismatch:"パスワードが一致しません",err_invalid_credentials:"メールアドレスまたはパスワードが正しくありません",err_email_taken:"このメールアドレスは既に登録されています",err_generic:"エラーが発生しました。再試行してください",my_factories_title:"マイ工場",add_factory_btn:"+ 工場を追加",edit_factory_btn:"編集",delete_factory_btn:"削除",confirm_delete_factory:"この工場を削除しますか？元に戻せません。",no_factories_yet:"まだ工場を追加していません。",factory_created_toast:"工場が作成・分析されました",factory_updated_toast:"工場が更新されました",factory_deleted_toast:"工場が削除されました",ai_insights_feed_title:"AIインサイトフィード",no_ai_insights_yet:"AIインサイトはまだありません。工場を追加してください。",reanalyze_btn:"再分析",view_insights_btn:"インサイトを見る",created_label:"作成日",cancel_btn:"キャンセル",save_btn:"変更を保存",role_label:"あなたの役割",role_engineer:"エンジニア",role_manager:"マネージャー",role_admin:"管理者"},
  ko: {tagline:"글로벌 산업 인텔리전스 플랫폼",live_label:"실시간",kpi_energy:"에너지 사용량",kpi_efficiency:"효율성",kpi_active:"가동 중인 기계",kpi_alerts:"경고",kwh_unit:"kWh",chart_title:"실시간 성능",machine_status_title:"기계 상태",status_running:"가동 중",status_warning:"경고",status_critical:"심각",form_title:"공장 데이터 입력",factory_name_label:"공장 이름",machine_count_label:"기계 수",energy_cost_label:"에너지 비용 ($/kWh)",machine_type_label:"기계 유형",temperature_label:"온도 (°C)",vibration_label:"진동 (mm/s)",load_label:"부하 (%)",submit_btn:"공장 분석",submitting:"업데이트 중...",ai_panel_title:"AI 인사이트",ai_placeholder:"AI 분석을 생성하려면 공장 데이터를 제출하세요.",ai_analyzing:"분석 중...",ai_risks:"위험 요소",ai_efficiency_insights:"효율성 분석",ai_optimizations:"최적화 제안",toast_updated:"공장 데이터가 업데이트되었습니다",toast_analysis_done:"AI 분석이 완료되었습니다",toast_error:"문제가 발생했습니다",nav_dashboard:"대시보드",nav_factories:"공장",nav_ai_insights:"AI 인사이트",logout_btn:"로그아웃",login_title:"다시 오신 것을 환영합니다",login_subtitle:"FactoryPulse AI 계정에 로그인하세요",ph_email:"이메일",ph_password:"비밀번호",remember_me:"로그인 상태 유지",login_btn:"로그인",login_link_register:"계정이 없으신가요? 계정 만들기",register_title:"계정 만들기",register_subtitle:"AI로 공장 모니터링을 시작하세요",ph_full_name:"성명",ph_confirm_password:"비밀번호 확인",register_btn:"계정 생성",register_link_login:"이미 계정이 있으신가요? 로그인",err_missing_fields:"모든 항목을 입력해주세요",err_invalid_email:"유효한 이메일 주소를 입력하세요",err_weak_password:"비밀번호는 8자 이상, 문자와 숫자를 포함해야 합니다",err_password_mismatch:"비밀번호가 일치하지 않습니다",err_invalid_credentials:"이메일 또는 비밀번호가 올바르지 않습니다",err_email_taken:"이미 등록된 이메일입니다",err_generic:"문제가 발생했습니다. 다시 시도해주세요",my_factories_title:"내 공장",add_factory_btn:"+ 공장 추가",edit_factory_btn:"수정",delete_factory_btn:"삭제",confirm_delete_factory:"이 공장을 삭제하시겠습니까? 되돌릴 수 없습니다.",no_factories_yet:"아직 추가된 공장이 없습니다.",factory_created_toast:"공장이 생성되고 분석되었습니다",factory_updated_toast:"공장이 업데이트되었습니다",factory_deleted_toast:"공장이 삭제되었습니다",ai_insights_feed_title:"AI 인사이트 피드",no_ai_insights_yet:"아직 AI 인사이트가 없습니다. 공장을 추가하세요.",reanalyze_btn:"다시 분석",view_insights_btn:"인사이트 보기",created_label:"생성일",cancel_btn:"취소",save_btn:"변경사항 저장",role_label:"귀하의 역할",role_engineer:"엔지니어",role_manager:"매니저",role_admin:"관리자"},
  hi: {tagline:"वैश्विक औद्योगिक बुद्धिमत्ता मंच",live_label:"लाइव",kpi_energy:"ऊर्जा उपयोग",kpi_efficiency:"दक्षता",kpi_active:"सक्रिय मशीनें",kpi_alerts:"अलर्ट",kwh_unit:"kWh",chart_title:"रीयल-टाइम प्रदर्शन",machine_status_title:"मशीन की स्थिति",status_running:"चल रहा है",status_warning:"चेतावनी",status_critical:"गंभीर",form_title:"फ़ैक्टरी डेटा इनपुट",factory_name_label:"फ़ैक्टरी का नाम",machine_count_label:"मशीनों की संख्या",energy_cost_label:"ऊर्जा लागत ($/kWh)",machine_type_label:"मशीन प्रकार",temperature_label:"तापमान (°C)",vibration_label:"कंपन (mm/s)",load_label:"लोड (%)",submit_btn:"फ़ैक्टरी का विश्लेषण करें",submitting:"अद्यतन हो रहा है...",ai_panel_title:"AI अंतर्दृष्टि",ai_placeholder:"AI विश्लेषण उत्पन्न करने के लिए फ़ैक्टरी डेटा सबमिट करें।",ai_analyzing:"विश्लेषण हो रहा है...",ai_risks:"जोखिम",ai_efficiency_insights:"दक्षता विश्लेषण",ai_optimizations:"अनुकूलन सुझाव",toast_updated:"फ़ैक्टरी डेटा अपडेट किया गया",toast_analysis_done:"AI विश्लेषण पूर्ण हुआ",toast_error:"कुछ गलत हो गया",nav_dashboard:"डैशबोर्ड",nav_factories:"फ़ैक्टरियाँ",nav_ai_insights:"AI अंतर्दृष्टि",logout_btn:"लॉग आउट",login_title:"वापसी पर स्वागत है",login_subtitle:"अपने FactoryPulse AI खाते में लॉग इन करें",ph_email:"ईमेल",ph_password:"पासवर्ड",remember_me:"मुझे याद रखें",login_btn:"लॉग इन करें",login_link_register:"खाता नहीं है? एक बनाएं",register_title:"अपना खाता बनाएं",register_subtitle:"AI के साथ अपनी फ़ैक्टरियों की निगरानी शुरू करें",ph_full_name:"पूरा नाम",ph_confirm_password:"पासवर्ड की पुष्टि करें",register_btn:"खाता बनाएं",register_link_login:"पहले से खाता है? लॉग इन करें",err_missing_fields:"कृपया सभी फ़ील्ड भरें",err_invalid_email:"कृपया एक मान्य ईमेल पता दर्ज करें",err_weak_password:"पासवर्ड कम से कम 8 अक्षर, एक अक्षर और एक अंक होना चाहिए",err_password_mismatch:"पासवर्ड मेल नहीं खाते",err_invalid_credentials:"गलत ईमेल या पासवर्ड",err_email_taken:"यह ईमेल पहले से पंजीकृत है",err_generic:"कुछ गलत हो गया। कृपया पुनः प्रयास करें",my_factories_title:"मेरी फ़ैक्टरियाँ",add_factory_btn:"+ फ़ैक्टरी जोड़ें",edit_factory_btn:"संपादित करें",delete_factory_btn:"हटाएं",confirm_delete_factory:"इस फ़ैक्टरी को हटाएं? इसे पूर्ववत नहीं किया जा सकता।",no_factories_yet:"आपने अभी तक कोई फ़ैक्टरी नहीं जोड़ी है।",factory_created_toast:"फ़ैक्टरी बनाई और विश्लेषित की गई",factory_updated_toast:"फ़ैक्टरी अपडेट की गई",factory_deleted_toast:"फ़ैक्टरी हटाई गई",ai_insights_feed_title:"AI अंतर्दृष्टि फ़ीड",no_ai_insights_yet:"अभी तक कोई AI अंतर्दृष्टि नहीं। शुरू करने के लिए एक फ़ैक्टरी जोड़ें।",reanalyze_btn:"पुनः विश्लेषण करें",view_insights_btn:"अंतर्दृष्टि देखें",created_label:"बनाया गया",cancel_btn:"रद्द करें",save_btn:"परिवर्तन सहेजें",role_label:"आपकी भूमिका",role_engineer:"इंजीनियर",role_manager:"प्रबंधक",role_admin:"व्यवस्थापक"},
  uz: {tagline:"Global sanoat intellekti platformasi",live_label:"Jonli",kpi_energy:"Energiya sarfi",kpi_efficiency:"Samaradorlik",kpi_active:"Faol stanoklar",kpi_alerts:"Ogohlantirishlar",kwh_unit:"kWh",chart_title:"Real vaqtdagi ko'rsatkichlar",machine_status_title:"Stanoklar holati",status_running:"Ishlamoqda",status_warning:"Ogohlantirish",status_critical:"Muhim",form_title:"Zavod ma'lumotlarini kiritish",factory_name_label:"Zavod nomi",machine_count_label:"Stanoklar soni",energy_cost_label:"Energiya narxi ($/kWh)",machine_type_label:"Stanok turi",temperature_label:"Harorat (°C)",vibration_label:"Tebranish (mm/s)",load_label:"Yuklama (%)",submit_btn:"Zavodni tahlil qilish",submitting:"Yangilanmoqda...",ai_panel_title:"AI tahlili",ai_placeholder:"AI tahlilini olish uchun zavod ma'lumotlarini yuboring.",ai_analyzing:"Tahlil qilinmoqda...",ai_risks:"Xavflar",ai_efficiency_insights:"Samaradorlik tahlili",ai_optimizations:"Optimallashtirish tavsiyalari",toast_updated:"Zavod ma'lumotlari yangilandi",toast_analysis_done:"AI tahlili yakunlandi",toast_error:"Xatolik yuz berdi",nav_dashboard:"Boshqaruv paneli",nav_factories:"Zavodlar",nav_ai_insights:"AI tahlili",logout_btn:"Chiqish",login_title:"Xush kelibsiz",login_subtitle:"FactoryPulse AI hisobingizga kiring",ph_email:"Elektron pochta",ph_password:"Parol",remember_me:"Meni eslab qol",login_btn:"Kirish",login_link_register:"Hisobingiz yo'qmi? Yarating",register_title:"Hisob yarating",register_subtitle:"Zavodlaringizni AI bilan kuzatishni boshlang",ph_full_name:"To'liq ism",ph_confirm_password:"Parolni tasdiqlang",register_btn:"Hisob yaratish",register_link_login:"Hisobingiz bormi? Kiring",err_missing_fields:"Barcha maydonlarni to'ldiring",err_invalid_email:"Yaroqli elektron pochta manzilini kiriting",err_weak_password:"Parol kamida 8 belgidan, harf va raqamdan iborat bo'lishi kerak",err_password_mismatch:"Parollar mos kelmaydi",err_invalid_credentials:"Elektron pochta yoki parol noto'g'ri",err_email_taken:"Bu elektron pochta allaqachon ro'yxatdan o'tgan",err_generic:"Xatolik yuz berdi. Qaytadan urinib ko'ring",my_factories_title:"Mening Zavodlarim",add_factory_btn:"+ Zavod qo'shish",edit_factory_btn:"Tahrirlash",delete_factory_btn:"O'chirish",confirm_delete_factory:"Bu zavodni o'chirasizmi? Buni bekor qilib bo'lmaydi.",no_factories_yet:"Siz hali hech qanday zavod qo'shmagansiz.",factory_created_toast:"Zavod yaratildi va tahlil qilindi",factory_updated_toast:"Zavod yangilandi",factory_deleted_toast:"Zavod o'chirildi",ai_insights_feed_title:"AI Tahlili Lentasi",no_ai_insights_yet:"Hali AI tahlili yo'q. Boshlash uchun zavod qo'shing.",reanalyze_btn:"Qayta tahlil qilish",view_insights_btn:"Tahlilni ko'rish",created_label:"Yaratilgan",cancel_btn:"Bekor qilish",save_btn:"O'zgarishlarni saqlash",role_label:"Sizning rolingiz",role_engineer:"Muhandis",role_manager:"Menejer",role_admin:"Administrator"},
  ky: {tagline:"Глобалдык өнөр жай интеллект платформасы",live_label:"Түз эфир",kpi_energy:"Энергия сарпталышы",kpi_efficiency:"Эффективдүүлүк",kpi_active:"Активдүү станоктор",kpi_alerts:"Дабылдар",kwh_unit:"кВт·саат",chart_title:"Реалдуу убакыттагы көрсөткүчтөр",machine_status_title:"Станоктордун абалы",status_running:"Иштеп жатат",status_warning:"Эскертүү",status_critical:"Олуттуу",form_title:"Завод маалыматтарын киргизүү",factory_name_label:"Заводдун аты",machine_count_label:"Станоктордун саны",energy_cost_label:"Энергия наркы ($/кВт·саат)",machine_type_label:"Станоктун түрү",temperature_label:"Температура (°C)",vibration_label:"Дирилдөө (мм/с)",load_label:"Жүктөм (%)",submit_btn:"Заводду талдоо",submitting:"Жаңыртылууда...",ai_panel_title:"AI-талдоо",ai_placeholder:"AI-талдоо алуу үчүн завод маалыматтарын жөнөтүңүз.",ai_analyzing:"Талдануда...",ai_risks:"Тобокелдиктер",ai_efficiency_insights:"Эффективдүүлүк талдоосу",ai_optimizations:"Оптималдаштыруу сунуштары",toast_updated:"Завод маалыматтары жаңыртылды",toast_analysis_done:"AI-талдоо аяктады",toast_error:"Ката кетти",nav_dashboard:"Башкаруу панели",nav_factories:"Заводдор",nav_ai_insights:"AI-талдоо",logout_btn:"Чыгуу",login_title:"Кайра кош келиңиз",login_subtitle:"FactoryPulse AI каттоо эсебиңизге кириңиз",ph_email:"Электрондук почта",ph_password:"Сырсөз",remember_me:"Мени эстеп кал",login_btn:"Кирүү",login_link_register:"Каттоо эсебиңиз жокпу? Түзүү",register_title:"Каттоо эсебин түзүү",register_subtitle:"Заводдоруңузду AI менен байкоону баштаңыз",ph_full_name:"Толук аты-жөнү",ph_confirm_password:"Сырсөздү ырастаңыз",register_btn:"Каттоо эсебин түзүү",register_link_login:"Каттоо эсебиңиз барбы? Кирүү",err_missing_fields:"Бардык талааларды толтуруңуз",err_invalid_email:"Жарактуу электрондук почта дарегин киргизиңиз",err_weak_password:"Сырсөз кеминде 8 белги, тамга жана сан камтышы керек",err_password_mismatch:"Сырсөздөр дал келбейт",err_invalid_credentials:"Электрондук почта же сырсөз туура эмес",err_email_taken:"Бул электрондук почта мурунтан катталган",err_generic:"Ката кетти. Кайра аракет кылыңыз",my_factories_title:"Менин Заводдорум",add_factory_btn:"+ Завод кошуу",edit_factory_btn:"Түзөтүү",delete_factory_btn:"Өчүрүү",confirm_delete_factory:"Бул заводду өчүрөсүзбү? Бул аракетти артка кайтарууга болбойт.",no_factories_yet:"Сиз азырынча эч кандай завод кошкон жоксуз.",factory_created_toast:"Завод түзүлдү жана талданды",factory_updated_toast:"Завод жаңыртылды",factory_deleted_toast:"Завод өчүрүлдү",ai_insights_feed_title:"AI-талдоо тизмеси",no_ai_insights_yet:"Азырынча AI-талдоо жок. Баштоо үчүн завод кошуңуз.",reanalyze_btn:"Кайра талдоо",view_insights_btn:"Талдоону көрүү",created_label:"Түзүлгөн күнү",cancel_btn:"Жокко чыгаруу",save_btn:"Өзгөртүүлөрдү сактоо",role_label:"Сиздин ролуңуз",role_engineer:"Инженер",role_manager:"Менеджер",role_admin:"Администратор"},
  uk: {tagline:"Глобальна платформа промислового інтелекту",live_label:"Наживо",kpi_energy:"Споживання енергії",kpi_efficiency:"Ефективність",kpi_active:"Активні верстати",kpi_alerts:"Сповіщення",kwh_unit:"кВт·год",chart_title:"Показники в реальному часі",machine_status_title:"Статус верстатів",status_running:"Працює",status_warning:"Попередження",status_critical:"Критично",form_title:"Введення даних заводу",factory_name_label:"Назва заводу",machine_count_label:"Кількість верстатів",energy_cost_label:"Вартість енергії ($/кВт·год)",machine_type_label:"Тип верстата",temperature_label:"Температура (°C)",vibration_label:"Вібрація (мм/с)",load_label:"Навантаження (%)",submit_btn:"Аналізувати завод",submitting:"Оновлення...",ai_panel_title:"AI-аналітика",ai_placeholder:"Надішліть дані заводу, щоб отримати AI-аналіз.",ai_analyzing:"Аналіз...",ai_risks:"Ризики",ai_efficiency_insights:"Аналіз ефективності",ai_optimizations:"Рекомендації з оптимізації",toast_updated:"Дані заводу оновлено",toast_analysis_done:"AI-аналіз завершено",toast_error:"Сталася помилка",nav_dashboard:"Панель",nav_factories:"Заводи",nav_ai_insights:"AI-аналітика",logout_btn:"Вийти",login_title:"З поверненням",login_subtitle:"Увійдіть у свій обліковий запис FactoryPulse AI",ph_email:"Електронна пошта",ph_password:"Пароль",remember_me:"Запам'ятати мене",login_btn:"Увійти",login_link_register:"Немає акаунту? Створити",register_title:"Створіть акаунт",register_subtitle:"Почніть моніторинг заводів за допомогою AI",ph_full_name:"Повне ім'я",ph_confirm_password:"Підтвердіть пароль",register_btn:"Створити акаунт",register_link_login:"Вже є акаунт? Увійти",err_missing_fields:"Будь ласка, заповніть усі поля",err_invalid_email:"Введіть дійсну електронну адресу",err_weak_password:"Пароль має містити щонайменше 8 символів, літеру та цифру",err_password_mismatch:"Паролі не збігаються",err_invalid_credentials:"Невірна електронна пошта або пароль",err_email_taken:"Ця електронна пошта вже зареєстрована",err_generic:"Сталася помилка. Спробуйте ще раз",my_factories_title:"Мої Заводи",add_factory_btn:"+ Додати завод",edit_factory_btn:"Редагувати",delete_factory_btn:"Видалити",confirm_delete_factory:"Видалити цей завод? Цю дію не можна скасувати.",no_factories_yet:"Ви ще не додали жодного заводу.",factory_created_toast:"Завод створено та проаналізовано",factory_updated_toast:"Завод оновлено",factory_deleted_toast:"Завод видалено",ai_insights_feed_title:"Стрічка AI-аналітики",no_ai_insights_yet:"Ще немає AI-аналітики. Додайте завод.",reanalyze_btn:"Проаналізувати знову",view_insights_btn:"Переглянути аналітику",created_label:"Створено",cancel_btn:"Скасувати",save_btn:"Зберегти зміни",role_label:"Ваша роль",role_engineer:"Інженер",role_manager:"Менеджер",role_admin:"Адміністратор"},
  pl: {tagline:"Globalna Platforma Inteligencji Przemysłowej",live_label:"Na żywo",kpi_energy:"Zużycie Energii",kpi_efficiency:"Wydajność",kpi_active:"Aktywne Maszyny",kpi_alerts:"Alerty",kwh_unit:"kWh",chart_title:"Wydajność w Czasie Rzeczywistym",machine_status_title:"Status Maszyn",status_running:"Działa",status_warning:"Ostrzeżenie",status_critical:"Krytyczne",form_title:"Wprowadzanie Danych Fabryki",factory_name_label:"Nazwa Fabryki",machine_count_label:"Liczba Maszyn",energy_cost_label:"Koszt Energii ($/kWh)",machine_type_label:"Typ Maszyny",temperature_label:"Temperatura (°C)",vibration_label:"Wibracje (mm/s)",load_label:"Obciążenie (%)",submit_btn:"Analizuj Fabrykę",submitting:"Aktualizowanie...",ai_panel_title:"Analizy AI",ai_placeholder:"Prześlij dane fabryki, aby wygenerować analizę AI.",ai_analyzing:"Analizowanie...",ai_risks:"Ryzyka",ai_efficiency_insights:"Analiza Wydajności",ai_optimizations:"Sugestie Optymalizacji",toast_updated:"Dane fabryki zaktualizowane",toast_analysis_done:"Analiza AI zakończona",toast_error:"Coś poszło nie tak",nav_dashboard:"Panel",nav_factories:"Fabryki",nav_ai_insights:"Analizy AI",logout_btn:"Wyloguj",login_title:"Witamy z powrotem",login_subtitle:"Zaloguj się do swojego konta FactoryPulse AI",ph_email:"E-mail",ph_password:"Hasło",remember_me:"Zapamiętaj mnie",login_btn:"Zaloguj się",login_link_register:"Nie masz konta? Utwórz je",register_title:"Utwórz konto",register_subtitle:"Zacznij monitorować swoje fabryki z AI",ph_full_name:"Imię i Nazwisko",ph_confirm_password:"Potwierdź Hasło",register_btn:"Utwórz Konto",register_link_login:"Masz już konto? Zaloguj się",err_missing_fields:"Proszę wypełnić wszystkie pola",err_invalid_email:"Proszę podać prawidłowy adres e-mail",err_weak_password:"Hasło musi mieć min. 8 znaków, literę i cyfrę",err_password_mismatch:"Hasła nie pasują do siebie",err_invalid_credentials:"Nieprawidłowy e-mail lub hasło",err_email_taken:"Ten e-mail jest już zarejestrowany",err_generic:"Coś poszło nie tak. Spróbuj ponownie",my_factories_title:"Moje Fabryki",add_factory_btn:"+ Dodaj Fabrykę",edit_factory_btn:"Edytuj",delete_factory_btn:"Usuń",confirm_delete_factory:"Usunąć tę fabrykę? Tej czynności nie można cofnąć.",no_factories_yet:"Nie dodałeś jeszcze żadnej fabryki.",factory_created_toast:"Fabryka utworzona i przeanalizowana",factory_updated_toast:"Fabryka zaktualizowana",factory_deleted_toast:"Fabryka usunięta",ai_insights_feed_title:"Kanał Analiz AI",no_ai_insights_yet:"Brak analiz AI. Dodaj fabrykę, aby zacząć.",reanalyze_btn:"Analizuj Ponownie",view_insights_btn:"Zobacz Analizy",created_label:"Utworzono",cancel_btn:"Anuluj",save_btn:"Zapisz Zmiany",role_label:"Twoja Rola",role_engineer:"Inżynier",role_manager:"Menedżer",role_admin:"Administrator"},
  nl: {tagline:"Wereldwijd Industrieel Intelligentieplatform",live_label:"Live",kpi_energy:"Energieverbruik",kpi_efficiency:"Efficiëntie",kpi_active:"Actieve Machines",kpi_alerts:"Meldingen",kwh_unit:"kWh",chart_title:"Realtime Prestaties",machine_status_title:"Machinestatus",status_running:"Actief",status_warning:"Waarschuwing",status_critical:"Kritiek",form_title:"Fabrieksgegevens Invoeren",factory_name_label:"Fabrieksnaam",machine_count_label:"Aantal Machines",energy_cost_label:"Energiekosten ($/kWh)",machine_type_label:"Machinetype",temperature_label:"Temperatuur (°C)",vibration_label:"Trilling (mm/s)",load_label:"Belasting (%)",submit_btn:"Fabriek Analyseren",submitting:"Bijwerken...",ai_panel_title:"AI-inzichten",ai_placeholder:"Verzend fabrieksgegevens om een AI-analyse te genereren.",ai_analyzing:"Analyseren...",ai_risks:"Risico's",ai_efficiency_insights:"Efficiëntieanalyse",ai_optimizations:"Optimalisatiesuggesties",toast_updated:"Fabrieksgegevens bijgewerkt",toast_analysis_done:"AI-analyse voltooid",toast_error:"Er is iets misgegaan",nav_dashboard:"Dashboard",nav_factories:"Fabrieken",nav_ai_insights:"AI-inzichten",logout_btn:"Uitloggen",login_title:"Welkom terug",login_subtitle:"Log in op uw FactoryPulse AI-account",ph_email:"E-mail",ph_password:"Wachtwoord",remember_me:"Onthoud mij",login_btn:"Inloggen",login_link_register:"Geen account? Maak er een",register_title:"Maak uw account aan",register_subtitle:"Begin met AI-monitoring van uw fabrieken",ph_full_name:"Volledige Naam",ph_confirm_password:"Bevestig Wachtwoord",register_btn:"Account Aanmaken",register_link_login:"Heeft u al een account? Inloggen",err_missing_fields:"Vul alle velden in",err_invalid_email:"Voer een geldig e-mailadres in",err_weak_password:"Wachtwoord moet minimaal 8 tekens, een letter en een cijfer bevatten",err_password_mismatch:"Wachtwoorden komen niet overeen",err_invalid_credentials:"Ongeldige e-mail of wachtwoord",err_email_taken:"Dit e-mailadres is al geregistreerd",err_generic:"Er is iets misgegaan. Probeer het opnieuw",my_factories_title:"Mijn Fabrieken",add_factory_btn:"+ Fabriek Toevoegen",edit_factory_btn:"Bewerken",delete_factory_btn:"Verwijderen",confirm_delete_factory:"Deze fabriek verwijderen? Dit kan niet ongedaan worden gemaakt.",no_factories_yet:"U heeft nog geen fabrieken toegevoegd.",factory_created_toast:"Fabriek aangemaakt en geanalyseerd",factory_updated_toast:"Fabriek bijgewerkt",factory_deleted_toast:"Fabriek verwijderd",ai_insights_feed_title:"AI-inzichten Feed",no_ai_insights_yet:"Nog geen AI-inzichten. Voeg een fabriek toe.",reanalyze_btn:"Opnieuw Analyseren",view_insights_btn:"Bekijk Inzichten",created_label:"Aangemaakt",cancel_btn:"Annuleren",save_btn:"Wijzigingen Opslaan",role_label:"Uw Rol",role_engineer:"Ingenieur",role_manager:"Manager",role_admin:"Beheerder"},
  sv: {tagline:"Global Industriell Intelligensplattform",live_label:"Live",kpi_energy:"Energiförbrukning",kpi_efficiency:"Effektivitet",kpi_active:"Aktiva Maskiner",kpi_alerts:"Varningar",kwh_unit:"kWh",chart_title:"Realtidsprestanda",machine_status_title:"Maskinstatus",status_running:"Igång",status_warning:"Varning",status_critical:"Kritisk",form_title:"Fabriksdatainmatning",factory_name_label:"Fabriksnamn",machine_count_label:"Antal Maskiner",energy_cost_label:"Energikostnad ($/kWh)",machine_type_label:"Maskintyp",temperature_label:"Temperatur (°C)",vibration_label:"Vibration (mm/s)",load_label:"Belastning (%)",submit_btn:"Analysera Fabrik",submitting:"Uppdaterar...",ai_panel_title:"AI-insikter",ai_placeholder:"Skicka fabriksdata för att generera en AI-analys.",ai_analyzing:"Analyserar...",ai_risks:"Risker",ai_efficiency_insights:"Effektivitetsanalys",ai_optimizations:"Optimeringsförslag",toast_updated:"Fabriksdata uppdaterad",toast_analysis_done:"AI-analys klar",toast_error:"Något gick fel",nav_dashboard:"Instrumentpanel",nav_factories:"Fabriker",nav_ai_insights:"AI-insikter",logout_btn:"Logga ut",login_title:"Välkommen tillbaka",login_subtitle:"Logga in på ditt FactoryPulse AI-konto",ph_email:"E-post",ph_password:"Lösenord",remember_me:"Kom ihåg mig",login_btn:"Logga in",login_link_register:"Inget konto? Skapa ett",register_title:"Skapa ditt konto",register_subtitle:"Börja övervaka dina fabriker med AI",ph_full_name:"Fullständigt Namn",ph_confirm_password:"Bekräfta Lösenord",register_btn:"Skapa Konto",register_link_login:"Har du redan ett konto? Logga in",err_missing_fields:"Vänligen fyll i alla fält",err_invalid_email:"Ange en giltig e-postadress",err_weak_password:"Lösenordet måste vara minst 8 tecken med en bokstav och en siffra",err_password_mismatch:"Lösenorden matchar inte",err_invalid_credentials:"Felaktig e-post eller lösenord",err_email_taken:"Denna e-post är redan registrerad",err_generic:"Något gick fel. Försök igen",my_factories_title:"Mina Fabriker",add_factory_btn:"+ Lägg till Fabrik",edit_factory_btn:"Redigera",delete_factory_btn:"Ta bort",confirm_delete_factory:"Ta bort denna fabrik? Detta kan inte ångras.",no_factories_yet:"Du har inte lagt till några fabriker än.",factory_created_toast:"Fabrik skapad och analyserad",factory_updated_toast:"Fabrik uppdaterad",factory_deleted_toast:"Fabrik borttagen",ai_insights_feed_title:"AI-insikter Flöde",no_ai_insights_yet:"Inga AI-insikter än. Lägg till en fabrik.",reanalyze_btn:"Analysera Igen",view_insights_btn:"Visa Insikter",created_label:"Skapad",cancel_btn:"Avbryt",save_btn:"Spara Ändringar",role_label:"Din Roll",role_engineer:"Ingenjör",role_manager:"Chef",role_admin:"Administratör"},
};

let currentLang = localStorage.getItem("fp_lang") || "en";
if (!translations[currentLang]) currentLang = "en";
const RTL_LANGS = ["ar"];
function t(key) { return (translations[currentLang] && translations[currentLang][key]) || translations.en[key] || key; }
function applyTranslations() {
  document.documentElement.lang = currentLang;
  document.documentElement.dir = RTL_LANGS.includes(currentLang) ? "rtl" : "ltr";
  document.querySelectorAll("[data-t]").forEach(el => { el.textContent = t(el.getAttribute("data-t")); });
}
function buildLangSelector() {
  const langNames = { en:"English", ru:"\u0420\u0443\u0441\u0441\u043a\u0438\u0439", kk:"\u049a\u0430\u0437\u0430\u049b\u0448\u0430", de:"Deutsch", fr:"Fran\u00e7ais", es:"Espa\u00f1ol", zh:"\u4e2d\u6587", ar:"\u0627\u0644\u0639\u0631\u0628\u064a\u0629", tr:"T\u00fcrk\u00e7e", it:"Italiano", pt:"Portugu\u00eas", ja:"\u65e5\u672c\u8a9e", ko:"\ud55c\uad6d\uc5b4", hi:"\u0939\u093f\u0928\u094d\u0926\u0940", uz:"O\u02bbzbekcha", ky:"\u041a\u044b\u0440\u0433\u044b\u0437\u0447\u0430", uk:"\u0423\u043a\u0440\u0430\u0457\u043d\u0441\u044c\u043a\u0430", pl:"Polski", nl:"Nederlands", sv:"Svenska" };
  const sel = document.getElementById("lang-select");
  sel.innerHTML = "";
  Object.keys(translations).forEach(code => {
    const opt = document.createElement("option");
    opt.value = code; opt.textContent = langNames[code] || code;
    if (code === currentLang) opt.selected = true;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", e => { currentLang = e.target.value; localStorage.setItem("fp_lang", currentLang); applyTranslations(); });
}
function showError(msg) {
  const el = document.getElementById("form-error");
  el.textContent = msg;
  el.classList.remove("hidden");
  el.classList.remove("anim-error"); void el.offsetWidth; el.classList.add("anim-error");
}
function hideError() { document.getElementById("form-error").classList.add("hidden"); }

const EYE_ICON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7Z"/><circle cx="12" cy="12" r="3"/></svg>';
const EYE_OFF_ICON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.9 4.24A10.94 10.94 0 0 1 12 4c7 0 11 7 11 7a13.16 13.16 0 0 1-1.67 2.68M6.61 6.61A13.526 13.526 0 0 0 1 12s4 7 11 7a10.94 10.94 0 0 0 5.11-1.24M14.12 14.12a3 3 0 1 1-4.24-4.24"/><path d="m1 1 22 22"/></svg>';
function initPasswordToggles() {
  document.querySelectorAll(".toggle-eye").forEach(btn => {
    btn.addEventListener("click", () => {
      const input = document.getElementById(btn.dataset.target);
      if (!input) return;
      const isPw = input.type === "password";
      input.type = isPw ? "text" : "password";
      btn.innerHTML = isPw ? EYE_OFF_ICON : EYE_ICON;
    });
  });
}

function isValidEmail(email) { return /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email); }
function isStrongPassword(pw) { return pw.length >= 8 && /[A-Za-z]/.test(pw) && /[0-9]/.test(pw); }

document.addEventListener("contextmenu", (e) => e.preventDefault());

document.addEventListener("DOMContentLoaded", () => {
  buildLangSelector();
  applyTranslations();
  initPasswordToggles();
  if (localStorage.getItem("fp_token")) { window.location.href = "/dashboard"; return; }

  document.getElementById("register-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError();
    const full_name = document.getElementById("full-name").value.trim();
    const email = document.getElementById("email").value.trim();
    const password = document.getElementById("password").value;
    const confirm_password = document.getElementById("confirm-password").value;

    if (!full_name || !email || !password || !confirm_password) { showError(t("err_missing_fields")); return; }
    if (!isValidEmail(email)) { showError(t("err_invalid_email")); return; }
    if (!isStrongPassword(password)) { showError(t("err_weak_password")); return; }
    if (password !== confirm_password) { showError(t("err_password_mismatch")); return; }

    const btn = document.getElementById("submit-btn");
    const spinner = document.getElementById("submit-spinner");
    btn.classList.add("opacity-70");
    spinner.classList.remove("hidden");
    try {
      const res = await fetch("/api/register", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ full_name, email, password, confirm_password, role: document.getElementById("reg-role").value, lang: currentLang }),
      });
      const data = await res.json();
      if (!res.ok) {
        if (data.error === "email_taken") showError(t("err_email_taken"));
        else if (data.error === "invalid_email") showError(t("err_invalid_email"));
        else if (data.error === "weak_password") showError(t("err_weak_password"));
        else if (data.error === "password_mismatch") showError(t("err_password_mismatch"));
        else showError(t("err_generic"));
        return;
      }
      localStorage.setItem("fp_token", data.token);
      window.location.href = "/dashboard";
    } catch (err) {
      showError(t("err_generic"));
    } finally {
      btn.classList.remove("opacity-70");
      spinner.classList.add("hidden");
    }
  });
});
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    # A first-time visitor should see registration, not a dashboard shell that
    # flashes and then bounces. The register page itself forwards anyone who is
    # already signed in straight to /dashboard.
    return redirect("/register")


def _no_cache_html(html):
    """Serves a page with caching disabled.

    The whole app (markup + JavaScript) lives inside these HTML responses, so a
    cached copy means the browser keeps running an old build after an update -
    which looks exactly like a bug that was already fixed."""
    response = make_response(html)
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return _no_cache_html(INDEX_HTML)


@app.route("/login", methods=["GET"])
def login_page():
    return _no_cache_html(LOGIN_HTML)


@app.route("/register", methods=["GET"])
def register_page():
    return _no_cache_html(REGISTER_HTML)


@app.route("/healthz", methods=["GET"])
def healthz():
    """Lightweight health check for load balancers and uptime monitors.
    Deliberately does no heavy work so it stays fast under load."""
    try:
        with engine.connect():
            db_ok = True
    except Exception:
        db_ok = False
    return jsonify({
        "status": "ok" if db_ok else "degraded",
        "database": "up" if db_ok else "down",
        "gemini": GEMINI_ENABLED,
        "websocket": SOCKETIO_ENABLED,
        "data_mode": DATA_MODE,
        "active_users": len(_active_users),
    }), (200 if db_ok else 503)


# ==================================================================
# BUILT-IN LOAD GENERATOR
#   python app.py loadtest --users 200 --seconds 30 --url http://localhost:5000
#
# Spawns concurrent virtual users that register, log in, and then hammer the
# read-heavy endpoints a real dashboard hits, so you can see actual latency and
# error rates before real traffic does.
# ==================================================================

def run_load_test(base_url, n_users, duration_seconds, ramp_seconds=5):
    import threading as _th
    import urllib.request
    import urllib.error
    import json as _json
    import time as _time
    import random as _random

    base_url = base_url.rstrip("/")
    results = {"ok": 0, "fail": 0, "latencies": [], "errors": {}, "codes": {}}
    lock = _th.Lock()
    stop_at = _time.time() + duration_seconds

    def http(method, path, body=None, token=None, timeout=20):
        url = base_url + path
        data = _json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("Authorization", "Bearer " + token)
        started = _time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
                elapsed = (_time.time() - started) * 1000
                return resp.status, payload, elapsed
        except urllib.error.HTTPError as e:
            return e.code, e.read(), (_time.time() - started) * 1000
        except Exception as e:
            return None, str(e).encode(), (_time.time() - started) * 1000

    def record(status, elapsed, err=None):
        with lock:
            results["latencies"].append(elapsed)
            if status and 200 <= status < 400:
                results["ok"] += 1
            else:
                results["fail"] += 1
                key = err or f"HTTP {status}"
                results["errors"][key] = results["errors"].get(key, 0) + 1
            if status:
                results["codes"][status] = results["codes"].get(status, 0) + 1

    def virtual_user(index):
        # Stagger start so we ramp up instead of thundering-herd on second zero.
        _time.sleep(_random.uniform(0, ramp_seconds))

        email = f"loadtest_{index}_{int(_time.time())}@example.com"
        password = "LoadTest123"
        token = None

        status, body, elapsed = http("POST", "/api/register", {
            "full_name": f"Load User {index}", "email": email,
            "password": password, "confirm_password": password, "role": "engineer",
        })
        record(status, elapsed)
        if status and 200 <= status < 300:
            try:
                token = _json.loads(body).get("token")
            except Exception:
                token = None

        if not token:
            status, body, elapsed = http("POST", "/api/login", {"email": email, "password": password})
            record(status, elapsed)
            if status and 200 <= status < 300:
                try:
                    token = _json.loads(body).get("token")
                except Exception:
                    pass

        if not token:
            return  # this virtual user could not authenticate; its failures are recorded

        # Each user owns one machine, like a real operator would.
        http("POST", "/api/machine", {
            "machine_code": f"LT-{index:04d}", "machine_name": f"LoadTest {index}",
            "factory_section": f"Line-{index % 4}",
            "temperature": _random.uniform(60, 90), "vibration": _random.uniform(3, 11),
            "load": _random.uniform(50, 95), "pressure": 1.2,
            "voltage": 220, "current": _random.uniform(4, 12),
            "status": "running", "daily_output_units": 500,
        }, token=token)

        # Steady-state read traffic, mirroring what an open dashboard generates.
        endpoints = ["/api/machines", "/api/alerts", "/api/system/intelligence", "/api/business/roi"]
        while _time.time() < stop_at:
            path = _random.choice(endpoints)
            status, body, elapsed = http("GET", path, token=token)
            record(status, elapsed, None if status else body.decode()[:60])
            _time.sleep(_random.uniform(0.5, 2.0))

    print("=" * 70)
    print(f"LOAD TEST  ->  {base_url}")
    print(f"virtual users: {n_users} | duration: {duration_seconds}s | ramp: {ramp_seconds}s")
    print("=" * 70)

    threads = [_th.Thread(target=virtual_user, args=(i,), daemon=True) for i in range(n_users)]
    wall_start = _time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=duration_seconds + 60)
    wall = _time.time() - wall_start

    lat = sorted(results["latencies"])
    total = results["ok"] + results["fail"]

    def pct(p):
        if not lat:
            return 0.0
        return lat[min(len(lat) - 1, int(len(lat) * p))]

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  requests      : {total}")
    print(f"  successful    : {results['ok']}")
    print(f"  failed        : {results['fail']}")
    if total:
        print(f"  error rate    : {results['fail'] / total * 100:.2f}%")
        print(f"  throughput    : {total / max(wall, 0.001):.1f} req/s")
    if lat:
        print(f"  latency avg   : {sum(lat) / len(lat):.0f} ms")
        print(f"  latency p50   : {pct(0.50):.0f} ms")
        print(f"  latency p95   : {pct(0.95):.0f} ms")
        print(f"  latency p99   : {pct(0.99):.0f} ms")
        print(f"  latency max   : {lat[-1]:.0f} ms")
    if results["codes"]:
        print(f"  status codes  : {dict(sorted(results['codes'].items()))}")
    if results["errors"]:
        print("  errors:")
        for k, v in sorted(results["errors"].items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {v:5}x  {k}")

    print()
    if total and results["fail"] / total > 0.05:
        print("  VERDICT: FAILING - error rate above 5%. Reduce load or scale up.")
    elif lat and pct(0.95) > 2000:
        print("  VERDICT: SLOW - p95 above 2s. Users will notice lag.")
    elif total == 0:
        print("  VERDICT: NO TRAFFIC - could not reach the server at all.")
    else:
        print("  VERDICT: HEALTHY at this load level.")
    print("=" * 70)
    return results


if __name__ == "__main__":
    import sys

    # ---- Mode: load test ----------------------------------------------
    if len(sys.argv) > 1 and sys.argv[1] == "loadtest":
        args = sys.argv[2:]

        def arg(name, default):
            if name in args:
                return args[args.index(name) + 1]
            return default

        run_load_test(
            base_url=arg("--url", os.environ.get("LOADTEST_URL", "http://localhost:5000")),
            n_users=int(arg("--users", "50")),
            duration_seconds=int(arg("--seconds", "30")),
            ramp_seconds=int(arg("--ramp", "5")),
        )
        sys.exit(0)

    # ---- Mode: web server ---------------------------------------------
    CERT_FILE = "cert.pem"
    KEY_FILE = "key.pem"
    PORT = int(os.environ.get("PORT", 5000))
    # Debug/reloader mode is meant for local development only. On a host like
    # Render there's no cert.pem/key.pem (they terminate HTTPS for you), and
    # newer Flask-SocketIO refuses to run its dev server with debug=True there.
    DEBUG_MODE = os.environ.get("FLASK_DEBUG", "true").lower() in ("1", "true", "yes")

    print("=" * 70)
    print("FactoryPulse AI is starting...")
    print(f"Gemini AI enabled: {GEMINI_ENABLED}")
    if not GEMINI_ENABLED:
        print("Set GEMINI_API_KEY env var to enable live Gemini analysis.")
        print("Running on deterministic fallback analysis engine for demo purposes.")
    print(f"Data mode: {DATA_MODE}  (USB available: {USB_AVAILABLE}, PLC available: {PLC_AVAILABLE})")
    print(f"Real-time WebSocket push: {'enabled' if SOCKETIO_ENABLED else 'disabled (falling back to polling)'}")
    _smtp = smtp_config_status()
    if _smtp["configured"]:
        print(f"Email: ENABLED via {_smtp['host']}:{_smtp['port']} as {_smtp['from']}")
        print("       Password-reset codes and critical alerts will be delivered.")
    else:
        print("Email: DISABLED - password-reset codes will NOT reach anyone's inbox.")
        print(f"       Missing: {', '.join(_smtp['missing'])}")
        print("       Set these env vars to enable (Gmail needs a 16-char App Password):")
        print("         SMTP_HOST=smtp.gmail.com  SMTP_PORT=587")
        print("         SMTP_USER=you@gmail.com   SMTP_PASSWORD=<app password>")
    print(f"Broadcast interval: {BROADCAST_INTERVAL_SECONDS}s | DB pool: {DB_POOL_SIZE}+{DB_MAX_OVERFLOW}"
          f" | Redis: {'yes' if REDIS_URL else 'no'}")
    print("NOTE: this is the development server (fine for demos and pilots).")
    print("      For production traffic run it behind gunicorn instead:")
    print("      gunicorn --worker-class eventlet -w 1 -b 0.0.0.0:$PORT app:app")

    def _run(**kwargs):
        if SOCKETIO_ENABLED:
            # allow_unsafe_werkzeug is required on hosts (Render, etc.) where
            # Flask-SocketIO detects it isn't running via a "real" production
            # server and would otherwise refuse to start at all.
            socketio.run(app, allow_unsafe_werkzeug=True, **kwargs)
        else:
            app.run(**kwargs)

    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        print("SSL certificate found - starting with HTTPS enabled.")
        print(f"Visit: https://localhost:{PORT}")
        print("=" * 70)
        _run(host="0.0.0.0", port=PORT, ssl_context=(CERT_FILE, KEY_FILE), debug=DEBUG_MODE)
    else:
        print("No SSL certificate found (cert.pem / key.pem).")
        print("Generate one with:")
        print("  openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes")
        print("Falling back to plain HTTP (hosts like Render provide HTTPS for you at the edge).")
        print(f"Visit: http://localhost:{PORT}")
        print("=" * 70)
        _run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)
