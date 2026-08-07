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
from flask import Flask, request, jsonify, g, redirect, send_file

REPORTLAB_AVAILABLE = False
try:
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
    REPORTLAB_AVAILABLE = True
except ImportError:
    print("reportlab not installed - PDF report export disabled (pip install reportlab to enable).")

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
        model = genai.GenerativeModel("gemini-pro")
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

try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    with engine.connect():
        pass
    print(f"Connected to database: {DATABASE_URL.split('@')[-1]}")
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
    role = Column(String(20), default="operator")  # "admin" or "operator"
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
    acknowledged = Column(Integer, default=0)  # 0/1 (SQLite-friendly boolean)
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
            ],
            "factories": [
                ("ai_insights", "TEXT"),
            ],
            "machines": [
                ("data_mode", "VARCHAR(20) DEFAULT 'SIMULATION'"),
                ("failure_risk", "FLOAT DEFAULT 0"),
                ("estimated_failure_time", "VARCHAR(120) DEFAULT ''"),
                ("daily_output_units", "FLOAT DEFAULT 0"),
            ],
            "alerts": [
                ("alert_temperature", "FLOAT DEFAULT 0"),
                ("alert_vibration", "FLOAT DEFAULT 0"),
                ("alert_status", "VARCHAR(20) DEFAULT ''"),
                ("alert_type", "VARCHAR(20) DEFAULT 'critical'"),
                ("alert_value", "FLOAT DEFAULT 0"),
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


def serialize_user(user):
    return {"id": user.id, "full_name": user.full_name, "email": user.email, "role": user.role or "operator"}


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


def serialize_alert(a):
    return {
        "id": a.id,
        "machine_id": a.machine_id,
        "machine_code": a.machine_code,
        "machine_name": a.machine_name,
        "severity": a.severity,
        "alert_type": a.alert_type,
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

factory_state = {
    "factory_name": "Demo Factory",
    "machine_count": 6,
    "energy_cost": 0.12,
    "machine_type": "CNC",
    "temperature": 65,
    "vibration": 3.5,
    "load": 60,
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
'en': {'risk_issue':"{critical} machine(s) in CRITICAL state and {warning} in WARNING state. Average load and temperature trends suggest elevated mechanical stress on {type} units.",'risk_ok':"No critical anomalies detected across {count} {type} machines. Overall risk level is low.",'efficiency':"Current plant efficiency is {eff}% with energy usage at {energy} kWh. {active} of {total} machines are running optimally.",'optimization':"Schedule preventive maintenance for flagged machines within 48 hours, recalibrate load balancing across the line, and consider reducing peak-hour load by 10-15% to lower energy cost without impacting throughput.",'pred_critical':"Motor failure likely in {hours} hour(s)",'pred_warning':"Elevated wear detected, potential failure within {hours} hours",'pred_ok':"No imminent failure expected",'rec_critical':"Stop the machine, reduce load immediately, and inspect bearings and cooling system.",'rec_warning':"Schedule inspection within 24 hours and reduce load by 10-15%.",'rec_ok':"Continue routine monitoring; no action needed."},
'ru': {'risk_issue':"{critical} станок(ов) в состоянии КРИТИЧНО и {warning} в состоянии ВНИМАНИЕ. Тенденции нагрузки и температуры указывают на повышенную механическую нагрузку на оборудование типа {type}.",'risk_ok':"Критических аномалий не обнаружено среди {count} станков типа {type}. Общий уровень риска низкий.",'efficiency':"Текущая эффективность завода составляет {eff}% при потреблении энергии {energy} кВт·ч. {active} из {total} станков работают оптимально.",'optimization':"Запланируйте профилактическое обслуживание отмеченных станков в течение 48 часов, перекалибруйте балансировку нагрузки по линии и рассмотрите снижение пиковой нагрузки на 10-15% для экономии энергии без потери производительности.",'pred_critical':"Вероятен отказ двигателя через {hours} час(ов)",'pred_warning':"Обнаружен повышенный износ, возможен отказ в течение {hours} часов",'pred_ok':"Неминуемый отказ не ожидается",'rec_critical':"Немедленно остановите станок, снизьте нагрузку и проверьте подшипники и систему охлаждения.",'rec_warning':"Запланируйте осмотр в течение 24 часов и снизьте нагрузку на 10-15%.",'rec_ok':"Продолжайте плановый мониторинг; действий не требуется."},
'kk': {'risk_issue':"{critical} станок СЫНИ күйде, {warning} станок ЕСКЕРТУ күйінде. Жүктеме мен температура үрдісі {type} жабдығына түсетін механикалық қысымның артқанын көрсетеді.",'risk_ok':"{count} {type} станогының арасында сыни ақаулар табылмады. Жалпы тәуекел деңгейі төмен.",'efficiency':"Зауыттың қазіргі тиімділігі {eff}%, энергия тұтыну {energy} кВт·сағ. {total} станоктың {active}-і оңтайлы жұмыс істеп тұр.",'optimization':"Белгіленген станоктарға 48 сағат ішінде алдын алу техникалық қызметін жоспарлаңыз, желі бойынша жүктеме теңгерімін қайта реттеңіз және өнімділікке нұқсан келтірмей энергия шығынын азайту үшін шыңдық сағаттардағы жүктемені 10-15%-ға төмендетуді қарастырыңыз.",'pred_critical':"{hours} сағаттан кейін қозғалтқыштың бұзылу ықтималдығы жоғары",'pred_warning':"Тозудың артқаны байқалды, {hours} сағат ішінде ақау болуы мүмкін",'pred_ok':"Жақын арада ақау күтілмейді",'rec_critical':"Станокты дереу тоқтатыңыз, жүктемені азайтыңыз және подшипниктер мен салқындату жүйесін тексеріңіз.",'rec_warning':"24 сағат ішінде тексеру жоспарлаңыз және жүктемені 10-15%-ға азайтыңыз.",'rec_ok':"Жоспарлы бақылауды жалғастырыңыз; әрекет қажет емес."},
'de': {'risk_issue':"{critical} Maschine(n) im KRITISCHEN Zustand und {warning} im WARNZUSTAND. Last- und Temperaturtrends deuten auf erhöhte mechanische Belastung der {type}-Einheiten hin.",'risk_ok':"Keine kritischen Anomalien bei {count} {type}-Maschinen festgestellt. Das Gesamtrisiko ist niedrig.",'efficiency':"Die aktuelle Anlageneffizienz beträgt {eff}% bei einem Energieverbrauch von {energy} kWh. {active} von {total} Maschinen laufen optimal.",'optimization':"Planen Sie innerhalb von 48 Stunden eine vorbeugende Wartung der markierten Maschinen, kalibrieren Sie die Lastverteilung neu und erwägen Sie eine Reduzierung der Spitzenlast um 10-15%, um Energiekosten zu senken.",'pred_critical':"Motorausfall wahrscheinlich in {hours} Stunde(n)",'pred_warning':"Erhöhter Verschleiß festgestellt, möglicher Ausfall innerhalb von {hours} Stunden",'pred_ok':"Kein unmittelbarer Ausfall erwartet",'rec_critical':"Maschine sofort stoppen, Last reduzieren und Lager sowie Kühlsystem prüfen.",'rec_warning':"Inspektion innerhalb von 24 Stunden planen und Last um 10-15% reduzieren.",'rec_ok':"Routineüberwachung fortsetzen; keine Maßnahme erforderlich."},
'fr': {'risk_issue':"{critical} machine(s) en état CRITIQUE et {warning} en état d'ALERTE. Les tendances de charge et de température indiquent un stress mécanique accru sur les unités {type}.",'risk_ok':"Aucune anomalie critique détectée parmi {count} machines {type}. Le niveau de risque global est faible.",'efficiency':"L'efficacité actuelle de l'usine est de {eff}% avec une consommation d'énergie de {energy} kWh. {active} machines sur {total} fonctionnent de manière optimale.",'optimization':"Planifiez une maintenance préventive pour les machines signalées dans les 48 heures, recalibrez l'équilibrage de charge et envisagez de réduire la charge aux heures de pointe de 10-15%.",'pred_critical':"Panne moteur probable dans {hours} heure(s)",'pred_warning':"Usure élevée détectée, panne possible dans {hours} heures",'pred_ok':"Aucune panne imminente prévue",'rec_critical':"Arrêtez la machine immédiatement, réduisez la charge et inspectez les roulements et le système de refroidissement.",'rec_warning':"Planifiez une inspection dans les 24 heures et réduisez la charge de 10-15%.",'rec_ok':"Poursuivez la surveillance de routine ; aucune action requise."},
'es': {'risk_issue':"{critical} máquina(s) en estado CRÍTICO y {warning} en estado de ADVERTENCIA. Las tendencias de carga y temperatura sugieren un mayor estrés mecánico en las unidades {type}.",'risk_ok':"No se detectaron anomalías críticas en {count} máquinas {type}. El nivel de riesgo general es bajo.",'efficiency':"La eficiencia actual de la planta es del {eff}% con un consumo de energía de {energy} kWh. {active} de {total} máquinas funcionan de manera óptima.",'optimization':"Programe mantenimiento preventivo para las máquinas marcadas en 48 horas, recalibre el equilibrio de carga y considere reducir la carga en horas pico un 10-15%.",'pred_critical':"Probable fallo del motor en {hours} hora(s)",'pred_warning':"Desgaste elevado detectado, posible fallo en {hours} horas",'pred_ok':"No se espera un fallo inminente",'rec_critical':"Detenga la máquina de inmediato, reduzca la carga e inspeccione los rodamientos y el sistema de refrigeración.",'rec_warning':"Programe una inspección en 24 horas y reduzca la carga en un 10-15%.",'rec_ok':"Continúe el monitoreo rutinario; no se requiere acción."},
'zh': {'risk_issue':"{critical} 台设备处于严重状态，{warning} 台处于警告状态。负载和温度趋势表明 {type} 设备的机械压力正在增加。",'risk_ok':"在 {count} 台 {type} 设备中未检测到严重异常。总体风险水平较低。",'efficiency':"当前工厂效率为 {eff}%，能耗为 {energy} kWh。{total} 台设备中有 {active} 台运行正常。",'optimization':"请在48小时内安排对标记设备的预防性维护，重新校准生产线负载平衡，并考虑将高峰时段负载降低10-15%以降低能源成本。",'pred_critical':"{hours} 小时内可能发生电机故障",'pred_warning':"检测到磨损加剧，{hours} 小时内可能发生故障",'pred_ok':"预计不会发生紧急故障",'rec_critical':"立即停机，降低负载，检查轴承和冷却系统。",'rec_warning':"请在24小时内安排检查，并将负载降低10-15%。",'rec_ok':"继续常规监测；无需采取行动。"},
'ar': {'risk_issue':"{critical} آلة في حالة حرجة و {warning} في حالة تحذير. تشير اتجاهات الحمل ودرجة الحرارة إلى زيادة الإجهاد الميكانيكي على وحدات {type}.",'risk_ok':"لم يتم اكتشاف أي حالات شاذة حرجة بين {count} آلة من نوع {type}. مستوى الخطر العام منخفض.",'efficiency':"كفاءة المصنع الحالية هي {eff}% باستهلاك طاقة {energy} كيلوواط. {active} من أصل {total} آلة تعمل بشكل مثالي.",'optimization':"جدولة الصيانة الوقائية للآلات المحددة خلال 48 ساعة، وإعادة معايرة توازن الحمل عبر الخط، والنظر في تقليل حمل ساعات الذروة بنسبة 10-15%.",'pred_critical':"عطل المحرك محتمل خلال {hours} ساعة",'pred_warning':"تم اكتشاف تآكل مرتفع، احتمال حدوث عطل خلال {hours} ساعة",'pred_ok':"لا يُتوقع حدوث عطل وشيك",'rec_critical':"أوقف الآلة فورًا، قلل الحمل، وافحص المحامل ونظام التبريد.",'rec_warning':"جدولة فحص خلال 24 ساعة وتقليل الحمل بنسبة 10-15%.",'rec_ok':"واصل المراقبة الروتينية؛ لا حاجة لأي إجراء."},
'tr': {'risk_issue':"{critical} makine KRİTİK durumda ve {warning} makine UYARI durumunda. Yük ve sıcaklık eğilimleri {type} ünitelerinde artan mekanik strese işaret ediyor.",'risk_ok':"{count} adet {type} makine arasında kritik anormallik tespit edilmedi. Genel risk seviyesi düşük.",'efficiency':"Mevcut tesis verimliliği {eff}%, enerji kullanımı {energy} kWh. {total} makineden {active} tanesi optimum çalışıyor.",'optimization':"İşaretli makineler için 48 saat içinde önleyici bakım planlayın, hat boyunca yük dengelemesini yeniden kalibre edin ve enerji maliyetini düşürmek için yoğun saat yükünü %10-15 azaltmayı düşünün.",'pred_critical':"{hours} saat içinde motor arızası olası",'pred_warning':"Artan aşınma tespit edildi, {hours} saat içinde arıza olasılığı",'pred_ok':"Yakın zamanda arıza beklenmiyor",'rec_critical':"Makineyi hemen durdurun, yükü azaltın ve rulmanları ile soğutma sistemini kontrol edin.",'rec_warning':"24 saat içinde muayene planlayın ve yükü %10-15 azaltın.",'rec_ok':"Rutin izlemeye devam edin; işlem gerekmiyor."},
'it': {'risk_issue':"{critical} macchina/e in stato CRITICO e {warning} in stato di AVVISO. Le tendenze di carico e temperatura suggeriscono uno stress meccanico elevato sulle unità {type}.",'risk_ok':"Nessuna anomalia critica rilevata tra {count} macchine {type}. Il livello di rischio complessivo è basso.",'efficiency':"L'efficienza attuale dello stabilimento è del {eff}% con un consumo energetico di {energy} kWh. {active} macchine su {total} funzionano in modo ottimale.",'optimization':"Pianifica la manutenzione preventiva per le macchine segnalate entro 48 ore, ricalibra il bilanciamento del carico e considera di ridurre il carico nelle ore di punta del 10-15%.",'pred_critical':"Probabile guasto del motore entro {hours} ora/e",'pred_warning':"Rilevata usura elevata, possibile guasto entro {hours} ore",'pred_ok':"Nessun guasto imminente previsto",'rec_critical':"Ferma immediatamente la macchina, riduci il carico e ispeziona i cuscinetti e il sistema di raffreddamento.",'rec_warning':"Pianifica un'ispezione entro 24 ore e riduci il carico del 10-15%.",'rec_ok':"Continua il monitoraggio di routine; nessuna azione richiesta."},
'pt': {'risk_issue':"{critical} máquina(s) em estado CRÍTICO e {warning} em estado de ALERTA. As tendências de carga e temperatura sugerem estresse mecânico elevado nas unidades {type}.",'risk_ok':"Nenhuma anomalia crítica detectada entre {count} máquinas {type}. O nível de risco geral é baixo.",'efficiency':"A eficiência atual da fábrica é de {eff}% com consumo de energia de {energy} kWh. {active} de {total} máquinas estão funcionando de forma otimizada.",'optimization':"Agende manutenção preventiva para as máquinas sinalizadas em 48 horas, recalibre o balanceamento de carga e considere reduzir a carga no horário de pico em 10-15%.",'pred_critical':"Falha do motor provável em {hours} hora(s)",'pred_warning':"Desgaste elevado detectado, possível falha em {hours} horas",'pred_ok':"Nenhuma falha iminente esperada",'rec_critical':"Pare a máquina imediatamente, reduza a carga e inspecione os rolamentos e o sistema de resfriamento.",'rec_warning':"Agende uma inspeção em 24 horas e reduza a carga em 10-15%.",'rec_ok':"Continue o monitoramento de rotina; nenhuma ação necessária."},
'ja': {'risk_issue':"{critical} 台の機械が重大状態、{warning} 台が警告状態です。負荷と温度の傾向から、{type} ユニットへの機械的ストレスの増加が示唆されます。",'risk_ok':"{count} 台の {type} 機械の中に重大な異常は検出されませんでした。全体的なリスクレベルは低いです。",'efficiency':"現在のプラント効率は {eff}%、エネルギー使用量は {energy} kWhです。{total} 台中 {active} 台が最適に稼働しています。",'optimization':"フラグの立った機械について48時間以内に予防保守を計画し、ライン全体の負荷バランスを再調整し、ピーク時の負荷を10〜15%削減することを検討してください。",'pred_critical':"{hours} 時間以内にモーター故障の可能性が高いです",'pred_warning':"摩耗の増加を検出、{hours} 時間以内に故障の可能性があります",'pred_ok':"差し迫った故障は予想されません",'rec_critical':"直ちに機械を停止し、負荷を減らし、軸受と冷却システムを点検してください。",'rec_warning':"24時間以内に点検を計画し、負荷を10〜15%削減してください。",'rec_ok':"通常の監視を継続してください。対応は不要です。"},
'ko': {'risk_issue':"{critical} 대의 기계가 심각 상태, {warning} 대가 경고 상태입니다. 부하 및 온도 추세는 {type} 장비의 기계적 스트레스 증가를 시사합니다.",'risk_ok':"{count} 대의 {type} 기계 중 심각한 이상이 발견되지 않았습니다. 전체 위험 수준은 낮습니다.",'efficiency':"현재 공장 효율성은 {eff}%이며 에너지 사용량은 {energy} kWh입니다. {total} 대 중 {active} 대가 최적으로 작동 중입니다.",'optimization':"표시된 기계에 대해 48시간 이내에 예방 정비를 예약하고, 라인 전체의 부하 균형을 재조정하며, 처리량에 영향을 주지 않으면서 에너지 비용을 낮추기 위해 피크 시간대 부하를 10-15% 줄이는 것을 고려하세요.",'pred_critical':"{hours} 시간 내 모터 고장 가능성이 높습니다",'pred_warning':"마모 증가가 감지되었으며, {hours} 시간 내 고장 가능성이 있습니다",'pred_ok':"임박한 고장은 예상되지 않습니다",'rec_critical':"즉시 기계를 정지하고 부하를 줄이며 베어링과 냉각 시스템을 점검하세요.",'rec_warning':"24시간 이내에 점검을 예약하고 부하를 10-15% 줄이세요.",'rec_ok':"정기 모니터링을 계속하세요. 조치가 필요하지 않습니다."},
'hi': {'risk_issue':"{critical} मशीन(ें) गंभीर स्थिति में और {warning} चेतावनी स्थिति में हैं। लोड और तापमान की प्रवृत्तियाँ {type} इकाइयों पर बढ़े हुए यांत्रिक तनाव का संकेत देती हैं।",'risk_ok':"{count} {type} मशीनों में कोई गंभीर विसंगति नहीं पाई गई। समग्र जोखिम स्तर कम है।",'efficiency':"वर्तमान संयंत्र दक्षता {eff}% है, ऊर्जा उपयोग {energy} kWh है। {total} में से {active} मशीनें इष्टतम रूप से चल रही हैं।",'optimization':"चिह्नित मशीनों के लिए 48 घंटों के भीतर निवारक रखरखाव शेड्यूल करें, लाइन भर में लोड संतुलन को पुनः कैलिब्रेट करें, और ऊर्जा लागत कम करने के लिए पीक-ऑवर लोड को 10-15% तक कम करने पर विचार करें।",'pred_critical':"{hours} घंटे में मोटर विफलता की संभावना है",'pred_warning':"बढ़ी हुई घिसावट का पता चला, {hours} घंटों के भीतर विफलता संभव",'pred_ok':"तत्काल विफलता की उम्मीद नहीं है",'rec_critical':"मशीन को तुरंत रोकें, लोड कम करें, और बियरिंग तथा कूलिंग सिस्टम की जांच करें।",'rec_warning':"24 घंटों के भीतर निरीक्षण शेड्यूल करें और लोड 10-15% कम करें।",'rec_ok':"नियमित निगरानी जारी रखें; किसी कार्रवाई की आवश्यकता नहीं है।"},
'uz': {'risk_issue':"{critical} ta stanok TANQIDIY holatda va {warning} ta OGOHLANTIRISH holatida. Yuklama va harorat tendentsiyalari {type} uskunalariga mexanik zo'riqish oshganini ko'rsatadi.",'risk_ok':"{count} ta {type} stanok orasida jiddiy anomaliyalar aniqlanmadi. Umumiy xavf darajasi past.",'efficiency':"Joriy zavod samaradorligi {eff}%, energiya sarfi {energy} kVt·soat. {total} tadan {active} tasi optimal ishlamoqda.",'optimization':"Belgilangan stanoklar uchun 48 soat ichida profilaktik texnik xizmatni rejalashtiring, liniya bo'ylab yuklama balansini qayta sozlang va energiya xarajatlarini kamaytirish uchun cho'qqi soatlardagi yuklamani 10-15% ga kamaytirishni ko'rib chiqing.",'pred_critical':"{hours} soatdan keyin dvigatel buzilishi ehtimoli yuqori",'pred_warning':"Ortgan eskirish aniqlandi, {hours} soat ichida nosozlik ehtimoli bor",'pred_ok':"Yaqin orada nosozlik kutilmaydi",'rec_critical':"Stanokni darhol to'xtating, yuklamani kamaytiring va podshipniklar bilan sovutish tizimini tekshiring.",'rec_warning':"24 soat ichida tekshiruv rejalashtiring va yuklamani 10-15% ga kamaytiring.",'rec_ok':"Rejali monitoringni davom ettiring; harakat talab qilinmaydi."},
'ky': {'risk_issue':"{critical} станок КРИТИКАЛЫК абалда, {warning} ЭСКЕРТҮҮ абалында. Жүктөм жана температура тенденциялары {type} жабдыктарына механикалык стресстин көбөйгөнүн көрсөтөт.",'risk_ok':"{count} {type} станок арасында олуттуу аномалиялар табылган жок. Жалпы тобокелдик деңгээли төмөн.",'efficiency':"Учурдагы заводдун эффективдүүлүгү {eff}%, энергия сарптоо {energy} кВт·саат. {total} станоктон {active} нормалдуу иштеп жатат.",'optimization':"Белгиленген станоктор үчүн 48 саат ичинде алдын алуучу тейлөөнү пландаштырыңыз, линия боюнча жүктөм тең салмактуулугун кайра тууралаңыз жана энергия чыгымдарын азайтуу үчүн чоку сааттардагы жүктөмдү 10-15% га азайтууну карап көрүңүз.",'pred_critical':"{hours} сааттан кийин мотордун бузулушу ыктымал",'pred_warning':"Тозуунун көбөйгөнү аныкталды, {hours} саат ичинде бузулуу мүмкүн",'pred_ok':"Жакынкы мезгилде бузулуу күтүлбөйт",'rec_critical':"Станокту дароо токтотуңуз, жүктөмдү азайтыңыз жана подшипниктерди, муздатуу тутумун текшериңиз.",'rec_warning':"24 саат ичинде текшерүү пландаштырыңыз жана жүктөмдү 10-15% азайтыңыз.",'rec_ok':"Пландуу байкоону улантыңыз; аракет талап кылынбайт."},
'uk': {'risk_issue':"{critical} верстат(и) у КРИТИЧНОМУ стані та {warning} у стані ПОПЕРЕДЖЕННЯ. Тенденції навантаження та температури вказують на підвищене механічне навантаження на обладнання {type}.",'risk_ok':"Критичних аномалій серед {count} верстатів {type} не виявлено. Загальний рівень ризику низький.",'efficiency':"Поточна ефективність заводу становить {eff}% при споживанні енергії {energy} кВт·год. {active} з {total} верстатів працюють оптимально.",'optimization':"Заплануйте профілактичне обслуговування позначених верстатів протягом 48 годин, перекалібруйте баланс навантаження по лінії та розгляньте зниження пікового навантаження на 10-15%.",'pred_critical':"Ймовірна відмова двигуна через {hours} годину(и)",'pred_warning':"Виявлено підвищений знос, можлива відмова протягом {hours} годин",'pred_ok':"Неминучої відмови не очікується",'rec_critical':"Негайно зупиніть верстат, зменшіть навантаження та перевірте підшипники й систему охолодження.",'rec_warning':"Заплануйте огляд протягом 24 годин і зменшіть навантаження на 10-15%.",'rec_ok':"Продовжуйте планове спостереження; дії не потрібні."},
'pl': {'risk_issue':"{critical} maszyn(y) w stanie KRYTYCZNYM i {warning} w stanie OSTRZEŻENIA. Trendy obciążenia i temperatury wskazują na zwiększone naprężenia mechaniczne jednostek {type}.",'risk_ok':"Nie wykryto krytycznych anomalii wśród {count} maszyn {type}. Ogólny poziom ryzyka jest niski.",'efficiency':"Obecna wydajność zakładu wynosi {eff}% przy zużyciu energii {energy} kWh. {active} z {total} maszyn działa optymalnie.",'optimization':"Zaplanuj konserwację zapobiegawczą oznaczonych maszyn w ciągu 48 godzin, przekalibruj równoważenie obciążenia na linii i rozważ zmniejszenie obciążenia szczytowego o 10-15%.",'pred_critical':"Prawdopodobna awaria silnika w ciągu {hours} godzin(y)",'pred_warning':"Wykryto zwiększone zużycie, możliwa awaria w ciągu {hours} godzin",'pred_ok':"Nie przewiduje się rychłej awarii",'rec_critical':"Natychmiast zatrzymaj maszynę, zmniejsz obciążenie i sprawdź łożyska oraz układ chłodzenia.",'rec_warning':"Zaplanuj przegląd w ciągu 24 godzin i zmniejsz obciążenie o 10-15%.",'rec_ok':"Kontynuuj rutynowe monitorowanie; żadne działanie nie jest wymagane."},
'nl': {'risk_issue':"{critical} machine(s) in KRITIEKE staat en {warning} in WAARSCHUWINGSstaat. Belasting- en temperatuurtrends wijzen op verhoogde mechanische stress op {type}-eenheden.",'risk_ok':"Geen kritieke afwijkingen gedetecteerd bij {count} {type}-machines. Algeheel risiconiveau is laag.",'efficiency':"De huidige fabrieksefficiëntie is {eff}% met een energieverbruik van {energy} kWh. {active} van de {total} machines draaien optimaal.",'optimization':"Plan binnen 48 uur preventief onderhoud voor gemarkeerde machines, herijk de belastingsverdeling en overweeg de piekbelasting met 10-15% te verlagen.",'pred_critical':"Motorstoring waarschijnlijk binnen {hours} uur",'pred_warning':"Verhoogde slijtage gedetecteerd, mogelijke storing binnen {hours} uur",'pred_ok':"Geen dreigende storing verwacht",'rec_critical':"Stop de machine onmiddellijk, verminder de belasting en inspecteer lagers en koelsysteem.",'rec_warning':"Plan een inspectie binnen 24 uur en verminder de belasting met 10-15%.",'rec_ok':"Ga door met routinebewaking; geen actie vereist."},
'sv': {'risk_issue':"{critical} maskin(er) i KRITISKT tillstånd och {warning} i VARNINGSläge. Belastnings- och temperaturtrender tyder på ökad mekanisk påfrestning på {type}-enheter.",'risk_ok':"Inga kritiska avvikelser upptäcktes bland {count} {type}-maskiner. Den totala risknivån är låg.",'efficiency':"Nuvarande anläggningseffektivitet är {eff}% med en energiförbrukning på {energy} kWh. {active} av {total} maskiner körs optimalt.",'optimization':"Schemalägg förebyggande underhåll för flaggade maskiner inom 48 timmar, kalibrera om lastbalanseringen och överväg att minska belastningen under högtrafik med 10-15%.",'pred_critical':"Motorhaveri troligt inom {hours} timme(ar)",'pred_warning':"Ökat slitage upptäckt, möjligt haveri inom {hours} timmar",'pred_ok':"Inget omedelbart haveri förväntas",'rec_critical':"Stoppa maskinen omedelbart, minska belastningen och inspektera lager och kylsystem.",'rec_warning':"Schemalägg en inspektion inom 24 timmar och minska belastningen med 10-15%.",'rec_ok':"Fortsätt rutinmässig övervakning; ingen åtgärd krävs."},
}

LANG_NAMES = {
'en':'English','ru':'Russian','kk':'Kazakh','de':'German','fr':'French','es':'Spanish','zh':'Chinese',
'ar':'Arabic','tr':'Turkish','it':'Italian','pt':'Portuguese','ja':'Japanese','ko':'Korean','hi':'Hindi',
'uz':'Uzbek','ky':'Kyrgyz','uk':'Ukrainian','pl':'Polish','nl':'Dutch','sv':'Swedish',
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
def _rule_based_machine_analysis(reading, lang="en"):
    tpl = RULE_TEMPLATES.get(lang, RULE_TEMPLATES["en"])
    temperature = reading.get("temperature", 0)
    vibration = reading.get("vibration", 0)
    load = reading.get("load", 0)

    score = max(0, temperature - 60) * 1.3 + max(0, vibration - 4) * 8 + max(0, load - 75) * 1.1
    risk = int(max(0, min(99, score)))
    anomaly = risk > 55 or temperature > 85 or vibration > 10

    if risk > 75:
        hours = max(1, int(6 - risk / 20))
        prediction = tpl["pred_critical"].format(hours=hours)
        recommendation = tpl["rec_critical"]
    elif risk > 40:
        hours = max(4, int(48 - risk / 2))
        prediction = tpl["pred_warning"].format(hours=hours)
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
        "current_load_pct": load,
        "at_optimal_load": at_optimal_load,
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
    return reading


# ------------------------------------------------------------------
# FLASK APP
# ------------------------------------------------------------------
app = Flask(__name__)

SOCKETIO_ENABLED = False
socketio = None
try:
    from flask_socketio import SocketIO, join_room
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
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


def send_alert_email(subject, body):
    """Best-effort SMTP send. Silently skipped if SMTP_HOST/ALERT_EMAIL_TO aren't set."""
    if not (SMTP_HOST and ALERT_EMAIL_TO):
        return
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM
        msg["To"] = ALERT_EMAIL_TO
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            if SMTP_USER:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, [ALERT_EMAIL_TO], msg.as_string())
    except Exception as e:
        print("Alert email failed to send:", e)


def maybe_create_alert(db, m, reading):
    """Checks a live reading against critical thresholds and, respecting a per-machine
    cooldown, persists an Alert, pushes it instantly over WebSocket (if enabled), and
    emails it (if SMTP is configured). Called from both the WebSocket broadcast loop
    and the HTTP-polling live-reading route, so alerts fire either way."""
    temperature = reading.get("temperature", 0)
    vibration = reading.get("vibration", 0)
    status = reading.get("status", "running")
    is_critical = temperature > 85 or vibration > 10 or status == "stopped"
    if not is_critical:
        return None

    now = time.time()
    if now - _last_alert_at.get(m.id, 0) < ALERT_COOLDOWN_SECONDS:
        return None
    _last_alert_at[m.id] = now

    message = (
        f"{m.machine_name} ({m.machine_code}): temperature {temperature}°C, "
        f"vibration {vibration} mm/s, status {status}."
    )
    alert = Alert(
        user_id=m.user_id, machine_id=m.id, machine_code=m.machine_code,
        machine_name=m.machine_name, severity="critical", alert_type="critical", message=message,
        alert_temperature=temperature, alert_vibration=vibration, alert_status=status,
    )
    db.add(alert)
    db.commit()

    if SOCKETIO_ENABLED:
        socketio.emit("critical_alert", serialize_alert(alert), room=f"user_{m.user_id}")
    send_alert_email(f"FactoryPulse AI - Critical Alert: {m.machine_name}", message)
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

    def _broadcast_loop():
        tick = 0
        while True:
            time.sleep(1)
            tick += 1
            db = SessionLocal()
            try:
                machines = db.query(Machine).all()
                for m in machines:
                    reading = get_live_reading(m)
                    socketio.emit("machine_reading", reading, room=f"user_{m.user_id}")
                    maybe_create_alert(db, m, reading)
                    maybe_create_idle_alert(db, m, reading)
                    db.commit()  # persist any in-place mutations from get_live_reading (e.g. error_code)

                    # Every 10s, re-run AI analysis for machines on a genuine live
                    # SCADA/PLC/protocol feed (not the manual-baseline simulation),
                    # so predictive maintenance tracks real incoming telemetry.
                    if tick % 10 == 0 and reading.get("source") == "auto":
                        try:
                            ai_result = analyze_machine_reading(reading, "en")
                            m.failure_risk = ai_result["risk"]
                            m.estimated_failure_time = ai_result["prediction"]
                            db.commit()
                            socketio.emit("machine_ai_update", {
                                "machine_id": m.id, "machine_code": m.machine_code, **ai_result,
                            }, room=f"user_{m.user_id}")
                        except Exception as e:
                            print("auto AI re-analysis error:", e)
            except Exception as e:
                print("broadcast loop error:", e)
            finally:
                SessionLocal.remove()

    threading.Thread(target=_broadcast_loop, daemon=True).start()


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

    user = User(full_name=full_name, email=email, password_hash=hash_password(password))
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
def _build_pdf_report(user, factories, machines, alerts, days):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=18 * mm, bottomMargin=18 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("FPTitle", parent=styles["Title"], textColor=colors.HexColor("#0f172a"), fontSize=22)
    h2_style = ParagraphStyle("FPH2", parent=styles["Heading2"], textColor=colors.HexColor("#0891b2"), spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("FPBody", parent=styles["Normal"], fontSize=10, leading=14)
    small_style = ParagraphStyle("FPSmall", parent=styles["Normal"], fontSize=8, textColor=colors.grey)

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
    elements.append(Paragraph("FactoryPulse AI", title_style))
    elements.append(Paragraph("Pilot Program Summary Report", styles["Heading3"]))
    elements.append(Spacer(1, 4 * mm))
    elements.append(Paragraph(
        f"Prepared for: {user.full_name} ({user.email})<br/>"
        f"Reporting period: last {days} days<br/>"
        f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        body_style,
    ))
    elements.append(Spacer(1, 6 * mm))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#e2e8f0")))

    elements.append(Paragraph("Overview", h2_style))
    overview_data = [
        ["Factories monitored", str(len(factories))],
        ["Machines monitored", str(total_machines)],
        ["Total energy usage", f"{total_energy} kWh"],
        ["Average load", f"{avg_load}%"],
        ["Average temperature", f"{avg_temp}°C"],
        ["Critical alerts (incidents caught)", str(incidents_caught)],
        ["Estimated energy savings identified", f"{estimated_energy_savings_kwh} kWh"],
    ]
    overview_table = Table(overview_data, colWidths=[95 * mm, 70 * mm])
    overview_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#475569")),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    elements.append(overview_table)

    elements.append(Paragraph("Factories", h2_style))
    if factories:
        factory_rows = [["Factory", "Machines", "Type", "Energy Cost", "Avg Load", "Avg Temp"]]
        for f in factories:
            factory_rows.append([
                f.factory_name, str(f.machines), f.machine_type,
                f"${f.energy_cost}/kWh", f"{f.load}%", f"{f.temperature}°C",
            ])
        factory_table = Table(factory_rows, colWidths=[45 * mm, 22 * mm, 25 * mm, 28 * mm, 22 * mm, 22 * mm])
        factory_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ]))
        elements.append(factory_table)
    else:
        elements.append(Paragraph("No factories recorded for this account yet.", body_style))

    elements.append(Paragraph("Critical Alerts Log", h2_style))
    if critical_alerts:
        alert_rows = [["Machine", "Message", "Date"]]
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
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fef2f2")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ]))
        elements.append(alert_table)
    else:
        elements.append(Paragraph("No critical incidents were recorded during this period.", body_style))

    elements.append(Spacer(1, 10 * mm))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#e2e8f0")))
    elements.append(Spacer(1, 3 * mm))
    elements.append(Paragraph(
        "Generated automatically by FactoryPulse AI. Figures are derived from monitored "
        "machine telemetry and the built-in predictive-maintenance engine.", small_style,
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

    buffer = _build_pdf_report(g.user, factories, machines, alerts, days)
    filename = f"FactoryPulseAI_Report_{datetime.datetime.utcnow().strftime('%Y%m%d')}.pdf"
    return send_file(buffer, mimetype="application/pdf", as_attachment=True, download_name=filename)


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
def api_data():
    machines = generate_machines(factory_state)
    kpis = compute_kpis(machines, factory_state)
    return jsonify({
        "factory_name": factory_state["factory_name"],
        "kpis": kpis,
        "machines": machines,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "gemini_enabled": GEMINI_ENABLED,
    })


@app.route("/api/factory", methods=["POST"])
def api_factory():
    data = request.get_json(force=True, silent=True) or {}
    try:
        factory_state["factory_name"] = str(data.get("factory_name", "")).strip() or "Demo Factory"
        factory_state["machine_count"] = max(1, min(30, int(data.get("machine_count", 6))))
        factory_state["energy_cost"] = max(0.01, float(data.get("energy_cost", 0.12)))
        factory_state["machine_type"] = str(data.get("machine_type", "CNC")).strip() or "CNC"
        factory_state["temperature"] = float(data.get("temperature", 65))
        factory_state["vibration"] = float(data.get("vibration", 3.5))
        factory_state["load"] = float(data.get("load", 60))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid input"}), 400

    machines = generate_machines(factory_state)
    kpis = compute_kpis(machines, factory_state)
    return jsonify({
        "success": True,
        "factory_name": factory_state["factory_name"],
        "kpis": kpis,
        "machines": machines,
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(force=True, silent=True) or {}
    lang = str(data.get("lang", "en"))
    machines = generate_machines(factory_state)
    kpis = compute_kpis(machines, factory_state)
    analysis = analyze_factory(factory_state, kpis, machines, lang)
    return jsonify({"success": True, "analysis": analysis, "kpis": kpis})


@app.route("/api/meta", methods=["GET"])
def api_meta():
    return jsonify({"machine_types": MACHINE_TYPES})





INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>FactoryPulse AI</title>
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
      <input id="sm-code" placeholder="Machine ID (e.g. M-01)" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
      <input id="sm-name" placeholder="Machine Name" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
      <input id="sm-section" placeholder="Factory Section" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
      <input id="sm-operator" placeholder="Operator Name" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />

      <div class="text-xs uppercase tracking-wide text-cyan-400 font-semibold mt-2" data-t="section_sensor_data">Sensor Data</div>
      <div class="grid grid-cols-2 gap-3">
        <input id="sm-temp" type="number" placeholder="Temperature (°C)" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        <input id="sm-vibration" type="number" step="0.1" placeholder="Vibration (mm/s)" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        <input id="sm-load" type="number" placeholder="Load (%)" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        <input id="sm-pressure" type="number" step="0.1" placeholder="Pressure (bar)" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        <input id="sm-voltage" type="number" placeholder="Voltage (V)" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        <input id="sm-current" type="number" step="0.1" placeholder="Current (A)" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
      </div>

      <div class="text-xs uppercase tracking-wide text-cyan-400 font-semibold mt-2" data-t="section_status">Status</div>
      <select id="sm-status" class="input-field w-full rounded-xl text-sm px-3 py-2.5">
        <option value="running" data-t="status_running">Running</option>
        <option value="stopped" data-t="status_stopped">Stopped</option>
        <option value="maintenance" data-t="status_maintenance">Maintenance</option>
      </select>
      <div class="grid grid-cols-2 gap-3">
        <input id="sm-error" placeholder="Error Code" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
        <select id="sm-priority" class="input-field w-full rounded-xl text-sm px-3 py-2.5">
          <option value="low" data-t="priority_low">Low</option>
          <option value="normal" selected data-t="priority_normal">Normal</option>
          <option value="high" data-t="priority_high">High</option>
          <option value="critical" data-t="priority_critical">Critical</option>
        </select>
      </div>

      <div class="text-xs uppercase tracking-wide text-cyan-400 font-semibold mt-2" data-t="section_energy_intel">Energy Intelligence</div>
      <input id="sm-output" type="number" step="1" placeholder="Daily Output (units)" class="input-field w-full rounded-xl text-sm px-3 py-2.5" />
      <p class="text-xs text-slate-500" data-t="daily_output_hint">Used to calculate Specific Energy Consumption (kWh per unit).</p>

      <div class="text-xs uppercase tracking-wide text-cyan-400 font-semibold mt-2" data-t="section_notes">Notes</div>
      <textarea id="sm-notes" rows="3" class="input-field w-full rounded-xl text-sm px-3 py-2.5" placeholder="Notes..."></textarea>

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
  en: {tagline:"Global Industrial Intelligence Platform",live_label:"Live",kpi_energy:"Energy Usage",kpi_efficiency:"Efficiency",kpi_active:"Active Machines",kpi_alerts:"Alerts",kwh_unit:"kWh",chart_title:"Real-Time Performance",machine_status_title:"Machine Status",status_running:"Running",status_warning:"Warning",status_critical:"Critical",form_title:"Factory Data Input",factory_name_label:"Factory Name",machine_count_label:"Number of Machines",energy_cost_label:"Energy Cost ($/kWh)",machine_type_label:"Machine Type",temperature_label:"Temperature (°C)",vibration_label:"Vibration (mm/s)",load_label:"Load (%)",submit_btn:"Analyze Factory",submitting:"Updating...",ai_panel_title:"AI Insights",ai_placeholder:"Submit factory data to generate an AI analysis.",ai_analyzing:"Analyzing...",ai_risks:"Risks",ai_efficiency_insights:"Efficiency Insights",ai_optimizations:"Optimization Suggestions",toast_updated:"Factory data updated",toast_analysis_done:"AI analysis complete",toast_error:"Something went wrong",nav_dashboard:"Dashboard",nav_factories:"Factories",nav_ai_insights:"AI Insights",logout_btn:"Log Out",login_title:"Welcome back",login_subtitle:"Sign in to your FactoryPulse AI account",ph_email:"Email",ph_password:"Password",remember_me:"Remember me",login_btn:"Log In",login_link_register:"Don't have an account? Create one",register_title:"Create your account",register_subtitle:"Start monitoring your factories with AI",ph_full_name:"Full Name",ph_confirm_password:"Confirm Password",register_btn:"Create Account",register_link_login:"Already have an account? Sign in",err_missing_fields:"Please fill in all fields",err_invalid_email:"Please enter a valid email address",err_weak_password:"Password must be at least 8 characters with a letter and a number",err_password_mismatch:"Passwords do not match",err_invalid_credentials:"Invalid email or password",err_email_taken:"This email is already registered",err_generic:"Something went wrong. Please try again",my_factories_title:"My Factories",add_factory_btn:"+ Add Factory",edit_factory_btn:"Edit",delete_factory_btn:"Delete",confirm_delete_factory:"Delete this factory? This cannot be undone.",no_factories_yet:"You haven't added any factories yet.",factory_created_toast:"Factory created and analyzed",factory_updated_toast:"Factory updated",factory_deleted_toast:"Factory deleted",ai_insights_feed_title:"AI Insights Feed",no_ai_insights_yet:"No AI insights yet. Add a factory to get started.",reanalyze_btn:"Re-analyze",view_insights_btn:"View Insights",created_label:"Created",cancel_btn:"Cancel",save_btn:"Save Changes",nav_live_monitor:"Live Monitor",add_machine_scada_btn:"+ Add Machine",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Polling",live_chart_title:"Live Sensor Chart",machines_table_title:"Machines",machine_code_col:"Code",machine_name_col:"Name",status_col:"Status",risk_col:"Risk",no_machines_yet:"No machines yet. Click \"+ Add Machine\".",section_machine_info:"Machine Info",section_sensor_data:"Sensor Data",section_status:"Status",section_notes:"Notes",status_stopped:"Stopped",status_maintenance:"Maintenance",priority_low:"Low",priority_normal:"Normal",priority_high:"High",priority_critical:"Critical",save_and_analyze_btn:"Save & Analyze",source_col:"Source",source_auto:"Auto (SCADA)",source_manual:"Manual",nav_alerts:"Alerts",acknowledge_btn:"Acknowledge",acknowledged_label:"Acknowledged",acknowledge_all_btn:"Acknowledge All",no_alerts_yet:"No alerts. Everything is running smoothly.",download_report_btn:"Report",alert_details_template:"Temperature {temp}°C, vibration {vib} mm/s, status: {status}",section_energy_intel:"Energy Intelligence",daily_output_hint:"Used to calculate Specific Energy Consumption (kWh per unit).",energy_insights_title:"Energy Intelligence",idle_power_title:"Idle Power Detection",idle_active_msg:"Machine is idle - wasting approx. {kw} kW right now.",idle_none_msg:"No idle power waste detected.",friction_loss_title:"Predictive Energy Loss",friction_active_msg:"Elevated friction detected: +{pct}% power overhead (~{kw} kW extra). Schedule maintenance to prevent losses.",friction_none_msg:"No abnormal friction losses detected.",sec_title:"Specific Energy Consumption",sec_label:"kWh per unit",sec_unit:"kWh/unit",sec_no_data_msg:"Enter Daily Output when adding this machine to see this metric.",optimal_load_title:"Optimal Load Zone",optimal_load_label:"Optimal load",current_load_label:"Current load",at_optimal_msg:"Running in the optimal load zone.",adjust_to_optimal_msg:"Adjust load toward {pct}% to minimize energy per unit.",nav_digital_twin:"Digital Twin",twin_hint:"Drag to rotate, scroll to zoom, click a machine to see its live details.",twin_unavailable_msg:"3D view could not load (check your internet connection for the Three.js library)."},
  ru: {tagline:"Глобальная платформа промышленного интеллекта",live_label:"Live",kpi_energy:"Потребление энергии",kpi_efficiency:"Эффективность",kpi_active:"Активные станки",kpi_alerts:"Оповещения",kwh_unit:"кВт·ч",chart_title:"Показатели в реальном времени",machine_status_title:"Статус станков",status_running:"Работает",status_warning:"Внимание",status_critical:"Критично",form_title:"Ввод данных завода",factory_name_label:"Название завода",machine_count_label:"Количество станков",energy_cost_label:"Стоимость энергии ($/кВт·ч)",machine_type_label:"Тип станка",temperature_label:"Температура (°C)",vibration_label:"Вибрация (мм/с)",load_label:"Нагрузка (%)",submit_btn:"Анализировать завод",submitting:"Обновление...",ai_panel_title:"AI-аналитика",ai_placeholder:"Отправьте данные завода, чтобы получить AI-анализ.",ai_analyzing:"Анализ...",ai_risks:"Риски",ai_efficiency_insights:"Анализ эффективности",ai_optimizations:"Рекомендации по оптимизации",toast_updated:"Данные завода обновлены",toast_analysis_done:"AI-анализ завершён",toast_error:"Произошла ошибка",nav_dashboard:"Панель",nav_factories:"Заводы",nav_ai_insights:"AI-аналитика",logout_btn:"Выход",login_title:"С возвращением",login_subtitle:"Войдите в аккаунт FactoryPulse AI",ph_email:"Email",ph_password:"Пароль",remember_me:"Запомнить меня",login_btn:"Войти",login_link_register:"Нет аккаунта? Создать",register_title:"Создать аккаунт",register_subtitle:"Начните мониторинг заводов с помощью AI",ph_full_name:"Полное имя",ph_confirm_password:"Подтвердите пароль",register_btn:"Создать аккаунт",register_link_login:"Уже есть аккаунт? Войти",err_missing_fields:"Заполните все поля",err_invalid_email:"Введите корректный email",err_weak_password:"Пароль должен быть от 8 символов, с буквой и цифрой",err_password_mismatch:"Пароли не совпадают",err_invalid_credentials:"Неверный email или пароль",err_email_taken:"Этот email уже зарегистрирован",err_generic:"Что-то пошло не так. Попробуйте снова",my_factories_title:"Мои заводы",add_factory_btn:"+ Добавить завод",edit_factory_btn:"Изменить",delete_factory_btn:"Удалить",confirm_delete_factory:"Удалить этот завод? Это действие нельзя отменить.",no_factories_yet:"Вы ещё не добавили ни одного завода.",factory_created_toast:"Завод создан и проанализирован",factory_updated_toast:"Завод обновлён",factory_deleted_toast:"Завод удалён",ai_insights_feed_title:"Лента AI-аналитики",no_ai_insights_yet:"Пока нет AI-аналитики. Добавьте завод, чтобы начать.",reanalyze_btn:"Проанализировать снова",view_insights_btn:"Смотреть аналитику",created_label:"Создано",cancel_btn:"Отмена",save_btn:"Сохранить изменения",nav_live_monitor:"Мониторинг",add_machine_scada_btn:"+ Добавить станок",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Опрос",live_chart_title:"График датчиков в реальном времени",machines_table_title:"Станки",machine_code_col:"Код",machine_name_col:"Название",status_col:"Статус",risk_col:"Риск",no_machines_yet:"Станков пока нет. Нажмите «+ Добавить станок».",section_machine_info:"Информация о станке",section_sensor_data:"Данные датчиков",section_status:"Статус",section_notes:"Заметки",status_stopped:"Остановлен",status_maintenance:"Обслуживание",priority_low:"Низкий",priority_normal:"Обычный",priority_high:"Высокий",priority_critical:"Критический",save_and_analyze_btn:"Сохранить и анализировать",source_col:"Источник",source_auto:"Авто (SCADA)",source_manual:"Вручную",nav_alerts:"Оповещения",acknowledge_btn:"Подтвердить",acknowledged_label:"Подтверждено",acknowledge_all_btn:"Подтвердить все",no_alerts_yet:"Оповещений нет. Всё работает штатно.",download_report_btn:"Отчёт",alert_details_template:"Температура {temp}°C, вибрация {vib} мм/с, статус: {status}",section_energy_intel:"Энергетический интеллект",daily_output_hint:"Используется для расчёта удельного энергопотребления (кВт·ч на единицу).",energy_insights_title:"Энергетический интеллект",idle_power_title:"Обнаружение холостого хода",idle_active_msg:"Станок простаивает — расходуется примерно {kw} кВт впустую.",idle_none_msg:"Потерь энергии на холостом ходу не обнаружено.",friction_loss_title:"Прогноз потерь энергии",friction_active_msg:"Обнаружено повышенное трение: +{pct}% лишней мощности (~{kw} кВт). Запланируйте обслуживание, чтобы избежать потерь.",friction_none_msg:"Аномального трения не обнаружено.",sec_title:"Удельное энергопотребление",sec_label:"кВт·ч на единицу",sec_unit:"кВт·ч/ед.",sec_no_data_msg:"Укажите суточный объём выпуска при добавлении станка, чтобы увидеть этот показатель.",optimal_load_title:"Оптимальная зона нагрузки",optimal_load_label:"Оптимальная нагрузка",current_load_label:"Текущая нагрузка",at_optimal_msg:"Работает в оптимальной зоне нагрузки.",adjust_to_optimal_msg:"Приблизьте нагрузку к {pct}%, чтобы минимизировать расход энергии на единицу.",nav_digital_twin:"Цифровой двойник",twin_hint:"Перетаскивайте для поворота, прокручивайте для масштаба, нажмите на станок для подробностей.",twin_unavailable_msg:"Не удалось загрузить 3D-вид (проверьте подключение к интернету для библиотеки Three.js)."},
  kk: {tagline:"Жаһандық өнеркәсіптік интеллект платформасы",live_label:"Тікелей эфир",kpi_energy:"Энергия тұтыну",kpi_efficiency:"Тиімділік",kpi_active:"Белсенді станоктар",kpi_alerts:"Дабылдар",kwh_unit:"кВт·сағ",chart_title:"Нақты уақыттағы көрсеткіштер",machine_status_title:"Станоктар күйі",status_running:"Жұмыс істеп тұр",status_warning:"Ескерту",status_critical:"Сыни",form_title:"Зауыт деректерін енгізу",factory_name_label:"Зауыт атауы",machine_count_label:"Станоктар саны",energy_cost_label:"Энергия құны ($/кВт·сағ)",machine_type_label:"Станок түрі",temperature_label:"Температура (°C)",vibration_label:"Діріл (мм/с)",load_label:"Жүктеме (%)",submit_btn:"Зауытты талдау",submitting:"Жаңартылуда...",ai_panel_title:"AI-талдау",ai_placeholder:"AI-талдау алу үшін зауыт деректерін жіберіңіз.",ai_analyzing:"Талдануда...",ai_risks:"Тәуекелдер",ai_efficiency_insights:"Тиімділік талдауы",ai_optimizations:"Оңтайландыру ұсыныстары",toast_updated:"Зауыт деректері жаңартылды",toast_analysis_done:"AI-талдау аяқталды",toast_error:"Қате орын алды",nav_dashboard:"Басқару тақтасы",nav_factories:"Зауыттар",nav_ai_insights:"AI-талдау",logout_btn:"Шығу",login_title:"Қайта қош келдіңіз",login_subtitle:"FactoryPulse AI аккаунтыңызға кіріңіз",ph_email:"Email",ph_password:"Құпия сөз",remember_me:"Мені есте сақтау",login_btn:"Кіру",login_link_register:"Аккаунтыңыз жоқ па? Тіркелу",register_title:"Аккаунт құру",register_subtitle:"Зауыттарды AI арқылы бақылауды бастаңыз",ph_full_name:"Толық аты-жөні",ph_confirm_password:"Құпия сөзді қайталаңыз",register_btn:"Аккаунт құру",register_link_login:"Аккаунтыңыз бар ма? Кіру",err_missing_fields:"Барлық өрістерді толтырыңыз",err_invalid_email:"Дұрыс email мекенжайын енгізіңіз",err_weak_password:"Құпия сөз кемінде 8 таңба, әріп пен сан болуы керек",err_password_mismatch:"Құпия сөздер сәйкес келмейді",err_invalid_credentials:"Қате email немесе құпия сөз",err_email_taken:"Бұл email тіркелген",err_generic:"Қате орын алды. Қайталап көріңіз",my_factories_title:"Менің зауыттарым",add_factory_btn:"+ Зауыт қосу",edit_factory_btn:"Өзгерту",delete_factory_btn:"Жою",confirm_delete_factory:"Бұл зауытты жоясыз ба? Бұл әрекетті кері қайтару мүмкін емес.",no_factories_yet:"Сіз әлі ешбір зауыт қосқан жоқсыз.",factory_created_toast:"Зауыт құрылды және талданды",factory_updated_toast:"Зауыт жаңартылды",factory_deleted_toast:"Зауыт жойылды",ai_insights_feed_title:"AI-талдау таспасы",no_ai_insights_yet:"AI-талдау әлі жоқ. Бастау үшін зауыт қосыңыз.",reanalyze_btn:"Қайта талдау",view_insights_btn:"Талдауды көру",created_label:"Құрылған күні",cancel_btn:"Бас тарту",save_btn:"Өзгерістерді сақтау",nav_live_monitor:"Тікелей мониторинг",add_machine_scada_btn:"+ Станок қосу",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Сұрау",live_chart_title:"Нақты уақыттағы сенсор графигі",machines_table_title:"Станоктар",machine_code_col:"Код",machine_name_col:"Атауы",status_col:"Күй",risk_col:"Тәуекел",no_machines_yet:"Станоктар әлі жоқ. «+ Станок қосу» басыңыз.",section_machine_info:"Станок туралы ақпарат",section_sensor_data:"Сенсор деректері",section_status:"Күй",section_notes:"Ескертпелер",status_stopped:"Тоқтатылды",status_maintenance:"Техникалық қызмет",priority_low:"Төмен",priority_normal:"Қалыпты",priority_high:"Жоғары",priority_critical:"Сыни",save_and_analyze_btn:"Сақтау және талдау",source_col:"Дереккөз",source_auto:"Авто (SCADA)",source_manual:"Қолмен",nav_alerts:"Дабылдар",acknowledge_btn:"Растау",acknowledged_label:"Расталды",acknowledge_all_btn:"Барлығын растау",no_alerts_yet:"Дабылдар жоқ. Бәрі қалыпты жұмыс істеп тұр.",download_report_btn:"Есеп",alert_details_template:"Температура {temp}°C, діріл {vib} мм/с, күйі: {status}",section_energy_intel:"Энергетикалық интеллект",daily_output_hint:"Бірлік өнімге кететін энергияны есептеу үшін қолданылады (кВт·сағ/бірлік).",energy_insights_title:"Энергетикалық интеллект",idle_power_title:"Бос жүрісті анықтау",idle_active_msg:"Станок бос тұр — шамамен {kw} кВт босқа кетіп жатыр.",idle_none_msg:"Бос жүріс шығыны табылмады.",friction_loss_title:"Энергия шығынын болжау",friction_active_msg:"Артық үйкеліс табылды: +{pct}% қосымша қуат (~{kw} кВт). Шығынды болдырмау үшін техникалық қызмет жоспарлаңыз.",friction_none_msg:"Ауытқыған үйкеліс табылмады.",sec_title:"Бірлік өнімге кететін қуат",sec_label:"кВт·сағ/бірлік",sec_unit:"кВт·сағ/бірлік",sec_no_data_msg:"Бұл көрсеткішті көру үшін станок қосқанда күнделікті өнім санын енгізіңіз.",optimal_load_title:"Оңтайлы жұмыс режимі",optimal_load_label:"Оңтайлы жүктеме",current_load_label:"Ағымдағы жүктеме",at_optimal_msg:"Оңтайлы жүктеме аймағында жұмыс істеп тұр.",adjust_to_optimal_msg:"Бірлікке кететін энергияны азайту үшін жүктемені {pct}%-ға жақындатыңыз.",nav_digital_twin:"Цифрлық егіз",twin_hint:"Айналдыру үшін сүйреңіз, масштабтау үшін айналдырыңыз, толық мәлімет үшін станокты басыңыз.",twin_unavailable_msg:"3D көрініс жүктелмеді (Three.js кітапханасы үшін интернет байланысын тексеріңіз)."},
  de: {tagline:"Globale Industrielle Intelligenzplattform",live_label:"Live",kpi_energy:"Energieverbrauch",kpi_efficiency:"Effizienz",kpi_active:"Aktive Maschinen",kpi_alerts:"Warnungen",kwh_unit:"kWh",chart_title:"Echtzeit-Leistung",machine_status_title:"Maschinenstatus",status_running:"Läuft",status_warning:"Warnung",status_critical:"Kritisch",form_title:"Fabrikdateneingabe",factory_name_label:"Fabrikname",machine_count_label:"Anzahl der Maschinen",energy_cost_label:"Energiekosten ($/kWh)",machine_type_label:"Maschinentyp",temperature_label:"Temperatur (°C)",vibration_label:"Vibration (mm/s)",load_label:"Last (%)",submit_btn:"Fabrik Analysieren",submitting:"Aktualisieren...",ai_panel_title:"KI-Einblicke",ai_placeholder:"Senden Sie Fabrikdaten, um eine KI-Analyse zu erstellen.",ai_analyzing:"Analysiere...",ai_risks:"Risiken",ai_efficiency_insights:"Effizienzanalyse",ai_optimizations:"Optimierungsvorschläge",toast_updated:"Fabrikdaten aktualisiert",toast_analysis_done:"KI-Analyse abgeschlossen",toast_error:"Etwas ist schiefgelaufen",nav_dashboard:"Übersicht",nav_factories:"Fabriken",nav_ai_insights:"KI-Einblicke",logout_btn:"Abmelden",login_title:"Willkommen zurück",login_subtitle:"Melden Sie sich bei Ihrem FactoryPulse AI-Konto an",ph_email:"E-Mail",ph_password:"Passwort",remember_me:"Angemeldet bleiben",login_btn:"Einloggen",login_link_register:"Kein Konto? Jetzt erstellen",register_title:"Konto erstellen",register_subtitle:"Beginnen Sie mit der KI-Überwachung Ihrer Fabriken",ph_full_name:"Vollständiger Name",ph_confirm_password:"Passwort bestätigen",register_btn:"Konto erstellen",register_link_login:"Bereits ein Konto? Anmelden",err_missing_fields:"Bitte füllen Sie alle Felder aus",err_invalid_email:"Bitte geben Sie eine gültige E-Mail-Adresse ein",err_weak_password:"Passwort muss mind. 8 Zeichen, einen Buchstaben und eine Zahl enthalten",err_password_mismatch:"Passwörter stimmen nicht überein",err_invalid_credentials:"Ungültige E-Mail oder Passwort",err_email_taken:"Diese E-Mail ist bereits registriert",err_generic:"Etwas ist schiefgelaufen. Bitte erneut versuchen",my_factories_title:"Meine Fabriken",add_factory_btn:"+ Fabrik Hinzufügen",edit_factory_btn:"Bearbeiten",delete_factory_btn:"Löschen",confirm_delete_factory:"Diese Fabrik löschen? Dies kann nicht rückgängig gemacht werden.",no_factories_yet:"Sie haben noch keine Fabriken hinzugefügt.",factory_created_toast:"Fabrik erstellt und analysiert",factory_updated_toast:"Fabrik aktualisiert",factory_deleted_toast:"Fabrik gelöscht",ai_insights_feed_title:"KI-Einblicke Feed",no_ai_insights_yet:"Noch keine KI-Einblicke. Fügen Sie eine Fabrik hinzu.",reanalyze_btn:"Erneut analysieren",view_insights_btn:"Einblicke Anzeigen",created_label:"Erstellt",cancel_btn:"Abbrechen",save_btn:"Änderungen Speichern",nav_live_monitor:"Live-Überwachung",add_machine_scada_btn:"+ Maschine Hinzufügen",usb_status:"USB:",plc_status:"SPS:",polling_mode:"Abfrage",live_chart_title:"Live-Sensordiagramm",machines_table_title:"Maschinen",machine_code_col:"Code",machine_name_col:"Name",status_col:"Status",risk_col:"Risiko",no_machines_yet:"Noch keine Maschinen. Klicken Sie auf „+ Maschine Hinzufügen“.",section_machine_info:"Maschineninformationen",section_sensor_data:"Sensordaten",section_status:"Status",section_notes:"Notizen",status_stopped:"Gestoppt",status_maintenance:"Wartung",priority_low:"Niedrig",priority_normal:"Normal",priority_high:"Hoch",priority_critical:"Kritisch",save_and_analyze_btn:"Speichern & Analysieren",source_col:"Quelle",source_auto:"Auto (SCADA)",source_manual:"Manuell",nav_alerts:"Warnungen",acknowledge_btn:"Bestätigen",acknowledged_label:"Bestätigt",acknowledge_all_btn:"Alle Bestätigen",no_alerts_yet:"Keine Warnungen. Alles läuft reibungslos.",download_report_btn:"Bericht",alert_details_template:"Temperatur {temp}°C, Vibration {vib} mm/s, Status: {status}",section_energy_intel:"Energieintelligenz",daily_output_hint:"Wird zur Berechnung des spezifischen Energieverbrauchs verwendet (kWh pro Einheit).",energy_insights_title:"Energieintelligenz",idle_power_title:"Leerlaufenergie-Erkennung",idle_active_msg:"Maschine im Leerlauf – aktuell werden etwa {kw} kW verschwendet.",idle_none_msg:"Kein Leerlaufenergieverlust festgestellt.",friction_loss_title:"Vorausschauender Energieverlust",friction_active_msg:"Erhöhte Reibung festgestellt: +{pct}% Mehrleistung (~{kw} kW extra). Wartung einplanen, um Verluste zu vermeiden.",friction_none_msg:"Keine abnormale Reibung festgestellt.",sec_title:"Spezifischer Energieverbrauch",sec_label:"kWh pro Einheit",sec_unit:"kWh/Einheit",sec_no_data_msg:"Geben Sie die Tagesproduktion beim Hinzufügen der Maschine an, um diese Kennzahl zu sehen.",optimal_load_title:"Optimale Lastzone",optimal_load_label:"Optimale Last",current_load_label:"Aktuelle Last",at_optimal_msg:"Läuft in der optimalen Lastzone.",adjust_to_optimal_msg:"Last auf {pct}% anpassen, um den Energieverbrauch pro Einheit zu minimieren.",nav_digital_twin:"Digitaler Zwilling",twin_hint:"Ziehen zum Drehen, Scrollen zum Zoomen, Maschine anklicken für Live-Details.",twin_unavailable_msg:"3D-Ansicht konnte nicht geladen werden (Internetverbindung für Three.js prüfen)."},
  fr: {tagline:"Plateforme mondiale d'intelligence industrielle",live_label:"En direct",kpi_energy:"Consommation d'Énergie",kpi_efficiency:"Efficacité",kpi_active:"Machines Actives",kpi_alerts:"Alertes",kwh_unit:"kWh",chart_title:"Performance en Temps Réel",machine_status_title:"État des Machines",status_running:"En marche",status_warning:"Avertissement",status_critical:"Critique",form_title:"Saisie des Données d'Usine",factory_name_label:"Nom de l'Usine",machine_count_label:"Nombre de Machines",energy_cost_label:"Coût de l'Énergie ($/kWh)",machine_type_label:"Type de Machine",temperature_label:"Température (°C)",vibration_label:"Vibration (mm/s)",load_label:"Charge (%)",submit_btn:"Analyser l'Usine",submitting:"Mise à jour...",ai_panel_title:"Analyses IA",ai_placeholder:"Envoyez les données de l'usine pour générer une analyse IA.",ai_analyzing:"Analyse en cours...",ai_risks:"Risques",ai_efficiency_insights:"Analyse d'Efficacité",ai_optimizations:"Suggestions d'Optimisation",toast_updated:"Données d'usine mises à jour",toast_analysis_done:"Analyse IA terminée",toast_error:"Une erreur est survenue",nav_dashboard:"Tableau de Bord",nav_factories:"Usines",nav_ai_insights:"Analyses IA",logout_btn:"Déconnexion",login_title:"Content de vous revoir",login_subtitle:"Connectez-vous à votre compte FactoryPulse AI",ph_email:"E-mail",ph_password:"Mot de passe",remember_me:"Se souvenir de moi",login_btn:"Se connecter",login_link_register:"Pas de compte ? Créez-en un",register_title:"Créer votre compte",register_subtitle:"Commencez à surveiller vos usines avec l'IA",ph_full_name:"Nom Complet",ph_confirm_password:"Confirmer le Mot de Passe",register_btn:"Créer un Compte",register_link_login:"Déjà un compte ? Se connecter",err_missing_fields:"Veuillez remplir tous les champs",err_invalid_email:"Veuillez entrer une adresse e-mail valide",err_weak_password:"Le mot de passe doit contenir 8 caractères min., une lettre et un chiffre",err_password_mismatch:"Les mots de passe ne correspondent pas",err_invalid_credentials:"E-mail ou mot de passe incorrect",err_email_taken:"Cet e-mail est déjà enregistré",err_generic:"Une erreur est survenue. Veuillez réessayer",my_factories_title:"Mes Usines",add_factory_btn:"+ Ajouter une Usine",edit_factory_btn:"Modifier",delete_factory_btn:"Supprimer",confirm_delete_factory:"Supprimer cette usine ? Cette action est irréversible.",no_factories_yet:"Vous n'avez pas encore ajouté d'usine.",factory_created_toast:"Usine créée et analysée",factory_updated_toast:"Usine mise à jour",factory_deleted_toast:"Usine supprimée",ai_insights_feed_title:"Flux d'Analyses IA",no_ai_insights_yet:"Aucune analyse IA pour l'instant. Ajoutez une usine.",reanalyze_btn:"Réanalyser",view_insights_btn:"Voir les Analyses",created_label:"Créée le",cancel_btn:"Annuler",save_btn:"Enregistrer les Modifications",nav_live_monitor:"Surveillance en Direct",add_machine_scada_btn:"+ Ajouter une Machine",usb_status:"USB :",plc_status:"API :",polling_mode:"Interrogation",live_chart_title:"Graphique des Capteurs en Direct",machines_table_title:"Machines",machine_code_col:"Code",machine_name_col:"Nom",status_col:"Statut",risk_col:"Risque",no_machines_yet:"Aucune machine pour l'instant. Cliquez sur « + Ajouter une Machine ».",section_machine_info:"Informations sur la Machine",section_sensor_data:"Données des Capteurs",section_status:"Statut",section_notes:"Notes",status_stopped:"Arrêtée",status_maintenance:"Maintenance",priority_low:"Faible",priority_normal:"Normale",priority_high:"Élevée",priority_critical:"Critique",save_and_analyze_btn:"Enregistrer et Analyser",source_col:"Source",source_auto:"Auto (SCADA)",source_manual:"Manuel",nav_alerts:"Alertes",acknowledge_btn:"Confirmer",acknowledged_label:"Confirmé",acknowledge_all_btn:"Tout Confirmer",no_alerts_yet:"Aucune alerte. Tout fonctionne normalement.",download_report_btn:"Rapport",alert_details_template:"Température {temp}°C, vibration {vib} mm/s, statut : {status}",section_energy_intel:"Intelligence Énergétique",daily_output_hint:"Utilisé pour calculer la consommation énergétique spécifique (kWh par unité).",energy_insights_title:"Intelligence Énergétique",idle_power_title:"Détection de Puissance au Ralenti",idle_active_msg:"Machine au ralenti - environ {kw} kW gaspillés actuellement.",idle_none_msg:"Aucun gaspillage d'énergie au ralenti détecté.",friction_loss_title:"Perte d'Énergie Prédictive",friction_active_msg:"Friction élevée détectée : +{pct}% de surcharge (~{kw} kW en plus). Planifiez une maintenance pour éviter les pertes.",friction_none_msg:"Aucune friction anormale détectée.",sec_title:"Consommation Énergétique Spécifique",sec_label:"kWh par unité",sec_unit:"kWh/unité",sec_no_data_msg:"Indiquez la production quotidienne lors de l'ajout de la machine pour voir cette mesure.",optimal_load_title:"Zone de Charge Optimale",optimal_load_label:"Charge optimale",current_load_label:"Charge actuelle",at_optimal_msg:"Fonctionne dans la zone de charge optimale.",adjust_to_optimal_msg:"Ajustez la charge vers {pct}% pour minimiser l'énergie par unité.",nav_digital_twin:"Jumeau Numérique",twin_hint:"Faites glisser pour pivoter, défilez pour zoomer, cliquez sur une machine pour ses détails en direct.",twin_unavailable_msg:"Impossible de charger la vue 3D (vérifiez votre connexion internet pour Three.js)."},
  es: {tagline:"Plataforma Global de Inteligencia Industrial",live_label:"En vivo",kpi_energy:"Uso de Energía",kpi_efficiency:"Eficiencia",kpi_active:"Máquinas Activas",kpi_alerts:"Alertas",kwh_unit:"kWh",chart_title:"Rendimiento en Tiempo Real",machine_status_title:"Estado de Máquinas",status_running:"Funcionando",status_warning:"Advertencia",status_critical:"Crítico",form_title:"Entrada de Datos de Fábrica",factory_name_label:"Nombre de Fábrica",machine_count_label:"Número de Máquinas",energy_cost_label:"Costo de Energía ($/kWh)",machine_type_label:"Tipo de Máquina",temperature_label:"Temperatura (°C)",vibration_label:"Vibración (mm/s)",load_label:"Carga (%)",submit_btn:"Analizar Fábrica",submitting:"Actualizando...",ai_panel_title:"Perspectivas IA",ai_placeholder:"Envíe datos de fábrica para generar un análisis IA.",ai_analyzing:"Analizando...",ai_risks:"Riesgos",ai_efficiency_insights:"Análisis de Eficiencia",ai_optimizations:"Sugerencias de Optimización",toast_updated:"Datos de fábrica actualizados",toast_analysis_done:"Análisis IA completo",toast_error:"Algo salió mal",nav_dashboard:"Panel",nav_factories:"Fábricas",nav_ai_insights:"Perspectivas IA",logout_btn:"Cerrar Sesión",login_title:"Bienvenido de nuevo",login_subtitle:"Inicia sesión en tu cuenta de FactoryPulse AI",ph_email:"Correo electrónico",ph_password:"Contraseña",remember_me:"Recuérdame",login_btn:"Iniciar Sesión",login_link_register:"¿No tienes cuenta? Crea una",register_title:"Crea tu cuenta",register_subtitle:"Empieza a monitorear tus fábricas con IA",ph_full_name:"Nombre Completo",ph_confirm_password:"Confirmar Contraseña",register_btn:"Crear Cuenta",register_link_login:"¿Ya tienes cuenta? Inicia sesión",err_missing_fields:"Por favor complete todos los campos",err_invalid_email:"Por favor ingrese un correo válido",err_weak_password:"La contraseña debe tener mín. 8 caracteres, una letra y un número",err_password_mismatch:"Las contraseñas no coinciden",err_invalid_credentials:"Correo o contraseña incorrectos",err_email_taken:"Este correo ya está registrado",err_generic:"Algo salió mal. Inténtalo de nuevo",my_factories_title:"Mis Fábricas",add_factory_btn:"+ Añadir Fábrica",edit_factory_btn:"Editar",delete_factory_btn:"Eliminar",confirm_delete_factory:"¿Eliminar esta fábrica? Esta acción no se puede deshacer.",no_factories_yet:"Aún no has añadido ninguna fábrica.",factory_created_toast:"Fábrica creada y analizada",factory_updated_toast:"Fábrica actualizada",factory_deleted_toast:"Fábrica eliminada",ai_insights_feed_title:"Feed de Perspectivas IA",no_ai_insights_yet:"Aún no hay perspectivas IA. Añade una fábrica.",reanalyze_btn:"Reanalizar",view_insights_btn:"Ver Perspectivas",created_label:"Creada",cancel_btn:"Cancelar",save_btn:"Guardar Cambios",nav_live_monitor:"Monitor en Vivo",add_machine_scada_btn:"+ Añadir Máquina",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Sondeo",live_chart_title:"Gráfico de Sensores en Vivo",machines_table_title:"Máquinas",machine_code_col:"Código",machine_name_col:"Nombre",status_col:"Estado",risk_col:"Riesgo",no_machines_yet:"Aún no hay máquinas. Haga clic en «+ Añadir Máquina».",section_machine_info:"Información de la Máquina",section_sensor_data:"Datos de Sensores",section_status:"Estado",section_notes:"Notas",status_stopped:"Detenida",status_maintenance:"Mantenimiento",priority_low:"Baja",priority_normal:"Normal",priority_high:"Alta",priority_critical:"Crítica",save_and_analyze_btn:"Guardar y Analizar",source_col:"Fuente",source_auto:"Auto (SCADA)",source_manual:"Manual",nav_alerts:"Alertas",acknowledge_btn:"Reconocer",acknowledged_label:"Reconocido",acknowledge_all_btn:"Reconocer Todo",no_alerts_yet:"Sin alertas. Todo funciona correctamente.",download_report_btn:"Informe",alert_details_template:"Temperatura {temp}°C, vibración {vib} mm/s, estado: {status}",section_energy_intel:"Inteligencia Energética",daily_output_hint:"Se usa para calcular el consumo energético específico (kWh por unidad).",energy_insights_title:"Inteligencia Energética",idle_power_title:"Detección de Potencia en Inactividad",idle_active_msg:"Máquina inactiva - se desperdician aprox. {kw} kW ahora mismo.",idle_none_msg:"No se detectó desperdicio de energía en inactividad.",friction_loss_title:"Pérdida de Energía Predictiva",friction_active_msg:"Fricción elevada detectada: +{pct}% de sobrecarga (~{kw} kW extra). Programe mantenimiento para evitar pérdidas.",friction_none_msg:"No se detectó fricción anormal.",sec_title:"Consumo Energético Específico",sec_label:"kWh por unidad",sec_unit:"kWh/unidad",sec_no_data_msg:"Ingrese la producción diaria al añadir esta máquina para ver esta métrica.",optimal_load_title:"Zona de Carga Óptima",optimal_load_label:"Carga óptima",current_load_label:"Carga actual",at_optimal_msg:"Funcionando en la zona de carga óptima.",adjust_to_optimal_msg:"Ajuste la carga hacia {pct}% para minimizar la energía por unidad.",nav_digital_twin:"Gemelo Digital",twin_hint:"Arrastre para rotar, desplácese para acercar, haga clic en una máquina para ver sus detalles en vivo.",twin_unavailable_msg:"No se pudo cargar la vista 3D (verifique su conexión a internet para Three.js)."},
  zh: {tagline:"全球工业智能平台",live_label:"实时",kpi_energy:"能源使用量",kpi_efficiency:"效率",kpi_active:"运行中设备",kpi_alerts:"警报",kwh_unit:"kWh",chart_title:"实时性能",machine_status_title:"设备状态",status_running:"运行中",status_warning:"警告",status_critical:"严重",form_title:"工厂数据输入",factory_name_label:"工厂名称",machine_count_label:"设备数量",energy_cost_label:"能源成本 ($/kWh)",machine_type_label:"设备类型",temperature_label:"温度 (°C)",vibration_label:"振动 (mm/s)",load_label:"负载 (%)",submit_btn:"分析工厂",submitting:"更新中...",ai_panel_title:"AI 洞察",ai_placeholder:"提交工厂数据以生成AI分析。",ai_analyzing:"分析中...",ai_risks:"风险",ai_efficiency_insights:"效率分析",ai_optimizations:"优化建议",toast_updated:"工厂数据已更新",toast_analysis_done:"AI分析已完成",toast_error:"出现错误",nav_dashboard:"仪表盘",nav_factories:"工厂",nav_ai_insights:"AI洞察",logout_btn:"退出",login_title:"欢迎回来",login_subtitle:"登录您的 FactoryPulse AI 账户",ph_email:"电子邮件",ph_password:"密码",remember_me:"记住我",login_btn:"登录",login_link_register:"没有账户？创建一个",register_title:"创建账户",register_subtitle:"开始使用AI监控您的工厂",ph_full_name:"全名",ph_confirm_password:"确认密码",register_btn:"创建账户",register_link_login:"已有账户？登录",err_missing_fields:"请填写所有字段",err_invalid_email:"请输入有效的电子邮件地址",err_weak_password:"密码至少8位，需包含字母和数字",err_password_mismatch:"两次密码不一致",err_invalid_credentials:"电子邮件或密码错误",err_email_taken:"该电子邮件已被注册",err_generic:"出现错误，请重试",my_factories_title:"我的工厂",add_factory_btn:"+ 添加工厂",edit_factory_btn:"编辑",delete_factory_btn:"删除",confirm_delete_factory:"删除此工厂？此操作无法撤销。",no_factories_yet:"您还没有添加任何工厂。",factory_created_toast:"工厂已创建并分析",factory_updated_toast:"工厂已更新",factory_deleted_toast:"工厂已删除",ai_insights_feed_title:"AI洞察动态",no_ai_insights_yet:"暂无AI洞察。请添加工厂开始。",reanalyze_btn:"重新分析",view_insights_btn:"查看洞察",created_label:"创建于",cancel_btn:"取消",save_btn:"保存更改",nav_live_monitor:"实时监控",add_machine_scada_btn:"+ 添加设备",usb_status:"USB:",plc_status:"PLC:",polling_mode:"轮询",live_chart_title:"实时传感器图表",machines_table_title:"设备",machine_code_col:"编号",machine_name_col:"名称",status_col:"状态",risk_col:"风险",no_machines_yet:"暂无设备。点击“+ 添加设备”。",section_machine_info:"设备信息",section_sensor_data:"传感器数据",section_status:"状态",section_notes:"备注",status_stopped:"已停止",status_maintenance:"维护中",priority_low:"低",priority_normal:"正常",priority_high:"高",priority_critical:"严重",save_and_analyze_btn:"保存并分析",source_col:"数据来源",source_auto:"自动 (SCADA)",source_manual:"手动",nav_alerts:"警报",acknowledge_btn:"确认",acknowledged_label:"已确认",acknowledge_all_btn:"全部确认",no_alerts_yet:"暂无警报，一切运行正常。",download_report_btn:"报告",alert_details_template:"温度 {temp}°C，振动 {vib} mm/s，状态：{status}",section_energy_intel:"能源智能",daily_output_hint:"用于计算单位能耗（kWh/单位）。",energy_insights_title:"能源智能",idle_power_title:"空转功率检测",idle_active_msg:"设备处于空转状态 — 目前大约浪费 {kw} kW。",idle_none_msg:"未检测到空转能耗浪费。",friction_loss_title:"预测性能量损耗",friction_active_msg:"检测到摩擦增加：额外功率 +{pct}%（约 {kw} kW）。请安排维护以避免损耗。",friction_none_msg:"未检测到异常摩擦。",sec_title:"单位能耗",sec_label:"每单位kWh",sec_unit:"kWh/单位",sec_no_data_msg:"添加设备时请输入日产量以查看此指标。",optimal_load_title:"最佳负载区间",optimal_load_label:"最佳负载",current_load_label:"当前负载",at_optimal_msg:"正在最佳负载区间运行。",adjust_to_optimal_msg:"将负载调整至 {pct}% 以最小化单位能耗。",nav_digital_twin:"数字孪生",twin_hint:"拖动旋转，滚动缩放，点击设备查看实时详情。",twin_unavailable_msg:"无法加载3D视图（请检查Three.js库的网络连接）。"},
  ar: {tagline:"منصة الذكاء الصناعي العالمية",live_label:"مباشر",kpi_energy:"استهلاك الطاقة",kpi_efficiency:"الكفاءة",kpi_active:"الآلات النشطة",kpi_alerts:"التنبيهات",kwh_unit:"kWh",chart_title:"الأداء في الوقت الفعلي",machine_status_title:"حالة الآلات",status_running:"تعمل",status_warning:"تحذير",status_critical:"حرج",form_title:"إدخال بيانات المصنع",factory_name_label:"اسم المصنع",machine_count_label:"عدد الآلات",energy_cost_label:"تكلفة الطاقة ($/kWh)",machine_type_label:"نوع الآلة",temperature_label:"درجة الحرارة (°C)",vibration_label:"الاهتزاز (مم/ث)",load_label:"الحمل (%)",submit_btn:"تحليل المصنع",submitting:"جارٍ التحديث...",ai_panel_title:"رؤى الذكاء الاصطناعي",ai_placeholder:"أرسل بيانات المصنع لإنشاء تحليل بالذكاء الاصطناعي.",ai_analyzing:"جارٍ التحليل...",ai_risks:"المخاطر",ai_efficiency_insights:"تحليل الكفاءة",ai_optimizations:"اقتراحات التحسين",toast_updated:"تم تحديث بيانات المصنع",toast_analysis_done:"اكتمل تحليل الذكاء الاصطناعي",toast_error:"حدث خطأ ما",nav_dashboard:"لوحة التحكم",nav_factories:"المصانع",nav_ai_insights:"رؤى الذكاء الاصطناعي",logout_btn:"تسجيل الخروج",login_title:"مرحباً بعودتك",login_subtitle:"سجل الدخول إلى حساب FactoryPulse AI الخاص بك",ph_email:"البريد الإلكتروني",ph_password:"كلمة المرور",remember_me:"تذكرني",login_btn:"تسجيل الدخول",login_link_register:"ليس لديك حساب؟ أنشئ واحداً",register_title:"إنشاء حسابك",register_subtitle:"ابدأ بمراقبة مصانعك بالذكاء الاصطناعي",ph_full_name:"الاسم الكامل",ph_confirm_password:"تأكيد كلمة المرور",register_btn:"إنشاء حساب",register_link_login:"لديك حساب بالفعل؟ سجل الدخول",err_missing_fields:"يرجى ملء جميع الحقول",err_invalid_email:"يرجى إدخال بريد إلكتروني صالح",err_weak_password:"يجب أن تكون كلمة المرور 8 أحرف على الأقل وتحتوي على حرف ورقم",err_password_mismatch:"كلمتا المرور غير متطابقتين",err_invalid_credentials:"البريد الإلكتروني أو كلمة المرور غير صحيحة",err_email_taken:"هذا البريد الإلكتروني مسجل بالفعل",err_generic:"حدث خطأ ما. يرجى المحاولة مرة أخرى",my_factories_title:"مصانعي",add_factory_btn:"+ إضافة مصنع",edit_factory_btn:"تعديل",delete_factory_btn:"حذف",confirm_delete_factory:"هل تريد حذف هذا المصنع؟ لا يمكن التراجع عن هذا.",no_factories_yet:"لم تقم بإضافة أي مصنع بعد.",factory_created_toast:"تم إنشاء المصنع وتحليله",factory_updated_toast:"تم تحديث المصنع",factory_deleted_toast:"تم حذف المصنع",ai_insights_feed_title:"موجز رؤى الذكاء الاصطناعي",no_ai_insights_yet:"لا توجد رؤى بعد. أضف مصنعاً للبدء.",reanalyze_btn:"إعادة التحليل",view_insights_btn:"عرض الرؤى",created_label:"تاريخ الإنشاء",cancel_btn:"إلغاء",save_btn:"حفظ التغييرات",nav_live_monitor:"المراقبة المباشرة",add_machine_scada_btn:"+ إضافة آلة",usb_status:"USB:",plc_status:"PLC:",polling_mode:"استطلاع",live_chart_title:"مخطط المستشعرات المباشر",machines_table_title:"الآلات",machine_code_col:"الرمز",machine_name_col:"الاسم",status_col:"الحالة",risk_col:"الخطر",no_machines_yet:"لا توجد آلات بعد. انقر على «+ إضافة آلة».",section_machine_info:"معلومات الآلة",section_sensor_data:"بيانات المستشعر",section_status:"الحالة",section_notes:"ملاحظات",status_stopped:"متوقفة",status_maintenance:"صيانة",priority_low:"منخفضة",priority_normal:"عادية",priority_high:"عالية",priority_critical:"حرجة",save_and_analyze_btn:"حفظ وتحليل",source_col:"المصدر",source_auto:"تلقائي (SCADA)",source_manual:"يدوي",nav_alerts:"التنبيهات",acknowledge_btn:"إقرار",acknowledged_label:"تم الإقرار",acknowledge_all_btn:"إقرار الكل",no_alerts_yet:"لا توجد تنبيهات. كل شيء يعمل بسلاسة.",download_report_btn:"تقرير",alert_details_template:"درجة الحرارة {temp}°C، الاهتزاز {vib} مم/ث، الحالة: {status}",section_energy_intel:"ذكاء الطاقة",daily_output_hint:"يُستخدم لحساب استهلاك الطاقة النوعي (kWh لكل وحدة).",energy_insights_title:"ذكاء الطاقة",idle_power_title:"كشف طاقة الخمول",idle_active_msg:"الآلة خاملة - يُهدر حاليًا حوالي {kw} كيلوواط.",idle_none_msg:"لم يتم اكتشاف هدر طاقة أثناء الخمول.",friction_loss_title:"فقدان الطاقة التنبؤي",friction_active_msg:"تم اكتشاف احتكاك مرتفع: +{pct}% زيادة في الطاقة (~{kw} كيلوواط إضافية). جدولة الصيانة لتجنب الخسائر.",friction_none_msg:"لم يتم اكتشاف احتكاك غير طبيعي.",sec_title:"استهلاك الطاقة النوعي",sec_label:"kWh لكل وحدة",sec_unit:"kWh/وحدة",sec_no_data_msg:"أدخل الإنتاج اليومي عند إضافة هذه الآلة لرؤية هذا المقياس.",optimal_load_title:"منطقة الحمل الأمثل",optimal_load_label:"الحمل الأمثل",current_load_label:"الحمل الحالي",at_optimal_msg:"يعمل في منطقة الحمل الأمثل.",adjust_to_optimal_msg:"اضبط الحمل نحو {pct}% لتقليل الطاقة لكل وحدة.",nav_digital_twin:"التوأم الرقمي",twin_hint:"اسحب للتدوير، مرر للتكبير، انقر على آلة لرؤية تفاصيلها المباشرة.",twin_unavailable_msg:"تعذر تحميل العرض ثلاثي الأبعاد (تحقق من اتصال الإنترنت لمكتبة Three.js)."},
  tr: {tagline:"Küresel Endüstriyel Zeka Platformu",live_label:"Canlı",kpi_energy:"Enerji Kullanımı",kpi_efficiency:"Verimlilik",kpi_active:"Aktif Makineler",kpi_alerts:"Uyarılar",kwh_unit:"kWh",chart_title:"Gerçek Zamanlı Performans",machine_status_title:"Makine Durumu",status_running:"Çalışıyor",status_warning:"Uyarı",status_critical:"Kritik",form_title:"Fabrika Veri Girişi",factory_name_label:"Fabrika Adı",machine_count_label:"Makine Sayısı",energy_cost_label:"Enerji Maliyeti ($/kWh)",machine_type_label:"Makine Türü",temperature_label:"Sıcaklık (°C)",vibration_label:"Titreşim (mm/s)",load_label:"Yük (%)",submit_btn:"Fabrikayı Analiz Et",submitting:"Güncelleniyor...",ai_panel_title:"AI Analizleri",ai_placeholder:"AI analizi oluşturmak için fabrika verilerini gönderin.",ai_analyzing:"Analiz ediliyor...",ai_risks:"Riskler",ai_efficiency_insights:"Verimlilik Analizi",ai_optimizations:"Optimizasyon Önerileri",toast_updated:"Fabrika verileri güncellendi",toast_analysis_done:"AI analizi tamamlandı",toast_error:"Bir şeyler ters gitti",nav_dashboard:"Panel",nav_factories:"Fabrikalar",nav_ai_insights:"AI Analizleri",logout_btn:"Çıkış Yap",login_title:"Tekrar hoş geldiniz",login_subtitle:"FactoryPulse AI hesabınıza giriş yapın",ph_email:"E-posta",ph_password:"Şifre",remember_me:"Beni hatırla",login_btn:"Giriş Yap",login_link_register:"Hesabınız yok mu? Oluşturun",register_title:"Hesabınızı oluşturun",register_subtitle:"Fabrikalarınızı AI ile izlemeye başlayın",ph_full_name:"Ad Soyad",ph_confirm_password:"Şifreyi Onayla",register_btn:"Hesap Oluştur",register_link_login:"Zaten hesabınız var mı? Giriş yapın",err_missing_fields:"Lütfen tüm alanları doldurun",err_invalid_email:"Lütfen geçerli bir e-posta adresi girin",err_weak_password:"Şifre en az 8 karakter, bir harf ve bir rakam içermeli",err_password_mismatch:"Şifreler eşleşmiyor",err_invalid_credentials:"E-posta veya şifre hatalı",err_email_taken:"Bu e-posta zaten kayıtlı",err_generic:"Bir şeyler ters gitti. Tekrar deneyin",my_factories_title:"Fabrikalarım",add_factory_btn:"+ Fabrika Ekle",edit_factory_btn:"Düzenle",delete_factory_btn:"Sil",confirm_delete_factory:"Bu fabrika silinsin mi? Bu işlem geri alınamaz.",no_factories_yet:"Henüz fabrika eklemediniz.",factory_created_toast:"Fabrika oluşturuldu ve analiz edildi",factory_updated_toast:"Fabrika güncellendi",factory_deleted_toast:"Fabrika silindi",ai_insights_feed_title:"AI Analiz Akışı",no_ai_insights_yet:"Henüz AI analizi yok. Başlamak için fabrika ekleyin.",reanalyze_btn:"Yeniden Analiz Et",view_insights_btn:"Analizleri Görüntüle",created_label:"Oluşturulma",cancel_btn:"İptal",save_btn:"Değişiklikleri Kaydet",nav_live_monitor:"Canlı İzleme",add_machine_scada_btn:"+ Makine Ekle",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Sorgulama",live_chart_title:"Canlı Sensör Grafiği",machines_table_title:"Makineler",machine_code_col:"Kod",machine_name_col:"Ad",status_col:"Durum",risk_col:"Risk",no_machines_yet:"Henüz makine yok. \"+ Makine Ekle\"ye tıklayın.",section_machine_info:"Makine Bilgisi",section_sensor_data:"Sensör Verileri",section_status:"Durum",section_notes:"Notlar",status_stopped:"Durduruldu",status_maintenance:"Bakımda",priority_low:"Düşük",priority_normal:"Normal",priority_high:"Yüksek",priority_critical:"Kritik",save_and_analyze_btn:"Kaydet ve Analiz Et",source_col:"Kaynak",source_auto:"Otomatik (SCADA)",source_manual:"Manuel",nav_alerts:"Uyarılar",acknowledge_btn:"Onayla",acknowledged_label:"Onaylandı",acknowledge_all_btn:"Tümünü Onayla",no_alerts_yet:"Uyarı yok. Her şey sorunsuz çalışıyor.",download_report_btn:"Rapor",alert_details_template:"Sıcaklık {temp}°C, titreşim {vib} mm/s, durum: {status}",section_energy_intel:"Enerji Zekası",daily_output_hint:"Özgül enerji tüketimini hesaplamak için kullanılır (birim başına kWh).",energy_insights_title:"Enerji Zekası",idle_power_title:"Boşta Güç Tespiti",idle_active_msg:"Makine boşta - şu anda yaklaşık {kw} kW israf ediliyor.",idle_none_msg:"Boşta enerji israfı tespit edilmedi.",friction_loss_title:"Öngörülü Enerji Kaybı",friction_active_msg:"Artan sürtünme tespit edildi: +%{pct} fazla güç (~{kw} kW ekstra). Kayıpları önlemek için bakım planlayın.",friction_none_msg:"Anormal sürtünme tespit edilmedi.",sec_title:"Özgül Enerji Tüketimi",sec_label:"birim başına kWh",sec_unit:"kWh/birim",sec_no_data_msg:"Bu metriği görmek için makineyi eklerken günlük üretimi girin.",optimal_load_title:"Optimal Yük Bölgesi",optimal_load_label:"Optimal yük",current_load_label:"Mevcut yük",at_optimal_msg:"Optimal yük bölgesinde çalışıyor.",adjust_to_optimal_msg:"Birim başına enerjiyi en aza indirmek için yükü %{pct}'e ayarlayın.",nav_digital_twin:"Dijital İkiz",twin_hint:"Döndürmek için sürükleyin, yakınlaştırmak için kaydırın, canlı detaylar için bir makineye tıklayın.",twin_unavailable_msg:"3D görünüm yüklenemedi (Three.js kütüphanesi için internet bağlantınızı kontrol edin)."},
  it: {tagline:"Piattaforma Globale di Intelligenza Industriale",live_label:"In diretta",kpi_energy:"Consumo Energetico",kpi_efficiency:"Efficienza",kpi_active:"Macchine Attive",kpi_alerts:"Avvisi",kwh_unit:"kWh",chart_title:"Prestazioni in Tempo Reale",machine_status_title:"Stato delle Macchine",status_running:"In funzione",status_warning:"Avviso",status_critical:"Critico",form_title:"Inserimento Dati Fabbrica",factory_name_label:"Nome Fabbrica",machine_count_label:"Numero di Macchine",energy_cost_label:"Costo Energia ($/kWh)",machine_type_label:"Tipo di Macchina",temperature_label:"Temperatura (°C)",vibration_label:"Vibrazione (mm/s)",load_label:"Carico (%)",submit_btn:"Analizza Fabbrica",submitting:"Aggiornamento...",ai_panel_title:"Analisi IA",ai_placeholder:"Invia i dati della fabbrica per generare un'analisi IA.",ai_analyzing:"Analisi in corso...",ai_risks:"Rischi",ai_efficiency_insights:"Analisi dell'Efficienza",ai_optimizations:"Suggerimenti di Ottimizzazione",toast_updated:"Dati fabbrica aggiornati",toast_analysis_done:"Analisi IA completata",toast_error:"Qualcosa è andato storto",nav_dashboard:"Dashboard",nav_factories:"Fabbriche",nav_ai_insights:"Analisi IA",logout_btn:"Esci",login_title:"Bentornato",login_subtitle:"Accedi al tuo account FactoryPulse AI",ph_email:"Email",ph_password:"Password",remember_me:"Ricordami",login_btn:"Accedi",login_link_register:"Non hai un account? Creane uno",register_title:"Crea il tuo account",register_subtitle:"Inizia a monitorare le tue fabbriche con l'IA",ph_full_name:"Nome Completo",ph_confirm_password:"Conferma Password",register_btn:"Crea Account",register_link_login:"Hai già un account? Accedi",err_missing_fields:"Si prega di compilare tutti i campi",err_invalid_email:"Inserisci un indirizzo email valido",err_weak_password:"La password deve avere almeno 8 caratteri, una lettera e un numero",err_password_mismatch:"Le password non corrispondono",err_invalid_credentials:"Email o password errati",err_email_taken:"Questa email è già registrata",err_generic:"Qualcosa è andato storto. Riprova",my_factories_title:"Le Mie Fabbriche",add_factory_btn:"+ Aggiungi Fabbrica",edit_factory_btn:"Modifica",delete_factory_btn:"Elimina",confirm_delete_factory:"Eliminare questa fabbrica? Questa azione non può essere annullata.",no_factories_yet:"Non hai ancora aggiunto nessuna fabbrica.",factory_created_toast:"Fabbrica creata e analizzata",factory_updated_toast:"Fabbrica aggiornata",factory_deleted_toast:"Fabbrica eliminata",ai_insights_feed_title:"Feed di Analisi IA",no_ai_insights_yet:"Nessuna analisi IA ancora. Aggiungi una fabbrica.",reanalyze_btn:"Rianalizza",view_insights_btn:"Vedi Analisi",created_label:"Creata il",cancel_btn:"Annulla",save_btn:"Salva Modifiche",nav_live_monitor:"Monitoraggio Live",add_machine_scada_btn:"+ Aggiungi Macchina",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Polling",live_chart_title:"Grafico Sensori in Tempo Reale",machines_table_title:"Macchine",machine_code_col:"Codice",machine_name_col:"Nome",status_col:"Stato",risk_col:"Rischio",no_machines_yet:"Nessuna macchina ancora. Fai clic su «+ Aggiungi Macchina».",section_machine_info:"Informazioni Macchina",section_sensor_data:"Dati dei Sensori",section_status:"Stato",section_notes:"Note",status_stopped:"Ferma",status_maintenance:"Manutenzione",priority_low:"Bassa",priority_normal:"Normale",priority_high:"Alta",priority_critical:"Critica",save_and_analyze_btn:"Salva e Analizza",source_col:"Origine",source_auto:"Auto (SCADA)",source_manual:"Manuale",nav_alerts:"Avvisi",acknowledge_btn:"Conferma",acknowledged_label:"Confermato",acknowledge_all_btn:"Conferma Tutti",no_alerts_yet:"Nessun avviso. Tutto funziona correttamente.",download_report_btn:"Rapporto",alert_details_template:"Temperatura {temp}°C, vibrazione {vib} mm/s, stato: {status}",section_energy_intel:"Intelligenza Energetica",daily_output_hint:"Usato per calcolare il consumo energetico specifico (kWh per unità).",energy_insights_title:"Intelligenza Energetica",idle_power_title:"Rilevamento Potenza in Inattività",idle_active_msg:"Macchina inattiva - circa {kw} kW sprecati ora.",idle_none_msg:"Nessuno spreco di energia in inattività rilevato.",friction_loss_title:"Perdita di Energia Predittiva",friction_active_msg:"Attrito elevato rilevato: +{pct}% di sovraccarico (~{kw} kW extra). Pianifica la manutenzione per evitare perdite.",friction_none_msg:"Nessun attrito anomalo rilevato.",sec_title:"Consumo Energetico Specifico",sec_label:"kWh per unità",sec_unit:"kWh/unità",sec_no_data_msg:"Inserisci la produzione giornaliera aggiungendo questa macchina per vedere questa metrica.",optimal_load_title:"Zona di Carico Ottimale",optimal_load_label:"Carico ottimale",current_load_label:"Carico attuale",at_optimal_msg:"In funzione nella zona di carico ottimale.",adjust_to_optimal_msg:"Regola il carico verso {pct}% per minimizzare l'energia per unità.",nav_digital_twin:"Gemello Digitale",twin_hint:"Trascina per ruotare, scorri per zoomare, clicca su una macchina per i dettagli in tempo reale.",twin_unavailable_msg:"Impossibile caricare la vista 3D (controlla la connessione internet per Three.js)."},
  pt: {tagline:"Plataforma Global de Inteligência Industrial",live_label:"Ao vivo",kpi_energy:"Uso de Energia",kpi_efficiency:"Eficiência",kpi_active:"Máquinas Ativas",kpi_alerts:"Alertas",kwh_unit:"kWh",chart_title:"Desempenho em Tempo Real",machine_status_title:"Status das Máquinas",status_running:"Em funcionamento",status_warning:"Aviso",status_critical:"Crítico",form_title:"Entrada de Dados da Fábrica",factory_name_label:"Nome da Fábrica",machine_count_label:"Número de Máquinas",energy_cost_label:"Custo de Energia ($/kWh)",machine_type_label:"Tipo de Máquina",temperature_label:"Temperatura (°C)",vibration_label:"Vibração (mm/s)",load_label:"Carga (%)",submit_btn:"Analisar Fábrica",submitting:"Atualizando...",ai_panel_title:"Insights de IA",ai_placeholder:"Envie os dados da fábrica para gerar uma análise de IA.",ai_analyzing:"Analisando...",ai_risks:"Riscos",ai_efficiency_insights:"Análise de Eficiência",ai_optimizations:"Sugestões de Otimização",toast_updated:"Dados da fábrica atualizados",toast_analysis_done:"Análise de IA concluída",toast_error:"Algo deu errado",nav_dashboard:"Painel",nav_factories:"Fábricas",nav_ai_insights:"Insights de IA",logout_btn:"Sair",login_title:"Bem-vindo de volta",login_subtitle:"Entre na sua conta FactoryPulse AI",ph_email:"E-mail",ph_password:"Senha",remember_me:"Lembrar de mim",login_btn:"Entrar",login_link_register:"Não tem conta? Crie uma",register_title:"Crie sua conta",register_subtitle:"Comece a monitorar suas fábricas com IA",ph_full_name:"Nome Completo",ph_confirm_password:"Confirmar Senha",register_btn:"Criar Conta",register_link_login:"Já tem conta? Entrar",err_missing_fields:"Por favor preencha todos os campos",err_invalid_email:"Por favor insira um e-mail válido",err_weak_password:"A senha deve ter no mínimo 8 caracteres, uma letra e um número",err_password_mismatch:"As senhas não coincidem",err_invalid_credentials:"E-mail ou senha incorretos",err_email_taken:"Este e-mail já está registrado",err_generic:"Algo deu errado. Tente novamente",my_factories_title:"Minhas Fábricas",add_factory_btn:"+ Adicionar Fábrica",edit_factory_btn:"Editar",delete_factory_btn:"Excluir",confirm_delete_factory:"Excluir esta fábrica? Esta ação não pode ser desfeita.",no_factories_yet:"Você ainda não adicionou nenhuma fábrica.",factory_created_toast:"Fábrica criada e analisada",factory_updated_toast:"Fábrica atualizada",factory_deleted_toast:"Fábrica excluída",ai_insights_feed_title:"Feed de Insights de IA",no_ai_insights_yet:"Ainda sem insights de IA. Adicione uma fábrica.",reanalyze_btn:"Reanalisar",view_insights_btn:"Ver Insights",created_label:"Criada em",cancel_btn:"Cancelar",save_btn:"Salvar Alterações",nav_live_monitor:"Monitor ao Vivo",add_machine_scada_btn:"+ Adicionar Máquina",usb_status:"USB:",plc_status:"CLP:",polling_mode:"Sondagem",live_chart_title:"Gráfico de Sensores ao Vivo",machines_table_title:"Máquinas",machine_code_col:"Código",machine_name_col:"Nome",status_col:"Status",risk_col:"Risco",no_machines_yet:"Ainda sem máquinas. Clique em «+ Adicionar Máquina».",section_machine_info:"Informações da Máquina",section_sensor_data:"Dados do Sensor",section_status:"Status",section_notes:"Notas",status_stopped:"Parada",status_maintenance:"Manutenção",priority_low:"Baixa",priority_normal:"Normal",priority_high:"Alta",priority_critical:"Crítica",save_and_analyze_btn:"Salvar e Analisar",source_col:"Origem",source_auto:"Auto (SCADA)",source_manual:"Manual",nav_alerts:"Alertas",acknowledge_btn:"Confirmar",acknowledged_label:"Confirmado",acknowledge_all_btn:"Confirmar Todos",no_alerts_yet:"Sem alertas. Tudo funcionando normalmente.",download_report_btn:"Relatório",alert_details_template:"Temperatura {temp}°C, vibração {vib} mm/s, status: {status}",section_energy_intel:"Inteligência Energética",daily_output_hint:"Usado para calcular o consumo energético específico (kWh por unidade).",energy_insights_title:"Inteligência Energética",idle_power_title:"Detecção de Potência Ociosa",idle_active_msg:"Máquina ociosa - cerca de {kw} kW desperdiçados agora.",idle_none_msg:"Nenhum desperdício de energia ociosa detectado.",friction_loss_title:"Perda de Energia Preditiva",friction_active_msg:"Atrito elevado detectado: +{pct}% de sobrecarga (~{kw} kW extra). Agende manutenção para evitar perdas.",friction_none_msg:"Nenhum atrito anormal detectado.",sec_title:"Consumo Energético Específico",sec_label:"kWh por unidade",sec_unit:"kWh/unidade",sec_no_data_msg:"Insira a produção diária ao adicionar esta máquina para ver esta métrica.",optimal_load_title:"Zona de Carga Ideal",optimal_load_label:"Carga ideal",current_load_label:"Carga atual",at_optimal_msg:"Funcionando na zona de carga ideal.",adjust_to_optimal_msg:"Ajuste a carga para {pct}% para minimizar a energia por unidade.",nav_digital_twin:"Gêmeo Digital",twin_hint:"Arraste para girar, role para ampliar, clique em uma máquina para ver detalhes ao vivo.",twin_unavailable_msg:"Não foi possível carregar a visualização 3D (verifique sua conexão com a internet para o Three.js)."},
  ja: {tagline:"グローバル産業インテリジェンスプラットフォーム",live_label:"ライブ",kpi_energy:"エネルギー使用量",kpi_efficiency:"効率",kpi_active:"稼働中の機械",kpi_alerts:"アラート",kwh_unit:"kWh",chart_title:"リアルタイムパフォーマンス",machine_status_title:"機械の状態",status_running:"稼働中",status_warning:"警告",status_critical:"重大",form_title:"工場データ入力",factory_name_label:"工場名",machine_count_label:"機械の数",energy_cost_label:"エネルギーコスト ($/kWh)",machine_type_label:"機械の種類",temperature_label:"温度 (°C)",vibration_label:"振動 (mm/s)",load_label:"負荷 (%)",submit_btn:"工場を分析",submitting:"更新中...",ai_panel_title:"AIインサイト",ai_placeholder:"工場データを送信してAI分析を生成してください。",ai_analyzing:"分析中...",ai_risks:"リスク",ai_efficiency_insights:"効率分析",ai_optimizations:"最適化提案",toast_updated:"工場データが更新されました",toast_analysis_done:"AI分析が完了しました",toast_error:"問題が発生しました",nav_dashboard:"ダッシュボード",nav_factories:"工場",nav_ai_insights:"AIインサイト",logout_btn:"ログアウト",login_title:"おかえりなさい",login_subtitle:"FactoryPulse AI アカウントにログイン",ph_email:"メールアドレス",ph_password:"パスワード",remember_me:"ログイン状態を保持",login_btn:"ログイン",login_link_register:"アカウントをお持ちでないですか？作成する",register_title:"アカウントを作成",register_subtitle:"AIで工場の監視を始めましょう",ph_full_name:"氏名",ph_confirm_password:"パスワードの確認",register_btn:"アカウント作成",register_link_login:"すでにアカウントをお持ちですか？ログイン",err_missing_fields:"すべての項目を入力してください",err_invalid_email:"有効なメールアドレスを入力してください",err_weak_password:"パスワードは8文字以上で、文字と数字を含める必要があります",err_password_mismatch:"パスワードが一致しません",err_invalid_credentials:"メールアドレスまたはパスワードが正しくありません",err_email_taken:"このメールアドレスは既に登録されています",err_generic:"エラーが発生しました。再試行してください",my_factories_title:"マイ工場",add_factory_btn:"+ 工場を追加",edit_factory_btn:"編集",delete_factory_btn:"削除",confirm_delete_factory:"この工場を削除しますか？元に戻せません。",no_factories_yet:"まだ工場を追加していません。",factory_created_toast:"工場が作成・分析されました",factory_updated_toast:"工場が更新されました",factory_deleted_toast:"工場が削除されました",ai_insights_feed_title:"AIインサイトフィード",no_ai_insights_yet:"AIインサイトはまだありません。工場を追加してください。",reanalyze_btn:"再分析",view_insights_btn:"インサイトを見る",created_label:"作成日",cancel_btn:"キャンセル",save_btn:"変更を保存",nav_live_monitor:"ライブモニター",add_machine_scada_btn:"+ 機械を追加",usb_status:"USB:",plc_status:"PLC:",polling_mode:"ポーリング",live_chart_title:"ライブセンサーチャート",machines_table_title:"機械",machine_code_col:"コード",machine_name_col:"名前",status_col:"ステータス",risk_col:"リスク",no_machines_yet:"まだ機械がありません。「+ 機械を追加」をクリックしてください。",section_machine_info:"機械情報",section_sensor_data:"センサーデータ",section_status:"ステータス",section_notes:"メモ",status_stopped:"停止中",status_maintenance:"メンテナンス中",priority_low:"低",priority_normal:"通常",priority_high:"高",priority_critical:"重大",save_and_analyze_btn:"保存して分析",source_col:"データ元",source_auto:"自動 (SCADA)",source_manual:"手動",nav_alerts:"アラート",acknowledge_btn:"確認",acknowledged_label:"確認済み",acknowledge_all_btn:"すべて確認",no_alerts_yet:"アラートはありません。すべて正常に稼働しています。",download_report_btn:"レポート",alert_details_template:"温度 {temp}°C、振動 {vib} mm/s、状態：{status}",section_energy_intel:"エネルギーインテリジェンス",daily_output_hint:"単位あたりのエネルギー消費量（kWh/単位）を計算するために使用します。",energy_insights_title:"エネルギーインテリジェンス",idle_power_title:"アイドル電力検出",idle_active_msg:"機械がアイドル状態です - 現在約{kw} kWが無駄になっています。",idle_none_msg:"アイドル時のエネルギー浪費は検出されていません。",friction_loss_title:"予測エネルギー損失",friction_active_msg:"摩擦増加を検出：電力オーバーヘッド +{pct}%（約{kw} kW増加）。損失を防ぐためメンテナンスを計画してください。",friction_none_msg:"異常な摩擦は検出されていません。",sec_title:"原単位エネルギー消費量",sec_label:"単位あたりkWh",sec_unit:"kWh/単位",sec_no_data_msg:"この指標を表示するには、機械追加時に1日の生産量を入力してください。",optimal_load_title:"最適負荷ゾーン",optimal_load_label:"最適負荷",current_load_label:"現在の負荷",at_optimal_msg:"最適負荷ゾーンで稼働中です。",adjust_to_optimal_msg:"単位あたりのエネルギーを最小化するには、負荷を{pct}%に調整してください。",nav_digital_twin:"デジタルツイン",twin_hint:"ドラッグで回転、スクロールでズーム、機械をクリックするとライブ詳細が表示されます。",twin_unavailable_msg:"3Dビューを読み込めませんでした（Three.jsライブラリのインターネット接続を確認してください）。"},
  ko: {tagline:"글로벌 산업 인텔리전스 플랫폼",live_label:"실시간",kpi_energy:"에너지 사용량",kpi_efficiency:"효율성",kpi_active:"가동 중인 기계",kpi_alerts:"경고",kwh_unit:"kWh",chart_title:"실시간 성능",machine_status_title:"기계 상태",status_running:"가동 중",status_warning:"경고",status_critical:"심각",form_title:"공장 데이터 입력",factory_name_label:"공장 이름",machine_count_label:"기계 수",energy_cost_label:"에너지 비용 ($/kWh)",machine_type_label:"기계 유형",temperature_label:"온도 (°C)",vibration_label:"진동 (mm/s)",load_label:"부하 (%)",submit_btn:"공장 분석",submitting:"업데이트 중...",ai_panel_title:"AI 인사이트",ai_placeholder:"AI 분석을 생성하려면 공장 데이터를 제출하세요.",ai_analyzing:"분석 중...",ai_risks:"위험 요소",ai_efficiency_insights:"효율성 분석",ai_optimizations:"최적화 제안",toast_updated:"공장 데이터가 업데이트되었습니다",toast_analysis_done:"AI 분석이 완료되었습니다",toast_error:"문제가 발생했습니다",nav_dashboard:"대시보드",nav_factories:"공장",nav_ai_insights:"AI 인사이트",logout_btn:"로그아웃",login_title:"다시 오신 것을 환영합니다",login_subtitle:"FactoryPulse AI 계정에 로그인하세요",ph_email:"이메일",ph_password:"비밀번호",remember_me:"로그인 상태 유지",login_btn:"로그인",login_link_register:"계정이 없으신가요? 계정 만들기",register_title:"계정 만들기",register_subtitle:"AI로 공장 모니터링을 시작하세요",ph_full_name:"성명",ph_confirm_password:"비밀번호 확인",register_btn:"계정 생성",register_link_login:"이미 계정이 있으신가요? 로그인",err_missing_fields:"모든 항목을 입력해주세요",err_invalid_email:"유효한 이메일 주소를 입력하세요",err_weak_password:"비밀번호는 8자 이상, 문자와 숫자를 포함해야 합니다",err_password_mismatch:"비밀번호가 일치하지 않습니다",err_invalid_credentials:"이메일 또는 비밀번호가 올바르지 않습니다",err_email_taken:"이미 등록된 이메일입니다",err_generic:"문제가 발생했습니다. 다시 시도해주세요",my_factories_title:"내 공장",add_factory_btn:"+ 공장 추가",edit_factory_btn:"수정",delete_factory_btn:"삭제",confirm_delete_factory:"이 공장을 삭제하시겠습니까? 되돌릴 수 없습니다.",no_factories_yet:"아직 추가된 공장이 없습니다.",factory_created_toast:"공장이 생성되고 분석되었습니다",factory_updated_toast:"공장이 업데이트되었습니다",factory_deleted_toast:"공장이 삭제되었습니다",ai_insights_feed_title:"AI 인사이트 피드",no_ai_insights_yet:"아직 AI 인사이트가 없습니다. 공장을 추가하세요.",reanalyze_btn:"다시 분석",view_insights_btn:"인사이트 보기",created_label:"생성일",cancel_btn:"취소",save_btn:"변경사항 저장",nav_live_monitor:"실시간 모니터",add_machine_scada_btn:"+ 기계 추가",usb_status:"USB:",plc_status:"PLC:",polling_mode:"폴링",live_chart_title:"실시간 센서 차트",machines_table_title:"기계",machine_code_col:"코드",machine_name_col:"이름",status_col:"상태",risk_col:"위험",no_machines_yet:"아직 기계가 없습니다. \"+ 기계 추가\"를 클릭하세요.",section_machine_info:"기계 정보",section_sensor_data:"센서 데이터",section_status:"상태",section_notes:"메모",status_stopped:"정지됨",status_maintenance:"유지보수 중",priority_low:"낮음",priority_normal:"보통",priority_high:"높음",priority_critical:"긴급",save_and_analyze_btn:"저장 및 분석",source_col:"소스",source_auto:"자동 (SCADA)",source_manual:"수동",nav_alerts:"경고",acknowledge_btn:"확인",acknowledged_label:"확인됨",acknowledge_all_btn:"모두 확인",no_alerts_yet:"경고가 없습니다. 모든 것이 정상 작동 중입니다.",download_report_btn:"보고서",alert_details_template:"온도 {temp}°C, 진동 {vib} mm/s, 상태: {status}",section_energy_intel:"에너지 인텔리전스",daily_output_hint:"단위당 에너지 소비량(kWh/단위)을 계산하는 데 사용됩니다.",energy_insights_title:"에너지 인텔리전스",idle_power_title:"유휴 전력 감지",idle_active_msg:"기계가 유휴 상태입니다 - 현재 약 {kw} kW가 낭비되고 있습니다.",idle_none_msg:"유휴 에너지 낭비가 감지되지 않았습니다.",friction_loss_title:"예측 에너지 손실",friction_active_msg:"마찰 증가 감지: 전력 오버헤드 +{pct}%(약 {kw} kW 추가). 손실을 방지하려면 정비를 예약하세요.",friction_none_msg:"비정상적인 마찰이 감지되지 않았습니다.",sec_title:"단위당 에너지 소비량",sec_label:"단위당 kWh",sec_unit:"kWh/단위",sec_no_data_msg:"이 지표를 보려면 기계 추가 시 일일 생산량을 입력하세요.",optimal_load_title:"최적 부하 구간",optimal_load_label:"최적 부하",current_load_label:"현재 부하",at_optimal_msg:"최적 부하 구간에서 작동 중입니다.",adjust_to_optimal_msg:"단위당 에너지를 최소화하려면 부하를 {pct}%로 조정하세요.",nav_digital_twin:"디지털 트윈",twin_hint:"드래그하여 회전, 스크롤하여 확대/축소, 기계를 클릭하면 실시간 세부정보를 볼 수 있습니다.",twin_unavailable_msg:"3D 보기를 로드할 수 없습니다 (Three.js 라이브러리의 인터넷 연결을 확인하세요)."},
  hi: {tagline:"वैश्विक औद्योगिक बुद्धिमत्ता मंच",live_label:"लाइव",kpi_energy:"ऊर्जा उपयोग",kpi_efficiency:"दक्षता",kpi_active:"सक्रिय मशीनें",kpi_alerts:"अलर्ट",kwh_unit:"kWh",chart_title:"रीयल-टाइम प्रदर्शन",machine_status_title:"मशीन की स्थिति",status_running:"चल रहा है",status_warning:"चेतावनी",status_critical:"गंभीर",form_title:"फ़ैक्टरी डेटा इनपुट",factory_name_label:"फ़ैक्टरी का नाम",machine_count_label:"मशीनों की संख्या",energy_cost_label:"ऊर्जा लागत ($/kWh)",machine_type_label:"मशीन प्रकार",temperature_label:"तापमान (°C)",vibration_label:"कंपन (mm/s)",load_label:"लोड (%)",submit_btn:"फ़ैक्टरी का विश्लेषण करें",submitting:"अद्यतन हो रहा है...",ai_panel_title:"AI अंतर्दृष्टि",ai_placeholder:"AI विश्लेषण उत्पन्न करने के लिए फ़ैक्टरी डेटा सबमिट करें।",ai_analyzing:"विश्लेषण हो रहा है...",ai_risks:"जोखिम",ai_efficiency_insights:"दक्षता विश्लेषण",ai_optimizations:"अनुकूलन सुझाव",toast_updated:"फ़ैक्टरी डेटा अपडेट किया गया",toast_analysis_done:"AI विश्लेषण पूर्ण हुआ",toast_error:"कुछ गलत हो गया",nav_dashboard:"डैशबोर्ड",nav_factories:"फ़ैक्टरियाँ",nav_ai_insights:"AI अंतर्दृष्टि",logout_btn:"लॉग आउट",login_title:"वापसी पर स्वागत है",login_subtitle:"अपने FactoryPulse AI खाते में लॉग इन करें",ph_email:"ईमेल",ph_password:"पासवर्ड",remember_me:"मुझे याद रखें",login_btn:"लॉग इन करें",login_link_register:"खाता नहीं है? एक बनाएं",register_title:"अपना खाता बनाएं",register_subtitle:"AI के साथ अपनी फ़ैक्टरियों की निगरानी शुरू करें",ph_full_name:"पूरा नाम",ph_confirm_password:"पासवर्ड की पुष्टि करें",register_btn:"खाता बनाएं",register_link_login:"पहले से खाता है? लॉग इन करें",err_missing_fields:"कृपया सभी फ़ील्ड भरें",err_invalid_email:"कृपया एक मान्य ईमेल पता दर्ज करें",err_weak_password:"पासवर्ड कम से कम 8 अक्षर, एक अक्षर और एक अंक होना चाहिए",err_password_mismatch:"पासवर्ड मेल नहीं खाते",err_invalid_credentials:"गलत ईमेल या पासवर्ड",err_email_taken:"यह ईमेल पहले से पंजीकृत है",err_generic:"कुछ गलत हो गया। कृपया पुनः प्रयास करें",my_factories_title:"मेरी फ़ैक्टरियाँ",add_factory_btn:"+ फ़ैक्टरी जोड़ें",edit_factory_btn:"संपादित करें",delete_factory_btn:"हटाएं",confirm_delete_factory:"इस फ़ैक्टरी को हटाएं? इसे पूर्ववत नहीं किया जा सकता।",no_factories_yet:"आपने अभी तक कोई फ़ैक्टरी नहीं जोड़ी है।",factory_created_toast:"फ़ैक्टरी बनाई और विश्लेषित की गई",factory_updated_toast:"फ़ैक्टरी अपडेट की गई",factory_deleted_toast:"फ़ैक्टरी हटाई गई",ai_insights_feed_title:"AI अंतर्दृष्टि फ़ीड",no_ai_insights_yet:"अभी तक कोई AI अंतर्दृष्टि नहीं। शुरू करने के लिए एक फ़ैक्टरी जोड़ें।",reanalyze_btn:"पुनः विश्लेषण करें",view_insights_btn:"अंतर्दृष्टि देखें",created_label:"बनाया गया",cancel_btn:"रद्द करें",save_btn:"परिवर्तन सहेजें",nav_live_monitor:"लाइव मॉनिटर",add_machine_scada_btn:"+ मशीन जोड़ें",usb_status:"USB:",plc_status:"PLC:",polling_mode:"पोलिंग",live_chart_title:"लाइव सेंसर चार्ट",machines_table_title:"मशीनें",machine_code_col:"कोड",machine_name_col:"नाम",status_col:"स्थिति",risk_col:"जोखिम",no_machines_yet:"अभी तक कोई मशीन नहीं। \"+ मशीन जोड़ें\" पर क्लिक करें।",section_machine_info:"मशीन जानकारी",section_sensor_data:"सेंसर डेटा",section_status:"स्थिति",section_notes:"नोट्स",status_stopped:"रुकी हुई",status_maintenance:"रखरखाव",priority_low:"कम",priority_normal:"सामान्य",priority_high:"उच्च",priority_critical:"गंभीर",save_and_analyze_btn:"सहेजें और विश्लेषण करें",source_col:"स्रोत",source_auto:"स्वचालित (SCADA)",source_manual:"मैन्युअल",nav_alerts:"अलर्ट",acknowledge_btn:"स्वीकार करें",acknowledged_label:"स्वीकृत",acknowledge_all_btn:"सभी स्वीकार करें",no_alerts_yet:"कोई अलर्ट नहीं। सब कुछ सुचारू रूप से चल रहा है।",download_report_btn:"रिपोर्ट",alert_details_template:"तापमान {temp}°C, कंपन {vib} mm/s, स्थिति: {status}",section_energy_intel:"ऊर्जा इंटेलिजेंस",daily_output_hint:"विशिष्ट ऊर्जा खपत (प्रति इकाई kWh) की गणना के लिए उपयोग किया जाता है।",energy_insights_title:"ऊर्जा इंटेलिजेंस",idle_power_title:"निष्क्रिय शक्ति का पता लगाना",idle_active_msg:"मशीन निष्क्रिय है - अभी लगभग {kw} kW बर्बाद हो रहा है।",idle_none_msg:"कोई निष्क्रिय ऊर्जा बर्बादी नहीं पाई गई।",friction_loss_title:"पूर्वानुमानित ऊर्जा हानि",friction_active_msg:"बढ़ा हुआ घर्षण पाया गया: +{pct}% अतिरिक्त शक्ति (~{kw} kW अतिरिक्त)। हानि रोकने के लिए रखरखाव शेड्यूल करें।",friction_none_msg:"कोई असामान्य घर्षण नहीं पाया गया।",sec_title:"विशिष्ट ऊर्जा खपत",sec_label:"प्रति इकाई kWh",sec_unit:"kWh/इकाई",sec_no_data_msg:"यह मीट्रिक देखने के लिए मशीन जोड़ते समय दैनिक उत्पादन दर्ज करें।",optimal_load_title:"इष्टतम लोड ज़ोन",optimal_load_label:"इष्टतम लोड",current_load_label:"वर्तमान लोड",at_optimal_msg:"इष्टतम लोड ज़ोन में चल रहा है।",adjust_to_optimal_msg:"प्रति इकाई ऊर्जा कम करने के लिए लोड को {pct}% की ओर समायोजित करें।",nav_digital_twin:"डिजिटल ट्विन",twin_hint:"घुमाने के लिए खींचें, ज़ूम करने के लिए स्क्रॉल करें, लाइव विवरण देखने के लिए मशीन पर क्लिक करें।",twin_unavailable_msg:"3D दृश्य लोड नहीं हो सका (Three.js लाइब्रेरी के लिए अपना इंटरनेट कनेक्शन जांचें)।"},
  uz: {tagline:"Global sanoat intellekti platformasi",live_label:"Jonli",kpi_energy:"Energiya sarfi",kpi_efficiency:"Samaradorlik",kpi_active:"Faol stanoklar",kpi_alerts:"Ogohlantirishlar",kwh_unit:"kWh",chart_title:"Real vaqtdagi ko'rsatkichlar",machine_status_title:"Stanoklar holati",status_running:"Ishlamoqda",status_warning:"Ogohlantirish",status_critical:"Muhim",form_title:"Zavod ma'lumotlarini kiritish",factory_name_label:"Zavod nomi",machine_count_label:"Stanoklar soni",energy_cost_label:"Energiya narxi ($/kWh)",machine_type_label:"Stanok turi",temperature_label:"Harorat (°C)",vibration_label:"Tebranish (mm/s)",load_label:"Yuklama (%)",submit_btn:"Zavodni tahlil qilish",submitting:"Yangilanmoqda...",ai_panel_title:"AI tahlili",ai_placeholder:"AI tahlilini olish uchun zavod ma'lumotlarini yuboring.",ai_analyzing:"Tahlil qilinmoqda...",ai_risks:"Xavflar",ai_efficiency_insights:"Samaradorlik tahlili",ai_optimizations:"Optimallashtirish tavsiyalari",toast_updated:"Zavod ma'lumotlari yangilandi",toast_analysis_done:"AI tahlili yakunlandi",toast_error:"Xatolik yuz berdi",nav_dashboard:"Boshqaruv paneli",nav_factories:"Zavodlar",nav_ai_insights:"AI tahlili",logout_btn:"Chiqish",login_title:"Xush kelibsiz",login_subtitle:"FactoryPulse AI hisobingizga kiring",ph_email:"Elektron pochta",ph_password:"Parol",remember_me:"Meni eslab qol",login_btn:"Kirish",login_link_register:"Hisobingiz yo'qmi? Yarating",register_title:"Hisob yarating",register_subtitle:"Zavodlaringizni AI bilan kuzatishni boshlang",ph_full_name:"To'liq ism",ph_confirm_password:"Parolni tasdiqlang",register_btn:"Hisob yaratish",register_link_login:"Hisobingiz bormi? Kiring",err_missing_fields:"Barcha maydonlarni to'ldiring",err_invalid_email:"Yaroqli elektron pochta manzilini kiriting",err_weak_password:"Parol kamida 8 belgidan, harf va raqamdan iborat bo'lishi kerak",err_password_mismatch:"Parollar mos kelmaydi",err_invalid_credentials:"Elektron pochta yoki parol noto'g'ri",err_email_taken:"Bu elektron pochta allaqachon ro'yxatdan o'tgan",err_generic:"Xatolik yuz berdi. Qaytadan urinib ko'ring",my_factories_title:"Mening Zavodlarim",add_factory_btn:"+ Zavod qo'shish",edit_factory_btn:"Tahrirlash",delete_factory_btn:"O'chirish",confirm_delete_factory:"Bu zavodni o'chirasizmi? Buni bekor qilib bo'lmaydi.",no_factories_yet:"Siz hali hech qanday zavod qo'shmagansiz.",factory_created_toast:"Zavod yaratildi va tahlil qilindi",factory_updated_toast:"Zavod yangilandi",factory_deleted_toast:"Zavod o'chirildi",ai_insights_feed_title:"AI Tahlili Lentasi",no_ai_insights_yet:"Hali AI tahlili yo'q. Boshlash uchun zavod qo'shing.",reanalyze_btn:"Qayta tahlil qilish",view_insights_btn:"Tahlilni ko'rish",created_label:"Yaratilgan",cancel_btn:"Bekor qilish",save_btn:"O'zgarishlarni saqlash",nav_live_monitor:"Jonli monitoring",add_machine_scada_btn:"+ Stanok qo'shish",usb_status:"USB:",plc_status:"PLC:",polling_mode:"So'rov",live_chart_title:"Jonli sensor grafigi",machines_table_title:"Stanoklar",machine_code_col:"Kod",machine_name_col:"Nomi",status_col:"Holat",risk_col:"Xavf",no_machines_yet:"Hali stanoklar yo'q. \"+ Stanok qo'shish\"ni bosing.",section_machine_info:"Stanok ma'lumoti",section_sensor_data:"Sensor ma'lumotlari",section_status:"Holat",section_notes:"Eslatmalar",status_stopped:"To'xtatilgan",status_maintenance:"Texnik xizmat",priority_low:"Past",priority_normal:"Oddiy",priority_high:"Yuqori",priority_critical:"Muhim",save_and_analyze_btn:"Saqlash va tahlil qilish",source_col:"Manba",source_auto:"Avtomatik (SCADA)",source_manual:"Qo'lda",nav_alerts:"Ogohlantirishlar",acknowledge_btn:"Tasdiqlash",acknowledged_label:"Tasdiqlangan",acknowledge_all_btn:"Barchasini Tasdiqlash",no_alerts_yet:"Ogohlantirishlar yo'q. Hammasi yaxshi ishlamoqda.",download_report_btn:"Hisobot",alert_details_template:"Harorat {temp}°C, tebranish {vib} mm/s, holat: {status}",section_energy_intel:"Energiya intellekti",daily_output_hint:"Solishtirma energiya sarfini (birlik uchun kVt·soat) hisoblash uchun ishlatiladi.",energy_insights_title:"Energiya intellekti",idle_power_title:"Bo'sh yurish quvvatini aniqlash",idle_active_msg:"Stanok bo'sh turibdi - hozir taxminan {kw} kVt behuda sarflanmoqda.",idle_none_msg:"Bo'sh yurish energiya isrofi aniqlanmadi.",friction_loss_title:"Bashoratli energiya yo'qotishi",friction_active_msg:"Oshgan ishqalanish aniqlandi: +{pct}% ortiqcha quvvat (~{kw} kVt qo'shimcha). Isrofni oldini olish uchun texnik xizmatni rejalashtiring.",friction_none_msg:"Anomal ishqalanish aniqlanmadi.",sec_title:"Solishtirma energiya sarfi",sec_label:"birlik uchun kVt·soat",sec_unit:"kVt·soat/birlik",sec_no_data_msg:"Bu ko'rsatkichni ko'rish uchun stanok qo'shishda kunlik ishlab chiqarishni kiriting.",optimal_load_title:"Optimal yuklama zonasi",optimal_load_label:"Optimal yuklama",current_load_label:"Joriy yuklama",at_optimal_msg:"Optimal yuklama zonasida ishlamoqda.",adjust_to_optimal_msg:"Birlik uchun energiyani minimallashtirish uchun yuklamani {pct}% ga moslang.",nav_digital_twin:"Raqamli egizak",twin_hint:"Aylantirish uchun torting, kattalashtirish uchun aylantiring, jonli tafsilotlar uchun stanokni bosing.",twin_unavailable_msg:"3D ko'rinishni yuklab bo'lmadi (Three.js kutubxonasi uchun internet aloqangizni tekshiring)."},
  ky: {tagline:"Глобалдык өнөр жай интеллект платформасы",live_label:"Түз эфир",kpi_energy:"Энергия сарпталышы",kpi_efficiency:"Эффективдүүлүк",kpi_active:"Активдүү станоктор",kpi_alerts:"Дабылдар",kwh_unit:"кВт·саат",chart_title:"Реалдуу убакыттагы көрсөткүчтөр",machine_status_title:"Станоктордун абалы",status_running:"Иштеп жатат",status_warning:"Эскертүү",status_critical:"Олуттуу",form_title:"Завод маалыматтарын киргизүү",factory_name_label:"Заводдун аты",machine_count_label:"Станоктордун саны",energy_cost_label:"Энергия наркы ($/кВт·саат)",machine_type_label:"Станоктун түрү",temperature_label:"Температура (°C)",vibration_label:"Дирилдөө (мм/с)",load_label:"Жүктөм (%)",submit_btn:"Заводду талдоо",submitting:"Жаңыртылууда...",ai_panel_title:"AI-талдоо",ai_placeholder:"AI-талдоо алуу үчүн завод маалыматтарын жөнөтүңүз.",ai_analyzing:"Талдануда...",ai_risks:"Тобокелдиктер",ai_efficiency_insights:"Эффективдүүлүк талдоосу",ai_optimizations:"Оптималдаштыруу сунуштары",toast_updated:"Завод маалыматтары жаңыртылды",toast_analysis_done:"AI-талдоо аяктады",toast_error:"Ката кетти",nav_dashboard:"Башкаруу панели",nav_factories:"Заводдор",nav_ai_insights:"AI-талдоо",logout_btn:"Чыгуу",login_title:"Кайра кош келиңиз",login_subtitle:"FactoryPulse AI каттоо эсебиңизге кириңиз",ph_email:"Электрондук почта",ph_password:"Сырсөз",remember_me:"Мени эстеп кал",login_btn:"Кирүү",login_link_register:"Каттоо эсебиңиз жокпу? Түзүү",register_title:"Каттоо эсебин түзүү",register_subtitle:"Заводдоруңузду AI менен байкоону баштаңыз",ph_full_name:"Толук аты-жөнү",ph_confirm_password:"Сырсөздү ырастаңыз",register_btn:"Каттоо эсебин түзүү",register_link_login:"Каттоо эсебиңиз барбы? Кирүү",err_missing_fields:"Бардык талааларды толтуруңуз",err_invalid_email:"Жарактуу электрондук почта дарегин киргизиңиз",err_weak_password:"Сырсөз кеминде 8 белги, тамга жана сан камтышы керек",err_password_mismatch:"Сырсөздөр дал келбейт",err_invalid_credentials:"Электрондук почта же сырсөз туура эмес",err_email_taken:"Бул электрондук почта мурунтан катталган",err_generic:"Ката кетти. Кайра аракет кылыңыз",my_factories_title:"Менин Заводдорум",add_factory_btn:"+ Завод кошуу",edit_factory_btn:"Түзөтүү",delete_factory_btn:"Өчүрүү",confirm_delete_factory:"Бул заводду өчүрөсүзбү? Бул аракетти артка кайтарууга болбойт.",no_factories_yet:"Сиз азырынча эч кандай завод кошкон жоксуз.",factory_created_toast:"Завод түзүлдү жана талданды",factory_updated_toast:"Завод жаңыртылды",factory_deleted_toast:"Завод өчүрүлдү",ai_insights_feed_title:"AI-талдоо тизмеси",no_ai_insights_yet:"Азырынча AI-талдоо жок. Баштоо үчүн завод кошуңуз.",reanalyze_btn:"Кайра талдоо",view_insights_btn:"Талдоону көрүү",created_label:"Түзүлгөн күнү",cancel_btn:"Жокко чыгаруу",save_btn:"Өзгөртүүлөрдү сактоо",nav_live_monitor:"Түз мониторинг",add_machine_scada_btn:"+ Станок кошуу",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Сурам",live_chart_title:"Реалдуу убакыттагы сенсор графиги",machines_table_title:"Станоктор",machine_code_col:"Код",machine_name_col:"Аты",status_col:"Абалы",risk_col:"Тобокелдик",no_machines_yet:"Азырынча станоктор жок. \"+ Станок кошуу\"ну басыңыз.",section_machine_info:"Станок маалыматы",section_sensor_data:"Сенсор маалыматтары",section_status:"Абалы",section_notes:"Эскертүүлөр",status_stopped:"Токтотулган",status_maintenance:"Тейлөө",priority_low:"Төмөн",priority_normal:"Кадимки",priority_high:"Жогору",priority_critical:"Олуттуу",save_and_analyze_btn:"Сактоо жана талдоо",source_col:"Булак",source_auto:"Автоматтык (SCADA)",source_manual:"Кол менен",nav_alerts:"Дабылдар",acknowledge_btn:"Ырастоо",acknowledged_label:"Ырасталды",acknowledge_all_btn:"Баарын ырастоо",no_alerts_yet:"Дабылдар жок. Баары жакшы иштеп жатат.",download_report_btn:"Отчет",alert_details_template:"Температура {temp}°C, дирилдөө {vib} мм/с, абалы: {status}",section_energy_intel:"Энергия интеллекти",daily_output_hint:"Бирдикке кеткен энергия сарптоону (бирдик үчүн кВт·саат) эсептөө үчүн колдонулат.",energy_insights_title:"Энергия интеллекти",idle_power_title:"Бош жүрүштү аныктоо",idle_active_msg:"Станок бош турат - учурда болжол менен {kw} кВт бекер коротулуп жатат.",idle_none_msg:"Бош жүрүш чыгымы табылган жок.",friction_loss_title:"Болжолдуу энергия жоготуусу",friction_active_msg:"Жогорулаган үйкөлүш табылды: +{pct}% кошумча кубат (~{kw} кВт кошумча). Чыгымдын алдын алуу үчүн тейлөөнү пландаштырыңыз.",friction_none_msg:"Аномалдуу үйкөлүш табылган жок.",sec_title:"Бирдикке кеткен энергия",sec_label:"бирдик үчүн кВт·саат",sec_unit:"кВт·саат/бирдик",sec_no_data_msg:"Бул көрсөткүчтү көрүү үчүн станок кошууда күндөлүк өндүрүштү киргизиңиз.",optimal_load_title:"Оптималдуу жүктөм аймагы",optimal_load_label:"Оптималдуу жүктөм",current_load_label:"Учурдагы жүктөм",at_optimal_msg:"Оптималдуу жүктөм аймагында иштеп жатат.",adjust_to_optimal_msg:"Бирдикке кеткен энергияны азайтуу үчүн жүктөмдү {pct}%га жакындатыңыз.",nav_digital_twin:"Санарип эгиз",twin_hint:"Айландыруу үчүн сүйрөңүз, чоңойтуу үчүн айландырыңыз, түз маалымат үчүн станокту басыңыз.",twin_unavailable_msg:"3D көрүнүш жүктөлгөн жок (Three.js китепканасы үчүн интернет байланышын текшериңиз)."},
  uk: {tagline:"Глобальна платформа промислового інтелекту",live_label:"Наживо",kpi_energy:"Споживання енергії",kpi_efficiency:"Ефективність",kpi_active:"Активні верстати",kpi_alerts:"Сповіщення",kwh_unit:"кВт·год",chart_title:"Показники в реальному часі",machine_status_title:"Статус верстатів",status_running:"Працює",status_warning:"Попередження",status_critical:"Критично",form_title:"Введення даних заводу",factory_name_label:"Назва заводу",machine_count_label:"Кількість верстатів",energy_cost_label:"Вартість енергії ($/кВт·год)",machine_type_label:"Тип верстата",temperature_label:"Температура (°C)",vibration_label:"Вібрація (мм/с)",load_label:"Навантаження (%)",submit_btn:"Аналізувати завод",submitting:"Оновлення...",ai_panel_title:"AI-аналітика",ai_placeholder:"Надішліть дані заводу, щоб отримати AI-аналіз.",ai_analyzing:"Аналіз...",ai_risks:"Ризики",ai_efficiency_insights:"Аналіз ефективності",ai_optimizations:"Рекомендації з оптимізації",toast_updated:"Дані заводу оновлено",toast_analysis_done:"AI-аналіз завершено",toast_error:"Сталася помилка",nav_dashboard:"Панель",nav_factories:"Заводи",nav_ai_insights:"AI-аналітика",logout_btn:"Вийти",login_title:"З поверненням",login_subtitle:"Увійдіть у свій обліковий запис FactoryPulse AI",ph_email:"Електронна пошта",ph_password:"Пароль",remember_me:"Запам'ятати мене",login_btn:"Увійти",login_link_register:"Немає акаунту? Створити",register_title:"Створіть акаунт",register_subtitle:"Почніть моніторинг заводів за допомогою AI",ph_full_name:"Повне ім'я",ph_confirm_password:"Підтвердіть пароль",register_btn:"Створити акаунт",register_link_login:"Вже є акаунт? Увійти",err_missing_fields:"Будь ласка, заповніть усі поля",err_invalid_email:"Введіть дійсну електронну адресу",err_weak_password:"Пароль має містити щонайменше 8 символів, літеру та цифру",err_password_mismatch:"Паролі не збігаються",err_invalid_credentials:"Невірна електронна пошта або пароль",err_email_taken:"Ця електронна пошта вже зареєстрована",err_generic:"Сталася помилка. Спробуйте ще раз",my_factories_title:"Мої Заводи",add_factory_btn:"+ Додати завод",edit_factory_btn:"Редагувати",delete_factory_btn:"Видалити",confirm_delete_factory:"Видалити цей завод? Цю дію не можна скасувати.",no_factories_yet:"Ви ще не додали жодного заводу.",factory_created_toast:"Завод створено та проаналізовано",factory_updated_toast:"Завод оновлено",factory_deleted_toast:"Завод видалено",ai_insights_feed_title:"Стрічка AI-аналітики",no_ai_insights_yet:"Ще немає AI-аналітики. Додайте завод.",reanalyze_btn:"Проаналізувати знову",view_insights_btn:"Переглянути аналітику",created_label:"Створено",cancel_btn:"Скасувати",save_btn:"Зберегти зміни",nav_live_monitor:"Моніторинг",add_machine_scada_btn:"+ Додати верстат",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Опитування",live_chart_title:"Графік датчиків у реальному часі",machines_table_title:"Верстати",machine_code_col:"Код",machine_name_col:"Назва",status_col:"Статус",risk_col:"Ризик",no_machines_yet:"Верстатів поки немає. Натисніть «+ Додати верстат».",section_machine_info:"Інформація про верстат",section_sensor_data:"Дані датчиків",section_status:"Статус",section_notes:"Примітки",status_stopped:"Зупинено",status_maintenance:"Обслуговування",priority_low:"Низький",priority_normal:"Звичайний",priority_high:"Високий",priority_critical:"Критичний",save_and_analyze_btn:"Зберегти та проаналізувати",source_col:"Джерело",source_auto:"Авто (SCADA)",source_manual:"Вручну",nav_alerts:"Сповіщення",acknowledge_btn:"Підтвердити",acknowledged_label:"Підтверджено",acknowledge_all_btn:"Підтвердити все",no_alerts_yet:"Сповіщень немає. Все працює нормально.",download_report_btn:"Звіт",alert_details_template:"Температура {temp}°C, вібрація {vib} мм/с, статус: {status}",section_energy_intel:"Енергетичний інтелект",daily_output_hint:"Використовується для розрахунку питомого енергоспоживання (кВт·год на одиницю).",energy_insights_title:"Енергетичний інтелект",idle_power_title:"Виявлення холостого ходу",idle_active_msg:"Верстат простоює — зараз витрачається приблизно {kw} кВт даремно.",idle_none_msg:"Втрат енергії на холостому ходу не виявлено.",friction_loss_title:"Прогноз втрат енергії",friction_active_msg:"Виявлено підвищене тертя: +{pct}% зайвої потужності (~{kw} кВт). Заплануйте обслуговування, щоб уникнути втрат.",friction_none_msg:"Аномального тертя не виявлено.",sec_title:"Питоме енергоспоживання",sec_label:"кВт·год на одиницю",sec_unit:"кВт·год/од.",sec_no_data_msg:"Вкажіть добовий обсяг випуску при додаванні верстата, щоб побачити цей показник.",optimal_load_title:"Оптимальна зона навантаження",optimal_load_label:"Оптимальне навантаження",current_load_label:"Поточне навантаження",at_optimal_msg:"Працює в оптимальній зоні навантаження.",adjust_to_optimal_msg:"Наблизьте навантаження до {pct}%, щоб мінімізувати енергію на одиницю.",nav_digital_twin:"Цифровий двійник",twin_hint:"Перетягуйте для повороту, прокручуйте для масштабування, натисніть на верстат для деталей у реальному часі.",twin_unavailable_msg:"Не вдалося завантажити 3D-вигляд (перевірте підключення до інтернету для Three.js)."},
  pl: {tagline:"Globalna Platforma Inteligencji Przemysłowej",live_label:"Na żywo",kpi_energy:"Zużycie Energii",kpi_efficiency:"Wydajność",kpi_active:"Aktywne Maszyny",kpi_alerts:"Alerty",kwh_unit:"kWh",chart_title:"Wydajność w Czasie Rzeczywistym",machine_status_title:"Status Maszyn",status_running:"Działa",status_warning:"Ostrzeżenie",status_critical:"Krytyczne",form_title:"Wprowadzanie Danych Fabryki",factory_name_label:"Nazwa Fabryki",machine_count_label:"Liczba Maszyn",energy_cost_label:"Koszt Energii ($/kWh)",machine_type_label:"Typ Maszyny",temperature_label:"Temperatura (°C)",vibration_label:"Wibracje (mm/s)",load_label:"Obciążenie (%)",submit_btn:"Analizuj Fabrykę",submitting:"Aktualizowanie...",ai_panel_title:"Analizy AI",ai_placeholder:"Prześlij dane fabryki, aby wygenerować analizę AI.",ai_analyzing:"Analizowanie...",ai_risks:"Ryzyka",ai_efficiency_insights:"Analiza Wydajności",ai_optimizations:"Sugestie Optymalizacji",toast_updated:"Dane fabryki zaktualizowane",toast_analysis_done:"Analiza AI zakończona",toast_error:"Coś poszło nie tak",nav_dashboard:"Panel",nav_factories:"Fabryki",nav_ai_insights:"Analizy AI",logout_btn:"Wyloguj",login_title:"Witamy z powrotem",login_subtitle:"Zaloguj się do swojego konta FactoryPulse AI",ph_email:"E-mail",ph_password:"Hasło",remember_me:"Zapamiętaj mnie",login_btn:"Zaloguj się",login_link_register:"Nie masz konta? Utwórz je",register_title:"Utwórz konto",register_subtitle:"Zacznij monitorować swoje fabryki z AI",ph_full_name:"Imię i Nazwisko",ph_confirm_password:"Potwierdź Hasło",register_btn:"Utwórz Konto",register_link_login:"Masz już konto? Zaloguj się",err_missing_fields:"Proszę wypełnić wszystkie pola",err_invalid_email:"Proszę podać prawidłowy adres e-mail",err_weak_password:"Hasło musi mieć min. 8 znaków, literę i cyfrę",err_password_mismatch:"Hasła nie pasują do siebie",err_invalid_credentials:"Nieprawidłowy e-mail lub hasło",err_email_taken:"Ten e-mail jest już zarejestrowany",err_generic:"Coś poszło nie tak. Spróbuj ponownie",my_factories_title:"Moje Fabryki",add_factory_btn:"+ Dodaj Fabrykę",edit_factory_btn:"Edytuj",delete_factory_btn:"Usuń",confirm_delete_factory:"Usunąć tę fabrykę? Tej czynności nie można cofnąć.",no_factories_yet:"Nie dodałeś jeszcze żadnej fabryki.",factory_created_toast:"Fabryka utworzona i przeanalizowana",factory_updated_toast:"Fabryka zaktualizowana",factory_deleted_toast:"Fabryka usunięta",ai_insights_feed_title:"Kanał Analiz AI",no_ai_insights_yet:"Brak analiz AI. Dodaj fabrykę, aby zacząć.",reanalyze_btn:"Analizuj Ponownie",view_insights_btn:"Zobacz Analizy",created_label:"Utworzono",cancel_btn:"Anuluj",save_btn:"Zapisz Zmiany",nav_live_monitor:"Monitoring na Żywo",add_machine_scada_btn:"+ Dodaj Maszynę",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Odpytywanie",live_chart_title:"Wykres Czujników na Żywo",machines_table_title:"Maszyny",machine_code_col:"Kod",machine_name_col:"Nazwa",status_col:"Status",risk_col:"Ryzyko",no_machines_yet:"Brak maszyn. Kliknij „+ Dodaj Maszynę”.",section_machine_info:"Informacje o Maszynie",section_sensor_data:"Dane Czujników",section_status:"Status",section_notes:"Notatki",status_stopped:"Zatrzymana",status_maintenance:"Konserwacja",priority_low:"Niski",priority_normal:"Normalny",priority_high:"Wysoki",priority_critical:"Krytyczny",save_and_analyze_btn:"Zapisz i Analizuj",source_col:"Źródło",source_auto:"Auto (SCADA)",source_manual:"Ręcznie",nav_alerts:"Alerty",acknowledge_btn:"Potwierdź",acknowledged_label:"Potwierdzone",acknowledge_all_btn:"Potwierdź Wszystkie",no_alerts_yet:"Brak alertów. Wszystko działa prawidłowo.",download_report_btn:"Raport",alert_details_template:"Temperatura {temp}°C, wibracje {vib} mm/s, status: {status}",section_energy_intel:"Inteligencja Energetyczna",daily_output_hint:"Używane do obliczania jednostkowego zużycia energii (kWh na jednostkę).",energy_insights_title:"Inteligencja Energetyczna",idle_power_title:"Wykrywanie Mocy Jałowej",idle_active_msg:"Maszyna bezczynna - obecnie marnowane jest ok. {kw} kW.",idle_none_msg:"Nie wykryto marnowania energii w trybie bezczynności.",friction_loss_title:"Predykcyjna Utrata Energii",friction_active_msg:"Wykryto zwiększone tarcie: +{pct}% dodatkowej mocy (~{kw} kW więcej). Zaplanuj konserwację, aby zapobiec stratom.",friction_none_msg:"Nie wykryto nieprawidłowego tarcia.",sec_title:"Jednostkowe Zużycie Energii",sec_label:"kWh na jednostkę",sec_unit:"kWh/jednostkę",sec_no_data_msg:"Podaj dzienną produkcję podczas dodawania maszyny, aby zobaczyć ten wskaźnik.",optimal_load_title:"Optymalna Strefa Obciążenia",optimal_load_label:"Optymalne obciążenie",current_load_label:"Obecne obciążenie",at_optimal_msg:"Działa w optymalnej strefie obciążenia.",adjust_to_optimal_msg:"Dostosuj obciążenie do {pct}%, aby zminimalizować energię na jednostkę.",nav_digital_twin:"Cyfrowy Bliźniak",twin_hint:"Przeciągnij, aby obrócić, przewiń, aby powiększyć, kliknij maszynę, aby zobaczyć szczegóły na żywo.",twin_unavailable_msg:"Nie udało się załadować widoku 3D (sprawdź połączenie internetowe dla biblioteki Three.js)."},
  nl: {tagline:"Wereldwijd Industrieel Intelligentieplatform",live_label:"Live",kpi_energy:"Energieverbruik",kpi_efficiency:"Efficiëntie",kpi_active:"Actieve Machines",kpi_alerts:"Meldingen",kwh_unit:"kWh",chart_title:"Realtime Prestaties",machine_status_title:"Machinestatus",status_running:"Actief",status_warning:"Waarschuwing",status_critical:"Kritiek",form_title:"Fabrieksgegevens Invoeren",factory_name_label:"Fabrieksnaam",machine_count_label:"Aantal Machines",energy_cost_label:"Energiekosten ($/kWh)",machine_type_label:"Machinetype",temperature_label:"Temperatuur (°C)",vibration_label:"Trilling (mm/s)",load_label:"Belasting (%)",submit_btn:"Fabriek Analyseren",submitting:"Bijwerken...",ai_panel_title:"AI-inzichten",ai_placeholder:"Verzend fabrieksgegevens om een AI-analyse te genereren.",ai_analyzing:"Analyseren...",ai_risks:"Risico's",ai_efficiency_insights:"Efficiëntieanalyse",ai_optimizations:"Optimalisatiesuggesties",toast_updated:"Fabrieksgegevens bijgewerkt",toast_analysis_done:"AI-analyse voltooid",toast_error:"Er is iets misgegaan",nav_dashboard:"Dashboard",nav_factories:"Fabrieken",nav_ai_insights:"AI-inzichten",logout_btn:"Uitloggen",login_title:"Welkom terug",login_subtitle:"Log in op uw FactoryPulse AI-account",ph_email:"E-mail",ph_password:"Wachtwoord",remember_me:"Onthoud mij",login_btn:"Inloggen",login_link_register:"Geen account? Maak er een",register_title:"Maak uw account aan",register_subtitle:"Begin met AI-monitoring van uw fabrieken",ph_full_name:"Volledige Naam",ph_confirm_password:"Bevestig Wachtwoord",register_btn:"Account Aanmaken",register_link_login:"Heeft u al een account? Inloggen",err_missing_fields:"Vul alle velden in",err_invalid_email:"Voer een geldig e-mailadres in",err_weak_password:"Wachtwoord moet minimaal 8 tekens, een letter en een cijfer bevatten",err_password_mismatch:"Wachtwoorden komen niet overeen",err_invalid_credentials:"Ongeldige e-mail of wachtwoord",err_email_taken:"Dit e-mailadres is al geregistreerd",err_generic:"Er is iets misgegaan. Probeer het opnieuw",my_factories_title:"Mijn Fabrieken",add_factory_btn:"+ Fabriek Toevoegen",edit_factory_btn:"Bewerken",delete_factory_btn:"Verwijderen",confirm_delete_factory:"Deze fabriek verwijderen? Dit kan niet ongedaan worden gemaakt.",no_factories_yet:"U heeft nog geen fabrieken toegevoegd.",factory_created_toast:"Fabriek aangemaakt en geanalyseerd",factory_updated_toast:"Fabriek bijgewerkt",factory_deleted_toast:"Fabriek verwijderd",ai_insights_feed_title:"AI-inzichten Feed",no_ai_insights_yet:"Nog geen AI-inzichten. Voeg een fabriek toe.",reanalyze_btn:"Opnieuw Analyseren",view_insights_btn:"Bekijk Inzichten",created_label:"Aangemaakt",cancel_btn:"Annuleren",save_btn:"Wijzigingen Opslaan",nav_live_monitor:"Live Monitor",add_machine_scada_btn:"+ Machine Toevoegen",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Polling",live_chart_title:"Live Sensorgrafiek",machines_table_title:"Machines",machine_code_col:"Code",machine_name_col:"Naam",status_col:"Status",risk_col:"Risico",no_machines_yet:"Nog geen machines. Klik op „+ Machine Toevoegen”.",section_machine_info:"Machine-informatie",section_sensor_data:"Sensorgegevens",section_status:"Status",section_notes:"Notities",status_stopped:"Gestopt",status_maintenance:"Onderhoud",priority_low:"Laag",priority_normal:"Normaal",priority_high:"Hoog",priority_critical:"Kritiek",save_and_analyze_btn:"Opslaan & Analyseren",source_col:"Bron",source_auto:"Auto (SCADA)",source_manual:"Handmatig",nav_alerts:"Meldingen",acknowledge_btn:"Bevestigen",acknowledged_label:"Bevestigd",acknowledge_all_btn:"Alles Bevestigen",no_alerts_yet:"Geen meldingen. Alles werkt naar behoren.",download_report_btn:"Rapport",alert_details_template:"Temperatuur {temp}°C, trilling {vib} mm/s, status: {status}",section_energy_intel:"Energie-intelligentie",daily_output_hint:"Wordt gebruikt om het specifieke energieverbruik te berekenen (kWh per eenheid).",energy_insights_title:"Energie-intelligentie",idle_power_title:"Stationair Vermogen Detectie",idle_active_msg:"Machine is inactief - momenteel wordt ongeveer {kw} kW verspild.",idle_none_msg:"Geen stationaire energieverspilling gedetecteerd.",friction_loss_title:"Voorspellend Energieverlies",friction_active_msg:"Verhoogde wrijving gedetecteerd: +{pct}% extra vermogen (~{kw} kW extra). Plan onderhoud om verliezen te voorkomen.",friction_none_msg:"Geen abnormale wrijving gedetecteerd.",sec_title:"Specifiek Energieverbruik",sec_label:"kWh per eenheid",sec_unit:"kWh/eenheid",sec_no_data_msg:"Voer de dagelijkse output in bij het toevoegen van deze machine om deze metriek te zien.",optimal_load_title:"Optimale Belastingzone",optimal_load_label:"Optimale belasting",current_load_label:"Huidige belasting",at_optimal_msg:"Draait in de optimale belastingzone.",adjust_to_optimal_msg:"Pas de belasting aan naar {pct}% om energie per eenheid te minimaliseren.",nav_digital_twin:"Digitale Tweeling",twin_hint:"Sleep om te draaien, scroll om te zoomen, klik op een machine voor live details.",twin_unavailable_msg:"3D-weergave kon niet worden geladen (controleer uw internetverbinding voor Three.js)."},
  sv: {tagline:"Global Industriell Intelligensplattform",live_label:"Live",kpi_energy:"Energiförbrukning",kpi_efficiency:"Effektivitet",kpi_active:"Aktiva Maskiner",kpi_alerts:"Varningar",kwh_unit:"kWh",chart_title:"Realtidsprestanda",machine_status_title:"Maskinstatus",status_running:"Igång",status_warning:"Varning",status_critical:"Kritisk",form_title:"Fabriksdatainmatning",factory_name_label:"Fabriksnamn",machine_count_label:"Antal Maskiner",energy_cost_label:"Energikostnad ($/kWh)",machine_type_label:"Maskintyp",temperature_label:"Temperatur (°C)",vibration_label:"Vibration (mm/s)",load_label:"Belastning (%)",submit_btn:"Analysera Fabrik",submitting:"Uppdaterar...",ai_panel_title:"AI-insikter",ai_placeholder:"Skicka fabriksdata för att generera en AI-analys.",ai_analyzing:"Analyserar...",ai_risks:"Risker",ai_efficiency_insights:"Effektivitetsanalys",ai_optimizations:"Optimeringsförslag",toast_updated:"Fabriksdata uppdaterad",toast_analysis_done:"AI-analys klar",toast_error:"Något gick fel",nav_dashboard:"Instrumentpanel",nav_factories:"Fabriker",nav_ai_insights:"AI-insikter",logout_btn:"Logga ut",login_title:"Välkommen tillbaka",login_subtitle:"Logga in på ditt FactoryPulse AI-konto",ph_email:"E-post",ph_password:"Lösenord",remember_me:"Kom ihåg mig",login_btn:"Logga in",login_link_register:"Inget konto? Skapa ett",register_title:"Skapa ditt konto",register_subtitle:"Börja övervaka dina fabriker med AI",ph_full_name:"Fullständigt Namn",ph_confirm_password:"Bekräfta Lösenord",register_btn:"Skapa Konto",register_link_login:"Har du redan ett konto? Logga in",err_missing_fields:"Vänligen fyll i alla fält",err_invalid_email:"Ange en giltig e-postadress",err_weak_password:"Lösenordet måste vara minst 8 tecken med en bokstav och en siffra",err_password_mismatch:"Lösenorden matchar inte",err_invalid_credentials:"Felaktig e-post eller lösenord",err_email_taken:"Denna e-post är redan registrerad",err_generic:"Något gick fel. Försök igen",my_factories_title:"Mina Fabriker",add_factory_btn:"+ Lägg till Fabrik",edit_factory_btn:"Redigera",delete_factory_btn:"Ta bort",confirm_delete_factory:"Ta bort denna fabrik? Detta kan inte ångras.",no_factories_yet:"Du har inte lagt till några fabriker än.",factory_created_toast:"Fabrik skapad och analyserad",factory_updated_toast:"Fabrik uppdaterad",factory_deleted_toast:"Fabrik borttagen",ai_insights_feed_title:"AI-insikter Flöde",no_ai_insights_yet:"Inga AI-insikter än. Lägg till en fabrik.",reanalyze_btn:"Analysera Igen",view_insights_btn:"Visa Insikter",created_label:"Skapad",cancel_btn:"Avbryt",save_btn:"Spara Ändringar",nav_live_monitor:"Livemonitor",add_machine_scada_btn:"+ Lägg till Maskin",usb_status:"USB:",plc_status:"PLC:",polling_mode:"Polling",live_chart_title:"Live Sensordiagram",machines_table_title:"Maskiner",machine_code_col:"Kod",machine_name_col:"Namn",status_col:"Status",risk_col:"Risk",no_machines_yet:"Inga maskiner än. Klicka på \"+ Lägg till Maskin\".",section_machine_info:"Maskininformation",section_sensor_data:"Sensordata",section_status:"Status",section_notes:"Anteckningar",status_stopped:"Stoppad",status_maintenance:"Underhåll",priority_low:"Låg",priority_normal:"Normal",priority_high:"Hög",priority_critical:"Kritisk",save_and_analyze_btn:"Spara & Analysera",source_col:"Källa",source_auto:"Auto (SCADA)",source_manual:"Manuell",nav_alerts:"Varningar",acknowledge_btn:"Bekräfta",acknowledged_label:"Bekräftad",acknowledge_all_btn:"Bekräfta Alla",no_alerts_yet:"Inga varningar. Allt fungerar smidigt.",download_report_btn:"Rapport",alert_details_template:"Temperatur {temp}°C, vibration {vib} mm/s, status: {status}",section_energy_intel:"Energiintelligens",daily_output_hint:"Används för att beräkna specifik energiförbrukning (kWh per enhet).",energy_insights_title:"Energiintelligens",idle_power_title:"Detektering av Tomgångseffekt",idle_active_msg:"Maskinen står i tomgång - ungefär {kw} kW slösas just nu.",idle_none_msg:"Ingen tomgångsenergiförlust upptäckt.",friction_loss_title:"Prediktiv Energiförlust",friction_active_msg:"Ökad friktion upptäckt: +{pct}% extra effekt (~{kw} kW extra). Schemalägg underhåll för att förhindra förluster.",friction_none_msg:"Ingen onormal friktion upptäckt.",sec_title:"Specifik Energiförbrukning",sec_label:"kWh per enhet",sec_unit:"kWh/enhet",sec_no_data_msg:"Ange daglig produktion när du lägger till denna maskin för att se detta mått.",optimal_load_title:"Optimal Belastningszon",optimal_load_label:"Optimal belastning",current_load_label:"Aktuell belastning",at_optimal_msg:"Körs i den optimala belastningszonen.",adjust_to_optimal_msg:"Justera belastningen mot {pct}% för att minimera energi per enhet.",nav_digital_twin:"Digital Tvilling",twin_hint:"Dra för att rotera, scrolla för att zooma, klicka på en maskin för live-detaljer.",twin_unavailable_msg:"3D-vyn kunde inte laddas (kontrollera din internetanslutning för Three.js)."},
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
  });
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
    const data = await api("/api/data");
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
    await api("/api/factory", { method: "POST", body: JSON.stringify(payload) });
    showToast(t("toast_updated"), "success");
    await refreshData();
    label.textContent = t("ai_analyzing");
    const result = await api("/api/analyze", { method: "POST", body: JSON.stringify(payload) });
    renderAiAnalysis(result.analysis);
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
  if (!authToken) { window.location.href = "/login"; return false; }
  try {
    const data = await authApi("/api/me");
    currentUser = data.user;
    document.getElementById("user-name").textContent = currentUser.full_name;
    document.getElementById("user-avatar").textContent = (currentUser.full_name || "?").trim().charAt(0).toUpperCase();
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
  ["dashboard", "factories", "live", "twin", "alerts", "ai"].forEach(p => {
    document.getElementById("page-" + p).classList.toggle("hidden", p !== page);
  });
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.toggle("active", b.dataset.page === page));
  document.querySelectorAll(".nav-btn-m").forEach(b => b.classList.toggle("active", b.dataset.page === page));
  if (page === "factories") loadFactories();
  if (page === "ai") loadAiFeed();
  if (page === "live") initLiveMonitor();
  if (page === "alerts") loadAlerts();
  if (page === "twin") initDigitalTwin();
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
    cell.textContent = update.risk + "%";
    cell.style.color = riskColor(update.risk);
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
      <td class="py-2 pr-3 font-semibold gauge-value cell-risk" style="color:${riskColor(m.failure_risk)}">${m.failure_risk}%</td>
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
  } catch (e) {}
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
    card.innerHTML = `
      <div>
        <div class="font-semibold text-sm" style="color:${severityColor(a.severity)}">${(a.machine_name || a.machine_code)} — ${a.severity.toUpperCase()}</div>
        <div class="text-sm text-slate-300 mt-1">${localizedAlertMessage(a)}</div>
        <div class="text-xs text-slate-500 mt-1 gauge-value">${time}</div>
      </div>
      ${a.acknowledged
        ? `<span class="text-xs text-slate-500 whitespace-nowrap">✓ ${t("acknowledged_label")}</span>`
        : `<button class="btn-ack-alert input-field rounded-lg px-3 py-1.5 text-xs whitespace-nowrap" data-id="${a.id}">${t("acknowledge_btn")}</button>`}
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
      new Notification("FactoryPulse AI — Critical Alert", { body: localizedMsg, tag: "fp-alert-" + alert.id });
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
    const res = await fetch("/api/report/pdf?days=30", { headers: { "Authorization": "Bearer " + authToken } });
    if (!res.ok) throw new Error("report_failed");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "FactoryPulseAI_Report.pdf";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
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
  document.getElementById("f-name").value = "Demo Factory";
  document.getElementById("f-count").value = 6;
  document.getElementById("f-cost").value = 0.12;
  document.getElementById("f-temp").value = 65;
  document.getElementById("f-vibration").value = 3.5;
  document.getElementById("f-load").value = 60;
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
<title>FactoryPulse AI - Login</title>
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
        <label class="flex items-center gap-2 text-xs text-slate-400 select-none cursor-pointer">
          <input id="remember" type="checkbox" class="w-4 h-4 rounded accent-cyan-400" />
          <span data-t="remember_me">Remember me</span>
        </label>
        <div id="form-error" class="hidden anim-error text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded-xl px-3 py-2"></div>
        <button type="submit" id="submit-btn" class="glow-btn rounded-xl py-3 text-sm font-semibold flex items-center justify-center gap-2 mt-1">
          <span id="submit-spinner" class="spinner hidden"></span>
          <span id="submit-label" data-t="login_btn">Log In</span>
        </button>
        <p class="text-xs text-slate-400 text-center mt-1">
          <a href="/register" class="text-cyan-400 hover:text-cyan-300 font-medium" data-t="login_link_register">Don't have an account? Create one</a>
        </p>
      </form>
    </div>
  </div>
</div>

<script>
const translations = {
  en: {tagline:"Global Industrial Intelligence Platform",live_label:"Live",kpi_energy:"Energy Usage",kpi_efficiency:"Efficiency",kpi_active:"Active Machines",kpi_alerts:"Alerts",kwh_unit:"kWh",chart_title:"Real-Time Performance",machine_status_title:"Machine Status",status_running:"Running",status_warning:"Warning",status_critical:"Critical",form_title:"Factory Data Input",factory_name_label:"Factory Name",machine_count_label:"Number of Machines",energy_cost_label:"Energy Cost ($/kWh)",machine_type_label:"Machine Type",temperature_label:"Temperature (°C)",vibration_label:"Vibration (mm/s)",load_label:"Load (%)",submit_btn:"Analyze Factory",submitting:"Updating...",ai_panel_title:"AI Insights",ai_placeholder:"Submit factory data to generate an AI analysis.",ai_analyzing:"Analyzing...",ai_risks:"Risks",ai_efficiency_insights:"Efficiency Insights",ai_optimizations:"Optimization Suggestions",toast_updated:"Factory data updated",toast_analysis_done:"AI analysis complete",toast_error:"Something went wrong",nav_dashboard:"Dashboard",nav_factories:"Factories",nav_ai_insights:"AI Insights",logout_btn:"Log Out",login_title:"Welcome back",login_subtitle:"Sign in to your FactoryPulse AI account",ph_email:"Email",ph_password:"Password",remember_me:"Remember me",login_btn:"Log In",login_link_register:"Don't have an account? Create one",register_title:"Create your account",register_subtitle:"Start monitoring your factories with AI",ph_full_name:"Full Name",ph_confirm_password:"Confirm Password",register_btn:"Create Account",register_link_login:"Already have an account? Sign in",err_missing_fields:"Please fill in all fields",err_invalid_email:"Please enter a valid email address",err_weak_password:"Password must be at least 8 characters with a letter and a number",err_password_mismatch:"Passwords do not match",err_invalid_credentials:"Invalid email or password",err_email_taken:"This email is already registered",err_generic:"Something went wrong. Please try again",my_factories_title:"My Factories",add_factory_btn:"+ Add Factory",edit_factory_btn:"Edit",delete_factory_btn:"Delete",confirm_delete_factory:"Delete this factory? This cannot be undone.",no_factories_yet:"You haven't added any factories yet.",factory_created_toast:"Factory created and analyzed",factory_updated_toast:"Factory updated",factory_deleted_toast:"Factory deleted",ai_insights_feed_title:"AI Insights Feed",no_ai_insights_yet:"No AI insights yet. Add a factory to get started.",reanalyze_btn:"Re-analyze",view_insights_btn:"View Insights",created_label:"Created",cancel_btn:"Cancel",save_btn:"Save Changes"},
  ru: {tagline:"Глобальная платформа промышленного интеллекта",live_label:"Live",kpi_energy:"Потребление энергии",kpi_efficiency:"Эффективность",kpi_active:"Активные станки",kpi_alerts:"Оповещения",kwh_unit:"кВт·ч",chart_title:"Показатели в реальном времени",machine_status_title:"Статус станков",status_running:"Работает",status_warning:"Внимание",status_critical:"Критично",form_title:"Ввод данных завода",factory_name_label:"Название завода",machine_count_label:"Количество станков",energy_cost_label:"Стоимость энергии ($/кВт·ч)",machine_type_label:"Тип станка",temperature_label:"Температура (°C)",vibration_label:"Вибрация (мм/с)",load_label:"Нагрузка (%)",submit_btn:"Анализировать завод",submitting:"Обновление...",ai_panel_title:"AI-аналитика",ai_placeholder:"Отправьте данные завода, чтобы получить AI-анализ.",ai_analyzing:"Анализ...",ai_risks:"Риски",ai_efficiency_insights:"Анализ эффективности",ai_optimizations:"Рекомендации по оптимизации",toast_updated:"Данные завода обновлены",toast_analysis_done:"AI-анализ завершён",toast_error:"Произошла ошибка",nav_dashboard:"Панель",nav_factories:"Заводы",nav_ai_insights:"AI-аналитика",logout_btn:"Выход",login_title:"С возвращением",login_subtitle:"Войдите в аккаунт FactoryPulse AI",ph_email:"Email",ph_password:"Пароль",remember_me:"Запомнить меня",login_btn:"Войти",login_link_register:"Нет аккаунта? Создать",register_title:"Создать аккаунт",register_subtitle:"Начните мониторинг заводов с помощью AI",ph_full_name:"Полное имя",ph_confirm_password:"Подтвердите пароль",register_btn:"Создать аккаунт",register_link_login:"Уже есть аккаунт? Войти",err_missing_fields:"Заполните все поля",err_invalid_email:"Введите корректный email",err_weak_password:"Пароль должен быть от 8 символов, с буквой и цифрой",err_password_mismatch:"Пароли не совпадают",err_invalid_credentials:"Неверный email или пароль",err_email_taken:"Этот email уже зарегистрирован",err_generic:"Что-то пошло не так. Попробуйте снова",my_factories_title:"Мои заводы",add_factory_btn:"+ Добавить завод",edit_factory_btn:"Изменить",delete_factory_btn:"Удалить",confirm_delete_factory:"Удалить этот завод? Это действие нельзя отменить.",no_factories_yet:"Вы ещё не добавили ни одного завода.",factory_created_toast:"Завод создан и проанализирован",factory_updated_toast:"Завод обновлён",factory_deleted_toast:"Завод удалён",ai_insights_feed_title:"Лента AI-аналитики",no_ai_insights_yet:"Пока нет AI-аналитики. Добавьте завод, чтобы начать.",reanalyze_btn:"Проанализировать снова",view_insights_btn:"Смотреть аналитику",created_label:"Создано",cancel_btn:"Отмена",save_btn:"Сохранить изменения"},
  kk: {tagline:"Жаһандық өнеркәсіптік интеллект платформасы",live_label:"Тікелей эфир",kpi_energy:"Энергия тұтыну",kpi_efficiency:"Тиімділік",kpi_active:"Белсенді станоктар",kpi_alerts:"Дабылдар",kwh_unit:"кВт·сағ",chart_title:"Нақты уақыттағы көрсеткіштер",machine_status_title:"Станоктар күйі",status_running:"Жұмыс істеп тұр",status_warning:"Ескерту",status_critical:"Сыни",form_title:"Зауыт деректерін енгізу",factory_name_label:"Зауыт атауы",machine_count_label:"Станоктар саны",energy_cost_label:"Энергия құны ($/кВт·сағ)",machine_type_label:"Станок түрі",temperature_label:"Температура (°C)",vibration_label:"Діріл (мм/с)",load_label:"Жүктеме (%)",submit_btn:"Зауытты талдау",submitting:"Жаңартылуда...",ai_panel_title:"AI-талдау",ai_placeholder:"AI-талдау алу үшін зауыт деректерін жіберіңіз.",ai_analyzing:"Талдануда...",ai_risks:"Тәуекелдер",ai_efficiency_insights:"Тиімділік талдауы",ai_optimizations:"Оңтайландыру ұсыныстары",toast_updated:"Зауыт деректері жаңартылды",toast_analysis_done:"AI-талдау аяқталды",toast_error:"Қате орын алды",nav_dashboard:"Басқару тақтасы",nav_factories:"Зауыттар",nav_ai_insights:"AI-талдау",logout_btn:"Шығу",login_title:"Қайта қош келдіңіз",login_subtitle:"FactoryPulse AI аккаунтыңызға кіріңіз",ph_email:"Email",ph_password:"Құпия сөз",remember_me:"Мені есте сақтау",login_btn:"Кіру",login_link_register:"Аккаунтыңыз жоқ па? Тіркелу",register_title:"Аккаунт құру",register_subtitle:"Зауыттарды AI арқылы бақылауды бастаңыз",ph_full_name:"Толық аты-жөні",ph_confirm_password:"Құпия сөзді қайталаңыз",register_btn:"Аккаунт құру",register_link_login:"Аккаунтыңыз бар ма? Кіру",err_missing_fields:"Барлық өрістерді толтырыңыз",err_invalid_email:"Дұрыс email мекенжайын енгізіңіз",err_weak_password:"Құпия сөз кемінде 8 таңба, әріп пен сан болуы керек",err_password_mismatch:"Құпия сөздер сәйкес келмейді",err_invalid_credentials:"Қате email немесе құпия сөз",err_email_taken:"Бұл email тіркелген",err_generic:"Қате орын алды. Қайталап көріңіз",my_factories_title:"Менің зауыттарым",add_factory_btn:"+ Зауыт қосу",edit_factory_btn:"Өзгерту",delete_factory_btn:"Жою",confirm_delete_factory:"Бұл зауытты жоясыз ба? Бұл әрекетті кері қайтару мүмкін емес.",no_factories_yet:"Сіз әлі ешбір зауыт қосқан жоқсыз.",factory_created_toast:"Зауыт құрылды және талданды",factory_updated_toast:"Зауыт жаңартылды",factory_deleted_toast:"Зауыт жойылды",ai_insights_feed_title:"AI-талдау таспасы",no_ai_insights_yet:"AI-талдау әлі жоқ. Бастау үшін зауыт қосыңыз.",reanalyze_btn:"Қайта талдау",view_insights_btn:"Талдауды көру",created_label:"Құрылған күні",cancel_btn:"Бас тарту",save_btn:"Өзгерістерді сақтау"},
  de: {tagline:"Globale Industrielle Intelligenzplattform",live_label:"Live",kpi_energy:"Energieverbrauch",kpi_efficiency:"Effizienz",kpi_active:"Aktive Maschinen",kpi_alerts:"Warnungen",kwh_unit:"kWh",chart_title:"Echtzeit-Leistung",machine_status_title:"Maschinenstatus",status_running:"Läuft",status_warning:"Warnung",status_critical:"Kritisch",form_title:"Fabrikdateneingabe",factory_name_label:"Fabrikname",machine_count_label:"Anzahl der Maschinen",energy_cost_label:"Energiekosten ($/kWh)",machine_type_label:"Maschinentyp",temperature_label:"Temperatur (°C)",vibration_label:"Vibration (mm/s)",load_label:"Last (%)",submit_btn:"Fabrik Analysieren",submitting:"Aktualisieren...",ai_panel_title:"KI-Einblicke",ai_placeholder:"Senden Sie Fabrikdaten, um eine KI-Analyse zu erstellen.",ai_analyzing:"Analysiere...",ai_risks:"Risiken",ai_efficiency_insights:"Effizienzanalyse",ai_optimizations:"Optimierungsvorschläge",toast_updated:"Fabrikdaten aktualisiert",toast_analysis_done:"KI-Analyse abgeschlossen",toast_error:"Etwas ist schiefgelaufen",nav_dashboard:"Übersicht",nav_factories:"Fabriken",nav_ai_insights:"KI-Einblicke",logout_btn:"Abmelden",login_title:"Willkommen zurück",login_subtitle:"Melden Sie sich bei Ihrem FactoryPulse AI-Konto an",ph_email:"E-Mail",ph_password:"Passwort",remember_me:"Angemeldet bleiben",login_btn:"Einloggen",login_link_register:"Kein Konto? Jetzt erstellen",register_title:"Konto erstellen",register_subtitle:"Beginnen Sie mit der KI-Überwachung Ihrer Fabriken",ph_full_name:"Vollständiger Name",ph_confirm_password:"Passwort bestätigen",register_btn:"Konto erstellen",register_link_login:"Bereits ein Konto? Anmelden",err_missing_fields:"Bitte füllen Sie alle Felder aus",err_invalid_email:"Bitte geben Sie eine gültige E-Mail-Adresse ein",err_weak_password:"Passwort muss mind. 8 Zeichen, einen Buchstaben und eine Zahl enthalten",err_password_mismatch:"Passwörter stimmen nicht überein",err_invalid_credentials:"Ungültige E-Mail oder Passwort",err_email_taken:"Diese E-Mail ist bereits registriert",err_generic:"Etwas ist schiefgelaufen. Bitte erneut versuchen",my_factories_title:"Meine Fabriken",add_factory_btn:"+ Fabrik Hinzufügen",edit_factory_btn:"Bearbeiten",delete_factory_btn:"Löschen",confirm_delete_factory:"Diese Fabrik löschen? Dies kann nicht rückgängig gemacht werden.",no_factories_yet:"Sie haben noch keine Fabriken hinzugefügt.",factory_created_toast:"Fabrik erstellt und analysiert",factory_updated_toast:"Fabrik aktualisiert",factory_deleted_toast:"Fabrik gelöscht",ai_insights_feed_title:"KI-Einblicke Feed",no_ai_insights_yet:"Noch keine KI-Einblicke. Fügen Sie eine Fabrik hinzu.",reanalyze_btn:"Erneut analysieren",view_insights_btn:"Einblicke Anzeigen",created_label:"Erstellt",cancel_btn:"Abbrechen",save_btn:"Änderungen Speichern"},
  fr: {tagline:"Plateforme mondiale d'intelligence industrielle",live_label:"En direct",kpi_energy:"Consommation d'Énergie",kpi_efficiency:"Efficacité",kpi_active:"Machines Actives",kpi_alerts:"Alertes",kwh_unit:"kWh",chart_title:"Performance en Temps Réel",machine_status_title:"État des Machines",status_running:"En marche",status_warning:"Avertissement",status_critical:"Critique",form_title:"Saisie des Données d'Usine",factory_name_label:"Nom de l'Usine",machine_count_label:"Nombre de Machines",energy_cost_label:"Coût de l'Énergie ($/kWh)",machine_type_label:"Type de Machine",temperature_label:"Température (°C)",vibration_label:"Vibration (mm/s)",load_label:"Charge (%)",submit_btn:"Analyser l'Usine",submitting:"Mise à jour...",ai_panel_title:"Analyses IA",ai_placeholder:"Envoyez les données de l'usine pour générer une analyse IA.",ai_analyzing:"Analyse en cours...",ai_risks:"Risques",ai_efficiency_insights:"Analyse d'Efficacité",ai_optimizations:"Suggestions d'Optimisation",toast_updated:"Données d'usine mises à jour",toast_analysis_done:"Analyse IA terminée",toast_error:"Une erreur est survenue",nav_dashboard:"Tableau de Bord",nav_factories:"Usines",nav_ai_insights:"Analyses IA",logout_btn:"Déconnexion",login_title:"Content de vous revoir",login_subtitle:"Connectez-vous à votre compte FactoryPulse AI",ph_email:"E-mail",ph_password:"Mot de passe",remember_me:"Se souvenir de moi",login_btn:"Se connecter",login_link_register:"Pas de compte ? Créez-en un",register_title:"Créer votre compte",register_subtitle:"Commencez à surveiller vos usines avec l'IA",ph_full_name:"Nom Complet",ph_confirm_password:"Confirmer le Mot de Passe",register_btn:"Créer un Compte",register_link_login:"Déjà un compte ? Se connecter",err_missing_fields:"Veuillez remplir tous les champs",err_invalid_email:"Veuillez entrer une adresse e-mail valide",err_weak_password:"Le mot de passe doit contenir 8 caractères min., une lettre et un chiffre",err_password_mismatch:"Les mots de passe ne correspondent pas",err_invalid_credentials:"E-mail ou mot de passe incorrect",err_email_taken:"Cet e-mail est déjà enregistré",err_generic:"Une erreur est survenue. Veuillez réessayer",my_factories_title:"Mes Usines",add_factory_btn:"+ Ajouter une Usine",edit_factory_btn:"Modifier",delete_factory_btn:"Supprimer",confirm_delete_factory:"Supprimer cette usine ? Cette action est irréversible.",no_factories_yet:"Vous n'avez pas encore ajouté d'usine.",factory_created_toast:"Usine créée et analysée",factory_updated_toast:"Usine mise à jour",factory_deleted_toast:"Usine supprimée",ai_insights_feed_title:"Flux d'Analyses IA",no_ai_insights_yet:"Aucune analyse IA pour l'instant. Ajoutez une usine.",reanalyze_btn:"Réanalyser",view_insights_btn:"Voir les Analyses",created_label:"Créée le",cancel_btn:"Annuler",save_btn:"Enregistrer les Modifications"},
  es: {tagline:"Plataforma Global de Inteligencia Industrial",live_label:"En vivo",kpi_energy:"Uso de Energía",kpi_efficiency:"Eficiencia",kpi_active:"Máquinas Activas",kpi_alerts:"Alertas",kwh_unit:"kWh",chart_title:"Rendimiento en Tiempo Real",machine_status_title:"Estado de Máquinas",status_running:"Funcionando",status_warning:"Advertencia",status_critical:"Crítico",form_title:"Entrada de Datos de Fábrica",factory_name_label:"Nombre de Fábrica",machine_count_label:"Número de Máquinas",energy_cost_label:"Costo de Energía ($/kWh)",machine_type_label:"Tipo de Máquina",temperature_label:"Temperatura (°C)",vibration_label:"Vibración (mm/s)",load_label:"Carga (%)",submit_btn:"Analizar Fábrica",submitting:"Actualizando...",ai_panel_title:"Perspectivas IA",ai_placeholder:"Envíe datos de fábrica para generar un análisis IA.",ai_analyzing:"Analizando...",ai_risks:"Riesgos",ai_efficiency_insights:"Análisis de Eficiencia",ai_optimizations:"Sugerencias de Optimización",toast_updated:"Datos de fábrica actualizados",toast_analysis_done:"Análisis IA completo",toast_error:"Algo salió mal",nav_dashboard:"Panel",nav_factories:"Fábricas",nav_ai_insights:"Perspectivas IA",logout_btn:"Cerrar Sesión",login_title:"Bienvenido de nuevo",login_subtitle:"Inicia sesión en tu cuenta de FactoryPulse AI",ph_email:"Correo electrónico",ph_password:"Contraseña",remember_me:"Recuérdame",login_btn:"Iniciar Sesión",login_link_register:"¿No tienes cuenta? Crea una",register_title:"Crea tu cuenta",register_subtitle:"Empieza a monitorear tus fábricas con IA",ph_full_name:"Nombre Completo",ph_confirm_password:"Confirmar Contraseña",register_btn:"Crear Cuenta",register_link_login:"¿Ya tienes cuenta? Inicia sesión",err_missing_fields:"Por favor complete todos los campos",err_invalid_email:"Por favor ingrese un correo válido",err_weak_password:"La contraseña debe tener mín. 8 caracteres, una letra y un número",err_password_mismatch:"Las contraseñas no coinciden",err_invalid_credentials:"Correo o contraseña incorrectos",err_email_taken:"Este correo ya está registrado",err_generic:"Algo salió mal. Inténtalo de nuevo",my_factories_title:"Mis Fábricas",add_factory_btn:"+ Añadir Fábrica",edit_factory_btn:"Editar",delete_factory_btn:"Eliminar",confirm_delete_factory:"¿Eliminar esta fábrica? Esta acción no se puede deshacer.",no_factories_yet:"Aún no has añadido ninguna fábrica.",factory_created_toast:"Fábrica creada y analizada",factory_updated_toast:"Fábrica actualizada",factory_deleted_toast:"Fábrica eliminada",ai_insights_feed_title:"Feed de Perspectivas IA",no_ai_insights_yet:"Aún no hay perspectivas IA. Añade una fábrica.",reanalyze_btn:"Reanalizar",view_insights_btn:"Ver Perspectivas",created_label:"Creada",cancel_btn:"Cancelar",save_btn:"Guardar Cambios"},
  zh: {tagline:"全球工业智能平台",live_label:"实时",kpi_energy:"能源使用量",kpi_efficiency:"效率",kpi_active:"运行中设备",kpi_alerts:"警报",kwh_unit:"kWh",chart_title:"实时性能",machine_status_title:"设备状态",status_running:"运行中",status_warning:"警告",status_critical:"严重",form_title:"工厂数据输入",factory_name_label:"工厂名称",machine_count_label:"设备数量",energy_cost_label:"能源成本 ($/kWh)",machine_type_label:"设备类型",temperature_label:"温度 (°C)",vibration_label:"振动 (mm/s)",load_label:"负载 (%)",submit_btn:"分析工厂",submitting:"更新中...",ai_panel_title:"AI 洞察",ai_placeholder:"提交工厂数据以生成AI分析。",ai_analyzing:"分析中...",ai_risks:"风险",ai_efficiency_insights:"效率分析",ai_optimizations:"优化建议",toast_updated:"工厂数据已更新",toast_analysis_done:"AI分析已完成",toast_error:"出现错误",nav_dashboard:"仪表盘",nav_factories:"工厂",nav_ai_insights:"AI洞察",logout_btn:"退出",login_title:"欢迎回来",login_subtitle:"登录您的 FactoryPulse AI 账户",ph_email:"电子邮件",ph_password:"密码",remember_me:"记住我",login_btn:"登录",login_link_register:"没有账户？创建一个",register_title:"创建账户",register_subtitle:"开始使用AI监控您的工厂",ph_full_name:"全名",ph_confirm_password:"确认密码",register_btn:"创建账户",register_link_login:"已有账户？登录",err_missing_fields:"请填写所有字段",err_invalid_email:"请输入有效的电子邮件地址",err_weak_password:"密码至少8位，需包含字母和数字",err_password_mismatch:"两次密码不一致",err_invalid_credentials:"电子邮件或密码错误",err_email_taken:"该电子邮件已被注册",err_generic:"出现错误，请重试",my_factories_title:"我的工厂",add_factory_btn:"+ 添加工厂",edit_factory_btn:"编辑",delete_factory_btn:"删除",confirm_delete_factory:"删除此工厂？此操作无法撤销。",no_factories_yet:"您还没有添加任何工厂。",factory_created_toast:"工厂已创建并分析",factory_updated_toast:"工厂已更新",factory_deleted_toast:"工厂已删除",ai_insights_feed_title:"AI洞察动态",no_ai_insights_yet:"暂无AI洞察。请添加工厂开始。",reanalyze_btn:"重新分析",view_insights_btn:"查看洞察",created_label:"创建于",cancel_btn:"取消",save_btn:"保存更改"},
  ar: {tagline:"منصة الذكاء الصناعي العالمية",live_label:"مباشر",kpi_energy:"استهلاك الطاقة",kpi_efficiency:"الكفاءة",kpi_active:"الآلات النشطة",kpi_alerts:"التنبيهات",kwh_unit:"kWh",chart_title:"الأداء في الوقت الفعلي",machine_status_title:"حالة الآلات",status_running:"تعمل",status_warning:"تحذير",status_critical:"حرج",form_title:"إدخال بيانات المصنع",factory_name_label:"اسم المصنع",machine_count_label:"عدد الآلات",energy_cost_label:"تكلفة الطاقة ($/kWh)",machine_type_label:"نوع الآلة",temperature_label:"درجة الحرارة (°C)",vibration_label:"الاهتزاز (مم/ث)",load_label:"الحمل (%)",submit_btn:"تحليل المصنع",submitting:"جارٍ التحديث...",ai_panel_title:"رؤى الذكاء الاصطناعي",ai_placeholder:"أرسل بيانات المصنع لإنشاء تحليل بالذكاء الاصطناعي.",ai_analyzing:"جارٍ التحليل...",ai_risks:"المخاطر",ai_efficiency_insights:"تحليل الكفاءة",ai_optimizations:"اقتراحات التحسين",toast_updated:"تم تحديث بيانات المصنع",toast_analysis_done:"اكتمل تحليل الذكاء الاصطناعي",toast_error:"حدث خطأ ما",nav_dashboard:"لوحة التحكم",nav_factories:"المصانع",nav_ai_insights:"رؤى الذكاء الاصطناعي",logout_btn:"تسجيل الخروج",login_title:"مرحباً بعودتك",login_subtitle:"سجل الدخول إلى حساب FactoryPulse AI الخاص بك",ph_email:"البريد الإلكتروني",ph_password:"كلمة المرور",remember_me:"تذكرني",login_btn:"تسجيل الدخول",login_link_register:"ليس لديك حساب؟ أنشئ واحداً",register_title:"إنشاء حسابك",register_subtitle:"ابدأ بمراقبة مصانعك بالذكاء الاصطناعي",ph_full_name:"الاسم الكامل",ph_confirm_password:"تأكيد كلمة المرور",register_btn:"إنشاء حساب",register_link_login:"لديك حساب بالفعل؟ سجل الدخول",err_missing_fields:"يرجى ملء جميع الحقول",err_invalid_email:"يرجى إدخال بريد إلكتروني صالح",err_weak_password:"يجب أن تكون كلمة المرور 8 أحرف على الأقل وتحتوي على حرف ورقم",err_password_mismatch:"كلمتا المرور غير متطابقتين",err_invalid_credentials:"البريد الإلكتروني أو كلمة المرور غير صحيحة",err_email_taken:"هذا البريد الإلكتروني مسجل بالفعل",err_generic:"حدث خطأ ما. يرجى المحاولة مرة أخرى",my_factories_title:"مصانعي",add_factory_btn:"+ إضافة مصنع",edit_factory_btn:"تعديل",delete_factory_btn:"حذف",confirm_delete_factory:"هل تريد حذف هذا المصنع؟ لا يمكن التراجع عن هذا.",no_factories_yet:"لم تقم بإضافة أي مصنع بعد.",factory_created_toast:"تم إنشاء المصنع وتحليله",factory_updated_toast:"تم تحديث المصنع",factory_deleted_toast:"تم حذف المصنع",ai_insights_feed_title:"موجز رؤى الذكاء الاصطناعي",no_ai_insights_yet:"لا توجد رؤى بعد. أضف مصنعاً للبدء.",reanalyze_btn:"إعادة التحليل",view_insights_btn:"عرض الرؤى",created_label:"تاريخ الإنشاء",cancel_btn:"إلغاء",save_btn:"حفظ التغييرات"},
  tr: {tagline:"Küresel Endüstriyel Zeka Platformu",live_label:"Canlı",kpi_energy:"Enerji Kullanımı",kpi_efficiency:"Verimlilik",kpi_active:"Aktif Makineler",kpi_alerts:"Uyarılar",kwh_unit:"kWh",chart_title:"Gerçek Zamanlı Performans",machine_status_title:"Makine Durumu",status_running:"Çalışıyor",status_warning:"Uyarı",status_critical:"Kritik",form_title:"Fabrika Veri Girişi",factory_name_label:"Fabrika Adı",machine_count_label:"Makine Sayısı",energy_cost_label:"Enerji Maliyeti ($/kWh)",machine_type_label:"Makine Türü",temperature_label:"Sıcaklık (°C)",vibration_label:"Titreşim (mm/s)",load_label:"Yük (%)",submit_btn:"Fabrikayı Analiz Et",submitting:"Güncelleniyor...",ai_panel_title:"AI Analizleri",ai_placeholder:"AI analizi oluşturmak için fabrika verilerini gönderin.",ai_analyzing:"Analiz ediliyor...",ai_risks:"Riskler",ai_efficiency_insights:"Verimlilik Analizi",ai_optimizations:"Optimizasyon Önerileri",toast_updated:"Fabrika verileri güncellendi",toast_analysis_done:"AI analizi tamamlandı",toast_error:"Bir şeyler ters gitti",nav_dashboard:"Panel",nav_factories:"Fabrikalar",nav_ai_insights:"AI Analizleri",logout_btn:"Çıkış Yap",login_title:"Tekrar hoş geldiniz",login_subtitle:"FactoryPulse AI hesabınıza giriş yapın",ph_email:"E-posta",ph_password:"Şifre",remember_me:"Beni hatırla",login_btn:"Giriş Yap",login_link_register:"Hesabınız yok mu? Oluşturun",register_title:"Hesabınızı oluşturun",register_subtitle:"Fabrikalarınızı AI ile izlemeye başlayın",ph_full_name:"Ad Soyad",ph_confirm_password:"Şifreyi Onayla",register_btn:"Hesap Oluştur",register_link_login:"Zaten hesabınız var mı? Giriş yapın",err_missing_fields:"Lütfen tüm alanları doldurun",err_invalid_email:"Lütfen geçerli bir e-posta adresi girin",err_weak_password:"Şifre en az 8 karakter, bir harf ve bir rakam içermeli",err_password_mismatch:"Şifreler eşleşmiyor",err_invalid_credentials:"E-posta veya şifre hatalı",err_email_taken:"Bu e-posta zaten kayıtlı",err_generic:"Bir şeyler ters gitti. Tekrar deneyin",my_factories_title:"Fabrikalarım",add_factory_btn:"+ Fabrika Ekle",edit_factory_btn:"Düzenle",delete_factory_btn:"Sil",confirm_delete_factory:"Bu fabrika silinsin mi? Bu işlem geri alınamaz.",no_factories_yet:"Henüz fabrika eklemediniz.",factory_created_toast:"Fabrika oluşturuldu ve analiz edildi",factory_updated_toast:"Fabrika güncellendi",factory_deleted_toast:"Fabrika silindi",ai_insights_feed_title:"AI Analiz Akışı",no_ai_insights_yet:"Henüz AI analizi yok. Başlamak için fabrika ekleyin.",reanalyze_btn:"Yeniden Analiz Et",view_insights_btn:"Analizleri Görüntüle",created_label:"Oluşturulma",cancel_btn:"İptal",save_btn:"Değişiklikleri Kaydet"},
  it: {tagline:"Piattaforma Globale di Intelligenza Industriale",live_label:"In diretta",kpi_energy:"Consumo Energetico",kpi_efficiency:"Efficienza",kpi_active:"Macchine Attive",kpi_alerts:"Avvisi",kwh_unit:"kWh",chart_title:"Prestazioni in Tempo Reale",machine_status_title:"Stato delle Macchine",status_running:"In funzione",status_warning:"Avviso",status_critical:"Critico",form_title:"Inserimento Dati Fabbrica",factory_name_label:"Nome Fabbrica",machine_count_label:"Numero di Macchine",energy_cost_label:"Costo Energia ($/kWh)",machine_type_label:"Tipo di Macchina",temperature_label:"Temperatura (°C)",vibration_label:"Vibrazione (mm/s)",load_label:"Carico (%)",submit_btn:"Analizza Fabbrica",submitting:"Aggiornamento...",ai_panel_title:"Analisi IA",ai_placeholder:"Invia i dati della fabbrica per generare un'analisi IA.",ai_analyzing:"Analisi in corso...",ai_risks:"Rischi",ai_efficiency_insights:"Analisi dell'Efficienza",ai_optimizations:"Suggerimenti di Ottimizzazione",toast_updated:"Dati fabbrica aggiornati",toast_analysis_done:"Analisi IA completata",toast_error:"Qualcosa è andato storto",nav_dashboard:"Dashboard",nav_factories:"Fabbriche",nav_ai_insights:"Analisi IA",logout_btn:"Esci",login_title:"Bentornato",login_subtitle:"Accedi al tuo account FactoryPulse AI",ph_email:"Email",ph_password:"Password",remember_me:"Ricordami",login_btn:"Accedi",login_link_register:"Non hai un account? Creane uno",register_title:"Crea il tuo account",register_subtitle:"Inizia a monitorare le tue fabbriche con l'IA",ph_full_name:"Nome Completo",ph_confirm_password:"Conferma Password",register_btn:"Crea Account",register_link_login:"Hai già un account? Accedi",err_missing_fields:"Si prega di compilare tutti i campi",err_invalid_email:"Inserisci un indirizzo email valido",err_weak_password:"La password deve avere almeno 8 caratteri, una lettera e un numero",err_password_mismatch:"Le password non corrispondono",err_invalid_credentials:"Email o password errati",err_email_taken:"Questa email è già registrata",err_generic:"Qualcosa è andato storto. Riprova",my_factories_title:"Le Mie Fabbriche",add_factory_btn:"+ Aggiungi Fabbrica",edit_factory_btn:"Modifica",delete_factory_btn:"Elimina",confirm_delete_factory:"Eliminare questa fabbrica? Questa azione non può essere annullata.",no_factories_yet:"Non hai ancora aggiunto nessuna fabbrica.",factory_created_toast:"Fabbrica creata e analizzata",factory_updated_toast:"Fabbrica aggiornata",factory_deleted_toast:"Fabbrica eliminata",ai_insights_feed_title:"Feed di Analisi IA",no_ai_insights_yet:"Nessuna analisi IA ancora. Aggiungi una fabbrica.",reanalyze_btn:"Rianalizza",view_insights_btn:"Vedi Analisi",created_label:"Creata il",cancel_btn:"Annulla",save_btn:"Salva Modifiche"},
  pt: {tagline:"Plataforma Global de Inteligência Industrial",live_label:"Ao vivo",kpi_energy:"Uso de Energia",kpi_efficiency:"Eficiência",kpi_active:"Máquinas Ativas",kpi_alerts:"Alertas",kwh_unit:"kWh",chart_title:"Desempenho em Tempo Real",machine_status_title:"Status das Máquinas",status_running:"Em funcionamento",status_warning:"Aviso",status_critical:"Crítico",form_title:"Entrada de Dados da Fábrica",factory_name_label:"Nome da Fábrica",machine_count_label:"Número de Máquinas",energy_cost_label:"Custo de Energia ($/kWh)",machine_type_label:"Tipo de Máquina",temperature_label:"Temperatura (°C)",vibration_label:"Vibração (mm/s)",load_label:"Carga (%)",submit_btn:"Analisar Fábrica",submitting:"Atualizando...",ai_panel_title:"Insights de IA",ai_placeholder:"Envie os dados da fábrica para gerar uma análise de IA.",ai_analyzing:"Analisando...",ai_risks:"Riscos",ai_efficiency_insights:"Análise de Eficiência",ai_optimizations:"Sugestões de Otimização",toast_updated:"Dados da fábrica atualizados",toast_analysis_done:"Análise de IA concluída",toast_error:"Algo deu errado",nav_dashboard:"Painel",nav_factories:"Fábricas",nav_ai_insights:"Insights de IA",logout_btn:"Sair",login_title:"Bem-vindo de volta",login_subtitle:"Entre na sua conta FactoryPulse AI",ph_email:"E-mail",ph_password:"Senha",remember_me:"Lembrar de mim",login_btn:"Entrar",login_link_register:"Não tem conta? Crie uma",register_title:"Crie sua conta",register_subtitle:"Comece a monitorar suas fábricas com IA",ph_full_name:"Nome Completo",ph_confirm_password:"Confirmar Senha",register_btn:"Criar Conta",register_link_login:"Já tem conta? Entrar",err_missing_fields:"Por favor preencha todos os campos",err_invalid_email:"Por favor insira um e-mail válido",err_weak_password:"A senha deve ter no mínimo 8 caracteres, uma letra e um número",err_password_mismatch:"As senhas não coincidem",err_invalid_credentials:"E-mail ou senha incorretos",err_email_taken:"Este e-mail já está registrado",err_generic:"Algo deu errado. Tente novamente",my_factories_title:"Minhas Fábricas",add_factory_btn:"+ Adicionar Fábrica",edit_factory_btn:"Editar",delete_factory_btn:"Excluir",confirm_delete_factory:"Excluir esta fábrica? Esta ação não pode ser desfeita.",no_factories_yet:"Você ainda não adicionou nenhuma fábrica.",factory_created_toast:"Fábrica criada e analisada",factory_updated_toast:"Fábrica atualizada",factory_deleted_toast:"Fábrica excluída",ai_insights_feed_title:"Feed de Insights de IA",no_ai_insights_yet:"Ainda sem insights de IA. Adicione uma fábrica.",reanalyze_btn:"Reanalisar",view_insights_btn:"Ver Insights",created_label:"Criada em",cancel_btn:"Cancelar",save_btn:"Salvar Alterações"},
  ja: {tagline:"グローバル産業インテリジェンスプラットフォーム",live_label:"ライブ",kpi_energy:"エネルギー使用量",kpi_efficiency:"効率",kpi_active:"稼働中の機械",kpi_alerts:"アラート",kwh_unit:"kWh",chart_title:"リアルタイムパフォーマンス",machine_status_title:"機械の状態",status_running:"稼働中",status_warning:"警告",status_critical:"重大",form_title:"工場データ入力",factory_name_label:"工場名",machine_count_label:"機械の数",energy_cost_label:"エネルギーコスト ($/kWh)",machine_type_label:"機械の種類",temperature_label:"温度 (°C)",vibration_label:"振動 (mm/s)",load_label:"負荷 (%)",submit_btn:"工場を分析",submitting:"更新中...",ai_panel_title:"AIインサイト",ai_placeholder:"工場データを送信してAI分析を生成してください。",ai_analyzing:"分析中...",ai_risks:"リスク",ai_efficiency_insights:"効率分析",ai_optimizations:"最適化提案",toast_updated:"工場データが更新されました",toast_analysis_done:"AI分析が完了しました",toast_error:"問題が発生しました",nav_dashboard:"ダッシュボード",nav_factories:"工場",nav_ai_insights:"AIインサイト",logout_btn:"ログアウト",login_title:"おかえりなさい",login_subtitle:"FactoryPulse AI アカウントにログイン",ph_email:"メールアドレス",ph_password:"パスワード",remember_me:"ログイン状態を保持",login_btn:"ログイン",login_link_register:"アカウントをお持ちでないですか？作成する",register_title:"アカウントを作成",register_subtitle:"AIで工場の監視を始めましょう",ph_full_name:"氏名",ph_confirm_password:"パスワードの確認",register_btn:"アカウント作成",register_link_login:"すでにアカウントをお持ちですか？ログイン",err_missing_fields:"すべての項目を入力してください",err_invalid_email:"有効なメールアドレスを入力してください",err_weak_password:"パスワードは8文字以上で、文字と数字を含める必要があります",err_password_mismatch:"パスワードが一致しません",err_invalid_credentials:"メールアドレスまたはパスワードが正しくありません",err_email_taken:"このメールアドレスは既に登録されています",err_generic:"エラーが発生しました。再試行してください",my_factories_title:"マイ工場",add_factory_btn:"+ 工場を追加",edit_factory_btn:"編集",delete_factory_btn:"削除",confirm_delete_factory:"この工場を削除しますか？元に戻せません。",no_factories_yet:"まだ工場を追加していません。",factory_created_toast:"工場が作成・分析されました",factory_updated_toast:"工場が更新されました",factory_deleted_toast:"工場が削除されました",ai_insights_feed_title:"AIインサイトフィード",no_ai_insights_yet:"AIインサイトはまだありません。工場を追加してください。",reanalyze_btn:"再分析",view_insights_btn:"インサイトを見る",created_label:"作成日",cancel_btn:"キャンセル",save_btn:"変更を保存"},
  ko: {tagline:"글로벌 산업 인텔리전스 플랫폼",live_label:"실시간",kpi_energy:"에너지 사용량",kpi_efficiency:"효율성",kpi_active:"가동 중인 기계",kpi_alerts:"경고",kwh_unit:"kWh",chart_title:"실시간 성능",machine_status_title:"기계 상태",status_running:"가동 중",status_warning:"경고",status_critical:"심각",form_title:"공장 데이터 입력",factory_name_label:"공장 이름",machine_count_label:"기계 수",energy_cost_label:"에너지 비용 ($/kWh)",machine_type_label:"기계 유형",temperature_label:"온도 (°C)",vibration_label:"진동 (mm/s)",load_label:"부하 (%)",submit_btn:"공장 분석",submitting:"업데이트 중...",ai_panel_title:"AI 인사이트",ai_placeholder:"AI 분석을 생성하려면 공장 데이터를 제출하세요.",ai_analyzing:"분석 중...",ai_risks:"위험 요소",ai_efficiency_insights:"효율성 분석",ai_optimizations:"최적화 제안",toast_updated:"공장 데이터가 업데이트되었습니다",toast_analysis_done:"AI 분석이 완료되었습니다",toast_error:"문제가 발생했습니다",nav_dashboard:"대시보드",nav_factories:"공장",nav_ai_insights:"AI 인사이트",logout_btn:"로그아웃",login_title:"다시 오신 것을 환영합니다",login_subtitle:"FactoryPulse AI 계정에 로그인하세요",ph_email:"이메일",ph_password:"비밀번호",remember_me:"로그인 상태 유지",login_btn:"로그인",login_link_register:"계정이 없으신가요? 계정 만들기",register_title:"계정 만들기",register_subtitle:"AI로 공장 모니터링을 시작하세요",ph_full_name:"성명",ph_confirm_password:"비밀번호 확인",register_btn:"계정 생성",register_link_login:"이미 계정이 있으신가요? 로그인",err_missing_fields:"모든 항목을 입력해주세요",err_invalid_email:"유효한 이메일 주소를 입력하세요",err_weak_password:"비밀번호는 8자 이상, 문자와 숫자를 포함해야 합니다",err_password_mismatch:"비밀번호가 일치하지 않습니다",err_invalid_credentials:"이메일 또는 비밀번호가 올바르지 않습니다",err_email_taken:"이미 등록된 이메일입니다",err_generic:"문제가 발생했습니다. 다시 시도해주세요",my_factories_title:"내 공장",add_factory_btn:"+ 공장 추가",edit_factory_btn:"수정",delete_factory_btn:"삭제",confirm_delete_factory:"이 공장을 삭제하시겠습니까? 되돌릴 수 없습니다.",no_factories_yet:"아직 추가된 공장이 없습니다.",factory_created_toast:"공장이 생성되고 분석되었습니다",factory_updated_toast:"공장이 업데이트되었습니다",factory_deleted_toast:"공장이 삭제되었습니다",ai_insights_feed_title:"AI 인사이트 피드",no_ai_insights_yet:"아직 AI 인사이트가 없습니다. 공장을 추가하세요.",reanalyze_btn:"다시 분석",view_insights_btn:"인사이트 보기",created_label:"생성일",cancel_btn:"취소",save_btn:"변경사항 저장"},
  hi: {tagline:"वैश्विक औद्योगिक बुद्धिमत्ता मंच",live_label:"लाइव",kpi_energy:"ऊर्जा उपयोग",kpi_efficiency:"दक्षता",kpi_active:"सक्रिय मशीनें",kpi_alerts:"अलर्ट",kwh_unit:"kWh",chart_title:"रीयल-टाइम प्रदर्शन",machine_status_title:"मशीन की स्थिति",status_running:"चल रहा है",status_warning:"चेतावनी",status_critical:"गंभीर",form_title:"फ़ैक्टरी डेटा इनपुट",factory_name_label:"फ़ैक्टरी का नाम",machine_count_label:"मशीनों की संख्या",energy_cost_label:"ऊर्जा लागत ($/kWh)",machine_type_label:"मशीन प्रकार",temperature_label:"तापमान (°C)",vibration_label:"कंपन (mm/s)",load_label:"लोड (%)",submit_btn:"फ़ैक्टरी का विश्लेषण करें",submitting:"अद्यतन हो रहा है...",ai_panel_title:"AI अंतर्दृष्टि",ai_placeholder:"AI विश्लेषण उत्पन्न करने के लिए फ़ैक्टरी डेटा सबमिट करें।",ai_analyzing:"विश्लेषण हो रहा है...",ai_risks:"जोखिम",ai_efficiency_insights:"दक्षता विश्लेषण",ai_optimizations:"अनुकूलन सुझाव",toast_updated:"फ़ैक्टरी डेटा अपडेट किया गया",toast_analysis_done:"AI विश्लेषण पूर्ण हुआ",toast_error:"कुछ गलत हो गया",nav_dashboard:"डैशबोर्ड",nav_factories:"फ़ैक्टरियाँ",nav_ai_insights:"AI अंतर्दृष्टि",logout_btn:"लॉग आउट",login_title:"वापसी पर स्वागत है",login_subtitle:"अपने FactoryPulse AI खाते में लॉग इन करें",ph_email:"ईमेल",ph_password:"पासवर्ड",remember_me:"मुझे याद रखें",login_btn:"लॉग इन करें",login_link_register:"खाता नहीं है? एक बनाएं",register_title:"अपना खाता बनाएं",register_subtitle:"AI के साथ अपनी फ़ैक्टरियों की निगरानी शुरू करें",ph_full_name:"पूरा नाम",ph_confirm_password:"पासवर्ड की पुष्टि करें",register_btn:"खाता बनाएं",register_link_login:"पहले से खाता है? लॉग इन करें",err_missing_fields:"कृपया सभी फ़ील्ड भरें",err_invalid_email:"कृपया एक मान्य ईमेल पता दर्ज करें",err_weak_password:"पासवर्ड कम से कम 8 अक्षर, एक अक्षर और एक अंक होना चाहिए",err_password_mismatch:"पासवर्ड मेल नहीं खाते",err_invalid_credentials:"गलत ईमेल या पासवर्ड",err_email_taken:"यह ईमेल पहले से पंजीकृत है",err_generic:"कुछ गलत हो गया। कृपया पुनः प्रयास करें",my_factories_title:"मेरी फ़ैक्टरियाँ",add_factory_btn:"+ फ़ैक्टरी जोड़ें",edit_factory_btn:"संपादित करें",delete_factory_btn:"हटाएं",confirm_delete_factory:"इस फ़ैक्टरी को हटाएं? इसे पूर्ववत नहीं किया जा सकता।",no_factories_yet:"आपने अभी तक कोई फ़ैक्टरी नहीं जोड़ी है।",factory_created_toast:"फ़ैक्टरी बनाई और विश्लेषित की गई",factory_updated_toast:"फ़ैक्टरी अपडेट की गई",factory_deleted_toast:"फ़ैक्टरी हटाई गई",ai_insights_feed_title:"AI अंतर्दृष्टि फ़ीड",no_ai_insights_yet:"अभी तक कोई AI अंतर्दृष्टि नहीं। शुरू करने के लिए एक फ़ैक्टरी जोड़ें।",reanalyze_btn:"पुनः विश्लेषण करें",view_insights_btn:"अंतर्दृष्टि देखें",created_label:"बनाया गया",cancel_btn:"रद्द करें",save_btn:"परिवर्तन सहेजें"},
  uz: {tagline:"Global sanoat intellekti platformasi",live_label:"Jonli",kpi_energy:"Energiya sarfi",kpi_efficiency:"Samaradorlik",kpi_active:"Faol stanoklar",kpi_alerts:"Ogohlantirishlar",kwh_unit:"kWh",chart_title:"Real vaqtdagi ko'rsatkichlar",machine_status_title:"Stanoklar holati",status_running:"Ishlamoqda",status_warning:"Ogohlantirish",status_critical:"Muhim",form_title:"Zavod ma'lumotlarini kiritish",factory_name_label:"Zavod nomi",machine_count_label:"Stanoklar soni",energy_cost_label:"Energiya narxi ($/kWh)",machine_type_label:"Stanok turi",temperature_label:"Harorat (°C)",vibration_label:"Tebranish (mm/s)",load_label:"Yuklama (%)",submit_btn:"Zavodni tahlil qilish",submitting:"Yangilanmoqda...",ai_panel_title:"AI tahlili",ai_placeholder:"AI tahlilini olish uchun zavod ma'lumotlarini yuboring.",ai_analyzing:"Tahlil qilinmoqda...",ai_risks:"Xavflar",ai_efficiency_insights:"Samaradorlik tahlili",ai_optimizations:"Optimallashtirish tavsiyalari",toast_updated:"Zavod ma'lumotlari yangilandi",toast_analysis_done:"AI tahlili yakunlandi",toast_error:"Xatolik yuz berdi",nav_dashboard:"Boshqaruv paneli",nav_factories:"Zavodlar",nav_ai_insights:"AI tahlili",logout_btn:"Chiqish",login_title:"Xush kelibsiz",login_subtitle:"FactoryPulse AI hisobingizga kiring",ph_email:"Elektron pochta",ph_password:"Parol",remember_me:"Meni eslab qol",login_btn:"Kirish",login_link_register:"Hisobingiz yo'qmi? Yarating",register_title:"Hisob yarating",register_subtitle:"Zavodlaringizni AI bilan kuzatishni boshlang",ph_full_name:"To'liq ism",ph_confirm_password:"Parolni tasdiqlang",register_btn:"Hisob yaratish",register_link_login:"Hisobingiz bormi? Kiring",err_missing_fields:"Barcha maydonlarni to'ldiring",err_invalid_email:"Yaroqli elektron pochta manzilini kiriting",err_weak_password:"Parol kamida 8 belgidan, harf va raqamdan iborat bo'lishi kerak",err_password_mismatch:"Parollar mos kelmaydi",err_invalid_credentials:"Elektron pochta yoki parol noto'g'ri",err_email_taken:"Bu elektron pochta allaqachon ro'yxatdan o'tgan",err_generic:"Xatolik yuz berdi. Qaytadan urinib ko'ring",my_factories_title:"Mening Zavodlarim",add_factory_btn:"+ Zavod qo'shish",edit_factory_btn:"Tahrirlash",delete_factory_btn:"O'chirish",confirm_delete_factory:"Bu zavodni o'chirasizmi? Buni bekor qilib bo'lmaydi.",no_factories_yet:"Siz hali hech qanday zavod qo'shmagansiz.",factory_created_toast:"Zavod yaratildi va tahlil qilindi",factory_updated_toast:"Zavod yangilandi",factory_deleted_toast:"Zavod o'chirildi",ai_insights_feed_title:"AI Tahlili Lentasi",no_ai_insights_yet:"Hali AI tahlili yo'q. Boshlash uchun zavod qo'shing.",reanalyze_btn:"Qayta tahlil qilish",view_insights_btn:"Tahlilni ko'rish",created_label:"Yaratilgan",cancel_btn:"Bekor qilish",save_btn:"O'zgarishlarni saqlash"},
  ky: {tagline:"Глобалдык өнөр жай интеллект платформасы",live_label:"Түз эфир",kpi_energy:"Энергия сарпталышы",kpi_efficiency:"Эффективдүүлүк",kpi_active:"Активдүү станоктор",kpi_alerts:"Дабылдар",kwh_unit:"кВт·саат",chart_title:"Реалдуу убакыттагы көрсөткүчтөр",machine_status_title:"Станоктордун абалы",status_running:"Иштеп жатат",status_warning:"Эскертүү",status_critical:"Олуттуу",form_title:"Завод маалыматтарын киргизүү",factory_name_label:"Заводдун аты",machine_count_label:"Станоктордун саны",energy_cost_label:"Энергия наркы ($/кВт·саат)",machine_type_label:"Станоктун түрү",temperature_label:"Температура (°C)",vibration_label:"Дирилдөө (мм/с)",load_label:"Жүктөм (%)",submit_btn:"Заводду талдоо",submitting:"Жаңыртылууда...",ai_panel_title:"AI-талдоо",ai_placeholder:"AI-талдоо алуу үчүн завод маалыматтарын жөнөтүңүз.",ai_analyzing:"Талдануда...",ai_risks:"Тобокелдиктер",ai_efficiency_insights:"Эффективдүүлүк талдоосу",ai_optimizations:"Оптималдаштыруу сунуштары",toast_updated:"Завод маалыматтары жаңыртылды",toast_analysis_done:"AI-талдоо аяктады",toast_error:"Ката кетти",nav_dashboard:"Башкаруу панели",nav_factories:"Заводдор",nav_ai_insights:"AI-талдоо",logout_btn:"Чыгуу",login_title:"Кайра кош келиңиз",login_subtitle:"FactoryPulse AI каттоо эсебиңизге кириңиз",ph_email:"Электрондук почта",ph_password:"Сырсөз",remember_me:"Мени эстеп кал",login_btn:"Кирүү",login_link_register:"Каттоо эсебиңиз жокпу? Түзүү",register_title:"Каттоо эсебин түзүү",register_subtitle:"Заводдоруңузду AI менен байкоону баштаңыз",ph_full_name:"Толук аты-жөнү",ph_confirm_password:"Сырсөздү ырастаңыз",register_btn:"Каттоо эсебин түзүү",register_link_login:"Каттоо эсебиңиз барбы? Кирүү",err_missing_fields:"Бардык талааларды толтуруңуз",err_invalid_email:"Жарактуу электрондук почта дарегин киргизиңиз",err_weak_password:"Сырсөз кеминде 8 белги, тамга жана сан камтышы керек",err_password_mismatch:"Сырсөздөр дал келбейт",err_invalid_credentials:"Электрондук почта же сырсөз туура эмес",err_email_taken:"Бул электрондук почта мурунтан катталган",err_generic:"Ката кетти. Кайра аракет кылыңыз",my_factories_title:"Менин Заводдорум",add_factory_btn:"+ Завод кошуу",edit_factory_btn:"Түзөтүү",delete_factory_btn:"Өчүрүү",confirm_delete_factory:"Бул заводду өчүрөсүзбү? Бул аракетти артка кайтарууга болбойт.",no_factories_yet:"Сиз азырынча эч кандай завод кошкон жоксуз.",factory_created_toast:"Завод түзүлдү жана талданды",factory_updated_toast:"Завод жаңыртылды",factory_deleted_toast:"Завод өчүрүлдү",ai_insights_feed_title:"AI-талдоо тизмеси",no_ai_insights_yet:"Азырынча AI-талдоо жок. Баштоо үчүн завод кошуңуз.",reanalyze_btn:"Кайра талдоо",view_insights_btn:"Талдоону көрүү",created_label:"Түзүлгөн күнү",cancel_btn:"Жокко чыгаруу",save_btn:"Өзгөртүүлөрдү сактоо"},
  uk: {tagline:"Глобальна платформа промислового інтелекту",live_label:"Наживо",kpi_energy:"Споживання енергії",kpi_efficiency:"Ефективність",kpi_active:"Активні верстати",kpi_alerts:"Сповіщення",kwh_unit:"кВт·год",chart_title:"Показники в реальному часі",machine_status_title:"Статус верстатів",status_running:"Працює",status_warning:"Попередження",status_critical:"Критично",form_title:"Введення даних заводу",factory_name_label:"Назва заводу",machine_count_label:"Кількість верстатів",energy_cost_label:"Вартість енергії ($/кВт·год)",machine_type_label:"Тип верстата",temperature_label:"Температура (°C)",vibration_label:"Вібрація (мм/с)",load_label:"Навантаження (%)",submit_btn:"Аналізувати завод",submitting:"Оновлення...",ai_panel_title:"AI-аналітика",ai_placeholder:"Надішліть дані заводу, щоб отримати AI-аналіз.",ai_analyzing:"Аналіз...",ai_risks:"Ризики",ai_efficiency_insights:"Аналіз ефективності",ai_optimizations:"Рекомендації з оптимізації",toast_updated:"Дані заводу оновлено",toast_analysis_done:"AI-аналіз завершено",toast_error:"Сталася помилка",nav_dashboard:"Панель",nav_factories:"Заводи",nav_ai_insights:"AI-аналітика",logout_btn:"Вийти",login_title:"З поверненням",login_subtitle:"Увійдіть у свій обліковий запис FactoryPulse AI",ph_email:"Електронна пошта",ph_password:"Пароль",remember_me:"Запам'ятати мене",login_btn:"Увійти",login_link_register:"Немає акаунту? Створити",register_title:"Створіть акаунт",register_subtitle:"Почніть моніторинг заводів за допомогою AI",ph_full_name:"Повне ім'я",ph_confirm_password:"Підтвердіть пароль",register_btn:"Створити акаунт",register_link_login:"Вже є акаунт? Увійти",err_missing_fields:"Будь ласка, заповніть усі поля",err_invalid_email:"Введіть дійсну електронну адресу",err_weak_password:"Пароль має містити щонайменше 8 символів, літеру та цифру",err_password_mismatch:"Паролі не збігаються",err_invalid_credentials:"Невірна електронна пошта або пароль",err_email_taken:"Ця електронна пошта вже зареєстрована",err_generic:"Сталася помилка. Спробуйте ще раз",my_factories_title:"Мої Заводи",add_factory_btn:"+ Додати завод",edit_factory_btn:"Редагувати",delete_factory_btn:"Видалити",confirm_delete_factory:"Видалити цей завод? Цю дію не можна скасувати.",no_factories_yet:"Ви ще не додали жодного заводу.",factory_created_toast:"Завод створено та проаналізовано",factory_updated_toast:"Завод оновлено",factory_deleted_toast:"Завод видалено",ai_insights_feed_title:"Стрічка AI-аналітики",no_ai_insights_yet:"Ще немає AI-аналітики. Додайте завод.",reanalyze_btn:"Проаналізувати знову",view_insights_btn:"Переглянути аналітику",created_label:"Створено",cancel_btn:"Скасувати",save_btn:"Зберегти зміни"},
  pl: {tagline:"Globalna Platforma Inteligencji Przemysłowej",live_label:"Na żywo",kpi_energy:"Zużycie Energii",kpi_efficiency:"Wydajność",kpi_active:"Aktywne Maszyny",kpi_alerts:"Alerty",kwh_unit:"kWh",chart_title:"Wydajność w Czasie Rzeczywistym",machine_status_title:"Status Maszyn",status_running:"Działa",status_warning:"Ostrzeżenie",status_critical:"Krytyczne",form_title:"Wprowadzanie Danych Fabryki",factory_name_label:"Nazwa Fabryki",machine_count_label:"Liczba Maszyn",energy_cost_label:"Koszt Energii ($/kWh)",machine_type_label:"Typ Maszyny",temperature_label:"Temperatura (°C)",vibration_label:"Wibracje (mm/s)",load_label:"Obciążenie (%)",submit_btn:"Analizuj Fabrykę",submitting:"Aktualizowanie...",ai_panel_title:"Analizy AI",ai_placeholder:"Prześlij dane fabryki, aby wygenerować analizę AI.",ai_analyzing:"Analizowanie...",ai_risks:"Ryzyka",ai_efficiency_insights:"Analiza Wydajności",ai_optimizations:"Sugestie Optymalizacji",toast_updated:"Dane fabryki zaktualizowane",toast_analysis_done:"Analiza AI zakończona",toast_error:"Coś poszło nie tak",nav_dashboard:"Panel",nav_factories:"Fabryki",nav_ai_insights:"Analizy AI",logout_btn:"Wyloguj",login_title:"Witamy z powrotem",login_subtitle:"Zaloguj się do swojego konta FactoryPulse AI",ph_email:"E-mail",ph_password:"Hasło",remember_me:"Zapamiętaj mnie",login_btn:"Zaloguj się",login_link_register:"Nie masz konta? Utwórz je",register_title:"Utwórz konto",register_subtitle:"Zacznij monitorować swoje fabryki z AI",ph_full_name:"Imię i Nazwisko",ph_confirm_password:"Potwierdź Hasło",register_btn:"Utwórz Konto",register_link_login:"Masz już konto? Zaloguj się",err_missing_fields:"Proszę wypełnić wszystkie pola",err_invalid_email:"Proszę podać prawidłowy adres e-mail",err_weak_password:"Hasło musi mieć min. 8 znaków, literę i cyfrę",err_password_mismatch:"Hasła nie pasują do siebie",err_invalid_credentials:"Nieprawidłowy e-mail lub hasło",err_email_taken:"Ten e-mail jest już zarejestrowany",err_generic:"Coś poszło nie tak. Spróbuj ponownie",my_factories_title:"Moje Fabryki",add_factory_btn:"+ Dodaj Fabrykę",edit_factory_btn:"Edytuj",delete_factory_btn:"Usuń",confirm_delete_factory:"Usunąć tę fabrykę? Tej czynności nie można cofnąć.",no_factories_yet:"Nie dodałeś jeszcze żadnej fabryki.",factory_created_toast:"Fabryka utworzona i przeanalizowana",factory_updated_toast:"Fabryka zaktualizowana",factory_deleted_toast:"Fabryka usunięta",ai_insights_feed_title:"Kanał Analiz AI",no_ai_insights_yet:"Brak analiz AI. Dodaj fabrykę, aby zacząć.",reanalyze_btn:"Analizuj Ponownie",view_insights_btn:"Zobacz Analizy",created_label:"Utworzono",cancel_btn:"Anuluj",save_btn:"Zapisz Zmiany"},
  nl: {tagline:"Wereldwijd Industrieel Intelligentieplatform",live_label:"Live",kpi_energy:"Energieverbruik",kpi_efficiency:"Efficiëntie",kpi_active:"Actieve Machines",kpi_alerts:"Meldingen",kwh_unit:"kWh",chart_title:"Realtime Prestaties",machine_status_title:"Machinestatus",status_running:"Actief",status_warning:"Waarschuwing",status_critical:"Kritiek",form_title:"Fabrieksgegevens Invoeren",factory_name_label:"Fabrieksnaam",machine_count_label:"Aantal Machines",energy_cost_label:"Energiekosten ($/kWh)",machine_type_label:"Machinetype",temperature_label:"Temperatuur (°C)",vibration_label:"Trilling (mm/s)",load_label:"Belasting (%)",submit_btn:"Fabriek Analyseren",submitting:"Bijwerken...",ai_panel_title:"AI-inzichten",ai_placeholder:"Verzend fabrieksgegevens om een AI-analyse te genereren.",ai_analyzing:"Analyseren...",ai_risks:"Risico's",ai_efficiency_insights:"Efficiëntieanalyse",ai_optimizations:"Optimalisatiesuggesties",toast_updated:"Fabrieksgegevens bijgewerkt",toast_analysis_done:"AI-analyse voltooid",toast_error:"Er is iets misgegaan",nav_dashboard:"Dashboard",nav_factories:"Fabrieken",nav_ai_insights:"AI-inzichten",logout_btn:"Uitloggen",login_title:"Welkom terug",login_subtitle:"Log in op uw FactoryPulse AI-account",ph_email:"E-mail",ph_password:"Wachtwoord",remember_me:"Onthoud mij",login_btn:"Inloggen",login_link_register:"Geen account? Maak er een",register_title:"Maak uw account aan",register_subtitle:"Begin met AI-monitoring van uw fabrieken",ph_full_name:"Volledige Naam",ph_confirm_password:"Bevestig Wachtwoord",register_btn:"Account Aanmaken",register_link_login:"Heeft u al een account? Inloggen",err_missing_fields:"Vul alle velden in",err_invalid_email:"Voer een geldig e-mailadres in",err_weak_password:"Wachtwoord moet minimaal 8 tekens, een letter en een cijfer bevatten",err_password_mismatch:"Wachtwoorden komen niet overeen",err_invalid_credentials:"Ongeldige e-mail of wachtwoord",err_email_taken:"Dit e-mailadres is al geregistreerd",err_generic:"Er is iets misgegaan. Probeer het opnieuw",my_factories_title:"Mijn Fabrieken",add_factory_btn:"+ Fabriek Toevoegen",edit_factory_btn:"Bewerken",delete_factory_btn:"Verwijderen",confirm_delete_factory:"Deze fabriek verwijderen? Dit kan niet ongedaan worden gemaakt.",no_factories_yet:"U heeft nog geen fabrieken toegevoegd.",factory_created_toast:"Fabriek aangemaakt en geanalyseerd",factory_updated_toast:"Fabriek bijgewerkt",factory_deleted_toast:"Fabriek verwijderd",ai_insights_feed_title:"AI-inzichten Feed",no_ai_insights_yet:"Nog geen AI-inzichten. Voeg een fabriek toe.",reanalyze_btn:"Opnieuw Analyseren",view_insights_btn:"Bekijk Inzichten",created_label:"Aangemaakt",cancel_btn:"Annuleren",save_btn:"Wijzigingen Opslaan"},
  sv: {tagline:"Global Industriell Intelligensplattform",live_label:"Live",kpi_energy:"Energiförbrukning",kpi_efficiency:"Effektivitet",kpi_active:"Aktiva Maskiner",kpi_alerts:"Varningar",kwh_unit:"kWh",chart_title:"Realtidsprestanda",machine_status_title:"Maskinstatus",status_running:"Igång",status_warning:"Varning",status_critical:"Kritisk",form_title:"Fabriksdatainmatning",factory_name_label:"Fabriksnamn",machine_count_label:"Antal Maskiner",energy_cost_label:"Energikostnad ($/kWh)",machine_type_label:"Maskintyp",temperature_label:"Temperatur (°C)",vibration_label:"Vibration (mm/s)",load_label:"Belastning (%)",submit_btn:"Analysera Fabrik",submitting:"Uppdaterar...",ai_panel_title:"AI-insikter",ai_placeholder:"Skicka fabriksdata för att generera en AI-analys.",ai_analyzing:"Analyserar...",ai_risks:"Risker",ai_efficiency_insights:"Effektivitetsanalys",ai_optimizations:"Optimeringsförslag",toast_updated:"Fabriksdata uppdaterad",toast_analysis_done:"AI-analys klar",toast_error:"Något gick fel",nav_dashboard:"Instrumentpanel",nav_factories:"Fabriker",nav_ai_insights:"AI-insikter",logout_btn:"Logga ut",login_title:"Välkommen tillbaka",login_subtitle:"Logga in på ditt FactoryPulse AI-konto",ph_email:"E-post",ph_password:"Lösenord",remember_me:"Kom ihåg mig",login_btn:"Logga in",login_link_register:"Inget konto? Skapa ett",register_title:"Skapa ditt konto",register_subtitle:"Börja övervaka dina fabriker med AI",ph_full_name:"Fullständigt Namn",ph_confirm_password:"Bekräfta Lösenord",register_btn:"Skapa Konto",register_link_login:"Har du redan ett konto? Logga in",err_missing_fields:"Vänligen fyll i alla fält",err_invalid_email:"Ange en giltig e-postadress",err_weak_password:"Lösenordet måste vara minst 8 tecken med en bokstav och en siffra",err_password_mismatch:"Lösenorden matchar inte",err_invalid_credentials:"Felaktig e-post eller lösenord",err_email_taken:"Denna e-post är redan registrerad",err_generic:"Något gick fel. Försök igen",my_factories_title:"Mina Fabriker",add_factory_btn:"+ Lägg till Fabrik",edit_factory_btn:"Redigera",delete_factory_btn:"Ta bort",confirm_delete_factory:"Ta bort denna fabrik? Detta kan inte ångras.",no_factories_yet:"Du har inte lagt till några fabriker än.",factory_created_toast:"Fabrik skapad och analyserad",factory_updated_toast:"Fabrik uppdaterad",factory_deleted_toast:"Fabrik borttagen",ai_insights_feed_title:"AI-insikter Flöde",no_ai_insights_yet:"Inga AI-insikter än. Lägg till en fabrik.",reanalyze_btn:"Analysera Igen",view_insights_btn:"Visa Insikter",created_label:"Skapad",cancel_btn:"Avbryt",save_btn:"Spara Ändringar"},
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
<title>FactoryPulse AI - Register</title>
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
  en: {tagline:"Global Industrial Intelligence Platform",live_label:"Live",kpi_energy:"Energy Usage",kpi_efficiency:"Efficiency",kpi_active:"Active Machines",kpi_alerts:"Alerts",kwh_unit:"kWh",chart_title:"Real-Time Performance",machine_status_title:"Machine Status",status_running:"Running",status_warning:"Warning",status_critical:"Critical",form_title:"Factory Data Input",factory_name_label:"Factory Name",machine_count_label:"Number of Machines",energy_cost_label:"Energy Cost ($/kWh)",machine_type_label:"Machine Type",temperature_label:"Temperature (°C)",vibration_label:"Vibration (mm/s)",load_label:"Load (%)",submit_btn:"Analyze Factory",submitting:"Updating...",ai_panel_title:"AI Insights",ai_placeholder:"Submit factory data to generate an AI analysis.",ai_analyzing:"Analyzing...",ai_risks:"Risks",ai_efficiency_insights:"Efficiency Insights",ai_optimizations:"Optimization Suggestions",toast_updated:"Factory data updated",toast_analysis_done:"AI analysis complete",toast_error:"Something went wrong",nav_dashboard:"Dashboard",nav_factories:"Factories",nav_ai_insights:"AI Insights",logout_btn:"Log Out",login_title:"Welcome back",login_subtitle:"Sign in to your FactoryPulse AI account",ph_email:"Email",ph_password:"Password",remember_me:"Remember me",login_btn:"Log In",login_link_register:"Don't have an account? Create one",register_title:"Create your account",register_subtitle:"Start monitoring your factories with AI",ph_full_name:"Full Name",ph_confirm_password:"Confirm Password",register_btn:"Create Account",register_link_login:"Already have an account? Sign in",err_missing_fields:"Please fill in all fields",err_invalid_email:"Please enter a valid email address",err_weak_password:"Password must be at least 8 characters with a letter and a number",err_password_mismatch:"Passwords do not match",err_invalid_credentials:"Invalid email or password",err_email_taken:"This email is already registered",err_generic:"Something went wrong. Please try again",my_factories_title:"My Factories",add_factory_btn:"+ Add Factory",edit_factory_btn:"Edit",delete_factory_btn:"Delete",confirm_delete_factory:"Delete this factory? This cannot be undone.",no_factories_yet:"You haven't added any factories yet.",factory_created_toast:"Factory created and analyzed",factory_updated_toast:"Factory updated",factory_deleted_toast:"Factory deleted",ai_insights_feed_title:"AI Insights Feed",no_ai_insights_yet:"No AI insights yet. Add a factory to get started.",reanalyze_btn:"Re-analyze",view_insights_btn:"View Insights",created_label:"Created",cancel_btn:"Cancel",save_btn:"Save Changes"},
  ru: {tagline:"Глобальная платформа промышленного интеллекта",live_label:"Live",kpi_energy:"Потребление энергии",kpi_efficiency:"Эффективность",kpi_active:"Активные станки",kpi_alerts:"Оповещения",kwh_unit:"кВт·ч",chart_title:"Показатели в реальном времени",machine_status_title:"Статус станков",status_running:"Работает",status_warning:"Внимание",status_critical:"Критично",form_title:"Ввод данных завода",factory_name_label:"Название завода",machine_count_label:"Количество станков",energy_cost_label:"Стоимость энергии ($/кВт·ч)",machine_type_label:"Тип станка",temperature_label:"Температура (°C)",vibration_label:"Вибрация (мм/с)",load_label:"Нагрузка (%)",submit_btn:"Анализировать завод",submitting:"Обновление...",ai_panel_title:"AI-аналитика",ai_placeholder:"Отправьте данные завода, чтобы получить AI-анализ.",ai_analyzing:"Анализ...",ai_risks:"Риски",ai_efficiency_insights:"Анализ эффективности",ai_optimizations:"Рекомендации по оптимизации",toast_updated:"Данные завода обновлены",toast_analysis_done:"AI-анализ завершён",toast_error:"Произошла ошибка",nav_dashboard:"Панель",nav_factories:"Заводы",nav_ai_insights:"AI-аналитика",logout_btn:"Выход",login_title:"С возвращением",login_subtitle:"Войдите в аккаунт FactoryPulse AI",ph_email:"Email",ph_password:"Пароль",remember_me:"Запомнить меня",login_btn:"Войти",login_link_register:"Нет аккаунта? Создать",register_title:"Создать аккаунт",register_subtitle:"Начните мониторинг заводов с помощью AI",ph_full_name:"Полное имя",ph_confirm_password:"Подтвердите пароль",register_btn:"Создать аккаунт",register_link_login:"Уже есть аккаунт? Войти",err_missing_fields:"Заполните все поля",err_invalid_email:"Введите корректный email",err_weak_password:"Пароль должен быть от 8 символов, с буквой и цифрой",err_password_mismatch:"Пароли не совпадают",err_invalid_credentials:"Неверный email или пароль",err_email_taken:"Этот email уже зарегистрирован",err_generic:"Что-то пошло не так. Попробуйте снова",my_factories_title:"Мои заводы",add_factory_btn:"+ Добавить завод",edit_factory_btn:"Изменить",delete_factory_btn:"Удалить",confirm_delete_factory:"Удалить этот завод? Это действие нельзя отменить.",no_factories_yet:"Вы ещё не добавили ни одного завода.",factory_created_toast:"Завод создан и проанализирован",factory_updated_toast:"Завод обновлён",factory_deleted_toast:"Завод удалён",ai_insights_feed_title:"Лента AI-аналитики",no_ai_insights_yet:"Пока нет AI-аналитики. Добавьте завод, чтобы начать.",reanalyze_btn:"Проанализировать снова",view_insights_btn:"Смотреть аналитику",created_label:"Создано",cancel_btn:"Отмена",save_btn:"Сохранить изменения"},
  kk: {tagline:"Жаһандық өнеркәсіптік интеллект платформасы",live_label:"Тікелей эфир",kpi_energy:"Энергия тұтыну",kpi_efficiency:"Тиімділік",kpi_active:"Белсенді станоктар",kpi_alerts:"Дабылдар",kwh_unit:"кВт·сағ",chart_title:"Нақты уақыттағы көрсеткіштер",machine_status_title:"Станоктар күйі",status_running:"Жұмыс істеп тұр",status_warning:"Ескерту",status_critical:"Сыни",form_title:"Зауыт деректерін енгізу",factory_name_label:"Зауыт атауы",machine_count_label:"Станоктар саны",energy_cost_label:"Энергия құны ($/кВт·сағ)",machine_type_label:"Станок түрі",temperature_label:"Температура (°C)",vibration_label:"Діріл (мм/с)",load_label:"Жүктеме (%)",submit_btn:"Зауытты талдау",submitting:"Жаңартылуда...",ai_panel_title:"AI-талдау",ai_placeholder:"AI-талдау алу үшін зауыт деректерін жіберіңіз.",ai_analyzing:"Талдануда...",ai_risks:"Тәуекелдер",ai_efficiency_insights:"Тиімділік талдауы",ai_optimizations:"Оңтайландыру ұсыныстары",toast_updated:"Зауыт деректері жаңартылды",toast_analysis_done:"AI-талдау аяқталды",toast_error:"Қате орын алды",nav_dashboard:"Басқару тақтасы",nav_factories:"Зауыттар",nav_ai_insights:"AI-талдау",logout_btn:"Шығу",login_title:"Қайта қош келдіңіз",login_subtitle:"FactoryPulse AI аккаунтыңызға кіріңіз",ph_email:"Email",ph_password:"Құпия сөз",remember_me:"Мені есте сақтау",login_btn:"Кіру",login_link_register:"Аккаунтыңыз жоқ па? Тіркелу",register_title:"Аккаунт құру",register_subtitle:"Зауыттарды AI арқылы бақылауды бастаңыз",ph_full_name:"Толық аты-жөні",ph_confirm_password:"Құпия сөзді қайталаңыз",register_btn:"Аккаунт құру",register_link_login:"Аккаунтыңыз бар ма? Кіру",err_missing_fields:"Барлық өрістерді толтырыңыз",err_invalid_email:"Дұрыс email мекенжайын енгізіңіз",err_weak_password:"Құпия сөз кемінде 8 таңба, әріп пен сан болуы керек",err_password_mismatch:"Құпия сөздер сәйкес келмейді",err_invalid_credentials:"Қате email немесе құпия сөз",err_email_taken:"Бұл email тіркелген",err_generic:"Қате орын алды. Қайталап көріңіз",my_factories_title:"Менің зауыттарым",add_factory_btn:"+ Зауыт қосу",edit_factory_btn:"Өзгерту",delete_factory_btn:"Жою",confirm_delete_factory:"Бұл зауытты жоясыз ба? Бұл әрекетті кері қайтару мүмкін емес.",no_factories_yet:"Сіз әлі ешбір зауыт қосқан жоқсыз.",factory_created_toast:"Зауыт құрылды және талданды",factory_updated_toast:"Зауыт жаңартылды",factory_deleted_toast:"Зауыт жойылды",ai_insights_feed_title:"AI-талдау таспасы",no_ai_insights_yet:"AI-талдау әлі жоқ. Бастау үшін зауыт қосыңыз.",reanalyze_btn:"Қайта талдау",view_insights_btn:"Талдауды көру",created_label:"Құрылған күні",cancel_btn:"Бас тарту",save_btn:"Өзгерістерді сақтау"},
  de: {tagline:"Globale Industrielle Intelligenzplattform",live_label:"Live",kpi_energy:"Energieverbrauch",kpi_efficiency:"Effizienz",kpi_active:"Aktive Maschinen",kpi_alerts:"Warnungen",kwh_unit:"kWh",chart_title:"Echtzeit-Leistung",machine_status_title:"Maschinenstatus",status_running:"Läuft",status_warning:"Warnung",status_critical:"Kritisch",form_title:"Fabrikdateneingabe",factory_name_label:"Fabrikname",machine_count_label:"Anzahl der Maschinen",energy_cost_label:"Energiekosten ($/kWh)",machine_type_label:"Maschinentyp",temperature_label:"Temperatur (°C)",vibration_label:"Vibration (mm/s)",load_label:"Last (%)",submit_btn:"Fabrik Analysieren",submitting:"Aktualisieren...",ai_panel_title:"KI-Einblicke",ai_placeholder:"Senden Sie Fabrikdaten, um eine KI-Analyse zu erstellen.",ai_analyzing:"Analysiere...",ai_risks:"Risiken",ai_efficiency_insights:"Effizienzanalyse",ai_optimizations:"Optimierungsvorschläge",toast_updated:"Fabrikdaten aktualisiert",toast_analysis_done:"KI-Analyse abgeschlossen",toast_error:"Etwas ist schiefgelaufen",nav_dashboard:"Übersicht",nav_factories:"Fabriken",nav_ai_insights:"KI-Einblicke",logout_btn:"Abmelden",login_title:"Willkommen zurück",login_subtitle:"Melden Sie sich bei Ihrem FactoryPulse AI-Konto an",ph_email:"E-Mail",ph_password:"Passwort",remember_me:"Angemeldet bleiben",login_btn:"Einloggen",login_link_register:"Kein Konto? Jetzt erstellen",register_title:"Konto erstellen",register_subtitle:"Beginnen Sie mit der KI-Überwachung Ihrer Fabriken",ph_full_name:"Vollständiger Name",ph_confirm_password:"Passwort bestätigen",register_btn:"Konto erstellen",register_link_login:"Bereits ein Konto? Anmelden",err_missing_fields:"Bitte füllen Sie alle Felder aus",err_invalid_email:"Bitte geben Sie eine gültige E-Mail-Adresse ein",err_weak_password:"Passwort muss mind. 8 Zeichen, einen Buchstaben und eine Zahl enthalten",err_password_mismatch:"Passwörter stimmen nicht überein",err_invalid_credentials:"Ungültige E-Mail oder Passwort",err_email_taken:"Diese E-Mail ist bereits registriert",err_generic:"Etwas ist schiefgelaufen. Bitte erneut versuchen",my_factories_title:"Meine Fabriken",add_factory_btn:"+ Fabrik Hinzufügen",edit_factory_btn:"Bearbeiten",delete_factory_btn:"Löschen",confirm_delete_factory:"Diese Fabrik löschen? Dies kann nicht rückgängig gemacht werden.",no_factories_yet:"Sie haben noch keine Fabriken hinzugefügt.",factory_created_toast:"Fabrik erstellt und analysiert",factory_updated_toast:"Fabrik aktualisiert",factory_deleted_toast:"Fabrik gelöscht",ai_insights_feed_title:"KI-Einblicke Feed",no_ai_insights_yet:"Noch keine KI-Einblicke. Fügen Sie eine Fabrik hinzu.",reanalyze_btn:"Erneut analysieren",view_insights_btn:"Einblicke Anzeigen",created_label:"Erstellt",cancel_btn:"Abbrechen",save_btn:"Änderungen Speichern"},
  fr: {tagline:"Plateforme mondiale d'intelligence industrielle",live_label:"En direct",kpi_energy:"Consommation d'Énergie",kpi_efficiency:"Efficacité",kpi_active:"Machines Actives",kpi_alerts:"Alertes",kwh_unit:"kWh",chart_title:"Performance en Temps Réel",machine_status_title:"État des Machines",status_running:"En marche",status_warning:"Avertissement",status_critical:"Critique",form_title:"Saisie des Données d'Usine",factory_name_label:"Nom de l'Usine",machine_count_label:"Nombre de Machines",energy_cost_label:"Coût de l'Énergie ($/kWh)",machine_type_label:"Type de Machine",temperature_label:"Température (°C)",vibration_label:"Vibration (mm/s)",load_label:"Charge (%)",submit_btn:"Analyser l'Usine",submitting:"Mise à jour...",ai_panel_title:"Analyses IA",ai_placeholder:"Envoyez les données de l'usine pour générer une analyse IA.",ai_analyzing:"Analyse en cours...",ai_risks:"Risques",ai_efficiency_insights:"Analyse d'Efficacité",ai_optimizations:"Suggestions d'Optimisation",toast_updated:"Données d'usine mises à jour",toast_analysis_done:"Analyse IA terminée",toast_error:"Une erreur est survenue",nav_dashboard:"Tableau de Bord",nav_factories:"Usines",nav_ai_insights:"Analyses IA",logout_btn:"Déconnexion",login_title:"Content de vous revoir",login_subtitle:"Connectez-vous à votre compte FactoryPulse AI",ph_email:"E-mail",ph_password:"Mot de passe",remember_me:"Se souvenir de moi",login_btn:"Se connecter",login_link_register:"Pas de compte ? Créez-en un",register_title:"Créer votre compte",register_subtitle:"Commencez à surveiller vos usines avec l'IA",ph_full_name:"Nom Complet",ph_confirm_password:"Confirmer le Mot de Passe",register_btn:"Créer un Compte",register_link_login:"Déjà un compte ? Se connecter",err_missing_fields:"Veuillez remplir tous les champs",err_invalid_email:"Veuillez entrer une adresse e-mail valide",err_weak_password:"Le mot de passe doit contenir 8 caractères min., une lettre et un chiffre",err_password_mismatch:"Les mots de passe ne correspondent pas",err_invalid_credentials:"E-mail ou mot de passe incorrect",err_email_taken:"Cet e-mail est déjà enregistré",err_generic:"Une erreur est survenue. Veuillez réessayer",my_factories_title:"Mes Usines",add_factory_btn:"+ Ajouter une Usine",edit_factory_btn:"Modifier",delete_factory_btn:"Supprimer",confirm_delete_factory:"Supprimer cette usine ? Cette action est irréversible.",no_factories_yet:"Vous n'avez pas encore ajouté d'usine.",factory_created_toast:"Usine créée et analysée",factory_updated_toast:"Usine mise à jour",factory_deleted_toast:"Usine supprimée",ai_insights_feed_title:"Flux d'Analyses IA",no_ai_insights_yet:"Aucune analyse IA pour l'instant. Ajoutez une usine.",reanalyze_btn:"Réanalyser",view_insights_btn:"Voir les Analyses",created_label:"Créée le",cancel_btn:"Annuler",save_btn:"Enregistrer les Modifications"},
  es: {tagline:"Plataforma Global de Inteligencia Industrial",live_label:"En vivo",kpi_energy:"Uso de Energía",kpi_efficiency:"Eficiencia",kpi_active:"Máquinas Activas",kpi_alerts:"Alertas",kwh_unit:"kWh",chart_title:"Rendimiento en Tiempo Real",machine_status_title:"Estado de Máquinas",status_running:"Funcionando",status_warning:"Advertencia",status_critical:"Crítico",form_title:"Entrada de Datos de Fábrica",factory_name_label:"Nombre de Fábrica",machine_count_label:"Número de Máquinas",energy_cost_label:"Costo de Energía ($/kWh)",machine_type_label:"Tipo de Máquina",temperature_label:"Temperatura (°C)",vibration_label:"Vibración (mm/s)",load_label:"Carga (%)",submit_btn:"Analizar Fábrica",submitting:"Actualizando...",ai_panel_title:"Perspectivas IA",ai_placeholder:"Envíe datos de fábrica para generar un análisis IA.",ai_analyzing:"Analizando...",ai_risks:"Riesgos",ai_efficiency_insights:"Análisis de Eficiencia",ai_optimizations:"Sugerencias de Optimización",toast_updated:"Datos de fábrica actualizados",toast_analysis_done:"Análisis IA completo",toast_error:"Algo salió mal",nav_dashboard:"Panel",nav_factories:"Fábricas",nav_ai_insights:"Perspectivas IA",logout_btn:"Cerrar Sesión",login_title:"Bienvenido de nuevo",login_subtitle:"Inicia sesión en tu cuenta de FactoryPulse AI",ph_email:"Correo electrónico",ph_password:"Contraseña",remember_me:"Recuérdame",login_btn:"Iniciar Sesión",login_link_register:"¿No tienes cuenta? Crea una",register_title:"Crea tu cuenta",register_subtitle:"Empieza a monitorear tus fábricas con IA",ph_full_name:"Nombre Completo",ph_confirm_password:"Confirmar Contraseña",register_btn:"Crear Cuenta",register_link_login:"¿Ya tienes cuenta? Inicia sesión",err_missing_fields:"Por favor complete todos los campos",err_invalid_email:"Por favor ingrese un correo válido",err_weak_password:"La contraseña debe tener mín. 8 caracteres, una letra y un número",err_password_mismatch:"Las contraseñas no coinciden",err_invalid_credentials:"Correo o contraseña incorrectos",err_email_taken:"Este correo ya está registrado",err_generic:"Algo salió mal. Inténtalo de nuevo",my_factories_title:"Mis Fábricas",add_factory_btn:"+ Añadir Fábrica",edit_factory_btn:"Editar",delete_factory_btn:"Eliminar",confirm_delete_factory:"¿Eliminar esta fábrica? Esta acción no se puede deshacer.",no_factories_yet:"Aún no has añadido ninguna fábrica.",factory_created_toast:"Fábrica creada y analizada",factory_updated_toast:"Fábrica actualizada",factory_deleted_toast:"Fábrica eliminada",ai_insights_feed_title:"Feed de Perspectivas IA",no_ai_insights_yet:"Aún no hay perspectivas IA. Añade una fábrica.",reanalyze_btn:"Reanalizar",view_insights_btn:"Ver Perspectivas",created_label:"Creada",cancel_btn:"Cancelar",save_btn:"Guardar Cambios"},
  zh: {tagline:"全球工业智能平台",live_label:"实时",kpi_energy:"能源使用量",kpi_efficiency:"效率",kpi_active:"运行中设备",kpi_alerts:"警报",kwh_unit:"kWh",chart_title:"实时性能",machine_status_title:"设备状态",status_running:"运行中",status_warning:"警告",status_critical:"严重",form_title:"工厂数据输入",factory_name_label:"工厂名称",machine_count_label:"设备数量",energy_cost_label:"能源成本 ($/kWh)",machine_type_label:"设备类型",temperature_label:"温度 (°C)",vibration_label:"振动 (mm/s)",load_label:"负载 (%)",submit_btn:"分析工厂",submitting:"更新中...",ai_panel_title:"AI 洞察",ai_placeholder:"提交工厂数据以生成AI分析。",ai_analyzing:"分析中...",ai_risks:"风险",ai_efficiency_insights:"效率分析",ai_optimizations:"优化建议",toast_updated:"工厂数据已更新",toast_analysis_done:"AI分析已完成",toast_error:"出现错误",nav_dashboard:"仪表盘",nav_factories:"工厂",nav_ai_insights:"AI洞察",logout_btn:"退出",login_title:"欢迎回来",login_subtitle:"登录您的 FactoryPulse AI 账户",ph_email:"电子邮件",ph_password:"密码",remember_me:"记住我",login_btn:"登录",login_link_register:"没有账户？创建一个",register_title:"创建账户",register_subtitle:"开始使用AI监控您的工厂",ph_full_name:"全名",ph_confirm_password:"确认密码",register_btn:"创建账户",register_link_login:"已有账户？登录",err_missing_fields:"请填写所有字段",err_invalid_email:"请输入有效的电子邮件地址",err_weak_password:"密码至少8位，需包含字母和数字",err_password_mismatch:"两次密码不一致",err_invalid_credentials:"电子邮件或密码错误",err_email_taken:"该电子邮件已被注册",err_generic:"出现错误，请重试",my_factories_title:"我的工厂",add_factory_btn:"+ 添加工厂",edit_factory_btn:"编辑",delete_factory_btn:"删除",confirm_delete_factory:"删除此工厂？此操作无法撤销。",no_factories_yet:"您还没有添加任何工厂。",factory_created_toast:"工厂已创建并分析",factory_updated_toast:"工厂已更新",factory_deleted_toast:"工厂已删除",ai_insights_feed_title:"AI洞察动态",no_ai_insights_yet:"暂无AI洞察。请添加工厂开始。",reanalyze_btn:"重新分析",view_insights_btn:"查看洞察",created_label:"创建于",cancel_btn:"取消",save_btn:"保存更改"},
  ar: {tagline:"منصة الذكاء الصناعي العالمية",live_label:"مباشر",kpi_energy:"استهلاك الطاقة",kpi_efficiency:"الكفاءة",kpi_active:"الآلات النشطة",kpi_alerts:"التنبيهات",kwh_unit:"kWh",chart_title:"الأداء في الوقت الفعلي",machine_status_title:"حالة الآلات",status_running:"تعمل",status_warning:"تحذير",status_critical:"حرج",form_title:"إدخال بيانات المصنع",factory_name_label:"اسم المصنع",machine_count_label:"عدد الآلات",energy_cost_label:"تكلفة الطاقة ($/kWh)",machine_type_label:"نوع الآلة",temperature_label:"درجة الحرارة (°C)",vibration_label:"الاهتزاز (مم/ث)",load_label:"الحمل (%)",submit_btn:"تحليل المصنع",submitting:"جارٍ التحديث...",ai_panel_title:"رؤى الذكاء الاصطناعي",ai_placeholder:"أرسل بيانات المصنع لإنشاء تحليل بالذكاء الاصطناعي.",ai_analyzing:"جارٍ التحليل...",ai_risks:"المخاطر",ai_efficiency_insights:"تحليل الكفاءة",ai_optimizations:"اقتراحات التحسين",toast_updated:"تم تحديث بيانات المصنع",toast_analysis_done:"اكتمل تحليل الذكاء الاصطناعي",toast_error:"حدث خطأ ما",nav_dashboard:"لوحة التحكم",nav_factories:"المصانع",nav_ai_insights:"رؤى الذكاء الاصطناعي",logout_btn:"تسجيل الخروج",login_title:"مرحباً بعودتك",login_subtitle:"سجل الدخول إلى حساب FactoryPulse AI الخاص بك",ph_email:"البريد الإلكتروني",ph_password:"كلمة المرور",remember_me:"تذكرني",login_btn:"تسجيل الدخول",login_link_register:"ليس لديك حساب؟ أنشئ واحداً",register_title:"إنشاء حسابك",register_subtitle:"ابدأ بمراقبة مصانعك بالذكاء الاصطناعي",ph_full_name:"الاسم الكامل",ph_confirm_password:"تأكيد كلمة المرور",register_btn:"إنشاء حساب",register_link_login:"لديك حساب بالفعل؟ سجل الدخول",err_missing_fields:"يرجى ملء جميع الحقول",err_invalid_email:"يرجى إدخال بريد إلكتروني صالح",err_weak_password:"يجب أن تكون كلمة المرور 8 أحرف على الأقل وتحتوي على حرف ورقم",err_password_mismatch:"كلمتا المرور غير متطابقتين",err_invalid_credentials:"البريد الإلكتروني أو كلمة المرور غير صحيحة",err_email_taken:"هذا البريد الإلكتروني مسجل بالفعل",err_generic:"حدث خطأ ما. يرجى المحاولة مرة أخرى",my_factories_title:"مصانعي",add_factory_btn:"+ إضافة مصنع",edit_factory_btn:"تعديل",delete_factory_btn:"حذف",confirm_delete_factory:"هل تريد حذف هذا المصنع؟ لا يمكن التراجع عن هذا.",no_factories_yet:"لم تقم بإضافة أي مصنع بعد.",factory_created_toast:"تم إنشاء المصنع وتحليله",factory_updated_toast:"تم تحديث المصنع",factory_deleted_toast:"تم حذف المصنع",ai_insights_feed_title:"موجز رؤى الذكاء الاصطناعي",no_ai_insights_yet:"لا توجد رؤى بعد. أضف مصنعاً للبدء.",reanalyze_btn:"إعادة التحليل",view_insights_btn:"عرض الرؤى",created_label:"تاريخ الإنشاء",cancel_btn:"إلغاء",save_btn:"حفظ التغييرات"},
  tr: {tagline:"Küresel Endüstriyel Zeka Platformu",live_label:"Canlı",kpi_energy:"Enerji Kullanımı",kpi_efficiency:"Verimlilik",kpi_active:"Aktif Makineler",kpi_alerts:"Uyarılar",kwh_unit:"kWh",chart_title:"Gerçek Zamanlı Performans",machine_status_title:"Makine Durumu",status_running:"Çalışıyor",status_warning:"Uyarı",status_critical:"Kritik",form_title:"Fabrika Veri Girişi",factory_name_label:"Fabrika Adı",machine_count_label:"Makine Sayısı",energy_cost_label:"Enerji Maliyeti ($/kWh)",machine_type_label:"Makine Türü",temperature_label:"Sıcaklık (°C)",vibration_label:"Titreşim (mm/s)",load_label:"Yük (%)",submit_btn:"Fabrikayı Analiz Et",submitting:"Güncelleniyor...",ai_panel_title:"AI Analizleri",ai_placeholder:"AI analizi oluşturmak için fabrika verilerini gönderin.",ai_analyzing:"Analiz ediliyor...",ai_risks:"Riskler",ai_efficiency_insights:"Verimlilik Analizi",ai_optimizations:"Optimizasyon Önerileri",toast_updated:"Fabrika verileri güncellendi",toast_analysis_done:"AI analizi tamamlandı",toast_error:"Bir şeyler ters gitti",nav_dashboard:"Panel",nav_factories:"Fabrikalar",nav_ai_insights:"AI Analizleri",logout_btn:"Çıkış Yap",login_title:"Tekrar hoş geldiniz",login_subtitle:"FactoryPulse AI hesabınıza giriş yapın",ph_email:"E-posta",ph_password:"Şifre",remember_me:"Beni hatırla",login_btn:"Giriş Yap",login_link_register:"Hesabınız yok mu? Oluşturun",register_title:"Hesabınızı oluşturun",register_subtitle:"Fabrikalarınızı AI ile izlemeye başlayın",ph_full_name:"Ad Soyad",ph_confirm_password:"Şifreyi Onayla",register_btn:"Hesap Oluştur",register_link_login:"Zaten hesabınız var mı? Giriş yapın",err_missing_fields:"Lütfen tüm alanları doldurun",err_invalid_email:"Lütfen geçerli bir e-posta adresi girin",err_weak_password:"Şifre en az 8 karakter, bir harf ve bir rakam içermeli",err_password_mismatch:"Şifreler eşleşmiyor",err_invalid_credentials:"E-posta veya şifre hatalı",err_email_taken:"Bu e-posta zaten kayıtlı",err_generic:"Bir şeyler ters gitti. Tekrar deneyin",my_factories_title:"Fabrikalarım",add_factory_btn:"+ Fabrika Ekle",edit_factory_btn:"Düzenle",delete_factory_btn:"Sil",confirm_delete_factory:"Bu fabrika silinsin mi? Bu işlem geri alınamaz.",no_factories_yet:"Henüz fabrika eklemediniz.",factory_created_toast:"Fabrika oluşturuldu ve analiz edildi",factory_updated_toast:"Fabrika güncellendi",factory_deleted_toast:"Fabrika silindi",ai_insights_feed_title:"AI Analiz Akışı",no_ai_insights_yet:"Henüz AI analizi yok. Başlamak için fabrika ekleyin.",reanalyze_btn:"Yeniden Analiz Et",view_insights_btn:"Analizleri Görüntüle",created_label:"Oluşturulma",cancel_btn:"İptal",save_btn:"Değişiklikleri Kaydet"},
  it: {tagline:"Piattaforma Globale di Intelligenza Industriale",live_label:"In diretta",kpi_energy:"Consumo Energetico",kpi_efficiency:"Efficienza",kpi_active:"Macchine Attive",kpi_alerts:"Avvisi",kwh_unit:"kWh",chart_title:"Prestazioni in Tempo Reale",machine_status_title:"Stato delle Macchine",status_running:"In funzione",status_warning:"Avviso",status_critical:"Critico",form_title:"Inserimento Dati Fabbrica",factory_name_label:"Nome Fabbrica",machine_count_label:"Numero di Macchine",energy_cost_label:"Costo Energia ($/kWh)",machine_type_label:"Tipo di Macchina",temperature_label:"Temperatura (°C)",vibration_label:"Vibrazione (mm/s)",load_label:"Carico (%)",submit_btn:"Analizza Fabbrica",submitting:"Aggiornamento...",ai_panel_title:"Analisi IA",ai_placeholder:"Invia i dati della fabbrica per generare un'analisi IA.",ai_analyzing:"Analisi in corso...",ai_risks:"Rischi",ai_efficiency_insights:"Analisi dell'Efficienza",ai_optimizations:"Suggerimenti di Ottimizzazione",toast_updated:"Dati fabbrica aggiornati",toast_analysis_done:"Analisi IA completata",toast_error:"Qualcosa è andato storto",nav_dashboard:"Dashboard",nav_factories:"Fabbriche",nav_ai_insights:"Analisi IA",logout_btn:"Esci",login_title:"Bentornato",login_subtitle:"Accedi al tuo account FactoryPulse AI",ph_email:"Email",ph_password:"Password",remember_me:"Ricordami",login_btn:"Accedi",login_link_register:"Non hai un account? Creane uno",register_title:"Crea il tuo account",register_subtitle:"Inizia a monitorare le tue fabbriche con l'IA",ph_full_name:"Nome Completo",ph_confirm_password:"Conferma Password",register_btn:"Crea Account",register_link_login:"Hai già un account? Accedi",err_missing_fields:"Si prega di compilare tutti i campi",err_invalid_email:"Inserisci un indirizzo email valido",err_weak_password:"La password deve avere almeno 8 caratteri, una lettera e un numero",err_password_mismatch:"Le password non corrispondono",err_invalid_credentials:"Email o password errati",err_email_taken:"Questa email è già registrata",err_generic:"Qualcosa è andato storto. Riprova",my_factories_title:"Le Mie Fabbriche",add_factory_btn:"+ Aggiungi Fabbrica",edit_factory_btn:"Modifica",delete_factory_btn:"Elimina",confirm_delete_factory:"Eliminare questa fabbrica? Questa azione non può essere annullata.",no_factories_yet:"Non hai ancora aggiunto nessuna fabbrica.",factory_created_toast:"Fabbrica creata e analizzata",factory_updated_toast:"Fabbrica aggiornata",factory_deleted_toast:"Fabbrica eliminata",ai_insights_feed_title:"Feed di Analisi IA",no_ai_insights_yet:"Nessuna analisi IA ancora. Aggiungi una fabbrica.",reanalyze_btn:"Rianalizza",view_insights_btn:"Vedi Analisi",created_label:"Creata il",cancel_btn:"Annulla",save_btn:"Salva Modifiche"},
  pt: {tagline:"Plataforma Global de Inteligência Industrial",live_label:"Ao vivo",kpi_energy:"Uso de Energia",kpi_efficiency:"Eficiência",kpi_active:"Máquinas Ativas",kpi_alerts:"Alertas",kwh_unit:"kWh",chart_title:"Desempenho em Tempo Real",machine_status_title:"Status das Máquinas",status_running:"Em funcionamento",status_warning:"Aviso",status_critical:"Crítico",form_title:"Entrada de Dados da Fábrica",factory_name_label:"Nome da Fábrica",machine_count_label:"Número de Máquinas",energy_cost_label:"Custo de Energia ($/kWh)",machine_type_label:"Tipo de Máquina",temperature_label:"Temperatura (°C)",vibration_label:"Vibração (mm/s)",load_label:"Carga (%)",submit_btn:"Analisar Fábrica",submitting:"Atualizando...",ai_panel_title:"Insights de IA",ai_placeholder:"Envie os dados da fábrica para gerar uma análise de IA.",ai_analyzing:"Analisando...",ai_risks:"Riscos",ai_efficiency_insights:"Análise de Eficiência",ai_optimizations:"Sugestões de Otimização",toast_updated:"Dados da fábrica atualizados",toast_analysis_done:"Análise de IA concluída",toast_error:"Algo deu errado",nav_dashboard:"Painel",nav_factories:"Fábricas",nav_ai_insights:"Insights de IA",logout_btn:"Sair",login_title:"Bem-vindo de volta",login_subtitle:"Entre na sua conta FactoryPulse AI",ph_email:"E-mail",ph_password:"Senha",remember_me:"Lembrar de mim",login_btn:"Entrar",login_link_register:"Não tem conta? Crie uma",register_title:"Crie sua conta",register_subtitle:"Comece a monitorar suas fábricas com IA",ph_full_name:"Nome Completo",ph_confirm_password:"Confirmar Senha",register_btn:"Criar Conta",register_link_login:"Já tem conta? Entrar",err_missing_fields:"Por favor preencha todos os campos",err_invalid_email:"Por favor insira um e-mail válido",err_weak_password:"A senha deve ter no mínimo 8 caracteres, uma letra e um número",err_password_mismatch:"As senhas não coincidem",err_invalid_credentials:"E-mail ou senha incorretos",err_email_taken:"Este e-mail já está registrado",err_generic:"Algo deu errado. Tente novamente",my_factories_title:"Minhas Fábricas",add_factory_btn:"+ Adicionar Fábrica",edit_factory_btn:"Editar",delete_factory_btn:"Excluir",confirm_delete_factory:"Excluir esta fábrica? Esta ação não pode ser desfeita.",no_factories_yet:"Você ainda não adicionou nenhuma fábrica.",factory_created_toast:"Fábrica criada e analisada",factory_updated_toast:"Fábrica atualizada",factory_deleted_toast:"Fábrica excluída",ai_insights_feed_title:"Feed de Insights de IA",no_ai_insights_yet:"Ainda sem insights de IA. Adicione uma fábrica.",reanalyze_btn:"Reanalisar",view_insights_btn:"Ver Insights",created_label:"Criada em",cancel_btn:"Cancelar",save_btn:"Salvar Alterações"},
  ja: {tagline:"グローバル産業インテリジェンスプラットフォーム",live_label:"ライブ",kpi_energy:"エネルギー使用量",kpi_efficiency:"効率",kpi_active:"稼働中の機械",kpi_alerts:"アラート",kwh_unit:"kWh",chart_title:"リアルタイムパフォーマンス",machine_status_title:"機械の状態",status_running:"稼働中",status_warning:"警告",status_critical:"重大",form_title:"工場データ入力",factory_name_label:"工場名",machine_count_label:"機械の数",energy_cost_label:"エネルギーコスト ($/kWh)",machine_type_label:"機械の種類",temperature_label:"温度 (°C)",vibration_label:"振動 (mm/s)",load_label:"負荷 (%)",submit_btn:"工場を分析",submitting:"更新中...",ai_panel_title:"AIインサイト",ai_placeholder:"工場データを送信してAI分析を生成してください。",ai_analyzing:"分析中...",ai_risks:"リスク",ai_efficiency_insights:"効率分析",ai_optimizations:"最適化提案",toast_updated:"工場データが更新されました",toast_analysis_done:"AI分析が完了しました",toast_error:"問題が発生しました",nav_dashboard:"ダッシュボード",nav_factories:"工場",nav_ai_insights:"AIインサイト",logout_btn:"ログアウト",login_title:"おかえりなさい",login_subtitle:"FactoryPulse AI アカウントにログイン",ph_email:"メールアドレス",ph_password:"パスワード",remember_me:"ログイン状態を保持",login_btn:"ログイン",login_link_register:"アカウントをお持ちでないですか？作成する",register_title:"アカウントを作成",register_subtitle:"AIで工場の監視を始めましょう",ph_full_name:"氏名",ph_confirm_password:"パスワードの確認",register_btn:"アカウント作成",register_link_login:"すでにアカウントをお持ちですか？ログイン",err_missing_fields:"すべての項目を入力してください",err_invalid_email:"有効なメールアドレスを入力してください",err_weak_password:"パスワードは8文字以上で、文字と数字を含める必要があります",err_password_mismatch:"パスワードが一致しません",err_invalid_credentials:"メールアドレスまたはパスワードが正しくありません",err_email_taken:"このメールアドレスは既に登録されています",err_generic:"エラーが発生しました。再試行してください",my_factories_title:"マイ工場",add_factory_btn:"+ 工場を追加",edit_factory_btn:"編集",delete_factory_btn:"削除",confirm_delete_factory:"この工場を削除しますか？元に戻せません。",no_factories_yet:"まだ工場を追加していません。",factory_created_toast:"工場が作成・分析されました",factory_updated_toast:"工場が更新されました",factory_deleted_toast:"工場が削除されました",ai_insights_feed_title:"AIインサイトフィード",no_ai_insights_yet:"AIインサイトはまだありません。工場を追加してください。",reanalyze_btn:"再分析",view_insights_btn:"インサイトを見る",created_label:"作成日",cancel_btn:"キャンセル",save_btn:"変更を保存"},
  ko: {tagline:"글로벌 산업 인텔리전스 플랫폼",live_label:"실시간",kpi_energy:"에너지 사용량",kpi_efficiency:"효율성",kpi_active:"가동 중인 기계",kpi_alerts:"경고",kwh_unit:"kWh",chart_title:"실시간 성능",machine_status_title:"기계 상태",status_running:"가동 중",status_warning:"경고",status_critical:"심각",form_title:"공장 데이터 입력",factory_name_label:"공장 이름",machine_count_label:"기계 수",energy_cost_label:"에너지 비용 ($/kWh)",machine_type_label:"기계 유형",temperature_label:"온도 (°C)",vibration_label:"진동 (mm/s)",load_label:"부하 (%)",submit_btn:"공장 분석",submitting:"업데이트 중...",ai_panel_title:"AI 인사이트",ai_placeholder:"AI 분석을 생성하려면 공장 데이터를 제출하세요.",ai_analyzing:"분석 중...",ai_risks:"위험 요소",ai_efficiency_insights:"효율성 분석",ai_optimizations:"최적화 제안",toast_updated:"공장 데이터가 업데이트되었습니다",toast_analysis_done:"AI 분석이 완료되었습니다",toast_error:"문제가 발생했습니다",nav_dashboard:"대시보드",nav_factories:"공장",nav_ai_insights:"AI 인사이트",logout_btn:"로그아웃",login_title:"다시 오신 것을 환영합니다",login_subtitle:"FactoryPulse AI 계정에 로그인하세요",ph_email:"이메일",ph_password:"비밀번호",remember_me:"로그인 상태 유지",login_btn:"로그인",login_link_register:"계정이 없으신가요? 계정 만들기",register_title:"계정 만들기",register_subtitle:"AI로 공장 모니터링을 시작하세요",ph_full_name:"성명",ph_confirm_password:"비밀번호 확인",register_btn:"계정 생성",register_link_login:"이미 계정이 있으신가요? 로그인",err_missing_fields:"모든 항목을 입력해주세요",err_invalid_email:"유효한 이메일 주소를 입력하세요",err_weak_password:"비밀번호는 8자 이상, 문자와 숫자를 포함해야 합니다",err_password_mismatch:"비밀번호가 일치하지 않습니다",err_invalid_credentials:"이메일 또는 비밀번호가 올바르지 않습니다",err_email_taken:"이미 등록된 이메일입니다",err_generic:"문제가 발생했습니다. 다시 시도해주세요",my_factories_title:"내 공장",add_factory_btn:"+ 공장 추가",edit_factory_btn:"수정",delete_factory_btn:"삭제",confirm_delete_factory:"이 공장을 삭제하시겠습니까? 되돌릴 수 없습니다.",no_factories_yet:"아직 추가된 공장이 없습니다.",factory_created_toast:"공장이 생성되고 분석되었습니다",factory_updated_toast:"공장이 업데이트되었습니다",factory_deleted_toast:"공장이 삭제되었습니다",ai_insights_feed_title:"AI 인사이트 피드",no_ai_insights_yet:"아직 AI 인사이트가 없습니다. 공장을 추가하세요.",reanalyze_btn:"다시 분석",view_insights_btn:"인사이트 보기",created_label:"생성일",cancel_btn:"취소",save_btn:"변경사항 저장"},
  hi: {tagline:"वैश्विक औद्योगिक बुद्धिमत्ता मंच",live_label:"लाइव",kpi_energy:"ऊर्जा उपयोग",kpi_efficiency:"दक्षता",kpi_active:"सक्रिय मशीनें",kpi_alerts:"अलर्ट",kwh_unit:"kWh",chart_title:"रीयल-टाइम प्रदर्शन",machine_status_title:"मशीन की स्थिति",status_running:"चल रहा है",status_warning:"चेतावनी",status_critical:"गंभीर",form_title:"फ़ैक्टरी डेटा इनपुट",factory_name_label:"फ़ैक्टरी का नाम",machine_count_label:"मशीनों की संख्या",energy_cost_label:"ऊर्जा लागत ($/kWh)",machine_type_label:"मशीन प्रकार",temperature_label:"तापमान (°C)",vibration_label:"कंपन (mm/s)",load_label:"लोड (%)",submit_btn:"फ़ैक्टरी का विश्लेषण करें",submitting:"अद्यतन हो रहा है...",ai_panel_title:"AI अंतर्दृष्टि",ai_placeholder:"AI विश्लेषण उत्पन्न करने के लिए फ़ैक्टरी डेटा सबमिट करें।",ai_analyzing:"विश्लेषण हो रहा है...",ai_risks:"जोखिम",ai_efficiency_insights:"दक्षता विश्लेषण",ai_optimizations:"अनुकूलन सुझाव",toast_updated:"फ़ैक्टरी डेटा अपडेट किया गया",toast_analysis_done:"AI विश्लेषण पूर्ण हुआ",toast_error:"कुछ गलत हो गया",nav_dashboard:"डैशबोर्ड",nav_factories:"फ़ैक्टरियाँ",nav_ai_insights:"AI अंतर्दृष्टि",logout_btn:"लॉग आउट",login_title:"वापसी पर स्वागत है",login_subtitle:"अपने FactoryPulse AI खाते में लॉग इन करें",ph_email:"ईमेल",ph_password:"पासवर्ड",remember_me:"मुझे याद रखें",login_btn:"लॉग इन करें",login_link_register:"खाता नहीं है? एक बनाएं",register_title:"अपना खाता बनाएं",register_subtitle:"AI के साथ अपनी फ़ैक्टरियों की निगरानी शुरू करें",ph_full_name:"पूरा नाम",ph_confirm_password:"पासवर्ड की पुष्टि करें",register_btn:"खाता बनाएं",register_link_login:"पहले से खाता है? लॉग इन करें",err_missing_fields:"कृपया सभी फ़ील्ड भरें",err_invalid_email:"कृपया एक मान्य ईमेल पता दर्ज करें",err_weak_password:"पासवर्ड कम से कम 8 अक्षर, एक अक्षर और एक अंक होना चाहिए",err_password_mismatch:"पासवर्ड मेल नहीं खाते",err_invalid_credentials:"गलत ईमेल या पासवर्ड",err_email_taken:"यह ईमेल पहले से पंजीकृत है",err_generic:"कुछ गलत हो गया। कृपया पुनः प्रयास करें",my_factories_title:"मेरी फ़ैक्टरियाँ",add_factory_btn:"+ फ़ैक्टरी जोड़ें",edit_factory_btn:"संपादित करें",delete_factory_btn:"हटाएं",confirm_delete_factory:"इस फ़ैक्टरी को हटाएं? इसे पूर्ववत नहीं किया जा सकता।",no_factories_yet:"आपने अभी तक कोई फ़ैक्टरी नहीं जोड़ी है।",factory_created_toast:"फ़ैक्टरी बनाई और विश्लेषित की गई",factory_updated_toast:"फ़ैक्टरी अपडेट की गई",factory_deleted_toast:"फ़ैक्टरी हटाई गई",ai_insights_feed_title:"AI अंतर्दृष्टि फ़ीड",no_ai_insights_yet:"अभी तक कोई AI अंतर्दृष्टि नहीं। शुरू करने के लिए एक फ़ैक्टरी जोड़ें।",reanalyze_btn:"पुनः विश्लेषण करें",view_insights_btn:"अंतर्दृष्टि देखें",created_label:"बनाया गया",cancel_btn:"रद्द करें",save_btn:"परिवर्तन सहेजें"},
  uz: {tagline:"Global sanoat intellekti platformasi",live_label:"Jonli",kpi_energy:"Energiya sarfi",kpi_efficiency:"Samaradorlik",kpi_active:"Faol stanoklar",kpi_alerts:"Ogohlantirishlar",kwh_unit:"kWh",chart_title:"Real vaqtdagi ko'rsatkichlar",machine_status_title:"Stanoklar holati",status_running:"Ishlamoqda",status_warning:"Ogohlantirish",status_critical:"Muhim",form_title:"Zavod ma'lumotlarini kiritish",factory_name_label:"Zavod nomi",machine_count_label:"Stanoklar soni",energy_cost_label:"Energiya narxi ($/kWh)",machine_type_label:"Stanok turi",temperature_label:"Harorat (°C)",vibration_label:"Tebranish (mm/s)",load_label:"Yuklama (%)",submit_btn:"Zavodni tahlil qilish",submitting:"Yangilanmoqda...",ai_panel_title:"AI tahlili",ai_placeholder:"AI tahlilini olish uchun zavod ma'lumotlarini yuboring.",ai_analyzing:"Tahlil qilinmoqda...",ai_risks:"Xavflar",ai_efficiency_insights:"Samaradorlik tahlili",ai_optimizations:"Optimallashtirish tavsiyalari",toast_updated:"Zavod ma'lumotlari yangilandi",toast_analysis_done:"AI tahlili yakunlandi",toast_error:"Xatolik yuz berdi",nav_dashboard:"Boshqaruv paneli",nav_factories:"Zavodlar",nav_ai_insights:"AI tahlili",logout_btn:"Chiqish",login_title:"Xush kelibsiz",login_subtitle:"FactoryPulse AI hisobingizga kiring",ph_email:"Elektron pochta",ph_password:"Parol",remember_me:"Meni eslab qol",login_btn:"Kirish",login_link_register:"Hisobingiz yo'qmi? Yarating",register_title:"Hisob yarating",register_subtitle:"Zavodlaringizni AI bilan kuzatishni boshlang",ph_full_name:"To'liq ism",ph_confirm_password:"Parolni tasdiqlang",register_btn:"Hisob yaratish",register_link_login:"Hisobingiz bormi? Kiring",err_missing_fields:"Barcha maydonlarni to'ldiring",err_invalid_email:"Yaroqli elektron pochta manzilini kiriting",err_weak_password:"Parol kamida 8 belgidan, harf va raqamdan iborat bo'lishi kerak",err_password_mismatch:"Parollar mos kelmaydi",err_invalid_credentials:"Elektron pochta yoki parol noto'g'ri",err_email_taken:"Bu elektron pochta allaqachon ro'yxatdan o'tgan",err_generic:"Xatolik yuz berdi. Qaytadan urinib ko'ring",my_factories_title:"Mening Zavodlarim",add_factory_btn:"+ Zavod qo'shish",edit_factory_btn:"Tahrirlash",delete_factory_btn:"O'chirish",confirm_delete_factory:"Bu zavodni o'chirasizmi? Buni bekor qilib bo'lmaydi.",no_factories_yet:"Siz hali hech qanday zavod qo'shmagansiz.",factory_created_toast:"Zavod yaratildi va tahlil qilindi",factory_updated_toast:"Zavod yangilandi",factory_deleted_toast:"Zavod o'chirildi",ai_insights_feed_title:"AI Tahlili Lentasi",no_ai_insights_yet:"Hali AI tahlili yo'q. Boshlash uchun zavod qo'shing.",reanalyze_btn:"Qayta tahlil qilish",view_insights_btn:"Tahlilni ko'rish",created_label:"Yaratilgan",cancel_btn:"Bekor qilish",save_btn:"O'zgarishlarni saqlash"},
  ky: {tagline:"Глобалдык өнөр жай интеллект платформасы",live_label:"Түз эфир",kpi_energy:"Энергия сарпталышы",kpi_efficiency:"Эффективдүүлүк",kpi_active:"Активдүү станоктор",kpi_alerts:"Дабылдар",kwh_unit:"кВт·саат",chart_title:"Реалдуу убакыттагы көрсөткүчтөр",machine_status_title:"Станоктордун абалы",status_running:"Иштеп жатат",status_warning:"Эскертүү",status_critical:"Олуттуу",form_title:"Завод маалыматтарын киргизүү",factory_name_label:"Заводдун аты",machine_count_label:"Станоктордун саны",energy_cost_label:"Энергия наркы ($/кВт·саат)",machine_type_label:"Станоктун түрү",temperature_label:"Температура (°C)",vibration_label:"Дирилдөө (мм/с)",load_label:"Жүктөм (%)",submit_btn:"Заводду талдоо",submitting:"Жаңыртылууда...",ai_panel_title:"AI-талдоо",ai_placeholder:"AI-талдоо алуу үчүн завод маалыматтарын жөнөтүңүз.",ai_analyzing:"Талдануда...",ai_risks:"Тобокелдиктер",ai_efficiency_insights:"Эффективдүүлүк талдоосу",ai_optimizations:"Оптималдаштыруу сунуштары",toast_updated:"Завод маалыматтары жаңыртылды",toast_analysis_done:"AI-талдоо аяктады",toast_error:"Ката кетти",nav_dashboard:"Башкаруу панели",nav_factories:"Заводдор",nav_ai_insights:"AI-талдоо",logout_btn:"Чыгуу",login_title:"Кайра кош келиңиз",login_subtitle:"FactoryPulse AI каттоо эсебиңизге кириңиз",ph_email:"Электрондук почта",ph_password:"Сырсөз",remember_me:"Мени эстеп кал",login_btn:"Кирүү",login_link_register:"Каттоо эсебиңиз жокпу? Түзүү",register_title:"Каттоо эсебин түзүү",register_subtitle:"Заводдоруңузду AI менен байкоону баштаңыз",ph_full_name:"Толук аты-жөнү",ph_confirm_password:"Сырсөздү ырастаңыз",register_btn:"Каттоо эсебин түзүү",register_link_login:"Каттоо эсебиңиз барбы? Кирүү",err_missing_fields:"Бардык талааларды толтуруңуз",err_invalid_email:"Жарактуу электрондук почта дарегин киргизиңиз",err_weak_password:"Сырсөз кеминде 8 белги, тамга жана сан камтышы керек",err_password_mismatch:"Сырсөздөр дал келбейт",err_invalid_credentials:"Электрондук почта же сырсөз туура эмес",err_email_taken:"Бул электрондук почта мурунтан катталган",err_generic:"Ката кетти. Кайра аракет кылыңыз",my_factories_title:"Менин Заводдорум",add_factory_btn:"+ Завод кошуу",edit_factory_btn:"Түзөтүү",delete_factory_btn:"Өчүрүү",confirm_delete_factory:"Бул заводду өчүрөсүзбү? Бул аракетти артка кайтарууга болбойт.",no_factories_yet:"Сиз азырынча эч кандай завод кошкон жоксуз.",factory_created_toast:"Завод түзүлдү жана талданды",factory_updated_toast:"Завод жаңыртылды",factory_deleted_toast:"Завод өчүрүлдү",ai_insights_feed_title:"AI-талдоо тизмеси",no_ai_insights_yet:"Азырынча AI-талдоо жок. Баштоо үчүн завод кошуңуз.",reanalyze_btn:"Кайра талдоо",view_insights_btn:"Талдоону көрүү",created_label:"Түзүлгөн күнү",cancel_btn:"Жокко чыгаруу",save_btn:"Өзгөртүүлөрдү сактоо"},
  uk: {tagline:"Глобальна платформа промислового інтелекту",live_label:"Наживо",kpi_energy:"Споживання енергії",kpi_efficiency:"Ефективність",kpi_active:"Активні верстати",kpi_alerts:"Сповіщення",kwh_unit:"кВт·год",chart_title:"Показники в реальному часі",machine_status_title:"Статус верстатів",status_running:"Працює",status_warning:"Попередження",status_critical:"Критично",form_title:"Введення даних заводу",factory_name_label:"Назва заводу",machine_count_label:"Кількість верстатів",energy_cost_label:"Вартість енергії ($/кВт·год)",machine_type_label:"Тип верстата",temperature_label:"Температура (°C)",vibration_label:"Вібрація (мм/с)",load_label:"Навантаження (%)",submit_btn:"Аналізувати завод",submitting:"Оновлення...",ai_panel_title:"AI-аналітика",ai_placeholder:"Надішліть дані заводу, щоб отримати AI-аналіз.",ai_analyzing:"Аналіз...",ai_risks:"Ризики",ai_efficiency_insights:"Аналіз ефективності",ai_optimizations:"Рекомендації з оптимізації",toast_updated:"Дані заводу оновлено",toast_analysis_done:"AI-аналіз завершено",toast_error:"Сталася помилка",nav_dashboard:"Панель",nav_factories:"Заводи",nav_ai_insights:"AI-аналітика",logout_btn:"Вийти",login_title:"З поверненням",login_subtitle:"Увійдіть у свій обліковий запис FactoryPulse AI",ph_email:"Електронна пошта",ph_password:"Пароль",remember_me:"Запам'ятати мене",login_btn:"Увійти",login_link_register:"Немає акаунту? Створити",register_title:"Створіть акаунт",register_subtitle:"Почніть моніторинг заводів за допомогою AI",ph_full_name:"Повне ім'я",ph_confirm_password:"Підтвердіть пароль",register_btn:"Створити акаунт",register_link_login:"Вже є акаунт? Увійти",err_missing_fields:"Будь ласка, заповніть усі поля",err_invalid_email:"Введіть дійсну електронну адресу",err_weak_password:"Пароль має містити щонайменше 8 символів, літеру та цифру",err_password_mismatch:"Паролі не збігаються",err_invalid_credentials:"Невірна електронна пошта або пароль",err_email_taken:"Ця електронна пошта вже зареєстрована",err_generic:"Сталася помилка. Спробуйте ще раз",my_factories_title:"Мої Заводи",add_factory_btn:"+ Додати завод",edit_factory_btn:"Редагувати",delete_factory_btn:"Видалити",confirm_delete_factory:"Видалити цей завод? Цю дію не можна скасувати.",no_factories_yet:"Ви ще не додали жодного заводу.",factory_created_toast:"Завод створено та проаналізовано",factory_updated_toast:"Завод оновлено",factory_deleted_toast:"Завод видалено",ai_insights_feed_title:"Стрічка AI-аналітики",no_ai_insights_yet:"Ще немає AI-аналітики. Додайте завод.",reanalyze_btn:"Проаналізувати знову",view_insights_btn:"Переглянути аналітику",created_label:"Створено",cancel_btn:"Скасувати",save_btn:"Зберегти зміни"},
  pl: {tagline:"Globalna Platforma Inteligencji Przemysłowej",live_label:"Na żywo",kpi_energy:"Zużycie Energii",kpi_efficiency:"Wydajność",kpi_active:"Aktywne Maszyny",kpi_alerts:"Alerty",kwh_unit:"kWh",chart_title:"Wydajność w Czasie Rzeczywistym",machine_status_title:"Status Maszyn",status_running:"Działa",status_warning:"Ostrzeżenie",status_critical:"Krytyczne",form_title:"Wprowadzanie Danych Fabryki",factory_name_label:"Nazwa Fabryki",machine_count_label:"Liczba Maszyn",energy_cost_label:"Koszt Energii ($/kWh)",machine_type_label:"Typ Maszyny",temperature_label:"Temperatura (°C)",vibration_label:"Wibracje (mm/s)",load_label:"Obciążenie (%)",submit_btn:"Analizuj Fabrykę",submitting:"Aktualizowanie...",ai_panel_title:"Analizy AI",ai_placeholder:"Prześlij dane fabryki, aby wygenerować analizę AI.",ai_analyzing:"Analizowanie...",ai_risks:"Ryzyka",ai_efficiency_insights:"Analiza Wydajności",ai_optimizations:"Sugestie Optymalizacji",toast_updated:"Dane fabryki zaktualizowane",toast_analysis_done:"Analiza AI zakończona",toast_error:"Coś poszło nie tak",nav_dashboard:"Panel",nav_factories:"Fabryki",nav_ai_insights:"Analizy AI",logout_btn:"Wyloguj",login_title:"Witamy z powrotem",login_subtitle:"Zaloguj się do swojego konta FactoryPulse AI",ph_email:"E-mail",ph_password:"Hasło",remember_me:"Zapamiętaj mnie",login_btn:"Zaloguj się",login_link_register:"Nie masz konta? Utwórz je",register_title:"Utwórz konto",register_subtitle:"Zacznij monitorować swoje fabryki z AI",ph_full_name:"Imię i Nazwisko",ph_confirm_password:"Potwierdź Hasło",register_btn:"Utwórz Konto",register_link_login:"Masz już konto? Zaloguj się",err_missing_fields:"Proszę wypełnić wszystkie pola",err_invalid_email:"Proszę podać prawidłowy adres e-mail",err_weak_password:"Hasło musi mieć min. 8 znaków, literę i cyfrę",err_password_mismatch:"Hasła nie pasują do siebie",err_invalid_credentials:"Nieprawidłowy e-mail lub hasło",err_email_taken:"Ten e-mail jest już zarejestrowany",err_generic:"Coś poszło nie tak. Spróbuj ponownie",my_factories_title:"Moje Fabryki",add_factory_btn:"+ Dodaj Fabrykę",edit_factory_btn:"Edytuj",delete_factory_btn:"Usuń",confirm_delete_factory:"Usunąć tę fabrykę? Tej czynności nie można cofnąć.",no_factories_yet:"Nie dodałeś jeszcze żadnej fabryki.",factory_created_toast:"Fabryka utworzona i przeanalizowana",factory_updated_toast:"Fabryka zaktualizowana",factory_deleted_toast:"Fabryka usunięta",ai_insights_feed_title:"Kanał Analiz AI",no_ai_insights_yet:"Brak analiz AI. Dodaj fabrykę, aby zacząć.",reanalyze_btn:"Analizuj Ponownie",view_insights_btn:"Zobacz Analizy",created_label:"Utworzono",cancel_btn:"Anuluj",save_btn:"Zapisz Zmiany"},
  nl: {tagline:"Wereldwijd Industrieel Intelligentieplatform",live_label:"Live",kpi_energy:"Energieverbruik",kpi_efficiency:"Efficiëntie",kpi_active:"Actieve Machines",kpi_alerts:"Meldingen",kwh_unit:"kWh",chart_title:"Realtime Prestaties",machine_status_title:"Machinestatus",status_running:"Actief",status_warning:"Waarschuwing",status_critical:"Kritiek",form_title:"Fabrieksgegevens Invoeren",factory_name_label:"Fabrieksnaam",machine_count_label:"Aantal Machines",energy_cost_label:"Energiekosten ($/kWh)",machine_type_label:"Machinetype",temperature_label:"Temperatuur (°C)",vibration_label:"Trilling (mm/s)",load_label:"Belasting (%)",submit_btn:"Fabriek Analyseren",submitting:"Bijwerken...",ai_panel_title:"AI-inzichten",ai_placeholder:"Verzend fabrieksgegevens om een AI-analyse te genereren.",ai_analyzing:"Analyseren...",ai_risks:"Risico's",ai_efficiency_insights:"Efficiëntieanalyse",ai_optimizations:"Optimalisatiesuggesties",toast_updated:"Fabrieksgegevens bijgewerkt",toast_analysis_done:"AI-analyse voltooid",toast_error:"Er is iets misgegaan",nav_dashboard:"Dashboard",nav_factories:"Fabrieken",nav_ai_insights:"AI-inzichten",logout_btn:"Uitloggen",login_title:"Welkom terug",login_subtitle:"Log in op uw FactoryPulse AI-account",ph_email:"E-mail",ph_password:"Wachtwoord",remember_me:"Onthoud mij",login_btn:"Inloggen",login_link_register:"Geen account? Maak er een",register_title:"Maak uw account aan",register_subtitle:"Begin met AI-monitoring van uw fabrieken",ph_full_name:"Volledige Naam",ph_confirm_password:"Bevestig Wachtwoord",register_btn:"Account Aanmaken",register_link_login:"Heeft u al een account? Inloggen",err_missing_fields:"Vul alle velden in",err_invalid_email:"Voer een geldig e-mailadres in",err_weak_password:"Wachtwoord moet minimaal 8 tekens, een letter en een cijfer bevatten",err_password_mismatch:"Wachtwoorden komen niet overeen",err_invalid_credentials:"Ongeldige e-mail of wachtwoord",err_email_taken:"Dit e-mailadres is al geregistreerd",err_generic:"Er is iets misgegaan. Probeer het opnieuw",my_factories_title:"Mijn Fabrieken",add_factory_btn:"+ Fabriek Toevoegen",edit_factory_btn:"Bewerken",delete_factory_btn:"Verwijderen",confirm_delete_factory:"Deze fabriek verwijderen? Dit kan niet ongedaan worden gemaakt.",no_factories_yet:"U heeft nog geen fabrieken toegevoegd.",factory_created_toast:"Fabriek aangemaakt en geanalyseerd",factory_updated_toast:"Fabriek bijgewerkt",factory_deleted_toast:"Fabriek verwijderd",ai_insights_feed_title:"AI-inzichten Feed",no_ai_insights_yet:"Nog geen AI-inzichten. Voeg een fabriek toe.",reanalyze_btn:"Opnieuw Analyseren",view_insights_btn:"Bekijk Inzichten",created_label:"Aangemaakt",cancel_btn:"Annuleren",save_btn:"Wijzigingen Opslaan"},
  sv: {tagline:"Global Industriell Intelligensplattform",live_label:"Live",kpi_energy:"Energiförbrukning",kpi_efficiency:"Effektivitet",kpi_active:"Aktiva Maskiner",kpi_alerts:"Varningar",kwh_unit:"kWh",chart_title:"Realtidsprestanda",machine_status_title:"Maskinstatus",status_running:"Igång",status_warning:"Varning",status_critical:"Kritisk",form_title:"Fabriksdatainmatning",factory_name_label:"Fabriksnamn",machine_count_label:"Antal Maskiner",energy_cost_label:"Energikostnad ($/kWh)",machine_type_label:"Maskintyp",temperature_label:"Temperatur (°C)",vibration_label:"Vibration (mm/s)",load_label:"Belastning (%)",submit_btn:"Analysera Fabrik",submitting:"Uppdaterar...",ai_panel_title:"AI-insikter",ai_placeholder:"Skicka fabriksdata för att generera en AI-analys.",ai_analyzing:"Analyserar...",ai_risks:"Risker",ai_efficiency_insights:"Effektivitetsanalys",ai_optimizations:"Optimeringsförslag",toast_updated:"Fabriksdata uppdaterad",toast_analysis_done:"AI-analys klar",toast_error:"Något gick fel",nav_dashboard:"Instrumentpanel",nav_factories:"Fabriker",nav_ai_insights:"AI-insikter",logout_btn:"Logga ut",login_title:"Välkommen tillbaka",login_subtitle:"Logga in på ditt FactoryPulse AI-konto",ph_email:"E-post",ph_password:"Lösenord",remember_me:"Kom ihåg mig",login_btn:"Logga in",login_link_register:"Inget konto? Skapa ett",register_title:"Skapa ditt konto",register_subtitle:"Börja övervaka dina fabriker med AI",ph_full_name:"Fullständigt Namn",ph_confirm_password:"Bekräfta Lösenord",register_btn:"Skapa Konto",register_link_login:"Har du redan ett konto? Logga in",err_missing_fields:"Vänligen fyll i alla fält",err_invalid_email:"Ange en giltig e-postadress",err_weak_password:"Lösenordet måste vara minst 8 tecken med en bokstav och en siffra",err_password_mismatch:"Lösenorden matchar inte",err_invalid_credentials:"Felaktig e-post eller lösenord",err_email_taken:"Denna e-post är redan registrerad",err_generic:"Något gick fel. Försök igen",my_factories_title:"Mina Fabriker",add_factory_btn:"+ Lägg till Fabrik",edit_factory_btn:"Redigera",delete_factory_btn:"Ta bort",confirm_delete_factory:"Ta bort denna fabrik? Detta kan inte ångras.",no_factories_yet:"Du har inte lagt till några fabriker än.",factory_created_toast:"Fabrik skapad och analyserad",factory_updated_toast:"Fabrik uppdaterad",factory_deleted_toast:"Fabrik borttagen",ai_insights_feed_title:"AI-insikter Flöde",no_ai_insights_yet:"Inga AI-insikter än. Lägg till en fabrik.",reanalyze_btn:"Analysera Igen",view_insights_btn:"Visa Insikter",created_label:"Skapad",cancel_btn:"Avbryt",save_btn:"Spara Ändringar"},
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
        body: JSON.stringify({ full_name, email, password, confirm_password }),
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
    return redirect("/dashboard")


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return INDEX_HTML


@app.route("/login", methods=["GET"])
def login_page():
    return LOGIN_HTML


@app.route("/register", methods=["GET"])
def register_page():
    return REGISTER_HTML


if __name__ == "__main__":
    CERT_FILE = "cert.pem"
    KEY_FILE = "key.pem"

    print("=" * 70)
    print("FactoryPulse AI is starting...")
    print(f"Gemini AI enabled: {GEMINI_ENABLED}")
    if not GEMINI_ENABLED:
        print("Set GEMINI_API_KEY env var to enable live Gemini analysis.")
        print("Running on deterministic fallback analysis engine for demo purposes.")
    print(f"Data mode: {DATA_MODE}  (USB available: {USB_AVAILABLE}, PLC available: {PLC_AVAILABLE})")
    print(f"Real-time WebSocket push: {'enabled' if SOCKETIO_ENABLED else 'disabled (falling back to polling)'}")

    def _run(**kwargs):
        if SOCKETIO_ENABLED:
          socketio.run(app, allow_unsafe_werkzeug=True, **kwargs)
        else:
            app.run(**kwargs)

    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        print("SSL certificate found - starting with HTTPS enabled.")
        print("Visit: https://localhost:5000")
        print("=" * 70)
        _run(host="0.0.0.0", port=5000, ssl_context=(CERT_FILE, KEY_FILE), debug=True)
    else:
        print("No SSL certificate found (cert.pem / key.pem).")
        print("Generate one with:")
        print("  openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes")
        print("Falling back to plain HTTP for local development.")
        print("Visit: http://localhost:5000")
        print("=" * 70)
        _run(host="0.0.0.0", port=5000, debug=True)
