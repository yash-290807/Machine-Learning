import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import mysql.connector


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = r"C:\AI Projects\HelmetViolationSystem"

MODEL_PATH = os.path.join(
    ROOT,
    "runs",
    "helmet_yolo11n-3",
    "weights",
    "best.pt"
)

EVIDENCE_DIR = os.path.join(
    ROOT,
    "outputs",
    "evidence"
)

CONFIDENCE_THRESHOLD = 0.25
FINE_AMOUNT = 500

os.makedirs(EVIDENCE_DIR, exist_ok=True)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Helmet Violation Detection System",
    page_icon="🪖",
    layout="wide"
)


# ============================================================
# MYSQL CONNECTION
# ============================================================

def get_db_connection():
    return mysql.connector.connect(
        host=st.secrets["MYSQL_HOST"],
        port=int(st.secrets["MYSQL_PORT"]),
        user=st.secrets["MYSQL_USER"],
        password=st.secrets["MYSQL_PASSWORD"],
        database=st.secrets["MYSQL_DATABASE"]
    )


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_database():

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS violations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME NOT NULL,
            helmet_status VARCHAR(50) NOT NULL,
            confidence FLOAT NOT NULL,
            evidence_image VARCHAR(500) NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS challans (
            id INT AUTO_INCREMENT PRIMARY KEY,
            violation_id INT NOT NULL,
            challan_number VARCHAR(100) UNIQUE NOT NULL,
            generated_at DATETIME NOT NULL,
            fine_amount INT NOT NULL,
            status VARCHAR(50) NOT NULL,
            FOREIGN KEY (violation_id)
                REFERENCES violations(id)
        )
        """
    )

    conn.commit()

    cursor.close()
    conn.close()


# ============================================================
# SAVE VIOLATION
# ============================================================

def save_violation(
    timestamp,
    confidence,
    evidence_image
):

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO violations
        (
            timestamp,
            helmet_status,
            confidence,
            evidence_image
        )
        VALUES (%s, %s, %s, %s)
        """,
        (
            timestamp,
            "No Helmet",
            float(confidence),
            evidence_image
        )
    )

    violation_id = cursor.lastrowid

    conn.commit()

    cursor.close()
    conn.close()

    return violation_id


# ============================================================
# GET VIOLATIONS
# ============================================================

def get_all_violations():

    conn = get_db_connection()

    query = """
        SELECT
            id,
            timestamp,
            helmet_status,
            confidence,
            evidence_image
        FROM violations
        ORDER BY id DESC
    """

    df = pd.read_sql(
        query,
        conn
    )

    conn.close()

    return df


# ============================================================
# GENERATE CHALLAN
# ============================================================

def generate_challan(
    violation_id
):

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT challan_number
        FROM challans
        WHERE violation_id = %s
        """,
        (violation_id,)
    )

    existing = cursor.fetchone()

    if existing:

        cursor.close()
        conn.close()

        return existing[0]

    now = datetime.now()

    challan_number = (
        f"ECH-"
        f"{now.strftime('%Y%m%d')}-"
        f"{violation_id:04d}"
    )

    cursor.execute(
        """
        INSERT INTO challans
        (
            violation_id,
            challan_number,
            generated_at,
            fine_amount,
            status
        )
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            violation_id,
            challan_number,
            now,
            FINE_AMOUNT,
            "Generated"
        )
    )

    conn.commit()

    cursor.close()
    conn.close()

    return challan_number


# ============================================================
# GET CHALLAN
# ============================================================

def get_challan(
    violation_id
):

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            challan_number,
            generated_at,
            fine_amount,
            status
        FROM challans
        WHERE violation_id = %s
        """,
        (violation_id,)
    )

    row = cursor.fetchone()

    cursor.close()
    conn.close()

    return row


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():

    return YOLO(
        MODEL_PATH
    )


# ============================================================
# DETECTION
# ============================================================

def detect_image(
    image
):

    model = load_model()

    image_rgb = np.array(
        image.convert("RGB")
    )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    results = model.predict(
        source=image_bgr,
        conf=CONFIDENCE_THRESHOLD,
        device=0,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            class_name = str(
                result.names[class_id]
            ).lower()

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            detections.append(
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            )

    return detections, image_bgr


# ============================================================
# SAVE FULL FRAME EVIDENCE
# ============================================================

def save_full_frame_evidence(
    image_bgr,
    detections
):

    evidence = image_bgr.copy()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Draw detections
    for detection in detections:

        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]

        class_name = detection["class_name"]
        confidence = detection["confidence"]

        is_violation = class_name in [
            "no_helmet",
            "not_helmet",
            "no helmet",
            "not helmet"
        ]

        if is_violation:

            box_color = (0, 0, 255)

            label = (
                f"NO HELMET "
                f"{confidence * 100:.1f}%"
            )

        else:

            box_color = (0, 255, 0)

            label = (
                f"HELMET "
                f"{confidence * 100:.1f}%"
            )

        cv2.rectangle(
            evidence,
            (x1, y1),
            (x2, y2),
            box_color,
            3
        )

        cv2.putText(
            evidence,
            label,
            (
                x1,
                max(30, y1 - 10)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            box_color,
            2,
            cv2.LINE_AA
        )

    # Timestamp banner
    banner_height = 55

    cv2.rectangle(
        evidence,
        (0, 0),
        (
            evidence.shape[1],
            banner_height
        ),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        evidence,
        f"HELMET VIOLATION EVIDENCE | {timestamp}",
        (15, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    filename = (
        "violation_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        + ".jpg"
    )

    full_path = os.path.join(
        EVIDENCE_DIR,
        filename
    )

    cv2.imwrite(
        full_path,
        evidence
    )

    return full_path


# ============================================================
# INITIALIZE MYSQL
# ============================================================

try:

    init_database()

except Exception as e:

    st.error(
        "Unable to connect to MySQL."
    )

    st.error(
        str(e)
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🪖 Helmet Violation System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Detect Violation",
        "Violation History",
        "E-Challan"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title(
        "🪖 Helmet Violation Detection System"
    )

    st.subheader(
        "Dashboard"
    )

    df = get_all_violations()

    total = len(df)

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    if total:

        today_count = sum(
            df["timestamp"]
            .astype(str)
            .str.startswith(today)
        )

        avg_confidence = (
            df["confidence"].mean()
            * 100
        )

    else:

        today_count = 0
        avg_confidence = 0

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM challans"
    )

    challan_count = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Total Violations",
            total
        )

    with c2:

        st.metric(
            "Today's Violations",
            int(today_count)
        )

    with c3:

        st.metric(
            "Avg Confidence",
            f"{avg_confidence:.1f}%"
        )

    with c4:

        st.metric(
            "Challans Generated",
            challan_count
        )

    st.markdown("---")

    st.subheader(
        "System Workflow"
    )

    st.markdown(
        """
        **Image Input**
        → **YOLO Helmet Detection**
        → **No Helmet Detection**
        → **Evidence Snapshot**
        → **Timestamp**
        → **MySQL Database**
        → **E-Challan**
        """
    )

    st.markdown("---")

    if total:

        st.subheader(
            "Recent Violations"
        )

        recent = df.head(5).copy()

        recent["confidence"] = (
            recent["confidence"]
            * 100
        ).round(2).astype(str) + "%"

        recent = recent.rename(
            columns={
                "id": "ID",
                "timestamp": "Timestamp",
                "helmet_status": "Status",
                "confidence": "Confidence"
            }
        )

        st.dataframe(
            recent[
                [
                    "ID",
                    "Timestamp",
                    "Status",
                    "Confidence"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No violations recorded yet."
        )


# ============================================================
# DETECT VIOLATION
# ============================================================

elif page == "Detect Violation":

    st.title(
        "🔍 Detect Helmet Violation"
    )

    st.write(
        "Upload a traffic image to detect helmet "
        "violations and capture digital evidence."
    )

    uploaded = st.file_uploader(
        "Upload Image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded:

        image = Image.open(
            uploaded
        ).convert("RGB")

        st.subheader(
            "Input Image"
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown("---")

        if st.button(
            "🚨 Analyze Image",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Running YOLO helmet detection..."
            ):

                detections, image_bgr = (
                    detect_image(image)
                )

            no_helmet = [
                d for d in detections
                if d["class_name"] in [
                    "no_helmet",
                    "not_helmet",
                    "no helmet",
                    "not helmet"
                ]
            ]

            helmets = [
                d for d in detections
                if d["class_name"] == "helmet"
            ]

            a, b = st.columns(2)

            with a:

                st.metric(
                    "Helmet",
                    len(helmets)
                )

            with b:

                st.metric(
                    "No Helmet",
                    len(no_helmet)
                )

            # ------------------------------------------------
            # NO VIOLATION
            # ------------------------------------------------

            if not no_helmet:

                st.success(
                    "✅ No helmet violation detected."
                )

                model = load_model()

                result = model.predict(
                    source=image_bgr,
                    conf=CONFIDENCE_THRESHOLD,
                    device=0,
                    verbose=False
                )[0]

                annotated = result.plot()

                annotated = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    annotated,
                    caption="Detection Result",
                    use_container_width=True
                )

            # ------------------------------------------------
            # VIOLATION
            # ------------------------------------------------

            else:

                st.error(
                    f"🚨 {len(no_helmet)} "
                    "helmet violation(s) detected!"
                )

                evidence_path = (
                    save_full_frame_evidence(
                        image_bgr,
                        detections
                    )
                )

                strongest = max(
                    no_helmet,
                    key=lambda x: x["confidence"]
                )

                timestamp = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                violation_id = save_violation(
                    timestamp,
                    strongest["confidence"],
                    evidence_path
                )

                st.success(
                    f"Violation #{violation_id} "
                    "recorded successfully."
                )

                st.subheader(
                    "📸 Evidence Snapshot"
                )

                st.image(
                    evidence_path,
                    caption=(
                        f"Violation #{violation_id} | "
                        f"{timestamp}"
                    ),
                    use_container_width=True
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Violation ID",
                        violation_id
                    )

                with c2:

                    st.metric(
                        "Confidence",
                        f"{strongest['confidence'] * 100:.1f}%"
                    )

                with c3:

                    st.metric(
                        "Status",
                        "No Helmet"
                    )

                st.info(
                    "Evidence and timestamp have been "
                    "saved to MySQL."
                )


# ============================================================
# VIOLATION HISTORY
# ============================================================

elif page == "Violation History":

    st.title(
        "📋 Violation History"
    )

    df = get_all_violations()

    if df.empty:

        st.info(
            "No violation records found."
        )

    else:

        st.write(
            f"Total violations: **{len(df)}**"
        )

        display = df.copy()

        display["confidence"] = (
            display["confidence"]
            * 100
        ).round(2).astype(str) + "%"

        display["evidence_image"] = (
            display["evidence_image"]
            .apply(os.path.basename)
        )

        display = display.rename(
            columns={
                "id": "ID",
                "timestamp": "Timestamp",
                "helmet_status": "Status",
                "confidence": "Confidence",
                "evidence_image": "Evidence"
            }
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        st.subheader(
            "Evidence Viewer"
        )

        selected_id = st.number_input(
            "Violation ID",
            min_value=1,
            step=1
        )

        selected = df[
            df["id"] == selected_id
        ]

        if not selected.empty:

            row = selected.iloc[0]

            if os.path.exists(
                row["evidence_image"]
            ):

                st.image(
                    row["evidence_image"],
                    caption=(
                        f"Violation #{int(row['id'])} | "
                        f"{row['timestamp']}"
                    ),
                    use_container_width=True
                )

            else:

                st.warning(
                    "Evidence image not found."
                )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.write(
                    f"**Status:** "
                    f"{row['helmet_status']}"
                )

            with c2:

                st.write(
                    f"**Confidence:** "
                    f"{row['confidence'] * 100:.2f}%"
                )

            with c3:

                st.write(
                    f"**Timestamp:** "
                    f"{row['timestamp']}"
                )


# ============================================================
# E-CHALLAN
# ============================================================

elif page == "E-Challan":

    st.title(
        "🧾 E-Challan Management"
    )

    df = get_all_violations()

    if df.empty:

        st.warning(
            "No violations are available. "
            "Detect a violation first."
        )

    else:

        violation_id = st.selectbox(
            "Select Violation",
            df["id"].tolist()
        )

        row = df[
            df["id"] == violation_id
        ].iloc[0]

        st.subheader(
            f"Violation #{violation_id}"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**Violation:** "
                f"{row['helmet_status']}"
            )

            st.write(
                f"**Confidence:** "
                f"{row['confidence'] * 100:.2f}%"
            )

            st.write(
                f"**Detected At:** "
                f"{row['timestamp']}"
            )

            st.write(
                f"**Fine Amount:** "
                f"₹{FINE_AMOUNT}"
            )

        with col2:

            if os.path.exists(
                row["evidence_image"]
            ):

                st.image(
                    row["evidence_image"],
                    caption="Violation Evidence",
                    use_container_width=True
                )

        st.markdown("---")

        existing = get_challan(
            violation_id
        )

        if existing:

            st.success(
                "✅ Challan already generated."
            )

            challan_number = existing[0]

        else:

            if st.button(
                "🧾 Generate E-Challan",
                type="primary",
                use_container_width=True
            ):

                generate_challan(
                    violation_id
                )

                st.success(
                    "E-Challan generated successfully!"
                )

                st.rerun()

            challan_number = None

        if challan_number:

            existing = get_challan(
                violation_id
            )

            challan_number = existing[0]
            generated_at = existing[1]
            fine = existing[2]
            status = existing[3]

            st.markdown("---")

            st.subheader(
                "E-Challan Details"
            )

            st.write(
                f"**Challan Number:** "
                f"`{challan_number}`"
            )

            st.write(
                f"**Violation ID:** "
                f"`{violation_id}`"
            )

            st.write(
                "**Violation:** `No Helmet`"
            )

            st.write(
                f"**Generated At:** "
                f"`{generated_at}`"
            )

            st.write(
                f"**Fine:** "
                f"`₹{fine}`"
            )

            st.write(
                f"**Status:** "
                f"`{status}`"
            )

            st.success(
                "Challan record stored in MySQL database."
            )