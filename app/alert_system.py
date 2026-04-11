"""
alert_system.py — Gmail SMTP alert for PipelineGuard
Fires when prediction probability > 0.75.
Email contains: pipeline name, risk score, top 3 SHAP features.

Usage:
    from app.alert_system import send_alert
    send_alert(pipeline_name, risk_score, top_shap_features)

Environment variables required:
    ALERT_EMAIL_SENDER   — Gmail address sending the alert
    ALERT_EMAIL_PASSWORD — Gmail App Password (not your login password)
    ALERT_EMAIL_RECEIVER — Email address to receive alerts
"""

import smtplib
import os
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

logger = logging.getLogger(__name__)

RISK_THRESHOLD = 0.75


def send_alert(
    pipeline_name: str,
    risk_score: float,
    top_shap_features: list,
    sender_email: str = None,
    sender_password: str = None,
    receiver_email: str = None,
) -> bool:
    """
    Send a Gmail SMTP alert when risk_score > RISK_THRESHOLD.

    Returns True if email was sent, False otherwise.
    """
    if risk_score <= RISK_THRESHOLD:
        logger.info(
            f"Risk score {risk_score:.4f} <= {RISK_THRESHOLD} — no alert sent."
        )
        return False

    sender   = sender_email    or os.getenv("ALERT_EMAIL_SENDER",   "lakhsmikranthiamathi@gmail.com")
    password = sender_password or os.getenv("ALERT_EMAIL_PASSWORD", "")
    receiver = receiver_email  or os.getenv("ALERT_EMAIL_RECEIVER", "lakhsmikranthiamathi@gmail.com")

    if not sender or not password:
        logger.warning(
            "Alert credentials not set. "
            "Set ALERT_EMAIL_SENDER and ALERT_EMAIL_PASSWORD env vars."
        )
        return False

    risk_level = _get_risk_level(risk_score)
    shap_str   = _format_shap(top_shap_features)
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    subject = f"🚨 PipelineGuard Alert — {pipeline_name} | Risk: {risk_score:.1%}"
    html_body = f"""
    <html><body style="font-family: Arial, sans-serif; max-width: 600px; margin: auto;">
      <div style="background:#d32f2f;color:white;padding:16px;border-radius:8px 8px 0 0;">
        <h2 style="margin:0;">🚨 PipelineGuard High-Risk Alert</h2>
      </div>
      <div style="border:1px solid #ddd;border-top:none;padding:20px;border-radius:0 0 8px 8px;">
        <table style="width:100%;border-collapse:collapse;">
          <tr>
            <td style="padding:8px;font-weight:bold;color:#555;">Pipeline</td>
            <td style="padding:8px;">{pipeline_name}</td>
          </tr>
          <tr style="background:#f9f9f9;">
            <td style="padding:8px;font-weight:bold;color:#555;">Risk Score</td>
            <td style="padding:8px;">
              <span style="font-size:1.4em;font-weight:bold;color:#d32f2f;">
                {risk_score:.1%}
              </span>
              &nbsp;({risk_level})
            </td>
          </tr>
          <tr>
            <td style="padding:8px;font-weight:bold;color:#555;">Top Risk Factors</td>
            <td style="padding:8px;">{shap_str}</td>
          </tr>
          <tr style="background:#f9f9f9;">
            <td style="padding:8px;font-weight:bold;color:#555;">Timestamp</td>
            <td style="padding:8px;">{timestamp}</td>
          </tr>
        </table>
        <p style="margin-top:16px;color:#777;font-size:0.85em;">
          This is an automated alert from PipelineGuard. 
          Please investigate the pipeline immediately.
        </p>
      </div>
    </body></html>
    """

    plain_body = (
        f"PipelineGuard High-Risk Alert\n"
        f"{'='*40}\n"
        f"Pipeline   : {pipeline_name}\n"
        f"Risk Score : {risk_score:.1%} ({risk_level})\n"
        f"Top Factors: {', '.join(top_shap_features[:3]) if top_shap_features else 'N/A'}\n"
        f"Timestamp  : {timestamp}\n"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = receiver
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body,  "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        logger.info(
            f"Alert sent → {receiver} | Pipeline: {pipeline_name} | Risk: {risk_score:.4f}"
        )
        return True
    except smtplib.SMTPAuthenticationError:
        logger.error(
            "SMTP authentication failed. "
            "Make sure you are using a Gmail App Password, not your account password."
        )
        return False
    except Exception as e:
        logger.error(f"Failed to send alert email: {e}")
        return False


def _get_risk_level(score: float) -> str:
    if score >= 0.90:
        return "CRITICAL"
    elif score >= 0.75:
        return "HIGH"
    elif score >= 0.50:
        return "MEDIUM"
    return "LOW"


def _format_shap(features: list) -> str:
    if not features:
        return "N/A"
    top3 = features[:3]
    items = "".join(
        f'<li style="margin:4px 0;">'
        f'<span style="color:#d32f2f;font-weight:bold;">{i+1}.</span> {f}'
        f'</li>'
        for i, f in enumerate(top3)
    )
    return f"<ol style='margin:0;padding-left:20px;'>{items}</ol>"
