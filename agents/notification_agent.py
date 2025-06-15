
import asyncio
import json
import logging
import smtplib
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import secrets
import aiohttp
import aiofiles
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import jinja2
from pathlib import Path

# LangGraph and LangSmith imports
from langgraph.graph import StateGraph, END
from langsmith import traceable, Client as LangSmithClient

# MCP imports
from mcp_client import MCPClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    MOBILE_PUSH = "mobile_push"


class NotificationPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    URGENT = "urgent"


class NotificationStatus(Enum):
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


class NotificationTemplate(Enum):
    COMPLIANCE_ALERT = "compliance_alert"
    DORMANCY_REPORT = "dormancy_report"
    RISK_WARNING = "risk_warning"
    WORKFLOW_COMPLETION = "workflow_completion"
    ERROR_NOTIFICATION = "error_notification"
    EXECUTIVE_SUMMARY = "executive_summary"
    ARTICLE_3_REMINDER = "article_3_reminder"
    HIGH_VALUE_ALERT = "high_value_alert"


@dataclass
class NotificationState:
    """Comprehensive state for notification workflow"""

    session_id: str
    user_id: str
    notification_id: str
    timestamp: datetime

    # Input data
    report_results: Optional[Dict] = None
    supervisor_decisions: Optional[Dict] = None
    alert_data: Optional[Dict] = None
    notification_config: Optional[Dict] = None

    # Notification processing
    notification_requests: List[Dict] = None
    delivery_results: Dict = None
    notification_results: Optional[Dict] = None

    # Status tracking
    notification_status: NotificationStatus = NotificationStatus.PENDING
    total_notifications: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0

    # Memory context
    memory_context: Dict = None
    retrieved_patterns: Dict = None
    user_preferences: Dict = None

    # Error handling and audit
    notification_log: List[Dict] = None
    error_log: List[Dict] = None
    delivery_tracking: Dict = None
    performance_metrics: Dict = None

    def __post_init__(self):
        if self.notification_requests is None:
            self.notification_requests = []
        if self.delivery_results is None:
            self.delivery_results = {}
        if self.memory_context is None:
            self.memory_context = {}
        if self.retrieved_patterns is None:
            self.retrieved_patterns = {}
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.notification_log is None:
            self.notification_log = []
        if self.error_log is None:
            self.error_log = []
        if self.delivery_tracking is None:
            self.delivery_tracking = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}


class NotificationChannelManager:
    """Manages different notification channels and their configurations"""

    def __init__(self, config: Dict):
        self.config = config
        self.email_client = None
        self.sms_client = None
        self.slack_client = None
        self.teams_client = None

        # Initialize template engine
        template_dir = Path(__file__).parent / "templates"
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

    async def initialize_clients(self):
        """Initialize notification channel clients"""

        try:
            # Initialize email client
            if self.config.get("email", {}).get("enabled"):
                self.email_client = EmailClient(self.config["email"])

            # Initialize SMS client
            if self.config.get("sms", {}).get("enabled"):
                self.sms_client = SMSClient(self.config["sms"])

            # Initialize Slack client
            if self.config.get("slack", {}).get("enabled"):
                self.slack_client = SlackClient(self.config["slack"])

            # Initialize Teams client
            if self.config.get("teams", {}).get("enabled"):
                self.teams_client = TeamsClient(self.config["teams"])

        except Exception as e:
            logger.error(f"Failed to initialize notification clients: {str(e)}")

    async def send_notification(self, channel: NotificationChannel, message_data: Dict) -> Dict:
        """Send notification through specified channel"""

        try:
            if channel == NotificationChannel.EMAIL and self.email_client:
                return await self.email_client.send_email(message_data)

            elif channel == NotificationChannel.SMS and self.sms_client:
                return await self.sms_client.send_sms(message_data)

            elif channel == NotificationChannel.SLACK and self.slack_client:
                return await self.slack_client.send_message(message_data)

            elif channel == NotificationChannel.TEAMS and self.teams_client:
                return await self.teams_client.send_message(message_data)

            elif channel == NotificationChannel.WEBHOOK:
                return await self._send_webhook(message_data)

            elif channel == NotificationChannel.DASHBOARD:
                return await self._send_dashboard_notification(message_data)

            else:
                return {
                    "success": False,
                    "error": f"Channel {channel.value} not available or not configured"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "channel": channel.value
            }

    async def _send_webhook(self, message_data: Dict) -> Dict:
        """Send webhook notification"""

        try:
            webhook_url = message_data.get("webhook_url")
            if not webhook_url:
                return {"success": False, "error": "Webhook URL not provided"}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                        webhook_url,
                        json=message_data.get("payload", {}),
                        headers=message_data.get("headers", {}),
                        timeout=30
                ) as response:
                    if response.status == 200:
                        return {
                            "success": True,
                            "status_code": response.status,
                            "response": await response.text()
                        }
                    else:
                        return {
                            "success": False,
                            "status_code": response.status,
                            "error": await response.text()
                        }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _send_dashboard_notification(self, message_data: Dict) -> Dict:
        """Send dashboard notification (in-app notification)"""

        try:
            # Store notification in user's dashboard queue
            dashboard_notification = {
                "id": secrets.token_hex(8),
                "user_id": message_data.get("user_id"),
                "title": message_data.get("title"),
                "message": message_data.get("message"),
                "priority": message_data.get("priority", "medium"),
                "timestamp": datetime.now().isoformat(),
                "read": False,
                "category": message_data.get("category", "general")
            }

            # In a real implementation, this would be stored in a database
            # For now, we'll return success
            return {
                "success": True,
                "notification_id": dashboard_notification["id"],
                "delivery_method": "dashboard"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class EmailClient:
    """Email notification client"""

    def __init__(self, config: Dict):
        self.smtp_server = config.get("smtp_server")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config.get("username")
        self.password = config.get("password")
        self.use_tls = config.get("use_tls", True)
        self.from_email = config.get("from_email")

    async def send_email(self, message_data: Dict) -> Dict:
        """Send email notification"""

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = message_data.get("subject")
            msg['From'] = self.from_email
            msg['To'] = ", ".join(message_data.get("recipients", []))

            # Add text content
            if message_data.get("text_content"):
                text_part = MIMEText(message_data["text_content"], 'plain')
                msg.attach(text_part)

            # Add HTML content
            if message_data.get("html_content"):
                html_part = MIMEText(message_data["html_content"], 'html')
                msg.attach(html_part)

            # Add attachments
            for attachment in message_data.get("attachments", []):
                await self._add_attachment(msg, attachment)

            # Send email
            context = ssl.create_default_context()

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls(context=context)
                server.login(self.username, self.password)
                server.send_message(msg)

            return {
                "success": True,
                "recipients": message_data.get("recipients"),
                "message_id": msg.get("Message-ID"),
                "delivery_method": "email"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "delivery_method": "email"
            }

    async def _add_attachment(self, msg: MIMEMultipart, attachment: Dict):
        """Add attachment to email message"""

        try:
            file_path = attachment.get("file_path")
            filename = attachment.get("filename")

            if file_path and Path(file_path).exists():
                async with aiofiles.open(file_path, "rb") as f:
                    file_data = await f.read()

                part = MIMEBase('application', 'octet-stream')
                part.set_payload(file_data)
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {filename or Path(file_path).name}'
                )
                msg.attach(part)

        except Exception as e:
            logger.warning(f"Failed to add attachment: {str(e)}")


class SMSClient:
    """SMS notification client"""

    def __init__(self, config: Dict):
        self.api_key = config.get("api_key")
        self.api_url = config.get("api_url")
        self.from_number = config.get("from_number")

    async def send_sms(self, message_data: Dict) -> Dict:
        """Send SMS notification"""

        try:
            payload = {
                "from": self.from_number,
                "to": message_data.get("phone_number"),
                "text": message_data.get("message")
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.api_url,
                        json=payload,
                        headers=headers,
                        timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "message_id": result.get("message_id"),
                            "delivery_method": "sms"
                        }
                    else:
                        return {
                            "success": False,
                            "status_code": response.status,
                            "error": await response.text(),
                            "delivery_method": "sms"
                        }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "delivery_method": "sms"
            }


class SlackClient:
    """Slack notification client"""

    def __init__(self, config: Dict):
        self.webhook_url = config.get("webhook_url")
        self.bot_token = config.get("bot_token")

    async def send_message(self, message_data: Dict) -> Dict:
        """Send Slack notification"""

        try:
            # Prepare Slack message format
            slack_message = {
                "text": message_data.get("title", "Banking Compliance Notification"),
                "attachments": [
                    {
                        "color": self._get_color_by_priority(message_data.get("priority", "medium")),
                        "fields": [
                            {
                                "title": "Message",
                                "value": message_data.get("message"),
                                "short": False
                            }
                        ],
                        "footer": "Banking Compliance System",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }

            # Add additional fields if provided
            if message_data.get("fields"):
                slack_message["attachments"][0]["fields"].extend(message_data["fields"])

            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.webhook_url,
                        json=slack_message,
                        timeout=30
                ) as response:
                    if response.status == 200:
                        return {
                            "success": True,
                            "delivery_method": "slack"
                        }
                    else:
                        return {
                            "success": False,
                            "status_code": response.status,
                            "error": await response.text(),
                            "delivery_method": "slack"
                        }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "delivery_method": "slack"
            }

    def _get_color_by_priority(self, priority: str) -> str:
        """Get Slack message color by priority"""

        color_map = {
            "low": "#36a64f",  # Green
            "medium": "#ff9500",  # Orange
            "high": "#ff0000",  # Red
            "critical": "#8b0000",  # Dark Red
            "urgent": "#ff1493"  # Deep Pink
        }

        return color_map.get(priority.lower(), "#36a64f")


class TeamsClient:
    """Microsoft Teams notification client"""

    def __init__(self, config: Dict):
        self.webhook_url = config.get("webhook_url")

    async def send_message(self, message_data: Dict) -> Dict:
        """Send Teams notification"""

        try:
            # Prepare Teams message format (Adaptive Card)
            teams_message = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": self._get_color_by_priority(message_data.get("priority", "medium")),
                "summary": message_data.get("title", "Banking Compliance Notification"),
                "sections": [
                    {
                        "activityTitle": message_data.get("title"),
                        "activitySubtitle": f"Priority: {message_data.get('priority', 'medium').upper()}",
                        "facts": [
                            {
                                "name": "Timestamp",
                                "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            },
                            {
                                "name": "User",
                                "value": message_data.get("user_id", "System")
                            }
                        ],
                        "markdown": True,
                        "text": message_data.get("message")
                    }
                ]
            }

            # Add action buttons if provided
            if message_data.get("actions"):
                teams_message["potentialAction"] = message_data["actions"]

            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.webhook_url,
                        json=teams_message,
                        timeout=30
                ) as response:
                    if response.status == 200:
                        return {
                            "success": True,
                            "delivery_method": "teams"
                        }
                    else:
                        return {
                            "success": False,
                            "status_code": response.status,
                            "error": await response.text(),
                            "delivery_method": "teams"
                        }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "delivery_method": "teams"
            }

    def _get_color_by_priority(self, priority: str) -> str:
        """Get Teams message color by priority"""

        color_map = {
            "low": "28a745",
            "medium": "ff9500",
            "high": "ff0000",
            "critical": "8b0000",
            "urgent": "ff1493"
        }

        return color_map.get(priority.lower(), "28a745")


class NotificationTemplateManager:
    """Manages notification templates and content generation"""

    def __init__(self, template_env: jinja2.Environment):
        self.template_env = template_env

        # Default templates
        self.default_templates = {
            NotificationTemplate.COMPLIANCE_ALERT: {
                "subject": "🚨 Banking Compliance Alert - Immediate Attention Required",
                "template": "compliance_alert.html"
            },
            NotificationTemplate.DORMANCY_REPORT: {
                "subject": "📊 Dormancy Analysis Report - {{report_date}}",
                "template": "dormancy_report.html"
            },
            NotificationTemplate.RISK_WARNING: {
                "subject": "⚠️ Risk Assessment Warning - {{risk_level}} Risk Detected",
                "template": "risk_warning.html"
            },
            NotificationTemplate.WORKFLOW_COMPLETION: {
                "subject": "✅ Banking Compliance Workflow Completed",
                "template": "workflow_completion.html"
            },
            NotificationTemplate.ERROR_NOTIFICATION: {
                "subject": "❌ System Error Notification - {{error_type}}",
                "template": "error_notification.html"
            },
            NotificationTemplate.EXECUTIVE_SUMMARY: {
                "subject": "📈 Executive Summary - Banking Compliance {{period}}",
                "template": "executive_summary.html"
            },
            NotificationTemplate.ARTICLE_3_REMINDER: {
                "subject": "📋 Article 3 Process Reminder - Action Required",
                "template": "article_3_reminder.html"
            },
            NotificationTemplate.HIGH_VALUE_ALERT: {
                "subject": "💰 High Value Dormant Account Alert",
                "template": "high_value_alert.html"
            }
        }

    def generate_notification_content(self, template_type: NotificationTemplate,
                                      data: Dict, format_type: str = "html") -> Dict:
        """Generate notification content from template"""

        try:
            template_config = self.default_templates.get(template_type)
            if not template_config:
                return self._generate_default_content(data, format_type)

            # Render subject
            subject_template = jinja2.Template(template_config["subject"])
            subject = subject_template.render(**data)

            # Render content
            if format_type == "html":
                try:
                    template = self.template_env.get_template(template_config["template"])
                    html_content = template.render(**data)
                    text_content = self._html_to_text(html_content)
                except jinja2.TemplateNotFound:
                    # Fallback to default template
                    html_content, text_content = self._generate_default_html_content(template_type, data)
            else:
                text_content = self._generate_text_content(template_type, data)
                html_content = self._text_to_html(text_content)

            return {
                "subject": subject,
                "html_content": html_content,
                "text_content": text_content,
                "template_used": template_config["template"]
            }

        except Exception as e:
            logger.error(f"Template generation failed: {str(e)}")
            return self._generate_default_content(data, format_type)

    def _generate_default_content(self, data: Dict, format_type: str) -> Dict:
        """Generate default notification content"""

        subject = f"Banking Compliance Notification - {datetime.now().strftime('%Y-%m-%d')}"

        if format_type == "html":
            html_content = f"""
            <html>
                <body>
                    <h2>Banking Compliance Notification</h2>
                    <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Message:</strong> {data.get('message', 'No message provided')}</p>
                    <hr>
                    <p><em>Generated by Banking Compliance System</em></p>
                </body>
            </html>
            """
            text_content = self._html_to_text(html_content)
        else:
            text_content = f"""
Banking Compliance Notification

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Message: {data.get('message', 'No message provided')}

---
Generated by Banking Compliance System
            """.strip()
            html_content = self._text_to_html(text_content)

        return {
            "subject": subject,
            "html_content": html_content,
            "text_content": text_content,
            "template_used": "default"
        }

    def _generate_default_html_content(self, template_type: NotificationTemplate, data: Dict) -> tuple:
        """Generate default HTML content for specific template types"""

        if template_type == NotificationTemplate.COMPLIANCE_ALERT:
            html_content = f"""
            <html>
                <body style="font-family: Arial, sans-serif;">
                    <div style="background-color: #ff4444; color: white; padding: 20px; border-radius: 5px;">
                        <h2>🚨 Compliance Alert</h2>
                    </div>
                    <div style="padding: 20px;">
                        <p><strong>Alert Type:</strong> {data.get('alert_type', 'General Compliance')}</p>
                        <p><strong>Severity:</strong> {data.get('severity', 'High')}</p>
                        <p><strong>Accounts Affected:</strong> {data.get('affected_accounts', 'N/A')}</p>
                        <p><strong>Description:</strong></p>
                        <p>{data.get('description', 'Compliance violation detected')}</p>
                        <p><strong>Required Action:</strong></p>
                        <p>{data.get('required_action', 'Review and remediate immediately')}</p>
                    </div>
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 20px;">
                        <p><em>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
                    </div>
                </body>
            </html>
            """

        elif template_type == NotificationTemplate.DORMANCY_REPORT:
            html_content = f"""
            <html>
                <body style="font-family: Arial, sans-serif;">
                    <div style="background-color: #007bff; color: white; padding: 20px; border-radius: 5px;">
                        <h2>📊 Dormancy Analysis Report</h2>
                    </div>
                    <div style="padding: 20px;">
                        <h3>Summary</h3>
                        <ul>
                            <li><strong>Total Accounts Analyzed:</strong> {data.get('total_accounts', 'N/A')}</li>
                            <li><strong>Dormant Accounts Found:</strong> {data.get('dormant_accounts', 'N/A')}</li>
                            <li><strong>High Risk Accounts:</strong> {data.get('high_risk_accounts', 'N/A')}</li>
                            <li><strong>Compliance Issues:</strong> {data.get('compliance_issues', 'N/A')}</li>
                        </ul>
                        <h3>Key Findings</h3>
                        <p>{data.get('key_findings', 'No significant findings')}</p>
                        <h3>Recommendations</h3>
                        <p>{data.get('recommendations', 'No specific recommendations')}</p>
                    </div>
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 20px;">
                        <p><em>Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
                    </div>
                </body>
            </html>
            """

        else:
            html_content = self._generate_default_content(data, "html")["html_content"]

        text_content = self._html_to_text(html_content)
        return html_content, text_content

    def _generate_text_content(self, template_type: NotificationTemplate, data: Dict) -> str:
        """Generate plain text content for specific template types"""

        if template_type == NotificationTemplate.COMPLIANCE_ALERT:
            return f"""
COMPLIANCE ALERT - IMMEDIATE ATTENTION REQUIRED

Alert Type: {data.get('alert_type', 'General Compliance')}
Severity: {data.get('severity', 'High')}
Accounts Affected: {data.get('affected_accounts', 'N/A')}

Description:
{data.get('description', 'Compliance violation detected')}

Required Action:
{data.get('required_action', 'Review and remediate immediately')}

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()

        elif template_type == NotificationTemplate.DORMANCY_REPORT:
            return f"""
DORMANCY ANALYSIS REPORT

Summary:
- Total Accounts Analyzed: {data.get('total_accounts', 'N/A')}
- Dormant Accounts Found: {data.get('dormant_accounts', 'N/A')}
- High Risk Accounts: {data.get('high_risk_accounts', 'N/A')}
- Compliance Issues: {data.get('compliance_issues', 'N/A')}

Key Findings:
{data.get('key_findings', 'No significant findings')}

Recommendations:
{data.get('recommendations', 'No specific recommendations')}

Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()

        else:
            return self._generate_default_content(data, "text")["text_content"]

    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML content to plain text"""

        try:
            # Simple HTML to text conversion
            import re
            # Remove HTML tags
            text = re.sub('<[^<]+?>', '', html_content)
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        except:
            return html_content

    def _text_to_html(self, text_content: str) -> str:
        """Convert plain text to basic HTML"""

        try:
            # Simple text to HTML conversion
            html = text_content.replace('\n', '<br>')
            return f"""
            <html>
                <body style="font-family: Arial, sans-serif;">
                    <div style="padding: 20px;">
                        {html}
                    </div>
                </body>
            </html>
            """
        except:
            return text_content


class NotificationAgent:
    """Enhanced notification agent with multi-channel delivery"""

    def __init__(self, memory_agent, mcp_client: MCPClient, db_session=None):
        self.memory_agent = memory_agent
        self.mcp_client = mcp_client
        self.db_session = db_session
        self.langsmith_client = LangSmithClient()

        # Load configuration
        self.notification_config = self._load_notification_config()

        # Initialize channel manager
        self.channel_manager = NotificationChannelManager(self.notification_config)

        # Initialize template manager
        template_dir = Path(__file__).parent / "templates"
        template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        self.template_manager = NotificationTemplateManager(template_env)

        # Delivery tracking
        self.delivery_history = {}

    def _load_notification_config(self) -> Dict:
        """Load notification configuration"""

        # Default configuration - would typically be loaded from config file or environment
        return {
            "email": {
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "banking-compliance@company.com",
                "password": "app_password",  # Use app password for production
                "use_tls": True,
                "from_email": "banking-compliance@company.com"
            },
            "sms": {
                "enabled": False,  # Configure with SMS provider
                "api_key": "sms_api_key",
                "api_url": "https://api.sms-provider.com/send",
                "from_number": "+1234567890"
            },
            "slack": {
                "enabled": True,
                "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                "bot_token": "xoxb-your-bot-token"
            },
            "teams": {
                "enabled": True,
                "webhook_url": "https://outlook.office.com/webhook/YOUR-TEAMS-WEBHOOK"
            },
            "default_channels": {
                "critical": [NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.SMS],
                "high": [NotificationChannel.EMAIL, NotificationChannel.SLACK],
                "medium": [NotificationChannel.EMAIL],
                "low": [NotificationChannel.DASHBOARD]
            },
            "retry_policy": {
                "max_retries": 3,
                "retry_delay": 60,  # seconds
                "exponential_backoff": True
            }
        }

    @traceable(name="notification_pre_hook")
    async def pre_notification_hook(self, state: NotificationState) -> NotificationState:
        """Enhanced pre-notification memory hook"""

        try:
            # Retrieve notification preferences
            user_preferences = await self.memory_agent.retrieve_memory(
                bucket="session",
                filter_criteria={
                    "type": "notification_preferences",
                    "user_id": state.user_id
                }
            )

            if user_preferences.get("success"):
                state.user_preferences = user_preferences.get("data", {})
                logger.info("Retrieved notification preferences from memory")

            # Retrieve notification patterns
            notification_patterns = await self.memory_agent.retrieve_memory(
                bucket="knowledge",
                filter_criteria={
                    "type": "notification_patterns",
                    "user_id": state.user_id
                }
            )

            if notification_patterns.get("success"):
                state.retrieved_patterns["notifications"] = notification_patterns.get("data", {})
                logger.info("Retrieved notification patterns from memory")

            # Retrieve delivery history for optimization
            delivery_history = await self.memory_agent.retrieve_memory(
                bucket="knowledge",
                filter_criteria={
                    "type": "delivery_history",
                    "user_id": state.user_id
                }
            )

            if delivery_history.get("success"):
                state.retrieved_patterns["delivery"] = delivery_history.get("data", {})

            # Log pre-hook execution
            state.notification_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "pre_notification_hook",
                "action": "memory_retrieval",
                "status": "completed",
                "preferences_loaded": len(state.user_preferences),
                "patterns_retrieved": len(state.retrieved_patterns)
            })

        except Exception as e:
            logger.error(f"Pre-notification hook failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "pre_notification_hook",
                "error": str(e)
            })

        return state

    @traceable(name="send_notifications")
    async def send_notifications(self, state: NotificationState) -> NotificationState:
        """Main notification sending workflow"""

        try:
            start_time = datetime.now()
            state.notification_status = NotificationStatus.SENDING

            # Initialize notification channel clients
            await self.channel_manager.initialize_clients()

            # Generate notification requests based on input data
            notification_requests = await self._generate_notification_requests(state)
            state.notification_requests = notification_requests
            state.total_notifications = len(notification_requests)

            # Process each notification request
            delivery_results = {}
            successful_count = 0
            failed_count = 0

            for request in notification_requests:
                try:
                    # Generate content
                    content = self.template_manager.generate_notification_content(
                        NotificationTemplate(request["template"]),
                        request["data"],
                        request.get("format", "html")
                    )

                    # Prepare message data
                    message_data = {
                        **request["message_data"],
                        **content
                    }

                    # Send through each channel
                    for channel in request["channels"]:
                        channel_enum = NotificationChannel(channel)
                        result = await self.channel_manager.send_notification(channel_enum, message_data)

                        delivery_key = f"{request['notification_id']}_{channel}"
                        delivery_results[delivery_key] = {
                            "notification_id": request["notification_id"],
                            "channel": channel,
                            "result": result,
                            "timestamp": datetime.now().isoformat()
                        }

                        if result.get("success"):
                            successful_count += 1
                        else:
                            failed_count += 1

                            # Retry failed notifications if configured
                            if self.notification_config.get("retry_policy", {}).get("max_retries", 0) > 0:
                                await self._schedule_retry(request, channel, result)

                except Exception as e:
                    failed_count += 1
                    delivery_results[f"{request['notification_id']}_error"] = {
                        "notification_id": request["notification_id"],
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }

                    state.error_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "stage": "notification_sending",
                        "notification_id": request["notification_id"],
                        "error": str(e)
                    })

            # Update state with results
            state.delivery_results = delivery_results
            state.successful_deliveries = successful_count
            state.failed_deliveries = failed_count
            state.notification_status = NotificationStatus.SENT if failed_count == 0 else NotificationStatus.FAILED

            # Call MCP tool for additional notification processing
            mcp_result = await self.mcp_client.call_tool("send_notification", {
                "notification_requests": notification_requests,
                "delivery_results": delivery_results,
                "user_id": state.user_id,
                "session_id": state.session_id
            })

            if mcp_result.get("success"):
                state.notification_results = mcp_result.get("data", {})

            # Calculate performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            state.performance_metrics = {
                "processing_time_seconds": processing_time,
                "notifications_per_second": state.total_notifications / processing_time if processing_time > 0 else 0,
                "success_rate": successful_count / state.total_notifications if state.total_notifications > 0 else 0,
                "failure_rate": failed_count / state.total_notifications if state.total_notifications > 0 else 0,
                "channel_performance": self._calculate_channel_performance(delivery_results)
            }

            # Log successful completion
            state.notification_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "notification_sending",
                "action": "send_notifications",
                "status": state.notification_status.value,
                "total_notifications": state.total_notifications,
                "successful_deliveries": successful_count,
                "failed_deliveries": failed_count,
                "processing_time": processing_time
            })

        except Exception as e:
            state.notification_status = NotificationStatus.FAILED
            error_msg = str(e)
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "notification_sending",
                "error": error_msg,
                "error_type": type(e).__name__
            })
            logger.error(f"Notification sending failed: {error_msg}")

        return state

    async def _generate_notification_requests(self, state: NotificationState) -> List[Dict]:
        """Generate notification requests based on state data"""

        requests = []

        try:
            # Process report results for notifications
            if state.report_results:
                requests.extend(await self._generate_report_notifications(state))

            # Process supervisor decisions for notifications
            if state.supervisor_decisions:
                requests.extend(await self._generate_supervisor_notifications(state))

            # Process alert data for notifications
            if state.alert_data:
                requests.extend(await self._generate_alert_notifications(state))

            # Apply user preferences and filtering
            requests = await self._apply_notification_preferences(requests, state.user_preferences)

        except Exception as e:
            logger.error(f"Failed to generate notification requests: {str(e)}")

        return requests

    async def _generate_report_notifications(self, state: NotificationState) -> List[Dict]:
        """Generate notifications based on report results"""

        notifications = []
        report_data = state.report_results

        if not report_data:
            return notifications

        # Executive summary notification
        if report_data.get("executive_summary"):
            notifications.append({
                "notification_id": secrets.token_hex(8),
                "template": NotificationTemplate.EXECUTIVE_SUMMARY.value,
                "priority": NotificationPriority.MEDIUM.value,
                "channels": self._get_channels_by_priority(NotificationPriority.MEDIUM),
                "data": {
                    "period": "Current Analysis",
                    "total_accounts": report_data["executive_summary"].get("total_accounts_analyzed", 0),
                    "dormant_accounts": report_data["executive_summary"].get("dormant_accounts_identified", 0),
                    "risk_score": report_data["executive_summary"].get("high_risk_accounts", 0),
                    "compliance_status": report_data["executive_summary"].get("compliance_status", "Unknown"),
                    "key_findings": self._extract_key_findings(report_data),
                    "recommendations": self._extract_recommendations(report_data),
                    "user_id": state.user_id
                },
                "message_data": {
                    "recipients": self._get_recipients_by_role("compliance"),
                    "user_id": state.user_id,
                    "priority": NotificationPriority.MEDIUM.value
                }
            })

        return notifications

    async def _generate_supervisor_notifications(self, state: NotificationState) -> List[Dict]:
        """Generate notifications based on supervisor decisions"""

        notifications = []
        supervisor_data = state.supervisor_decisions

        if not supervisor_data:
            return notifications

        # Critical decision notifications
        if supervisor_data.get("critical_decisions"):
            for decision in supervisor_data["critical_decisions"]:
                priority = NotificationPriority.CRITICAL if decision.get(
                    "severity") == "critical" else NotificationPriority.HIGH

                notifications.append({
                    "notification_id": secrets.token_hex(8),
                    "template": NotificationTemplate.COMPLIANCE_ALERT.value,
                    "priority": priority.value,
                    "channels": self._get_channels_by_priority(priority),
                    "data": {
                        "alert_type": decision.get("decision_type", "Supervisor Decision"),
                        "severity": decision.get("severity", "High"),
                        "affected_accounts": decision.get("affected_accounts", "Multiple"),
                        "description": decision.get("description", "Supervisor intervention required"),
                        "required_action": decision.get("required_action", "Review and take appropriate action"),
                        "deadline": decision.get("deadline", "Immediate"),
                        "user_id": state.user_id
                    },
                    "message_data": {
                        "recipients": self._get_recipients_by_role("supervisor"),
                        "user_id": state.user_id,
                        "priority": priority.value,
                        "category": "supervisor_decision"
                    }
                })

        # Escalation notifications
        if supervisor_data.get("escalations"):
            for escalation in supervisor_data["escalations"]:
                notifications.append({
                    "notification_id": secrets.token_hex(8),
                    "template": NotificationTemplate.COMPLIANCE_ALERT.value,
                    "priority": NotificationPriority.URGENT.value,
                    "channels": self._get_channels_by_priority(NotificationPriority.URGENT),
                    "data": {
                        "alert_type": "Escalation Required",
                        "severity": "Urgent",
                        "affected_accounts": escalation.get("account_count", "Unknown"),
                        "description": escalation.get("reason", "Manual escalation required"),
                        "required_action": "Immediate management review and intervention required",
                        "escalation_level": escalation.get("level", "Management"),
                        "user_id": state.user_id
                    },
                    "message_data": {
                        "recipients": self._get_recipients_by_role("management"),
                        "user_id": state.user_id,
                        "priority": NotificationPriority.URGENT.value,
                        "category": "escalation"
                    }
                })

        return notifications

    async def _generate_alert_notifications(self, state: NotificationState) -> List[Dict]:
        """Generate notifications based on alert data"""

        notifications = []
        alert_data = state.alert_data

        if not alert_data:
            return notifications

        # High value dormant account alerts
        if alert_data.get("high_value_dormant"):
            for account in alert_data["high_value_dormant"][:5]:  # Limit to top 5
                notifications.append({
                    "notification_id": secrets.token_hex(8),
                    "template": NotificationTemplate.HIGH_VALUE_ALERT.value,
                    "priority": NotificationPriority.HIGH.value,
                    "channels": self._get_channels_by_priority(NotificationPriority.HIGH),
                    "data": {
                        "account_id": account.get("account_id", "Unknown"),
                        "balance": account.get("balance", 0),
                        "dormancy_period": account.get("dormancy_days", 0),
                        "last_activity": account.get("last_activity", "Unknown"),
                        "account_type": account.get("account_type", "Unknown"),
                        "risk_factors": account.get("risk_factors", []),
                        "user_id": state.user_id
                    },
                    "message_data": {
                        "recipients": self._get_recipients_by_role("risk_management"),
                        "user_id": state.user_id,
                        "priority": NotificationPriority.HIGH.value,
                        "category": "high_value_alert"
                    }
                })

        # Article 3 process reminders
        if alert_data.get("article_3_required"):
            notifications.append({
                "notification_id": secrets.token_hex(8),
                "template": NotificationTemplate.ARTICLE_3_REMINDER.value,
                "priority": NotificationPriority.MEDIUM.value,
                "channels": self._get_channels_by_priority(NotificationPriority.MEDIUM),
                "data": {
                    "account_count": len(alert_data["article_3_required"]),
                    "accounts": alert_data["article_3_required"][:10],  # Sample accounts
                    "deadline": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                    "process_steps": [
                        "Contact customer using last known information",
                        "Wait 90 days for customer response",
                        "Transfer to dormant accounts ledger if no response",
                        "Continue monitoring for reactivation"
                    ],
                    "user_id": state.user_id
                },
                "message_data": {
                    "recipients": self._get_recipients_by_role("compliance"),
                    "user_id": state.user_id,
                    "priority": NotificationPriority.MEDIUM.value,
                    "category": "article_3_reminder"
                }
            })

        return notifications

    def _get_channels_by_priority(self, priority: NotificationPriority) -> List[str]:
        """Get notification channels based on priority level"""

        default_channels = self.notification_config.get("default_channels", {})
        channels = default_channels.get(priority.value, [NotificationChannel.EMAIL.value])

        # Convert enum values to strings
        return [channel.value if hasattr(channel, 'value') else str(channel) for channel in channels]

    def _get_recipients_by_role(self, role: str) -> List[str]:
        """Get notification recipients based on role"""

        # Default recipients mapping - would typically come from user management system
        role_recipients = {
            "executive": ["ceo@company.com", "cfo@company.com"],
            "compliance": ["compliance@company.com", "audit@company.com"],
            "risk_management": ["risk@company.com", "compliance@company.com"],
            "supervisor": ["supervisor@company.com", "manager@company.com"],
            "management": ["manager@company.com", "director@company.com"],
            "default": ["admin@company.com"]
        }

        return role_recipients.get(role, role_recipients["default"])

    async def _apply_notification_preferences(self, requests: List[Dict], preferences: Dict) -> List[Dict]:
        """Apply user notification preferences to filter and modify requests"""

        if not preferences:
            return requests

        filtered_requests = []

        for request in requests:
            # Check if notification type is enabled
            notification_type = request.get("template")
            if preferences.get("disabled_types") and notification_type in preferences["disabled_types"]:
                continue

            # Check priority threshold
            priority_threshold = preferences.get("priority_threshold", "low")
            request_priority = request.get("priority", "medium")

            priority_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4, "urgent": 5}
            if priority_levels.get(request_priority, 2) < priority_levels.get(priority_threshold, 1):
                continue

            # Modify channels based on preferences
            preferred_channels = preferences.get("preferred_channels", [])
            if preferred_channels:
                request["channels"] = [ch for ch in request["channels"] if ch in preferred_channels]

            # Apply quiet hours
            quiet_hours = preferences.get("quiet_hours")
            if quiet_hours and self._is_quiet_hours(quiet_hours):
                # Defer non-critical notifications
                if request_priority not in ["critical", "urgent"]:
                    request["defer_until"] = self._calculate_quiet_hours_end(quiet_hours)

            filtered_requests.append(request)

        return filtered_requests

    def _is_quiet_hours(self, quiet_hours: Dict) -> bool:
        """Check if current time is within quiet hours"""

        try:
            current_time = datetime.now().time()
            start_time = datetime.strptime(quiet_hours.get("start", "22:00"), "%H:%M").time()
            end_time = datetime.strptime(quiet_hours.get("end", "06:00"), "%H:%M").time()

            if start_time <= end_time:
                return start_time <= current_time <= end_time
            else:  # Overnight quiet hours
                return current_time >= start_time or current_time <= end_time

        except Exception:
            return False

    def _calculate_quiet_hours_end(self, quiet_hours: Dict) -> datetime:
        """Calculate when quiet hours end"""

        try:
            end_time = datetime.strptime(quiet_hours.get("end", "06:00"), "%H:%M").time()
            today = datetime.now().date()
            end_datetime = datetime.combine(today, end_time)

            # If end time is before current time, it's tomorrow
            if end_datetime <= datetime.now():
                end_datetime += timedelta(days=1)

            return end_datetime

        except Exception:
            return datetime.now() + timedelta(hours=8)  # Default 8 hours from now

    async def _schedule_retry(self, request: Dict, channel: str, failure_result: Dict):
        """Schedule retry for failed notification"""

        retry_policy = self.notification_config.get("retry_policy", {})
        max_retries = retry_policy.get("max_retries", 3)

        # Implementation would depend on task queue system (Celery, etc.)
        # For now, just log the retry attempt
        logger.info(f"Scheduling retry for notification {request['notification_id']} on channel {channel}")

    def _calculate_channel_performance(self, delivery_results: Dict) -> Dict:
        """Calculate performance metrics by channel"""

        channel_stats = {}

        for delivery_key, result in delivery_results.items():
            channel = result.get("channel")
            if not channel:
                continue

            if channel not in channel_stats:
                channel_stats[channel] = {"sent": 0, "delivered": 0, "failed": 0}

            if result.get("result", {}).get("success"):
                channel_stats[channel]["delivered"] += 1
            else:
                channel_stats[channel]["failed"] += 1

            channel_stats[channel]["sent"] += 1

        # Calculate success rates
        for channel, stats in channel_stats.items():
            total = stats["sent"]
            if total > 0:
                stats["success_rate"] = stats["delivered"] / total
                stats["failure_rate"] = stats["failed"] / total
            else:
                stats["success_rate"] = 0.0
                stats["failure_rate"] = 0.0

        return channel_stats

    def _extract_key_findings(self, report_data: Dict) -> str:
        """Extract key findings from report data"""

        findings = []

        if report_data.get("detailed_findings"):
            detailed = report_data["detailed_findings"]

            if detailed.get("dormancy_by_type"):
                total_dormant = sum(detailed["dormancy_by_type"].values())
                findings.append(f"Total dormant accounts identified: {total_dormant}")

            if detailed.get("risk_analysis"):
                high_risk = detailed["risk_analysis"].get("high_risk_count", 0)
                findings.append(f"High-risk accounts requiring attention: {high_risk}")

            if detailed.get("compliance_breakdown"):
                critical_items = detailed["compliance_breakdown"].get("critical_items", 0)
                findings.append(f"Critical compliance items: {critical_items}")

        return " | ".join(findings) if findings else "No significant findings identified"

    def _extract_recommendations(self, report_data: Dict) -> str:
        """Extract recommendations from report data"""

        recommendations = []

        if report_data.get("action_items"):
            for item in report_data["action_items"][:3]:  # Top 3 recommendations
                recommendations.append(f"• {item.get('action', 'Review required')}")

        if not recommendations:
            recommendations = [
                "• Review dormant account procedures",
                "• Ensure compliance with CBUAE regulations",
                "• Monitor high-risk accounts closely"
            ]

        return "\n".join(recommendations)

    @traceable(name="notification_post_hook")
    async def post_notification_hook(self, state: NotificationState) -> NotificationState:
        """Enhanced post-notification memory hook"""

        try:
            # Store notification session data
            session_data = {
                "session_id": state.session_id,
                "notification_id": state.notification_id,
                "user_id": state.user_id,
                "notification_results": {
                    "status": state.notification_status.value,
                    "total_notifications": state.total_notifications,
                    "successful_deliveries": state.successful_deliveries,
                    "failed_deliveries": state.failed_deliveries,
                    "processing_time": state.performance_metrics.get("processing_time_seconds", 0)
                },
                "delivery_summary": {
                    "channels_used": list(
                        set([r.get("channel") for r in state.delivery_results.values() if r.get("channel")])),
                    "success_rate": state.performance_metrics.get("success_rate", 0),
                    "channel_performance": state.performance_metrics.get("channel_performance", {})
                }
            }

            await self.memory_agent.store_memory(
                bucket="session",
                data=session_data,
                encrypt_sensitive=True
            )

            # Store notification patterns in knowledge memory
            if state.notification_status == NotificationStatus.SENT:
                pattern_data = {
                    "type": "notification_patterns",
                    "user_id": state.user_id,
                    "successful_patterns": {
                        "notification_types": [req.get("template") for req in state.notification_requests],
                        "channel_preferences": self._analyze_channel_effectiveness(state.delivery_results),
                        "timing_patterns": {
                            "sent_at": datetime.now().hour,
                            "day_of_week": datetime.now().weekday(),
                            "response_optimization": state.performance_metrics.get("notifications_per_second", 0)
                        },
                        "user_engagement": {
                            "preferred_templates": self._get_most_effective_templates(state),
                            "optimal_channels": self._get_best_performing_channels(state)
                        }
                    },
                    "performance_metrics": state.performance_metrics,
                    "timestamp": datetime.now().isoformat()
                }

                await self.memory_agent.store_memory(
                    bucket="knowledge",
                    data=pattern_data
                )

            # Store delivery history for future optimization
            delivery_history_data = {
                "type": "delivery_history",
                "user_id": state.user_id,
                "delivery_batch": {
                    "batch_id": state.notification_id,
                    "timestamp": datetime.now().isoformat(),
                    "deliveries": state.delivery_results,
                    "performance": state.performance_metrics,
                    "success_metrics": {
                        "total_sent": state.total_notifications,
                        "successful": state.successful_deliveries,
                        "failed": state.failed_deliveries,
                        "success_rate": state.performance_metrics.get("success_rate", 0)
                    }
                }
            }

            await self.memory_agent.store_memory(
                bucket="knowledge",
                data=delivery_history_data
            )

            # Log post-hook completion
            state.notification_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "post_notification_hook",
                "action": "memory_storage",
                "status": "completed",
                "session_data_stored": True,
                "pattern_data_stored": state.notification_status == NotificationStatus.SENT,
                "delivery_history_stored": True
            })

        except Exception as e:
            logger.error(f"Post-notification hook failed: {str(e)}")
            state.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "stage": "post_notification_hook",
                "error": str(e)
            })

        return state

    def _analyze_channel_effectiveness(self, delivery_results: Dict) -> Dict:
        """Analyze which channels performed best"""

        channel_effectiveness = {}

        for delivery_key, result in delivery_results.items():
            channel = result.get("channel")
            if not channel:
                continue

            if channel not in channel_effectiveness:
                channel_effectiveness[channel] = {"attempts": 0, "successes": 0}

            channel_effectiveness[channel]["attempts"] += 1
            if result.get("result", {}).get("success"):
                channel_effectiveness[channel]["successes"] += 1

        # Calculate effectiveness scores
        for channel, stats in channel_effectiveness.items():
            if stats["attempts"] > 0:
                stats["effectiveness_score"] = stats["successes"] / stats["attempts"]
            else:
                stats["effectiveness_score"] = 0.0

        return channel_effectiveness

    def _get_most_effective_templates(self, state: NotificationState) -> List[str]:
        """Get templates that had the highest success rates"""

        template_success = {}

        for request in state.notification_requests:
            template = request.get("template")
            notification_id = request.get("notification_id")

            # Count successes for this template
            successes = sum(1 for k, v in state.delivery_results.items()
                            if k.startswith(notification_id) and v.get("result", {}).get("success"))
            attempts = sum(1 for k, v in state.delivery_results.items()
                           if k.startswith(notification_id))

            if template not in template_success:
                template_success[template] = {"successes": 0, "attempts": 0}

            template_success[template]["successes"] += successes
            template_success[template]["attempts"] += attempts

        # Sort by success rate
        effective_templates = []
        for template, stats in template_success.items():
            if stats["attempts"] > 0:
                success_rate = stats["successes"] / stats["attempts"]
                effective_templates.append((template, success_rate))

        effective_templates.sort(key=lambda x: x[1], reverse=True)
        return [template for template, _ in effective_templates[:5]]

    def _get_best_performing_channels(self, state: NotificationState) -> List[str]:
        """Get channels with the highest success rates"""

        channel_performance = state.performance_metrics.get("channel_performance", {})

        performing_channels = []
        for channel, stats in channel_performance.items():
            success_rate = stats.get("success_rate", 0)
            performing_channels.append((channel, success_rate))

        performing_channels.sort(key=lambda x: x[1], reverse=True)
        return [channel for channel, _ in performing_channels[:3]]

    @traceable(name="execute_notification_workflow")
    async def execute_workflow(self, user_id: str, notification_data: Dict,
                               notification_options: Dict = None) -> Dict:
        """Execute complete notification workflow"""

        try:
            # Initialize notification state
            notification_id = secrets.token_hex(16)
            session_id = secrets.token_hex(16)

            state = NotificationState(
                session_id=session_id,
                user_id=user_id,
                notification_id=notification_id,
                timestamp=datetime.now(),
                report_results=notification_data.get("report_results"),
                supervisor_decisions=notification_data.get("supervisor_decisions"),
                alert_data=notification_data.get("alert_data"),
                notification_config=notification_options or {}
            )

            # Execute workflow stages
            state = await self.pre_notification_hook(state)
            state = await self.send_notifications(state)
            state = await self.post_notification_hook(state)

            # Prepare response
            response = {
                "success": state.notification_status in [NotificationStatus.SENT, NotificationStatus.DELIVERED],
                "notification_id": notification_id,
                "session_id": session_id,
                "status": state.notification_status.value,
                "total_notifications": state.total_notifications,
                "successful_deliveries": state.successful_deliveries,
                "failed_deliveries": state.failed_deliveries,
                "processing_time": state.performance_metrics.get("processing_time_seconds", 0),
                "delivery_results": state.delivery_results,
                "performance_metrics": state.performance_metrics,
                "notification_log": state.notification_log[-10:],  # Last 10 entries
                "channels_used": list(
                    set([r.get("channel") for r in state.delivery_results.values() if r.get("channel")]))
            }

            if state.notification_status == NotificationStatus.FAILED:
                response["errors"] = state.error_log

            return response

        except Exception as e:
            logger.error(f"Notification workflow failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }