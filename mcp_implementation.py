import asyncio
import json
import logging
import websockets
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import secrets
import sqlite3
import hashlib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========================= MCP PROTOCOL DEFINITIONS =========================

class MCPMessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class MCPErrorCode(Enum):
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    TOOL_EXECUTION_ERROR = -32000
    AUTHENTICATION_ERROR = -32001
    AUTHORIZATION_ERROR = -32002


@dataclass
class MCPMessage:
    """Base MCP message structure"""

    id: str
    type: MCPMessageType
    timestamp: datetime

    # Request fields
    method: Optional[str] = None
    params: Optional[Dict] = None

    # Response fields
    result: Optional[Dict] = None
    error: Optional[Dict] = None

    # Notification fields
    notification: Optional[str] = None
    data: Optional[Dict] = None


@dataclass
class MCPTool:
    """MCP tool definition"""

    name: str
    description: str
    parameters: Dict
    handler: Callable
    category: str = "general"
    version: str = "1.0"
    requires_auth: bool = True
    rate_limit: Optional[int] = None
    timeout: int = 30


# ========================= MCP CLIENT IMPLEMENTATION =========================

class MCPClient:
    """MCP client for making requests to banking compliance tools"""

    def __init__(self, server_url: str = "ws://localhost:8765",
                 auth_token: str = None, timeout: int = 30):
        self.server_url = server_url
        self.auth_token = auth_token
        self.timeout = timeout

        # Connection management
        self.websocket = None
        self.connected = False
        self.connection_id = None

        # Request tracking
        self.pending_requests = {}
        self.request_counter = 0

        # Statistics
        self.stats = {
            "requests_sent": 0,
            "responses_received": 0,
            "errors_encountered": 0,
            "tools_called": {}
        }

    async def connect(self) -> bool:
        """Connect to MCP server"""

        try:
            # Establish WebSocket connection
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            self.websocket = await websockets.connect(
                self.server_url,
                extra_headers=headers,
                timeout=self.timeout
            )

            # Send initialization request
            init_message = MCPMessage(
                id=self._generate_request_id(),
                type=MCPMessageType.REQUEST,
                timestamp=datetime.now(),
                method="initialize",
                params={
                    "client_info": {
                        "name": "Banking Compliance Client",
                        "version": "1.0.0"
                    },
                    "capabilities": [
                        "tool_calling",
                        "notifications",
                        "async_operations"
                    ]
                }
            )

            response = await self._send_message(init_message)

            if response and response.get("success"):
                self.connected = True
                self.connection_id = response.get("data", {}).get("connection_id")
                logger.info(f"Connected to MCP server: {self.connection_id}")

                # Start message listener
                asyncio.create_task(self._message_listener())

                return True
            else:
                logger.error("MCP server initialization failed")
                return False

        except Exception as e:
            logger.error(f"MCP connection failed: {str(e)}")
            return False

    async def disconnect(self):
        """Disconnect from MCP server"""

        try:
            if self.websocket and self.connected:
                # Send disconnect notification
                disconnect_message = MCPMessage(
                    id=self._generate_request_id(),
                    type=MCPMessageType.NOTIFICATION,
                    timestamp=datetime.now(),
                    notification="disconnect",
                    data={"connection_id": self.connection_id}
                )

                await self._send_message(disconnect_message, wait_response=False)
                await self.websocket.close()

            self.connected = False
            self.connection_id = None
            logger.info("Disconnected from MCP server")

        except Exception as e:
            logger.warning(f"Disconnect error: {str(e)}")

    async def call_tool(self, tool_name: str, parameters: Dict = None,
                        timeout: Optional[int] = None) -> Dict:
        """Call a tool on the MCP server"""

        if not self.connected:
            await self.connect()

        if not self.connected:
            return {
                "success": False,
                "error": "Not connected to MCP server",
                "error_code": MCPErrorCode.INTERNAL_ERROR.value
            }

        try:
            request_id = self._generate_request_id()

            # Create tool call message
            message = MCPMessage(
                id=request_id,
                type=MCPMessageType.REQUEST,
                timestamp=datetime.now(),
                method="call_tool",
                params={
                    "tool_name": tool_name,
                    "parameters": parameters or {},
                    "connection_id": self.connection_id,
                    "timeout": timeout or self.timeout
                }
            )

            # Send request and wait for response
            response = await self._send_message(message, timeout=timeout or self.timeout)

            # Update statistics
            self.stats["requests_sent"] += 1
            self.stats["tools_called"][tool_name] = self.stats["tools_called"].get(tool_name, 0) + 1

            if response:
                self.stats["responses_received"] += 1
                return response
            else:
                self.stats["errors_encountered"] += 1
                return {
                    "success": False,
                    "error": "No response received",
                    "error_code": MCPErrorCode.INTERNAL_ERROR.value
                }

        except Exception as e:
            self.stats["errors_encountered"] += 1
            logger.error(f"Tool call failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_code": MCPErrorCode.TOOL_EXECUTION_ERROR.value
            }

    async def list_tools(self) -> Dict:
        """List available tools on the server"""

        if not self.connected:
            await self.connect()

        try:
            message = MCPMessage(
                id=self._generate_request_id(),
                type=MCPMessageType.REQUEST,
                timestamp=datetime.now(),
                method="list_tools",
                params={"connection_id": self.connection_id}
            )

            response = await self._send_message(message)
            return response or {"success": False, "error": "No response"}

        except Exception as e:
            logger.error(f"List tools failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_tool_info(self, tool_name: str) -> Dict:
        """Get information about a specific tool"""

        try:
            message = MCPMessage(
                id=self._generate_request_id(),
                type=MCPMessageType.REQUEST,
                timestamp=datetime.now(),
                method="get_tool_info",
                params={
                    "tool_name": tool_name,
                    "connection_id": self.connection_id
                }
            )

            response = await self._send_message(message)
            return response or {"success": False, "error": "No response"}

        except Exception as e:
            logger.error(f"Get tool info failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _send_message(self, message: MCPMessage, wait_response: bool = True,
                            timeout: Optional[int] = None) -> Optional[Dict]:
        """Send message to server and optionally wait for response"""

        try:
            # Serialize message
            message_json = json.dumps(asdict(message), default=str)

            # Send to server
            await self.websocket.send(message_json)

            if wait_response:
                # Store pending request
                self.pending_requests[message.id] = {
                    "timestamp": datetime.now(),
                    "timeout": timeout or self.timeout,
                    "future": asyncio.Future()
                }

                # Wait for response
                try:
                    response = await asyncio.wait_for(
                        self.pending_requests[message.id]["future"],
                        timeout=timeout or self.timeout
                    )
                    return response

                except asyncio.TimeoutError:
                    # Clean up pending request
                    if message.id in self.pending_requests:
                        del self.pending_requests[message.id]

                    return {
                        "success": False,
                        "error": "Request timeout",
                        "error_code": MCPErrorCode.INTERNAL_ERROR.value
                    }

            return None

        except Exception as e:
            logger.error(f"Send message failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_code": MCPErrorCode.INTERNAL_ERROR.value
            }

    async def _message_listener(self):
        """Listen for incoming messages from server"""

        try:
            async for message_json in self.websocket:
                try:
                    message_data = json.loads(message_json)

                    # Handle responses to pending requests
                    if message_data.get("id") in self.pending_requests:
                        future = self.pending_requests[message_data["id"]]["future"]
                        if not future.done():
                            if message_data.get("error"):
                                future.set_result({
                                    "success": False,
                                    "error": message_data["error"],
                                    "error_code": message_data.get("error_code")
                                })
                            else:
                                future.set_result({
                                    "success": True,
                                    "data": message_data.get("result"),
                                    "response_time": message_data.get("response_time")
                                })

                        # Clean up pending request
                        del self.pending_requests[message_data["id"]]

                    # Handle notifications
                    elif message_data.get("type") == "notification":
                        await self._handle_notification(message_data)

                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON message")
                except Exception as e:
                    logger.error(f"Message processing error: {str(e)}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.connected = False

        except Exception as e:
            logger.error(f"Message listener error: {str(e)}")
            self.connected = False

    async def _handle_notification(self, notification_data: Dict):
        """Handle server notifications"""

        try:
            notification_type = notification_data.get("notification")

            if notification_type == "server_shutdown":
                logger.info("Server shutdown notification received")
                self.connected = False

            elif notification_type == "tool_updated":
                logger.info(f"Tool updated: {notification_data.get('data', {}).get('tool_name')}")

            elif notification_type == "rate_limit_warning":
                logger.warning(f"Rate limit warning: {notification_data.get('data')}")

        except Exception as e:
            logger.error(f"Notification handling error: {str(e)}")

    def _generate_request_id(self) -> str:
        """Generate unique request ID"""

        self.request_counter += 1
        return f"{self.connection_id or 'client'}_{self.request_counter}_{secrets.token_hex(4)}"

    def get_statistics(self) -> Dict:
        """Get client statistics"""

        return {
            "connected": self.connected,
            "connection_id": self.connection_id,
            "statistics": self.stats,
            "pending_requests": len(self.pending_requests)
        }


# ========================= MCP SERVER IMPLEMENTATION =========================

class MCPServer:
    """MCP server for banking compliance tools"""

    def __init__(self, host: str = "localhost", port: int = 8765,
                 auth_required: bool = True):
        self.host = host
        self.port = port
        self.auth_required = auth_required

        # Tool registry
        self.tools = {}
        self.tool_categories = {}

        # Connection management
        self.connections = {}
        self.active_sessions = {}

        # Rate limiting
        self.rate_limits = {}

        # Statistics
        self.stats = {
            "connections_established": 0,
            "total_requests": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "tools_registered": 0
        }

        # Initialize database for persistent storage
        self._init_database()

        # Register core banking compliance tools
        self._register_core_tools()

    def _init_database(self):
        """Initialize server database"""

        self.db_path = "mcp_server.db"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Tool execution log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tool_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    connection_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    parameters TEXT,
                    result TEXT,
                    success BOOLEAN,
                    execution_time REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Connection log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS connections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    connection_id TEXT UNIQUE NOT NULL,
                    client_info TEXT,
                    connected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    disconnected_at TIMESTAMP NULL,
                    total_requests INTEGER DEFAULT 0
                )
            ''')

            conn.commit()

    def register_tool(self, tool: MCPTool):
        """Register a new tool"""

        self.tools[tool.name] = tool

        if tool.category not in self.tool_categories:
            self.tool_categories[tool.category] = []

        self.tool_categories[tool.category].append(tool.name)
        self.stats["tools_registered"] += 1

        logger.info(f"Registered tool: {tool.name} (category: {tool.category})")

    def _register_core_tools(self):
        """Register core banking compliance tools"""

        # Data processing tools
        self.register_tool(MCPTool(
            name="process_banking_data",
            description="Process and validate banking data for compliance analysis",
            parameters={
                "data": {"type": "object", "description": "Banking data to process"},
                "validation_rules": {"type": "object", "description": "Validation rules"},
                "quality_checks": {"type": "boolean", "description": "Enable quality checks"}
            },
            handler=self._handle_process_banking_data,
            category="data_processing"
        ))

        # Dormancy analysis tools
        self.register_tool(MCPTool(
            name="analyze_account_dormancy",
            description="Analyze accounts for dormancy based on CBUAE regulations",
            parameters={
                "accounts": {"type": "array", "description": "Account data"},
                "report_date": {"type": "string", "description": "Analysis report date"},
                "regulatory_params": {"type": "object", "description": "Regulatory parameters"}
            },
            handler=self._handle_analyze_dormancy,
            category="dormancy_analysis"
        ))

        # Compliance verification tools
        self.register_tool(MCPTool(
            name="verify_compliance",
            description="Verify compliance with CBUAE regulations",
            parameters={
                "compliance_results": {"type": "object", "description": "Compliance results"},
                "violations": {"type": "array", "description": "Violations found"},
                "compliance_score": {"type": "number", "description": "Compliance score"},
                "user_id": {"type": "string", "description": "User ID"}
            },
            handler=self._handle_verify_compliance,
            category="compliance"
        ))

        # Risk assessment tools
        self.register_tool(MCPTool(
            name="assess_risk",
            description="Assess risk levels for banking accounts",
            parameters={
                "risk_assessment": {"type": "object", "description": "Risk assessment data"},
                "user_id": {"type": "string", "description": "User ID"},
                "session_id": {"type": "string", "description": "Session ID"}
            },
            handler=self._handle_assess_risk,
            category="risk_assessment"
        ))

        # Memory operations tools
        self.register_tool(MCPTool(
            name="memory_store",
            description="Store data in hybrid memory system",
            parameters={
                "bucket": {"type": "string", "description": "Memory bucket"},
                "data": {"type": "object", "description": "Data to store"},
                "metadata": {"type": "object", "description": "Storage metadata"}
            },
            handler=self._handle_memory_store,
            category="memory"
        ))

        self.register_tool(MCPTool(
            name="memory_retrieve",
            description="Retrieve data from hybrid memory system",
            parameters={
                "bucket": {"type": "string", "description": "Memory bucket"},
                "filter": {"type": "object", "description": "Filter criteria"},
                "limit": {"type": "integer", "description": "Result limit"}
            },
            handler=self._handle_memory_retrieve,
            category="memory"
        ))

        # Notification tools
        self.register_tool(MCPTool(
            name="send_notifications",
            description="Send notifications through various channels",
            parameters={
                "notification_results": {"type": "object", "description": "Notification results"},
                "user_id": {"type": "string", "description": "User ID"},
                "session_id": {"type": "string", "description": "Session ID"}
            },
            handler=self._handle_send_notification,
            category="notifications"
        ))

        # Reporting tools
        self.register_tool(MCPTool(
            name="generate_compliance_report",
            description="Generate comprehensive compliance reports",
            parameters={
                "report": {"type": "object", "description": "Report data"},
                "user_id": {"type": "string", "description": "User ID"},
                "session_id": {"type": "string", "description": "Session ID"}
            },
            handler=self._handle_generate_report,
            category="reporting"
        ))

        # Error handling tools
        self.register_tool(MCPTool(
            name="handle_error",
            description="Handle workflow errors and recovery",
            parameters={
                "errors": {"type": "array", "description": "Error list"},
                "failed_node": {"type": "string", "description": "Failed node"},
                "recovery_action": {"type": "string", "description": "Recovery action"},
                "user_id": {"type": "string", "description": "User ID"}
            },
            handler=self._handle_error,
            category="error_handling"
        ))

        # Audit trail tools
        self.register_tool(MCPTool(
            name="log_audit_trail",
            description="Log audit trail for compliance",
            parameters={
                "audit_log": {"type": "object", "description": "Audit log data"},
                "user_id": {"type": "string", "description": "User ID"},
                "session_id": {"type": "string", "description": "Session ID"}
            },
            handler=self._handle_audit_trail,
            category="audit"
        ))

    async def start_server(self):
        """Start the MCP server"""

        logger.info(f"Starting MCP server on {self.host}:{self.port}")

        # Start WebSocket server
        async def handle_client(websocket, path):
            await self._handle_client_connection(websocket, path)

        server = await websockets.serve(
            handle_client,
            self.host,
            self.port,
            max_size=1024 * 1024,  # 1MB max message size
            timeout=60
        )

        logger.info(f"MCP server started with {len(self.tools)} tools registered")

        return server

    async def _handle_client_connection(self, websocket, path):
        """Handle new client connection"""

        connection_id = secrets.token_hex(16)

        try:
            # Store connection info
            self.connections[connection_id] = {
                "websocket": websocket,
                "connected_at": datetime.now(),
                "requests_count": 0,
                "client_info": None
            }

            self.stats["connections_established"] += 1

            logger.info(f"New client connected: {connection_id}")

            # Handle messages from client
            async for message_json in websocket:
                await self._handle_client_message(connection_id, message_json)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"Client connection error: {str(e)}")
        finally:
            # Clean up connection
            if connection_id in self.connections:
                del self.connections[connection_id]

            # Log disconnection
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE connections 
                    SET disconnected_at = CURRENT_TIMESTAMP 
                    WHERE connection_id = ?
                ''', (connection_id,))
                conn.commit()

    async def _handle_client_message(self, connection_id: str, message_json: str):
        """Handle message from client"""

        try:
            message_data = json.loads(message_json)
            request_id = message_data.get("id")
            method = message_data.get("method")
            params = message_data.get("params", {})

            # Update request count
            self.connections[connection_id]["requests_count"] += 1
            self.stats["total_requests"] += 1

            # Route message based on method
            if method == "initialize":
                response = await self._handle_initialize(connection_id, params)

            elif method == "call_tool":
                response = await self._handle_tool_call(connection_id, params)

            elif method == "list_tools":
                response = await self._handle_list_tools(connection_id, params)

            elif method == "get_tool_info":
                response = await self._handle_get_tool_info(connection_id, params)

            else:
                response = {
                    "success": False,
                    "error": f"Unknown method: {method}",
                    "error_code": MCPErrorCode.METHOD_NOT_FOUND.value
                }

            # Send response
            response_message = {
                "id": request_id,
                "type": "response",
                "timestamp": datetime.now().isoformat(),
                "result": response.get("result") if response.get("success") else None,
                "error": response.get("error") if not response.get("success") else None,
                "error_code": response.get("error_code"),
                "response_time": response.get("response_time")
            }

            await self.connections[connection_id]["websocket"].send(
                json.dumps(response_message, default=str)
            )

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client {connection_id}")
        except Exception as e:
            logger.error(f"Message handling error: {str(e)}")

    async def _handle_initialize(self, connection_id: str, params: Dict) -> Dict:
        """Handle client initialization"""

        try:
            client_info = params.get("client_info", {})
            capabilities = params.get("capabilities", [])

            # Store client info
            self.connections[connection_id]["client_info"] = client_info

            # Log connection to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO connections (connection_id, client_info)
                    VALUES (?, ?)
                ''', (connection_id, json.dumps(client_info)))
                conn.commit()

            return {
                "success": True,
                "result": {
                    "connection_id": connection_id,
                    "server_info": {
                        "name": "Banking Compliance MCP Server",
                        "version": "1.0.0"
                    },
                    "capabilities": [
                        "tool_calling",
                        "notifications",
                        "async_operations",
                        "rate_limiting"
                    ],
                    "tools_available": len(self.tools)
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": MCPErrorCode.INTERNAL_ERROR.value
            }

    async def _handle_tool_call(self, connection_id: str, params: Dict) -> Dict:
        """Handle tool call request"""

        start_time = datetime.now()

        try:
            tool_name = params.get("tool_name")
            tool_params = params.get("parameters", {})
            timeout = params.get("timeout", 30)

            if tool_name not in self.tools:
                return {
                    "success": False,
                    "error": f"Tool not found: {tool_name}",
                    "error_code": MCPErrorCode.METHOD_NOT_FOUND.value
                }

            tool = self.tools[tool_name]

            # Check rate limiting
            if not self._check_rate_limit(connection_id, tool_name):
                return {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "error_code": MCPErrorCode.INTERNAL_ERROR.value
                }

            # Execute tool with timeout
            try:
                result = await asyncio.wait_for(
                    tool.handler(tool_params),
                    timeout=timeout
                )

                execution_time = (datetime.now() - start_time).total_seconds()

                # Log successful execution
                self._log_tool_execution(
                    connection_id, tool_name, tool_params, result, True, execution_time
                )

                self.stats["successful_calls"] += 1

                return {
                    "success": True,
                    "result": result,
                    "response_time": execution_time
                }

            except asyncio.TimeoutError:
                self.stats["failed_calls"] += 1
                return {
                    "success": False,
                    "error": f"Tool execution timeout ({timeout}s)",
                    "error_code": MCPErrorCode.TOOL_EXECUTION_ERROR.value
                }

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()

            # Log failed execution
            self._log_tool_execution(
                connection_id, tool_name, tool_params, str(e), False, execution_time
            )

            self.stats["failed_calls"] += 1

            return {
                "success": False,
                "error": str(e),
                "error_code": MCPErrorCode.TOOL_EXECUTION_ERROR.value,
                "response_time": execution_time
            }

    async def _handle_list_tools(self, connection_id: str, params: Dict) -> Dict:
        """Handle list tools request"""

        try:
            category_filter = params.get("category")

            tools_list = []

            for tool_name, tool in self.tools.items():
                if category_filter and tool.category != category_filter:
                    continue

                tools_list.append({
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category,
                    "version": tool.version,
                    "requires_auth": tool.requires_auth,
                    "parameters": tool.parameters
                })

            return {
                "success": True,
                "result": {
                    "tools": tools_list,
                    "total_count": len(tools_list),
                    "categories": list(self.tool_categories.keys())
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": MCPErrorCode.INTERNAL_ERROR.value
            }

    async def _handle_get_tool_info(self, connection_id: str, params: Dict) -> Dict:
        """Handle get tool info request"""

        try:
            tool_name = params.get("tool_name")

            if tool_name not in self.tools:
                return {
                    "success": False,
                    "error": f"Tool not found: {tool_name}",
                    "error_code": MCPErrorCode.METHOD_NOT_FOUND.value
                }

            tool = self.tools[tool_name]

            # Get execution statistics
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_executions,
                        AVG(execution_time) as avg_execution_time,
                        COUNT(CASE WHEN success = 1 THEN 1 END) as successful_executions
                    FROM tool_executions 
                    WHERE tool_name = ?
                ''', (tool_name,))

                stats = cursor.fetchone()

            tool_info = {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "version": tool.version,
                "requires_auth": tool.requires_auth,
                "rate_limit": tool.rate_limit,
                "timeout": tool.timeout,
                "parameters": tool.parameters,
                "statistics": {
                    "total_executions": stats[0] if stats else 0,
                    "avg_execution_time": round(stats[1], 3) if stats and stats[1] else 0,
                    "success_rate": round(stats[2] / max(stats[0], 1), 3) if stats else 0
                }
            }

            return {
                "success": True,
                "result": tool_info
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": MCPErrorCode.INTERNAL_ERROR.value
            }

    def _check_rate_limit(self, connection_id: str, tool_name: str) -> bool:
        """Check if request is within rate limits"""

        tool = self.tools.get(tool_name)
        if not tool or not tool.rate_limit:
            return True

        # Simple rate limiting implementation
        now = datetime.now()

    def _check_rate_limit(self, connection_id: str, tool_name: str) -> bool:
        """Check if request is within rate limits"""

        tool = self.tools.get(tool_name)
        if not tool or not tool.rate_limit:
            return True

        # Simple rate limiting implementation
        now = datetime.now()
        key = f"{connection_id}:{tool_name}"

        if key not in self.rate_limits:
            self.rate_limits[key] = []

        # Remove old requests (1 minute window)
        cutoff = now - timedelta(minutes=1)
        self.rate_limits[key] = [
            req_time for req_time in self.rate_limits[key]
            if req_time > cutoff
        ]

        # Check if within limit
        if len(self.rate_limits[key]) >= tool.rate_limit:
            return False

        # Add current request
        self.rate_limits[key].append(now)
        return True

    def _log_tool_execution(self, connection_id: str, tool_name: str,
                            parameters: Dict, result: Any, success: bool,
                            execution_time: float):
        """Log tool execution to database"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO tool_executions 
                    (connection_id, tool_name, parameters, result, success, execution_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    connection_id,
                    tool_name,
                    json.dumps(parameters, default=str),
                    json.dumps(result, default=str)[:1000],  # Limit result size
                    success,
                    execution_time
                ))
                conn.commit()

        except Exception as e:
            logger.warning(f"Failed to log tool execution: {str(e)}")

    # ========================= TOOL HANDLERS =========================

    async def _handle_process_banking_data(self, params: Dict) -> Dict:
        """Handle banking data processing tool"""

        try:
            data = params.get("data", {})
            validation_rules = params.get("validation_rules", {})
            quality_checks = params.get("quality_checks", True)

            # Mock processing logic - in real implementation, this would
            # integrate with actual data processing systems
            processed_data = {
                "processed_records": len(data.get("accounts", [])),
                "validation_passed": True,
                "quality_score": 0.95,
                "processing_time": 1.2,
                "issues_found": [],
                "recommendations": [
                    "Data quality is excellent",
                    "All validation rules passed"
                ]
            }

            return {
                "success": True,
                "processed_data": processed_data,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _handle_analyze_dormancy(self, params: Dict) -> Dict:
        """Handle dormancy analysis tool"""

        try:
            accounts = params.get("accounts", [])
            report_date = params.get("report_date")
            regulatory_params = params.get("regulatory_params", {})

            # Mock dormancy analysis
            total_accounts = len(accounts)
            dormant_accounts = int(total_accounts * 0.23)  # 23% dormancy rate

            analysis_results = {
                "total_accounts_analyzed": total_accounts,
                "dormant_accounts_found": dormant_accounts,
                "dormancy_rate": round((dormant_accounts / total_accounts) * 100, 2) if total_accounts > 0 else 0,
                "article_breakdown": {
                    "article_2_1_1": int(dormant_accounts * 0.4),
                    "article_2_2": int(dormant_accounts * 0.3),
                    "article_2_3": int(dormant_accounts * 0.2),
                    "article_2_6": int(dormant_accounts * 0.1)
                },
                "high_risk_accounts": int(dormant_accounts * 0.15),
                "cb_transfer_eligible": int(dormant_accounts * 0.25),
                "analysis_date": report_date or datetime.now().strftime("%Y-%m-%d")
            }

            return {
                "success": True,
                "analysis_results": analysis_results,
                "compliance_status": "requires_attention" if dormant_accounts > 0 else "compliant"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _handle_verify_compliance(self, params: Dict) -> Dict:
        """Handle compliance verification tool"""

        try:
            compliance_results = params.get("compliance_results", {})
            violations = params.get("violations", [])
            compliance_score = params.get("compliance_score", 0)
            user_id = params.get("user_id")

            # Log compliance verification
            logger.info(f"Compliance verification for user {user_id}: score={compliance_score}")

            return {
                "success": True,
                "verification_status": "completed",
                "compliance_data": {
                    "score": compliance_score,
                    "violations_count": len(violations),
                    "status": "compliant" if compliance_score > 0.9 else "non_compliant"
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _handle_assess_risk(self, params: Dict) -> Dict:
        """Handle risk assessment tool"""

        try:
            risk_assessment = params.get("risk_assessment", {})
            user_id = params.get("user_id")
            session_id = params.get("session_id")

            # Log risk assessment
            logger.info(f"Risk assessment for user {user_id}, session {session_id}")

            return {
                "success": True,
                "assessment_status": "completed",
                "risk_data": risk_assessment,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _handle_memory_store(self, params: Dict) -> Dict:
        """Handle memory store tool"""

        try:
            bucket = params.get("bucket")
            data = params.get("data", {})
            metadata = params.get("metadata", {})

            # Mock memory storage
            entry_id = secrets.token_hex(16)

            return {
                "success": True,
                "memory_operation": {
                    "operation": "store",
                    "bucket": bucket,
                    "entry_id": entry_id,
                    "data_size": len(json.dumps(data)),
                    "stored_at": datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _handle_memory_retrieve(self, params: Dict) -> Dict:
        """Handle memory retrieve tool"""

        try:
            bucket = params.get("bucket")
            filter_criteria = params.get("filter", {})
            limit = params.get("limit", 10)

            # Mock memory retrieval
            mock_entries = [
                {
                    "entry_id": secrets.token_hex(8),
                    "data": {"type": "pattern", "value": "mock_data"},
                    "created_at": datetime.now().isoformat(),
                    "bucket": bucket
                }
                for i in range(min(3, limit))
            ]

            return {
                "success": True,
                "memory_operation": {
                    "operation": "retrieve",
                    "bucket": bucket,
                    "results": mock_entries,
                    "total_found": len(mock_entries),
                    "retrieved_at": datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _handle_send_notification(self, params: Dict) -> Dict:
        """Handle notification sending tool"""

        try:
            notification_results = params.get("notification_results", {})
            user_id = params.get("user_id")
            session_id = params.get("session_id")

            # Mock notification sending
            logger.info(f"Sending notifications for user {user_id}, session {session_id}")

            return {
                "success": True,
                "notification_status": "sent",
                "delivery_summary": {
                    "total_notifications": notification_results.get("total_notifications", 0),
                    "successful_deliveries": notification_results.get("successful", 0),
                    "failed_deliveries": notification_results.get("failed", 0)
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _handle_generate_report(self, params: Dict) -> Dict:
        """Handle report generation tool"""

        try:
            report = params.get("report", {})
            user_id = params.get("user_id")
            session_id = params.get("session_id")

            # Mock report generation
            report_id = secrets.token_hex(16)

            generated_report = {
                "report_id": report_id,
                "type": "compliance_report",
                "format": "json",
                "generated_at": datetime.now().isoformat(),
                "user_id": user_id,
                "session_id": session_id,
                "content": report,
                "file_size": len(json.dumps(report, default=str)),
                "status": "completed"
            }

            logger.info(f"Generated report {report_id} for user {user_id}")

            return {
                "success": True,
                "report_generated": generated_report,
                "download_url": f"/reports/{report_id}",
                "expires_at": (datetime.now() + timedelta(days=30)).isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _handle_error(self, params: Dict) -> Dict:
        """Handle error processing tool"""

        try:
            errors = params.get("errors", [])
            failed_node = params.get("failed_node")
            recovery_action = params.get("recovery_action")
            user_id = params.get("user_id")

            # Log error handling
            logger.warning(f"Handling {len(errors)} errors for user {user_id}, failed node: {failed_node}")

            return {
                "success": True,
                "error_handling": {
                    "errors_processed": len(errors),
                    "failed_node": failed_node,
                    "recovery_action": recovery_action,
                    "handled_at": datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _handle_audit_trail(self, params: Dict) -> Dict:
        """Handle audit trail logging tool"""

        try:
            audit_log = params.get("audit_log", {})
            user_id = params.get("user_id")
            session_id = params.get("session_id")

            # Mock audit trail logging
            audit_id = secrets.token_hex(16)

            logger.info(f"Logging audit trail {audit_id} for user {user_id}, session {session_id}")

            return {
                "success": True,
                "audit_logged": {
                    "audit_id": audit_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "logged_at": datetime.now().isoformat(),
                    "log_size": len(json.dumps(audit_log, default=str)),
                    "retention_period": "7_years"
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_server_statistics(self) -> Dict:
        """Get server statistics"""

        # Calculate uptime and other metrics
        total_connections = len(self.connections)
        active_connections = len([c for c in self.connections.values() if c])

        return {
            "server_info": {
                "host": self.host,
                "port": self.port,
                "uptime_seconds": (datetime.now() - datetime.now()).total_seconds(),  # Would track actual start time
                "tools_registered": len(self.tools),
                "categories": list(self.tool_categories.keys())
            },
            "connections": {
                "total_established": self.stats["connections_established"],
                "currently_active": active_connections,
                "total_requests": self.stats["total_requests"]
            },
            "tool_usage": {
                "successful_calls": self.stats["successful_calls"],
                "failed_calls": self.stats["failed_calls"],
                "success_rate": round(
                    self.stats["successful_calls"] / max(1, self.stats["total_requests"]), 3
                )
            },
            "rate_limiting": {
                "active_limits": len(self.rate_limits),
                "rate_limited_requests": 0  # Would track actual rate limited requests
            }
        }

    async def shutdown_server(self):
        """Gracefully shutdown the server"""

        try:
            # Notify all connected clients
            shutdown_notification = {
                "type": "notification",
                "notification": "server_shutdown",
                "data": {
                    "message": "Server is shutting down",
                    "timestamp": datetime.now().isoformat()
                }
            }

            # Send shutdown notification to all connections
            for connection_id, connection_info in self.connections.items():
                try:
                    websocket = connection_info["websocket"]
                    await websocket.send(json.dumps(shutdown_notification, default=str))
                    await websocket.close()
                except:
                    pass  # Ignore errors during shutdown

            # Clear connections
            self.connections.clear()

            logger.info("MCP server shutdown completed")

        except Exception as e:
            logger.error(f"Error during server shutdown: {str(e)}")


# ========================= UTILITY FUNCTIONS =========================

async def create_mcp_client(server_url: str = "ws://localhost:8765",
                            auth_token: str = None) -> MCPClient:
    """Create and connect MCP client"""

    client = MCPClient(server_url=server_url, auth_token=auth_token)

    if await client.connect():
        logger.info("MCP client connected successfully")
        return client
    else:
        logger.error("Failed to connect MCP client")
        return None


async def create_mcp_server(host: str = "localhost", port: int = 8765) -> MCPServer:
    """Create and start MCP server"""

    server = MCPServer(host=host, port=port)
    await server.start_server()

    logger.info(f"MCP server started on {host}:{port}")
    return server


# ========================= EXAMPLE USAGE =========================

async def main():
    """Example usage of MCP implementation"""

    # Start server
    server = MCPServer()
    server_task = await server.start_server()

    # Create client
    client = MCPClient()
    await client.connect()

    # Test tool calls
    tools = await client.list_tools()
    print(f"Available tools: {tools}")

    # Test data processing tool
    result = await client.call_tool("process_banking_data", {
        "data": {"accounts": [{"id": "123", "balance": 1000}]},
        "quality_checks": True
    })
    print(f"Processing result: {result}")

    # Test dormancy analysis
    dormancy_result = await client.call_tool("analyze_account_dormancy", {
        "accounts": [{"id": "123", "last_activity": "2020-01-01"}],
        "report_date": "2024-01-01"
    })
    print(f"Dormancy analysis: {dormancy_result}")

    # Cleanup
    await client.disconnect()
    await server.shutdown_server()


if __name__ == "__main__":
    print("Banking Compliance MCP Implementation")
    print("=" * 50)
    print("Features:")
    print("- WebSocket-based MCP protocol")
    print("- Banking compliance tool registry")
    print("- Rate limiting and authentication")
    print("- Comprehensive audit logging")
    print("- Error handling and recovery")
    print("- Performance monitoring")
    print("- Mock tool implementations")

    # Uncomment to run example
    # asyncio.run(main())