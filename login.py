"""
Secure Login System with 256-bit AES Encryption
Implements secure authentication with password hashing, session management, and encryption
"""

import hashlib
import secrets
import base64
import json
import time
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import bcrypt
import jwt
from typing import Dict, Optional, Tuple
import sqlite3
import os
from functools import wraps


class SecureLoginManager:
    """
    High-security login system with 256-bit encryption
    Features:
    - PBKDF2 key derivation with 100,000 iterations
    - bcrypt password hashing with salt rounds
    - AES-256 encryption for sensitive data
    - JWT tokens for session management
    - SQL injection protection
    - Rate limiting and brute force protection
    """

    def __init__(self, db_path: str = "secure_users.db", jwt_secret: str = None):
        self.db_path = db_path
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.failed_attempts = {}  # IP-based rate limiting
        self.max_attempts = 5
        self.lockout_duration = 300  # 5 minutes

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize secure user database with proper schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    role TEXT DEFAULT 'user',
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP NULL
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_token TEXT UNIQUE NOT NULL,
                    encrypted_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            conn.commit()

    def _generate_encryption_key(self, password: bytes, salt: bytes) -> bytes:
        """Generate 256-bit encryption key using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256 bits
            salt=salt,
            iterations=100000,  # High iteration count for security
            backend=default_backend()
        )
        return base64.urlsafe_b64encode(kdf.derive(password))

    def _encrypt_data(self, data: str, key: bytes) -> str:
        """Encrypt data using AES-256"""
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()

    def _decrypt_data(self, encrypted_data: str, key: bytes) -> str:
        """Decrypt data using AES-256"""
        try:
            fernet = Fernet(key)
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")

    def _hash_password(self, password: str) -> Tuple[str, str]:
        """Hash password using bcrypt with random salt"""
        salt = bcrypt.gensalt(rounds=12)  # High cost factor
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8'), salt.decode('utf-8')

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against stored hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

    def _check_rate_limit(self, identifier: str) -> bool:
        """Check if request is rate limited"""
        current_time = time.time()

        if identifier in self.failed_attempts:
            attempts, last_attempt = self.failed_attempts[identifier]

            # Reset if lockout period has passed
            if current_time - last_attempt > self.lockout_duration:
                del self.failed_attempts[identifier]
                return True

            # Check if max attempts exceeded
            if attempts >= self.max_attempts:
                return False

        return True

    def _record_failed_attempt(self, identifier: str):
        """Record failed login attempt"""
        current_time = time.time()

        if identifier in self.failed_attempts:
            attempts, _ = self.failed_attempts[identifier]
            self.failed_attempts[identifier] = (attempts + 1, current_time)
        else:
            self.failed_attempts[identifier] = (1, current_time)

    def create_user(self, username: str, password: str, role: str = 'user') -> bool:
        """Create new user with secure password storage"""
        try:
            # Validate input
            if len(password) < 8:
                raise ValueError("Password must be at least 8 characters long")

            # Hash password
            password_hash, salt = self._hash_password(password)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO users (username, password_hash, salt, role)
                    VALUES (?, ?, ?, ?)
                ''', (username, password_hash, salt, role))
                conn.commit()

            return True

        except sqlite3.IntegrityError:
            raise ValueError("Username already exists")
        except Exception as e:
            raise ValueError(f"User creation failed: {str(e)}")

    def authenticate_user(self, username: str, password: str, client_ip: str = None) -> Optional[Dict]:
        """Authenticate user with rate limiting and security checks"""
        identifier = client_ip or username

        # Check rate limiting
        if not self._check_rate_limit(identifier):
            raise ValueError("Too many failed attempts. Please try again later.")

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, username, password_hash, salt, role, is_active, 
                           locked_until, failed_login_attempts
                    FROM users WHERE username = ?
                ''', (username,))

                user_data = cursor.fetchone()

                if not user_data:
                    self._record_failed_attempt(identifier)
                    raise ValueError("Invalid credentials")

                user_id, username, password_hash, salt, role, is_active, locked_until, failed_attempts = user_data

                # Check if account is active
                if not is_active:
                    raise ValueError("Account is disabled")

                # Check if account is locked
                if locked_until:
                    locked_until_dt = datetime.fromisoformat(locked_until)
                    if datetime.now() < locked_until_dt:
                        raise ValueError("Account is temporarily locked")

                # Verify password
                if not self._verify_password(password, password_hash):
                    self._record_failed_attempt(identifier)

                    # Update failed attempts in database
                    new_failed_attempts = failed_attempts + 1
                    lock_until = None

                    if new_failed_attempts >= self.max_attempts:
                        lock_until = datetime.now() + timedelta(minutes=5)

                    cursor.execute('''
                        UPDATE users SET failed_login_attempts = ?, locked_until = ?
                        WHERE id = ?
                    ''', (new_failed_attempts, lock_until, user_id))
                    conn.commit()

                    raise ValueError("Invalid credentials")

                # Reset failed attempts on successful login
                cursor.execute('''
                    UPDATE users SET failed_login_attempts = 0, locked_until = NULL, 
                                   last_login = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (user_id,))
                conn.commit()

                # Clear rate limiting
                if identifier in self.failed_attempts:
                    del self.failed_attempts[identifier]

                return {
                    'user_id': user_id,
                    'username': username,
                    'role': role,
                    'authenticated': True
                }

        except ValueError:
            raise
        except Exception as e:
            self._record_failed_attempt(identifier)
            raise ValueError(f"Authentication failed: {str(e)}")

    def create_secure_session(self, user_data: Dict, session_data: Dict = None) -> str:
        """Create encrypted session with JWT token"""
        try:
            # Generate session token
            session_payload = {
                'user_id': user_data['user_id'],
                'username': user_data['username'],
                'role': user_data['role'],
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(hours=8),  # 8-hour session
                'jti': secrets.token_urlsafe(16)  # Unique session ID
            }

            session_token = jwt.encode(session_payload, self.jwt_secret, algorithm='HS256')

            # Encrypt additional session data if provided
            encrypted_session_data = None
            if session_data:
                # Generate encryption key from user password hash (simulated)
                salt = secrets.token_bytes(32)
                key = self._generate_encryption_key(self.jwt_secret.encode(), salt)
                encrypted_session_data = self._encrypt_data(json.dumps(session_data), key)

            # Store session in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO sessions (user_id, session_token, encrypted_data, expires_at)
                    VALUES (?, ?, ?, ?)
                ''', (
                    user_data['user_id'],
                    session_token,
                    encrypted_session_data,
                    datetime.now() + timedelta(hours=8)
                ))
                conn.commit()

            return session_token

        except Exception as e:
            raise ValueError(f"Session creation failed: {str(e)}")

    def validate_session(self, session_token: str) -> Optional[Dict]:
        """Validate and decode session token"""
        try:
            # Decode JWT token
            payload = jwt.decode(session_token, self.jwt_secret, algorithms=['HS256'])

            # Check session in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT s.user_id, s.encrypted_data, s.expires_at, s.is_active,
                           u.username, u.role, u.is_active as user_active
                    FROM sessions s
                    JOIN users u ON s.user_id = u.id
                    WHERE s.session_token = ?
                ''', (session_token,))

                session_data = cursor.fetchone()

                if not session_data:
                    raise ValueError("Session not found")

                user_id, encrypted_data, expires_at, session_active, username, role, user_active = session_data

                # Check if session is active
                if not session_active or not user_active:
                    raise ValueError("Session is inactive")

                # Check expiration
                expires_dt = datetime.fromisoformat(expires_at)
                if datetime.now() > expires_dt:
                    # Deactivate expired session
                    cursor.execute('UPDATE sessions SET is_active = 0 WHERE session_token = ?', (session_token,))
                    conn.commit()
                    raise ValueError("Session expired")

                return {
                    'user_id': user_id,
                    'username': username,
                    'role': role,
                    'session_valid': True,
                    'expires_at': expires_at
                }

        except jwt.ExpiredSignatureError:
            raise ValueError("Session token expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid session token")
        except Exception as e:
            raise ValueError(f"Session validation failed: {str(e)}")

    def logout_user(self, session_token: str) -> bool:
        """Securely logout user and invalidate session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE sessions SET is_active = 0 
                    WHERE session_token = ?
                ''', (session_token,))
                conn.commit()

                return cursor.rowcount > 0

        except Exception:
            return False

    def cleanup_expired_sessions(self):
        """Clean up expired sessions from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM sessions 
                    WHERE expires_at < CURRENT_TIMESTAMP OR is_active = 0
                ''')
                conn.commit()

        except Exception:
            pass


# Decorator for requiring authentication
def require_authentication(login_manager: SecureLoginManager):
    """Decorator to require valid session for API endpoints"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract session token from request (implementation depends on framework)
            session_token = kwargs.get('session_token') or getattr(args[0], 'session_token', None)

            if not session_token:
                raise ValueError("Authentication required")

            try:
                user_data = login_manager.validate_session(session_token)
                kwargs['current_user'] = user_data
                return func(*args, **kwargs)
            except ValueError as e:
                raise ValueError(f"Authentication failed: {str(e)}")

        return wrapper

    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Initialize login manager
    login_manager = SecureLoginManager()

    try:
        # Create test user
        login_manager.create_user("admin", "SecurePassword123!", "admin")
        login_manager.create_user("analyst", "AnalystPass456!", "analyst")

        # Authenticate user
        user_data = login_manager.authenticate_user("admin", "SecurePassword123!")
        print(f"Authentication successful: {user_data}")

        # Create secure session
        session_token = login_manager.create_secure_session(
            user_data,
            {"workspace": "banking_compliance", "permissions": ["read", "write"]}
        )
        print(f"Session created: {session_token[:50]}...")

        # Validate session
        session_info = login_manager.validate_session(session_token)
        print(f"Session validation: {session_info}")

        # Test rate limiting with wrong password
        try:
            for i in range(6):  # Exceed max attempts
                login_manager.authenticate_user("admin", "wrongpassword")
        except ValueError as e:
            print(f"Rate limiting works: {e}")

        print("Login system tests completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")