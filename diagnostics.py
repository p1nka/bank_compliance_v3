"""
Diagnostic version to debug login issues
"""

import streamlit as st
import pandas as pd
import time
import traceback
from datetime import datetime
import sqlite3
import os

# Import secure login system
try:
    from login import SecureLoginManager, require_authentication

    LOGIN_MODULE_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Failed to import login module: {e}")
    LOGIN_MODULE_AVAILABLE = False

st.set_page_config(
    page_title="Login Diagnostic Tool",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Login System Diagnostic Tool")


def check_database_status():
    """Check database and user status"""
    st.subheader("📊 Database Status")

    if not os.path.exists("secure_users.db"):
        st.warning("⚠️ Database file 'secure_users.db' does not exist")
        return

    st.success("✅ Database file exists")

    try:
        with sqlite3.connect("secure_users.db") as conn:
            cursor = conn.cursor()

            # Check users table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            if cursor.fetchone():
                st.success("✅ Users table exists")

                # Get user count and details
                cursor.execute("SELECT username, role, is_active, failed_login_attempts, locked_until FROM users")
                users = cursor.fetchall()

                if users:
                    st.write(f"**Users in database: {len(users)}**")
                    for username, role, is_active, failed_attempts, locked_until in users:
                        status = "🟢 Active" if is_active else "🔴 Inactive"
                        lock_status = f"🔒 Locked until {locked_until}" if locked_until else "🔓 Unlocked"
                        st.write(
                            f"- **{username}** ({role}) - {status} - Failed attempts: {failed_attempts} - {lock_status}")
                else:
                    st.warning("⚠️ No users found in database")
            else:
                st.error("❌ Users table does not exist")

            # Check sessions table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
            if cursor.fetchone():
                st.success("✅ Sessions table exists")

                cursor.execute("SELECT COUNT(*) FROM sessions WHERE is_active = 1")
                active_sessions = cursor.fetchone()[0]
                st.write(f"**Active sessions: {active_sessions}**")
            else:
                st.error("❌ Sessions table does not exist")

    except Exception as e:
        st.error(f"❌ Database error: {e}")


def reset_database():
    """Reset and recreate database with fresh users"""
    st.subheader("🔄 Database Reset")

    if st.button("🗑️ Delete Database and Recreate", type="secondary"):
        try:
            # Delete existing database
            if os.path.exists("secure_users.db"):
                os.remove("secure_users.db")
                st.success("✅ Old database deleted")

            # Create new login manager (this will create fresh database)
            login_manager = SecureLoginManager()
            st.success("✅ New database created")

            # Create users with detailed logging
            users_to_create = [
                ("admin", "SecurePassword123!", "admin"),
                ("analyst", "AnalystPass456!", "analyst"),
                ("compliance", "ComplianceOfficer789!", "compliance")
            ]

            for username, password, role in users_to_create:
                try:
                    result = login_manager.create_user(username, password, role)
                    st.success(f"✅ Created user: {username}")
                except Exception as e:
                    st.error(f"❌ Failed to create user {username}: {e}")

            st.success("🎉 Database reset complete! Please try logging in again.")
            time.sleep(2)
            st.rerun()

        except Exception as e:
            st.error(f"❌ Reset failed: {e}")
            st.write(traceback.format_exc())


def test_manual_authentication():
    """Test authentication manually with detailed logging"""
    st.subheader("🧪 Manual Authentication Test")

    col1, col2 = st.columns(2)

    with col1:
        test_username = st.text_input("Test Username", value="admin")
        test_password = st.text_input("Test Password", value="SecurePassword123!", type="password")

    with col2:
        if st.button("🧪 Test Authentication", type="primary"):
            if not LOGIN_MODULE_AVAILABLE:
                st.error("❌ Login module not available")
                return

            try:
                st.write("🔧 Creating SecureLoginManager...")
                login_manager = SecureLoginManager()
                st.success("✅ SecureLoginManager created")

                st.write(f"🔧 Testing authentication for user: {test_username}")
                st.write(f"🔧 Password length: {len(test_password)}")

                # Test authentication with detailed error handling
                try:
                    user_data = login_manager.authenticate_user(test_username, test_password, "127.0.0.1")
                    st.success(f"✅ Authentication successful!")
                    st.json(user_data)

                    # Test session creation
                    try:
                        session_token = login_manager.create_secure_session(
                            user_data,
                            {"test": "session_data"}
                        )
                        st.success(f"✅ Session created: {session_token[:20]}...")

                        # Test session validation
                        try:
                            session_info = login_manager.validate_session(session_token)
                            st.success("✅ Session validation successful!")
                            st.json(session_info)
                        except Exception as e:
                            st.error(f"❌ Session validation failed: {e}")

                    except Exception as e:
                        st.error(f"❌ Session creation failed: {e}")

                except ValueError as e:
                    st.error(f"❌ Authentication failed: {e}")

                    # Check if it's a rate limiting issue
                    if "too many failed attempts" in str(e).lower():
                        st.warning("🚨 Rate limiting detected. Trying to clear failed attempts...")
                        # Clear failed attempts manually if needed

                except Exception as e:
                    st.error(f"❌ Unexpected authentication error: {e}")
                    st.write(traceback.format_exc())

            except Exception as e:
                st.error(f"❌ Failed to create SecureLoginManager: {e}")
                st.write(traceback.format_exc())


def check_password_verification():
    """Test password hashing and verification"""
    st.subheader("🔐 Password Verification Test")

    if st.button("🔐 Test Password Hashing", type="secondary"):
        try:
            login_manager = SecureLoginManager()

            # Test password hashing
            test_password = "SecurePassword123!"
            password_hash, salt = login_manager._hash_password(test_password)

            st.write(f"**Original Password:** {test_password}")
            st.write(f"**Password Hash:** {password_hash[:50]}...")
            st.write(f"**Salt:** {salt[:20]}...")

            # Test verification
            verification_result = login_manager._verify_password(test_password, password_hash)

            if verification_result:
                st.success("✅ Password verification works correctly")
            else:
                st.error("❌ Password verification failed")

            # Test wrong password
            wrong_verification = login_manager._verify_password("wrongpassword", password_hash)
            if not wrong_verification:
                st.success("✅ Wrong password correctly rejected")
            else:
                st.error("❌ Wrong password incorrectly accepted")

        except Exception as e:
            st.error(f"❌ Password test failed: {e}")
            st.write(traceback.format_exc())


def main():
    """Main diagnostic interface"""

    # Initialize session state
    if 'diagnostic_login_manager' not in st.session_state:
        if LOGIN_MODULE_AVAILABLE:
            try:
                st.session_state.diagnostic_login_manager = SecureLoginManager()
            except Exception as e:
                st.error(f"❌ Failed to create SecureLoginManager: {e}")
                st.session_state.diagnostic_login_manager = None
        else:
            st.session_state.diagnostic_login_manager = None

    # Show system status
    col1, col2 = st.columns(2)

    with col1:
        if LOGIN_MODULE_AVAILABLE:
            st.success("✅ Login module imported successfully")
        else:
            st.error("❌ Login module import failed")

    with col2:
        if st.session_state.diagnostic_login_manager:
            st.success("✅ SecureLoginManager created")
        else:
            st.error("❌ SecureLoginManager creation failed")

    # Database diagnostics
    check_database_status()

    # Reset option
    reset_database()

    # Manual authentication test
    test_manual_authentication()

    # Password verification test
    check_password_verification()

    # File system check
    st.subheader("📁 File System Check")
    st.write(f"**Current directory:** {os.getcwd()}")
    st.write(f"**Database exists:** {os.path.exists('secure_users.db')}")
    if os.path.exists('login.py'):
        st.success("✅ login.py file found")
        st.write(f"**login.py size:** {os.path.getsize('login.py')} bytes")
    else:
        st.error("❌ login.py file not found")


if __name__ == "__main__":
    main()