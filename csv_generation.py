#!/usr/bin/env python3
"""
Standalone CBUAE Banking Compliance CSV Generator
================================================

Pure Python CSV generator without Streamlit dependencies.
Can be run directly from command line.

Usage:
    python standalone_csv_generator.py
    python standalone_csv_generator.py --rows 500 --output test_data.csv
"""

import csv
import random
import secrets
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


class StandaloneCBUAECSVGenerator:
    """Standalone CBUAE Banking Compliance CSV Generator - No Streamlit Dependencies"""

    def __init__(self, num_records: int = 1000):
        self.num_records = num_records
        self.setup_data_sources()

    def setup_data_sources(self):
        """Initialize data sources for realistic UAE banking data"""

        # UAE Customer Names (English and Arabic pairs)
        self.customer_names = [
            ("Ahmed Al-Rashid", "أحمد الراشد"),
            ("Fatima Al-Zahra", "فاطمة الزهراء"),
            ("Omar Al-Suwaidi", "عمر السويدي"),
            ("Hassan Al-Maktoum", "حسان المكتوم"),
            ("Zahra Al-Falasi", "زهرة الفلاسي"),
            ("Khalifa Al-Dhaheri", "خليفة الظاهري"),
            ("Majid Al-Qasimi", "ماجد القاسمي"),
            ("Dana Al-Maktoum", "دانا المكتوم"),
            ("Reem Al-Suwaidi", "ريم السويدي"),
            ("Faisal Al-Kaabi", "فيصل الكعبي"),
            ("Saeed Al-Mazrouei", "سعيد المزروعي"),
            ("Hind Al-Nuaimi", "هند النعيمي"),
            ("Abdullah Al-Kaabi", "عبدالله الكعبي"),
            ("Latifa Al-Shamsi", "لطيفة الشامسي"),
            ("Rashid Al-Falasi", "راشد الفلاسي"),
            ("Maryam Al-Mansoori", "مريم المنصوري"),
            ("Noura Al-Suwaidi", "نورا السويدي"),
            ("Salem Al-Qasimi", "سالم القاسمي"),
            ("Amira Al-Dhaheri", "أميرة الظاهري"),
            ("Mohammad Al-Shehhi", "محمد الشحي"),
        ]

        # UAE Address Data
        self.address_data = [
            ("Villa 123 Al Wasl Road", "Dubai", "Dubai"),
            ("Apartment 45B Marina Walk", "Dubai", "Dubai"),
            ("Villa 156 Palm Jumeirah", "Dubai", "Dubai"),
            ("Palace Road Villa 1", "Dubai", "Dubai"),
            ("Villa 88 Al Safa", "Dubai", "Dubai"),
            ("Villa 123 Khalifa City", "Abu Dhabi", "Abu Dhabi"),
            ("House 77 Al Khaleej", "Sharjah", "Sharjah"),
            ("Apartment 99 Business Bay", "Dubai", "Dubai"),
            ("House 456 Al Barsha", "Dubai", "Dubai"),
            ("Apartment 789 Marina", "Dubai", "Dubai"),
            ("House 654 Al Qusais", "Dubai", "Dubai"),
            ("Apartment 987 Downtown", "Dubai", "Dubai"),
            ("Villa 111 Jumeirah", "Dubai", "Dubai"),
            ("House 321 Al Wasl", "Dubai", "Dubai"),
            ("Villa 234 Al Safa", "Dubai", "Dubai"),
            ("", "Abu Dhabi", "Abu Dhabi"),  # Unknown address scenario
            ("House 99 Al Khaleej", "Sharjah", "Sharjah"),
            ("Apartment 567 DIFC", "Dubai", "Dubai"),
            ("Villa 890 Al Barsha", "Dubai", "Dubai"),
            ("House 123 Al Mizhar", "Dubai", "Dubai"),
        ]

        # Account Type Scenarios for Comprehensive Testing (All CBUAE Articles)
        self.account_scenarios = [
            # Demand Deposit Scenarios (Article 2.1.1)
            {"type": "CURRENT", "subtype": "PERSONAL", "purpose": "Demand Deposit Dormancy", "weight": 0.20},
            {"type": "SAVINGS", "subtype": "PREMIUM", "purpose": "Demand Deposit Dormancy", "weight": 0.15},
            {"type": "CURRENT", "subtype": "SALARY", "purpose": "Demand Deposit Dormancy", "weight": 0.12},
            {"type": "SAVINGS", "subtype": "REGULAR", "purpose": "Demand Deposit Dormancy", "weight": 0.12},
            {"type": "CURRENT", "subtype": "JOINT", "purpose": "Demand Deposit Dormancy", "weight": 0.08},

            # Fixed Deposit Scenarios (Article 2.2)
            {"type": "FIXED_DEPOSIT", "subtype": "STANDARD", "purpose": "Fixed Deposit Dormancy", "weight": 0.08},
            {"type": "FIXED_DEPOSIT", "subtype": "PREMIUM", "purpose": "Fixed Deposit Dormancy", "weight": 0.05},
            {"type": "FIXED_DEPOSIT", "subtype": "ISLAMIC", "purpose": "Fixed Deposit Dormancy", "weight": 0.03},

            # Investment Scenarios (Article 2.3)
            {"type": "INVESTMENT", "subtype": "MUTUAL_FUND", "purpose": "Investment Dormancy", "weight": 0.04},
            {"type": "INVESTMENT", "subtype": "PORTFOLIO", "purpose": "Investment Dormancy", "weight": 0.03},
            {"type": "INVESTMENT", "subtype": "SECURITIES", "purpose": "Investment Dormancy", "weight": 0.02},

            # Payment Instruments (Article 2.4) - CRITICAL FOR TESTING
            {"type": "CURRENT", "subtype": "INSTRUMENT_LINKED", "purpose": "Payment Instruments", "weight": 0.03},
            {"type": "SAVINGS", "subtype": "INSTRUMENT_LINKED", "purpose": "Payment Instruments", "weight": 0.02},

            # Safe Deposit Box (Article 2.6) - CRITICAL FOR TESTING
            {"type": "SAVINGS", "subtype": "SDB_LINKED", "purpose": "Safe Deposit Box", "weight": 0.02},
            {"type": "CURRENT", "subtype": "SDB_LINKED", "purpose": "Safe Deposit Box", "weight": 0.01},

            # High Value & Special Scenarios
            {"type": "CURRENT", "subtype": "BUSINESS", "purpose": "High Value Account", "weight": 0.02},
            {"type": "CURRENT", "subtype": "VIP", "purpose": "High Value Account", "weight": 0.01},
            {"type": "SAVINGS", "subtype": "CORPORATE", "purpose": "Corporate Account", "weight": 0.01},
        ]

        # Nationalities for diverse testing
        self.nationalities = [
            "UAE", "INDIA", "PAKISTAN", "PHILIPPINES", "EGYPT",
            "JORDAN", "LEBANON", "SYRIA", "BANGLADESH", "SRI_LANKA",
            "NEPAL", "SUDAN", "YEMEN", "SOMALIA", "ETHIOPIA"
        ]

        # Emirates
        self.emirates = [
            "DUBAI", "ABU_DHABI", "SHARJAH", "AJMAN",
            "UMM_AL_QUWAIN", "RAS_AL_KHAIMAH", "FUJAIRAH"
        ]

        # Currencies
        self.currencies = ["AED", "USD", "EUR", "GBP", "SAR", "INR"]

        # Risk Ratings
        self.risk_ratings = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

        # Dormancy Stages
        self.dormancy_stages = [
            "ACTIVE", "FLAGGED", "CONTACTED", "WAITING",
            "TRANSFER_READY", "TRANSFERRED", "CLOSED"
        ]

        # Contact Methods
        self.contact_methods = ["EMAIL", "PHONE", "SMS", "LETTER", "BRANCH_VISIT"]

        # Statement Frequencies
        self.statement_frequencies = ["MONTHLY", "QUARTERLY", "SEMI_ANNUAL", "ANNUAL"]

        # 66 Column Headers (Complete CBUAE Schema)
        self.column_headers = [
            "customer_id", "customer_type", "full_name_en", "full_name_ar",
            "id_number", "id_type", "date_of_birth", "nationality",
            "address_line1", "address_line2", "city", "emirate", "country",
            "postal_code", "phone_primary", "phone_secondary", "email_primary",
            "email_secondary", "address_known", "last_contact_date",
            "last_contact_method", "kyc_status", "kyc_expiry_date", "risk_rating",
            "account_id", "account_type", "account_subtype", "account_name",
            "currency", "account_status", "dormancy_status", "opening_date",
            "closing_date", "last_transaction_date", "last_system_transaction_date",
            "balance_current", "balance_available", "balance_minimum",
            "interest_rate", "interest_accrued", "is_joint_account",
            "joint_account_holders", "has_outstanding_facilities", "maturity_date",
            "auto_renewal", "last_statement_date", "statement_frequency",
            "tracking_id", "dormancy_trigger_date", "dormancy_period_start",
            "dormancy_period_months", "dormancy_classification_date",
            "transfer_eligibility_date", "current_stage", "contact_attempts_made",
            "last_contact_attempt_date", "waiting_period_start", "waiting_period_end",
            "transferred_to_ledger_date", "transferred_to_cb_date",
            "cb_transfer_amount", "cb_transfer_reference", "exclusion_reason",
            "created_date", "updated_date", "updated_by"
        ]

    def generate_customer_data(self, index: int) -> Dict[str, Any]:
        """Generate realistic customer information"""

        # Select name (cycle through available names)
        name_index = index % len(self.customer_names)
        name_en, name_ar = self.customer_names[name_index]

        # Select address (cycle through available addresses)
        address_index = index % len(self.address_data)
        address_line1, city, emirate = self.address_data[address_index]

        # Generate Emirates ID
        emirates_id = f"784{random.randint(1000000000, 9999999999)}"

        # Generate realistic birth date (18-80 years old)
        birth_date = datetime.now() - timedelta(days=random.randint(18 * 365, 80 * 365))

        # Generate phone numbers
        phone_primary = f"971{random.choice(['50', '52', '54', '55', '56'])}{random.randint(1000000, 9999999)}"
        # 70% chance of having secondary phone
        phone_secondary = f"971{random.choice(['50', '52', '54', '55', '56'])}{random.randint(1000000, 9999999)}" if random.random() < 0.7 else ""

        # Generate emails
        email_domains = ['gmail.com', 'hotmail.com', 'yahoo.com', 'outlook.com', 'email.ae', 'adcb.com', 'nbad.com']
        clean_name = name_en.lower().replace(' ', '.').replace('-', '.')
        email_primary = f"{clean_name}@{random.choice(email_domains)}"
        # 40% chance of having secondary email
        email_secondary = f"{clean_name}.{random.randint(1, 99)}@{random.choice(email_domains)}" if random.random() < 0.4 else ""

        return {
            'customer_id': f"CUS{index + 1:06d}",
            'customer_type': random.choice(["INDIVIDUAL", "CORPORATE"]),
            'full_name_en': name_en,
            'full_name_ar': name_ar,
            'id_number': emirates_id,
            'id_type': "EMIRATES_ID",
            'date_of_birth': birth_date.strftime('%Y-%m-%d'),
            'nationality': random.choice(self.nationalities),
            'address_line1': address_line1,
            'address_line2': f"Area {random.randint(1, 20)}" if address_line1 else "",
            'city': city,
            'emirate': emirate,
            'country': "UAE",
            'postal_code': f"{random.randint(10000, 99999)}" if address_line1 else "",
            'phone_primary': phone_primary,
            'phone_secondary': phone_secondary,
            'email_primary': email_primary,
            'email_secondary': email_secondary,
            'address_known': "NO" if not address_line1 else "YES"
        }

    def generate_balance(self, account_type: str, account_subtype: str) -> float:
        """Generate realistic balance based on account type and subtype"""

        if account_type == "CURRENT":
            if account_subtype == "BUSINESS":
                return random.uniform(50000, 2000000)  # Business accounts
            elif account_subtype == "VIP":
                return random.uniform(500000, 3000000)  # VIP accounts
            elif account_subtype == "PREMIUM":
                return random.uniform(25000, 500000)  # Premium accounts
            elif account_subtype == "INSTRUMENT_LINKED":
                return random.uniform(10000, 80000)  # Payment instruments (Article 2.4)
            elif account_subtype == "SDB_LINKED":
                return random.uniform(20000, 100000)  # Safe deposit box linked (Article 2.6)
            elif account_subtype == "SALARY":
                return random.uniform(5000, 75000)  # Salary accounts
            elif account_subtype == "JOINT":
                return random.uniform(15000, 200000)  # Joint accounts
            else:
                return random.uniform(1000, 100000)  # Regular current accounts

        elif account_type == "SAVINGS":
            if account_subtype == "PREMIUM":
                return random.uniform(50000, 750000)  # Premium savings
            elif account_subtype == "CORPORATE":
                return random.uniform(100000, 1500000)  # Corporate savings
            elif account_subtype == "ISLAMIC":
                return random.uniform(15000, 300000)  # Islamic savings
            elif account_subtype == "SDB_LINKED":
                return random.uniform(25000, 150000)  # Safe deposit box linked (Article 2.6)
            elif account_subtype == "INSTRUMENT_LINKED":
                return random.uniform(15000, 90000)  # Payment instruments linked (Article 2.4)
            else:
                return random.uniform(5000, 200000)  # Regular savings

        elif account_type == "FIXED_DEPOSIT":
            if account_subtype == "PREMIUM":
                return random.uniform(500000, 3000000)  # Premium fixed deposits
            elif account_subtype == "ISLAMIC":
                return random.uniform(100000, 1000000)  # Islamic fixed deposits
            else:
                return random.uniform(100000, 2000000)  # Standard fixed deposits

        elif account_type == "INVESTMENT":
            if account_subtype == "PORTFOLIO":
                return random.uniform(200000, 2000000)  # Portfolio management
            elif account_subtype == "SECURITIES":
                return random.uniform(100000, 1500000)  # Securities trading
            elif account_subtype == "MUTUAL_FUND":
                return random.uniform(50000, 800000)  # Mutual funds
            else:
                return random.uniform(50000, 1000000)  # General investment accounts

        else:
            return random.uniform(1000, 50000)  # Default fallback

    def generate_dormancy_scenario(self, scenario: Dict[str, Any], current_date: datetime) -> Dict[str, Any]:
        """Generate realistic dormancy scenario based on account type and CBUAE articles"""

        # Determine dormancy characteristics
        if "Demand Deposit" in scenario['purpose']:
            # Article 2.1.1: 12+ months inactive
            days_inactive = random.randint(400, 2000)  # 1-5 years
            last_transaction = current_date - timedelta(days=days_inactive)

            return {
                'account_status': 'ACTIVE',
                'dormancy_status': 'DORMANT' if days_inactive >= 365 else 'ACTIVE',
                'last_transaction_date': last_transaction.strftime('%Y-%m-%d'),
                'last_statement_date': (last_transaction + timedelta(days=30)).strftime('%Y-%m-%d'),
                'dormancy_trigger_date': (last_transaction + timedelta(days=365)).strftime(
                    '%Y-%m-%d') if days_inactive >= 365 else '',
                'dormancy_period_start': (last_transaction + timedelta(days=365)).strftime(
                    '%Y-%m-%d') if days_inactive >= 365 else '',
                'dormancy_period_months': max(0, int((current_date - last_transaction - timedelta(
                    days=365)).days / 30.44)) if days_inactive >= 365 else 0,
                'dormancy_classification_date': (last_transaction + timedelta(days=395)).strftime(
                    '%Y-%m-%d') if days_inactive >= 365 else '',
                'current_stage': random.choice(
                    ['FLAGGED', 'CONTACTED', 'WAITING']) if days_inactive >= 365 else 'ACTIVE',
                'contact_attempts_made': random.randint(0, 5) if days_inactive >= 365 else 0,
                'last_contact_attempt_date': (last_transaction + timedelta(days=random.randint(30, 200))).strftime(
                    '%Y-%m-%d') if days_inactive >= 365 else '',
                'waiting_period_start': (last_transaction + timedelta(days=random.randint(400, 600))).strftime(
                    '%Y-%m-%d') if days_inactive >= 730 else '',
                'waiting_period_end': '',
                'transferred_to_ledger_date': '',
                'transferred_to_cb_date': self.generate_cb_transfer_date(days_inactive, last_transaction),
                'cb_transfer_amount': self.generate_cb_transfer_amount(days_inactive),
                'cb_transfer_reference': self.generate_cb_transfer_ref(days_inactive),
                'transfer_eligibility_date': (last_transaction + timedelta(days=1460)).strftime(
                    '%Y-%m-%d') if days_inactive >= 1460 else '',  # 4 years
                'exclusion_reason': ''
            }

        elif "Fixed Deposit" in scenario['purpose']:
            # Article 2.2: Post-maturity dormancy
            days_inactive = random.randint(400, 1500)  # 1-4 years
            last_transaction = current_date - timedelta(days=days_inactive)
            maturity_date = last_transaction + timedelta(days=random.randint(180, 365))  # 6 months to 1 year term

            return {
                'account_status': 'DORMANT',
                'dormancy_status': 'DORMANT',
                'last_transaction_date': last_transaction.strftime('%Y-%m-%d'),
                'maturity_date': maturity_date.strftime('%Y-%m-%d'),
                'last_statement_date': maturity_date.strftime('%Y-%m-%d'),
                'dormancy_trigger_date': maturity_date.strftime('%Y-%m-%d'),
                'dormancy_period_start': maturity_date.strftime('%Y-%m-%d'),
                'dormancy_period_months': int((current_date - maturity_date).days / 30.44),
                'dormancy_classification_date': (maturity_date + timedelta(days=30)).strftime('%Y-%m-%d'),
                'current_stage': random.choice(['CONTACTED', 'WAITING', 'TRANSFER_READY']),
                'contact_attempts_made': random.randint(1, 4),
                'last_contact_attempt_date': (maturity_date + timedelta(days=random.randint(30, 100))).strftime(
                    '%Y-%m-%d'),
                'waiting_period_start': (maturity_date + timedelta(days=90)).strftime('%Y-%m-%d'),
                'waiting_period_end': (maturity_date + timedelta(days=180)).strftime('%Y-%m-%d'),
                'transferred_to_ledger_date': '',
                'transferred_to_cb_date': self.generate_cb_transfer_date(days_inactive, maturity_date),
                'cb_transfer_amount': self.generate_cb_transfer_amount(days_inactive),
                'cb_transfer_reference': self.generate_cb_transfer_ref(days_inactive),
                'transfer_eligibility_date': (maturity_date + timedelta(days=1095)).strftime('%Y-%m-%d'),
                # 3 years post-maturity
                'exclusion_reason': ''
            }

        elif "Investment" in scenario['purpose']:
            # Article 2.3: Investment dormancy
            days_inactive = random.randint(400, 1200)  # 1-3 years
            last_transaction = current_date - timedelta(days=days_inactive)

            return {
                'account_status': 'ACTIVE',
                'dormancy_status': 'DORMANT',
                'last_transaction_date': last_transaction.strftime('%Y-%m-%d'),
                'last_statement_date': (last_transaction + timedelta(days=30)).strftime('%Y-%m-%d'),
                'dormancy_trigger_date': (last_transaction + timedelta(days=365)).strftime('%Y-%m-%d'),
                'dormancy_period_start': (last_transaction + timedelta(days=365)).strftime('%Y-%m-%d'),
                'dormancy_period_months': max(0, int((current_date - last_transaction - timedelta(
                    days=365)).days / 30.44)),
                'dormancy_classification_date': (last_transaction + timedelta(days=395)).strftime('%Y-%m-%d'),
                'current_stage': random.choice(['CONTACTED', 'WAITING']),
                'contact_attempts_made': random.randint(1, 3),
                'last_contact_attempt_date': (last_transaction + timedelta(days=random.randint(30, 150))).strftime(
                    '%Y-%m-%d'),
                'waiting_period_start': (last_transaction + timedelta(days=random.randint(400, 500))).strftime(
                    '%Y-%m-%d'),
                'waiting_period_end': (last_transaction + timedelta(days=random.randint(600, 700))).strftime(
                    '%Y-%m-%d'),
                'transferred_to_ledger_date': '',
                'transferred_to_cb_date': '',
                'cb_transfer_amount': '',
                'cb_transfer_reference': '',
                'transfer_eligibility_date': (last_transaction + timedelta(days=1095)).strftime('%Y-%m-%d'),  # 3 years
                'exclusion_reason': ''
            }

        elif "Payment Instruments" in scenario['purpose']:
            # Article 2.4: Unclaimed payment instruments (1+ year)
            days_inactive = random.randint(400, 800)  # 1-2 years
            last_transaction = current_date - timedelta(days=days_inactive)

            return {
                'account_status': 'ACTIVE',
                'dormancy_status': 'DORMANT',
                'last_transaction_date': last_transaction.strftime('%Y-%m-%d'),
                'last_statement_date': (last_transaction + timedelta(days=30)).strftime('%Y-%m-%d'),
                'dormancy_trigger_date': (last_transaction + timedelta(days=365)).strftime('%Y-%m-%d'),
                'dormancy_period_start': (last_transaction + timedelta(days=365)).strftime('%Y-%m-%d'),
                'dormancy_period_months': max(0, int((current_date - last_transaction - timedelta(
                    days=365)).days / 30.44)),
                'dormancy_classification_date': (last_transaction + timedelta(days=395)).strftime('%Y-%m-%d'),
                'current_stage': 'FLAGGED',
                'contact_attempts_made': random.randint(1, 2),
                'last_contact_attempt_date': (last_transaction + timedelta(days=random.randint(30, 120))).strftime(
                    '%Y-%m-%d'),
                'waiting_period_start': (last_transaction + timedelta(days=400)).strftime('%Y-%m-%d'),
                'waiting_period_end': (last_transaction + timedelta(days=490)).strftime('%Y-%m-%d'),
                'transferred_to_ledger_date': '',
                'transferred_to_cb_date': '',
                'cb_transfer_amount': '',
                'cb_transfer_reference': '',
                'transfer_eligibility_date': (last_transaction + timedelta(days=730)).strftime('%Y-%m-%d'),  # 2 years
                'exclusion_reason': ''
            }

        elif "Safe Deposit" in scenario['purpose']:
            # Article 2.6: Safe deposit box fees (3+ years outstanding)
            days_inactive = random.randint(1095, 1460)  # 3-4 years
            last_transaction = current_date - timedelta(days=days_inactive)

            return {
                'account_status': 'ACTIVE',
                'dormancy_status': 'DORMANT',
                'last_transaction_date': last_transaction.strftime('%Y-%m-%d'),
                'last_statement_date': (last_transaction + timedelta(days=30)).strftime('%Y-%m-%d'),
                'dormancy_trigger_date': (last_transaction + timedelta(days=1095)).strftime('%Y-%m-%d'),
                # 3 years for SDB
                'dormancy_period_start': (last_transaction + timedelta(days=1095)).strftime('%Y-%m-%d'),
                'dormancy_period_months': max(0, int((current_date - last_transaction - timedelta(
                    days=1095)).days / 30.44)),
                'dormancy_classification_date': (last_transaction + timedelta(days=1125)).strftime('%Y-%m-%d'),
                'current_stage': 'CONTACTED',
                'contact_attempts_made': 1,  # SDB typically has limited contact attempts
                'last_contact_attempt_date': (last_transaction + timedelta(days=random.randint(30, 100))).strftime(
                    '%Y-%m-%d'),
                'waiting_period_start': (last_transaction + timedelta(days=1200)).strftime('%Y-%m-%d'),
                'waiting_period_end': (last_transaction + timedelta(days=1290)).strftime('%Y-%m-%d'),
                'transferred_to_ledger_date': '',
                'transferred_to_cb_date': '',
                'cb_transfer_amount': '',
                'cb_transfer_reference': '',
                'transfer_eligibility_date': '',  # SDB handled differently
                'exclusion_reason': 'SDB_SPECIAL_PROCESS'
            }

        else:
            # Default active account
            days_inactive = random.randint(1, 60)  # Recent activity
            last_transaction = current_date - timedelta(days=days_inactive)

            return {
                'account_status': 'ACTIVE',
                'dormancy_status': 'ACTIVE',
                'last_transaction_date': last_transaction.strftime('%Y-%m-%d'),
                'last_statement_date': (last_transaction + timedelta(days=30)).strftime('%Y-%m-%d'),
                'dormancy_trigger_date': '',
                'dormancy_period_start': '',
                'dormancy_period_months': 0,
                'dormancy_classification_date': '',
                'current_stage': 'ACTIVE',
                'contact_attempts_made': 0,
                'last_contact_attempt_date': '',
                'waiting_period_start': '',
                'waiting_period_end': '',
                'transferred_to_ledger_date': '',
                'transferred_to_cb_date': '',
                'cb_transfer_amount': '',
                'cb_transfer_reference': '',
                'transfer_eligibility_date': '',
                'exclusion_reason': ''
            }

    def generate_cb_transfer_date(self, days_inactive: int, reference_date: datetime) -> str:
        """Generate Central Bank transfer date for long-dormant accounts (5+ years)"""
        if days_inactive >= 1825:  # 5+ years
            if random.random() < 0.3:  # 30% chance of already transferred
                transfer_date = reference_date + timedelta(days=random.randint(1500, 1700))
                return transfer_date.strftime('%Y-%m-%d')
        return ''

    def generate_cb_transfer_amount(self, days_inactive: int) -> str:
        """Generate CB transfer amount for transferred accounts"""
        if days_inactive >= 1825 and random.random() < 0.3:  # 5+ years, 30% transferred
            return str(random.uniform(10000, 500000))  # Will be overridden by actual balance
        return ''

    def generate_cb_transfer_ref(self, days_inactive: int) -> str:
        """Generate CB transfer reference number"""
        if days_inactive >= 1825 and random.random() < 0.3:  # 5+ years, 30% transferred
            return f"CBT{datetime.now().year}{random.randint(100000, 999999)}"
        return ''

    def generate_account_data(self, customer_data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Generate realistic account information with dormancy scenarios"""

        # Select account scenario based on weights
        scenario = random.choices(
            self.account_scenarios,
            weights=[s['weight'] for s in self.account_scenarios]
        )[0]

        # Current date for calculations
        current_date = datetime.now()

        # Account opening date (1-10 years ago)
        opening_date = current_date - timedelta(days=random.randint(365, 3650))

        # Generate dormancy scenario based on account type
        dormancy_scenario = self.generate_dormancy_scenario(scenario, current_date)

        # Generate balance based on account type
        balance_current = self.generate_balance(scenario['type'], scenario['subtype'])
        balance_available = balance_current * random.uniform(0.85, 0.98)

        # CRITICAL: Ensure contact_attempts_made is properly set
        contact_attempts = dormancy_scenario.get('contact_attempts_made', 0)
        if contact_attempts == '':
            contact_attempts = 0

        # CRITICAL: Handle CB transfer data properly
        cb_transfer_date = dormancy_scenario.get('transferred_to_cb_date', '')
        cb_transfer_amount = ''
        cb_transfer_reference = ''

        if cb_transfer_date:  # If there's a CB transfer date, set the amount and reference
            cb_transfer_amount = balance_current
            cb_transfer_reference = dormancy_scenario.get('cb_transfer_reference',
                                                          f"CBT{current_date.year}{random.randint(100000, 999999)}")

        # Generate account-specific data
        account_data = {
            'account_id': f"ACC{index + 1:06d}",
            'account_type': scenario['type'],
            'account_subtype': scenario['subtype'],
            'account_name': f"{customer_data['full_name_en'].split()[0]} {scenario['subtype'].title()} Account",
            'currency': random.choice(self.currencies),
            'account_status': dormancy_scenario['account_status'],
            'dormancy_status': dormancy_scenario['dormancy_status'],
            'opening_date': opening_date.strftime('%Y-%m-%d'),
            'closing_date': dormancy_scenario.get('closing_date', ''),
            'last_transaction_date': dormancy_scenario['last_transaction_date'],
            'last_system_transaction_date': dormancy_scenario['last_transaction_date'],
            'balance_current': round(balance_current, 2),
            'balance_available': round(balance_available, 2),
            'balance_minimum': random.randint(500, 10000),
            'interest_rate': round(random.uniform(0.25, 5.0), 2),
            'interest_accrued': round(balance_current * random.uniform(0.001, 0.05), 2),
            'is_joint_account': random.choice(["YES", "NO"]),
            'joint_account_holders': random.randint(0, 3),
            'has_outstanding_facilities': random.choice(["YES", "NO"]),
            'maturity_date': dormancy_scenario.get('maturity_date', ''),
            'auto_renewal': random.choice(["YES", "NO"]) if scenario['type'] == 'FIXED_DEPOSIT' else '',
            'last_statement_date': dormancy_scenario.get('last_statement_date', ''),
            'statement_frequency': random.choice(self.statement_frequencies),
            'tracking_id': f"TRK{index + 1:06d}",

            # CRITICAL: Dormancy fields with proper data types
            'dormancy_trigger_date': dormancy_scenario.get('dormancy_trigger_date', ''),
            'dormancy_period_start': dormancy_scenario.get('dormancy_period_start', ''),
            'dormancy_period_months': dormancy_scenario.get('dormancy_period_months', 0),
            'dormancy_classification_date': dormancy_scenario.get('dormancy_classification_date', ''),
            'transfer_eligibility_date': dormancy_scenario.get('transfer_eligibility_date', ''),
            'current_stage': dormancy_scenario.get('current_stage', 'ACTIVE'),

            # CRITICAL: Contact fields with proper data types
            'contact_attempts_made': contact_attempts,
            'last_contact_attempt_date': dormancy_scenario.get('last_contact_attempt_date', ''),
            'waiting_period_start': dormancy_scenario.get('waiting_period_start', ''),
            'waiting_period_end': dormancy_scenario.get('waiting_period_end', ''),

            # CRITICAL: Transfer fields with proper data types
            'transferred_to_ledger_date': dormancy_scenario.get('transferred_to_ledger_date', ''),
            'transferred_to_cb_date': cb_transfer_date,
            'cb_transfer_amount': cb_transfer_amount,
            'cb_transfer_reference': cb_transfer_reference,
            'exclusion_reason': dormancy_scenario.get('exclusion_reason', '')
        }

        return account_data

    def generate_contact_data(self, dormancy_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Generate contact-related data"""

        current_date = datetime.now()

        # Determine if contact was attempted
        contact_attempts = dormancy_scenario.get('contact_attempts_made', 0)

        if contact_attempts > 0:
            # Generate last contact date
            last_contact = current_date - timedelta(days=random.randint(30, 365))
            last_contact_method = random.choice(self.contact_methods)
        else:
            last_contact = current_date - timedelta(days=random.randint(500, 1500))
            last_contact_method = random.choice(self.contact_methods)

        return {
            'last_contact_date': last_contact.strftime('%Y-%m-%d'),
            'last_contact_method': last_contact_method,
            'kyc_status': random.choice(["VALID", "EXPIRED", "PENDING", "COMPLIANT"]),
            'kyc_expiry_date': (current_date + timedelta(days=random.randint(30, 730))).strftime('%Y-%m-%d'),
            'risk_rating': random.choice(self.risk_ratings)
        }

    def generate_metadata(self) -> Dict[str, Any]:
        """Generate metadata fields"""

        current_date = datetime.now()

        return {
            'created_date': (current_date - timedelta(days=random.randint(1000, 3000))).strftime('%Y-%m-%d'),
            'updated_date': current_date.strftime('%Y-%m-%d'),
            'updated_by': random.choice(["SYSTEM", "ADMIN", "USER", "BATCH_PROCESS"])
        }

    def fix_data_types(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Fix data types to prevent validation errors"""

        # Ensure integer fields are integers
        integer_fields = [
            'contact_attempts_made', 'joint_account_holders', 'dormancy_period_months',
            'postal_code'
        ]

        for field in integer_fields:
            if field in record:
                try:
                    if record[field] == '' or record[field] is None:
                        record[field] = 0
                    else:
                        record[field] = int(float(record[field]))
                except (ValueError, TypeError):
                    record[field] = 0

        # Ensure float fields are floats
        float_fields = [
            'balance_current', 'balance_available', 'balance_minimum',
            'interest_rate', 'interest_accrued', 'cb_transfer_amount'
        ]

        for field in float_fields:
            if field in record:
                try:
                    if record[field] == '' or record[field] is None:
                        if field == 'cb_transfer_amount':
                            record[field] = ''  # CB transfer amount can be empty
                        else:
                            record[field] = 0.0
                    else:
                        record[field] = float(record[field])
                except (ValueError, TypeError):
                    if field == 'cb_transfer_amount':
                        record[field] = ''
                    else:
                        record[field] = 0.0

        # Ensure string fields are strings
        string_fields = [
            'customer_id', 'account_id', 'full_name_en', 'full_name_ar',
            'account_type', 'account_subtype', 'currency', 'current_stage'
        ]

        for field in string_fields:
            if field in record:
                if record[field] is None:
                    record[field] = ''
                else:
                    record[field] = str(record[field])

        return record

    def generate_single_record(self, index: int) -> Dict[str, Any]:
        """Generate a complete record with all 66 columns"""

        # Generate customer data
        customer_data = self.generate_customer_data(index)

        # Generate account data
        account_data = self.generate_account_data(customer_data, index)

        # Generate contact data
        contact_data = self.generate_contact_data(account_data)

        # Generate metadata
        metadata = self.generate_metadata()

        # Combine all data
        record = {**customer_data, **account_data, **contact_data, **metadata}

        # CRITICAL: Ensure all 66 columns are present with proper data types
        for header in self.column_headers:
            if header not in record:
                record[header] = self.get_default_value_for_column(header)

        # CRITICAL: Fix data type issues
        record = self.fix_data_types(record)

        return record

    def get_default_value_for_column(self, column_name: str) -> Any:
        """Get appropriate default value for missing columns"""

        # Numeric columns that should never be empty
        if column_name in ['contact_attempts_made', 'joint_account_holders', 'dormancy_period_months']:
            return 0

        # Balance columns
        elif column_name in ['balance_current', 'balance_available', 'balance_minimum', 'interest_rate',
                             'interest_accrued']:
            return 0.0

        # Date columns
        elif 'date' in column_name.lower():
            return ''

        # Text columns
        else:
            return ''

    def generate_dataset(self, output_file: str = None) -> pd.DataFrame:
        """Generate complete dataset with specified number of records"""

        print(f"🏛️  CBUAE Banking Compliance Dataset Generator")
        print(f"=" * 50)
        print(f"📊 Generating {self.num_records:,} records...")
        print(f"📋 Total columns: {len(self.column_headers)}")

        records = []

        for i in range(self.num_records):
            record = self.generate_single_record(i)
            records.append(record)

            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"✅ Generated {i + 1:,} records...")

        # Create DataFrame with strict column order and data types
        df = pd.DataFrame(records)

        # CRITICAL: Ensure exact column order (no extra spaces)
        df = df[self.column_headers]

        # CRITICAL: Fix data types to prevent validation errors
        df = self.fix_dataframe_types(df)

        # CRITICAL: Validate data quality
        validation_results = self.validate_generated_data(df)

        print(f"\n📈 Dataset Generation Complete!")
        print(f"📁 Records generated: {len(df):,}")
        print(f"📊 Columns: {len(df.columns)}")

        # Display validation results
        if validation_results['passed']:
            print(f"✅ Data validation: PASSED")
        else:
            print(f"⚠️  Data validation: FAILED")
            for issue in validation_results['issues']:
                print(f"   - {issue}")

        # Save to CSV if output file specified
        if output_file:
            # CRITICAL: Save with proper formatting
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"💾 Saved to: {output_file}")

        # Analyze dataset
        self.analyze_dataset(df)

        return df

    def fix_dataframe_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix DataFrame column types to match expected validation"""

        # Integer columns
        integer_columns = [
            'contact_attempts_made', 'joint_account_holders', 'dormancy_period_months'
        ]

        for col in integer_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # Float columns
        float_columns = [
            'balance_current', 'balance_available', 'balance_minimum',
            'interest_rate', 'interest_accrued'
        ]

        for col in float_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)

        # String columns (ensure no None values)
        string_columns = [
            'customer_id', 'account_id', 'account_type', 'dormancy_status',
            'current_stage', 'currency'
        ]

        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)

        return df

    def validate_generated_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the generated data quality"""

        issues = []

        # Check for missing critical columns
        critical_columns = [
            'customer_id', 'account_id', 'balance_current', 'contact_attempts_made'
        ]

        for col in critical_columns:
            if col not in df.columns:
                issues.append(f"Missing critical column: {col}")
            elif df[col].isnull().any():
                null_count = df[col].isnull().sum()
                issues.append(f"Column {col} has {null_count} null values")

        # Check data types
        if 'contact_attempts_made' in df.columns:
            if not pd.api.types.is_integer_dtype(df['contact_attempts_made']):
                issues.append("contact_attempts_made should be integer type")

        if 'balance_current' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['balance_current']):
                issues.append("balance_current should be numeric type")

        # Check for reasonable data ranges
        if 'contact_attempts_made' in df.columns:
            invalid_attempts = df[(df['contact_attempts_made'] < 0) | (df['contact_attempts_made'] > 10)]
            if len(invalid_attempts) > 0:
                issues.append(f"{len(invalid_attempts)} records have invalid contact_attempts_made values")

        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'total_records': len(df),
            'columns_validated': len(df.columns)
        }

    def analyze_dataset(self, df: pd.DataFrame):
        """Analyze the generated dataset for coverage"""

        print(f"\n🎯 DATASET ANALYSIS:")
        print(f"=" * 25)

        # Account type distribution
        print(f"\n📋 Account Types:")
        account_types = df['account_type'].value_counts()
        for acc_type, count in account_types.items():
            percentage = (count / len(df)) * 100
            print(f"   {acc_type}: {count:,} ({percentage:.1f}%)")

        # Dormancy status distribution
        print(f"\n🔍 Dormancy Status:")
        dormancy_status = df['dormancy_status'].value_counts()
        for status, count in dormancy_status.items():
            percentage = (count / len(df)) * 100
            print(f"   {status}: {count:,} ({percentage:.1f}%)")

        # Currency distribution
        print(f"\n💱 Currency Distribution:")
        currencies = df['currency'].value_counts()
        for currency, count in currencies.items():
            percentage = (count / len(df)) * 100
            print(f"   {currency}: {count:,} ({percentage:.1f}%)")

        # Balance analysis
        print(f"\n💰 Balance Analysis:")
        total_balance = df['balance_current'].sum()
        avg_balance = df['balance_current'].mean()
        print(f"   Total Balance: AED {total_balance:,.2f}")
        print(f"   Average Balance: AED {avg_balance:,.2f}")
        print(f"   High Value Accounts (>500K): {len(df[df['balance_current'] > 500000]):,}")

        # Contact attempts analysis
        print(f"\n📞 Contact Attempts Analysis:")
        avg_attempts = df['contact_attempts_made'].mean()
        no_attempts = len(df[df['contact_attempts_made'] == 0])
        print(f"   Average Attempts: {avg_attempts:.1f}")
        print(f"   No Attempts: {no_attempts:,} ({(no_attempts / len(df) * 100):.1f}%)")

        # CB Transfer analysis
        print(f"\n🏦 CB Transfer Analysis:")
        cb_transfers = len(df[df['transferred_to_cb_date'] != ''])
        print(f"   Accounts Transferred: {cb_transfers:,} ({(cb_transfers / len(df) * 100):.1f}%)")


def main():
    """Main function with command-line interface"""

    parser = argparse.ArgumentParser(
        description="Standalone CBUAE Banking Compliance CSV Generator"
    )
    parser.add_argument(
        '--rows', '-r',
        type=int,
        default=1000,
        help='Number of records to generate (default: 1000)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='cbuae_banking_compliance_dataset.csv',
        help='Output CSV filename (default: cbuae_banking_compliance_dataset.csv)'
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Generate a quick small dataset (100 records) for testing'
    )

    args = parser.parse_args()

    # Use quick mode if specified
    if args.quick:
        num_records = 100
        output_file = 'quick_test_data.csv'
        print("🚀 Quick Mode: Generating small test dataset")
    else:
        num_records = args.rows
        output_file = args.output

    # Create generator
    generator = StandaloneCBUAECSVGenerator(num_records=num_records)

    # Generate dataset
    df = generator.generate_dataset(output_file=output_file)

    print(f"\n🚀 USAGE INSTRUCTIONS:")
    print(f"=" * 22)
    print(f"1. Import the generated CSV into your CBUAE compliance system")
    print(f"2. Run dormancy analysis agents to test Article 2.1.1, 2.2, 2.3")
    print(f"3. Execute compliance verification agents for comprehensive testing")
    print(f"4. Test data processing and mapping functionality")
    print(f"5. Validate reporting and export capabilities")

    return df


if __name__ == "__main__":
    main()