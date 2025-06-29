"""
Complete Unified Banking Schema System - All 67 Columns
Handles: Data Mapping + Database Schema + Table Creation
Production-ready implementation for banking compliance with all required fields
"""

import json
import re
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CORE ENUMS AND DATA CLASSES
# =============================================================================

class FieldType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    UUID = "uuid"
    JSON = "json"


class DatabaseType(Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLSERVER = "sqlserver"
    ORACLE = "oracle"
    SQLITE = "sqlite"


class ConstraintType(Enum):
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    UNIQUE = "unique"
    NOT_NULL = "not_null"
    CHECK = "check"
    DEFAULT = "default"


class IndexType(Enum):
    BTREE = "btree"
    HASH = "hash"
    GIN = "gin"
    GIST = "gist"
    BRIN = "brin"


@dataclass
class ValidationRule:
    """Validation rule for field data"""
    type: str  # pattern, range, enum, custom
    value: Any
    error_message: str
    severity: str = "error"  # error, warning, info


@dataclass
class DatabaseConstraint:
    """Database constraint definition"""
    type: ConstraintType
    value: Any = None
    error_message: str = ""


@dataclass
class IndexDefinition:
    """Database index definition"""
    name: str
    columns: List[str]
    type: IndexType = IndexType.BTREE
    unique: bool = False
    where_clause: Optional[str] = None


@dataclass
class UnifiedFieldDefinition:
    """Unified field definition for all three purposes"""
    # Basic field information
    name: str
    description: str
    type: FieldType
    required: bool = False

    # Data mapping properties
    keywords: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    validation_rules: List[ValidationRule] = field(default_factory=list)

    # Database properties
    db_type_mapping: Dict[str, str] = field(default_factory=dict)  # db_type -> sql_type
    constraints: List[DatabaseConstraint] = field(default_factory=list)
    default_value: Any = None

    # Business properties
    category: str = "general"
    regulatory_requirement: Optional[str] = None
    business_rules: List[str] = field(default_factory=list)
    sensitive_data: bool = False

    # Metadata
    version: str = "1.0.0"
    created_date: Optional[datetime] = None

    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now()


@dataclass
class EntityDefinition:
    """Business entity definition"""
    name: str
    description: str
    table_name: str
    fields: List[str]  # Field names that belong to this entity
    primary_key: List[str]
    relationships: List[Dict] = field(default_factory=list)
    business_rules: List[str] = field(default_factory=list)
    indexes: List[IndexDefinition] = field(default_factory=list)
    partitioning: Optional[Dict] = None


@dataclass
class RelationshipDefinition:
    """Entity relationship definition"""
    name: str
    from_entity: str
    to_entity: str
    from_fields: List[str]
    to_fields: List[str]
    relationship_type: str  # one_to_one, one_to_many, many_to_many
    on_delete: str = "RESTRICT"
    on_update: str = "CASCADE"


# =============================================================================
# COMPLETE BANKING SCHEMA MANAGER
# =============================================================================

class CompleteBankingSchemaManager:
    """
    Complete schema manager with all 67 banking compliance fields
    """

    def __init__(self, schema_config_path: Optional[str] = None):
        self.schema_config_path = schema_config_path
        self.fields: Dict[str, UnifiedFieldDefinition] = {}
        self.entities: Dict[str, EntityDefinition] = {}
        self.relationships: List[RelationshipDefinition] = []
        self.db_type_mappings: Dict[DatabaseType, Dict[FieldType, str]] = {}

        # Initialize with complete banking schema
        self._initialize_complete_schema()
        self._initialize_db_type_mappings()
        self._initialize_entities()

        # Load custom schema if provided
        if schema_config_path:
            self.load_schema_from_file(schema_config_path)

    def _initialize_complete_schema(self):
        """Initialize with all 67 banking compliance fields"""

        # =============================================================================
        # CUSTOMER INFORMATION FIELDS (8 fields)
        # =============================================================================

        self.add_field(UnifiedFieldDefinition(
            name="customer_id",
            description="Unique customer identifier",
            type=FieldType.STRING,
            required=True,
            keywords=["customer", "client", "id", "identifier", "cust"],
            aliases=["client_id", "cust_id", "customer_number", "clientid"],
            validation_rules=[
                ValidationRule("pattern", r"^[A-Za-z0-9]{6,20}$", "Invalid customer ID format"),
                ValidationRule("custom", "unique_check", "Customer ID must be unique")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(20)",
                "mysql": "VARCHAR(20)",
                "sqlserver": "NVARCHAR(20)"
            },
            constraints=[
                DatabaseConstraint(ConstraintType.PRIMARY_KEY),
                DatabaseConstraint(ConstraintType.NOT_NULL)
            ],
            category="customer",
            regulatory_requirement="CBUAE Article 2.1",
            sensitive_data=True
        ))

        self.add_field(UnifiedFieldDefinition(
            name="customer_type",
            description="Type of customer (Individual/Corporate)",
            type=FieldType.STRING,
            required=True,
            keywords=["customer_type", "client_type", "type", "individual", "corporate"],
            aliases=["cust_type", "clienttype"],
            validation_rules=[
                ValidationRule("enum", ["individual", "corporate"], "Invalid customer type")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(20)",
                "mysql": "ENUM('individual', 'corporate')",
                "sqlserver": "NVARCHAR(20)"
            },
            constraints=[
                DatabaseConstraint(ConstraintType.NOT_NULL),
                DatabaseConstraint(ConstraintType.CHECK, "customer_type IN ('individual', 'corporate')")
            ],
            category="customer"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="full_name_en",
            description="Customer full name in English",
            type=FieldType.STRING,
            required=True,
            keywords=["name", "full_name", "customer_name", "client_name", "english"],
            aliases=["fullname", "customername", "clientname", "name_en"],
            validation_rules=[
                ValidationRule("range", {"min": 2, "max": 200}, "Name length must be 2-200 characters")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(200)",
                "mysql": "VARCHAR(200)",
                "sqlserver": "NVARCHAR(200)"
            },
            constraints=[DatabaseConstraint(ConstraintType.NOT_NULL)],
            category="customer",
            sensitive_data=True
        ))

        self.add_field(UnifiedFieldDefinition(
            name="full_name_ar",
            description="Customer full name in Arabic",
            type=FieldType.STRING,
            required=False,
            keywords=["name_ar", "full_name_ar", "arabic", "name_arabic"],
            aliases=["arabic_name", "customer_name_ar"],
            db_type_mapping={
                "postgresql": "VARCHAR(200)",
                "mysql": "VARCHAR(200) CHARACTER SET utf8mb4",
                "sqlserver": "NVARCHAR(200)"
            },
            category="customer",
            sensitive_data=True
        ))

        self.add_field(UnifiedFieldDefinition(
            name="id_number",
            description="Customer identification number",
            type=FieldType.STRING,
            required=True,
            keywords=["id_number", "identification", "passport", "emirates_id", "national_id"],
            aliases=["id_no", "identification_number", "document_number"],
            validation_rules=[
                ValidationRule("pattern", r"^[A-Za-z0-9\-]{8,20}$", "Invalid ID number format")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(50)",
                "mysql": "VARCHAR(50)",
                "sqlserver": "NVARCHAR(50)"
            },
            constraints=[DatabaseConstraint(ConstraintType.NOT_NULL)],
            category="customer",
            sensitive_data=True
        ))

        self.add_field(UnifiedFieldDefinition(
            name="id_type",
            description="Type of identification document",
            type=FieldType.STRING,
            required=True,
            keywords=["id_type", "document_type", "passport", "emirates_id", "visa"],
            aliases=["identification_type", "doc_type"],
            validation_rules=[
                ValidationRule("enum", ["emirates_id", "passport", "visa", "gcc_id"], "Invalid ID type")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(20)",
                "mysql": "ENUM('emirates_id', 'passport', 'visa', 'gcc_id')",
                "sqlserver": "NVARCHAR(20)"
            },
            constraints=[DatabaseConstraint(ConstraintType.NOT_NULL)],
            category="customer"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="date_of_birth",
            description="Customer date of birth",
            type=FieldType.DATE,
            required=False,
            keywords=["birth", "born", "dob", "date_of_birth"],
            aliases=["birth_date", "dob"],
            validation_rules=[
                ValidationRule("custom", "age_validation", "Customer must be at least 18 years old")
            ],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            category="customer",
            sensitive_data=True
        ))

        self.add_field(UnifiedFieldDefinition(
            name="nationality",
            description="Customer nationality",
            type=FieldType.STRING,
            required=False,
            keywords=["nationality", "citizen", "country", "origin"],
            aliases=["citizenship", "nation"],
            validation_rules=[
                ValidationRule("pattern", r"^[A-Z]{2,3}$", "Nationality code must be 2-3 uppercase letters")
            ],
            db_type_mapping={
                "postgresql": "CHAR(3)",
                "mysql": "CHAR(3)",
                "sqlserver": "CHAR(3)"
            },
            category="customer"
        ))

        # =============================================================================
        # ADDRESS INFORMATION FIELDS (6 fields)
        # =============================================================================

        self.add_field(UnifiedFieldDefinition(
            name="address_line1",
            description="Primary address line",
            type=FieldType.STRING,
            required=False,
            keywords=["address", "address_line1", "street", "location"],
            aliases=["addr1", "street_address", "primary_address"],
            db_type_mapping={
                "postgresql": "VARCHAR(200)",
                "mysql": "VARCHAR(200)",
                "sqlserver": "NVARCHAR(200)"
            },
            category="address",
            sensitive_data=True
        ))

        self.add_field(UnifiedFieldDefinition(
            name="address_line2",
            description="Secondary address line",
            type=FieldType.STRING,
            required=False,
            keywords=["address_line2", "apartment", "suite", "floor"],
            aliases=["addr2", "apartment", "suite"],
            db_type_mapping={
                "postgresql": "VARCHAR(200)",
                "mysql": "VARCHAR(200)",
                "sqlserver": "NVARCHAR(200)"
            },
            category="address",
            sensitive_data=True
        ))

        self.add_field(UnifiedFieldDefinition(
            name="city",
            description="City of residence",
            type=FieldType.STRING,
            required=False,
            keywords=["city", "town", "municipality"],
            aliases=["location_city", "residence_city"],
            db_type_mapping={
                "postgresql": "VARCHAR(100)",
                "mysql": "VARCHAR(100)",
                "sqlserver": "NVARCHAR(100)"
            },
            category="address"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="emirate",
            description="UAE Emirate",
            type=FieldType.STRING,
            required=False,
            keywords=["emirate", "state", "province"],
            aliases=["emirate_name", "state"],
            validation_rules=[
                ValidationRule("enum", ["abu_dhabi", "dubai", "sharjah", "ajman", "fujairah", "rak", "uaq"],
                              "Invalid emirate")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(50)",
                "mysql": "VARCHAR(50)",
                "sqlserver": "NVARCHAR(50)"
            },
            category="address"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="country",
            description="Country of residence",
            type=FieldType.STRING,
            required=False,
            keywords=["country", "nation", "residence_country"],
            aliases=["country_code", "nation_code"],
            validation_rules=[
                ValidationRule("pattern", r"^[A-Z]{2}$", "Country code must be 2 uppercase letters")
            ],
            db_type_mapping={
                "postgresql": "CHAR(2)",
                "mysql": "CHAR(2)",
                "sqlserver": "CHAR(2)"
            },
            category="address"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="postal_code",
            description="Postal/ZIP code",
            type=FieldType.STRING,
            required=False,
            keywords=["postal", "zip", "code", "postal_code"],
            aliases=["zip_code", "post_code"],
            validation_rules=[
                ValidationRule("pattern", r"^[A-Za-z0-9\s\-]{3,10}$", "Invalid postal code format")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(10)",
                "mysql": "VARCHAR(10)",
                "sqlserver": "NVARCHAR(10)"
            },
            category="address"
        ))

        # =============================================================================
        # CONTACT INFORMATION FIELDS (7 fields)
        # =============================================================================

        self.add_field(UnifiedFieldDefinition(
            name="phone_primary",
            description="Primary phone number",
            type=FieldType.STRING,
            required=False,
            keywords=["phone", "mobile", "telephone", "contact", "primary_phone"],
            aliases=["phone_primary", "mobile_number", "contact_number"],
            validation_rules=[
                ValidationRule("pattern", r"^\+?[1-9]\d{1,14}$", "Invalid phone format")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(20)",
                "mysql": "VARCHAR(20)",
                "sqlserver": "NVARCHAR(20)"
            },
            category="contact",
            sensitive_data=True
        ))

        self.add_field(UnifiedFieldDefinition(
            name="phone_secondary",
            description="Secondary phone number",
            type=FieldType.STRING,
            required=False,
            keywords=["phone_secondary", "second", "alternative", "backup"],
            aliases=["phone2", "secondary_phone", "alt_phone"],
            validation_rules=[
                ValidationRule("pattern", r"^\+?[1-9]\d{1,14}$", "Invalid phone format")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(20)",
                "mysql": "VARCHAR(20)",
                "sqlserver": "NVARCHAR(20)"
            },
            category="contact",
            sensitive_data=True
        ))

        self.add_field(UnifiedFieldDefinition(
            name="email_primary",
            description="Primary email address",
            type=FieldType.STRING,
            required=False,
            keywords=["email", "email_address", "contact_email", "primary_email"],
            aliases=["email_primary", "primary_email_address"],
            validation_rules=[
                ValidationRule("pattern", r"^[^@]+@[^@]+\.[^@]+$", "Invalid email format")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(255)",
                "mysql": "VARCHAR(255)",
                "sqlserver": "NVARCHAR(255)"
            },
            category="contact",
            sensitive_data=True
        ))

        self.add_field(UnifiedFieldDefinition(
            name="email_secondary",
            description="Secondary email address",
            type=FieldType.STRING,
            required=False,
            keywords=["email_secondary", "second", "alternative", "backup"],
            aliases=["email2", "secondary_email", "alt_email"],
            validation_rules=[
                ValidationRule("pattern", r"^[^@]+@[^@]+\.[^@]+$", "Invalid email format")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(255)",
                "mysql": "VARCHAR(255)",
                "sqlserver": "NVARCHAR(255)"
            },
            category="contact",
            sensitive_data=True
        ))

        self.add_field(UnifiedFieldDefinition(
            name="address_known",
            description="Whether customer address is known/verified",
            type=FieldType.BOOLEAN,
            required=False,
            keywords=["address_known", "verified", "confirmed"],
            aliases=["address_verified", "known_address"],
            db_type_mapping={
                "postgresql": "BOOLEAN",
                "mysql": "BOOLEAN",
                "sqlserver": "BIT"
            },
            constraints=[DatabaseConstraint(ConstraintType.DEFAULT, "false")],
            category="contact"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="last_contact_date",
            description="Date of last customer contact",
            type=FieldType.DATE,
            required=False,
            keywords=["last_contact", "contact_date", "communication"],
            aliases=["last_communication", "contact_dt"],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            category="contact",
            regulatory_requirement="CBUAE Article 3.1"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="last_contact_method",
            description="Method of last contact attempt",
            type=FieldType.STRING,
            required=False,
            keywords=["contact_method", "method", "phone", "email", "letter"],
            aliases=["communication_method", "contact_type"],
            validation_rules=[
                ValidationRule("enum", ["phone", "email", "sms", "letter", "in_person", "system"],
                              "Invalid contact method")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(20)",
                "mysql": "ENUM('phone', 'email', 'sms', 'letter', 'in_person', 'system')",
                "sqlserver": "NVARCHAR(20)"
            },
            category="contact"
        ))

        # =============================================================================
        # KYC & RISK FIELDS (3 fields)
        # =============================================================================

        self.add_field(UnifiedFieldDefinition(
            name="kyc_status",
            description="Know Your Customer status",
            type=FieldType.STRING,
            required=False,
            keywords=["kyc", "know_your_customer", "verification"],
            aliases=["kyc_verified", "verification_status"],
            validation_rules=[
                ValidationRule("enum", ["complete", "incomplete", "pending", "expired"], "Invalid KYC status")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(20)",
                "mysql": "ENUM('complete', 'incomplete', 'pending', 'expired')",
                "sqlserver": "NVARCHAR(20)"
            },
            category="kyc_risk",
            regulatory_requirement="AML/CFT Requirements"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="kyc_expiry_date",
            description="KYC documentation expiry date",
            type=FieldType.DATE,
            required=False,
            keywords=["kyc_expiry", "expiry", "expire", "valid_until"],
            aliases=["kyc_expires", "verification_expiry"],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            category="kyc_risk",
            regulatory_requirement="AML/CFT Requirements"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="risk_rating",
            description="Customer risk rating",
            type=FieldType.STRING,
            required=False,
            keywords=["risk", "rating", "assessment", "level"],
            aliases=["risk_level", "risk_score"],
            validation_rules=[
                ValidationRule("enum", ["low", "medium", "high", "very_high"], "Invalid risk rating")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(20)",
                "mysql": "ENUM('low', 'medium', 'high', 'very_high')",
                "sqlserver": "NVARCHAR(20)"
            },
            category="kyc_risk",
            regulatory_requirement="AML/CFT Requirements"
        ))

        # =============================================================================
        # ACCOUNT BASIC FIELDS (6 fields)
        # =============================================================================

        self.add_field(UnifiedFieldDefinition(
            name="account_id",
            description="Unique account identifier",
            type=FieldType.STRING,
            required=True,
            keywords=["account", "account_id", "account_number", "acc_id"],
            aliases=["accountid", "acc_no", "account_no"],
            validation_rules=[
                ValidationRule("pattern", r"^[A-Za-z0-9]{8,20}$", "Invalid account ID format")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(20)",
                "mysql": "VARCHAR(20)",
                "sqlserver": "NVARCHAR(20)"
            },
            constraints=[
                DatabaseConstraint(ConstraintType.PRIMARY_KEY),
                DatabaseConstraint(ConstraintType.NOT_NULL)
            ],
            category="account"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="account_type",
            description="Type of account (Savings/Current/Fixed)",
            type=FieldType.STRING,
            required=True,
            keywords=["account_type", "type", "savings", "current", "fixed"],
            aliases=["acc_type", "accounttype", "product_type"],
            validation_rules=[
                ValidationRule("enum", ["savings", "current", "fixed_deposit", "credit"], "Invalid account type")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(50)",
                "mysql": "ENUM('savings', 'current', 'fixed_deposit', 'credit')",
                "sqlserver": "NVARCHAR(50)"
            },
            constraints=[DatabaseConstraint(ConstraintType.NOT_NULL)],
            category="account"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="account_subtype",
            description="Account subtype or product variant",
            type=FieldType.STRING,
            required=False,
            keywords=["subtype", "variant", "product", "subcategory"],
            aliases=["product_subtype", "account_variant"],
            db_type_mapping={
                "postgresql": "VARCHAR(50)",
                "mysql": "VARCHAR(50)",
                "sqlserver": "NVARCHAR(50)"
            },
            category="account"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="account_name",
            description="Account name or title",
            type=FieldType.STRING,
            required=False,
            keywords=["account_name", "title", "description"],
            aliases=["account_title", "name"],
            db_type_mapping={
                "postgresql": "VARCHAR(200)",
                "mysql": "VARCHAR(200)",
                "sqlserver": "NVARCHAR(200)"
            },
            category="account"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="currency",
            description="Account currency code",
            type=FieldType.STRING,
            required=True,
            keywords=["currency", "currency_code", "ccy"],
            aliases=["ccy", "currency_cd"],
            validation_rules=[
                ValidationRule("pattern", r"^[A-Z]{3}$", "Currency must be 3-letter ISO code"),
                ValidationRule("enum", ["AED", "USD", "EUR", "GBP", "SAR"], "Unsupported currency")
            ],
            db_type_mapping={
                "postgresql": "CHAR(3)",
                "mysql": "CHAR(3)",
                "sqlserver": "CHAR(3)"
            },
            constraints=[DatabaseConstraint(ConstraintType.NOT_NULL)],
            category="account"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="account_status",
            description="Current status of account",
            type=FieldType.STRING,
            required=True,
            keywords=["status", "account_status", "active", "inactive", "closed"],
            aliases=["acc_status", "state", "account_state"],
            validation_rules=[
                ValidationRule("enum", ["active", "inactive", "closed", "suspended"], "Invalid account status")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(20)",
                "mysql": "ENUM('active', 'inactive', 'closed', 'suspended')",
                "sqlserver": "NVARCHAR(20)"
            },
            constraints=[DatabaseConstraint(ConstraintType.NOT_NULL)],
            category="account"
        ))

        # =============================================================================
        # DORMANCY STATUS FIELD (1 field)
        # =============================================================================

        self.add_field(UnifiedFieldDefinition(
            name="dormancy_status",
            description="Dormancy classification status",
            type=FieldType.STRING,
            required=False,
            keywords=["dormancy", "dormant", "dormancy_status", "classification"],
            aliases=["dormant_status", "dormancy_class"],
            validation_rules=[
                ValidationRule("enum", ["active", "dormant", "pre_dormant"], "Invalid dormancy status")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(20)",
                "mysql": "ENUM('active', 'dormant', 'pre_dormant')",
                "sqlserver": "NVARCHAR(20)"
            },
            category="dormancy",
            regulatory_requirement="CBUAE Article 3"
        ))

        # =============================================================================
        # ACCOUNT DATES FIELDS (4 fields)
        # =============================================================================

        self.add_field(UnifiedFieldDefinition(
            name="opening_date",
            description="Account opening date",
            type=FieldType.DATE,
            required=False,
            keywords=["opening_date", "opened", "start_date", "creation_date"],
            aliases=["open_date", "created_date", "account_opened"],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            category="account"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="closing_date",
            description="Account closing date",
            type=FieldType.DATE,
            required=False,
            keywords=["closing_date", "closed", "end_date", "closure"],
            aliases=["close_date", "closure_date", "account_closed"],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            category="account"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="last_transaction_date",
            description="Date of last customer transaction",
            type=FieldType.DATE,
            required=True,
            keywords=["transaction_date", "last_transaction", "activity_date", "last_activity"],
            aliases=["lasttransaction", "transaction_dt", "activity_dt"],
            validation_rules=[
                ValidationRule("custom", "date_not_future", "Transaction date cannot be in future")
            ],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            constraints=[DatabaseConstraint(ConstraintType.NOT_NULL)],
            category="account",
            regulatory_requirement="CBUAE Article 3.2"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="last_system_transaction_date",
            description="Date of last system-generated transaction",
            type=FieldType.DATE,
            required=False,
            keywords=["system_transaction", "automated", "system", "auto"],
            aliases=["last_auto_transaction", "system_activity"],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            category="account"
        ))

        # =============================================================================
        # ACCOUNT FINANCIAL FIELDS (5 fields)
        # =============================================================================

        self.add_field(UnifiedFieldDefinition(
            name="balance_current",
            description="Current account balance",
            type=FieldType.DECIMAL,
            required=True,
            keywords=["balance", "current_balance", "amount", "bal"],
            aliases=["currentbalance", "balance_current", "account_balance"],
            validation_rules=[
                ValidationRule("range", {"min": -1000000, "max": 1000000000}, "Balance out of range")
            ],
            db_type_mapping={
                "postgresql": "DECIMAL(15,2)",
                "mysql": "DECIMAL(15,2)",
                "sqlserver": "DECIMAL(15,2)"
            },
            constraints=[
                DatabaseConstraint(ConstraintType.NOT_NULL),
                DatabaseConstraint(ConstraintType.DEFAULT, "0.00")
            ],
            category="account"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="balance_available",
            description="Available balance for transactions",
            type=FieldType.DECIMAL,
            required=False,
            keywords=["available", "available_balance", "usable"],
            aliases=["available_bal", "usable_balance"],
            db_type_mapping={
                "postgresql": "DECIMAL(15,2)",
                "mysql": "DECIMAL(15,2)",
                "sqlserver": "DECIMAL(15,2)"
            },
            category="account"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="balance_minimum",
            description="Minimum required balance",
            type=FieldType.DECIMAL,
            required=False,
            keywords=["minimum", "min_balance", "required"],
            aliases=["min_bal", "minimum_balance"],
            db_type_mapping={
                "postgresql": "DECIMAL(15,2)",
                "mysql": "DECIMAL(15,2)",
                "sqlserver": "DECIMAL(15,2)"
            },
            category="account"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="interest_rate",
            description="Interest rate applicable to account",
            type=FieldType.DECIMAL,
            required=False,
            keywords=["interest", "rate", "percentage"],
            aliases=["int_rate", "rate"],
            validation_rules=[
                ValidationRule("range", {"min": 0, "max": 100}, "Interest rate must be 0-100%")
            ],
            db_type_mapping={
                "postgresql": "DECIMAL(5,4)",
                "mysql": "DECIMAL(5,4)",
                "sqlserver": "DECIMAL(5,4)"
            },
            category="account"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="interest_accrued",
            description="Accrued interest amount",
            type=FieldType.DECIMAL,
            required=False,
            keywords=["accrued", "earned", "interest_earned"],
            aliases=["accrued_interest", "earned_interest"],
            db_type_mapping={
                "postgresql": "DECIMAL(15,2)",
                "mysql": "DECIMAL(15,2)",
                "sqlserver": "DECIMAL(15,2)"
            },
            category="account"
        ))

        # =============================================================================
        # ACCOUNT FEATURES FIELDS (5 fields)
        # =============================================================================

        self.add_field(UnifiedFieldDefinition(
            name="is_joint_account",
            description="Whether account is jointly held",
            type=FieldType.BOOLEAN,
            required=False,
            keywords=["joint", "shared", "multiple_holders"],
            aliases=["joint_account", "shared_account"],
            db_type_mapping={
                "postgresql": "BOOLEAN",
                "mysql": "BOOLEAN",
                "sqlserver": "BIT"
            },
            constraints=[DatabaseConstraint(ConstraintType.DEFAULT, "false")],
            category="account"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="joint_account_holders",
            description="List of joint account holders",
            type=FieldType.STRING,
            required=False,
            keywords=["joint_holders", "co_holders", "secondary_holders"],
            aliases=["co_holders", "additional_holders"],
            db_type_mapping={
                "postgresql": "TEXT",
                "mysql": "TEXT",
                "sqlserver": "NVARCHAR(MAX)"
            },
            category="account",
            sensitive_data=True
        ))

        self.add_field(UnifiedFieldDefinition(
            name="has_outstanding_facilities",
            description="Whether account has outstanding credit facilities",
            type=FieldType.BOOLEAN,
            required=False,
            keywords=["facilities", "credit", "loan", "outstanding"],
            aliases=["credit_facilities", "loans"],
            db_type_mapping={
                "postgresql": "BOOLEAN",
                "mysql": "BOOLEAN",
                "sqlserver": "BIT"
            },
            constraints=[DatabaseConstraint(ConstraintType.DEFAULT, "false")],
            category="account"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="maturity_date",
            description="Account maturity date (for fixed deposits)",
            type=FieldType.DATE,
            required=False,
            keywords=["maturity", "expiry", "term_end"],
            aliases=["maturity_dt", "term_end"],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            category="account"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="auto_renewal",
            description="Whether account auto-renews at maturity",
            type=FieldType.BOOLEAN,
            required=False,
            keywords=["auto_renewal", "renewal", "automatic"],
            aliases=["auto_renew", "automatic_renewal"],
            db_type_mapping={
                "postgresql": "BOOLEAN",
                "mysql": "BOOLEAN",
                "sqlserver": "BIT"
            },
            constraints=[DatabaseConstraint(ConstraintType.DEFAULT, "false")],
            category="account"
        ))

        # =============================================================================
        # STATEMENTS FIELDS (2 fields)
        # =============================================================================

        self.add_field(UnifiedFieldDefinition(
            name="last_statement_date",
            description="Date of last statement generation",
            type=FieldType.DATE,
            required=False,
            keywords=["statement", "last_statement", "statement_date"],
            aliases=["statement_dt", "last_stmt"],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            category="statements"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="statement_frequency",
            description="Frequency of statement generation",
            type=FieldType.STRING,
            required=False,
            keywords=["frequency", "statement_frequency", "periodic"],
            aliases=["stmt_frequency", "statement_cycle"],
            validation_rules=[
                ValidationRule("enum", ["monthly", "quarterly", "annually", "on_demand"],
                              "Invalid statement frequency")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(20)",
                "mysql": "ENUM('monthly', 'quarterly', 'annually', 'on_demand')",
                "sqlserver": "NVARCHAR(20)"
            },
            category="statements"
        ))

        # =============================================================================
        # DORMANCY TRACKING FIELDS (5 fields)
        # =============================================================================

        self.add_field(UnifiedFieldDefinition(
            name="tracking_id",
            description="Unique dormancy tracking identifier",
            type=FieldType.STRING,
            required=False,
            keywords=["tracking", "tracking_id", "dormancy_id"],
            aliases=["dormancy_tracking_id", "track_id"],
            db_type_mapping={
                "postgresql": "VARCHAR(50)",
                "mysql": "VARCHAR(50)",
                "sqlserver": "NVARCHAR(50)"
            },
            category="dormancy_tracking"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="dormancy_trigger_date",
            description="Date when dormancy was first triggered",
            type=FieldType.DATE,
            required=False,
            keywords=["trigger", "dormancy_trigger", "start"],
            aliases=["dormancy_start", "trigger_dt"],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            category="dormancy_tracking",
            regulatory_requirement="CBUAE Article 3"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="dormancy_period_start",
            description="Start date of dormancy period",
            type=FieldType.DATE,
            required=False,
            keywords=["period_start", "dormancy_start", "begin"],
            aliases=["dormancy_begin", "period_begin"],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            category="dormancy_tracking"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="dormancy_period_months",
            description="Number of months in dormancy",
            type=FieldType.INTEGER,
            required=False,
            keywords=["months", "period", "duration"],
            aliases=["dormancy_months", "duration_months"],
            validation_rules=[
                ValidationRule("range", {"min": 0, "max": 1200}, "Dormancy period must be 0-1200 months")
            ],
            db_type_mapping={
                "postgresql": "INTEGER",
                "mysql": "INT",
                "sqlserver": "INT"
            },
            category="dormancy_tracking"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="dormancy_classification_date",
            description="Date of dormancy classification",
            type=FieldType.DATE,
            required=False,
            keywords=["classification", "classified", "determination"],
            aliases=["classification_dt", "determined_date"],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            category="dormancy_tracking"
        ))

        # =============================================================================
        # TRANSFER PROCESS FIELDS (6 fields)
        # =============================================================================

        self.add_field(UnifiedFieldDefinition(
            name="transfer_eligibility_date",
            description="Date when transfer eligibility was determined",
            type=FieldType.DATE,
            required=False,
            keywords=["eligibility", "transfer_eligible", "eligible"],
            aliases=["eligible_date", "eligibility_dt"],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            category="transfer_process"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="current_stage",
            description="Current stage in dormancy process",
            type=FieldType.STRING,
            required=False,
            keywords=["stage", "phase", "step", "current"],
            aliases=["process_stage", "workflow_stage"],
            validation_rules=[
                ValidationRule("enum", ["identified", "contact_attempts", "waiting_period", "transfer_ready",
                              "transferred", "excluded"], "Invalid process stage")
            ],
            db_type_mapping={
                "postgresql": "VARCHAR(30)",
                "mysql": "VARCHAR(30)",
                "sqlserver": "NVARCHAR(30)"
            },
            category="transfer_process"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="contact_attempts_made",
            description="Number of contact attempts made",
            type=FieldType.INTEGER,
            required=False,
            keywords=["attempts", "contact", "tried"],
            aliases=["attempt_count", "contact_count"],
            validation_rules=[
                ValidationRule("range", {"min": 0, "max": 10}, "Contact attempts must be 0-10")
            ],
            db_type_mapping={
                "postgresql": "INTEGER",
                "mysql": "INT",
                "sqlserver": "INT"
            },
            category="transfer_process",
            regulatory_requirement="CBUAE Article 3.1"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="last_contact_attempt_date",
            description="Date of last contact attempt",
            type=FieldType.DATE,
            required=False,
            keywords=["last_attempt", "contact_attempt", "recent"],
            aliases=["last_attempt_dt", "recent_contact"],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            category="transfer_process"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="waiting_period_start",
            description="Start date of waiting period",
            type=FieldType.DATE,
            required=False,
            keywords=["waiting", "wait_start", "period"],
            aliases=["wait_period_start", "waiting_start"],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            category="transfer_process",
            regulatory_requirement="CBUAE Article 3.3"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="waiting_period_end",
            description="End date of waiting period",
            type=FieldType.DATE,
            required=False,
            keywords=["waiting_end", "wait_end", "period_end"],
            aliases=["wait_period_end", "waiting_end"],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            category="transfer_process"
        ))

        # =============================================================================
        # CB TRANSFER FIELDS (5 fields)
        # =============================================================================

        self.add_field(UnifiedFieldDefinition(
            name="transferred_to_ledger_date",
            description="Date transferred to dormancy ledger",
            type=FieldType.DATE,
            required=False,
            keywords=["ledger", "transferred", "dormancy_ledger"],
            aliases=["ledger_transfer_date", "ledger_dt"],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            category="cb_transfer"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="transferred_to_cb_date",
            description="Date transferred to Central Bank",
            type=FieldType.DATE,
            required=False,
            keywords=["cb_transfer", "central_bank", "transferred"],
            aliases=["cb_transfer_date", "cb_dt"],
            db_type_mapping={
                "postgresql": "DATE",
                "mysql": "DATE",
                "sqlserver": "DATE"
            },
            category="cb_transfer",
            regulatory_requirement="CBUAE Article 3.4"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="cb_transfer_amount",
            description="Amount transferred to Central Bank",
            type=FieldType.DECIMAL,
            required=False,
            keywords=["transfer_amount", "cb_amount", "transferred"],
            aliases=["cb_amount", "transfer_amt"],
            validation_rules=[
                ValidationRule("range", {"min": 0, "max": 1000000000}, "Transfer amount out of range")
            ],
            db_type_mapping={
                "postgresql": "DECIMAL(15,2)",
                "mysql": "DECIMAL(15,2)",
                "sqlserver": "DECIMAL(15,2)"
            },
            category="cb_transfer"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="cb_transfer_reference",
            description="Central Bank transfer reference number",
            type=FieldType.STRING,
            required=False,
            keywords=["reference", "cb_reference", "transfer_ref"],
            aliases=["cb_ref", "transfer_reference"],
            db_type_mapping={
                "postgresql": "VARCHAR(50)",
                "mysql": "VARCHAR(50)",
                "sqlserver": "NVARCHAR(50)"
            },
            category="cb_transfer"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="exclusion_reason",
            description="Reason for exclusion from transfer",
            type=FieldType.STRING,
            required=False,
            keywords=["exclusion", "excluded", "reason"],
            aliases=["exclude_reason", "not_transferred"],
            db_type_mapping={
                "postgresql": "TEXT",
                "mysql": "TEXT",
                "sqlserver": "NVARCHAR(MAX)"
            },
            category="cb_transfer"
        ))

        # =============================================================================
        # SYSTEM FIELDS (3 fields)
        # =============================================================================

        self.add_field(UnifiedFieldDefinition(
            name="created_date",
            description="Record creation timestamp",
            type=FieldType.DATETIME,
            required=False,
            keywords=["created", "created_date", "creation_date", "timestamp"],
            aliases=["created_at", "date_created", "record_created"],
            db_type_mapping={
                "postgresql": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "mysql": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "sqlserver": "DATETIME2 DEFAULT CURRENT_TIMESTAMP"
            },
            constraints=[DatabaseConstraint(ConstraintType.DEFAULT, "CURRENT_TIMESTAMP")],
            category="system"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="updated_date",
            description="Record last update timestamp",
            type=FieldType.DATETIME,
            required=False,
            keywords=["updated", "updated_date", "modified", "last_modified"],
            aliases=["updated_at", "date_updated", "last_update"],
            db_type_mapping={
                "postgresql": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "mysql": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP",
                "sqlserver": "DATETIME2 DEFAULT CURRENT_TIMESTAMP"
            },
            category="system"
        ))

        self.add_field(UnifiedFieldDefinition(
            name="updated_by",
            description="User or system that last updated the record",
            type=FieldType.STRING,
            required=False,
            keywords=["updated_by", "modified_by", "user"],
            aliases=["last_updated_by", "modifier"],
            db_type_mapping={
                "postgresql": "VARCHAR(100)",
                "mysql": "VARCHAR(100)",
                "sqlserver": "NVARCHAR(100)"
            },
            category="system"
        ))

    def _initialize_entities(self):
        """Initialize comprehensive business entities"""
        # Customer entity with all customer fields
        customer_fields = [
            "customer_id", "customer_type", "full_name_en", "full_name_ar",
            "id_number", "id_type", "date_of_birth", "nationality",
            "address_line1", "address_line2", "city", "emirate", "country", "postal_code",
            "phone_primary", "phone_secondary", "email_primary", "email_secondary",
            "address_known", "last_contact_date", "last_contact_method",
            "kyc_status", "kyc_expiry_date", "risk_rating",
            "created_date", "updated_date", "updated_by"
        ]

        self.add_entity(EntityDefinition(
            name="Customer",
            description="Comprehensive customer master entity",
            table_name="customers",
            fields=customer_fields,
            primary_key=["customer_id"],
            indexes=[
                IndexDefinition("idx_customer_type", ["customer_type"]),
                IndexDefinition("idx_customer_email", ["email_primary"], unique=True),
                IndexDefinition("idx_customer_id_number", ["id_number"], unique=True),
                IndexDefinition("idx_customer_name", ["full_name_en"], type=IndexType.GIN),
                IndexDefinition("idx_customer_kyc", ["kyc_status", "kyc_expiry_date"]),
                IndexDefinition("idx_customer_risk", ["risk_rating"])
            ]
        ))

        # Account entity with all account fields
        account_fields = [
            "account_id", "customer_id", "account_type", "account_subtype", "account_name",
            "currency", "account_status", "dormancy_status",
            "opening_date", "closing_date", "last_transaction_date", "last_system_transaction_date",
            "balance_current", "balance_available", "balance_minimum", "interest_rate", "interest_accrued",
            "is_joint_account", "joint_account_holders", "has_outstanding_facilities",
            "maturity_date", "auto_renewal", "last_statement_date", "statement_frequency",
            "created_date", "updated_date", "updated_by"
        ]

        self.add_entity(EntityDefinition(
            name="Account",
            description="Comprehensive account master entity",
            table_name="accounts",
            fields=account_fields,
            primary_key=["account_id"],
            indexes=[
                IndexDefinition("idx_account_customer", ["customer_id"]),
                IndexDefinition("idx_account_status", ["account_status", "account_type"]),
                IndexDefinition("idx_account_dormancy", ["dormancy_status"]),
                IndexDefinition("idx_account_balance", ["balance_current"]),
                IndexDefinition("idx_account_currency", ["currency"]),
                IndexDefinition("idx_account_last_transaction", ["last_transaction_date"]),
                IndexDefinition("idx_account_maturity", ["maturity_date"])
            ]
        ))

        # Dormancy Tracking entity
        dormancy_fields = [
            "account_id", "tracking_id", "dormancy_trigger_date", "dormancy_period_start",
            "dormancy_period_months", "dormancy_classification_date",
            "transfer_eligibility_date", "current_stage", "contact_attempts_made",
            "last_contact_attempt_date", "waiting_period_start", "waiting_period_end",
            "transferred_to_ledger_date", "transferred_to_cb_date", "cb_transfer_amount",
            "cb_transfer_reference", "exclusion_reason",
            "created_date", "updated_date", "updated_by"
        ]

        self.add_entity(EntityDefinition(
            name="DormancyTracking",
            description="Dormancy process tracking entity",
            table_name="dormancy_tracking",
            fields=dormancy_fields,
            primary_key=["tracking_id"],
            indexes=[
                IndexDefinition("idx_dormancy_account", ["account_id"], unique=True),
                IndexDefinition("idx_dormancy_stage", ["current_stage"]),
                IndexDefinition("idx_dormancy_trigger", ["dormancy_trigger_date"]),
                IndexDefinition("idx_dormancy_transfer_date", ["transferred_to_cb_date"]),
                IndexDefinition("idx_dormancy_waiting_period", ["waiting_period_start", "waiting_period_end"])
            ]
        ))

        # Add relationships
        self.add_relationship(RelationshipDefinition(
            name="customer_accounts",
            from_entity="Account",
            to_entity="Customer",
            from_fields=["customer_id"],
            to_fields=["customer_id"],
            relationship_type="many_to_one",
            on_delete="CASCADE",
            on_update="CASCADE"
        ))

        self.add_relationship(RelationshipDefinition(
            name="account_dormancy_tracking",
            from_entity="DormancyTracking",
            to_entity="Account",
            from_fields=["account_id"],
            to_fields=["account_id"],
            relationship_type="one_to_one",
            on_delete="CASCADE",
            on_update="CASCADE"
        ))

    def _initialize_db_type_mappings(self):
        """Initialize database type mappings"""
        self.db_type_mappings = {
            DatabaseType.POSTGRESQL: {
                FieldType.STRING: "VARCHAR",
                FieldType.INTEGER: "INTEGER",
                FieldType.FLOAT: "REAL",
                FieldType.DECIMAL: "DECIMAL",
                FieldType.DATE: "DATE",
                FieldType.DATETIME: "TIMESTAMP",
                FieldType.BOOLEAN: "BOOLEAN",
                FieldType.UUID: "UUID",
                FieldType.JSON: "JSONB"
            },
            DatabaseType.MYSQL: {
                FieldType.STRING: "VARCHAR",
                FieldType.INTEGER: "INT",
                FieldType.FLOAT: "FLOAT",
                FieldType.DECIMAL: "DECIMAL",
                FieldType.DATE: "DATE",
                FieldType.DATETIME: "DATETIME",
                FieldType.BOOLEAN: "BOOLEAN",
                FieldType.UUID: "CHAR(36)",
                FieldType.JSON: "JSON"
            },
            DatabaseType.SQLSERVER: {
                FieldType.STRING: "NVARCHAR",
                FieldType.INTEGER: "INT",
                FieldType.FLOAT: "FLOAT",
                FieldType.DECIMAL: "DECIMAL",
                FieldType.DATE: "DATE",
                FieldType.DATETIME: "DATETIME2",
                FieldType.BOOLEAN: "BIT",
                FieldType.UUID: "UNIQUEIDENTIFIER",
                FieldType.JSON: "NVARCHAR(MAX)"
            }
        }

    # =============================================================================
    # FIELD MANAGEMENT
    # =============================================================================

    def add_field(self, field_def: UnifiedFieldDefinition):
        """Add a field definition"""
        self.fields[field_def.name] = field_def
        logger.info(f"Added field: {field_def.name}")

    def get_field(self, field_name: str) -> Optional[UnifiedFieldDefinition]:
        """Get field definition by name"""
        return self.fields.get(field_name)

    def get_fields_by_category(self, category: str) -> List[UnifiedFieldDefinition]:
        """Get all fields in a category"""
        return [field for field in self.fields.values() if field.category == category]

    def get_required_fields(self) -> List[str]:
        """Get list of required field names"""
        return [name for name, field in self.fields.items() if field.required]

    def get_all_field_names(self) -> List[str]:
        """Get list of all field names"""
        return list(self.fields.keys())

    # =============================================================================
    # ENTITY MANAGEMENT
    # =============================================================================

    def add_entity(self, entity_def: EntityDefinition):
        """Add entity definition"""
        self.entities[entity_def.name] = entity_def
        logger.info(f"Added entity: {entity_def.name}")

    def add_relationship(self, relationship: RelationshipDefinition):
        """Add relationship definition"""
        self.relationships.append(relationship)
        logger.info(f"Added relationship: {relationship.name}")

    # =============================================================================
    # DATA MAPPING CAPABILITIES
    # =============================================================================

    def map_columns_to_fields(self, columns: List[str]) -> Dict[str, Tuple[str, float]]:
        """
        Map incoming columns to standard fields
        Returns: {column_name: (field_name, confidence_score)}
        """
        mappings = {}

        for column in columns:
            best_match, confidence = self._find_best_field_match(column)
            if best_match and confidence > 0.4:  # Minimum confidence threshold
                mappings[column] = (best_match, confidence)

        return mappings

    def _find_best_field_match(self, column: str) -> Tuple[Optional[str], float]:
        """Find best matching field for a column"""
        column_clean = self._clean_string(column)
        best_score = 0.0
        best_match = None

        for field_name, field_def in self.fields.items():
            # Check exact match with field name
            if column_clean == self._clean_string(field_name):
                return field_name, 1.0

            # Check keywords and aliases
            all_terms = field_def.keywords + field_def.aliases + [field_name]

            for term in all_terms:
                term_clean = self._clean_string(term)

                # Exact match
                if column_clean == term_clean:
                    return field_name, 0.95

                # Substring match
                if term_clean in column_clean or column_clean in term_clean:
                    score = min(len(term_clean), len(column_clean)) / max(len(term_clean), len(column_clean))
                    if score > best_score:
                        best_score = score
                        best_match = field_name

                # Fuzzy match (Levenshtein-like)
                similarity = self._calculate_similarity(column_clean, term_clean)
                if similarity > best_score and similarity > 0.6:
                    best_score = similarity
                    best_match = field_name

        return best_match, best_score

    def _clean_string(self, s: str) -> str:
        """Clean string for comparison"""
        return re.sub(r'[^a-z0-9]', '', s.lower())

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity (simplified Levenshtein)"""
        if not s1 or not s2:
            return 0.0

        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if abs(len1 - len2) > max(len1, len2) * 0.5:
            return 0.0

        # Simple character overlap calculation
        common_chars = set(s1) & set(s2)
        return len(common_chars) / max(len(set(s1)), len(set(s2)))

    def validate_data_against_schema(self, data: pd.DataFrame, mappings: Dict[str, str]) -> List[str]:
        """Validate data against schema rules"""
        errors = []

        # Check required fields
        required_fields = self.get_required_fields()
        mapped_fields = set(mappings.values())
        missing_required = [f for f in required_fields if f not in mapped_fields]

        if missing_required:
            errors.append(f"Missing required fields: {missing_required}")

        # Validate field data
        for column, field_name in mappings.items():
            if field_name in self.fields and column in data.columns:
                field_def = self.fields[field_name]
                field_errors = self._validate_field_data(data[column], field_def)
                errors.extend([f"{column} -> {field_name}: {error}" for error in field_errors])

        return errors

    def _validate_field_data(self, series: pd.Series, field_def: UnifiedFieldDefinition) -> List[str]:
        """Validate series data against field definition"""
        errors = []

        # Type validation
        if not self._validate_field_type(series, field_def.type):
            errors.append(f"Invalid data type. Expected {field_def.type.value}")

        # Validation rules
        for rule in field_def.validation_rules:
            rule_errors = self._apply_validation_rule(series, rule)
            errors.extend(rule_errors)

        return errors

    def _validate_field_type(self, series: pd.Series, expected_type: FieldType) -> bool:
        """Validate series against expected field type"""
        # Sample non-null values for type checking
        non_null_values = series.dropna()
        if non_null_values.empty:
            return True

        sample_value = non_null_values.iloc[0]

        if expected_type == FieldType.STRING:
            return isinstance(sample_value, str)
        elif expected_type == FieldType.INTEGER:
            return pd.api.types.is_integer_dtype(series)
        elif expected_type in [FieldType.FLOAT, FieldType.DECIMAL]:
            return pd.api.types.is_numeric_dtype(series)
        elif expected_type in [FieldType.DATE, FieldType.DATETIME]:
            try:
                pd.to_datetime(series, errors='raise')
                return True
            except:
                return False
        elif expected_type == FieldType.BOOLEAN:
            return series.dtype == bool or set(non_null_values.unique()).issubset({0, 1, True, False})

        return True

    def _apply_validation_rule(self, series: pd.Series, rule: ValidationRule) -> List[str]:
        """Apply validation rule to series"""
        errors = []

        if rule.type == "pattern":
            pattern = rule.value
            invalid_count = 0
            for value in series.dropna():
                if not re.match(pattern, str(value)):
                    invalid_count += 1

            if invalid_count > 0:
                errors.append(f"{invalid_count} values don't match pattern: {rule.error_message}")

        elif rule.type == "range":
            range_def = rule.value
            min_val = range_def.get("min")
            max_val = range_def.get("max")

            numeric_series = pd.to_numeric(series, errors='coerce')

            if min_val is not None:
                below_min = (numeric_series < min_val).sum()
                if below_min > 0:
                    errors.append(f"{below_min} values below minimum {min_val}")

            if max_val is not None:
                above_max = (numeric_series > max_val).sum()
                if above_max > 0:
                    errors.append(f"{above_max} values above maximum {max_val}")

        elif rule.type == "enum":
            allowed_values = rule.value
            invalid_values = set(series.dropna()) - set(allowed_values)
            if invalid_values:
                errors.append(f"Invalid values found: {invalid_values}")

        return errors

    # =============================================================================
    # DATABASE DDL GENERATION
    # =============================================================================

    def generate_ddl(self, db_type: DatabaseType = DatabaseType.POSTGRESQL) -> List[str]:
        """Generate complete DDL for database creation"""
        ddl_statements = []

        # Generate table creation statements
        for entity in self.entities.values():
            table_ddl = self._generate_table_ddl(entity, db_type)
            ddl_statements.append(table_ddl)

        # Generate foreign key constraints (after all tables)
        for relationship in self.relationships:
            fk_ddl = self._generate_foreign_key_ddl(relationship, db_type)
            ddl_statements.append(fk_ddl)

        # Generate indexes
        for entity in self.entities.values():
            for index in entity.indexes:
                index_ddl = self._generate_index_ddl(entity.table_name, index, db_type)
                ddl_statements.append(index_ddl)

        return ddl_statements

    def _generate_table_ddl(self, entity: EntityDefinition, db_type: DatabaseType) -> str:
        """Generate CREATE TABLE statement"""
        lines = [f"CREATE TABLE {entity.table_name} ("]

        # Generate column definitions
        column_lines = []
        for field_name in entity.fields:
            if field_name in self.fields:
                field_def = self.fields[field_name]
                column_def = self._generate_column_definition(field_def, db_type)
                column_lines.append(f"    {column_def}")

        # Add primary key
        if entity.primary_key:
            pk_cols = ", ".join(entity.primary_key)
            column_lines.append(f"    PRIMARY KEY ({pk_cols})")

        lines.extend([",\n".join(column_lines)])
        lines.append(");")

        # Add comments
        comment_lines = [
            f"",
            f"COMMENT ON TABLE {entity.table_name} IS '{entity.description}';",
            f""
        ]

        for field_name in entity.fields:
            if field_name in self.fields:
                field_def = self.fields[field_name]
                comment_lines.append(
                    f"COMMENT ON COLUMN {entity.table_name}.{field_name} IS '{field_def.description}';"
                )

        return "\n".join(lines + comment_lines)

    def _generate_column_definition(self, field_def: UnifiedFieldDefinition, db_type: DatabaseType) -> str:
        """Generate column definition"""
        # Get database-specific type
        if db_type.value in field_def.db_type_mapping:
            sql_type = field_def.db_type_mapping[db_type.value]
        else:
            # Use default type mapping
            base_type = self.db_type_mappings[db_type][field_def.type]
            if field_def.type == FieldType.STRING:
                sql_type = f"{base_type}(255)"  # Default length
            else:
                sql_type = base_type

        definition = f"{field_def.name} {sql_type}"

        # Add constraints
        for constraint in field_def.constraints:
            if constraint.type == ConstraintType.NOT_NULL:
                definition += " NOT NULL"
            elif constraint.type == ConstraintType.UNIQUE:
                definition += " UNIQUE"
            elif constraint.type == ConstraintType.DEFAULT and constraint.value:
                if isinstance(constraint.value, str) and constraint.value != "CURRENT_TIMESTAMP":
                    definition += f" DEFAULT '{constraint.value}'"
                else:
                    definition += f" DEFAULT {constraint.value}"
            elif constraint.type == ConstraintType.CHECK and constraint.value:
                definition += f" CHECK ({constraint.value})"

        return definition

    def _generate_foreign_key_ddl(self, relationship: RelationshipDefinition, db_type: DatabaseType) -> str:
        """Generate foreign key constraint"""
        from_entity = self.entities[relationship.from_entity]
        to_entity = self.entities[relationship.to_entity]

        from_cols = ", ".join(relationship.from_fields)
        to_cols = ", ".join(relationship.to_fields)

        return (f"ALTER TABLE {from_entity.table_name} "
                f"ADD CONSTRAINT fk_{relationship.name} "
                f"FOREIGN KEY ({from_cols}) "
                f"REFERENCES {to_entity.table_name}({to_cols}) "
                f"ON DELETE {relationship.on_delete} "
                f"ON UPDATE {relationship.on_update};")

    def _generate_index_ddl(self, table_name: str, index: IndexDefinition, db_type: DatabaseType) -> str:
        """Generate index creation statement"""
        unique_clause = "UNIQUE " if index.unique else ""
        columns = ", ".join(index.columns)

        if db_type == DatabaseType.POSTGRESQL:
            index_type_clause = f" USING {index.type.value}" if index.type != IndexType.BTREE else ""
            where_clause = f" WHERE {index.where_clause}" if index.where_clause else ""
            return f"CREATE {unique_clause}INDEX {index.name} ON {table_name}{index_type_clause} ({columns}){where_clause};"
        else:
            return f"CREATE {unique_clause}INDEX {index.name} ON {table_name} ({columns});"

    # =============================================================================
    # QUERY GENERATION
    # =============================================================================

    def generate_analysis_queries(self) -> Dict[str, str]:
        """Generate comprehensive banking analysis queries"""
        return {
            "dormancy_summary": """
                SELECT 
                    account_type,
                    dormancy_status,
                    COUNT(*) as account_count,
                    SUM(balance_current) as total_balance,
                    AVG(balance_current) as avg_balance,
                    COUNT(CASE WHEN current_stage = 'transfer_ready' THEN 1 END) as transfer_ready
                FROM accounts 
                WHERE account_status IN ('active', 'inactive')
                GROUP BY account_type, dormancy_status
                ORDER BY account_type, dormancy_status;
            """,

            "customer_account_summary": """
                SELECT 
                    c.customer_type,
                    c.risk_rating,
                    COUNT(DISTINCT c.customer_id) as customer_count,
                    COUNT(a.account_id) as account_count,
                    SUM(a.balance_current) as total_balance,
                    COUNT(CASE WHEN a.dormancy_status = 'dormant' THEN 1 END) as dormant_accounts
                FROM customers c
                LEFT JOIN accounts a ON c.customer_id = a.customer_id
                GROUP BY c.customer_type, c.risk_rating
                ORDER BY c.customer_type, c.risk_rating;
            """,

            "dormant_accounts_detail": """
                SELECT 
                    c.customer_id,
                    c.full_name_en,
                    c.risk_rating,
                    a.account_id,
                    a.account_type,
                    a.balance_current,
                    a.last_transaction_date,
                    dt.current_stage,
                    dt.contact_attempts_made,
                    dt.cb_transfer_amount,
                    CURRENT_DATE - a.last_transaction_date as days_dormant
                FROM customers c
                JOIN accounts a ON c.customer_id = a.customer_id
                LEFT JOIN dormancy_tracking dt ON a.account_id = dt.account_id
                WHERE a.dormancy_status = 'dormant'
                ORDER BY days_dormant DESC, a.balance_current DESC;
            """,

            "cb_transfer_ready": """
                SELECT 
                    a.account_id,
                    a.account_type,
                    a.balance_current,
                    dt.transfer_eligibility_date,
                    dt.waiting_period_end,
                    dt.contact_attempts_made
                FROM accounts a
                JOIN dormancy_tracking dt ON a.account_id = dt.account_id
                WHERE dt.current_stage = 'transfer_ready'
                  AND dt.waiting_period_end <= CURRENT_DATE
                  AND dt.transferred_to_cb_date IS NULL
                ORDER BY dt.waiting_period_end ASC;
            """,

            "compliance_summary": """
                SELECT 
                    COUNT(*) as total_accounts,
                    COUNT(CASE WHEN a.dormancy_status = 'dormant' THEN 1 END) as dormant_count,
                    COUNT(CASE WHEN dt.contact_attempts_made >= 3 THEN 1 END) as contact_completed,
                    COUNT(CASE WHEN dt.transferred_to_cb_date IS NOT NULL THEN 1 END) as transferred_to_cb,
                    SUM(CASE WHEN a.dormancy_status = 'dormant' THEN a.balance_current ELSE 0 END) as total_dormant_balance,
                    SUM(dt.cb_transfer_amount) as total_transferred_amount
                FROM accounts a
                LEFT JOIN dormancy_tracking dt ON a.account_id = dt.account_id
                WHERE a.account_status IN ('active', 'inactive');
            """,

            "kyc_expiry_report": """
                SELECT 
                    c.customer_id,
                    c.full_name_en,
                    c.kyc_status,
                    c.kyc_expiry_date,
                    COUNT(a.account_id) as account_count,
                    SUM(a.balance_current) as total_balance,
                    CASE 
                        WHEN c.kyc_expiry_date < CURRENT_DATE THEN 'Expired'
                        WHEN c.kyc_expiry_date < CURRENT_DATE + INTERVAL '30 days' THEN 'Expiring Soon'
                        ELSE 'Valid'
                    END as kyc_status_category
                FROM customers c
                LEFT JOIN accounts a ON c.customer_id = a.customer_id
                WHERE c.kyc_expiry_date IS NOT NULL
                GROUP BY c.customer_id, c.full_name_en, c.kyc_status, c.kyc_expiry_date
                ORDER BY c.kyc_expiry_date ASC;
            """
        }

    # =============================================================================
    # SCHEMA PERSISTENCE
    # =============================================================================

    def save_schema_to_file(self, file_path: str):
        """Save schema to JSON file"""
        schema_data = {
            "version": "2.0.0",
            "description": "Complete Banking Compliance Schema - All 67 Fields",
            "created_date": datetime.now().isoformat(),
            "field_count": len(self.fields),
            "fields": {name: asdict(field_def) for name, field_def in self.fields.items()},
            "entities": {name: asdict(entity) for name, entity in self.entities.items()},
            "relationships": [asdict(rel) for rel in self.relationships],
            "categories": list(set(field.category for field in self.fields.values()))
        }

        # Convert datetime objects to strings
        schema_data = self._serialize_datetime_objects(schema_data)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(schema_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Complete schema with {len(self.fields)} fields saved to {file_path}")

    def load_schema_from_file(self, file_path: str):
        """Load schema from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)

        # Load fields
        for field_name, field_data in schema_data.get("fields", {}).items():
            field_def = self._deserialize_field_definition(field_data)
            self.fields[field_name] = field_def

        # Load entities
        for entity_name, entity_data in schema_data.get("entities", {}).items():
            entity_def = self._deserialize_entity_definition(entity_data)
            self.entities[entity_name] = entity_def

        # Load relationships
        for rel_data in schema_data.get("relationships", []):
            rel_def = self._deserialize_relationship_definition(rel_data)
            self.relationships.append(rel_def)

        logger.info(f"Schema loaded from {file_path} - {len(self.fields)} fields")

    def _serialize_datetime_objects(self, obj):
        """Convert datetime objects to ISO format strings"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._serialize_datetime_objects(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetime_objects(item) for item in obj]
        else:
            return obj

    def _deserialize_field_definition(self, field_data: Dict) -> UnifiedFieldDefinition:
        """Deserialize field definition from dict"""
        # Convert string enums back to enum objects
        field_data['type'] = FieldType(field_data['type'])

        # Convert validation rules
        validation_rules = []
        for rule_data in field_data.get('validation_rules', []):
            validation_rules.append(ValidationRule(**rule_data))
        field_data['validation_rules'] = validation_rules

        # Convert constraints
        constraints = []
        for constraint_data in field_data.get('constraints', []):
            constraint_data['type'] = ConstraintType(constraint_data['type'])
            constraints.append(DatabaseConstraint(**constraint_data))
        field_data['constraints'] = constraints

        # Convert datetime strings back to datetime objects
        if 'created_date' in field_data and field_data['created_date']:
            field_data['created_date'] = datetime.fromisoformat(field_data['created_date'])

        return UnifiedFieldDefinition(**field_data)

    def _deserialize_entity_definition(self, entity_data: Dict) -> EntityDefinition:
        """Deserialize entity definition from dict"""
        # Convert indexes
        indexes = []
        for index_data in entity_data.get('indexes', []):
            index_data['type'] = IndexType(index_data['type'])
            indexes.append(IndexDefinition(**index_data))
        entity_data['indexes'] = indexes

        return EntityDefinition(**entity_data)

    def _deserialize_relationship_definition(self, rel_data: Dict) -> RelationshipDefinition:
        """Deserialize relationship definition from dict"""
        return RelationshipDefinition(**rel_data)

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    def get_schema_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the schema"""
        return {
            "total_fields": len(self.fields),
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "fields_by_category": {
                category: len(self.get_fields_by_category(category))
                for category in set(field.category for field in self.fields.values())
            },
            "required_fields": len(self.get_required_fields()),
            "sensitive_fields": len([f for f in self.fields.values() if f.sensitive_data]),
            "categories": sorted(list(set(field.category for field in self.fields.values()))),
            "all_field_names": sorted(self.get_all_field_names())
        }

    def export_field_mapping_template(self, format: str = "csv") -> str:
        """Export comprehensive field mapping template"""
        if format.lower() == "csv":
            import io
            import csv

            output = io.StringIO()
            writer = csv.writer(output)

            # Header
            writer.writerow([
                "Field Name", "Category", "Description", "Type", "Required",
                "Keywords", "Aliases", "Sample Value", "Validation Rules", "Regulatory Requirement"
            ])

            # Data
            for field_name, field_def in sorted(self.fields.items()):
                writer.writerow([
                    field_name,
                    field_def.category,
                    field_def.description,
                    field_def.type.value,
                    "Yes" if field_def.required else "No",
                    "; ".join(field_def.keywords),
                    "; ".join(field_def.aliases),
                    self._generate_sample_value(field_def),
                    "; ".join([f"{rule.type}:{rule.value}" for rule in field_def.validation_rules]),
                    field_def.regulatory_requirement or ""
                ])

            return output.getvalue()

        return ""

    def _generate_sample_value(self, field_def: UnifiedFieldDefinition) -> str:
        """Generate sample value for field"""
        if field_def.type == FieldType.STRING:
            if 'id' in field_def.name.lower():
                return "SAMPLE123"
            elif 'email' in field_def.name.lower():
                return "sample@email.com"
            elif 'name' in field_def.name.lower():
                return "Sample Name"
            elif 'phone' in field_def.name.lower():
                return "+971501234567"
            elif field_def.name == "currency":
                return "AED"
            elif field_def.name == "emirate":
                return "dubai"
            else:
                return "Sample Text"
        elif field_def.type in [FieldType.INTEGER]:
            if 'months' in field_def.name.lower():
                return "6"
            elif 'attempts' in field_def.name.lower():
                return "3"
            else:
                return "100"
        elif field_def.type in [FieldType.FLOAT, FieldType.DECIMAL]:
            if 'balance' in field_def.name.lower():
                return "50000.00"
            elif 'rate' in field_def.name.lower():
                return "2.5"
            else:
                return "100.50"
        elif field_def.type in [FieldType.DATE, FieldType.DATETIME]:
            return "2024-01-15"
        elif field_def.type == FieldType.BOOLEAN:
            return "false"

        return "sample_value"

    def get_dormancy_related_fields(self) -> List[str]:
        """Get all fields related to dormancy processing"""
        dormancy_categories = ["dormancy", "dormancy_tracking", "transfer_process", "cb_transfer"]
        dormancy_fields = []

        for field_name, field_def in self.fields.items():
            if (field_def.category in dormancy_categories or
                "dormancy" in field_name.lower() or
                "transfer" in field_name.lower() or
                "contact" in field_name.lower()):
                dormancy_fields.append(field_name)

        return sorted(dormancy_fields)

    def get_compliance_related_fields(self) -> List[str]:
        """Get all fields related to compliance requirements"""
        compliance_fields = []

        for field_name, field_def in self.fields.items():
            if (field_def.regulatory_requirement or
                field_def.category in ["kyc_risk", "compliance"] or
                "kyc" in field_name.lower() or
                "risk" in field_name.lower() or
                "compliance" in field_name.lower()):
                compliance_fields.append(field_name)

        return sorted(compliance_fields)


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Initialize the complete schema manager
    schema_manager = CompleteBankingSchemaManager()

    print("=== COMPLETE BANKING COMPLIANCE SCHEMA SYSTEM ===\n")

    # 1. Schema Summary
    print("1. COMPREHENSIVE SCHEMA SUMMARY:")
    summary = schema_manager.get_schema_summary()
    for key, value in summary.items():
        if key == "all_field_names":
            print(f"   {key}: {len(value)} fields")
            # Print first 10 fields as example
            print(f"   Sample fields: {', '.join(value[:10])}...")
        else:
            print(f"   {key}: {value}")
    print()

    # 2. Field Categories
    print("2. FIELDS BY CATEGORY:")
    for category in summary["categories"]:
        category_fields = schema_manager.get_fields_by_category(category)
        print(f"   {category}: {len(category_fields)} fields")
        print(f"     {', '.join([f.name for f in category_fields[:5]])}...")
    print()

    # 3. Data Mapping Example
    print("3. DATA MAPPING EXAMPLE:")
    sample_columns = [
        "cust_id", "customer_name", "acc_number", "account_balance",
        "last_activity", "email_addr", "phone_num", "acc_type",
        "dormancy_flag", "cb_transfer_amt", "contact_attempts"
    ]

    mappings = schema_manager.map_columns_to_fields(sample_columns)
    print("   Column Mappings:")
    for column, (field, confidence) in mappings.items():
        print(f"   '{column}' -> '{field}' (confidence: {confidence:.2f})")
    print()

    # 4. Database DDL Generation
    print("4. DATABASE DDL GENERATION:")
    ddl_statements = schema_manager.generate_ddl(DatabaseType.POSTGRESQL)
    print(f"   Generated {len(ddl_statements)} DDL statements")
    print("   Sample CREATE TABLE statement:")
    print("   " + "\n   ".join(ddl_statements[0].split('\n')[:10]) + "...")
    print()

    # 5. Analysis Queries
    print("5. ANALYSIS QUERIES:")
    queries = schema_manager.generate_analysis_queries()
    for name, query in list(queries.items())[:2]:  # Show first 2
        print(f"   {name}:")
        print("   " + query.strip().replace('\n', '\n   ')[:200] + "...")
        print()

    # 6. Dormancy & Compliance Fields
    print("6. SPECIALIZED FIELD GROUPS:")
    dormancy_fields = schema_manager.get_dormancy_related_fields()
    compliance_fields = schema_manager.get_compliance_related_fields()

    print(f"   Dormancy-related fields ({len(dormancy_fields)}): ")
    print(f"     {', '.join(dormancy_fields[:8])}...")

    print(f"   Compliance-related fields ({len(compliance_fields)}): ")
    print(f"     {', '.join(compliance_fields[:8])}...")
    print()

    # 7. Save schema
    print("7. SCHEMA PERSISTENCE:")
    try:
        schema_manager.save_schema_to_file("complete_banking_schema.json")
        print("   ✅ Complete schema saved to 'complete_banking_schema.json'")
    except Exception as e:
        print(f"   ❌ Error saving schema: {e}")

    print("\n=== COMPLETE BANKING SCHEMA SYSTEM READY ===")
    print("📊 Schema Statistics:")
    print(f"✅ Total Fields: {summary['total_fields']}")
    print(f"✅ Categories: {len(summary['categories'])}")
    print(f"✅ Required Fields: {summary['required_fields']}")
    print(f"✅ Sensitive Fields: {summary['sensitive_fields']}")
    print(f"✅ Database Entities: {summary['total_entities']}")
    print(f"✅ Relationships: {summary['total_relationships']}")

    print("\n🎯 Features Available:")
    print("✅ Complete data mapping for all 67 banking fields")
    print("✅ Multi-database DDL generation (PostgreSQL, MySQL, SQL Server)")
    print("✅ Comprehensive validation rules and constraints")
    print("✅ CBUAE compliance field mapping")
    print("✅ Dormancy and CB transfer tracking")
    print("✅ KYC and risk management fields")
    print("✅ Contact and communication tracking")
    print("✅ Schema export and import capabilities")