import streamlit as st
import pandas as pd
import re
import json
from typing import Dict, List, Optional, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Import the unified schema system
from unified_schema import UnifiedBankingSchemaManager, DatabaseType, FieldType


class EnhancedSQLBot:
    """Enhanced SQL Bot with Unified Schema Integration"""

    def __init__(self):
        self.schema_manager = UnifiedBankingSchemaManager()
        self.llm = None
        self.initialize_prompts()

    def initialize_prompts(self):
        """Initialize schema-aware SQL generation prompts"""

        self.SCHEMA_AWARE_SQL_PROMPT = """
        You are an expert SQL query generator for banking compliance systems with deep knowledge of banking data structures.

        BANKING SCHEMA INFORMATION:
        {schema_info}

        FIELD MAPPINGS AND ALIASES:
        {field_mappings}

        VALIDATION RULES:
        {validation_rules}

        USER QUESTION: {question}

        INSTRUCTIONS:
        1. Generate a precise SQL query that answers the user's question
        2. Use the correct field names from the schema (not aliases)
        3. Apply appropriate filters based on validation rules
        4. Include relevant JOINs if multiple entities are needed
        5. Add comments explaining complex logic
        6. Ensure the query follows banking compliance best practices
        7. Use appropriate date ranges and business logic

        IMPORTANT BANKING CONTEXT:
        - Dormant accounts typically have no activity for 90+ days
        - Account statuses: active, inactive, closed, suspended
        - Dormancy status: active, pre_dormant, dormant
        - Customer types: individual, corporate
        - Account types: savings, current, fixed_deposit, credit

        Generate only the SQL query without explanations:
        """

        self.SCHEMA_VALIDATION_PROMPT = """
        You are a banking data validation expert. Analyze the following SQL query for correctness:

        QUERY TO VALIDATE:
        {sql_query}

        SCHEMA INFORMATION:
        {schema_info}

        VALIDATION RULES:
        {validation_rules}

        Check for:
        1. Correct table and column names
        2. Proper data types and constraints
        3. Business logic compliance
        4. Performance considerations
        5. Security best practices

        Provide:
        1. Validation status (VALID/INVALID)
        2. Issues found (if any)
        3. Suggestions for improvement
        4. Corrected query (if needed)

        Response format:
        STATUS: [VALID/INVALID]
        ISSUES: [List of issues]
        SUGGESTIONS: [Improvement suggestions]
        CORRECTED_QUERY: [If applicable]
        """

        self.INTELLIGENT_MAPPING_PROMPT = """
        You are an expert at mapping user queries to banking schema fields.

        USER QUERY: {user_query}

        AVAILABLE SCHEMA FIELDS:
        {schema_fields}

        FIELD ALIASES AND KEYWORDS:
        {field_aliases}

        Identify which schema fields are most relevant to answer the user's query.
        Consider:
        1. Direct field name matches
        2. Alias and keyword matches
        3. Business context and relationships
        4. Required JOINs between entities

        Return JSON format:
        {{
            "primary_fields": ["field1", "field2"],
            "secondary_fields": ["field3", "field4"],
            "required_entities": ["entity1", "entity2"],
            "suggested_filters": ["filter1", "filter2"],
            "business_context": "explanation"
        }}
        """

    def initialize_llm(self):
        """Initialize the LLM with proper configuration"""
        try:
            self.llm = ChatOpenAI(
                model_name="gpt-4-turbo",
                temperature=0.1,
                max_tokens=3000
            )
            return True
        except Exception as e:
            st.error(f"LLM initialization error: {e}")
            return False

    def get_schema_info_for_llm(self) -> Dict:
        """Get comprehensive schema information for LLM"""
        schema_info = {
            "entities": {},
            "fields": {},
            "relationships": [],
            "field_mappings": {},
            "validation_rules": {}
        }

        # Get entities information
        for entity_name, entity in self.schema_manager.entities.items():
            schema_info["entities"][entity_name] = {
                "table_name": entity.table_name,
                "description": entity.description,
                "fields": entity.fields,
                "primary_key": entity.primary_key
            }

        # Get field information with mappings
        for field_name, field in self.schema_manager.fields.items():
            schema_info["fields"][field_name] = {
                "description": field.description,
                "type": field.type.value,
                "required": field.required,
                "category": field.category,
                "keywords": field.keywords,
                "aliases": field.aliases
            }

            # Create mapping information
            all_terms = [field_name] + field.keywords + field.aliases
            for term in all_terms:
                schema_info["field_mappings"][term.lower()] = field_name

            # Add validation rules
            if field.validation_rules:
                schema_info["validation_rules"][field_name] = [
                    {
                        "type": rule.type,
                        "value": rule.value,
                        "message": rule.error_message
                    } for rule in field.validation_rules
                ]

        # Get relationships
        for rel in self.schema_manager.relationships:
            schema_info["relationships"].append({
                "name": rel.name,
                "from_entity": rel.from_entity,
                "to_entity": rel.to_entity,
                "from_fields": rel.from_fields,
                "to_fields": rel.to_fields,
                "type": rel.relationship_type
            })

        return schema_info

    def map_user_query_to_fields(self, user_query: str) -> Dict:
        """Use LLM to intelligently map user query to schema fields"""
        if not self.llm:
            return {}

        schema_info = self.get_schema_info_for_llm()

        try:
            mapping_prompt = PromptTemplate.from_template(self.INTELLIGENT_MAPPING_PROMPT)
            mapping_chain = mapping_prompt | self.llm | StrOutputParser()

            response = mapping_chain.invoke({
                "user_query": user_query,
                "schema_fields": json.dumps(schema_info["fields"], indent=2),
                "field_aliases": json.dumps(schema_info["field_mappings"], indent=2)
            })

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

        except Exception as e:
            st.warning(f"Field mapping error: {e}")

        return {}

    def generate_schema_aware_sql(self, user_query: str, connection_info: Dict = None) -> str:
        """Generate SQL query using schema intelligence"""
        if not self.llm:
            return None

        schema_info = self.get_schema_info_for_llm()

        # Get intelligent field mapping
        field_mapping = self.map_user_query_to_fields(user_query)

        try:
            sql_prompt = PromptTemplate.from_template(self.SCHEMA_AWARE_SQL_PROMPT)
            sql_chain = sql_prompt | self.llm | StrOutputParser()

            response = sql_chain.invoke({
                "question": user_query,
                "schema_info": json.dumps(schema_info["entities"], indent=2),
                "field_mappings": json.dumps(schema_info["field_mappings"], indent=2),
                "validation_rules": json.dumps(schema_info["validation_rules"], indent=2)
            })

            # Clean and extract SQL
            sql_query = self.clean_sql_query(response)

            # Validate the generated SQL
            if sql_query:
                validation_result = self.validate_sql_query(sql_query, schema_info)
                if validation_result.get("status") == "INVALID":
                    st.warning("Generated SQL has issues. Attempting to correct...")
                    corrected_query = validation_result.get("corrected_query")
                    if corrected_query:
                        sql_query = corrected_query

            return sql_query

        except Exception as e:
            st.error(f"SQL generation error: {e}")
            return None

    def validate_sql_query(self, sql_query: str, schema_info: Dict) -> Dict:
        """Validate SQL query against schema"""
        if not self.llm:
            return {"status": "UNKNOWN"}

        try:
            validation_prompt = PromptTemplate.from_template(self.SCHEMA_VALIDATION_PROMPT)
            validation_chain = validation_prompt | self.llm | StrOutputParser()

            response = validation_chain.invoke({
                "sql_query": sql_query,
                "schema_info": json.dumps(schema_info["entities"], indent=2),
                "validation_rules": json.dumps(schema_info["validation_rules"], indent=2)
            })

            # Parse validation response
            result = {"status": "VALID"}

            if "STATUS:" in response:
                status_line = re.search(r"STATUS:\s*(\w+)", response)
                if status_line:
                    result["status"] = status_line.group(1)

            if "ISSUES:" in response:
                issues_match = re.search(r"ISSUES:\s*(.*?)(?=SUGGESTIONS:|CORRECTED_QUERY:|$)", response, re.DOTALL)
                if issues_match:
                    result["issues"] = issues_match.group(1).strip()

            if "CORRECTED_QUERY:" in response:
                corrected_match = re.search(r"CORRECTED_QUERY:\s*(.*?)$", response, re.DOTALL)
                if corrected_match:
                    result["corrected_query"] = self.clean_sql_query(corrected_match.group(1))

            return result

        except Exception as e:
            st.warning(f"Validation error: {e}")
            return {"status": "UNKNOWN"}

    def clean_sql_query(self, sql_text: str) -> str:
        """Clean and extract valid SQL from response"""
        if not sql_text:
            return None

        # Remove markdown code blocks
        sql_text = re.sub(r"^```sql\s*|\s*```$", "", sql_text, flags=re.MULTILINE).strip()

        # Look for SELECT statement
        match = re.search(r"SELECT.*?(?=;|$)", sql_text, re.IGNORECASE | re.DOTALL)
        if match:
            sql_query = match.group(0).strip()
            # Add semicolon if missing
            if not sql_query.endswith(';'):
                sql_query += ';'
            return sql_query

        return sql_text.strip()

    def suggest_example_queries(self) -> List[str]:
        """Generate contextual example queries based on schema"""
        examples = [
            "Show me all dormant accounts with high balances",
            "Find customers who haven't been contacted in 6 months",
            "List accounts that need to be transferred to Central Bank",
            "Show compliance violations by account type",
            "Find accounts with missing contact information",
            "Display risk assessment for corporate customers",
            "Show accounts that became dormant in the last 90 days",
            "List customers with multiple dormant accounts",
            "Find accounts with suspicious activity patterns",
            "Show compliance status by customer segment"
        ]
        return examples

    def get_field_suggestions(self, partial_query: str) -> List[str]:
        """Get field suggestions based on partial user input"""
        suggestions = []
        query_lower = partial_query.lower()

        # Check for field matches
        for field_name, field in self.schema_manager.fields.items():
            # Check field name
            if query_lower in field_name.lower():
                suggestions.append(f"{field_name} - {field.description}")

            # Check keywords and aliases
            for keyword in field.keywords + field.aliases:
                if query_lower in keyword.lower():
                    suggestions.append(f"{field_name} (matched: {keyword}) - {field.description}")

        return suggestions[:10]  # Limit to top 10 suggestions


def render_enhanced_sqlbot():
    """Render the enhanced SQL Bot UI"""

    st.header("🤖 Enhanced SQL Bot with Schema Intelligence")

    # Initialize SQL Bot
    sql_bot = EnhancedSQLBot()

    # Initialize LLM
    if not sql_bot.llm:
        if sql_bot.initialize_llm():
            st.success("✅ AI Assistant with Schema Intelligence initialized")
        else:
            st.warning("⚠️ AI Assistant initialization failed. Limited functionality available.")
    else:
        st.info("🧠 Schema Intelligence Active")



    # Schema Overview
    with st.expander("🗄️ Banking Schema Overview"):
        schema_summary = sql_bot.schema_manager.get_schema_summary()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Fields", schema_summary["total_fields"])
        with col2:
            st.metric("Entities", schema_summary["total_entities"])
        with col3:
            st.metric("Required Fields", schema_summary["required_fields"])
        with col4:
            st.metric("Sensitive Fields", schema_summary["sensitive_fields"])

        # Show field categories
        st.subheader("Field Categories")
        for category, count in schema_summary["fields_by_category"].items():
            st.write(f"**{category.title()}**: {count} fields")

        # Show sample field mappings
        st.subheader("Sample Field Mappings")
        sample_fields = list(sql_bot.schema_manager.fields.items())[:5]
        for field_name, field_def in sample_fields:
            st.write(f"**{field_name}**: {field_def.description}")
            if field_def.aliases:
                st.write(f"  *Aliases*: {', '.join(field_def.aliases)}")

    # Query Mode Selection
    query_mode = st.radio(
        "Select Query Mode:",
        ["🧠 AI-Powered Natural Language", "✏️ Manual SQL Editor", "🎯 Smart Query Builder"],
        horizontal=True
    )

    if query_mode == "🧠 AI-Powered Natural Language":
        render_natural_language_mode(sql_bot, conn)
    elif query_mode == "✏️ Manual SQL Editor":
        render_manual_sql_mode(sql_bot, conn)
    else:
        render_smart_query_builder(sql_bot, conn)


def render_natural_language_mode(sql_bot: EnhancedSQLBot, conn):
    """Render natural language query interface"""

    st.subheader("💬 Ask Questions in Natural Language")

    # Example queries
    st.write("**Example questions you can ask:**")
    examples = sql_bot.suggest_example_queries()

    # Display examples in columns
    cols = st.columns(3)
    for i, example in enumerate(examples):
        with cols[i % 3]:
            if st.button(f"💡 {example}", key=f"example_{i}"):
                st.session_state.nl_query = example

    # Natural language input
    nl_query = st.text_area(
        "Ask your question:",
        value=st.session_state.get('nl_query', ''),
        placeholder="e.g., Show me all dormant accounts with balances over 10,000 AED that haven't been contacted",
        height=100,
        key="nl_query_input"
    )

    # Query suggestions as user types
    if nl_query and len(nl_query) > 3:
        suggestions = sql_bot.get_field_suggestions(nl_query)
        if suggestions:
            with st.expander("💡 Field Suggestions"):
                for suggestion in suggestions:
                    st.write(f"• {suggestion}")

    # Generate and execute SQL
    col1, col2 = st.columns([1, 1])

    with col1:
        generate_sql = st.button("🧠 Generate SQL", type="primary")

    with col2:
        execute_mode = st.selectbox("Execution Mode", ["Generate & Execute", "Generate Only"])

    if generate_sql and nl_query:
        with st.spinner("🤖 Analyzing your question and generating SQL..."):

            # Show field mapping analysis
            field_mapping = sql_bot.map_user_query_to_fields(nl_query)
            if field_mapping:
                with st.expander("🔍 Query Analysis"):
                    st.json(field_mapping)

            # Generate SQL
            generated_sql = sql_bot.generate_schema_aware_sql(nl_query)

            if generated_sql:
                st.subheader("📝 Generated SQL Query")
                st.code(generated_sql, language='sql')

                # Save to session for potential execution
                st.session_state.generated_sql = generated_sql
                st.session_state.original_query = nl_query

                # Execute if requested
                if execute_mode == "Generate & Execute":
                    execute_sql_query(generated_sql, conn, nl_query)
                else:
                    if st.button("▶️ Execute Query"):
                        execute_sql_query(generated_sql, conn, nl_query)
            else:
                st.error("❌ Failed to generate SQL query")


def render_manual_sql_mode(sql_bot: EnhancedSQLBot, conn):
    """Render manual SQL editor with schema assistance"""

    st.subheader("✏️ Manual SQL Editor with Schema Intelligence")

    # SQL Editor with syntax highlighting
    default_sql = """-- Example: Find dormant accounts with high balances
SELECT 
    c.customer_id,
    c.full_name_en,
    a.account_id,
    a.account_type,
    a.balance_current,
    a.last_transaction_date,
    a.dormancy_status
FROM customers c
JOIN accounts a ON c.customer_id = a.customer_id
WHERE a.dormancy_status = 'dormant'
  AND a.balance_current > 10000
ORDER BY a.balance_current DESC;"""

    manual_sql = st.text_area(
        "Enter your SQL query:",
        value=default_sql,
        height=200,
        help="Use the schema field names. The system will validate your query."
    )

    # Query validation
    if manual_sql.strip():
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("✅ Validate Query"):
                schema_info = sql_bot.get_schema_info_for_llm()
                validation = sql_bot.validate_sql_query(manual_sql, schema_info)

                if validation["status"] == "VALID":
                    st.success("✅ Query is valid!")
                else:
                    st.warning("⚠️ Query has issues:")
                    if "issues" in validation:
                        st.write(validation["issues"])

        with col2:
            if st.button("▶️ Execute Query", type="primary"):
                execute_sql_query(manual_sql, conn, "Manual SQL Query")


def render_smart_query_builder(sql_bot: EnhancedSQLBot, conn):
    """Render smart query builder interface"""

    st.subheader("🎯 Smart Query Builder")

    # Entity selection
    entities = list(sql_bot.schema_manager.entities.keys())
    selected_entity = st.selectbox("Select Primary Entity:", entities)

    if selected_entity:
        entity = sql_bot.schema_manager.entities[selected_entity]

        # Field selection
        available_fields = entity.fields
        selected_fields = st.multiselect(
            "Select Fields:",
            available_fields,
            default=available_fields[:5],
            help="Choose which fields to include in your query"
        )

        # Filter builder
        st.subheader("🔍 Add Filters")
        filters = []

        num_filters = st.number_input("Number of filters:", min_value=0, max_value=5, value=1)

        for i in range(int(num_filters)):
            col1, col2, col3 = st.columns([2, 1, 2])

            with col1:
                filter_field = st.selectbox(f"Field {i + 1}:", available_fields, key=f"filter_field_{i}")

            with col2:
                operator = st.selectbox(f"Operator {i + 1}:", ["=", "!=", ">", "<", ">=", "<=", "LIKE", "IN"],
                                        key=f"operator_{i}")

            with col3:
                value = st.text_input(f"Value {i + 1}:", key=f"value_{i}")

            if filter_field and value:
                filters.append(f"{filter_field} {operator} '{value}'")

        # Sort options
        sort_field = st.selectbox("Sort by:", [""] + available_fields)
        sort_order = st.selectbox("Sort order:", ["ASC", "DESC"])

        # Limit
        limit = st.number_input("Limit results:", min_value=0, max_value=1000, value=100)

        # Generate query
        if st.button("🏗️ Build Query"):
            query_parts = []

            # SELECT clause
            fields_str = ", ".join(selected_fields)
            query_parts.append(f"SELECT {fields_str}")

            # FROM clause
            query_parts.append(f"FROM {entity.table_name}")

            # WHERE clause
            if filters:
                where_clause = " AND ".join(filters)
                query_parts.append(f"WHERE {where_clause}")

            # ORDER BY clause
            if sort_field:
                query_parts.append(f"ORDER BY {sort_field} {sort_order}")

            # LIMIT clause
            if limit > 0:
                query_parts.append(f"LIMIT {limit}")

            built_query = "\n".join(query_parts) + ";"

            st.subheader("🔧 Built Query")
            st.code(built_query, language='sql')

            if st.button("▶️ Execute Built Query"):
                execute_sql_query(built_query, conn, "Smart Query Builder")


def execute_sql_query(sql_query: str, conn, query_source: str = "Unknown"):
    """Execute SQL query and display results"""

    try:
        with st.spinner("⏳ Executing query..."):
            # Execute query
            results_df = pd.read_sql(sql_query, conn)



            # Display results
            st.subheader("📊 Query Results")

            if not results_df.empty:
                # Results metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows Returned", len(results_df))
                with col2:
                    st.metric("Columns", len(results_df.columns))
                with col3:
                    query_time = datetime.now().strftime("%H:%M:%S")
                    st.metric("Executed At", query_time)

                # Data display
                st.dataframe(results_df, use_container_width=True)

                # Quick insights
                if len(results_df) > 0:
                    with st.expander("📈 Quick Insights"):
                        # Numeric columns summary
                        numeric_cols = results_df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 0:
                            st.write("**Numeric Summary:**")
                            st.dataframe(results_df[numeric_cols].describe())

                        # Data types
                        st.write("**Data Types:**")
                        st.write(results_df.dtypes.to_dict())

                # Visualization options
                if len(results_df) <= 1000 and len(results_df.columns) >= 2:
                    render_quick_visualization(results_df)

                # Download option
                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name=f"sql_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            else:
                st.info("✅ Query executed successfully but returned no results")

    except Exception as e:
        st.error(f"❌ Query execution failed: {str(e)}")

        # Provide help for common errors
        error_str = str(e).lower()
        if "column" in error_str and "does not exist" in error_str:
            st.help(
                "💡 **Tip**: Check if the column name exists in the schema. Use the Schema Overview to see available fields.")
        elif "table" in error_str and "does not exist" in error_str:
            st.help("💡 **Tip**: Check if the table name is correct. Available tables: customers, accounts")
        elif "syntax error" in error_str:
            st.help("💡 **Tip**: Check your SQL syntax. Try using the Query Validation feature.")


def render_quick_visualization(df: pd.DataFrame):
    """Render quick visualization options for query results"""

    st.subheader("📊 Quick Visualization")

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()

    if len(numeric_cols) > 0:
        viz_type = st.selectbox(
            "Visualization Type:",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Histogram"]
        )

        if viz_type == "Bar Chart" and len(text_cols) > 0:
            x_col = st.selectbox("X-axis (Category):", text_cols)
            y_col = st.selectbox("Y-axis (Value):", numeric_cols)

            if st.button("Generate Bar Chart"):
                fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Histogram":
            col = st.selectbox("Column for Histogram:", numeric_cols)

            if st.button("Generate Histogram"):
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)


# Query History Management
def show_enhanced_query_history(sql_bot: EnhancedSQLBot):
    """Show enhanced query history with schema insights"""

    st.subheader("📚 Query History & Analytics")

    # Get recent history
    try:
        history = get_recent_sql_history(limit=10)

        if history:
            for i, (timestamp, nl_query, sql_query) in enumerate(history):
                with st.expander(f"Query {i + 1}: {timestamp}"):
                    st.write(f"**Original Question**: {nl_query}")
                    st.code(sql_query, language='sql')

                    # Re-run option
                    if st.button(f"🔄 Re-run Query {i + 1}", key=f"rerun_{i}"):

                            execute_sql_query(sql_query, conn, "History Re-run")
        else:
            st.info("No query history available yet.")

    except Exception as e:
        st.warning(f"Could not load query history: {e}")


# Main render function
def render_sqlbot():
    """Main function to render the enhanced SQL bot"""
    render_enhanced_sqlbot()

    # Add query history section
    st.markdown("---")
    sql_bot = EnhancedSQLBot()
    show_enhanced_query_history(sql_bot)


if __name__ == "__main__":
    render_sqlbot()