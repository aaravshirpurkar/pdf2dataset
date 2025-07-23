"""Data validation and consistency checks for extracted content."""
from typing import Dict, Any, List, Optional, Union, Tuple
import re
from datetime import datetime
import pandas as pd

class DataValidator:
    def __init__(self):
        # Common validation patterns
        self.patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?\d{9,15}$',
            'date': r'^\d{4}-\d{2}-\d{2}$',
            'url': r'^https?:\/\/[\w\-]+(\.[\w\-]+)+[/#?]?.*$',
            'number': r'^-?\d*\.?\d+$',
            'currency': r'^\$?\d{1,3}(,\d{3})*(\.\d{2})?$'
        }
        
        # Field-specific validation rules
        self.field_rules = {
            'email': {'type': 'email', 'required': True},
            'phone': {'type': 'phone', 'required': False},
            'date': {'type': 'date', 'required': True},
            'price': {'type': 'currency', 'min': 0},
            'quantity': {'type': 'number', 'min': 0},
            'percentage': {'type': 'number', 'min': 0, 'max': 100}
        }
    
    def validate_field(self, value: str, field_type: str) -> Tuple[bool, Optional[str]]:
        """Validate a single field value against its type rules."""
        if not value:
            return False, "Empty value"
        
        pattern = self.patterns.get(field_type)
        if not pattern:
            return True, None
        
        if re.match(pattern, str(value)):
            # Additional type-specific validations
            if field_type == 'date':
                try:
                    datetime.strptime(value, '%Y-%m-%d')
                    return True, None
                except ValueError:
                    return False, "Invalid date format"
            
            if field_type in ['number', 'currency']:
                try:
                    num = float(re.sub(r'[^\d.-]', '', value))
                    rules = self.field_rules.get(field_type, {})
                    if 'min' in rules and num < rules['min']:
                        return False, f"Value below minimum {rules['min']}"
                    if 'max' in rules and num > rules['max']:
                        return False, f"Value above maximum {rules['max']}"
                except ValueError:
                    return False, "Invalid number format"
            
            return True, None
        
        return False, f"Does not match {field_type} pattern"

    def validate_record(self, record: Dict[str, Any], rules: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Validate a complete record against specified rules."""
        errors = {}
        
        # Check required fields
        for field, field_rules in rules.items():
            if field_rules.get('required', False) and field not in record:
                errors[field] = ["Required field missing"]
                continue
            
            if field in record:
                value = record[field]
                valid, error = self.validate_field(value, field_rules.get('type', 'text'))
                if not valid:
                    errors[field] = [error]
        
        # Cross-field validations
        if 'start_date' in record and 'end_date' in record:
            try:
                start = datetime.strptime(record['start_date'], '%Y-%m-%d')
                end = datetime.strptime(record['end_date'], '%Y-%m-%d')
                if end < start:
                    errors['end_date'] = ["End date cannot be before start date"]
            except ValueError:
                pass  # Date format errors will be caught by individual field validation
        
        return errors

    def validate_dataset(self, df: pd.DataFrame, rules: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate an entire dataset and return cleaned data with error report."""
        error_report = {
            'total_records': len(df),
            'invalid_records': 0,
            'errors_by_field': {},
            'error_samples': {}
        }
        
        # Initialize error tracking
        for field in rules.keys():
            error_report['errors_by_field'][field] = 0
        
        # Validate each record
        valid_records = []
        for idx, row in df.iterrows():
            record_errors = self.validate_record(row.to_dict(), rules)
            
            if record_errors:
                error_report['invalid_records'] += 1
                for field, errors in record_errors.items():
                    error_report['errors_by_field'][field] += 1
                    if field not in error_report['error_samples']:
                        error_report['error_samples'][field] = []
                    if len(error_report['error_samples'][field]) < 3:  # Keep up to 3 examples
                        error_report['error_samples'][field].append({
                            'value': row.get(field),
                            'errors': errors
                        })
            else:
                valid_records.append(row)
        
        # Create cleaned DataFrame with only valid records
        cleaned_df = pd.DataFrame(valid_records)
        
        # Calculate error percentages
        total = len(df)
        if total > 0:
            error_report['error_rate'] = error_report['invalid_records'] / total
            for field in error_report['errors_by_field']:
                error_report['errors_by_field'][field] = {
                    'count': error_report['errors_by_field'][field],
                    'percentage': error_report['errors_by_field'][field] / total * 100
                }
        
        return cleaned_df, error_report

    def suggest_corrections(self, value: str, field_type: str) -> Optional[str]:
        """Suggest corrections for common data issues."""
        if not value:
            return None
        
        if field_type == 'phone':
            # Normalize phone numbers
            digits = re.sub(r'\D', '', value)
            if len(digits) == 10:
                return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            elif len(digits) == 11 and digits.startswith('1'):
                return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        
        elif field_type == 'date':
            # Try common date formats
            for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y']:
                try:
                    date = datetime.strptime(value, fmt)
                    return date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
        
        elif field_type == 'currency':
            # Normalize currency values
            try:
                num = float(re.sub(r'[^\d.-]', '', value))
                return f"${num:,.2f}"
            except ValueError:
                pass
        
        return None
