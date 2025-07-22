"""
Document classifier module for pdf2dataset application.
Detects document types based on textual and layout patterns.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

class DocumentClassifier:
    """Class for classifying document types based on content patterns."""
    
    def __init__(self):
        """Initialize document patterns and classifications."""
        # Document patterns mapping
        self.patterns = {
            "invoice": {
                "keywords": ["invoice", "bill to", "payment due", "invoice no", "invoice number", "invoice date"],
                "patterns": [
                    r"invoice\s+#?\s*[0-9]+",
                    r"invoice\s+date\s*:?",
                    r"due\s+date\s*:?",
                    r"bill\s+to\s*:",
                ],
                "tables": ["description", "qty", "quantity", "price", "amount", "total"],
                "confidence_threshold": 0.6,
            },
            "bank_statement": {
                "keywords": ["statement of account", "account statement", "balance", "transaction", "deposit", "withdrawal"],
                "patterns": [
                    r"statement\s+of\s+account",
                    r"account\s+statement",
                    r"opening\s+balance",
                    r"closing\s+balance",
                ],
                "tables": ["date", "description", "amount", "balance"],
                "confidence_threshold": 0.5,
            },
            "insurance_document": {
                "keywords": ["policy", "coverage", "insured", "premium", "insurance", "policyholder"],
                "patterns": [
                    r"policy\s+no[.:]*\s*[a-zA-Z0-9]+",
                    r"insured\s+name",
                    r"coverage\s+(amount|period|type)",
                ],
                "tables": ["coverage", "limit", "deductible", "premium"],
                "confidence_threshold": 0.6,
            },
            "legal_contract": {
                "keywords": ["agreement", "contract", "parties", "terms", "conditions", "clause", "hereby", "thereto"],
                "patterns": [
                    r"§\s*\d+\.\d+",
                    r"section\s+\d+\.\d+",
                    r"signed\s+by",
                    r"dated\s+this",
                    r"in\s+witness\s+whereof",
                ],
                "tables": [],
                "confidence_threshold": 0.5,
            },
            "academic_paper": {
                "keywords": ["abstract", "introduction", "method", "results", "conclusion", "references", "et al"],
                "patterns": [
                    r"abstract",
                    r"\[\d+\]",
                    r"\([A-Za-z]+,\s+\d{4}\)",
                    r"keywords\s*:",
                ],
                "tables": ["figure", "table", "data"],
                "confidence_threshold": 0.7,
            },
            "resume_cv": {
                "keywords": ["resume", "curriculum vitae", "experience", "education", "skills", "employment"],
                "patterns": [
                    r"curriculum\s+vitae",
                    r"professional\s+experience",
                    r"work\s+experience",
                    r"education",
                    r"skills",
                ],
                "tables": ["experience", "education", "skills"],
                "confidence_threshold": 0.6,
            },
            "presentation": {
                "keywords": ["slide", "presentation", "powerpoint", "deck"],
                "patterns": [],
                "tables": [],
                "confidence_threshold": 0.4,
            },
            "exam_paper": {
                "keywords": ["exam", "question", "answer", "marks", "points"],
                "patterns": [
                    r"question\s+\d+",
                    r"q\.\s*\d+",
                    r"\(\s*\d+\s+marks\s*\)",
                    r"\(\s*\d+\s+points\s*\)",
                ],
                "tables": [],
                "confidence_threshold": 0.6,
            },
            "multiple_choice": {
                "keywords": ["multiple choice", "select one", "circle the", "choose the"],
                "patterns": [
                    r"[A-Da-d]\s*\)",
                    r"\(\s*[A-Da-d]\s*\)",
                    r"□\s*[A-Da-d]",
                    r"○\s*[A-Da-d]",
                ],
                "tables": [],
                "confidence_threshold": 0.5,
            },
            "legal_opinion": {
                "keywords": ["court", "plaintiff", "defendant", "ruling", "judgment", "opinion", "case"],
                "patterns": [
                    r"v\.\s+",
                    r"case\s+no\.",
                    r"plaintiff",
                    r"defendant",
                ],
                "tables": [],
                "confidence_threshold": 0.6,
            },
            "medical_record": {
                "keywords": ["patient", "diagnosis", "assessment", "plan", "medical", "treatment", "symptoms"],
                "patterns": [
                    r"patient\s+info",
                    r"assessment",
                    r"plan",
                    r"diagnosis",
                    r"[A-Z]\d+\.\d+", # ICD-10 codes
                ],
                "tables": ["medication", "dose", "frequency"],
                "confidence_threshold": 0.7,
            },
            "boarding_pass": {
                "keywords": ["boarding pass", "flight", "passenger", "gate", "seat"],
                "patterns": [
                    r"boarding\s+pass",
                    r"flight\s+[A-Z]{2}\d+",
                    r"seat\s+\d+[A-Z]",
                    r"gate\s+[A-Z]?\d+",
                ],
                "tables": [],
                "confidence_threshold": 0.8,
            },
            "election_results": {
                "keywords": ["votes", "election", "ballot", "precinct", "candidate", "party", "results"],
                "patterns": [
                    r"total\s+votes",
                    r"precinct",
                    r"\d+\s*%\s*of\s*votes",
                ],
                "tables": ["candidate", "votes", "percentage", "party"],
                "confidence_threshold": 0.7,
            },
            "business_letter": {
                "keywords": ["dear", "sincerely", "regards", "address", "subject"],
                "patterns": [
                    r"dear\s+[a-zA-Z\s\.]+",
                    r"sincerely",
                    r"regards",
                ],
                "tables": [],
                "confidence_threshold": 0.5,
            },
            "financial_report": {
                "keywords": ["balance sheet", "income statement", "cash flow", "assets", "liabilities", "equity"],
                "patterns": [
                    r"balance\s+sheet",
                    r"income\s+statement",
                    r"cash\s+flow",
                ],
                "tables": ["assets", "liabilities", "revenue", "expense"],
                "confidence_threshold": 0.7,
            },
            "technical_manual": {
                "keywords": ["manual", "guide", "instructions", "troubleshooting", "version", "revision"],
                "patterns": [
                    r"version\s+\d+\.\d+",
                    r"revision\s+history",
                ],
                "tables": [],
                "confidence_threshold": 0.5,
            },
            "form": {
                "keywords": ["form", "fill", "complete", "submit"],
                "patterns": [
                    r"name\s*:",
                    r"date\s+of\s+birth",
                    r"ssn",
                    r"social\s+security",
                    r"address\s*:",
                    r"signature\s*:",
                ],
                "tables": [],
                "confidence_threshold": 0.5,
            },
        }
    
    def classify_document(self, text: str) -> Dict[str, Any]:
        """
        Classify a document based on its text content.
        
        Args:
            text: The extracted text from the document
            
        Returns:
            Dictionary with document type, confidence score, and metadata
        """
        text_lower = text.lower()
        results = {}
        
        for doc_type, pattern_set in self.patterns.items():
            score = 0
            matches = []
            
            # Check for keywords
            keyword_count = 0
            for keyword in pattern_set["keywords"]:
                if keyword.lower() in text_lower:
                    keyword_count += 1
                    matches.append(f"Keyword: {keyword}")
            
            if len(pattern_set["keywords"]) > 0:
                keyword_score = keyword_count / len(pattern_set["keywords"])
            else:
                keyword_score = 0
                
            # Check for patterns
            pattern_count = 0
            for pattern in pattern_set["patterns"]:
                if re.search(pattern, text_lower):
                    pattern_count += 1
                    matches.append(f"Pattern: {pattern}")
            
            if len(pattern_set["patterns"]) > 0:
                pattern_score = pattern_count / len(pattern_set["patterns"])
            else:
                pattern_score = 0
                
            # Check for tables/columns
            table_count = 0
            for table_term in pattern_set["tables"]:
                if table_term.lower() in text_lower:
                    table_count += 1
                    matches.append(f"Table term: {table_term}")
            
            if len(pattern_set["tables"]) > 0:
                table_score = table_count / len(pattern_set["tables"])
            else:
                table_score = 0
                
            # Calculate weighted score
            if len(pattern_set["keywords"]) > 0 and len(pattern_set["patterns"]) > 0 and len(pattern_set["tables"]) > 0:
                score = (keyword_score * 0.4) + (pattern_score * 0.4) + (table_score * 0.2)
            elif len(pattern_set["keywords"]) > 0 and len(pattern_set["patterns"]) > 0:
                score = (keyword_score * 0.5) + (pattern_score * 0.5)
            elif len(pattern_set["keywords"]) > 0:
                score = keyword_score
            else:
                score = 0
                
            # Store results if score meets threshold
            if score >= pattern_set["confidence_threshold"]:
                results[doc_type] = {
                    "score": score,
                    "matches": matches,
                    "confidence_threshold": pattern_set["confidence_threshold"]
                }
        
        # Find the best match
        if results:
            best_match = max(results.items(), key=lambda x: x[1]["score"])
            return {
                "document_type": best_match[0],
                "confidence": best_match[1]["score"],
                "matches": best_match[1]["matches"],
                "all_matches": results
            }
        else:
            return {
                "document_type": "unknown",
                "confidence": 0,
                "matches": [],
                "all_matches": {}
            }


def detect_document_type(text: str) -> Dict[str, Any]:
    """
    Helper function to detect document type from text.
    
    Args:
        text: The extracted text content
        
    Returns:
        Dictionary with document type and confidence information
    """
    classifier = DocumentClassifier()
    return classifier.classify_document(text)


def get_extraction_hints(document_type: str) -> Dict[str, Any]:
    """
    Get extraction hints and guidance for a given document type.
    
    Args:
        document_type: The detected document type
        
    Returns:
        Dictionary with extraction guidance, expected fields, and structure
    """
    hints = {
        "invoice": {
            "fields": ["invoice_number", "invoice_date", "due_date", "vendor_name", "customer_name", 
                      "line_items", "subtotal", "tax", "total"],
            "structure": "header_with_table",
            "extraction_tips": "Focus on the line items table and extract each row as a separate record."
        },
        "bank_statement": {
            "fields": ["account_number", "statement_period", "opening_balance", "closing_balance", 
                      "transactions", "date", "description", "amount"],
            "structure": "header_with_transactions",
            "extraction_tips": "Extract each transaction as a separate record with date, description and amount."
        },
        "insurance_document": {
            "fields": ["policy_number", "insured_name", "coverage_period", "premium", "coverage_details"],
            "structure": "key_value_pairs",
            "extraction_tips": "Extract the key policy information and coverage details as structured data."
        },
        "legal_contract": {
            "fields": ["parties", "effective_date", "terms", "clauses", "signatures"],
            "structure": "sectioned_document",
            "extraction_tips": "Identify major sections, clauses and definitions in the document."
        },
        "academic_paper": {
            "fields": ["title", "authors", "abstract", "keywords", "sections", "references"],
            "structure": "sectioned_with_citations",
            "extraction_tips": "Extract the paper structure with separate fields for abstract, methods, results, etc."
        },
        "resume_cv": {
            "fields": ["name", "contact_info", "experience", "education", "skills"],
            "structure": "sectioned_document",
            "extraction_tips": "Extract work experiences as separate records with company, position and dates."
        },
        "presentation": {
            "fields": ["title", "slides", "content", "bullet_points"],
            "structure": "slides",
            "extraction_tips": "Identify individual slides and their content."
        },
        "exam_paper": {
            "fields": ["title", "instructions", "questions", "options", "marks"],
            "structure": "questions_with_options",
            "extraction_tips": "Extract each question as a separate record with its options if applicable."
        },
        "multiple_choice": {
            "fields": ["questions", "options", "correct_answer"],
            "structure": "questions_with_options",
            "extraction_tips": "Extract each question and its multiple choice options."
        },
        "legal_opinion": {
            "fields": ["case_name", "case_number", "court", "date", "opinion", "citations"],
            "structure": "sectioned_with_citations",
            "extraction_tips": "Identify the case details and the legal reasoning sections."
        },
        "medical_record": {
            "fields": ["patient_name", "date", "symptoms", "diagnosis", "treatment", "medications"],
            "structure": "key_value_with_sections",
            "extraction_tips": "Extract patient information and separate assessment, diagnosis and plan sections."
        },
        "boarding_pass": {
            "fields": ["passenger_name", "flight_number", "date", "from", "to", "seat", "gate", "boarding_time"],
            "structure": "key_value_pairs",
            "extraction_tips": "Extract the flight details and passenger information."
        },
        "election_results": {
            "fields": ["election_name", "date", "jurisdiction", "candidates", "votes", "percentage"],
            "structure": "table_data",
            "extraction_tips": "Extract each candidate result as a separate record with votes and percentages."
        },
        "business_letter": {
            "fields": ["sender", "recipient", "date", "subject", "body", "signature"],
            "structure": "letter_format",
            "extraction_tips": "Identify the sender, recipient and key content of the letter."
        },
        "financial_report": {
            "fields": ["company_name", "period", "balance_sheet", "income_statement", "cash_flow"],
            "structure": "sectioned_with_tables",
            "extraction_tips": "Extract financial data from tables and organize by statement type."
        },
        "technical_manual": {
            "fields": ["title", "version", "chapters", "sections", "procedures", "troubleshooting"],
            "structure": "sectioned_document",
            "extraction_tips": "Identify major sections and procedural steps in the document."
        },
        "form": {
            "fields": ["form_title", "form_fields", "instructions", "signature_fields"],
            "structure": "form_fields",
            "extraction_tips": "Extract form fields as key-value pairs."
        },
        "unknown": {
            "fields": ["content", "sections", "tables"],
            "structure": "unknown",
            "extraction_tips": "Extract any visible structure such as sections, tables or key information."
        }
    }
    
    return hints.get(document_type, hints["unknown"]) 