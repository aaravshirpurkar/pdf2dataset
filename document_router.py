"""Document type detection and specialized routing."""
from typing import Dict, Any, List, Tuple
import re
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

@dataclass
class DocumentType:
    name: str
    confidence: float
    features: Dict[str, Any]

class DocumentRouter:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.classifier = MultinomialNB()
        self._load_or_initialize_model()
        
        # Document type patterns and rules
        self.patterns = {
            'invoice': {
                'keywords': ['invoice', 'bill to', 'payment terms', 'due date', 'total amount'],
                'required_fields': ['invoice_number', 'date', 'total_amount'],
                'table_headers': ['description', 'quantity', 'price', 'amount']
            },
            'resume': {
                'keywords': ['experience', 'education', 'skills', 'objective', 'references'],
                'sections': ['work experience', 'education', 'skills', 'contact'],
                'exclude': ['invoice', 'payment']
            },
            'scientific_paper': {
                'keywords': ['abstract', 'introduction', 'methodology', 'results', 'conclusion'],
                'sections': ['abstract', 'references'],
                'patterns': [r'\bcite[ds]?\b', r'\breference[s]?\b']
            },
            'contract': {
                'keywords': ['agreement', 'parties', 'terms', 'conditions', 'signature'],
                'required_sections': ['parties', 'terms', 'signatures'],
                'patterns': [r'witness[es]*\s+whereof', r'here[by|in|to]\s+agree[s]?']
            }
        }
    
    def _load_or_initialize_model(self):
        """Load or initialize the document classification model."""
        model_path = 'doc_classifier_model.joblib'
        if os.path.exists(model_path):
            try:
                loaded_model = joblib.load(model_path)
                self.vectorizer = loaded_model['vectorizer']
                self.classifier = loaded_model['classifier']
            except Exception:
                pass  # Use default initialized models if loading fails
    
    def detect_document_type(self, text: str, metadata: Dict[str, Any] = None) -> DocumentType:
        """Detect document type using rules, patterns, and ML."""
        scores = {}
        
        # 1. Rule-based scoring
        for doc_type, rules in self.patterns.items():
            score = 0
            text_lower = text.lower()
            
            # Keyword matching
            keywords = rules.get('keywords', [])
            keyword_matches = sum(1 for kw in keywords if kw in text_lower)
            score += keyword_matches / len(keywords) if keywords else 0
            
            # Required fields/sections
            if 'required_fields' in rules:
                fields_found = sum(1 for field in rules['required_fields'] 
                                 if any(field.replace('_', ' ') in line.lower() 
                                       for line in text.split('\n')))
                score += fields_found / len(rules['required_fields'])
            
            # Pattern matching
            if 'patterns' in rules:
                pattern_matches = sum(1 for pattern in rules['patterns'] 
                                   if re.search(pattern, text_lower))
                score += pattern_matches / len(rules['patterns'])
            
            # Exclusion rules
            if 'exclude' in rules and any(excl in text_lower for excl in rules['exclude']):
                score *= 0.5
            
            scores[doc_type] = score
        
        # 2. ML-based classification (if model is trained)
        try:
            X = self.vectorizer.transform([text])
            ml_probs = self.classifier.predict_proba(X)[0]
            for doc_type, prob in zip(self.classifier.classes_, ml_probs):
                if doc_type in scores:
                    # Combine rule-based and ML scores
                    scores[doc_type] = (scores[doc_type] + prob) / 2
                else:
                    scores[doc_type] = prob
        except Exception:
            pass  # Continue with just rule-based scores if ML fails
        
        # 3. Consider metadata if available
        if metadata:
            for doc_type, rules in self.patterns.items():
                if any(key in metadata for key in rules.get('required_fields', [])):
                    scores[doc_type] = scores.get(doc_type, 0) * 1.2
        
        # Select best match
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            return DocumentType(
                name=best_type[0],
                confidence=best_type[1],
                features={
                    'patterns_matched': [p for p in self.patterns[best_type[0]].get('patterns', [])
                                       if re.search(p, text.lower())],
                    'keywords_found': [k for k in self.patterns[best_type[0]].get('keywords', [])
                                     if k in text.lower()]
                }
            )
        
        return DocumentType(name='unknown', confidence=0.0, features={})
    
    def get_extraction_pipeline(self, doc_type: str) -> Dict[str, Any]:
        """Get specialized extraction settings for document type."""
        pipelines = {
            'invoice': {
                'extractors': ['pdfplumber', 'camelot'],
                'focus_areas': ['header', 'table', 'totals'],
                'validation_rules': {
                    'invoice_number': r'^[A-Za-z0-9-]+$',
                    'date': r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
                    'amount': r'\$?\d+,?\d*\.?\d*'
                }
            },
            'resume': {
                'extractors': ['pdfminer', 'unstructured'],
                'focus_areas': ['sections', 'lists'],
                'section_keywords': ['experience', 'education', 'skills']
            },
            'scientific_paper': {
                'extractors': ['grobid', 'nougat'],
                'focus_areas': ['text', 'equations', 'references'],
                'preserve_sections': True
            },
            'contract': {
                'extractors': ['pdfplumber', 'tika'],
                'focus_areas': ['text', 'signatures'],
                'preserve_formatting': True
            }
        }
        
        return pipelines.get(doc_type, {
            'extractors': ['pdfplumber', 'pdfminer'],
            'focus_areas': ['text', 'tables'],
            'preserve_formatting': False
        })
