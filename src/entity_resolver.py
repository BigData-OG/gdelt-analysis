import spacy
from spacy.matcher import PhraseMatcher
import json
from pathlib import Path
from typing import List, Dict, Set
import logging

logger = logging.getLogger(__name__)

class EntityResolver:
    """
    Resolves company name variations to canonical names using spaCy.
    """
    
    def __init__(self, config_path: str = None):

        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy model: en_core_web_sm")

        
        # Load company aliases
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'company_aliases.json'
        
        with open(config_path, 'r') as f:
            self.company_aliases = json.load(f)
        
        # Build phrase matcher
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self._build_matcher()
        
        logger.info(f"Loaded {len(self.company_aliases)} companies with aliases")
    
    def _build_matcher(self):
        """Build phrase matcher from company aliases"""
        for canonical_name, aliases in self.company_aliases.items():
            patterns = [self.nlp.make_doc(alias) for alias in aliases]
            self.matcher.add(canonical_name, patterns)
    
    def resolve_text(self, text: str) -> Set[str]:
        """
        Extract and resolve company mentions in text.

        """
        if not text:
            return set()
        
        doc = self.nlp(text)
        matches = self.matcher(doc)
        
        companies = set()
        for match_id, start, end in matches:
            canonical_name = self.nlp.vocab.strings[match_id]
            companies.add(canonical_name)
        
        return companies
    
    def get_regex_pattern(self, company_name: str, ticker: str) -> str:
        """
        Generate enhanced regex pattern including all aliases.
        """
        # Get aliases for this company
        aliases = self.company_aliases.get(company_name, [company_name])
        
        # Add ticker if not already in aliases
        if ticker not in aliases:
            aliases.append(ticker)
        
        # Build regex pattern: (alias1|alias2|ticker)
        escaped_aliases = [self._escape_regex(alias) for alias in aliases]
        pattern = '|'.join(escaped_aliases)
        
        return pattern
    
    def _escape_regex(self, text: str) -> str:
        """Escape special regex characters"""
        special_chars = r'\.^$*+?{}[]|()'
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text
    
    def extract_entities_batch(self, texts: List[str]) -> List[Set[str]]:
        """
        Batch process multiple texts efficiently.
        """
        results = []
        
        # Use nlp.pipe for efficient batch processing
        for doc in self.nlp.pipe(texts, batch_size=50):
            matches = self.matcher(doc)
            companies = set()
            for match_id, start, end in matches:
                canonical_name = self.nlp.vocab.strings[match_id]
                companies.add(canonical_name)
            results.append(companies)
        
        return results