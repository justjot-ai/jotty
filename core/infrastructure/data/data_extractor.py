"""
 SMART DATA EXTRACTOR
======================

Extracts data from ANY format - NO hardcoding!

Supports:
- Direct strings
- Objects with methods (introspection)
- Callables
- File paths
- Dicts
- Mixed formats

The user provides data in THEIR format, ReVal figures out how to extract it!
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class SmartDataExtractor:
    """
    Intelligently extracts data from ANY format.
    
    NO assumptions about structure!
    NO hardcoded method names!
    NO configuration needed!
    
    Just works!
    """
    
    def __init__(self) -> None:
        self.extraction_stats = []  # Track which strategies work
    
    async def extract(
        self,
        data_source: Any,
        param_name: str,
        context_key: Optional[str] = None
    ) -> Any:
        """
        Extract data using multiple strategies until one works.
        
        Args:
            data_source: Data in ANY format
            param_name: What we're looking for (e.g., 'table_metadata')
            context_key: Optional key in context_providers (e.g., 'metadata')
        
        Returns:
            Extracted data (or None if all strategies fail)
        """
        logger.debug(f" SmartDataExtractor: Extracting '{param_name}' from {type(data_source).__name__}")
        
        strategies = [
            ('direct_string', self._extract_as_string),
            ('callable', self._extract_as_callable),
            ('dict_exact', self._extract_from_dict_exact),
            ('method_exact', self._extract_via_method_exact),
            ('attribute', self._extract_as_attribute),
            ('file_path', self._extract_from_file),
            ('json_string', self._extract_as_json),
            ('str_fallback', self._extract_str_fallback),
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                result = await strategy_func(data_source, param_name, context_key)
                if result is not None:
                    logger.info(f" Extracted '{param_name}' using strategy: {strategy_name}")
                    self.extraction_stats.append({
                        'param': param_name,
                        'strategy': strategy_name,
                        'success': True
                    })
                    return result
            except Exception as e:
                logger.debug(f"  Strategy '{strategy_name}' failed: {e}")
                continue
        
        # All strategies failed
        logger.warning(f" Could not extract '{param_name}' from {type(data_source).__name__}")
        self.extraction_stats.append({
            'param': param_name,
            'strategy': 'none',
            'success': False
        })
        return None
    
    async def _extract_as_string(self, data_source: Any, param_name: Any, context_key: Any) -> Optional[str]:
        """Strategy 1: Direct string"""
        if isinstance(data_source, str) and len(data_source) > 0:
            return data_source
        return None
    
    async def _extract_as_callable(self, data_source: Any, param_name: Any, context_key: Any) -> Optional[Any]:
        """Strategy 2: Callable (function/lambda)"""
        if callable(data_source) and not isinstance(data_source, type):
            result = data_source()
            # Handle async callables
            if asyncio.iscoroutine(result):
                result = await result
            return result
        return None
    
    async def _extract_from_dict_exact(self, data_source: Any, param_name: Any, context_key: Any) -> Optional[Any]:
        """Strategy 3: Dict with exact key match"""
        if isinstance(data_source, dict):
            if param_name in data_source:
                value = data_source[param_name]
                # If value is callable, call it
                if callable(value):
                    result = value()
                    return await result if asyncio.iscoroutine(result) else result
                return value
        return None
    
    async def _extract_via_method_exact(self, data_source: Any, param_name: Any, context_key: Any) -> Optional[Any]:
        """Strategy 5: Object with exact method name"""
        if hasattr(data_source, param_name):
            attr = getattr(data_source, param_name)
            if callable(attr):
                result = attr()
                return await result if asyncio.iscoroutine(result) else result
            return attr
        return None
    
    async def _extract_as_attribute(self, data_source: Any, param_name: Any, context_key: Any) -> Optional[Any]:
        """Strategy 7: Object attribute (non-callable)"""
        if hasattr(data_source, param_name):
            attr = getattr(data_source, param_name)
            if not callable(attr):
                return attr
        return None
    
    async def _extract_from_file(self, data_source: Any, param_name: Any, context_key: Any) -> Optional[str]:
        """Strategy 8: File path"""
        if isinstance(data_source, (str, Path)):
            path = Path(data_source)
            if path.exists() and path.is_file():
                try:
                    with open(path, 'r') as f:
                        return f.read()
                except Exception as e:
                    logger.debug(f"  Could not read file {path}: {e}")
        return None
    
    async def _extract_as_json(self, data_source: Any, param_name: Any, context_key: Any) -> Optional[Any]:
        """Strategy 9: JSON string"""
        if isinstance(data_source, str):
            try:
                data = json.loads(data_source)
                # Try to extract param_name from parsed JSON
                if isinstance(data, dict) and param_name in data:
                    return data[param_name]
                return data  # Return whole thing
            except json.JSONDecodeError:
                pass
        return None
    
    async def _extract_str_fallback(self, data_source: Any, param_name: Any, context_key: Any) -> Optional[str]:
        """Strategy 10: Convert to string (last resort)"""
        try:
            result = str(data_source)
            # Don't return object repr strings
            if not (result.startswith('<') and result.endswith('>')):
                return result
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"String conversion failed: {e}")
            pass
        return None
    
    def get_stats(self) -> Dict:
        """Get extraction statistics for debugging."""
        if not self.extraction_stats:
            return {}
        
        successful = [s for s in self.extraction_stats if s['success']]
        failed = [s for s in self.extraction_stats if not s['success']]
        
        strategy_counts = {}
        for stat in successful:
            strategy = stat['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'total_extractions': len(self.extraction_stats),
            'successful': len(successful),
            'failed': len(failed),
            'strategy_usage': strategy_counts,
            'failed_params': [s['param'] for s in failed]
        }

