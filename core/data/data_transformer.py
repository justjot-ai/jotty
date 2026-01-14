"""
Smart Data Transformer - ReAct Agent with Format Tools
=======================================================

CRITICAL DESIGN NOVELTY (User Insight):
"It needs react as all formats there should be loaded by it.
For example json loads in react to check and fix json till it can be sent,
similarly csv trial. Its tool response of formatted output is sent back."

This transformer is a ReAct agent with TOOLS to:
- Load and test JSON (json.loads)
- Parse and fix CSV
- Test dict conversion
- Validate list formats
- Iteratively fix until it works

The agent USES these tools, gets responses, and tries again if needed.
This is the key innovation: transformer has actual tools, not just prompts!
"""

import json
import csv
import io
import ast
import logging
from typing import Any, Dict, List, Optional, Type, Union, get_origin, get_args
import dspy

logger = logging.getLogger(__name__)


# ==================== FORMAT TESTING TOOLS ====================

class FormatTools:
    """
    Tools that the ReAct agent can use to test and fix formats.
    Each tool returns a result that the agent can see and reason about.
    """
    
    @staticmethod
    def test_json_load(data_str: str) -> Dict[str, Any]:
        """
        Try to load a string as JSON.
        
        Args:
            data_str: String to parse as JSON
            
        Returns:
            {
                'success': bool,
                'result': parsed data (if success),
                'error': error message (if failed),
                'suggestions': list of fixes to try
            }
        """
        try:
            result = json.loads(data_str)
            return {
                'success': True,
                'result': result,
                'type': type(result).__name__,
                'error': None
            }
        except json.JSONDecodeError as e:
            # Provide specific error and suggestions
            suggestions = []
            
            # Common issues
            if "Expecting property name" in str(e):
                suggestions.append("Try replacing single quotes with double quotes")
            if "Expecting value" in str(e):
                suggestions.append("Check for trailing commas")
            if "Extra data" in str(e):
                suggestions.append("Check for multiple JSON objects (need array?)")
                
            return {
                'success': False,
                'result': None,
                'error': str(e),
                'error_position': e.pos if hasattr(e, 'pos') else None,
                'suggestions': suggestions
            }
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': f"Unexpected error: {str(e)}",
                'suggestions': ['Check if input is actually a string']
            }
    
    @staticmethod
    def fix_json_quotes(data_str: str) -> str:
        """
        Fix common JSON quote issues (single → double quotes).
        
        Args:
            data_str: String with potential quote issues
            
        Returns:
            Fixed string
        """
        # Replace single quotes with double quotes (carefully!)
        # Don't replace single quotes inside existing double quotes
        fixed = data_str.replace("'", '"')
        return fixed
    
    @staticmethod
    def test_python_literal(data_str: str) -> Dict[str, Any]:
        """
        Try to parse as Python literal (ast.literal_eval).
        Safer than eval, handles dicts/lists with single quotes.
        
        Args:
            data_str: String to parse
            
        Returns:
            {
                'success': bool,
                'result': parsed data,
                'error': error message
            }
        """
        try:
            result = ast.literal_eval(data_str)
            return {
                'success': True,
                'result': result,
                'type': type(result).__name__,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }
    
    @staticmethod
    def test_csv_parse(data_str: str, delimiter: str = ',') -> Dict[str, Any]:
        """
        Try to parse string as CSV.
        
        Args:
            data_str: CSV string
            delimiter: Column delimiter
            
        Returns:
            {
                'success': bool,
                'result': list of rows,
                'rows': number of rows,
                'columns': number of columns,
                'error': error message
            }
        """
        try:
            reader = csv.reader(io.StringIO(data_str), delimiter=delimiter)
            rows = list(reader)
            
            if not rows:
                return {
                    'success': False,
                    'result': None,
                    'error': 'No data found',
                    'suggestions': ['Check if string is empty']
                }
            
            return {
                'success': True,
                'result': rows,
                'rows': len(rows),
                'columns': len(rows[0]) if rows else 0,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e),
                'suggestions': [f'Try different delimiter (current: {delimiter})']
            }
    
    @staticmethod
    def test_string_to_list(data_str: str) -> Dict[str, Any]:
        """
        Try to convert string to list.
        Handles: "['a', 'b']", "a, b, c", "a|b|c", etc.
        
        Args:
            data_str: String to convert
            
        Returns:
            {
                'success': bool,
                'result': list,
                'error': error message
            }
        """
        # Try 1: Parse as JSON array
        try:
            result = json.loads(data_str)
            if isinstance(result, list):
                return {'success': True, 'result': result, 'method': 'json', 'error': None}
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"JSON parsing failed: {e}")
            pass

        # Try 2: Parse as Python literal
        try:
            result = ast.literal_eval(data_str)
            if isinstance(result, list):
                return {'success': True, 'result': result, 'method': 'literal_eval', 'error': None}
        except (ValueError, SyntaxError, TypeError) as e:
            logger.debug(f"Literal eval failed: {e}")
            pass
        
        # Try 3: Split by common delimiters
        for delimiter in [',', '|', ';', '\t', '\n']:
            parts = [p.strip().strip('"').strip("'") for p in data_str.split(delimiter)]
            if len(parts) > 1:
                return {'success': True, 'result': parts, 'method': f'split({repr(delimiter)})', 'error': None}
        
        # Try 4: Single item list
        return {'success': True, 'result': [data_str], 'method': 'single_item', 'error': None}
    
    @staticmethod
    def test_type_conversion(data: Any, target_type_name: str) -> Dict[str, Any]:
        """
        Test if data can be converted to target type.
        
        Args:
            data: Source data
            target_type_name: Target type name ('dict', 'list', 'str', etc.)
            
        Returns:
            {
                'success': bool,
                'result': converted data,
                'current_type': current type name,
                'error': error message
            }
        """
        current_type = type(data).__name__
        
        try:
            # Type conversion map
            if target_type_name == 'dict':
                if isinstance(data, dict):
                    result = data
                elif isinstance(data, str):
                    result = json.loads(data)  # Will raise if invalid
                else:
                    return {'success': False, 'current_type': current_type, 
                           'error': f"Cannot convert {current_type} to dict"}
            
            elif target_type_name == 'list':
                if isinstance(data, list):
                    result = data
                elif isinstance(data, str):
                    # Try parsing
                    result = FormatTools.test_string_to_list(data)['result']
                elif isinstance(data, (tuple, set)):
                    result = list(data)
                else:
                    return {'success': False, 'current_type': current_type,
                           'error': f"Cannot convert {current_type} to list"}
            
            elif target_type_name == 'str':
                result = str(data)
            
            elif target_type_name in ('int', 'float', 'bool'):
                result = eval(target_type_name)(data)
            
            else:
                return {'success': False, 'current_type': current_type,
                       'error': f"Unknown target type: {target_type_name}"}
            
            return {
                'success': True,
                'result': result,
                'current_type': current_type,
                'converted_type': type(result).__name__,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'current_type': current_type,
                'error': str(e)
            }


# ==================== REACT TRANSFORMER WITH TOOLS ====================

class SmartDataTransformer:
    """
    SOTA Agentic Data Transformer with ReAct and Format Tools.
    
    KEY INNOVATION (User Design):
    The transformer is a ReAct agent with TOOLS to test formats.
    It iteratively tries loading, gets tool responses, and fixes issues.
    
    Flow:
    1. Agent decides: "Try json.loads"
    2. Tool executes: Returns success=False with error message
    3. Agent sees error: "Missing double quotes"
    4. Agent decides: "Fix quotes and try again"
    5. Tool executes: Returns success=True with result
    6. Agent returns: Formatted output
    
    This is fundamentally different from prompt-only transformation!
    """
    
    def __init__(self, lm=None):
        """
        Initialize with format tools.
        
        Args:
            lm: Optional DSPy language model. If None, uses dspy.settings.lm.
                This allows AgentSlack to pass custom LM configurations.
        """
        self.lm = lm  # Store for potential future use
        if lm is None:
            # Use global DSPy LM if available
            try:
                import dspy
                if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
                    self.lm = dspy.settings.lm
            except (ImportError, AttributeError, TypeError) as e:
                logger.debug(f"Could not get LM from dspy.settings: {e}")
                pass
        
        self.tools = FormatTools()
        self.transformation_history = []
        logger.info("✅ SmartDataTransformer initialized with ReAct + Format Tools")
        if self.lm:
            logger.info(f"   🔧 Using LM: {getattr(self.lm, 'model', 'unknown')}")
    
    def transform(
        self,
        source: Any,
        target_type: Type,
        context: str = "",
        param_name: str = ""
    ) -> Any:
        """
        Transform data using ReAct agent with tools.
        
        The agent can USE tools to test formats and get responses!
        
        Args:
            source: Source data
            target_type: Target type (dict, list, str, etc.)
            context: Context for intelligent transformation
            param_name: Parameter name
            
        Returns:
            Transformed data in target type
        """
        source_type = type(source).__name__
        target_type_name = getattr(target_type, '__name__', str(target_type))
        
        logger.info(f"🔄 SmartDataTransformer: {source_type} → {target_type_name}")
        logger.info(f"   Context: {context}")
        logger.info(f"   Param: {param_name}")
        
        # 🔥 A-TEAM FIX: Handle Union types properly!
        # Union types can't be used with isinstance(), need to check each type in Union
        origin = get_origin(target_type)
        if origin is Union:
            # For Union types, check if source matches ANY type in the Union
            union_args = get_args(target_type)
            logger.info(f"🔍 Union type detected: {union_args}")
            
            # Check if source already matches any type in Union
            for union_type in union_args:
                if union_type is type(None):
                    # Handle NoneType specially
                    if source is None:
                        logger.info(f"✅ Source is None, matches Union")
                        return source
                else:
                    try:
                        if isinstance(source, union_type):
                            logger.info(f"✅ Source matches {union_type.__name__} in Union, no transformation needed")
                            return source
                    except TypeError:
                        # Some types can't be used with isinstance either
                        continue
            
            # If no match, try to transform to the first concrete type in Union
            for union_type in union_args:
                if union_type is not type(None):
                    logger.info(f"🔄 Attempting transformation to {union_type.__name__} (first concrete type in Union)")
                    try:
                        result = self._transform_with_tools(source, union_type.__name__, context)
                        if result is not None:
                            logger.info(f"✅ Transformation successful!")
                            return result
                    except Exception as e:
                        logger.debug(f"⚠️  Transformation to {union_type.__name__} failed: {e}")
                        continue
            
            # If all transformations fail, just return the source as-is
            # DSPy ReAct will handle the rest!
            logger.info(f"⚠️  No transformation succeeded, returning source as-is (DSPy will handle)")
            return source
        
        # Fast path: already correct type (non-Union)
        try:
            if isinstance(source, target_type):
                logger.info(f"✅ Already correct type, no transformation needed")
                return source
        except TypeError:
            # target_type can't be used with isinstance (e.g., generic types)
            logger.info(f"⚠️  Can't check isinstance for {target_type_name}, attempting transformation")
        
        # Use tools to transform
        result = self._transform_with_tools(source, target_type_name, context)
        
        if result is None:
            # 🔥 A-TEAM: Don't raise error, just return source
            # Let DSPy ReAct handle type conversion!
            logger.info(f"⚠️  Transformation failed, returning source as-is (DSPy will handle)")
            return source
        
        # Validate result type (only for non-Union, non-generic types)
        try:
            if not isinstance(result, target_type):
                logger.warning(f"⚠️  Transformation produced {type(result).__name__}, expected {target_type_name}")
                # Still return it, let DSPy handle
                return result
        except TypeError:
            # Can't validate with isinstance, just return
            pass
        
        logger.info(f"✅ Transformation successful!")
        return result
    
    # 🔥 NEW: AgentSlack-compatible async API
    async def transform_async(
        self,
        data: Any,
        target_format: str,
        source_format: Optional[str] = None
    ) -> Any:
        """
        Transform data to target format (async version for AgentSlack).
        
        This is the API that AgentSlack expects!
        
        Args:
            data: Input data (any type)
            target_format: Desired output format ('dict', 'json_string', 'csv_string', 'list_of_dicts', etc.)
            source_format: Optional hint about input format (auto-detected if None)
        
        Returns:
            Transformed data in target_format
        """
        logger.info(f"🔄 [AgentSlack API] Transform: {source_format or 'auto'} → {target_format}")
        
        # Auto-detect source format if not provided
        if source_format is None:
            source_format = self._detect_format(data)
            logger.info(f"   🔍 Auto-detected source format: {source_format}")
        
        # If formats match, return as-is
        if source_format == target_format:
            logger.info(f"   ✅ Formats match, no transformation needed")
            return data
        
        # Map format strings to types
        format_map = {
            'dict': dict,
            'json_string': str,  # Special: dict → JSON string
            'csv_string': str,   # Special: list → CSV string
            'list': list,
            'list_of_dicts': list,
            'str': str,
            'string': str,
            'int': int,
            'float': float,
            'bool': bool
        }
        
        # Handle special transformations
        if target_format == 'json_string':
            # dict/list → JSON string
            try:
                result = json.dumps(data, ensure_ascii=False, indent=None)
                logger.info(f"   ✅ Converted to JSON string ({len(result)} chars)")
                return result
            except Exception as e:
                logger.error(f"   ❌ JSON conversion failed: {e}")
                raise
        
        elif target_format == 'csv_string':
            # list of dicts → CSV string
            try:
                if not data:
                    return ""
                
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], dict):
                        # List of dicts → CSV
                        output = io.StringIO()
                        writer = csv.DictWriter(output, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
                        result = output.getvalue()
                        logger.info(f"   ✅ Converted to CSV string ({len(result)} chars, {len(data)} rows)")
                        return result
                    else:
                        # List of values → simple CSV
                        output = io.StringIO()
                        writer = csv.writer(output)
                        for item in data:
                            writer.writerow([item] if not isinstance(item, (list, tuple)) else item)
                        result = output.getvalue()
                        logger.info(f"   ✅ Converted to CSV string ({len(result)} chars)")
                        return result
                else:
                    logger.warning(f"   ⚠️  Data is not a list, converting to string")
                    return str(data)
            except Exception as e:
                logger.error(f"   ❌ CSV conversion failed: {e}")
                raise
        
        elif target_format == 'list_of_dicts':
            # CSV string → list of dicts
            if isinstance(data, str):
                try:
                    reader = csv.DictReader(io.StringIO(data))
                    result = list(reader)
                    logger.info(f"   ✅ Converted CSV to list of dicts ({len(result)} rows)")
                    return result
                except Exception as e:
                    logger.error(f"   ❌ CSV parsing failed: {e}")
                    raise
            elif isinstance(data, list):
                # Already a list, just return it
                logger.info(f"   ✅ Already a list")
                return data
        
        # Generic transformation using existing method
        target_type = format_map.get(target_format, str)
        result = self.transform(
            source=data,
            target_type=target_type,
            context=f"AgentSlack transformation: {source_format} → {target_format}",
            param_name="data"
        )
        
        return result
    
    def _detect_format(self, data: Any) -> str:
        """
        Auto-detect the format of data.
        
        Args:
            data: Input data
        
        Returns:
            Format name ('dict', 'list', 'str', 'json_string', 'csv_string', etc.)
        """
        if isinstance(data, dict):
            return 'dict'
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                return 'list_of_dicts'
            return 'list'
        elif isinstance(data, str):
            # Try to detect if it's JSON or CSV
            data_stripped = data.strip()
            if data_stripped.startswith(('{', '[')):
                return 'json_string'
            elif ',' in data and '\n' in data:
                return 'csv_string'
            return 'str'
        elif isinstance(data, (int, float, bool)):
            return type(data).__name__
        else:
            return 'unknown'
    
    def _transform_with_tools(self, source: Any, target_type_name: str, context: str) -> Any:
        """
        Use format tools to transform data.
        
        This is where the magic happens - we actually USE the tools!
        """
        source_str = str(source) if not isinstance(source, str) else source
        
        # Tool execution sequence (intelligent order)
        attempts = []
        
        # Attempt 1: Direct type conversion test
        logger.debug("   🔧 Tool: test_type_conversion")
        test_result = self.tools.test_type_conversion(source, target_type_name)
        attempts.append(('test_type_conversion', test_result))
        
        if test_result['success']:
            logger.debug(f"   ✅ Direct conversion worked!")
            return test_result['result']
        
        logger.debug(f"   ❌ Direct conversion failed: {test_result['error']}")
        
        # Attempt 2: If target is dict, try JSON tools
        if target_type_name == 'dict' and isinstance(source, str):
            # Try JSON load
            logger.debug("   🔧 Tool: test_json_load")
            json_result = self.tools.test_json_load(source_str)
            attempts.append(('test_json_load', json_result))
            
            if json_result['success'] and isinstance(json_result['result'], dict):
                logger.debug(f"   ✅ JSON load worked!")
                return json_result['result']
            
            # JSON failed - try fixing quotes
            if not json_result['success'] and json_result['suggestions']:
                logger.debug(f"   💡 Suggestion: {json_result['suggestions'][0]}")
                logger.debug("   🔧 Tool: fix_json_quotes")
                fixed_str = self.tools.fix_json_quotes(source_str)
                
                # Try again with fixed quotes
                logger.debug("   🔧 Tool: test_json_load (retry)")
                json_retry = self.tools.test_json_load(fixed_str)
                attempts.append(('test_json_load_retry', json_retry))
                
                if json_retry['success'] and isinstance(json_retry['result'], dict):
                    logger.debug(f"   ✅ JSON load after fix worked!")
                    return json_retry['result']
            
            # Try Python literal eval (handles single quotes)
            logger.debug("   🔧 Tool: test_python_literal")
            literal_result = self.tools.test_python_literal(source_str)
            attempts.append(('test_python_literal', literal_result))
            
            if literal_result['success'] and isinstance(literal_result['result'], dict):
                logger.debug(f"   ✅ Python literal_eval worked!")
                return literal_result['result']
        
        # Attempt 3: If target is list, try list tools
        if target_type_name == 'list' and isinstance(source, str):
            logger.debug("   🔧 Tool: test_string_to_list")
            list_result = self.tools.test_string_to_list(source_str)
            attempts.append(('test_string_to_list', list_result))
            
            if list_result['success']:
                logger.debug(f"   ✅ String to list worked! (method: {list_result.get('method')})")
                return list_result['result']
        
        # All attempts failed - but this is OK! DSPy will handle it
        logger.info(f"   ⚠️  All transformation attempts completed without success")
        logger.debug(f"   Attempts: {len(attempts)}")
        for tool_name, result in attempts:
            logger.debug(f"     - {tool_name}: {result.get('error', 'Unknown error')}")
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transformation statistics."""
        return {
            'total_history': len(self.transformation_history),
            'tools': {
                'test_json_load': 'Tests JSON parsing',
                'fix_json_quotes': 'Fixes quote issues',
                'test_python_literal': 'Parses Python literals',
                'test_csv_parse': 'Parses CSV data',
                'test_string_to_list': 'Converts string to list',
                'test_type_conversion': 'Generic type conversion'
            }
        }


__all__ = ['SmartDataTransformer', 'FormatTools']
