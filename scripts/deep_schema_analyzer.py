#!/usr/bin/env python3
"""
DEEP API Response Schema Analyzer
Performs comprehensive analysis of all API responses with full depth exploration
"""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict, OrderedDict
import re
from datetime import datetime

class DeepSchemaAnalyzer:
    def __init__(self):
        self.base_path = Path("data/api_responses")
        self.schemas = {}
        self.field_stats = defaultdict(lambda: {
            'occurrences': 0,
            'types': set(),
            'examples': [],
            'paths': set(),
            'nullable': False,
            'formats': set(),
            'min_value': None,
            'max_value': None,
            'min_length': None,
            'max_length': None,
            'unique_values': set(),
            'patterns': set()
        })
        
    def detect_format(self, value: str) -> Optional[str]:
        """Detect specific format patterns in string values"""
        if not isinstance(value, str):
            return None
            
        # Date patterns
        if re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            return 'date:YYYY-MM-DD'
        if re.match(r'^\d{8}T\d{6}$', value):
            return 'timestamp:YYYYMMDDTHHMMSS'
        if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', value):
            return 'datetime:YYYY-MM-DD HH:MM:SS'
            
        # Financial patterns
        if re.match(r'^-?\d+\.\d+$', value):
            return 'decimal'
        if re.match(r'^-?\d+$', value):
            return 'integer_string'
        if re.match(r'^[A-Z]{1,5}$', value):
            return 'ticker_symbol'
        if re.match(r'^[A-Z]+\d{6}[CP]\d{8}$', value):
            return 'option_contract_id'
            
        # URLs and identifiers
        if value.startswith('http://') or value.startswith('https://'):
            return 'url'
        if re.match(r'^[A-Z0-9]{8}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{12}$', value, re.I):
            return 'uuid'
            
        # Special values
        if value in ['None', 'null', 'N/A', 'n/a']:
            return 'null_string'
            
        return None
    
    def analyze_value(self, value: Any, path: str = "", depth: int = 0, max_depth: int = 20) -> Dict[str, Any]:
        """Deep analysis of a value with all its characteristics"""
        
        if depth > max_depth:
            return {"type": "deep_nested", "depth_exceeded": True}
            
        result = {
            "type": None,
            "nullable": value is None or value == "None",
            "path": path,
            "depth": depth
        }
        
        if value is None or value == "None":
            result["type"] = "null"
            return result
            
        elif isinstance(value, bool):
            result.update({
                "type": "boolean",
                "value": value
            })
            
        elif isinstance(value, (int, float)):
            result.update({
                "type": "number" if isinstance(value, float) else "integer",
                "value": value,
                "numeric_range": {"min": value, "max": value}
            })
            
        elif isinstance(value, str):
            format_type = self.detect_format(value)
            result.update({
                "type": "string",
                "length": len(value),
                "format": format_type,
                "sample": value[:100] if len(value) <= 100 else value[:97] + "...",
                "is_empty": len(value) == 0
            })
            
            # Try to parse as number
            try:
                float_val = float(value)
                result["parseable_as_number"] = True
                result["numeric_value"] = float_val
            except:
                pass
                
        elif isinstance(value, list):
            result["type"] = "array"
            result["length"] = len(value)
            
            if value:
                # Analyze array items deeply
                item_schemas = []
                unique_types = set()
                
                # Sample up to 100 items for analysis
                sample_size = min(len(value), 100)
                for i in range(sample_size):
                    item_schema = self.analyze_value(value[i], f"{path}[{i}]", depth + 1)
                    item_schemas.append(item_schema)
                    unique_types.add(item_schema.get("type"))
                
                result["items"] = {
                    "schemas": item_schemas[:5],  # Keep first 5 for reference
                    "unique_types": list(unique_types),
                    "homogeneous": len(unique_types) == 1,
                    "sample_size": sample_size,
                    "total_items": len(value)
                }
                
                # If homogeneous and object type, merge schemas
                if len(unique_types) == 1 and "object" in unique_types:
                    merged_properties = self.merge_object_schemas(item_schemas)
                    result["items"]["merged_schema"] = merged_properties
                    
        elif isinstance(value, dict):
            result["type"] = "object"
            result["property_count"] = len(value)
            
            properties = {}
            required = []
            optional = []
            
            for key, val in value.items():
                prop_schema = self.analyze_value(val, f"{path}.{key}", depth + 1)
                properties[key] = prop_schema
                
                if val is not None and val != "None":
                    required.append(key)
                else:
                    optional.append(key)
                    
                # Track field statistics
                self.update_field_stats(key, val, f"{path}.{key}")
            
            result.update({
                "properties": properties,
                "required": required,
                "optional": optional,
                "property_names": list(value.keys())
            })
            
        else:
            result["type"] = f"unknown:{type(value).__name__}"
            
        return result
    
    def update_field_stats(self, field_name: str, value: Any, path: str):
        """Update global statistics for a field"""
        stats = self.field_stats[field_name]
        stats['occurrences'] += 1
        stats['paths'].add(path)
        
        if value is None or value == "None":
            stats['nullable'] = True
            
        value_type = type(value).__name__
        stats['types'].add(value_type)
        
        # Collect examples (up to 5 unique)
        if len(stats['examples']) < 5:
            if value not in stats['examples']:
                stats['examples'].append(value)
                
        # String-specific stats
        if isinstance(value, str):
            format_type = self.detect_format(value)
            if format_type:
                stats['formats'].add(format_type)
                
            if stats['min_length'] is None or len(value) < stats['min_length']:
                stats['min_length'] = len(value)
            if stats['max_length'] is None or len(value) > stats['max_length']:
                stats['max_length'] = len(value)
                
            # Collect unique values for enums (if not too many)
            if len(stats['unique_values']) < 50:
                stats['unique_values'].add(value)
                
        # Numeric stats
        elif isinstance(value, (int, float)):
            if stats['min_value'] is None or value < stats['min_value']:
                stats['min_value'] = value
            if stats['max_value'] is None or value > stats['max_value']:
                stats['max_value'] = value
    
    def merge_object_schemas(self, schemas: List[Dict]) -> Dict:
        """Merge multiple object schemas to find common structure"""
        if not schemas:
            return {}
            
        merged = {
            "all_properties": set(),
            "always_present": set(),
            "sometimes_present": set(),
            "property_types": defaultdict(set)
        }
        
        # Collect all properties
        for schema in schemas:
            if "properties" in schema:
                props = set(schema["properties"].keys())
                merged["all_properties"].update(props)
                
                for prop, prop_schema in schema["properties"].items():
                    merged["property_types"][prop].add(prop_schema.get("type"))
        
        # Find always vs sometimes present
        for prop in merged["all_properties"]:
            present_count = sum(1 for s in schemas if "properties" in s and prop in s["properties"])
            if present_count == len(schemas):
                merged["always_present"].add(prop)
            else:
                merged["sometimes_present"].add(prop)
                
        merged["all_properties"] = list(merged["all_properties"])
        merged["always_present"] = list(merged["always_present"])
        merged["sometimes_present"] = list(merged["sometimes_present"])
        
        # Convert property types to list
        merged["property_types"] = {k: list(v) for k, v in merged["property_types"].items()}
        
        return merged
    
    def analyze_file(self, file_path: Path) -> Optional[Dict]:
        """Analyze a single file comprehensively"""
        print(f"  Analyzing: {file_path.name}")
        
        try:
            if file_path.suffix == ".json":
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    schema = self.analyze_value(data, str(file_path.stem))
                    
                    # Add file metadata
                    schema["_metadata"] = {
                        "file": str(file_path),
                        "size_bytes": file_path.stat().st_size,
                        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    }
                    
                    return schema
                    
            elif file_path.suffix == ".csv":
                with open(file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    
                    schema = {
                        "type": "csv",
                        "row_count": len(rows),
                        "columns": list(rows[0].keys()) if rows else [],
                        "_metadata": {
                            "file": str(file_path),
                            "size_bytes": file_path.stat().st_size
                        }
                    }
                    
                    if rows:
                        # Analyze column types from first 100 rows
                        column_analysis = {}
                        sample_size = min(len(rows), 100)
                        
                        for col in rows[0].keys():
                            col_values = [row[col] for row in rows[:sample_size]]
                            col_types = set()
                            col_formats = set()
                            
                            for val in col_values:
                                if val:
                                    format_type = self.detect_format(val)
                                    if format_type:
                                        col_formats.add(format_type)
                                    
                                    # Try to determine type
                                    try:
                                        float(val)
                                        col_types.add("numeric")
                                    except:
                                        col_types.add("string")
                                        
                            column_analysis[col] = {
                                "types": list(col_types),
                                "formats": list(col_formats),
                                "sample_values": col_values[:5]
                            }
                            
                        schema["column_analysis"] = column_analysis
                        schema["sample_rows"] = rows[:3]
                    
                    return schema
                    
        except Exception as e:
            print(f"    ERROR: {e}")
            return {"error": str(e), "file": str(file_path)}
    
    def analyze_directory(self, directory: Path, level: int = 0) -> Dict[str, Any]:
        """Recursively analyze all files in directory"""
        indent = "  " * level
        print(f"{indent}📁 {directory.name}/")
        
        results = {}
        
        for item in sorted(directory.iterdir()):
            if item.is_dir():
                results[item.name] = self.analyze_directory(item, level + 1)
            elif item.is_file() and item.suffix in ['.json', '.csv']:
                schema = self.analyze_file(item)
                if schema:
                    results[item.stem] = schema
                    
        return results
    
    def generate_detailed_report(self, schemas: Dict) -> str:
        """Generate a detailed markdown report"""
        lines = ["# Deep API Schema Analysis Report", ""]
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("")
        
        # Table of contents
        lines.append("## Table of Contents")
        for category in schemas.keys():
            lines.append(f"- [{category}](#{category})")
        lines.append("")
        
        # Detailed schemas
        for category, endpoints in schemas.items():
            if not isinstance(endpoints, dict):
                continue
                
            lines.append(f"## {category}")
            lines.append("")
            
            for endpoint, schema in endpoints.items():
                if not isinstance(schema, dict):
                    continue
                    
                lines.append(f"### {endpoint}")
                
                # File metadata
                if "_metadata" in schema:
                    meta = schema["_metadata"]
                    lines.append(f"**File**: `{meta['file']}`")
                    lines.append(f"**Size**: {meta['size_bytes']:,} bytes")
                    lines.append("")
                
                # Schema details
                lines.append("#### Structure")
                lines.append("```json")
                lines.extend(self.format_schema_tree(schema, max_depth=5))
                lines.append("```")
                lines.append("")
                
        # Field statistics
        lines.append("## Global Field Statistics")
        lines.append("")
        
        for field, stats in sorted(self.field_stats.items()):
            if stats['occurrences'] > 1:  # Only show fields that appear multiple times
                lines.append(f"### {field}")
                lines.append(f"- **Occurrences**: {stats['occurrences']}")
                lines.append(f"- **Types**: {', '.join(stats['types'])}")
                if stats['formats']:
                    lines.append(f"- **Formats**: {', '.join(stats['formats'])}")
                lines.append(f"- **Nullable**: {stats['nullable']}")
                if stats['examples']:
                    lines.append(f"- **Examples**: {stats['examples'][:3]}")
                lines.append("")
        
        return "\n".join(lines)
    
    def format_schema_tree(self, schema: Dict, indent: int = 0, max_depth: int = 10) -> List[str]:
        """Format schema as tree structure"""
        if indent > max_depth:
            return ["  " * indent + "..."]
            
        lines = []
        prefix = "  " * indent
        
        if schema.get("type") == "object":
            if "properties" in schema:
                for prop, prop_schema in schema["properties"].items():
                    type_str = prop_schema.get("type", "unknown")
                    format_str = prop_schema.get("format", "")
                    
                    if format_str:
                        lines.append(f"{prefix}{prop}: {type_str} ({format_str})")
                    else:
                        lines.append(f"{prefix}{prop}: {type_str}")
                        
                    if type_str == "object" and indent < max_depth:
                        lines.extend(self.format_schema_tree(prop_schema, indent + 1, max_depth))
                    elif type_str == "array" and "items" in prop_schema:
                        lines.append(f"{prefix}  [{prop_schema['items'].get('unique_types', ['unknown'])}]")
                        
        elif schema.get("type") == "array":
            if "items" in schema:
                items = schema["items"]
                lines.append(f"{prefix}Array ({items.get('total_items', 0)} items)")
                if "merged_schema" in items:
                    merged = items["merged_schema"]
                    lines.append(f"{prefix}  Properties: {', '.join(merged.get('all_properties', []))}")
                    
        return lines
    
    def run_deep_analysis(self):
        """Run the complete deep analysis"""
        print("\n" + "=" * 80)
        print("DEEP API RESPONSE SCHEMA ANALYSIS")
        print("=" * 80 + "\n")
        
        # Analyze all files
        schemas = self.analyze_directory(self.base_path)
        
        # Save complete deep schema
        print("\n📊 Saving results...")
        
        with open("data/deep_api_schemas.json", "w") as f:
            json.dump(schemas, f, indent=2, default=str)
            
        # Generate detailed report
        report = self.generate_detailed_report(schemas)
        with open("data/DEEP_SCHEMA_ANALYSIS.md", "w") as f:
            f.write(report)
            
        # Save field statistics
        field_stats_clean = {}
        for field, stats in self.field_stats.items():
            field_stats_clean[field] = {
                "occurrences": stats["occurrences"],
                "types": list(stats["types"]),
                "formats": list(stats["formats"]),
                "nullable": stats["nullable"],
                "examples": stats["examples"][:5],
                "paths_count": len(stats["paths"]),
                "min_value": stats["min_value"],
                "max_value": stats["max_value"],
                "min_length": stats["min_length"],
                "max_length": stats["max_length"],
                "unique_values_count": len(stats["unique_values"])
            }
            
        with open("data/field_statistics.json", "w") as f:
            json.dump(field_stats_clean, f, indent=2, default=str)
            
        print("\n✅ Deep analysis complete!")
        print("📁 Deep schemas: data/deep_api_schemas.json")
        print("📄 Detailed report: data/DEEP_SCHEMA_ANALYSIS.md")
        print("📊 Field statistics: data/field_statistics.json")
        
        # Print summary
        self.print_analysis_summary(schemas)
    
    def print_analysis_summary(self, schemas: Dict):
        """Print analysis summary"""
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        
        total_files = 0
        total_fields = len(self.field_stats)
        categories = {}
        
        def count_files(obj, cat_name=""):
            nonlocal total_files
            for key, value in obj.items():
                if isinstance(value, dict):
                    if "_metadata" in value:
                        total_files += 1
                        if cat_name:
                            categories[cat_name] = categories.get(cat_name, 0) + 1
                    else:
                        count_files(value, key if not cat_name else cat_name)
                        
        count_files(schemas)
        
        print(f"\n📊 Total files analyzed: {total_files}")
        print(f"📋 Unique fields discovered: {total_fields}")
        print(f"\n📁 Categories:")
        for cat, count in categories.items():
            print(f"  • {cat}: {count} files")
            
        # Most common fields
        print(f"\n🔍 Most common fields:")
        sorted_fields = sorted(self.field_stats.items(), key=lambda x: x[1]['occurrences'], reverse=True)
        for field, stats in sorted_fields[:10]:
            print(f"  • {field}: {stats['occurrences']} occurrences")

if __name__ == "__main__":
    analyzer = DeepSchemaAnalyzer()
    analyzer.run_deep_analysis()