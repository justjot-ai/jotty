"""
File Organizer Skill - Intelligently organize files and folders.

Helps maintain clean file structures by understanding context,
finding duplicates, and suggesting better organization.
"""
import asyncio
import logging
import inspect
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import os
import shutil

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("file-organizer")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def organize_files_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Organize files and folders intelligently.
    
    Args:
        params:
            - target_directory (str): Directory to organize
            - organization_strategy (str, optional): Strategy type
            - find_duplicates (bool, optional): Find duplicates
            - archive_old_files (bool, optional): Archive old files
            - age_threshold_days (int, optional): Days threshold
            - dry_run (bool, optional): Preview only
            - output_report (str, optional): Report path
    
    Returns:
        Dictionary with changes, duplicates, statistics
    """
    status.set_callback(params.pop('_status_callback', None))

    target_dir = params.get('target_directory', '')
    strategy = params.get('organization_strategy', 'auto')
    find_duplicates = params.get('find_duplicates', True)
    archive_old_files = params.get('archive_old_files', False)
    age_threshold_days = params.get('age_threshold_days', 180)
    dry_run = params.get('dry_run', False)
    output_report = params.get('output_report', None)
    
    if not target_dir:
        return {
            'success': False,
            'error': 'target_directory is required'
        }
    
    target_path = Path(os.path.expanduser(target_dir))
    if not target_path.exists():
        return {
            'success': False,
            'error': f'Directory does not exist: {target_dir}'
        }
    
    # Analyze current state
    analysis = await _analyze_directory(target_path)
    
    # Find duplicates if requested
    duplicates = []
    if find_duplicates:
        duplicates = await _find_duplicates(target_path)
    
    # Generate organization plan
    plan = await _generate_organization_plan(
        analysis, strategy, duplicates, archive_old_files, age_threshold_days
    )
    
    # Execute or preview
    changes = []
    if not dry_run:
        changes = await _execute_organization(plan, target_path)
    else:
        changes = plan.get('proposed_changes', [])
    
    # Generate statistics
    statistics = {
        'total_files': analysis.get('total_files', 0),
        'total_folders': analysis.get('total_folders', 0),
        'duplicates_found': len(duplicates),
        'files_to_move': len([c for c in changes if c.get('action') == 'move']),
        'files_to_rename': len([c for c in changes if c.get('action') == 'rename']),
        'files_to_delete': len([c for c in changes if c.get('action') == 'delete'])
    }
    
    # Generate report
    report_content = _generate_organization_report(analysis, duplicates, changes, statistics, dry_run)
    
    # Save report if requested
    report_path = None
    if output_report:
        report_path = Path(output_report)
        report_path.write_text(report_content, encoding='utf-8')
    
    return {
        'success': True,
        'changes': changes,
        'duplicates_found': duplicates,
        'statistics': statistics,
        'report_path': str(report_path) if report_path else None,
        'report': report_content
    }


async def _analyze_directory(directory: Path) -> Dict:
    """Analyze directory structure and file patterns."""
    
    files = []
    folders = []
    file_types = {}
    total_size = 0
    
    for item in directory.rglob('*'):
        if item.is_file():
            files.append(item)
            ext = item.suffix.lower() or 'no-extension'
            file_types[ext] = file_types.get(ext, 0) + 1
            try:
                total_size += item.stat().st_size
            except:
                pass
        elif item.is_dir():
            folders.append(item)
    
    return {
        'total_files': len(files),
        'total_folders': len(folders),
        'file_types': file_types,
        'total_size': total_size,
        'files': files[:100],  # Limit for processing
        'folders': folders
    }


async def _find_duplicates(directory: Path) -> List[Dict]:
    """Find duplicate files by content hash."""
    
    file_hashes = {}
    duplicates = []
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            try:
                # Calculate hash
                hash_md5 = hashlib.md5()
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        hash_md5.update(chunk)
                file_hash = hash_md5.hexdigest()
                
                if file_hash in file_hashes:
                    # Found duplicate
                    if file_hash not in [d['hash'] for d in duplicates]:
                        duplicates.append({
                            'hash': file_hash,
                            'files': [file_hashes[file_hash], str(file_path)]
                        })
                    else:
                        # Add to existing duplicate set
                        for dup in duplicates:
                            if dup['hash'] == file_hash:
                                dup['files'].append(str(file_path))
                                break
                else:
                    file_hashes[file_hash] = str(file_path)
            except Exception as e:
                logger.debug(f"Failed to hash {file_path}: {e}")
    
    return duplicates


async def _generate_organization_plan(
    analysis: Dict,
    strategy: str,
    duplicates: List[Dict],
    archive_old: bool,
    age_threshold: int
) -> Dict:
    """Generate organization plan."""
    
    proposed_changes = []
    
    # Categorize files by type
    type_categories = {
        'Documents': ['.pdf', '.doc', '.docx', '.txt', '.md', '.rtf'],
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp'],
        'Videos': ['.mp4', '.mov', '.avi', '.mkv', '.webm'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z', '.dmg'],
        'Code': ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs'],
        'Spreadsheets': ['.xlsx', '.xls', '.csv'],
        'Presentations': ['.pptx', '.ppt', '.key']
    }
    
    files = analysis.get('files', [])
    cutoff_date = datetime.now() - timedelta(days=age_threshold)
    
    for file_path in files:
        ext = file_path.suffix.lower()
        
        # Determine category
        category = 'Other'
        for cat, exts in type_categories.items():
            if ext in exts:
                category = cat
                break
        
        # Determine new location
        new_dir = file_path.parent / category
        new_path = new_dir / file_path.name
        
        if new_path != file_path:
            proposed_changes.append({
                'action': 'move',
                'from': str(file_path),
                'to': str(new_path),
                'category': category
            })
        
        # Check if should archive
        if archive_old:
            try:
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime < cutoff_date:
                    archive_dir = file_path.parent / 'Archive' / str(mtime.year)
                    archive_path = archive_dir / file_path.name
                    proposed_changes.append({
                        'action': 'archive',
                        'from': str(file_path),
                        'to': str(archive_path)
                    })
            except:
                pass
    
    return {
        'proposed_changes': proposed_changes,
        'strategy': strategy
    }


async def _execute_organization(plan: Dict, base_dir: Path) -> List[Dict]:
    """Execute organization plan."""
    
    changes = []
    executed = []
    
    for change in plan.get('proposed_changes', []):
        action = change.get('action')
        from_path = Path(change['from'])
        to_path = Path(change['to'])
        
        try:
            if action == 'move':
                # Create destination directory
                to_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move file
                if from_path.exists():
                    shutil.move(str(from_path), str(to_path))
                    executed.append({
                        'action': 'move',
                        'from': str(from_path),
                        'to': str(to_path),
                        'success': True
                    })
            
            elif action == 'archive':
                to_path.parent.mkdir(parents=True, exist_ok=True)
                if from_path.exists():
                    shutil.move(str(from_path), str(to_path))
                    executed.append({
                        'action': 'archive',
                        'from': str(from_path),
                        'to': str(to_path),
                        'success': True
                    })
        except Exception as e:
            logger.error(f"Failed to {action} {from_path}: {e}")
            executed.append({
                'action': action,
                'from': str(from_path),
                'to': str(to_path),
                'success': False,
                'error': str(e)
            })
    
    return executed


def _generate_organization_report(
    analysis: Dict,
    duplicates: List[Dict],
    changes: List[Dict],
    statistics: Dict,
    dry_run: bool
) -> str:
    """Generate organization report."""
    
    lines = [
        "# File Organization Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Mode:** {'Dry Run (Preview)' if dry_run else 'Executed'}",
        "",
        "## Summary",
        "",
        f"- Total files analyzed: {statistics.get('total_files', 0)}",
        f"- Duplicates found: {statistics.get('duplicates_found', 0)}",
        f"- Files to move: {statistics.get('files_to_move', 0)}",
        f"- Files to rename: {statistics.get('files_to_rename', 0)}",
        "",
        "## Duplicates Found",
        ""
    ]
    
    if duplicates:
        for i, dup in enumerate(duplicates[:10], 1):  # Top 10
            lines.append(f"### Duplicate Set {i}")
            for file_path in dup.get('files', []):
                lines.append(f"- {file_path}")
            lines.append("")
    else:
        lines.append("No duplicates found.")
        lines.append("")
    
    # Changes
    if changes:
        lines.append("## Changes Made")
        lines.append("")
        for change in changes[:20]:  # Top 20
            action = change.get('action', 'unknown')
            lines.append(f"- **{action.upper()}**: {change.get('from', '')} → {change.get('to', '')}")
        lines.append("")
    
    return "\n".join(lines)
