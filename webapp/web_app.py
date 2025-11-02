"""
Web interface for prtime - GitHub ETA tracker
Provides real-time monitoring of PR/Issue ETA tables
"""
import os
import sys
import json
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import logging

# Add parent directory (for prtime module) to path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
sys.path.insert(0, _parent_dir)

from prtime import (
    Github, load_settings, parse_eta, get_pr_id, 
    is_issue, was_updated, prev_monday, hours_row
)

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    format='%(asctime)s %(levelname).4s: %(message)s',
    level=logging.INFO
)
_logger = logging.getLogger(__name__)

# Global state
progress_state = {
    'status': 'idle',
    'current_repo': '',
    'current_item': '',
    'processed': 0,
    'total': 0,
    'errors': [],
    'warnings': [],
    'results': {
        'open_issues': [],
        'closed_issues': [],
        'validation_errors': [],
        'this_week_updates': [],
        'missing_eta_tables': [],
        'summary': {}
    },
    'start_time': None,
    'end_time': None
}

def reset_progress():
    """Reset progress state"""
    global progress_state
    progress_state.update({
        'status': 'idle',
        'current_repo': '',
        'current_item': '',
        'processed': 0,
        'total': 0,
        'errors': [],
        'warnings': [],
        'results': {
            'open_issues': [],
            'closed_issues': [],
            'validation_errors': [],
            'this_week_updates': [],
            'missing_eta_tables': [],
            'summary': {}
        },
        'start_time': None,
        'end_time': None
    })


def analyze_repository(settings_file='../settings.json', max_items=400):
    """Main analysis function that runs in background thread"""
    global progress_state
    
    try:
        progress_state['status'] = 'running'
        progress_state['start_time'] = datetime.now().isoformat()
        
        # Load settings - this updates the global settings dict in prtime module
        import prtime
        # Resolve relative path from webapp directory
        if not os.path.isabs(settings_file):
            settings_file = os.path.join(_parent_dir, settings_file)
        settings = load_settings(settings_file)
        prtime.settings = settings  # Update the global in prtime module
        hours_row.init()
        
        # Initialize GitHub client
        gh_key = "GITHUB_PAT"
        if gh_key not in os.environ:
            progress_state['errors'].append("GITHUB_PAT not found in environment")
            progress_state['status'] = 'error'
            return
        
        gh = Github(os.environ[gh_key])
        start_date = settings["start_time"]
        monday_this_week = prev_monday(force_prev=False)
        
        # Set total to max_items for progress bar
        progress_state['total'] = max_items
        
        # Counter for processed items
        items_processed = 0
        
        # Process all projects
        for p, p_dict in settings["projects"]:
            if items_processed >= max_items:
                _logger.info(f"Reached max items limit ({max_items}), stopping")
                break
                
            try:
                repo = gh.get_repo(p)
                progress_state['current_repo'] = repo.name
                _logger.info(f"Processing {repo.name}")
                
                ignored_pr = p_dict.get("ignored_pr", [])
                
                # Process issues with ETA label (sorted by created, like prtime.py)
                try:
                    issues = repo.get_issues(state='all', sort='created',
                                            direction="desc", labels=["ETA"])
                    
                    for issue in issues:
                        if items_processed >= max_items:
                            _logger.info(f"Reached max items limit ({max_items}) in issues")
                            break
                            
                        if issue.created_at < start_date:
                            break
                        
                        # Check if has ETA table in body (like prtime.py)
                        if not (issue.body and '|' in issue.body and 'ETA' in issue.body):
                            continue
                        
                        progress_state['processed'] += 1
                        progress_state['current_item'] = f"{repo.name}#{issue.number}"
                        
                        process_issue_or_pr(repo.name, issue, settings, monday_this_week)
                        items_processed += 1
                        
                except Exception as e:
                    error_msg = f"Error processing issues for {repo.name}: {str(e)}"
                    _logger.exception(error_msg)
                    progress_state['errors'].append(error_msg)
                
                # Process PRs (sorted by created, like prtime.py)
                if items_processed < max_items:
                    try:
                        pulls = repo.get_pulls(state='all', sort='created',
                                              direction="desc", base=p_dict.get("pr_base", "master"))
                        
                        for pr in pulls:
                            if items_processed >= max_items:
                                _logger.info(f"Reached max items limit ({max_items}) in PRs")
                                break
                                
                            if pr.number in ignored_pr:
                                progress_state['processed'] += 1
                                continue
                            
                            if pr.created_at < start_date:
                                break
                            
                            # Check if has ETA table in body (like prtime.py)
                            if not (pr.body and '|' in pr.body and 'ETA' in pr.body):
                                continue
                            
                            progress_state['processed'] += 1
                            progress_state['current_item'] = f"{repo.name}#{pr.number}"
                            
                            process_issue_or_pr(repo.name, pr, settings, monday_this_week)
                            items_processed += 1
                            
                    except Exception as e:
                        error_msg = f"Error processing PRs for {repo.name}: {str(e)}"
                        _logger.exception(error_msg)
                        progress_state['errors'].append(error_msg)
                        
            except Exception as e:
                error_msg = f"Error accessing repo {p}: {str(e)}"
                _logger.exception(error_msg)
                progress_state['errors'].append(error_msg)
        
        # Calculate summary
        calculate_summary()
        
        progress_state['status'] = 'completed'
        progress_state['end_time'] = datetime.now().isoformat()
        _logger.info(f"Analysis completed. Processed {items_processed} items.")
        
    except Exception as e:
        error_msg = f"Fatal error during analysis: {str(e)}"
        _logger.exception(error_msg)
        progress_state['errors'].append(error_msg)
        progress_state['status'] = 'error'
        progress_state['end_time'] = datetime.now().isoformat()


def process_issue_or_pr(repo_name, item, settings, monday_this_week):
    """Process a single issue or PR"""
    global progress_state
    
    pr_id = get_pr_id(repo_name, item)
    
    # Check if has ETA table in body
    if not item.body or "|" not in item.body or "ETA" not in item.body:
        progress_state['results']['missing_eta_tables'].append({
            'repo': repo_name,
            'number': item.number,
            'title': item.title,
            'url': item.html_url,
            'state': item.state,
            'type': 'issue' if is_issue(item) else 'pr',
            'created_at': item.created_at.isoformat(),
            'updated_at': item.updated_at.isoformat()
        })
        return
    
    # Try to parse ETA
    try:
        eta = parse_eta(item, pr_id)
    except Exception as e:
        _logger.warning(f"Exception parsing ETA for {pr_id}: {e}")
        eta = None
    
    if eta is None:
        progress_state['warnings'].append(f"Cannot parse ETA for {pr_id}")
        progress_state['results']['missing_eta_tables'].append({
            'repo': repo_name,
            'number': item.number,
            'title': item.title,
            'url': item.html_url,
            'state': item.state,
            'type': 'issue' if is_issue(item) else 'pr',
            'created_at': item.created_at.isoformat(),
            'updated_at': item.updated_at.isoformat(),
            'parse_error': True
        })
        return
    
    # Validate ETA
    try:
        valid, errors = eta.validate_hours(item)
    except Exception as e:
        _logger.error(f"Error validating {pr_id}: {e}")
        valid = False
        errors = [f"validation_exception: {str(e)}"]
    
    # Check if updated this week using simple updated_at comparison
    # This is more intuitive than checking timeline events
    try:
        item_updated_date = item.updated_at.date()
        updated = item_updated_date >= monday_this_week
        update_week = item.updated_at.isocalendar()[1]
    except Exception as e:
        _logger.warning(f"Error checking updates for {pr_id}: {e}")
        updated = False
        update_week = -1
    
    # Build result object
    result = {
        'repo': repo_name,
        'number': item.number,
        'title': item.title,
        'url': item.html_url,
        'state': item.state,
        'type': 'issue' if is_issue(item) else 'pr',
        'created_at': item.created_at.isoformat(),
        'updated_at': item.updated_at.isoformat(),
        'closed_at': item.closed_at.isoformat() if item.closed_at else None,
        'eta_valid': valid,
        'eta_errors': errors if not valid else [],
        'updated_this_week': updated,
        'eta_data': {
            'estimate': eta.est,
            'customer_estimate': eta.cust_est,
            'total_reported': eta.total_reported,
            'dev_hours_total': eta.dev_hours_total(),
            'stage_totals': dict(eta.stage_totals),
            'dev_hours': {dev: {stage: hours for stage, hours in stages.items()} 
                         for dev, stages in eta.dev_hours.items()}
        }
    }
    
    # Categorize
    if item.state == 'open':
        progress_state['results']['open_issues'].append(result)
    else:
        progress_state['results']['closed_issues'].append(result)
    
    if not valid:
        progress_state['results']['validation_errors'].append(result)
    
    if updated:
        progress_state['results']['this_week_updates'].append(result)


def calculate_summary():
    """Calculate summary statistics"""
    results = progress_state['results']
    
    summary = {
        'total_items': (len(results['open_issues']) + 
                       len(results['closed_issues']) + 
                       len(results['missing_eta_tables'])),
        'open_count': len(results['open_issues']),
        'closed_count': len(results['closed_issues']),
        'validation_errors_count': len(results['validation_errors']),
        'this_week_updates_count': len(results['this_week_updates']),
        'missing_eta_count': len(results['missing_eta_tables']),
        'total_hours': {
            'estimate': 0,
            'customer_estimate': 0,
            'reported': 0,
            'by_dev': {}
        }
    }
    
    # Calculate totals
    all_with_eta = results['open_issues'] + results['closed_issues']
    for item in all_with_eta:
        eta_data = item['eta_data']
        summary['total_hours']['estimate'] += eta_data['estimate']
        summary['total_hours']['customer_estimate'] += eta_data['customer_estimate']
        summary['total_hours']['reported'] += eta_data['total_reported']
        
        for dev, stages in eta_data['dev_hours'].items():
            if dev not in summary['total_hours']['by_dev']:
                summary['total_hours']['by_dev'][dev] = 0
            summary['total_hours']['by_dev'][dev] += sum(stages.values())
    
    progress_state['results']['summary'] = summary


# Flask routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/start', methods=['POST'])
def start_analysis():
    """Start background analysis"""
    global progress_state
    
    if progress_state['status'] == 'running':
        return jsonify({'error': 'Analysis already running'}), 400
    
    data = request.get_json() or {}
    settings_file = data.get('settings_file', '../settings.json')
    max_items = data.get('max_items', 400)  # Default to 400 to catch more items
    
    reset_progress()
    
    # Start background thread
    thread = threading.Thread(target=analyze_repository, args=(settings_file, max_items))
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Analysis started', 'max_items': max_items})


@app.route('/api/progress')
def get_progress():
    """Get current progress"""
    return jsonify(progress_state)


@app.route('/api/results')
def get_results():
    """Get final results"""
    return jsonify(progress_state['results'])


@app.route('/api/validate', methods=['POST'])
def validate_specific():
    """Validate specific issues"""
    data = request.get_json()
    issue_numbers = data.get('issues', [])
    
    # TODO: Implement specific validation
    return jsonify({'message': 'Validation started', 'issues': issue_numbers})


if __name__ == '__main__':
    # Check for .env
    env_file = os.path.join(_this_dir, ".env")
    if os.path.exists(env_file):
        with open(env_file, mode="r") as fin:
            lines = [x.strip().split("=", maxsplit=1)
                    for x in fin.readlines() if 3 < len(x.strip())]
            for k, v in lines:
                if k not in os.environ:
                    os.environ[k] = v
    
    print("\n" + "="*60)
    print(">>> PRTime Web Dashboard Starting")
    print("="*60)
    print(f"Open: http://localhost:5000")
    print(f"Directory: {_this_dir}")
    print(f"Default limit: 20 items (adjustable in UI)")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
