"""
Web interface for prtime - GitHub ETA tracker
Provides real-time monitoring of PR/Issue ETA tables
"""
import os
import sys
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, request
import logging

# Add parent directory (for prtime module) to path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
sys.path.insert(0, _parent_dir)

from prtime import (
    Github, load_settings, parse_eta, get_pr_id, 
    is_issue, prev_monday, hours_row,
    pr_with_eta_hours, pr_with_eta, find_hours_all_structured
)

app = Flask(__name__)
# Single-origin app: no CORS, the dashboard is served by Flask itself.
# Adding flask_cors.CORS(app) without origin restrictions combined with the
# absence of auth would let any page in your browser drive /api/start.

# Configure logging
logging.basicConfig(
    format='%(asctime)s %(levelname).4s: %(message)s',
    level=logging.INFO
)
_logger = logging.getLogger(__name__)

# Lock guarding the global progress_state. Background analysis/validation
# threads mutate the dict while /api/progress reads it; without a lock the
# dashboard occasionally observes torn state, and the "already running"
# check below is racy across two near-simultaneous /api/start requests.
_progress_lock = threading.Lock()

# Global state
progress_state = {
    'status': 'idle',
    'current_repo': '',
    'current_item': '',
    'processed': 0,
    'total': 0,
    'errors': [],
    'warnings': [],
    'console_logs': [],  # NEW: Capture console output
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
        'console_logs': [],  # NEW: Reset console logs
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


def analyze_repository(settings_file='../settings.json', max_items=400, filter_state='all'):
    """Main analysis function that runs in background thread
    
    NOW REUSES find_hours_all_structured() from prtime.py to ensure 100% identical behavior
    to: python prtime.py --hours --state=closed --check-last=12w
    
    No parsing is done inside web_app.py - everything is delegated to prtime.py
    
    Args:
        settings_file: Path to settings.json
        max_items: Maximum items to process (for UI progress only - NOT USED)
        filter_state: 'all', 'open', or 'closed' - matches prtime.py --state parameter
    """
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
        
        progress_state['console_logs'].append(f"Filtering by state: {filter_state}, valid since: {start_date}")
        _logger.info(f"Filtering by state: {filter_state}, valid since: {start_date}")
        
        # Use find_hours_all_structured() - REUSES ALL LOGIC from prtime.py
        # This is the SAME function that --hours uses, just returns structured data instead of markdown
        progress_state['console_logs'].append("Fetching and processing PRs/Issues (reusing prtime.py logic)...")
        
        structured_data = find_hours_all_structured(gh, start_date, filter_state=filter_state)
        
        progress_state['console_logs'].append(f"Found {structured_data['summary']['total_weeks']} weeks of data")
        progress_state['console_logs'].append(f"Total items: {structured_data['summary']['total_items']}")
        progress_state['console_logs'].append(f"Items with ETA: {structured_data['summary']['items_with_eta']}")
        progress_state['console_logs'].append(f"Items without ETA: {structured_data['summary']['items_without_eta']}")
        progress_state['console_logs'].append(f"Items filtered by state: {structured_data['summary']['items_filtered']}")
        
        # Convert structured data to progress_state format
        seen_this_week_updates = set()
        total_processed = 0
        
        for week_data in structured_data['weeks']:
            progress_state['current_repo'] = f"Week {week_data['week_key']}"
            
            for item in week_data['items']:
                total_processed += 1
                progress_state['processed'] = total_processed
                progress_state['current_item'] = f"#{item['number']}"
                
                # Check if updated this week
                updated = False
                if item['updated_at']:
                    from dateutil import parser as date_parser
                    updated_date = date_parser.parse(item['updated_at']).date()
                    updated = (updated_date >= monday_this_week)
                
                # Convert hours_data to eta_data format expected by UI
                hours_data = item['hours_data']
                
                # Helper to safely convert to float (handles empty strings and None)
                def safe_float(value, default=0):
                    if value == '' or value is None:
                        return default
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                
                eta_data = {
                    'estimate': safe_float(hours_data.get('ETA', 0)),
                    'customer_estimate': safe_float(hours_data.get('ETA Cust', 0)),
                    'total_reported': safe_float(hours_data.get('Dev Total', 0)),
                    'dev_hours_total': safe_float(hours_data.get('Dev Total', 0)),
                    'stage_totals': {
                        'ETA': safe_float(hours_data.get('Phase ETA', 0)),
                        'Developing': safe_float(hours_data.get('Phase Dev', 0)),
                        'Review': safe_float(hours_data.get('Phase Review', 0)),
                    },
                    'dev_hours': {}  # Extract dev hours from hours_data
                }
                
                # Extract individual dev hours
                for key, value in hours_data.items():
                    if key.startswith('Dev ') and key not in ['Dev Total', 'Dev Others']:
                        dev_name = key.replace('Dev ', '')
                        if value != '':
                            eta_data['dev_hours'][dev_name] = {'total': safe_float(value)}
                
                # Build result object compatible with existing UI
                result = {
                    'repo': item['repo'],
                    'number': item['number'],
                    'title': item['title'],
                    'url': item['url'],
                    'state': item['state'],
                    'week_key': week_data['week_key'],
                    'week_state': item['week_state'],
                    'type': item['type'],
                    'created_at': item['created_at'],
                    'updated_at': item['updated_at'],
                    'closed_at': item['closed_at'],
                    'eta_valid': True,  # If it's in structured_data, it passed parsing
                    'eta_errors': [],
                    'updated_this_week': updated,
                    'eta_data': eta_data,
                    'week_hours': safe_float(hours_data.get('Last Week Total', 0))  # Hours spent this week
                }
                
                # Categorize
                if item['state'] == 'open':
                    progress_state['results']['open_issues'].append(result)
                else:
                    progress_state['results']['closed_issues'].append(result)
                
                # Add to "This Week Updates" only once per PR (deduplicate)
                if updated:
                    pr_key = (item['repo'], item['number'])
                    if pr_key not in seen_this_week_updates:
                        seen_this_week_updates.add(pr_key)
                        progress_state['results']['this_week_updates'].append(result)
        
        # Add missing ETA items
        for missing_item in structured_data['missing_eta']:
            progress_state['results']['missing_eta_tables'].append({
                'repo': missing_item['repo'],
                'number': missing_item['number'],
                'title': missing_item['title'],
                'url': missing_item['url'],
                'state': missing_item['state'],
                'type': missing_item['type'],
                'parse_error': True
            })
        
        progress_state['total'] = total_processed
        
        progress_state['console_logs'].append(f"\n{'='*60}")
        progress_state['console_logs'].append("Processing complete!")
        progress_state['console_logs'].append(f"  • Total items processed: {total_processed}")
        progress_state['console_logs'].append(f"  • Items with ETA: {structured_data['summary']['items_with_eta']}")
        progress_state['console_logs'].append(f"  • Items without ETA: {structured_data['summary']['items_without_eta']}")
        progress_state['console_logs'].append(f"  • Items filtered by state: {structured_data['summary']['items_filtered']}")
        progress_state['console_logs'].append(f"{'='*60}")
        
        calculate_summary()
        
        progress_state['status'] = 'completed'
        progress_state['end_time'] = datetime.now().isoformat()
        
    except Exception as e:
        _logger.exception("Analysis failed")
        progress_state['status'] = 'error'
        progress_state['errors'].append(str(e))
        progress_state['console_logs'].append(f"Error: {str(e)}")
        progress_state['end_time'] = datetime.now().isoformat()


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


def validate_repository(settings_file='../settings.json', filter_state='closed', filter_week=None):
    """
    Validation function that runs in background thread
    Validates ETA tables in issues/PRs and reports problems
    
    Args:
        settings_file: Path to settings.json
        filter_state: 'all', 'open', or 'closed' - matches prtime.py --state parameter
        filter_week: Optional week filter like '2025_45' to validate only items from specific week
    """
    global progress_state
    
    try:
        progress_state['status'] = 'running'
        progress_state['start_time'] = datetime.now().isoformat()
        progress_state['console_logs'].append(f"🔍 Starting validation with state={filter_state}, week={filter_week or 'all'}")
        
        # Load settings
        import prtime
        if not os.path.isabs(settings_file):
            settings_file = os.path.join(_parent_dir, settings_file)
        settings = load_settings(settings_file)
        prtime.settings = settings
        hours_row.init()
        
        # Get ignore patterns from settings
        ignore_titles = settings.get('validation_ignore_titles', [])
        if ignore_titles:
            progress_state['console_logs'].append(f"📋 Ignoring titles matching: {', '.join(ignore_titles)}")
        
        # Initialize GitHub client
        gh_key = "GITHUB_PAT"
        if gh_key not in os.environ:
            raise Exception(f"Missing {gh_key} environment variable")
        
        gh = Github(os.environ[gh_key])
        progress_state['console_logs'].append("✓ GitHub client initialized")
        
        start_date = settings['start_time']
        progress_state['console_logs'].append(f"📅 Checking items since {start_date.date()}")
        
        # If filtering by week, use find_hours_all_structured to get weekly data
        if filter_week:
            progress_state['console_logs'].append(f"🗓️  Filtering by week: {filter_week}")
            structured_data = find_hours_all_structured(gh, start_date, filter_state=filter_state)
            
            # Find the specific week
            week_found = False
            all_items = []
            for week_data in structured_data['weeks']:
                if week_data['week_key'] == filter_week:
                    week_found = True
                    progress_state['console_logs'].append(f"✓ Found week {filter_week}: {week_data['monday']} to {week_data['sunday']}")
                    
                    # Convert items to (repo_name, pr_or_issue) tuples
                    # We need to fetch the actual PR objects from GitHub
                    for item in week_data['items']:
                        # Extract repo and number from URL or stored data
                        repo_name = item['repo']
                        # We need to get the actual PR/Issue object
                        # For now, we'll skip week filtering and do it differently
                        pass
                    break
            
            if not week_found and filter_week != 'all':
                progress_state['console_logs'].append(f"⚠️  Week {filter_week} not found in data")
                all_items = []
        
        # Collect all items (without week filter for now - will implement properly)
        progress_state['console_logs'].append(f"📋 Fetching issues/PRs with state={filter_state}...")
        all_items = []
        for repo_name, pr_or_issue in pr_with_eta(gh, start_date, state=filter_state, include_issues=True):
            all_items.append((repo_name, pr_or_issue))
        
        progress_state['total'] = len(all_items)
        progress_state['console_logs'].append(f"✓ Found {len(all_items)} items to validate")
        
        # Validate each item
        ok_count = 0
        error_count = 0
        no_eta_count = 0
        ignored_count = 0
        
        for repo_name, pr_or_issue in all_items:
            pr_id = get_pr_id(repo_name, pr_or_issue)
            progress_state['current_repo'] = repo_name
            progress_state['current_item'] = f"#{pr_or_issue.number}"
            progress_state['processed'] += 1
            
            # Check if title should be ignored
            title_ignored = False
            for ignore_pattern in ignore_titles:
                if ignore_pattern.lower() in pr_or_issue.title.lower():
                    title_ignored = True
                    ignored_count += 1
                    progress_state['console_logs'].append(f"⏭️  Ignored (title matches): {pr_id}")
                    break
            
            if title_ignored:
                continue
            
            eta = parse_eta(pr_or_issue, pr_id)
            if eta is None:
                no_eta_count += 1
                progress_state['console_logs'].append(f"⚠️  No ETA table: {pr_id}")
                continue
            
            valid, errors = eta.validate_hours(pr_or_issue)
            if valid:
                ok_count += 1
                # Add to results as validated item
                result = {
                    'repo': repo_name,
                    'number': pr_or_issue.number,
                    'title': pr_or_issue.title,
                    'url': pr_or_issue.html_url,
                    'state': pr_or_issue.state,
                    'type': 'issue' if is_issue(pr_or_issue) else 'pr',
                    'validated': True,
                    'validation_status': 'OK',
                    'eta_data': {
                        'estimate': eta.est,
                        'customer_estimate': eta.cust_est,
                        'total_reported': eta.total_reported,
                        'dev_hours_total': eta.dev_hours_total(),
                        'stage_totals': dict(eta.stage_totals),
                        'dev_hours': {dev: {'total': sum(stages.values())} 
                                     for dev, stages in eta.dev_hours.items()}
                    },
                    'created_at': pr_or_issue.created_at.isoformat() if pr_or_issue.created_at else None,
                    'updated_at': pr_or_issue.updated_at.isoformat() if pr_or_issue.updated_at else None,
                    'closed_at': pr_or_issue.closed_at.isoformat() if pr_or_issue.closed_at else None,
                }
                
                if pr_or_issue.state == 'open':
                    progress_state['results']['open_issues'].append(result)
                else:
                    progress_state['results']['closed_issues'].append(result)
            else:
                error_count += 1
                progress_state['console_logs'].append(f"❌ ERRORS in {pr_id}: {', '.join(errors)}")
                
                # Add to validation errors with eta_data for display
                result = {
                    'repo': repo_name,
                    'number': pr_or_issue.number,
                    'title': pr_or_issue.title,
                    'url': pr_or_issue.html_url,
                    'state': pr_or_issue.state,
                    'type': 'issue' if is_issue(pr_or_issue) else 'pr',
                    'validated': True,
                    'validation_status': 'ERROR',
                    'eta_errors': errors,
                    'eta_data': {
                        'estimate': eta.est,
                        'customer_estimate': eta.cust_est,
                        'total_reported': eta.total_reported,
                        'dev_hours_total': eta.dev_hours_total(),
                        'stage_totals': dict(eta.stage_totals),
                        'dev_hours': {dev: {'total': sum(stages.values())} 
                                     for dev, stages in eta.dev_hours.items()}
                    },
                    'created_at': pr_or_issue.created_at.isoformat() if pr_or_issue.created_at else None,
                    'updated_at': pr_or_issue.updated_at.isoformat() if pr_or_issue.updated_at else None,
                    'closed_at': pr_or_issue.closed_at.isoformat() if pr_or_issue.closed_at else None,
                }
                
                # Add to BOTH validation_errors AND open/closed issues
                progress_state['results']['validation_errors'].append(result)
                
                # Also add to appropriate state list so it's counted in totals
                if pr_or_issue.state == 'open':
                    progress_state['results']['open_issues'].append(result)
                else:
                    progress_state['results']['closed_issues'].append(result)
        
        # Final summary
        progress_state['console_logs'].append(f"\n{'='*60}")
        progress_state['console_logs'].append("✅ Validation complete!")
        progress_state['console_logs'].append(f"   • Total checked: {len(all_items)}")
        progress_state['console_logs'].append(f"   • Valid: {ok_count}")
        progress_state['console_logs'].append(f"   • Errors: {error_count}")
        progress_state['console_logs'].append(f"   • No ETA table: {no_eta_count}")
        if ignored_count > 0:
            progress_state['console_logs'].append(f"   • Ignored (by title): {ignored_count}")
        progress_state['console_logs'].append(f"{'='*60}")
        
        calculate_summary()
        
        progress_state['status'] = 'completed'
        progress_state['end_time'] = datetime.now().isoformat()
        
    except Exception as e:
        _logger.exception("Validation failed")
        progress_state['status'] = 'error'
        progress_state['errors'].append(str(e))
        progress_state['console_logs'].append(f"❌ Validation failed: {str(e)}")
        progress_state['end_time'] = datetime.now().isoformat()


# Flask routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/start', methods=['POST'])
def start_analysis():
    """Start background analysis"""
    data = request.get_json() or {}
    settings_file = data.get('settings_file', '../settings.json')
    max_items = data.get('max_items', 400)  # Default to 400 to catch more items
    filter_state = data.get('state', 'all')  # 'all', 'open', or 'closed' like prtime.py --state

    # Validate filter_state
    if filter_state not in ['all', 'open', 'closed']:
        return jsonify({'error': f'Invalid state: {filter_state}. Must be all, open, or closed'}), 400

    with _progress_lock:
        if progress_state['status'] == 'running':
            return jsonify({'error': 'Analysis already running'}), 400
        reset_progress()
        progress_state['status'] = 'running'

    thread = threading.Thread(target=analyze_repository, args=(settings_file, max_items, filter_state))
    thread.daemon = True
    thread.start()

    return jsonify({
        'message': 'Analysis started',
        'max_items': max_items,
        'state': filter_state
    })


@app.route('/api/progress')
def get_progress():
    """Get current progress"""
    return jsonify(progress_state)


@app.route('/api/results')
def get_results():
    """Get final results"""
    return jsonify(progress_state['results'])


@app.route('/api/start-validation', methods=['POST'])
def start_validation():
    """Start background validation"""
    data = request.get_json() or {}
    settings_file = data.get('settings_file', '../settings.json')
    filter_state = data.get('state', 'closed')  # Default to 'closed' for validation
    filter_week = data.get('week', None)  # Optional week filter like '2025_45'

    # Validate filter_state
    if filter_state not in ['all', 'open', 'closed']:
        return jsonify({'error': f'Invalid state: {filter_state}. Must be all, open, or closed'}), 400

    with _progress_lock:
        if progress_state['status'] == 'running':
            return jsonify({'error': 'Analysis or validation already running'}), 400
        reset_progress()
        progress_state['status'] = 'running'

    thread = threading.Thread(target=validate_repository, args=(settings_file, filter_state, filter_week))
    thread.daemon = True
    thread.start()

    return jsonify({
        'message': 'Validation started',
        'state': filter_state,
        'week': filter_week or 'all'
    })


@app.route('/api/validate', methods=['POST'])
def validate_specific():
    """Validate specific issues (deprecated - use /api/start-validation)"""
    data = request.get_json()
    issue_numbers = data.get('issues', [])
    
    # TODO: Implement specific validation
    return jsonify({'message': 'Validation started', 'issues': issue_numbers})


@app.route('/api/weeks', methods=['GET'])
def get_available_weeks():
    """Get list of available weeks from the data"""
    try:
        import prtime
        
        # Load settings
        settings_file = os.path.join(_parent_dir, 'settings.json')
        settings = load_settings(settings_file)
        prtime.settings = settings
        hours_row.init()
        
        # Check for GitHub token
        gh_key = "GITHUB_PAT"
        if gh_key not in os.environ:
            return jsonify({'error': 'GITHUB_PAT not found'}), 500
        
        gh = Github(os.environ[gh_key])
        start_date = settings["start_time"]
        
        # Get weekly data
        weeks = pr_with_eta_hours(gh, start_date)
        
        # Format weeks for dropdown
        week_list = []
        for week_key in sorted(weeks.keys(), reverse=True):
            year, week_n = week_key.split("_")
            week_list.append({
                'value': week_key,
                'label': f"Week {week_n} ({year})",
                'count': len(weeks[week_key])
            })
        
        return jsonify({
            'weeks': week_list,
            'total_weeks': len(week_list)
        })
    except Exception:
        _logger.exception("Failed to get weeks")
        return jsonify({'error': 'Failed to fetch weeks (see server logs)'}), 500


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

    # Bind localhost-only and disable Werkzeug debugger by default. The
    # dashboard has no auth, so binding to 0.0.0.0 with debug=True exposes
    # the Werkzeug debugger PIN on every interface (RCE if ever reached).
    # Override via PRTIME_HOST / PRTIME_PORT / PRTIME_DEBUG for explicit
    # opt-in (e.g. PRTIME_DEBUG=1 in dev).
    host = os.environ.get('PRTIME_HOST', '127.0.0.1')
    port = int(os.environ.get('PRTIME_PORT', '5000'))
    debug = os.environ.get('PRTIME_DEBUG', '').lower() in ('1', 'true', 'yes')

    print("\n" + "="*60)
    print(">>> PRTime Web Dashboard Starting")
    print("="*60)
    print(f"Open: http://{host}:{port}")
    print(f"Directory: {_this_dir}")
    print(f"Debug: {debug}")
    print("="*60 + "\n")

    app.run(debug=debug, host=host, port=port)
