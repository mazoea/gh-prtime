"""
ETA parser for github issues/PRs.

| Phases            | JH  |  JP  | TM |   JM | Total  |
|-----------------|----:|----:|-----:|-----:|-------:|
| ETA                  |  2  |   0.5 |     0 |      0 |        0 |
| Developing      |  2+  |    0 |    0 |      0 |         0 |
| Review             |  0  |    0 |    0 |      0 |         0 |
| Total                |   -  |   -   |  -    |   -    |         0 |
| ETA est.             |      |       |       |         |     32  |
| ETA cust.           |   -  |   -  |   -   |   -     |        24 |
"""
import os
import sys
import copy
import logging
import argparse
import re
import glob
import json
import typing
import shutil

import tqdm
from pprint import pformat
from collections import defaultdict, OrderedDict
from github import Github

from datetime import datetime, timedelta, date

_ts = datetime.now().strftime("%Y_%m_%d__%H.%M.%S")
_logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(levelname).4s: %(message)s', level=logging.INFO)
_this_dir = os.path.dirname(os.path.abspath(__file__))

# read .env
env_file = os.path.join(_this_dir, ".env")
if os.path.exists(env_file):
    with open(env_file, mode="r") as fin:
        lines = [x.strip().split("=", maxsplit=1)
                 for x in fin.readlines() if 3 < len(x.strip())]
        for k, v in lines:
            if k in os.environ:
                continue
            os.environ[k] = v


# =================


def load_settings(file_str: str):
    if not os.path.exists(file_str):
        base_f = os.path.basename(file_str)
        for f in glob.glob("./___*.json"):
            if base_f in os.path.basename(f):
                file_str = f
                break

    with open(file_str, "r") as fin:
        cfg = json.load(fin)
    cfg["start_time"] = datetime.strptime(cfg["start_time"], r"%Y-%m-%d")
    # abs paths
    for k, v in settings.items():
        if isinstance(v, str) and v.startswith("./"):
            settings[k] = os.path.join(_this_dir, v[2:])

    cfg["TRACKER_LINK"] = os.environ.get('TRACKER_LINK', 'XXXlink')
    cfg["CUSTOMER"] = os.environ.get('CUSTOMER', 'XXXcustomer')

    return cfg


# init in `load_settings`
settings = {}


# =================

def prev_monday(force_prev=True) -> date:
    """
        `force_prev` - if today is monday, return last monday
    """
    today = datetime.today()
    if force_prev:
        weeks = 0 if today.weekday() != 0 else -1
    else:
        weeks = 0
    monday = today + timedelta(days=-today.weekday(), weeks=weeks)
    return monday.date()


# =================


class ETA:
    stages = ("ETA", "Developing", "Review")
    key_phase = "phases"
    key_eta = "eta"
    key_eta_cust = "ETA cust"
    key_total = "Total"


# =================

def is_issue(pr):
    """ Return True if `pr` is an issue (and not a PullRequest). """
    from github import Issue
    return isinstance(pr, Issue.Issue)


def log_err(msg, pr, pr_id):
    tmpl = "\n" + ("<" * 10) + "\n%s [%s]\n\t->%s\n" + (">" * 10)
    _logger.info(tmpl, msg, pr_id, pr.html_url)


def sum_hours(s, pr_id, pr_html=None):
    """
        Try to sum the cell.
    """
    try:
        return float(eval(s))
    except Exception:
        _logger.info(f"Cannot parse [{s}] in [{pr_id}] [{pr_html or ''}]")
    return -1.


def get_pr_id(repo_name, pr):
    return "%s:%s:%s" % (repo_name, pr.number, pr.title)


class hours_row(object):
    h_week = "#Week"
    h_customer = "Cust"
    h_issue = "Issue"
    h_link_gh = "Link GH"
    h_link_jira = "Link JIRA"
    h_state = "State"
    h_tracked = "Tracked"
    h_eta = "ETA"
    h_eta_cust = "ETA Cust"
    h_phase_eta = "Phase ETA"
    h_phase_dev = "Phase Dev"
    h_phase_review = "Phase Review"
    h_phase_total = "Phase Total"
    h_dev_total = "Dev Total"
    h_dev_others = "Dev Others"
    h_closed = "Closed"
    h_created = "Created"
    h_days_open = "Days Opened"
    h_last_week = "Last Week Total"
    h_exp_dev_k = "Dev %s"
    # TODO(jm): filled based on settings in `init`
    # order headers
    h_spec = None
    # header string with separators
    header = None
    header_sep = None

    @staticmethod
    def init():
        h_devs_arr = [hours_row.h_exp_dev_k % x for x in settings["devs"]]
        hours_row.h_spec = [
            hours_row.h_week, hours_row.h_customer, hours_row.h_issue,
            hours_row.h_link_gh, hours_row.h_link_jira,
            hours_row.h_state, hours_row.h_tracked,
            hours_row.h_eta, hours_row.h_eta_cust, hours_row.h_phase_eta, hours_row.h_phase_dev, hours_row.h_phase_review, hours_row.h_phase_total, hours_row.h_dev_total
        ] + h_devs_arr + [
            hours_row.h_dev_others,
            hours_row.h_closed, hours_row.h_created, hours_row.h_days_open, hours_row.h_last_week
        ]
        hours_row.header = "|" + \
            ("|".join([f" {x:10s} " for x in hours_row.h_spec]))[1:] + " |"
        header_len = hours_row.header.count("|") - 1
        hours_row.header_sep = header_len * ("| %10s " % "----:") + " |"

    def __init__(self):
        self._d = OrderedDict()
        for k in hours_row.h_spec:
            self._d[k] = ""

    def __getitem__(self, k):
        return self._d[k]

    def __setitem__(self, k, v):
        self._d[k] = v

    def dev(self, key, value):
        dev_k = hours_row.h_exp_dev_k % key
        for k in self._d.keys():
            if k == dev_k:
                self._d[k] = value
                return
        if self._d[self.h_dev_others] == "":
            self._d[self.h_dev_others] = value
        else:
            self._d[self.h_dev_others] += value

    def __str__(self):
        s = ""
        for k, v in self._d.items():
            if isinstance(v, float):
                v = f"{v}"
            s += f"| {v} "
        return s + "|"


class eta_row:
    """
        Row of an ETA
    """

    def __init__(self, arr, pr_id, pr_html=None):
        self.arr = arr
        self.pr_id = pr_id
        self.pr_html = pr_html

    @property
    def name(self): return self.arr[1]

    @property
    def total(self): return self.arr[-2]

    @property
    def dev_hours(self): return [sum_hours(
        x, self.pr_id, self.pr_html) for x in self.arr[2:-2]]

    @property
    def dev_names(self): return self.arr[2:-2]


class eta_table:
    label_prefix = "ETA_err_"

    def __init__(self, pr, pr_id, cols):
        self.pr = pr
        self.pr_id = pr_id
        self.rows = []
        self._valid = False
        try:
            if len(cols) > 0:
                self.rows = [eta_row(x, pr_id, pr.html_url) for x in cols]
                self._valid = self._validate_keys()
        except Exception as e:
            _logger.exception(
                f"Could not validate ETA table for [{pr.html_url}] [{self.rows}]")
        if not self._valid:
            return
        self._d = self._parse()
        self._valid = self._d is not None

        # validate
        if self._valid:
            stage_totals = copy.deepcopy(self.stage_totals)
            stage_totals_1 = self.compute_stage_totals()
            if stage_totals_1 != stage_totals:
                _logger.warning(f"Computed stage totals do not match in [{self.pr_id}]!")

    def copy(self):
        cp = eta_table(self.pr, self.pr_id, [])
        cp.rows = self.rows
        cp._valid = self._valid
        cp._d = self._d
        return cp

    @property
    def d(self):
        return self._d

    @property
    def valid(self):
        return self._valid

    @property
    def total_reported(self):
        return self._d["total_reported"]

    @property
    def cust_est(self):
        return self._d["customer_estimate"]

    @property
    def est(self):
        return self._d["estimate"]

    @property
    def dev_hours(self):
        return self._d["dev_hours"]

    @dev_hours.setter
    def dev_hours(self, d):
        """
            Overwrite dev hours, used when relative ETA is needed.
        """
        self._d["dev_hours"] = d

    @property
    def stage_totals(self):
        return self._d["stage_totals"]

    def dev_hours_total(self):
        t = 0
        for dev, hours in self.dev_hours.items():
            t += sum(hours.values())
        return t

    def _parse(self):
        d = {}
        try:
            d["customer_estimate"] = float(self.rows[-1].total)
        except Exception:
            return None
        try:
            d["estimate"] = float(self.rows[-2].total)
        except Exception:
            return None
        try:
            d["total_reported"] = float(self.rows[-3].total)
        except Exception:
            return None

        dev_hours = {k: {} for k in self.rows[0].dev_names}
        stage_totals = {}
        for stage in ETA.stages:
            found = False
            for row in self.rows:
                if row.name.lower() != stage.lower():
                    continue
                for i, h in enumerate(row.dev_hours):
                    dev_hours[self.rows[0].dev_names[i]][stage] = h
                try:
                    stage_totals[stage] = float(row.total)
                except Exception as e:
                    _logger.critical(
                        f"Cannot parse stage total [{row.total}] [{stage}] [{self.pr_id}]\n{self.pr}")
                    raise
                found = True
                break
            if not found:
                _logger.critical("Did not find stage [%s] in [%s]", stage, self.pr_id)
                return None

        for k, v in dev_hours.items():
            for stage, hours in v.items():
                if hours == -1:
                    _logger.debug("-1 in hours [%s] [%s] [%s]", k, stage, self.pr_id)

        d["dev_hours"] = dev_hours
        d["stage_totals"] = stage_totals
        return d

    def compute_stage_totals(self):
        stage_totals = defaultdict(float)
        for stage in ETA.stages:
            for k, v in self.dev_hours.items():
                stage_totals[stage] += v[stage]
        self._d["stage_totals"] = stage_totals
        self._d["total_reported"] = sum(stage_totals.values())
        return stage_totals

    def _validate_keys(self):
        ver_1 = self.rows[0].name.lower() == ETA.key_phase
        ver_2 = ETA.key_eta in self.rows[-1].name.lower()
        ver_3 = True
        try:
            float(self.rows[2].total)
        except Exception:
            ver_3 = False
        if not ver_1 or not ver_2 or not ver_3:
            _logger.critical(
                "Cannot parse, verification failed [%s]\n\t->[%s]", self.pr_id, self.pr.html_url)
            return False

        def has_name(row, name):
            return name in row.name.lower()

        # verification #2
        if has_name(self.rows[-1], ETA.key_eta_cust):
            _logger.critical("Cannot find ETA cust [%s]", self.pr_id)
            return False

        if has_name(self.rows[-3], ETA.key_total):
            _logger.critical("Cannot find Total [%s]", self.pr_id)
            return False

        return True

    def validate_hours(self, pr=None):
        errors = []
        valid = True
        html_url = pr.html_url if pr is not None else ""

        def _add_error(s: str):
            errors.append("%s%s" % (eta_table.label_prefix, s))

        def _err_msg(s: str):
            return f"{s} in [{self.pr_id}]\n\t->[{html_url}]"

        if self.total_reported == 0:
            msg = "Total reported is 0!"
            _logger.critical(_err_msg(msg))
            _add_error("total_is_0")
            valid = False
        # if self.cust_est == 0:
        #     msg = f"Customer ETA is 0!"
        #     _logger.critical(_err_msg(msg))
        #     _add_error("cust_is_0")
        #     valid = False

        calc_stage_totals = defaultdict(int)
        for stage in ETA.stages:
            for dev, hours in self.dev_hours.items():
                calc_stage_totals[stage] += hours[stage]
        stage_totals = self.stage_totals

        for stage in ETA.stages:
            if stage_totals[stage] != calc_stage_totals[stage]:
                msg = f"Check stage totals: stage [{stage}]=[{calc_stage_totals[stage]}], NOT [{stage_totals[stage]}]"
                _logger.critical(_err_msg(msg))
                valid = False
                _add_error("stage_%s" % stage)

        calc_total_reported = sum(stage_totals.values())
        if calc_total_reported != self.total_reported:
            msg = f"Check totals: =[{self.total_reported}], NOT [{calc_total_reported}]"
            _logger.critical(_err_msg(msg))
            valid = False
            _add_error("totals")

        return valid, errors

    def md_hours(self, this_week_n, in_progress: bool = False) -> hours_row:
        """
            Return filled out `hours_row`
        """
        pr = self.pr
        eta = self

        r = hours_row()
        r[r.h_week] = this_week_n
        r[r.h_customer] = "internal"
        r[r.h_issue] = f"{'/'.join(pr.html_url.split('/')[-3:])}:{pr.title}"
        r[r.h_link_gh] = pr.html_url
        r[r.h_state] = pr.state if not in_progress else "open"
        r[r.h_closed] = pr.closed_at
        r[r.h_created] = pr.created_at
        total_days = -1
        if pr.closed_at is not None:
            try:
                total_days = (pr.closed_at - pr.created_at).days
                r[r.h_days_open] = total_days
            except Exception as e:
                pass

        r[r.h_eta] = eta.est
        r[r.h_eta_cust] = eta.cust_est

        for dev, hours in eta.dev_hours.items():
            t = sum(hours.values())
            r.dev(dev, t)

        # fill out phases when closed
        if in_progress is False:
            p0, p1, p2 = \
                eta.stage_totals[ETA.stages[0]], \
                eta.stage_totals[ETA.stages[1]], \
                eta.stage_totals[ETA.stages[2]]
            r[r.h_phase_eta] = p0
            r[r.h_phase_dev] = p1
            r[r.h_phase_review] = p2
            r[r.h_phase_total] = p0 + p1 + p2
            r[r.h_dev_total] = self.dev_hours_total()
        else:
            r[r.h_last_week] = self.dev_hours_total()

        rec = re.compile(r"jira\D?(\d+)", re.IGNORECASE)
        m = rec.search(r[r.h_issue])
        if m:
            r[r.h_link_jira] = f"{settings['TRACKER_LINK']}{m.group(1)}"
            r[r.h_customer] = settings['CUSTOMER']

        return r

    def relative_eta(self, from_monday):
        """
            Get values for a week starting at `from_monday`.
        """
        till_sunday = from_monday + timedelta(days=6)

        if self.pr.created_at.date() > till_sunday:
            _logger.critical(
                f"INVALID relative_eta for [{self.pr_id}] [{self.pr.html_url}]!!!")
            raise Exception("INVALID relative_eta")

        if is_issue(self.pr):
            _logger.critical(
                f"Cannot do relative ETA for ISSUE (only PR)! [{self.pr_id}] -> [{self.pr.html_url}]")
            return None

        created_that_week = from_monday <= self.pr.created_at.date()
        for_not_finished_week = datetime.now().date() <= till_sunday
        closed_that_week = False
        if self.pr.closed_at is not None and self.pr.closed_at.date() <= till_sunday:
            closed_that_week = True

        # 1. PR created that week
        if created_that_week:

            # created this week or already closed
            if for_not_finished_week or closed_that_week:
                return self

            # next ETA is the one we want
            chckp_eta_d_end = parse_checkpoint_eta(self.pr, till_sunday)
            if chckp_eta_d_end is None:
                _logger.info(
                    f"Missing ETA checkpoint ~[{from_monday}]: {self.pr_id} -> [{self.pr.html_url}]!!")
                return None

            rel_eta = self.copy()
            rel_eta.dev_hours = chckp_eta_d_end["dev_hours"]
            rel_eta.compute_stage_totals()
            return rel_eta

        # 2. PR created before that week, start eta should be valid
        chckp_eta_d_start = parse_checkpoint_eta(self.pr, from_monday)
        if chckp_eta_d_start is None:
            _logger.info(
                f"Missing ETA checkpoint ~[{from_monday}]: {self.pr_id} -> [{self.pr.html_url}]!!")
            return None

        if for_not_finished_week or closed_that_week:
            chckp_eta_d_end = self._d
        else:
            chckp_eta_d_end = parse_checkpoint_eta(self.pr, till_sunday)
            if chckp_eta_d_end is None:
                _logger.info(
                    f"Missing ETA checkpoint ~[{till_sunday}]: {self.pr_id} -> [{self.pr.html_url}]!!")
                return None

        rel_eta = self.copy()
        dev_hours_k = "dev_hours"
        dev_hours_start = chckp_eta_d_start[dev_hours_k]
        dev_hours_end = chckp_eta_d_end[dev_hours_k]
        for dev, hours_d in rel_eta.dev_hours.items():
            for stage in hours_d.keys():
                hours_d[stage] = dev_hours_end[dev][stage] - dev_hours_start[dev][stage]
        rel_eta.compute_stage_totals()
        return rel_eta


def parse_eta_lines(pr) -> tuple:
    """
        Return list of lines that contain the ETA table.
    """
    rec_start = re.compile("phases.*total", re.I)
    rec_line = re.compile("[|].*[|]", re.I)
    l_arr = []
    state = 0
    for l in pr.body.splitlines():
        if state == 0:
            if rec_start.search(l):
                state = 1
        if state != 1:
            continue
        if rec_line.search(l) is None:
            break
        l_arr.append(l)

    # check if we should ignore this table
    ignored = 0 < len(l_arr) and "ignore" in l_arr[0].lower()

    return l_arr, ignored


def parse_eta(pr, pr_id) -> typing.Optional[eta_table]:
    """
| Phases            | JH  |  JP  | TM |   JM | Total  |
|-----------------|----:|----:|-----:|-----:|-------:|
| ETA                  |  0  |    |     3 |      0 |        3 |
| Developing      |  5+4,5+8+8+5+8+5  |    0,5 + 1 |    0 |      0 |         45 |
| Review             |  4+4  |     2.5 + 0.5 + 1 + 1 |    0 |      0 |         13 |
| Total                |   -  |   -   |  -    |   -    |         61 |
| ETA est.             |      |       |       |         |     40  |
| ETA cust.           |   -  |   -  |   -   |   -     |        40 |
    """
    l_arr, ignored = parse_eta_lines(pr)
    if ignored:
        return None

    if len(l_arr) == 0:
        _logger.critical(f"Cannot parse ETA table [{pr_id}]\n{pr.html_url}")
        return None

    # parse
    cols = [[x.strip() for x in x.split("|")] for x in l_arr]
    try:
        eta = eta_table(pr, pr_id, cols)
    except Exception as e:
        _logger.critical(f"Cannot parse ETA table [{pr_id}]\n{pr.html_url}")
        return None

    # verification #1
    if not eta.valid:
        return None

    return eta


def parse_checkpoint_eta(pr, around_date) -> dict:
    rec = re.compile(r"/summary>\s*(.*)\s*</details", re.M)
    comments = pr.get_issue_comments()
    for c in comments:
        diff_days = (c.created_at.date() - around_date).days
        if abs(diff_days) > 1:
            continue
        if "ETA checkpoint" in c.body:
            s = " ".join(c.body.splitlines())
            m = rec.search(s)
            if m is not None:
                d = json.loads(m.group(1))
                return d
            else:
                _logger.critical(f"Cannot parse ETA comments [{c.body}] in {pr.html_url}")
                return None
    return None


def pr_with_eta(gh, start_at: datetime, state: str = None, base: str = None, include_issues: bool = False):
    """
    state: "all", "closed"
    """

    rec_pr_time = re.compile(r"[|]\s*ETA")

    for p, ignored_pr in settings["projects"]:
        repo = gh.get_repo(p)
        _logger.info(repo.name)
        if include_issues:
            issues = repo.get_issues(state='all', sort='created',
                                     direction="desc", labels=["ETA"])
            _logger.info("Total ISSUES count: [%d]", issues.totalCount)
            for issue in tqdm.tqdm(issues, total=issues.totalCount):
                if issue.created_at < start_at:
                    break
                if rec_pr_time.search(issue.body or "") is None:
                    continue
                yield repo.name, issue

        pulls = repo.get_pulls(state=state or 'all', sort='created',
                               direction="desc", base=base or settings["base"])
        _logger.info("Total PR count: [%d]", pulls.totalCount)
        for pr in tqdm.tqdm(pulls, total=pulls.totalCount):
            if pr.number in ignored_pr:
                continue

            if pr.created_at < start_at:
                break
            if rec_pr_time.search(pr.body or "") is None:
                continue
            yield repo.name, pr


def pr_with_eta_hours(gh, start_at: datetime):
    """
        Find all Issues/PRs with ETA tables valid for the report.
    """
    sort_by = "updated"
    rec_pr_time = re.compile(r"[|]\s*ETA")

    weeks = defaultdict(list)

    def _filter(iss_or_pr, ign_arr: list):
        if iss_or_pr.number in ign_arr:
            return True
        if iss_or_pr.updated_at < start_at:
            raise StopIteration()
        if rec_pr_time.search(iss_or_pr.body or "") is None:
            return True
        # ignore issues that are too old
        if iss_or_pr.created_at < start_at:
            return True
        return False

    def process_one(repo_name, iss_or_pr):
        created = iss_or_pr.created_at
        closed = iss_or_pr.closed_at
        week_d_start, year_start = created.isocalendar()[1], created.isocalendar()[0]
        end_d = closed.isocalendar() if closed else datetime.now().isocalendar()
        week_d_end, year_end = end_d[1], end_d[0]
        is_closed = closed is not None

        week_year = []
        use_year = year_start
        week_d_i = week_d_start
        while week_d_i != week_d_end + 1:
            if week_d_i == 0:
                use_year = year_end
            week_year.append((week_d_i, use_year))
            week_d_i = (week_d_i + 1) % 53
        #
        if len(week_year) > settings["warn_if_opened_longer_than"]:
            lately = datetime.now() - timedelta(days=14)
            updated, week = was_updated(iss_or_pr, since=lately.date())
            # only ignore if not updated lately
            if not updated:
                if iss_or_pr.state != "closed":
                    _logger.warning(
                        f"IGNORING: OPENED for too long [{len(week_year)} weeks][{iss_or_pr.created_at}]: {repo_name}:{iss_or_pr.number} [{iss_or_pr.html_url}] [{iss_or_pr.title}]")
                return

        for week_d, year in week_year:
            week_state = 'open'
            if week_d == week_d_start:
                week_state = 'created'
            if week_d == week_d_end and is_closed:
                week_state = 'closed'
            weeks[f"{year}_{week_d:02}"].append((repo_name, iss_or_pr, week_state))

    def process(arr, ignored_pr: list, ftor):
        if arr.totalCount == 0:
            return
        try:
            for iss_or_pr in tqdm.tqdm(arr, total=arr.totalCount):
                ign = [] if is_issue(iss_or_pr) else ignored_pr
                if _filter(iss_or_pr, ign):
                    continue
                ftor(repo.name, iss_or_pr)
        except StopIteration:
            pass

    for p, p_dict in settings["projects"]:
        repo = gh.get_repo(p)
        _logger.info(repo.name)

        try:
            issues = repo.get_issues(state='all', sort=sort_by,
                                     direction="desc", labels=["ETA"])
            _logger.info(f"[{p}] ISSUES: [{issues.totalCount:03d}]")
            process(issues, p_dict.get("ignored_issues", []), process_one)
        except StopIteration:
            pass

        try:
            pulls = repo.get_pulls(state='all', sort=sort_by,
                                   direction="desc", base=p_dict["pr_base"])
            _logger.info(f"[{p}] PR: [{pulls.totalCount:03d}]")
            process(pulls, p_dict.get("ignored_pr", []), process_one)
        except StopIteration:
            pass
    week_keys = sorted(weeks.keys(), reverse=True)
    for key in week_keys:
        _logger.info(f"Week {key}: {len(weeks[key]):03d} items")
        for repo_name, iss_or_pr, week_state in weeks[key]:
            _logger.info(
                f"\t{week_state: >8}{repo_name: >20}: [{iss_or_pr.html_url: >55}] [{iss_or_pr.title}]")
    _logger.info(40 * "=")

    return weeks


def find_cust_est(gh, issue_arr, state=None):
    issue_arr = sorted(issue_arr)
    issue_d = {k: None for k in issue_arr}

    _logger.info("Looking for %s", issue_arr)
    for repo_name, pr in pr_with_eta(gh, settings["start_time"], state=state):
        pr_id = get_pr_id(repo_name, pr)

        for i, issue in enumerate(issue_arr):
            if issue not in pr.title:
                continue
            eta = parse_eta(pr, pr_id)
            if eta is None:
                log_err("Cannot parse", pr, pr_id)
                continue
            if eta.total_reported == 0 or eta.cust_est == 0:
                log_err("Empty times", pr, pr_id)
                continue

            if issue_d[issue] is not None:
                _logger.warning("Found >1 issues with billable times for [%s]", pr_id)

            issue_d[issue] = (eta, pr, pr_id)

    totals = {
        "cust_est": 0,
        "total_reported": 0,
    }

    for k, v in issue_d.items():
        if v is None:
            _logger.info("Cannot find time for [%s]", k)
            continue
        eta, _1, _2 = v
        totals["cust_est"] += eta.cust_est
        totals["total_reported"] += eta.total_reported

    issue_d = {k: v for k, v in issue_d.items() if v is not None}

    _logger.info(20 * "=")
    _logger.info("Looked for [%d] issues, found [%d] issues",
                 len(issue_arr), len(issue_d))

    _logger.info(20 * "=")

    abbrev = {
        "c-image-to-text": "i2t"
    }

    s = ""
    for k, v in sorted(issue_d.items(), key=lambda x: x[1][2]):
        if v is None:
            continue
        eta, pr, pr_id = v
        proj, proj_s = pr_id.split(":", maxsplit=1)
        pr_id_s = "%s:%s" % (abbrev.get(proj, proj), proj_s)

        s += "%-40s: [ETA:%4.1f][REPORTED:%4.1f]->[%s]\n" % (
            pr_id_s[:40], eta.cust_est, eta.total_reported, pr.html_url
        )

    _logger.info("\n\n" + s)
    _logger.info(20 * "=")
    _logger.info("Found [cust_est: %4.1f] [total_reported: %4.1f]",
                 totals["cust_est"], totals["total_reported"])


def find_eta_sum(gh, issue_arr):
    issue_arr = sorted(issue_arr)

    found_etas = []
    _logger.info("Looking for %s", issue_arr)
    for repo_name, pr in pr_with_eta(gh, settings["start_time"]):
        pr_id = get_pr_id(repo_name, pr)

        for i, issue in enumerate(issue_arr):
            if issue not in pr.title:
                continue
            eta = parse_eta(pr, pr_id)
            if eta is None:
                continue

            stages = {k: [] for k in ETA.stages}
            for stage in stages.keys():
                for dev, hours in eta.dev_hours.items():
                    stages[stage].append(hours[stage])

            stages_sum = {k: sum(v) for k, v in stages.items()}
            _logger.info("%s\n\t->%s", pr_id, pformat(stages_sum))
            found_etas.append(issue)

    missing = set(issue_arr) - set(found_etas)
    if 0 < len(missing):
        _logger.critical("Did not find ETAs for issues [%s]", missing)


def find_hours(gh, input_arr, input_are_ids):
    input_arr = sorted(input_arr)

    if input_are_ids:
        input_arr = [float(x) for x in input_arr]

    devs = defaultdict(dict)
    _logger.info("Looking for %s", input_arr)
    for repo_name, pr in pr_with_eta(gh, settings["start_time"]):
        pr_id = get_pr_id(repo_name, pr)

        if input_are_ids is False:
            for i, issue in enumerate(input_arr):
                if issue not in pr.title:
                    continue
                eta = parse_eta(pr, pr_id)
                if eta is None:
                    continue
                for dev, hours in eta.dev_hours.items():
                    devs[dev][issue] = hours
        else:
            if pr.number in input_arr:
                eta = parse_eta(pr, pr_id)
                if eta is None:
                    continue
                for dev, hours in eta.dev_hours.items():
                    devs[dev][pr.number] = hours

    _logger.info(pformat(devs))
    if 0 == len(devs):
        return

    got_issues = set(devs[list(devs.keys())[0]].keys())
    missing = set(input_arr) - got_issues
    if 0 < len(missing):
        _logger.critical("Did not find hours for issues [%s]", missing)


def validate(gh, start_date: datetime, state: str = "closed", sort_by=None):
    ok_status = []

    for repo_name, pr_or_issue in pr_with_eta(gh, start_date, state=state, include_issues=True):
        pr_id = get_pr_id(repo_name, pr_or_issue)
        eta = parse_eta(pr_or_issue, pr_id)
        if eta is None:
            continue

        valid, _1 = eta.validate_hours(pr_or_issue)
        if valid:
            msg = "Issue [%-70s] is OK" % (eta.pr_id,)
            ok_status.append((msg, pr_or_issue))
        else:
            _logger.error(f"PROBLEM: Issue [{eta.pr_id:70s}] has issues!!!")

    if 0 == len(ok_status):
        return

    # sort
    sort_by = sort_by or 'closed_at'
    ok_status.sort(key=lambda x: getattr(x[1], sort_by), reverse=True)
    last_week_n = -1
    for msg, pr in ok_status:
        cal = getattr(pr, sort_by).isocalendar()
        this_week_n = cal[1]
        if last_week_n != this_week_n:
            _logger.info(f"=== {cal[0]}: week #{this_week_n:02d}   " + 20 * "=")
            last_week_n = this_week_n

        msg += " [%s:%%s]\t%s" % (sort_by, pr.html_url)
        _logger.info(msg, getattr(pr, sort_by))


def was_updated(pr_or_issue, since=None, update_events=None):
    update_events = update_events or ("committed",)
    # when was the last commit
    issue = pr_or_issue
    try:
        issue = pr_or_issue.as_issue()
    except Exception as e:
        pass
    updated = [x for x in issue.get_timeline() if x.event in update_events]
    if len(updated) == 0:
        return False, -1

    last_update = updated[-1]
    try:
        if last_update.created_at is not None:
            dt = last_update.created_at
        else:
            dt = last_update.raw_data["author"]["date"]
            dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%SZ")
        week = dt.isocalendar()[1]
        if since is None:
            return True, week
        return since <= dt.date(), week

    except Exception as e:
        _logger.critical(f"Cannot parse commit events for {pr_or_issue.html_url}: {e}")
    return False, -1


def store_checkpoint(gh, start_date: datetime, dry=False):
    """
        Find ETA, check if in the last week there was an update, if so, store the checkpoint.
    """
    today = datetime.today().date()
    monday = prev_monday()
    _logger.info(
        f"Checking updates since [{monday}] till [{today}] totalling [{today - monday}] days.")

    stale = []
    for repo_name, pr in pr_with_eta(gh, start_date, state="open"):
        pr_id = get_pr_id(repo_name, pr)
        eta = parse_eta(pr, pr_id)
        if eta is None:
            _logger.info(f"No ETA for [{pr_id}] [{pr.html_url}]")
            continue

        updated, _1 = was_updated(
            pr, since=monday, update_events=("committed", "commented"))

        # add new checkpoint
        body = f"""<details><summary>ETA checkpoint [{monday}]-[{today}]</summary>
        {json.dumps(eta.d, indent=2)}
        </details>"""
        _logger.info(f"Checkpoint for [{pr_id}] [{pr.html_url}]")
        if dry:
            continue
        pr.as_issue().create_comment(body)

        if not updated:
            stale.append((pr_id, pr.html_url))

    _logger.info("=====")
    for pr_id, html_url in stale:
        _logger.info(
            f"Issue not updated lately (checkpoint made): [{pr_id}] [{html_url}]")


def find_hours_all(gh, start_date: datetime, output_md: str = None):
    """
        Show hours of all specified issues.
    """
    def _monday_date(year, week):
        first = date(year, 1, 1)
        base = 1 if first.isocalendar()[1] == 1 else 8
        return first + timedelta(days=base - first.isocalendar()[2] + 7 * (week - 1))

    weeks = pr_with_eta_hours(gh, start_date)

    with open(output_md, "w+", encoding="utf-8") as fout:
        fout.write(f"# Hours of all issues (generated at {_ts})\n\n")

        # sort helper
        sort_states = {'closed': 0, 'created': 1, 'open': 2, }

        # iterate through weeks
        week_keys = sorted(weeks.keys(), reverse=True)
        for week in week_keys:
            year, week_n = week.split("_")
            year = int(year)
            week_n = int(week_n)
            monday = _monday_date(year, week_n)
            sunday = monday + timedelta(days=6)
            fout.write(f"\n\n## {week}: [{monday} - {sunday}]\n\n")
            fout.write(f"{hours_row.header}\n")
            fout.write(f"{hours_row.header_sep}\n")

            # sort based on open/closed states
            iss_pr_arr = sorted(weeks[week], key=lambda x: sort_states.get(x[2], 10))
            for repo_name, iss_pr, week_state in tqdm.tqdm(iss_pr_arr):
                iss_pr_id = get_pr_id(repo_name, iss_pr)
                eta = parse_eta(iss_pr, iss_pr_id)
                if eta is None:
                    continue
                # when was the last commit
                updated, week = was_updated(eta.pr)
                if not updated:
                    _logger.info(
                        f"PR/ISSUE {eta.pr.html_url} not updated in the last week")

                eta_rel = eta.relative_eta(monday)
                if eta_rel is None:
                    continue
                r = eta_rel.md_hours(week_n, in_progress=(week_state != "closed"))
                fout.write(f"{str(r)}\n")


def store(gh, out_file):
    d = {
        "state": {
        },
        "timestamp": _ts
    }
    for repo_name, pr in pr_with_eta(gh, settings["start_time"], state="all"):
        pr_id = get_pr_id(repo_name, pr)
        eta_lines, _1 = parse_eta_lines(pr)
        if 0 == len(eta_lines):
            _logger.critical("No ETA for [%s]", pr_id)
            continue

        if repo_name not in d["state"]:
            d["state"][repo_name] = {}

        d["state"][repo_name][pr_id] = {
            "updated": str(pr.updated_at),
            "state": pr.state,
            "eta": eta_lines,
        }

    _logger.info("Saving to [%s]", out_file)
    with open(out_file, mode="w+") as fout:
        json.dump(d, fout, sort_keys=True, indent=2)


def get_output_file(output_d: str, ts, file_str: str):
    """ Get output file name. """
    if not os.path.exists(output_d):
        os.makedirs(output_d)
    return os.path.join(output_d, f"{ts}.{file_str}")


# =================

if __name__ == '__main__':
    UNSPECIFIED = object()

    parser = argparse.ArgumentParser(description='Github Helper Tools')
    parser.add_argument(
        '--eta-cust', help='Print customer ETA for these issues', type=str, required=False)
    parser.add_argument(
        '--hours', help='Print hours for these issues', type=str, default=UNSPECIFIED, required=False, nargs='?')
    parser.add_argument(
        '--list', help='List all issues with ETA', action="store_true", required=False)
    parser.add_argument(
        '--store', help='Serialize current state', action="store_true", required=False)
    parser.add_argument(
        '--validate', help='Validate closed ETA issues', action="store_true", required=False)
    parser.add_argument(
        '--eta-sum', help='Print ETA sum for these issues', type=str, required=False)
    parser.add_argument(
        '--state', help='Filter issues based on the state (open, closed, all)', type=str, default=None, required=False)
    parser.add_argument(
        '--sort', help='Sort by (closed_ar, merged_at) date', type=str, default=None, required=False)
    parser.add_argument(
        '--input-pr-id', help='Input are pr ids not description', action="store_true", required=False)
    parser.add_argument(
        '--settings', help='Input settings json file', required=False, default="settings.json")
    parser.add_argument(
        '--check-last', help='Check last N weeks .e.g, `4w`', required=False)
    parser.add_argument(
        '--checkpoint', help='Add new comment with a checkpoint of hours', required=False, action="store_true")
    parser.add_argument(
        '--dry-run', help='Do not make any changes (valid for --checkpoint)', required=False, action="store_true")
    parser.add_argument(
        '--output-file', help='Output file (valid for --hours)', required=False, default=None, type=str)
    flags = parser.parse_args()

    _logger.info('Started at [%s]', datetime.now())

    gh_key = "GITHUB_PAT"
    if gh_key not in os.environ:
        _logger.critical(f"Cannot find [{gh_key}] in env")
        sys.exit(1)

    # load settings
    settings_file = os.path.join(_this_dir, flags.settings)
    _logger.info("Using [%s] settings file", settings_file)
    settings = load_settings(settings_file)

    # init based on settings
    hours_row.init()

    gh = Github(os.environ[gh_key])

    if flags.check_last:
        rec = re.compile(r"^(\d+)w$")
        m = rec.match(flags.check_last)
        if m:
            since = datetime.now() - timedelta(weeks=int(m.group(1)))
            settings["start_time"] = since
        else:
            _logger.critical("Unknown format f{flags.check_last}")
            sys.exit(1)
        _logger.info(f"Valid issues/PRs since {settings['start_time']}")

    if flags.store:
        if not os.path.exists(settings["state_dir"]):
            os.makedirs(settings["state_dir"])
        out_f = os.path.join(settings["state_dir"], "%s.json" % _ts)
        store(gh, out_f)
        sys.exit(0)

    if flags.validate:
        start_date = settings["start_time"]
        validate(gh, start_date, state=flags.state, sort_by=flags.sort)
        sys.exit(0)

    if flags.eta_cust is not None:
        find_cust_est(gh, flags.eta_cust.split(","), flags.state)
        sys.exit(0)

    # store hours
    if flags.checkpoint:
        start_date = settings["start_time"]
        store_checkpoint(gh, start_date, dry=flags.dry_run)
        sys.exit(0)

    # iterate and show hours for all PRs conforming to the input
    if flags.hours is None:
        if flags.output_file is not None:
            output_f = flags.output_file
        else:
            output_f = get_output_file(settings["result_dir"], _ts, "hours.md")
        start_date = settings["start_time"]
        find_hours_all(gh, start_date, output_md=output_f)
        if os.path.exists(output_f):
            if not os.path.exists(settings["result_dir"]):
                os.makedirs(settings["result_dir"])
            shutil.copy(output_f, os.path.join(settings["result_dir"], "hours.md"))
        sys.exit(0)

    if isinstance(flags.hours, str):
        find_hours(gh, flags.hours.split(","), flags.input_pr_id)
        sys.exit(0)

    if flags.eta_sum is not None:
        find_eta_sum(gh, flags.eta_sum.split(","))
        sys.exit(0)

    if flags.list:
        last_repo_name = None
        last_week_n = -1
        for repo_name, pr in pr_with_eta(gh, settings["start_time"]):
            # reset
            if repo_name != repo_name:
                repo_name = repo_name
                last_week_n = -1

            this_week_n = pr.created_at.isocalendar()[1]
            if last_week_n != this_week_n:
                _logger.info(("\n\nWeek #%02d   " + 40 * "="), this_week_n)
                last_week_n = this_week_n

            assignee = pr.assignee.login if pr.assignee else "unknown"
            merged_by = pr.merged_by.login if pr.merged_by else "unknown"
            _logger.info(
                "[created:%20s -> closed:%20s] assigned to [%10s] state [%8s] merged [%5s] merged by [%10s]\n\t%s\n\t%s",
                pr.created_at, pr.closed_at, assignee, pr.state, pr.merged, merged_by,
                str(pr),
                pr.html_url
            )
