"""
https://pygithub.readthedocs.io/en/latest/examples.html

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
import logging
import argparse
import re
import glob
import json
import tqdm
from pprint import pformat
from collections import defaultdict, OrderedDict
from github import Github

from datetime import datetime, timedelta

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

    return cfg


settings = {}


# =================


class ETA:
    stages = ("ETA", "Developing", "Review")
    key_phase = "phases"
    key_eta = "eta"
    key_eta_cust = "ETA cust"
    key_total = "Total"


# =================

def log_err(msg, pr, pr_id):
    tmpl = "\n" + ("<" * 10) + "\n%s [%s]\n\t->%s\n" + (">" * 10)
    _logger.info(tmpl, msg, pr_id, pr.html_url)


def sum_hours(s, pr_id):
    """
        Try to sum the cell.
    """
    try:
        return float(eval(s))
    except:
        _logger.info("Cannot parse [%s] in [%s]", s, pr_id)
    return -1.


def get_pr_id(repo_name, pr):
    return "%s:%s:%s" % (repo_name, pr.number, pr.title)


class eta_row:
    """
        Row of an ETA
    """

    def __init__(self, arr, pr_id):
        self.arr = arr
        self.pr_id = pr_id

    @property
    def name(self): return self.arr[1]

    @property
    def total(self): return self.arr[-2]

    @property
    def dev_hours(self): return [sum_hours(x, self.pr_id) for x in self.arr[2:-2]]

    @property
    def dev_names(self): return self.arr[2:-2]


class eta_table:

    label_prefix = "ETA_err_"

    def __init__(self, pr, pr_id, cols):
        self.pr = pr
        self.pr_id = pr_id
        self.rows = [eta_row(x, pr_id) for x in cols]
        self._valid = self._validate_keys()
        if not self._valid:
            return
        self._d = self._parse()
        self._valid = self._d is not None

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

    @property
    def stage_totals(self):
        return self._d["stage_totals"]

    def _parse(self):
        d = {}
        try:
            d["customer_estimate"] = float(self.rows[-1].total)
        except:
            return None
        try:
            d["estimate"] = float(self.rows[-2].total)
        except:
            return None
        try:
            d["total_reported"] = float(self.rows[-3].total)
        except:
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
                    _logger.critical(f"Cannot parse stage total [{row.total}] [{stage}] [{self.pr_id}]\n{self.pr}")
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

    def _validate_keys(self):
        ver_1 = self.rows[0].name.lower() == ETA.key_phase
        ver_2 = ETA.key_eta in self.rows[-1].name.lower()
        ver_3 = True
        try:
            float(self.rows[2].total)
        except:
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

        def _add_error(s: str): errors.append("%s%s" % (eta_table.label_prefix, s))

        if self.total_reported == 0:
            _logger.critical("Total reported is 0!")
            _add_error("total_is_0")
            valid = False
        if self.cust_est == 0:
            _logger.critical("Customer ETA is 0!")
            _add_error("cust_is_0")
            valid = False

        calc_stage_totals = defaultdict(int)
        for stage in ETA.stages:
            for dev, hours in self.dev_hours.items():
                calc_stage_totals[stage] += hours[stage]
        stage_totals = self.stage_totals

        html_url = pr.html_url if pr is not None else ""

        for stage in ETA.stages:
            if stage_totals[stage] != calc_stage_totals[stage]:
                _logger.critical("Check stage totals: stage [%s]=[%s], NOT [%s] in [%s]\n\t->[%s]",
                                 stage, calc_stage_totals[stage], stage_totals[stage], self.pr_id, html_url)
                valid = False
                _add_error("stage_%s" % stage)

        calc_total_reported = sum(stage_totals.values())
        if calc_total_reported != self.total_reported:
            _logger.critical("Check totals: =[%s], NOT [%s] in [%s]\n\t->[%s]",
                             self.total_reported, calc_total_reported, self.pr_id, html_url)
            valid = False
            _add_error("totals")

        return valid, errors


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


def parse_eta(pr, pr_id):
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


def pr_with_eta(gh, start_at: datetime, state: str = None, base: str = None):
    """
    state: "all", "closed"
    """

    rec_pr_time = re.compile(r"[|]\s*ETA")

    for p, ignored_pr in settings["projects"]:
        repo = gh.get_repo(p)
        _logger.info(repo.name)
        pulls = repo.get_pulls(state=state or 'all', sort='created',
                               direction="desc", base=base or settings["base"])
        _logger.info("Total PR count: [%d]", pulls.totalCount)
        for pr in tqdm.tqdm(pulls):
            if pr.number in ignored_pr:
                continue

            if pr.created_at < start_at:
                break
            if rec_pr_time.search(pr.body or "") is None:
                continue
            yield repo.name, pr


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

    for repo_name, pr in pr_with_eta(gh, start_date, state=state):
        pr_id = get_pr_id(repo_name, pr)
        eta = parse_eta(pr, pr_id)
        if eta is None:
            continue

        err, _1 = eta.validate_hours(pr)
        if not err:
            msg = "Issue [%-70s] is OK" % (eta.pr_id,)
            ok_status.append((msg, pr))

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
    h_closed = "Closed"
    h_phase_eta = "Phase ETA"
    h_phase_dev = "Phase Dev"
    h_phase_review = "Phase Review"
    h_phase_total = "Phase Total"
    h_dev_total = "Dev Total"
    h_dev_jh = "Dev JH"
    h_dev_jp = "Dev JP"
    h_dev_tm = "Dev TM"
    h_dev_others = "Dev Others"

    h_spec = (
        h_week, h_customer, h_issue, h_link_gh, h_link_jira, h_state, h_tracked, h_eta, h_eta_cust, h_closed,
        h_phase_eta, h_phase_dev, h_phase_review, h_phase_total, h_dev_total,
        h_dev_jh, h_dev_jp, h_dev_tm, h_dev_others
    )
    header = "|" + ("|".join([f" {x:10s} " for x in h_spec]))[1:] + " |"
    header_len = header.count("|") - 1
    header_sep = header_len * ("| %10s " % "----:") + " |"

    def __init__(self):
        self._d = OrderedDict()
        for k in self.h_spec:
            self._d[k] = ""

    def __getitem__(self, k): return self._d[k]
    def __setitem__(self, k, v): self._d[k] = v

    def dev(self, key, value):
        exp_k = f"Dev {key}"
        for k in self._d.keys():
            if k == exp_k:
                self._d[k] = value
                return
        if self._d[self.h_dev_others] == "":
            self._d[self.h_dev_others] = value
        else:
            self._d[self.h_dev_others] += value

    def __str__(self):
        s = ""
        for k, v in self._d.items():
            s += f"| {v} "
        return s + "|"


def find_hours_all(gh, start_date: datetime, state: str = "closed", sort_by=None, output_md: str = None):
    """
        Show hours of all specified issues.
    """
    ok_status = []
    in_progress = []

    def pr_to_md_hours(pr, eta, this_week_n):
        r = hours_row()
        r[r.h_week] = this_week_n
        r[r.h_customer] = "internal"
        r[r.h_issue] = f"{'/'.join(pr.html_url.split('/')[-3:])}:{pr.title}"
        r[r.h_link_gh] = pr.html_url
        r[r.h_state] = pr.state
        r[r.h_closed] = pr.closed_at

        if pr.state == "closed":
            assert eta is not None
            r[r.h_eta] = eta.est
            r[r.h_eta_cust] = eta.cust_est

            p0, p1, p2 = \
                eta.stage_totals[ETA.stages[0]], \
                eta.stage_totals[ETA.stages[1]], \
                eta.stage_totals[ETA.stages[2]]
            r[r.h_phase_eta] = p0
            r[r.h_phase_dev] = p1
            r[r.h_phase_review] = p2
            r[r.h_phase_total] = p0 + p1 + p2

            dev_total = 0
            for dev, hours in eta.dev_hours.items():
                t = sum(hours.values())
                r.dev(dev, t)
                dev_total += t
            r[r.h_dev_total] = dev_total

        rec = re.compile(r"jira\D?(\d+)", re.IGNORECASE)
        m = rec.search(r[r.h_issue])
        if m:
            r[r.h_link_jira] = f"https://adventhp.atlassian.net/browse/OCR-{m.group(1)}"
            r[r.h_customer] = "advent"

        return r

    for repo_name, pr in pr_with_eta(gh, start_date, state="all"):
        if pr.state != state:
            if pr.state == "open":
                in_progress.append(pr)
            continue

        pr_id = get_pr_id(repo_name, pr)
        eta = parse_eta(pr, pr_id)
        if eta is None:
            continue
        ok_status.append((eta, pr))

    if 0 == len(ok_status):
        return

    # sort by week
    in_progress_d = defaultdict(list)
    for pr in in_progress:
        # when was the last commit
        commit_events = [x for x in pr.as_issue().get_timeline() if x.event == "committed"]
        last_commit = commit_events[-1]
        try:
            dt = last_commit.raw_data["author"]["date"]
            dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%SZ")
            week = dt.isocalendar()[1]
            update_week = pr.updated_at.isocalendar()[1]
            in_progress_d[week].append(pr)
        except Exception as e:
            _logger.critical(f"Cannot parse commit events for {pr.html_url}: {e}")

    # sort
    sort_by = sort_by or 'closed_at'
    ok_status.sort(key=lambda x: getattr(x[1], sort_by), reverse=True)
    last_week_n = -1
    with open(output_md, "w+", encoding="utf-8") as fout:
        fout.write(f"# Hours of all issues (generated at {_ts})\n\n")

        for eta, pr in ok_status:
            cal = getattr(pr, sort_by).isocalendar()
            this_week_n = cal[1]
            if last_week_n != this_week_n:

                # print out in progress last week
                for pr_prg in in_progress_d[last_week_n]:
                    r = pr_to_md_hours(pr_prg, None, last_week_n)
                    fout.write(f"{str(r)}\n")

                # print new header
                d = f"{cal[0]}-W{this_week_n:02d}"
                monday = datetime.strptime(d + '-1', "%Y-W%W-%w")
                sunday = monday + timedelta(days=6)
                fout.write(f"\n\n## {cal[0]}: week #{this_week_n:02d} - {monday.date()} - {sunday.date()}\n\n")
                last_week_n = this_week_n
                fout.write(f"{hours_row.header}\n")
                fout.write(f"{hours_row.header_sep}\n")

            # print closed issue
            r = pr_to_md_hours(pr, eta, last_week_n)
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

    # iterate and show hours for all PRs conforming to the input
    if flags.hours is None:
        # output_f = get_output_file(settings["result_dir"], _ts, "hours.md")
        output_f = get_output_file(settings["result_dir"], "", "hours.md")
        start_date = settings["start_time"]
        find_hours_all(gh, start_date, state=flags.state, sort_by=flags.sort, output_md=output_f)
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
