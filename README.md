# How to start

Create .env with 
```
GITHUB_PAT=yourPATfromGITHUBaccount
```

## List all ETA issues

```
python prtime.py --list

Started at [2022-03-21 23:25:07.034820]
c-image-to-text
Total PR count: [262]
Total PR count with time: [17]
[created: 2022-03-18 12:53:44 -> closed:                None] assigned to [     ku-bo] state [    open] merged [False] merged by [   unknown]
...
```


## Number of customer ETA per issue

```
python prtime.py --eta-cust=949,960,939,944,943,956
python prtime.py --eta-cust=944,949,960,943,939,942,968,957,958 --state=closed
```

## Dev hours per issue

```
python prtime.py --hours=949,960,939,944,943,956
```

## Validate calculations

```
python prtime.py --validate
python prtime.py --validate --state=closed
python prtime.py --validate --state=closed --sort=merged_at
```

## Store checkpoint ETAs

```
python prtime.py --checkpoint
```

## Write the weekly tab straight into the timesheet .xlsx

Copies the `template` sheet into a new `od <date>` tab and fills the ETA-tracked
rows (the same data as `--hours`), so you don't hand-transcribe the Markdown.

```
# previous Monday's week, into the shared timesheet
python prtime.py --xlsx "Shared time sheets.xlsx" --check-last=2w

# an explicit week; --force overwrites an existing tab
python prtime.py --xlsx "Shared time sheets.xlsx" --week=2026-08-10 --check-last=2w --force
```

Still filled by hand (not derivable from PR ETA tables): non-PR lines
(Release / Tier), OFF/leave hours, and any multi-week split deltas.

# Other projects

```
python prtime.py --validate --state=closed --settings=___settings.dq.json
```