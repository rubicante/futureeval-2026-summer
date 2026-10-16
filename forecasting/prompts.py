"""Prompt templates for each question type.

Templates reference canonical ForecastQuestion fields: {title}, {body}, etc.
"""

BINARY_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

Question background:
{body}

Your research assistant says:
{summary_report}

Today is {today}. The question resolves at {resolve_time}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A brief description of a scenario that results in a No outcome.
(d) A brief description of a scenario that results in a Yes outcome.

You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""

NUMERIC_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

Background:
{body}

Units for answer: {units}

Your research assistant says:
{summary_report}

Today is {today}. The question resolves at {resolve_time}.

{lower_bound_message}
{upper_bound_message}


Formatting Instructions:
- Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1m).
- Never use scientific notation.
- Always start with a smaller number (more negative if negative) and then increase from there

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The outcome if nothing changed.
(c) The outcome if the current trend continued.
(d) The expectations of experts and markets.
(e) A brief description of an unexpected scenario that results in a low outcome.
(f) A brief description of an unexpected scenario that results in a high outcome.

You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unkowns.

The last thing you write is your final answer as:
"
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"
"""

DATE_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

Background:
{body}

Your research assistant says:
{summary_report}

Today is {today}. The question resolves at {resolve_time}.

{lower_bound_message}
{upper_bound_message}

Formatting Instructions:
- This is a date question, and as such, the answer must be expressed in terms of dates.
- The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
- Always start with a lower date chronologically and then increase from there.
- Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The outcome if nothing changed.
(c) The outcome if the current trend continued.
(d) The expectations of experts and markets.
(e) A brief description of an unexpected scenario that results in a low outcome.
(f) A brief description of an unexpected scenario that results in a high outcome.

You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

The last thing you write is your final answer as:
"
Percentile 10: YYYY-MM-DD (oldest date)
Percentile 20: YYYY-MM-DD
Percentile 40: YYYY-MM-DD
Percentile 60: YYYY-MM-DD
Percentile 80: YYYY-MM-DD
Percentile 90: YYYY-MM-DD (newest date)
"
"""

MULTIPLE_CHOICE_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

The options are: {options}


Background:
{body}

Your research assistant says:
{summary_report}

Today is {today}. The question resolves at {resolve_time}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A description of an scenario that results in an unexpected outcome.

You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

The last thing you write is your final probabilities for the N options in this order {options} as:
<option 1 text exactly as listed above>: <probability>
<option 2 text exactly as listed above>: <probability>
...
<option N text exactly as listed above>: <probability>

Write each option's text exactly as it appears in the list above — never
placeholders like "Option_A".
"""
