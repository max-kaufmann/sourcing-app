from sourcing_app.task import source_authors_task
from sourcing_app.agents import candidate_rater_agent
from sourcing_app.arxiv import arxiv_id_from_url
from inspect_ai import eval_set
import uuid
from datetime import datetime
from pathlib import Path
from sourcing_app.task import task_output_to_candidate_df
from dotenv import load_dotenv


ARXIV_URLS = [
    "https://arxiv.org/abs/2310.10077",
    "https://arxiv.org/abs/2306.05499",
    "https://arxiv.org/abs/2311.01011",
    "https://arxiv.org/abs/2311.11538",
    "https://arxiv.org/abs/2311.11415"
]

TOP_N_AUTHORS = 2 # Select only the top N authors for each paper. If None, all authors are selected.
MODEL_NAME = "anthropic/claude-3-7-sonnet-20250219"
DEFAULT_RATER_SYSTEM_PROMPT = """You are a candidate finder bot, built to find potential candidates for the UK's AI Safety Institute, a set of machine learning researchers in the UK government. Your job is to find background information about a candidate, where you are given the candidate's name and the url for  a paper which the candidate has appeared in. You should rate the candidate on a scale of 1 to 5, where 1 is the worst and 5 is the best. As well as looking at general competence and experience at machine learning research and engineering, you should also particularly focus on:

- Whether the author has a track record of publishing in ML conferences and a clear interest in adversarial ML or AI safety work (eg as demonstrated by publishing a at least few papers in either area).
- Whether the author has worked at frontier ai labs, has finished a PhD, or is towards the end of their PhD.

Use your submit tool to return a rating in the following format. Your submission in the tool should be a json object with the following schema:

{model_json_schema}

Make sure to first use your web search too to get some background information about the candidate, and then use your web browser to find other information on the candidate. DONT GIVE UP! Only once you are certain that you cannot find the relevant information about the candidate, then you should submit a rating - however, think carefully about the information that you have found."""

load_dotenv()

run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{str(uuid.uuid4())[:4]}"
run_dir = Path(f"./outputs/{run_name}")
run_dir.mkdir(parents=True, exist_ok=True)

task = source_authors_task(
    arxiv_ids=[arxiv_id_from_url(url) for url in ARXIV_URLS],
    top_n_authors=TOP_N_AUTHORS,
    solver=candidate_rater_agent(system_prompt=DEFAULT_RATER_SYSTEM_PROMPT),
)


log_dir = run_dir / "inspect_logs"
log_dir.mkdir(parents=True, exist_ok=True)

_, logs = eval_set(tasks=[task], log_dir=log_dir.as_posix(), model=MODEL_NAME)
output_df = task_output_to_candidate_df(logs[0])
output_df.to_csv(run_dir / "output.csv", index=False)


