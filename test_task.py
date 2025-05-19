from sourcing_app.task import source_authors, arxiv_id_from_url
from inspect_ai import eval_set
import uuid
from datetime import datetime
from pathlib import Path
from sourcing_app.task import task_output_to_candidate_df
from dotenv import load_dotenv


ARXIV_URLS = ["https://arxiv.org/pdf/2307.08678"]
MAX_AUTHORS = None # Maximum authors to source for each paper
MODEL_NAME = "anthropic/claude-3-7-sonnet-20250219"


load_dotenv()

run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{str(uuid.uuid4())[:4]}"
run_dir = Path(f"./outputs/{run_name}")
run_dir.mkdir(parents=True, exist_ok=True)

task = source_authors(
    arxiv_ids=[arxiv_id_from_url(url) for url in ARXIV_URLS],
    maximum_authors=MAX_AUTHORS,
)


log_dir = run_dir / "inspect_logs"
log_dir.mkdir(parents=True, exist_ok=True)

_, logs = eval_set(tasks=[task], log_dir=log_dir.as_posix(), model=MODEL_NAME)
log = logs[0]
assert log.samples is not None
outputs = [task_output_to_candidate_df(log) for log in logs]
