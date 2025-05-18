from sourcing_app.sourcing import source_authors, arxiv_id_from_url
from inspect_ai import eval_set
import uuid
import re

ARXIV_URLS = ["https://arxiv.org/pdf/2307.08678"]
MAX_AUTHORS = 1

task = source_authors(
    arxiv_ids=[arxiv_id_from_url(url) for url in ARXIV_URLS],
    maximum_authors=MAX_AUTHORS
)
log_dir = f"./logs/{uuid.uuid4()}"

_, logs = eval_set(tasks=[task], log_dir=log_dir, model="anthropic/claude-3-5-sonnet-20240620")
