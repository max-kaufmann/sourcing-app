# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sourcing application built with Python, buil for the UK's AI Safety Institute (AISI). We are currently creating a small mvp of the project, which lives in (`mvp.py`), which is a jupyter notebook that uses the ArXiv API to query for authors of academic papers. 


## Plan for sourcing application

Workflow we’re trying to automate: Here is the workflow we want to automate:

1) Take in a list of arxiv papers where we suspect we might want to hire the authors.  
2) Parse out a list of the authors from those papers.  
3) Classify whether these authors are of interest to the organisation. Classifying authors should involve a custom prompt, but some important considerations are:  
   1) *They would be interested in working for AISI*. Most common thing is they have shown an interest in Safety.  
   2) *We would be interested in them.* They seem generally competent / have some hard skills that we need / have done some specific work which we are interested in.  
4) For the classified authors, extract some data about them in a structured format. Again, customizable, but some classic things might be: Current role, years of experience, links to websites.  
5) Put this data into a database. My understanding is that the current plan is Airtable.

Code workflow for the for the mvp. This will all be wrapped in an inspect Task object, which will take in a list of arxiv ids, and create a single "Sample" for each author in that list. The current best guess for the interfact of the task is:

'''python
@task
def source_authors(arxiv_ids: List[str],model_name : str = "claude-3-5-sonnet-20240620" ) -> Task:
    """Inspect Task definition for the sourcing task. Returns a task where each Sample is a single author, created from its arxiv id.

    Args:
        fewshot (int): The number of few shots to include
        fewshot_seed (int): The seed for generating few shots
    """

    ...

    return Task(...#list of authors)
'''
The task creation code should be as follows
- Step 1 \- Go from a list of arxiv papers, to their authors, using the arxiv package. mvp.py currently has an (untested example of how to do this)
- Step 2 \- Create a list of samples, where each sample has the information from the Author dataclass in mvp.py. Turn those samples into a Task.

The task then needs a Solver, which is going to be the meat of the workflow - in inspect_ai the Solvers define the LLM workflow. Here is the current plan for the Solver:
- Step 1 \- Use claude's web search tool to find the authors. See web search tool docs in docs/claude_web_search.txt. The output of this should be a list of authors with urls to their personal websites (in particular, linkedin, github, google scholar), and a high level summary of their work. In inspect_ai this is just the same as passing web_search() as a tool to the Solver.
- Step 2  \- Given the summary of the authors, rate the authors based on the criteria above. They should be given a score between 0 and 10 for each criteria - this should use inspect's "scorer" feature to parse the LLMs' output.

We will then process the data, but we can start there.

## Important consideration

- We should use the inspect_ai library to do all of the LLM calls, and general orchestration. There is a copy of the inspect_ai documentation in docs/inspect_ai_docs/. There is also an example usage of the library in the synth_pretraining.py file.

