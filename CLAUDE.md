# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sourcing application built with Python. We are currently creating a small mvp of the project, which lives in (`mvp.py`), which is a jupyter notebook that uses the ArXiv API to query for authors of academic papers. 


## Plan for sourcing application

Workflow we’re trying to automate: Here is the workflow we want to automate:

1) Take in a list of arxiv papers where we suspect we might want to hire the authors:  
2) Parse out a list of the authors from those papers.  
3) Classify whether these authors are of interest to AISI. Classifying authors should involve a custom prompt, but some important considerations are:  
   1) *They would be interested in working for AISI*. Most common thing is they have shown an interest in Safety.  
   2) *We would be interested in them.* They seem generally competent / have some hard skills that we need / have done some specific work which we are interested in.  
4) For the classified authors, extract some data about them in a structured format. Again, customizable, but some classic things might be: Current role, years of experience, links to websites.  
5) Put this data into a database. My understanding is that the current plan is Airtable.

Technical plan:

- Step one \- go from a list of arxiv papers, to their authors, using the arxiv package.
- Step two \- Use claude's web search tool to find the author's .  
- Step 3a \- Classification about these authors based on this classification, probably parse out a range of specific metrics. Can then do data analysis on these metrics.  
- Step 3b \- Parse out structured data about these authors.

## Important consideration

- We should use the inspect_ai library to do all of the LLM calls, and general orchestration. There is a copy of the inspect_ai documentation in docs/inspect_ai_docs/
- 

