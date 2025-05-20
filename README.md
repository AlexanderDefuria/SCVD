# SCVD 

## Classification Process
1. Get Commit
    a. Checkout before given commit (Devign is a commit fixes dataset).
    b. Calculate CPG for project.
    c. Collect scope of commit changes using diff.
    d. Query joern CPG for commit changes.
    e. Define commit boundaries.
2. Agentic Exploration
    a. Based on current state query joern.
    b. Decide to add or not. 
    c. Finish exploration (or goto 2a).
3. Classification
    a. Message passing for graph embedding.
    b. LineVul embedding of source code.
    c. Classify.

## Preprocessing Requirements
1. Generate CPG for each before fix commit project state.
2. Calculate scope of each commit.
3. Collect project wide abstraction dictionaries (train split).
4. Generate CPG for each commit scope.
