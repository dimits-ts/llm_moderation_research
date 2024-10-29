# Mitigating Polarization in Online Discussions Through Adaptive Moderation Techniques

This repository houses the code, documentation, paper and summplementary materials for my thesis conducted between the AUEB MSc in Data Science and Archimedes/Athena RC.

The subject of this thesis is the development of a framework capable of generating synthetic discussion between LLM user-agents and LLM moderators/facilitators as well as the automated annotation of conversations by LLM annotator-agents with different socio-demographic backgrounds. 

Apart from the framework itself, we include the experiments and analysis presented in the paper, as well as the produced synthetic conversation datasets.


## Abstract

Online discussion moderation/facilitation is crucial for discussions to flourish and prevent polarization and toxicity, which nowdays seem omnipresent. However, being heavily based on humans, this moderation/facilitation proves costly, time-consuming and non-scalable, which has led many to turn to LLMs for discourse facilitation. In this thesis, we explore the use of LLMs as pseudo-users in online discussions, as a cost-efficient, realistic and scalable way of substituting initial LLM facilitation experiments, which would ordinarily necessitate costly human involvement. Furthermore, we show that including socio-demographic backgrounds in our LLM users leads to more realistic discussions. We explore the use of LLM annotators to estimate discussion quality, using a new statistical test to gauge annotator polarization, and prove that using socio-demographic backgrounds in LLM annotators does not meaningfully affect their judgments. Finally, we release a synthetic-discussion creation and annotation framework, three synthetic datasets resulting from our experiments, as well as subsequent analysis and findings from these datasets.

## Concepts

`The subject of this thesis; developing a framework where many LLM user-agents can simulate online discussions. We prime the LLM user-agents to lower the quality of the conversation by any means, while concurrently instructing the LLM-moderator/facilitator to keep the conversation quality as high as possible.`

![Alt text](./paper/resources/research_goal_3.svg)


`Our framework further incorporates automated LLM-based annotations of these synthetic discussions, allowing for an inexpensive comparison of the effects of various factors such as moderator strategy, moderator presence, and LLM user prompts. Ordinarily, using LLMs for annotation presents two distinct issues; the model's inherent biases and the question of how representative their annotations are in comparison with ones that would be made by humans. While the latter concern can only be conclusively addressed by a correlation study, we attempt to address the former by using annotators with different SDBs. This also allows us to assess whether and how different LLM personalities influence the annotation process.` 
![Alt text](./paper/resources/research_goal_4.svg)

## Requirements & Usage

Refer to [src/README.md](src/README.md) for usage instruction and software requirements.

## Structure

* `src/`: Code, input/output data, results, data analysis
* `paper/`: Source code and PDF for the thesis
* `presentations/`: Presentations concerning various aspects of this research