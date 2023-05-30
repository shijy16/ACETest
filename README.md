# ACETest
This repo maintains the source code and other data for our research paper, "ACETest: Automated Constraint Extraction for Testing Deep Learning Operators", which is accepted by ISSTA 2023. ([preprint](https://arxiv.org/abs/2305.17914))



## About ACETest

ACETest is a technique which automatically extracts input constraints from the source code and generates valid test cases to test the deep functional logic of DL operators.

ACETest  works in three main steps:

+ **Input validation path extraction**: Identify the input validation code in target DL operator and explore paths in input validation code.
+ **Constraint extraction**: Extract constraints related to user controllable inputs from the paths extracted in the last step by leveraging a constraint model, a set of controllability propagation rules and a set of constraint construction rules.
+ **Testing**: Generate solutions for the extracted constraints with Z3 and use the solutions to generate python scripts to execute the target DL operator.

We have used ACETest to detect 108 previously unknown bugs on TensorFlow and PyTorch, with 87 of them confirmed by the developers and 5 CVEs assigned.



## Source Code

We are currently in the process of code cleanup, and we will make it available as soon as we finish, expected around September 2023.



## Detected Bugs

We partly list the bugs detected by ACETest [here](https://docs.google.com/spreadsheets/d/1KiyqIXJ2ZKS-5zz3QhPP4WX_qWS9WF5jk0Gr5W4meUw/edit?usp=sharing). 

Some bugs are security-related and were reported to Google OSS VRP Panel. Regarding to the vulnerability disclosure policy, they are not listed in the table.
