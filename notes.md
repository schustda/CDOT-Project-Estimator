# CDOT - Don't blow the budget

Seems like estimating construction costs is more art than science.

Can we use machine learning for estimating construction projects (tons of asphalt, linear feet of guardrail, pounds of rebar)?

Help us develop better pricing strategies.




##### Initial call (170605)

* Attendees:
    * Lekshmy Sankar - CDOT - Engineering applications manager
    * Kyle Dilbert - CDOT - Engineering Contracts Manager
* Questions
    * How quickly can get access to the data?
    * What does the data look like?
        * Each row is a project with different price breakouts?
    * How, ideally, would you use this?
        * What is the estimating process? How are projects handed to you?
        * Would you want a unit rate? unit rate range? lump sum?
* Notes:
    * Kyle - manages contracts, PSAs
    * not enough ppl bidding on projects
    * 7 on their estimating team
    * design engineer designs the project
        * they list all of the items
    * Kyle's group puts out the bid.
    * PROBLEM: when you look at the data, nothing is consistent. Estimators
        are typical.
    * CDOT is broken into 5 regions
    * Machine Learning
    * Currently using:
        * Excel
        * Omen system
        * Preconstruction
    * One estimator for each r
    * Oracle database

L
* Questions
    * How do you estimate? Have a take-off and apply unit rate?
    * Problems
        * Change orders will be worked in to the final price
        * Can you classify each change order? Unforseen conditions, addtl scope,
            missed scope.
    * Idea number one - Predict actual final construction costs
        * How to deal with change orders.
        * Factor it in to each project
    * Idea number two - Use natural language processing to find trends in Change Orders
    * Idea number three - Predict how many companies will submit a proposal
    * Idea number four - Will a specific contrator bid on this proposal
