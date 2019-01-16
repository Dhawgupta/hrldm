1. We have to  initialize each instance of env with a set of intents that have to be served in that instance
2. The reset will return a 2 states values first will be the confidence values for different slots eg. [.98, 0.12, 0, ..]
    - The second part will be the intent i.e. the goal [0,0,0,1,0]
3. As we are stoeing the intent currently being served we have to keep in mind the action that is taken , hence we can award it approprotealy , currently I am not taking the goal values as the input in the step call
