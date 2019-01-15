## This contains the basic structure for code
### Date : 14/1/19 
Working on hDQN with following specs :
**Meta Policy** :
- State : The all possible intents i.e. size of 5 for each intent
- Action : ALl the options (i.e. policy for each intent)
**Controller Policy** : 
- State : Intent space (5) + ALl slots (8) = 5 + 8 => 13
- Action : All the primitive action (8[ask each slot] + 3[hybrids] + 8[reask slots] + 1[terminate task] )
- Point : Currently I am not restructing the controller to take actions that might not be relevant to the controller (hence airfare controller will be able to take actions regardign ground service also , whoch can be made easier by restricing the controller )
- Point 2 : When the agent takes an action not meant for a specific goal the state does not change and it gets a negative iteration award (in the current model)

Thoughts I am having
1. Suppose a person inquired about the ground service like transport type like the agent asks, and the human replies as follows
"What transport type do you require"
"What all transport is available"
At this point our agent should be able to show him the appropraite transport available which can only be done if the agent has the context of what it asked in the question , for that it needs to retain its last action.
So we can have a system , where the NLU is able to detect Answer / Question in the reply and approriately take the value or give the options.
This will need other set of actions which will include showing the availabel transport, (in here also there can be 2 cases) : 
Case 1 : The Agent knows the Ground Service City and can put the appropraite options forward
Case 2 : The agent does not know the transport city and will put the options forward after asking the appropriate city. (if this also becomces a request slot , then there can be an issue in this )
