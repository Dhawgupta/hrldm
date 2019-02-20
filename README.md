# HRL-DM
Repository and Code for the Heirarhcial Reinforcement Learning based Dialogue Management System


## Terminology 

### Meta Policy :
**intent-state mod1** : This is the case where the intent state is changed after the user_agent by replacing all the previous intents with the new intents, not retaining the old one. Also in each subsequent option picked the intent is not removed from the state pertaining to that option
```python
# intent state
[0,1,0,1,0] -> option 1 -> [0,1,0,1,0]
[0,1,0,1,0] -> option 5 -> [1,0,1,0,0] # new set of intents
```
**intent-state-mod2** : This will be same in terms of the mutation of intent state in each subsequent user_action (5), but after each selection of an option, the intent pertaining to that option will be removed.
```python
# intent state
[0,1,0,1,0] -> option 1 -> [0,0,0,1,0] # the intent 1 is removed
[0,1,0,1,0] -> option 5 -> [1,0,1,0,0] # new set of intents
```

**meta-reward-1** : In this reward function the meta state is rewarded in terms of teh relevenat slots that It had filled with repectd to its intents and slots after the user agent action. Even if it was acheived using a wrong option

**meta-reward-2** : In this situation , the meta policy is only rewarded if it picks up the option that corresponds to an intent in the intent state.

**meta-state-1**: In this case the state space of meta contains the union of both the intent space and the confidence values of the slot space.
```buildoutcfg
intent-space : [i0,i1,i2,i3,i4]
slot-space : [s0,s1,s2,s3,s4,s5,s6,s7]
meta-state : [i0,i1,i2,i3,i4,s0,s1,s2,s3,s4,s5,s6,s7]
```

**meta-state-2** : In this case the state space of meta contains only the intent space and no confidence values of slot space
```buildoutcfg
intent-space : [i0,i1,i2,i3,i4]
slot-space : [s0,s1,s2,s3,s4,s5,s6,s7]
meta-state : [i0,i1,i2,i3,i4]
```



