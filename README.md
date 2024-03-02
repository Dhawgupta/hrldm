# HRL-DM
Repository and Code for the Hierarchical Reinforcement Learning-based Dialogue Management System introduced in the following paper

**Paper Name:-** Towards Integrated Dialogue Policy Learning for Multiple Domains and Intents using Hierarchical Deep Reinforcement Learning
>This paper presents an expert, unified and a generic Deep Reinforcement Learning (DRL) framework that creates dialogue managers competent for managing task-oriented conversations embodying multiple domains along with their various intents and provide the user with an expert system which is a one stop for all queries. In order to address these multiple aspects, the dialogue exchange between the user and the VA is split into hierarchies, so as to isolate and identify subtasks belonging to different domains. The notion of Hierarchical Reinforcement Learning (HRL) specifically options is employed to learn optimal policies in these hierarchies that operate at varying time steps to accomplish the user goal. The dialogue manager encompasses a top-level domain meta-policy, intermediate-level intent meta-policies in order to select amongst varied and multiple subtasks or options and low-level controller policies to select primitive actions to complete the subtask given by the higher-level meta-policies in varying intents and domains. Sharing of controller policies among overlapping subtasks enables the meta-policies to be generic. The proposed expert framework has been demonstrated in the domains of “Air Travel” and “Restaurant”. Experiments as compared to several strong baselines and a state of the art model establish the efficiency of the learned policies and the need for such expert models capable of handling complex and composite tasks.

* **Authors:** Tulika Saha, Dhawal Gupta, Sriparna Saha and Pushpak Bhattacharyya
* **Affiliation:** Indian Institute of Technology Patna, India
* **Corresponding Author:** [Dhawal Gupta] (dhawgupta@gmail.com), [Tulika Saha] (sahatulika15@gmail.com)
* **Accepted(December, 2020):**  [Expert Systems with Applications](https://www.sciencedirect.com/science/article/abs/pii/S0957417420304747)

If you consider this dataset useful, please cite it as

```bash
@article{saha2020towards,
  title={Towards integrated dialogue policy learning for multiple domains and intents using hierarchical deep reinforcement learning},
  author={Saha, Tulika and Gupta, Dhawal and Saha, Sriparna and Bhattacharyya, Pushpak},
  journal={Expert Systems with Applications},
  volume={162},
  pages={113650},
  year={2020},
  publisher={Elsevier}
}
```

## Terminology 

### Meta Policy :
**intent-state mod1**: This is the case where the intent state is changed after the user_agent by replacing all the previous intents with the new intents, not retaining the old one. Also in each subsequent option picked the intent is not removed from the state of that option
```python
# intent state
[0,1,0,1,0] -> option 1 -> [0,1,0,1,0]
[0,1,0,1,0] -> option 5 -> [1,0,1,0,0] # new set of intents
```
**intent-state-mod2**: This will be the same in terms of the mutation of intent state in each subsequent user_action (5), but after each selection of an option, the intent of that option will be removed.
```python
# intent state
[0,1,0,1,0] -> option 1 -> [0,0,0,1,0] # the intent 1 is removed
[0,1,0,1,0] -> option 5 -> [1,0,1,0,0] # new set of intents
```

**meta-reward-1**: In this reward function the meta state is rewarded in terms of the relevant slots that It had filled with respect to its intents and slots after the user agent action. Even if it was achieved using a wrong option

**meta-reward-2**: In this situation, the meta policy is only rewarded if it picks up the option that corresponds to an intent in the intent state.

**meta-state-1**: In this case, the state space of meta contains the union of both the intent space and the confidence values of the slot space.
```buildoutcfg
intent-space : [i0,i1,i2,i3,i4]
slot-space : [s0,s1,s2,s3,s4,s5,s6,s7]
meta-state : [i0,i1,i2,i3,i4,s0,s1,s2,s3,s4,s5,s6,s7]
```

**meta-state-2**: In this case, the state space of meta contains only the intent space and no confidence values of slot space
```buildoutcfg
intent-space : [i0,i1,i2,i3,i4]
slot-space : [s0,s1,s2,s3,s4,s5,s6,s7]
meta-state : [i0,i1,i2,i3,i4]
```

**controller-type-1**: This is the single neural network-based controller policy that will be used for training the meta policy

**controller-type-2**: In this case the individual neural net will be used for each intent in the controller policy part.


# Contact

For any queries, feel free to contact the corresponding authors.

