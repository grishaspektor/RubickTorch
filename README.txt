------ TO RUN TRAINING -------
# create save directory:

>> mkdir save

# train all three networks (for more options, see parser documentations):

>> python reinforcement_Cube_V14.py --num_episodes 1000000

------ TO TEST NETWORK -------

>> python test_Cube.py --filename './save/sim4/net_V13_Reverse_TD_1000000_@_20_07_18_Time_10_16' --test_level 10

output:
Loading network ./save/sim4/net_V13_Reverse_TD_1000000_@_20_07_18_Time_10_16
Level 1, Success rate 1.0
Level 2, Success rate 0.996
Level 3, Success rate 0.978
Level 4, Success rate 0.926
Level 5, Success rate 0.698
Level 6, Success rate 0.512
Level 7, Success rate 0.38
Level 8, Success rate 0.228
Level 9, Success rate 0.132
Level 10, Success rate 0.094
