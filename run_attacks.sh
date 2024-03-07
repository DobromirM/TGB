export PYTHONPATH=`pwd`

set -x;

declare -a attack=(
'None'
'RandomAttack(attack_dataset="train",rate=0.3)'
'RandomAttackOnline(attack_dataset="train",rate=0.3,batch_size=200)'
'ConstrainedAttack(attack_dataset="train",rate=0.3,kde_bandwidth=0.1,time_window=100,max_node_degree_strat="median")'
'ConstrainedAttackOnline(attack_dataset="train",rate=0.3,batch_size=200,kde_bandwidth=0.1,time_window=100,max_node_degree_strat="median")'
'EvolutionaryAttack(attack_dataset="train,validation",rate=0.3,mutation_rate=0.25,mutation_range=100)'
'EvolutionaryAttackOnline(attack_dataset="train",rate=0.3,batch_size=200,mutation_rate=0.25,mutation_range=100)'
)

for a in "${attack[@]}"
do
   python examples/linkproppred/tgbl-coin/tgn.py --seed 1 --reduce_ratio 0.1 --attack $a
done

for a in "${attack[@]}"
do
   python examples/linkproppred/tgbl-coin/dyrep.py --seed 1 --reduce_ratio 0.01 --attack $a
done
