export PYTHONPATH=`pwd`

set -x;

declare -a attack=(
'None'
'RandomAttack(attack_dataset="train",rate=0.1)'
'RandomAttack(attack_dataset="train",rate=0.2)'
'RandomAttackOnline(attack_dataset="train",rate=0.1,batch_size=200)'
'RandomAttackOnline(attack_dataset="train",rate=0.2,batch_size=200)'
)

for a in "${attack[@]}"
do
   python examples/linkproppred/tgbl-coin/dyrep.py --seed 1 --reduce_ratio 0.01 --attack $a
done


