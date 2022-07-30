./gan/bin/python main.py --log_dir=AdvTest --norm=True --lidc=True
./gan/bin/python main.py --log_dir=AdvTest --lidc=True
./gan/bin/python main.py --log_dir=TrTest --triplet=True --norm=True --lidc=True
./gan/bin/python main.py --log_dir=ClTest --cl=True --norm=True --lidc=True
./run.sh
#./gan/bin/python main.py --log_dir=AdvNF --lidc=True
#./gan/bin/python main.py --log_dir=AdvOne --temp_iter=2 --norm=True --one_disc=True
