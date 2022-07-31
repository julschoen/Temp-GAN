./gan/bin/python main.py --log_dir=AdvNFNorm --norm=True --lidc=True
./gan/bin/python main.py --log_dir=TrNFNorm --triplet=True --norm=True --lidc=True
./gan/bin/python main.py --log_dir=ClNFNorm --cl=True --norm=True --lidc=True
./run.sh
#./gan/bin/python main.py --log_dir=AdvNF --lidc=True
#./gan/bin/python main.py --log_dir=AdvOne --temp_iter=2 --norm=True --one_disc=True
