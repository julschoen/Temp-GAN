./gan/bin/python main.py --log_dir=AdvCBCT --norm=True --batch_size=16 --z_size=256 --cbct=True
./gan/bin/python main.py --log_dir=Adv4D --norm=True --batch_size=16 --z_size=256
./gan/bin/python main.py --log_dir=AdvOne --one_disc=True --norm=True --batch_size=12 --temp_iter=2 --lidc=True
./run.sh