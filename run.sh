./gan/bin/python main.py --log_dir=AdvCBCT2 --norm=True --z_size=256 --batch_size=16 --cbct=True
./gan/bin/python main.py --log_dir=Adv4D2 --norm=True --z_size=256 --batch_size=16
./gan/bin/python main.py --log_dir=AdvCBCT512 --norm=True --batch_size=16 --cbct=True
./gan/bin/python main.py --log_dir=Adv4D512 --norm=True --batch_size=16
./run.sh